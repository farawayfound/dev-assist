"""Offline tests for the Assistant's tool loop (2026-10-07, capabilities): exocortex.llm.assistant
with a `tools=` runner.

A duck-typed FakeRunner (wants / specs / budget / run / map_text, handing back the real
`exocortex.capabilities` dataclasses) and a scripted FakeClient stand in for the policy layer and
the model; the EventBus, the Transcript, the ContextManager and the prompt builders are real. The
client is scripted one batch of pieces per stream, e.g. a call and then an answer:

    [['<tool>{"name": "read_file", "args": {"path": "a.py"}}'], ["The file defines Foo."]]

and, like llama-server, it cuts a stream at the first stop string it was sent and does not yield
the stop string itself.
"""
from __future__ import annotations

import queue
import threading
import time
from types import SimpleNamespace
from typing import Any, Callable

import pytest

from exocortex.bus import (
    AssistDeltaEvent,
    AssistDoneEvent,
    AssistErrorEvent,
    AssistStartedEvent,
    EventBus,
    Source,
    ToolCallEvent,
    TranscriptSegment,
)
from exocortex.capabilities import ToolBudget, ToolCall, ToolResult, ToolRunner, ToolSpec
from exocortex.config import Config
from exocortex.context import ContextManager, ContextStore
from exocortex.llm import prompts
from exocortex.llm.assistant import Assistant, _ToolSplitter
from exocortex.llm.client import ChatDelta, Health, LLMError, ModelInfo, StreamStats
from exocortex.transcript import Transcript

OVERFLOW = LLMError(
    "HTTP 400: request (5000 tokens) exceeds the available context size (4096 tokens), try increasing it"
)
READ_CALL = '<tool>{"name": "read_file", "args": {"path": "a.py"}}'
READ_CALL_FULL = READ_CALL + "</tool>"
FILE_TEXT = "class Foo:\n    pass"
READ_BLOCK = prompts.tool_result_block("read_file", FILE_TEXT)
#: 2026-10-08: a pass that may make a call stops at the closing tag (server-side: the hop completes
#: on its slot and the continuation reuses its cache); the answer pass, with no calls left, stops at
#: an opening tag instead.
CALL_STOP = prompts.STOP + [prompts.TOOL_CLOSE]
FINAL_STOP = prompts.STOP + [prompts.TOOL_OPEN]


def turn(block: str) -> str:
    """How a result continues the raw prompt (2026-10-08): the assistant turn that wrote the call is
    closed, the result block comes back as a USER turn, and a fresh assistant turn opens with the
    no-think prefix -- a tool's reply is a message to the model, never text inside its own turn."""
    return (f"{prompts.IM_END}\n{prompts.IM_START}user\n{block.strip(chr(10))}{prompts.IM_END}\n"
            f"{prompts.IM_START}assistant\n{prompts.NO_THINK_PREFIX}")


NUDGED_BLOCK = prompts.tool_result_block("read_file", FILE_TEXT + "\n(write the answer now)")

# --------------------------------------------------------------------------------------------
# doubles
# --------------------------------------------------------------------------------------------


class FakeClient:
    """Scripted LMStudioClient: `batches` (one list of pieces per stream call, "ok" when it runs
    out), `reasons` (finish_reason per call, "stop" by default), `reports` ((prompt_tokens,
    cached_tokens) per call), `fail_with` (an LLMError or None per call: raised before the batch is
    consumed). Every request is recorded in `calls`. Like a real server it cuts the stream at the
    first stop string (`honor_stop=False` simulates one that does not)."""

    def __init__(self, batches: list[list[str]] | None = None, *, reasons: list[str] | None = None,
                 reports: list[tuple[int | None, int | None]] | None = None, honor_stop: bool = True) -> None:
        self.cfg = SimpleNamespace(model="maple-preview", base_url="http://fake.test", auto_load_model=True)
        self._local = threading.local()
        self.batches = list(batches or [])
        self.reasons = list(reasons or [])
        self.reports = list(reports or [])
        self.honor_stop = honor_stop
        self.fail_with: list[LLMError | None] = []
        self.calls: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    @property
    def last_stats(self) -> StreamStats | None:
        return getattr(self._local, "value", None)

    @last_stats.setter
    def last_stats(self, value: StreamStats | None) -> None:
        self._local.value = value

    def health(self) -> Health:
        return Health(reachable=True, model_listed=True, model_loaded=True, detail="")

    def ensure_loaded(self) -> bool:
        return True

    def list_models(self) -> list[ModelInfo]:
        return [ModelInfo(id="maple-preview", loaded=True)]

    def context_window(self) -> int | None:
        return None

    def stream_completion(self, prompt: str, *, max_tokens: int, temperature: float | None = None,
                          top_p: float | None = None, stop: list[str] | None = None,
                          cancel: threading.Event | None = None, extra_body: dict[str, Any] | None = None):
        record = {"kind": "completion", "prompt": prompt, "max_tokens": max_tokens, "stop": list(stop or []),
                  "extra_body": extra_body}
        yield from self._run(record, cancel)

    def stream_chat(self, messages: list[dict[str, str]], *, max_tokens: int, temperature: float | None = None,
                    top_p: float | None = None, cancel: threading.Event | None = None,
                    stop: list[str] | None = None):
        record = {"kind": "chat", "messages": [dict(m) for m in messages], "max_tokens": max_tokens,
                  "stop": list(stop or [])}
        for piece in self._run(record, cancel):
            yield ChatDelta(content=piece, reasoning="")

    def _run(self, record: dict[str, Any], cancel: threading.Event | None):
        with self._lock:
            n = len(self.calls)
            self.calls.append(record)
            failure = self.fail_with.pop(0) if self.fail_with else None
            if failure is None:
                batch = self.batches.pop(0) if self.batches else ["ok"]
            reason = self.reasons[n] if n < len(self.reasons) else "stop"
            report = self.reports[n] if n < len(self.reports) else (None, None)
        if failure is not None:
            raise failure
        stops = record["stop"] if self.honor_stop else []
        produced = ""
        count = 0
        try:
            for piece in batch:
                if cancel is not None and cancel.is_set():
                    break
                candidate = produced + piece
                cuts = [candidate.find(s) for s in stops if s in candidate]
                if cuts:
                    head = candidate[len(produced):min(cuts)]
                    if head:
                        count += 1
                        yield head
                    break
                produced = candidate
                count += 1
                yield piece
        finally:
            self.last_stats = StreamStats(
                ttft_s=0.05 + 0.01 * n, tokens=count, elapsed_s=0.1, tok_per_s=float(count) / 0.1, cancelled=False,
                prompt_tokens=report[0], cached_tokens=report[1], finish_reason=reason,
            )


READ_SPECS = [
    ToolSpec("find_files", "Find files by name pattern.", (("pattern", "a glob"), ("path", "a folder, optional"))),
    ToolSpec("list_dir", "List a folder.", (("path", "the folder"),)),
    ToolSpec("read_file", "Read a text file.", (("path", "the file"), ("start_line", "first line, optional"))),
    ToolSpec("search_text", "Search the files for a literal text.", (("query", "the text"),)),
]
ACTIVE_SPECS = [ToolSpec("run_command", "Run an allowed command.", (("command", "the command line"),))]
EDIT_SPECS = [
    ToolSpec("edit_file", "Replace one text in a file.", (("path", "the file"), ("old", "text"), ("new", "text"))),
    ToolSpec("write_file", "Write a whole file.", (("path", "the file"), ("content", "the text"))),
]


class FakeRunner:
    """Duck-typed ToolRunner. Read tools for every kind, `run_command` (and the edit tools with
    `edit=True`) for the ask box only; the ask box gets `calls` calls, every other kind
    `live_calls`. `run` records the call, charges EVERY call (a repeat too -- unlike the real
    `ToolRunner`, where an identical repeat is free, which is what the loop's `hard_cap` net is for:
    see `test_a_model_that_repeats_one_call_forever_...`), honours `names` -> results (a ToolResult,
    a list of them consumed in order, or a callable) and calls `on_run` first."""

    def __init__(self, *, results: dict[str, Any] | None = None, calls: int = 2, live_calls: int = 2,
                 chars: int = 2000, map_text: str = "", wants: bool = True, edit: bool = False,
                 deadline_s: float = 60.0, on_run: Callable[[ToolCall, ToolBudget], None] | None = None) -> None:
        self.results = results or {}
        self.default = ToolResult(FILE_TEXT, "done", "20 B", "a.py")
        self.calls, self.live_calls, self.chars = calls, live_calls, chars
        self._map, self._wants, self._edit, self.deadline_s, self.on_run = map_text, wants, edit, deadline_s, on_run
        self.wants_calls: list[str] = []
        self.budgets: list[ToolBudget] = []
        self.runs: list[tuple[ToolCall, ToolBudget, threading.Event | None]] = []

    def wants(self, kind: str) -> bool:
        self.wants_calls.append(kind)
        return self._wants

    def specs(self, kind: str) -> list[ToolSpec]:
        out = list(READ_SPECS)
        if kind == "ask":
            out += ACTIVE_SPECS + (EDIT_SPECS if self._edit else [])
        return sorted(out, key=lambda s: s.name)

    def budget(self, kind: str) -> ToolBudget:
        budget = ToolBudget(
            kind=kind, allowed=frozenset(s.name for s in self.specs(kind)),
            calls_left=self.calls if kind == "ask" else self.live_calls, chars=self.chars,
            deadline=time.monotonic() + self.deadline_s, web_left=3,
        )
        self.budgets.append(budget)
        return budget

    def run(self, call: ToolCall, *, budget: ToolBudget, cancel: threading.Event | None = None) -> ToolResult:
        self.runs.append((call, budget, cancel))
        if self.on_run is not None:
            self.on_run(call, budget)
        budget.calls_left = max(0, budget.calls_left - 1)
        if call.name not in budget.allowed:
            return ToolResult("denied: tool not available for this kind of answer", "denied",
                              "tool not available for this kind of answer", call.name)
        scripted = self.results.get(call.name, self.default)
        if callable(scripted):
            return scripted(call, budget)
        if isinstance(scripted, list):
            return scripted.pop(0) if len(scripted) > 1 else scripted[0]
        return scripted

    def map_text(self) -> str:
        return self._map


# --------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------


def make_segment(seg_id: int, *, text: str = "hello", source: Source = Source.SYSTEM,
                 is_question: bool = False) -> TranscriptSegment:
    return TranscriptSegment(
        id=seg_id, source=source, speaker_id="spk1", speaker_label="Speaker 1", text=text,
        t_start=float(seg_id), t_end=float(seg_id) + 1.0, confidence=0.9, is_question=is_question,
    )


def drain_until(sub: "queue.Queue[Any]", predicate: Callable[[Any], bool], timeout: float = 5.0) -> list[Any]:
    deadline = time.monotonic() + timeout
    collected: list[Any] = []
    while time.monotonic() < deadline:
        try:
            ev = sub.get(timeout=0.05)
        except queue.Empty:
            continue
        collected.append(ev)
        if predicate(ev):
            return collected
    raise AssertionError(f"timed out waiting for event; collected so far: {collected!r}")


def finished(ev: Any) -> bool:
    return isinstance(ev, (AssistDoneEvent, AssistErrorEvent))


@pytest.fixture()
def cfg() -> Config:
    c = Config()
    c.llm.thinking = "off"            # the raw-completion path: prompts are rendered ChatML
    return c


@pytest.fixture()
def bus() -> EventBus:
    return EventBus()


@pytest.fixture()
def transcript() -> Transcript:
    t = Transcript()
    t.add(make_segment(1, text="We are presenting the Atlas project to the platform team today."))
    return t


def make_assistant(cfg: Config, bus: EventBus, client: FakeClient, transcript: Transcript,
                   runner: FakeRunner | None) -> Assistant:
    manager = ContextManager(cfg, ContextStore(None), transcript)
    return Assistant(cfg, bus, client, transcript, clock=time.monotonic, context=manager, tools=runner)


def ask(assistant: Assistant, sub: "queue.Queue[Any]", text: str = "What does a.py define?") -> list[Any]:
    """Ask `text` and return every event up to and including the job's Done / Error."""
    assistant.ask(text)
    return drain_until(sub, finished)


def run_ask(cfg: Config, bus: EventBus, transcript: Transcript, client: FakeClient, runner: FakeRunner | None,
            text: str = "What does a.py define?") -> tuple[Assistant, list[Any]]:
    a = make_assistant(cfg, bus, client, transcript, runner)
    sub = bus.subscribe()
    a.start()
    try:
        events = ask(a, sub, text)
    finally:
        a.stop()
    return a, events


def system_of(prompt: str) -> str:
    return prompt.split("<|im_start|>system\n", 1)[1].split("<|im_end|>", 1)[0]


def user_of(prompt: str) -> str:
    return prompt.split("<|im_start|>user\n", 1)[1].split("<|im_end|>", 1)[0]


def tool_events(events: list[Any]) -> list[ToolCallEvent]:
    return [e for e in events if isinstance(e, ToolCallEvent)]


def done_of(events: list[Any]) -> AssistDoneEvent:
    return next(e for e in events if isinstance(e, AssistDoneEvent))


def prompts_of(client: FakeClient) -> list[str]:
    return [c["prompt"] for c in client.calls]


# --------------------------------------------------------------------------------------------
# the basic loop
# --------------------------------------------------------------------------------------------


def test_one_read_file_call_continues_the_same_completion(cfg, bus, transcript) -> None:
    client = FakeClient([[READ_CALL], ["The file defines Foo."]])
    runner = FakeRunner()
    a, events = run_ask(cfg, bus, transcript, client, runner)
    first, second = prompts_of(client)
    # the continuation is the first prompt + what the model generated (the KV prefix is intact) +
    # the closing tag the server stripped, then the result as a user turn and a fresh assistant turn
    assert second.startswith(first + READ_CALL)
    assert second == first + READ_CALL_FULL + turn(READ_BLOCK)
    assert first.endswith(prompts.NO_THINK_PREFIX)
    # a call pass stops at the closing tag; the second pass may still call (one call left)
    assert [c["stop"] for c in client.calls] == [CALL_STOP, CALL_STOP]
    assert done_of(events).text == "The file defines Foo."
    assert not done_of(events).cancelled
    # what the runner was handed
    (call, budget, cancel), = runner.runs
    assert (call.name, call.args) == ("read_file", {"path": "a.py"})
    assert call.raw == '{"name": "read_file", "args": {"path": "a.py"}}'
    assert budget.kind == "ask" and isinstance(cancel, threading.Event)


def test_events_come_in_order_and_only_visible_text_reaches_the_deltas(cfg, bus, transcript) -> None:
    client = FakeClient([[READ_CALL], ["The file ", "defines Foo."]])
    a, events = run_ask(cfg, bus, transcript, client, FakeRunner())
    kinds = [type(e).__name__ for e in events
             if isinstance(e, (AssistStartedEvent, ToolCallEvent, AssistDeltaEvent, AssistDoneEvent))]
    assert kinds[0] == "AssistStartedEvent"
    assert kinds[1:3] == ["ToolCallEvent", "ToolCallEvent"]
    assert kinds[-1] == "AssistDoneEvent"
    assert set(kinds[3:-1]) == {"AssistDeltaEvent"} and len(kinds[3:-1]) >= 1
    running, done = tool_events(events)
    assert (running.state, done.state) == ("running", "done")
    assert (running.index, done.index) == (1, 1)
    assert (done.name, done.summary, done.detail) == ("read_file", "a.py", "20 B")
    assert running.summary == "a.py" and done.elapsed_s >= 0.0
    deltas = "".join(e.delta for e in events if isinstance(e, AssistDeltaEvent))
    assert deltas == "The file defines Foo."
    assert "<tool>" not in deltas and "<result" not in deltas
    assert done_of(events).text == "The file defines Foo."


def test_the_history_holds_the_visible_text_only(cfg, bus, transcript) -> None:
    client = FakeClient([[READ_CALL], ["The file defines Foo."]])
    a, events = run_ask(cfg, bus, transcript, client, FakeRunner())
    (_t, question, answer), = a.answers
    assert question == "What does a.py define?" and answer == "The file defines Foo."


def test_the_system_prompt_ends_with_the_rule_and_the_tools_block(cfg, bus, transcript) -> None:
    client = FakeClient([["Plain answer."]])
    runner = FakeRunner(calls=3)
    run_ask(cfg, bus, transcript, client, runner)
    system = system_of(client.calls[0]["prompt"])
    block = prompts.tools_block(runner.specs("ask"), max_calls=3)
    assert system.endswith("\n" + prompts._TOOL_RULE + "\n" + block)
    assert "- run_command(command: the command line): Run an allowed command." in system
    assert "At most 3 calls per answer." in system


def test_the_user_turn_carries_the_map_between_earlier_and_the_transcript(cfg, bus, transcript) -> None:
    client = FakeClient([["Plain answer."]])
    run_ask(cfg, bus, transcript, client, FakeRunner(map_text="src/\n  app.py -- the entry point"))
    user = user_of(client.calls[0]["prompt"])
    assert user.startswith("<map>\nsrc/\n  app.py -- the entry point\n</map>\n\n<transcript>")


def test_the_map_is_cut_to_map_tokens_on_a_line_boundary(cfg, bus, transcript) -> None:
    cfg.capabilities.map_tokens = 20
    long_map = "\n".join(f"src/module_{i}.py -- does thing number {i}" for i in range(40))
    client = FakeClient([["Plain answer."]])
    a, _events = run_ask(cfg, bus, transcript, client, FakeRunner(map_text=long_map))
    user = user_of(client.calls[0]["prompt"])
    body = user.split("<map>\n", 1)[1].split("\n</map>", 1)[0]
    limit = 4 * int(20 * cfg.context.chars_per_token)          # the safety net: 4x the configured ratio
    assert 0 < len(body) <= limit < len(long_map)
    assert all(line in long_map.split("\n") for line in body.split("\n"))      # no line is cut in half


def map_body(prompt: str) -> str:
    return user_of(prompt).split("<map>\n", 1)[1].split("\n</map>", 1)[0]


def test_a_map_within_four_times_the_configured_figure_is_used_verbatim(cfg, bus, transcript) -> None:
    cfg.capabilities.map_tokens = 20
    text = "\n".join(f"src/m{i}.py" for i in range(12))
    assert int(20 * cfg.context.chars_per_token) < len(text) < 4 * int(20 * cfg.context.chars_per_token)
    client = FakeClient([["Plain answer."]])
    run_ask(cfg, bus, transcript, client, FakeRunner(map_text=text))
    assert map_body(client.calls[0]["prompt"]) == text            # the engine cut it once; no second cut here


def test_the_map_block_is_byte_identical_across_jobs_whatever_the_calibration_does(cfg, bus, transcript) -> None:
    cfg.capabilities.map_tokens = 100                              # 360 configured characters, 1440 as the net
    text = "\n".join(f"src/module_{i}.py -- does thing number {i}" for i in range(30))
    assert 360 < len(text) < 1440
    client = FakeClient([["One."], ["Two."], ["Three."]], reports=[(3000, 0)] * 3)    # the server counts MORE tokens
    a = make_assistant(cfg, bus, client, transcript, FakeRunner(map_text=text))
    cpt = [a.context.estimator.chars_per_token]
    sub = bus.subscribe()
    a.start()
    try:
        for question in ("First?", "Second?", "Third?"):
            ask(a, sub, question)
            cpt.append(a.context.estimator.chars_per_token)
    finally:
        a.stop()
    assert cpt[0] > cpt[1] > cpt[2] > cpt[3]                      # the calibration really moved, every time
    assert [map_body(c["prompt"]) for c in client.calls] == [text] * 3
    heads = [user_of(c["prompt"]).split("<transcript>", 1)[0] for c in client.calls]
    assert heads[0] == heads[1] == heads[2] and heads[0].endswith("</map>\n\n")


def test_a_map_over_the_safety_net_is_cut_on_a_line_boundary_with_the_configured_ratio(cfg, bus, transcript) -> None:
    cfg.capabilities.map_tokens = 20
    cfg.context.chars_per_token = 2.0                              # net = 4 * int(20 * 2.0) = 160 characters
    text = "\n".join(f"src/module_{i}.py -- does thing number {i}" for i in range(40))
    client = FakeClient([["Plain answer."]])
    run_ask(cfg, bus, transcript, client, FakeRunner(map_text=text))
    body = map_body(client.calls[0]["prompt"])
    assert 0 < len(body) <= 160 and all(line in text.split("\n") for line in body.split("\n"))


def test_no_map_block_without_a_map_or_with_zero_map_tokens(cfg, bus, transcript) -> None:
    client = FakeClient([["x"]])
    run_ask(cfg, bus, transcript, client, FakeRunner(map_text=""))
    assert "<map>" not in user_of(client.calls[0]["prompt"])
    cfg.capabilities.map_tokens = 0
    client = FakeClient([["x"]])
    run_ask(cfg, bus, Transcript(), client, FakeRunner(map_text="src/\n  a.py"))
    assert "<map>" not in user_of(client.calls[0]["prompt"])


def test_the_budget_chars_and_the_map_are_reserved_in_the_window(cfg, bus, transcript) -> None:
    seen: list[int] = []

    def run_once(runner: FakeRunner | None) -> int:
        client = FakeClient([["x"]])
        a = make_assistant(cfg, bus, client, transcript, runner)
        real = a.context.build

        def spy(**kw):
            seen.append(kw["fixed_chars"])
            return real(**kw)

        a.context.build = spy              # type: ignore[method-assign]
        sub = bus.subscribe()
        a.start()
        try:
            ask(a, sub)
        finally:
            a.stop()
        return seen[-1]

    plain = run_once(None)
    with_tools = run_once(FakeRunner(chars=5000, map_text="m" * 300))
    assert with_tools == plain + 5000 + 300


# --------------------------------------------------------------------------------------------
# denied / malformed / exhausted
# --------------------------------------------------------------------------------------------


def test_a_denied_call_is_reported_and_the_loop_continues(cfg, bus, transcript) -> None:
    denied = ToolResult("denied: that path is outside the readable folders", "denied", "outside the folders", "../x")
    client = FakeClient([['<tool>{"name": "read_file", "args": {"path": "../x"}}'], ["I could not read that file."]])
    a, events = run_ask(cfg, bus, transcript, client, FakeRunner(results={"read_file": denied}))
    running, outcome = tool_events(events)
    assert (outcome.state, outcome.detail, outcome.summary) == ("denied", "outside the folders", "../x")
    assert done_of(events).text == "I could not read that file."
    second = client.calls[1]["prompt"]
    assert turn(prompts.tool_result_block("read_file", denied.text)) in second
    assert len(client.calls) == 2


def test_an_error_result_event_falls_back_to_the_first_line_of_the_text(cfg, bus, transcript) -> None:
    err = ToolResult("error: no such file\nmore lines", "error", "", "")
    client = FakeClient([[READ_CALL], ["Sorry."]])
    a, events = run_ask(cfg, bus, transcript, client, FakeRunner(results={"read_file": err}))
    outcome = tool_events(events)[1]
    assert outcome.state == "error" and outcome.detail == "error: no such file"
    assert outcome.summary == "a.py"                    # the call's own argument when the result names none


def test_the_first_malformed_call_is_free_and_the_second_counts(cfg, bus, transcript) -> None:
    client = FakeClient([["<tool>this is not json"], ["<tool>{still: bad json"], [READ_CALL], ["Done."]])
    runner = FakeRunner(calls=2)
    a, events = run_ask(cfg, bus, transcript, client, runner)
    assert len(client.calls) == 4
    budget = runner.budgets[0]
    # free, counted, then the real call spent the last of 2... (2 -> 2 -> 1 -> 0)
    assert len(runner.runs) == 1 and budget.calls_left == 0
    notice = ('(the call could not be parsed: the call is not valid JSON; expected '
              '{"name": "<tool name>", "args": {"<argument>": <value>}}. Write exactly: '
              '<tool>{"name": ..., "args": {...}}</tool>)')
    third = client.calls[2]["prompt"]
    assert third.count('<result name="tool">') == 2 and notice in third      # unsanitised: it names <tool> on purpose
    assert "<tool>this is not json</tool>" in third and "<tool>{still: bad json</tool>" in third
    # the call that spent the last one says so, and the answer pass has no call stop
    last = client.calls[3]
    assert "(no calls left: write the answer now)" in last["prompt"]
    assert last["stop"] == FINAL_STOP
    assert [c["stop"] for c in client.calls[:3]] == [CALL_STOP] * 3
    assert done_of(events).text == "Done."
    assert tool_events(events) and all(e.name == "read_file" for e in tool_events(events))   # a bad call has no event


def test_a_malformed_call_alone_does_not_spend_a_call(cfg, bus, transcript) -> None:
    client = FakeClient([["<tool>nope"], ["Fine, no tools needed."]])
    runner = FakeRunner(calls=1)
    a, events = run_ask(cfg, bus, transcript, client, runner)
    assert runner.budgets[0].calls_left == 1 and not runner.runs
    assert done_of(events).text == "Fine, no tools needed."
    assert client.calls[1]["stop"] == CALL_STOP          # the call is still unspent: a call pass


def test_a_second_malformed_call_that_spends_the_last_call_ends_the_calling(cfg, bus, transcript) -> None:
    client = FakeClient([["<tool>nope"], ["<tool>nope again"], ["Answer without tools."]])
    runner = FakeRunner(calls=1)
    a, events = run_ask(cfg, bus, transcript, client, runner)
    assert not runner.runs and runner.budgets[0].calls_left == 0
    assert client.calls[2]["stop"] == FINAL_STOP
    assert "(no calls left: write the answer now)" in client.calls[2]["prompt"]
    assert done_of(events).text == "Answer without tools."


def test_a_stream_cut_by_the_length_limit_with_a_call_is_malformed(cfg, bus, transcript) -> None:
    # the JSON happens to be complete, but the model was cut off: treat the call as untrustworthy
    client = FakeClient([[READ_CALL], ["Answer."]], reasons=["length", "stop"])
    runner = FakeRunner()
    a, events = run_ask(cfg, bus, transcript, client, runner)
    assert not runner.runs
    assert "the call was cut off by the length limit" in client.calls[1]["prompt"]
    assert done_of(events).text == "Answer."


def test_calls_left_exhausted_gives_the_no_calls_left_line_and_a_stop_only_stream(cfg, bus, transcript) -> None:
    client = FakeClient([[READ_CALL], ["The file defines Foo."]])
    a, events = run_ask(cfg, bus, transcript, client, FakeRunner(calls=1))
    second = client.calls[1]
    assert second["stop"] == FINAL_STOP
    assert second["prompt"].endswith(READ_CALL_FULL + turn(
        prompts.tool_result_block("read_file", FILE_TEXT + "\n(no calls left: write the answer now)")))
    assert done_of(events).text == "The file defines Foo."


def test_a_call_still_started_when_none_are_left_is_cut_and_its_visible_part_kept(cfg, bus, transcript) -> None:
    client = FakeClient([[READ_CALL], ['It defines Foo.\n<tool>{"name": "read_file", "args": {"path": "b.py"}}', " more"]])
    runner = FakeRunner(calls=1)
    a, events = run_ask(cfg, bus, transcript, client, runner)
    assert len(runner.runs) == 1 and len(client.calls) == 2
    assert done_of(events).text == "It defines Foo.\n"
    deltas = "".join(e.delta for e in events if isinstance(e, AssistDeltaEvent))
    assert deltas == "It defines Foo.\n"


def test_the_time_budget_running_out_during_a_tool_gives_the_same_line(cfg, bus, transcript) -> None:
    def expire(_call: ToolCall, budget: ToolBudget) -> None:
        budget.deadline = time.monotonic() - 1.0

    client = FakeClient([[READ_CALL], ["Short answer."]])
    a, events = run_ask(cfg, bus, transcript, client, FakeRunner(calls=4, on_run=expire))
    assert client.calls[1]["stop"] == FINAL_STOP
    assert "(no calls left: write the answer now)" in client.calls[1]["prompt"]
    assert done_of(events).text == "Short answer."


def test_a_call_that_arrives_after_the_deadline_is_not_run(cfg, bus, transcript) -> None:
    runner = FakeRunner(calls=4)
    client = FakeClient([[READ_CALL], ["Answer."]])
    a = make_assistant(cfg, bus, client, transcript, runner)
    real_budget = runner.budget

    def short_budget(kind: str) -> ToolBudget:
        b = real_budget(kind)
        b.deadline = time.monotonic() + 0.4
        return b

    runner.budget = short_budget          # type: ignore[method-assign]
    real_create = client.stream_completion

    def slow(prompt, **kw):
        if not client.calls:
            time.sleep(0.6)               # the first pass outlasts the budget
        yield from real_create(prompt, **kw)

    client.stream_completion = slow       # type: ignore[method-assign]
    sub = bus.subscribe()
    a.start()
    try:
        events = ask(a, sub)
    finally:
        a.stop()
    assert not runner.runs
    running, outcome = tool_events(events)
    assert (running.state, outcome.state) == ("running", "denied")
    assert "(no calls left: write the answer now)" in client.calls[1]["prompt"]
    assert done_of(events).text == "Answer."


# --------------------------------------------------------------------------------------------
# the splitter in the loop: narration, fences, lead markers, two calls
# --------------------------------------------------------------------------------------------


def test_text_before_the_call_stays_visible_and_in_the_continuation(cfg, bus, transcript) -> None:
    client = FakeClient([["Checking the file.\n" + READ_CALL], ["Done."]])
    a, events = run_ask(cfg, bus, transcript, client, FakeRunner())
    first, second = prompts_of(client)
    assert second == first + "Checking the file.\n" + READ_CALL_FULL + turn(READ_BLOCK)
    assert done_of(events).text == "Checking the file.\nDone."


def test_a_special_token_the_model_wrote_is_defused_in_the_continuation(cfg, bus, transcript) -> None:
    client = FakeClient([["Odd <|fim_prefix|> text.\n" + READ_CALL], ["Done."]])
    a, events = run_ask(cfg, bus, transcript, client, FakeRunner())
    second = client.calls[1]["prompt"]
    assert "<|fim_prefix|>" not in second and "Odd <¦fim_prefix|> text.\n" in second
    assert done_of(events).text.endswith("Done.")


def test_a_fenced_and_bulleted_call_is_cleaned_in_the_continuation(cfg, bus, transcript) -> None:
    client = FakeClient([["```json\n- " + READ_CALL], ["The file defines Foo."]])
    a, events = run_ask(cfg, bus, transcript, client, FakeRunner())
    first, second = prompts_of(client)
    assert second == first + READ_CALL_FULL + turn(READ_BLOCK)        # no fence, no bullet
    assert done_of(events).text == "The file defines Foo."
    assert "```" not in "".join(e.delta for e in events if isinstance(e, AssistDeltaEvent))


def test_two_calls_in_one_stream_run_only_the_first(cfg, bus, transcript) -> None:
    second_call = '<tool>{"name": "list_dir", "args": {"path": "."}}'
    client = FakeClient([[READ_CALL + "</tool>\n" + second_call + "</tool>"], ["Done."]], honor_stop=False)
    runner = FakeRunner()
    a, events = run_ask(cfg, bus, transcript, client, runner, "Look around.")
    assert [c.name for c, _b, _c in runner.runs] == ["read_file"]
    assert "list_dir" not in client.calls[1]["prompt"].split("<|im_start|>assistant\n", 1)[1]


def test_a_server_that_ignores_the_stop_string_is_cut_at_the_closing_tag(cfg, bus, transcript) -> None:
    client = FakeClient([[READ_CALL_FULL, "\nThe file defines Bar (made up).", ""], ["The file defines Foo."]],
                        honor_stop=False)
    runner = FakeRunner()
    a, events = run_ask(cfg, bus, transcript, client, runner)
    assert len(runner.runs) == 1
    assert done_of(events).text == "The file defines Foo."
    assert "made up" not in client.calls[1]["prompt"]


def test_a_quoted_tool_marker_in_the_middle_of_a_line_is_ordinary_text_cut_at_the_closing_tag(
        cfg, bus, transcript) -> None:
    # 2026-10-08: </tool> is the server's stop string for a call pass (what keeps a hop on its slot
    # and its cache); a model that spells the tag out in its own prose loses the rest of the
    # sentence -- accepted, the rule tells it never to mention the tools. A name-only object is
    # still no call: nothing runs, the text before the tag is the answer.
    text = 'Use <tool>{"name": "read_file"}</tool>-style calls sparingly.'
    client = FakeClient([[text]])
    runner = FakeRunner()
    a, events = run_ask(cfg, bus, transcript, client, runner)
    assert not runner.runs and len(client.calls) == 1
    cut = text.split("</tool>", 1)[0]
    assert done_of(events).text == cut
    assert "".join(e.delta for e in events if isinstance(e, AssistDeltaEvent)) == cut


@pytest.mark.parametrize("text", [
    "A call is finished by the </tool> tag, and the result follows it.",         # a bare closing tag
    'Sure. <tool>{"name": "read_file"}</tool> shows the shape.',                 # a name with no arguments
    "Write <tool>{}</tool> to start. More text after it.",                       # an empty object
    "Write <tool> to open a call and </tool> to close it.",                      # the markers as words
])
def test_a_closing_tag_or_a_quoted_marker_in_prose_runs_nothing_and_is_cut_at_the_tag(
        cfg, bus, transcript, text: str) -> None:
    # none of these is a call (no arguments, an empty object, the markers as words): nothing runs;
    # the server's stop string cuts the prose at the first </tool> (2026-10-08, see above)
    client = FakeClient([[text]])
    runner = FakeRunner()
    a, events = run_ask(cfg, bus, transcript, client, runner)
    assert not runner.runs and len(client.calls) == 1
    assert done_of(events).text == text.split("</tool>", 1)[0]


def test_a_closing_tag_in_the_answer_after_a_tool_ran_is_cut_there_while_calls_are_left(
        cfg, bus, transcript) -> None:
    answer = "The call ends with </tool> and then the result follows."
    client = FakeClient([[READ_CALL], [answer]])
    a, events = run_ask(cfg, bus, transcript, client, FakeRunner())
    assert len(client.calls) == 2 and done_of(events).text == "The call ends with "


def test_a_call_with_the_pieces_split_anywhere_never_leaks_a_partial_marker(cfg, bus, transcript) -> None:
    pieces = list(READ_CALL)                      # one character per piece
    client = FakeClient([pieces, ["The file defines Foo."]])
    a, events = run_ask(cfg, bus, transcript, client, FakeRunner())
    deltas = "".join(e.delta for e in events if isinstance(e, AssistDeltaEvent))
    assert deltas == "The file defines Foo."
    assert len(client.calls) == 2


# --------------------------------------------------------------------------------------------
# a call after prose on the same line, a call with no closing tag
# --------------------------------------------------------------------------------------------

SECOND_CALL = '<tool>{"name": "read_file", "args": {"path": "b.py"}}'


def raw_deltas(events: list[Any]) -> str:
    return "".join(e.delta for e in events if isinstance(e, AssistDeltaEvent))


def test_a_call_after_prose_on_the_same_line_runs_and_the_prose_stays_visible(cfg, bus, transcript) -> None:
    client = FakeClient([["One moment: " + READ_CALL_FULL], ["The file defines Foo."]])
    runner = FakeRunner()
    a, events = run_ask(cfg, bus, transcript, client, runner)
    first, second = prompts_of(client)
    assert len(runner.runs) == 1 and runner.runs[0][0].args == {"path": "a.py"}
    assert second == first + "One moment: " + READ_CALL_FULL + turn(READ_BLOCK)
    deltas = raw_deltas(events)
    assert "<tool>" not in deltas and '"name"' not in deltas           # the raw JSON is never published
    assert done_of(events).text == "One moment: The file defines Foo."
    assert a.answers[-1][2] == "One moment: The file defines Foo."


def test_a_call_after_prose_needs_no_closing_tag(cfg, bus, transcript) -> None:
    client = FakeClient([["Checking now. " + READ_CALL], ["The file defines Foo."]])
    a, events = run_ask(cfg, bus, transcript, client, FakeRunner())
    assert len(client.calls) == 2 and "<tool>" not in raw_deltas(events)
    assert done_of(events).text == "Checking now. The file defines Foo."


def test_a_call_after_prose_on_a_later_hop_runs_too(cfg, bus, transcript) -> None:
    client = FakeClient([[READ_CALL], ["Now the second one. " + SECOND_CALL], ["Done."]])
    runner = FakeRunner(calls=4)
    a, events = run_ask(cfg, bus, transcript, client, runner)
    assert [c.args["path"] for c, _b, _c in runner.runs] == ["a.py", "b.py"]
    assert len(client.calls) == 3 and "<tool>" not in raw_deltas(events)
    assert done_of(events).text == "Now the second one. Done."


def test_a_call_after_a_bullet_and_prose_runs(cfg, bus, transcript) -> None:
    client = FakeClient([["- Reading a.py: " + READ_CALL], ["It defines Foo."]])
    runner = FakeRunner()
    a, events = run_ask(cfg, bus, transcript, client, runner)
    assert len(runner.runs) == 1 and "<tool>" not in raw_deltas(events)
    assert done_of(events).text == "- Reading a.py: It defines Foo."


@pytest.mark.parametrize("thinking", ["off", "on"])
def test_a_mid_line_call_with_the_pieces_split_character_by_character(cfg, bus, transcript, thinking: str) -> None:
    cfg.llm.thinking = thinking
    client = FakeClient([list("Checking now. " + READ_CALL_FULL), ["The file defines Foo."]])
    runner = FakeRunner()
    a, events = run_ask(cfg, bus, transcript, client, runner)
    assert len(runner.runs) == 1 and len(client.calls) == 2
    deltas = raw_deltas(events)
    assert "<" not in deltas and "{" not in deltas                        # no partial marker, no JSON
    assert deltas == "Checking now. The file defines Foo."
    assert done_of(events).text == "Checking now. The file defines Foo."


def test_the_first_call_on_a_line_runs_and_a_second_one_after_it_does_not(cfg, bus, transcript) -> None:
    line = "First " + READ_CALL_FULL + " and then " + SECOND_CALL + "</tool>"
    client = FakeClient([[line], ["Done."]])
    runner = FakeRunner()
    a, events = run_ask(cfg, bus, transcript, client, runner)
    assert [c.args["path"] for c, _b, _c in runner.runs] == ["a.py"]
    assert "b.py" not in client.calls[1]["prompt"].split("<|im_start|>assistant\n", 1)[1]


def test_prose_before_an_unfinished_mid_line_call_waits_and_the_stream_ending_makes_it_text(cfg, bus, transcript) -> None:
    text = 'The shape is <tool>{"name": "read_file", "args": {"path": '          # cut off for good
    client = FakeClient([[text]])
    runner = FakeRunner()
    a, events = run_ask(cfg, bus, transcript, client, runner)
    assert not runner.runs and len(client.calls) == 1
    assert done_of(events).text == text


def test_a_call_with_no_closing_tag_is_cut_shortly_after_its_json(cfg, bus, transcript) -> None:
    # it forgot </tool> and wrote on: the loop gives the tag _CLOSE_GRACE_CHARS (80) characters to
    # arrive, then abandons the stream (the one client-side cut left, 2026-10-08) -- 32-char pieces:
    # the third one takes the tail past 80
    pieces = list(READ_CALL) + ["The file defines Bar (made up). "] * 300
    client = FakeClient([pieces, ["The file defines Foo."]])
    runner = FakeRunner()
    a, events = run_ask(cfg, bus, transcript, client, runner)
    assert len(runner.runs) == 1 and runner.runs[0][0].args == {"path": "a.py"}
    done = done_of(events)
    assert done.tokens == len(READ_CALL) + 3 + 1
    assert "made up" not in client.calls[1]["prompt"] and done.text == "The file defines Foo."


def test_a_forced_call_with_no_closing_tag_is_cut_at_its_json_too(cfg, bus, transcript) -> None:
    forced_call = '{"name": "read_file", "args": {"path": "config.toml"}}'
    client = FakeClient([
        ["Let me check the config file."], list(forced_call) + ["It sets the port. "] * 200, ["It sets the port to 8080."],
    ])
    runner = FakeRunner()
    a, events = run_ask(cfg, bus, transcript, client, runner)
    assert runner.runs[0][0].args == {"path": "config.toml"}
    assert done_of(events).tokens == 1 + len(forced_call) + 5 + 1        # 18-char pieces: five pass the 80-char grace
    assert done_of(events).text == "It sets the port to 8080."


# --------------------------------------------------------------------------------------------
# intent without a call, blank answers
# --------------------------------------------------------------------------------------------


def test_narration_without_a_call_is_re_streamed_once_with_the_call_opened(cfg, bus, transcript) -> None:
    client = FakeClient([
        ["Let me check the config file."],
        ['{"name": "read_file", "args": {"path": "config.toml"}}'],
        ["It sets the port to 8080."],
    ])
    runner = FakeRunner()
    a, events = run_ask(cfg, bus, transcript, client, runner)
    first, second, third = prompts_of(client)
    assert second == first + "Let me check the config file.\n<tool>"
    call_text = '{"name": "read_file", "args": {"path": "config.toml"}}</tool>'
    assert third == second + call_text + turn(prompts.tool_result_block("read_file", FILE_TEXT))
    assert runner.runs[0][0].args == {"path": "config.toml"}
    assert client.calls[1]["stop"] == CALL_STOP
    # the narration is dropped from the answer (history and Done), whatever was streamed meanwhile
    assert done_of(events).text == "It sets the port to 8080."
    assert a.answers[-1][2] == "It sets the port to 8080."


def test_the_forced_retry_happens_once_per_answer(cfg, bus, transcript) -> None:
    client = FakeClient([
        ["I will look at the file."],
        ["I'm not sure what to read."],          # the forced pass ignores the opened call: malformed (free)
        ["Let me think again."],                 # a normal pass: narration again, but no second forcing
    ])
    a, events = run_ask(cfg, bus, transcript, client, FakeRunner())
    assert len(client.calls) == 3
    assert not tool_events(events)
    assert done_of(events).text == "Let me think again."


def test_a_forced_call_the_model_declines_keeps_the_narration(cfg, bus, transcript) -> None:
    client = FakeClient([["I need to check this file."], []])
    a, events = run_ask(cfg, bus, transcript, client, FakeRunner())
    assert len(client.calls) == 2
    assert done_of(events).text == "I need to check this file."


ANNOUNCEMENTS = [
    "Okay, let me check a.py.", "Let me run git status.", "Okay, let me check the config file.",
    "Sure, I'll look at the file.", "I'll start by reading a.py.", "Alright. I will read the file now.",
    "Checking the file.", "Reading a.py now.", "Looking at a.py now.", "Let me take a look at the file.",
    "Let me check",                                           # unfinished: no full stop at its end
]
REAL_ANSWERS = [
    "The answer is 42 and I am sure.",                                                     # no intent words
    "Let me know if you want more detail on that one. " + "x" * 260,                       # too long to be a bare announcement
    "Looking at the transcript, Atlas ships Friday.", "We're looking at Friday for the release.",
    "I'll follow up after the call.", "I will be presenting Atlas to the platform team.",
    "- Reading the spec, the limit is 5 MB.", "Let me know if you need more detail.", "I need to leave at 5.",
    "Reading the spec: the limit is 5 MB.", "Checking the transcript: Atlas ships Friday.",
    "Reading the spec. The limit is 5 MB.", "Opening the call at 5 is fine.", "I'll see you Friday.",
    "Let me get back to you on that.",
    "- Looking at the transcript, the Atlas project goes to the platform team today.\n",
    "Let me check that.", "I need to check this.",           # finished and naming nothing: as likely an answer
]


@pytest.mark.parametrize("answer", REAL_ANSWERS)
def test_no_forced_retry_for_a_real_answer(cfg, bus, transcript, answer: str) -> None:
    client = FakeClient([[answer]])
    a, events = run_ask(cfg, bus, transcript, client, FakeRunner())
    assert len(client.calls) == 1 and done_of(events).text == answer
    assert not tool_events(events)


@pytest.mark.parametrize("narration", ANNOUNCEMENTS)
def test_an_announcement_of_a_lookup_is_re_streamed_once_with_the_call_opened(
    cfg, bus, transcript, narration: str
) -> None:
    client = FakeClient([[narration], ['{"name": "read_file", "args": {"path": "a.py"}}'], ["It defines Foo."]])
    runner = FakeRunner()
    a, events = run_ask(cfg, bus, transcript, client, runner)
    assert len(client.calls) == 3 and len(runner.runs) == 1
    assert client.calls[1]["prompt"].endswith(narration + "\n<tool>")
    assert done_of(events).text == "It defines Foo."


def test_the_gate_on_its_own() -> None:
    from exocortex.llm.assistant import _announces_lookup
    assert all(_announces_lookup(t) for t in ANNOUNCEMENTS)
    assert not any(_announces_lookup(t) for t in REAL_ANSWERS)
    assert not _announces_lookup("") and not _announces_lookup("   \n")


def test_no_forced_retry_once_a_tool_has_run(cfg, bus, transcript) -> None:
    client = FakeClient([[READ_CALL], ["Let me summarise: it defines Foo."]])
    a, events = run_ask(cfg, bus, transcript, client, FakeRunner())
    assert len(client.calls) == 2
    assert done_of(events).text == "Let me summarise: it defines Foo."


def test_a_blank_answer_after_a_tool_is_asked_for_once_more(cfg, bus, transcript) -> None:
    client = FakeClient([[READ_CALL], [], ["The file defines Foo."]])
    a, events = run_ask(cfg, bus, transcript, client, FakeRunner())
    first, second, third = client.calls
    # the retry repeats the continuation with the nudge inside the last result block (a user turn),
    # and no further call may be made
    assert second["prompt"] == first["prompt"] + READ_CALL_FULL + turn(READ_BLOCK)
    assert third["prompt"] == first["prompt"] + READ_CALL_FULL + turn(NUDGED_BLOCK)
    assert third["stop"] == FINAL_STOP
    assert done_of(events).text == "The file defines Foo."


def test_a_blank_answer_twice_is_an_error(cfg, bus, transcript) -> None:
    client = FakeClient([[READ_CALL], [], ["  \n"]])
    a, events = run_ask(cfg, bus, transcript, client, FakeRunner())
    error = events[-1]
    assert isinstance(error, AssistErrorEvent)
    assert error.message == "the model made 1 tool call(s) but wrote no answer"
    assert not any(isinstance(e, AssistDoneEvent) for e in events)
    assert a.answers == []


LEAD = "Let me check the file.\n"


def test_a_lead_in_before_the_call_does_not_hide_a_blank_answer(cfg, bus, transcript) -> None:
    client = FakeClient([[LEAD + READ_CALL], [], ["  \n"]])
    a, events = run_ask(cfg, bus, transcript, client, FakeRunner())
    error = events[-1]
    assert isinstance(error, AssistErrorEvent)
    assert error.message == "the model made 1 tool call(s) but wrote no answer"
    assert len(client.calls) == 3 and a.answers == []
    assert not any(isinstance(e, AssistDoneEvent) for e in events)


def test_a_lead_in_then_a_blank_pass_then_an_answer_gets_the_extra_pass(cfg, bus, transcript) -> None:
    client = FakeClient([[LEAD + READ_CALL], [], ["The file defines Foo."]])
    a, events = run_ask(cfg, bus, transcript, client, FakeRunner())
    assert len(client.calls) == 3
    assert client.calls[2]["prompt"] == client.calls[0]["prompt"] + LEAD + READ_CALL_FULL + turn(NUDGED_BLOCK)
    assert done_of(events).text == "Let me check the file.\nThe file defines Foo."
    assert a.answers[-1][2] == "Let me check the file.\nThe file defines Foo."


def test_a_lead_in_before_the_call_and_a_blank_chat_pass_is_asked_once_more(cfg, bus, transcript) -> None:
    cfg.llm.thinking = "on"
    client = FakeClient([[LEAD + READ_CALL], [], ["The file defines Foo."]])
    a, events = run_ask(cfg, bus, transcript, client, FakeRunner())
    assert len(client.calls) == 3
    assert client.calls[2]["messages"][-1]["content"].endswith("(write the answer now)\n</result>")
    assert done_of(events).text == "Let me check the file.\nThe file defines Foo."


def test_two_calls_with_a_lead_in_on_the_first_only_still_get_the_blank_retry(cfg, bus, transcript) -> None:
    client = FakeClient([[LEAD + READ_CALL], [SECOND_CALL], ["   "], ["Both files define Foo."]])
    a, events = run_ask(cfg, bus, transcript, client, FakeRunner(calls=4))
    assert len(client.calls) == 4
    assert done_of(events).text == "Let me check the file.\nBoth files define Foo."


def test_a_final_pass_that_only_writes_another_call_after_a_lead_in_is_an_error(cfg, bus, transcript) -> None:
    # one call allowed: the answer pass has no calls left, the model writes a call anyway (cut at it)
    client = FakeClient([[LEAD + READ_CALL], [READ_CALL], [READ_CALL]])
    a, events = run_ask(cfg, bus, transcript, client, FakeRunner(calls=1))
    assert isinstance(events[-1], AssistErrorEvent) and len(client.calls) == 3
    assert not any(isinstance(e, AssistDoneEvent) for e in events)


def test_a_blank_answer_without_any_tool_is_not_an_error(cfg, bus, transcript) -> None:
    client = FakeClient([[]])
    a, events = run_ask(cfg, bus, transcript, client, FakeRunner())
    assert isinstance(events[-1], AssistDoneEvent) and len(client.calls) == 1


# --------------------------------------------------------------------------------------------
# cancel
# --------------------------------------------------------------------------------------------


def test_cancel_during_a_tool_runs_no_further_tool(cfg, bus, transcript) -> None:
    holder: dict[str, Assistant] = {}

    def cancel_now(_call: ToolCall, _budget: ToolBudget) -> None:
        holder["a"].cancel()

    client = FakeClient([[READ_CALL], [READ_CALL], ["never reached"]])
    runner = FakeRunner(calls=4, on_run=cancel_now)
    a = make_assistant(cfg, bus, client, transcript, runner)
    holder["a"] = a
    sub = bus.subscribe()
    a.start()
    try:
        events = ask(a, sub)
    finally:
        a.stop()
    assert len(runner.runs) == 1 and len(client.calls) == 1
    done = done_of(events)
    assert done.cancelled is True
    assert a.answers == []
    assert [e.state for e in tool_events(events)] == ["running", "done"]


def test_the_runner_gets_the_jobs_cancel_event(cfg, bus, transcript) -> None:
    client = FakeClient([[READ_CALL], ["ok"]])
    runner = FakeRunner()
    a, events = run_ask(cfg, bus, transcript, client, runner)
    (_call, _budget, cancel), = runner.runs
    assert cancel is not None and not cancel.is_set()


# --------------------------------------------------------------------------------------------
# live kinds, other kinds, no tools
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["auto", "manual"])
def test_a_live_job_gets_the_live_budget_and_read_tools_only(cfg, bus, transcript, kind: str) -> None:
    """An answer to the OTHER speaker is untrusted text: read tools, the live limits, and a forged
    `run_command` comes back denied without a second tool running for it."""
    transcript.add(make_segment(2, text="What does a.py define?", source=Source.SYSTEM, is_question=True))
    cmd = '<tool>{"name": "run_command", "args": {"command": "dir"}}'
    client = FakeClient([[cmd], ["It defines Foo."]])
    runner = FakeRunner(calls=4, live_calls=2)
    a = make_assistant(cfg, bus, client, transcript, runner)
    sub = bus.subscribe()
    a.start()
    try:
        if kind == "auto":
            a._submit_assist("auto", "What does a.py define?")       # what the coalesce timer submits
        else:
            a.answer_last()
        events = drain_until(sub, finished)
    finally:
        a.stop()
    assert runner.budgets[0].kind == kind and runner.budgets[0].calls_left == 1       # live: 2, one spent
    system = system_of(client.calls[0]["prompt"])
    assert "- read_file(" in system and "run_command" not in system
    assert "At most 2 calls per answer." in system
    (call, budget, _cancel), = runner.runs
    assert call.name == "run_command"
    running, outcome = tool_events(events)
    assert outcome.state == "denied" and outcome.name == "run_command"
    assert len(runner.runs) == 1 and len(client.calls) == 2
    assert done_of(events).text == "It defines Foo."


def test_a_solve_job_gets_the_tools_and_the_map(cfg, bus, transcript) -> None:
    client = FakeClient([[READ_CALL], ["def f(): pass"]])
    runner = FakeRunner(map_text="src/\n  a.py -- thing")
    a = make_assistant(cfg, bus, client, transcript, runner)
    a.set_screen_provider(lambda: "Write f.")
    sub = bus.subscribe()
    a.start()
    try:
        a.solve("")
        events = drain_until(sub, finished)
    finally:
        a.stop()
    first = client.calls[0]["prompt"]
    assert "<tools>\n" in system_of(first) and system_of(first).endswith("</tools>")
    assert "<map>\nsrc/" in user_of(first)
    assert runner.budgets[0].kind == "solve"
    assert done_of(events).text == "def f(): pass"


def test_no_runner_means_the_request_the_assistant_always_sent(cfg, bus, transcript) -> None:
    client = FakeClient([["Plain answer."]])
    a, events = run_ask(cfg, bus, transcript, client, None)
    (call,) = client.calls
    assert call["stop"] == prompts.STOP and call["extra_body"] is None
    assert "<tools>" not in call["prompt"] and "<map>" not in call["prompt"]
    assert call["prompt"].endswith(prompts.NO_THINK_PREFIX)
    assert done_of(events).text == "Plain answer."


def test_a_runner_that_does_not_want_the_kind_changes_nothing(cfg, bus, transcript) -> None:
    plain_client = FakeClient([["Plain answer."]])
    run_ask(cfg, bus, transcript, plain_client, None)
    runner = FakeRunner(wants=False, map_text="src/")
    client = FakeClient([["Plain answer."]])
    run_ask(cfg, bus, Transcript(), client, runner)
    # (the transcripts differ in object only: both hold the same single segment)
    assert runner.wants_calls == ["ask"] and not runner.budgets
    assert system_of(client.calls[0]["prompt"]) == system_of(plain_client.calls[0]["prompt"])
    assert client.calls[0]["stop"] == prompts.STOP


def test_a_runner_with_no_calls_allowed_changes_nothing(cfg, bus, transcript) -> None:
    client = FakeClient([["Plain answer."]])
    run_ask(cfg, bus, transcript, client, FakeRunner(calls=0))
    assert "<tools>" not in client.calls[0]["prompt"] and client.calls[0]["stop"] == prompts.STOP


def test_a_runner_that_raises_means_no_tools_not_a_failed_job(cfg, bus, transcript) -> None:
    class Broken(FakeRunner):
        def specs(self, kind: str):
            raise RuntimeError("boom")

    client = FakeClient([["Plain answer."]])
    a, events = run_ask(cfg, bus, transcript, client, Broken())
    assert done_of(events).text == "Plain answer." and "<tools>" not in client.calls[0]["prompt"]


def test_complete_and_explain_never_get_tools(cfg, bus, transcript) -> None:
    runner = FakeRunner()
    client = FakeClient([[" today"]])
    a = make_assistant(cfg, bus, client, transcript, runner)
    sub = bus.subscribe()
    a.start()
    try:
        job_id = a.complete("The codename is", "", "Chat message")
        drain_until(sub, lambda e: isinstance(e, AssistDoneEvent) and e.job_id == job_id)
    finally:
        a.stop()
    assert runner.wants_calls == [] and not runner.budgets
    assert "<tools>" not in client.calls[0]["prompt"]
    assert client.calls[0]["prompt"].endswith(prompts.NO_THINK_PREFIX + "The codename is")


def test_explain_never_gets_tools(cfg, bus, transcript) -> None:
    runner = FakeRunner()                                  # wants=True for every kind: only the Assistant's own guard is left
    client = FakeClient([["It loops over the list."]])
    a = make_assistant(cfg, bus, client, transcript, runner)
    sub = bus.subscribe()
    a.start()
    try:
        a.explain_now()                                    # no plan / screen provider needed
        drain_until(sub, lambda e: isinstance(e, AssistDoneEvent))
    finally:
        a.stop()
    assert runner.wants_calls == [] and not runner.budgets
    assert "<tools>" not in client.calls[0]["prompt"]


# --------------------------------------------------------------------------------------------
# the loop with the REAL ToolRunner: free repeats and the hard_cap net
# --------------------------------------------------------------------------------------------


def real_runner(cfg: Config, tmp_path: Any, *, calls: int) -> tuple[ToolRunner, list[ToolBudget]]:
    """A real ToolRunner over `tmp_path/proj` (holding a.py), plus the budgets it hands out."""
    root = tmp_path / "proj"
    root.mkdir()
    (root / "a.py").write_text(FILE_TEXT, encoding="utf-8")
    cfg.paths.models_dir = str(tmp_path / "models")
    cfg.paths.sessions_dir = str(tmp_path / "sessions")
    cfg.capabilities.read_roots = [str(root)]
    cfg.capabilities.max_tool_calls = calls
    runner = ToolRunner(cfg, session_folder=lambda: tmp_path / "sess")
    budgets: list[ToolBudget] = []
    real_budget = runner.budget

    def spy(kind: str) -> ToolBudget:
        b = real_budget(kind)
        budgets.append(b)
        return b

    runner.budget = spy                     # type: ignore[method-assign]
    return runner, budgets


def test_a_model_that_repeats_one_call_forever_is_stopped_by_the_loops_net(cfg, bus, transcript, tmp_path) -> None:
    runner, budgets = real_runner(cfg, tmp_path, calls=3)         # hard_cap = 2 * 3 + 2 = 8
    client = FakeClient([[READ_CALL]] * 20)
    a, events = run_ask(cfg, bus, transcript, client, runner)     # type: ignore[arg-type]
    # 8 call passes, then the answer pass (which writes a call again, cut at it) and its blank retry
    assert len(client.calls) == 10
    done = [e for e in tool_events(events) if e.state == "done"]
    assert len(done) == 8 and sum(e.detail == "same call as before" for e in done) == 7
    assert budgets[0].calls_left == 2                             # only the first call was charged: the repeats are free
    last = events[-1]
    assert isinstance(last, AssistErrorEvent)
    assert last.message == "the model made 8 tool call(s) but wrote no answer"
    assert not any(isinstance(e, AssistDoneEvent) for e in events) and a.answers == []
    assert "(no calls left: write the answer now)" in client.calls[8]["prompt"]       # the 8th call hit the net
    assert "(no calls left" not in client.calls[7]["prompt"]
    assert "(write the answer now)\n</result>" in client.calls[9]["prompt"]      # the retry's nudge
    assert "(write the answer now)" not in client.calls[8]["prompt"]
    assert all(c["stop"] == FINAL_STOP for c in client.calls[-2:])


def test_free_repeated_calls_below_the_net_still_end_in_an_answer(cfg, bus, transcript, tmp_path) -> None:
    runner, budgets = real_runner(cfg, tmp_path, calls=3)         # hard_cap = 8
    client = FakeClient([[READ_CALL]] * 7 + [["The file defines Foo."]])
    a, events = run_ask(cfg, bus, transcript, client, runner)     # type: ignore[arg-type]
    assert len(client.calls) == 8 and budgets[0].calls_left == 2
    assert len([e for e in tool_events(events) if e.state == "done"]) == 7
    assert "(no calls left" not in client.calls[7]["prompt"]       # the net did not trigger early
    assert done_of(events).text == "The file defines Foo."


# --------------------------------------------------------------------------------------------
# the chat path (thinking on)
# --------------------------------------------------------------------------------------------


def test_thinking_on_goes_through_stream_chat_with_the_stop_list_and_appended_messages(
    cfg, bus, transcript
) -> None:
    cfg.llm.thinking = "on"
    client = FakeClient([[READ_CALL], ["The file defines Foo."]])
    a, events = run_ask(cfg, bus, transcript, client, FakeRunner())
    first, second = client.calls
    assert first["kind"] == second["kind"] == "chat"
    assert first["stop"] == second["stop"] == CALL_STOP
    assert [m["role"] for m in first["messages"]] == ["system", "user"]
    assert [m["role"] for m in second["messages"]] == ["system", "user", "assistant", "user"]
    assert second["messages"][:2] == first["messages"]
    assert second["messages"][2]["content"] == READ_CALL_FULL
    assert second["messages"][3]["content"] == READ_BLOCK.strip("\n")
    assert "<tools>" in first["messages"][0]["content"]
    assert done_of(events).text == "The file defines Foo."


def test_thinking_on_exhausted_calls_use_the_answer_stop_list(cfg, bus, transcript) -> None:
    cfg.llm.thinking = "on"
    client = FakeClient([[READ_CALL], ["Done."]])
    run_ask(cfg, bus, transcript, client, FakeRunner(calls=1))
    assert client.calls[1]["stop"] == FINAL_STOP
    assert client.calls[1]["messages"][3]["content"].endswith("(no calls left: write the answer now)\n</result>")


def test_thinking_on_the_forced_retry_asks_in_words(cfg, bus, transcript) -> None:
    cfg.llm.thinking = "on"
    client = FakeClient([["Let me check the config."], ['{"name": "read_file", "args": {"path": "c"}}'], ["Port 80."]])
    a, events = run_ask(cfg, bus, transcript, client, FakeRunner())
    second = client.calls[1]["messages"]
    assert [m["role"] for m in second] == ["system", "user", "assistant", "user"]
    assert second[2]["content"] == "Let me check the config."
    assert "make the tool call now" in second[3]["content"]
    assert done_of(events).text == "Port 80."


def test_thinking_on_the_blank_retry_keeps_the_roles_alternating(cfg, bus, transcript) -> None:
    cfg.llm.thinking = "on"
    client = FakeClient([[READ_CALL], [], ["Foo."]])
    a, events = run_ask(cfg, bus, transcript, client, FakeRunner())
    third = client.calls[2]["messages"]
    assert [m["role"] for m in third] == ["system", "user", "assistant", "user"]
    assert third[3]["content"].endswith("(write the answer now)\n</result>")
    assert done_of(events).text == "Foo."


# --------------------------------------------------------------------------------------------
# overflow
# --------------------------------------------------------------------------------------------


def test_an_overflow_on_a_continuation_drops_the_oldest_result_and_retries_that_hop(cfg, bus, transcript) -> None:
    client = FakeClient([[READ_CALL], ["The file defines Foo."]])
    client.fail_with = [None, OVERFLOW, None]
    runner = FakeRunner()
    a, events = run_ask(cfg, bus, transcript, client, runner)
    assert len(client.calls) == 3 and len(runner.runs) == 1
    assert FILE_TEXT in client.calls[1]["prompt"]
    assert turn(prompts.tool_result_block("read_file", "(result dropped to fit the context)")) in client.calls[2]["prompt"]
    assert FILE_TEXT not in client.calls[2]["prompt"].split("<|im_start|>assistant\n", 1)[1]
    assert done_of(events).text == "The file defines Foo."


def test_a_second_overflow_on_the_hop_is_an_error_and_never_re_runs_the_tool(cfg, bus, transcript) -> None:
    client = FakeClient([[READ_CALL], ["unused"]])
    client.fail_with = [None, OVERFLOW, OVERFLOW]
    runner = FakeRunner()
    a, events = run_ask(cfg, bus, transcript, client, runner)
    assert isinstance(events[-1], AssistErrorEvent) and "exceeds" in events[-1].message
    assert len(runner.runs) == 1 and len(client.calls) == 3          # no outer rebuild-and-retry after a tool ran


def test_an_overflow_before_any_tool_still_gets_the_outer_retry_with_the_same_budget(cfg, bus, transcript) -> None:
    client = FakeClient([["Plain answer."]])
    client.fail_with = [OVERFLOW, None]
    runner = FakeRunner(calls=3)
    a, events = run_ask(cfg, bus, transcript, client, runner)
    assert len(client.calls) == 2 and not runner.runs
    assert len(runner.budgets) == 1                                  # taken once per job
    assert system_of(client.calls[0]["prompt"]) == system_of(client.calls[1]["prompt"])
    assert done_of(events).text == "Plain answer."


def test_the_oldest_result_is_the_one_dropped_first(cfg, bus, transcript) -> None:
    second_call = '<tool>{"name": "read_file", "args": {"path": "b.py"}}'
    results = {"read_file": [ToolResult("OLDEST BODY", "done", "", "a.py"), ToolResult("NEWER BODY", "done", "", "b.py")]}
    client = FakeClient([[READ_CALL], [second_call], ["Done."]])
    client.fail_with = [None, None, OVERFLOW, None]
    a, events = run_ask(cfg, bus, transcript, client, FakeRunner(calls=4, results=results))
    retried = client.calls[3]["prompt"]
    assert "OLDEST BODY" not in retried and "NEWER BODY" in retried
    assert "(result dropped to fit the context)" in retried
    assert done_of(events).text == "Done."


# --------------------------------------------------------------------------------------------
# prompt stability, usage, stats, max tokens
# --------------------------------------------------------------------------------------------


def test_the_system_prompt_with_tools_is_byte_identical_across_two_jobs(cfg, bus, transcript) -> None:
    client = FakeClient([["One."], ["Two."]])
    runner = FakeRunner(calls=3, map_text="src/\n  a.py")
    a = make_assistant(cfg, bus, client, transcript, runner)
    sub = bus.subscribe()
    a.start()
    try:
        ask(a, sub, "First question?")
        transcript.add(make_segment(2, text="More talk happened in the meantime."))
        ask(a, sub, "Second question?")
    finally:
        a.stop()
    s1, s2 = (system_of(c["prompt"]) for c in client.calls)
    assert s1 == s2 and s1.count(prompts._TOOL_RULE) == 1
    assert len(runner.budgets) == 2                                  # one budget per job


def test_usage_calibration_pairs_the_last_requests_chars_with_its_prompt_tokens(cfg, bus, transcript) -> None:
    client = FakeClient([[READ_CALL], ["The file defines Foo."]], reports=[(111, 0), (222, 100)])
    a = make_assistant(cfg, bus, client, transcript, FakeRunner())
    observed: list[tuple[int, int | None]] = []
    real = a.context.observe_prompt
    a.context.observe_prompt = lambda chars, tokens: (observed.append((chars, tokens)), real(chars, tokens))[1]
    sub = bus.subscribe()
    a.start()
    try:
        events = ask(a, sub)
    finally:
        a.stop()
    assert observed == [(len(client.calls[-1]["prompt"]), 222)]
    usage = [e for e in events if type(e).__name__ == "ContextUsageEvent"]
    assert usage[-1].prompt_tokens == 222 and usage[-1].cached_tokens == 100


def test_chat_usage_counts_the_characters_of_every_message(cfg, bus, transcript) -> None:
    cfg.llm.thinking = "on"
    client = FakeClient([[READ_CALL], ["Foo."]], reports=[(111, 0), (222, 0)])
    a = make_assistant(cfg, bus, client, transcript, FakeRunner())
    observed: list[tuple[int, int | None]] = []
    real = a.context.observe_prompt
    a.context.observe_prompt = lambda chars, tokens: (observed.append((chars, tokens)), real(chars, tokens))[1]
    sub = bus.subscribe()
    a.start()
    try:
        ask(a, sub)
    finally:
        a.stop()
    assert observed == [(sum(len(m["content"]) for m in client.calls[-1]["messages"]), 222)]


def test_the_stats_of_the_hops_are_merged(cfg, bus, transcript) -> None:
    client = FakeClient([[READ_CALL], ["The file ", "defines Foo."]])
    a, events = run_ask(cfg, bus, transcript, client, FakeRunner())
    done = done_of(events)
    assert done.ttft_s == pytest.approx(0.05)              # the first stream's
    assert done.tokens == 1 + 2                             # summed over both streams
    assert done.tok_per_s > 0


def test_max_tokens_per_hop_for_an_ask_with_edit_tools(cfg, bus, transcript) -> None:
    cfg.llm.ask_max_tokens = 300
    client = FakeClient([[READ_CALL], ["Done."]])
    run_ask(cfg, bus, transcript, client, FakeRunner(calls=1, edit=True))
    assert [c["max_tokens"] for c in client.calls] == [1200, 300]         # the call hop roomy, the answer hop not


def test_max_tokens_stay_the_answer_budget_without_edit_tools(cfg, bus, transcript) -> None:
    cfg.llm.ask_max_tokens = 300
    client = FakeClient([[READ_CALL], ["Done."]])
    run_ask(cfg, bus, transcript, client, FakeRunner(calls=4))
    assert [c["max_tokens"] for c in client.calls] == [300, 300]


def test_max_tokens_are_not_lowered_below_a_bigger_answer_budget(cfg, bus, transcript) -> None:
    cfg.llm.ask_max_tokens = 2000
    client = FakeClient([[READ_CALL], ["Done."]])
    run_ask(cfg, bus, transcript, client, FakeRunner(calls=4, edit=True))
    assert [c["max_tokens"] for c in client.calls] == [2000, 2000]


# --------------------------------------------------------------------------------------------
# the splitter on its own
# --------------------------------------------------------------------------------------------


def split(chunks: list[str]) -> tuple[list[str], _ToolSplitter]:
    out: list[str] = []
    s = _ToolSplitter(out.append)
    for chunk in chunks:
        s.feed(chunk)
    s.finish()
    return out, s


def test_splitter_plain_text_passes_through_in_one_delta_at_the_end() -> None:
    out, s = split(["Hello ", "there, ", "friend."])
    assert "".join(out) == "Hello there, friend." and s.visible == "Hello there, friend." and s.held is None


def test_splitter_buffers_until_a_newline_or_120_characters() -> None:
    out: list[str] = []
    s = _ToolSplitter(out.append)
    s.feed("short line")
    assert out == []
    s.feed(" and a newline\nrest")
    assert out == ["short line and a newline\n"]               # published up to the newline; "rest" waits
    s.feed("x" * 130)
    assert len(out) == 2 and out[1] == "rest" + "x" * 130
    s.feed("tail")
    assert len(out) == 2
    s.finish()
    assert out[-1] == "tail"


def test_splitter_holds_a_call_at_the_very_start_without_leaking_a_prefix() -> None:
    chunks = list('<tool>{"name": "read_file", "args": {}}')
    out: list[str] = []
    s = _ToolSplitter(out.append)
    for ch in chunks:
        s.feed(ch)
        assert out == [] and s.visible == ""
    s.finish()
    assert out == [] and s.held == '<tool>{"name": "read_file", "args": {}}'


@pytest.mark.parametrize("lead", ["", "  ", "- ", "* ", "1. ", "> ", "  - ", "12) "])
def test_splitter_holds_a_call_after_a_lead_marker(lead: str) -> None:
    out, s = split(["Looking.\n", lead, "<tool>{}"])
    assert "".join(out) == "Looking.\n"
    assert s.held == "<tool>{}"


def test_splitter_a_fence_line_right_before_a_call_is_swallowed() -> None:
    out, s = split(["```json\n", "<tool>{}"])
    assert out == [] and s.visible == "" and s.held == "<tool>{}"
    out, s = split(["Text.\n```\n<tool>{}"])
    assert "".join(out) == "Text.\n" and s.held == "<tool>{}"


def test_splitter_a_fence_line_before_ordinary_text_stays() -> None:
    out, s = split(["Here:\n```python\n", "print(1)\n", "```\n", "Done."])
    assert "".join(out) == "Here:\n```python\nprint(1)\n```\nDone." and s.held is None


def test_splitter_a_fence_line_at_the_end_of_the_stream_stays() -> None:
    out, s = split(["Code:\n```"])
    assert "".join(out) == "Code:\n```" and s.held is None


def test_splitter_a_marker_in_the_middle_of_a_line_is_text() -> None:
    out, s = split(["Call it like <tool>{}</tool> today.\nNext."])
    assert "".join(out) == "Call it like <tool>{}</tool> today.\nNext." and s.held is None


def test_splitter_a_call_in_the_middle_of_a_line_is_held_and_the_prose_before_it_kept() -> None:
    out, s = split(["Checking now. ", READ_CALL])
    assert "".join(out) == "Checking now. " and s.visible == "Checking now. " and s.held == READ_CALL


def test_splitter_a_mid_line_call_split_anywhere_never_leaks() -> None:
    out, s = split(list("Checking now. " + READ_CALL_FULL + " more"))
    assert "".join(out) == "Checking now. " and s.held == READ_CALL_FULL + " more"


def test_splitter_the_first_mid_line_call_wins() -> None:
    out, s = split(["A " + READ_CALL_FULL + " B " + SECOND_CALL + "</tool>"])
    assert s.visible == "A " and s.held.startswith(READ_CALL_FULL)


@pytest.mark.parametrize("text", [
    'Use <tool>{"name": "read_file"}</tool> here', "Write <tool>{} now", "The <tool> marker", "x <tool>{'a': 1} y",
    "a </tool> b", 'see <tool>{"name": 1, "args": {}}</tool>', "x <tool>```\nnot json\n``` y",
])
def test_splitter_a_mid_line_marker_that_is_not_a_call_is_text(text: str) -> None:
    for chunks in ([text, "\nnext line"], list(text + "\nnext line")):
        out, s = split(chunks)
        assert "".join(out) == text + "\nnext line" and s.held is None


def test_splitter_an_unfinished_mid_line_call_waits_and_the_end_of_the_stream_decides() -> None:
    out: list[str] = []
    s = _ToolSplitter(out.append)
    s.feed('Hi. <tool>{"name": "read_file", "args": ')
    assert s.held is None and s.visible == "Hi. "                  # waiting: the object is not complete
    s.feed('{"path": "a.py"}}')
    assert s.held == '<tool>{"name": "read_file", "args": {"path": "a.py"}}'
    out2: list[str] = []
    s2 = _ToolSplitter(out2.append)
    s2.feed('Hi. <tool>{"name": "read_file", "args": ')
    s2.finish()                                                    # the stream ended inside the object
    assert s2.held is None and "".join(out2) == 'Hi. <tool>{"name": "read_file", "args": '


def test_splitter_gives_up_waiting_for_a_mid_line_object_after_the_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_ToolSplitter, "MID_CALL_WAIT_CHARS", 40)
    out: list[str] = []
    s = _ToolSplitter(out.append)
    s.feed("Look: <tool>{" + "x" * 60)
    assert s.held is None and s.visible.startswith("Look: <tool>{x")


def test_splitter_a_partial_marker_at_the_end_of_a_prose_chunk_waits_for_the_next_one() -> None:
    out: list[str] = []
    s = _ToolSplitter(out.append)
    s.feed("Hello <to")
    assert s.visible == "Hello "                                    # "<to" is held back: it may become "<tool>"
    s.feed("day is fine")
    s.finish()
    assert s.visible == "Hello <today is fine" and s.held is None


def test_splitter_a_mid_line_marker_before_a_newline_is_judged_by_the_text_after_it() -> None:
    out, s = split(["Sure. <tool>\n", '{"name": "read_file", "args": {"path": "a.py"}}'])
    assert s.visible == "Sure. " and s.held.startswith("<tool>\n{")


def test_splitter_a_marker_after_ordinary_text_on_a_later_line_is_a_call() -> None:
    out, s = split(["One line.\n", "Another.\n<tool>{}</tool>"])
    assert "".join(out) == "One line.\nAnother.\n" and s.held == "<tool>{}</tool>"


def test_splitter_everything_after_the_call_is_held() -> None:
    out, s = split(["<tool>{}", "</tool>", "\nand more text"])
    assert out == [] and s.held == "<tool>{}</tool>\nand more text"


def test_splitter_a_partial_marker_at_the_end_of_the_stream_is_held_not_shown() -> None:
    out, s = split(["Fine.\n<too"])
    assert "".join(out) == "Fine.\n" and s.held == "<too"


def test_splitter_look_alikes_are_ordinary_text() -> None:
    for text in ["<b>bold</b>", "**Note** this", "-1 is the sentinel", "12 apples", "1.5 liters", "<tools> are listed",
                 "`<tool>` is the marker", "- <b>item</b>"]:
        out, s = split([text, "\nmore"])
        assert "".join(out) == text + "\nmore" and s.held is None, text


def test_splitter_a_lead_only_line_at_the_end_is_visible() -> None:
    out, s = split(["Items:\n- "])
    assert "".join(out) == "Items:\n- " and s.held is None


def test_splitter_started_inside_a_call_holds_everything() -> None:
    out: list[str] = []
    s = _ToolSplitter(out.append, held=prompts.TOOL_OPEN)
    s.feed('{"name": "read_file"}')
    s.finish()
    assert out == [] and s.visible == "" and s.held == '<tool>{"name": "read_file"}'


def test_splitter_reports_whether_a_delta_went_out() -> None:
    out: list[str] = []
    s = _ToolSplitter(out.append)
    s.feed("pending")
    assert s.published is False
    s.finish()
    assert s.published is True


# --------------------------------------------------------------------------------------------
# 2026-10-08: the shape the first live session forced (see _raw_prompt)
# --------------------------------------------------------------------------------------------


def test_every_continuation_extends_the_previous_request_plus_what_the_model_wrote(cfg, bus, transcript) -> None:
    """The KV-cache invariant across two calls: request k+1 starts with request k followed by the
    text the model generated in it (the stop string the server stripped is re-appended), so the
    server's slot holds everything but the new turn."""
    client = FakeClient([[READ_CALL], [SECOND_CALL], ["Both define Foo."]])
    a, events = run_ask(cfg, bus, transcript, client, FakeRunner(calls=4))
    p1, p2, p3 = prompts_of(client)
    assert p2.startswith(p1 + READ_CALL) and p3.startswith(p2 + SECOND_CALL)
    second_block = prompts.tool_result_block("read_file", FILE_TEXT)
    assert p3 == p2 + SECOND_CALL + "</tool>" + turn(second_block)
    assert p3.count(f"{prompts.IM_START}assistant\n{prompts.NO_THINK_PREFIX}") == 3     # one turn per pass
    assert p3.count(f"{prompts.IM_START}user\n") == 3                                   # the question + two results
    assert done_of(events).text == "Both define Foo."


def test_a_result_is_never_inside_the_assistant_turn(cfg, bus, transcript) -> None:
    """What the 27B did live: with the result appended inside its own turn it ended the turn at
    once. Every result block now sits in a user turn, right after the turn that made the call."""
    client = FakeClient([[READ_CALL], ["The file defines Foo."]])
    a, events = run_ask(cfg, bus, transcript, client, FakeRunner())
    second = prompts_of(client)[1]
    before, after = second.split('<result name="read_file">', 1)
    assert before.endswith(f"{READ_CALL_FULL}{prompts.IM_END}\n{prompts.IM_START}user\n")
    assert after.startswith(f"\n{FILE_TEXT}\n</result>{prompts.IM_END}\n{prompts.IM_START}assistant\n{prompts.NO_THINK_PREFIX}")
    assert second.endswith(prompts.NO_THINK_PREFIX)

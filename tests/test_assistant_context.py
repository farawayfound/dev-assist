"""Offline tests for the Assistant's context-library integration (2026-10-06).

Covers what `exocortex.llm.assistant` does around `exocortex.context.ContextManager`: injected /
retrieved reference material reaching the prompt, `include_in` gating, the transcript compaction
job (and the `<earlier>` block it feeds), document digests, the two `ContextUsageEvent`s per job,
calibration from the server's prompt-token count, the one-shot overflow retry, the server's context
window being read at health-check time, and the notes-lane dispatch order. A duck-typed FakeClient
(copied from tests/test_assistant.py, extended with a prompt-driven responder, scripted failures
and reported token counts) stands in for the LLM; everything else is real.
"""
from __future__ import annotations

import queue
import threading
import time
from types import SimpleNamespace
from typing import Any, Callable

import pytest

from exocortex.bus import (
    AssistDoneEvent,
    AssistErrorEvent,
    AssistStartedEvent,
    ContextLibraryEvent,
    ContextUsageEvent,
    EventBus,
    Source,
    TranscriptSegment,
)
from exocortex.config import Config
from exocortex.context import ContextManager, ContextStore
from exocortex.llm import prompts
from exocortex.llm.assistant import Assistant, _CompactJob, _DigestJob, _NotesJob
from exocortex.llm.client import ChatDelta, Health, LLMError, ModelInfo, StreamStats
from exocortex.transcript import Transcript

OVERFLOW = LLMError(
    "HTTP 400: request (5000 tokens) exceeds the available context size (4096 tokens), try increasing it"
)


# --------------------------------------------------------------------------------------------
# test doubles / helpers
# --------------------------------------------------------------------------------------------


class FakeClient:
    """Duck-typed LMStudioClient.

    `responder(text) -> list[str]` picks the streamed pieces from the prompt (completion path) or
    the joined messages (chat path); `fail_with` is consumed one entry per stream call (an
    `LLMError` is raised, `None` lets the call through); `report` (prompt_tokens, cached_tokens) is
    what `last_stats` claims after a call; `window` is what `context_window()` returns (an
    exception instance is raised instead). `last_stats` is thread-local like the real client's.
    """

    def __init__(self, *, responder: Callable[[str], list[str]] | None = None, chunk_delay: float = 0.0) -> None:
        self.cfg = SimpleNamespace(model="maple-preview", base_url="http://fake.test", auto_load_model=True)
        self._local = threading.local()
        self._responder = responder or (lambda _text: ["ok"])
        self._chunk_delay = chunk_delay
        self.calls: list[dict[str, Any]] = []
        self.fail_with: list[LLMError | None] = []
        self.report: tuple[int | None, int | None] | None = None
        self.window: Any = None
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
        if isinstance(self.window, Exception):
            raise self.window
        return self.window

    def _run(self, record: dict[str, Any], text: str, cancel: threading.Event | None):
        with self._lock:
            self.calls.append(record)
            failure = self.fail_with.pop(0) if self.fail_with else None
        if failure is not None:
            raise failure
        pieces = self._responder(text)
        count = 0
        for piece in pieces:
            if cancel is not None and cancel.is_set():
                break
            if self._chunk_delay:
                time.sleep(self._chunk_delay)
            count += 1
            yield piece
        prompt_tokens, cached = self.report if self.report is not None else (None, None)
        self.last_stats = StreamStats(
            ttft_s=0.01, tokens=count, elapsed_s=0.02, tok_per_s=float(count) / 0.02, cancelled=False,
            prompt_tokens=prompt_tokens, cached_tokens=cached,
        )

    def stream_completion(self, prompt: str, *, max_tokens: int, temperature: float | None = None,
                          top_p: float | None = None, stop: list[str] | None = None,
                          cancel: threading.Event | None = None, extra_body: dict[str, Any] | None = None):
        yield from self._run({"kind": "completion", "prompt": prompt, "max_tokens": max_tokens}, prompt, cancel)

    def stream_chat(self, messages: list[dict[str, str]], *, max_tokens: int, temperature: float | None = None,
                    top_p: float | None = None, cancel: threading.Event | None = None):
        joined = "\n".join(m["content"] for m in messages)
        record = {"kind": "chat", "messages": messages, "max_tokens": max_tokens,
                  "prompt": messages[0]["content"] + messages[1]["content"]}
        for piece in self._run(record, joined, cancel):
            yield ChatDelta(content=piece, reasoning="")


def user_turn(prompt: str) -> str:
    """The user turn of a rendered ChatML prompt (the system prompt also NAMES the tags)."""
    return prompt.split("<|im_start|>user" + chr(10), 1)[1]


def make_segment(seg_id: int, *, text: str = "hello", source: Source = Source.SYSTEM) -> TranscriptSegment:
    return TranscriptSegment(
        id=seg_id, source=source, speaker_id="spk1", speaker_label="Speaker 1", text=text,
        t_start=float(seg_id), t_end=float(seg_id) + 1.0, confidence=0.9, is_question=False,
    )


def fill(transcript: Transcript, n: int, *, prefix: str = "We keep discussing the rollout plan item") -> None:
    for i in range(1, n + 1):
        transcript.add(make_segment(i, text=f"{prefix} {i}."))


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


def wait_for(cond: Callable[[], bool], timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return
        time.sleep(0.01)
    raise AssertionError("timed out waiting for a condition")


def run_ask(assistant: Assistant, bus_sub: "queue.Queue[Any]", text: str) -> list[Any]:
    """Ask `text` and return every event up to and including that job's Done."""
    assistant.ask(text)
    events = drain_until(bus_sub, lambda e: isinstance(e, AssistDoneEvent) or isinstance(e, AssistErrorEvent))
    return events


@pytest.fixture()
def cfg() -> Config:
    c = Config()
    c.llm.thinking = "off"            # raw-completion path: the prompt is the rendered ChatML
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
                   store: ContextStore | None = None) -> Assistant:
    manager = ContextManager(cfg, store if store is not None else ContextStore(None), transcript)
    return Assistant(cfg, bus, client, transcript, clock=time.monotonic, context=manager)


# --------------------------------------------------------------------------------------------
# construction
# --------------------------------------------------------------------------------------------


def test_old_five_argument_constructor_builds_a_private_manager(cfg: Config, bus: EventBus, transcript: Transcript) -> None:
    a = Assistant(cfg, bus, FakeClient(), transcript, time.monotonic)
    assert isinstance(a.context, ContextManager)
    assert a.context.store.documents() == []


def test_the_given_manager_is_the_one_used(cfg: Config, bus: EventBus, transcript: Transcript) -> None:
    manager = ContextManager(cfg, ContextStore(None), transcript)
    a = Assistant(cfg, bus, FakeClient(), transcript, time.monotonic, context=manager)
    assert a.context is manager


# --------------------------------------------------------------------------------------------
# reference material in the prompt
# --------------------------------------------------------------------------------------------


def test_injected_document_sits_before_the_transcript(cfg: Config, bus: EventBus, transcript: Transcript) -> None:
    store = ContextStore(None)
    store.add_text("Atlas README", "Atlas is the internal routing service. Codename: WOMBAT-7.",
                   mode="inject", max_chars=10000, digest_wanted=False)
    client = FakeClient()
    a = make_assistant(cfg, bus, client, transcript, store)
    sub = bus.subscribe()
    a.start()
    try:
        run_ask(a, sub, "What is Atlas?")
    finally:
        a.stop()
    prompt = user_turn(client.calls[-1]["prompt"])
    assert "<reference>" in prompt and "WOMBAT-7" in prompt
    assert prompt.index("<reference>") < prompt.index("<transcript>")
    assert "### Atlas README" in prompt
    assert "<lookup>" + chr(10) not in prompt


def test_reference_passage_appears_in_lookup_only_for_a_matching_question(
    cfg: Config, bus: EventBus, transcript: Transcript
) -> None:
    cfg.context.lookup_min_score = 0.1
    store = ContextStore(None)
    store.add_text(
        "Design doc",
        "# Cache\n\nThe frobnicator invalidates its cache with a generation counter per shard.\n\n"
        "# Deploy\n\nRollouts go through a canary stage first.",
        mode="reference", max_chars=10000, digest_wanted=False,
    )
    store.add_text("Other doc", "Completely unrelated notes about lunch menus and parking.",
                   mode="reference", max_chars=10000, digest_wanted=False)
    client = FakeClient()
    a = make_assistant(cfg, bus, client, transcript, store)
    sub = bus.subscribe()
    a.start()
    try:
        run_ask(a, sub, "How does the frobnicator invalidate its cache?")
        matching = user_turn(client.calls[-1]["prompt"])
        run_ask(a, sub, "Zxqv wibble?")
        unrelated = user_turn(client.calls[-1]["prompt"])
    finally:
        a.stop()
    block = "<lookup>" + chr(10)          # the catalogue sentence also NAMES the tag
    assert block in matching and "generation counter" in matching
    assert "Design doc" in matching.split(block)[1]
    assert matching.index("</transcript>") < matching.index(block) < matching.index("<question>")
    assert block not in unrelated
    # The catalogue (the stable <reference> block) is there either way.
    assert "Reference documents available" in unrelated and "Design doc" in unrelated


def test_include_in_questions_keeps_the_reference_out_of_a_solve(cfg: Config, bus: EventBus, transcript: Transcript) -> None:
    cfg.context.include_in = "questions"
    store = ContextStore(None)
    store.add_text("Atlas README", "Atlas codename WOMBAT-7 lives here.",
                   mode="inject", max_chars=10000, digest_wanted=False)
    client = FakeClient()
    a = make_assistant(cfg, bus, client, transcript, store)
    sub = bus.subscribe()
    a.start()
    try:
        a.solve("write it")
        drain_until(sub, lambda e: isinstance(e, AssistDoneEvent))
        solve_prompt = user_turn(client.calls[-1]["prompt"])
        run_ask(a, sub, "What is Atlas?")
        ask_prompt = user_turn(client.calls[-1]["prompt"])
    finally:
        a.stop()
    assert "WOMBAT-7" not in solve_prompt and "<reference>" not in solve_prompt
    assert "<transcript>" in solve_prompt           # the transcript is still budgeted and present
    assert "WOMBAT-7" in ask_prompt


def test_include_in_none_keeps_it_out_of_everything(cfg: Config, bus: EventBus, transcript: Transcript) -> None:
    cfg.context.include_in = "none"
    store = ContextStore(None)
    store.add_text("Atlas README", "Atlas codename WOMBAT-7 lives here.",
                   mode="inject", max_chars=10000, digest_wanted=False)
    client = FakeClient()
    a = make_assistant(cfg, bus, client, transcript, store)
    sub = bus.subscribe()
    a.start()
    try:
        run_ask(a, sub, "What is Atlas?")
    finally:
        a.stop()
    assert "WOMBAT-7" not in user_turn(client.calls[-1]["prompt"])


def test_reference_reaches_an_inline_completion_inside_its_context_block(
    cfg: Config, bus: EventBus, transcript: Transcript
) -> None:
    store = ContextStore(None)
    store.add_text("Atlas README", "Atlas codename WOMBAT-7 lives here.",
                   mode="inject", max_chars=10000, digest_wanted=False)
    client = FakeClient()
    a = make_assistant(cfg, bus, client, transcript, store)
    sub = bus.subscribe()
    a.start()
    try:
        job_id = a.complete("The codename is ", "", "Chat message")
        drain_until(sub, lambda e: isinstance(e, AssistDoneEvent) and e.job_id == job_id)
    finally:
        a.stop()
    full = client.calls[-1]["prompt"]
    prompt = user_turn(full)
    assert "<context>" in prompt
    assert prompt.index("<reference>") < prompt.index("<transcript>")
    assert full.endswith(prompts.NO_THINK_PREFIX + "The codename is ")


def test_a_context_manager_bug_degrades_to_the_plain_transcript_window(
    cfg: Config, bus: EventBus, transcript: Transcript, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = FakeClient()
    a = make_assistant(cfg, bus, client, transcript)

    def boom(**_kw: Any) -> None:
        raise RuntimeError("manager bug")

    monkeypatch.setattr(a.context, "build", boom)
    sub = bus.subscribe()
    a.start()
    try:
        events = run_ask(a, sub, "What is Atlas?")
    finally:
        a.stop()
    assert isinstance(events[-1], AssistDoneEvent) and not any(isinstance(e, AssistErrorEvent) for e in events)
    assert "Atlas project" in client.calls[-1]["prompt"]       # the plain window


# --------------------------------------------------------------------------------------------
# usage events + calibration
# --------------------------------------------------------------------------------------------


def usage_events(events: list[Any]) -> list[ContextUsageEvent]:
    return [e for e in events if isinstance(e, ContextUsageEvent)]


def test_two_usage_events_per_job_the_second_with_the_servers_counts(
    cfg: Config, bus: EventBus, transcript: Transcript
) -> None:
    client = FakeClient()
    client.report = (900, 700)
    a = make_assistant(cfg, bus, client, transcript)
    sub = bus.subscribe()
    a.start()
    try:
        events = run_ask(a, sub, "What is Atlas?")
    finally:
        a.stop()
    started = [e for e in events if isinstance(e, AssistStartedEvent)]
    done = [e for e in events if isinstance(e, AssistDoneEvent)]
    usage = usage_events(events)
    assert len(started) == 1 and len(done) == 1 and len(usage) == 2
    assert all(u.job_id == started[0].job_id and u.kind == "ask" for u in usage)
    first, second = usage
    assert first.prompt_tokens is None and first.cached_tokens is None
    assert (second.prompt_tokens, second.cached_tokens) == (900, 700)
    assert first.transcript_tokens > 0 and first.system_tokens > 0 and first.slot_tokens > 0
    assert second.transcript_tokens == first.transcript_tokens
    # The first one is published right after the prompt was built, i.e. before any delta.
    first_idx = events.index(first)
    assert first_idx < next(i for i, e in enumerate(events) if type(e).__name__ == "AssistDeltaEvent")
    snap = a.context.usage_snapshot()
    assert snap["has_usage"] and snap["prompt_tokens"] == 900 and snap["cached_tokens"] == 700


def test_usage_second_event_keeps_none_when_the_client_reports_nothing(
    cfg: Config, bus: EventBus, transcript: Transcript
) -> None:
    client = FakeClient()
    a = make_assistant(cfg, bus, client, transcript)
    sub = bus.subscribe()
    a.start()
    try:
        events = run_ask(a, sub, "What is Atlas?")
    finally:
        a.stop()
    usage = usage_events(events)
    assert len(usage) == 2
    assert usage[1].prompt_tokens is None and usage[1].cached_tokens is None
    assert a.context.usage_snapshot()["prompt_tokens"] is None


def test_calibration_uses_the_exact_text_sent_on_the_completion_path(
    cfg: Config, bus: EventBus, transcript: Transcript, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = FakeClient()
    client.report = (1000, 0)
    a = make_assistant(cfg, bus, client, transcript)
    seen: list[tuple[int, int | None]] = []
    real = a.context.observe_prompt

    def spy(prompt_chars: int, prompt_tokens: int | None) -> None:
        seen.append((prompt_chars, prompt_tokens))
        real(prompt_chars, prompt_tokens)

    monkeypatch.setattr(a.context, "observe_prompt", spy)
    before = a.context.estimator.chars_per_token
    sub = bus.subscribe()
    a.start()
    try:
        events = run_ask(a, sub, "What is Atlas?")
    finally:
        a.stop()
    assert seen and seen[-1] == (len(client.calls[-1]["prompt"]), 1000)
    after = a.context.estimator.chars_per_token
    assert after != before
    assert usage_events(events)[-1].chars_per_token == pytest.approx(after)


def test_calibration_uses_system_plus_user_on_the_chat_path(
    cfg: Config, bus: EventBus, transcript: Transcript, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg.llm.thinking = "on"
    client = FakeClient()
    client.report = (1000, None)
    a = make_assistant(cfg, bus, client, transcript)
    seen: list[int] = []
    real = a.context.observe_prompt
    monkeypatch.setattr(a.context, "observe_prompt", lambda chars, toks: (seen.append(chars), real(chars, toks))[1])
    sub = bus.subscribe()
    a.start()
    try:
        run_ask(a, sub, "What is Atlas?")
    finally:
        a.stop()
    messages = client.calls[-1]["messages"]
    assert seen == [len(messages[0]["content"]) + len(messages[1]["content"])]


def test_no_calibration_without_a_reported_count(cfg: Config, bus: EventBus, transcript: Transcript) -> None:
    client = FakeClient()
    a = make_assistant(cfg, bus, client, transcript)
    before = a.context.estimator.chars_per_token
    sub = bus.subscribe()
    a.start()
    try:
        run_ask(a, sub, "What is Atlas?")
    finally:
        a.stop()
    assert a.context.estimator.chars_per_token == before


# --------------------------------------------------------------------------------------------
# overflow retry
# --------------------------------------------------------------------------------------------


def test_overflow_error_retries_once_with_a_smaller_prompt(cfg: Config, bus: EventBus) -> None:
    cfg.llm.parallel = 1
    cfg.context.trim_step = 0.0                 # slide: a smaller budget visibly shrinks the window
    transcript = Transcript()
    fill(transcript, 300)
    client = FakeClient()
    client.fail_with = [OVERFLOW]
    a = make_assistant(cfg, bus, client, transcript)
    client.window = 2600
    cpt_before = a.context.estimator.chars_per_token
    sub = bus.subscribe()
    a.start()
    try:
        events = run_ask(a, sub, "What is Atlas?")
    finally:
        a.stop()
    assert len(client.calls) == 2
    assert len([e for e in events if isinstance(e, AssistStartedEvent)]) == 1
    assert isinstance(events[-1], AssistDoneEvent) and not any(isinstance(e, AssistErrorEvent) for e in events)
    assert len(client.calls[1]["prompt"]) < len(client.calls[0]["prompt"]) - 200
    assert a.context.estimator.chars_per_token == pytest.approx(cpt_before * 0.85)


def test_a_second_overflow_is_reported_as_an_error(cfg: Config, bus: EventBus, transcript: Transcript) -> None:
    client = FakeClient()
    client.fail_with = [OVERFLOW, OVERFLOW]
    a = make_assistant(cfg, bus, client, transcript)
    sub = bus.subscribe()
    a.start()
    try:
        events = run_ask(a, sub, "What is Atlas?")
    finally:
        a.stop()
    assert len(client.calls) == 2
    assert isinstance(events[-1], AssistErrorEvent) and "exceeds" in events[-1].message


def test_an_unrelated_llm_error_is_not_retried(cfg: Config, bus: EventBus, transcript: Transcript) -> None:
    client = FakeClient()
    client.fail_with = [LLMError("connection refused")]
    a = make_assistant(cfg, bus, client, transcript)
    sub = bus.subscribe()
    a.start()
    try:
        events = run_ask(a, sub, "What is Atlas?")
    finally:
        a.stop()
    assert len(client.calls) == 1
    assert isinstance(events[-1], AssistErrorEvent)


# --------------------------------------------------------------------------------------------
# the server's context window
# --------------------------------------------------------------------------------------------


def test_slot_tokens_come_from_the_clients_context_window(cfg: Config, bus: EventBus, transcript: Transcript) -> None:
    client = FakeClient()
    client.window = 3072
    a = make_assistant(cfg, bus, client, transcript)
    a.start()
    try:
        assert a.context.slot_tokens == 3072
    finally:
        a.stop()


@pytest.mark.parametrize("window", [None, RuntimeError("boom")])
def test_unknown_context_window_falls_back_to_the_configured_ctx_size(
    cfg: Config, bus: EventBus, transcript: Transcript, window: Any
) -> None:
    cfg.llm.ctx_size = 5000
    client = FakeClient()
    client.window = window
    a = make_assistant(cfg, bus, client, transcript)
    a.context.set_slot_tokens(1234)             # a stale value from an earlier model
    a.start()
    try:
        assert a.context.slot_tokens == 5000
    finally:
        a.stop()


def test_set_model_rereads_the_context_window(cfg: Config, bus: EventBus, transcript: Transcript) -> None:
    client = FakeClient()
    client.window = 4096
    a = make_assistant(cfg, bus, client, transcript)
    a.start()
    try:
        client.window = 8192
        a.set_model("other-model")
        assert a.context.slot_tokens == 8192
    finally:
        a.stop()


# --------------------------------------------------------------------------------------------
# compaction
# --------------------------------------------------------------------------------------------


def compaction_responder(summary: str = "Earlier they agreed on the Kafka migration plan.") -> Callable[[str], list[str]]:
    return lambda text: [summary] if "<previous_summary>" in text else ["ok"]


def small_window(cfg: Config) -> None:
    """A transcript window of about 360 characters: a 40-line meeting overflows it at once."""
    cfg.llm.context_tokens = 100


def test_compaction_runs_after_an_anchor_move_and_the_next_prompt_carries_earlier(
    cfg: Config, bus: EventBus
) -> None:
    small_window(cfg)
    transcript = Transcript()
    fill(transcript, 40)
    client = FakeClient(responder=compaction_responder())
    a = make_assistant(cfg, bus, client, transcript)
    sub = bus.subscribe()
    a.start()
    try:
        events = run_ask(a, sub, "What is the plan?")
        assert usage_events(events)[0].anchor_moved
        assert "<earlier>" not in user_turn(client.calls[0]["prompt"])
        wait_for(lambda: a.context.window.staged_summary(None) != "")
        assert a.context.window.staged_summary(None) == "Earlier they agreed on the Kafka migration plan."
        assert a.context.window.earlier_summary(None) == ""     # staged, not yet in the prompt
        compact_calls = [c for c in client.calls if "<previous_summary>" in c["prompt"]]
        assert len(compact_calls) == 1
        assert compact_calls[0]["max_tokens"] == cfg.context.summary_tokens + 32
        assert "<dropped_transcript>" in compact_calls[0]["prompt"]
        # the same window: no anchor move, so the prompt (and its cached prefix) is unchanged
        run_ask(a, sub, "And what next?")
        assert "<earlier>" not in user_turn(client.calls[-1]["prompt"])
        # more speech makes the anchor move again: the staged summary is promoted on that turn
        for i in range(41, 81):
            transcript.add(make_segment(i, text=f"We keep discussing the rollout plan item {i}."))
        events = run_ask(a, sub, "And after that?")
        assert usage_events(events)[0].anchor_moved
    finally:
        a.stop()
    last = user_turn(client.calls[-1]["prompt"])
    assert "<earlier>" in last and "Kafka migration plan" in last
    assert last.index("<earlier>") < last.index("<transcript>")
    assert a.context.window.earlier_summary(None) == "Earlier they agreed on the Kafka migration plan."
    assert a.context.window.staged_summary(None) == ""


def test_compaction_off_never_queues_a_compaction(cfg: Config, bus: EventBus) -> None:
    small_window(cfg)
    cfg.context.compaction = "off"
    transcript = Transcript()
    fill(transcript, 40)
    client = FakeClient(responder=compaction_responder())
    a = make_assistant(cfg, bus, client, transcript)
    sub = bus.subscribe()
    a.start()
    try:
        run_ask(a, sub, "What is the plan?")
        time.sleep(0.3)
    finally:
        a.stop()
    assert not any("<previous_summary>" in c["prompt"] for c in client.calls)
    assert a.context.window.earlier_summary(None) == ""


def test_a_failed_compaction_gives_the_dropped_text_back(cfg: Config, bus: EventBus) -> None:
    small_window(cfg)
    transcript = Transcript()
    fill(transcript, 40)
    client = FakeClient(responder=compaction_responder())
    a = make_assistant(cfg, bus, client, transcript)
    sub = bus.subscribe()
    a.start()
    try:
        client.fail_with = [None, LLMError("model went away")]      # the answer works, the summary fails
        run_ask(a, sub, "What is the plan?")
        wait_for(lambda: any("<previous_summary>" in c["prompt"] for c in client.calls))
        wait_for(lambda: a.context.window.has_pending(None))
        time.sleep(0.1)
    finally:
        a.stop()
    assert a.context.window.has_pending(None)
    assert a.context.window.earlier_summary(None) == ""


def test_a_long_backlog_is_compacted_in_chunks_that_requeue_themselves(cfg: Config, bus: EventBus) -> None:
    cfg.llm.parallel = 1
    transcript = Transcript()
    fill(transcript, 80)
    client = FakeClient(responder=compaction_responder("Running summary so far."))
    a = make_assistant(cfg, bus, client, transcript)
    client.window = 1024                        # a tiny pool: the notes lane has almost no room
    sub = bus.subscribe()
    a.start()
    try:
        run_ask(a, sub, "What is the plan?")
        wait_for(lambda: a.context.window.staged_summary(None) != "" and not a.context.window.has_pending(None))
    finally:
        a.stop()
    compact_calls = [c for c in client.calls if "<previous_summary>" in c["prompt"]]
    assert len(compact_calls) >= 2
    # the chained jobs built on each other: every claim after the first carried the staged summary
    assert all("Running summary so far." in c["prompt"] for c in compact_calls[1:])


def test_reset_drops_summaries_and_pending_compactions(cfg: Config, bus: EventBus) -> None:
    small_window(cfg)
    transcript = Transcript()
    fill(transcript, 40)
    client = FakeClient(responder=compaction_responder())
    a = make_assistant(cfg, bus, client, transcript)
    sub = bus.subscribe()
    a.start()
    try:
        run_ask(a, sub, "What is the plan?")
        wait_for(lambda: a.context.window.staged_summary(None) != "")
        a.reset()
        assert a.context.window.earlier_summary(None) == "" and a.context.window.staged_summary(None) == ""
        assert a.context.window.anchor(None) is None
    finally:
        a.stop()


def test_a_compaction_finishing_after_a_reset_does_not_leak_into_the_new_session(
    cfg: Config, bus: EventBus
) -> None:
    small_window(cfg)
    transcript = Transcript()
    fill(transcript, 40)
    gate = threading.Event()
    started = threading.Event()

    def responder(text: str) -> list[str]:
        if "<previous_summary>" in text:
            started.set()
            gate.wait(3.0)
            return ["Stale summary of the old session."]
        return ["ok"]

    client = FakeClient(responder=responder)
    a = make_assistant(cfg, bus, client, transcript)
    sub = bus.subscribe()
    a.start()
    try:
        run_ask(a, sub, "What is the plan?")
        assert started.wait(3.0)
        a.reset()                               # while the compaction is mid-stream
        gate.set()
        time.sleep(0.3)
    finally:
        gate.set()
        a.stop()
    assert a.context.window.earlier_summary(None) == "" and a.context.window.staged_summary(None) == ""


# --------------------------------------------------------------------------------------------
# digests
# --------------------------------------------------------------------------------------------


def digest_responder(text: str) -> list[str]:
    return ["A design doc for the Atlas router: topics, shards, canary rollouts."] if "<document name=" in text else ["ok"]


def test_digest_job_stores_the_digest_and_publishes_a_library_event(
    cfg: Config, bus: EventBus, transcript: Transcript
) -> None:
    store = ContextStore(None)
    doc = store.add_text("Atlas design", "# Atlas\n\nRouting, shards and canary rollouts.",
                         mode="reference", max_chars=10000, digest_wanted=True)
    assert store.get(doc.id).digest_state == "pending"
    client = FakeClient(responder=digest_responder)
    a = make_assistant(cfg, bus, client, transcript, store)
    sub = bus.subscribe()
    a.start()
    try:
        # start() already queued it: pending documents are digested once the model is up.
        events = drain_until(sub, lambda e: isinstance(e, ContextLibraryEvent))
    finally:
        a.stop()
    ev = events[-1]
    assert ev.reason == "digest" and ev.doc_id == doc.id
    got = store.get(doc.id)
    assert got.digest_state == "ready" and "Atlas router" in got.digest
    call = next(c for c in client.calls if "<document name=" in c["prompt"])
    assert 'name="Atlas design"' in call["prompt"] and "canary rollouts" in call["prompt"]
    assert call["max_tokens"] == cfg.context.digest_tokens + 32
    assert prompts.system_digest() in call["prompt"]


def test_digest_document_queues_an_explicit_request(cfg: Config, bus: EventBus, transcript: Transcript) -> None:
    store = ContextStore(None)
    doc = store.add_text("Atlas design", "Routing and shards.", mode="reference", max_chars=10000, digest_wanted=False)
    client = FakeClient(responder=digest_responder)
    a = make_assistant(cfg, bus, client, transcript, store)
    sub = bus.subscribe()
    a.start()
    try:
        a.digest_document(doc.id)
        drain_until(sub, lambda e: isinstance(e, ContextLibraryEvent) and e.doc_id == doc.id)
    finally:
        a.stop()
    assert store.get(doc.id).digest_state == "ready"


def test_a_burst_of_pending_documents_is_digested_one_after_another(
    cfg: Config, bus: EventBus, transcript: Transcript
) -> None:
    store = ContextStore(None)
    ids = [
        store.add_text(f"Doc {i}", f"Content number {i} about topic {i}.", mode="reference",
                       max_chars=10000, digest_wanted=True).id
        for i in range(3)
    ]
    client = FakeClient(responder=digest_responder)
    a = make_assistant(cfg, bus, client, transcript, store)
    sub = bus.subscribe()
    a.start()
    try:
        for doc_id in ids:                      # the slot holds one job; the worker re-queues the rest
            a.digest_document(doc_id)
        wait_for(lambda: all(store.get(i).digest_state == "ready" for i in ids))
    finally:
        a.stop()
    assert store.pending_digests() == []


def test_a_failed_digest_is_marked_failed_and_announced(cfg: Config, bus: EventBus, transcript: Transcript) -> None:
    store = ContextStore(None)
    doc = store.add_text("Atlas design", "Routing and shards.", mode="reference", max_chars=10000, digest_wanted=False)
    client = FakeClient(responder=digest_responder)
    client.fail_with = [LLMError("model not loaded")]
    a = make_assistant(cfg, bus, client, transcript, store)
    sub = bus.subscribe()
    a.start()
    try:
        a.digest_document(doc.id)
        events = drain_until(sub, lambda e: isinstance(e, ContextLibraryEvent))
    finally:
        a.stop()
    assert events[-1].reason == "digest failed" and events[-1].doc_id == doc.id
    assert store.get(doc.id).digest_state == "failed"
    assert store.pending_digests() == []        # a failed document is not retried in a loop


def test_digest_for_an_unknown_document_is_a_no_op(cfg: Config, bus: EventBus, transcript: Transcript) -> None:
    client = FakeClient(responder=digest_responder)
    a = make_assistant(cfg, bus, client, transcript)
    a.start()
    try:
        a.digest_document("nope")
        time.sleep(0.3)
    finally:
        a.stop()
    assert not any("<document name=" in c["prompt"] for c in client.calls)


def digest_prompt_length(cfg: Config, bus: EventBus, transcript: Transcript, slot: int | None) -> int:
    store = ContextStore(None)
    doc = store.add_text("Big doc", "\n".join(f"Line {i} of a very long reference document." for i in range(800)),
                         mode="reference", max_chars=200000, digest_wanted=False)
    client = FakeClient(responder=digest_responder)
    a = make_assistant(cfg, bus, client, transcript, store)
    client.window = slot                        # read by the health check at start()
    sub = bus.subscribe()
    a.start()
    try:
        a.digest_document(doc.id)
        drain_until(sub, lambda e: isinstance(e, ContextLibraryEvent))
    finally:
        a.stop()
    return len(next(c for c in client.calls if "<document name=" in c["prompt"])["prompt"]) - len(prompts.system_digest())


def test_digest_excerpt_is_capped_and_shrinks_with_the_notes_lane(cfg: Config, transcript: Transcript) -> None:
    roomy = digest_prompt_length(cfg, EventBus(), transcript, None)
    assert roomy < 6000 + 600               # the excerpt is at most 6000 chars (+ the tags)
    tight = digest_prompt_length(cfg, EventBus(), transcript, 1024)
    assert tight < roomy - 1500


# --------------------------------------------------------------------------------------------
# notes-lane dispatch and lifecycle
# --------------------------------------------------------------------------------------------


def test_next_notes_job_serves_notes_then_compaction_then_digest(cfg: Config, bus: EventBus, transcript: Transcript) -> None:
    a = make_assistant(cfg, bus, FakeClient(), transcript)
    a._digest_slot.put(_DigestJob(id=3, doc_id="d"))
    a._compact_slot.put(_CompactJob(id=2, scope=None))
    a._notes_slot.put(_NotesJob(id=1, force=True))
    assert [type(a._next_notes_job()).__name__ for _ in range(3)] == ["_NotesJob", "_CompactJob", "_DigestJob"]


def test_next_notes_job_returns_none_once_the_notes_slot_is_closed(cfg: Config, bus: EventBus, transcript: Transcript) -> None:
    a = make_assistant(cfg, bus, FakeClient(), transcript)
    a._notes_slot.close()
    assert a._next_notes_job() is None


def test_next_notes_job_wakes_for_a_job_queued_while_waiting(cfg: Config, bus: EventBus, transcript: Transcript) -> None:
    a = make_assistant(cfg, bus, FakeClient(), transcript)
    got: list[Any] = []
    t = threading.Thread(target=lambda: got.append(a._next_notes_job()), daemon=True)
    t.start()
    time.sleep(0.1)
    a._compact_slot.put(_CompactJob(id=9, scope=None))
    t.join(2.0)
    assert got and isinstance(got[0], _CompactJob)


def test_stop_closes_and_start_reopens_the_new_slots(cfg: Config, bus: EventBus, transcript: Transcript) -> None:
    a = make_assistant(cfg, bus, FakeClient(), transcript)
    a.start()
    a.stop()
    assert a._compact_slot.closed and a._digest_slot.closed
    a.start()
    try:
        assert not a._compact_slot.closed and not a._digest_slot.closed
    finally:
        a.stop()
    assert not any(t.is_alive() for t in a._threads)


def test_notes_chunk_is_bounded_by_the_notes_lane_and_still_advances(cfg: Config, bus: EventBus) -> None:
    cfg.llm.context_tokens = 2500                       # cap: 10000 chars
    transcript = Transcript()
    fill(transcript, 400)                               # ~20000 chars of speech
    client = FakeClient(responder=lambda text: ["# Notes"])
    a = make_assistant(cfg, bus, client, transcript)
    client.window = 2048                                # notes lane: ~614 tokens in the pool
    sub = bus.subscribe()
    a.start()
    try:
        a.generate_notes(force=True)
        drain_until(sub, lambda e: type(e).__name__ == "NotesUpdatedEvent")
    finally:
        a.stop()
    prompt = next(c["prompt"] for c in client.calls if "<new_transcript>" in c["prompt"])
    chunk = prompt.split("<new_transcript>\n", 1)[1].split("\n</new_transcript>", 1)[0]
    assert 0 < len(chunk) < 2000                          # far under the 10000-char cap, not empty


# --------------------------------------------------------------------------------------------
# post-review fixes (2026-10-06): notes sized to the lane, digest prep failures, retry safety
# --------------------------------------------------------------------------------------------


def _notes_call(client: FakeClient) -> dict[str, Any]:
    return next(c for c in client.calls if "<new_transcript>" in c["prompt"])


def test_notes_keep_the_full_answer_budget_when_the_lane_has_room(cfg: Config, bus: EventBus) -> None:
    """The default 8192-token context with two slots leaves the notes lane ~2457 tokens: the first
    pass (no previous notes) keeps the configured 900-token answer and states no word limit."""
    transcript = Transcript()
    fill(transcript, 60)
    client = FakeClient(responder=lambda text: ["# Notes"])
    a = make_assistant(cfg, bus, client, transcript)
    client.window = 8192
    sub = bus.subscribe()
    a.start()
    try:
        a.generate_notes(force=True)
        drain_until(sub, lambda e: type(e).__name__ == "NotesUpdatedEvent")
    finally:
        a.stop()
    call = _notes_call(client)
    assert call["max_tokens"] == cfg.llm.notes_max_tokens
    assert "Keep the whole document under" not in call["prompt"]


def test_notes_scale_down_to_a_small_lane_and_say_so(cfg: Config, bus: EventBus) -> None:
    """The owner's 4096-token profile: the notes lane (~1228 tokens) cannot hold a 900-token
    document twice, so the answer budget shrinks, the prompt states the word limit, and the chunk
    still advances the notes (review finding, 2026-10-06)."""
    transcript = Transcript()
    fill(transcript, 200)
    client = FakeClient(responder=lambda text: ["# Notes"])
    a = make_assistant(cfg, bus, client, transcript)
    client.window = 4096
    sub = bus.subscribe()
    a.start()
    try:
        a.generate_notes(force=True)
        drain_until(sub, lambda e: type(e).__name__ == "NotesUpdatedEvent")
    finally:
        a.stop()
    call = _notes_call(client)
    assert 200 <= call["max_tokens"] < cfg.llm.notes_max_tokens
    assert f"Keep the whole document under {prompts.words_for_tokens(call['max_tokens'])} words." in call["prompt"]
    chunk = call["prompt"].split("<new_transcript>\n", 1)[1].split("\n</new_transcript>", 1)[0]
    assert len(chunk) >= 200
    # the whole prompt + answer fits the notes lane (with the estimator's own arithmetic)
    est = a.context.estimator
    assert est.tokens(call["prompt"]) + call["max_tokens"] <= a.context.job_limit("notes") + 8


def test_previous_notes_longer_than_the_lane_share_are_cut_on_a_line_boundary(cfg: Config, bus: EventBus) -> None:
    transcript = Transcript()
    fill(transcript, 50)
    client = FakeClient(responder=lambda text: ["# Notes"])
    a = make_assistant(cfg, bus, client, transcript)
    client.window = 4096
    a._notes_markdown = "\n".join(f"- point number {i} of the previous notes" for i in range(200))   # ~7800 chars
    sub = bus.subscribe()
    a.start()
    try:
        a.generate_notes(force=True)
        drain_until(sub, lambda e: type(e).__name__ == "NotesUpdatedEvent")
    finally:
        a.stop()
    call = _notes_call(client)
    prev = call["prompt"].split("<previous_notes>\n", 1)[1].split("\n</previous_notes>", 1)[0]
    assert prev.startswith("- point number 0")                       # the head survives
    assert len(prev) < 7800 and prev.endswith("of the previous notes")   # cut on a whole line


def test_digest_that_cannot_be_prepared_is_marked_failed_not_requeued_forever(
    cfg: Config, bus: EventBus, transcript: Transcript, monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ContextStore(None)
    doc = store.add_text("a.md", "alpha document body", mode="reference", max_chars=10000, digest_wanted=True)
    doc2 = store.add_text("b.md", "beta document body", mode="reference", max_chars=10000, digest_wanted=True)
    client = FakeClient(responder=lambda text: ["a digest"])
    a = make_assistant(cfg, bus, client, transcript, store)
    monkeypatch.setattr(store, "text", lambda _id: (_ for _ in ()).throw(RuntimeError("disk gone")))
    sub = bus.subscribe()
    a.start()
    try:
        drain_until(sub, lambda e: isinstance(e, ContextLibraryEvent) and e.reason == "digest failed" and e.doc_id == doc2.id, timeout=5.0)
        wait_for(lambda: {d.digest_state for d in store.documents()} == {"failed"})
    finally:
        a.stop()
    assert store.pending_digests() == []
    assert not any(c for c in client.calls if "<document" in c["prompt"])   # nothing ever reached the model
    assert {d.id for d in store.documents()} == {doc.id, doc2.id}


def test_a_failure_while_rebuilding_the_retry_prompt_still_ends_in_an_error_event(
    cfg: Config, bus: EventBus, transcript: Transcript, monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeClient()
    client.fail_with = [OVERFLOW]
    a = make_assistant(cfg, bus, client, transcript)
    original = a._prepare
    calls = {"n": 0}

    def flaky(*args: Any, **kwargs: Any):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("prepare blew up on the retry")
        return original(*args, **kwargs)

    monkeypatch.setattr(a, "_prepare", flaky)
    sub = bus.subscribe()
    a.start()
    try:
        events = run_ask(a, sub, "What is the plan?")
    finally:
        a.stop()
    assert sum(isinstance(e, AssistStartedEvent) for e in events) == 1
    errors = [e for e in events if isinstance(e, AssistErrorEvent)]
    assert len(errors) == 1 and "prepare blew up" in errors[0].message
    assert not any(isinstance(e, AssistDoneEvent) for e in events)


# --------------------------------------------------------------------------------------------
# yielding to a live answer (2026-10-06, second live run)
# --------------------------------------------------------------------------------------------


def test_a_compaction_steps_aside_for_an_answer_and_finishes_afterwards(cfg: Config, bus: EventBus) -> None:
    """Measured live: a compaction streaming beside an answer took the answer's first token from
    0.3 s to 4.9 s. The compaction is cancelled when an answer starts, hands its text back, and
    runs again once the answer lane is idle."""
    small_window(cfg)
    transcript = Transcript()
    fill(transcript, 40)
    compact_started = threading.Event()

    def responder(text: str) -> list[str]:
        if "<previous_summary>" in text:
            compact_started.set()
            return ["Earlier ", "they ", "agreed ", "on ", "the ", "Kafka ", "plan."]
        return ["ok"]

    client = FakeClient(responder=responder, chunk_delay=0.05)
    a = make_assistant(cfg, bus, client, transcript)
    sub = bus.subscribe()
    a.start()
    try:
        run_ask(a, sub, "What is the plan?")              # moves the anchor -> a compaction is queued
        assert compact_started.wait(3.0)
        time.sleep(0.08)                                   # mid-stream
        run_ask(a, sub, "Another question?")               # an answer starts: the compaction yields
        wait_for(lambda: a.context.window.staged_summary(None) != "", timeout=6.0)
    finally:
        a.stop()
    compact_calls = [c for c in client.calls if "<previous_summary>" in c["prompt"]]
    assert len(compact_calls) >= 2                        # the first attempt yielded, the second ran
    assert a.context.window.staged_summary(None) == "Earlier they agreed on the Kafka plan."
    assert not a.context.window.has_pending(None)


def test_lane_cancel_stops_yielding_after_the_attempt_limit(cfg: Config, bus: EventBus, transcript: Transcript) -> None:
    client = FakeClient()
    a = make_assistant(cfg, bus, client, transcript)
    assert a._lane_cancel(0) is a._lane_yield
    assert a._lane_cancel(1) is a._lane_yield
    assert a._lane_cancel(2) is a._stop_event              # runs to completion regardless


def test_a_digest_steps_aside_for_an_answer_and_is_requeued(cfg: Config, bus: EventBus, transcript: Transcript) -> None:
    store = ContextStore(None)
    doc = store.add_text("atlas.md", "# Atlas\nrouter design " * 50, mode="reference", max_chars=100000, digest_wanted=True)
    digest_started = threading.Event()

    def responder(text: str) -> list[str]:
        if "<document name=" in text:
            digest_started.set()
            return ["A ", "design ", "doc ", "for ", "Atlas."]
        return ["ok"]

    client = FakeClient(responder=responder, chunk_delay=0.05)
    a = make_assistant(cfg, bus, client, transcript, store)
    sub = bus.subscribe()
    a.start()
    try:
        assert digest_started.wait(3.0)
        time.sleep(0.08)
        run_ask(a, sub, "What is the plan?")
        wait_for(lambda: store.get(doc.id).digest_state == "ready", timeout=6.0)   # type: ignore[union-attr]
    finally:
        a.stop()
    digest_calls = [c for c in client.calls if "<document name=" in c["prompt"]]
    assert len(digest_calls) >= 2
    assert store.get(doc.id).digest == "A design doc for Atlas."   # type: ignore[union-attr]

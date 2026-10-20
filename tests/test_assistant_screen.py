"""Tests for the screen-reading extension of exocortex.llm.assistant.Assistant:
set_screen_provider(), suggest_now(), auto-suggest, and screen-triggered questions.

Fully offline: a FakeClient (duck-typed, no httpx/LMStudioClient subclassing) stands in for
exocortex.llm.client.LMStudioClient, and a real EventBus + real Transcript are used throughout,
exactly as in tests/test_assistant.py. FakeClient/make_segment/drain_until are copied from that
file rather than imported, so this file does not depend on another module's private test
internals staying stable.
"""
from __future__ import annotations

import queue
import threading
import time
from types import SimpleNamespace
from typing import Any, Callable

import pytest

import exocortex.llm.assistant as assistant_module
from exocortex.bus import (
    AssistDeltaEvent,
    AssistDoneEvent,
    AssistStartedEvent,
    EventBus,
    ScreenTextEvent,
    Source,
    TranscriptSegment,
    drain,
)
from exocortex.config import Config
from exocortex.llm.assistant import Assistant
from exocortex.llm.client import ChatDelta, Health, LLMError, ModelInfo, StreamStats
from exocortex.transcript import Transcript

# --------------------------------------------------------------------------------------------
# test doubles / helpers (copied from tests/test_assistant.py, see module docstring)
# --------------------------------------------------------------------------------------------


class FakeClient:
    """Duck-typed stand-in for LMStudioClient. No httpx, no subclassing.

    `token_batches` is consumed one batch per stream_completion/stream_chat call, in order. Each
    item is yielded after a `chunk_delay` sleep so cancellation has time to land mid-stream, and
    `cancel.is_set()` is honoured exactly like the real client.
    """

    def __init__(
        self,
        *,
        model: str = "maple-preview",
        token_batches: list[list[Any]] | None = None,
        chunk_delay: float = 0.02,
    ) -> None:
        self.cfg = SimpleNamespace(model=model, base_url="http://fake.test", auto_load_model=True)
        self._local = threading.local()
        self._batches = list(token_batches) if token_batches is not None else None
        self._chunk_delay = chunk_delay
        self.calls: list[dict[str, Any]] = []
        self.health_result = Health(reachable=True, model_listed=True, model_loaded=True, detail="")
        self.list_models_result = [ModelInfo(id=model, loaded=True)]
        self.raise_error: LLMError | None = None
        self.ensure_loaded_calls = 0

    @property
    def last_stats(self) -> StreamStats | None:
        return getattr(self._local, "value", None)

    @last_stats.setter
    def last_stats(self, value: StreamStats | None) -> None:
        self._local.value = value

    def health(self) -> Health:
        return self.health_result

    def ensure_loaded(self) -> bool:
        self.ensure_loaded_calls += 1
        return True

    def list_models(self) -> list[ModelInfo]:
        return self.list_models_result

    def stream_completion(
        self, prompt: str, *, max_tokens: int, temperature: float | None = None,
        top_p: float | None = None, stop: list[str] | None = None, cancel: threading.Event | None = None,
    ):
        self.calls.append({"kind": "completion", "prompt": prompt, "max_tokens": max_tokens, "stop": stop})
        if self.raise_error is not None:
            raise self.raise_error
        for item in self._emit(self._next_batch(), cancel):
            yield item if isinstance(item, str) else item[0]

    def stream_chat(
        self, messages: list[dict[str, str]], *, max_tokens: int, temperature: float | None = None,
        top_p: float | None = None, cancel: threading.Event | None = None,
    ):
        self.calls.append({"kind": "chat", "messages": messages, "max_tokens": max_tokens})
        if self.raise_error is not None:
            raise self.raise_error
        for item in self._emit(self._next_batch(), cancel):
            if isinstance(item, tuple):
                yield ChatDelta(content=item[0], reasoning=item[1])
            else:
                yield ChatDelta(content=item, reasoning="")

    def _next_batch(self) -> list[Any]:
        if self._batches:
            return self._batches.pop(0)
        return ["ok"]

    def _emit(self, batch: list[Any], cancel: threading.Event | None):
        start = time.monotonic()
        ttft: float | None = None
        count = 0
        cancelled = False
        for item in batch:
            if cancel is not None and cancel.is_set():
                cancelled = True
                break
            time.sleep(self._chunk_delay)
            if cancel is not None and cancel.is_set():
                cancelled = True
                break
            if ttft is None:
                ttft = time.monotonic() - start
            count += 1
            yield item
        elapsed = max(time.monotonic() - start, 1e-9)
        self.last_stats = StreamStats(
            ttft_s=ttft or 0.0, tokens=count, elapsed_s=elapsed, tok_per_s=count / elapsed, cancelled=cancelled
        )


def make_segment(
    seg_id: int,
    *,
    source: Source = Source.SYSTEM,
    text: str = "hello",
    t_start: float = 0.0,
    t_end: float = 1.0,
    is_question: bool = False,
    speaker_id: str = "spk1",
    speaker_label: str = "Speaker 1",
) -> TranscriptSegment:
    return TranscriptSegment(
        id=seg_id, source=source, speaker_id=speaker_id, speaker_label=speaker_label, text=text,
        t_start=t_start, t_end=t_end, confidence=0.9, is_question=is_question,
    )


def make_screen_event(
    *,
    lines: tuple[str, ...],
    text: str | None = None,
    changed: bool = True,
    mode: str = "auto",
    source_name: str = "Fake Window",
    rect: tuple[int, int, int, int] = (0, 0, 100, 100),
    ocr_ms: float = 1.0,
    captured_at: float = 0.0,
) -> ScreenTextEvent:
    return ScreenTextEvent(
        mode=mode,
        source_name=source_name,
        rect=rect,
        text=text if text is not None else "\n".join(lines),
        lines=tuple(lines),
        changed=changed,
        ocr_ms=ocr_ms,
        captured_at=captured_at,
    )


def drain_until(sub: "queue.Queue[Any]", predicate: Callable[[Any], bool], timeout: float = 3.0) -> list[Any]:
    """Collect events from `sub` until one satisfies `predicate` (inclusive), or raise on timeout."""
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


def _make_assistant(cfg: Config, bus: EventBus, transcript: Transcript, client: FakeClient) -> Assistant:
    return Assistant(cfg, bus, client, transcript, clock=time.monotonic)


# --------------------------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------------------------


@pytest.fixture()
def bus() -> EventBus:
    return EventBus()


@pytest.fixture()
def transcript() -> Transcript:
    return Transcript()


# --------------------------------------------------------------------------------------------
# (a) set_screen_provider(): screen_text in assist/ask prompts, gated by use_in_answers
# --------------------------------------------------------------------------------------------


def test_screen_text_included_between_transcript_and_question_when_use_in_answers(
    bus: EventBus, transcript: Transcript
) -> None:
    cfg = Config()
    cfg.screen.use_in_answers = True
    transcript.add(make_segment(1, text="What is the deployment plan?", t_end=1.0, is_question=True))
    client = FakeClient(token_batches=[["ok"]])
    assistant = _make_assistant(cfg, bus, transcript, client)
    assistant.set_screen_provider(lambda: "Line one\nLine two")
    sub = bus.subscribe()
    assistant.start()
    try:
        assistant.answer_last()
        drain_until(sub, lambda e: isinstance(e, AssistDoneEvent))

        prompt = client.calls[0]["prompt"]
        # "<screen>" alone also appears in the system-prompt rule describing the block, so use
        # "<screen>\n" (the actual block's opening tag, from prompts._screen_block) to be sure an
        # actual block -- not just the rule mentioning it -- was inserted.
        assert "<screen>\n" in prompt
        assert "Line one" in prompt
        transcript_end = prompt.index("</transcript>")
        screen_start = prompt.index("<screen>\n")
        question_start = prompt.index("<question>")
        assert transcript_end < screen_start < question_start
        assert assistant.last_screen_text == "Line one\nLine two"
    finally:
        assistant.stop()


def test_screen_text_excluded_when_use_in_answers_is_false(bus: EventBus, transcript: Transcript) -> None:
    cfg = Config()
    cfg.screen.use_in_answers = False
    transcript.add(make_segment(1, text="What is the deployment plan?", t_end=1.0, is_question=True))
    client = FakeClient(token_batches=[["ok"]])
    assistant = _make_assistant(cfg, bus, transcript, client)
    assistant.set_screen_provider(lambda: "Line one")
    sub = bus.subscribe()
    assistant.start()
    try:
        assistant.answer_last()
        drain_until(sub, lambda e: isinstance(e, AssistDoneEvent))

        assert "Line one" not in client.calls[0]["prompt"]
        assert "<screen>\n" not in client.calls[0]["prompt"]
        assert assistant.last_screen_text == ""
    finally:
        assistant.stop()


def test_screen_provider_exception_yields_empty_screen_text(bus: EventBus, transcript: Transcript) -> None:
    cfg = Config()
    client = FakeClient(token_batches=[["ok"]])
    assistant = _make_assistant(cfg, bus, transcript, client)

    def _boom() -> str:
        raise RuntimeError("ocr broke")

    assistant.set_screen_provider(_boom)
    sub = bus.subscribe()
    assistant.start()
    try:
        assistant.ask("status?")
        drain_until(sub, lambda e: isinstance(e, AssistDoneEvent))

        assert "<screen>\n" not in client.calls[0]["prompt"]
        assert assistant.last_screen_text == ""
    finally:
        assistant.stop()


def test_screen_text_truncated_to_max_text_chars(bus: EventBus, transcript: Transcript) -> None:
    cfg = Config()
    cfg.screen.max_text_chars = 5
    client = FakeClient(token_batches=[["ok"]])
    assistant = _make_assistant(cfg, bus, transcript, client)
    assistant.set_screen_provider(lambda: "abcdefghij")
    sub = bus.subscribe()
    assistant.start()
    try:
        assistant.ask("status?")
        drain_until(sub, lambda e: isinstance(e, AssistDoneEvent))

        assert assistant.last_screen_text == "abcde"
        assert "abcde" in client.calls[0]["prompt"]
        assert "abcdef" not in client.calls[0]["prompt"]
    finally:
        assistant.stop()


def test_no_screen_provider_means_no_screen_block(bus: EventBus, transcript: Transcript) -> None:
    cfg = Config()
    client = FakeClient(token_batches=[["ok"]])
    assistant = _make_assistant(cfg, bus, transcript, client)
    sub = bus.subscribe()
    assistant.start()
    try:
        assistant.ask("status?")
        drain_until(sub, lambda e: isinstance(e, AssistDoneEvent))
        assert "<screen>\n" not in client.calls[0]["prompt"]
        assert assistant.last_screen_text == ""
    finally:
        assistant.stop()


# --------------------------------------------------------------------------------------------
# (b) suggest_now(): kind, prompt content, suggestions property
# --------------------------------------------------------------------------------------------


def test_suggest_now_publishes_suggest_kind_with_screen_block_and_records_suggestion(
    bus: EventBus, transcript: Transcript
) -> None:
    cfg = Config()
    client = FakeClient(token_batches=[["- Mention the Q3 numbers."]])
    assistant = _make_assistant(cfg, bus, transcript, client)
    assistant.set_screen_provider(lambda: "Some screen text")
    sub = bus.subscribe()
    assistant.start()
    try:
        assistant.suggest_now("focus on pricing")

        events = drain_until(sub, lambda e: isinstance(e, AssistDoneEvent))
        started = next(e for e in events if isinstance(e, AssistStartedEvent))
        done = next(e for e in events if isinstance(e, AssistDoneEvent))

        assert started.kind == "suggest"
        assert started.question == "focus on pricing"
        assert done.cancelled is False

        call = client.calls[0]
        assert call["max_tokens"] == cfg.llm.assist_max_tokens
        assert "Propose what" in call["prompt"]  # fragment unique to system_suggest
        assert "<screen>" in call["prompt"]
        assert "Some screen text" in call["prompt"]

        suggestions = assistant.suggestions
        assert len(suggestions) == 1
        assert suggestions[0][1] == done.text
        assert assistant.answers == []  # suggestions are never recorded as answers
    finally:
        assistant.stop()


def test_suggest_now_with_no_focus_defaults_question_to_suggestions(bus: EventBus, transcript: Transcript) -> None:
    cfg = Config()
    client = FakeClient(token_batches=[["ok"]])
    assistant = _make_assistant(cfg, bus, transcript, client)
    sub = bus.subscribe()
    assistant.start()
    try:
        assistant.suggest_now()
        events = drain_until(sub, lambda e: isinstance(e, AssistDoneEvent))
        started = next(e for e in events if isinstance(e, AssistStartedEvent))
        assert started.question == "Suggestions"
    finally:
        assistant.stop()


def test_suggestions_property_capped_at_50(bus: EventBus, transcript: Transcript) -> None:
    cfg = Config()
    client = FakeClient(token_batches=[["s"] for _ in range(55)], chunk_delay=0.0)
    assistant = _make_assistant(cfg, bus, transcript, client)
    sub = bus.subscribe()
    assistant.start()
    try:
        for _ in range(55):
            assistant.suggest_now()
            drain_until(sub, lambda e: isinstance(e, AssistDoneEvent))
        assert len(assistant.suggestions) <= 50
    finally:
        assistant.stop()


# --------------------------------------------------------------------------------------------
# (c) priority: suggest never cancels a running main-lane job; a main-lane job cancels a
# running suggest job
# --------------------------------------------------------------------------------------------


def test_suggest_never_cancels_a_running_manual_job_and_runs_after_it(
    bus: EventBus, transcript: Transcript
) -> None:
    cfg = Config()
    transcript.add(make_segment(1, text="Tell me about the migration plan.", t_end=1.0, is_question=True))
    client = FakeClient(token_batches=[list("abcdef"), ["s", "g"]], chunk_delay=0.05)
    assistant = _make_assistant(cfg, bus, transcript, client)
    sub = bus.subscribe()
    assistant.start()
    try:
        assistant.answer_last()  # manual job: slow stream (~0.3s)
        started1 = drain_until(sub, lambda e: isinstance(e, AssistStartedEvent))
        manual_started = next(e for e in started1 if isinstance(e, AssistStartedEvent))
        assert manual_started.kind == "manual"

        time.sleep(0.1)  # well into the manual stream
        assistant.suggest_now()  # must NOT cancel the manual job

        batch1 = drain_until(sub, lambda e: isinstance(e, AssistDoneEvent))
        manual_done = next(e for e in batch1 if isinstance(e, AssistDoneEvent))
        assert manual_done.job_id == manual_started.job_id
        assert manual_done.cancelled is False

        # the suggest job only starts once the manual job is done, and runs to completion
        batch2 = drain_until(sub, lambda e: isinstance(e, AssistStartedEvent) and e.kind == "suggest")
        suggest_started = next(e for e in batch2 if isinstance(e, AssistStartedEvent))
        batch3 = drain_until(sub, lambda e: isinstance(e, AssistDoneEvent))
        suggest_done = next(e for e in batch3 if isinstance(e, AssistDoneEvent))
        assert suggest_done.job_id == suggest_started.job_id
        assert suggest_done.cancelled is False
        assert suggest_done.text == "sg"
    finally:
        assistant.stop()


def test_second_suggest_now_cancels_a_running_first_suggest_job(bus: EventBus, transcript: Transcript) -> None:
    """`_enqueue_side_job` cancels the current job when it is itself a suggest/solve job, so a rapid
    second `suggest_now()` supersedes the first instead of queuing uselessly behind it (the
    first job's own Done event must still be published, marked cancelled)."""
    cfg = Config()
    client = FakeClient(token_batches=[list("abcdef"), ["ok"]], chunk_delay=0.05)
    assistant = _make_assistant(cfg, bus, transcript, client)
    sub = bus.subscribe()
    assistant.start()
    try:
        assistant.suggest_now("first")
        started1 = drain_until(sub, lambda e: isinstance(e, AssistStartedEvent))
        first_started = next(e for e in started1 if isinstance(e, AssistStartedEvent))
        assert first_started.kind == "suggest"

        time.sleep(0.1)  # well into the first suggest stream
        assistant.suggest_now("second")  # must cancel the running first suggest job

        batch1 = drain_until(sub, lambda e: isinstance(e, AssistDoneEvent))
        first_done = next(e for e in batch1 if isinstance(e, AssistDoneEvent))
        assert first_done.job_id == first_started.job_id
        assert first_done.cancelled is True

        batch2 = drain_until(sub, lambda e: isinstance(e, AssistStartedEvent) and e.question == "second")
        second_started = next(e for e in batch2 if isinstance(e, AssistStartedEvent))
        batch3 = drain_until(sub, lambda e: isinstance(e, AssistDoneEvent))
        second_done = next(e for e in batch3 if isinstance(e, AssistDoneEvent))
        assert second_done.job_id == second_started.job_id
        assert second_done.cancelled is False
        assert second_done.text == "ok"
    finally:
        assistant.stop()


def test_screen_provider_exception_during_suggest_yields_empty_screen_block(
    bus: EventBus, transcript: Transcript
) -> None:
    """A broken screen provider must not break a suggest job either: no crash, no screen text,
    and `user_suggest` still renders its own placeholder <screen> block instead of the real one."""
    cfg = Config()
    client = FakeClient(token_batches=[["- ok"]])
    assistant = _make_assistant(cfg, bus, transcript, client)

    def _boom() -> str:
        raise RuntimeError("ocr broke")

    assistant.set_screen_provider(_boom)
    sub = bus.subscribe()
    assistant.start()
    try:
        assistant.suggest_now("focus")
        drain_until(sub, lambda e: isinstance(e, AssistDoneEvent))

        prompt = client.calls[0]["prompt"]
        assert "(nothing readable on screen)" in prompt
        assert assistant.last_screen_text == ""
    finally:
        assistant.stop()


def test_ask_cancels_a_running_suggest_job(bus: EventBus, transcript: Transcript) -> None:
    cfg = Config()
    client = FakeClient(token_batches=[list("abcdef"), ["ok"]], chunk_delay=0.05)
    assistant = _make_assistant(cfg, bus, transcript, client)
    sub = bus.subscribe()
    assistant.start()
    try:
        assistant.suggest_now()
        started1 = drain_until(sub, lambda e: isinstance(e, AssistStartedEvent))
        suggest_started = next(e for e in started1 if isinstance(e, AssistStartedEvent))
        assert suggest_started.kind == "suggest"

        time.sleep(0.1)  # well into the suggest stream
        assistant.ask("what now?")  # must cancel the running suggest job

        batch1 = drain_until(sub, lambda e: isinstance(e, AssistDoneEvent))
        suggest_done = next(e for e in batch1 if isinstance(e, AssistDoneEvent))
        assert suggest_done.job_id == suggest_started.job_id
        assert suggest_done.cancelled is True

        batch2 = drain_until(sub, lambda e: isinstance(e, AssistStartedEvent) and e.kind == "ask")
        ask_started = next(e for e in batch2 if isinstance(e, AssistStartedEvent))
        batch3 = drain_until(sub, lambda e: isinstance(e, AssistDoneEvent))
        ask_done = next(e for e in batch3 if isinstance(e, AssistDoneEvent))
        assert ask_done.job_id == ask_started.job_id
        assert ask_done.cancelled is False
        assert ask_done.text == "ok"
    finally:
        assistant.stop()


# --------------------------------------------------------------------------------------------
# (d) auto-suggest: triggered by a changed ScreenTextEvent, throttled by interval + similarity
# --------------------------------------------------------------------------------------------


def test_auto_suggest_respects_interval_and_similarity(bus: EventBus, transcript: Transcript) -> None:
    cfg = Config()
    cfg.assist.autonomous = False   # exercise the "suggest" throttle path (see solve test below)
    cfg.screen.auto_suggest = True
    cfg.screen.suggest_min_interval_s = 1
    client = FakeClient(token_batches=[["s1"], ["s2"]], chunk_delay=0.01)
    assistant = _make_assistant(cfg, bus, transcript, client)
    sub = bus.subscribe()
    assistant.start()
    try:
        same_lines = ("Quarterly revenue grew twelve percent this year.",)
        bus.publish(make_screen_event(lines=same_lines))

        events = drain_until(sub, lambda e: isinstance(e, AssistStartedEvent) and e.kind == "suggest")
        first_started = next(e for e in events if isinstance(e, AssistStartedEvent) and e.kind == "suggest")
        drain_until(sub, lambda e: isinstance(e, AssistDoneEvent) and e.job_id == first_started.job_id)

        # a second event within suggest_min_interval_s: no new suggest job
        bus.publish(make_screen_event(lines=same_lines))
        time.sleep(0.3)
        assert not any(isinstance(e, AssistStartedEvent) and e.kind == "suggest" for e in drain(sub))

        # wait out the interval, but the lines are still near-identical (Jaccard >= 0.6): still no trigger
        time.sleep(cfg.screen.suggest_min_interval_s + 0.2)
        bus.publish(make_screen_event(lines=same_lines))
        time.sleep(0.3)
        assert not any(isinstance(e, AssistStartedEvent) and e.kind == "suggest" for e in drain(sub))

        # genuinely different lines after the interval: triggers a second suggest job
        bus.publish(make_screen_event(lines=("Completely different topic about staffing levels now.",)))
        events2 = drain_until(sub, lambda e: isinstance(e, AssistStartedEvent) and e.kind == "suggest")
        second_started = next(e for e in events2 if isinstance(e, AssistStartedEvent) and e.kind == "suggest")
        assert second_started.job_id != first_started.job_id
    finally:
        assistant.stop()


def test_auto_suggest_similarity_check_ignores_ocr_case_and_whitespace_noise(
    bus: EventBus, transcript: Transcript
) -> None:
    """Two OCR passes of the SAME on-screen content routinely differ only in case or incidental
    whitespace (different confidence run to run). The Jaccard similarity check must compare
    normalised lines so that noise alone never re-triggers a second suggestion for content the
    user was already just given a suggestion for."""
    cfg = Config()
    cfg.assist.autonomous = False   # exercise the "suggest" throttle path (see solve test below)
    cfg.screen.auto_suggest = True
    cfg.screen.suggest_min_interval_s = 0
    client = FakeClient(token_batches=[["s1"], ["s2"]], chunk_delay=0.01)
    assistant = _make_assistant(cfg, bus, transcript, client)
    sub = bus.subscribe()
    assistant.start()
    try:
        bus.publish(make_screen_event(lines=("Quarterly revenue grew twelve percent this year.",)))
        events = drain_until(sub, lambda e: isinstance(e, AssistStartedEvent) and e.kind == "suggest")
        first = next(e for e in events if isinstance(e, AssistStartedEvent) and e.kind == "suggest")
        drain_until(sub, lambda e: isinstance(e, AssistDoneEvent) and e.job_id == first.job_id)

        # Same content: different case, an extra internal space, and a trailing space.
        bus.publish(make_screen_event(lines=("quarterly revenue  grew twelve percent this year. ",)))
        time.sleep(0.3)
        extra = drain(sub)
        assert not any(isinstance(e, AssistStartedEvent) and e.kind == "suggest" for e in extra)
    finally:
        assistant.stop()


def test_autonomous_screen_change_triggers_solve_not_suggest(
    bus: EventBus, transcript: Transcript
) -> None:
    """With autonomous mode on (the default), a screen change fires a kind "solve" job (whose
    result is an answer), not a kind "suggest" one."""
    cfg = Config()
    assert cfg.assist.autonomous is True
    cfg.screen.auto_suggest = True
    cfg.screen.suggest_min_interval_s = 0
    client = FakeClient(token_batches=[["def two_sum(): ..."]], chunk_delay=0.0)
    assistant = _make_assistant(cfg, bus, transcript, client)
    assistant.set_screen_provider(lambda: "Two Sum: return indices of the two numbers that add up to target.")
    sub = bus.subscribe()
    assistant.start()
    try:
        bus.publish(make_screen_event(lines=("Two Sum", "return indices adding to target")))
        events = drain_until(sub, lambda e: isinstance(e, AssistStartedEvent))
        started = next(e for e in events if isinstance(e, AssistStartedEvent))
        assert started.kind == "solve"
        drain_until(sub, lambda e: isinstance(e, AssistDoneEvent) and e.job_id == started.job_id)
        # a "solve" result is stored as an answer, never as a suggestion
        assert assistant.answers and assistant.answers[-1][2] == "def two_sum(): ..."
        assert assistant.suggestions == []
    finally:
        assistant.stop()


def test_paused_assistant_ignores_screen_and_auto(bus: EventBus, transcript: Transcript) -> None:
    """set_paused(True) suppresses auto-answer, screen questions and auto-solve/suggest."""
    cfg = Config()
    cfg.screen.auto_suggest = True
    cfg.screen.suggest_min_interval_s = 0
    cfg.screen.answer_screen_questions = True
    client = FakeClient(token_batches=[["x"] for _ in range(5)], chunk_delay=0.0)
    assistant = _make_assistant(cfg, bus, transcript, client)
    assistant.set_screen_provider(lambda: "some screen text")
    sub = bus.subscribe()
    assistant.start()
    try:
        assistant.set_paused(True)
        bus.publish(make_screen_event(lines=("How would you implement a thread-safe LRU cache here?",)))
        time.sleep(0.3)
        assert not any(isinstance(e, AssistStartedEvent) for e in drain(sub))
        # resuming lets the next screen change through
        assistant.set_paused(False)
        bus.publish(make_screen_event(lines=("A completely different problem statement appears now.",)))
        events = drain_until(sub, lambda e: isinstance(e, AssistStartedEvent))
        assert any(isinstance(e, AssistStartedEvent) for e in events)
    finally:
        assistant.stop()


def test_auto_suggest_disabled_never_triggers(bus: EventBus, transcript: Transcript) -> None:
    cfg = Config()
    cfg.screen.auto_suggest = False
    client = FakeClient()
    assistant = _make_assistant(cfg, bus, transcript, client)
    sub = bus.subscribe()
    assistant.start()
    try:
        bus.publish(make_screen_event(lines=("Anything at all on the screen right now.",)))
        time.sleep(0.3)
        assert not any(isinstance(e, AssistStartedEvent) for e in drain(sub))
    finally:
        assistant.stop()


def test_unchanged_screen_event_never_triggers_auto_suggest(bus: EventBus, transcript: Transcript) -> None:
    cfg = Config()
    cfg.screen.auto_suggest = True
    client = FakeClient()
    assistant = _make_assistant(cfg, bus, transcript, client)
    sub = bus.subscribe()
    assistant.start()
    try:
        bus.publish(make_screen_event(lines=("Some unchanged text on screen.",), changed=False))
        time.sleep(0.3)
        assert not any(isinstance(e, AssistStartedEvent) for e in drain(sub))
    finally:
        assistant.stop()


# --------------------------------------------------------------------------------------------
# (e) screen questions: new question lines trigger kind "screen"; dedup; non-questions ignored
# --------------------------------------------------------------------------------------------


def test_screen_question_job_includes_screen_text_even_when_use_in_answers_is_false(
    bus: EventBus, transcript: Transcript, monkeypatch
) -> None:
    """A kind="screen" job exists specifically to answer a question the OCR just found ON the
    screen; without the surrounding screen text the model has no antecedent for something like
    "how would you implement this?". `cfg.screen.use_in_answers` is documented (config.py) as
    gating "assist/ask prompts" only, so it must not blank out a screen job's <screen> block --
    the same way it never gates a "suggest" job's."""
    monkeypatch.setattr(assistant_module, "SCREEN_QUESTION_MIN_INTERVAL_S", 0.05)
    cfg = Config()
    cfg.screen.use_in_answers = False
    cfg.screen.answer_screen_questions = True
    cfg.screen.auto_suggest = False
    client = FakeClient(token_batches=[["- answer"]])
    assistant = _make_assistant(cfg, bus, transcript, client)
    question = "Can you explain how the rate limiter design handles bursts?"
    assistant.set_screen_provider(lambda: f"Surrounding code context.\n{question}")
    sub = bus.subscribe()
    assistant.start()
    try:
        bus.publish(make_screen_event(lines=(question,)))
        events = drain_until(sub, lambda e: isinstance(e, AssistDoneEvent))
        started = next(e for e in events if isinstance(e, AssistStartedEvent))
        assert started.kind == "screen"

        prompt = client.calls[0]["prompt"]
        assert "<screen>\n" in prompt
        assert "Surrounding code context" in prompt
        assert assistant.last_screen_text != ""
    finally:
        assistant.stop()


def test_new_screen_question_triggers_screen_job(bus: EventBus, transcript: Transcript, monkeypatch) -> None:
    monkeypatch.setattr(assistant_module, "SCREEN_QUESTION_MIN_INTERVAL_S", 0.05)
    cfg = Config()
    cfg.screen.answer_screen_questions = True
    cfg.screen.auto_suggest = False  # isolate the screen-question path from auto-suggest
    client = FakeClient(token_batches=[["- Bursts are smoothed via a token bucket."]])
    assistant = _make_assistant(cfg, bus, transcript, client)
    sub = bus.subscribe()
    assistant.start()
    try:
        question = "Can you explain how your rate limiter design handles bursts?"
        bus.publish(make_screen_event(lines=(question,)))

        events = drain_until(sub, lambda e: isinstance(e, AssistDoneEvent))
        started = next(e for e in events if isinstance(e, AssistStartedEvent))
        assert started.kind == "screen"
        assert started.question == question
    finally:
        assistant.stop()


def test_same_screen_question_is_not_answered_twice(bus: EventBus, transcript: Transcript, monkeypatch) -> None:
    monkeypatch.setattr(assistant_module, "SCREEN_QUESTION_MIN_INTERVAL_S", 0.05)
    cfg = Config()
    cfg.screen.answer_screen_questions = True
    cfg.screen.auto_suggest = False  # isolate the screen-question path from auto-suggest
    client = FakeClient(token_batches=[["ok"]])
    assistant = _make_assistant(cfg, bus, transcript, client)
    sub = bus.subscribe()
    assistant.start()
    try:
        question = "Can you explain how your rate limiter design handles bursts?"
        bus.publish(make_screen_event(lines=(question,)))
        events = drain_until(sub, lambda e: isinstance(e, AssistDoneEvent))
        started = next(e for e in events if isinstance(e, AssistStartedEvent))
        assert started.kind == "screen"

        time.sleep(0.15)  # past the (monkeypatched) throttle
        bus.publish(make_screen_event(lines=(question + "?",)))  # same question, extra punctuation
        time.sleep(0.3)
        extra = drain(sub)
        assert not any(isinstance(e, AssistStartedEvent) and e.kind == "screen" for e in extra)
    finally:
        assistant.stop()


def test_mic_statement_and_short_line_do_not_trigger_screen_job(
    bus: EventBus, transcript: Transcript, monkeypatch
) -> None:
    monkeypatch.setattr(assistant_module, "SCREEN_QUESTION_MIN_INTERVAL_S", 0.05)
    cfg = Config()
    cfg.screen.answer_screen_questions = True
    cfg.screen.auto_suggest = False  # isolate the screen-question path from auto-suggest
    client = FakeClient()
    assistant = _make_assistant(cfg, bus, transcript, client)
    sub = bus.subscribe()
    assistant.start()
    try:
        bus.publish(make_screen_event(lines=("Me: I think that's a genuinely great idea honestly.",)))
        time.sleep(0.15)
        bus.publish(make_screen_event(lines=("What now?",)))  # 2 words, below min_words=6
        time.sleep(0.3)
        assert not any(isinstance(e, AssistStartedEvent) for e in drain(sub))
        assert client.calls == []
    finally:
        assistant.stop()


def test_answer_screen_questions_disabled_never_triggers(bus: EventBus, transcript: Transcript) -> None:
    cfg = Config()
    cfg.screen.answer_screen_questions = False
    cfg.screen.auto_suggest = False  # isolate the screen-question path from auto-suggest
    client = FakeClient()
    assistant = _make_assistant(cfg, bus, transcript, client)
    sub = bus.subscribe()
    assistant.start()
    try:
        bus.publish(make_screen_event(lines=("Can you explain how the retry policy works here?",)))
        time.sleep(0.3)
        assert not any(isinstance(e, AssistStartedEvent) for e in drain(sub))
    finally:
        assistant.stop()


# --------------------------------------------------------------------------------------------
# (f) lifecycle: stop() leaves no threads/timers alive even with jobs queued in both lanes
# --------------------------------------------------------------------------------------------


def test_stop_leaves_no_assistant_threads_alive_with_suggest_lane_active(
    bus: EventBus, transcript: Transcript
) -> None:
    cfg = Config()
    cfg.assist.coalesce_ms = 60_000  # long enough that it can't fire during the test
    transcript.add(make_segment(1, text="What is the plan?", t_end=1.0, is_question=True))
    client = FakeClient(token_batches=[list("abcdef")], chunk_delay=0.05)
    assistant = _make_assistant(cfg, bus, transcript, client)
    assistant.start()
    assistant.ask("hello")           # running, slow stream
    assistant.suggest_now("focus")   # queued behind it, never gets to run
    assistant._arm_coalesce_timer()  # exercise the coalesce timer too

    assistant.stop()

    names = {t.name for t in threading.enumerate()}
    for dead_name in ("assist-worker", "notes-worker", "bus-listener", "notes-auto-timer", "assist-coalesce"):
        assert dead_name not in names


def test_assistant_can_be_restarted_after_stop_with_suggest_lane(bus: EventBus, transcript: Transcript) -> None:
    cfg = Config()
    client = FakeClient(token_batches=[["first"], ["second"]])
    assistant = _make_assistant(cfg, bus, transcript, client)
    sub = bus.subscribe()

    assistant.start()
    try:
        assistant.suggest_now()
        events = drain_until(sub, lambda e: isinstance(e, AssistDoneEvent))
        assert next(e for e in events if isinstance(e, AssistDoneEvent)).text == "first"
    finally:
        assistant.stop()

    assistant.start()
    try:
        assistant.suggest_now()
        events = drain_until(sub, lambda e: isinstance(e, AssistDoneEvent))
        assert next(e for e in events if isinstance(e, AssistDoneEvent)).text == "second"
    finally:
        assistant.stop()


def test_irrelevant_screen_question_is_ignored_and_screen_job_never_cancels_running_solve(
    bus: EventBus, transcript: Transcript
) -> None:
    """LeetCode's poll ("Have you seen this in an interview before?") must not trigger an answer,
    and a real on-screen question must queue behind a running solve instead of cancelling it."""
    cfg = Config()
    cfg.screen.answer_screen_questions = True
    cfg.screen.auto_suggest = False
    client = FakeClient(token_batches=[list("solving..."), ["answer"]], chunk_delay=0.05)
    assistant = _make_assistant(cfg, bus, transcript, client)
    assistant.set_screen_provider(lambda: "Two Sum")
    sub = bus.subscribe()
    assistant.start()
    try:
        bus.publish(make_screen_event(lines=("Q - Have you seen this in an interview before?",)))
        time.sleep(0.3)
        assert not any(isinstance(e, AssistStartedEvent) for e in drain(sub))
        # a solve is running; a relevant on-screen question arrives
        assistant.solve()
        events = drain_until(sub, lambda e: isinstance(e, AssistStartedEvent) and e.kind == "solve")
        solve_id = next(e.job_id for e in events if isinstance(e, AssistStartedEvent))
        bus.publish(make_screen_event(lines=("What is the time complexity of the two-pointer approach here?",)))
        done = drain_until(sub, lambda e: isinstance(e, AssistDoneEvent) and e.job_id == solve_id)
        assert not done[-1].cancelled and done[-1].text == "solving..."
        started = drain_until(sub, lambda e: isinstance(e, AssistStartedEvent) and e.kind == "screen")
        assert any(isinstance(e, AssistStartedEvent) and e.kind == "screen" for e in started)
    finally:
        assistant.stop()


# =====================================================================================
# Multiple-choice questions (exocortex.llm.choices wired into the assistant)
# =====================================================================================
MCQ_SCREEN = "Question 3 of 20\nWhat is 2 ** 2?\n1. 4\n2. 8\n3. 16\n4. 32\nSubmit"
PRIME_SCREEN = "Which of the following is a prime number?\nA. 4\nB. 6\nC. 7\nD. 9\nSubmit"


def _run_solve(cfg: Config, bus: EventBus, transcript: Transcript, client: FakeClient, screen: str) -> tuple[Assistant, list[Any]]:
    """Start `assistant`, solve `screen`, and return it (still running) with the events up to Done."""
    assistant = _make_assistant(cfg, bus, transcript, client)
    assistant.set_screen_provider(lambda: screen)
    sub = bus.subscribe()
    assistant.start()
    assistant.solve()
    events = drain_until(sub, lambda e: isinstance(e, AssistDoneEvent))
    return assistant, events


def test_solve_on_screen_mcq_gets_the_choices_block_and_ends_with_the_verdict(bus: EventBus, transcript: Transcript) -> None:
    client = FakeClient(token_batches=[["2 squared is 4, ", "so option 1."]], chunk_delay=0.0)
    assistant, events = _run_solve(Config(), bus, transcript, client, MCQ_SCREEN)
    try:
        done = events[-1]
        assert done.text == "2 squared is 4, so option 1.\n\n**Answer: 1 — 4**"
        # the verdict is streamed too (the headless console only prints deltas), after the answer
        deltas = [e.delta for e in events if isinstance(e, AssistDeltaEvent)]
        assert deltas == ["2 squared is 4, ", "so option 1.", "\n\n**Answer: 1 — 4**"]
        assert assistant.answers[-1][2] == done.text
        prompt = client.calls[-1]["prompt"]
        assert "<choices>" in prompt and "1: 4\n2: 8\n3: 16\n4: 32" in prompt
        # the screen block carries "Choice 1: 4", never the decimal-looking "1. 4"
        assert "Choice 1: 4\nChoice 2: 8" in prompt and "\n1. 4\n" not in prompt
        assert "Submit\n</screen>" in prompt
        # extra room for the mandatory last line (on top of the default comment level's headroom)
        assert client.calls[-1]["max_tokens"] == (
            Config().llm.solve_max_tokens + assistant_module.COMMENT_TOKEN_HEADROOM["normal"]
            + assistant_module.CHOICES_TOKEN_HEADROOM
        )
    finally:
        assistant.stop()


def test_model_answer_line_is_normalised_not_duplicated(bus: EventBus, transcript: Transcript) -> None:
    client = FakeClient(token_batches=[["2 squared, which is 1.4.\nAnswer: 1.4"]], chunk_delay=0.0)
    assistant, events = _run_solve(Config(), bus, transcript, client, MCQ_SCREEN)
    try:
        assert events[-1].text == "2 squared, which is 1.4.\n\n**Answer: 1 — 4**"
        assert events[-1].text.count("Answer:") == 1
    finally:
        assistant.stop()


def test_verdict_is_streamed_even_when_the_model_wrote_the_plain_line(bus: EventBus, transcript: Transcript) -> None:
    """A model that follows the instruction to the letter ends with "Answer: 1 — 4"; the console
    only shows deltas, so the bold verdict is streamed anyway (the Done text carries it once)."""
    client = FakeClient(token_batches=[["2 squared is 4.\nAnswer: 1 — 4"]], chunk_delay=0.0)
    assistant, events = _run_solve(Config(), bus, transcript, client, MCQ_SCREEN)
    try:
        assert events[-1].text == "2 squared is 4.\n\n**Answer: 1 — 4**"
        deltas = [e.delta for e in events if isinstance(e, AssistDeltaEvent)]
        assert deltas[-1] == "\n\n**Answer: 1 — 4**"
        client2 = FakeClient(token_batches=[["2 squared is 4.\n\n**Answer: 1 — 4**"]], chunk_delay=0.0)
        assistant.stop()
        assistant2, events2 = _run_solve(Config(), bus, transcript, client2, MCQ_SCREEN)
        try:
            assert events2[-1].text == "2 squared is 4.\n\n**Answer: 1 — 4**"
            assert not any(e.delta.startswith("\n\n**Answer") for e in events2 if isinstance(e, AssistDeltaEvent))
        finally:
            assistant2.stop()
    finally:
        assistant.stop()


def test_skip_marker_with_an_mcq_on_screen_is_still_a_quiet_skip(bus: EventBus, transcript: Transcript) -> None:
    client = FakeClient(token_batches=[["(nothing to do)"]], chunk_delay=0.0)
    assistant, events = _run_solve(Config(), bus, transcript, client, MCQ_SCREEN)
    try:
        assert events[-1].cancelled and events[-1].text == ""
        assert assistant.answers == []
    finally:
        assistant.stop()


def test_fork_answer_is_left_alone_even_with_an_mcq_on_screen(bus: EventBus, transcript: Transcript) -> None:
    fork = "[[CHOICE]] Solve in Python\n[[CHOICE]] Solve in C++"
    client = FakeClient(token_batches=[[fork]], chunk_delay=0.0)
    assistant, events = _run_solve(Config(), bus, transcript, client, MCQ_SCREEN)
    try:
        assert events[-1].text == fork
    finally:
        assistant.stop()


def test_cancelled_mcq_job_is_not_finalised(bus: EventBus, transcript: Transcript) -> None:
    client = FakeClient(token_batches=[["tok "] * 60], chunk_delay=0.02)
    assistant = _make_assistant(Config(), bus, transcript, client)
    assistant.set_screen_provider(lambda: MCQ_SCREEN)
    sub = bus.subscribe()
    assistant.start()
    try:
        assistant.solve()
        drain_until(sub, lambda e: isinstance(e, AssistDeltaEvent))
        assistant._cancel_running_of(("solve",))
        events = drain_until(sub, lambda e: isinstance(e, AssistDoneEvent))
        assert events[-1].cancelled
        assert "Answer:" not in events[-1].text
        assert assistant.answers == []
    finally:
        assistant.stop()


def test_ask_with_options_in_the_question_annotates_the_question(bus: EventBus, transcript: Transcript) -> None:
    client = FakeClient(token_batches=[["7 is prime.\nAnswer: C"]], chunk_delay=0.0)
    assistant = _make_assistant(Config(), bus, transcript, client)
    sub = bus.subscribe()
    assistant.start()
    try:
        assistant.ask("Which is prime? A) 4 B) 6 C) 7")
        events = drain_until(sub, lambda e: isinstance(e, AssistDoneEvent))
        assert events[-1].text == "7 is prime.\n\n**Answer: C — 7**"
        prompt = client.calls[-1]["prompt"]
        assert "<question>\nWhich is prime?\nChoice A: 4\nChoice B: 6\nChoice C: 7\n</question>" in prompt
        assert "<choices>" in prompt and "A: 4\nB: 6\nC: 7" in prompt
        assert client.calls[-1]["max_tokens"] == Config().llm.ask_max_tokens + assistant_module.CHOICES_TOKEN_HEADROOM
        assert assistant.answers[-1][1] == "Which is prime? A) 4 B) 6 C) 7"   # the history keeps the raw question
    finally:
        assistant.stop()


def test_ask_unrelated_to_the_screen_ignores_an_on_screen_quiz(bus: EventBus, transcript: Transcript) -> None:
    cfg = Config()
    assert cfg.screen.use_in_answers is True
    client = FakeClient(token_batches=[["Network Address Translation."]], chunk_delay=0.0)
    assistant = _make_assistant(cfg, bus, transcript, client)
    assistant.set_screen_provider(lambda: MCQ_SCREEN)
    sub = bus.subscribe()
    assistant.start()
    try:
        assistant.ask("remind me what NAT stands for")
        events = drain_until(sub, lambda e: isinstance(e, AssistDoneEvent))
        assert events[-1].text == "Network Address Translation."
        prompt = client.calls[-1]["prompt"]
        assert "<choices>\nThis is" not in prompt and "\n1. 4\n" in prompt      # the screen is passed verbatim
    finally:
        assistant.stop()


def test_ask_referring_to_the_on_screen_options_uses_them(bus: EventBus, transcript: Transcript) -> None:
    client = FakeClient(token_batches=[["2 squared is 4, so the first one."]], chunk_delay=0.0)
    assistant = _make_assistant(Config(), bus, transcript, client)
    assistant.set_screen_provider(lambda: MCQ_SCREEN)
    sub = bus.subscribe()
    assistant.start()
    try:
        assistant.ask("which one is it?")
        events = drain_until(sub, lambda e: isinstance(e, AssistDoneEvent))
        assert "<choices>\nThis is" in client.calls[-1]["prompt"]
        assert events[-1].text.endswith("**Answer: 1 — 4**")
    finally:
        assistant.stop()


def test_on_screen_question_with_options_ends_with_the_verdict(bus: EventBus, transcript: Transcript) -> None:
    cfg = Config()
    cfg.screen.answer_screen_questions = True
    cfg.screen.auto_suggest = False
    client = FakeClient(token_batches=[["4, 6 and 9 are composite, so it is C."]], chunk_delay=0.0)
    assistant = _make_assistant(cfg, bus, transcript, client)
    assistant.set_screen_provider(lambda: PRIME_SCREEN)
    sub = bus.subscribe()
    assistant.start()
    try:
        bus.publish(make_screen_event(lines=tuple(PRIME_SCREEN.split("\n"))))
        events = drain_until(sub, lambda e: isinstance(e, AssistDoneEvent))
        started = next(e for e in events if isinstance(e, AssistStartedEvent))
        assert started.kind == "screen"
        assert events[-1].text == "4, 6 and 9 are composite, so it is C.\n\n**Answer: C — 7**"
        prompt = client.calls[-1]["prompt"]
        assert "Choice A: 4\nChoice B: 6\nChoice C: 7\nChoice D: 9" in prompt
        # the de-dup memory keys off the raw question line, not any annotated copy
        assert assistant._screen_answered_questions == [assistant_module._normalize_question(PRIME_SCREEN.split("\n")[0])]
    finally:
        assistant.stop()


def test_plan_keeps_the_raw_answer_and_raw_screen_lines(bus: EventBus, transcript: Transcript) -> None:
    """Explanations on: the verdict line goes to the bus and the history, never into the plan the
    completer follows; the plan's screen fingerprint is the raw OCR text."""
    cfg = Config()
    cfg.explain.enabled = True
    raw = "Walkthrough:\n2 squared is 4.\n\nIt is 4.\nAnswer: 1"
    client = FakeClient(token_batches=[[raw]], chunk_delay=0.0)
    assistant, events = _run_solve(cfg, bus, transcript, client, MCQ_SCREEN)
    try:
        assert events[-1].text == "Walkthrough:\n2 squared is 4.\n\nIt is 4.\n\n**Answer: 1 — 4**"
        view = assistant.plan()
        assert view is not None and view.done
        assert "**Answer" not in view.solution and view.solution == "It is 4.\nAnswer: 1"
        assert assistant.plan_walkthrough == "2 squared is 4."
        assert "1. 4" in assistant._plan.screen_lines and "choice 1: 4" not in assistant._plan.screen_lines
    finally:
        assistant.stop()


def test_model_skip_marker_is_dropped(bus: EventBus, transcript: Transcript) -> None:
    cfg = Config()
    client = FakeClient(token_batches=[["(nothing to do)"]], chunk_delay=0.0)
    assistant = _make_assistant(cfg, bus, transcript, client)
    assistant.set_screen_provider(lambda: "chrome")
    sub = bus.subscribe()
    assistant.start()
    try:
        assistant.solve()
        events = drain_until(sub, lambda e: isinstance(e, AssistDoneEvent))
        assert events[-1].cancelled and events[-1].text == ""
        assert assistant.answers == []
    finally:
        assistant.stop()

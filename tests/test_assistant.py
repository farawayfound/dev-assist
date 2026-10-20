"""Tests for exocortex.llm.assistant.Assistant.

Fully offline: a FakeClient (duck-typed, no httpx/LMStudioClient subclassing) stands in for
exocortex.llm.client.LMStudioClient, and a real EventBus + real Transcript are used throughout so
the tests exercise the actual prompt builders (exocortex.llm.prompts) and question heuristics
(exocortex.llm.questions) the module is built on.

The one @pytest.mark.live test talks to a real LM Studio server at http://127.0.0.1:1234 with
model "maple-preview" and is skipped by default (see pytest.ini).
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
    AutoAssistChangedEvent,
    ComponentStatusEvent,
    EventBus,
    FinalTranscriptEvent,
    NotesErrorEvent,
    NotesUpdatedEvent,
    Source,
    TranscriptSegment,
    drain,
)
from exocortex.config import Config
from exocortex.llm import prompts
from exocortex.llm.assistant import Assistant, _AssistJob, _JobSlot
from exocortex.llm.client import ChatDelta, Health, LLMError, ModelInfo, StreamStats
from exocortex.transcript import Transcript

# --------------------------------------------------------------------------------------------
# test doubles / helpers
# --------------------------------------------------------------------------------------------


class FakeClient:
    """Duck-typed stand-in for LMStudioClient. No httpx, no subclassing.

    `token_batches` is consumed one batch per stream_completion/stream_chat call, in order (a
    batch is a list of either plain strings -- the visible content -- or `(content, reasoning)`
    tuples when the test needs to script reasoning text for a `stream_chat` call). Each item is
    yielded after a `chunk_delay` sleep so cancellation has time to land mid-stream, and
    `cancel.is_set()` is honoured exactly like the real client. `last_stats` is thread-local, like
    the real client's, since assist-worker and notes-worker are different threads.
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


@pytest.fixture()
def bus() -> EventBus:
    return EventBus()


@pytest.fixture()
def transcript() -> Transcript:
    return Transcript()


# --------------------------------------------------------------------------------------------
# manual answer_last() / ask()
# --------------------------------------------------------------------------------------------


def test_answer_last_publishes_lifecycle_and_builds_prompt_correctly(bus: EventBus, transcript: Transcript) -> None:
    cfg = Config()
    transcript.add(make_segment(1, text="We need to decide on Kafka versus RabbitMQ.", t_start=0.0, t_end=3.0))
    transcript.add(
        make_segment(2, text="What would you say is the main trade-off?", t_start=4.0, t_end=6.0, is_question=True)
    )
    client = FakeClient(token_batches=[["- Kafka handles", " throughput better."]])
    assistant = Assistant(cfg, bus, client, transcript, clock=time.monotonic)
    sub = bus.subscribe()
    assistant.start()
    try:
        assistant.answer_last()

        events = drain_until(sub, lambda e: isinstance(e, AssistDoneEvent))
        started = next(e for e in events if isinstance(e, AssistStartedEvent))
        deltas = [e for e in events if isinstance(e, AssistDeltaEvent)]
        done = next(e for e in events if isinstance(e, AssistDoneEvent))

        assert started.kind == "manual"
        assert started.job_id == done.job_id
        assert [d.delta for d in deltas] == ["- Kafka handles", " throughput better."]
        assert done.text == "- Kafka handles throughput better."
        assert done.cancelled is False

        call = client.calls[0]
        assert call["kind"] == "completion"
        assert call["prompt"].endswith(prompts.NO_THINK_PREFIX)
        assert "Kafka versus RabbitMQ" in call["prompt"]
        assert "main trade-off" in call["prompt"]
        assert call["max_tokens"] == cfg.llm.assist_max_tokens

        answers = assistant.answers
        assert len(answers) == 1
        assert answers[0][2] == done.text
    finally:
        assistant.stop()


def test_ask_uses_ask_max_tokens_and_system_ask(bus: EventBus, transcript: Transcript) -> None:
    cfg = Config()
    client = FakeClient(token_batches=[["Sure thing."]])
    assistant = Assistant(cfg, bus, client, transcript, clock=time.monotonic)
    sub = bus.subscribe()
    assistant.start()
    try:
        assistant.ask("What's our budget?")

        events = drain_until(sub, lambda e: isinstance(e, AssistDoneEvent))
        started = next(e for e in events if isinstance(e, AssistStartedEvent))

        assert started.kind == "ask"
        assert started.question == "What's our budget?"
        call = client.calls[0]
        assert call["max_tokens"] == cfg.llm.ask_max_tokens
        assert "just typed you a question of their own" in call["prompt"]  # unique to system_ask
        assert "What's our budget?" in call["prompt"]
    finally:
        assistant.stop()


def test_answer_last_with_no_system_segments_is_a_noop(bus: EventBus, transcript: Transcript) -> None:
    cfg = Config()
    client = FakeClient()
    assistant = Assistant(cfg, bus, client, transcript, clock=time.monotonic)
    sub = bus.subscribe()
    assistant.start()
    try:
        assistant.answer_last()
        time.sleep(0.2)
        assert not any(isinstance(e, AssistStartedEvent) for e in drain(sub))
        assert client.calls == []
    finally:
        assistant.stop()


# --------------------------------------------------------------------------------------------
# cancellation: a new job cancels the running one
# --------------------------------------------------------------------------------------------


def test_new_job_cancels_the_running_one(bus: EventBus, transcript: Transcript) -> None:
    cfg = Config()
    transcript.add(make_segment(1, text="Tell me about the migration plan.", t_end=1.0, is_question=True))
    client = FakeClient(token_batches=[list("abcdefgh"), ["x", "y"]], chunk_delay=0.05)
    assistant = Assistant(cfg, bus, client, transcript, clock=time.monotonic)
    sub = bus.subscribe()
    assistant.start()
    try:
        assistant.answer_last()  # job 1: long stream
        drain_until(sub, lambda e: isinstance(e, AssistStartedEvent))
        time.sleep(0.12)  # let a couple of chunks flow before superseding it
        assistant.ask("second question")  # job 2: supersedes job 1

        first_batch = drain_until(sub, lambda e: isinstance(e, AssistDoneEvent))
        first_done = next(e for e in first_batch if isinstance(e, AssistDoneEvent))
        second_batch = drain_until(sub, lambda e: isinstance(e, AssistDoneEvent))
        second_done = next(e for e in second_batch if isinstance(e, AssistDoneEvent))

        assert first_done.job_id != second_done.job_id
        assert first_done.cancelled is True
        assert second_done.cancelled is False
        assert second_done.text == "xy"
        # the cancelled job's partial text must not be recorded as an answer
        assert [a[2] for a in assistant.answers] == ["xy"]
    finally:
        assistant.stop()


# --------------------------------------------------------------------------------------------
# auto mode: coalescing, MIC exclusion, no double-answering
# --------------------------------------------------------------------------------------------


def test_auto_mode_combines_two_questions_into_one_job(bus: EventBus, transcript: Transcript) -> None:
    cfg = Config()
    cfg.assist.coalesce_ms = 60
    cfg.assist.auto_answer_questions = True
    client = FakeClient(token_batches=[["ok"]])
    assistant = Assistant(cfg, bus, client, transcript, clock=time.monotonic)
    sub = bus.subscribe()
    assistant.start()
    try:
        seg1 = make_segment(1, text="Quick question.", t_end=10.0, is_question=True)
        transcript.add(seg1)
        bus.publish(FinalTranscriptEvent(segment=seg1))
        time.sleep(0.02)  # well inside the coalesce window -> restarts, doesn't double-fire
        seg2 = make_segment(2, text="What's the timeline for the migration?", t_end=10.4, is_question=True)
        transcript.add(seg2)
        bus.publish(FinalTranscriptEvent(segment=seg2))

        events = drain_until(sub, lambda e: isinstance(e, AssistStartedEvent) and e.kind == "auto")
        started = next(e for e in events if isinstance(e, AssistStartedEvent))
        assert "Quick question." in started.question
        assert "What's the timeline for the migration?" in started.question

        # exactly one auto job ever fires for these two segments
        time.sleep(0.3)
        extra = drain(sub)
        assert not any(isinstance(e, AssistStartedEvent) and e.kind == "auto" for e in extra)
    finally:
        assistant.stop()


def test_auto_mode_never_answers_the_same_segment_twice(bus: EventBus, transcript: Transcript) -> None:
    cfg = Config()
    cfg.assist.coalesce_ms = 50
    client = FakeClient(token_batches=[["ok"]])
    assistant = Assistant(cfg, bus, client, transcript, clock=time.monotonic)
    sub = bus.subscribe()
    assistant.start()
    try:
        seg = make_segment(1, text="What is the plan?", t_end=1.0, is_question=True)
        transcript.add(seg)
        bus.publish(FinalTranscriptEvent(segment=seg))

        events = drain_until(sub, lambda e: isinstance(e, AssistStartedEvent) and e.kind == "auto")
        assert sum(1 for e in events if isinstance(e, AssistStartedEvent) and e.kind == "auto") == 1

        bus.publish(FinalTranscriptEvent(segment=seg))  # redelivery of the same final
        time.sleep(0.3)
        extra = drain(sub)
        assert not any(isinstance(e, AssistStartedEvent) and e.kind == "auto" for e in extra)
    finally:
        assistant.stop()


def test_mic_questions_never_trigger_auto(bus: EventBus, transcript: Transcript) -> None:
    cfg = Config()
    cfg.assist.coalesce_ms = 50
    client = FakeClient()
    assistant = Assistant(cfg, bus, client, transcript, clock=time.monotonic)
    sub = bus.subscribe()
    assistant.start()
    try:
        seg = make_segment(1, source=Source.MIC, text="What time is it?", is_question=True, speaker_id="me", speaker_label="Me")
        transcript.add(seg)
        bus.publish(FinalTranscriptEvent(segment=seg))
        time.sleep(0.3)
        assert not any(isinstance(e, AssistStartedEvent) for e in drain(sub))
    finally:
        assistant.stop()


def test_set_auto_publishes_auto_assist_changed_event(bus: EventBus, transcript: Transcript) -> None:
    cfg = Config()
    client = FakeClient()
    assistant = Assistant(cfg, bus, client, transcript, clock=time.monotonic)
    sub = bus.subscribe()
    assistant.start()
    try:
        assistant.set_auto(False)
        assert assistant.auto_enabled is False
        events = drain_until(sub, lambda e: isinstance(e, AutoAssistChangedEvent))
        change = next(e for e in events if isinstance(e, AutoAssistChangedEvent))
        assert change.enabled is False
    finally:
        assistant.stop()


# --------------------------------------------------------------------------------------------
# notes
# --------------------------------------------------------------------------------------------


def test_generate_notes_force_true_publishes_notes_updated_with_exact_headings(
    bus: EventBus, transcript: Transcript
) -> None:
    cfg = Config()
    transcript.add(make_segment(1, text="We discussed the roadmap.", t_end=1.0))
    client = FakeClient(token_batches=[["## Summary\nStuff.\n"]])
    assistant = Assistant(cfg, bus, client, transcript, clock=time.monotonic)
    sub = bus.subscribe()
    assistant.start()
    try:
        assistant.generate_notes(force=True)

        events = drain_until(sub, lambda e: isinstance(e, NotesUpdatedEvent))
        updated = next(e for e in events if isinstance(e, NotesUpdatedEvent))
        assert updated.markdown == "## Summary\nStuff."
        assert assistant.notes_markdown == "## Summary\nStuff."

        prompt = client.calls[0]["prompt"]
        for heading in ("## Summary", "## Key points", "## Decisions", "## Action items", "## Open questions"):
            assert heading in prompt
    finally:
        assistant.stop()


def test_generate_notes_force_false_skipped_when_too_few_new_words(bus: EventBus, transcript: Transcript) -> None:
    cfg = Config()
    cfg.notes.min_new_words = 1000
    transcript.add(make_segment(1, text="short text", t_end=1.0))
    client = FakeClient()
    assistant = Assistant(cfg, bus, client, transcript, clock=time.monotonic)
    sub = bus.subscribe()
    assistant.start()
    try:
        assistant.generate_notes(force=False)
        time.sleep(0.3)
        assert not any(isinstance(e, (NotesUpdatedEvent, NotesErrorEvent)) for e in drain(sub))
        assert client.calls == []
    finally:
        assistant.stop()


def test_generate_notes_uses_whole_transcript_the_first_time(bus: EventBus, transcript: Transcript) -> None:
    cfg = Config()
    transcript.add(make_segment(1, text="the very first thing said", t_end=1.0))
    client = FakeClient(token_batches=[["## Summary\nok\n"]])
    assistant = Assistant(cfg, bus, client, transcript, clock=time.monotonic)
    sub = bus.subscribe()
    assistant.start()
    try:
        assistant.generate_notes(force=True)
        drain_until(sub, lambda e: isinstance(e, NotesUpdatedEvent))
        assert "the very first thing said" in client.calls[0]["prompt"]
    finally:
        assistant.stop()


def test_notes_backlog_is_processed_oldest_first_in_bounded_chunks(bus: EventBus, transcript: Transcript) -> None:
    """upstream finding 3 (ported from mac-cortex, adjusted for Windows): the notes job's
    transcript slice is bounded to cfg.llm.context_tokens * 4 chars -- as the OLDEST chunk that
    fits (mac-cortex keeps the newest and silently skips the rest), with the checkpoint advancing
    exactly to the last segment sent and the next chunk queued at once, so a backlog is processed
    in order and nothing is skipped."""
    cfg = Config()
    cfg.llm.context_tokens = 50  # a 200-char budget: a few segments per chunk
    for i in range(30):
        transcript.add(make_segment(i, text=f"the {i:02d}th thing that was said in the meeting", t_end=float(i + 1)))
    client = FakeClient(token_batches=[["## Summary\nok\n"]])   # later calls fall back to "ok"
    assistant = Assistant(cfg, bus, client, transcript, clock=time.monotonic)
    sub = bus.subscribe()
    assistant.start()
    try:
        assistant.generate_notes(force=True)
        drain_until(sub, lambda e: isinstance(e, NotesUpdatedEvent) and assistant._notes_last_id == 29, timeout=10.0)
        prompts_sent = [c["prompt"] for c in client.calls]
        assert len(prompts_sent) >= 2, "a backlog bigger than one chunk must take several calls"
        assert "the 00th thing" in prompts_sent[0] and "the 29th thing" not in prompts_sent[0], "oldest first"
        assert "the 29th thing" in prompts_sent[-1]
        for prompt in prompts_sent:
            assert prompt.count("th thing that was said") <= 5, "every chunk stays within the budget"
        joined = "".join(prompts_sent)
        positions = [joined.index(f"the {i:02d}th thing") for i in range(30)]
        assert positions == sorted(positions), "nothing skipped, everything in order"
    finally:
        assistant.stop()


def test_notes_backlog_after_a_failure_is_not_skipped(bus: EventBus, transcript: Transcript) -> None:
    """upstream finding 3 (Windows fix): the scenario the audit's reviewers worried about -- a
    cycle fails, a long stretch of transcript piles up, then the LLM recovers. Neither the content
    of the failed cycle nor the backlog may be silently skipped: the recovery processes everything
    from the untouched checkpoint onward, oldest first, in bounded chunks."""
    cfg = Config()
    cfg.llm.context_tokens = 50
    transcript.add(make_segment(0, text="the very first thing said before the outage", t_end=1.0))
    client = FakeClient(token_batches=[["## Summary\nok\n"]])
    client.raise_error = LLMError("boom")
    assistant = Assistant(cfg, bus, client, transcript, clock=time.monotonic)
    sub = bus.subscribe()
    assistant.start()
    try:
        assistant.generate_notes(force=True)
        drain_until(sub, lambda e: isinstance(e, NotesErrorEvent))
        failed_calls = len(client.calls)
        for i in range(1, 31):
            transcript.add(make_segment(i, text=f"the {i:02d}th thing said during the outage", t_end=float(i + 1)))
        client.raise_error = None
        assistant.generate_notes(force=True)
        drain_until(sub, lambda e: isinstance(e, NotesUpdatedEvent) and assistant._notes_last_id == 30, timeout=10.0)
        recovery = [c["prompt"] for c in client.calls[failed_calls:]]
        assert "the very first thing said before the outage" in recovery[0]
        joined = "".join(recovery)
        positions = [joined.index(f"the {i:02d}th thing said during the outage") for i in range(1, 31)]
        assert positions == sorted(positions), "the backlog must be processed in order, nothing skipped"
    finally:
        assistant.stop()


def test_notes_checkpoint_is_not_advanced_after_an_llm_error(bus: EventBus, transcript: Transcript) -> None:
    """upstream finding 3 (Windows fix): a failed notes cycle must leave `_notes_last_id` alone,
    so the content spoken during a transient failure (backend hiccup, timeout, ...) is retried
    -- and not silently dropped from the notes forever -- once the failure clears. The bounded
    chunk (`Transcript.head_text`, the actual fix for finding 3's unbounded-prompt claim) already
    guarantees the retried prompt cannot grow past `context_tokens * 4` chars no matter how long
    the checkpoint stays put, so
    advancing it on an ordinary failure would only lose content without preventing anything."""
    cfg = Config()
    transcript.add(make_segment(1, text="first sentence before the failure", t_end=1.0))
    client = FakeClient()
    client.raise_error = LLMError("boom")
    assistant = Assistant(cfg, bus, client, transcript, clock=time.monotonic)
    sub = bus.subscribe()
    assistant.start()
    try:
        assistant.generate_notes(force=True)
        drain_until(sub, lambda e: isinstance(e, NotesErrorEvent))

        client.raise_error = None
        client._batches = [["## Summary\nok\n"]]
        transcript.add(make_segment(2, text="second sentence after the failure", t_end=2.0))
        assistant.generate_notes(force=True)
        drain_until(sub, lambda e: isinstance(e, NotesUpdatedEvent))

        prompt = client.calls[-1]["prompt"]
        assert "second sentence after the failure" in prompt
        assert "first sentence before the failure" in prompt, (
            "the checkpoint must NOT have advanced past segment 1 on the earlier failure, or its "
            "content is silently and permanently lost from the notes"
        )
    finally:
        assistant.stop()


def test_notes_checkpoint_is_not_advanced_after_an_unexpected_error(bus: EventBus, transcript: Transcript) -> None:
    """upstream finding 3 (Windows fix): same guarantee as the LLMError case above, but for the
    generic `except Exception` branch (an arbitrary bug), which must not be folded into a
    'discard this window forever' behavior either -- it is even less likely to be caused by
    prompt size than an `LLMError` is."""
    cfg = Config()
    transcript.add(make_segment(1, text="first sentence before the failure", t_end=1.0))
    client = FakeClient()
    client.raise_error = RuntimeError("unexpected bug")
    assistant = Assistant(cfg, bus, client, transcript, clock=time.monotonic)
    sub = bus.subscribe()
    assistant.start()
    try:
        assistant.generate_notes(force=True)
        drain_until(sub, lambda e: isinstance(e, NotesErrorEvent))

        client.raise_error = None
        client._batches = [["## Summary\nok\n"]]
        transcript.add(make_segment(2, text="second sentence after the failure", t_end=2.0))
        assistant.generate_notes(force=True)
        drain_until(sub, lambda e: isinstance(e, NotesUpdatedEvent))

        prompt = client.calls[-1]["prompt"]
        assert "second sentence after the failure" in prompt
        assert "first sentence before the failure" in prompt, (
            "the checkpoint must NOT have advanced past segment 1 on the earlier unexpected error"
        )
    finally:
        assistant.stop()


# --------------------------------------------------------------------------------------------
# thinking "on"
# --------------------------------------------------------------------------------------------


def test_thinking_on_uses_stream_chat_and_never_publishes_reasoning(bus: EventBus, transcript: Transcript) -> None:
    cfg = Config()
    cfg.llm.thinking = "on"
    transcript.add(make_segment(1, text="Some context here.", t_end=1.0))
    client = FakeClient(token_batches=[[("", "Let me think privately. "), ("Answer: ", ""), ("42", "")]])
    assistant = Assistant(cfg, bus, client, transcript, clock=time.monotonic)
    sub = bus.subscribe()
    assistant.start()
    try:
        assistant.ask("What is the answer?")

        events = drain_until(sub, lambda e: isinstance(e, AssistDoneEvent))
        done = next(e for e in events if isinstance(e, AssistDoneEvent))
        deltas = [e for e in events if isinstance(e, AssistDeltaEvent)]

        assert done.text == "Answer: 42"
        assert all("Let me think" not in d.delta for d in deltas)
        assert "Let me think" not in done.text

        call = client.calls[0]
        assert call["kind"] == "chat"
        assert call["messages"][0] == {"role": "system", "content": prompts.system_ask(cfg.assist.persona)}
    finally:
        assistant.stop()


# --------------------------------------------------------------------------------------------
# errors
# --------------------------------------------------------------------------------------------


def test_llm_error_publishes_assist_error_event(bus: EventBus, transcript: Transcript) -> None:
    cfg = Config()
    client = FakeClient()
    client.raise_error = LLMError("boom")
    assistant = Assistant(cfg, bus, client, transcript, clock=time.monotonic)
    sub = bus.subscribe()
    assistant.start()
    try:
        assistant.ask("hello?")
        events = drain_until(sub, lambda e: isinstance(e, AssistErrorEvent))
        err = next(e for e in events if isinstance(e, AssistErrorEvent))
        assert "boom" in err.message
    finally:
        assistant.stop()


def test_llm_error_in_notes_publishes_notes_error_event(bus: EventBus, transcript: Transcript) -> None:
    cfg = Config()
    transcript.add(make_segment(1, text="hello there", t_end=1.0))
    client = FakeClient()
    client.raise_error = LLMError("kaboom")
    assistant = Assistant(cfg, bus, client, transcript, clock=time.monotonic)
    sub = bus.subscribe()
    assistant.start()
    try:
        assistant.generate_notes(force=True)
        events = drain_until(sub, lambda e: isinstance(e, NotesErrorEvent))
        err = next(e for e in events if isinstance(e, NotesErrorEvent))
        assert "kaboom" in err.message
    finally:
        assistant.stop()


# --------------------------------------------------------------------------------------------
# lifecycle: start()/stop()
# --------------------------------------------------------------------------------------------


def test_start_health_check_publishes_ready_when_reachable_and_loaded(bus: EventBus, transcript: Transcript) -> None:
    cfg = Config()
    client = FakeClient()
    assistant = Assistant(cfg, bus, client, transcript, clock=time.monotonic)
    sub = bus.subscribe()
    assistant.start()
    try:
        events = drain_until(sub, lambda e: isinstance(e, ComponentStatusEvent))
        status = next(e for e in events if isinstance(e, ComponentStatusEvent))
        assert status.component == "llm"
        assert status.state == "ready"
    finally:
        assistant.stop()


def test_start_health_check_publishes_error_when_unreachable(bus: EventBus, transcript: Transcript) -> None:
    cfg = Config()
    client = FakeClient()
    client.health_result = Health(reachable=False, model_listed=False, model_loaded=None, detail="refused")
    assistant = Assistant(cfg, bus, client, transcript, clock=time.monotonic)
    sub = bus.subscribe()
    assistant.start()
    try:
        events = drain_until(sub, lambda e: isinstance(e, ComponentStatusEvent))
        status = next(e for e in events if isinstance(e, ComponentStatusEvent))
        assert status.state == "error"
        assert cfg.llm.base_url in status.detail
    finally:
        assistant.stop()


def test_start_health_check_loads_model_when_not_loaded(bus: EventBus, transcript: Transcript) -> None:
    cfg = Config()
    client = FakeClient()
    client.health_result = Health(reachable=True, model_listed=True, model_loaded=False, detail="")
    assistant = Assistant(cfg, bus, client, transcript, clock=time.monotonic)
    sub = bus.subscribe()
    assistant.start()
    try:
        events = drain_until(sub, lambda e: isinstance(e, ComponentStatusEvent) and e.state == "ready")
        states = [e.state for e in events if isinstance(e, ComponentStatusEvent)]
        assert "loading" in states
        assert states[-1] == "ready"
        assert client.ensure_loaded_calls == 1
    finally:
        assistant.stop()


def test_set_model_updates_cfg_and_client_and_rechecks_health(bus: EventBus, transcript: Transcript) -> None:
    cfg = Config()
    client = FakeClient()
    assistant = Assistant(cfg, bus, client, transcript, clock=time.monotonic)
    sub = bus.subscribe()
    assistant.start()
    try:
        drain_until(sub, lambda e: isinstance(e, ComponentStatusEvent))  # the start() health check
        assistant.set_model("other-model")
        assert cfg.llm.model == "other-model"
        assert client.cfg.model == "other-model"
        events = drain_until(sub, lambda e: isinstance(e, ComponentStatusEvent))
        assert any(isinstance(e, ComponentStatusEvent) for e in events)
    finally:
        assistant.stop()


def test_stop_leaves_no_assistant_threads_alive(bus: EventBus, transcript: Transcript) -> None:
    cfg = Config()
    client = FakeClient()
    assistant = Assistant(cfg, bus, client, transcript, clock=time.monotonic)
    assistant.start()
    assistant.stop()
    names = {t.name for t in threading.enumerate()}
    for dead_name in ("assist-worker", "notes-worker", "bus-listener", "notes-auto-timer", "assist-coalesce"):
        assert dead_name not in names


def test_cancel_clears_queued_job_without_running_it(bus: EventBus, transcript: Transcript) -> None:
    cfg = Config()
    transcript.add(make_segment(1, text="Tell me something.", t_end=1.0, is_question=True))
    client = FakeClient(token_batches=[list("abcdef"), ["never", "runs"]], chunk_delay=0.05)
    assistant = Assistant(cfg, bus, client, transcript, clock=time.monotonic)
    sub = bus.subscribe()
    assistant.start()
    try:
        assistant.answer_last()  # job 1: running
        drain_until(sub, lambda e: isinstance(e, AssistStartedEvent))
        assistant.ask("queued job")  # job 2: queued, superseding nothing
        time.sleep(0.02)
        assistant.cancel()  # cancels job 1, drops queued job 2

        events = drain_until(sub, lambda e: isinstance(e, AssistDoneEvent))
        done = next(e for e in events if isinstance(e, AssistDoneEvent))
        assert done.cancelled is True

        # job 2 must never start: give the worker time, then confirm no second Started/Done
        time.sleep(0.3)
        extra = drain(sub)
        assert not any(isinstance(e, (AssistStartedEvent, AssistDoneEvent)) for e in extra)
    finally:
        assistant.stop()


# --------------------------------------------------------------------------------------------
# _JobSlot: the shutdown race a bare put(None) sentinel is vulnerable to
# --------------------------------------------------------------------------------------------


def test_job_slot_put_after_close_is_a_noop_and_never_resurrects_the_worker() -> None:
    """A job submitted concurrently with (or after) close() must never overwrite the stop
    sentinel -- otherwise a worker loop's get() would return that job, run it, then block in
    get() forever since no further sentinel is coming (a permanent thread leak)."""
    slot: _JobSlot[str] = _JobSlot()
    slot.close()
    slot.put("late job, submitted right as shutdown was happening")
    assert slot.get() is None


def test_job_slot_reopen_allows_reuse_after_close() -> None:
    slot: _JobSlot[str] = _JobSlot()
    slot.close()
    slot.reopen()
    slot.put("job")
    assert slot.get() == "job"


# --------------------------------------------------------------------------------------------
# restart: stop() then start() again on the same instance
# --------------------------------------------------------------------------------------------


def test_assistant_can_be_restarted_after_stop(bus: EventBus, transcript: Transcript) -> None:
    cfg = Config()
    client = FakeClient(token_batches=[["first"], ["second"]])
    assistant = Assistant(cfg, bus, client, transcript, clock=time.monotonic)
    sub = bus.subscribe()

    assistant.start()
    try:
        assistant.ask("one")
        events = drain_until(sub, lambda e: isinstance(e, AssistDoneEvent))
        assert next(e for e in events if isinstance(e, AssistDoneEvent)).text == "first"
    finally:
        assistant.stop()

    # A closed _JobSlot must not leave the instance permanently unable to run new jobs.
    assistant.start()
    try:
        assistant.ask("two")
        events = drain_until(sub, lambda e: isinstance(e, AssistDoneEvent))
        assert next(e for e in events if isinstance(e, AssistDoneEvent)).text == "second"
    finally:
        assistant.stop()


# --------------------------------------------------------------------------------------------
# coalesce timer: the threading.Timer.cancel() race
# --------------------------------------------------------------------------------------------


def test_stale_coalesce_callback_does_not_fire_after_a_rearm(bus: EventBus, transcript: Transcript) -> None:
    """Simulates the threading.Timer.cancel() race directly: a timer callback that reads its
    generation number *after* a newer timer has already been armed (as if its wait elapsed and
    cancel() lost the race) must be a no-op, not a second/duplicate auto-answer."""
    cfg = Config()
    client = FakeClient()
    assistant = Assistant(cfg, bus, client, transcript, clock=time.monotonic)
    sub = bus.subscribe()
    assistant.start()
    try:
        transcript.add(make_segment(1, text="What is the plan?", t_end=1.0, is_question=True))
        assistant._arm_coalesce_timer()
        stale_generation = assistant._coalesce_generation
        assistant._arm_coalesce_timer()  # a follow-on question re-arms with a newer generation

        assistant._fire_auto(stale_generation)  # the stale (raced) callback fires anyway

        time.sleep(0.05)
        assert not any(isinstance(e, AssistStartedEvent) for e in drain(sub))
    finally:
        assistant.stop()


def test_coalesce_timer_is_cancelled_and_joined_by_stop(bus: EventBus, transcript: Transcript) -> None:
    cfg = Config()
    cfg.assist.coalesce_ms = 60_000  # long enough that it can't fire during the test
    client = FakeClient()
    assistant = Assistant(cfg, bus, client, transcript, clock=time.monotonic)
    assistant.start()
    transcript.add(make_segment(1, text="What is the plan?", t_end=1.0, is_question=True))
    assistant._arm_coalesce_timer()
    assert assistant._coalesce_timer is not None

    assistant.stop()

    assert "assist-coalesce" not in {t.name for t in threading.enumerate()}


# --------------------------------------------------------------------------------------------
# a job cancelled before it starts streaming must not waste an LLM request
# --------------------------------------------------------------------------------------------


def test_already_cancelled_job_skips_the_llm_call(bus: EventBus, transcript: Transcript) -> None:
    cfg = Config()
    client = FakeClient()
    assistant = Assistant(cfg, bus, client, transcript, clock=time.monotonic)
    sub = bus.subscribe()
    assistant.start()
    try:
        # Simulates cancel()/a newer submission landing between the worker claiming this job
        # and _run_assist_job actually starting to stream.
        job = _AssistJob(id=999, kind="manual", question="Tell me something.", cancel=threading.Event())
        job.cancel.set()
        assistant._run_assist_job(job)

        events = drain_until(sub, lambda e: isinstance(e, AssistDoneEvent))
        done = next(e for e in events if isinstance(e, AssistDoneEvent))
        assert done.job_id == 999
        assert done.cancelled is True
        assert done.text == ""
        assert client.calls == []
    finally:
        assistant.stop()


# --------------------------------------------------------------------------------------------
# cancel() must only ever touch the assist job, never a running/queued notes job
# --------------------------------------------------------------------------------------------


def test_cancel_does_not_affect_a_running_notes_job(bus: EventBus, transcript: Transcript) -> None:
    cfg = Config()
    transcript.add(make_segment(1, text="Some content to summarize today.", t_end=1.0))
    client = FakeClient(token_batches=[["## Summary\nok\n"]], chunk_delay=0.06)
    assistant = Assistant(cfg, bus, client, transcript, clock=time.monotonic)
    sub = bus.subscribe()
    assistant.start()
    try:
        assistant.generate_notes(force=True)
        time.sleep(0.02)  # let the notes job actually start streaming
        assistant.cancel()  # must be a no-op for the notes job

        events = drain_until(sub, lambda e: isinstance(e, NotesUpdatedEvent))
        updated = next(e for e in events if isinstance(e, NotesUpdatedEvent))
        assert updated.markdown == "## Summary\nok"
    finally:
        assistant.stop()


# --------------------------------------------------------------------------------------------
# the prompt window is a snapshot taken once at job start (KV prefix-cache stability)
# --------------------------------------------------------------------------------------------


def test_prompt_window_is_a_snapshot_taken_at_job_start(bus: EventBus, transcript: Transcript) -> None:
    cfg = Config()
    transcript.add(make_segment(1, text="Original question about staffing levels.", t_end=1.0, is_question=True))
    client = FakeClient(token_batches=[["a", "b", "c"]], chunk_delay=0.08)
    assistant = Assistant(cfg, bus, client, transcript, clock=time.monotonic)
    sub = bus.subscribe()
    assistant.start()
    try:
        assistant.answer_last()
        drain_until(sub, lambda e: isinstance(e, AssistStartedEvent))
        # Arrives while the job above is still streaming -- must not appear in ITS prompt.
        transcript.add(make_segment(2, text="A brand new unrelated sentence spoken mid-answer.", t_end=5.0))

        drain_until(sub, lambda e: isinstance(e, AssistDoneEvent))
        prompt = client.calls[0]["prompt"]
        assert "Original question about staffing" in prompt
        assert "brand new unrelated sentence" not in prompt
    finally:
        assistant.stop()


# --------------------------------------------------------------------------------------------
# cfg.llm.thinking is read fresh per job, not cached at construction
# --------------------------------------------------------------------------------------------


def test_thinking_mode_is_read_per_job(bus: EventBus, transcript: Transcript) -> None:
    cfg = Config()
    cfg.llm.thinking = "off"
    client = FakeClient(token_batches=[["first answer"], [("", "thinking..."), ("second answer", "")]])
    assistant = Assistant(cfg, bus, client, transcript, clock=time.monotonic)
    sub = bus.subscribe()
    assistant.start()
    try:
        assistant.ask("one")
        drain_until(sub, lambda e: isinstance(e, AssistDoneEvent))
        assert client.calls[0]["kind"] == "completion"

        cfg.llm.thinking = "on"  # flipped between jobs
        assistant.ask("two")
        events = drain_until(sub, lambda e: isinstance(e, AssistDoneEvent))
        done = next(e for e in events if isinstance(e, AssistDoneEvent))
        assert client.calls[1]["kind"] == "chat"
        assert done.text == "second answer"
        assert "thinking" not in done.text
    finally:
        assistant.stop()


# --------------------------------------------------------------------------------------------
# list_models() background refresh
# --------------------------------------------------------------------------------------------


def test_list_models_populated_after_background_refresh(bus: EventBus, transcript: Transcript) -> None:
    cfg = Config()
    client = FakeClient()
    client.list_models_result = [ModelInfo(id="model-a", loaded=True), ModelInfo(id="model-b", loaded=False)]
    assistant = Assistant(cfg, bus, client, transcript, clock=time.monotonic)
    assistant.start()
    try:
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and assistant.list_models() == []:
            time.sleep(0.02)
        assert assistant.list_models() == ["model-a", "model-b"]
    finally:
        assistant.stop()


# --------------------------------------------------------------------------------------------
# live (real LM Studio, skipped by default)
# --------------------------------------------------------------------------------------------


@pytest.mark.live
def test_live_ask_against_lm_studio() -> None:
    from exocortex.llm.client import LMStudioClient

    cfg = Config()
    cfg.llm.base_url = "http://127.0.0.1:1234"
    cfg.llm.model = "maple-preview"
    cfg.llm.ask_max_tokens = 100  # keep live-test generations short

    client = LMStudioClient(cfg.llm)
    try:
        client.ensure_loaded()
    except LLMError as exc:
        pytest.skip(f"LM Studio not reachable: {exc}")

    bus = EventBus()
    transcript = Transcript()
    assistant = Assistant(cfg, bus, client, transcript, clock=time.monotonic)
    sub = bus.subscribe()
    assistant.start()
    try:
        assistant.ask("In one sentence, what is a mutex?")
        events = drain_until(sub, lambda e: isinstance(e, AssistDoneEvent), timeout=30.0)
        done = next(e for e in events if isinstance(e, AssistDoneEvent))

        assert done.text.strip() != ""
        assert done.ttft_s < 1.5
        assert done.tok_per_s > 20
    finally:
        assistant.stop()

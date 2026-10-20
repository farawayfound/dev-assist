"""Offline tests for the Assistant's kind "complete" job (completion mode) and auto-suppression."""
from __future__ import annotations

import queue
import threading
import time
from types import SimpleNamespace
from typing import Any, Callable

import pytest

from exocortex.bus import (
    AssistDoneEvent,
    AssistStartedEvent,
    EventBus,
    FinalTranscriptEvent,
    ScreenTextEvent,
    Source,
    TranscriptSegment,
    drain,
)
from exocortex.config import Config
from exocortex.llm import prompts
from exocortex.llm.assistant import Assistant
from exocortex.llm.client import Health, ModelInfo, StreamStats
from exocortex.transcript import Transcript


class FakeClient:
    """Records every raw completion request; streams `tokens` slowly enough to be cancellable."""

    def __init__(self, tokens: list[str] | None = None, chunk_delay: float = 0.01) -> None:
        self.cfg = SimpleNamespace(model="maple-preview", base_url="http://fake.test", auto_load_model=True)
        self.tokens = tokens or ["ld", " today"]
        self.chunk_delay = chunk_delay
        self.calls: list[dict[str, Any]] = []
        self._local = threading.local()

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

    def stream_completion(self, prompt: str, *, max_tokens: int, temperature: float | None = None,
                          top_p: float | None = None, stop: list[str] | None = None,
                          cancel: threading.Event | None = None, extra_body: dict[str, Any] | None = None):
        self.calls.append({"prompt": prompt, "max_tokens": max_tokens, "temperature": temperature,
                           "extra_body": extra_body, "stop": stop})
        start = time.monotonic()
        count = 0
        cancelled = False
        for tok in self.tokens:
            time.sleep(self.chunk_delay)
            if cancel is not None and cancel.is_set():
                cancelled = True
                break
            count += 1
            yield tok
        elapsed = max(time.monotonic() - start, 1e-9)
        self.last_stats = StreamStats(ttft_s=0.01, tokens=count, elapsed_s=elapsed, tok_per_s=count / elapsed, cancelled=cancelled)

    def stream_chat(self, *a: Any, **k: Any):  # pragma: no cover - completion never uses chat
        raise AssertionError("chat path must not be used for completions")


def drain_until(sub: "queue.Queue[Any]", predicate: Callable[[Any], bool], timeout: float = 3.0) -> list[Any]:
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
    raise AssertionError(f"timed out; collected {collected!r}")


@pytest.fixture()
def setup() -> tuple[Config, EventBus, Transcript, FakeClient]:
    cfg = Config()
    cfg.completion.hint = "Python 3, terse"
    bus = EventBus()
    tr = Transcript()
    tr.add(TranscriptSegment(id=1, source=Source.SYSTEM, speaker_id="spk1", speaker_label="Priya",
                             text="The board wants the hiring plan by Friday.", t_start=0.0, t_end=2.0, confidence=0.9))
    return cfg, bus, tr, FakeClient()


def test_complete_job_continues_typed_prefix(setup: tuple[Config, EventBus, Transcript, FakeClient]) -> None:
    cfg, bus, tr, client = setup
    assistant = Assistant(cfg, bus, client, tr, clock=time.monotonic)
    assistant.set_screen_provider(lambda: "OCR: some slide text")
    sub = bus.subscribe()
    assistant.start()
    try:
        job_id = assistant.complete("hello wor", "after", "Code.exe - Message input")
        events = drain_until(sub, lambda e: isinstance(e, AssistDoneEvent) and e.job_id == job_id)
        started = next(e for e in events if isinstance(e, AssistStartedEvent) and e.job_id == job_id)
        assert started.kind == "complete"
        done = events[-1]
        assert done.text == "ld today" and not done.cancelled
        call = client.calls[-1]
        assert call["prompt"].endswith(prompts.NO_THINK_PREFIX + "hello wor")
        assert call["temperature"] == pytest.approx(cfg.completion.temperature)
        assert call["max_tokens"] == cfg.completion.max_tokens
        assert call["extra_body"] == {"repeat_penalty": pytest.approx(cfg.completion.repeat_penalty)}
        assert "<transcript>" in call["prompt"] and "hiring plan by Friday" in call["prompt"]
        assert "<screen>" in call["prompt"] and "some slide text" in call["prompt"]
        assert "<after_cursor>\nafter" in call["prompt"]
        assert "Python 3, terse" in call["prompt"]
        # ephemeral: never stored as an answer or a suggestion
        assert assistant.answers == [] and assistant.suggestions == []
    finally:
        assistant.stop()


def test_complete_context_sources_are_configurable(setup: tuple[Config, EventBus, Transcript, FakeClient]) -> None:
    cfg, bus, tr, client = setup
    cfg.completion.use_transcript = False
    cfg.completion.use_screen = False
    cfg.completion.repeat_penalty = 1.0
    assistant = Assistant(cfg, bus, client, tr, clock=time.monotonic)
    assistant.set_screen_provider(lambda: "OCR text")
    sub = bus.subscribe()
    assistant.start()
    try:
        job_id = assistant.complete("x", "", "f")
        drain_until(sub, lambda e: isinstance(e, AssistDoneEvent) and e.job_id == job_id)
        prompt = client.calls[-1]["prompt"]
        assert "<transcript>" not in prompt and "<screen>" not in prompt
        assert "(no session context yet)" in prompt
        assert client.calls[-1]["extra_body"] is None
    finally:
        assistant.stop()


def test_cancel_job_cancels_running_completion(setup: tuple[Config, EventBus, Transcript, FakeClient]) -> None:
    cfg, bus, tr, _ = setup
    client = FakeClient(tokens=["a"] * 40, chunk_delay=0.03)
    assistant = Assistant(cfg, bus, client, tr, clock=time.monotonic)
    sub = bus.subscribe()
    assistant.start()
    try:
        job_id = assistant.complete("hello", "", "f")
        drain_until(sub, lambda e: isinstance(e, AssistStartedEvent) and e.job_id == job_id)
        time.sleep(0.1)
        assistant.cancel_job(job_id)
        events = drain_until(sub, lambda e: isinstance(e, AssistDoneEvent) and e.job_id == job_id)
        assert events[-1].cancelled
    finally:
        assistant.stop()


def test_auto_suppressed_blocks_automatic_lanes(setup: tuple[Config, EventBus, Transcript, FakeClient]) -> None:
    cfg, bus, tr, client = setup
    cfg.screen.auto_suggest = True
    cfg.screen.suggest_min_interval_s = 0
    cfg.assist.coalesce_ms = 50
    assistant = Assistant(cfg, bus, client, tr, clock=time.monotonic)
    assistant.set_screen_provider(lambda: "text")
    sub = bus.subscribe()
    assistant.start()
    try:
        assistant.set_auto_suppressed(True)
        bus.publish(ScreenTextEvent(mode="auto", source_name="w", rect=(0, 0, 1, 1), text="Two Sum problem",
                                    lines=("Two Sum problem",), changed=True, ocr_ms=1.0, captured_at=0.0))
        seg = TranscriptSegment(id=2, source=Source.SYSTEM, speaker_id="spk1", speaker_label="Priya",
                                text="What do you think about the hiring plan for the platform team?",
                                t_start=3.0, t_end=5.0, confidence=0.9, is_question=True)
        tr.add(seg)
        bus.publish(FinalTranscriptEvent(segment=seg))
        time.sleep(0.4)
        assert not any(isinstance(e, AssistStartedEvent) for e in drain(sub))
        # manual actions still work while suppressed
        assistant.ask("typed question")
        events = drain_until(sub, lambda e: isinstance(e, AssistStartedEvent))
        assert any(e.kind == "ask" for e in events if isinstance(e, AssistStartedEvent))
    finally:
        assistant.stop()

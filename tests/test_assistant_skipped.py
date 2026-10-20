"""Tests for `AssistDoneEvent.skipped` (the model answered exactly `prompts.SKIP_MARKER`,
"(nothing to do)") on the "code" job, and for `Assistant.code(focus, label=...)`: the label
becomes `AssistStartedEvent.question` while the user prompt still carries the (possibly long)
`focus` text.

Fully offline: a FakeClient (duck-typed, no httpx/LMStudioClient subclassing) stands in for
exocortex.llm.client.LMStudioClient. FakeClient/drain_until/make_segment are imported from
tests/test_assistant.py rather than redefined here, exactly as tests/test_assistant_text.py does
-- see that file's module docstring.
"""
from __future__ import annotations

import time

from exocortex.bus import AssistDoneEvent, AssistStartedEvent, EventBus
from exocortex.config import Config
from exocortex.llm import prompts
from exocortex.llm.assistant import Assistant
from exocortex.transcript import Transcript
from tests.test_assistant import FakeClient, drain_until


def _dones(events: list) -> list[AssistDoneEvent]:
    return [e for e in events if isinstance(e, AssistDoneEvent)]


def _started(events: list) -> list[AssistStartedEvent]:
    return [e for e in events if isinstance(e, AssistStartedEvent)]


# --------------------------------------------------------------------------------------------
# AssistDoneEvent.skipped
# --------------------------------------------------------------------------------------------


def test_code_skip_marker_sets_cancelled_and_skipped() -> None:
    cfg = Config()
    bus = EventBus()
    transcript = Transcript()
    client = FakeClient(token_batches=[["(nothing to do)"]], chunk_delay=0.0)
    a = Assistant(cfg, bus, client, transcript, clock=time.monotonic)
    a.set_screen_provider(lambda: "chrome, nothing technical")
    sub = bus.subscribe()
    a.start()
    try:
        a.code("")
        events = drain_until(sub, lambda e: isinstance(e, AssistDoneEvent))
        done = _dones(events)[-1]
        assert done.cancelled is True
        assert done.skipped is True
        assert done.text == ""
        assert a.answers == []
    finally:
        a.stop()


def test_code_normal_answer_has_skipped_false() -> None:
    cfg = Config()
    bus = EventBus()
    transcript = Transcript()
    client = FakeClient(token_batches=[["def add(a, b): return a + b"]], chunk_delay=0.0)
    a = Assistant(cfg, bus, client, transcript, clock=time.monotonic)
    a.set_screen_provider(lambda: "write a function that adds two numbers")
    sub = bus.subscribe()
    a.start()
    try:
        a.code("")
        events = drain_until(sub, lambda e: isinstance(e, AssistDoneEvent))
        done = _dones(events)[-1]
        assert done.cancelled is False
        assert done.skipped is False
        assert "def add" in done.text
    finally:
        a.stop()


def test_skip_marker_exact_text_is_the_documented_marker() -> None:
    """`prompts.SKIP_MARKER` is literally what the model is told to answer -- pin the exact
    string so a future rewording of the prompt doesn't silently stop the drop-quietly path."""
    assert prompts.SKIP_MARKER == "(nothing to do)"


# --------------------------------------------------------------------------------------------
# Assistant.code(focus, label=...): label drives AssistStartedEvent.question, focus stays in
# the user prompt sent to the model.
# --------------------------------------------------------------------------------------------


def test_code_with_label_publishes_started_with_the_label() -> None:
    cfg = Config()
    bus = EventBus()
    transcript = Transcript()
    client = FakeClient(token_batches=[["def solve(): pass"]], chunk_delay=0.0)
    a = Assistant(cfg, bus, client, transcript, clock=time.monotonic)
    a.set_screen_provider(lambda: "")
    sub = bus.subscribe()
    a.start()
    try:
        a.code("some long focus instruction from hands-free mode", label="the coding task on screen")
        started_events = drain_until(sub, lambda e: isinstance(e, AssistStartedEvent) and e.kind == "code")
        started = _started(started_events)[-1]
        assert started.question == "the coding task on screen"

        drain_until(sub, lambda e: isinstance(e, AssistDoneEvent))
        # the FakeClient records the rendered prompt it was actually sent
        completion_calls = [c for c in client.calls if c["kind"] == "completion"]
        assert completion_calls
        assert "some long focus instruction from hands-free mode" in completion_calls[-1]["prompt"]
    finally:
        a.stop()


def test_code_without_label_falls_back_to_focus_or_default() -> None:
    cfg = Config()
    bus = EventBus()
    transcript = Transcript()
    client = FakeClient(token_batches=[["def solve(): pass"]], chunk_delay=0.0)
    a = Assistant(cfg, bus, client, transcript, clock=time.monotonic)
    a.set_screen_provider(lambda: "")
    sub = bus.subscribe()
    a.start()
    try:
        a.code("fix the off-by-one bug")
        events = drain_until(sub, lambda e: isinstance(e, AssistStartedEvent) and e.kind == "code")
        started = _started(events)[-1]
        assert started.question == "fix the off-by-one bug"
    finally:
        a.stop()

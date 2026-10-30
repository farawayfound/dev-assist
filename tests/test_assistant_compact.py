"""Offline tests for ``Assistant.compact`` (the overlay's "Compact" control, 2026-10-07): it
folds all but the newest lines of the current context window into the running summary straight
away and queues a compaction job on the notes lane. Mirrors tests/test_assistant_checkpoint.py's
harness (the FakeClient / make_segment pattern from tests/test_assistant.py)."""
from __future__ import annotations

import time

import pytest

from exocortex.config import Config
from exocortex.llm.assistant import Assistant
from exocortex.bus import EventBus
from exocortex.transcript import Transcript
from tests.test_assistant import FakeClient, make_segment


@pytest.fixture()
def bus() -> EventBus:
    return EventBus()


@pytest.fixture()
def transcript() -> Transcript:
    return Transcript()


def _fill(tr: Transcript, start: int, stop: int) -> None:
    """Add segments with ids ``start``..``stop``-1 (continuing ids, never reused)."""
    for i in range(start, stop):
        tr.add(make_segment(i, text=f"utterance number {i:03d}", t_end=float(i)))


def test_compact_folds_the_window_and_queues_a_job(bus: EventBus, transcript: Transcript) -> None:
    cfg = Config()
    cfg.context.compaction = "summary"
    a = Assistant(cfg, bus, FakeClient(), transcript, clock=time.monotonic)
    _fill(transcript, 1, 13)
    dropped = a.compact()
    assert dropped > 0
    # The whole-meeting window (no checkpoint set) now carries pending text to be summarised.
    assert a._context.window.has_pending(None)
    # A compaction job is queued (latest-wins slot); claim it to prove one was put.
    job = a._compact_slot.try_get()
    assert job is not None and job.scope is None


def test_compact_uses_the_whole_meeting_scope_even_with_a_checkpoint(bus: EventBus, transcript: Transcript) -> None:
    cfg = Config()
    cfg.context.compaction = "summary"
    a = Assistant(cfg, bus, FakeClient(), transcript, clock=time.monotonic)
    _fill(transcript, 1, 7)
    a.checkpoint()                         # boundary at the last segment so far
    _fill(transcript, 7, 13)               # six more after the checkpoint
    scope = a._checkpoint_boundary()
    assert scope is not None
    dropped = a.compact()
    assert dropped > 0
    # Compact always folds the whole-meeting scope (None) -- what ask/auto/screen and the Monitor
    # summary read -- not the (already checkpoint-bounded) coding scope.
    assert a._context.window.has_pending(None)
    assert not a._context.window.has_pending(scope)


def test_compact_job_installs_the_summary_actively(bus: EventBus, transcript: Transcript) -> None:
    """A user-driven Compact queues a _CompactJob with force=True, so the summary is installed as
    the ACTIVE earlier-summary (not merely staged for a future overflow)."""
    cfg = Config()
    cfg.context.compaction = "summary"
    a = Assistant(cfg, bus, FakeClient(), transcript, clock=time.monotonic)
    _fill(transcript, 1, 13)
    a.compact()
    job = a._compact_slot.try_get()
    assert job is not None and job.force is True and job.scope is None


def test_compact_is_a_noop_when_compaction_is_off(bus: EventBus, transcript: Transcript) -> None:
    cfg = Config()
    cfg.context.compaction = "off"
    a = Assistant(cfg, bus, FakeClient(), transcript, clock=time.monotonic)
    _fill(transcript, 1, 13)
    assert a.compact() == 0
    assert not a._context.window.has_pending(None)       # nothing dropped: no silent data loss


def test_compact_is_a_noop_on_an_empty_window(bus: EventBus, transcript: Transcript) -> None:
    cfg = Config()
    cfg.context.compaction = "summary"
    a = Assistant(cfg, bus, FakeClient(), transcript, clock=time.monotonic)
    _fill(transcript, 1, 2)
    assert a.compact() == 0

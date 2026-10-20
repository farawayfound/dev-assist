"""Additional regression tests for exocortex.speech.pipeline.SpeechPipeline.

Targets gaps not covered by tests/test_pipeline.py: partial capture-start failure, the
diarization status-event matrix (clusterer present/absent x enabled/disabled), symmetric
echo suppression (SYSTEM duplicate of an earlier MIC final), the bounded frame/ASR job
queues under overflow, lifecycle idempotence (start-twice, stop-before-start, restart),
and rename_speaker with no clusterer configured. See tests/test_pipeline.py for the shared
fake AudioCapture/Transcriber/SpeakerClusterer implementations and `_make_cfg` helper,
reused here via direct import (this file is local to the same tests/ package).
"""
from __future__ import annotations

import queue
import threading
import time
from typing import Any, Callable

import numpy as np
import pytest

from exocortex.audio.capture import DeviceInfo
from exocortex.bus import (
    ComponentStatusEvent,
    EventBus,
    FinalTranscriptEvent,
    Source,
    SpeakerRenamedEvent,
    drain,
)
from exocortex.speech.pipeline import SpeechPipeline

from test_pipeline import (
    FakeClusterer,
    FakeTranscriber,
    FixedTextTranscriber,
    _ManualCapture,
    _make_cfg,
    _make_pipeline,
    _wait_until,
)


# --------------------------------------------------------------------------------- #
# capture start failure: one source fails, the other still starts
# --------------------------------------------------------------------------------- #


class _StartFailsCapture:
    """Fakes AudioCapture whose `start()` always raises, as if the device were missing."""

    def __init__(
        self,
        source: Source,
        spec: str,
        *,
        target_rate: int,
        frame_samples: int,
        on_frames: Callable[[np.ndarray], None],
        on_level: Callable[[float, float], None] | None = None,
        on_status: Callable[[str, str, str], None] | None = None,
        keepalive: bool = True,
    ) -> None:
        self._source = source

    def start(self) -> DeviceInfo:
        raise RuntimeError("no such system device")

    def stop(self) -> None:
        pass

    def set_muted(self, muted: bool) -> None:
        pass

    @property
    def device(self) -> DeviceInfo | None:
        return None

    @property
    def running(self) -> bool:
        return False


def test_capture_failure_for_one_source_still_starts_other_and_reports_error(silero_path):
    cfg = _make_cfg(capture_system=True, capture_mic=True)
    bus = EventBus()
    q = bus.subscribe()

    holder: dict[Source, _ManualCapture] = {}

    def factory(source: Source, spec: str, **kw: Any) -> Any:
        if source is Source.SYSTEM:
            return _StartFailsCapture(source, spec, **kw)
        cap = _ManualCapture(source, spec, device_name="fake-mic", **kw)
        holder[source] = cap
        return cap

    pipeline = _make_pipeline(cfg, bus, FakeTranscriber(), None, factory)
    pipeline.start()
    try:
        assert _wait_until(lambda: Source.MIC in holder, timeout=2.0)
        time.sleep(0.2)
    finally:
        pipeline.stop()

    events = drain(q, max_items=10000)
    audio_status = [e for e in events if isinstance(e, ComponentStatusEvent) and e.component == "audio"]
    ready = [e for e in audio_status if e.state == "ready"]
    errors = [e for e in audio_status if e.state == "error"]

    # The MIC capture started fine: exactly one "ready" event, naming only the MIC device
    # (the failed SYSTEM capture never contributes a device name).
    assert len(ready) == 1
    assert "fake-mic" in ready[0].detail
    # The failed SYSTEM capture reports its own error; the MIC capture is unaffected.
    assert any(e.detail.startswith("system:") for e in errors)


# --------------------------------------------------------------------------------- #
# diarization status matrix
# --------------------------------------------------------------------------------- #


def _start_simple_pipeline(cfg, bus, clusterer) -> SpeechPipeline:
    def factory(source: Source, spec: str, **kw: Any) -> _ManualCapture:
        return _ManualCapture(source, spec, device_name="fake", **kw)

    pipeline = _make_pipeline(cfg, bus, FakeTranscriber(), clusterer, factory)
    pipeline.start()
    return pipeline


def test_no_diarization_status_when_disabled_and_no_clusterer(silero_path):
    cfg = _make_cfg(capture_system=True, capture_mic=False)
    cfg.diarization.enabled = False
    bus = EventBus()
    q = bus.subscribe()

    pipeline = _start_simple_pipeline(cfg, bus, None)
    try:
        time.sleep(0.2)
    finally:
        pipeline.stop()

    diar = [e for e in drain(q, max_items=10000) if isinstance(e, ComponentStatusEvent) and e.component == "diarization"]
    assert diar == [], "no clusterer was ever expected (diarization turned off) -> no status event"


def test_diarization_status_error_when_enabled_but_no_clusterer(silero_path):
    cfg = _make_cfg(capture_system=True, capture_mic=False)
    cfg.diarization.enabled = True
    bus = EventBus()
    q = bus.subscribe()

    pipeline = _start_simple_pipeline(cfg, bus, None)
    try:
        time.sleep(0.2)
    finally:
        pipeline.stop()

    diar = [e for e in drain(q, max_items=10000) if isinstance(e, ComponentStatusEvent) and e.component == "diarization"]
    assert len(diar) == 1
    assert diar[0].state == "error"


def test_diarization_status_ready_when_clusterer_present(silero_path):
    cfg = _make_cfg(capture_system=True, capture_mic=False)
    bus = EventBus()
    q = bus.subscribe()

    pipeline = _start_simple_pipeline(cfg, bus, FakeClusterer())
    try:
        time.sleep(0.2)
    finally:
        pipeline.stop()

    diar = [e for e in drain(q, max_items=10000) if isinstance(e, ComponentStatusEvent) and e.component == "diarization"]
    assert len(diar) == 1
    assert diar[0].state == "ready"


# --------------------------------------------------------------------------------- #
# echo suppression: symmetric direction (SYSTEM duplicate of an earlier MIC final)
# --------------------------------------------------------------------------------- #


def test_echo_suppression_symmetric_system_after_mic(two_speakers_audio, silero_path):
    audio, sr = two_speakers_audio
    cfg = _make_cfg(capture_system=True, capture_mic=True, echo_suppression=True)
    bus = EventBus()
    q = bus.subscribe()
    transcriber = FixedTextTranscriber("we should ship the migration plan by friday")

    holder: dict[Source, _ManualCapture] = {}

    def factory(source: Source, spec: str, **kw: Any) -> _ManualCapture:
        cap = _ManualCapture(source, spec, device_name=f"fake-{source.value}", **kw)
        holder[source] = cap
        return cap

    pipeline = _make_pipeline(cfg, bus, transcriber, FakeClusterer(), factory)
    pipeline.start()
    try:
        mic_snippet = audio[int(9.03 * sr) : int(13.81 * sr)]
        sys_snippet = audio[int(3.17 * sr) : int(6.86 * sr)]

        # MIC speaks first this time (opposite order from test_pipeline.py's echo test).
        holder[Source.MIC].push_audio(mic_snippet)
        holder[Source.MIC].push_silence()
        assert _wait_until(lambda: transcriber.calls >= 1, timeout=5.0)
        time.sleep(0.3)

        holder[Source.SYSTEM].push_audio(sys_snippet)
        holder[Source.SYSTEM].push_silence()
        time.sleep(0.6)
    finally:
        pipeline.stop()

    finals = [e.segment for e in drain(q, max_items=10000) if isinstance(e, FinalTranscriptEvent)]
    assert len(finals) == 1, f"expected the later SYSTEM duplicate to be echo-suppressed, got {finals!r}"
    assert finals[0].source is Source.MIC


# --------------------------------------------------------------------------------- #
# bounded queues: frame queue overflow, ASR job queue cap
# --------------------------------------------------------------------------------- #


def test_frame_queue_overflow_drops_oldest_frame_and_counts_it():
    """Whitebox: exercises `_make_on_frames`'s overflow path directly and deterministically
    (an end-to-end overflow via real threads would be timing-dependent/flaky)."""
    cfg = _make_cfg()
    bus = EventBus()
    pipeline = _make_pipeline(
        cfg, bus, FakeTranscriber(), None, lambda source, spec, **kw: _ManualCapture(source, spec, **kw)
    )
    small_queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=3)
    pipeline._frame_queues[Source.SYSTEM] = small_queue  # type: ignore[attr-defined]

    on_frames = pipeline._make_on_frames(Source.SYSTEM)  # type: ignore[attr-defined]
    frames = [np.full(512, float(i), dtype=np.float32) for i in range(5)]
    for f in frames:
        on_frames(f)

    assert small_queue.qsize() == 3
    remaining = [small_queue.get_nowait()[0] for _ in range(3)]
    assert remaining == [2.0, 3.0, 4.0], "the two oldest frames (0, 1) should have been dropped"
    assert pipeline._frame_drop_counts[Source.SYSTEM] == 2  # type: ignore[attr-defined]


def test_asr_job_queue_has_a_bound():
    """Regression test: the ASR job priority queue must be bounded (finals get backpressure,
    partials get dropped) rather than growing without limit if the ASR worker falls behind."""
    cfg = _make_cfg()
    bus = EventBus()
    pipeline = _make_pipeline(
        cfg, bus, FakeTranscriber(), None, lambda source, spec, **kw: _ManualCapture(source, spec, **kw)
    )
    assert pipeline._asr_queue.maxsize > 0  # type: ignore[attr-defined]


# --------------------------------------------------------------------------------- #
# lifecycle idempotence
# --------------------------------------------------------------------------------- #


def test_start_called_twice_does_not_open_a_second_set_of_captures(silero_path):
    cfg = _make_cfg(capture_system=True, capture_mic=False)
    bus = EventBus()
    starts = {"n": 0}

    def factory(source: Source, spec: str, **kw: Any) -> _ManualCapture:
        starts["n"] += 1
        return _ManualCapture(source, spec, device_name="fake", **kw)

    pipeline = _make_pipeline(cfg, bus, FakeTranscriber(), None, factory)
    pipeline.start()
    pipeline.start()  # must be a no-op (logs a warning), not open a second capture
    try:
        time.sleep(0.1)
    finally:
        pipeline.stop()

    assert starts["n"] == 1


def test_stop_before_start_is_a_safe_noop():
    cfg = _make_cfg()
    bus = EventBus()
    pipeline = _make_pipeline(
        cfg, bus, FakeTranscriber(), None, lambda source, spec, **kw: _ManualCapture(source, spec, **kw)
    )
    pipeline.stop()  # must not raise
    pipeline.stop()  # nor on a second call


def test_restart_after_stop_starts_cleanly_with_no_leftover_threads(silero_path):
    cfg = _make_cfg(capture_system=True, capture_mic=False)
    bus = EventBus()

    def factory(source: Source, spec: str, **kw: Any) -> _ManualCapture:
        return _ManualCapture(source, spec, device_name="fake", **kw)

    pipeline = _make_pipeline(cfg, bus, FakeTranscriber(), None, factory)

    pipeline.start()
    time.sleep(0.1)
    pipeline.stop()

    pipeline.start()
    time.sleep(0.1)
    pipeline.stop()

    names = {t.name for t in threading.enumerate()}
    assert "asr-worker" not in names
    assert "vad-system" not in names


# --------------------------------------------------------------------------------- #
# rename_speaker with no clusterer configured
# --------------------------------------------------------------------------------- #


def test_rename_speaker_without_clusterer_still_publishes_event():
    cfg = _make_cfg()
    bus = EventBus()
    q = bus.subscribe()
    pipeline = _make_pipeline(
        cfg, bus, FakeTranscriber(), None, lambda source, spec, **kw: _ManualCapture(source, spec, **kw)
    )

    pipeline.rename_speaker("spk2", "Bob")  # must not raise despite clusterer=None

    events = drain(q)
    renamed = [e for e in events if isinstance(e, SpeakerRenamedEvent)]
    assert len(renamed) == 1
    assert renamed[0].speaker_id == "spk2"
    assert renamed[0].new_label == "Bob"


# --------------------------------------------------------------------------------- #
# partial-transcription error clears the stale partial row (mirrors the final job's
# error handling; see the fix in exocortex/speech/pipeline.py's _process_partial_job)
# --------------------------------------------------------------------------------- #


class _RaisingPartialTranscriber:
    """Raises on partial calls, succeeds on final calls (real text)."""

    def __init__(self) -> None:
        self.partial_calls = 0
        self.final_calls = 0

    def transcribe(self, audio: np.ndarray, *, partial: bool = False):
        from exocortex.speech.asr import AsrResult

        if partial:
            self.partial_calls += 1
            raise RuntimeError("boom: partial transcription blew up")
        self.final_calls += 1
        return AsrResult(
            text="hello there friend", avg_logprob=-0.2, no_speech_prob=0.0,
            audio_s=len(audio) / 16000.0, elapsed_s=0.0, dropped=False, reason="",
        )


def test_partial_transcription_error_clears_stale_partial_row(two_speakers_audio, silero_path):
    from exocortex.bus import PartialTranscriptEvent

    audio, sr = two_speakers_audio
    cfg = _make_cfg(capture_system=True, capture_mic=False, partials_enabled=True, partial_interval_s=0.2)
    bus = EventBus()
    q = bus.subscribe()
    transcriber = _RaisingPartialTranscriber()

    holder: dict[Source, _ManualCapture] = {}

    def factory(source: Source, spec: str, **kw: Any) -> _ManualCapture:
        cap = _ManualCapture(source, spec, device_name="fake-sys", **kw)
        holder[source] = cap
        return cap

    pipeline = _make_pipeline(cfg, bus, transcriber, None, factory)
    pipeline.start()
    try:
        snippet = audio[int(3.17 * sr) : int(6.86 * sr)]
        holder[Source.SYSTEM].push_audio(snippet)
        assert _wait_until(lambda: transcriber.partial_calls >= 1, timeout=5.0)
        holder[Source.SYSTEM].push_silence()
        assert _wait_until(lambda: transcriber.final_calls >= 1, timeout=5.0)
        time.sleep(0.3)
    finally:
        pipeline.stop()

    partials = [e for e in drain(q, max_items=10000) if isinstance(e, PartialTranscriptEvent)]
    # Every partial call raised, so every published partial row must be the empty-text clear
    # (never a stale non-empty partial left dangling because the transcribe call errored).
    assert partials, "expected at least one PartialTranscriptEvent (the clears)"
    assert all(p.text == "" for p in partials)


# --------------------------------------------------------------------------------- #
# 2026-09-26 (asr-sherpa item 4): a slow transcribe() must never publish, or even run, a stale
# partial. SpeechPipeline already has this guard end to end -- the per-source `version` counter
# bumped by `_bump_version()` on every new partial/final, checked in `_process_partial_job()`
# both BEFORE calling `transcribe()` (a partial superseded while still queued behind a slow
# transcribe is never even given to the model) and AFTER it returns (a partial superseded WHILE
# its own transcribe() was running is computed but never published) -- so no pipeline.py change
# was needed for this item; this test pins that existing behaviour against a slow engine.
# --------------------------------------------------------------------------------- #


class _BlockingPartialTranscriber:
    """`transcribe(partial=True)` blocks on `release` after the first call, standing in for an
    engine slower than `[asr].partial_interval_s` (e.g. faster-whisper's fixed 30 s window on a
    weak CPU). `calls` records every call actually made (partial or final) in order."""

    def __init__(self) -> None:
        from exocortex.speech.asr import AsrResult  # local: keeps this fixture import-light

        self._AsrResult = AsrResult
        self.calls: list[bool] = []
        self.started = threading.Event()
        self.release = threading.Event()
        self._blocked_once = False

    def transcribe(self, audio: np.ndarray, *, partial: bool = False):
        self.calls.append(partial)
        if partial and not self._blocked_once:
            self._blocked_once = True
            self.started.set()
            self.release.wait(timeout=5.0)
        return self._AsrResult(
            text=f"partial-{len(self.calls)}", avg_logprob=-0.2, no_speech_prob=0.0,
            audio_s=len(audio) / 16000.0, elapsed_s=0.0, dropped=False, reason="",
        )


def test_slow_transcribe_never_publishes_or_runs_a_superseded_partial(silero_path) -> None:
    from exocortex.bus import PartialTranscriptEvent
    from exocortex.speech.vad import SpeechOngoing

    cfg = _make_cfg(capture_system=True, capture_mic=False, partial_interval_s=0.1)
    bus = EventBus()
    q = bus.subscribe()
    transcriber = _BlockingPartialTranscriber()

    def factory(source: Source, spec: str, **kw: Any):
        return _ManualCapture(source, spec, device_name="fake-sys", **kw)

    pipeline = _make_pipeline(cfg, bus, transcriber, None, factory)
    pipeline.start()
    try:
        audio_1s = np.zeros(16000, dtype=np.float32)

        # job #1 (version 1): the worker dequeues it and blocks inside transcribe().
        pipeline._enqueue_partial(Source.SYSTEM, SpeechOngoing(t_start=0.0, audio=audio_1s))
        assert _wait_until(lambda: transcriber.started.is_set(), timeout=5.0)

        # jobs #2 and #3 (versions 2, 3) arrive while #1 is still in flight -- each supersedes
        # the last; only version 4 (job #4) will still be "current" once the worker catches up.
        pipeline._enqueue_partial(Source.SYSTEM, SpeechOngoing(t_start=0.1, audio=audio_1s))
        pipeline._enqueue_partial(Source.SYSTEM, SpeechOngoing(t_start=0.2, audio=audio_1s))
        pipeline._enqueue_partial(Source.SYSTEM, SpeechOngoing(t_start=0.3, audio=audio_1s))

        transcriber.release.set()  # let job #1's transcribe() return
        assert _wait_until(lambda: len(transcriber.calls) >= 2, timeout=5.0)
        time.sleep(0.2)  # let the asr-worker drain whatever it is going to drain
    finally:
        pipeline.stop()

    # Jobs #2 and #3 were skipped BEFORE transcribe() ever ran for them (the pre-check in
    # _process_partial_job): only #1 and #4 were actually transcribed.
    assert transcriber.calls == [True, True], (
        f"expected exactly 2 transcribe() calls (job #1 and the still-current job #4), got {transcriber.calls}"
    )

    partials = [e for e in drain(q, max_items=10000) if isinstance(e, PartialTranscriptEvent)]
    non_empty = [p.text for p in partials if p.text]
    # Job #1's result ("partial-1") was computed but superseded before it could be published
    # (the post-check in _process_partial_job); only job #4's result ("partial-2", the 2nd call
    # actually made) reaches the bus.
    assert non_empty == ["partial-2"], f"a stale partial result leaked onto the bus: {non_empty}"

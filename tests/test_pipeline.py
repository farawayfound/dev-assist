"""Unit tests for exocortex.speech.pipeline.SpeechPipeline.

Offline tests run on CPU with the real Silero VAD ONNX model (models/silero_vad.onnx,
skipped automatically if missing) driving the real VadSegmenter, but use fake AudioCapture
implementations (injected via `capture_factory`), a fake Transcriber and a fake
SpeakerClusterer so no GPU / audio devices / LM Studio are required. See tests/conftest.py
for `two_speakers_audio`, `silero_path` and the `TWO_SPEAKERS_TRUTH` ground truth.

A `@pytest.mark.live` test at the bottom exercises the real Transcriber (large-v3-turbo,
CUDA) and a real SpeakerClusterer end to end; it is skipped by default (see pytest.ini).
"""
from __future__ import annotations

import itertools
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pytest

from exocortex.audio.capture import DeviceInfo
from exocortex.bus import (
    AudioLevelEvent,
    AudioStatusEvent,
    ComponentStatusEvent,
    EventBus,
    FinalTranscriptEvent,
    PartialTranscriptEvent,
    Source,
    drain,
)
from exocortex.config import Config
from exocortex.speech.asr import AsrResult
from exocortex.speech.pipeline import SpeechPipeline
from exocortex.speech.vad import SpeechEnd, SpeechOngoing

from conftest import TWO_SPEAKERS_TRUTH

_QUESTION_TRUTH_INDICES = (5, 9)  # 1-indexed rows of TWO_SPEAKERS_TRUTH that are questions


# --------------------------------------------------------------------------------- #
# config helper
# --------------------------------------------------------------------------------- #


def _make_cfg(
    *,
    capture_system: bool = True,
    capture_mic: bool = False,
    min_silence_s: float = 0.3,
    min_speech_s: float = 0.25,
    pre_roll_s: float = 0.3,
    max_utterance_s: float = 20.0,
    echo_suppression: bool = True,
    partials_enabled: bool = True,
    partial_interval_s: float = 0.5,
) -> Config:
    """A Config tuned like tests/test_vad.py (min_silence_s=0.3 reliably yields the full 9
    TWO_SPEAKERS_TRUTH segments; SPEC.md's own AudioConfig default merges the first two)."""
    cfg = Config()
    cfg.audio.capture_system = capture_system
    cfg.audio.capture_mic = capture_mic
    cfg.audio.min_silence_s = min_silence_s
    cfg.audio.min_speech_s = min_speech_s
    cfg.audio.pre_roll_s = pre_roll_s
    cfg.audio.max_utterance_s = max_utterance_s
    cfg.audio.echo_suppression = echo_suppression
    cfg.audio.loopback_keepalive = False
    cfg.asr.partials_enabled = partials_enabled
    cfg.asr.partial_interval_s = partial_interval_s
    return cfg


def _chunk(audio: np.ndarray, size: int = 512) -> list[np.ndarray]:
    """Split `audio` into exact `size`-sample frames, dropping any short remainder (matches
    how a real AudioCapture only ever calls on_frames with exact-size chunks)."""
    n = (len(audio) // size) * size
    return [audio[i : i + size].astype(np.float32) for i in range(0, n, size)]


# --------------------------------------------------------------------------------- #
# fake Transcriber / SpeakerClusterer
# --------------------------------------------------------------------------------- #


@dataclass
class _Call:
    partial: bool
    audio_len: int


class FakeTranscriber:
    """Stand-in for Transcriber: never drops (avg_logprob=-0.2, no_speech_prob=0.0).

    Final calls return "utt-<n>" (n = 1, 2, ... in call order) unless `final_text_overrides`
    maps that 1-indexed final call number to a specific text (used to feed real question text
    through for the is_question wiring test). Partial calls return "utt-<n>-partial" using the
    number of the final currently in progress. `calls` records every call for assertions.
    """

    def __init__(self, final_text_overrides: dict[int, str] | None = None) -> None:
        self.final_text_overrides = final_text_overrides or {}
        self.calls: list[_Call] = []
        self._final_n = 0
        self._lock = threading.Lock()

    def transcribe(self, audio: np.ndarray, *, partial: bool = False) -> AsrResult:
        with self._lock:
            self.calls.append(_Call(partial=partial, audio_len=len(audio)))
            if partial:
                text = f"utt-{self._final_n + 1}-partial"
            else:
                self._final_n += 1
                text = self.final_text_overrides.get(self._final_n, f"utt-{self._final_n}")
        return AsrResult(
            text=text, avg_logprob=-0.2, no_speech_prob=0.0,
            audio_s=len(audio) / 16000.0, elapsed_s=0.0, dropped=False, reason="",
        )


class FixedTextTranscriber:
    """Stand-in for Transcriber that always returns the same fixed text, regardless of audio
    or source - used to make two independently-detected utterances (one MIC, one SYSTEM)
    compare as duplicates for the echo-suppression test."""

    def __init__(self, text: str) -> None:
        self._text = text
        self.calls = 0

    def transcribe(self, audio: np.ndarray, *, partial: bool = False) -> AsrResult:
        self.calls += 1
        return AsrResult(
            text=self._text, avg_logprob=-0.2, no_speech_prob=0.0,
            audio_s=len(audio) / 16000.0, elapsed_s=0.0, dropped=False, reason="",
        )


class FakeClusterer:
    """Stand-in for SpeakerClusterer: alternates spk1/spk2 on every assign() call."""

    def __init__(self) -> None:
        self._n = 0
        self.renames: list[tuple[str, str]] = []

    def assign(self, audio: np.ndarray) -> tuple[str, str]:
        self._n += 1
        return ("spk1", "Speaker 1") if self._n % 2 == 1 else ("spk2", "Speaker 2")

    def rename(self, speaker_id: str, label: str) -> None:
        self.renames.append((speaker_id, label))


# --------------------------------------------------------------------------------- #
# fake AudioCapture implementations
# --------------------------------------------------------------------------------- #


def _fake_device(source: Source, name: str) -> DeviceInfo:
    return DeviceInfo(
        index=0, name=name, is_loopback=(source is Source.SYSTEM), is_input=True,
        sample_rate=16000, channels=1, is_default=True,
    )


class _AutoFeedCapture:
    """Fakes AudioCapture: replays a fixed `audio` array as 512-sample frames from a
    background thread at `speed`x real-time, via the callbacks the pipeline supplied."""

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
        audio: np.ndarray,
        speed: float = 10.0,
        device_name: str = "fake",
    ) -> None:
        self._source = source
        self._on_frames = on_frames
        self._on_level = on_level
        self._on_status = on_status
        self._audio = audio
        self._frame_samples = frame_samples
        self._speed = speed
        self._device = _fake_device(source, device_name)
        self._running = False
        self._muted = False
        self._done = threading.Event()
        self._stop_evt = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> DeviceInfo:
        self._running = True
        self._stop_evt.clear()
        self._done.clear()
        if self._on_status is not None:
            self._on_status("started", self._device.name, "")
        self._thread = threading.Thread(target=self._run, daemon=True, name=f"fake-capture-{self._source.value}")
        self._thread.start()
        return self._device

    def _run(self) -> None:
        n = self._frame_samples
        period = (n / 16000.0) / self._speed
        i = 0
        total = len(self._audio)
        while i + n <= total and not self._stop_evt.is_set():
            frame = np.zeros(n, dtype=np.float32) if self._muted else self._audio[i : i + n].astype(np.float32)
            self._on_frames(frame)
            if self._on_level is not None:
                rms = float(np.sqrt(np.mean(np.square(frame)))) if frame.size else 0.0
                self._on_level(rms, float(np.max(np.abs(frame))) if frame.size else 0.0)
            i += n
            time.sleep(period)
        self._done.set()

    def is_playing(self) -> bool:
        return not self._done.is_set()

    def stop(self) -> None:
        self._running = False
        self._stop_evt.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._on_status is not None:
            self._on_status("stopped", self._device.name, "")

    def set_muted(self, muted: bool) -> None:
        self._muted = muted
        if self._on_status is not None:
            self._on_status("muted" if muted else "unmuted", self._device.name, "")

    @property
    def device(self) -> DeviceInfo:
        return self._device

    @property
    def running(self) -> bool:
        return self._running


class _ManualCapture:
    """Fakes AudioCapture with no auto-playback: the test pushes frames directly via
    `.push(frame)` (zeroed while muted, mirroring the real AudioCapture)."""

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
        device_name: str = "fake",
    ) -> None:
        self._source = source
        self._on_frames = on_frames
        self._on_status = on_status
        self._device = _fake_device(source, device_name)
        self._running = False
        self._muted = False

    def start(self) -> DeviceInfo:
        self._running = True
        if self._on_status is not None:
            self._on_status("started", self._device.name, "")
        return self._device

    def stop(self) -> None:
        self._running = False
        if self._on_status is not None:
            self._on_status("stopped", self._device.name, "")

    def set_muted(self, muted: bool) -> None:
        self._muted = muted
        if self._on_status is not None:
            self._on_status("muted" if muted else "unmuted", self._device.name, "")

    def push(self, frame: np.ndarray) -> None:
        f = np.zeros_like(frame) if self._muted else frame
        self._on_frames(f.astype(np.float32))

    def push_audio(self, audio: np.ndarray) -> None:
        for frame in _chunk(audio):
            self.push(frame)

    def push_silence(self, n_frames: int = 20) -> None:
        silence = np.zeros(512, dtype=np.float32)
        for _ in range(n_frames):
            self.push(silence)

    @property
    def device(self) -> DeviceInfo:
        return self._device

    @property
    def running(self) -> bool:
        return self._running


# --------------------------------------------------------------------------------- #
# common test plumbing
# --------------------------------------------------------------------------------- #


def _make_pipeline(
    cfg: Config,
    bus: EventBus,
    transcriber: Any,
    clusterer: Any,
    capture_factory: Callable[..., Any],
    *,
    clock: Callable[[], float] | None = None,
) -> SpeechPipeline:
    ids = itertools.count(1)
    t0 = time.monotonic()
    return SpeechPipeline(
        cfg, bus, transcriber, clusterer,
        clock=clock or (lambda: time.monotonic() - t0),
        next_id=lambda: next(ids),
        capture_factory=capture_factory,
    )


def _wait_until(predicate: Callable[[], bool], timeout: float = 10.0, interval: float = 0.05) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


# --------------------------------------------------------------------------------- #
# main end-to-end test: full file -> ordered finals matching ground truth
# --------------------------------------------------------------------------------- #


def test_finals_in_order_matching_truth_with_partials_first(two_speakers_audio, silero_path):
    audio, _sr = two_speakers_audio
    cfg = _make_cfg(capture_system=True, capture_mic=False)
    bus = EventBus()
    q = bus.subscribe()
    transcriber = FakeTranscriber()
    clusterer = FakeClusterer()

    holder: dict[Source, _AutoFeedCapture] = {}

    def factory(source: Source, spec: str, **kw: Any) -> _AutoFeedCapture:
        cap = _AutoFeedCapture(source, spec, audio=audio, speed=12.0, device_name="Fake Loopback", **kw)
        holder[source] = cap
        return cap

    pipeline = _make_pipeline(cfg, bus, transcriber, clusterer, factory)
    pipeline.start()
    try:
        assert _wait_until(lambda: Source.SYSTEM in holder, timeout=2.0)
        assert _wait_until(lambda: not holder[Source.SYSTEM].is_playing(), timeout=15.0)
        time.sleep(0.5)  # let the asr worker drain the tail of the queue
    finally:
        pipeline.stop()

    events = drain(q, max_items=10000)
    finals = [e.segment for e in events if isinstance(e, FinalTranscriptEvent)]
    partials = [e for e in events if isinstance(e, PartialTranscriptEvent)]

    assert len(finals) >= 8
    ids_seen = [s.id for s in finals]
    assert ids_seen == sorted(ids_seen)
    assert len(set(ids_seen)) == len(ids_seen), "segment ids must be unique"
    assert all(s.source is Source.SYSTEM for s in finals)
    assert not any(s.is_question for s in finals), "default fake text ('utt-N') is never a question"

    for got, truth in zip(finals, TWO_SPEAKERS_TRUTH):
        t_start_truth, t_end_truth = truth[0], truth[1]
        assert got.t_start == pytest.approx(t_start_truth, abs=0.3)
        assert got.t_end == pytest.approx(t_end_truth, abs=0.3)

    nonempty_partials = [p for p in partials if p.text]
    assert nonempty_partials, "expected at least one non-empty PartialTranscriptEvent"
    first_partial_ts = min(p.ts for p in nonempty_partials)
    first_final_ts = min(e.ts for e in events if isinstance(e, FinalTranscriptEvent))
    assert first_partial_ts < first_final_ts

    audio_status = [e for e in events if isinstance(e, ComponentStatusEvent) and e.component == "audio"]
    assert any(e.state == "ready" and "Fake Loopback" in e.detail for e in audio_status)

    diar_status = [e for e in events if isinstance(e, ComponentStatusEvent) and e.component == "diarization"]
    assert any(e.state == "ready" for e in diar_status)

    levels = [e for e in events if isinstance(e, AudioLevelEvent)]
    assert levels, "expected at least one throttled AudioLevelEvent"


# --------------------------------------------------------------------------------- #
# is_question wiring
# --------------------------------------------------------------------------------- #


@pytest.mark.parametrize("truth_index", _QUESTION_TRUTH_INDICES)
def test_is_question_true_for_question_segments(two_speakers_audio, silero_path, truth_index):
    audio, _sr = two_speakers_audio
    truth_text = TWO_SPEAKERS_TRUTH[truth_index - 1][3]
    cfg = _make_cfg(capture_system=True, capture_mic=False)
    bus = EventBus()
    q = bus.subscribe()
    transcriber = FakeTranscriber(final_text_overrides={truth_index: truth_text})

    holder: dict[Source, _AutoFeedCapture] = {}

    def factory(source: Source, spec: str, **kw: Any) -> _AutoFeedCapture:
        cap = _AutoFeedCapture(source, spec, audio=audio, speed=15.0, device_name="Fake Loopback", **kw)
        holder[source] = cap
        return cap

    pipeline = _make_pipeline(cfg, bus, transcriber, None, factory)
    pipeline.start()
    try:
        assert _wait_until(lambda: Source.SYSTEM in holder, timeout=2.0)
        assert _wait_until(lambda: not holder[Source.SYSTEM].is_playing(), timeout=15.0)
        time.sleep(0.5)
    finally:
        pipeline.stop()

    finals = [e.segment for e in drain(q, max_items=10000) if isinstance(e, FinalTranscriptEvent)]
    assert len(finals) >= truth_index
    segment = finals[truth_index - 1]
    assert segment.text == truth_text
    assert segment.is_question is True


# --------------------------------------------------------------------------------- #
# echo suppression
# --------------------------------------------------------------------------------- #


def test_echo_suppression_drops_duplicate_mic_final(two_speakers_audio, silero_path):
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
        sys_snippet = audio[int(3.17 * sr) : int(6.86 * sr)]
        mic_snippet = audio[int(9.03 * sr) : int(13.81 * sr)]

        holder[Source.SYSTEM].push_audio(sys_snippet)
        holder[Source.SYSTEM].push_silence()
        assert _wait_until(lambda: transcriber.calls >= 1, timeout=5.0)
        time.sleep(0.3)

        holder[Source.MIC].push_audio(mic_snippet)
        holder[Source.MIC].push_silence()
        time.sleep(0.6)
    finally:
        pipeline.stop()

    events = drain(q, max_items=10000)
    finals = [e.segment for e in events if isinstance(e, FinalTranscriptEvent)]
    assert len(finals) == 1, f"expected the MIC duplicate to be echo-suppressed, got {finals!r}"
    assert finals[0].source is Source.SYSTEM


def test_echo_suppression_disabled_keeps_both_finals(two_speakers_audio, silero_path):
    audio, sr = two_speakers_audio
    cfg = _make_cfg(capture_system=True, capture_mic=True, echo_suppression=False)
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
        sys_snippet = audio[int(3.17 * sr) : int(6.86 * sr)]
        mic_snippet = audio[int(9.03 * sr) : int(13.81 * sr)]

        holder[Source.SYSTEM].push_audio(sys_snippet)
        holder[Source.SYSTEM].push_silence()
        assert _wait_until(lambda: transcriber.calls >= 1, timeout=5.0)
        time.sleep(0.3)

        holder[Source.MIC].push_audio(mic_snippet)
        holder[Source.MIC].push_silence()
        time.sleep(0.6)
    finally:
        pipeline.stop()

    finals = [e.segment for e in drain(q, max_items=10000) if isinstance(e, FinalTranscriptEvent)]
    assert len(finals) == 2
    assert {s.source for s in finals} == {Source.SYSTEM, Source.MIC}


# upstream finding 7 (ported from mac-cortex): the echo-suppressed-final log line must never
# carry the recognised text, at any level - only its length. Drives the real echo-suppression
# path (two sources, same text, within `_ECHO_WINDOW_S`; see `_should_suppress_echo`) and
# inspects every log record emitted while it runs.
def test_echo_suppressed_log_never_contains_recognized_text(two_speakers_audio, silero_path, caplog):
    audio, sr = two_speakers_audio
    cfg = _make_cfg(capture_system=True, capture_mic=True, echo_suppression=True)
    bus = EventBus()
    q = bus.subscribe()
    spoken_text = "we should ship the migration plan by friday"
    transcriber = FixedTextTranscriber(spoken_text)

    holder: dict[Source, _ManualCapture] = {}

    def factory(source: Source, spec: str, **kw: Any) -> _ManualCapture:
        cap = _ManualCapture(source, spec, device_name=f"fake-{source.value}", **kw)
        holder[source] = cap
        return cap

    pipeline = _make_pipeline(cfg, bus, transcriber, FakeClusterer(), factory)
    caplog.set_level(logging.DEBUG, logger="exocortex.speech.pipeline")
    pipeline.start()
    try:
        sys_snippet = audio[int(3.17 * sr) : int(6.86 * sr)]
        mic_snippet = audio[int(9.03 * sr) : int(13.81 * sr)]

        holder[Source.SYSTEM].push_audio(sys_snippet)
        holder[Source.SYSTEM].push_silence()
        assert _wait_until(lambda: transcriber.calls >= 1, timeout=5.0)
        time.sleep(0.3)

        holder[Source.MIC].push_audio(mic_snippet)
        holder[Source.MIC].push_silence()
        time.sleep(0.6)
    finally:
        pipeline.stop()

    events = drain(q, max_items=10000)
    finals = [e.segment for e in events if isinstance(e, FinalTranscriptEvent)]
    assert len(finals) == 1, f"expected the MIC duplicate to be echo-suppressed, got {finals!r}"

    messages = [record.getMessage() for record in caplog.records]
    assert any("echo-suppressed" in m for m in messages), "expected an echo-suppressed log record"
    assert not any(spoken_text in m for m in messages), (
        f"spoken text leaked into the log: {[m for m in messages if spoken_text in m]!r}"
    )


# --------------------------------------------------------------------------------- #
# mute
# --------------------------------------------------------------------------------- #


def test_mute_publishes_events_and_suppresses_mic_finals(two_speakers_audio, silero_path):
    audio, sr = two_speakers_audio
    cfg = _make_cfg(capture_system=False, capture_mic=True)
    bus = EventBus()
    q = bus.subscribe()
    transcriber = FakeTranscriber()

    holder: dict[Source, _ManualCapture] = {}

    def factory(source: Source, spec: str, **kw: Any) -> _ManualCapture:
        cap = _ManualCapture(source, spec, device_name="fake-mic", **kw)
        holder[source] = cap
        return cap

    pipeline = _make_pipeline(cfg, bus, transcriber, None, factory)
    pipeline.start()
    try:
        assert _wait_until(lambda: Source.MIC in holder, timeout=2.0)
        assert pipeline.mic_muted is False

        pipeline.set_mic_muted(True)
        assert pipeline.mic_muted is True

        snippet = audio[int(3.17 * sr) : int(6.86 * sr)]
        holder[Source.MIC].push_audio(snippet)
        holder[Source.MIC].push_silence()
        time.sleep(0.5)
    finally:
        pipeline.stop()

    events = drain(q, max_items=10000)
    finals = [e for e in events if isinstance(e, FinalTranscriptEvent)]
    assert not finals, "a muted capture delivers silence; no speech should ever be detected"

    mute_events = [e for e in events if isinstance(e, AudioStatusEvent) and e.state == "muted"]
    assert len(mute_events) == 1
    assert mute_events[0].source is Source.MIC


# --------------------------------------------------------------------------------- #
# upstream finding 12 (ported from mac-cortex): `_asr_busy` makes `_enqueue_partial` stash
# (never queue) a partial - never a final - while the one ASR worker thread is mid-transcribe;
# `_asr_worker_loop` flushes whatever is stashed onto `_asr_queue` the moment it frees up, so
# the *last* partial for a source is always eventually queued, never silently lost. These
# drive `_enqueue_partial` / `_enqueue_final` / `_flush_pending_partials` directly (no running
# pipeline needed) and inspect `_asr_queue` / `_pending_partial`.
# --------------------------------------------------------------------------------- #


def test_enqueue_partial_skipped_while_asr_worker_busy():
    cfg = _make_cfg()
    bus = EventBus()
    pipeline = _make_pipeline(
        cfg, bus, FakeTranscriber(), None, lambda source, spec, **kw: _ManualCapture(source, spec, **kw)
    )
    event = SpeechOngoing(t_start=0.0, audio=np.zeros(1600, dtype=np.float32))

    pipeline._asr_busy.set()
    pipeline._enqueue_partial(Source.SYSTEM, event)
    assert pipeline._asr_queue.qsize() == 0, "a partial must not be queued while the worker is busy"

    pipeline._asr_busy.clear()
    pipeline._enqueue_partial(Source.SYSTEM, event)
    assert pipeline._asr_queue.qsize() == 1, "a partial must still be queued once the worker is free"


def test_enqueue_final_not_skipped_while_asr_worker_busy():
    cfg = _make_cfg()
    bus = EventBus()
    pipeline = _make_pipeline(
        cfg, bus, FakeTranscriber(), None, lambda source, spec, **kw: _ManualCapture(source, spec, **kw)
    )
    event = SpeechEnd(t_start=0.0, t_end=0.1, audio=np.zeros(1600, dtype=np.float32))

    pipeline._asr_busy.set()
    pipeline._enqueue_final(Source.SYSTEM, event)
    assert pipeline._asr_queue.qsize() == 1, "finals must never be skipped, busy worker or not"


def test_pending_partial_flushed_once_asr_worker_frees_up():
    """Regression test for a bug in an earlier version of the finding-12 mitigation: gating
    `_enqueue_partial` purely on "worker busy" (dropping the job outright instead of stashing
    it) silently discarded the *last* partial too whenever it happened to arrive while busy,
    even though nothing later would ever supersede it - freezing that source's live caption
    until the next SpeechOngoing event. Only a genuinely superseded partial may be dropped;
    the still-current one must always get queued once the worker frees up. This is what
    tests/test_pipeline_extra.py::
    test_slow_transcribe_never_publishes_or_runs_a_superseded_partial pins end to end with a
    real blocked-in-transcribe() worker; this test isolates the same contract at the
    `_enqueue_partial` / `_flush_pending_partials` level, with no running worker thread."""
    cfg = _make_cfg()
    bus = EventBus()
    pipeline = _make_pipeline(
        cfg, bus, FakeTranscriber(), None, lambda source, spec, **kw: _ManualCapture(source, spec, **kw)
    )
    audio = np.zeros(1600, dtype=np.float32)

    pipeline._asr_busy.set()
    # Three partials arrive while busy; each supersedes the last. Only the third (t_start=0.2)
    # is still current once the worker frees up.
    pipeline._enqueue_partial(Source.SYSTEM, SpeechOngoing(t_start=0.0, audio=audio))
    pipeline._enqueue_partial(Source.SYSTEM, SpeechOngoing(t_start=0.1, audio=audio))
    pipeline._enqueue_partial(Source.SYSTEM, SpeechOngoing(t_start=0.2, audio=audio))
    assert pipeline._asr_queue.qsize() == 0, "none of them may be queued while the worker is busy"
    assert len(pipeline._pending_partial) == 1, "only the most recent stashed job survives per source"

    pipeline._asr_busy.clear()
    pipeline._flush_pending_partials()

    assert pipeline._asr_queue.qsize() == 1, "the still-current (most recent) partial must reach the queue"
    queued = pipeline._asr_queue.get_nowait()
    assert queued.t_start == pytest.approx(0.2), (
        f"expected the last (still-current) partial to be flushed, got t_start={queued.t_start}"
    )
    assert not pipeline._pending_partial, "the pending slot must be cleared after flushing"


# --------------------------------------------------------------------------------- #
# stop() joins every worker thread
# --------------------------------------------------------------------------------- #


def test_stop_joins_all_worker_threads(two_speakers_audio, silero_path):
    cfg = _make_cfg(capture_system=True, capture_mic=True)
    bus = EventBus()
    transcriber = FakeTranscriber()
    clusterer = FakeClusterer()

    def factory(source: Source, spec: str, **kw: Any) -> _ManualCapture:
        return _ManualCapture(source, spec, device_name=f"fake-{source.value}", **kw)

    pipeline = _make_pipeline(cfg, bus, transcriber, clusterer, factory)
    pipeline.start()
    time.sleep(0.1)
    pipeline.stop()

    names = {t.name for t in threading.enumerate()}
    assert "asr-worker" not in names
    assert "vad-mic" not in names
    assert "vad-system" not in names


# --------------------------------------------------------------------------------- #
# force-split of a long, gap-free monologue
# --------------------------------------------------------------------------------- #


def test_force_splits_long_monologue_into_multiple_finals(two_speakers_audio, silero_path):
    audio, sr = two_speakers_audio
    # A single long sentence repeated back-to-back with zero silence between repeats: sherpa's
    # VAD never sees a pause, so without our own force-split this would stay one utterance
    # (verified empirically against the real model) - exactly the case SPEC.md calls out.
    sentence = audio[int(3.17 * sr) : int(6.86 * sr)]
    continuous = np.concatenate([sentence] * 6)  # ~22s, no internal silence

    cfg = _make_cfg(capture_system=True, capture_mic=False, max_utterance_s=4.0, min_silence_s=0.3)
    bus = EventBus()
    q = bus.subscribe()
    transcriber = FakeTranscriber()

    holder: dict[Source, _AutoFeedCapture] = {}

    def factory(source: Source, spec: str, **kw: Any) -> _AutoFeedCapture:
        cap = _AutoFeedCapture(source, spec, audio=continuous, speed=20.0, device_name="fake-sys", **kw)
        holder[source] = cap
        return cap

    pipeline = _make_pipeline(cfg, bus, transcriber, None, factory)
    pipeline.start()
    try:
        assert _wait_until(lambda: Source.SYSTEM in holder, timeout=2.0)
        assert _wait_until(lambda: not holder[Source.SYSTEM].is_playing(), timeout=15.0)
        time.sleep(0.5)
    finally:
        pipeline.stop()

    finals = [e.segment for e in drain(q, max_items=10000) if isinstance(e, FinalTranscriptEvent)]
    assert len(finals) >= 3, f"expected the monologue to be force-split into several finals, got {len(finals)}"
    for seg in finals[:-1]:
        assert seg.duration <= cfg.audio.max_utterance_s + 0.75


# --------------------------------------------------------------------------------- #
# time-origin realignment after a frame-arrival gap
# --------------------------------------------------------------------------------- #


def test_gap_realignment_after_frame_silence(two_speakers_audio, silero_path):
    audio, sr = two_speakers_audio
    cfg = _make_cfg(capture_system=True, capture_mic=False, min_silence_s=0.3, pre_roll_s=0.3, min_speech_s=0.25)
    bus = EventBus()
    q = bus.subscribe()
    transcriber = FakeTranscriber()
    t0 = time.monotonic()
    clock = lambda: time.monotonic() - t0

    holder: dict[Source, _ManualCapture] = {}

    def factory(source: Source, spec: str, **kw: Any) -> _ManualCapture:
        cap = _ManualCapture(source, spec, device_name="fake-sys", **kw)
        holder[source] = cap
        return cap

    pipeline = _make_pipeline(cfg, bus, transcriber, None, factory, clock=clock)
    pipeline.start()
    try:
        assert _wait_until(lambda: Source.SYSTEM in holder, timeout=2.0)
        holder[Source.SYSTEM].push_silence(10)

        time.sleep(1.5)  # > _GAP_REALIGN_S with no frames delivered, and not mid-utterance
        resume_clock = clock()

        snippet = audio[int(3.17 * sr) : int(6.86 * sr)]
        holder[Source.SYSTEM].push_audio(snippet)
        time.sleep(0.4)  # let the vad worker register speech before stop() flushes it
    finally:
        pipeline.stop()  # flushes the still-in-progress speech into a trailing final

    finals = [e.segment for e in drain(q, max_items=10000) if isinstance(e, FinalTranscriptEvent)]
    assert finals, "expected the trailing speech to be flushed into a final on stop()"
    assert finals[0].t_start == pytest.approx(resume_clock, abs=0.6)


# --------------------------------------------------------------------------------- #
# rename_speaker
# --------------------------------------------------------------------------------- #


def test_rename_speaker_forwards_to_clusterer_and_publishes_event():
    from exocortex.bus import SpeakerRenamedEvent

    cfg = _make_cfg()
    bus = EventBus()
    q = bus.subscribe()
    clusterer = FakeClusterer()
    pipeline = _make_pipeline(cfg, bus, FakeTranscriber(), clusterer, lambda source, spec, **kw: _ManualCapture(source, spec, **kw))

    pipeline.rename_speaker("spk1", "Alice")

    assert clusterer.renames == [("spk1", "Alice")]
    events = drain(q)
    renamed = [e for e in events if isinstance(e, SpeakerRenamedEvent)]
    assert len(renamed) == 1
    assert renamed[0].speaker_id == "spk1"
    assert renamed[0].new_label == "Alice"


# --------------------------------------------------------------------------------- #
# live: real Transcriber + real SpeakerClusterer end to end
# --------------------------------------------------------------------------------- #


@pytest.mark.live
def test_live_real_transcriber_and_clusterer(two_speakers_audio, silero_path, speaker_model_path):
    from exocortex.config import ASRConfig, DiarizationConfig
    from exocortex.speech.diarize import SpeakerClusterer
    from exocortex.speech.asr import Transcriber

    audio, _sr = two_speakers_audio

    asr_cfg = ASRConfig(model="large-v3-turbo", device="cuda", compute_type="float16", beam_size=1)
    transcriber = Transcriber(asr_cfg)
    transcriber.load()

    diar_cfg = DiarizationConfig()
    clusterer = SpeakerClusterer(
        speaker_model_path, threshold=diar_cfg.threshold, enroll_min_s=diar_cfg.enroll_min_s,
        max_speakers=diar_cfg.max_speakers, num_threads=diar_cfg.num_threads,
    )

    cfg = _make_cfg(capture_system=True, capture_mic=False)
    bus = EventBus()
    q = bus.subscribe()

    holder: dict[Source, _AutoFeedCapture] = {}

    def factory(source: Source, spec: str, **kw: Any) -> _AutoFeedCapture:
        cap = _AutoFeedCapture(source, spec, audio=audio, speed=4.0, device_name="Fake Loopback", **kw)
        holder[source] = cap
        return cap

    pipeline = _make_pipeline(cfg, bus, transcriber, clusterer, factory)
    pipeline.start()
    try:
        assert _wait_until(lambda: Source.SYSTEM in holder, timeout=2.0)
        assert _wait_until(lambda: not holder[Source.SYSTEM].is_playing(), timeout=30.0)
        time.sleep(2.0)
    finally:
        pipeline.stop()

    finals = [e.segment for e in drain(q, max_items=10000) if isinstance(e, FinalTranscriptEvent)]
    assert len(finals) >= 8

    long_truth = [t for t in TWO_SPEAKERS_TRUTH if t[1] - t[0] >= 1.5]
    matched = 0
    for seg, truth in zip(finals, TWO_SPEAKERS_TRUTH):
        truth_words = set(truth[3].lower().rstrip(".?!").split())
        seg_words = set(seg.text.lower().rstrip(".?!").split())
        if not truth_words:
            continue
        overlap = len(truth_words & seg_words) / len(truth_words)
        if overlap >= 0.7:
            matched += 1
    assert matched / len(TWO_SPEAKERS_TRUTH) >= 0.7

    speaker_ids = {seg.speaker_id for seg in finals if seg.duration >= 1.5}
    assert len(speaker_ids) >= 2

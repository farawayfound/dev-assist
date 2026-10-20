"""Silero VAD segmentation via sherpa-onnx.

`VadSegmenter` turns a stream of fixed-size 16 kHz mono float32 frames into
`SpeechStart` / `SpeechOngoing` / `SpeechEnd` events. It wraps
`sherpa_onnx.VoiceActivityDetector` (see SPEC.md "Verified facts" for the exact
API used) and adds:
  - a small pre-roll ring buffer so `SpeechOngoing`/`SpeechStart` audio includes
    a bit of audio from just before the VAD flagged speech,
  - a fed-sample counter translated into session-clock time via `set_time_origin`,
  - throttling of `SpeechOngoing` to at most once per `ongoing_interval_s`.

This module has no import-time side effects: constructing a `sherpa_onnx`
config and a `VoiceActivityDetector` only happens inside `VadSegmenter.__init__`,
never at import time.
"""
from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass

import numpy as np
import sherpa_onnx

log = logging.getLogger(__name__)

#: The Silero VAD model bundled in models/ (and sherpa-onnx generally) operates
#: on fixed 512-sample windows at 16 kHz. `process()` requires frames of exactly
#: this length.
WINDOW_SIZE = 512


@dataclass(frozen=True)
class SpeechStart:
    """A speech onset was detected. `t_start` includes the configured pre-roll."""

    t_start: float


@dataclass(frozen=True)
class SpeechOngoing:
    """Periodic snapshot of the audio accumulated since the matching `SpeechStart`."""

    t_start: float
    audio: np.ndarray  # audio since start (incl. pre-roll)


@dataclass(frozen=True)
class SpeechEnd:
    """A finalised speech segment, as reported by the underlying sherpa-onnx VAD."""

    t_start: float
    t_end: float
    audio: np.ndarray


VadEvent = SpeechStart | SpeechOngoing | SpeechEnd


class VadSegmenter:
    """Wraps sherpa-onnx's Silero VAD and turns raw frames into `VadEvent`s.

    `process(frame)` takes exactly `WINDOW_SIZE` (512) float32 samples at
    `sample_rate` and returns zero or more events. Time is derived from the
    number of samples fed so far, offset by `set_time_origin` so the caller can
    align event timestamps to its own session clock.
    """

    def __init__(
        self,
        model_path: str,
        *,
        sample_rate: int = 16000,
        threshold: float = 0.5,
        min_silence_s: float = 0.45,
        min_speech_s: float = 0.25,
        max_speech_s: float = 20.0,
        pre_roll_s: float = 0.3,
        ongoing_interval_s: float = 1.2,
        num_threads: int = 1,
    ) -> None:
        cfg = sherpa_onnx.VadModelConfig()
        cfg.silero_vad.model = model_path
        cfg.silero_vad.threshold = threshold
        cfg.silero_vad.min_silence_duration = min_silence_s
        cfg.silero_vad.min_speech_duration = min_speech_s
        cfg.silero_vad.max_speech_duration = max_speech_s
        cfg.silero_vad.window_size = WINDOW_SIZE
        cfg.sample_rate = sample_rate
        cfg.num_threads = num_threads
        self._vad = sherpa_onnx.VoiceActivityDetector(cfg, buffer_size_in_seconds=60)

        self._sample_rate = sample_rate
        self._min_speech_s = min_speech_s
        self._max_speech_s = max_speech_s
        self._ongoing_interval_s = ongoing_interval_s
        self._pre_roll_samples = int(round(pre_roll_s * sample_rate))

        # Ring buffer covers pre-roll + the longest possible in-progress utterance,
        # with a little slack so trimming never eats into needed samples.
        capacity_s = pre_roll_s + max_speech_s
        self._ring_capacity = int(math.ceil(capacity_s * sample_rate)) + WINDOW_SIZE
        self._ring_frames: deque[np.ndarray] = deque()
        self._ring_len = 0
        self._ring_first_sample = 0

        # Time mapping: time_of(sample_index) = t0 + (sample_index - origin_sample) / sample_rate
        self._t0 = 0.0
        self._origin_sample = 0

        self._total_fed = 0
        self._was_speech = False
        self._speech_start_sample: int | None = None
        self._last_ongoing_sample: int | None = None

        log.debug(
            "VadSegmenter: sr=%d threshold=%.2f min_silence_s=%.2f min_speech_s=%.2f "
            "max_speech_s=%.2f pre_roll_s=%.2f ongoing_interval_s=%.2f",
            sample_rate, threshold, min_silence_s, min_speech_s, max_speech_s,
            pre_roll_s, ongoing_interval_s,
        )

    def set_time_origin(self, t0: float) -> None:
        """Declare `t0` as the session-clock time of the *next* sample fed to `process`."""
        self._t0 = t0
        self._origin_sample = self._total_fed

    def process(self, frame: np.ndarray) -> list[VadEvent]:
        """Feed exactly `WINDOW_SIZE` samples and return the events this produced."""
        assert len(frame) == WINDOW_SIZE, (
            f"VadSegmenter.process expects exactly {WINDOW_SIZE} samples, got {len(frame)}"
        )
        frame = np.asarray(frame, dtype=np.float32)
        events: list[VadEvent] = []

        self._push_ring(frame)
        self._vad.accept_waveform(frame)
        self._total_fed += WINDOW_SIZE

        is_speech = self._vad.is_speech_detected()
        if is_speech and not self._was_speech:
            start_sample = max(0, self._total_fed - self._pre_roll_samples)
            self._speech_start_sample = start_sample
            self._last_ongoing_sample = None
            events.append(SpeechStart(t_start=self._time_of(start_sample)))
            log.debug("SpeechStart at sample %d", start_sample)
        self._was_speech = is_speech

        if is_speech and self._speech_start_sample is not None:
            events.extend(self._maybe_ongoing())

        events.extend(self._drain_finished())
        return events

    def flush(self) -> list[VadEvent]:
        """Finalise any buffered tail audio and drain the resulting segment(s)."""
        self._vad.flush()
        events = self._drain_finished()
        self._was_speech = False
        self._speech_start_sample = None
        self._last_ongoing_sample = None
        return events

    def reset(self) -> None:
        """Reset the VAD and all counters. The time origin (`t0`) is preserved."""
        self._vad.reset()
        self._total_fed = 0
        self._origin_sample = 0
        self._was_speech = False
        self._speech_start_sample = None
        self._last_ongoing_sample = None
        self._ring_frames.clear()
        self._ring_len = 0
        self._ring_first_sample = 0

    @property
    def in_speech(self) -> bool:
        """Whether the VAD currently considers itself inside a speech segment."""
        return self._vad.is_speech_detected()

    # ---- internals -----------------------------------------------------

    def _time_of(self, sample_index: int) -> float:
        return self._t0 + (sample_index - self._origin_sample) / self._sample_rate

    def _maybe_ongoing(self) -> list[VadEvent]:
        assert self._speech_start_sample is not None
        elapsed_s = (self._total_fed - self._speech_start_sample) / self._sample_rate
        if elapsed_s < self._min_speech_s:
            return []
        due = self._last_ongoing_sample is None or (
            (self._total_fed - self._last_ongoing_sample) / self._sample_rate
            >= self._ongoing_interval_s
        )
        if not due:
            return []
        audio = self._slice_ring(self._speech_start_sample, self._total_fed)
        self._last_ongoing_sample = self._total_fed
        return [SpeechOngoing(t_start=self._time_of(self._speech_start_sample), audio=audio)]

    def _drain_finished(self) -> list[VadEvent]:
        events: list[VadEvent] = []
        while not self._vad.empty():
            front = self._vad.front
            samples = np.asarray(front.samples, dtype=np.float32)
            t_start = self._time_of(front.start)
            t_end = t_start + len(samples) / self._sample_rate
            events.append(SpeechEnd(t_start=t_start, t_end=t_end, audio=samples))
            self._vad.pop()
            # A segment just ended: the next SpeechStart begins a fresh utterance.
            self._speech_start_sample = None
            self._last_ongoing_sample = None
        return events

    def _push_ring(self, frame: np.ndarray) -> None:
        self._ring_frames.append(frame)
        self._ring_len += len(frame)
        while self._ring_frames and self._ring_len - len(self._ring_frames[0]) >= self._ring_capacity:
            popped = self._ring_frames.popleft()
            self._ring_len -= len(popped)
            self._ring_first_sample += len(popped)

    def _slice_ring(self, start_sample: int, end_sample: int) -> np.ndarray:
        start = max(start_sample, self._ring_first_sample)
        cap_samples = int(round(self._max_speech_s * self._sample_rate))
        if end_sample - start > cap_samples:
            start = end_sample - cap_samples
        if not self._ring_frames:
            return np.zeros(0, dtype=np.float32)
        full = np.concatenate(self._ring_frames)
        offset_start = max(0, start - self._ring_first_sample)
        offset_end = min(len(full), end_sample - self._ring_first_sample)
        if offset_end <= offset_start:
            return np.zeros(0, dtype=np.float32)
        return np.array(full[offset_start:offset_end], dtype=np.float32)

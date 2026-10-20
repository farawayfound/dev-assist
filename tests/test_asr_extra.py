"""Verifier's additional focused tests for exocortex.speech.asr: boundary conditions,
aggregation edge cases, and thread-safety not covered by tests/test_asr.py.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest

from exocortex.config import ASRConfig
from exocortex.speech.asr import Transcriber

from test_asr import FakeModel, FakeSegment, audio_of, make_transcriber


# ---- hallucination boundary at exactly 1.5 s -------------------------------------------------

def test_hallucination_blocklist_boundary_exactly_1_5s_not_checked() -> None:
    """SPEC: blocklist applies to clips '< 1.5 s'. A clip of exactly 1.5 s must NOT be
    blocklist-checked, even though the text would match if it were."""
    t, model = make_transcriber()
    model.segments = [FakeSegment(text="Thank you.")]
    result = t.transcribe(audio_of(1.5))
    assert result.dropped is False


def test_hallucination_blocklist_boundary_just_under_1_5s_checked() -> None:
    t, model = make_transcriber()
    model.segments = [FakeSegment(text="Thank you.")]
    result = t.transcribe(audio_of(1.499))
    assert result.dropped is True
    assert "hallucination" in result.reason


# ---- avg_logprob / no_speech_prob aggregation edge cases -------------------------------------

def test_avg_logprob_falls_back_to_unweighted_mean_when_total_duration_zero() -> None:
    """All segments have zero duration (start == end): weighting by duration would divide by
    zero, so the implementation must fall back to a plain arithmetic mean."""
    t, model = make_transcriber()
    model.segments = [
        FakeSegment(text="a", start=0.0, end=0.0, avg_logprob=-0.2),
        FakeSegment(text="b", start=5.0, end=5.0, avg_logprob=-0.8),
    ]
    result = t.transcribe(audio_of(2.0))
    assert result.avg_logprob == pytest.approx(-0.5)


def test_no_segments_at_all_yields_zero_logprob_and_no_speech_prob() -> None:
    t, model = make_transcriber()
    model.segments = []
    result = t.transcribe(audio_of(1.0))
    assert result.dropped is True
    assert result.reason == "empty text"
    assert result.avg_logprob == 0.0
    assert result.no_speech_prob == 0.0


# ---- punctuation / music-only text -------------------------------------------------------------

def test_music_note_only_text_is_dropped() -> None:
    t, model = make_transcriber()
    model.segments = [FakeSegment(text="♪♪♪")]
    result = t.transcribe(audio_of(1.0))
    assert result.dropped is True
    assert "punctuation" in result.reason


def test_curly_quotes_and_ellipsis_only_text_is_dropped() -> None:
    t, model = make_transcriber()
    model.segments = [FakeSegment(text="“…”")]
    result = t.transcribe(audio_of(1.0))
    assert result.dropped is True


def test_punctuation_only_rule_not_applied_to_long_clips() -> None:
    """The punctuation/music-only rule is part of the < 1.5 s hallucination check; a long
    clip whose text happens to be only punctuation should not be dropped by it (matches the
    blocklist's own duration gate)."""
    t, model = make_transcriber()
    model.segments = [FakeSegment(text="...")]
    result = t.transcribe(audio_of(3.0))
    assert result.dropped is False


# ---- hotwords edge cases -----------------------------------------------------------------------

def test_whitespace_only_hotwords_treated_as_no_prompt() -> None:
    cfg = ASRConfig(hotwords=["   ", ""])
    t, model = make_transcriber(cfg)
    model.segments = [FakeSegment(text="hello")]
    t.transcribe(audio_of(2.0))
    assert model.calls[-1]["initial_prompt"] is None


# ---- properties before load() ------------------------------------------------------------------

def test_device_and_compute_type_used_are_empty_before_load() -> None:
    def factory(model_size_or_path: str, device: str, compute_type: str, cpu_threads: int) -> FakeModel:
        return FakeModel()

    t = Transcriber(ASRConfig(), model_factory=factory)
    assert t.device_used == ""
    assert t.compute_type_used == ""


# ---- empty audio array ---------------------------------------------------------------------------

def test_transcribe_zero_length_audio_does_not_crash() -> None:
    t, model = make_transcriber()
    model.segments = [FakeSegment(text="")]
    result = t.transcribe(np.zeros(0, dtype=np.float32))
    assert result.audio_s == 0.0
    assert result.dropped is True


# ---- thread-safety: transcribe() serializes concurrent callers ------------------------------------

@dataclass
class _SlowFakeModel:
    """Records the max number of simultaneously-active `transcribe()` calls, to verify that
    `Transcriber.transcribe()` really serializes access to the underlying model via its lock."""

    active: int = 0
    max_active: int = 0
    calls: int = 0
    _state_lock: threading.Lock = field(default_factory=threading.Lock)

    def transcribe(self, audio: np.ndarray, **kwargs: Any) -> tuple[list[FakeSegment], None]:
        with self._state_lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.calls += 1
        time.sleep(0.02)
        with self._state_lock:
            self.active -= 1
        return [FakeSegment(text="hi")], None


def test_transcribe_serializes_concurrent_calls_via_internal_lock() -> None:
    slow_model = _SlowFakeModel()

    def factory(model_size_or_path: str, device: str, compute_type: str, cpu_threads: int) -> _SlowFakeModel:
        return slow_model

    # engine="faster-whisper": see test_asr.py::make_transcriber's docstring -- this test builds
    # its own Transcriber directly (not via make_transcriber) but must pin the engine the same way.
    t = Transcriber(ASRConfig(engine="faster-whisper"), model_factory=factory)
    t.load()  # warm-up call included

    errors: list[BaseException] = []

    def worker() -> None:
        try:
            t.transcribe(audio_of(1.0))
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for th in threads:
        th.start()
    for th in threads:
        th.join(timeout=5)

    assert not errors
    assert slow_model.max_active == 1, "concurrent transcribe() calls were not serialized by the lock"
    assert slow_model.calls == 9  # 1 warm-up + 8 concurrent

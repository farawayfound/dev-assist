"""Unit tests for exocortex.speech.vad — offline, CPU only, using the real Silero VAD ONNX
model (models/silero_vad.onnx) and tests/data/two_speakers_16k.wav. Ground truth and fixtures
come from tests/conftest.py.
"""
from __future__ import annotations

import numpy as np
import pytest

from exocortex.speech.vad import (
    WINDOW_SIZE,
    SpeechEnd,
    SpeechOngoing,
    SpeechStart,
    VadEvent,
    VadSegmenter,
)

from conftest import TWO_SPEAKERS_TRUTH

# SPEC.md's default min_silence_s (0.45s, matching AudioConfig.min_silence_s) merges the
# first two sentences of the test file ("Hi everyone!" / "Thanks for joining.", separated by
# only ~0.37s of silence) into a single VAD segment, giving 8 segments instead of the 9 in
# TWO_SPEAKERS_TRUTH (test_default_min_silence_matches_spec_default locks this in). A shorter
# gap splits them correctly and still matches every other boundary within a few ms, so the
# rest of these tests use that value. See spec_issues in the module report.
MIN_SILENCE_S = 0.3
PRE_ROLL_S = 0.3


def _feed(segmenter: VadSegmenter, audio: np.ndarray) -> list[VadEvent]:
    """Push `audio` through `segmenter` in exact WINDOW_SIZE frames, collecting all events."""
    events: list[VadEvent] = []
    i = 0
    n = len(audio)
    while i + WINDOW_SIZE <= n:
        events.extend(segmenter.process(audio[i : i + WINDOW_SIZE]))
        i += WINDOW_SIZE
    return events


def _make_segmenter(silero_path: str, **overrides: float) -> VadSegmenter:
    kwargs: dict = dict(min_silence_s=MIN_SILENCE_S, pre_roll_s=PRE_ROLL_S)
    kwargs.update(overrides)
    return VadSegmenter(silero_path, **kwargs)


def _ends(events: list[VadEvent]) -> list[SpeechEnd]:
    return [e for e in events if isinstance(e, SpeechEnd)]


def _starts(events: list[VadEvent]) -> list[SpeechStart]:
    return [e for e in events if isinstance(e, SpeechStart)]


def test_full_file_produces_nine_segments_matching_truth(two_speakers_audio, silero_path):
    audio, _sr = two_speakers_audio
    seg = _make_segmenter(silero_path)
    events = _feed(seg, audio)
    events.extend(seg.flush())

    ends = _ends(events)
    assert len(ends) == 9
    for got, truth in zip(ends, TWO_SPEAKERS_TRUTH):
        t_start_truth, t_end_truth = truth[0], truth[1]
        assert got.t_start == pytest.approx(t_start_truth, abs=0.15)
        assert got.t_end == pytest.approx(t_end_truth, abs=0.15)
        assert got.audio.dtype == np.float32
        assert len(got.audio) > 0
        assert len(got.audio) / 16000 == pytest.approx(got.t_end - got.t_start, abs=1e-3)


def test_default_min_silence_matches_spec_default(two_speakers_audio, silero_path):
    """Documents the spec_issue: SPEC.md's own default (min_silence_s=0.45) merges two
    closely-spoken sentences in this file, yielding 8 segments rather than 9."""
    audio, _sr = two_speakers_audio
    seg = VadSegmenter(silero_path)  # all defaults, incl. min_silence_s=0.45
    events = _feed(seg, audio)
    events.extend(seg.flush())
    assert len(_ends(events)) == 8


def test_speech_start_precedes_each_speech_end(two_speakers_audio, silero_path):
    audio, _sr = two_speakers_audio
    seg = _make_segmenter(silero_path)
    events = _feed(seg, audio)
    events.extend(seg.flush())

    assert len(_ends(events)) == 9

    open_starts = 0
    for event in events:
        if isinstance(event, SpeechStart):
            open_starts += 1
        elif isinstance(event, SpeechEnd):
            assert open_starts > 0, "SpeechEnd emitted without a preceding SpeechStart"
            open_starts -= 1
    assert open_starts == 0, "a SpeechStart was never matched by a SpeechEnd"


def test_ongoing_grows_monotonically_and_aligns_with_speech_end(two_speakers_audio, silero_path):
    audio, _sr = two_speakers_audio
    seg = _make_segmenter(silero_path)
    events = _feed(seg, audio)
    events.extend(seg.flush())

    # Group events into runs: SpeechStart, zero or more SpeechOngoing, then SpeechEnd.
    runs: list[dict] = []
    current: dict | None = None
    for event in events:
        if isinstance(event, SpeechStart):
            current = {"ongoing": [], "end": None}
        elif isinstance(event, SpeechOngoing):
            assert current is not None
            current["ongoing"].append(event)
        elif isinstance(event, SpeechEnd):
            assert current is not None
            current["end"] = event
            runs.append(current)
            current = None

    assert len(runs) == 9

    runs_with_multiple_ongoing = 0
    for run in runs:
        ongoing: list[SpeechOngoing] = run["ongoing"]
        end: SpeechEnd = run["end"]
        for snapshot in ongoing:
            assert snapshot.t_start == pytest.approx(end.t_start, abs=PRE_ROLL_S + 0.1)
        if len(ongoing) >= 2:
            runs_with_multiple_ongoing += 1
            lengths = [len(o.audio) for o in ongoing]
            assert all(b > a for a, b in zip(lengths, lengths[1:])), (
                "SpeechOngoing.audio should grow monotonically within a run"
            )

    # The longer sentences ("Today we need to decide...", "What would you say...", etc.)
    # comfortably exceed ongoing_interval_s (1.2s default) and should snapshot more than once.
    assert runs_with_multiple_ongoing >= 2


def test_time_origin_offset_shifts_all_event_times(two_speakers_audio, silero_path):
    audio, _sr = two_speakers_audio
    offset = 1000.0

    seg_a = _make_segmenter(silero_path)
    events_a = _feed(seg_a, audio)
    events_a.extend(seg_a.flush())

    seg_b = _make_segmenter(silero_path)
    seg_b.set_time_origin(offset)
    events_b = _feed(seg_b, audio)
    events_b.extend(seg_b.flush())

    ends_a, ends_b = _ends(events_a), _ends(events_b)
    assert len(ends_a) == len(ends_b) == 9
    for a, b in zip(ends_a, ends_b):
        assert b.t_start == pytest.approx(a.t_start + offset, abs=1e-6)
        assert b.t_end == pytest.approx(a.t_end + offset, abs=1e-6)

    starts_a, starts_b = _starts(events_a), _starts(events_b)
    assert len(starts_a) == len(starts_b)
    for a, b in zip(starts_a, starts_b):
        assert b.t_start == pytest.approx(a.t_start + offset, abs=1e-6)


def test_flush_completes_the_file_to_exactly_nine(two_speakers_audio, silero_path):
    audio, _sr = two_speakers_audio
    seg = _make_segmenter(silero_path)
    events = _feed(seg, audio)
    ends_before_flush = _ends(events)

    flush_events = seg.flush()
    ends_from_flush = _ends(flush_events)

    assert len(ends_before_flush) + len(ends_from_flush) == 9
    # Flushing again is a no-op: no trailing audio left to finalize.
    assert seg.flush() == []


def test_reset_clears_state_but_preserves_time_origin(two_speakers_audio, silero_path):
    audio, sr = two_speakers_audio
    seg = _make_segmenter(silero_path)
    seg.set_time_origin(50.0)

    _feed(seg, audio[: sr * 2])  # partial feed to move internal counters/state
    seg.reset()
    assert seg.in_speech is False

    events = _feed(seg, audio)
    events.extend(seg.flush())
    ends = _ends(events)
    assert len(ends) == 9
    # sample 0 after reset should still map to the preserved time origin (50.0).
    assert ends[0].t_start == pytest.approx(TWO_SPEAKERS_TRUTH[0][0] + 50.0, abs=0.15)


def test_process_rejects_wrong_frame_size(silero_path):
    seg = _make_segmenter(silero_path)
    with pytest.raises(AssertionError):
        seg.process(np.zeros(256, dtype=np.float32))


def test_in_speech_reflects_vad_state(two_speakers_audio, silero_path):
    audio, sr = two_speakers_audio
    seg = _make_segmenter(silero_path)
    assert seg.in_speech is False

    # "Hi everyone!" starts around 0.13s; well before that we should still be silent.
    silent_prefix = audio[: int(0.05 * sr) // WINDOW_SIZE * WINDOW_SIZE]
    _feed(seg, silent_prefix)
    assert seg.in_speech is False

    _feed(seg, audio)
    seg.flush()
    # by the end of the (silent) file tail, speech should have ended again.
    assert seg.in_speech is False

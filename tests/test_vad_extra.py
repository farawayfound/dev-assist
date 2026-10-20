"""Additional verifier-written coverage for exocortex.speech.vad: boundary conditions,
resource-aliasing/lifetime, import side effects, and multi-instance isolation that
tests/test_vad.py does not exercise. Offline, CPU only, same fixtures as test_vad.py.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

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

MIN_SILENCE_S = 0.3
PRE_ROLL_S = 0.3

_ROOT = Path(__file__).resolve().parent.parent


def _feed(segmenter: VadSegmenter, audio: np.ndarray) -> list[VadEvent]:
    events: list[VadEvent] = []
    i = 0
    n = len(audio)
    while i + WINDOW_SIZE <= n:
        events.extend(segmenter.process(audio[i : i + WINDOW_SIZE]))
        i += WINDOW_SIZE
    return events


def test_import_has_no_side_effects() -> None:
    """Importing the module must never construct a sherpa_onnx VAD config/detector
    (no model loading at import time, per SPEC's coding rules).

    Run in a fresh subprocess (rather than importlib.reload() on the module this test
    session already imported) so patching sherpa_onnx's constructors cannot leak a
    second, distinct set of event classes into the current process and break isinstance
    checks in every other test that imported the real classes at collection time.
    """
    script = (
        "import sherpa_onnx\n"
        "calls = []\n"
        "sherpa_onnx.VadModelConfig = lambda *a, **k: calls.append('VadModelConfig')\n"
        "sherpa_onnx.VoiceActivityDetector = "
        "lambda *a, **k: calls.append('VoiceActivityDetector')\n"
        "import exocortex.speech.vad\n"
        "assert calls == [], f'import triggered sherpa_onnx VAD construction: {calls}'\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(_ROOT),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"


def test_fresh_segmenter_flush_and_reset_are_safe(silero_path: str) -> None:
    """flush()/reset() on a segmenter that never received a frame must not raise."""
    seg = VadSegmenter(silero_path)
    assert seg.flush() == []
    assert seg.in_speech is False

    seg2 = VadSegmenter(silero_path)
    seg2.reset()  # must not raise even though nothing was ever fed
    assert seg2.in_speech is False
    assert seg2.flush() == []


def test_process_rejects_zero_length_frame(silero_path: str) -> None:
    seg = VadSegmenter(silero_path)
    with pytest.raises(AssertionError):
        seg.process(np.zeros(0, dtype=np.float32))


def test_read_only_frame_is_accepted(two_speakers_audio, silero_path: str) -> None:
    """Real capture callbacks may hand VadSegmenter a read-only view (e.g. a slice of a
    buffer PortAudio owns); process() must not need to mutate its input."""
    audio, _sr = two_speakers_audio
    seg = VadSegmenter(silero_path, min_silence_s=MIN_SILENCE_S, pre_roll_s=PRE_ROLL_S)
    ro_audio = audio[: WINDOW_SIZE * 40].copy()
    ro_audio.setflags(write=False)

    events: list[VadEvent] = []
    i = 0
    while i + WINDOW_SIZE <= len(ro_audio):
        frame = ro_audio[i : i + WINDOW_SIZE]
        assert not frame.flags["WRITEABLE"]
        events.extend(seg.process(frame))
        i += WINDOW_SIZE
    # "Hi everyone!" (0.13-0.98s) is fully within this 1.28s prefix, so we expect at
    # least a start/end pair with no exception raised along the way.
    assert any(isinstance(e, SpeechStart) for e in events)


def test_reused_capture_buffer_does_not_corrupt_ongoing_audio(
    two_speakers_audio, silero_path: str
) -> None:
    """A real-time callback commonly reuses a single fixed-size buffer per frame instead
    of allocating a new array each time. SpeechOngoing.audio must be captured as an
    independent copy at the moment of the call, not a view that later mutation of the
    caller's buffer could silently corrupt."""
    audio, _sr = two_speakers_audio
    seg = VadSegmenter(
        silero_path, min_silence_s=MIN_SILENCE_S, pre_roll_s=PRE_ROLL_S, ongoing_interval_s=1.2
    )
    reusable = np.zeros(WINDOW_SIZE, dtype=np.float32)
    events: list[VadEvent] = []
    i = 0
    n = len(audio)
    while i + WINDOW_SIZE <= n:
        reusable[:] = audio[i : i + WINDOW_SIZE]
        events.extend(seg.process(reusable))
        i += WINDOW_SIZE
    events.extend(seg.flush())

    ongoing = [e for e in events if isinstance(e, SpeechOngoing)]
    assert ongoing, "expected at least one SpeechOngoing snapshot in this file"
    snapshot = ongoing[0].audio.copy()
    assert ongoing[0].audio.flags["OWNDATA"], "SpeechOngoing.audio must own its data"

    # Mutate the shared capture buffer as the audio thread would for its next frame.
    reusable[:] = 999.0
    assert np.array_equal(ongoing[0].audio, snapshot), (
        "SpeechOngoing.audio changed after the caller's buffer was reused/mutated"
    )


def test_independent_instances_do_not_share_state(two_speakers_audio, silero_path: str) -> None:
    """SpeechPipeline owns one VadSegmenter per source (mic + system); two instances built
    from the same model must segment independently and identically."""
    audio, _sr = two_speakers_audio
    seg_a = VadSegmenter(silero_path, min_silence_s=MIN_SILENCE_S, pre_roll_s=PRE_ROLL_S)
    seg_b = VadSegmenter(silero_path, min_silence_s=MIN_SILENCE_S, pre_roll_s=PRE_ROLL_S)

    events_a = _feed(seg_a, audio)
    events_a.extend(seg_a.flush())
    events_b = _feed(seg_b, audio)
    events_b.extend(seg_b.flush())

    ends_a = [e for e in events_a if isinstance(e, SpeechEnd)]
    ends_b = [e for e in events_b if isinstance(e, SpeechEnd)]
    assert len(ends_a) == len(ends_b) == 9
    for a, b in zip(ends_a, ends_b):
        assert a.t_start == pytest.approx(b.t_start, abs=1e-6)
        assert a.t_end == pytest.approx(b.t_end, abs=1e-6)


@pytest.mark.parametrize("gap_s", [0.30, 0.32, 0.36, 0.45, 0.60])
def test_close_utterance_gap_still_yields_matched_start_ongoing_end(
    two_speakers_audio, silero_path: str, gap_s: float
) -> None:
    """Regression guard for the SpeechEnd-drain / next-SpeechStart interleaving boundary:
    when two utterances are separated by a gap at or only slightly above min_silence_s,
    the finalisation of the first segment and the detection of the next must not clobber
    each other's bookkeeping (e.g. losing the second utterance's SpeechOngoing events)."""
    audio, sr = two_speakers_audio
    clip_a = audio[int(3.17 * sr) : int(6.86 * sr)]  # ~3.69s, long enough for >=1 Ongoing
    clip_b = audio[int(9.03 * sr) : int(13.81 * sr)]  # ~4.78s, long enough for >=2 Ongoing
    gap = np.zeros(int(round(gap_s * sr)), dtype=np.float32)
    sig = np.concatenate(
        [np.zeros(int(0.5 * sr), dtype=np.float32), clip_a, gap, clip_b, np.zeros(sr, dtype=np.float32)]
    )

    seg = VadSegmenter(
        silero_path, min_silence_s=MIN_SILENCE_S, min_speech_s=0.25, pre_roll_s=PRE_ROLL_S,
        ongoing_interval_s=1.2,
    )
    events = _feed(seg, sig)
    events.extend(seg.flush())

    starts = [e for e in events if isinstance(e, SpeechStart)]
    ongoing = [e for e in events if isinstance(e, SpeechOngoing)]
    ends = [e for e in events if isinstance(e, SpeechEnd)]
    assert len(starts) == 2
    assert len(ends) == 2
    # Both utterances comfortably exceed ongoing_interval_s; each run must keep emitting
    # SpeechOngoing, not just the first one (that's the failure mode this test guards).
    assert len(ongoing) >= 4


def test_full_file_processing_is_fast(two_speakers_audio, silero_path: str) -> None:
    """Regression guard against an accidental quadratic hot path (e.g. re-scanning the
    whole ring buffer per sample): SPEC's verified facts measure ~0.08s for this file."""
    audio, _sr = two_speakers_audio
    seg = VadSegmenter(silero_path, min_silence_s=MIN_SILENCE_S, pre_roll_s=PRE_ROLL_S)
    start = time.perf_counter()
    events = _feed(seg, audio)
    events.extend(seg.flush())
    elapsed = time.perf_counter() - start
    assert elapsed < 3.0, f"processing the 27s test file took {elapsed:.2f}s (expected << 1s)"

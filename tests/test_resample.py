"""Tests for exocortex.audio.resample: Resampler, downmix, FrameChunker.

All offline / CPU-only, no live marker needed.
"""
from __future__ import annotations

import time

import numpy as np
import pytest

from exocortex.audio.resample import FrameChunker, Resampler, downmix

RATE_PAIRS = [(48000, 16000), (44100, 16000), (32000, 16000), (96000, 16000)]


def _sine(rate: int, seconds: float, freq: float = 440.0) -> np.ndarray:
    t = np.arange(int(rate * seconds), dtype=np.float64) / rate
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


# ---------------------------------------------------------------------------
# Resampler: identity
# ---------------------------------------------------------------------------


def test_identity_passthrough_values() -> None:
    r = Resampler(16000, 16000)
    x = _sine(16000, 0.5)
    out = r.process(x)
    assert out.dtype == np.float32
    np.testing.assert_array_equal(out, x)


def test_identity_zero_copy() -> None:
    r = Resampler(16000, 16000)
    x = np.zeros(1000, dtype=np.float32)
    out = r.process(x)
    assert out is x  # no allocation/copy for the identity path


def test_identity_reset_is_noop_safe() -> None:
    r = Resampler(16000, 16000)
    r.reset()
    out = r.process(_sine(16000, 0.1))
    assert out.size == 1600


# ---------------------------------------------------------------------------
# Resampler: output length tracks in_len * out_rate / in_rate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("in_rate,out_rate", RATE_PAIRS)
def test_output_length_matches_ratio_single_call(in_rate: int, out_rate: int) -> None:
    r = Resampler(in_rate, out_rate)
    x = _sine(in_rate, 2.0)
    out = r.process(x)
    expected = round(x.size * out_rate / in_rate)
    assert abs(out.size - expected) <= 2


@pytest.mark.parametrize("in_rate,out_rate", RATE_PAIRS)
def test_output_length_drift_over_long_stream_chunked(in_rate: int, out_rate: int) -> None:
    duration_s = 5.0
    x = _sine(in_rate, duration_s)
    r = Resampler(in_rate, out_rate)

    rng = np.random.default_rng(0)
    total_out = 0
    i = 0
    while i < x.size:
        step = int(rng.integers(64, 900))
        chunk = x[i : i + step]
        total_out += r.process(chunk).size
        i += step

    expected = round(x.size * out_rate / in_rate)
    drift_budget = 2 * duration_s  # SPEC: drift < 2 samples per second
    assert abs(total_out - expected) < drift_budget


# ---------------------------------------------------------------------------
# Resampler: seamless across chunk boundaries
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("in_rate,out_rate", RATE_PAIRS)
def test_chunked_matches_single_call_no_click(in_rate: int, out_rate: int) -> None:
    x = _sine(in_rate, 3.0)

    single = Resampler(in_rate, out_rate).process(x)

    chunked_r = Resampler(in_rate, out_rate)
    chunk_len = max(1, round(in_rate * 0.020))  # 20 ms chunks
    pieces = [
        chunked_r.process(x[i : i + chunk_len]) for i in range(0, x.size, chunk_len)
    ]
    chunked = np.concatenate(pieces)

    n = min(single.size, chunked.size)
    assert n > 200
    a, b = single[100:n], chunked[100:n]
    max_diff = np.max(np.abs(a - b))
    assert max_diff < 1e-3, f"in_rate={in_rate}: max abs diff {max_diff} at boundary-joined output"

    # no click: sample-to-sample jumps in the chunked output should not be
    # meaningfully larger than in the reference single-call output.
    single_step = np.max(np.abs(np.diff(a)))
    chunked_step = np.max(np.abs(np.diff(b)))
    assert chunked_step <= single_step + 1e-2


def test_reset_restarts_stream_cleanly() -> None:
    in_rate, out_rate = 48000, 16000
    x = _sine(in_rate, 1.0)
    r = Resampler(in_rate, out_rate)
    first = r.process(x)
    r.reset()
    second = r.process(x)
    np.testing.assert_allclose(first, second, atol=1e-6)


# ---------------------------------------------------------------------------
# Resampler: performance
# ---------------------------------------------------------------------------


def test_resample_performance_under_5ms() -> None:
    in_rate, out_rate = 48000, 16000
    stereo = (_sine(in_rate, 1.0) * 0.5)
    interleaved = np.empty(stereo.size * 2, dtype=np.float32)
    interleaved[0::2] = stereo
    interleaved[1::2] = -stereo
    r = Resampler(in_rate, out_rate)

    best = float("inf")
    for _ in range(10):
        t0 = time.perf_counter()
        mono = downmix(interleaved, 2)
        r.process(mono)
        best = min(best, time.perf_counter() - t0)

    assert best < 0.005, f"resample of 1s @48kHz stereo took {best * 1000:.3f} ms (budget 5 ms)"


# ---------------------------------------------------------------------------
# downmix
# ---------------------------------------------------------------------------


def test_downmix_stereo_float32_cancels() -> None:
    left = np.full(100, 1.0, dtype=np.float32)
    right = np.full(100, -1.0, dtype=np.float32)
    interleaved = np.empty(200, dtype=np.float32)
    interleaved[0::2] = left
    interleaved[1::2] = right
    mono = downmix(interleaved, 2)
    assert mono.dtype == np.float32
    assert mono.shape == (100,)
    np.testing.assert_allclose(mono, np.zeros(100, dtype=np.float32))


def test_downmix_int16_scales_to_float() -> None:
    interleaved = np.array([32767, 32767, -32768, -32768], dtype=np.int16)
    mono = downmix(interleaved, 2)
    assert mono.dtype == np.float32
    np.testing.assert_allclose(mono, [32767 / 32768.0, -1.0], atol=1e-6)


def test_downmix_mono_passthrough() -> None:
    x = np.array([0.1, -0.2, 0.3], dtype=np.float32)
    mono = downmix(x, 1)
    np.testing.assert_array_equal(mono, x)


def test_downmix_three_channels_mean() -> None:
    interleaved = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
    mono = downmix(interleaved, 3)
    np.testing.assert_allclose(mono, [2.0, 5.0])


# ---------------------------------------------------------------------------
# FrameChunker
# ---------------------------------------------------------------------------


def test_frame_chunker_exact_multiple() -> None:
    fc = FrameChunker(512)
    frames = fc.push(np.arange(1024, dtype=np.float32))
    assert len(frames) == 2
    assert all(f.shape == (512,) for f in frames)
    np.testing.assert_array_equal(frames[0], np.arange(512, dtype=np.float32))
    np.testing.assert_array_equal(frames[1], np.arange(512, 1024, dtype=np.float32))
    assert fc.flush() is None


def test_frame_chunker_accumulates_partial_pushes() -> None:
    fc = FrameChunker(512)
    assert fc.push(np.zeros(300, dtype=np.float32)) == []
    frames = fc.push(np.zeros(300, dtype=np.float32))
    assert len(frames) == 1
    assert frames[0].shape == (512,)
    remainder = fc.flush()
    assert remainder is not None
    assert remainder.shape == (88,)  # 300 + 300 - 512


def test_frame_chunker_flush_empty_is_none() -> None:
    fc = FrameChunker(256)
    assert fc.flush() is None


def test_frame_chunker_flush_clears_buffer() -> None:
    fc = FrameChunker(256)
    fc.push(np.zeros(100, dtype=np.float32))
    first = fc.flush()
    assert first is not None and first.shape == (100,)
    assert fc.flush() is None


def test_frame_chunker_rejects_nonpositive_frame_size() -> None:
    with pytest.raises(ValueError):
        FrameChunker(0)

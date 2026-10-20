"""Additional edge-case coverage for exocortex.audio.resample, written by an independent
verifier pass over the implementation in resample.py.

Focus: boundary conditions, dtype/shape robustness, input validation, and the
single-sample-chunk continuity claim from the implementer's notes (verified here
as a committed test rather than left as an ad hoc claim).
"""
from __future__ import annotations

import numpy as np
import pytest

from exocortex.audio.resample import FrameChunker, Resampler, downmix

NON_IDENTITY_RATE_PAIRS = [(48000, 16000), (44100, 16000), (32000, 16000), (96000, 16000)]


def _sine(rate: int, seconds: float, freq: float = 440.0) -> np.ndarray:
    t = np.arange(int(rate * seconds), dtype=np.float64) / rate
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


# ---------------------------------------------------------------------------
# Resampler: construction validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("in_rate,out_rate", [(0, 16000), (16000, 0), (-48000, 16000), (48000, -16000)])
def test_resampler_rejects_nonpositive_rates(in_rate: int, out_rate: int) -> None:
    with pytest.raises(ValueError):
        Resampler(in_rate, out_rate)


# ---------------------------------------------------------------------------
# Resampler: single-sample chunks stay bit-identical to a single-call resample
# (the strongest test of the seamless-chunking contract; exercises the decimate
# offset math and the interpolation carry every possible way).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("in_rate,out_rate", NON_IDENTITY_RATE_PAIRS)
def test_single_sample_chunks_bit_identical_to_single_call(in_rate: int, out_rate: int) -> None:
    x = _sine(in_rate, 0.05)  # short: per-sample chunking is O(n) calls
    single = Resampler(in_rate, out_rate).process(x)

    r = Resampler(in_rate, out_rate)
    pieces = [r.process(x[i : i + 1]) for i in range(x.size)]
    chunked = np.concatenate(pieces) if pieces else np.empty(0, dtype=np.float32)

    n = min(single.size, chunked.size)
    assert n > 0
    assert np.max(np.abs(single[:n] - chunked[:n])) == 0.0


# ---------------------------------------------------------------------------
# Resampler: dtype / shape robustness of process()
# ---------------------------------------------------------------------------


def test_process_accepts_float64_and_returns_float32() -> None:
    r = Resampler(48000, 16000)
    x = _sine(48000, 0.1).astype(np.float64)
    out = r.process(x)
    assert out.dtype == np.float32
    assert out.size > 0


def test_process_flattens_column_vector_input() -> None:
    r = Resampler(48000, 16000)
    x = _sine(48000, 0.1)
    out_flat = Resampler(48000, 16000).process(x)
    out_2d = r.process(x.reshape(-1, 1))
    np.testing.assert_array_equal(out_2d, out_flat)


def test_identity_converts_non_float32_dtype() -> None:
    r = Resampler(16000, 16000)
    x = np.linspace(-1.0, 1.0, 100, dtype=np.float64)
    out = r.process(x)
    assert out.dtype == np.float32


# ---------------------------------------------------------------------------
# Resampler: empty input mid-stream must not raise or corrupt state
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("in_rate,out_rate", [(16000, 16000)] + NON_IDENTITY_RATE_PAIRS)
def test_empty_chunk_mid_stream_is_harmless(in_rate: int, out_rate: int) -> None:
    r = Resampler(in_rate, out_rate)
    x = _sine(in_rate, 0.1)
    first = r.process(x)
    empty = r.process(np.empty(0, dtype=np.float32))
    second = r.process(x)
    assert empty.size == 0

    # resuming after the empty call must behave exactly as if it had not happened
    r_ref = Resampler(in_rate, out_rate)
    ref_first = r_ref.process(x)
    ref_second = r_ref.process(x)
    np.testing.assert_array_equal(first, ref_first)
    np.testing.assert_array_equal(second, ref_second)


def test_process_empty_array_on_fresh_instance() -> None:
    r = Resampler(44100, 16000)
    out = r.process(np.empty(0, dtype=np.float32))
    assert out.dtype == np.float32
    assert out.size == 0


# ---------------------------------------------------------------------------
# downmix: shape/edge robustness
# ---------------------------------------------------------------------------


def test_downmix_drops_trailing_incomplete_frame() -> None:
    # 5 samples, 2 channels -> 2 full frames + 1 leftover sample, silently dropped
    interleaved = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    mono = downmix(interleaved, 2)
    np.testing.assert_allclose(mono, [1.5, 3.5])


def test_downmix_empty_input() -> None:
    mono = downmix(np.empty(0, dtype=np.float32), 2)
    assert mono.dtype == np.float32
    assert mono.size == 0


def test_downmix_fewer_samples_than_channels_returns_empty() -> None:
    interleaved = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    mono = downmix(interleaved, 4)
    assert mono.size == 0


# ---------------------------------------------------------------------------
# FrameChunker: edge cases
# ---------------------------------------------------------------------------


def test_frame_chunker_push_empty_array_is_noop() -> None:
    fc = FrameChunker(512)
    assert fc.push(np.empty(0, dtype=np.float32)) == []
    assert fc.flush() is None


def test_frame_chunker_rejects_negative_frame_size() -> None:
    with pytest.raises(ValueError):
        FrameChunker(-8)


def test_frame_chunker_many_small_pushes_preserve_order() -> None:
    fc = FrameChunker(512)
    x = np.arange(512 * 3, dtype=np.float32)
    frames: list[np.ndarray] = []
    for i in range(0, x.size, 7):  # odd, deliberately non-aligned push size
        frames.extend(fc.push(x[i : i + 7]))
    tail = fc.flush()
    assert tail is None
    rebuilt = np.concatenate(frames)
    np.testing.assert_array_equal(rebuilt, x)

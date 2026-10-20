"""Stateful audio resampling, channel downmixing and fixed-size frame chunking.

Numpy only (no scipy): the low-pass filter used by ``Resampler`` is a windowed-sinc
FIR designed by hand (Hann window). All arrays are float32 mono unless noted.

``Resampler`` carries two small pieces of state across calls so that consecutive
chunks of a live audio stream resample as if they had been processed in one shot:
  - a raw-input history buffer (the last ``numtaps - 1`` samples) so the FIR
    convolution is continuous across the chunk boundary, and
  - an exact sample-count based phase (integer counters, not an accumulated
    float step) so the decimation/interpolation position never drifts over a
    long session.
"""
from __future__ import annotations

import logging
import math

import numpy as np

log = logging.getLogger(__name__)

_NUMTAPS = 63  # odd -> integer group delay; SPEC only requires "windowed-sinc low-pass FIR",
# this tap count is our design choice (good stopband attenuation, ~0.66 ms group delay at 48 kHz)
_CUTOFF_FRACTION = 0.45  # low-pass cutoff = 0.45 * out_rate


def _design_lowpass_fir(cutoff_hz: float, sample_rate: int, numtaps: int = _NUMTAPS) -> np.ndarray:
    """Windowed-sinc low-pass FIR, unity DC gain, Hann window.

    ``cutoff_hz`` and ``sample_rate`` are both in Hz; the design is done at
    ``sample_rate`` (the *input* rate of the signal the filter will be applied to).
    """
    if numtaps % 2 == 0:
        numtaps += 1
    fc = cutoff_hz / sample_rate  # normalised cutoff, fraction of sample_rate (0..0.5)
    m = numtaps - 1
    n = np.arange(numtaps, dtype=np.float64) - m / 2.0
    h = 2.0 * fc * np.sinc(2.0 * fc * n)
    h *= np.hanning(numtaps)
    h /= h.sum()
    return h.astype(np.float32)


class Resampler:
    """Stateful mono resampler. Keeps fractional phase and filter history across calls
    so consecutive chunks join seamlessly. Numpy only.

    - For integer ratios (e.g. 48000->16000 = 3:1) a windowed-sinc low-pass FIR is
      applied at the input rate and every Mth filtered sample is kept (decimation).
    - For non-integer ratios (e.g. 44100->16000) the same FIR is applied at the input
      rate and the result is linearly interpolated at a carried fractional position.
    - 16000->16000 (or any in_rate == out_rate) is the identity: samples pass through
      unfiltered and unchanged (zero-copy).

    The resample position is tracked with exact integer sample counters (not an
    accumulated float step), so output length tracks ``in_len * out_rate / in_rate``
    with no meaningful drift even over long sessions.
    """

    def __init__(self, in_rate: int, out_rate: int) -> None:
        if in_rate <= 0 or out_rate <= 0:
            raise ValueError(f"rates must be positive: in_rate={in_rate}, out_rate={out_rate}")
        self.in_rate = int(in_rate)
        self.out_rate = int(out_rate)
        self._identity = self.in_rate == self.out_rate
        self._integer_ratio = (not self._identity) and (self.in_rate % self.out_rate == 0)
        self._ratio = self.in_rate / self.out_rate

        if not self._identity:
            cutoff_hz = _CUTOFF_FRACTION * min(self.in_rate, self.out_rate)
            self._taps = _design_lowpass_fir(cutoff_hz, self.in_rate)
            self._numtaps = self._taps.size
            self._decim = self.in_rate // self.out_rate if self._integer_ratio else 0
        else:
            self._taps = np.empty(0, dtype=np.float32)
            self._numtaps = 0
            self._decim = 1

        log.debug(
            "Resampler in_rate=%d out_rate=%d identity=%s integer_ratio=%s",
            self.in_rate, self.out_rate, self._identity, self._integer_ratio,
        )
        self.reset()

    def reset(self) -> None:
        """Clear filter history and phase; the next call starts a fresh stream."""
        self._history = np.zeros(max(self._numtaps - 1, 0), dtype=np.float32)
        self._in_count = 0   # total raw input samples consumed so far (this stream)
        self._out_count = 0  # total output samples produced so far (this stream)
        self._last_filtered: float | None = None  # for interpolation continuity

    def process(self, mono: np.ndarray) -> np.ndarray:
        """Resample one chunk of mono float32 audio. Safe to call repeatedly with
        arbitrarily sized chunks; state carries across calls seamlessly."""
        mono = np.asarray(mono, dtype=np.float32)
        if mono.ndim != 1:
            mono = mono.reshape(-1)
        if self._identity:
            return mono
        n = mono.size
        if n == 0:
            return np.empty(0, dtype=np.float32)

        full = np.concatenate([self._history, mono])
        filtered = np.convolve(full, self._taps, mode="valid").astype(np.float32)
        self._history = full[-(self._numtaps - 1):] if self._numtaps > 1 else self._history

        if self._integer_ratio:
            out = self._decimate(filtered, n)
        else:
            out = self._interpolate(filtered, n)

        self._in_count += n
        return out

    def _decimate(self, filtered: np.ndarray, n: int) -> np.ndarray:
        m = self._decim
        next_global = self._out_count * m
        offset = next_global - self._in_count
        if offset < 0 or offset >= n:
            return np.empty(0, dtype=np.float32)
        idx = np.arange(offset, n, m)
        out = filtered[idx]
        self._out_count += idx.size
        return out

    def _interpolate(self, filtered: np.ndarray, n: int) -> np.ndarray:
        ratio = self._ratio
        in_count_before = self._in_count
        # max j (output index) with local position strictly < n - 1
        max_j_exclusive = (n - 1 + in_count_before) / ratio
        n_out = math.ceil(max_j_exclusive - 1e-9) - self._out_count
        n_out = max(0, n_out)
        if n_out == 0:
            self._last_filtered = filtered[-1]
            return np.empty(0, dtype=np.float32)

        first = filtered[0] if self._last_filtered is None else self._last_filtered
        ext = np.concatenate([np.array([first], dtype=np.float32), filtered])

        j = self._out_count + np.arange(n_out, dtype=np.int64)
        local_pos = j * ratio - in_count_before
        ext_pos = local_pos + 1.0
        floor_idx = np.floor(ext_pos).astype(np.int64)
        floor_idx = np.clip(floor_idx, 0, n)  # safety against float edge cases
        frac = (ext_pos - floor_idx).astype(np.float32)
        out = ext[floor_idx] * (1.0 - frac) + ext[np.clip(floor_idx + 1, 0, n)] * frac

        self._out_count += n_out
        self._last_filtered = filtered[-1]
        return out.astype(np.float32)


def downmix(interleaved: np.ndarray, channels: int) -> np.ndarray:
    """Downmix an interleaved multi-channel array to mono float32 (mean over channels).

    Accepts float32 (assumed already in -1..1) or int16 (scaled by 1/32768) input.
    ``channels == 1`` returns the input as mono float32 unchanged.
    """
    arr = np.asarray(interleaved)
    if arr.dtype == np.int16:
        mono_scale = arr.astype(np.float32) / 32768.0
    else:
        mono_scale = arr.astype(np.float32, copy=False)
    flat = mono_scale.reshape(-1)
    if channels <= 1:
        return flat.astype(np.float32, copy=False)
    frames = flat.size // channels
    usable = flat[: frames * channels].reshape(frames, channels)
    return usable.mean(axis=1).astype(np.float32)


class FrameChunker:
    """Re-buffers arbitrary-length arrays into exact fixed-size frames (e.g. 512 samples).

    ``push`` returns as many exact-size frames as are available (possibly zero, one,
    or more); any leftover samples are kept internally until the next ``push`` or
    returned by ``flush``.
    """

    def __init__(self, frame_size: int) -> None:
        if frame_size <= 0:
            raise ValueError(f"frame_size must be positive, got {frame_size}")
        self._frame_size = frame_size
        self._buf = np.empty(0, dtype=np.float32)

    def push(self, x: np.ndarray) -> list[np.ndarray]:
        """Append ``x`` and return every complete ``frame_size`` frame now available."""
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        self._buf = np.concatenate([self._buf, x])
        n_frames = self._buf.size // self._frame_size
        if n_frames == 0:
            return []
        usable = n_frames * self._frame_size
        block = self._buf[:usable].reshape(n_frames, self._frame_size)
        frames = [row.copy() for row in block]
        self._buf = self._buf[usable:].copy()
        return frames

    def flush(self) -> np.ndarray | None:
        """Return whatever partial frame remains buffered (may be shorter than
        ``frame_size``), or ``None`` if nothing is buffered. Clears the buffer."""
        if self._buf.size == 0:
            return None
        out = self._buf
        self._buf = np.empty(0, dtype=np.float32)
        return out

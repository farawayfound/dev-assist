"""Device checks for Settings > Audio: is each feed live, and is the routing clean.

Speaker attribution is by source: the mic feed is "Me" and is never clustered, the output
(loopback) feed is everyone else and is clustered into Speaker 1, 2, ... (``speech/pipeline.py``).
That is exact only when two things hold, and this module checks both:

1. **The mic delivers sound.** The Audio page's Test (exo-cortex's
   :func:`exocortex.audio.capture.probe_device`) says whether audio arrived; a mic delivering
   digital silence opened fine but Windows feeds it nothing (endpoint muted, privacy, a hardware
   mic-mute key) -- :mod:`exocortex.audio.endpoint` reads and clears that mute.
2. **What the speakers play reaches the output feed, not the mic.** :func:`routing_check` plays
   a short chirp on the render device behind the chosen loopback while recording that loopback
   and the mic, then compares the chirp band's energy against the moment before it. A strong
   rise on the loopback and none on the mic: the mic hears only you. A rise on the mic is speaker
   bleed -- remote voices would be transcribed as "Me" (echo suppression then only catches the
   exact duplicates).

Qt-free and import-light like ``capture.py``: ``pyaudiowpatch`` is reached only through
``capture``'s lazy helpers, and ``pa_module`` / ``sleep`` are injectable for tests.
"""
from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from exocortex.audio.capture import (
    pa_guard,
    DeviceInfo,
    _resolve_pa_module,
    _safe_terminate,
    list_devices,
    resolve_device,
)
from exocortex.audio.resample import downmix

log = logging.getLogger(__name__)

#: Peak below this (~ -100 dBFS) is digital silence: no real microphone is that quiet.
SILENCE_PEAK = 1e-5

_LOOPBACK_SUFFIX = " [Loopback]"

# routing check timing (seconds) and the chirp: three 300 -> 3000 Hz sweeps. A sweep, not a steady
# tone, because laptop mic arrays run noise suppression that removes stationary tones.
_BASELINE_S = 0.6
_TAIL_S = 0.5
_SWEEP_S = 0.25
_GAP_S = 0.1
_SWEEPS = 3
_CHIRP_AMPLITUDE = 0.3
_BAND_HZ = (300.0, 3400.0)
_FLOOR = 1e-9

#: Output feed: the chirp must rise this far above the moment before, and be at least this loud.
OUTPUT_RISE_DB = 10.0
OUTPUT_MIN_DBFS = -50.0
#: Mic: a rise below this is no bleed; at or above STRONG it is plainly audible to the mic.
BLEED_SLIGHT_DB = 3.0
BLEED_STRONG_DB = 10.0


def dbfs(value: float) -> float:
    """A linear 0..1 level in dBFS (floored, so silence is a finite, very negative number)."""
    return 20.0 * math.log10(max(float(value), _FLOOR))


# --------------------------------------------------------------------------- #
# routing check
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RoutingResult:
    """What :func:`routing_check` measured. Levels are in dBFS in the 300-3400 Hz band."""

    output_device: str
    system_device: str
    mic_device: str
    system_before_db: float
    system_during_db: float
    mic_before_db: float
    mic_during_db: float
    mic_silent: bool

    @property
    def system_rise_db(self) -> float:
        return self.system_during_db - self.system_before_db

    @property
    def mic_rise_db(self) -> float:
        return self.mic_during_db - self.mic_before_db

    @property
    def output_ok(self) -> bool:
        """The chirp reached the output feed: speakers are captured where they should be."""
        return self.system_rise_db >= OUTPUT_RISE_DB and self.system_during_db >= OUTPUT_MIN_DBFS

    @property
    def bleed(self) -> str:
        """``"none"`` | ``"slight"`` | ``"strong"`` speaker sound in the mic, or ``"unknown"``
        (no mic, or a mic that delivered digital silence)."""
        if not self.mic_device or self.mic_silent:
            return "unknown"
        rise = self.mic_rise_db
        if rise >= BLEED_STRONG_DB:
            return "strong"
        if rise >= BLEED_SLIGHT_DB:
            return "slight"
        return "none"

    @property
    def isolated(self) -> bool:
        """Both guarantees hold: speakers -> output feed only, mic -> you only."""
        return self.output_ok and self.bleed == "none"


def chirp(rate: int, channels: int) -> np.ndarray:
    """The test sound, interleaved float32 for ``channels``: three short log sweeps with gaps and
    10 ms fades (no clicks)."""
    n = max(1, int(round(rate * _SWEEP_S)))
    t = np.arange(n, dtype=np.float64) / rate
    f0, f1 = 300.0, 3000.0
    k = math.log(f1 / f0)
    phase = 2.0 * math.pi * f0 * _SWEEP_S / k * (np.exp(t / _SWEEP_S * k) - 1.0)
    sweep = np.sin(phase)
    fade = min(n // 2, max(1, int(rate * 0.01)))
    ramp = np.linspace(0.0, 1.0, fade)
    sweep[:fade] *= ramp
    sweep[-fade:] *= ramp[::-1]
    gap = np.zeros(int(round(rate * _GAP_S)))
    mono = np.concatenate([np.concatenate([sweep, gap]) for _ in range(_SWEEPS)])
    mono = (_CHIRP_AMPLITUDE * mono).astype(np.float32)
    if channels <= 1:
        return mono
    return np.repeat(mono.reshape(-1, 1), channels, axis=1).reshape(-1).astype(np.float32)


def band_rms(mono: np.ndarray, rate: int, band: tuple[float, float] = _BAND_HZ) -> float:
    """RMS of ``mono`` restricted to ``band`` (an FFT band-pass; Parseval), 0.0 when empty."""
    x = np.asarray(mono, dtype=np.float64).reshape(-1)
    n = x.size
    if n == 0:
        return 0.0
    spectrum = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, d=1.0 / rate)
    keep = (freqs >= band[0]) & (freqs <= band[1])
    power = np.abs(spectrum[keep]) ** 2
    # one-sided spectrum: every bin but DC / Nyquist stands for two
    return float(math.sqrt(2.0 * float(power.sum())) / n)


def render_device_for(loopback: DeviceInfo, devices: list[DeviceInfo]) -> DeviceInfo | None:
    """The WASAPI output device a loopback device records ("X [Loopback]" -> "X")."""
    name = loopback.name[: -len(_LOOPBACK_SUFFIX)] if loopback.name.endswith(_LOOPBACK_SUFFIX) else loopback.name
    outputs = [d for d in devices if not d.is_input and not d.is_loopback]
    exact = next((d for d in outputs if d.name == name), None)
    if exact is not None:
        return exact
    low = name.lower()
    return next((d for d in outputs if low in d.name.lower() or d.name.lower() in low), None)


class _Recorder:
    """Collects one input stream's buffers, tagged with the phase they arrived in."""

    def __init__(self, device: DeviceInfo) -> None:
        self.device = device
        self.chunks: list[tuple[str, bytes]] = []
        self.phase = "before"
        self._lock = threading.Lock()

    def callback(self, mod: Any) -> Callable[[bytes, int, Any, int], tuple[None, int]]:
        def _cb(in_data: bytes, _n: int, _ti: Any, _st: int) -> tuple[None, int]:
            with self._lock:
                self.chunks.append((self.phase, in_data))
            return (None, mod.paContinue)

        return _cb

    def set_phase(self, phase: str) -> None:
        with self._lock:
            self.phase = phase

    def levels(self) -> tuple[float, float, float]:
        """``(before_band_rms, during_band_rms, overall_peak)`` of the downmixed signal."""
        with self._lock:
            chunks = list(self.chunks)
        out: dict[str, float] = {}
        peak = 0.0
        for phase in ("before", "during"):
            raw = b"".join(data for p, data in chunks if p == phase)
            mono = downmix(np.frombuffer(raw, dtype=np.float32), max(1, self.device.channels)) if raw else np.zeros(0)
            out[phase] = band_rms(mono, self.device.sample_rate)
            if mono.size:
                peak = max(peak, float(np.max(np.abs(mono))))
        return out["before"], out["during"], peak


def _open_input(pa: Any, mod: Any, rec: _Recorder) -> Any:
    d = rec.device
    return pa.open(
        format=mod.paFloat32, channels=max(1, d.channels), rate=d.sample_rate, input=True,
        input_device_index=d.index, frames_per_buffer=max(1, int(round(d.sample_rate * 0.02))),
        stream_callback=rec.callback(mod),
    )


def _close(stream: Any) -> None:
    for step in ("stop_stream", "close"):
        try:
            getattr(stream, step)()
        except Exception:  # noqa: BLE001
            log.debug("stream.%s() raised", step, exc_info=True)


def routing_check(
    system_spec: str = "default",
    mic_spec: str = "default",
    *,
    pa_module: Any | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> RoutingResult:
    """Play the chirp on the output behind ``system_spec``'s loopback, record that loopback and
    ``mic_spec``'s mic, and measure where the chirp arrived. Blocks ~2 s: call it off the UI
    thread. Raises ``ValueError`` / ``RuntimeError`` when the output feed cannot be resolved or
    opened; a mic that cannot be opened is reported as ``mic_device == ""`` instead."""
    mod = _resolve_pa_module(pa_module)
    devices = list_devices(pa_module=mod)
    loopback = resolve_device("system", system_spec, pa_module=mod)
    try:
        mic: DeviceInfo | None = resolve_device("mic", mic_spec, pa_module=mod)
    except ValueError:
        log.info("routing check: no mic for %r; measuring the output feed only", mic_spec)
        mic = None
    render = render_device_for(loopback, devices)

    # PortAudio calls run in pa_guard() (other captures run meanwhile); the waits in between do not
    with pa_guard():
        pa = mod.PyAudio()
    streams: list[Any] = []
    try:
        with pa_guard():
            if render is not None:
                out_index, out_rate, out_ch, out_name = render.index, render.sample_rate, render.channels, render.name
            else:
                info = pa.get_default_output_device_info()
                out_index = info.get("index")
                out_rate = int(round(float(info.get("defaultSampleRate", 48000) or 48000)))
                out_ch = max(1, int(info.get("maxOutputChannels", 2) or 2))
                out_name = str(info.get("name", ""))
                log.info("routing check: no output device named like %r; using the default output", loopback.name)
            loop_rec = _Recorder(loopback)
            streams.append(_open_input(pa, mod, loop_rec))
            mic_rec: _Recorder | None = None
            if mic is not None:
                try:
                    mic_rec = _Recorder(mic)
                    streams.append(_open_input(pa, mod, mic_rec))
                except Exception:  # noqa: BLE001 - the output feed check still stands on its own
                    log.warning("routing check: could not open mic %r", mic.name, exc_info=True)
                    mic_rec = None
        recorders = [r for r in (loop_rec, mic_rec) if r is not None]

        sleep(_BASELINE_S)
        sound = chirp(out_rate, max(1, out_ch))
        state = {"pos": 0}

        def _play(_in: Any, frame_count: int, _ti: Any, _st: int) -> tuple[bytes, int]:
            n = frame_count * max(1, out_ch)
            pos = state["pos"]
            chunk = sound[pos:pos + n]
            state["pos"] = pos + n
            if chunk.size < n:
                chunk = np.concatenate([chunk, np.zeros(n - chunk.size, dtype=np.float32)])
            return (chunk.tobytes(), mod.paContinue)

        for r in recorders:
            r.set_phase("during")
        with pa_guard():
            streams.append(pa.open(
                format=mod.paFloat32, channels=max(1, out_ch), rate=out_rate, output=True,
                output_device_index=out_index, frames_per_buffer=max(1, int(round(out_rate * 0.02))),
                stream_callback=_play,
            ))
        sleep(sound.size / max(1, out_ch) / out_rate + _TAIL_S)
        for r in recorders:
            r.set_phase("after")
    finally:
        with pa_guard():
            for s in reversed(streams):
                _close(s)
            _safe_terminate(pa)

    sys_before, sys_during, _ = loop_rec.levels()
    if mic_rec is not None:
        mic_before, mic_during, mic_peak = mic_rec.levels()
        mic_name = mic_rec.device.name
    else:
        mic_before = mic_during = mic_peak = 0.0
        mic_name = ""
    result = RoutingResult(
        output_device=out_name,
        system_device=loopback.name,
        mic_device=mic_name,
        system_before_db=dbfs(sys_before),
        system_during_db=dbfs(sys_during),
        mic_before_db=dbfs(mic_before),
        mic_during_db=dbfs(mic_during),
        mic_silent=bool(mic_name) and mic_peak < SILENCE_PEAK,
    )
    log.info(
        "routing check: output feed %+.1f dB (%.1f dBFS), mic %+.1f dB%s -> %s",
        result.system_rise_db, result.system_during_db, result.mic_rise_db,
        " (digital silence)" if result.mic_silent else "", "isolated" if result.isolated else f"bleed={result.bleed}",
    )
    return result

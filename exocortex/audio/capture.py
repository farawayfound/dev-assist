"""WASAPI audio capture: device discovery plus a live 16 kHz mono float32 frame source.

Wraps ``pyaudiowpatch`` (a WASAPI-aware fork of PyAudio) to open either the loopback
analogue of the default output device ("system" audio) or the default microphone
("mic"), and delivers fixed-size resampled mono frames to a caller-supplied callback.

``pyaudiowpatch`` is Windows-only and imported lazily inside functions so this module
imports cleanly (and tests run) on any platform; :func:`AudioCapture.start` raises a
clear ``RuntimeError`` if it is missing.

Threading model (see SPEC.md): the PortAudio callback (a realtime audio thread) does
nothing but copy the raw bytes into a bounded queue. A daemon consumer thread drains
that queue, does the downmix -> resample -> frame-chunk -> ``on_frames`` pipeline, and
computes throttled RMS/peak levels. A second daemon watchdog thread ticks every two
seconds for the cheap checks (a dead stream, a plugged/unplugged device) and reopens as
needed; upstream finding 4 (Windows fix): the expensive full device re-enumeration
(rebuilding a whole WASAPI host) is rate-limited separately -- see
``_DEVICE_POLL_INTERVAL_S`` and ``_check_device_changed`` below.

2026-10-29 (Settings > Audio): :func:`candidates_for` (the devices a spec can name, shared by
:func:`resolve_device` and the device pickers) and :func:`probe_device` (open a spec as a real
:class:`AudioCapture` for a moment and report what it delivered -- the "Test" button, never raises)
were added; nothing about the capture itself changed.
"""
from __future__ import annotations

import ctypes
import logging
import queue
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Literal

import numpy as np

from exocortex.audio.resample import FrameChunker, Resampler, downmix
from exocortex.bus import Source

log = logging.getLogger(__name__)

_RAW_QUEUE_MAXSIZE = 200  # ~4 s of 20 ms buffers; bounded so a stalled consumer can't grow memory
_WATCHDOG_INTERVAL_S = 2.0
_STREAM_DEAD_SILENCE_S = 3.0  # SYSTEM-without-keepalive silence-is-normal log threshold; never triggers a reopen
_EXCLUSIVE_MODE_SILENCE_S = 10.0
_KEEPALIVE_FREQ_HZ = 100.0
_KEEPALIVE_AMPLITUDE = 1e-4  # ~ -80 dBFS

# upstream finding 4 (Windows fix): PortAudio (via pyaudiowpatch) snapshots its device
# list once at Pa_Initialize -- inside `mod.PyAudio()` -- and never refreshes it until
# Pa_Terminate + a fresh Pa_Initialize. That means the audit's suggested fix (reuse the
# capture's already-open `self._pa` inside the watchdog) cannot work: a `self._pa` kept
# open since `_open()` will never see a newly plugged device or a default-device switch
# made in Windows Settings, because its snapshot was taken back when the stream opened.
# A fresh `mod.PyAudio()` (full WASAPI host init/enumerate/terminate, `resolve_device()`)
# really is the only way to see a change -- so the fix is to do that rarely, not never.
# `_check_device_changed()` now only pays for that full re-enumeration once per this many
# seconds per stream (instead of every `_WATCHDOG_INTERVAL_S` tick), while a plug/unplug
# event is still caught within one 2s tick via the cheap `_winmm_device_count()` probe
# below. Only a pure default-device change with no device-count change (e.g. picking a
# different already-connected device as default in Settings) waits up to this cadence.
_DEVICE_POLL_INTERVAL_S = 10.0

#: PortAudio's control calls -- Pa_Initialize (inside ``mod.PyAudio()``), device enumeration,
#: opening / stopping / closing a stream, Pa_Terminate -- are not thread-safe, and two threads in
#: Pa_Initialize at once is an access violation that takes the whole process down. Every one of
#: them in this package runs inside :func:`pa_guard`: each capture's watchdog re-enumerates on its
#: own thread, and Settings > Audio opens devices, lists them and runs the routing check while the
#: pipeline's captures run. Reentrant (``_open`` resolves inside it). Audio callbacks never take it.
PA_LOCK = threading.RLock()

_COINIT_APARTMENTTHREADED = 0x2
_com_threads = threading.local()


def _ensure_com() -> None:
    """Initialise COM on this thread once, for PortAudio's WASAPI host API.

    PortAudio initialises COM only on the thread whose Pa_Initialize takes its reference count
    from 0 to 1. While another PyAudio instance keeps it initialised (the other source's capture),
    opening a stream from any other thread without COM fails with ``[Errno -9999] Unanticipated
    host error`` -- the watchdog's reopen, a live device switch and the Settings check all open from
    their own threads (verified on this machine: fails without, opens with STA or MTA). STA, the
    mode PortAudio's own init and comtypes' import both use: an MTA thread would make a later
    ``import comtypes`` on it fail with RPC_E_CHANGED_MODE. A thread that already has COM is left
    as it is. Never uninitialised: streams opened on the thread may outlive any scope here.
    """
    if sys.platform != "win32" or getattr(_com_threads, "done", False):
        return
    _com_threads.done = True
    try:
        ctypes.windll.ole32.CoInitializeEx(None, _COINIT_APARTMENTTHREADED)  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001 - PortAudio then reports its own error on open
        log.debug("CoInitializeEx failed", exc_info=True)


@contextmanager
def pa_guard() -> Iterator[None]:
    """Hold :data:`PA_LOCK` with COM ready on this thread: wrap every PortAudio control call."""
    with PA_LOCK:
        _ensure_com()
        yield

#: exo-cortex's name for the same lock (2026-10-29): its probe / device-scan code takes it directly.
_PA_LOCK = PA_LOCK


def _winmm_device_count() -> int | None:
    """Cheap Windows-only probe: total waveIn + waveOut device count via winmm.

    upstream finding 4 (Windows fix): stdlib-only (``ctypes.windll.winmm``) stand-in for a
    full WASAPI re-enumeration, used by the watchdog to notice a plug/unplug event -- which
    changes this count -- within one `_WATCHDOG_INTERVAL_S` tick, without paying for
    `resolve_device()`'s full PyAudio() init/enumerate/terminate cycle every tick. It
    cannot see a pure default-device change (no device added or removed, e.g. picking a
    different already-connected device as default in Windows Settings); that still relies
    on the `_DEVICE_POLL_INTERVAL_S` cadence. Returns ``None`` off Windows, or if winmm is
    unavailable or raises for any reason, in which case the caller must fall back to the
    time-based cadence alone -- a ``None`` result must never by itself be treated as "the
    count changed".
    """
    if sys.platform != "win32":
        return None
    try:
        winmm = ctypes.windll.winmm  # type: ignore[attr-defined]
        return int(winmm.waveInGetNumDevs()) + int(winmm.waveOutGetNumDevs())
    except Exception:
        return None


@dataclass(frozen=True)
class DeviceInfo:
    """One WASAPI device as PortAudio reports it."""

    index: int
    name: str
    is_loopback: bool
    is_input: bool
    sample_rate: int
    channels: int
    is_default: bool


# --------------------------------------------------------------------------- #
# pyaudiowpatch access (lazy, guarded)
# --------------------------------------------------------------------------- #


def _import_pyaudiowpatch() -> Any:
    """Import ``pyaudiowpatch``, raising a clear error if it is unavailable."""
    try:
        import pyaudiowpatch  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "pyaudiowpatch is required for audio capture but is not installed or "
            "failed to load (Windows only). Run `pip install pyaudiowpatch`."
        ) from exc
    return pyaudiowpatch


def _resolve_pa_module(pa_module: Any | None) -> Any:
    """Return ``pa_module`` if given (test injection), else the real pyaudiowpatch module."""
    return pa_module if pa_module is not None else _import_pyaudiowpatch()


def _safe_terminate(pa: Any) -> None:
    with _PA_LOCK:
        try:
            pa.terminate()
        except Exception:
            log.debug("PyAudio.terminate() raised", exc_info=True)


def _looks_like_index(spec: str) -> bool:
    s = spec.strip()
    return s.isdigit() or (s.startswith("-") and s[1:].isdigit())


def _describe_devices(devices: list[DeviceInfo]) -> str:
    if not devices:
        return "(none)"
    return ", ".join(f"{d.index}:{d.name!r}" for d in devices)


def _enumerate_wasapi(pa: Any, mod: Any) -> tuple[list[DeviceInfo], int | None, str | None]:
    """Build ``DeviceInfo`` rows for every WASAPI device on an open ``pa`` instance.

    Returns ``(devices, default_loopback_index, default_mic_name)``. The default mic is
    identified by *name* (not index) because ``get_default_input_device_info()`` can
    report a different, non-WASAPI host-API entry for the same physical device -- an MME
    one whose name is cut at 31 characters ("Microphone Array (Realtek(R) Au"). The WASAPI
    host API's own ``defaultInputDevice`` gives the full WASAPI name, so it is asked first;
    and whichever name comes back, ``is_default`` is set on exactly the mic
    :func:`resolve_device` would pick for ``"default"``: the exact name match, else the first
    WASAPI mic whose name contains it (2026-10-29).
    """
    wasapi = pa.get_host_api_info_by_type(mod.paWASAPI)
    wasapi_index = wasapi["index"]

    default_loopback_index: int | None = None
    try:
        default_loopback_index = int(pa.get_default_wasapi_loopback()["index"])
    except Exception:
        log.debug("no default WASAPI loopback device", exc_info=True)

    default_mic_name: str | None = None
    wasapi_default_in = wasapi.get("defaultInputDevice")
    if isinstance(wasapi_default_in, int) and wasapi_default_in >= 0:
        try:
            default_mic_name = str(pa.get_device_info_by_index(wasapi_default_in)["name"])
        except Exception:
            log.debug("WASAPI default input %r not readable", wasapi_default_in, exc_info=True)
    if default_mic_name is None:
        try:
            default_mic_name = str(pa.get_default_input_device_info()["name"])
        except Exception:
            log.debug("no default input device", exc_info=True)

    rows: list[dict[str, Any]] = []
    for i in range(pa.get_device_count()):
        d = pa.get_device_info_by_index(i)
        if d.get("hostApi") != wasapi_index:
            continue
        max_in = int(d.get("maxInputChannels", 0) or 0)
        max_out = int(d.get("maxOutputChannels", 0) or 0)
        rows.append({
            "index": int(d["index"]),
            "name": str(d.get("name", "")),
            "is_loopback": bool(d.get("isLoopbackDevice", False)),
            "is_input": max_in > 0,
            "sample_rate": int(round(float(d.get("defaultSampleRate", 0) or 0))),
            "channels": max_in if max_in > 0 else max_out,
        })

    default_mic_index = _default_mic_index(rows, default_mic_name)
    devices = [
        DeviceInfo(
            is_default=(r["index"] == default_loopback_index) if r["is_loopback"] else (r["index"] == default_mic_index),
            **r,
        )
        for r in rows
    ]
    return devices, default_loopback_index, default_mic_name


def _default_mic_index(rows: list[dict[str, Any]], default_mic_name: str | None) -> int | None:
    """The WASAPI mic ``"default"`` resolves to: exact name match, else the first mic whose name
    contains the reported (possibly truncated) default name; ``None`` when there is none."""
    if default_mic_name is None:
        return None
    mics = [r for r in rows if r["is_input"] and not r["is_loopback"]]
    exact = next((r for r in mics if r["name"] == default_mic_name), None)
    if exact is not None:
        return int(exact["index"])
    needle = default_mic_name.lower()
    partial = next((r for r in mics if needle in r["name"].lower()), None)
    return int(partial["index"]) if partial is not None else None


def list_devices(pa_module: Any | None = None) -> list[DeviceInfo]:
    """List every WASAPI device (input, output and loopback) PortAudio currently knows about.

    ``pa_module`` lets tests inject a fake pyaudiowpatch-shaped module; production callers
    omit it and the real module is imported lazily.
    """
    mod = _resolve_pa_module(pa_module)
    with pa_guard():
        pa = mod.PyAudio()
        try:
            devices, _, _ = _enumerate_wasapi(pa, mod)
            return devices
        finally:
            _safe_terminate(pa)


def candidates_for(kind: Literal["mic", "system"], devices: list[DeviceInfo]) -> list[DeviceInfo]:
    """The devices a ``kind`` spec can resolve to, in enumeration order: loopback devices for
    ``"system"``, non-loopback input devices for ``"mic"`` -- the one filter :func:`resolve_device`
    applies, shared with the Settings > Audio device pickers so they offer exactly what resolves."""
    if kind == "system":
        return [d for d in devices if d.is_loopback]
    return [d for d in devices if d.is_input and not d.is_loopback]


def resolve_device(kind: Literal["mic", "system"], spec: str, pa_module: Any | None = None) -> DeviceInfo:
    """Resolve a device spec to a concrete :class:`DeviceInfo`.

    ``spec`` is one of:
      - ``"default"`` (or empty): the current default loopback ("system") or mic ("mic").
      - an integer / int-like string: that PortAudio device index (must be an input
        device; for ``kind="system"`` it must also be a loopback device).
      - anything else: a case-insensitive substring match on device name among WASAPI
        input devices (loopback devices for "system", non-loopback for "mic").

    Raises ``ValueError`` with the available devices listed when nothing matches.
    """
    mod = _resolve_pa_module(pa_module)
    with pa_guard():
        pa = mod.PyAudio()
        try:
            return _resolve_on(pa, mod, kind, spec)
        finally:
            _safe_terminate(pa)


def _resolve_on(pa: Any, mod: Any, kind: Literal["mic", "system"], spec: str) -> DeviceInfo:
    """:func:`resolve_device` on an already-initialised ``pa`` (the caller holds :data:`_PA_LOCK`).
    ``AudioCapture._open`` resolves on the very instance it then opens the stream on: one host
    init per open instead of two, and no window for another thread's init in between."""
    devices, default_loopback_index, default_mic_name = _enumerate_wasapi(pa, mod)
    candidates = candidates_for(kind, devices)

    spec_clean = (spec or "default").strip()

    if spec_clean.lower() == "default" or not spec_clean:
        if kind == "system":
            match = next((d for d in candidates if d.index == default_loopback_index), None)
        else:
            match = next((d for d in candidates if d.is_default), None)   # see _default_mic_index
        if match is None:
            raise ValueError(f"could not resolve default {kind} device; available: {_describe_devices(candidates)}")
        return match

    if _looks_like_index(spec_clean):
        idx = int(spec_clean)
        match = next((d for d in devices if d.index == idx), None)
        if match is None:
            raise ValueError(f"no WASAPI device with index {idx}; available: {_describe_devices(devices)}")
        if not match.is_input:
            raise ValueError(f"device {idx} ({match.name!r}) is not an input device")
        if kind == "system" and not match.is_loopback:
            raise ValueError(f"device {idx} ({match.name!r}) is not a loopback device")
        if kind == "mic" and match.is_loopback:
            raise ValueError(f"device {idx} ({match.name!r}) is a loopback device, not a mic")
        return match

    needle = spec_clean.lower()
    matches = [d for d in candidates if needle in d.name.lower()]
    if not matches:
        raise ValueError(f"no {kind} device matching {spec_clean!r}; available: {_describe_devices(candidates)}")
    matches.sort(key=lambda d: (not d.is_default, d.index))
    return matches[0]


def default_devices_summary(pa_module: Any | None = None) -> str:
    """Human-readable summary of the default loopback and mic devices, for the doctor script."""
    try:
        loop = resolve_device("system", "default", pa_module=pa_module)
        loop_line = f"system (loopback): {loop.name!r} index={loop.index} {loop.sample_rate} Hz {loop.channels} ch"
    except Exception as exc:
        loop_line = f"system (loopback): unavailable ({exc})"
    try:
        mic = resolve_device("mic", "default", pa_module=pa_module)
        mic_line = f"mic: {mic.name!r} index={mic.index} {mic.sample_rate} Hz {mic.channels} ch"
    except Exception as exc:
        mic_line = f"mic: unavailable ({exc})"
    return f"{loop_line}\n{mic_line}"


# --------------------------------------------------------------------------- #
# probe: open a spec for a moment and report what it delivered (Settings > Audio "Test")
# --------------------------------------------------------------------------- #

_PROBE_SILENT_DBFS = -60.0  # a peak below this is "silent": the -80 dBFS keep-alive tone alone lands here


@dataclass(frozen=True)
class ProbeResult:
    """What :func:`probe_device` found for one spec: whether it opened, as which device, and what
    the stream delivered while it ran. ``frames`` counts the 16 kHz ``frame_samples`` frames the
    production path produced (0 = the stream opened but no callback ever fired -- a WASAPI loopback
    idling while nothing plays, or a microphone Windows is not handing out); ``rms`` / ``peak`` are
    linear 0..1 over the whole run."""

    kind: str
    spec: str
    ok: bool
    device: DeviceInfo | None
    seconds: float
    frames: int
    rms: float
    peak: float
    error: str = ""

    @property
    def peak_dbfs(self) -> float:
        """The peak in dBFS, floored at -120 (silence)."""
        return 20.0 * float(np.log10(max(self.peak, 1e-6)))

    @property
    def signal(self) -> str:
        """``error`` (did not open) | ``none`` (opened, nothing arrived) | ``silent`` (audio arrived,
        all of it below -60 dBFS) | ``ok``."""
        if not self.ok:
            return "error"
        if self.frames == 0:
            return "none"
        if self.peak_dbfs < _PROBE_SILENT_DBFS:
            return "silent"
        return "ok"

    def describe(self) -> str:
        """One plain sentence for a person: the device it opened and what came through."""
        if not self.ok:
            return f"Could not open {self.spec!r}: {self.error}"
        d = self.device
        where = f"{d.name} ({d.sample_rate} Hz, {d.channels} ch)" if d is not None else self.spec
        if self.frames == 0:
            if self.kind == "system":
                return (f"Opened {where}, but no audio arrived in {self.seconds:.1f} s. Nothing is playing "
                        "(WASAPI loopback idles while silent; the keep-alive tone keeps it flowing), or "
                        "another app holds the device in exclusive mode.")
            return (f"Opened {where}, but no audio arrived in {self.seconds:.1f} s. Check Windows Settings > "
                    "Privacy > Microphone, and that no app holds the device in exclusive mode.")
        if self.peak_dbfs < _PROBE_SILENT_DBFS:
            if self.kind == "system":
                return (f"Opened {where}: audio flows but it is silent (peak {self.peak_dbfs:.0f} dBFS). "
                        "Play something through this device and test again.")
            return (f"Opened {where}: audio flows but it is silent (peak {self.peak_dbfs:.0f} dBFS). "
                    "Speak into the microphone, or raise its level in Windows Settings > Sound.")
        return f"OK: {where} delivers audio, peak {self.peak_dbfs:.0f} dBFS over {self.seconds:.1f} s."


def probe_device(
    kind: Literal["mic", "system"],
    spec: str,
    *,
    seconds: float = 1.5,
    keepalive: bool = True,
    on_level: Callable[[float, float], None] | None = None,
    pa_module: Any | None = None,
) -> ProbeResult:
    """Open ``spec`` the way the app does -- a real :class:`AudioCapture`: resolve, open at the
    device's native rate, downmix, resample -- for ``seconds``, stop it, and report.

    Never raises: a spec that does not resolve or a device that will not open comes back as a
    result with ``ok=False`` and the error text. ``keepalive`` is the ``[audio].loopback_keepalive``
    value to test with (only a ``"system"`` probe ever starts the tone). ``on_level(rms, peak)``
    relays the capture's throttled levels for a live meter; it is called from the capture's
    consumer thread, never the caller's. ``pa_module`` is the usual test injection.
    """
    source = Source.SYSTEM if kind == "system" else Source.MIC
    lock = threading.Lock()
    stats = {"frames": 0, "sumsq": 0.0, "samples": 0, "peak": 0.0}

    def _on_frames(frame: np.ndarray) -> None:
        if frame.size == 0:
            return
        sumsq = float(np.sum(np.square(frame, dtype=np.float64)))
        peak = float(np.max(np.abs(frame)))
        with lock:
            stats["frames"] += 1
            stats["sumsq"] += sumsq
            stats["samples"] += int(frame.size)
            stats["peak"] = max(stats["peak"], peak)

    capture = AudioCapture(
        source, spec, on_frames=_on_frames, on_level=on_level, keepalive=keepalive, pa_module=pa_module,
    )
    try:
        device = capture.start()
    except Exception as exc:  # noqa: BLE001 - the whole point: report, never raise
        return ProbeResult(kind, spec, False, None, 0.0, 0, 0.0, 0.0, str(exc))
    t0 = time.monotonic()
    try:
        time.sleep(max(0.0, seconds))
    finally:
        elapsed = time.monotonic() - t0            # the listening window; stop() below takes its own time
        try:
            capture.stop()
        except Exception:  # noqa: BLE001
            log.debug("probe: capture.stop() raised", exc_info=True)
    with lock:
        frames, sumsq, samples, peak = stats["frames"], stats["sumsq"], stats["samples"], stats["peak"]
    rms = float(np.sqrt(sumsq / samples)) if samples else 0.0
    return ProbeResult(kind, spec, True, device, elapsed, frames, rms, peak)


# --------------------------------------------------------------------------- #
# loopback keep-alive (WASAPI idles the render endpoint when nothing plays)
# --------------------------------------------------------------------------- #


def _keepalive_cycle(rate: int, channels: int) -> np.ndarray:
    """One exact period of a quiet 100 Hz sine, interleaved for ``channels`` channels."""
    cycle_len = max(1, round(rate / _KEEPALIVE_FREQ_HZ))
    t = np.arange(cycle_len, dtype=np.float64) / rate
    mono = (_KEEPALIVE_AMPLITUDE * np.sin(2.0 * np.pi * _KEEPALIVE_FREQ_HZ * t)).astype(np.float32)
    if channels <= 1:
        return mono
    return np.tile(mono.reshape(-1, 1), (1, channels)).reshape(-1).astype(np.float32)


def _start_keepalive(pa: Any, mod: Any) -> Any | None:
    """Open a tiny, near-silent WASAPI output stream on the default render device.

    Keeps WASAPI loopback callbacks flowing while nothing else is playing. Never raises:
    any failure is logged and ``None`` is returned so capture continues without it.
    """
    try:
        out_info = pa.get_default_output_device_info()
        rate = int(round(float(out_info.get("defaultSampleRate", 48000) or 48000)))
        channels = max(1, int(out_info.get("maxOutputChannels", 2) or 2))
        cycle = _keepalive_cycle(rate, channels)
        state = {"pos": 0}

        def _cb(_in_data: Any, frame_count: int, _time_info: Any, _status: int) -> tuple[bytes, int]:
            pos = state["pos"]
            n = frame_count * channels
            idx = (np.arange(n, dtype=np.int64) + pos) % cycle.size
            state["pos"] = (pos + n) % cycle.size
            return (cycle[idx].tobytes(), mod.paContinue)

        stream = pa.open(
            format=mod.paFloat32,
            channels=channels,
            rate=rate,
            output=True,
            output_device_index=out_info.get("index"),
            frames_per_buffer=max(1, int(round(rate * 0.02))),
            stream_callback=_cb,
        )
        log.info("keep-alive output stream started on %r (%d Hz, %d ch)", out_info.get("name"), rate, channels)
        return stream
    except Exception:
        log.warning("loopback keep-alive output stream unavailable; continuing without it", exc_info=True)
        return None


# --------------------------------------------------------------------------- #
# AudioCapture
# --------------------------------------------------------------------------- #

_SOURCE_KIND: dict[Source, Literal["mic", "system"]] = {Source.MIC: "mic", Source.SYSTEM: "system"}


class AudioCapture:
    """Opens one pyaudiowpatch input stream (loopback or mic) and delivers 16 kHz mono frames.

    Converts the device's native rate/channels to ``target_rate`` mono float32 and calls
    ``on_frames`` with exactly ``frame_samples`` samples per call, never from the PortAudio
    callback thread. ``on_level(rms, peak)`` fires at most every 100 ms; ``on_status(state,
    device_name, detail)`` mirrors ``bus.AudioStatusEvent`` states ("started", "stopped",
    "error", "device_changed", "muted", "unmuted"). A daemon watchdog thread reopens the
    stream if the resolved default device changes or the stream dies.
    """

    def __init__(
        self,
        source: Source,
        spec: str = "default",
        *,
        target_rate: int = 16000,
        frame_samples: int = 512,
        on_frames: Callable[[np.ndarray], None],
        on_level: Callable[[float, float], None] | None = None,
        on_status: Callable[[str, str, str], None] | None = None,
        keepalive: bool = True,
        pa_module: Any | None = None,
        device_count_fn: Callable[[], int | None] | None = None,
    ) -> None:
        self._source = source
        self._kind: Literal["mic", "system"] = _SOURCE_KIND[source]
        self._spec = spec
        self._target_rate = target_rate
        self._frame_samples = frame_samples
        self._on_frames = on_frames
        self._on_level = on_level
        self._on_status = on_status
        self._keepalive_enabled = keepalive
        self._pa_module = pa_module
        # upstream finding 4 (Windows fix): injectable so tests can drive the cheap
        # device-count probe without touching real winmm/hardware state; None (the
        # default) uses the real `_winmm_device_count`.
        self._device_count_fn: Callable[[], int | None] = (
            device_count_fn if device_count_fn is not None else _winmm_device_count
        )

        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._raw_queue: "queue.Queue[bytes]" = queue.Queue(maxsize=_RAW_QUEUE_MAXSIZE)

        self._pa: Any | None = None
        self._stream: Any | None = None
        self._keepalive_stream: Any | None = None
        self._device: DeviceInfo | None = None
        self._resampler: Resampler | None = None
        self._chunker: FrameChunker = FrameChunker(frame_samples)

        self._consumer_thread: threading.Thread | None = None
        self._watchdog_thread: threading.Thread | None = None

        self._running = False
        self._muted = False
        self._last_callback_ts = time.monotonic()
        self._last_level_ts = 0.0
        self._stream_error_flag = False
        self._silence_info_logged = False
        self._exclusive_info_logged = False
        # upstream finding 4 (Windows fix): last full-enumeration time and last cheap
        # device-count probe reading, both baselined in `_open()` so the very first
        # watchdog tick after opening doesn't immediately re-enumerate.
        self._last_enum_ts = 0.0
        self._last_device_count: int | None = None

    # ---- lifecycle ----------------------------------------------------- #

    def start(self) -> DeviceInfo:
        """Resolve the device, open the stream and start the consumer/watchdog threads."""
        if self._running:
            if self._device is None:
                # A background watchdog reopen is currently retrying (the device was lost
                # and hasn't come back yet); there is nothing to return. This is not an
                # assertion because it depends on external device/timing state, not a
                # programming invariant -- raise a clear, descriptive error instead.
                raise RuntimeError(
                    f"capture-{self._source.value} is running but has no open device right "
                    "now (a background reopen is retrying); call stop() first, or wait for "
                    "the next on_status('device_changed', ...) notification"
                )
            return self._device
        self._stop_event.clear()
        try:
            with self._lock:
                self._open()
        except Exception as exc:
            self._notify_status("error", "", str(exc))
            raise
        self._running = True
        self._consumer_thread = threading.Thread(
            target=self._consume_loop, name=f"capture-{self._source.value}", daemon=True
        )
        self._consumer_thread.start()
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop, name=f"capture-{self._source.value}-watchdog", daemon=True
        )
        self._watchdog_thread.start()
        if self._device is None:
            raise RuntimeError(f"capture-{self._source.value}: _open() did not set a device")
        return self._device

    def stop(self) -> None:
        """Stop and close all streams, join the worker threads and terminate PyAudio."""
        if not self._running:
            return
        self._running = False
        self._stop_event.set()
        with self._lock:
            device_name = self._device.name if self._device else ""
            self._teardown_streams()
        consumer, watchdog = self._consumer_thread, self._watchdog_thread
        self._consumer_thread = None
        self._watchdog_thread = None
        for t in (consumer, watchdog):
            if t is not None and t.is_alive():
                t.join(timeout=2.0)
        self._notify_status("stopped", device_name)

    def set_muted(self, muted: bool) -> None:
        """Keep the stream open but deliver zero frames and suppress levels (or resume)."""
        if muted == self._muted:
            return
        self._muted = muted
        self._notify_status("muted" if muted else "unmuted", self._device.name if self._device else "")

    @property
    def device(self) -> DeviceInfo | None:
        return self._device

    @property
    def running(self) -> bool:
        return self._running

    # ---- opening / reopening (caller holds self._lock) ------------------ #

    def _get_pa_module(self) -> Any:
        return self._pa_module if self._pa_module is not None else _import_pyaudiowpatch()

    def _open(self) -> None:
        with pa_guard():
            self._open_locked()

    def _open_locked(self) -> None:
        mod = self._get_pa_module()
        with _PA_LOCK:   # 2026-10-29: init + resolve + open (+ keep-alive) as one transaction, see _PA_LOCK
            pa = mod.PyAudio()
            try:
                device = _resolve_on(pa, mod, self._kind, self._spec)
                frames_per_buffer = max(1, int(round(device.sample_rate * 0.02)))
                stream = pa.open(
                    format=mod.paFloat32,
                    channels=max(1, device.channels),
                    rate=device.sample_rate,
                    input=True,
                    input_device_index=device.index,
                    frames_per_buffer=frames_per_buffer,
                    stream_callback=self._make_input_callback(mod),
                )
            except Exception:
                _safe_terminate(pa)
                raise
            keepalive_stream = None
            if self._source == Source.SYSTEM and self._keepalive_enabled:
                keepalive_stream = _start_keepalive(pa, mod)

        self._drain_raw_queue()
        self._pa = pa
        self._stream = stream
        self._device = device
        self._resampler = Resampler(device.sample_rate, self._target_rate)
        self._chunker = FrameChunker(self._frame_samples)
        self._last_callback_ts = time.monotonic()
        self._stream_error_flag = False
        self._silence_info_logged = False
        self._exclusive_info_logged = False
        # upstream finding 4 (Windows fix): baseline the poll cadence and probe count
        # here so the first watchdog tick after opening doesn't immediately re-enumerate
        # -- the open above just resolved the device fresh.
        self._last_enum_ts = time.monotonic()
        self._last_device_count = self._safe_device_count()
        self._keepalive_stream = keepalive_stream

        log.info(
            "capture-%s: opened %r (%d Hz, %d ch)",
            self._source.value, device.name, device.sample_rate, device.channels,
        )
        self._notify_status("started", device.name)

    def _make_input_callback(self, mod: Any) -> Callable[[bytes, int, Any, int], tuple[None, int]]:
        """Build the PortAudio callback. Captures ``mod`` directly (not ``self``) so it
        never races the assignment of ``self._pa``/``self._device`` that follows ``pa.open()``."""

        def _cb(in_data: bytes, _frame_count: int, _time_info: Any, _status: int) -> tuple[None, int]:
            self._last_callback_ts = time.monotonic()
            try:
                self._raw_queue.put_nowait(in_data)
            except queue.Full:
                pass  # bounded queue: drop under sustained overload rather than block
            except Exception:
                self._stream_error_flag = True
                log.exception("capture-%s: callback error", self._source.value)
            return (None, mod.paContinue)

        return _cb

    def _drain_raw_queue(self) -> None:
        while True:
            try:
                self._raw_queue.get_nowait()
            except queue.Empty:
                break

    def _teardown_streams(self) -> None:
        stream, ka_stream, pa = self._stream, self._keepalive_stream, self._pa
        self._stream = None
        self._keepalive_stream = None
        self._pa = None
        with pa_guard():
            for s in (stream, ka_stream):
                if s is None:
                    continue
                try:
                    s.stop_stream()
                except Exception:
                    log.debug("stream.stop_stream() raised", exc_info=True)
                try:
                    s.close()
                except Exception:
                    log.debug("stream.close() raised", exc_info=True)
            if pa is not None:
                _safe_terminate(pa)

    # ---- consumer thread: downmix -> resample -> chunk -> on_frames ----- #

    def _consume_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                data = self._raw_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            with self._lock:
                device = self._device
                resampler = self._resampler
                chunker = self._chunker
            if device is None or resampler is None:
                continue
            try:
                self._process_chunk(data, device.channels, resampler, chunker)
            except Exception:
                log.exception("capture-%s: error processing audio chunk", self._source.value)

    def _process_chunk(self, data: bytes, channels: int, resampler: Resampler, chunker: FrameChunker) -> None:
        muted = self._muted
        if muted:
            raw = np.zeros(len(data) // 4, dtype=np.float32)
        else:
            raw = np.frombuffer(data, dtype=np.float32)
        mono = downmix(raw, channels)
        resampled = resampler.process(mono)

        for frame in chunker.push(resampled):
            try:
                self._on_frames(frame)
            except Exception:
                log.exception("capture-%s: on_frames callback raised", self._source.value)

        if muted:
            self._maybe_emit_level(0.0, 0.0)
        elif resampled.size:
            rms = float(np.sqrt(np.mean(np.square(resampled))))
            peak = float(np.max(np.abs(resampled)))
            self._maybe_emit_level(rms, peak)

    def _maybe_emit_level(self, rms: float, peak: float) -> None:
        if self._on_level is None:
            return
        now = time.monotonic()
        if now - self._last_level_ts < 0.1:
            return
        self._last_level_ts = now
        try:
            self._on_level(rms, peak)
        except Exception:
            log.exception("capture-%s: on_level callback raised", self._source.value)

    def _notify_status(self, state: str, device_name: str, detail: str = "") -> None:
        if self._on_status is None:
            return
        try:
            self._on_status(state, device_name, detail)
        except Exception:
            log.exception("capture-%s: on_status callback raised", self._source.value)

    # ---- watchdog thread: device changes + dead-stream recovery --------- #

    def _watchdog_loop(self) -> None:
        while not self._stop_event.wait(_WATCHDOG_INTERVAL_S):
            try:
                self._watchdog_tick()
            except Exception:
                log.exception("capture-%s: watchdog error", self._source.value)

    def _watchdog_tick(self) -> None:
        if not self._running:
            return
        device_changed = self._check_device_changed()
        stream_dead = self._stream_error_flag or not self._safe_is_active()
        if device_changed or stream_dead:
            self._reopen("device_changed" if device_changed else "stream_dead")
        else:
            self._log_silence_if_needed()

    def _safe_is_active(self) -> bool:
        stream = self._stream
        if stream is None:
            return False
        try:
            return bool(stream.is_active())
        except Exception:
            return True  # unknown -> don't force a reopen storm over a missing method

    def _safe_device_count(self) -> int | None:
        # upstream finding 4 (Windows fix): a broken/injected probe must degrade to "no
        # count available" (falls back to the time cadence alone), never raise into the
        # watchdog loop.
        try:
            return self._device_count_fn()
        except Exception:
            return None

    def _check_device_changed(self) -> bool:
        current = self._device
        if current is None:
            return False

        # upstream finding 4 (Windows fix): only pay for a full WASAPI re-enumeration
        # (`resolve_device` -> a fresh PyAudio() host init/enumerate/terminate cycle) when
        # the cheap device-count probe says a device was plugged/unplugged, or the poll
        # cadence has elapsed. See `_DEVICE_POLL_INTERVAL_S`'s docstring for why a fresh
        # PyAudio() instance is unavoidable at all -- this just makes it rare.
        count = self._safe_device_count()
        count_changed = count is not None and count != self._last_device_count
        if count is not None:
            self._last_device_count = count

        now = time.monotonic()
        if not count_changed and (now - self._last_enum_ts) < _DEVICE_POLL_INTERVAL_S:
            return False
        self._last_enum_ts = now

        try:
            fresh = resolve_device(self._kind, self._spec, pa_module=self._get_pa_module())
        except Exception:
            log.debug("capture-%s: watchdog could not resolve device", self._source.value, exc_info=True)
            return False
        return (fresh.index, fresh.name, fresh.sample_rate) != (current.index, current.name, current.sample_rate)

    def _log_silence_if_needed(self) -> None:
        """Silence alone never triggers a reopen; loopback idles by design when nothing
        plays. This only logs an informational note (never on_status: AudioStatusEvent
        has no "info" state) so it's easy to tell "normal" apart from a real problem."""
        silence_s = time.monotonic() - self._last_callback_ts
        if self._source == Source.SYSTEM and not self._keepalive_enabled:
            if silence_s > _STREAM_DEAD_SILENCE_S and not self._silence_info_logged:
                log.info(
                    "capture-system: no callbacks for %.1fs (normal - nothing playing, keep-alive is off)",
                    silence_s,
                )
                self._silence_info_logged = True
            elif silence_s <= _STREAM_DEAD_SILENCE_S:
                self._silence_info_logged = False
        elif self._source == Source.SYSTEM and self._keepalive_enabled:
            if silence_s > _EXCLUSIVE_MODE_SILENCE_S and not self._exclusive_info_logged:
                log.info(
                    "capture-system: no system audio for %.1fs despite keep-alive - "
                    "is another app using exclusive mode?",
                    silence_s,
                )
                self._exclusive_info_logged = True
            elif silence_s <= _EXCLUSIVE_MODE_SILENCE_S:
                self._exclusive_info_logged = False

    def _reopen(self, reason: str) -> None:
        with self._lock:
            old_name = self._device.name if self._device else ""
            self._teardown_streams()
            # Honestly reflect "no stream open right now" while we retry, rather than
            # leaving a stale DeviceInfo (from the device we just tore down) in place;
            # the consumer loop already treats `device is None` as "nothing to process yet".
            self._device = None
            try:
                self._open()
            except Exception as exc:
                log.exception("capture-%s: reopen failed", self._source.value)
                self._notify_status("error", old_name, f"reopen failed ({reason}): {exc}")
                return
            new_name = self._device.name if self._device else ""
        log.warning("capture-%s: reopened (%s) -> %r", self._source.value, reason, new_name)
        self._notify_status("device_changed", new_name, reason)

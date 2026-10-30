"""Tests for exocortex.audio.capture: DeviceInfo, list_devices, resolve_device, AudioCapture.

Offline tests inject a fake pyaudiowpatch-shaped module (no real audio devices needed).
Tests marked ``live`` exercise the real pyaudiowpatch / real devices on this machine and
are skipped by default (pytest.ini: ``-m "not live"``).
"""
from __future__ import annotations

import sys
import threading
import time
from typing import Any

import numpy as np
import pytest

from exocortex.audio import capture as capture_mod
from exocortex.audio.capture import AudioCapture, DeviceInfo, default_devices_summary, list_devices, resolve_device
from exocortex.bus import Source

# --------------------------------------------------------------------------- #
# Fake pyaudiowpatch: world state + PyAudio + Stream
# --------------------------------------------------------------------------- #


def _device_dict(
    index: int, name: str, *, max_in: int = 0, max_out: int = 0, rate: float = 48000.0, loopback: bool = False
) -> dict[str, Any]:
    return {
        "index": index,
        "name": name,
        "hostApi": 0,
        "maxInputChannels": max_in,
        "maxOutputChannels": max_out,
        "defaultSampleRate": rate,
        "isLoopbackDevice": loopback,
    }


def default_world() -> dict[str, Any]:
    return {
        "devices": [
            _device_dict(0, "Loopback A", max_in=2, rate=48000.0, loopback=True),
            _device_dict(1, "Mic A", max_in=1, rate=44100.0, loopback=False),
            _device_dict(2, "Speakers A", max_out=2, rate=48000.0, loopback=False),
        ],
        "default_loopback_index": 0,
        "default_input_name": "Mic A",
    }


class FakeStream:
    """Mimics a pyaudiowpatch Stream: holds the callback, can be told to feed data."""

    def __init__(
        self,
        *,
        format: Any = None,
        channels: int = 1,
        rate: int = 48000,
        input: bool = False,
        output: bool = False,
        input_device_index: int | None = None,
        output_device_index: int | None = None,
        frames_per_buffer: int = 960,
        stream_callback: Any = None,
        **_ignored: Any,
    ) -> None:
        self.channels = channels
        self.rate = rate
        self.is_input = bool(input)
        self.is_output = bool(output)
        self.frames_per_buffer = frames_per_buffer
        self.callback = stream_callback
        self._active = True
        self.closed = False
        self._feed_thread: threading.Thread | None = None
        self._stop_feed = threading.Event()

    def is_active(self) -> bool:
        return self._active

    def stop_stream(self) -> None:
        self._active = False
        self._stop_feed.set()
        if self._feed_thread is not None and self._feed_thread.is_alive():
            self._feed_thread.join(timeout=2.0)

    def close(self) -> None:
        self.closed = True

    def start_input_feed(self, seconds: float, *, freq: float = 440.0, amplitude: float = 0.3, speedup: float = 50.0) -> None:
        """Feed ``seconds`` of synthetic sine audio to the callback from a background thread."""

        def _run() -> None:
            n_total = int(self.rate * seconds)
            pos = 0
            chunk = self.frames_per_buffer
            while pos < n_total and not self._stop_feed.is_set():
                n = min(chunk, n_total - pos)
                t = (np.arange(n, dtype=np.float64) + pos) / self.rate
                mono = (amplitude * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)
                if self.channels > 1:
                    interleaved = np.repeat(mono.reshape(-1, 1), self.channels, axis=1).reshape(-1)
                else:
                    interleaved = mono
                data = interleaved.astype(np.float32).tobytes()
                if self.callback is not None:
                    self.callback(data, n, {}, 0)
                pos += n
                time.sleep((n / self.rate) / speedup)

        self._feed_thread = threading.Thread(target=_run, daemon=True)
        self._feed_thread.start()

    def join_feed(self, timeout: float = 5.0) -> None:
        if self._feed_thread is not None:
            self._feed_thread.join(timeout=timeout)


class FakePyAudio:
    def __init__(self, world: dict[str, Any]) -> None:
        self.world = world
        self.terminated = False
        self.streams: list[FakeStream] = []

    def get_host_api_info_by_type(self, _api: int) -> dict[str, Any]:
        return {"index": 0, "name": "Windows WASAPI"}

    def get_device_count(self) -> int:
        return len(self.world["devices"])

    def get_device_info_by_index(self, i: int) -> dict[str, Any]:
        return dict(self.world["devices"][i])

    def get_default_wasapi_loopback(self) -> dict[str, Any]:
        idx = self.world["default_loopback_index"]
        return dict(self.world["devices"][idx])

    def get_default_input_device_info(self) -> dict[str, Any]:
        name = self.world["default_input_name"]
        for d in self.world["devices"]:
            if d["name"] == name:
                return dict(d)
        raise RuntimeError("no default input configured")

    def get_default_output_device_info(self) -> dict[str, Any]:
        for d in self.world["devices"]:
            if d.get("maxOutputChannels", 0) > 0:
                return dict(d)
        raise RuntimeError("no output device configured")

    def open(self, **kwargs: Any) -> FakeStream:
        stream = FakeStream(**kwargs)
        self.streams.append(stream)
        return stream

    def terminate(self) -> None:
        self.terminated = True


class FakePaModule:
    paWASAPI = 1
    paFloat32 = 1
    paContinue = 0
    paAbort = 2

    def __init__(self, world: dict[str, Any]) -> None:
        self.world = world
        self.instances: list[FakePyAudio] = []

    def PyAudio(self) -> FakePyAudio:
        pa = FakePyAudio(self.world)
        self.instances.append(pa)
        return pa

    def latest_input_stream(self) -> FakeStream:
        for pa in reversed(self.instances):
            for s in reversed(pa.streams):
                if s.is_input:
                    return s
        raise AssertionError("no input stream opened yet")


# --------------------------------------------------------------------------- #
# list_devices / resolve_device
# --------------------------------------------------------------------------- #


def test_list_devices_filters_wasapi_and_marks_defaults() -> None:
    mod = FakePaModule(default_world())
    devices = list_devices(pa_module=mod)
    assert {d.name for d in devices} == {"Loopback A", "Mic A", "Speakers A"}

    loopback = next(d for d in devices if d.name == "Loopback A")
    assert loopback.is_loopback and loopback.is_input and loopback.is_default
    assert loopback.sample_rate == 48000 and loopback.channels == 2

    mic = next(d for d in devices if d.name == "Mic A")
    assert mic.is_input and not mic.is_loopback and mic.is_default
    assert mic.sample_rate == 44100 and mic.channels == 1

    speakers = next(d for d in devices if d.name == "Speakers A")
    assert not speakers.is_input and not speakers.is_default


def test_resolve_device_default_system_and_mic() -> None:
    mod = FakePaModule(default_world())
    system = resolve_device("system", "default", pa_module=mod)
    assert system.name == "Loopback A" and system.is_loopback

    mic = resolve_device("mic", "default", pa_module=mod)
    assert mic.name == "Mic A" and not mic.is_loopback


def test_resolve_device_by_index() -> None:
    mod = FakePaModule(default_world())
    d = resolve_device("mic", "1", pa_module=mod)
    assert d.index == 1 and d.name == "Mic A"


def test_resolve_device_by_name_substring_case_insensitive() -> None:
    mod = FakePaModule(default_world())
    d = resolve_device("system", "loopback", pa_module=mod)
    assert d.name == "Loopback A"


def test_resolve_device_index_wrong_kind_raises() -> None:
    mod = FakePaModule(default_world())
    with pytest.raises(ValueError, match="not a loopback device"):
        resolve_device("system", "1", pa_module=mod)  # index 1 is the mic, not loopback


def test_resolve_device_no_match_raises_with_devices_listed() -> None:
    mod = FakePaModule(default_world())
    with pytest.raises(ValueError) as exc_info:
        resolve_device("mic", "nonexistent device name", pa_module=mod)
    assert "Mic A" in str(exc_info.value)


def test_default_devices_summary_offline() -> None:
    mod = FakePaModule(default_world())
    summary = default_devices_summary(pa_module=mod)
    assert "Loopback A" in summary
    assert "Mic A" in summary


# --------------------------------------------------------------------------- #
# AudioCapture: offline pipeline behaviour
# --------------------------------------------------------------------------- #


def _make_capture(mod: FakePaModule, *, on_frames: Any, on_level: Any = None, on_status: Any = None, keepalive: bool = False) -> AudioCapture:
    return AudioCapture(
        Source.SYSTEM,
        "default",
        on_frames=on_frames,
        on_level=on_level,
        on_status=on_status,
        keepalive=keepalive,
        pa_module=mod,
    )


def test_capture_frames_are_512_float32_at_16k_cadence() -> None:
    mod = FakePaModule(default_world())
    frames: list[np.ndarray] = []
    lock = threading.Lock()

    def on_frames(f: np.ndarray) -> None:
        with lock:
            frames.append(f)

    cap = _make_capture(mod, on_frames=on_frames)
    cap.start()
    try:
        stream = mod.latest_input_stream()
        seconds = 2.0
        stream.start_input_feed(seconds, freq=440.0)
        stream.join_feed(timeout=5.0)
        time.sleep(0.3)  # let the consumer thread drain the queue
    finally:
        cap.stop()

    assert len(frames) > 0
    assert all(f.shape == (512,) for f in frames)
    assert all(f.dtype == np.float32 for f in frames)

    expected = seconds * 16000 / 512
    assert abs(len(frames) - expected) / expected <= 0.05, f"got {len(frames)} frames, expected ~{expected:.1f}"


def test_capture_downmix_and_resample_frequency_peak() -> None:
    mod = FakePaModule(default_world())
    frames: list[np.ndarray] = []

    cap = _make_capture(mod, on_frames=frames.append)
    cap.start()
    try:
        stream = mod.latest_input_stream()
        stream.start_input_feed(1.5, freq=440.0)
        stream.join_feed(timeout=5.0)
        time.sleep(0.3)
    finally:
        cap.stop()

    audio = np.concatenate(frames)
    assert audio.size > 1000

    target_rate = 16000
    spectrum = np.abs(np.fft.rfft(audio * np.hanning(audio.size)))
    freqs = np.fft.rfftfreq(audio.size, d=1.0 / target_rate)
    peak_freq = freqs[np.argmax(spectrum)]
    assert abs(peak_freq - 440.0) < 20.0, f"peak frequency {peak_freq} Hz, expected ~440 Hz"


def test_capture_levels_reported() -> None:
    mod = FakePaModule(default_world())
    levels: list[tuple[float, float]] = []

    cap = _make_capture(mod, on_frames=lambda f: None, on_level=lambda rms, peak: levels.append((rms, peak)))
    cap.start()
    try:
        stream = mod.latest_input_stream()
        stream.start_input_feed(1.0, freq=440.0, amplitude=0.5)
        stream.join_feed(timeout=5.0)
        time.sleep(0.3)
    finally:
        cap.stop()

    assert len(levels) > 0
    assert any(rms > 0.0 for rms, _ in levels)
    assert any(peak > 0.0 for _, peak in levels)


def test_capture_mute_produces_zero_frames_and_zero_level() -> None:
    mod = FakePaModule(default_world())
    frames: list[np.ndarray] = []
    levels: list[tuple[float, float]] = []

    cap = _make_capture(mod, on_frames=frames.append, on_level=lambda rms, peak: levels.append((rms, peak)))
    cap.start()
    try:
        cap.set_muted(True)
        stream = mod.latest_input_stream()
        stream.start_input_feed(1.0, freq=440.0, amplitude=0.5)
        stream.join_feed(timeout=5.0)
        time.sleep(0.3)
    finally:
        cap.stop()

    assert len(frames) > 0
    assert all(np.all(f == 0.0) for f in frames)
    assert len(levels) > 0
    assert all(rms == 0.0 and peak == 0.0 for rms, peak in levels)


def test_capture_unmute_resumes_nonzero_frames() -> None:
    mod = FakePaModule(default_world())
    frames: list[np.ndarray] = []

    cap = _make_capture(mod, on_frames=frames.append)
    cap.set_muted(True)
    cap.start()
    try:
        stream = mod.latest_input_stream()
        stream.start_input_feed(0.5, freq=440.0, amplitude=0.5)
        stream.join_feed(timeout=5.0)
        time.sleep(0.2)
        cap.set_muted(False)
        stream2 = mod.latest_input_stream()
        stream2.start_input_feed(0.5, freq=440.0, amplitude=0.5)
        stream2.join_feed(timeout=5.0)
        time.sleep(0.3)
    finally:
        cap.stop()

    assert any(np.any(f != 0.0) for f in frames), "expected some nonzero frames after unmuting"


def test_capture_status_started_and_stopped() -> None:
    mod = FakePaModule(default_world())
    statuses: list[tuple[str, str, str]] = []

    cap = _make_capture(mod, on_frames=lambda f: None, on_status=lambda s, n, d: statuses.append((s, n, d)))
    cap.start()
    cap.stop()

    states = [s[0] for s in statuses]
    assert states[0] == "started"
    assert "stopped" in states


def test_capture_stop_joins_threads_and_terminates() -> None:
    mod = FakePaModule(default_world())
    cap = _make_capture(mod, on_frames=lambda f: None)
    cap.start()
    assert cap.running

    thread_names_while_running = {t.name for t in threading.enumerate()}
    assert any(n.startswith("capture-system") for n in thread_names_while_running)

    cap.stop()  # called promptly, before the 2s watchdog tick fires

    assert not cap.running
    thread_names_after_stop = {t.name for t in threading.enumerate()}
    assert not any(n.startswith("capture-system") for n in thread_names_after_stop)
    assert all(pa.terminated for pa in mod.instances)


def test_capture_watchdog_reopens_on_device_change(monkeypatch: pytest.MonkeyPatch) -> None:
    # upstream finding 4 (Windows fix): full device re-enumeration is now rate-limited to
    # once per `_DEVICE_POLL_INTERVAL_S` (see `test_watchdog_device_poll_cadence_still_
    # detects_default_device_change` for that behaviour in isolation); this test is about
    # device-change *detection and reopen* in general, so it keeps that cadence effectively
    # per-tick by shortening it to match the (also default) 2s watchdog tick.
    monkeypatch.setattr(capture_mod, "_DEVICE_POLL_INTERVAL_S", 0.0)
    world = default_world()
    mod = FakePaModule(world)
    statuses: list[tuple[str, str, str]] = []

    cap = AudioCapture(
        Source.SYSTEM,
        "default",
        on_frames=lambda f: None,
        on_status=lambda s, n, d: statuses.append((s, n, d)),
        keepalive=False,
        pa_module=mod,
        device_count_fn=lambda: None,  # isolate this test from the real winmm device count
    )
    cap.start()
    try:
        assert cap.device is not None and cap.device.name == "Loopback A"

        # simulate the default output device changing (e.g. a Bluetooth headset connects)
        world["devices"][0] = _device_dict(0, "New Loopback B", max_in=2, rate=48000.0, loopback=True)

        deadline = time.monotonic() + 6.0
        while time.monotonic() < deadline:
            if cap.device is not None and cap.device.name == "New Loopback B":
                break
            time.sleep(0.1)
    finally:
        cap.stop()

    assert cap.device is not None and cap.device.name == "New Loopback B"
    assert any(s[0] == "device_changed" and s[1] == "New Loopback B" for s in statuses)


def test_capture_watchdog_reopen_failure_clears_stale_device() -> None:
    """If the device disappears and a reopen attempt cannot resolve any replacement, the
    capture must not keep reporting the old (no-longer-open) device as current: `.device`
    should become None (matching what the consumer loop already treats as "not ready") and
    `.running` should stay True since the watchdog keeps retrying rather than giving up."""
    world = default_world()
    mod = FakePaModule(world)
    statuses: list[tuple[str, str, str]] = []

    cap = AudioCapture(
        Source.SYSTEM,
        "default",
        on_frames=lambda f: None,
        on_status=lambda s, n, d: statuses.append((s, n, d)),
        keepalive=False,
        pa_module=mod,
    )
    cap.start()
    try:
        assert cap.device is not None and cap.device.name == "Loopback A"

        # simulate the device vanishing entirely (e.g. unplugged) and the stream dying
        stream = mod.latest_input_stream()
        stream._active = False
        world["devices"] = [d for d in world["devices"] if not d.get("isLoopbackDevice")]

        deadline = time.monotonic() + 6.0
        while time.monotonic() < deadline:
            if any(s[0] == "error" for s in statuses):
                break
            time.sleep(0.1)

        assert any(s[0] == "error" and "reopen failed" in s[2] for s in statuses)
        assert cap.device is None, "stale DeviceInfo must not survive a failed reopen"
        assert cap.running, "the watchdog should keep retrying, not give up"

        # start() must not silently misbehave (or hit a bare-assert crash) while retrying
        with pytest.raises(RuntimeError):
            cap.start()
    finally:
        cap.stop()

    assert not cap.running
    thread_names_after_stop = {t.name for t in threading.enumerate()}
    assert not any(n.startswith("capture-system") for n in thread_names_after_stop)


def test_capture_mute_persists_across_watchdog_reopen(monkeypatch: pytest.MonkeyPatch) -> None:
    """A device change (e.g. a Bluetooth profile switch) must not un-mute a muted source."""
    # upstream finding 4 (Windows fix): see test_capture_watchdog_reopens_on_device_change.
    monkeypatch.setattr(capture_mod, "_DEVICE_POLL_INTERVAL_S", 0.0)
    world = default_world()
    mod = FakePaModule(world)
    frames: list[np.ndarray] = []

    cap = AudioCapture(
        Source.SYSTEM,
        "default",
        on_frames=frames.append,
        keepalive=False,
        pa_module=mod,
        device_count_fn=lambda: None,
    )
    cap.start()
    try:
        cap.set_muted(True)

        world["devices"][0] = _device_dict(0, "New Loopback B", max_in=2, rate=48000.0, loopback=True)
        deadline = time.monotonic() + 6.0
        while time.monotonic() < deadline:
            if cap.device is not None and cap.device.name == "New Loopback B":
                break
            time.sleep(0.1)
        assert cap.device is not None and cap.device.name == "New Loopback B"

        stream = mod.latest_input_stream()
        stream.start_input_feed(0.5, freq=440.0, amplitude=0.5)
        stream.join_feed(timeout=5.0)
        time.sleep(0.3)
    finally:
        cap.stop()

    assert len(frames) > 0
    assert all(np.all(f == 0.0) for f in frames), "mute must survive a watchdog-triggered reopen"


def test_capture_watchdog_no_reenumeration_without_change(monkeypatch: pytest.MonkeyPatch) -> None:
    """upstream finding 4 (Windows fix): with the poll cadence left at its (large) default
    and a steady cheap device-count reading, repeated watchdog ticks must not build/tear
    down any additional PyAudio host beyond the one `_open()` itself already created --
    that full host init/enumerate/terminate cycle is exactly the per-tick cost this
    finding is about."""
    monkeypatch.setattr(capture_mod, "_WATCHDOG_INTERVAL_S", 0.02)
    # _DEVICE_POLL_INTERVAL_S is left at its real (10s) default, well above the sleep below.
    world = default_world()
    mod = FakePaModule(world)

    cap = AudioCapture(
        Source.SYSTEM,
        "default",
        on_frames=lambda f: None,
        keepalive=False,
        pa_module=mod,
        device_count_fn=lambda: 4,  # constant: nothing plugged/unplugged
    )
    cap.start()
    try:
        # `_open()` resolves fresh on the one PyAudio host it then opens the stream on and
        # keeps as `self._pa` (2026-10-29: one host init per open, under _PA_LOCK; before, a
        # second, immediately terminated host was built inside `resolve_device`); baseline
        # against that and check that ticks add nothing further.
        baseline = len(mod.instances)
        assert baseline == 1
        time.sleep(0.02 * 15)  # several watchdog ticks
    finally:
        cap.stop()

    assert len(mod.instances) == baseline, "no extra PyAudio host should be built without a device/count change"


def test_capture_watchdog_device_count_change_triggers_reopen(monkeypatch: pytest.MonkeyPatch) -> None:
    """upstream finding 4 (Windows fix): a changed cheap device-count probe reading (e.g. a
    headset plugged in) must trigger a full re-enumeration on the very next tick, even
    while `_DEVICE_POLL_INTERVAL_S` is still at its large default -- this is what keeps
    plug/unplug events caught within one `_WATCHDOG_INTERVAL_S` tick."""
    monkeypatch.setattr(capture_mod, "_WATCHDOG_INTERVAL_S", 0.02)
    world = default_world()
    mod = FakePaModule(world)
    statuses: list[tuple[str, str, str]] = []
    count = {"value": 3}

    cap = AudioCapture(
        Source.SYSTEM,
        "default",
        on_frames=lambda f: None,
        on_status=lambda s, n, d: statuses.append((s, n, d)),
        keepalive=False,
        pa_module=mod,
        device_count_fn=lambda: count["value"],
    )
    cap.start()
    try:
        assert cap.device is not None and cap.device.name == "Loopback A"

        # simulate a headset being plugged in: both the device list and the cheap winmm
        # count change together, as a real plug event would.
        world["devices"][0] = _device_dict(0, "New Loopback B", max_in=2, rate=48000.0, loopback=True)
        count["value"] = 4

        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if cap.device is not None and cap.device.name == "New Loopback B":
                break
            time.sleep(0.02)
    finally:
        cap.stop()

    assert cap.device is not None and cap.device.name == "New Loopback B"
    assert any(s[0] == "device_changed" and s[1] == "New Loopback B" for s in statuses)


def test_capture_watchdog_device_poll_cadence_still_detects_change(monkeypatch: pytest.MonkeyPatch) -> None:
    """upstream finding 4 (Windows fix): even with a perfectly steady device count (no
    plug/unplug at all), a pure default-device change (e.g. switched in Windows Settings)
    is still caught once the `_DEVICE_POLL_INTERVAL_S` cadence elapses."""
    monkeypatch.setattr(capture_mod, "_WATCHDOG_INTERVAL_S", 0.02)
    monkeypatch.setattr(capture_mod, "_DEVICE_POLL_INTERVAL_S", 0.05)
    world = default_world()
    mod = FakePaModule(world)
    statuses: list[tuple[str, str, str]] = []

    cap = AudioCapture(
        Source.SYSTEM,
        "default",
        on_frames=lambda f: None,
        on_status=lambda s, n, d: statuses.append((s, n, d)),
        keepalive=False,
        pa_module=mod,
        device_count_fn=lambda: 4,  # constant: no plug/unplug event at all
    )
    cap.start()
    try:
        assert cap.device is not None and cap.device.name == "Loopback A"

        world["devices"][0] = _device_dict(0, "New Loopback B", max_in=2, rate=48000.0, loopback=True)

        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            if cap.device is not None and cap.device.name == "New Loopback B":
                break
            time.sleep(0.02)
    finally:
        cap.stop()

    assert cap.device is not None and cap.device.name == "New Loopback B"
    assert any(s[0] == "device_changed" and s[1] == "New Loopback B" for s in statuses)


def test_capture_keepalive_opens_output_stream() -> None:
    mod = FakePaModule(default_world())
    cap = _make_capture(mod, on_frames=lambda f: None, keepalive=True)
    cap.start()
    try:
        assert any(s.is_output for pa in mod.instances for s in pa.streams)
    finally:
        cap.stop()


def test_capture_keepalive_failure_does_not_break_capture() -> None:
    world = default_world()
    mod = FakePaModule(world)

    def broken_get_default_output(_self: Any) -> dict[str, Any]:
        raise RuntimeError("no output device available")

    # break get_default_output_device_info for every PyAudio instance this module creates
    FakePyAudio.get_default_output_device_info = broken_get_default_output  # type: ignore[method-assign]
    try:
        statuses: list[tuple[str, str, str]] = []
        cap = _make_capture(mod, on_frames=lambda f: None, on_status=lambda s, n, d: statuses.append((s, n, d)), keepalive=True)
        device = cap.start()
        cap.stop()
        assert device.name == "Loopback A"
        assert statuses[0][0] == "started"
    finally:
        del FakePyAudio.get_default_output_device_info


def test_capture_missing_pyaudiowpatch_raises_runtime_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "pyaudiowpatch", None)
    with pytest.raises(RuntimeError, match="pyaudiowpatch"):
        capture_mod._import_pyaudiowpatch()

    cap = AudioCapture(Source.MIC, on_frames=lambda f: None)
    with pytest.raises(RuntimeError, match="pyaudiowpatch"):
        cap.start()


# --------------------------------------------------------------------------- #
# live tests: real pyaudiowpatch, real devices on this machine
# --------------------------------------------------------------------------- #


@pytest.mark.live
def test_live_list_devices_contains_loopback() -> None:
    devices = list_devices()
    assert any(d.is_loopback for d in devices), "expected at least one WASAPI loopback device"


@pytest.mark.live
def test_live_system_capture_with_keepalive_delivers_frames() -> None:
    frames: list[np.ndarray] = []
    cap = AudioCapture(Source.SYSTEM, on_frames=frames.append, keepalive=True)
    cap.start()
    try:
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and len(frames) < 20:
            time.sleep(0.05)
    finally:
        cap.stop()
    assert len(frames) >= 20, f"only {len(frames)} frames delivered in 3s with keep-alive on"


@pytest.mark.live
def test_live_mic_capture_starts_and_stops_without_error() -> None:
    cap = AudioCapture(Source.MIC, on_frames=lambda f: None)
    device = cap.start()
    assert device.is_input
    time.sleep(0.3)
    cap.stop()
    assert not cap.running

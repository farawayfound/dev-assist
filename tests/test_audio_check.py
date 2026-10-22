"""Tests for exocortex.audio.check (level verdicts, the speaker-routing check), the WASAPI default
mic in exocortex.audio.capture, and exocortex.audio.endpoint.

The routing check runs against a fake pyaudiowpatch-shaped module with no threads: the injected
``sleep`` advances a simulated clock, pulls that much audio from every open output stream's callback
and delivers it -- times a per-device gain, plus that device's noise -- to every open input stream.
"""
from __future__ import annotations

import sys
from typing import Any

import numpy as np
import pytest

from exocortex.audio import endpoint
from exocortex.audio.capture import DeviceInfo, list_devices, resolve_device
from exocortex.audio.check import (
    QUIET_PEAK,
    SILENCE_PEAK,
    LevelTracker,
    RoutingResult,
    band_rms,
    chirp,
    dbfs,
    render_device_for,
    routing_check,
)

RATE = 48000


# --------------------------------------------------------------------------- #
# LevelTracker
# --------------------------------------------------------------------------- #


class _Clock:
    def __init__(self) -> None:
        self.t = 100.0

    def __call__(self) -> float:
        return self.t


def _feed(tracker: LevelTracker, clock: _Clock, peak: float, seconds: float) -> None:
    for _ in range(int(seconds * 10)):
        tracker.push(peak / 2, peak)
        clock.t += 0.1


def test_tracker_waits_for_a_full_window_before_calling_a_feed_silent() -> None:
    clock = _Clock()
    tracker = LevelTracker(window_s=1.5, clock=clock)
    assert tracker.verdict() == "waiting"
    _feed(tracker, clock, 1e-9, 0.5)
    assert tracker.verdict() == "waiting"          # too early to call it
    _feed(tracker, clock, 1e-9, 1.5)
    assert tracker.verdict() == "silent"


def test_tracker_ok_at_once_and_quiet_below_minus_50_dbfs() -> None:
    clock = _Clock()
    tracker = LevelTracker(window_s=1.5, clock=clock)
    tracker.push(0.05, 0.1)
    assert tracker.verdict() == "ok"               # speech-level peak: no need to wait
    clock.t += 5.0
    assert tracker.verdict() == "waiting"          # nothing for a whole window (no callbacks)
    tracker.reset()
    _feed(tracker, clock, QUIET_PEAK / 3, 2.0)
    assert tracker.verdict() == "quiet"
    assert SILENCE_PEAK < QUIET_PEAK / 3


def test_tracker_one_loud_moment_in_the_window_is_enough() -> None:
    clock = _Clock()
    tracker = LevelTracker(window_s=1.5, clock=clock)
    _feed(tracker, clock, 1e-9, 2.0)
    tracker.push(0.1, 0.2)
    assert tracker.verdict() == "ok"
    assert tracker.peak() == pytest.approx(0.2)
    _feed(tracker, clock, 1e-9, 2.0)
    assert tracker.verdict() == "silent"


# --------------------------------------------------------------------------- #
# signal helpers
# --------------------------------------------------------------------------- #


def test_band_rms_keeps_the_voice_band_and_drops_rumble() -> None:
    t = np.arange(RATE) / RATE
    tone = 0.5 * np.sin(2 * np.pi * 1000 * t)
    rumble = 0.5 * np.sin(2 * np.pi * 50 * t)
    assert band_rms(tone, RATE) == pytest.approx(0.5 / np.sqrt(2), rel=0.02)
    assert band_rms(rumble, RATE) < 1e-3
    assert band_rms(np.zeros(0), RATE) == 0.0


def test_chirp_is_interleaved_bounded_and_mostly_in_band() -> None:
    sound = chirp(RATE, 2)
    assert sound.dtype == np.float32
    assert sound.size % 2 == 0
    left, right = sound[0::2], sound[1::2]
    assert np.array_equal(left, right)
    assert 0.25 <= float(np.max(np.abs(left))) <= 0.3001
    assert abs(left[0]) < 1e-3                     # faded in: no click
    assert band_rms(left, RATE) > 0.5 * float(np.sqrt(np.mean(left.astype(np.float64) ** 2)))
    assert 0.8 < left.size / RATE < 1.2            # ~1 s


def test_dbfs_is_finite_for_silence() -> None:
    assert dbfs(1.0) == pytest.approx(0.0)
    assert dbfs(0.1) == pytest.approx(-20.0)
    assert dbfs(0.0) == pytest.approx(-180.0)


def _dev(index: int, name: str, *, is_input: bool, loopback: bool = False) -> DeviceInfo:
    return DeviceInfo(index=index, name=name, is_loopback=loopback, is_input=is_input,
                      sample_rate=RATE, channels=2, is_default=False)


def test_render_device_for_maps_a_loopback_to_its_output() -> None:
    devices = [
        _dev(1, "Speakers (Realtek(R) Audio)", is_input=False),
        _dev(2, "Odyssey G61SH (NVIDIA High Definition Audio)", is_input=False),
        _dev(3, "Odyssey G61SH (NVIDIA High Definition Audio) [Loopback]", is_input=True, loopback=True),
        _dev(4, "Microphone Array (Realtek(R) Audio)", is_input=True),
    ]
    found = render_device_for(devices[2], devices)
    assert found is not None and found.index == 2
    assert render_device_for(_dev(9, "Nothing like it [Loopback]", is_input=True, loopback=True), devices) is None


def _result(**over: Any) -> RoutingResult:
    base = dict(output_device="Speakers A", system_device="Speakers A [Loopback]", mic_device="Mic A",
                system_before_db=-180.0, system_during_db=-15.0, mic_before_db=-70.0, mic_during_db=-69.0,
                mic_silent=False)
    base.update(over)
    return RoutingResult(**base)


def test_routing_result_classifies_bleed() -> None:
    assert _result().isolated and _result().bleed == "none"
    assert _result(mic_during_db=-64.0).bleed == "slight"
    assert _result(mic_during_db=-40.0).bleed == "strong" and not _result(mic_during_db=-40.0).isolated
    assert _result(mic_silent=True).bleed == "unknown"
    assert _result(mic_device="").bleed == "unknown"
    assert not _result(system_during_db=-175.0).output_ok          # no real rise
    assert not _result(system_before_db=-90.0, system_during_db=-60.0).output_ok   # rose, but far too faint


# --------------------------------------------------------------------------- #
# routing_check against a simulated room
# --------------------------------------------------------------------------- #


class _Stream:
    def __init__(self, world: "_World", **kw: Any) -> None:
        self.world = world
        self.kw = kw
        self.channels = int(kw["channels"])
        self.callback = kw["stream_callback"]
        self.is_output = bool(kw.get("output"))
        self.device_index = kw.get("output_device_index") if self.is_output else kw.get("input_device_index")
        self.closed = False

    def stop_stream(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True
        self.world.open_streams.remove(self)


class _PyAudio:
    def __init__(self, world: "_World") -> None:
        self.world = world

    def get_host_api_info_by_type(self, _api: int) -> dict[str, Any]:
        return {"index": 0, "name": "Windows WASAPI", "defaultInputDevice": self.world.default_input}

    def get_device_count(self) -> int:
        return len(self.world.devices)

    def get_device_info_by_index(self, i: int) -> dict[str, Any]:
        return dict(self.world.devices[i])

    def get_default_wasapi_loopback(self) -> dict[str, Any]:
        return dict(self.world.devices[0])

    def get_default_input_device_info(self) -> dict[str, Any]:
        # the MME entry: a different index and a name cut at 31 characters
        return {"index": 99, "name": self.world.devices[self.world.default_input]["name"][:31]}

    def get_default_output_device_info(self) -> dict[str, Any]:
        return dict(self.world.devices[2])

    def open(self, **kw: Any) -> _Stream:
        stream = _Stream(self.world, **kw)
        self.world.open_streams.append(stream)
        self.world.opened.append(stream)
        return stream

    def terminate(self) -> None:
        self.world.terminated += 1


class _World:
    """Devices: 0 = loopback of output 2, 1 = a mic with a long name, 2 = the output, 3 = another mic."""

    paWASAPI = 13
    paFloat32 = 1
    paContinue = 0

    def __init__(self, *, loop_gain: float = 1.0, mic_gain: float = 0.0, mic_noise: float = 1e-3) -> None:
        def d(i: int, name: str, max_in: int, max_out: int, loop: bool = False) -> dict[str, Any]:
            return {"index": i, "name": name, "hostApi": 0, "maxInputChannels": max_in,
                    "maxOutputChannels": max_out, "defaultSampleRate": float(RATE), "isLoopbackDevice": loop}

        self.devices = [
            d(0, "Speakers A [Loopback]", 2, 0, loop=True),
            d(1, "Microphone Array (Realtek(R) Audio)", 4, 0),
            d(2, "Speakers A", 0, 2),
            d(3, "Headset Mic", 1, 0),
        ]
        self.default_input = 1
        self.gain = {0: loop_gain, 1: mic_gain, 3: mic_gain}
        self.noise = {0: 0.0, 1: mic_noise, 3: mic_noise}
        self.open_streams: list[_Stream] = []
        self.opened: list[_Stream] = []
        self.terminated = 0
        self.rng = np.random.default_rng(0)

    def PyAudio(self) -> _PyAudio:  # noqa: N802 - mirrors pyaudiowpatch
        return _PyAudio(self)

    def sleep(self, seconds: float) -> None:
        n = int(round(seconds * RATE))
        played = np.zeros(n)
        for s in [s for s in self.open_streams if s.is_output]:
            data, _flag = s.callback(None, n, None, 0)
            played += np.frombuffer(data, dtype=np.float32).reshape(-1, s.channels).mean(axis=1)
        for s in [s for s in self.open_streams if not s.is_output]:
            idx = s.device_index
            mono = self.gain[idx] * played + self.noise[idx] * self.rng.standard_normal(n)
            frames = np.repeat(mono.reshape(-1, 1), s.channels, axis=1).astype(np.float32)
            s.callback(frames.tobytes(), n, None, 0)


def test_isolated_routing_reaches_the_output_feed_and_not_the_mic() -> None:
    world = _World(loop_gain=1.0, mic_gain=0.0)
    result = routing_check(pa_module=world, sleep=world.sleep)
    assert result.output_device == "Speakers A"
    assert result.system_device == "Speakers A [Loopback]"
    assert result.mic_device == "Microphone Array (Realtek(R) Audio)"   # the WASAPI default, full name
    assert result.output_ok and result.system_rise_db > 60
    assert result.bleed == "none" and result.isolated
    # the chirp went to the output behind the loopback, and everything was closed again
    out = [s for s in world.opened if s.is_output]
    assert len(out) == 1 and out[0].device_index == 2
    assert world.open_streams == [] and all(s.closed for s in world.opened)


def test_a_mic_that_hears_the_speakers_is_bleed() -> None:
    world = _World(loop_gain=1.0, mic_gain=0.1)
    result = routing_check(pa_module=world, sleep=world.sleep)
    assert result.output_ok
    assert result.bleed == "strong" and result.mic_rise_db > 20 and not result.isolated


def test_the_wrong_loopback_never_hears_the_chirp() -> None:
    world = _World(loop_gain=0.0, mic_gain=0.0)
    result = routing_check(pa_module=world, sleep=world.sleep)
    assert not result.output_ok and not result.isolated


def test_a_muted_mic_cannot_be_measured() -> None:
    world = _World(loop_gain=1.0, mic_gain=0.0, mic_noise=0.0)
    result = routing_check(pa_module=world, sleep=world.sleep)
    assert result.output_ok
    assert result.mic_silent and result.bleed == "unknown" and not result.isolated


def test_routing_check_uses_the_chosen_mic() -> None:
    world = _World(loop_gain=1.0, mic_gain=0.0)
    result = routing_check("default", "Headset", pa_module=world, sleep=world.sleep)
    assert result.mic_device == "Headset Mic"
    assert any(not s.is_output and s.device_index == 3 for s in world.opened)


def test_routing_check_without_a_mic_still_measures_the_output_feed() -> None:
    world = _World(loop_gain=1.0)
    result = routing_check("default", "no such mic", pa_module=world, sleep=world.sleep)
    assert result.mic_device == "" and result.output_ok and result.bleed == "unknown"


def test_routing_check_raises_for_an_unknown_output_feed() -> None:
    world = _World()
    with pytest.raises(ValueError):
        routing_check("no such loopback", pa_module=world, sleep=world.sleep)


# --------------------------------------------------------------------------- #
# capture: the WASAPI default mic, not the truncated MME name
# --------------------------------------------------------------------------- #


def test_the_default_mic_is_the_wasapi_default_by_its_full_name() -> None:
    world = _World()
    mics = [d for d in list_devices(pa_module=world) if d.is_input and not d.is_loopback]
    assert [d.name for d in mics if d.is_default] == ["Microphone Array (Realtek(R) Audio)"]
    assert resolve_device("mic", "default", pa_module=world).index == 1
    world.default_input = 3
    assert resolve_device("mic", "default", pa_module=world).name == "Headset Mic"


# --------------------------------------------------------------------------- #
# capture: PortAudio control calls are serialised, with COM on the calling thread
# --------------------------------------------------------------------------- #


def test_portaudio_never_initialises_on_two_threads_at_once() -> None:
    """Two threads in Pa_Initialize at once was an access violation (Settings opened the mic and the
    output feed in parallel while the captures' watchdogs re-enumerate on their own threads)."""
    import threading
    import time

    state = {"active": 0, "most": 0}
    guard = threading.Lock()

    class _Racy(_World):
        def PyAudio(self) -> _PyAudio:  # noqa: N802
            with guard:
                state["active"] += 1
                state["most"] = max(state["most"], state["active"])
            time.sleep(0.005)
            with guard:
                state["active"] -= 1
            return super().PyAudio()

    world = _Racy()
    threads = [threading.Thread(target=lambda: (list_devices(pa_module=world), resolve_device("mic", "default", pa_module=world)))
               for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(5.0)
    assert state["most"] == 1


def test_every_thread_gets_com_once_before_portaudio(monkeypatch: pytest.MonkeyPatch) -> None:
    """While another capture keeps PortAudio initialised, opening a stream from a thread without COM
    fails with -9999 (the watchdog's reopen, a live device switch, the Settings check)."""
    import threading

    from exocortex.audio import capture as capture_mod

    calls: list[tuple[str, int]] = []

    class _Ole32:
        def CoInitializeEx(self, _reserved: Any, flags: int) -> int:  # noqa: N802
            calls.append((threading.current_thread().name, flags))
            return 0

    class _WinDll:
        ole32 = _Ole32()

    monkeypatch.setattr(capture_mod.sys, "platform", "win32")
    monkeypatch.setattr(capture_mod.ctypes, "windll", _WinDll(), raising=False)

    def work() -> None:
        for _ in range(3):
            with capture_mod.pa_guard():
                pass

    threads = [threading.Thread(target=work, name=f"opener-{i}") for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(5.0)
    assert sorted(calls) == [("opener-0", 0x2), ("opener-1", 0x2)]   # once per thread, STA


# --------------------------------------------------------------------------- #
# endpoint
# --------------------------------------------------------------------------- #


def test_endpoint_calls_degrade_off_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(endpoint.sys, "platform", "linux")
    assert endpoint.capture_endpoint_state() is None
    assert endpoint.set_capture_muted("Mic", False) is False


def test_endpoint_failures_never_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(_name: str | None) -> Any:
        raise OSError("no Core Audio here")

    monkeypatch.setattr(endpoint, "_find_capture_endpoint", boom)
    assert endpoint.capture_endpoint_state("Mic") is None
    assert endpoint.set_capture_muted("Mic", False) is False


@pytest.mark.live
@pytest.mark.skipif(sys.platform != "win32", reason="Core Audio is Windows-only")
def test_live_default_capture_endpoint_state_is_readable() -> None:
    state = endpoint.capture_endpoint_state()
    assert state is not None and state.name and 0.0 <= state.volume <= 1.0
    assert endpoint.capture_endpoint_state(state.name) == state

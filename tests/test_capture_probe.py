"""Tests for the Settings > Audio helpers in exocortex.audio.capture (2026-10-29): ``candidates_for``
(the devices a spec of a kind can name) and ``probe_device`` / ``ProbeResult`` (open a spec as a real
``AudioCapture`` for a moment and report, never raise).

Offline: the fake pyaudiowpatch-shaped module from tests/test_capture.py, extended here so every input
stream it opens starts feeding synthetic audio at once (``probe_device`` blocks while the stream runs,
so the test cannot feed it from outside)."""
from __future__ import annotations

from typing import Any

import pytest

from exocortex.audio.capture import ProbeResult, candidates_for, list_devices, probe_device, resolve_device
from tests.test_capture import FakePaModule, FakeStream, _device_dict, default_world

_PROBE_S = 0.4


class _AutoFeedModule(FakePaModule):
    """A fake pyaudiowpatch whose input streams feed ``feed_seconds`` of a sine at ``amplitude`` as
    soon as they open (real time, so the probe's own window sees them)."""

    def __init__(self, world: dict[str, Any], *, feed_seconds: float = 0.25, amplitude: float = 0.3) -> None:
        super().__init__(world)
        self._feed_seconds = feed_seconds
        self._amplitude = amplitude

    def PyAudio(self) -> Any:  # noqa: N802 - mirrors the real module
        pa = super().PyAudio()
        original_open = pa.open

        def _open(**kwargs: Any) -> FakeStream:
            stream = original_open(**kwargs)
            if stream.is_input:
                stream.start_input_feed(self._feed_seconds, amplitude=self._amplitude, speedup=1.0)
            return stream

        pa.open = _open  # type: ignore[method-assign]
        return pa


def _output_streams(mod: FakePaModule) -> list[FakeStream]:
    return [s for pa in mod.instances for s in pa.streams if s.is_output]


# --------------------------------------------------------------------------- #
# candidates_for
# --------------------------------------------------------------------------- #


class _TruncatedDefaultModule(FakePaModule):
    """Windows reporting the default input under another host API, name cut at 31 characters."""

    def PyAudio(self) -> Any:  # noqa: N802
        pa = super().PyAudio()
        pa.get_default_input_device_info = lambda: {"name": self.world["default_input_name"]}  # type: ignore[method-assign]
        return pa


def test_default_mic_is_flagged_even_when_windows_reports_a_truncated_name() -> None:
    world = default_world()
    world["devices"] = [
        _device_dict(0, "Loopback A", max_in=2, loopback=True),
        _device_dict(1, "Microphone Array (Realtek(R) Audio)", max_in=2),
        _device_dict(2, "Microphone (Yeti Stereo Microphone)", max_in=2),
    ]
    world["default_input_name"] = "Microphone (Yeti Stereo Microph"
    mod = _TruncatedDefaultModule(world)
    flags = {d.name: d.is_default for d in list_devices(pa_module=mod)}
    assert flags == {"Loopback A": True, "Microphone Array (Realtek(R) Audio)": False,
                     "Microphone (Yeti Stereo Microphone)": True}
    assert resolve_device("mic", "default", pa_module=mod).name == "Microphone (Yeti Stereo Microphone)"

    world["default_input_name"] = "Microphone Array (Realtek(R) Audio)"          # an exact name still wins
    assert resolve_device("mic", "default", pa_module=mod).index == 1
    world["default_input_name"] = "Something unplugged"
    assert not any(d.is_default for d in list_devices(pa_module=mod) if not d.is_loopback)
    with pytest.raises(ValueError, match="could not resolve default mic"):
        resolve_device("mic", "default", pa_module=mod)


def test_capture_open_initialises_the_host_once() -> None:
    """``_open`` resolves on the instance it opens the stream on: one ``PyAudio()`` per open, not two."""
    from exocortex.audio.capture import AudioCapture
    from exocortex.bus import Source

    mod = FakePaModule(default_world())
    cap = AudioCapture(Source.MIC, "default", on_frames=lambda f: None, pa_module=mod)
    cap.start()
    try:
        assert len(mod.instances) == 1 and not mod.instances[0].terminated
    finally:
        cap.stop()
    assert mod.instances[0].terminated


def test_candidates_for_filters_by_kind_exactly_like_resolve_device() -> None:
    world = default_world()
    world["devices"].append(_device_dict(3, "Mic B", max_in=2, rate=48000.0))
    devices = list_devices(pa_module=FakePaModule(world))
    assert [d.name for d in candidates_for("system", devices)] == ["Loopback A"]
    assert [d.name for d in candidates_for("mic", devices)] == ["Mic A", "Mic B"]   # never the output device


# --------------------------------------------------------------------------- #
# probe_device
# --------------------------------------------------------------------------- #


def test_probe_reports_the_device_and_a_signal() -> None:
    mod = _AutoFeedModule(default_world(), amplitude=0.3)
    levels: list[tuple[float, float]] = []
    res = probe_device("mic", "default", seconds=_PROBE_S, pa_module=mod, on_level=lambda r, p: levels.append((r, p)))
    assert isinstance(res, ProbeResult) and res.ok and res.error == ""
    assert res.kind == "mic" and res.spec == "default"
    assert res.device is not None and res.device.name == "Mic A"
    assert res.frames > 0
    assert _PROBE_S - 0.05 < res.seconds < _PROBE_S + 0.5   # the listening window (not stop()); sleep may wake a hair early
    assert 0.2 < res.rms < 0.3 and 0.25 < res.peak <= 0.31          # a 0.3 sine: rms = 0.3 / sqrt(2)
    assert -12.5 < res.peak_dbfs < -10.0
    assert res.signal == "ok"
    assert res.describe().startswith("OK: Mic A (44100 Hz, 1 ch) delivers audio, peak -1")
    assert levels and all(r > 0 for r, _ in levels)                 # the live meter saw it
    assert all(pa.terminated for pa in mod.instances)               # nothing left open


def test_probe_classifies_a_near_silent_stream_as_silent() -> None:
    mod = _AutoFeedModule(default_world(), amplitude=1e-4)          # the keep-alive tone's level
    res = probe_device("system", "default", seconds=_PROBE_S, pa_module=mod)
    assert res.ok and res.frames > 0
    assert res.signal == "silent" and res.peak_dbfs < -60
    assert "silent" in res.describe() and "Play something" in res.describe()


def test_probe_classifies_no_callbacks_as_none_with_a_kind_specific_hint() -> None:
    res = probe_device("system", "default", seconds=_PROBE_S, pa_module=FakePaModule(default_world()))
    assert res.ok and res.frames == 0 and res.rms == 0.0 and res.peak == 0.0
    assert res.signal == "none" and res.peak_dbfs == -120.0
    assert "no audio arrived" in res.describe() and "Nothing is playing" in res.describe()

    res_mic = probe_device("mic", "default", seconds=_PROBE_S, pa_module=FakePaModule(default_world()))
    assert res_mic.signal == "none" and "Privacy" in res_mic.describe()


def test_probe_never_raises_for_a_spec_that_does_not_resolve() -> None:
    res = probe_device("mic", "Nonexistent", seconds=_PROBE_S, pa_module=FakePaModule(default_world()))
    assert not res.ok and res.device is None and res.frames == 0
    assert res.signal == "error"
    assert "no mic device matching 'Nonexistent'" in res.error
    assert res.describe().startswith("Could not open 'Nonexistent':")


def test_probe_never_raises_when_the_module_is_broken() -> None:
    class _Broken:
        def PyAudio(self) -> Any:  # noqa: N802
            raise RuntimeError("WASAPI host unavailable")

    res = probe_device("system", "default", seconds=_PROBE_S, pa_module=_Broken())
    assert not res.ok and "WASAPI host unavailable" in res.error


@pytest.mark.parametrize("keepalive", [True, False])
def test_probe_honours_the_keepalive_flag_for_system_audio(keepalive: bool) -> None:
    mod = _AutoFeedModule(default_world())
    res = probe_device("system", "default", seconds=_PROBE_S, keepalive=keepalive, pa_module=mod)
    assert res.ok
    assert bool(_output_streams(mod)) is keepalive


def test_probe_never_starts_a_keepalive_for_the_mic() -> None:
    mod = _AutoFeedModule(default_world())
    assert probe_device("mic", "default", seconds=_PROBE_S, keepalive=True, pa_module=mod).ok
    assert not _output_streams(mod)

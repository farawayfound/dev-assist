"""SpeechPipeline device handling: switching a source's device live (Settings > Audio, through
exo-cortex's `restart_source`, whose own cases are tests/test_pipeline_restart.py) and the warning
for a mic that delivers digital silence (muted in Windows: it opens, but hears nothing). Plus the
Engine side: `apply_settings` reopens the capture whose device changed, off the UI thread.
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest

from exocortex.bus import ComponentStatusEvent, EventBus, FinalTranscriptEvent, Source, drain
from exocortex.config import Config
from exocortex.speech import pipeline as pipeline_mod
from tests.test_pipeline import FakeTranscriber, _fake_device, _make_cfg, _make_pipeline, _wait_until


class _LevelCapture:
    """Fake AudioCapture that records its spec and hands the test its callbacks."""

    def __init__(self, source: Source, spec: str, *, on_frames: Callable[[np.ndarray], None],
                 on_level: Callable[[float, float], None] | None = None, **_kw: Any) -> None:
        self.source = source
        self.spec = spec
        self.on_frames = on_frames
        self.on_level = on_level
        self._device = _fake_device(source, f"{source.value}:{spec}")
        self.running = False
        self.stopped = False

    def start(self) -> Any:
        self.running = True
        return self._device

    def stop(self) -> None:
        self.running = False
        self.stopped = True

    def set_muted(self, muted: bool) -> None:
        pass

    @property
    def device(self) -> Any:
        return self._device

    def push_audio(self, audio: np.ndarray) -> None:
        n = (len(audio) // 512) * 512
        for i in range(0, n, 512):
            self.on_frames(audio[i:i + 512].astype(np.float32))


class _Factory:
    def __init__(self) -> None:
        self.made: list[_LevelCapture] = []

    def __call__(self, source: Source, spec: str, **kw: Any) -> _LevelCapture:
        cap = _LevelCapture(source, spec, **kw)
        self.made.append(cap)
        return cap

    def latest(self, source: Source) -> _LevelCapture:
        return [c for c in self.made if c.source is source][-1]


def _status(events: list[Any], state: str) -> list[ComponentStatusEvent]:
    return [e for e in events if isinstance(e, ComponentStatusEvent) and e.component == "audio" and e.state == state]


def test_restart_source_reopens_one_source_on_the_new_device(two_speakers_audio, silero_path) -> None:
    audio, _sr = two_speakers_audio
    cfg = _make_cfg(capture_system=True, capture_mic=True, echo_suppression=False, partials_enabled=False)
    bus = EventBus()
    q = bus.subscribe()
    factory = _Factory()
    pipe = _make_pipeline(cfg, bus, FakeTranscriber(), None, factory)
    pipe.start()
    try:
        old_mic, system = factory.latest(Source.MIC), factory.latest(Source.SYSTEM)
        assert old_mic.spec == "default"
        drain(q)

        cfg.audio.mic_device = "Headset"
        assert pipe.restart_source(Source.MIC) == "mic:Headset"

        new_mic = factory.latest(Source.MIC)
        assert new_mic is not old_mic and new_mic.spec == "Headset" and new_mic.running
        assert old_mic.stopped and not system.stopped          # the other feed is untouched
        ready = _status(drain(q), "ready")
        assert ready and ready[-1].detail == "system:default | mic:Headset"

        # the VAD worker and its queue carried on: speech on the new device is transcribed as Me
        new_mic.push_audio(audio[: 16000 * 6])
        new_mic.push_audio(np.zeros(16000, dtype=np.float32))
        finals: list[FinalTranscriptEvent] = []

        def _have_final() -> bool:
            finals.extend(e for e in drain(q) if isinstance(e, FinalTranscriptEvent))
            return bool(finals)

        assert _wait_until(_have_final)
        assert finals[0].segment.source is Source.MIC and finals[0].segment.speaker_id == "me"
    finally:
        pipe.stop()


def test_restart_source_is_a_noop_while_the_pipeline_is_stopped(silero_path) -> None:
    cfg = _make_cfg(capture_system=True, capture_mic=False)
    factory = _Factory()
    pipe = _make_pipeline(cfg, EventBus(), FakeTranscriber(), None, factory)
    assert pipe.restart_source(Source.SYSTEM) is None            # not running yet
    assert factory.made == []


def test_stop_during_a_slow_reopen_waits_for_it_and_closes_the_new_capture(silero_path) -> None:
    cfg = _make_cfg(capture_system=False, capture_mic=True)
    gate = threading.Event()
    made: list[_LevelCapture] = []

    class _Slow(_LevelCapture):
        def start(self) -> Any:
            gate.wait(5.0)
            return super().start()

    def factory(source: Source, spec: str, **kw: Any) -> _LevelCapture:
        cap = (_Slow if made else _LevelCapture)(source, spec, **kw)
        made.append(cap)
        return cap

    pipe = _make_pipeline(cfg, EventBus(), FakeTranscriber(), None, factory)
    pipe.start()
    cfg.audio.mic_device = "Headset"
    switch = threading.Thread(target=pipe.restart_source, args=(Source.MIC,))
    switch.start()
    assert _wait_until(lambda: len(made) == 2, timeout=5.0)
    stopper = threading.Thread(target=pipe.stop)
    stopper.start()
    stopper.join(0.2)
    assert stopper.is_alive()                                   # stop() waits for the reopen in flight...
    gate.set()
    switch.join(5.0)
    stopper.join(5.0)
    assert not stopper.is_alive() and made[1].stopped and not made[1].running   # ...then closes it


# --------------------------------------------------------------------------- #
# a mic delivering digital silence
# --------------------------------------------------------------------------- #


@pytest.fixture
def fast_silence(monkeypatch: pytest.MonkeyPatch) -> float:
    monkeypatch.setattr(pipeline_mod, "_MIC_SILENCE_WARN_S", 0.05)
    return 0.08


def _silent_for(cap: _LevelCapture, seconds: float, peak: float = 1e-9) -> None:
    cap.on_level(peak / 2, peak)
    threading.Event().wait(seconds)
    cap.on_level(peak / 2, peak)


def test_a_silent_mic_is_reported_once(silero_path, fast_silence: float) -> None:
    cfg = _make_cfg(capture_system=True, capture_mic=True)
    bus = EventBus()
    q = bus.subscribe()
    factory = _Factory()
    pipe = _make_pipeline(cfg, bus, FakeTranscriber(), None, factory)
    pipe.start()
    try:
        mic, system = factory.latest(Source.MIC), factory.latest(Source.SYSTEM)
        drain(q)
        _silent_for(system, fast_silence)                       # an idle output feed is normal
        assert _status(drain(q), "error") == []
        _silent_for(mic, fast_silence)
        errors = _status(drain(q), "error")
        assert len(errors) == 1 and "mic:default" in errors[0].detail and "muted in Windows" in errors[0].detail
        _silent_for(mic, fast_silence)
        assert _status(drain(q), "error") == []                 # once, not every level
        mic.on_level(0.05, 0.2)                                 # it hears something again...
        _silent_for(mic, fast_silence)
        assert len(_status(drain(q), "error")) == 1             # ...so a new silence is news
    finally:
        pipe.stop()


def test_a_mic_muted_in_the_app_is_not_reported(silero_path, fast_silence: float) -> None:
    cfg = _make_cfg(capture_system=False, capture_mic=True)
    bus = EventBus()
    q = bus.subscribe()
    factory = _Factory()
    pipe = _make_pipeline(cfg, bus, FakeTranscriber(), None, factory)
    pipe.start()
    try:
        pipe.set_mic_muted(True)
        drain(q)
        _silent_for(factory.latest(Source.MIC), fast_silence, peak=0.0)
        assert _status(drain(q), "error") == []
    finally:
        pipe.stop()


# --------------------------------------------------------------------------- #
# Engine.apply_settings
# --------------------------------------------------------------------------- #


def test_engine_reopens_the_capture_whose_device_changed(tmp_path: Path) -> None:
    from exocortex import app as app_mod

    cfg = Config()
    cfg.path = tmp_path / "config.toml"
    eng = app_mod.Engine(cfg, EventBus())
    calls: list[tuple[Source, str]] = []
    done = threading.Event()

    class _Pipeline:
        def restart_source(self, source: Source) -> None:
            calls.append((source, threading.current_thread().name))
            if len(calls) == 2:
                done.set()

        def set_mic_muted(self, muted: bool) -> None:
            pass

    eng.pipeline = _Pipeline()
    cfg.audio.mic_device = "Headset"
    cfg.audio.system_device = "Speakers"
    eng.apply_settings({"audio.mic_device": "Headset", "audio.system_device": "Speakers"})
    assert done.wait(5.0)
    assert sorted(s.value for s, _ in calls) == ["mic", "system"]
    assert all(name != threading.main_thread().name for _, name in calls)
    eng.apply_settings({"audio.echo_suppression": False})
    assert len(calls) == 2

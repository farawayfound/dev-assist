"""Tests for ``SpeechPipeline.restart_source`` / ``devices`` / ``running`` (2026-10-29, Settings > Audio):
reopening one source's capture against the current config while the pipeline runs -- a new device, the
keep-alive switch, a source switched off or on -- without touching the other source or the workers.

Offline, same fakes pattern as tests/test_pipeline.py (fake captures via ``capture_factory``, the fake
transcriber, the real Silero VAD so the vad-* workers are genuine)."""
from __future__ import annotations

import threading
import time
from typing import Any, Callable

import numpy as np
import pytest

from exocortex.audio.capture import DeviceInfo
from exocortex.bus import AudioStatusEvent, ComponentStatusEvent, EventBus, FinalTranscriptEvent, Source, drain
from exocortex.speech.pipeline import SpeechPipeline
from tests.test_pipeline import FakeTranscriber, _chunk, _make_cfg

_WAIT_S = 5.0


class _Capture:
    """Records what the pipeline asked for. ``fail`` makes ``start()`` raise."""

    def __init__(self, source: Source, spec: str, *, target_rate: int, frame_samples: int,
                 on_frames: Callable[[np.ndarray], None], on_level: Any = None, on_status: Any = None,
                 keepalive: bool = True, fail: bool = False) -> None:
        self.source, self.spec, self.keepalive, self.fail = source, spec, keepalive, fail
        self.on_frames, self.on_status = on_frames, on_status
        self.starts = 0
        self.stops = 0
        self.muted = False
        self._running = False

    def start(self) -> DeviceInfo:
        if self.fail:
            raise RuntimeError(f"cannot open {self.spec!r}")
        self.starts += 1
        self._running = True
        if self.on_status is not None:
            self.on_status("started", self.spec, "")
        return self.device

    def stop(self) -> None:
        self.stops += 1
        self._running = False
        if self.on_status is not None:
            self.on_status("stopped", self.spec, "")

    def set_muted(self, muted: bool) -> None:
        self.muted = muted

    def push(self, frame: np.ndarray) -> None:
        self.on_frames(frame)

    @property
    def device(self) -> DeviceInfo:
        return DeviceInfo(index=0, name=self.spec, is_loopback=self.source is Source.SYSTEM, is_input=True,
                          sample_rate=16000, channels=1, is_default=False)

    @property
    def running(self) -> bool:
        return self._running


class _Factory:
    def __init__(self, fail_specs: set[str] = frozenset()) -> None:
        self.made: list[_Capture] = []
        self.fail_specs = set(fail_specs)

    def __call__(self, source: Source, spec: str, **kw: Any) -> _Capture:
        cap = _Capture(source, spec, fail=spec in self.fail_specs, **kw)
        self.made.append(cap)
        return cap

    def for_source(self, source: Source) -> list[_Capture]:
        return [c for c in self.made if c.source is source]


def _pipeline(cfg: Any, factory: _Factory) -> tuple[SpeechPipeline, EventBus, Any]:
    bus = EventBus()
    sub = bus.subscribe()
    clock = time.monotonic
    ids = iter(range(1, 10_000))
    pipe = SpeechPipeline(cfg, bus, FakeTranscriber(), None, clock, lambda: next(ids), capture_factory=factory)
    return pipe, bus, sub


def _by_kind(events: list[Any], kind: type) -> list[Any]:
    return [e for e in events if isinstance(e, kind)]


def test_restart_source_swaps_the_capture_for_the_new_device(silero_path: str) -> None:
    cfg = _make_cfg(capture_system=True, capture_mic=False)
    cfg.audio.system_device = "Speakers A"
    factory = _Factory()
    pipe, _bus, sub = _pipeline(cfg, factory)
    pipe.start()
    try:
        old = factory.for_source(Source.SYSTEM)[0]
        assert pipe.devices() == {Source.SYSTEM: "Speakers A"} and pipe.running
        drain(sub)

        cfg.audio.system_device = "Odyssey"
        cfg.audio.loopback_keepalive = True
        assert pipe.restart_source(Source.SYSTEM) == "Odyssey"

        new = factory.for_source(Source.SYSTEM)[1]
        assert old.stops == 1 and new.starts == 1
        assert new.spec == "Odyssey" and new.keepalive is True      # the current config, not start()'s
        assert pipe.devices() == {Source.SYSTEM: "Odyssey"}
        assert set(pipe._vad_threads) == {Source.SYSTEM}             # the worker was kept, not duplicated

        events = drain(sub)
        audio = _by_kind(events, AudioStatusEvent)
        states = [(e.state, e.device_name) for e in audio if e.source is Source.SYSTEM]
        assert ("stopped", "Speakers A") not in states                 # the swap must not read as "audio stopped"
        assert ("started", "Odyssey") in states
        assert ("device_changed", "Odyssey") in states
        changed = next(e for e in audio if e.state == "device_changed")
        assert changed.detail == "settings"
        ready = [e for e in _by_kind(events, ComponentStatusEvent) if e.component == "audio" and e.state == "ready"]
        assert ready and ready[-1].detail == "Odyssey"
    finally:
        pipe.stop()


def test_restarted_capture_still_feeds_the_vad_and_asr(
    silero_path: str, two_speakers_audio: tuple[np.ndarray, int]
) -> None:
    cfg = _make_cfg(capture_system=True, capture_mic=False)
    factory = _Factory()
    pipe, _bus, sub = _pipeline(cfg, factory)
    pipe.start()
    try:
        cfg.audio.system_device = "Odyssey"
        pipe.restart_source(Source.SYSTEM)
        new = factory.for_source(Source.SYSTEM)[1]
        drain(sub)
        audio, _sr = two_speakers_audio
        for frame in _chunk(audio[: 16000 * 8]):                    # the first eight seconds: three utterances
            new.push(frame)
            time.sleep(0.0005)
        deadline = time.monotonic() + _WAIT_S
        finals: list[FinalTranscriptEvent] = []
        while time.monotonic() < deadline and not finals:
            finals = _by_kind(drain(sub), FinalTranscriptEvent)
            time.sleep(0.05)
        assert finals, "frames pushed through the reopened capture never became a final"
        assert finals[0].segment.source is Source.SYSTEM
    finally:
        pipe.stop()


def test_restart_source_disabled_stops_and_removes_the_capture(silero_path: str) -> None:
    cfg = _make_cfg(capture_system=True, capture_mic=False)
    factory = _Factory()
    pipe, _bus, sub = _pipeline(cfg, factory)
    pipe.start()
    try:
        drain(sub)
        cfg.audio.capture_system = False
        assert pipe.restart_source(Source.SYSTEM) is None
        old = factory.for_source(Source.SYSTEM)[0]
        assert old.stops == 1 and len(factory.made) == 1             # nothing new was opened
        assert pipe.devices() == {}
        events = drain(sub)
        states = [e.state for e in _by_kind(events, AudioStatusEvent) if e.source is Source.SYSTEM]
        assert states == ["stopped"]                                  # a real stop: not swallowed
        info = [e for e in _by_kind(events, ComponentStatusEvent) if e.component == "audio"]
        assert info and info[-1].state == "info" and "no source enabled" in info[-1].detail
    finally:
        pipe.stop()


def test_restart_source_enables_a_source_that_was_off_at_start(silero_path: str) -> None:
    cfg = _make_cfg(capture_system=True, capture_mic=False)
    cfg.audio.mic_device = "Yeti"
    factory = _Factory()
    pipe, _bus, sub = _pipeline(cfg, factory)
    pipe.set_mic_muted(True)                                          # remembered mute must carry over
    pipe.start()
    try:
        assert Source.MIC not in pipe._vad_threads
        drain(sub)
        cfg.audio.capture_mic = True
        assert pipe.restart_source(Source.MIC) == "Yeti"
        mic = factory.for_source(Source.MIC)
        assert len(mic) == 1 and mic[0].starts == 1 and mic[0].muted is True
        assert Source.MIC in pipe._vad_threads and pipe._vad_threads[Source.MIC].is_alive()
        assert pipe.devices() == {Source.SYSTEM: "default", Source.MIC: "Yeti"}
        ready = [e for e in _by_kind(drain(sub), ComponentStatusEvent) if e.component == "audio" and e.state == "ready"]
        assert ready and ready[-1].detail == "default | Yeti"
    finally:
        pipe.stop()


def test_restart_source_failure_restores_the_previous_capture_and_raises(silero_path: str) -> None:
    cfg = _make_cfg(capture_system=True, capture_mic=False)
    cfg.audio.system_device = "Speakers A"
    factory = _Factory(fail_specs={"Broken"})
    pipe, _bus, sub = _pipeline(cfg, factory)
    pipe.start()
    try:
        old = factory.for_source(Source.SYSTEM)[0]
        drain(sub)
        cfg.audio.system_device = "Broken"
        with pytest.raises(RuntimeError, match="cannot open 'Broken'"):
            pipe.restart_source(Source.SYSTEM)
        assert old.stops == 1 and old.starts == 2                     # stopped for the swap, started again
        assert pipe.devices() == {Source.SYSTEM: "Speakers A"}
        errors = [e for e in _by_kind(drain(sub), ComponentStatusEvent) if e.component == "audio" and e.state == "error"]
        assert errors and "system: cannot open 'Broken'" in errors[0].detail
    finally:
        pipe.stop()
    assert old.stops == 2                                             # stop() closed the restored capture


def test_restart_source_is_a_no_op_before_start_and_after_stop(silero_path: str) -> None:
    cfg = _make_cfg(capture_system=True, capture_mic=False)
    factory = _Factory()
    pipe, _bus, _sub = _pipeline(cfg, factory)
    assert pipe.restart_source(Source.SYSTEM) is None and not factory.made and not pipe.running
    pipe.start()
    pipe.stop()
    assert pipe.restart_source(Source.SYSTEM) is None and len(factory.made) == 1
    assert pipe.devices() == {}


def test_concurrent_restarts_are_serialised(silero_path: str) -> None:
    cfg = _make_cfg(capture_system=True, capture_mic=True)
    factory = _Factory()
    pipe, _bus, _sub = _pipeline(cfg, factory)
    pipe.start()
    try:
        threads = [threading.Thread(target=pipe.restart_source, args=(s,)) for s in (Source.SYSTEM, Source.MIC) * 3]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=_WAIT_S)
        assert pipe.devices() == {Source.SYSTEM: "default", Source.MIC: "default"}
        # every capture but the two live ones was stopped exactly once, and only the live ones run
        live = {id(c) for c in factory.made if c.running}
        assert len(live) == 2
        assert all(c.stops == 1 for c in factory.made if id(c) not in live)
    finally:
        pipe.stop()

"""Tests for the engine side of Settings > Audio (2026-10-29): ``Engine.apply_settings`` reopening the
affected capture off the UI thread when an ``[audio]`` device / keep-alive / on-off value changes, and
``Engine.audio_status`` for the page's status line. A fake pipeline stands in for ``SpeechPipeline``
(the pipeline's own behaviour is tests/test_pipeline_restart.py's business)."""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import pytest

from exocortex import app as app_mod
from exocortex.bus import EventBus, Source
from tests.test_app_capabilities import make_cfg

_WAIT_S = 3.0


class _FakePipeline:
    def __init__(self, *, fail: bool = False) -> None:
        self.restarts: list[Source] = []
        self.calls = threading.Event()
        self.fail = fail
        self.running = True
        self._devices = {Source.SYSTEM: "Odyssey G95SD [Loopback]", Source.MIC: "Yeti"}
        self.threads: list[str] = []

    def restart_source(self, source: Source) -> str | None:
        self.threads.append(threading.current_thread().name)
        self.restarts.append(source)
        self.calls.set()
        if self.fail:
            raise RuntimeError("boom")
        return self._devices.get(source)

    def devices(self) -> dict[Source, str]:
        return dict(self._devices)

    def set_mic_muted(self, muted: bool) -> None:
        pass


@pytest.fixture()
def eng(tmp_path: Path) -> app_mod.Engine:
    e = app_mod.Engine(make_cfg(tmp_path), EventBus())
    e.pipeline = _FakePipeline()
    return e


def _wait_restarts(pipe: _FakePipeline, n: int) -> None:
    deadline = threading.Event()
    for _ in range(int(_WAIT_S / 0.02)):
        if len(pipe.restarts) >= n:
            return
        deadline.wait(0.02)
    raise AssertionError(f"expected {n} restart(s), got {pipe.restarts}")


@pytest.mark.parametrize("key, source", [
    ("audio.system_device", Source.SYSTEM),
    ("audio.capture_system", Source.SYSTEM),
    ("audio.loopback_keepalive", Source.SYSTEM),
    ("audio.mic_device", Source.MIC),
    ("audio.capture_mic", Source.MIC),
])
def test_apply_settings_reopens_the_affected_capture_off_the_calling_thread(
    eng: app_mod.Engine, key: str, source: Source
) -> None:
    pipe: Any = eng.pipeline
    eng.apply_settings({key: "x" if key.endswith("_device") else True})
    _wait_restarts(pipe, 1)
    assert pipe.restarts == [source]
    assert pipe.threads == ["audio-restart"]                         # never the caller's thread
    assert eng.cfg.path is not None and eng.cfg.path.is_file()         # and the file was saved as usual


def test_apply_settings_reopens_each_source_once_for_several_keys(eng: app_mod.Engine) -> None:
    pipe: Any = eng.pipeline
    eng.apply_settings({"audio.system_device": "a", "audio.loopback_keepalive": False, "audio.mic_device": "b",
                        "audio.capture_mic": True})
    _wait_restarts(pipe, 2)
    assert sorted(pipe.restarts, key=lambda s: s.value) == [Source.MIC, Source.SYSTEM]


def test_apply_settings_leaves_the_captures_alone_for_other_audio_keys(eng: app_mod.Engine) -> None:
    pipe: Any = eng.pipeline
    eng.apply_settings({"audio.mic_start_muted": True, "audio.echo_suppression": False, "audio.vad_threshold": 0.6,
                        "screen.interval_s": 3.0})
    assert not pipe.calls.wait(0.3)
    assert pipe.restarts == []


def test_apply_settings_survives_a_failing_restart(eng: app_mod.Engine) -> None:
    pipe = _FakePipeline(fail=True)
    eng.pipeline = pipe
    eng.apply_settings({"audio.mic_device": "x"})                      # must not raise, must still save
    _wait_restarts(pipe, 1)
    assert eng.cfg.path is not None and eng.cfg.path.is_file()


def test_apply_settings_does_nothing_before_the_pipeline_exists(tmp_path: Path) -> None:
    e = app_mod.Engine(make_cfg(tmp_path), EventBus())
    assert e.pipeline is None
    e.apply_settings({"audio.system_device": "x"})                     # start() reads the new value later
    assert e.cfg.audio.system_device == "default"                      # the dialog writes cfg, not apply_settings


def test_audio_status_reports_what_is_open(eng: app_mod.Engine) -> None:
    eng.cfg.audio.capture_mic = False
    eng.cfg.audio.system_device = "Odyssey"
    status = eng.audio_status()
    assert status == {
        "running": True,
        "system": {"enabled": True, "spec": "Odyssey", "device": "Odyssey G95SD [Loopback]"},
        "mic": {"enabled": False, "spec": "default", "device": "Yeti"},
    }


def test_audio_status_without_a_pipeline_or_with_a_broken_one(tmp_path: Path) -> None:
    e = app_mod.Engine(make_cfg(tmp_path), EventBus())
    status = e.audio_status()
    assert status["running"] is False
    assert status["system"]["device"] == "" and status["mic"]["device"] == ""

    class _Broken:
        def devices(self) -> dict[Source, str]:
            raise RuntimeError("no")

    e.pipeline = _Broken()
    status = e.audio_status()
    assert status["running"] is False and status["system"]["device"] == ""

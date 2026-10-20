"""Tests for exocortex/gpu.py (NVML-free: the reader is injected or monkeypatched) and the
Engine's gpu-guard wiring. The guard's job: a sustained free-VRAM collapse -- the signature of a
second large model landing on the display GPU mid-meeting -- must surface as a status event
within seconds, and must never fire on a single dip, on missing readings, or twice in a row
inside the cooldown."""
from __future__ import annotations

import time
from typing import Any

import pytest

from exocortex import gpu
from exocortex.gpu import RECOVERY_MARGIN_MB, VramGuard


class Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


def scripted(values: list[Any]):
    """A reader that returns the scripted values, then repeats the last one forever."""
    it = list(values)

    def read() -> int | None:
        if len(it) > 1:
            return it.pop(0)
        return it[0]

    return read


# ------------------------------------------------------------------------------ VramGuard ---
def test_single_dip_never_alerts() -> None:
    g = VramGuard(scripted([400, 5000]), 1500, consecutive=2, clock=Clock())
    assert g.sample() is None          # first low sample: streak 1
    assert g.sample() is None          # recovered before the streak completed


def test_sustained_dip_alerts_once_with_the_reading_in_the_text() -> None:
    g = VramGuard(scripted([400, 300, 250]), 1500, consecutive=2, clock=Clock())
    assert g.sample() is None
    alert = g.sample()
    assert alert is not None and "GPU memory low: 300 MiB free" in alert
    assert g.sample() is None          # cooldown: no immediate repeat


def test_cooldown_allows_a_repeat_alert_while_pressure_persists() -> None:
    clock = Clock()
    g = VramGuard(scripted([100]), 1500, consecutive=1, cooldown_s=180, clock=clock)
    assert g.sample() is not None
    clock.t += 179
    assert g.sample() is None
    clock.t += 2
    assert g.sample() is not None      # 181 s after the first: alert again


def test_recovery_message_once_and_only_above_the_margin() -> None:
    g = VramGuard(scripted([100, 100, 1600, 1500 + RECOVERY_MARGIN_MB, 9000]), 1500,
                  consecutive=2, clock=Clock())
    g.sample()
    assert g.sample() is not None      # alerted
    assert g.sample() is None          # 1600 is above the threshold but inside the margin: no flap
    recovered = g.sample()
    assert recovered is not None and "GPU memory recovered" in recovered
    assert g.sample() is None          # recovery is announced once


def test_none_readings_reset_the_streak_and_never_alert() -> None:
    g = VramGuard(scripted([400, None, 400, None]), 1500, consecutive=2, clock=Clock())
    assert g.sample() is None
    assert g.sample() is None          # None: streak reset
    assert g.sample() is None          # streak restarts at 1
    assert g.sample() is None


def test_a_raising_reader_is_treated_as_no_reading() -> None:
    def bad() -> int | None:
        raise RuntimeError("driver reset")

    g = VramGuard(bad, 1500, consecutive=1, clock=Clock())
    assert g.sample() is None


# --------------------------------------------------------------------------- vram_free_mb ---
# NVML-side behaviour only: no non-NVIDIA primary is offered, so the PDH fall-through added for
# other vendors is never reached here (that ladder has its own tests in test_gpu_detect.py).
def test_vram_free_mb_is_none_when_nvml_cannot_load(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(gpu._state, "ok", None)
    monkeypatch.setitem(gpu._state, "lib", None)
    monkeypatch.setitem(gpu._state, "handle", None)
    monkeypatch.setattr(gpu, "primary_gpu", lambda: None)   # no fall-through target
    import ctypes

    def boom(*_a: Any, **_k: Any) -> Any:
        raise OSError("no nvml.dll")

    monkeypatch.setattr(ctypes, "WinDLL", boom, raising=False)
    assert gpu.vram_free_mb() is None
    assert gpu._state["ok"] is False   # parked off: later calls stay cheap
    assert gpu.vram_free_mb() is None


def test_vram_free_mb_parks_off_after_a_failed_reading(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeLib:
        def nvmlDeviceGetMemoryInfo(self, _h: Any, _m: Any) -> int:  # noqa: N802
            return 999                # NVML_ERROR_*: anything non-zero

    monkeypatch.setitem(gpu._state, "ok", True)
    monkeypatch.setitem(gpu._state, "lib", FakeLib())
    monkeypatch.setitem(gpu._state, "handle", object())
    monkeypatch.setattr(gpu, "primary_gpu", lambda: None)   # no fall-through target
    assert gpu.vram_free_mb() is None
    assert gpu._state["ok"] is False


# ------------------------------------------------------------------------- Engine wiring ---
from exocortex import app as app_mod  # noqa: E402
from exocortex.bus import ComponentStatusEvent, drain  # noqa: E402
from tests.test_app import engine_env  # noqa: E402,F401  (fixture reuse)


def _wait(pred: Any, timeout: float = 8.0) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.02)
    return False


def test_engine_publishes_the_alert_and_stops_the_guard(engine_env: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg, bus = engine_env
    monkeypatch.setattr(app_mod, "GPU_GUARD_INTERVAL_S", 0.03)
    monkeypatch.setattr(gpu, "vram_free_mb", lambda: 200)
    q = bus.subscribe()
    seen: list[str] = []
    eng = app_mod.Engine(cfg, bus)
    eng.start()
    try:
        def saw_alert() -> bool:
            seen.extend(e.detail for e in drain(q)
                        if isinstance(e, ComponentStatusEvent) and e.component == "app" and e.state == "error")
            return any("GPU memory low" in d for d in seen)

        assert eng._gpu_thread is not None
        assert _wait(saw_alert)
    finally:
        eng.stop()
    assert _wait(lambda: not eng._gpu_thread.is_alive(), 3.0)


def test_engine_guard_off_by_config(engine_env: Any) -> None:
    cfg, bus = engine_env
    cfg.llm.vram_guard = False
    eng = app_mod.Engine(cfg, bus)
    eng.start()
    try:
        assert eng._gpu_thread is None
    finally:
        eng.stop()


def test_engine_guard_exits_quietly_without_readings(engine_env: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg, bus = engine_env
    monkeypatch.setattr(gpu, "vram_free_mb", lambda: None)
    eng = app_mod.Engine(cfg, bus)
    eng.start()
    try:
        assert eng._gpu_thread is not None
        assert _wait(lambda: not eng._gpu_thread.is_alive(), 3.0)   # probe said None: thread ends
    finally:
        eng.stop()

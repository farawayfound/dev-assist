"""Offscreen tests for the Monitor section (View -> Monitor, 2026-10-07): the MonitorView widget
and the overlay wiring that feeds it performance, context and resource readouts.

Mirrors tests/test_overlay.py's fixtures; reuses the catch-all FakeActions."""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from exocortex.bus import (
    AssistDoneEvent,
    ComponentStatusEvent,
    ContextUsageEvent,
    EventBus,
)
from exocortex.config import Config
from exocortex.ui.overlay import OverlayWindow
from exocortex.ui.resources import ResourceSample
from exocortex.ui.widgets import MonitorView
from tests.test_overlay_completion import FakeActions

_WAIT_MS = 50


@pytest.fixture(scope="session")
def qapp() -> QApplication:
    return QApplication.instance() or QApplication(["exocortex-monitor-tests"])


@pytest.fixture()
def actions() -> FakeActions:
    return FakeActions()


@pytest.fixture()
def bus() -> EventBus:
    return EventBus()


@pytest.fixture()
def window(qapp: QApplication, bus: EventBus, actions: FakeActions):
    win = OverlayWindow(Config(), bus, actions)
    win.show()
    qapp.processEvents()
    QTest.qWait(30)
    yield win
    win.close_silently()
    qapp.processEvents()


# --------------------------------------------------- the MonitorView widget on its own ----
def test_monitorview_context_renders_used_and_slot(qapp: QApplication) -> None:
    m = MonitorView(Config())
    m.set_context({
        "slot_tokens": 4096, "system_tokens": 300, "transcript_tokens": 1200,
        "reference_tokens": 200, "lookup_tokens": 100, "earlier_tokens": 150,
        "screen_tokens": 80, "documents": 2, "trimmed": True,
        "earlier_summary": "earlier: the team picked Kafka.",
    })
    assert "4096" in m._ctx_label.text()
    assert "2030" in m._ctx_label.text()            # 300+1200+200+100+150+80
    assert "trimmed" in m._legend_label.text()
    assert "Kafka" in m._summary.toPlainText()


def test_monitorview_resources_handles_missing_fields(qapp: QApplication) -> None:
    m = MonitorView(Config())
    m.set_resources(ResourceSample())               # all None
    assert "—" in m._bars["cpu"].format()
    m.set_resources(ResourceSample(cpu_percent=42.0, ram_used_mb=8192, ram_total_mb=65536,
                                   vram_used_mb=3072, vram_total_mb=24576, vram_source="nvml",
                                   gpu_percent=30.0))
    assert "42" in m._bars["cpu"].format()
    assert "GPU" in m._bars["gpu"].format() and "24.0" in m._bars["gpu"].format()
    assert "RAM" in m._bars["ram"].format() and "64.0" in m._bars["ram"].format()
    # The dedicated GPU reported a reading, so its row is now shown.
    assert m._bars["gpu"].isVisibleTo(m)


def test_monitorview_optional_rows_hide_when_device_absent(qapp: QApplication) -> None:
    m = MonitorView(Config())
    m.show()
    # A sample with no iGPU / NPU readings keeps those rows hidden; CPU/RAM always show.
    m.set_resources(ResourceSample(cpu_percent=10.0, ram_used_mb=1, ram_total_mb=2))
    assert m._bars["cpu"].isVisibleTo(m) and m._bars["ram"].isVisibleTo(m)
    assert not m._bars["igpu"].isVisibleTo(m)
    assert not m._bars["npu"].isVisibleTo(m)
    # Once the iGPU reports (even 0%), its row appears.
    m.set_resources(ResourceSample(igpu_percent=0.0))
    assert m._bars["igpu"].isVisibleTo(m)


def test_monitorview_config_hides_sections(qapp: QApplication) -> None:
    cfg = Config()
    cfg.monitor.show_performance = False
    cfg.monitor.show_cpu = False
    m = MonitorView(cfg)
    m.show()
    assert not m._perf_box.isVisibleTo(m)
    assert not m._bars["cpu"].isVisibleTo(m)
    assert m._ctx_box.isVisibleTo(m)        # context still on by default


def test_monitorview_perf_and_llm_state(qapp: QApplication) -> None:
    m = MonitorView(Config())
    m.on_assist_done(0.35, 128, 48.0, cancelled=False)
    assert "tok/s" in m._perf_label.text() and "0.35" in m._perf_label.text()
    m.set_llm_state("ready", "gemma-3-12b")
    assert "gemma-3-12b" in m._model_label.text()
    # A cancelled job leaves the last good reading in place.
    m.on_assist_done(0.0, 0, 0.0, cancelled=True)
    assert "48" in m._perf_label.text()


def test_context_event_does_not_blank_the_summary(qapp: QApplication) -> None:
    m = MonitorView(Config())
    m.set_context({"slot_tokens": 100, "earlier_summary": "kept"})
    # A raw event-shaped dict (no earlier_summary key) must not wipe the running summary.
    m.set_context({"slot_tokens": 100, "transcript_tokens": 40})
    assert m._summary.toPlainText() == "kept"


# --------------------------------------------------- the overlay wiring -------------------
def test_llm_status_event_updates_the_monitor_model(window: OverlayWindow, bus: EventBus) -> None:
    window._set_view_option("monitor", True)
    bus.publish(ComponentStatusEvent(component="llm", state="ready", detail="my-model-q4"))
    QTest.qWait(_WAIT_MS)
    assert "my-model-q4" in window.monitor._model_label.text()


def test_assist_done_event_updates_the_monitor_perf(window: OverlayWindow, bus: EventBus) -> None:
    window._set_view_option("monitor", True)
    window._assist_job_kinds[7] = "solve"
    bus.publish(AssistDoneEvent(job_id=7, text="done", ttft_s=0.22, tokens=64, tok_per_s=55.0))
    QTest.qWait(_WAIT_MS)
    assert "tok/s" in window.monitor._perf_label.text()


def test_context_usage_event_updates_the_monitor(window: OverlayWindow, bus: EventBus) -> None:
    window._set_view_option("monitor", True)
    bus.publish(ContextUsageEvent(job_id=3, kind="solve", slot_tokens=4096,
                                  transcript_tokens=800, system_tokens=200))
    QTest.qWait(_WAIT_MS)
    assert "4096" in window.monitor._ctx_label.text()

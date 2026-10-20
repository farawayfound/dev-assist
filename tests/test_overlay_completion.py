"""Offscreen tests for completion mode in the overlay: the slim bar, the ghost window hookup,
the context menu and the geometry bookkeeping. Mirrors tests/test_overlay.py's fixtures."""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path
from typing import Any

import pytest
from PySide6.QtCore import QRect
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from exocortex.bus import (
    AssistDeltaEvent,
    AssistDoneEvent,
    AssistStartedEvent,
    CompletionEvent,
    CompletionModeEvent,
    EventBus,
)
from exocortex.config import Config
from exocortex.ui.overlay import OverlayWindow

_WAIT_MS = 80


class FakeActions:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)

        def f(*args: Any, **kw: Any) -> Any:
            self.calls.append((name, args))
            if name == "list_models":
                return []
            if name == "export_session":
                return Path(".")
            if name == "toggle_screen_paused":
                return False
            return None

        return f


@pytest.fixture(scope="session")
def qapp() -> QApplication:
    return QApplication.instance() or QApplication(["exocortex-tests"])


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


def test_completion_toggle_collapses_to_slim_bar_and_restores(window: OverlayWindow, actions: FakeActions) -> None:
    full = QRect(window.geometry())
    window.completion_btn.setChecked(True)
    QTest.qWait(_WAIT_MS)
    assert window.slim.isVisibleTo(window)
    assert window.stack.isHidden() and window.askbox.isHidden() and window.tabstrip.isHidden()
    assert window.height() < 80
    assert window.geometry().right() == full.right()          # stays parked at the same right edge
    assert ("set_completion_mode", (True,)) in actions.calls
    # the persisted geometry is the FULL one, never the slim bar's
    window._save_geometry()
    assert (window._cfg.ui.width, window._cfg.ui.height) == (full.width(), full.height())

    window.slim_full_btn.click()
    QTest.qWait(_WAIT_MS)
    assert window.slim.isHidden() and window.stack.isVisibleTo(window)
    assert window.geometry() == full
    assert ("set_completion_mode", (False,)) in actions.calls


def test_slim_pause_mirrors_main_pause(window: OverlayWindow, actions: FakeActions) -> None:
    window.completion_btn.setChecked(True)
    window.slim_pause_btn.setChecked(True)
    assert window.pause_btn.isChecked() and window.slim_pause_btn.text() == "Resume"
    assert ("set_paused", (True,)) in actions.calls
    window.pause_btn.setChecked(False)
    assert not window.slim_pause_btn.isChecked() and window.slim_pause_btn.text() == "Pause"


def test_completion_event_drives_ghost_only_in_completion_mode(window: OverlayWindow, bus: EventBus) -> None:
    ev = CompletionEvent(visible=True, text=" on the mat", x=500, y=400, line_h=23, font_px=13.3,
                         font_name="Segoe UI", current_line="the cat sat", app="Code.exe")
    bus.publish(ev)
    QTest.qWait(_WAIT_MS)
    assert window._ghost is None                       # mode off: nothing is drawn
    window.completion_btn.setChecked(True)
    bus.publish(ev)
    QTest.qWait(_WAIT_MS)
    assert window._ghost is not None and window._ghost.isVisible()
    assert window._ghost._font.pixelSize() == 13
    bus.publish(CompletionEvent(visible=False))
    QTest.qWait(_WAIT_MS)
    assert not window._ghost.isVisible()
    bus.publish(ev)
    QTest.qWait(_WAIT_MS)
    assert window._ghost.isVisible()
    window.completion_btn.setChecked(False)             # leaving completion mode hides it too
    assert not window._ghost.isVisible()


def test_context_menu_reflects_and_updates_sources(window: OverlayWindow, actions: FakeActions) -> None:
    window._cfg.completion.use_screen = False
    window._cfg.completion.hint = "terse"
    window._rebuild_context_menu()
    acts = {a.text(): a for a in window._context_menu.actions() if a.text()}
    assert acts["Transcript (what was said)"].isChecked()
    assert not acts["Screen text (OCR)"].isChecked()
    assert "Hint: terse" in acts
    acts["Screen text (OCR)"].setChecked(True)
    assert ("set_completion_context", ({"screen": True},)) in actions.calls
    acts["Clear hint"].trigger()
    assert ("set_completion_context", ({}, "")) in actions.calls


def test_complete_kind_events_never_touch_the_tabs(window: OverlayWindow, bus: EventBus) -> None:
    window._show_tab(0)
    bus.publish(AssistStartedEvent(job_id=9, kind="complete", question="Completion"))
    bus.publish(AssistDeltaEvent(job_id=9, delta="ghost"))
    bus.publish(AssistDoneEvent(job_id=9, text="ghost", ttft_s=0.1, tokens=1, tok_per_s=10.0))
    QTest.qWait(_WAIT_MS)
    assert window.stack.currentIndex() == 0
    assert "ghost" not in window.assist_view.answer_browser.toPlainText()


def test_completion_mode_event_syncs_button(window: OverlayWindow, bus: EventBus) -> None:
    bus.publish(CompletionModeEvent(enabled=True))
    QTest.qWait(_WAIT_MS)
    assert window.completion_btn.isChecked() and window.slim.isVisibleTo(window)


def test_starts_in_completion_mode_when_configured(qapp: QApplication, bus: EventBus, actions: FakeActions) -> None:
    cfg = Config()
    cfg.completion.enabled = True
    win = OverlayWindow(cfg, bus, actions)
    win.show()
    QTest.qWait(_WAIT_MS)
    try:
        assert win.completion_btn.isChecked() and win.slim.isVisibleTo(win) and win.height() < 80
    finally:
        win.close_silently()
        qapp.processEvents()

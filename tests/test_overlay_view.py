"""Offscreen tests for the "View" menu (2026-10-07): toggling which overlay strips show
(Tabs / Controls / Status / Monitor), persistence through the engine, and the interaction with
the hands-free view and completion mode. Mirrors tests/test_overlay.py's fixtures and reuses the
catch-all FakeActions from tests/test_overlay_completion.py."""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMenu

from exocortex.config import Config
from exocortex.ui.overlay import OverlayWindow
from tests.test_overlay_completion import FakeActions

_WAIT_MS = 50


@pytest.fixture(scope="session")
def qapp() -> QApplication:
    return QApplication.instance() or QApplication(["exocortex-view-tests"])


@pytest.fixture()
def actions() -> FakeActions:
    return FakeActions()


@pytest.fixture()
def window(qapp: QApplication, actions: FakeActions):
    from exocortex.bus import EventBus

    win = OverlayWindow(Config(), EventBus(), actions)
    win.show()
    qapp.processEvents()
    QTest.qWait(30)
    yield win
    win.close_silently()
    qapp.processEvents()


# --------------------------------------------------------------------- defaults ----
def test_defaults_show_tabs_controls_status_and_hide_monitor(window: OverlayWindow) -> None:
    assert window.tabstrip.isVisibleTo(window)
    assert window.control_row.isVisibleTo(window)
    assert window.status.isVisibleTo(window)
    assert not window.monitor.isVisibleTo(window)   # [ui].show_monitor defaults off


# --------------------------------------------------------------------- toggles ------
def test_toggling_each_strip_hides_it_and_persists(window: OverlayWindow, actions: FakeActions) -> None:
    window._set_view_option("status", False)
    assert not window.status.isVisibleTo(window)
    assert not window._cfg.ui.show_status
    assert ("apply_settings", ({"ui.show_status": False},)) in actions.calls

    window._set_view_option("controls", False)
    assert not window.control_row.isVisibleTo(window)

    window._set_view_option("monitor", True)
    assert window.monitor.isVisibleTo(window)
    assert window._cfg.ui.show_monitor is True

    # Turning them back on restores each strip.
    window._set_view_option("status", True)
    window._set_view_option("controls", True)
    assert window.status.isVisibleTo(window) and window.control_row.isVisibleTo(window)


def test_hiding_tabs_keeps_the_current_tab_content(window: OverlayWindow) -> None:
    window._show_tab(2)                     # Notes
    assert window.stack.currentIndex() == 2
    window._set_view_option("tabs", False)
    assert not window.tabstrip.isVisibleTo(window)
    # The content stack stays on the last selected tab, still visible.
    assert window.stack.isVisibleTo(window)
    assert window.stack.currentIndex() == 2


# --------------------------------------------------------------------- menu ---------
def test_view_submenu_has_the_four_strips_reflecting_config(window: OverlayWindow) -> None:
    window._cfg.ui.show_monitor = True
    menu = QMenu(window)
    window._build_view_submenu(menu)
    by_text = {a.text(): a for a in menu.actions()}
    assert set(by_text) == {"Tabs", "Controls", "Status", "Monitor"}
    assert all(a.isCheckable() for a in by_text.values())
    assert by_text["Monitor"].isChecked() is True
    assert by_text["Status"].isChecked() is True


def test_main_menu_carries_view_submenu_and_compact(window: OverlayWindow) -> None:
    window._rebuild_menu()
    texts = [a.text() for a in window._menu.actions()]
    assert "View" in texts
    assert "Compact context now" in texts
    view_action = next(a for a in window._menu.actions() if a.text() == "View")
    assert view_action.menu() is not None


# --------------------------------------------------------------- hands-free wins ----
def test_hands_free_overrides_view_then_restores_it(window: OverlayWindow) -> None:
    window._set_view_option("monitor", True)
    window._set_view_option("status", True)
    window._set_hands_free(True, notify_engine=False)
    QTest.qWait(_WAIT_MS)
    # Hands-free shows only its output + title + the switches: controls, status and monitor go,
    # and the tab strip stays (its switches are the way out) but the tab buttons do not.
    assert not window.control_row.isVisibleTo(window)
    assert not window.status.isVisibleTo(window)
    assert not window.monitor.isVisibleTo(window)
    assert window.tabstrip.isVisibleTo(window)
    assert not window._tab_buttons[0].isVisibleTo(window)

    window._set_hands_free(False, notify_engine=False)
    QTest.qWait(_WAIT_MS)
    # The View preferences come back.
    assert window.status.isVisibleTo(window)
    assert window.monitor.isVisibleTo(window)
    assert window._tab_buttons[0].isVisibleTo(window)


def test_completion_mode_hides_the_monitor_and_restores_the_view(window: OverlayWindow) -> None:
    window._set_view_option("monitor", True)
    assert window.monitor.isVisibleTo(window)
    window.completion_btn.setChecked(True)          # enter completion mode
    QTest.qWait(_WAIT_MS)
    assert not window.monitor.isVisibleTo(window)
    window.completion_btn.setChecked(False)         # back to the full overlay
    QTest.qWait(_WAIT_MS)
    assert window.monitor.isVisibleTo(window)       # the View pref is honoured again

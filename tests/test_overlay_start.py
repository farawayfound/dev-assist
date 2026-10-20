"""Offscreen tests: the overlay opens on the Assist tab on every launch -- not Live -- including when
it starts hidden, in hands-free mode, or in completion mode's slim bar. Reuses
tests/test_overlay_completion.py's catch-all FakeActions."""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from exocortex.bus import EventBus, HandsFreeEvent
from exocortex.config import Config
from exocortex.ui import overlay as overlay_mod
from exocortex.ui.overlay import OverlayWindow
from tests.test_overlay_completion import FakeActions

_WAIT_MS = 80


@pytest.fixture(scope="session")
def qapp() -> QApplication:
    return QApplication.instance() or QApplication(["exocortex-start-tab-tests"])


def _open(qapp: QApplication, cfg: Config | None = None) -> tuple[OverlayWindow, EventBus]:
    bus = EventBus()
    win = OverlayWindow(cfg or Config(), bus, FakeActions())
    win.show()
    qapp.processEvents()
    QTest.qWait(30)
    return win, bus


def _close(qapp: QApplication, win: OverlayWindow) -> None:
    win.close_silently()
    qapp.processEvents()


def test_the_start_tab_is_assist() -> None:
    assert overlay_mod._START_TAB == 1


def test_opens_on_the_assist_tab(qapp: QApplication) -> None:
    win, _bus = _open(qapp)
    try:
        assert win.stack.currentIndex() == 1
        assert win.stack.currentWidget() is win.assist_view
        assert [b.text() for b in win._tab_buttons if b.isChecked()] == ["Assist"]
        # The hands-free switch is only usable from the Assist tab: now it is, from the first frame.
        assert win.handsfree_switch.isEnabled()
    finally:
        _close(qapp, win)


def test_the_tab_is_chosen_at_construction_not_on_show(qapp: QApplication) -> None:
    cfg = Config()
    cfg.ui.start_hidden = True
    win = OverlayWindow(cfg, EventBus(), FakeActions())
    try:
        assert win.stack.currentIndex() == 1
    finally:
        _close(qapp, win)


def test_every_launch_opens_on_assist_whatever_tab_the_last_one_ended_on(qapp: QApplication) -> None:
    first, _bus = _open(qapp)
    first._show_tab(2)                      # left on Notes
    _close(qapp, first)
    second, _bus = _open(qapp)
    try:
        assert second.stack.currentIndex() == 1
    finally:
        _close(qapp, second)


def test_hands_free_at_start_stays_on_assist(qapp: QApplication) -> None:
    """With `[handsfree].enabled` saved on, the engine switches it on at start and the overlay
    mirrors that event: its view is the Assist tab's output."""
    win, bus = _open(qapp)
    try:
        bus.publish(HandsFreeEvent(enabled=True, state="watching"))
        QTest.qWait(_WAIT_MS)
        assert win._hands_free
        assert win.stack.currentIndex() == 1
    finally:
        _close(qapp, win)


def test_leaving_the_slim_bar_after_a_completion_mode_start_lands_on_assist(qapp: QApplication) -> None:
    cfg = Config()
    cfg.completion.enabled = True
    win, _bus = _open(qapp, cfg)
    try:
        QTest.qWait(_WAIT_MS)
        assert win._completion_mode             # started in the slim bar
        win.completion_btn.setChecked(False)
        QTest.qWait(_WAIT_MS)
        assert not win._completion_mode
        assert win.stack.currentIndex() == 1
    finally:
        _close(qapp, win)

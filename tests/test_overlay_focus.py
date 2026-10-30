"""Offscreen tests for the focus rules (2026-10-29): the shortcuts and the clicks on the panel leave
the keyboard with the app the user is working in.

The overlay is shown without activating; `focus_ask` is the one shortcut that activates the window
(it exists to type into the ask box); every other trigger-table callback leaves `activateWindow`
uncalled. Clicks: `_click_activates` answers Windows' WM_MOUSEACTIVATE question per widget class (a
button or switch never takes the keyboard; the ask box, the text panes, the drag handles, the
sliders and a menu button do; a right click always, a middle / thumb click never; the hands-free
view's click-through content never), and `_mouse_activate_answer` / `nativeEvent` decode the native
message. Mirrors tests/test_overlay_manual.py's fixtures (spy hooks: nothing global is ever touched).
"""
from __future__ import annotations

import ctypes
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from ctypes import wintypes

import pytest
from PySide6.QtCore import QByteArray, QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QAbstractButton, QApplication, QWidget
from shiboken6 import VoidPtr

from exocortex.bus import EventBus
from exocortex.config import Config
from exocortex.ui import overlay as overlay_mod
from exocortex.ui.overlay import OverlayWindow
from tests.test_overlay import _SpyHotkeys
from tests.test_overlay_completion import FakeActions
from tests.test_overlay_manual import SpyPressHoldHook, SpyTapHook

WM_MOUSEACTIVATE = 0x0021
WM_PAINT = 0x000F
WM_LBUTTONDOWN, WM_LBUTTONDBLCLK = 0x0201, 0x0203
WM_RBUTTONDOWN, WM_MBUTTONDOWN, WM_XBUTTONDOWN = 0x0204, 0x0207, 0x020B
HTCLIENT = 1
MA_NOACTIVATE = 3


@pytest.fixture(scope="session")
def qapp() -> QApplication:
    return QApplication.instance() or QApplication(["exocortex-focus-tests"])


@pytest.fixture()
def actions() -> FakeActions:
    return FakeActions()


@pytest.fixture()
def bus() -> EventBus:
    return EventBus()


@pytest.fixture()
def window(qapp: QApplication, bus: EventBus, actions: FakeActions):
    win = OverlayWindow(Config(), bus, actions)
    win.resize(640, 520)
    win.show()
    qapp.processEvents()
    QTest.qWait(30)
    yield win
    win.close_silently()
    qapp.processEvents()


def _center(w: QWidget) -> QPoint:
    """A widget's centre in global (screen) coordinates -- what WM_MOUSEACTIVATE's cursor would be."""
    return w.mapToGlobal(w.rect().center())


def _msg(message: int, mouse_msg: int = WM_LBUTTONDOWN) -> wintypes.MSG:
    msg = wintypes.MSG()
    msg.message = message
    msg.lParam = (mouse_msg << 16) | HTCLIENT
    return msg


# =====================================================================================
# the window itself
# =====================================================================================
def test_overlay_is_shown_without_activating(window: OverlayWindow) -> None:
    assert window.testAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
    assert window.windowFlags() & Qt.WindowType.Tool          # Windows shows a tool window SW_SHOWNOACTIVATE


def test_focus_ask_is_the_only_shortcut_that_activates_the_window(
    window: OverlayWindow, monkeypatch: pytest.MonkeyPatch, qapp: QApplication
) -> None:
    """Every trigger-table callback but ask_focus leaves `activateWindow` uncalled (quit would close
    the window, select_region is the picker's own business -- see tests/test_region_picker.py)."""
    monkeypatch.setattr(overlay_mod.RegionPicker, "begin", lambda self: None)
    activations: list[str] = []
    window.activateWindow = lambda: activations.append("activate")   # type: ignore[method-assign]
    skipped = {"quit", "ask_focus", "toggle_visible"}
    ran: list[str] = []
    for name, _binding, callback, _manual_only in window._trigger_table():
        if name in skipped:
            continue
        callback()
        qapp.processEvents()
        ran.append(name)
    assert len(ran) >= 20
    assert activations == []
    window.toggle_visible()
    window.toggle_visible()
    qapp.processEvents()
    assert window.isVisible() and activations == []
    window.focus_ask()
    assert activations == ["activate"]


# =====================================================================================
# the click policy
# =====================================================================================
def test_buttons_and_switches_do_not_take_the_focus(window: OverlayWindow) -> None:
    buttons = [
        w for w in (
            window.pause_btn, window.auto_btn, window.mic_btn, window.hide_btn, window.clickthrough_btn,
            window.completion_btn, window.manual_switch, window.handsfree_switch, window.askbox.send_btn,
        ) if w.isVisible()
    ]
    assert len(buttons) >= 6
    for w in buttons:
        assert isinstance(w, QAbstractButton)
        assert window.childAt(window.mapFromGlobal(_center(w))) is w, w
        assert window._click_activates(WM_LBUTTONDOWN, _center(w)) is False, w
        assert window._click_activates(WM_LBUTTONDBLCLK, _center(w)) is False, w


def test_a_button_that_opens_a_menu_takes_the_focus(window: OverlayWindow) -> None:
    """A popup menu needs its window to be the foreground one (only the foreground thread can grab
    the mouse, which is how an outside click closes the menu)."""
    assert window.menu_btn.menu() is not None
    assert window._click_activates(WM_LBUTTONDOWN, _center(window.menu_btn)) is True


def test_right_click_takes_the_focus_everywhere(window: OverlayWindow) -> None:
    for w in (window.pause_btn, window.manual_switch, window.askbox.edit, window.assist_view.answer_browser):
        assert window._click_activates(WM_RBUTTONDOWN, _center(w)) is True, w


def test_middle_and_thumb_clicks_never_take_the_focus(window: OverlayWindow) -> None:
    for w in (window.askbox.edit, window.assist_view.answer_browser.viewport(), window.title_label):
        for msg in (WM_MBUTTONDOWN, WM_XBUTTONDOWN):
            assert window._click_activates(msg, _center(w)) is False, (w, hex(msg))


def test_text_panes_inputs_handles_and_sliders_take_the_focus(window: OverlayWindow) -> None:
    takes = [window.askbox.edit, window.assist_view.answer_browser.viewport(), window.title_label,
             window.opacity_slider]
    takes += [g for g in window._edge_grips if g.isVisible()][:1]
    if window.size_grip.isVisible():
        takes.append(window.size_grip)
    assert len(takes) >= 5
    for w in takes:
        assert w.isVisible(), w
        assert window._click_activates(WM_LBUTTONDOWN, _center(w)) is True, w


def test_hands_free_content_click_never_takes_the_focus(window: OverlayWindow, qapp: QApplication) -> None:
    """In the hands-free view the content is click-through from the cursor position, re-evaluated
    every poll: a click that lands on it raced that flag and was meant for the app underneath."""
    window._set_hands_free(True, notify_engine=False)
    qapp.processEvents()
    QTest.qWait(40)
    pane = window.assist_view.answer_browser.viewport()
    assert pane.isVisible()
    assert window._click_activates(WM_LBUTTONDOWN, _center(pane)) is False
    assert window._click_activates(WM_RBUTTONDOWN, _center(pane)) is True       # the text menu
    assert window._click_activates(WM_LBUTTONDOWN, _center(window.title_label)) is True    # dragging
    assert window._click_activates(WM_LBUTTONDOWN, _center(window.manual_switch)) is False
    window._set_hands_free(False, notify_engine=False)
    qapp.processEvents()
    QTest.qWait(40)
    assert window._click_activates(WM_LBUTTONDOWN, _center(pane)) is True


def test_click_outside_every_child_does_not_take_the_focus(window: OverlayWindow) -> None:
    """A point on the window with no child under it (the margin): nothing there needs the keyboard."""
    assert window._click_activates(WM_LBUTTONDOWN, _center(window) + QPoint(100000, 100000)) is False


# =====================================================================================
# the native message
# =====================================================================================
def test_mouse_activate_answer_decodes_the_native_message(window: OverlayWindow) -> None:
    refused = _msg(WM_MOUSEACTIVATE, WM_LBUTTONDOWN)
    assert window._mouse_activate_answer(ctypes.addressof(refused), _center(window.pause_btn)) == MA_NOACTIVATE
    assert window._mouse_activate_answer(ctypes.addressof(refused), _center(window.askbox.edit)) is None
    right = _msg(WM_MOUSEACTIVATE, WM_RBUTTONDOWN)
    assert window._mouse_activate_answer(ctypes.addressof(right), _center(window.pause_btn)) is None
    other = _msg(WM_PAINT)
    assert window._mouse_activate_answer(ctypes.addressof(other), _center(window.pause_btn)) is None
    assert window._mouse_activate_answer(0) is None


def test_native_event_answers_ma_noactivate_and_leaves_the_rest_to_qt(
    window: OverlayWindow, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = _center(window.pause_btn)

    class _Cursor:
        @staticmethod
        def pos() -> QPoint:
            return target

    monkeypatch.setattr(overlay_mod, "QCursor", _Cursor)
    refused = _msg(WM_MOUSEACTIVATE, WM_LBUTTONDOWN)
    handled, result = window.nativeEvent(QByteArray(b"windows_generic_MSG"), VoidPtr(ctypes.addressof(refused)))
    assert (handled, result) == (True, MA_NOACTIVATE)
    other = _msg(WM_PAINT)
    assert tuple(window.nativeEvent(QByteArray(b"windows_generic_MSG"), VoidPtr(ctypes.addressof(other)))) == (False, 0)
    assert tuple(window.nativeEvent(QByteArray(b"xcb_generic_event_t"), VoidPtr(ctypes.addressof(refused)))) == (False, 0)


def test_spy_hooks_are_never_the_real_ones(monkeypatch: pytest.MonkeyPatch) -> None:
    """Guard for this file: the overlay fixtures above never install a real hook or hotkey under the
    offscreen platform, so no spy is needed -- but if the platform check were ever faked here, the
    spies from test_overlay_manual would be the ones to use."""
    assert SpyPressHoldHook is not None and SpyTapHook is not None and _SpyHotkeys is not None
    assert not overlay_mod._is_windows_platform()

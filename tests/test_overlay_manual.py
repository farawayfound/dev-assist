"""Offscreen tests for the manual-advance UI: the Manual switch in the tab strip (next to the
hands-free pair, usable from every tab), `_set_manual`'s engine notification and hook lifecycle
(through a spy PressHoldMouseHook -- no real Win32 hook is ever installed), the `ManualAdvanceEvent`
bus sync, the `ctrl+alt+j` hotkey binding, the slim bar's Context-menu entry, and `_manual_fire`'s
routing (including the own-window guard and that it never shows / raises / activates the window).
Mirrors tests/test_overlay_handsfree.py's fixtures; reuses tests/test_overlay.py's `_SpyHotkeys`
and tests/test_overlay_completion.py's `FakeActions`.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from typing import Any, Callable

import pytest
from PySide6.QtCore import QPoint
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from exocortex.bus import EventBus, ManualAdvanceEvent
from exocortex.config import Config
from exocortex.ui import overlay as overlay_mod
from exocortex.ui.overlay import OverlayWindow
from exocortex.ui.widgets import ToggleSwitch
from tests.test_overlay import _SpyHotkeys
from tests.test_overlay_completion import FakeActions

_WAIT_MS = 80


class SpyPressHoldHook:
    """Stand-in for llhooks.PressHoldMouseHook: records install/uninstall, hooks nothing."""

    instances: list["SpyPressHoldHook"] = []

    def __init__(self, bindings: dict[str, Callable[[], None]], **kw: Any) -> None:
        self.bindings = dict(bindings)
        self.kw = kw
        self.installed = False
        self.failed = False
        self.installs = 0
        self.uninstalls = 0
        SpyPressHoldHook.instances.append(self)

    def install(self) -> bool:
        self.installed = True
        self.installs += 1
        return True

    def uninstall(self) -> None:
        self.installed = False
        self.uninstalls += 1


@pytest.fixture(scope="session")
def qapp() -> QApplication:
    return QApplication.instance() or QApplication(["exocortex-manual-tests"])


@pytest.fixture()
def actions() -> FakeActions:
    return FakeActions()


@pytest.fixture()
def bus() -> EventBus:
    return EventBus()


@pytest.fixture()
def spy_hook(monkeypatch: pytest.MonkeyPatch):
    """The overlay believes it is on the real Windows platform, but the press-and-hold hook and
    the RegisterHotKey table are both spies -- nothing global is ever touched."""
    monkeypatch.setattr(overlay_mod, "_is_windows_platform", lambda: True)
    monkeypatch.setattr(overlay_mod, "PressHoldMouseHook", SpyPressHoldHook)
    monkeypatch.setattr(overlay_mod, "GlobalHotkeys", _SpyHotkeys)
    SpyPressHoldHook.instances = []
    _SpyHotkeys.instances = []
    return SpyPressHoldHook


@pytest.fixture()
def window(qapp: QApplication, bus: EventBus, actions: FakeActions):
    win = OverlayWindow(Config(), bus, actions)
    win.show()
    qapp.processEvents()
    QTest.qWait(30)
    yield win
    win.close_silently()
    qapp.processEvents()


def _make_window(qapp: QApplication, bus: EventBus, actions: FakeActions, cfg: Config | None = None):
    win = OverlayWindow(cfg or Config(), bus, actions)
    return win


# =====================================================================================
# the switch: placement, availability, engine notification
# =====================================================================================


def test_manual_switch_sits_next_to_the_handsfree_switch(window: OverlayWindow) -> None:
    assert isinstance(window.manual_switch, ToggleSwitch)
    assert window.manual_label.parent() is window.tabstrip
    assert not window.manual_switch.isChecked()   # cfg default: Auto


def test_manual_switch_is_enabled_on_every_tab(window: OverlayWindow) -> None:
    for tab in range(4):
        window._show_tab(tab)
        assert window.manual_switch.isEnabled(), f"tab {tab}"


def test_toggling_the_switch_notifies_the_engine_and_toasts(window: OverlayWindow, actions: FakeActions) -> None:
    window.manual_switch.setChecked(True)
    assert ("set_manual_advance", (True,)) in actions.calls
    assert window._manual is True
    assert "Manual advance ON" in window.toast.text()
    window.manual_switch.setChecked(False)
    assert ("set_manual_advance", (False,)) in actions.calls
    assert window._manual is False


def test_manual_switch_stays_visible_and_clickable_in_the_hands_free_view(window: OverlayWindow) -> None:
    window._show_tab(1)
    window._set_hands_free(True, notify_engine=False)
    assert window.manual_switch.isVisible()
    pos = window.manual_switch.mapToGlobal(QPoint(2, 2))
    assert window._hands_free_clickable_at(pos) is True


def test_manual_advance_event_syncs_the_switch_without_calling_back(
    qapp: QApplication, window: OverlayWindow, actions: FakeActions, bus: EventBus
) -> None:
    bus.publish(ManualAdvanceEvent(enabled=True))
    QTest.qWait(_WAIT_MS)
    qapp.processEvents()
    assert window._manual is True
    assert window.manual_switch.isChecked()
    assert ("set_manual_advance", (True,)) not in actions.calls   # mirror only, no echo


# =====================================================================================
# initial state from config
# =====================================================================================


def test_starts_checked_and_hooked_when_config_says_manual(
    qapp: QApplication, bus: EventBus, actions: FakeActions, spy_hook
) -> None:
    cfg = Config()
    cfg.handsfree.manual = True
    win = _make_window(qapp, bus, actions, cfg)
    try:
        assert win.manual_switch.isChecked()
        assert win._manual is True
        assert ("set_manual_advance", (True,)) not in actions.calls   # mirrored, not re-sent
        assert spy_hook.instances and spy_hook.instances[-1].installed
    finally:
        win.close_silently()
        qapp.processEvents()


def test_no_hook_under_the_offscreen_platform(window: OverlayWindow) -> None:
    window._set_manual(True)
    assert window._manual_hook is None            # _is_windows_platform() is False offscreen


# =====================================================================================
# hook lifecycle: toggle on/off, close
# =====================================================================================


def test_hook_installed_on_manual_and_removed_on_auto_and_close(
    qapp: QApplication, bus: EventBus, actions: FakeActions, spy_hook
) -> None:
    win = _make_window(qapp, bus, actions)
    try:
        assert not spy_hook.instances             # Auto at start: no hook object at all
        win._set_manual(True)
        hook = spy_hook.instances[-1]
        assert hook.installed
        assert set(hook.bindings) == {"left", "middle"}
        assert callable(hook.kw["hold_s"]) and callable(hook.kw["require_hold"])
        win._set_manual(False)
        assert not hook.installed and hook.uninstalls == 1
        win._set_manual(True)
        assert hook.installed                     # the same object is re-installed
    finally:
        win.close_silently()
        qapp.processEvents()
    assert not spy_hook.instances[-1].installed   # closeEvent uninstalled it


def test_hold_s_and_require_hold_callables_read_the_live_config(
    qapp: QApplication, bus: EventBus, actions: FakeActions, spy_hook
) -> None:
    cfg = Config()
    win = _make_window(qapp, bus, actions, cfg)
    try:
        win._set_manual(True)
        hook = spy_hook.instances[-1]
        assert hook.kw["hold_s"]() == pytest.approx(0.8)
        cfg.handsfree.manual_hold_s = 1.4          # a settings change: no re-install needed
        assert hook.kw["hold_s"]() == pytest.approx(1.4)
        require_hold = hook.kw["require_hold"]
        assert require_hold("left") is True and require_hold("middle") is False   # the defaults
        cfg.handsfree.manual_hold_text = False
        cfg.handsfree.manual_hold_code = True
        assert require_hold("left") is False and require_hold("middle") is True
    finally:
        win.close_silently()
        qapp.processEvents()


# =====================================================================================
# the hotkey and the slim bar's menu entry
# =====================================================================================


def test_hotkey_binding_table_contains_toggle_manual(
    qapp: QApplication, bus: EventBus, actions: FakeActions, spy_hook
) -> None:
    cfg = Config()
    win = _make_window(qapp, bus, actions, cfg)
    try:
        bindings = _SpyHotkeys.instances[-1].bindings
        assert cfg.hotkeys.toggle_manual in bindings
        bindings[cfg.hotkeys.toggle_manual]()
        assert win._manual is True
        assert ("set_manual_advance", (True,)) in actions.calls
    finally:
        win.close_silently()
        qapp.processEvents()


def test_context_menu_manual_action_reflects_and_toggles_state(window: OverlayWindow, actions: FakeActions) -> None:
    window._rebuild_context_menu()
    acts = {a.text(): a for a in window._context_menu.actions() if a.text()}
    manual = next(a for text, a in acts.items() if "Manual advance" in text)
    assert manual.isCheckable() and not manual.isChecked()
    manual.setChecked(True)
    assert window._manual is True
    assert ("set_manual_advance", (True,)) in actions.calls
    window._rebuild_context_menu()
    acts = {a.text(): a for a in window._context_menu.actions() if a.text()}
    manual = next(a for text, a in acts.items() if "Manual advance" in text)
    assert manual.isChecked()


# =====================================================================================
# _manual_fire: routing, the own-window guard, no window state change
# =====================================================================================


def test_manual_fire_routes_to_actions_and_never_touches_the_window(
    window: OverlayWindow, actions: FakeActions, monkeypatch: pytest.MonkeyPatch
) -> None:
    window._set_manual(True)
    actions.calls.clear()
    far_away = QPoint(-5000, -5000)               # not over any of our windows
    monkeypatch.setattr(overlay_mod.QCursor, "pos", staticmethod(lambda: far_away))
    visible_before = window.isVisible()
    active_before = window.isActiveWindow()
    window._manual_fire("text")
    window._manual_fire("code")
    assert [c for c in actions.calls if c[0] == "manual_advance"] == [
        ("manual_advance", ("text",)), ("manual_advance", ("code",))]
    assert "Manual · Code" in window.toast.text()
    assert window.isVisible() == visible_before
    assert window.isActiveWindow() == active_before


def test_manual_fire_is_ignored_over_our_own_window(
    window: OverlayWindow, actions: FakeActions, monkeypatch: pytest.MonkeyPatch
) -> None:
    window._set_manual(True)
    actions.calls.clear()
    inside = window.frameGeometry().center()
    monkeypatch.setattr(overlay_mod.QCursor, "pos", staticmethod(lambda: inside))
    window._manual_fire("text")
    assert not any(c[0] == "manual_advance" for c in actions.calls)


def test_manual_fire_works_over_the_click_through_content_of_the_hands_free_view(
    qapp: QApplication, window: OverlayWindow, actions: FakeActions, monkeypatch: pytest.MonkeyPatch
) -> None:
    """In the hands-free view the content is click-through -- a hold over the answer text is a
    hold on the app underneath, and must fire; only the title bar and the switches block."""
    window._show_tab(1)
    window._set_hands_free(True, notify_engine=False)
    window._set_manual(True)
    qapp.processEvents()
    actions.calls.clear()
    content = window.stack.mapToGlobal(window.stack.rect().center())
    monkeypatch.setattr(overlay_mod.QCursor, "pos", staticmethod(lambda: content))
    window._manual_fire("text")
    assert ("manual_advance", ("text",)) in actions.calls
    actions.calls.clear()
    over_switch = window.manual_switch.mapToGlobal(QPoint(2, 2))
    monkeypatch.setattr(overlay_mod.QCursor, "pos", staticmethod(lambda: over_switch))
    window._manual_fire("text")                    # the switch itself stays clickable: blocked
    assert not any(c[0] == "manual_advance" for c in actions.calls)


def test_manual_fire_works_over_the_window_when_click_through_is_on(
    window: OverlayWindow, actions: FakeActions, monkeypatch: pytest.MonkeyPatch
) -> None:
    window._set_manual(True)
    window._click_through = True                   # the user's own click-through mode
    actions.calls.clear()
    inside = window.frameGeometry().center()
    monkeypatch.setattr(overlay_mod.QCursor, "pos", staticmethod(lambda: inside))
    window._manual_fire("code")
    assert ("manual_advance", ("code",)) in actions.calls


def test_manual_fire_is_ignored_once_manual_is_off(
    window: OverlayWindow, actions: FakeActions, monkeypatch: pytest.MonkeyPatch
) -> None:
    far_away = QPoint(-5000, -5000)
    monkeypatch.setattr(overlay_mod.QCursor, "pos", staticmethod(lambda: far_away))
    window._set_manual(True)
    window._set_manual(False)
    actions.calls.clear()
    window._manual_fire("text")                    # a stale deferred fire after toggling off
    assert not any(c[0] == "manual_advance" for c in actions.calls)

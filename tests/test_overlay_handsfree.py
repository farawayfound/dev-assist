"""Offscreen tests for hands-free mode in the overlay: the ToggleSwitch widget, the tab strip's
switch (enabled only from the Assist tab / while on), the output-only view it drives (both the
window's own chrome -- ``_apply_layout`` -- and ``AssistView.set_hands_free_view``), the title
bar subtitle, the slim bar's Context menu entry, the completion-mode round trip, the
``ctrl+alt+h`` hotkey and the cursor-tracking click-through. Mirrors tests/test_overlay.py's and
tests/test_overlay_completion.py's fixtures; reuses their ``FakeActions``, ``wait_until``,
``_StubWin32`` / ``fake_windows`` and ``_SpyHotkeys``.
"""
from __future__ import annotations

import dataclasses
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from typing import Any

import pytest
from PySide6.QtCore import QPoint, QSize
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (
    QAbstractButton,
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
)

from exocortex.bus import CheckpointEvent, EventBus, HandsFreeEvent
from exocortex.config import Config, HandsFreeConfig
from exocortex.ui import overlay as overlay_mod
from exocortex.ui.overlay import OverlayWindow
from exocortex.ui.settings import FIELDS, SettingsDialog
from exocortex.ui.widgets import ToggleSwitch
from tests.test_overlay import _SpyHotkeys, _StubWin32, fake_windows, wait_until
from tests.test_overlay_completion import FakeActions

_WAIT_MS = 80


@pytest.fixture(scope="session")
def qapp() -> QApplication:
    return QApplication.instance() or QApplication(["exocortex-handsfree-tests"])


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


# =====================================================================================
# ToggleSwitch: a checkable QAbstractButton with a fixed 36x18 size
# =====================================================================================
def test_toggle_switch_is_a_checkable_abstract_button_of_the_right_size(qapp: QApplication) -> None:
    sw = ToggleSwitch()
    assert isinstance(sw, QAbstractButton)
    assert sw.isCheckable()
    assert sw.sizeHint() == QSize(36, 18)
    assert sw.minimumSizeHint() == QSize(36, 18)


def test_toggle_switch_toggled_signal_fires_on_check_state_changes(qapp: QApplication) -> None:
    sw = ToggleSwitch()
    seen: list[bool] = []
    sw.toggled.connect(seen.append)
    sw.setChecked(True)
    sw.setChecked(True)   # no-op: no second signal
    sw.setChecked(False)
    assert seen == [True, False]


def test_toggle_switch_paints_without_error_in_every_combination(qapp: QApplication) -> None:
    sw = ToggleSwitch()
    sw.show()
    try:
        for checked in (False, True):
            sw.setChecked(checked)
            for enabled in (True, False):
                sw.setEnabled(enabled)
                sw.repaint()
                qapp.processEvents()
    finally:
        sw.close()


# =====================================================================================
# tab strip: after the stretch come the Manual pair, then the hands-free pair at the end
# =====================================================================================
def test_tabstrip_ends_with_stretch_then_manual_then_handsfree_pairs(window: OverlayWindow) -> None:
    layout = window.tabstrip.layout()
    assert layout is not None
    widgets: list[object | None] = []
    for i in range(layout.count()):
        item = layout.itemAt(i)
        widgets.append(item.widget())
    assert widgets[-1] is window.handsfree_switch
    assert widgets[-2] is window.handsfree_label
    assert widgets[-3] is window.manual_switch
    assert widgets[-4] is window.manual_label
    assert widgets[-5] is None   # the addStretch(1) item has no widget


def test_handsfree_switch_enabled_only_on_the_assist_tab(window: OverlayWindow) -> None:
    window._show_tab(0)   # Live
    assert not window.handsfree_switch.isEnabled()
    window._show_tab(1)   # Assist
    assert window.handsfree_switch.isEnabled()
    window._show_tab(2)   # Notes
    assert not window.handsfree_switch.isEnabled()
    window._show_tab(3)   # Screen
    assert not window.handsfree_switch.isEnabled()


def test_handsfree_switch_stays_enabled_while_on_whatever_the_tab(window: OverlayWindow) -> None:
    window._show_tab(1)
    window.handsfree_switch.setChecked(True)
    assert window._hands_free is True
    window._show_tab(0)
    assert window.handsfree_switch.isEnabled()
    window._show_tab(2)
    assert window.handsfree_switch.isEnabled()
    window._show_tab(3)
    assert window.handsfree_switch.isEnabled()


# =====================================================================================
# flipping the switch on the Assist tab enters the output-only hands-free view
# =====================================================================================
def test_flipping_switch_on_assist_tab_calls_actions_once_and_enters_the_view(
    window: OverlayWindow, actions: FakeActions
) -> None:
    window._show_tab(1)
    window.handsfree_switch.setChecked(True)
    assert actions.calls.count(("set_hands_free", (True,))) == 1
    assert window._hands_free is True

    for w in (window.askbox, window.control_row, window.status, window.size_grip, window.clickthrough_btn):
        assert w.isHidden(), w
    for btn in window._tab_buttons:
        assert btn.isHidden()

    assert not window.titlebar.isHidden()
    assert not window.tabstrip.isHidden()
    assert not window.stack.isHidden()
    assert window.stack.currentIndex() == 1
    assert not window.handsfree_switch.isHidden()
    assert not window.handsfree_label.isHidden()


def test_turning_off_restores_the_full_view(window: OverlayWindow) -> None:
    window._show_tab(1)
    window.handsfree_switch.setChecked(True)
    window.handsfree_switch.setChecked(False)
    assert window._hands_free is False

    for w in (window.askbox, window.control_row, window.status, window.size_grip, window.clickthrough_btn):
        assert not w.isHidden()
    for btn in window._tab_buttons:
        assert not btn.isHidden()
    assert window.handsfree_switch.isEnabled()   # still on the Assist tab
    window._show_tab(0)
    assert not window.handsfree_switch.isEnabled()


def test_toasts_show_on_and_off_messages(window: OverlayWindow) -> None:
    window._show_tab(1)
    window.handsfree_switch.setChecked(True)
    assert "Hands-free ON" in window.toast.text()
    window.handsfree_switch.setChecked(False)
    assert window.toast.text() == "Hands-free OFF"


# =====================================================================================
# AssistView.set_hands_free_view
# =====================================================================================
def test_set_hands_free_view_hides_chrome_shows_answer_and_disables_splitter_handle(
    window: OverlayWindow,
) -> None:
    av = window.assist_view
    av.set_hands_free_view(True)
    try:
        assert av.hands_free_view is True
        for w in (av.question_label, av.footer_label, av.answer_last_btn, av.history_toggle,
                  av.choices_container):
            assert w.isHidden()
        assert not av.answer_browser.isHidden()
        assert not av.history_splitter.handle(1).isEnabled()
    finally:
        av.set_hands_free_view(False)


def test_history_scroll_visible_in_view_iff_it_was_expanded_before(window: OverlayWindow) -> None:
    av = window.assist_view

    av.set_hands_free_view(True)
    assert av.history_scroll.isHidden()          # collapsed by default
    av.set_hands_free_view(False)

    av.history_toggle.setChecked(True)
    assert not av.history_scroll.isHidden()
    av.set_hands_free_view(True)
    assert not av.history_scroll.isHidden()      # stays visible: it was expanded
    av.set_hands_free_view(False)


def test_on_started_in_the_view_never_shows_the_stop_button(window: OverlayWindow) -> None:
    av = window.assist_view
    av.set_hands_free_view(True)
    try:
        av.on_started(1, "code", "task")
        assert av.stop_btn.isHidden()
    finally:
        av.set_hands_free_view(False)


def test_leaving_the_view_shows_stop_button_iff_still_streaming(window: OverlayWindow) -> None:
    av = window.assist_view

    av.set_hands_free_view(True)
    av.on_started(1, "code", "task")            # still streaming
    av.set_hands_free_view(False)
    assert not av.stop_btn.isHidden()

    av.set_hands_free_view(True)
    av.on_started(2, "code", "task 2")
    av.on_done(2, "the solution", 0.2, 20, 40.0)  # streaming ends
    av.set_hands_free_view(False)
    assert av.stop_btn.isHidden()


def test_choices_from_on_done_stay_hidden_in_the_view_and_appear_after_leaving(
    window: OverlayWindow,
) -> None:
    av = window.assist_view
    av.set_hands_free_view(True)
    av.on_started(3, "code", "task")
    av.on_done(3, "pick a language\n\n[[CHOICE]] Python\n[[CHOICE]] C++", 0.1, 5, 50.0)
    assert av.choices_layout.count() == 2
    assert av.choices_container.isHidden()       # never shown while in the view
    av.set_hands_free_view(False)
    assert not av.choices_container.isHidden()


# =====================================================================================
# title bar subtitle: "hands-free · <state>", follows HandsFreeEvent, back to the model off
# =====================================================================================
def test_subtitle_shows_state_and_follows_handsfree_events(window: OverlayWindow, bus: EventBus) -> None:
    window._show_tab(1)
    window.handsfree_switch.setChecked(True)
    assert window.subtitle_label.toolTip().startswith("hands-free")

    bus.publish(HandsFreeEvent(enabled=True, state="checkpoint"))
    assert wait_until(lambda: window.subtitle_label.toolTip() == "hands-free · checkpoint")

    bus.publish(HandsFreeEvent(enabled=True, state="code", detail="retry 1/2"))
    assert wait_until(lambda: window.subtitle_label.toolTip() == "hands-free · code · retry 1/2")

    window.handsfree_switch.setChecked(False)
    assert window.subtitle_label.toolTip() == window._cfg.llm.model


# =====================================================================================
# a HandsFreeEvent from the bus turns the view on/off without calling actions.set_hands_free
# =====================================================================================
def test_bus_handsfree_event_toggles_the_view_without_notifying_the_engine(
    window: OverlayWindow, bus: EventBus, actions: FakeActions
) -> None:
    assert window._hands_free is False
    calls_before = len(actions.calls)

    bus.publish(HandsFreeEvent(enabled=True, state="watching"))
    assert wait_until(lambda: window._hands_free is True)
    assert len(actions.calls) == calls_before
    assert window.handsfree_switch.isChecked()
    assert window.askbox.isHidden()

    bus.publish(HandsFreeEvent(enabled=False, state="off"))
    assert wait_until(lambda: window._hands_free is False)
    assert len(actions.calls) == calls_before
    assert not window.handsfree_switch.isChecked()
    assert not window.askbox.isHidden()


# =====================================================================================
# the slim bar's Context menu: a checkable "Hands-free" entry that reflects and toggles it
# =====================================================================================
def test_context_menu_handsfree_action_reflects_and_toggles_state(
    window: OverlayWindow, actions: FakeActions
) -> None:
    window._show_tab(1)
    window._rebuild_context_menu()
    acts = {a.text(): a for a in window._context_menu.actions() if a.text()}
    label = next(t for t in acts if t.startswith("Hands-free"))
    action = acts[label]
    assert action.isCheckable()
    assert not action.isChecked()

    action.setChecked(True)
    assert window._hands_free is True
    assert ("set_hands_free", (True,)) in actions.calls

    window._rebuild_context_menu()
    acts = {a.text(): a for a in window._context_menu.actions() if a.text()}
    assert acts[label].isChecked()

    acts[label].setChecked(False)
    assert window._hands_free is False
    assert ("set_hands_free", (False,)) in actions.calls


# =====================================================================================
# completion mode round trip while hands-free is on
# =====================================================================================
def test_completion_mode_round_trip_with_hands_free_on(window: OverlayWindow) -> None:
    window._show_tab(1)
    window.handsfree_switch.setChecked(True)

    window.completion_btn.setChecked(True)
    QTest.qWait(_WAIT_MS)
    assert window.slim.isVisibleTo(window)
    assert window.tabstrip.isHidden()
    assert window._hands_free is True
    assert window.handsfree_switch.isChecked()

    window.completion_btn.setChecked(False)
    QTest.qWait(_WAIT_MS)
    assert not window.tabstrip.isHidden()
    assert window.askbox.isHidden()            # the hands-free view was re-applied
    assert window.handsfree_switch.isChecked()
    assert window.handsfree_switch.isEnabled()


# =====================================================================================
# the ctrl+alt+h hotkey
# =====================================================================================
def test_hotkeys_config_default_toggle_handsfree_is_ctrl_alt_h() -> None:
    assert Config().hotkeys.toggle_handsfree == "ctrl+alt+h"


def test_hotkey_binding_table_contains_toggle_handsfree(
    qapp: QApplication, bus: EventBus, actions: FakeActions, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(overlay_mod, "_is_windows_platform", lambda: True)
    monkeypatch.setattr(overlay_mod, "GlobalHotkeys", _SpyHotkeys)
    _SpyHotkeys.instances = []
    cfg = Config()
    win = OverlayWindow(cfg, bus, actions)
    try:
        assert win._hotkeys is not None
        bindings = _SpyHotkeys.instances[-1].bindings
        assert cfg.hotkeys.toggle_handsfree in bindings
        bindings[cfg.hotkeys.toggle_handsfree]()
        assert win._hands_free is True
    finally:
        win.close_silently()
        qapp.processEvents()


# =====================================================================================
# cursor-tracking click-through (fake win32)
# =====================================================================================
def test_click_through_tracks_the_cursor_between_titlebar_and_content(
    qapp: QApplication, bus: EventBus, actions: FakeActions, fake_windows, monkeypatch: pytest.MonkeyPatch
) -> None:
    stub = _StubWin32()
    cfg = fake_windows(stub)   # cfg.ui.click_through defaults to False
    win = OverlayWindow(cfg, bus, actions)
    win.show()
    qapp.processEvents()
    try:
        win._show_tab(1)
        win.handsfree_switch.setChecked(True)
        qapp.processEvents()
        hwnd = int(win.winId())
        stub.calls.clear()

        title_pt = win.titlebar.mapToGlobal(QPoint(5, 5))
        monkeypatch.setattr(overlay_mod.QCursor, "pos", staticmethod(lambda: title_pt))
        win._update_hands_free_click_through()
        assert ("set_click_through", (hwnd, False)) in stub.calls

        stub.calls.clear()
        content_pt = win.stack.mapToGlobal(QPoint(20, 20))
        monkeypatch.setattr(overlay_mod.QCursor, "pos", staticmethod(lambda: content_pt))
        win._update_hands_free_click_through()
        assert ("set_click_through", (hwnd, True)) in stub.calls

        stub.calls.clear()
        win._update_hands_free_click_through()   # same point again: no repeat call
        assert stub.calls == []

        stub.calls.clear()
        win.handsfree_switch.setChecked(False)   # leaving hands-free: back to the user's setting
        assert ("set_click_through", (hwnd, False)) in stub.calls
    finally:
        win.close_silently()
        qapp.processEvents()


def test_click_through_restore_on_leaving_uses_the_configured_value(
    qapp: QApplication, bus: EventBus, actions: FakeActions, fake_windows, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same as above but with ``cfg.ui.click_through = True``: leaving hands-free must restore
    *that* value, not a hardcoded False (which happens to be what the default-config test above
    also sees for the title-bar case, so this checks the value is actually read from config)."""
    stub = _StubWin32()
    cfg = fake_windows(stub, click_through=True)
    win = OverlayWindow(cfg, bus, actions)
    win.show()
    qapp.processEvents()
    try:
        win._show_tab(1)
        win.handsfree_switch.setChecked(True)
        qapp.processEvents()
        hwnd = int(win.winId())
        content_pt = win.stack.mapToGlobal(QPoint(20, 20))
        monkeypatch.setattr(overlay_mod.QCursor, "pos", staticmethod(lambda: content_pt))
        win._update_hands_free_click_through()   # give _hf_click_through a value to restore from
        stub.calls.clear()

        win.handsfree_switch.setChecked(False)
        assert ("set_click_through", (hwnd, True)) in stub.calls
    finally:
        win.close_silently()
        qapp.processEvents()


def test_click_through_poll_does_nothing_in_completion_mode(
    qapp: QApplication, bus: EventBus, actions: FakeActions, fake_windows, monkeypatch: pytest.MonkeyPatch
) -> None:
    stub = _StubWin32()
    cfg = fake_windows(stub)
    win = OverlayWindow(cfg, bus, actions)
    win.show()
    qapp.processEvents()
    try:
        win._show_tab(1)
        win.handsfree_switch.setChecked(True)
        win.completion_btn.setChecked(True)
        qapp.processEvents()
        stub.calls.clear()
        win._update_hands_free_click_through()
        assert stub.calls == []
    finally:
        win.close_silently()
        qapp.processEvents()


# =====================================================================================
# settings: the "Hands-free" tab and "screen.ocr_device"
# =====================================================================================
def test_handsfree_tab_field_keys_match_handsfreeconfig_fields() -> None:
    keys = {f.key for f in FIELDS["Hands-free"]}
    expected = {f.name for f in dataclasses.fields(HandsFreeConfig)}
    assert keys == expected


def test_handsfree_accurate_ocr_needs_restart_the_rest_do_not() -> None:
    by_key = {f.key: f for f in FIELDS["Hands-free"]}
    assert by_key["accurate_ocr"].restart is True
    for key in ("change_threshold", "min_changed_lines", "settle_s", "deviation_threshold",
                "context_chars", "run_text", "run_code", "code_first", "enabled"):
        assert by_key[key].restart is False, key


def test_screen_ocr_device_field_choices_and_restart() -> None:
    field = next(f for f in FIELDS["Screen"] if f.key == "ocr_device")
    assert field.kind == "choice"
    assert field.choices == ("auto", "igpu", "npu", "cpu")
    assert field.restart is True


def test_settings_dialog_builds_widgets_for_handsfree_and_ocr_device(qapp: QApplication) -> None:
    cfg = Config()
    dlg = SettingsDialog(cfg, lambda: [])
    try:
        w = dlg._widgets["handsfree.change_threshold"]
        assert isinstance(w, QDoubleSpinBox)
        assert w.value() == pytest.approx(cfg.handsfree.change_threshold)

        enabled_w = dlg._widgets["handsfree.enabled"]
        assert isinstance(enabled_w, QCheckBox)
        assert enabled_w.isChecked() == cfg.handsfree.enabled

        combo = dlg._widgets["screen.ocr_device"]
        assert isinstance(combo, QComboBox)
        assert [combo.itemText(i) for i in range(combo.count())] == ["auto", "igpu", "npu", "cpu"]
        assert combo.currentText() == cfg.screen.ocr_device
        combo.setCurrentText("igpu")
        assert dlg.changes() == {"screen.ocr_device": "igpu"}
    finally:
        dlg.deleteLater()


# =====================================================================================
# _hands_free_scroll_area_at: the pane (if any) under a global point in the hands-free view
# =====================================================================================
def test_hands_free_scroll_area_at_returns_answer_browser_inside_it(window: OverlayWindow) -> None:
    window._show_tab(1)
    window.handsfree_switch.setChecked(True)
    QTest.qWait(_WAIT_MS)
    pos = window.assist_view.answer_browser.mapToGlobal(QPoint(5, 5))
    assert window._hands_free_scroll_area_at(pos) is window.assist_view.answer_browser


def test_hands_free_scroll_area_at_returns_history_scroll_when_expanded_and_inside(
    window: OverlayWindow,
) -> None:
    window._show_tab(1)
    window.assist_view.history_toggle.setChecked(True)   # expand while still in the normal view
    window.handsfree_switch.setChecked(True)
    QTest.qWait(_WAIT_MS)
    pos = window.assist_view.history_scroll.mapToGlobal(QPoint(5, 5))
    assert window._hands_free_scroll_area_at(pos) is window.assist_view.history_scroll


def test_hands_free_scroll_area_at_none_when_hands_free_off(window: OverlayWindow) -> None:
    pos = window.assist_view.answer_browser.mapToGlobal(QPoint(5, 5))
    assert window._hands_free_scroll_area_at(pos) is None


def test_hands_free_scroll_area_at_none_for_a_point_outside_the_window(window: OverlayWindow) -> None:
    window._show_tab(1)
    window.handsfree_switch.setChecked(True)
    QTest.qWait(_WAIT_MS)
    assert window._hands_free_scroll_area_at(QPoint(-10_000, -10_000)) is None


# =====================================================================================
# _wheel_hook_usable / _hands_free_clickable_at over an output pane
# =====================================================================================
def test_hands_free_clickable_at_false_over_pane_when_wheel_hook_usable(window: OverlayWindow) -> None:
    window._show_tab(1)
    window.handsfree_switch.setChecked(True)
    QTest.qWait(_WAIT_MS)
    assert window._wheel_hook_usable is True   # the real llhooks.MouseWheelHook is importable
    pos = window.assist_view.answer_browser.mapToGlobal(QPoint(5, 5))
    assert window._hands_free_clickable_at(pos) is False


def test_hands_free_clickable_at_true_over_pane_when_mousewheelhook_module_missing(
    window: OverlayWindow, monkeypatch: pytest.MonkeyPatch
) -> None:
    window._show_tab(1)
    window.handsfree_switch.setChecked(True)
    QTest.qWait(_WAIT_MS)
    monkeypatch.setattr(overlay_mod, "MouseWheelHook", None)
    assert window._wheel_hook_usable is False
    pos = window.assist_view.answer_browser.mapToGlobal(QPoint(5, 5))
    assert window._hands_free_clickable_at(pos) is True


def test_hands_free_clickable_at_true_over_pane_when_wheel_hook_failed(window: OverlayWindow) -> None:
    window._show_tab(1)
    window.handsfree_switch.setChecked(True)
    QTest.qWait(_WAIT_MS)

    class _Failed:
        failed = True
        installed = False

    window._wheel_hook = _Failed()   # type: ignore[assignment]
    assert window._wheel_hook_usable is False
    pos = window.assist_view.answer_browser.mapToGlobal(QPoint(5, 5))
    assert window._hands_free_clickable_at(pos) is True


# =====================================================================================
# _on_hands_free_wheel: the MouseWheelHook callback
# =====================================================================================
def test_on_hands_free_wheel_scrolls_the_pane_under_the_cursor(
    window: OverlayWindow, monkeypatch: pytest.MonkeyPatch
) -> None:
    window._show_tab(1)
    window.handsfree_switch.setChecked(True)
    QTest.qWait(_WAIT_MS)
    browser = window.assist_view.answer_browser
    bar = browser.verticalScrollBar()
    bar.setRange(0, 1000)
    bar.setValue(500)
    pos = browser.mapToGlobal(QPoint(5, 5))
    monkeypatch.setattr(overlay_mod.QCursor, "pos", staticmethod(lambda: pos))

    step = max(1, bar.singleStep()) * 3
    assert window._on_hands_free_wheel(120, False) is True
    assert bar.value() == 500 - step   # -(delta/120)*3*singleStep, delta=120 -> one step back


def test_on_hands_free_wheel_returns_false_when_not_over_a_pane(
    window: OverlayWindow, monkeypatch: pytest.MonkeyPatch
) -> None:
    window._show_tab(1)
    window.handsfree_switch.setChecked(True)
    QTest.qWait(_WAIT_MS)
    monkeypatch.setattr(overlay_mod.QCursor, "pos", staticmethod(lambda: QPoint(-10_000, -10_000)))
    assert window._on_hands_free_wheel(120, False) is False


# =====================================================================================
# _update_hands_free_hover_hooks / _release_hover_hooks (fake hook classes -- no real hook
# ever installed here)
# =====================================================================================
class _FakeHoverHook:
    """Stand-in for llhooks.MouseWheelHook / BareKeyHook: records install()/uninstall() without
    ever touching a real Win32 hook."""

    instances: list["_FakeHoverHook"] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.installed = False
        self.failed = False
        self.install_calls = 0
        self.uninstall_calls = 0
        type(self).instances.append(self)

    def install(self) -> bool:
        self.install_calls += 1
        self.installed = True
        return True

    def uninstall(self) -> None:
        self.uninstall_calls += 1
        self.installed = False


class _FakeWheelHook(_FakeHoverHook):
    instances: list["_FakeWheelHook"] = []


class _FakeNavHook(_FakeHoverHook):
    instances: list["_FakeNavHook"] = []


def test_update_hands_free_hover_hooks_installs_over_panes_and_history(
    window: OverlayWindow, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(overlay_mod, "MouseWheelHook", _FakeWheelHook)
    monkeypatch.setattr(overlay_mod, "BareKeyHook", _FakeNavHook)
    _FakeWheelHook.instances = []
    _FakeNavHook.instances = []

    window._show_tab(1)
    window.assist_view.history_toggle.setChecked(True)
    window.handsfree_switch.setChecked(True)
    QTest.qWait(_WAIT_MS)

    answer_pos = window.assist_view.answer_browser.mapToGlobal(QPoint(5, 5))
    window._update_hands_free_hover_hooks(answer_pos)
    assert window._wheel_hook is not None and window._wheel_hook.installed is True
    assert window._nav_key_hook is None   # never created: the cursor is not over the history

    history_pos = window.assist_view.history_scroll.mapToGlobal(QPoint(5, 5))
    window._update_hands_free_hover_hooks(history_pos)
    assert window._wheel_hook.installed is True
    assert window._nav_key_hook is not None and window._nav_key_hook.installed is True

    outside_pos = window.titlebar.mapToGlobal(QPoint(5, 5))
    window._update_hands_free_hover_hooks(outside_pos)
    assert window._wheel_hook.installed is False
    assert window._nav_key_hook.installed is False


def test_release_hover_hooks_uninstalls_both(
    window: OverlayWindow, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(overlay_mod, "MouseWheelHook", _FakeWheelHook)
    monkeypatch.setattr(overlay_mod, "BareKeyHook", _FakeNavHook)
    window._show_tab(1)
    window.assist_view.history_toggle.setChecked(True)
    window.handsfree_switch.setChecked(True)
    QTest.qWait(_WAIT_MS)

    history_pos = window.assist_view.history_scroll.mapToGlobal(QPoint(5, 5))
    window._update_hands_free_hover_hooks(history_pos)
    assert window._wheel_hook.installed is True
    assert window._nav_key_hook.installed is True

    window._release_hover_hooks()
    assert window._wheel_hook.installed is False
    assert window._nav_key_hook.installed is False


def test_leaving_hands_free_uninstalls_hover_hooks(
    window: OverlayWindow, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(overlay_mod, "MouseWheelHook", _FakeWheelHook)
    monkeypatch.setattr(overlay_mod, "BareKeyHook", _FakeNavHook)
    window._show_tab(1)
    window.assist_view.history_toggle.setChecked(True)
    window.handsfree_switch.setChecked(True)
    QTest.qWait(_WAIT_MS)

    history_pos = window.assist_view.history_scroll.mapToGlobal(QPoint(5, 5))
    window._update_hands_free_hover_hooks(history_pos)
    assert window._wheel_hook.installed is True
    assert window._nav_key_hook.installed is True

    window.handsfree_switch.setChecked(False)   # _apply_layout releases the hover hooks
    assert window._wheel_hook.installed is False
    assert window._nav_key_hook.installed is False


def test_close_event_uninstalls_hover_hooks(
    qapp: QApplication, bus: EventBus, actions: FakeActions, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(overlay_mod, "MouseWheelHook", _FakeWheelHook)
    monkeypatch.setattr(overlay_mod, "BareKeyHook", _FakeNavHook)
    win = OverlayWindow(Config(), bus, actions)
    win.show()
    qapp.processEvents()
    try:
        win._show_tab(1)
        win.assist_view.history_toggle.setChecked(True)
        win.handsfree_switch.setChecked(True)
        qapp.processEvents()
        history_pos = win.assist_view.history_scroll.mapToGlobal(QPoint(5, 5))
        win._update_hands_free_hover_hooks(history_pos)
        assert win._wheel_hook.installed is True
        assert win._nav_key_hook.installed is True
    finally:
        win.close_silently()
        qapp.processEvents()
    assert win._wheel_hook.installed is False
    assert win._nav_key_hook.installed is False


# =====================================================================================
# _toggle_pause_hotkey: flips the Pause button, notifies actions.set_paused, toasts
# =====================================================================================
def test_toggle_pause_hotkey_toggles_pause_btn_notifies_actions_and_toasts(
    window: OverlayWindow, actions: FakeActions
) -> None:
    assert window.pause_btn.isChecked() is False

    window._toggle_pause_hotkey()
    assert window.pause_btn.isChecked() is True
    assert ("set_paused", (True,)) in actions.calls
    assert window.toast.text() == "Paused"

    window._toggle_pause_hotkey()
    assert window.pause_btn.isChecked() is False
    assert ("set_paused", (False,)) in actions.calls
    assert window.toast.text() == "Resumed"


# =====================================================================================
# _install_hotkeys: [hotkeys].toggle_pause routes to a RegisterHotKey combination or a bare
# key (llhooks.BareKeyHook); any real-platform monkeypatching stays local to each test.
# =====================================================================================
def test_install_hotkeys_combination_toggle_pause_binds_via_global_hotkeys(
    qapp: QApplication, bus: EventBus, actions: FakeActions, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(overlay_mod, "_is_windows_platform", lambda: True)
    monkeypatch.setattr(overlay_mod, "GlobalHotkeys", _SpyHotkeys)
    _SpyHotkeys.instances = []
    cfg = Config()
    cfg.hotkeys.toggle_pause = "ctrl+alt+p"
    win = OverlayWindow(cfg, bus, actions)
    try:
        bindings = _SpyHotkeys.instances[-1].bindings
        assert "ctrl+alt+p" in bindings
        assert win._pause_key_hook is None
    finally:
        win.close_silently()
        qapp.processEvents()


def test_install_hotkeys_bare_key_calls_install_pause_key(
    qapp: QApplication, bus: EventBus, actions: FakeActions, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(overlay_mod, "_is_windows_platform", lambda: True)
    monkeypatch.setattr(overlay_mod, "GlobalHotkeys", _SpyHotkeys)
    monkeypatch.setattr(overlay_mod, "BareKeyHook", _FakeNavHook)
    _SpyHotkeys.instances = []
    _FakeNavHook.instances = []
    cfg = Config()
    assert cfg.hotkeys.toggle_pause == "rshift"
    win = OverlayWindow(cfg, bus, actions)
    try:
        assert len(_FakeNavHook.instances) == 1
        assert win._pause_key_hook is _FakeNavHook.instances[0]
        assert win._pause_key_hook.installed is True
        bindings = _SpyHotkeys.instances[-1].bindings
        assert "rshift" not in bindings   # bound as a bare-key hook, not a RegisterHotKey combo
    finally:
        win.close_silently()
        qapp.processEvents()


# =====================================================================================
# AssistView.history_entries / history_snap
# =====================================================================================
def _long_answer() -> str:
    code = "\n".join(f"line_{j} = {j}" for j in range(25))
    return "\n\n".join(f"Step {i}: " + "word " * 60 + f"\n\n```python\n{code}\n```" for i in range(3))


def test_history_entries_excludes_checkpoint_dividers(window: OverlayWindow, bus: EventBus) -> None:
    av = window.assist_view
    av.on_started(1, "solve", "q1")
    av.on_done(1, "answer one", 0.1, 3, 60.0)
    bus.publish(CheckpointEvent(t=1.0, segment_id=1))
    QTest.qWait(_WAIT_MS)
    av.on_started(2, "solve", "q2")
    av.on_done(2, "answer two", 0.1, 3, 60.0)

    entries = av.history_entries()
    assert len(entries) == 2
    for entry in entries:
        assert "checkpoint" not in entry.text().lower()


def test_assist_view_history_snap_scrolls_and_marks_entries(window: OverlayWindow) -> None:
    window._show_tab(1)
    window.resize(400, 700)
    QTest.qWait(_WAIT_MS)
    av = window.assist_view
    for i in range(1, 5):
        av.on_started(i, "solve", f"q{i}")
        av.on_done(i, _long_answer(), 0.1, 700, 40.0)
    av.history_toggle.setChecked(True)
    QTest.qWait(_WAIT_MS)
    # force the pane small enough that four long answers actually overflow it
    av.history_splitter.moveSplitter(av.history_splitter.height() - 100, 1)
    QTest.qWait(_WAIT_MS)

    entries = av.history_entries()
    assert len(entries) == 4
    bar = av.history_scroll.verticalScrollBar()
    assert bar.maximum() > 0   # the pane is actually scrollable

    accent = overlay_mod.COLORS.get("accent", "#4c8bf5")

    # snap down twice: lands on entries[1], then entries[2], marking whichever is current
    assert av.history_snap(1) is True
    assert bar.value() == entries[1].y()
    assert accent in entries[1].styleSheet()

    assert av.history_snap(1) is True
    assert bar.value() == entries[2].y()
    assert accent in entries[2].styleSheet()
    assert entries[1].styleSheet() == ""   # the mark moved off the previous entry

    # snap up: back to the previous entry
    assert av.history_snap(-1) is True
    assert bar.value() == entries[1].y()
    assert accent in entries[1].styleSheet()
    assert entries[2].styleSheet() == ""

    # collapsed: no snap either direction
    av.history_toggle.setChecked(False)
    QTest.qWait(_WAIT_MS)
    assert av.history_snap(1) is False
    assert av.history_snap(-1) is False
    av.history_toggle.setChecked(True)
    QTest.qWait(_WAIT_MS)

    # at the top end: snapping up from the newest entry goes nowhere
    assert av.history_snap(-1) is True
    assert bar.value() == entries[0].y()
    assert av.history_snap(-1) is False

    # at the bottom end: snapping down from the oldest entry goes nowhere
    assert av.history_snap(1) is True
    assert av.history_snap(1) is True
    assert av.history_snap(1) is True
    assert bar.value() == entries[3].y()
    assert av.history_snap(1) is False

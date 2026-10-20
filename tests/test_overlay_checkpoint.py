"""Offscreen tests for checkpoints, the Code button and the Assist tab's resizable history
(2026-09-22 pass): control-row / slim-bar wiring, the CheckpointEvent -> UI update path, the
history QSplitter's collapse/expand behaviour, revert skipping dividers, and the size-floor
measurements the design calls out (control row width, slim bar width, the Assist tab's
minimumSizeHint before/after a long history).

Mirrors tests/test_overlay.py's fixtures; reuses tests/test_overlay_completion.py's catch-all
FakeActions (it already accepts arbitrary EngineActions calls, checkpoint()/code() included).
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from typing import Any

import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QPushButton

from exocortex.bus import CheckpointEvent, EventBus
from exocortex.config import Config
from exocortex.ui import overlay as overlay_mod
from exocortex.ui.overlay import OverlayWindow
from tests.test_overlay import _SpyHotkeys
from tests.test_overlay_completion import FakeActions

_WAIT_MS = 80


@pytest.fixture(scope="session")
def qapp() -> QApplication:
    return QApplication.instance() or QApplication(["exocortex-checkpoint-tests"])


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


def _long_answer() -> str:
    code = "\n".join(f"line_{j} = {j}" for j in range(25))
    return "\n\n".join(f"Step {i}: " + "word " * 60 + f"\n\n```python\n{code}\n```" for i in range(3))


# =====================================================================================
# checkpoint_btn / code_btn -> actions
# =====================================================================================
def test_checkpoint_button_calls_actions_checkpoint_and_nothing_else(
    window: OverlayWindow, actions: FakeActions
) -> None:
    QTest.mouseClick(window.checkpoint_btn, Qt.MouseButton.LeftButton)
    assert actions.calls == [("checkpoint", ())]


def test_code_button_calls_actions_code_and_switches_to_assist_tab(
    window: OverlayWindow, actions: FakeActions
) -> None:
    window._show_tab(0)
    assert window.stack.currentIndex() == 0
    window.code_btn.click()
    assert window.stack.currentIndex() == 1
    assert ("code", ()) in actions.calls


def test_code_button_does_not_switch_tabs_in_completion_mode(
    window: OverlayWindow, actions: FakeActions
) -> None:
    window.completion_btn.setChecked(True)
    QTest.qWait(_WAIT_MS)
    before = window.stack.currentIndex()
    window.slim_code_btn.click()
    assert window.stack.currentIndex() == before
    assert ("code", ()) in actions.calls


def test_slim_checkpoint_button_calls_actions_checkpoint(window: OverlayWindow, actions: FakeActions) -> None:
    window.completion_btn.setChecked(True)
    QTest.qWait(_WAIT_MS)
    window.slim_checkpoint_btn.click()
    assert ("checkpoint", ()) in actions.calls


# =====================================================================================
# CheckpointEvent -> panel updates (never driven directly by the button click)
# =====================================================================================
def test_checkpoint_event_adds_divider_toasts_and_clears_choices(
    window: OverlayWindow, bus: EventBus
) -> None:
    window.assist_view.on_started(1, "solve", "q1")
    window.assist_view.on_done(1, "pick a language\n\n[[CHOICE]] Python\n[[CHOICE]] C++", 0.1, 3, 60.0)
    assert window.assist_view.choices_container.isVisibleTo(window.assist_view)
    assert window.assist_view._history   # the fork's own body text was pushed to history

    bus.publish(CheckpointEvent(t=5.0, segment_id=3))
    QTest.qWait(_WAIT_MS)

    assert window.assist_view._history[0][0] is None
    assert "checkpoint" in window.assist_view._history[0][1]
    assert not window.assist_view.choices_container.isVisibleTo(window.assist_view)
    assert "starts fresh" in window.assist_view.footer_label.text()
    assert window.toast.isVisible()
    assert "Checkpoint" in window.toast.text()


def test_checkpoint_event_drops_plan_and_explanation_state(window: OverlayWindow, bus: EventBus) -> None:
    window.explain_btn.setChecked(True)
    window._plan_text = "some walkthrough"
    window._plan_job = 9
    window._explain_text = "some explanation"
    window._explain_job = 10
    window.slim_panel.set_text("walkthrough", "some walkthrough")
    window.slim_panel.show()

    bus.publish(CheckpointEvent(t=1.0, segment_id=None))
    QTest.qWait(_WAIT_MS)

    assert window._plan_text == "" and window._plan_job is None
    assert window._explain_text == "" and window._explain_job is None
    assert not window.slim_panel.has("walkthrough")
    assert window.slim_panel.isHidden()


def test_checkpoint_button_click_does_not_itself_touch_the_panel(
    window: OverlayWindow, actions: FakeActions
) -> None:
    """_on_checkpoint calls actions.checkpoint() ONLY; without a CheckpointEvent on the bus, the
    history/toast/plan state must be untouched (the panel updates come from the event handler)."""
    window.assist_view.on_started(1, "solve", "q1")
    window.assist_view.on_done(1, "answer", 0.1, 3, 60.0)
    before_history = list(window.assist_view._history)

    window.checkpoint_btn.click()

    assert window.assist_view._history == before_history
    assert not window.toast.isVisible()


# =====================================================================================
# control row / slim bar composition
# =====================================================================================
def test_slim_bar_has_checkpoint_and_code_buttons(window: OverlayWindow) -> None:
    assert window.slim_checkpoint_btn.text() == "Checkpoint"
    assert window.slim_code_btn.text() == "Code"
    assert window.slim_code_btn.objectName() == "primary"


def test_slim_bar_order_full_context_explain_reset_pause_checkpoint_code(window: OverlayWindow) -> None:
    window.explain_btn.setChecked(True)   # so slim_explain_btn is part of the visible order too
    order = [b.text() for b in window.slim.findChildren(QPushButton)]
    # explain button's label is "Explain"; every button we care about must appear, in this order.
    wanted = ["Full", "Context", "Explain", "Reset", "Pause", "Checkpoint", "Code"]
    assert [t for t in order if t in wanted] == wanted


def test_slim_context_menu_has_checkpoint_and_code_entries(window: OverlayWindow, actions: FakeActions) -> None:
    window.completion_btn.setChecked(True)
    window._rebuild_context_menu()
    texts = [a.text() for a in window._context_menu.actions() if a.text()]
    assert "Checkpoint (new coding task)" in texts
    assert "Code now" in texts
    by_text = {a.text(): a for a in window._context_menu.actions() if a.text()}
    by_text["Checkpoint (new coding task)"].trigger()
    assert ("checkpoint", ()) in actions.calls
    by_text["Code now"].trigger()
    assert ("code", ()) in actions.calls


def test_checkpoint_code_precede_settings_separator_in_context_menu(window: OverlayWindow) -> None:
    window._rebuild_context_menu()
    actions_list = window._context_menu.actions()
    idx = {a.text(): i for i, a in enumerate(actions_list) if a.text()}
    assert idx["Checkpoint (new coding task)"] < idx["Settings…"]
    assert idx["Code now"] < idx["Settings…"]


# =====================================================================================
# hotkeys: ctrl+alt+b / ctrl+alt+g bound to the checkpoint/code handlers
# =====================================================================================
def test_hotkey_bindings_include_checkpoint_and_code(
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
        assert cfg.hotkeys.checkpoint in bindings and cfg.hotkeys.code in bindings
        bindings[cfg.hotkeys.checkpoint]()
        assert ("checkpoint", ()) in actions.calls
        bindings[cfg.hotkeys.code]()
        assert ("code", ()) in actions.calls
    finally:
        win.close_silently()
        qapp.processEvents()


# =====================================================================================
# resizable history: QSplitter collapse/expand
# =====================================================================================
def test_history_splitter_collapsed_by_default_handle_disabled_and_bottom_capped(
    window: OverlayWindow,
) -> None:
    av = window.assist_view
    assert not av.history_toggle.isChecked()
    assert av.history_scroll.isHidden()
    assert not av.history_splitter.handle(1).isEnabled()
    bottom = av.history_splitter.widget(1)
    assert bottom.maximumHeight() == av.history_toggle.sizeHint().height()


def test_history_splitter_expand_enables_handle_and_uses_remembered_height(
    window: OverlayWindow,
) -> None:
    window._show_tab(1)
    av = window.assist_view
    av.history_toggle.setChecked(True)
    QTest.qWait(_WAIT_MS)
    assert av.history_scroll.isVisibleTo(av)
    assert av.history_splitter.handle(1).isEnabled()
    sizes = av.history_splitter.sizes()
    assert len(sizes) == 2 and sizes[1] == av._history_height


def test_history_splitter_remembers_dragged_height_across_collapse_expand(
    window: OverlayWindow,
) -> None:
    window._show_tab(1)
    window.resize(400, 700)
    QTest.qWait(_WAIT_MS)
    av = window.assist_view
    av.history_toggle.setChecked(True)
    QTest.qWait(_WAIT_MS)

    av.history_splitter.moveSplitter(av.history_splitter.height() - 150, 1)
    QTest.qWait(_WAIT_MS)
    dragged = av._history_height
    assert dragged != 200   # actually moved away from the initial default

    av.history_toggle.setChecked(False)
    QTest.qWait(_WAIT_MS)
    assert not av.history_splitter.handle(1).isEnabled()
    assert av.history_scroll.isHidden()

    av.history_toggle.setChecked(True)
    QTest.qWait(_WAIT_MS)
    sizes = av.history_splitter.sizes()
    assert sizes[1] == dragged


def test_history_splitter_panes_are_not_collapsible(window: OverlayWindow) -> None:
    av = window.assist_view
    assert av.history_splitter.isCollapsible(0) is False
    assert av.history_splitter.isCollapsible(1) is False


# =====================================================================================
# revert skips checkpoint dividers
# =====================================================================================
def test_checkpoint_with_no_intervening_job_does_not_break_the_next_revert(
    window: OverlayWindow, bus: EventBus
) -> None:
    """Regression: `mark_checkpoint()` used to insert the divider at `_history[0]` without
    shifting `_view_index`, so -- when a checkpoint happens with no solve/code job completing in
    between (an explicitly supported flow: the checkpoint's UI effect is independent of whether a
    new job has started) -- the very first `revert()` afterwards just redisplayed the answer
    already on screen (landing back on the divider's old neighbour) instead of stepping to a
    genuinely earlier one."""
    av = window.assist_view
    av.on_started(1, "solve", "q0")
    av.on_done(1, "answer zero", 0.1, 3, 60.0)
    av.on_started(2, "solve", "q1")
    av.on_done(2, "answer one", 0.1, 3, 60.0)
    assert "answer one" in av.answer_browser.toPlainText()

    bus.publish(CheckpointEvent(t=1.0, segment_id=2))   # no job runs between this and revert()
    QTest.qWait(_WAIT_MS)
    assert "answer one" in av.answer_browser.toPlainText()   # the on-screen text is unchanged

    av.revert()
    assert "answer zero" in av.answer_browser.toPlainText()
    assert "reverted" in av.footer_label.text()


def test_revert_skips_checkpoint_dividers(window: OverlayWindow, bus: EventBus) -> None:
    av = window.assist_view
    av.on_started(1, "solve", "q1")
    av.on_done(1, "answer one", 0.1, 3, 60.0)
    bus.publish(CheckpointEvent(t=1.0, segment_id=1))
    QTest.qWait(_WAIT_MS)
    av.on_started(2, "code", "q2")
    av.on_done(2, "answer two", 0.1, 3, 60.0)

    av.revert()
    assert "answer one" in av.answer_browser.toPlainText()
    assert "reverted" in av.footer_label.text()

    av.revert()
    assert av.footer_label.text() == "no earlier answer"


# =====================================================================================
# size-floor measurements the design calls out
# =====================================================================================
def test_control_row_never_raises_the_window_floor(window: OverlayWindow) -> None:
    """The six buttons fit the 340 px floor under the real Windows font (322 px measured, see
    `_build_control_row`) but not under the offscreen QPA's much wider fallback font, so the row's
    own minimum must stay out of the window's: whatever the font, the root's minimum width is the
    title bar's, not the control row's."""
    window.root.layout().activate()
    row_min = window.control_row.minimumSizeHint().width()
    assert window.root.minimumSizeHint().width() <= overlay_mod._MIN_WIDTH
    assert window.root.minimumSizeHint().width() < row_min + 16   # i.e. the row is not what sets it


def test_slim_bar_min_width_with_explanations_on_and_off(window: OverlayWindow) -> None:
    window.slim.layout().activate()
    off = window.slim.minimumSizeHint().width()
    window.explain_btn.setChecked(True)
    window.slim.layout().activate()
    on = window.slim.minimumSizeHint().width()
    assert on >= off > 0   # the absolute numbers are font-dependent (offscreen != Windows): not asserted


def test_completion_mode_applies_a_width_that_fits_the_slim_bars_own_minimum(
    window: OverlayWindow,
) -> None:
    """Regression: `_set_completion_mode` used to size the slim bar from the full-mode window's
    CURRENT width alone (`cfg.ui.width` defaults to 460), never checking that against
    `slim.minimumSizeHint()` (wider than that now that Checkpoint/Code/Explain all live in the
    slim bar), so toggling completion mode on a fresh, un-resized window squeezed every slim-bar
    button below its own content width instead of growing the window to fit them."""
    window.slim.layout().activate()
    needed = window.slim.minimumSizeHint().width()
    assert window.width() < needed, "fixture window must start narrower than the slim bar needs"

    window.completion_btn.setChecked(True)
    QTest.qWait(_WAIT_MS)
    window.slim.layout().activate()

    assert window.width() >= window.slim.minimumSizeHint().width()
    assert window.slim.width() >= window.slim.minimumSizeHint().width()


def test_assist_tab_min_size_hint_does_not_balloon_with_a_long_expanded_history(
    window: OverlayWindow,
) -> None:
    """Regression guard for the QSplitter change: the history list no longer has a fixed 200 px
    max, so a long history must not inflate the tab's (and so the window's) minimum size -- the
    splitter, not a height cap on the scroll area, is what bounds it now."""
    window.root.layout().activate()
    before = window.minimumSizeHint()

    for i in range(1, 15):
        window.assist_view.on_started(i, "solve", f"q{i}")
        window.assist_view.on_done(i, _long_answer(), 0.1, 700, 40.0)
    window.assist_view.history_toggle.setChecked(True)
    QTest.qWait(_WAIT_MS)
    window.root.layout().activate()
    after = window.minimumSizeHint()

    assert after.width() == before.width()
    assert after.height() == before.height()

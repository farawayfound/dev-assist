"""Offscreen tests for the Text button (2026-09-23 pass): the control-row button itself, its
position immediately left of Code, wiring to ``actions.text()``, completion-mode hiding (no Text
anywhere in the slim bar / slim context menu), the Assist tab's handling of a "text" job (question
prefix, history entry, footer), the bus wiring (AssistStartedEvent/AssistDeltaEvent landing in the
Assist tab) and the "text is not a plan" rule (no slim panel, no `_plan_job`, unlike solve/code --
see tests/test_overlay_explain.py for that comparison). Mirrors tests/test_overlay_checkpoint.py's
fixture style; reuses tests/test_overlay_completion.py's catch-all FakeActions.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QPushButton

from exocortex.bus import AssistDeltaEvent, AssistStartedEvent, EventBus
from exocortex.config import Config
from exocortex.ui.overlay import OverlayWindow
from tests.test_overlay import wait_until
from tests.test_overlay_completion import FakeActions

_WAIT_MS = 80


@pytest.fixture(scope="session")
def qapp() -> QApplication:
    return QApplication.instance() or QApplication(["exocortex-text-tests"])


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
# text_btn: label, style, tooltip
# =====================================================================================
def test_text_button_label_style_and_tooltip(window: OverlayWindow) -> None:
    assert window.text_btn.text() == "Text"
    assert window.text_btn.objectName() == "primary"          # same accent style as code_btn
    assert window.code_btn.objectName() == "primary"
    assert "plain English" in window.text_btn.toolTip()


# =====================================================================================
# control row layout position: Read, Revert, Reset, Pause, Checkpoint, stretch, Text, Code
# =====================================================================================
def test_control_row_order_ends_with_stretch_text_code(window: OverlayWindow) -> None:
    layout = window.control_row.layout()
    assert layout is not None
    widgets: list[QPushButton | None] = []
    stretch_index = None
    for i in range(layout.count()):
        item = layout.itemAt(i)
        w = item.widget()
        if w is None:
            stretch_index = i   # the addStretch(1) item has no widget
        widgets.append(w)

    labels = [w.text() if w is not None else "<stretch>" for w in widgets]
    assert labels == ["Read", "Revert", "Reset", "Pause", "Checkpoint", "<stretch>", "Text", "Code"]
    assert stretch_index == 5
    # Text sits immediately left of Code, and both come after the stretch.
    assert widgets[-2] is window.text_btn
    assert widgets[-1] is window.code_btn


def test_code_btn_is_the_last_control_row_widget(window: OverlayWindow) -> None:
    """Regression guard mirrored from tests/test_overlay.py's six-button-order test: adding Text
    must not displace Code from being the last widget in the control row's layout."""
    layout = window.control_row.layout()
    assert layout is not None
    last_widget = None
    for i in range(layout.count()):
        w = layout.itemAt(i).widget()
        if w is not None:
            last_widget = w
    assert last_widget is window.code_btn


# =====================================================================================
# clicking text_btn -> actions.text() + switches to the Assist tab
# =====================================================================================
def test_text_button_calls_actions_text_and_switches_to_assist_tab(
    window: OverlayWindow, actions: FakeActions
) -> None:
    window._show_tab(0)
    assert window.stack.currentIndex() == 0
    QTest.mouseClick(window.text_btn, Qt.MouseButton.LeftButton)
    assert window.stack.currentIndex() == 1
    assert ("text", ()) in actions.calls


def test_text_button_click_method_also_records_the_call(window: OverlayWindow, actions: FakeActions) -> None:
    window._show_tab(0)
    window.text_btn.click()
    assert window.stack.currentIndex() == 1
    assert ("text", ()) in actions.calls


# =====================================================================================
# completion mode: control row hidden, no Text anywhere in the slim bar / slim menu
# =====================================================================================
def test_completion_mode_hides_control_row_and_text_button(
    window: OverlayWindow, actions: FakeActions
) -> None:
    window.completion_btn.setChecked(True)
    QTest.qWait(_WAIT_MS)
    assert window.control_row.isHidden()

    # no QPushButton child of the slim bar reads "Text"
    slim_button_texts = [b.text() for b in window.slim.findChildren(QPushButton)]
    assert "Text" not in slim_button_texts

    # the slim context menu deliberately has no Text entry
    window._rebuild_context_menu()
    texts = [a.text() for a in window._context_menu.actions() if a.text()]
    assert not any("Text" in t for t in texts)


def test_on_text_in_completion_mode_records_call_without_changing_stack_index(
    window: OverlayWindow, actions: FakeActions
) -> None:
    window.completion_btn.setChecked(True)
    QTest.qWait(_WAIT_MS)
    before = window.stack.currentIndex()
    window._on_text()
    assert window.stack.currentIndex() == before
    assert ("text", ()) in actions.calls


def test_text_btn_is_not_a_descendant_of_the_slim_bar(window: OverlayWindow) -> None:
    """The slim bar's minimumSizeHint must be unaffected by text_btn's existence: it lives only
    in the control row, never reparented into `self.slim`."""
    assert window.text_btn not in window.slim.findChildren(QPushButton)


# =====================================================================================
# AssistView: on_started / on_done for a "text" job
# =====================================================================================
def test_assist_view_on_started_sets_text_prefixed_question_label(window: OverlayWindow) -> None:
    window.assist_view.on_started(1, "text", "the task being discussed")
    assert window.assist_view.question_label.text() == "Text: the task being discussed"


def test_assist_view_on_done_pushes_history_entry_with_stripped_prefix_and_text_footer(
    window: OverlayWindow,
) -> None:
    av = window.assist_view
    av.on_started(7, "text", "the task being discussed")
    av.on_done(7, "Here is the walkthrough.", 0.2, 40, 55.0)
    assert av._history                     # pushed onto history
    question, answer = av._history[0]
    assert question == "the task being discussed"     # the "Text: " prefix was stripped
    assert answer == "Here is the walkthrough."
    assert av.footer_label.text().startswith("text")


# =====================================================================================
# bus wiring: AssistStartedEvent(kind="text") + AssistDeltaEvent -> answer_browser + Assist tab
# =====================================================================================
def test_text_job_bus_events_land_in_answer_browser_and_switch_to_assist_tab(
    window: OverlayWindow, bus: EventBus
) -> None:
    window._show_tab(0)
    bus.publish(AssistStartedEvent(job_id=21, kind="text", question="the task being discussed"))
    bus.publish(AssistDeltaEvent(job_id=21, delta="Walking through the problem step by step."))
    # the delta reaches the browser only once AssistView's own refresh timer (60 ms) has ticked,
    # so poll instead of a single fixed-length qWait (avoids flakiness under a loaded test run).
    assert wait_until(
        lambda: "Walking through the problem step by step." in window.assist_view.answer_browser.toPlainText()
    )
    assert window.stack.currentIndex() == 1


def test_text_job_is_not_a_plan_with_explanations_and_completion_mode_on(
    window: OverlayWindow, bus: EventBus
) -> None:
    """Contrast with tests/test_overlay_explain.py's `test_code_events_fill_the_walkthrough_panel_like_solve`:
    solve/code jobs are `is_plan` and open the slim walkthrough panel; a "text" job must not."""
    window.explain_btn.setChecked(True)
    window.completion_btn.setChecked(True)
    QTest.qWait(_WAIT_MS)
    assert window.slim_panel.isHidden()
    assert window._plan_job is None

    bus.publish(AssistStartedEvent(job_id=22, kind="text", question="the task being discussed"))
    bus.publish(AssistDeltaEvent(job_id=22, delta="Some walkthrough prose."))
    QTest.qWait(_WAIT_MS)

    assert window.slim_panel.isHidden()      # never opened for a text job
    assert window._plan_job is None          # never set for a text job


# =====================================================================================
# hotkeys: no "text" attribute on HotkeysConfig
# =====================================================================================
def test_hotkeys_config_has_no_text_attribute() -> None:
    assert not hasattr(Config().hotkeys, "text")

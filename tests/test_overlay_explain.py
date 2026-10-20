"""Offscreen tests for the explanations UI: the 💡 toggle, the slim bar's Explain drop-down and its
walkthrough / explanation panel driven by solve + explain events."""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from exocortex.bus import AssistDeltaEvent, AssistDoneEvent, AssistStartedEvent, EventBus
from exocortex.config import Config
from exocortex.ui.overlay import OverlayWindow
from tests.test_overlay_completion import FakeActions
from tests.test_plan import ANSWER

_WAIT_MS = 80


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


def _done(job_id: int, text: str, cancelled: bool = False) -> AssistDoneEvent:
    return AssistDoneEvent(job_id=job_id, text=text, ttft_s=0.1, tokens=5, tok_per_s=50.0, cancelled=cancelled)


def test_explain_toggle_drives_actions_and_the_slim_button(window: OverlayWindow, actions: FakeActions) -> None:
    assert not window.explain_btn.isChecked() and window.slim_explain_btn.isHidden()
    window.explain_btn.setChecked(True)
    assert ("set_explanations", (True,)) in actions.calls
    assert not window.slim_explain_btn.isHidden()
    window.explain_btn.setChecked(False)
    assert ("set_explanations", (False,)) in actions.calls
    assert window.slim_explain_btn.isHidden()


def test_context_menu_mirrors_the_explanations_toggle(window: OverlayWindow) -> None:
    window._rebuild_context_menu()
    act = next(a for a in window._context_menu.actions() if a.text().startswith("Explanations"))
    assert act.isCheckable() and not act.isChecked()
    act.setChecked(True)
    assert window.explain_btn.isChecked()
    window.explain_btn.setChecked(False)
    window._rebuild_context_menu()
    act = next(a for a in window._context_menu.actions() if a.text().startswith("Explanations"))
    assert not act.isChecked()


def test_starts_with_explanations_on_when_configured(qapp: QApplication, bus: EventBus, actions: FakeActions) -> None:
    cfg = Config()
    cfg.explain.enabled = True
    win = OverlayWindow(cfg, bus, actions)
    try:
        assert win.explain_btn.isChecked() and not win.slim_explain_btn.isHidden()
        assert ("set_explanations", (True,)) not in actions.calls     # reflecting config is not a user action
    finally:
        win.close_silently()
        qapp.processEvents()


def test_plan_and_explanation_events_fill_the_slim_panel(window: OverlayWindow, bus: EventBus) -> None:
    window.explain_btn.setChecked(True)
    window.completion_btn.setChecked(True)
    QTest.qWait(_WAIT_MS)
    slim_h = window.height()
    assert window.slim_panel.isHidden()

    bus.publish(AssistStartedEvent(job_id=11, kind="solve", question="Solving"))
    for i in range(0, len(ANSWER), 40):
        bus.publish(AssistDeltaEvent(job_id=11, delta=ANSWER[i:i + 40]))
    QTest.qWait(_WAIT_MS)
    assert not window.slim_panel.isHidden() and window.slim_panel.section == "walkthrough"
    assert window.height() > slim_h                                    # the window grew to fit the drop-down
    assert "We need two indices" in window.slim_panel.browser.toPlainText()
    assert "seen = {}" not in window.slim_panel.browser.toPlainText()  # only the walkthrough, never the code
    bus.publish(_done(11, ANSWER))
    QTest.qWait(_WAIT_MS)
    assert window.slim_status.toolTip().startswith("solution ready")   # _ElidingLabel: full text in the tooltip
    assert window.slim_panel.has("walkthrough") and not window.slim_panel.has("explanation")

    window.slim_panel.close_btn.click()
    QTest.qWait(_WAIT_MS)
    assert window.slim_panel.isHidden() and window.height() == slim_h

    bus.publish(AssistStartedEvent(job_id=12, kind="explain", question="Explanation"))
    bus.publish(AssistDeltaEvent(job_id=12, delta="- It maps values to indices."))
    bus.publish(_done(12, "- It maps values to indices."))
    QTest.qWait(_WAIT_MS)
    assert not window.slim_panel.isHidden() and window.slim_panel.section == "explanation"
    assert "maps values to indices" in window.slim_panel.browser.toPlainText()
    assert window.slim_status.toolTip() == "explanation ready"
    # the Assist tab (hidden behind the slim bar) received it too, for when the user goes back to Full
    assert "maps values to indices" in window.assist_view.answer_browser.toPlainText()

    window.slim_reset_btn.click()
    QTest.qWait(_WAIT_MS)
    assert window.slim_panel.isHidden() and not window.slim_panel.has("walkthrough")
    assert window.height() == slim_h


def test_code_events_fill_the_walkthrough_panel_like_solve(window: OverlayWindow, bus: EventBus) -> None:
    """The Code button is `is_plan` just like solve (design: "same as solve"), so its streamed
    walkthrough must open and fill the slim panel exactly as a solve job's does."""
    window.explain_btn.setChecked(True)
    window.completion_btn.setChecked(True)
    QTest.qWait(_WAIT_MS)
    slim_h = window.height()
    assert window.slim_panel.isHidden()

    bus.publish(AssistStartedEvent(job_id=31, kind="code", question="Code: the task being discussed"))
    for i in range(0, len(ANSWER), 40):
        bus.publish(AssistDeltaEvent(job_id=31, delta=ANSWER[i:i + 40]))
    QTest.qWait(_WAIT_MS)
    assert not window.slim_panel.isHidden() and window.slim_panel.section == "walkthrough"
    assert window.height() > slim_h
    assert "We need two indices" in window.slim_panel.browser.toPlainText()

    bus.publish(_done(31, ANSWER))
    QTest.qWait(_WAIT_MS)
    assert window.slim_status.toolTip().startswith("solution ready")
    assert window.slim_panel.has("walkthrough")


def test_panel_never_opens_in_the_full_overlay_and_leaving_slim_hides_it(window: OverlayWindow, bus: EventBus) -> None:
    window.explain_btn.setChecked(True)
    bus.publish(AssistStartedEvent(job_id=21, kind="solve", question="Solving"))
    bus.publish(AssistDeltaEvent(job_id=21, delta=ANSWER))
    bus.publish(_done(21, ANSWER))
    QTest.qWait(_WAIT_MS)
    assert window.slim_panel.isHidden()
    assert "We need two indices" in window.assist_view.answer_browser.toPlainText()
    window.completion_btn.setChecked(True)
    QTest.qWait(_WAIT_MS)
    window._open_panel("walkthrough")
    assert not window.slim_panel.isHidden()
    assert "We need two indices" in window.slim_panel.browser.toPlainText()   # kept from full mode
    window.completion_btn.setChecked(False)
    QTest.qWait(_WAIT_MS)
    assert window.slim_panel.isHidden()


def test_explain_menu_entries(window: OverlayWindow, actions: FakeActions) -> None:
    window.explain_btn.setChecked(True)
    window.completion_btn.setChecked(True)
    window._rebuild_explain_menu()
    labels = [a.text() for a in window._explain_menu.actions() if a.text()]
    assert labels[:2] == ["Walkthrough (none yet)", "Explanation (none yet)"]
    assert "Explain now" in labels and "Rethink (solve again)" in labels
    acts = {a.text(): a for a in window._explain_menu.actions() if a.text()}
    acts["Explain now"].trigger()
    assert ("explain_now", ()) in actions.calls
    acts["Rethink (solve again)"].trigger()
    assert ("solve", ("",)) in actions.calls
    acts["Walkthrough (none yet)"].trigger()
    assert not window.slim_panel.isHidden()
    assert "no walkthrough yet" in window.slim_panel.browser.toPlainText()
    window._rebuild_explain_menu()
    assert "Close panel" in [a.text() for a in window._explain_menu.actions()]


def test_explanations_switched_off_mid_stream_never_reopens_the_panel(window: OverlayWindow, bus: EventBus) -> None:
    window.explain_btn.setChecked(True)
    window.completion_btn.setChecked(True)
    QTest.qWait(_WAIT_MS)
    bus.publish(AssistStartedEvent(job_id=31, kind="explain", question="Explanation"))
    bus.publish(AssistDeltaEvent(job_id=31, delta="- half"))
    QTest.qWait(_WAIT_MS)
    window.explain_btn.setChecked(False)                 # off while the explanation streams
    QTest.qWait(_WAIT_MS)
    assert window.slim_panel.isHidden()
    bus.publish(AssistDeltaEvent(job_id=31, delta=" way"))
    bus.publish(_done(31, "- half way"))
    QTest.qWait(_WAIT_MS)
    assert window.slim_panel.isHidden()
    assert not window.slim_panel.has("explanation") or window.slim_panel._texts["explanation"] == "- half"

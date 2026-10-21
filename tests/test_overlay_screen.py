"""Offline tests for the Screen tab: exocortex.ui.widgets.ScreenView and its wiring into
exocortex.ui.overlay.OverlayWindow.

Runs under QT_QPA_PLATFORM=offscreen (set before PySide6 is imported): no real screen, no
real windows, no LM Studio. The RegionPicker is a real QWidget but its ``begin()`` is
monkeypatched in the picker-flow tests so nothing tries to actually grab the desktop.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path
from typing import Any

import pytest

from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from exocortex.bus import (
    AssistDeltaEvent,
    AssistDoneEvent,
    AssistErrorEvent,
    AssistStartedEvent,
    EventBus,
    ScreenStatusEvent,
    ScreenTextEvent,
)
from exocortex.config import Config
from exocortex.ui import overlay as overlay_mod
from exocortex.ui.overlay import OverlayWindow
from exocortex.ui.theme import COLORS
from exocortex.ui.widgets import ScreenView

_WAIT_MS = 80


def wait_until(predicate, timeout_ms: int = 5000, step_ms: int = 20) -> bool:
    """Poll ``predicate`` while pumping the Qt event loop, for slower/loaded CI machines."""
    elapsed = 0
    while elapsed < timeout_ms:
        if predicate():
            return True
        QTest.qWait(step_ms)
        elapsed += step_ms
    return predicate()


# =====================================================================================
# fixtures
# =====================================================================================
@pytest.fixture(scope="session")
def qapp() -> QApplication:
    app = QApplication.instance() or QApplication(["exocortex-screen-tests"])
    return app


class FakeActions:
    """Records every call made through the EngineActions protocol (see exocortex.actions)."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...]]] = []
        self._screen_paused = False

    def answer_last(self) -> None:
        self.calls.append(("answer_last", ()))

    def ask(self, text: str) -> None:
        self.calls.append(("ask", (text,)))

    def generate_notes(self) -> None:
        self.calls.append(("generate_notes", ()))

    def cancel_assist(self) -> None:
        self.calls.append(("cancel_assist", ()))

    def set_auto(self, enabled: bool) -> None:
        self.calls.append(("set_auto", (enabled,)))

    def set_mic_muted(self, muted: bool) -> None:
        self.calls.append(("set_mic_muted", (muted,)))

    def rename_speaker(self, speaker_id: str, label: str) -> None:
        self.calls.append(("rename_speaker", (speaker_id, label)))

    def export_session(self) -> Path:
        self.calls.append(("export_session", ()))
        return Path(".")

    def new_session(self) -> None:
        self.calls.append(("new_session", ()))

    def set_model(self, model_id: str) -> None:
        self.calls.append(("set_model", (model_id,)))

    def list_models(self) -> list[str]:
        self.calls.append(("list_models", ()))
        return ["maple-preview"]

    def quit(self) -> None:
        self.calls.append(("quit", ()))

    # ---- screen reading ----
    def set_screen_mode(self, mode: str) -> None:
        self.calls.append(("set_screen_mode", (mode,)))

    def set_screen_region(self, x: int, y: int, w: int, h: int) -> None:
        self.calls.append(("set_screen_region", (x, y, w, h)))

    def capture_screen_now(self) -> None:
        self.calls.append(("capture_screen_now", ()))

    def suggest_now(self) -> None:
        self.calls.append(("suggest_now", ()))

    def checkpoint(self) -> None:
        self.calls.append(("checkpoint", ()))

    def code(self, focus: str = "") -> None:
        self.calls.append(("code", (focus,)))

    def set_auto_suggest(self, enabled: bool) -> None:
        self.calls.append(("set_auto_suggest", (enabled,)))

    def toggle_screen_paused(self) -> bool:
        self._screen_paused = not self._screen_paused
        self.calls.append(("toggle_screen_paused", ()))
        return self._screen_paused


@pytest.fixture()
def actions() -> FakeActions:
    return FakeActions()


@pytest.fixture()
def bus() -> EventBus:
    return EventBus()


@pytest.fixture()
def window(qapp: QApplication, bus: EventBus, actions: FakeActions):
    cfg = Config()
    win = OverlayWindow(cfg, bus, actions)
    win.show()
    qapp.processEvents()
    QTest.qWait(30)
    yield win
    win.close_silently()
    qapp.processEvents()


# =====================================================================================
# Screen tab exists and starts synced to cfg
# =====================================================================================
def test_screen_tab_present_and_synced_to_cfg(window: OverlayWindow) -> None:
    assert isinstance(window.screen_view, ScreenView)
    assert window._tab_buttons[3].text() == "Screen"
    # ScreenConfig() defaults to mode="auto" -> "Active window" is the checked mode button.
    assert window.screen_view._mode_buttons["auto"].isChecked()
    assert not window.screen_view._mode_buttons["off"].isChecked()
    assert window.screen_view.use_in_answers_cb.isChecked() == window._cfg.screen.use_in_answers
    assert window.screen_view.auto_suggest_cb.isChecked() == window._cfg.screen.auto_suggest
    assert (
        window.screen_view.answer_screen_questions_cb.isChecked()
        == window._cfg.screen.answer_screen_questions
    )


# =====================================================================================
# ScreenTextEvent / ScreenStatusEvent
# =====================================================================================
def test_screen_text_event_updates_view(window: OverlayWindow, bus: EventBus) -> None:
    bus.publish(
        ScreenTextEvent(
            mode="auto",
            source_name="Zoom Meeting",
            rect=(0, 0, 800, 600),
            text="line one\nline two",
            lines=("line one", "line two"),
            changed=True,
            ocr_ms=140.0,
            captured_at=3.0,
        )
    )
    QTest.qWait(_WAIT_MS)
    # The line count now lives on the collapsible toggle; the raw readout is hidden by default.
    assert window.screen_view.screen_text_toggle.text() == "Screen text (2 lines)"
    assert not window.screen_view.screen_text_toggle.isChecked()
    # Expanding it renders the latest OCR text.
    window.screen_view.screen_text_toggle.setChecked(True)
    window.screen_view.set_screen_text("line one\nline two", 2)
    assert window.screen_view.screen_text_edit.toPlainText() == "line one\nline two"
    assert window.screen_view.status_label.text() == "auto · Zoom Meeting · 140 ms"
    assert COLORS["ok"] in window.screen_view.status_label.styleSheet()


def test_screen_status_event_paused_shown_with_warn_colour(window: OverlayWindow, bus: EventBus) -> None:
    bus.publish(ScreenStatusEvent(state="paused"))
    QTest.qWait(_WAIT_MS)
    assert window.screen_view.status_label.text() == "paused"
    assert COLORS["warn"] in window.screen_view.status_label.styleSheet()
    assert window.status.screen_label.text() == "Screen paused"
    assert COLORS["warn"] in window.status.screen_label.styleSheet()


def test_screen_status_event_error_shown_with_err_colour(window: OverlayWindow, bus: EventBus) -> None:
    bus.publish(ScreenStatusEvent(state="error", detail="OCR engine unavailable"))
    QTest.qWait(_WAIT_MS)
    assert window.screen_view.status_label.text() == "OCR engine unavailable"
    assert COLORS["err"] in window.screen_view.status_label.styleSheet()
    assert window.status.screen_label.text() == "Screen !" and "OCR engine unavailable" in window.status.screen_label.toolTip()
    assert COLORS["err"] in window.status.screen_label.styleSheet()


def test_screen_status_event_active_shown_with_ok_colour(window: OverlayWindow, bus: EventBus) -> None:
    bus.publish(ScreenStatusEvent(state="active", detail="auto: Zoom Meeting (140 ms)"))
    QTest.qWait(_WAIT_MS)
    assert window.screen_view.status_label.text() == "auto: Zoom Meeting (140 ms)"
    assert COLORS["ok"] in window.screen_view.status_label.styleSheet()


# =====================================================================================
# AssistStartedEvent routing by kind
# =====================================================================================
def test_suggest_job_streams_into_screen_tab_and_switches_tab(window: OverlayWindow, bus: EventBus) -> None:
    assert window.stack.currentIndex() == 1   # the overlay opens on Assist
    bus.publish(AssistStartedEvent(job_id=50, kind="suggest", question=""))
    QTest.qWait(_WAIT_MS)
    assert window.stack.currentIndex() == 3

    bus.publish(AssistDeltaEvent(job_id=50, delta="- Ask about "))
    bus.publish(AssistDeltaEvent(job_id=50, delta="the budget"))
    assert wait_until(lambda: "budget" in window.screen_view.suggestion_browser.toPlainText())

    bus.publish(AssistDoneEvent(job_id=50, text="- Ask about the budget", ttft_s=0.2, tokens=10, tok_per_s=50.0))
    QTest.qWait(_WAIT_MS)
    assert "budget" in window.screen_view.suggestion_browser.toPlainText()
    assert "0.20" in window.screen_view.suggestion_footer.text()
    assert window.screen_view.suggestion_history_layout.count() == 1

    # A suggest job must never touch the Assist tab's own view.
    assert window.assist_view.answer_browser.toPlainText() == ""


def test_suggest_job_error_shown_in_screen_tab(window: OverlayWindow, bus: EventBus) -> None:
    bus.publish(AssistStartedEvent(job_id=60, kind="suggest", question=""))
    QTest.qWait(_WAIT_MS)
    bus.publish(AssistErrorEvent(job_id=60, message="LM Studio unreachable"))
    QTest.qWait(_WAIT_MS)
    assert "LM Studio unreachable" in window.screen_view.suggestion_browser.toPlainText()
    assert window.screen_view.suggestion_footer.text() == "error"


def test_screen_question_job_goes_to_assist_tab_with_prefixed_question(
    window: OverlayWindow, bus: EventBus
) -> None:
    bus.publish(AssistStartedEvent(job_id=51, kind="screen", question="what does this chart show?"))
    QTest.qWait(_WAIT_MS)
    assert window.stack.currentIndex() == 1
    assert window.assist_view.question_label.text().startswith("On screen:")
    assert "what does this chart show?" in window.assist_view.question_label.text()


def test_screen_question_job_delta_and_done_flow_through_assist_view(
    window: OverlayWindow, bus: EventBus
) -> None:
    bus.publish(AssistStartedEvent(job_id=52, kind="screen", question="what is this?"))
    bus.publish(AssistDeltaEvent(job_id=52, delta="It is a pricing table."))
    assert wait_until(lambda: "pricing table" in window.assist_view.answer_browser.toPlainText())
    bus.publish(AssistDoneEvent(job_id=52, text="It is a pricing table.", ttft_s=0.1, tokens=6, tok_per_s=30.0))
    QTest.qWait(_WAIT_MS)
    assert not window.assist_view.stop_btn.isVisible()
    # Must never leak into the Screen tab's suggestion panel.
    assert window.screen_view.suggestion_browser.toPlainText() == ""


def test_manual_assist_job_unaffected_by_screen_routing(window: OverlayWindow, bus: EventBus) -> None:
    """Regression: ordinary auto/manual/ask jobs must behave exactly as before."""
    bus.publish(AssistStartedEvent(job_id=1, kind="manual", question="what's the plan?"))
    QTest.qWait(_WAIT_MS)
    assert window.stack.currentIndex() == 1
    assert window.assist_view.question_label.text() == "Q: what's the plan?"


# =====================================================================================
# mode buttons
# =====================================================================================
@pytest.mark.parametrize("mode", ["off", "auto", "monitor"])
def test_mode_button_click_calls_set_screen_mode(window: OverlayWindow, actions: FakeActions, mode: str) -> None:
    QTest.mouseClick(window.screen_view._mode_buttons[mode], Qt.MouseButton.LeftButton)
    assert ("set_screen_mode", (mode,)) in actions.calls


def test_region_button_does_not_call_set_screen_mode_directly(
    window: OverlayWindow, actions: FakeActions, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The "Region..." button opens the picker; it must not call set_screen_mode("region")
    itself (that happens only once a region is actually picked)."""
    monkeypatch.setattr(overlay_mod.RegionPicker, "begin", lambda self: None)
    QTest.mouseClick(window.screen_view._mode_buttons["region"], Qt.MouseButton.LeftButton)
    assert not any(c == ("set_screen_mode", ("region",)) for c in actions.calls)


# =====================================================================================
# toggles
# =====================================================================================
def test_use_in_answers_toggle_writes_cfg(window: OverlayWindow) -> None:
    before = window._cfg.screen.use_in_answers
    window.screen_view.use_in_answers_cb.setChecked(not before)
    assert window._cfg.screen.use_in_answers == (not before)


def test_answer_screen_questions_toggle_writes_cfg(window: OverlayWindow) -> None:
    before = window._cfg.screen.answer_screen_questions
    window.screen_view.answer_screen_questions_cb.setChecked(not before)
    assert window._cfg.screen.answer_screen_questions == (not before)


def test_auto_suggest_toggle_calls_action(window: OverlayWindow, actions: FakeActions) -> None:
    window.screen_view.auto_suggest_cb.setChecked(not window.screen_view.auto_suggest_cb.isChecked())
    assert any(c[0] == "set_auto_suggest" for c in actions.calls)


# =====================================================================================
# Read (bottom control row) / Suggest now
# =====================================================================================
def test_read_button_calls_capture_screen_now(window: OverlayWindow, actions: FakeActions) -> None:
    # "Read" moved from the Screen tab to the overlay's always-visible bottom control row.
    QTest.mouseClick(window.read_btn, Qt.MouseButton.LeftButton)
    assert ("capture_screen_now", ()) in actions.calls


def test_suggest_now_button_calls_suggest_now(window: OverlayWindow, actions: FakeActions) -> None:
    QTest.mouseClick(window.screen_view.suggest_now_btn, Qt.MouseButton.LeftButton)
    assert ("suggest_now", ()) in actions.calls


# =====================================================================================
# Region picker flow
# =====================================================================================
def test_region_picker_selected_sets_region_and_mode(
    window: OverlayWindow, actions: FakeActions, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        overlay_mod.RegionPicker, "begin", lambda self: self.selected.emit(10, 20, 300, 200)
    )
    QTest.mouseClick(window.screen_view._mode_buttons["region"], Qt.MouseButton.LeftButton)
    QTest.qWait(_WAIT_MS)
    assert ("set_screen_region", (10, 20, 300, 200)) in actions.calls
    assert window.screen_view._mode_buttons["region"].isChecked()
    assert window.toast.isVisible()
    assert "300x200" in window.toast.text()


def test_region_picker_cancelled_shows_toast_and_reverts_mode(
    window: OverlayWindow, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(overlay_mod.RegionPicker, "begin", lambda self: self.cancelled.emit())
    QTest.mouseClick(window.screen_view._mode_buttons["region"], Qt.MouseButton.LeftButton)
    QTest.qWait(_WAIT_MS)
    assert window.toast.isVisible()
    assert "cancelled" in window.toast.text().lower()
    # cfg default mode is "auto" and nothing selected it away, so it reverts to "auto".
    assert window.screen_view._mode_buttons["auto"].isChecked()
    assert not window.screen_view._mode_buttons["region"].isChecked()


def test_region_picker_reused_across_calls(
    window: OverlayWindow, actions: FakeActions, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The picker is instantiated lazily once, not rebuilt on every open."""
    calls: list[int] = []
    monkeypatch.setattr(overlay_mod.RegionPicker, "begin", lambda self: calls.append(1))
    window._open_region_picker()
    window._open_region_picker()
    assert len(calls) == 2
    assert window._region_picker is not None


# =====================================================================================
# hotkey-bound methods (select_region, suggest_now, toggle_screen)
# =====================================================================================
def test_select_region_hotkey_method_opens_picker(
    window: OverlayWindow, monkeypatch: pytest.MonkeyPatch
) -> None:
    began: list[bool] = []
    monkeypatch.setattr(overlay_mod.RegionPicker, "begin", lambda self: began.append(True))
    window._open_region_picker()
    assert began == [True]


def test_toggle_screen_hotkey_method_pauses_and_toasts(window: OverlayWindow, actions: FakeActions) -> None:
    window._on_toggle_screen()
    assert ("toggle_screen_paused", ()) in actions.calls
    assert window.screen_view.status_label.text() == "paused"
    assert COLORS["warn"] in window.screen_view.status_label.styleSheet()
    assert window.toast.isVisible()
    assert "paused" in window.toast.text().lower()

    window._on_toggle_screen()
    assert window.toast.text().lower().count("resumed") == 1


def test_suggest_now_hotkey_binding_calls_action_directly(actions: FakeActions) -> None:
    """The suggest_now hotkey binds straight to actions.suggest_now (see _install_hotkeys)."""
    actions.suggest_now()
    assert ("suggest_now", ()) in actions.calls


# =====================================================================================
# menu entries
# =====================================================================================
def test_menu_has_screen_entries(window: OverlayWindow) -> None:
    """2026-10-21: the screen entries live in the View submenu (source as a radio list, pause)."""
    window._rebuild_menu()
    view = next(a.menu() for a in window._menu.actions() if a.menu() is not None and a.text() == "View")
    texts = [a.text() for a in view.actions() if a.text()]
    assert "Screen: pause/resume" in texts and "Screen source" in texts
    source = next(a.menu() for a in view.actions() if a.menu() is not None and a.text() == "Screen source")
    source_texts = [a.text() for a in source.actions()]
    assert source_texts[:2] == ["Off", "Active window"] and source_texts[-1].startswith("Region")
    assert any(t.startswith("Monitor 0 (primary)") for t in source_texts)
    assert not any("Screen region" in a.text() or "Screen: pause" in a.text() for a in window._menu.actions())

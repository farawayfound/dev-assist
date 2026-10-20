"""Additional offline tests for the Screen tab, written as an independent verifier pass over
exocortex.ui.overlay / exocortex.ui.widgets on top of tests/test_overlay_screen.py.

Covers gaps found while reviewing the "overlay-screen" module:
- The overlay's documented 340 px width floor (``overlay._MIN_WIDTH``) must actually be
  reachable; an idle, content-less status-bar label must not by itself push the window's
  Qt-computed minimum width past it.
- The status bar's "Screen: ..." label must reflect the *current* capture (mode / source /
  latency) as SPEC's UI bullet shows, not just state transitions (ScreenStatusEvent is only
  published on mode/source/pause changes and on errors, not on every capture).
- Pausing then resuming screen reading must not leave a stale "paused" (warn-coloured)
  indicator on the Screen tab once reading has actually resumed.
- A proactive "suggest" job switching the visible tab must never steal keyboard focus from
  whatever the user is doing in the AskBox.

Runs under QT_QPA_PLATFORM=offscreen (set before PySide6 is imported): no real screen, no
real windows, no LM Studio.
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

from exocortex.bus import AssistStartedEvent, EventBus, ScreenStatusEvent, ScreenTextEvent
from exocortex.config import Config
from exocortex.ui.overlay import OverlayWindow, _MIN_WIDTH
from exocortex.ui.theme import COLORS

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
# fixtures (mirrors tests/test_overlay_screen.py)
# =====================================================================================
@pytest.fixture(scope="session")
def qapp() -> QApplication:
    app = QApplication.instance() or QApplication(["exocortex-screen-extra-tests"])
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
# 340 px floor
# =====================================================================================
def test_idle_window_min_width_fits_the_340px_floor(window: OverlayWindow) -> None:
    """An idle, empty "Screen: ..." status-bar label must not by itself push the overlay's
    Qt-computed minimum width past its own documented floor (_MIN_WIDTH)."""
    window.root.layout().activate()
    assert window.root.minimumSizeHint().width() <= _MIN_WIDTH


def test_status_bar_screen_label_hidden_until_first_screen_event(window: OverlayWindow) -> None:
    assert not window.status.screen_label.isVisible()


# =====================================================================================
# status bar reflects the *current* capture, not just state transitions
# =====================================================================================
def test_screen_text_event_updates_status_bar_with_live_capture_info(
    window: OverlayWindow, bus: EventBus
) -> None:
    """ScreenStatusEvent only fires on mode/source/pause changes and errors (see
    screen/reader.py's contract), so the ongoing per-capture readout SPEC shows in the status
    bar ("Screen: auto · Zoom Meeting · 140 ms") has to come from ScreenTextEvent instead, or
    it would never update between foreground-window changes."""
    bus.publish(
        ScreenTextEvent(
            mode="auto",
            source_name="Zoom Meeting",
            rect=(0, 0, 800, 600),
            text="line one",
            lines=("line one",),
            changed=True,
            ocr_ms=140.0,
            captured_at=3.0,
        )
    )
    assert wait_until(lambda: window.status.screen_label.isVisible())
    assert window.status.screen_label.text() == "Screen 140 ms" and "Zoom Meeting" in window.status.screen_label.toolTip()
    assert COLORS["ok"] in window.status.screen_label.styleSheet()

    # A second capture (window title / latency changed) must refresh it again.
    bus.publish(
        ScreenTextEvent(
            mode="auto",
            source_name="Slides",
            rect=(0, 0, 800, 600),
            text="line two",
            lines=("line two",),
            changed=True,
            ocr_ms=90.0,
            captured_at=5.0,
        )
    )
    assert wait_until(
        lambda: window.status.screen_label.text() == "Screen 90 ms"
    )


def test_screen_status_event_formats_still_unchanged_for_non_active_states(
    window: OverlayWindow, bus: EventBus
) -> None:
    """Regression: the "active" formatting special-case must not affect paused/off/error."""
    bus.publish(ScreenStatusEvent(state="paused"))
    QTest.qWait(_WAIT_MS)
    assert window.status.screen_label.text() == "Screen paused"
    bus.publish(ScreenStatusEvent(state="error", detail="OCR engine unavailable"))
    QTest.qWait(_WAIT_MS)
    assert window.status.screen_label.text() == "Screen !"


# =====================================================================================
# pause / resume must not leave a stale indicator
# =====================================================================================
def test_toggle_screen_resume_clears_stale_paused_indicator(
    window: OverlayWindow, actions: FakeActions
) -> None:
    window._on_toggle_screen()  # pause
    assert window.screen_view.status_label.text() == "paused"
    assert COLORS["warn"] in window.screen_view.status_label.styleSheet()

    window._on_toggle_screen()  # resume
    assert window.screen_view.status_label.text() != "paused"
    assert COLORS["warn"] not in window.screen_view.status_label.styleSheet()


def test_toggle_screen_resume_in_off_mode_never_shows_paused_forever(
    window: OverlayWindow, actions: FakeActions
) -> None:
    """With mode "off" the reader never captures again, so nothing will ever refresh the tab's
    status label after a resume except ScreenView.set_paused itself - if that leaves the stale
    "paused" text, the Screen tab would claim "paused" forever after being resumed."""
    window.screen_view.set_mode("off")
    window._on_toggle_screen()  # pause
    window._on_toggle_screen()  # resume
    assert window.screen_view.status_label.text() != "paused"


# =====================================================================================
# suggest-triggered tab switch must never steal keyboard focus from the AskBox
# =====================================================================================
def test_suggest_job_tab_switch_does_not_steal_askbox_focus(
    window: OverlayWindow, bus: EventBus
) -> None:
    window.askbox.edit.setFocus(Qt.FocusReason.OtherFocusReason)
    window.askbox.edit.setText("what is the budget for")
    assert window.askbox.edit.hasFocus()

    bus.publish(AssistStartedEvent(job_id=99, kind="suggest", question=""))
    QTest.qWait(_WAIT_MS)

    assert window.stack.currentIndex() == 3  # switched, as designed
    assert window.askbox.edit.hasFocus()  # ... but never at the cost of the user's typing
    assert window.askbox.edit.text() == "what is the budget for"

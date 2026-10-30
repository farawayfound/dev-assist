"""Offscreen tests for the "Compact" control (2026-10-07): the Controls-row button, the ⋯ menu
entry and the completion-mode context-menu entry all call ``EngineActions.compact``. Mirrors
tests/test_overlay.py's fixtures; reuses the catch-all FakeActions."""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from typing import Any

import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from exocortex.bus import EventBus
from exocortex.config import Config
from exocortex.ui.overlay import OverlayWindow


class CompactActions:
    """Catch-all FakeActions whose ``compact`` returns a configurable character count."""

    def __init__(self, dropped: int = 123) -> None:
        self.calls: list[tuple[str, tuple[Any, ...]]] = []
        self._dropped = dropped

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)

        def f(*args: Any, **kw: Any) -> Any:
            self.calls.append((name, args))
            if name == "compact":
                return self._dropped
            if name == "list_models":
                return []
            if name == "toggle_screen_paused":
                return False
            return None

        return f


@pytest.fixture(scope="session")
def qapp() -> QApplication:
    return QApplication.instance() or QApplication(["exocortex-compact-tests"])


@pytest.fixture()
def actions() -> CompactActions:
    return CompactActions()


@pytest.fixture()
def window(qapp: QApplication, actions: CompactActions):
    win = OverlayWindow(Config(), EventBus(), actions)
    win.show()
    qapp.processEvents()
    QTest.qWait(30)
    yield win
    win.close_silently()
    qapp.processEvents()


def test_compact_button_exists_after_checkpoint(window: OverlayWindow) -> None:
    row = window.control_row.layout()
    widgets = [row.itemAt(i).widget() for i in range(row.count())]
    labels = [w.text() for w in widgets if w is not None]
    assert "Compact" in labels
    assert labels.index("Compact") == labels.index("Checkpoint") + 1


def test_compact_button_calls_actions_compact(window: OverlayWindow, actions: CompactActions) -> None:
    QTest.mouseClick(window.compact_btn, Qt.MouseButton.LeftButton)
    assert ("compact", ()) in actions.calls


def test_menu_compact_entry_calls_actions_compact(window: OverlayWindow, actions: CompactActions) -> None:
    window._rebuild_menu()
    entry = next(a for a in window._menu.actions() if a.text() == "Compact context now")
    entry.trigger()
    assert ("compact", ()) in actions.calls


def test_compact_toast_when_compaction_off(window: OverlayWindow, actions: CompactActions) -> None:
    actions._dropped = 0
    window._cfg.context.compaction = "off"
    window._on_compact()
    assert window.toast.isVisibleTo(window)
    assert "off" in window.toast.text().lower()

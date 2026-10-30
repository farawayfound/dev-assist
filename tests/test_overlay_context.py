"""Offscreen tests for the overlay's context-library wiring (2026-10-06): the "Context
documents…" action in both menus (it opens the Models window on its Context tab) and the
forwarding of ContextLibraryEvent / ContextUsageEvent to the open Models dialog. Mirrors
tests/test_overlay_models.py."""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from typing import Any

import pytest
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from exocortex.bus import (
    ContextLibraryEvent,
    ContextUsageEvent,
    EventBus,
    ModelLibraryEvent,
)
from exocortex.config import Config
from exocortex.ui.overlay import OverlayWindow
from tests.test_overlay_completion import FakeActions

_WAIT_MS = 80
_ACTION = "Context documents…"


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


class _FakeModelsDialog:
    """Stand-in for ``ModelsDialog``: records ``on_event`` calls and the tab it was told to show."""

    def __init__(self) -> None:
        self.events: list[Any] = []
        self.show_tab_calls: list[str] = []
        self.load_calls = 0
        self.shown = 0

    def on_event(self, ev: object) -> None:
        self.events.append(ev)

    def load(self) -> None:
        self.load_calls += 1

    def show_tab(self, name: str) -> None:
        self.show_tab_calls.append(name)

    def show(self) -> None:
        self.shown += 1

    def raise_(self) -> None:
        pass

    def activateWindow(self) -> None:
        pass


# =====================================================================================
# the menus
# =====================================================================================

def test_both_menus_have_a_context_documents_action_after_models(window: OverlayWindow) -> None:
    window._rebuild_menu()
    texts = [a.text() for a in window._menu.actions()]
    assert _ACTION in texts
    assert texts[texts.index("Models…") + 1] == _ACTION

    window._rebuild_context_menu()
    ctx_texts = [a.text() for a in window._context_menu.actions()]
    assert _ACTION in ctx_texts
    assert ctx_texts[ctx_texts.index("Models…") + 1] == _ACTION


def test_main_menu_action_opens_the_models_window_on_context(window: OverlayWindow) -> None:
    window._rebuild_menu()
    act = next(a for a in window._menu.actions() if a.text() == _ACTION)
    fake = _FakeModelsDialog()
    window._models_dialog = fake
    act.trigger()
    assert fake.load_calls == 1 and fake.shown == 1
    assert fake.show_tab_calls == ["Context"]


def test_slim_bar_menu_action_opens_the_models_window_on_context(window: OverlayWindow) -> None:
    window._rebuild_context_menu()
    act = next(a for a in window._context_menu.actions() if a.text() == _ACTION)
    fake = _FakeModelsDialog()
    window._models_dialog = fake
    act.trigger()
    assert fake.load_calls == 1 and fake.shown == 1
    assert fake.show_tab_calls == ["Context"]


def test_the_plain_models_action_still_opens_without_a_tab(window: OverlayWindow) -> None:
    window._rebuild_menu()
    act = next(a for a in window._menu.actions() if a.text() == "Models…")
    fake = _FakeModelsDialog()
    window._models_dialog = fake
    act.trigger()
    assert fake.shown == 1 and fake.show_tab_calls == []


def test_context_action_builds_a_real_dialog_on_first_use(window: OverlayWindow) -> None:
    """No fake pre-installed: the action builds the real ``ModelsDialog`` (whose ``show_tab``
    ignores a tab name it does not know, so this also holds before the Context tab exists)."""
    window._rebuild_menu()
    act = next(a for a in window._menu.actions() if a.text() == _ACTION)
    assert window._models_dialog is None
    act.trigger()
    dlg = window._models_dialog
    assert dlg is not None and dlg.isVisible()
    dlg.close()
    QTest.qWait(30)


# =====================================================================================
# event forwarding
# =====================================================================================

def test_context_events_are_forwarded_to_the_open_dialog(window: OverlayWindow, bus: EventBus) -> None:
    fake = _FakeModelsDialog()
    window._models_dialog = fake

    library = ContextLibraryEvent(reason="added", doc_id="abc123def456")
    usage = ContextUsageEvent(job_id=3, kind="assist", slot_tokens=4096, budget_tokens=2900, used_tokens=2100)
    bus.publish(library)
    bus.publish(usage)
    QTest.qWait(_WAIT_MS)

    assert library in fake.events
    assert usage in fake.events


def test_error_library_events_are_forwarded_too(window: OverlayWindow, bus: EventBus) -> None:
    fake = _FakeModelsDialog()
    window._models_dialog = fake
    ev = ContextLibraryEvent(reason="error: x.docx is not a zip file")
    bus.publish(ev)
    QTest.qWait(_WAIT_MS)
    assert fake.events == [ev]


def test_context_events_do_not_open_the_dialog_or_crash_before_it_exists(window: OverlayWindow, bus: EventBus) -> None:
    assert window._models_dialog is None
    bus.publish(ContextLibraryEvent(reason="added", doc_id="x"))
    bus.publish(ContextUsageEvent(job_id=1, kind="ask"))
    QTest.qWait(_WAIT_MS)
    assert window._models_dialog is None


def test_context_events_do_not_disturb_model_event_forwarding(window: OverlayWindow, bus: EventBus) -> None:
    fake = _FakeModelsDialog()
    window._models_dialog = fake
    model_ev = ModelLibraryEvent(reason="loaded", loaded_key="k")
    ctx_ev = ContextLibraryEvent(reason="removed", doc_id="d")
    bus.publish(ctx_ev)
    bus.publish(model_ev)
    QTest.qWait(_WAIT_MS)
    assert fake.events == [ctx_ev, model_ev]
    assert fake.show_tab_calls == []                 # forwarding never changes the visible tab


def test_a_usage_event_does_not_toast(window: OverlayWindow, bus: EventBus) -> None:
    """One usage event per assist job: it updates the Context tab's panel silently."""
    fake = _FakeModelsDialog()
    window._models_dialog = fake
    before = window.toast.text()
    bus.publish(ContextUsageEvent(job_id=1, kind="assist"))
    QTest.qWait(_WAIT_MS)
    assert window.toast.text() == before

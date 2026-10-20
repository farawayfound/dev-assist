"""Offscreen tests for the overlay's "Models…" window wiring: the menu entries, opening /
reusing the dialog, and forwarding the model-manager events (2026-09-27, portable-install)."""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from typing import Any

import pytest
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from exocortex.bus import (
    ComponentStatusEvent,
    EventBus,
    ModelInfoEvent,
    ModelLibraryEvent,
    ModelPullEvent,
    ModelSearchEvent,
)
from exocortex.config import Config
from exocortex.ui.overlay import OverlayWindow
from tests.test_overlay_completion import FakeActions

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


class _FakeModelsDialog:
    """Stand-in for ``ModelsDialog``: records ``on_event`` calls and the tab it was told to
    show, without building any real Qt window."""

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
# the menus: "Models…" instead of the old "Choose model" submenu
# =====================================================================================

def test_both_menus_have_a_models_action_and_no_choose_model_submenu(window: OverlayWindow) -> None:
    window._rebuild_menu()
    menu_texts = [a.text() for a in window._menu.actions()]
    assert "Models…" in menu_texts
    assert not any("Choose model" in t for t in menu_texts)
    # right after "Settings…", per the brief
    settings_idx = menu_texts.index("Settings…")
    assert menu_texts[settings_idx + 1] == "Models…"

    window._rebuild_context_menu()
    ctx_texts = [a.text() for a in window._context_menu.actions()]
    assert "Models…" in ctx_texts
    assert not any("Choose model" in t for t in ctx_texts)
    ctx_settings_idx = ctx_texts.index("Settings…")
    assert ctx_texts[ctx_settings_idx + 1] == "Models…"


def test_models_action_in_main_menu_opens_the_window(window: OverlayWindow) -> None:
    window._rebuild_menu()
    act = next(a for a in window._menu.actions() if a.text() == "Models…")
    fake = _FakeModelsDialog()
    window._models_dialog = fake
    act.trigger()
    assert fake.load_calls == 1 and fake.shown == 1


def test_models_action_in_context_menu_opens_the_window(window: OverlayWindow) -> None:
    window._rebuild_context_menu()
    act = next(a for a in window._context_menu.actions() if a.text() == "Models…")
    fake = _FakeModelsDialog()
    window._models_dialog = fake
    act.trigger()
    assert fake.load_calls == 1 and fake.shown == 1


# =====================================================================================
# open_models(): reuses one dialog, honours ``tab``
# =====================================================================================

def test_open_models_reuses_one_dialog_and_honours_tab(window: OverlayWindow) -> None:
    fake = _FakeModelsDialog()
    window._models_dialog = fake
    window.open_models()
    assert fake.load_calls == 1 and fake.shown == 1 and fake.show_tab_calls == []

    window.open_models("Discover")
    assert window._models_dialog is fake                 # reused, not rebuilt
    assert fake.load_calls == 2 and fake.shown == 2
    assert fake.show_tab_calls == ["Discover"]


def test_open_models_constructs_a_real_dialog_on_first_call(window: OverlayWindow) -> None:
    """Without a fake pre-installed, ``open_models`` builds the real ``ModelsDialog`` -- exactly
    the way ``open_settings`` builds ``SettingsDialog`` -- and reuses that same instance after."""
    assert window._models_dialog is None
    window.open_models()
    dlg = window._models_dialog
    assert dlg is not None and dlg.isVisible()
    window.open_models()
    assert window._models_dialog is dlg                   # reused
    dlg.close()
    QTest.qWait(30)


# =====================================================================================
# event forwarding: the five model-manager event types reach the open dialog
# =====================================================================================

def test_model_events_are_forwarded_to_the_open_dialog(window: OverlayWindow, bus: EventBus) -> None:
    fake = _FakeModelsDialog()
    window._models_dialog = fake

    pull = ModelPullEvent(repo_id="unsloth/gemma", state="downloading", done_bytes=10, total_bytes=100)
    library = ModelLibraryEvent(reason="pull-finished", loaded_key="unsloth/gemma")
    search = ModelSearchEvent(request_id=1, query="gemma")
    info = ModelInfoEvent(request_id=1, repo_id="unsloth/gemma")
    llm_status = ComponentStatusEvent(component="llm", state="ready", detail="loaded")

    for ev in (pull, library, search, info, llm_status):
        bus.publish(ev)
    QTest.qWait(_WAIT_MS)

    assert pull in fake.events
    assert library in fake.events
    assert search in fake.events
    assert info in fake.events
    assert llm_status in fake.events


def test_events_are_not_forwarded_before_the_dialog_ever_opened(window: OverlayWindow, bus: EventBus) -> None:
    assert window._models_dialog is None
    bus.publish(ModelPullEvent(repo_id="unsloth/gemma", state="queued"))
    QTest.qWait(_WAIT_MS)
    assert window._models_dialog is None                  # never crashes, never opens the window


# =====================================================================================
# "empty-library": open on Discover once per session, with a toast
# =====================================================================================

def test_empty_library_opens_the_dialog_on_discover_once(window: OverlayWindow, bus: EventBus) -> None:
    fake = _FakeModelsDialog()
    window._models_dialog = fake

    bus.publish(ModelLibraryEvent(reason="empty-library"))
    QTest.qWait(_WAIT_MS)
    assert fake.show_tab_calls == ["Discover"]
    assert "No model yet" in window.toast.text()
    assert "Models" in window.toast.text()

    bus.publish(ModelLibraryEvent(reason="empty-library"))
    QTest.qWait(_WAIT_MS)
    assert fake.show_tab_calls == ["Discover"]             # not opened a second time this session
    # the event itself is still forwarded every time, empty-library or not
    assert sum(1 for ev in fake.events if isinstance(ev, ModelLibraryEvent)) == 2


def test_empty_library_opens_a_real_dialog_when_none_was_open(window: OverlayWindow, bus: EventBus) -> None:
    """Without any fake pre-installed, the first "empty-library" event builds the real dialog
    (via ``open_models("Discover")``) instead of forwarding to a dialog that does not exist yet."""
    assert window._models_dialog is None
    bus.publish(ModelLibraryEvent(reason="empty-library"))
    QTest.qWait(_WAIT_MS)
    assert window._models_dialog is not None
    assert window._models_prompted is True
    window._models_dialog.close()
    QTest.qWait(30)

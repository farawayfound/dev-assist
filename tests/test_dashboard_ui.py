"""Dashboard mode in the overlay (2026-10-20): the DashboardView's columns, cards and actions,
the title-bar toggle (geometry in and out), the routing of "dashboard" jobs and snapshot events,
the menu, the hotkey, the settings dialog's Dashboard and Context tabs, the slim bar's Context
menu toggles. Offscreen QPA, FakeActions as in the other overlay tests."""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QRect
from PySide6.QtGui import QGuiApplication
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QCheckBox, QLineEdit, QMenu, QSpinBox

from exocortex.bus import (
    AssistDoneEvent,
    AssistErrorEvent,
    AssistStartedEvent,
    DashboardItem,
    DashboardModeEvent,
    DashboardUpdatedEvent,
    EventBus,
    NotesUpdatedEvent,
)
from exocortex.config import Config
from exocortex.ui import dashboard as dash_mod
from exocortex.ui import overlay as overlay_mod
from exocortex.ui.dashboard import DashboardView
from exocortex.ui.overlay import OverlayWindow, _MIN_WIDTH
from exocortex.ui.settings import FIELDS, SettingsDialog
from tests.test_overlay import _SpyHotkeys
from tests.test_overlay_completion import FakeActions

_WAIT_MS = 60


@pytest.fixture(scope="session")
def qapp() -> QApplication:
    return QApplication.instance() or QApplication(["exocortex-dashboard-tests"])


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


def _items() -> tuple[DashboardItem, ...]:
    return (
        DashboardItem(id=1, kind="question", text="Is the cache per tenant?", created_at=10.0, updated_at=10.0),
        DashboardItem(id=2, kind="question", text="Who owns the migration?", status="answered", answer="Sam", created_at=5.0, updated_at=70.0),
        DashboardItem(id=3, kind="blindspot", text="No test for an empty payload", created_at=12.0, updated_at=12.0, source="user"),
        DashboardItem(id=4, kind="nextstep", text="Write the failing test", created_at=13.0, updated_at=13.0),
        DashboardItem(id=5, kind="prompt", text="/repro-fix export() 500", created_at=14.0, updated_at=14.0),
        DashboardItem(id=6, kind="nextstep", text="Old step", status="answered", answer="", created_at=1.0, updated_at=60.0),
    )


# --------------------------------------------------------------------------------------------
# the view
# --------------------------------------------------------------------------------------------


def test_view_renders_columns_cards_and_status(qapp: QApplication) -> None:
    cfg = Config()
    view = DashboardView(cfg)
    try:
        assert view.visible_columns() == ["notes", "questions", "blindspots", "nextsteps"]
        assert "No open questions yet" in view.columns["questions"].empty_label.text()
        view.set_snapshot(_items(), "## Summary\nhello", 42.0, 100.0, "updated: +2 new")
        q = view.columns["questions"]
        assert [c.item.id for c in q.cards()] == [1] and [c.item.id for c in q.resolved_cards()] == [2]
        assert q.count_label.text() == "1 open · 1 done" and q.resolved_toggle.text() == "Answered (1)"
        assert not q.empty_label.isVisibleTo(view) and q.resolved_toggle.isVisibleTo(view) and not q.resolved_container.isVisibleTo(view)
        q.resolved_toggle.setChecked(True)
        assert q.resolved_container.isVisibleTo(view)
        answered = q.resolved_cards()[0]
        assert "answered 01:10 — Sam" in answered.status_label.text() and "reopen" in answered._buttons and "answer" not in answered._buttons
        open_card = q.cards()[0]
        assert "since 00:10" in open_card.status_label.text() and set(open_card._buttons) == {"answer", "dismiss"}
        b = view.columns["blindspots"]
        assert "added by you" in b.cards()[0].status_label.text()
        n = view.columns["nextsteps"]
        assert [c.item.id for c in n.cards()] == [4, 5] and n.resolved_toggle.text() == "Done (1)"
        prompt_card = n.cards()[1]
        assert "copy" in prompt_card._buttons and "monospace" in prompt_card.text_label.styleSheet()
        assert "hello" in view.notes_col.browser.toPlainText() and view.notes_col.updated_label.text() == "updated 00:42"
        assert view.status_label.text() == "updated 01:40 · updated: +2 new"
        view.begin_update(9, "Dashboard refresh")
        assert view.updating and view.status_label.text() == "dashboard refresh…"
        view.finish_update(8)                     # another job: ignored
        assert view.updating
        view.finish_update(9, cancelled=True)
        assert not view.updating and "refresh cancelled" in view.status_label.text()
        view.begin_update(10)
        view.update_error(10, "LLM down")
        assert view.status_label.text() == "error: LLM down"
        view.set_notes("new notes", 50.0)
        assert view.notes_col.markdown == "new notes"
        cfg.dashboard.show_notes = False
        cfg.dashboard.show_blindspots = False
        view.apply_config()
        assert view.visible_columns() == ["questions", "nextsteps"]
        view.clear()
        assert view.items == () and view.columns["questions"].cards() == [] and view.status_label.text() == ""
    finally:
        view.shutdown()


def test_view_actions_emit_signals(qapp: QApplication, monkeypatch: pytest.MonkeyPatch) -> None:
    view = DashboardView(Config())
    got: list = []
    view.item_action.connect(lambda i, a, t: got.append(("item", i, a, t)))
    view.item_added.connect(lambda k, t: got.append(("add", k, t)))
    view.refresh_requested.connect(lambda s: got.append(("refresh", tuple(s))))
    view.notes_requested.connect(lambda: got.append(("notes",)))
    try:
        view.set_snapshot(_items(), "", None, 1.0)
        q = view.columns["questions"]
        monkeypatch.setattr(dash_mod.QInputDialog, "getText", staticmethod(lambda *a, **kw: ("because Sam said so", True)))
        q.cards()[0]._buttons["answer"].click()
        q.cards()[0]._buttons["dismiss"].click()
        q.resolved_toggle.setChecked(True)
        q.resolved_cards()[0]._buttons["reopen"].click()
        n = view.columns["nextsteps"]
        n.cards()[1]._buttons["copy"].click()
        assert QGuiApplication.clipboard().text() == "/repro-fix export() 500"
        q.add_btn.click()
        monkeypatch.setattr(dash_mod.QInputDialog, "getText", staticmethod(lambda *a, **kw: ("/check", True)))
        n.add_btn.click()
        monkeypatch.setattr(dash_mod.QInputDialog, "getText", staticmethod(lambda *a, **kw: ("", False)))
        q.add_btn.click()                                               # cancelled: nothing
        q.cards()[0]._buttons["answer"].click()                         # cancelled: nothing
        q.refresh_btn.click()
        view.refresh_all_btn.click()
        view.notes_col.refresh_btn.click()
        assert got == [
            ("item", 1, "answer", "because Sam said so"), ("item", 1, "dismiss", ""), ("item", 2, "reopen", ""),
            ("add", "question", "because Sam said so"), ("add", "prompt", "/check"),
            ("refresh", ("questions",)), ("refresh", ()), ("notes",),
        ]
    finally:
        view.shutdown()


# --------------------------------------------------------------------------------------------
# the overlay
# --------------------------------------------------------------------------------------------


def test_toggle_enters_dashboard_mode_and_restores_the_window(window: OverlayWindow, actions: FakeActions) -> None:
    full = QRect(window.geometry())
    assert not window.dashboard_mode and window.dashboard_view is not None and window.dashboard_view.isHidden()
    window.dashboard_btn.setChecked(True)
    QTest.qWait(_WAIT_MS)
    assert window.dashboard_mode and window.dashboard_view.isVisibleTo(window)
    assert window.stack.isHidden() and window.tabstrip.isHidden()
    assert window.askbox.isVisibleTo(window) and window.control_row.isVisibleTo(window) and window.status.isVisibleTo(window)
    screen_w = QGuiApplication.primaryScreen().availableGeometry().width()
    assert window.width() == min(max(window._cfg.dashboard.width, full.width()), screen_w)
    assert window.geometry().right() <= full.right() + 1 or window.width() == screen_w
    assert ("set_dashboard_mode", (True,)) in actions.calls
    window._save_geometry()
    assert (window._cfg.ui.width, window._cfg.ui.height) == (full.width(), full.height())   # the normal size persists
    assert window._cfg.dashboard.width == window.width()
    window.dashboard_btn.setChecked(False)
    QTest.qWait(_WAIT_MS)
    assert not window.dashboard_mode and window.geometry() == full
    assert window.stack.isVisibleTo(window) and window.tabstrip.isVisibleTo(window) and window.dashboard_view.isHidden()
    assert ("set_dashboard_mode", (False,)) in actions.calls
    window.root.layout().activate()
    assert window.root.minimumSizeHint().width() <= _MIN_WIDTH                 # the ninth icon button fits the floor


def test_dashboard_events_route_to_the_view_without_switching_tabs(window: OverlayWindow, bus: EventBus, actions: FakeActions) -> None:
    window._show_tab(0)
    bus.publish(AssistStartedEvent(job_id=5, kind="dashboard", question="Dashboard refresh"))
    window._poll()
    assert window.stack.currentIndex() == 0 and window.dashboard_view.updating
    assert window.assist_view.question_label.text() == "Q: –"
    bus.publish(DashboardUpdatedEvent(items=_items(), notes="n", notes_at=1.0, generated_at=90.0, auto=True, detail="updated: +1 new"))
    bus.publish(AssistDoneEvent(job_id=5, text="...", ttft_s=0.1, tokens=3, tok_per_s=30.0))
    window._poll()
    assert not window.dashboard_view.updating
    assert [c.item.id for c in window.dashboard_view.columns["questions"].cards()] == [1]
    assert window.dashboard_view.status_label.text() == "updated 01:30 · updated: +1 new"
    bus.publish(AssistStartedEvent(job_id=6, kind="dashboard", question="Dashboard refresh"))
    bus.publish(AssistErrorEvent(job_id=6, message="boom"))
    window._poll()
    assert window.dashboard_view.status_label.text() == "error: boom"
    bus.publish(NotesUpdatedEvent(markdown="## Summary\nfresh", generated_at=99.0))
    window._poll()
    assert window.dashboard_view.notes_col.markdown == "## Summary\nfresh"
    bus.publish(DashboardModeEvent(enabled=True))
    window._poll()
    assert window.dashboard_mode and window.dashboard_btn.isChecked()
    assert not any(c[0] == "set_dashboard_mode" for c in actions.calls)        # mirrored, not echoed back
    bus.publish(DashboardModeEvent(enabled=False))
    window._poll()
    assert not window.dashboard_mode


def test_view_signals_reach_the_engine_through_the_overlay(window: OverlayWindow, actions: FakeActions) -> None:
    view = window.dashboard_view
    view.item_action.emit(3, "dismiss", "")
    view.item_added.emit("question", "Who?")
    view.refresh_requested.emit(("questions",))
    view.refresh_requested.emit(())
    view.notes_requested.emit()
    assert ("dashboard_item", (3, "dismiss", "")) in actions.calls
    assert ("dashboard_add", ("question", "Who?")) in actions.calls
    assert ("dashboard_refresh", (("questions",),)) in actions.calls and ("dashboard_refresh", ((),)) in actions.calls
    assert actions.calls.count(("generate_notes", ())) == 2                   # Refresh all + the notes column


def test_completion_mode_leaves_dashboard_mode_first(window: OverlayWindow, actions: FakeActions) -> None:
    full = QRect(window.geometry())
    window.dashboard_btn.setChecked(True)
    QTest.qWait(_WAIT_MS)
    window.completion_btn.setChecked(True)
    QTest.qWait(_WAIT_MS)
    assert not window.dashboard_mode and window._completion_mode and window.slim.isVisibleTo(window)
    assert window._normal_geometry == full                                     # the slim bar remembers the NORMAL size
    window.completion_btn.setChecked(False)
    QTest.qWait(_WAIT_MS)
    assert window.geometry() == full and not window.dashboard_mode
    window.dashboard_btn.setChecked(True)
    QTest.qWait(_WAIT_MS)
    window._on_reset()
    assert window.dashboard_view.items == ()


def test_menu_and_hotkey_toggle_dashboard_mode(qapp: QApplication, bus: EventBus, actions: FakeActions, monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.test_overlay import _StubWin32

    monkeypatch.setattr(overlay_mod, "win32", _StubWin32())
    monkeypatch.setattr(overlay_mod, "_is_windows_platform", lambda: True)
    monkeypatch.setattr(overlay_mod, "GlobalHotkeys", _SpyHotkeys)
    _SpyHotkeys.instances = []
    win = OverlayWindow(Config(), bus, actions)
    try:
        spy = _SpyHotkeys.instances[-1]
        assert "ctrl+alt+d" in spy.bindings
        spy.bindings["ctrl+alt+d"]()
        assert win.dashboard_mode
        win._rebuild_menu()
        entry = next(a for a in win._menu.actions() if a.text().startswith("Dashboard mode"))
        assert entry.text() == "Dashboard mode\tCtrl+Alt+D" and entry.isCheckable() and entry.isChecked()
        entry.trigger()
        assert not win.dashboard_mode
        assist = next(a.menu() for a in win._menu.actions() if a.menu() is not None and a.text() == "Assist")
        next(a for a in assist.actions() if a.text() == "Refresh the dashboard now").trigger()
        assert ("dashboard_refresh", ((),)) in actions.calls
    finally:
        win.close_silently()
        qapp.processEvents()


def test_settings_dialog_has_the_dashboard_and_context_tabs_and_the_overlay_mirrors_them(window: OverlayWindow, actions: FakeActions, qapp: QApplication) -> None:
    dashboard_keys = [f.key for f in FIELDS["Dashboard"]]
    for key in ("enabled", "track", "refresh_interval_s", "min_new_words", "max_open_items", "max_new_items",
                "suggest_prompts", "show_resolved", "resolved_keep", "show_notes", "show_questions", "show_blindspots",
                "show_nextsteps", "width", "view_refresh_s", "persist"):
        assert key in dashboard_keys, key
    context_keys = [f.key for f in FIELDS["Context"]]
    for key in ("repo_root", "repo_search", "repo_max_files", "repo_max_chars", "toolkit", "toolkit_lookup",
                "toolkit_in_answers", "toolkit_user_layer", "toolkit_dir", "toolkit_max_chars", "rescan_s"):
        assert key in context_keys, key
    assert list(FIELDS).index("Dashboard") == list(FIELDS).index("Notes") + 1
    assert list(FIELDS).index("Context") == list(FIELDS).index("Dashboard") + 1
    dlg = SettingsDialog(Config(), None)
    try:
        assert dlg.show_tab("Dashboard") and dlg.show_tab("Context")
        assert isinstance(dlg._widgets["context.repo_root"], QLineEdit)
        assert isinstance(dlg._widgets["context.toolkit"], QCheckBox)
        assert isinstance(dlg._widgets["dashboard.refresh_interval_s"], QSpinBox)
    finally:
        dlg.close()
        qapp.processEvents()
    window.open_settings("Dashboard")
    qapp.processEvents()
    dlg = window._settings
    assert dlg is not None and dlg.tabs.tabText(dlg.tabs.currentIndex()) == "Dashboard"
    dlg._widgets["dashboard.enabled"].setChecked(True)
    dlg._widgets["dashboard.show_notes"].setChecked(False)
    dlg.apply_btn.click()
    QTest.qWait(_WAIT_MS)
    assert window.dashboard_mode and window.dashboard_view.visible_columns() == ["questions", "blindspots", "nextsteps"]
    assert not any(c[0] == "set_dashboard_mode" for c in actions.calls)        # the engine reacts through apply_settings
    applied = [c for c in actions.calls if c[0] == "apply_settings"][-1][1][0]
    assert applied == {"dashboard.enabled": True, "dashboard.show_notes": False}


def test_the_slim_bar_context_menu_switches_the_lookups(window: OverlayWindow, actions: FakeActions) -> None:
    window._cfg.context.repo_search = False
    window._rebuild_context_menu()
    acts = {a.text(): a for a in window._context_menu.actions() if a.text()}
    assert not acts["Repository lookup (answers, dashboard)"].isChecked()
    assert acts["Toolkit catalog (answers, dashboard)"].isChecked()
    acts["Repository lookup (answers, dashboard)"].setChecked(True)
    acts["Toolkit catalog (answers, dashboard)"].setChecked(False)
    assert ("set_completion_context", ({"repo": True},)) in actions.calls
    assert ("set_completion_context", ({"toolkit": False},)) in actions.calls


def test_starting_in_dashboard_mode_opens_the_board(qapp: QApplication, bus: EventBus, actions: FakeActions) -> None:
    cfg = Config()
    cfg.dashboard.enabled = True
    win = OverlayWindow(cfg, bus, actions)
    try:
        win.show()
        qapp.processEvents()
        QTest.qWait(_WAIT_MS)
        assert win.dashboard_mode and win.dashboard_view.isVisibleTo(win)
    finally:
        win.close_silently()
        qapp.processEvents()

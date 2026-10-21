"""Dashboard mode in the overlay (2026-10-20, reworked 2026-10-21): the DashboardView's six
columns -- the action items as a task tree with the prompts under them, the goals grouped by
task, the constraints -- its cards and actions, the title-bar toggle (geometry in and out), the
routing of "dashboard" jobs and snapshot events, the ⋯ menu's View submenu (the current view's
options, the screen source, the repository), the hotkey, the settings dialog's Dashboard and
Context tabs, the slim bar's Context menu toggles. Offscreen QPA, FakeActions as in the other
overlay tests."""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QPoint, QRect
from PySide6.QtGui import QContextMenuEvent, QGuiApplication
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QCheckBox, QLineEdit, QMenu, QSpinBox

from exocortex.bus import (
    AssistDoneEvent,
    AssistErrorEvent,
    AssistStartedEvent,
    ComponentStatusEvent,
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
_ALL_COLUMNS = ["notes", "nextsteps", "goals", "constraints", "questions", "blindspots"]


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
        DashboardItem(id=4, kind="nextstep", text="Task 1: export() rate limiting", created_at=13.0, updated_at=13.0),
        DashboardItem(id=7, kind="nextstep", text="Write the failing test", created_at=14.0, updated_at=14.0, parent_id=4),
        DashboardItem(id=5, kind="prompt", text="/repro-fix export() 500", created_at=15.0, updated_at=15.0, parent_id=7),
        DashboardItem(id=6, kind="nextstep", text="Old step", status="answered", answer="", created_at=1.0, updated_at=60.0),
        DashboardItem(id=8, kind="goal", text="an empty tenant gets 200", created_at=16.0, updated_at=16.0, parent_id=4),
        DashboardItem(id=9, kind="goal", text="every response is JSON", created_at=17.0, updated_at=17.0),
        DashboardItem(id=10, kind="constraint", text="no new dependencies", created_at=18.0, updated_at=18.0),
    )


def _submenu(menu: QMenu, title: str) -> QMenu:
    return next(a.menu() for a in menu.actions() if a.menu() is not None and a.text() == title)


def _action(menu: QMenu, text: str):
    return next(a for a in menu.actions() if a.text() == text or a.text().startswith(text))


def _context_menu_labels(card: dash_mod.ItemCard) -> list[str]:
    card.contextMenuEvent(QContextMenuEvent(QContextMenuEvent.Reason.Mouse, QPoint(2, 2), card.mapToGlobal(QPoint(2, 2))))
    menu = card.findChildren(QMenu)[-1]
    labels = [a.text() for a in menu.actions()]
    menu.close()
    return labels


# --------------------------------------------------------------------------------------------
# the view
# --------------------------------------------------------------------------------------------


def test_view_renders_the_columns_the_task_tree_and_the_status(qapp: QApplication) -> None:
    cfg = Config()
    view = DashboardView(cfg)
    try:
        assert view.visible_columns() == _ALL_COLUMNS
        assert "No open questions yet" in view.columns["questions"].empty_label.text()
        assert "No action items yet" in view.columns["nextsteps"].empty_label.text()
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
        assert "added by you" in view.columns["blindspots"].cards()[0].status_label.text()

        n = view.columns["nextsteps"]
        assert [c.item.id for c in n.cards()] == [4, 7, 5]                 # the task, its sub-task, the sub-task's prompt
        assert n.count_label.text() == "1 task · 1 sub · 1 done" and n.resolved_toggle.text() == "Done (1)"
        task, sub, prompt = n.cards()
        assert "font-weight: 600" in task.text_label.styleSheet() and "0/1 steps done" in task.status_label.text()
        assert set(task._buttons) == {"add_sub", "done", "dismiss"} and set(sub._buttons) == {"done", "dismiss"}
        assert set(prompt._buttons) == {"copy", "done", "dismiss"} and "monospace" in prompt.text_label.styleSheet()
        assert _context_menu_labels(task) == ["Mark done…", "Dismiss", "Edit…", "Copy", "Add a sub-task…",
                                              "Add an acceptance criterion…", "Add a prompt…"]
        assert _context_menu_labels(sub) == ["Mark done…", "Dismiss", "Edit…", "Copy", "Add an acceptance criterion…", "Add a prompt…"]

        g = view.columns["goals"]
        assert [c.item.id for c in g.cards()] == [9, 8] and g.count_label.text() == "2 open"      # whole session first, then per task
        headings = [w.text() for i in range(g.open_layout.count()) if (w := g.open_layout.itemAt(i).widget()) is not None
                    and not isinstance(w, dash_mod.ItemCard) and hasattr(w, "text")]
        assert headings == ["Whole session", "For: Task 1: export() rate limiting"]
        assert set(g.cards()[0]._buttons) == {"met", "dismiss"}

        c = view.columns["constraints"]
        assert [c_.item.id for c_ in c.cards()] == [10] and c.count_label.text() == "1 in force"
        assert set(c.cards()[0]._buttons) == {"lift"} and not c.resolved_toggle.isVisibleTo(view)
        assert _context_menu_labels(c.cards()[0]) == ["Lift (no longer applies)", "Edit…", "Copy"]

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

        cfg.dashboard.show_prompts = False
        view.apply_config()
        assert [c_.item.id for c_ in n.cards()] == [4, 7]
        cfg.dashboard.show_notes = False
        cfg.dashboard.show_goals = False
        cfg.dashboard.show_constraints = False
        view.apply_config()
        assert view.visible_columns() == ["nextsteps", "questions", "blindspots"]
        view.clear()
        assert view.items == () and view.columns["questions"].cards() == [] and view.status_label.text() == ""
    finally:
        view.shutdown()


def test_a_sub_task_whose_task_is_done_shows_at_the_top_with_a_note(qapp: QApplication) -> None:
    view = DashboardView(Config())
    try:
        view.set_snapshot((
            DashboardItem(id=1, kind="nextstep", text="Task 1: done already", status="answered", created_at=1.0, updated_at=9.0),
            DashboardItem(id=2, kind="nextstep", text="a late step", created_at=2.0, updated_at=2.0, parent_id=1),
            DashboardItem(id=3, kind="prompt", text="/check", created_at=3.0, updated_at=3.0),
        ), "", None, 1.0)
        n = view.columns["nextsteps"]
        assert [c.item.id for c in n.cards()] == [2, 3]
        assert "step of “Task 1: done already”" in n.cards()[0].status_label.text()
        assert n.count_label.text() == "1 task · 1 done"
        labels = [w.text() for i in range(n.open_layout.count()) if (w := n.open_layout.itemAt(i).widget()) is not None
                  and not isinstance(w, dash_mod.ItemCard) and hasattr(w, "text") and not w.findChildren(dash_mod.ItemCard)]
        assert labels == ["Prompts to run"]                                           # a prompt that serves no task
    finally:
        view.shutdown()


def test_view_actions_emit_signals(qapp: QApplication, monkeypatch: pytest.MonkeyPatch) -> None:
    view = DashboardView(Config())
    got: list = []
    view.item_action.connect(lambda i, a, t: got.append(("item", i, a, t)))
    view.item_added.connect(lambda k, t, p: got.append(("add", k, t, p)))
    view.refresh_requested.connect(lambda s: got.append(("refresh", tuple(s))))
    view.notes_requested.connect(lambda: got.append(("notes",)))
    try:
        view.set_snapshot(_items(), "", None, 1.0)
        q, n, g, c = (view.columns[k] for k in ("questions", "nextsteps", "goals", "constraints"))
        monkeypatch.setattr(dash_mod.QInputDialog, "getText", staticmethod(lambda *a, **kw: ("because Sam said so", True)))
        q.cards()[0]._buttons["answer"].click()
        q.cards()[0]._buttons["dismiss"].click()
        q.resolved_toggle.setChecked(True)
        q.resolved_cards()[0]._buttons["reopen"].click()
        n.cards()[1]._buttons["done"].click()                            # a sub-task: "done" is "answer" for the engine
        g.cards()[0]._buttons["met"].click()
        c.cards()[0]._buttons["lift"].click()                            # a constraint: lifting is dismissing
        n.cards()[2]._buttons["copy"].click()
        assert QGuiApplication.clipboard().text() == "/repro-fix export() 500"
        n.cards()[0]._buttons["add_sub"].click()                         # under Task 1 (id 4)
        q.add_btn.click()
        monkeypatch.setattr(dash_mod.QInputDialog, "getText", staticmethod(lambda *a, **kw: ("/check", True)))
        n.add_btn.click()                                                # typed as a prompt: kind prompt, no task
        monkeypatch.setattr(dash_mod.QInputDialog, "getItem", staticmethod(lambda *a, **kw: ("(the whole session)", True)))
        g.add_btn.click()
        monkeypatch.setattr(dash_mod.QInputDialog, "getItem", staticmethod(lambda *a, **kw: ("Task 1: export() rate limiting", True)))
        g.add_btn.click()
        c.add_btn.click()
        monkeypatch.setattr(dash_mod.QInputDialog, "getText", staticmethod(lambda *a, **kw: ("", False)))
        q.add_btn.click()                                               # cancelled: nothing
        q.cards()[0]._buttons["answer"].click()                         # cancelled: nothing
        n.cards()[0]._buttons["add_sub"].click()                        # cancelled: nothing
        q.refresh_btn.click()
        c.refresh_btn.click()
        view.refresh_all_btn.click()
        view.notes_col.refresh_btn.click()
        assert got == [
            ("item", 1, "answer", "because Sam said so"), ("item", 1, "dismiss", ""), ("item", 2, "reopen", ""),
            ("item", 7, "answer", "because Sam said so"), ("item", 9, "answer", "because Sam said so"),
            ("item", 10, "dismiss", ""),
            ("add", "nextstep", "because Sam said so", 4),
            ("add", "question", "because Sam said so", -1), ("add", "prompt", "/check", -1),
            ("add", "goal", "/check", -1), ("add", "goal", "/check", 4), ("add", "constraint", "/check", -1),
            ("refresh", ("questions",)), ("refresh", ("constraints",)), ("refresh", ()), ("notes",),
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
    view.item_added.emit("question", "Who?", -1)
    view.item_added.emit("nextstep", "a step", 4)
    view.refresh_requested.emit(("questions",))
    view.refresh_requested.emit(())
    view.notes_requested.emit()
    assert ("dashboard_item", (3, "dismiss", "")) in actions.calls
    assert ("dashboard_add", ("question", "Who?", None)) in actions.calls
    assert ("dashboard_add", ("nextstep", "a step", 4)) in actions.calls
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
        assist = _submenu(win._menu, "Assist")
        next(a for a in assist.actions() if a.text() == "Refresh the dashboard now").trigger()
        assert ("dashboard_refresh", ((),)) in actions.calls
    finally:
        win.close_silently()
        qapp.processEvents()


def test_the_view_menu_shows_the_current_view_and_switches_it(window: OverlayWindow, actions: FakeActions, bus: EventBus) -> None:
    cfg = window._cfg
    window._rebuild_menu()
    view = _submenu(window._menu, "View")
    radios = [a for a in view.actions() if a.isCheckable() and a.text() in ("Live", "Assist", "Notes", "Screen", "Dashboard")]
    assert [a.text() for a in radios] == ["Live", "Assist", "Notes", "Screen", "Dashboard"]
    assert [a.isChecked() for a in radios] == [False, True, False, False, False]           # the Assist tab is in front
    texts = [a.text() for a in view.actions() if a.text()]
    assert "Answer history pane" in texts and any(t.startswith("Explanations") for t in texts)
    assert "Dashboard columns" not in texts and "Screen source" in texts and "Repository" in texts
    _action(view, "Answer history pane").setChecked(True)
    assert window.assist_view.history_toggle.isChecked()

    _action(view, "Dashboard").trigger()
    QTest.qWait(_WAIT_MS)
    assert window.dashboard_mode
    window._rebuild_menu()
    view = _submenu(window._menu, "View")
    texts = [a.text() for a in view.actions() if a.text()]
    assert _action(view, "Dashboard").isChecked() and "Dashboard columns" in texts
    for title in ("Session notes", "Action items", "Goals", "Constraints", "Questions to ask", "Blind spots & follow-ups"):
        assert _action(view, title).isCheckable() and _action(view, title).isChecked(), title
    _action(view, "Goals").setChecked(False)
    _action(view, "Suggested prompts under the action items").setChecked(False)
    assert not cfg.dashboard.show_goals and not cfg.dashboard.show_prompts
    assert window.dashboard_view.visible_columns() == ["notes", "nextsteps", "constraints", "questions", "blindspots"]
    assert ("apply_settings", ({"dashboard.show_goals": False},)) in actions.calls
    assert ("apply_settings", ({"dashboard.show_prompts": False},)) in actions.calls

    _action(view, "Live").trigger()
    QTest.qWait(_WAIT_MS)
    assert not window.dashboard_mode and window.stack.currentIndex() == 0
    window._rebuild_menu()
    view = _submenu(window._menu, "View")
    assert _action(view, "Live").isChecked() and "In-progress speech (partial transcripts)" in [a.text() for a in view.actions()]
    _action(view, "In-progress speech").setChecked(False)
    assert not cfg.asr.partials_enabled

    _action(view, "Screen").trigger()
    window._rebuild_menu()
    view = _submenu(window._menu, "View")
    assert _action(view, "OCR readout").isCheckable() and not _action(view, "OCR readout").isChecked()
    _action(view, "OCR readout").setChecked(True)
    assert window.screen_view.screen_text_toggle.isChecked()

    window.completion_btn.setChecked(True)
    QTest.qWait(_WAIT_MS)
    window._rebuild_menu()
    view = _submenu(window._menu, "View")
    assert not any(a.isChecked() for a in view.actions() if a.text() in ("Live", "Assist", "Notes", "Screen", "Dashboard"))
    assert _action(view, "Completion mode").isChecked() and not _action(view, "Completion mode").isEnabled()
    _action(view, "Notes").trigger()
    QTest.qWait(_WAIT_MS)
    assert not window._completion_mode and window.stack.currentIndex() == 2


def test_the_view_menu_sets_the_screen_source_and_the_repository(window: OverlayWindow, actions: FakeActions, bus: EventBus) -> None:
    cfg = window._cfg
    window._rebuild_menu()
    view = _submenu(window._menu, "View")
    source = _submenu(view, "Screen source")
    assert _action(source, "Active window").isChecked()                        # the default mode
    _action(source, "Monitor 0 (primary)").trigger()
    assert cfg.screen.monitor == 0 and ("apply_settings", ({"screen.monitor": 0},)) in actions.calls
    assert ("set_screen_mode", ("monitor",)) in actions.calls and window.screen_view._mode_buttons["monitor"].isChecked()
    window._rebuild_menu()
    source = _submenu(_submenu(window._menu, "View"), "Screen source")
    assert _action(source, "Monitor 0 (primary)").isChecked() and not _action(source, "Active window").isChecked()
    _action(source, "Off").trigger()
    assert ("set_screen_mode", ("off",)) in actions.calls
    _action(_submenu(window._menu, "View"), "Screen: pause/resume").trigger()
    assert ("toggle_screen_paused", ()) in actions.calls

    repo = _submenu(_submenu(window._menu, "View"), "Repository")
    texts = [a.text() for a in repo.actions() if a.text()]
    assert texts[0] == "Context: not known yet" and not repo.actions()[0].isEnabled()
    auto = _action(repo, "Auto-detect from Claude Code")
    assert auto.isChecked() and auto.isEnabled() and "Choose a folder…" in texts and "Clear the folder" not in " ".join(texts)
    auto.setChecked(False)
    assert not cfg.context.repo_auto and ("apply_settings", ({"context.repo_auto": False},)) in actions.calls
    bus.publish(ComponentStatusEvent(component="context", state="info", detail="Context: scenario-1 (detected from Claude Code): 8 skills"))
    window._poll()
    assert window.toast.text() == "Context: scenario-1 (detected from Claude Code): 8 skills"
    cfg.context.repo_root = "D:/repos/thing"
    window._rebuild_menu()
    repo = _submenu(_submenu(window._menu, "View"), "Repository")
    texts = [a.text() for a in repo.actions() if a.text()]
    assert texts[0] == "Context: scenario-1 (detected from Claude Code): 8 skills"
    assert "Folder: D:/repos/thing" in texts and not _action(repo, "Auto-detect").isEnabled()
    _action(repo, "Clear the folder").trigger()
    assert cfg.context.repo_root == "" and ("apply_settings", ({"context.repo_root": ""},)) in actions.calls


def test_settings_dialog_has_the_dashboard_and_context_tabs_and_the_overlay_mirrors_them(window: OverlayWindow, actions: FakeActions, qapp: QApplication) -> None:
    dashboard_keys = [f.key for f in FIELDS["Dashboard"]]
    for key in ("enabled", "track", "refresh_interval_s", "min_new_words", "max_open_items", "max_new_items",
                "suggest_prompts", "show_resolved", "resolved_keep", "show_notes", "show_nextsteps", "show_goals",
                "show_constraints", "show_questions", "show_blindspots", "show_prompts", "width", "view_refresh_s", "persist"):
        assert key in dashboard_keys, key
    context_keys = [f.key for f in FIELDS["Context"]]
    for key in ("repo_root", "repo_auto", "repo_auto_max_age_h", "repo_search", "repo_max_files", "repo_max_chars", "toolkit",
                "toolkit_lookup", "toolkit_in_answers", "toolkit_user_layer", "toolkit_dir", "toolkit_max_chars", "rescan_s"):
        assert key in context_keys, key
    assert "monitor" in [f.key for f in FIELDS["Screen"]]
    assert list(FIELDS).index("Dashboard") == list(FIELDS).index("Notes") + 1
    assert list(FIELDS).index("Context") == list(FIELDS).index("Dashboard") + 1
    dlg = SettingsDialog(Config(), None)
    try:
        assert dlg.show_tab("Dashboard") and dlg.show_tab("Context")
        assert isinstance(dlg._widgets["context.repo_root"], QLineEdit)
        assert isinstance(dlg._widgets["context.toolkit"], QCheckBox) and isinstance(dlg._widgets["context.repo_auto"], QCheckBox)
        assert isinstance(dlg._widgets["dashboard.refresh_interval_s"], QSpinBox) and isinstance(dlg._widgets["screen.monitor"], QSpinBox)
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
    assert window.dashboard_mode and window.dashboard_view.visible_columns() == _ALL_COLUMNS[1:]
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

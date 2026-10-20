"""The pair-programming co-pilot in the overlay (2026-10-20): the Notes tab's Insights pane, the
routing of "insights" / "recap" jobs (insights never switch tabs), the Notes-tab badge, the "⋯" >
Assist menu, the new hotkeys (and empty bindings left unregistered), the settings dialog's new
fields and the Screen tab's toggle sync. Offscreen QPA, FakeActions as in the other overlay tests."""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QCoreApplication, QEvent, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QComboBox, QLineEdit, QMenu

from exocortex.bus import AssistDeltaEvent, AssistDoneEvent, AssistStartedEvent, EventBus, InsightsUpdatedEvent
from exocortex.config import Config
from exocortex.llm import prompts
from exocortex.ui import overlay as overlay_mod
from exocortex.ui.overlay import OverlayWindow
from exocortex.ui.settings import FIELDS, SettingsDialog
from exocortex.ui.widgets import NotesView, ScreenView
from tests.test_overlay import _SpyHotkeys
from tests.test_overlay_completion import FakeActions


@pytest.fixture(scope="session")
def qapp() -> QApplication:
    return QApplication.instance() or QApplication(["exocortex-insights-tests"])


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


def _assist_menu(window: OverlayWindow) -> QMenu:
    window._rebuild_menu()
    return next(a.menu() for a in window._menu.actions() if a.menu() is not None and a.text() == "Assist")


def _action(menu: QMenu, text: str):
    return next(a for a in menu.actions() if a.text() == text)


# --------------------------------------------------------------------------------------------
# the Notes tab's Insights pane
# --------------------------------------------------------------------------------------------


def test_notes_view_insights_lifecycle(qapp: QApplication) -> None:
    view = NotesView(FakeActions())
    try:
        assert "No questions or blind spots yet" in view.insights_browser.toPlainText()
        view.set_insights("- q1", "- b1", 65.0)
        text = view.insights_browser.toPlainText()
        assert "Questions to ask" in text and "q1" in text and "b1" in text
        assert "updated 01:05" in view.insights_label.text()

        view.begin_insights(7, "Blind spots & follow-ups")
        assert view.streaming_insights
        assert "updating blind spots & follow-ups" in view.insights_label.text()
        view.append_insights(7, "## Blind spots & follow-ups\n- b2")
        view.append_insights(99, "## Questions to ask\n- from another job")   # ignored
        view._flush_insights()
        text = view.insights_browser.toPlainText()
        assert "b2" in text and "b1" not in text and "q1" in text      # the streamed section only
        assert "another job" not in text

        view.set_insights("- q1", "- b2", 70.0)                        # lands before the job's Done
        view.finish_insights(7)
        assert not view.streaming_insights
        text = view.insights_browser.toPlainText()
        assert "b2" in text and "q1" in text and "updated 01:10" in view.insights_label.text()

        view.begin_insights(8, "Questions to ask")
        view.insights_error(8, "LLM down")
        assert not view.streaming_insights and "error: LLM down" in view.insights_label.text()

        view.clear()
        assert view.insights == {} and "No questions" in view.insights_browser.toPlainText()
    finally:
        view.shutdown()


def test_notes_view_buttons_call_the_engine(qapp: QApplication) -> None:
    actions = FakeActions()
    view = NotesView(actions)
    try:
        view.questions_btn.click()
        view.blindspots_btn.click()
        view.recap_btn.click()
        assert ("insights_now", (("questions",),)) in actions.calls
        assert ("insights_now", (("blindspots",),)) in actions.calls
        assert ("recap", ()) in actions.calls
    finally:
        view.shutdown()


# --------------------------------------------------------------------------------------------
# routing + the Notes-tab badge
# --------------------------------------------------------------------------------------------


def test_an_insights_job_streams_into_the_notes_tab_without_switching_tabs(
    window: OverlayWindow, bus: EventBus
) -> None:
    window._show_tab(1)
    before = window.assist_view.question_label.text()
    bus.publish(AssistStartedEvent(job_id=5, kind="insights", question="Questions & blind spots"))
    bus.publish(AssistDeltaEvent(job_id=5, delta="## Questions to ask\n- Is the cache per tenant?"))
    window._poll()
    assert window.stack.currentIndex() == 1                       # still on Assist
    assert window.notes_view.streaming_insights
    assert window.assist_view.question_label.text() == before      # the Assist tab is untouched
    bus.publish(InsightsUpdatedEvent(questions="- Is the cache per tenant?", generated_at=12.0, auto=True))
    bus.publish(AssistDoneEvent(job_id=5, text="...", ttft_s=0.1, tokens=3, tok_per_s=30.0))
    window._poll()
    assert not window.notes_view.streaming_insights
    assert "Is the cache per tenant?" in window.notes_view.insights_browser.toPlainText()
    assert window._tab_buttons[2].text() == "Notes •"          # a dot instead of a tab switch
    window._show_tab(2)
    assert window._tab_buttons[2].text() == "Notes"


def test_no_badge_when_the_notes_tab_is_in_front_or_nothing_is_open(window: OverlayWindow, bus: EventBus) -> None:
    window._show_tab(2)
    bus.publish(InsightsUpdatedEvent(questions="- q", generated_at=1.0, auto=True))
    window._poll()
    assert window._tab_buttons[2].text() == "Notes"
    window._show_tab(0)
    bus.publish(InsightsUpdatedEvent(questions="", blindspots="", generated_at=2.0, auto=True))
    window._poll()
    assert window._tab_buttons[2].text() == "Notes"


def test_a_recap_streams_into_the_assist_tab(window: OverlayWindow, bus: EventBus) -> None:
    window._show_tab(2)
    bus.publish(AssistStartedEvent(job_id=9, kind="recap", question="Recap of the session so far"))
    window._poll()
    assert window.stack.currentIndex() == 1
    assert window.assist_view.question_label.text() == "Recap of the session so far"


# --------------------------------------------------------------------------------------------
# the "⋯" > Assist menu
# --------------------------------------------------------------------------------------------


def test_the_assist_menu_offers_the_copilot_actions_with_their_hotkeys(window: OverlayWindow, actions: FakeActions) -> None:
    menu = _assist_menu(window)
    texts = [a.text() for a in menu.actions() if not a.isSeparator()]
    assert texts[:4] == [
        "Suggest questions now\tCtrl+Alt+W",
        "Blind spots && follow-ups now",
        "Recap the session\tCtrl+Alt+Y",
        "Regenerate notes\tCtrl+Alt+N",
    ]
    _action(menu, "Suggest questions now\tCtrl+Alt+W").trigger()
    assert ("insights_now", (("questions",),)) in actions.calls
    assert window.stack.currentIndex() == 2                        # on demand: show where they land
    _action(menu, "Blind spots && follow-ups now").trigger()
    assert ("insights_now", (("blindspots",),)) in actions.calls
    _action(menu, "Recap the session\tCtrl+Alt+Y").trigger()
    assert ("recap", ()) in actions.calls and window.stack.currentIndex() == 1


def test_the_assist_menu_switches_reflect_and_change_the_config(window: OverlayWindow, actions: FakeActions) -> None:
    cfg = window._cfg
    menu = _assist_menu(window)
    questions = _action(menu, "Suggest questions automatically")
    blind = _action(menu, "Flag blind spots && follow-ups automatically")
    screen = _action(menu, "Solve / suggest when the screen changes")
    on_screen = _action(menu, "Answer questions that appear on screen")
    assert questions.isChecked() and blind.isChecked()
    assert not screen.isChecked() and not on_screen.isChecked()    # off by default while driving
    questions.trigger()
    assert cfg.assist.suggest_questions is False and cfg.assist.flag_blind_spots is True
    assert any(name == "set_insight_options" for name, _args in actions.calls)
    screen.trigger()
    on_screen.trigger()
    assert cfg.screen.auto_suggest is True and cfg.screen.answer_screen_questions is True
    assert ("set_auto_suggest", (True,)) in actions.calls
    assert window.screen_view.auto_suggest_cb.isChecked() and window.screen_view.answer_screen_questions_cb.isChecked()
    assert not _action(_assist_menu(window), "Suggest questions automatically").isChecked()   # rebuilt from cfg


def test_the_role_submenu_switches_in_one_click(window: OverlayWindow, actions: FakeActions) -> None:
    menu = _assist_menu(window)
    role_action = next(a for a in menu.actions() if a.menu() is not None and a.text().startswith("Role: "))
    assert role_action.text() == "Role: Driver"
    checked = [a.text() for a in role_action.menu().actions() if a.isChecked()]
    assert checked == ["Driver (I'm at the keyboard)"]
    next(a for a in role_action.menu().actions() if a.text().startswith("Navigator")).trigger()
    assert ("set_role", ("navigator",)) in actions.calls and window._cfg.assist.role == "navigator"
    assert any(a.text() == "Role: Navigator" for a in _assist_menu(window).actions())


def test_the_session_context_entry_prompts_and_sets_it(
    window: OverlayWindow, actions: FakeActions, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(overlay_mod.QInputDialog, "getText", staticmethod(lambda *a, **kw: ("payments  API & co", True)))
    entry = next(a for a in _assist_menu(window).actions() if a.text().startswith("Session context"))
    assert entry.text() == "Session context... (none set)"
    entry.trigger()
    assert ("set_session_context", ("payments  API & co",)) in actions.calls
    assert window._cfg.assist.session_context == "payments API & co"
    assert any(a.text() == "Session context: payments API && co" for a in _assist_menu(window).actions())


def test_assist_settings_opens_the_settings_dialog_on_the_assist_tab(window: OverlayWindow, qapp: QApplication) -> None:
    _action(_assist_menu(window), "Assist settings…").trigger()
    qapp.processEvents()
    dlg = window._settings
    assert dlg is not None and dlg.tabs.tabText(dlg.tabs.currentIndex()) == "Assist"


def test_rebuilding_the_menu_does_not_pile_up_submenus(window: OverlayWindow, qapp: QApplication) -> None:
    for _ in range(6):
        window._rebuild_menu()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        qapp.processEvents()
    direct = window._menu.findChildren(QMenu, options=Qt.FindChildOption.FindDirectChildrenOnly)
    assert len(direct) <= 2          # the current Assist + Rename speakers submenus only


def test_hotkey_hint() -> None:
    assert overlay_mod._hotkey_hint("ctrl+alt+w") == "\tCtrl+Alt+W"
    assert overlay_mod._hotkey_hint("ctrl+alt+shift+left") == "\tCtrl+Alt+Shift+Left"
    assert overlay_mod._hotkey_hint("") == "" and overlay_mod._hotkey_hint("  ") == ""


# --------------------------------------------------------------------------------------------
# hotkeys
# --------------------------------------------------------------------------------------------


def test_hotkeys_bind_insights_and_recap_and_never_register_an_empty_binding(
    qapp: QApplication, bus: EventBus, actions: FakeActions, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.test_overlay import _StubWin32

    monkeypatch.setattr(overlay_mod, "win32", _StubWin32())
    monkeypatch.setattr(overlay_mod, "_is_windows_platform", lambda: True)
    monkeypatch.setattr(overlay_mod, "GlobalHotkeys", _SpyHotkeys)
    _SpyHotkeys.instances = []
    cfg = Config()
    assert cfg.hotkeys.move_up == ""                    # the IDE-colliding nudge keys are off by default
    win = OverlayWindow(cfg, bus, actions)
    try:
        spy = _SpyHotkeys.instances[-1]
        assert "ctrl+alt+w" in spy.bindings and "ctrl+alt+y" in spy.bindings
        assert all(key.strip() for key in spy.bindings)  # "" is never handed to RegisterHotKey
        spy.bindings["ctrl+alt+y"]()
        spy.bindings["ctrl+alt+w"]()
        assert ("recap", ()) in actions.calls and ("insights_now", ((),)) in actions.calls
    finally:
        win.close_silently()
        qapp.processEvents()


# --------------------------------------------------------------------------------------------
# settings dialog + Screen tab sync
# --------------------------------------------------------------------------------------------


def test_the_settings_dialog_has_the_pair_programming_fields_and_jumps_to_a_tab(qapp: QApplication) -> None:
    keys = [f.key for f in FIELDS["Assist"]]
    for key in ("session_context", "role", "suggest_questions", "flag_blind_spots",
                "insight_interval_s", "insight_min_new_words", "insight_max_items"):
        assert key in keys
    assert keys[:4] == ["auto_answer_questions", "autonomous", "primary_language", "comments"]
    role = next(f for f in FIELDS["Assist"] if f.key == "role")
    assert role.kind == "choice" and role.choices == prompts.ROLES
    assert [f.key for f in FIELDS["Notes"]] == ["template", "auto_interval_s", "min_new_words"]
    assert next(f for f in FIELDS["Notes"] if f.key == "template").choices == prompts.NOTES_TEMPLATES
    assert any(f.key == "exclude_own_windows" for f in FIELDS["Screen"])
    dlg = SettingsDialog(Config(), None)
    try:
        assert dlg.show_tab("Assist") and dlg.tabs.tabText(dlg.tabs.currentIndex()) == "Assist"
        assert not dlg.show_tab("Nope")
        assert isinstance(dlg._widgets["assist.session_context"], QLineEdit)
        assert isinstance(dlg._widgets["assist.role"], QComboBox)
        assert isinstance(dlg._widgets["notes.template"], QComboBox)
    finally:
        dlg.close()
        qapp.processEvents()


def test_screen_view_sync_toggles_rereads_the_config_without_rerunning_handlers(qapp: QApplication) -> None:
    cfg = Config()
    actions = FakeActions()
    view = ScreenView(cfg, actions)
    cfg.screen.auto_suggest = True
    cfg.screen.answer_screen_questions = True
    cfg.screen.use_in_answers = False
    view.sync_toggles()
    assert view.auto_suggest_cb.isChecked() and view.answer_screen_questions_cb.isChecked()
    assert not view.use_in_answers_cb.isChecked()
    assert actions.calls == []

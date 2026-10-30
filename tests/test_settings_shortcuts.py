"""The Settings > Shortcuts tab (2026-10-29): every `[hotkeys]` entry as a row -- the binding in
words, **Set** (listen for the next key combination or mouse button, taken half a second after the
last key; Esc cancels) and **Default** -- the status line (invalid, used twice, a bare left button on
a global shortcut), the dialog refusing to apply an invalid row, the fixed in-overlay list, and the
two other new fields (Assist > Consider given code, Hands-free > the third gesture's hold). Offscreen;
the capture runs through the Qt event-filter fallback (no low-level hook ever exists here)."""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import dataclasses
from typing import Any

import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QCheckBox

from exocortex.config import Config, HotkeysConfig
from exocortex.ui.settings import (
    FIELDS,
    FIXED_SHORTCUTS,
    SettingsDialog,
    ShortcutsPage,
    _BindingEdit,
    all_fields,
)
from tests.test_settings import qapp  # noqa: F401 -- the shared offscreen QApplication

_WAIT_MS = 60
_QUIET_WAIT_MS = 800        # > CAPTURE_QUIET_S, with margin for the coarse offscreen timers


@pytest.fixture()
def dlg(qapp: QApplication):
    cfg = Config()
    dialog = SettingsDialog(cfg, None)
    dialog.show()
    qapp.processEvents()
    yield dialog
    dialog.close()
    qapp.processEvents()


# --------------------------------------------------------------------------------------------
# the table
# --------------------------------------------------------------------------------------------


def test_shortcuts_tab_lists_every_hotkeys_field_exactly_once() -> None:
    keys = [f.key for f in FIELDS["Shortcuts"]]
    assert sorted(keys) == sorted(f.name for f in dataclasses.fields(HotkeysConfig))
    assert len(keys) == len(set(keys))
    assert all(f.section == "hotkeys" for f in FIELDS["Shortcuts"])
    bindings = [f for f in FIELDS["Shortcuts"] if f.kind == "binding"]
    assert {f.key for f in FIELDS["Shortcuts"] if f.kind != "binding"} == {"swallow_thumb_buttons"}
    assert [f.key for f in bindings][-3:] == ["manual_text", "manual_code", "manual_code_fresh"]
    assert list(FIELDS)[-2:] == ["Shortcuts", "Optimize"]        # Optimize stays last, Completion first
    assert list(FIELDS)[0] == "Completion"


def test_the_other_new_fields_sit_on_their_tabs() -> None:
    by_path = {f.path: f for f in all_fields()}
    consider = by_path["assist.consider_given_code"]
    assert consider.kind == "bool" and consider in FIELDS["Assist"]
    assist_paths = [f.path for f in FIELDS["Assist"]]
    assert assist_paths[assist_paths.index("assist.comments") + 1] == "assist.consider_given_code"
    hold = by_path["handsfree.manual_hold_code_fresh"]
    assert hold.kind == "bool" and hold in FIELDS["Hands-free"]
    hf_paths = [f.path for f in FIELDS["Hands-free"]]
    assert hf_paths[hf_paths.index("handsfree.manual_hold_code") + 1] == "handsfree.manual_hold_code_fresh"


def test_fixed_shortcuts_cover_the_documented_in_overlay_keys() -> None:
    joined = " ".join(" ".join(row) for row in FIXED_SHORTCUTS)
    for needle in ("Ask box", "Region picker", "Rename the speaker", "Mouse wheel", "Tap Up / Down", "Right-click"):
        assert needle in joined
    assert all(len(row) == 3 and all(row) for row in FIXED_SHORTCUTS)


# --------------------------------------------------------------------------------------------
# the page: rows in words, apply, Default, invalid, duplicates
# --------------------------------------------------------------------------------------------


def test_dialog_builds_the_page_with_the_rows_in_words(dlg: SettingsDialog) -> None:
    page = dlg._shortcuts_page
    assert isinstance(page, ShortcutsPage)
    assert dlg.tabs.tabText(dlg.tabs.count() - 2) == "Shortcuts"
    assert dlg.tabs.widget(dlg.tabs.count() - 2) is page
    row = dlg._widgets["hotkeys.code"]
    assert isinstance(row, _BindingEdit)
    assert row.value() == "ctrl+alt+g" and row.label.text() == "Ctrl+Alt+G" and row.is_valid()
    assert dlg._widgets["hotkeys.toggle_pause"].label.text() == "Right Shift (tap)"
    assert dlg._widgets["hotkeys.manual_text"].label.text() == "Left button"
    assert dlg._widgets["hotkeys.manual_code_fresh"].label.text() == "Mouse 4 (thumb, back)"
    assert isinstance(dlg._widgets["hotkeys.swallow_thumb_buttons"], QCheckBox)
    assert dlg._widgets["hotkeys.swallow_thumb_buttons"].isChecked()
    assert page.status.text() == "Every shortcut has its own key."
    assert len(page.rows) == len([f for f in FIELDS["Shortcuts"] if f.kind == "binding"])
    assert dlg.changes() == {}


def test_set_value_normalizes_and_apply_writes_the_config(dlg: SettingsDialog) -> None:
    got: list[dict[str, Any]] = []
    dlg.applied.connect(got.append)
    row = dlg._widgets["hotkeys.code"]
    row.set_value("Alt+Ctrl+K")
    assert row.value() == "ctrl+alt+k" and row.label.text() == "Ctrl+Alt+K"
    dlg._widgets["hotkeys.manual_code_fresh"].set_value("button5")
    assert dlg.changes() == {"hotkeys.code": "ctrl+alt+k", "hotkeys.manual_code_fresh": "mouse5"}
    changed = dlg.apply()
    assert changed == {"hotkeys.code": "ctrl+alt+k", "hotkeys.manual_code_fresh": "mouse5"} and got == [changed]
    assert dlg._cfg.hotkeys.code == "ctrl+alt+k" and dlg._cfg.hotkeys.manual_code_fresh == "mouse5"
    assert dlg.note.text() == "Applied."
    assert dlg.changes() == {}


def test_unbinding_is_allowed_and_shown_as_none(dlg: SettingsDialog) -> None:
    row = dlg._widgets["hotkeys.notes"]
    row.set_value("")
    assert row.is_valid() and row.value() == "" and row.label.text() == "(none)"
    assert dlg.apply() == {"hotkeys.notes": ""}
    assert dlg._cfg.hotkeys.notes == ""


def test_default_button_restores_one_row_only(dlg: SettingsDialog) -> None:
    code, quit_ = dlg._widgets["hotkeys.code"], dlg._widgets["hotkeys.quit"]
    code.set_value("ctrl+alt+k")
    quit_.set_value("ctrl+alt+z")
    code.default_btn.click()
    assert code.value() == "ctrl+alt+g" and code.default == "ctrl+alt+g"
    assert quit_.value() == "ctrl+alt+z"                             # untouched
    assert dlg.changes() == {"hotkeys.quit": "ctrl+alt+z"}
    dlg.restore_defaults()                                           # the dialog-wide button: every row
    assert quit_.value() == "ctrl+alt+x" and dlg.changes() == {}


def test_an_invalid_binding_is_flagged_and_never_applied(dlg: SettingsDialog) -> None:
    page = dlg._shortcuts_page
    code = dlg._widgets["hotkeys.code"]
    code.set_value("ctrl+")
    assert not code.is_valid() and code.label.text() == "ctrl+ (invalid)"
    assert "Invalid: Code" in page.status.text()
    assert page.invalid() == {"hotkeys.code": "Code: write the code for the task"}
    dlg._widgets["hotkeys.quit"].set_value("ctrl+alt+z")
    changed = dlg.apply()
    assert changed == {"hotkeys.quit": "ctrl+alt+z"}                 # the valid change goes through
    assert dlg._cfg.hotkeys.code == "ctrl+alt+g"                     # the invalid one never does
    assert "Not applied, invalid shortcut: Code" in dlg.note.text()
    assert "hotkeys.code" in dlg.changes()                           # ... and stays pending
    code.set_value("ctrl+alt+g")
    assert code.is_valid() and dlg.changes() == {} and page.status.text() == "Every shortcut has its own key."


def test_a_binding_used_twice_is_reported(dlg: SettingsDialog) -> None:
    page = dlg._shortcuts_page
    dlg._widgets["hotkeys.quit"].set_value("ctrl+alt+g")
    assert page.duplicates() == [("ctrl+alt+g", ["Code: write the code for the task", "Quit (saves the session)"])]
    assert "Ctrl+Alt+G is used twice" in page.status.text()
    dlg._widgets["hotkeys.quit"].default_btn.click()
    assert page.duplicates() == [] and page.status.text() == "Every shortcut has its own key."


def test_a_bare_mouse_button_on_a_global_shortcut_warns_but_a_gesture_does_not(dlg: SettingsDialog) -> None:
    page = dlg._shortcuts_page
    assert "fires on every click" not in page.status.text()        # manual_text = mouse_left is fine
    dlg._widgets["hotkeys.code"].set_value("mouse_left")
    assert "fires on every click" in page.status.text() and "add a modifier" in page.status.text()
    dlg._widgets["hotkeys.code"].set_value("ctrl+mouse_left")
    assert "fires on every click" not in page.status.text()


# --------------------------------------------------------------------------------------------
# Set: listening, the chord, Esc, one row at a time, the dialog hiding
# --------------------------------------------------------------------------------------------


def test_set_listens_and_takes_the_chord_from_key_events(qapp: QApplication, dlg: SettingsDialog) -> None:
    page = dlg._shortcuts_page
    row = dlg._widgets["hotkeys.code"]
    row.set_btn.click()
    assert page.capturing and row.listening and row.set_btn.isChecked()
    assert row.label.text().startswith("press keys")
    assert "Listening for" in page.status.text() and "Code" in page.status.text()
    QTest.keyPress(dlg, Qt.Key.Key_Control)
    QTest.keyPress(dlg, Qt.Key.Key_Alt)
    QTest.keyPress(dlg, Qt.Key.Key_K)
    QTest.keyRelease(dlg, Qt.Key.Key_K)
    QTest.keyRelease(dlg, Qt.Key.Key_Alt)
    QTest.keyRelease(dlg, Qt.Key.Key_Control)
    assert row.listening                                             # the quiet time has not passed
    QTest.qWait(_QUIET_WAIT_MS)
    assert row.value() == "ctrl+alt+k" and row.label.text() == "Ctrl+Alt+K"
    assert not row.listening and not page.capturing and not row.set_btn.isChecked()
    assert dlg.changes() == {"hotkeys.code": "ctrl+alt+k"}
    assert dlg.isVisible()


def test_a_single_key_tap_is_captured(qapp: QApplication, dlg: SettingsDialog) -> None:
    row = dlg._widgets["hotkeys.toggle_pause"]
    row.set_btn.click()
    QTest.keyClick(dlg, Qt.Key.Key_F9)
    QTest.qWait(_QUIET_WAIT_MS)
    assert row.value() == "f9" and row.label.text() == "F9 (tap)"


def test_esc_cancels_the_capture_without_closing_the_dialog(qapp: QApplication, dlg: SettingsDialog) -> None:
    page = dlg._shortcuts_page
    row = dlg._widgets["hotkeys.code"]
    row.set_btn.click()
    QTest.keyPress(dlg, Qt.Key.Key_Control)
    QTest.keyRelease(dlg, Qt.Key.Key_Control)
    QTest.keyClick(dlg, Qt.Key.Key_Escape)
    QTest.qWait(_WAIT_MS)
    assert not page.capturing and not row.listening
    assert row.value() == "ctrl+alt+g" and dlg.changes() == {}
    assert dlg.isVisible()                                           # Esc went to the capture, not QDialog


def test_clicking_set_again_cancels_and_only_one_row_listens_at_a_time(qapp: QApplication, dlg: SettingsDialog) -> None:
    page = dlg._shortcuts_page
    code, quit_ = dlg._widgets["hotkeys.code"], dlg._widgets["hotkeys.quit"]
    code.set_btn.click()
    assert code.listening
    code.set_btn.click()                                             # toggled off: cancelled
    QTest.qWait(_WAIT_MS)
    assert not code.listening and not page.capturing
    code.set_btn.click()
    quit_.set_btn.click()                                            # a second row takes over
    QTest.qWait(_WAIT_MS)
    assert not code.listening and quit_.listening and page.capturing
    assert "Quit" in page.status.text()


def test_hiding_the_dialog_cancels_a_capture(qapp: QApplication, dlg: SettingsDialog) -> None:
    page = dlg._shortcuts_page
    row = dlg._widgets["hotkeys.code"]
    row.set_btn.click()
    dlg.hide()
    QTest.qWait(_WAIT_MS)
    assert not page.capturing and not row.listening and row.value() == "ctrl+alt+g"


def test_a_recorded_mouse_binding_lands_in_the_row(qapp: QApplication, dlg: SettingsDialog) -> None:
    """The hooks (real Windows only) hand the recorder mouse tokens; the row takes them like keys."""
    page = dlg._shortcuts_page
    row = dlg._widgets["hotkeys.manual_code_fresh"]
    page.start_capture(row)
    page._recorder.press("lctrl")
    page._recorder.press("mouse5")
    page._recorder.release("mouse5")
    page._recorder.release("lctrl")
    QTest.qWait(_QUIET_WAIT_MS)
    assert row.value() == "ctrl+mouse5" and row.label.text() == "Ctrl+Mouse 5"
    assert dlg.changes() == {"hotkeys.manual_code_fresh": "ctrl+mouse5"}


def test_load_refreshes_the_rows_and_the_status(qapp: QApplication, dlg: SettingsDialog) -> None:
    page = dlg._shortcuts_page
    dlg._widgets["hotkeys.quit"].set_value("ctrl+alt+g")
    assert "used twice" in page.status.text()
    dlg.load()                                                       # back to the config
    assert dlg._widgets["hotkeys.quit"].value() == "ctrl+alt+x"
    assert page.status.text() == "Every shortcut has its own key." and dlg.changes() == {}

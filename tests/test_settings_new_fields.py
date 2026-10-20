"""Tests for the settings-dialog fields added alongside the `[assist].focus` gradient, the
`[assist].libraries` levels and hands-free's `min_new_lines` / `on_code` config values:
`assist.focus` (choices auto/technical/all), `assist.libraries` (choices none/basic/all),
`handsfree.min_new_lines` (int) and `handsfree.on_code` (choices code/code_first/ignore).

Offscreen, same pattern as tests/test_settings.py (imported for its `qapp` fixture rather than
redefined -- see that file's `QT_QPA_PLATFORM=offscreen` setup, which must be set before any
PySide6 import).
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from typing import Any

import pytest
from PySide6.QtWidgets import QApplication, QComboBox, QSpinBox

from exocortex.config import Config
from exocortex.ui.settings import FIELDS, SettingsDialog, all_fields
from tests.test_settings import qapp  # noqa: F401 -- reused fixture, see module docstring

# --------------------------------------------------------------------------------------------
# the fields exist, on the right tab, with the right kind/choices
# --------------------------------------------------------------------------------------------


def test_assist_focus_field_is_a_choice_of_auto_technical_all() -> None:
    by_path = {f.path: f for f in all_fields()}
    f = by_path["assist.focus"]
    assert f.kind == "choice"
    assert f.choices == ("auto", "technical", "all")
    assert f in FIELDS["Assist"]


def test_assist_libraries_field_is_a_choice_of_none_basic_all() -> None:
    by_path = {f.path: f for f in all_fields()}
    f = by_path["assist.libraries"]
    assert f.kind == "choice"
    assert f.choices == ("none", "basic", "all")
    assert f in FIELDS["Assist"]


def test_handsfree_min_new_lines_field_is_an_int() -> None:
    by_path = {f.path: f for f in all_fields()}
    f = by_path["handsfree.min_new_lines"]
    assert f.kind == "int"
    assert f.lo == 1
    assert f in FIELDS["Hands-free"]


def test_handsfree_on_code_field_is_a_choice_of_code_modes() -> None:
    by_path = {f.path: f for f in all_fields()}
    f = by_path["handsfree.on_code"]
    assert f.kind == "choice"
    assert f.choices == ("code", "code_first", "ignore")
    assert f in FIELDS["Hands-free"]


# --------------------------------------------------------------------------------------------
# the dialog actually builds with these fields (widget types, initial values from Config)
# --------------------------------------------------------------------------------------------


def test_dialog_builds_with_the_new_fields(qapp: QApplication) -> None:
    cfg = Config()
    cfg.assist.focus = "technical"
    cfg.assist.libraries = "all"
    cfg.handsfree.min_new_lines = 3
    cfg.handsfree.on_code = "code_first"
    dlg = SettingsDialog(cfg, None)
    try:
        focus_w = dlg._widgets["assist.focus"]
        libraries_w = dlg._widgets["assist.libraries"]
        min_new_lines_w = dlg._widgets["handsfree.min_new_lines"]
        on_code_w = dlg._widgets["handsfree.on_code"]

        assert isinstance(focus_w, QComboBox) and focus_w.currentText() == "technical"
        assert isinstance(libraries_w, QComboBox) and libraries_w.currentText() == "all"
        assert isinstance(min_new_lines_w, QSpinBox) and min_new_lines_w.value() == 3
        assert isinstance(on_code_w, QComboBox) and on_code_w.currentText() == "code_first"

        assert [focus_w.itemText(i) for i in range(focus_w.count())] == ["auto", "technical", "all"]
        assert [libraries_w.itemText(i) for i in range(libraries_w.count())] == ["none", "basic", "all"]
        assert [on_code_w.itemText(i) for i in range(on_code_w.count())] == ["code", "code_first", "ignore"]
    finally:
        dlg.close()
        qapp.processEvents()


# --------------------------------------------------------------------------------------------
# applying a change round-trips into the Config object
# --------------------------------------------------------------------------------------------


def test_applying_a_change_round_trips_into_config(qapp: QApplication) -> None:
    cfg = Config()
    dlg = SettingsDialog(cfg, None)
    got: list[dict[str, Any]] = []
    dlg.applied.connect(got.append)
    try:
        dlg._widgets["assist.focus"].setCurrentText("all")
        dlg._widgets["assist.libraries"].setCurrentText("none")
        dlg._widgets["handsfree.min_new_lines"].setValue(5)
        dlg._widgets["handsfree.on_code"].setCurrentText("ignore")

        changes = dlg.changes()
        assert changes == {
            "assist.focus": "all",
            "assist.libraries": "none",
            "handsfree.min_new_lines": 5,
            "handsfree.on_code": "ignore",
        }

        applied = dlg.apply()
        assert applied == changes
        assert got == [changes]

        assert cfg.assist.focus == "all"
        assert cfg.assist.libraries == "none"
        assert cfg.handsfree.min_new_lines == 5
        assert cfg.handsfree.on_code == "ignore"

        # nothing left to apply, and reloading from cfg shows the just-applied values
        assert dlg.changes() == {}
        dlg.load()
        assert dlg._widgets["assist.focus"].currentText() == "all"
        assert dlg._widgets["handsfree.min_new_lines"].value() == 5
    finally:
        dlg.close()
        qapp.processEvents()


def test_restore_defaults_resets_the_new_fields(qapp: QApplication) -> None:
    cfg = Config()
    cfg.assist.focus = "technical"
    cfg.assist.libraries = "all"
    cfg.handsfree.min_new_lines = 7
    cfg.handsfree.on_code = "ignore"
    dlg = SettingsDialog(cfg, None)
    try:
        dlg.restore_defaults()
        defaults = Config()
        assert dlg._widgets["assist.focus"].currentText() == defaults.assist.focus
        assert dlg._widgets["assist.libraries"].currentText() == defaults.assist.libraries
        assert dlg._widgets["handsfree.min_new_lines"].value() == defaults.handsfree.min_new_lines
        assert dlg._widgets["handsfree.on_code"].currentText() == defaults.handsfree.on_code
    finally:
        dlg.close()
        qapp.processEvents()

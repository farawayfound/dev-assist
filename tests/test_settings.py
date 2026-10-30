"""Offscreen tests for the settings dialog and its wiring (overlay menus, engine apply/save)."""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QLineEdit, QSpinBox

from exocortex.bus import EventBus
from exocortex.config import Config, load_config
from exocortex.ui.overlay import OverlayWindow
from exocortex.ui.settings import (
    FIELDS,
    Field,
    FieldForm,
    SettingsDialog,
    all_fields,
    get_field_value,
    make_field_widget,
    sections_covered,
)
from tests.test_overlay_completion import FakeActions

_WAIT_MS = 60


@pytest.fixture(scope="session")
def qapp() -> QApplication:
    return QApplication.instance() or QApplication(["exocortex-tests"])


def test_every_field_names_a_real_config_value_of_the_right_kind() -> None:
    cfg = Config()
    kinds = {"bool": bool, "int": int, "float": float, "choice": str, "text": str, "model": str,
             "device": str, "binding": str}
    seen: set[str] = set()
    for f in all_fields():
        assert f.path not in seen, f.path
        seen.add(f.path)
        section = getattr(cfg, f.section)
        assert hasattr(section, f.key), f.path
        assert isinstance(getattr(section, f.key), kinds[f.kind]), f.path
        if f.kind in ("int", "float"):
            assert f.lo <= getattr(section, f.key) <= f.hi, f.path
        if f.kind == "choice":
            assert getattr(section, f.key) in f.choices, f.path
    # everything README's Configuration section shows is editable -- except [llm], whose knobs
    # moved to the "Models..." window's Server / Tuning tabs on 2026-09-27 (portable-install)
    for path in ("assist.auto_answer_questions", "assist.autonomous", "assist.style", "assist.persona",
                 "screen.show_text", "completion.enabled", "completion.debounce_ms", "completion.hint",
                 "completion.max_lines", "explain.enabled", "explain.typed_fraction",
                 "audio.loopback_keepalive", "audio.echo_suppression",
                 "ui.opacity"):
        assert path in seen, path
    assert {"completion", "explain", "assist", "screen", "ui", "audio", "monitor"} <= sections_covered()
    assert "llm" not in sections_covered()
    assert list(FIELDS)[0] == "Completion" and FIELDS["Completion"][0].key == "max_lines"


def test_monitor_tab_configures_the_monitor_section() -> None:
    """The View -> Monitor section's contents are editable on a "Monitor" settings tab."""
    assert "Monitor" in FIELDS
    keys = [f.key for f in FIELDS["Monitor"]]
    assert keys == ["show_performance", "show_context", "show_cpu", "show_ram",
                    "show_gpu", "show_igpu", "show_npu"]
    assert all(f.section == "monitor" and f.kind == "bool" for f in FIELDS["Monitor"])


def test_dialog_loads_edits_and_applies_only_the_changes(qapp: QApplication) -> None:
    cfg = Config()
    cfg.completion.hint = "terse"
    dlg = SettingsDialog(cfg, lambda: ["maple-preview", "other-model"])
    got: list[dict[str, Any]] = []
    dlg.applied.connect(got.append)
    try:
        assert isinstance(dlg._widgets["completion.max_lines"], QSpinBox)
        assert dlg._widgets["completion.max_lines"].value() == cfg.completion.max_lines
        assert dlg._widgets["completion.hint"].text() == "terse"
        assert dlg.changes() == {}
        dlg._widgets["completion.max_lines"].setValue(12)
        dlg._widgets["explain.enabled"].setChecked(True)
        dlg._widgets["ui.opacity"].setValue(0.8)
        dlg._widgets["ui.font_size"].setValue(cfg.ui.font_size + 1)   # a restart-only field
        dlg._widgets["assist.style"].setCurrentText("short")
        assert dlg.changes() == {"completion.max_lines": 12, "explain.enabled": True, "ui.opacity": 0.8,
                                 "ui.font_size": cfg.ui.font_size + 1, "assist.style": "short"}
        changed = dlg.apply()
        assert got == [changed]
        assert cfg.completion.max_lines == 12 and cfg.explain.enabled and cfg.ui.opacity == pytest.approx(0.8)
        assert cfg.ui.font_size == cfg.ui.font_size and cfg.assist.style == "short"
        assert "next launch" in dlg.note.text() and "Font size" in dlg.note.text()
        assert dlg.changes() == {} and dlg.apply() == {} and got == [changed]   # nothing new: no signal
        dlg.restore_defaults()
        assert dlg._widgets["completion.max_lines"].value() == Config().completion.max_lines
        assert dlg._widgets["completion.hint"].text() == ""
        assert "completion.hint" in dlg.changes()
        dlg.load()                                                          # back to the config
        assert dlg.changes() == {}
    finally:
        dlg.close()
        qapp.processEvents()


def test_widget_kinds(qapp: QApplication) -> None:
    dlg = SettingsDialog(Config(), None)
    try:
        assert isinstance(dlg._widgets["explain.enabled"], QCheckBox)
        assert isinstance(dlg._widgets["ui.opacity"], QDoubleSpinBox)
        assert isinstance(dlg._widgets["assist.style"], QComboBox)
        assert isinstance(dlg._widgets["assist.persona"], QLineEdit)
        assert dlg.tabs.count() == len(FIELDS)
    finally:
        dlg.close()
        qapp.processEvents()


def test_primary_language_is_a_text_field_on_the_assist_tab(qapp: QApplication) -> None:
    """The user asked for this on the Assist tab specifically (not Completion): it feeds solve,
    the Code button AND completions alike (see prompts._language_rule), so it belongs with the
    other assist-wide defaults, not tucked under one of its consumers."""
    assist_paths = [f.path for f in FIELDS["Assist"]]
    assert "assist.primary_language" in assist_paths
    assert not any(f.path == "assist.primary_language" for group in FIELDS.values()
                   for f in group if group is not FIELDS["Assist"])
    # placed right after "Autonomous", per the design
    assert [f.key for f in FIELDS["Assist"][:3]] == ["auto_answer_questions", "autonomous", "primary_language"]

    cfg = Config()
    cfg.assist.primary_language = "Rust"
    dlg = SettingsDialog(cfg, None)
    try:
        w = dlg._widgets["assist.primary_language"]
        assert isinstance(w, QLineEdit)
        assert w.text() == "Rust"
        w.setText("TypeScript")
        assert dlg.changes() == {"assist.primary_language": "TypeScript"}
        dlg.apply()
        assert cfg.assist.primary_language == "TypeScript"
    finally:
        dlg.close()
        qapp.processEvents()


def test_overlay_opens_settings_from_both_menus_and_mirrors_applied_values(qapp: QApplication) -> None:
    actions = FakeActions()
    cfg = Config()
    win = OverlayWindow(cfg, EventBus(), actions)
    win.show()
    QTest.qWait(_WAIT_MS)
    try:
        win._rebuild_menu()
        assert any(a.text().startswith("Settings") for a in win._menu.actions())
        win._rebuild_context_menu()
        acts = [a for a in win._context_menu.actions() if a.text().startswith("Settings")]
        assert acts
        acts[0].trigger()
        assert win._settings is not None and win._settings.isVisible()
        dlg = win._settings
        dlg._widgets["ui.opacity"].setValue(0.6)
        dlg._widgets["explain.enabled"].setChecked(True)
        dlg._widgets["assist.auto_answer_questions"].setChecked(False)
        dlg.apply_btn.click()
        QTest.qWait(_WAIT_MS)
        assert win.windowOpacity() == pytest.approx(0.6)
        assert win.explain_btn.isChecked() and not win.slim_explain_btn.isHidden()
        assert not win.auto_btn.isChecked()
        applied = [c for c in actions.calls if c[0] == "apply_settings"]
        assert len(applied) == 1
        assert applied[0][1][0] == {"ui.opacity": 0.6, "explain.enabled": True, "assist.auto_answer_questions": False}
        # the buttons were mirrored silently: the engine reacts through apply_settings alone
        assert ("set_explanations", (True,)) not in actions.calls
        assert ("set_auto", (False,)) not in actions.calls
        win.open_settings()                                                 # reused, reloaded
        assert win._settings is dlg and dlg.changes() == {}
        dlg.ok_btn.click()
        QTest.qWait(_WAIT_MS)
        assert not dlg.isVisible()
    finally:
        win.close_silently()
        qapp.processEvents()


def test_engine_apply_settings_reacts_and_saves(tmp_path: Path) -> None:
    from exocortex import app as app_mod

    cfg = Config()
    cfg.path = tmp_path / "config.toml"
    eng = app_mod.Engine(cfg, EventBus())
    calls: list[tuple[str, Any]] = []

    class _Assistant:
        def set_auto(self, enabled: bool) -> None:
            calls.append(("set_auto", enabled))

        def set_explanations(self, enabled: bool) -> None:
            calls.append(("set_explanations", enabled))

        def set_model(self, model_id: str) -> None:
            calls.append(("set_model", model_id))

        def ensure_plan(self) -> None:
            calls.append(("ensure_plan", None))

    class _Completer:
        def reset(self) -> None:
            calls.append(("completer.reset", None))

    eng.assistant = _Assistant()
    eng.completer = _Completer()
    cfg.completion.max_lines = 11
    cfg.assist.auto_answer_questions = False
    cfg.llm.model = "other"
    eng.apply_settings({"completion.max_lines": 11, "assist.auto_answer_questions": False, "llm.model": "other"})
    assert ("set_auto", False) in calls and ("completer.reset", None) in calls and ("set_model", "other") in calls
    saved = load_config(cfg.path)
    assert saved.completion.max_lines == 11 and saved.assist.auto_answer_questions is False and saved.llm.model == "other"


# =====================================================================================
# the "model" field kind: no FIELDS entry uses it any more (the whole Model tab moved to
# the "Models..." window's Server tab, 2026-09-27), but make_field_widget still builds and
# wires up a _ModelCombo for it -- exercised here through an ad-hoc Field, the way
# exocortex.ui.models could use it if it ever needs a live, editable model picker.
# =====================================================================================
_MODEL_FIELD = Field(
    "llm", "model", "Model", "model",
    help="auto = the largest GGUF that fits this GPU; or any GGUF in LM Studio's "
         "library / an Ollama model name.",
)
_LONG_ID = "unsloth__NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-UD-Q4_K_XL"


def _model_combo(models: list[str]) -> tuple[QComboBox, Callable[[], None]]:
    """A ``_ModelCombo`` (via ``make_field_widget``) whose ``refresh_models`` callback re-reads
    ``models``, preserving whatever is selected or typed -- the merge ``SettingsDialog`` used to
    do itself before the Model tab moved out into its own window."""
    state: dict[str, QComboBox] = {}

    def refresh() -> None:
        combo = state["combo"]
        current = combo.currentText().strip()
        items = list(models)
        for extra in (current,):
            if extra and extra not in items:
                items.append(extra)
        if items == [combo.itemText(i) for i in range(combo.count())]:
            return
        combo.blockSignals(True)
        combo.set_models(items)
        idx = combo.findText(current)
        if idx >= 0:
            combo.setCurrentIndex(idx)
        else:
            combo.setEditText(current)
        combo.blockSignals(False)
        combo.sync_tooltip()

    combo = make_field_widget(_MODEL_FIELD, None, refresh_models=refresh, list_models=lambda: list(models))
    state["combo"] = combo
    return combo, refresh


def test_model_list_is_re_read_when_the_dropdown_opens_and_when_reloaded(qapp: QApplication) -> None:
    """A model downloaded elsewhere while this widget sits on screen is discoverable through its
    own ``refresh_models`` callback, called both when the drop-down opens and whenever the owning
    dialog reloads (there is no dialog here -- calling the callback directly stands in for that)."""
    library = ["a/one", "b/two"]
    combo, refresh = _model_combo(library)
    try:
        items = lambda: [combo.itemText(i) for i in range(combo.count())]  # noqa: E731
        assert items() == ["a/one", "b/two"]

        library.append("c/just-downloaded")
        combo.showPopup()
        combo.hidePopup()
        assert "c/just-downloaded" in items()

        library.append("d/downloaded-later")
        refresh()                                        # what reopening the owning dialog would do
        assert "d/downloaded-later" in items()
    finally:
        combo.deleteLater()
        qapp.processEvents()


def test_model_combo_keeps_a_name_the_library_does_not_have(qapp: QApplication) -> None:
    """The combo is editable on purpose: a bare .gguf path, or a model served by a remote
    ``openai`` backend, is not in the local library and must not be dropped by a refresh."""
    combo, refresh = _model_combo(["a/one"])
    try:
        combo.setEditText("D:/models/local.gguf")
        refresh()
        assert "D:/models/local.gguf" in [combo.itemText(i) for i in range(combo.count())]
        assert get_field_value(combo, _MODEL_FIELD) == "D:/models/local.gguf"
    finally:
        combo.deleteLater()
        qapp.processEvents()


def test_a_failing_library_leaves_the_configured_model_in_place(qapp: QApplication) -> None:
    def boom() -> list[str]:
        raise RuntimeError("LM Studio library unreachable")

    def refresh() -> None:
        try:
            boom()
        except Exception:  # noqa: BLE001 - a real caller (e.g. a dialog) would log.exception here
            pass

    combo = make_field_widget(_MODEL_FIELD, None, refresh_models=refresh, list_models=boom)
    try:
        assert [combo.itemText(i) for i in range(combo.count())] == []   # list_models() failed: left empty
        combo.setEditText("configured-model")
        combo.showPopup()                                                # refresh_models() also fails
        combo.hidePopup()
        assert combo.currentText().strip() == "configured-model"
    finally:
        combo.deleteLater()
        qapp.processEvents()


def test_the_whole_model_id_is_reachable_although_the_field_elides_it(qapp: QApplication) -> None:
    """Model ids run to ~100 characters and the field is one column of a form: the open list is
    widened to the longest id, and the full id is the tool tip of both the field and every row,
    so hovering always shows the whole name."""
    combo, _refresh = _model_combo(["a/one", _LONG_ID])
    try:
        combo.setCurrentIndex(combo.findText(_LONG_ID))
        assert combo.toolTip().splitlines()[0] == _LONG_ID
        assert _MODEL_FIELD.help in combo.toolTip()          # the field's own help is kept below it
        assert combo.itemData(combo.findText(_LONG_ID), Qt.ItemDataRole.ToolTipRole) == _LONG_ID

        combo.showPopup()
        assert combo.view().minimumWidth() > combo.width()
        combo.hidePopup()
    finally:
        combo.deleteLater()
        qapp.processEvents()


# =====================================================================================
# assist.comments (comment level for generated code)
# =====================================================================================
from exocortex.llm import prompts  # noqa: E402 - kept with the block that uses it


def test_comments_is_a_choice_field_on_the_assist_tab_right_after_primary_language() -> None:
    """One knob controls how much the generated code is commented (none/minimal/normal/detailed),
    mirrored by ``prompts.COMMENT_LEVELS`` so the dialog and the prompt builder never drift apart."""
    assist_paths = [f.path for f in FIELDS["Assist"]]
    idx = assist_paths.index("assist.primary_language")
    assert assist_paths[idx + 1] == "assist.comments"
    field = next(f for f in FIELDS["Assist"] if f.path == "assist.comments")
    assert field.kind == "choice"
    assert field.choices == prompts.COMMENT_LEVELS
    # nowhere else in the table
    assert not any(f.path == "assist.comments" for name, group in FIELDS.items()
                   for f in group if name != "Assist")


def test_explain_comments_field_is_gone() -> None:
    """The old ``ExplainConfig.comments`` bool was replaced by ``assist.comments``: no Field in
    the table may still point at ``explain.comments``."""
    assert not any(f.path == "explain.comments" for f in all_fields())


def test_comments_combo_defaults_to_normal_and_applies(qapp: QApplication) -> None:
    cfg = Config()
    dlg = SettingsDialog(cfg, None)
    try:
        combo = dlg._widgets["assist.comments"]
        assert isinstance(combo, QComboBox)
        assert combo.currentText() == "normal"
        combo.setCurrentText("detailed")
        assert dlg.changes() == {"assist.comments": "detailed"}
        changed = dlg.apply()
        assert changed == {"assist.comments": "detailed"}
        assert cfg.assist.comments == "detailed"
        dlg.restore_defaults()
        assert dlg._widgets["assist.comments"].currentText() == "normal"
    finally:
        dlg.close()
        qapp.processEvents()


# =====================================================================================
# llama_backend + server_url: still edited from the Models window's Server tab (not this
# dialog any more), but Engine.apply_settings must still react to and persist them.
# =====================================================================================

def test_server_url_and_llama_backend_apply_and_persist(tmp_path: Path) -> None:
    from exocortex import app as app_mod

    cfg = Config()
    cfg.path = tmp_path / "config.toml"
    eng = app_mod.Engine(cfg, EventBus())

    class _Assistant:
        def set_auto(self, enabled: bool) -> None:
            pass

        def set_explanations(self, enabled: bool) -> None:
            pass

        def set_model(self, model_id: str) -> None:
            pass

        def ensure_plan(self) -> None:
            pass

    class _Completer:
        def reset(self) -> None:
            pass

    eng.assistant = _Assistant()
    eng.completer = _Completer()
    cfg.llm.server_url = "http://127.0.0.1:11434"
    cfg.llm.llama_backend = "vulkan"
    eng.apply_settings({"llm.server_url": "http://127.0.0.1:11434", "llm.llama_backend": "vulkan"})
    saved = load_config(cfg.path)
    assert saved.llm.server_url == "http://127.0.0.1:11434"
    assert saved.llm.llama_backend == "vulkan"


def test_ocr_device_help_no_longer_says_intel_only() -> None:
    field = next(f for f in FIELDS["Screen"] if f.path == "screen.ocr_device")
    assert "amd" in field.help.lower()
    assert "intel" in field.help.lower()
    assert "never the nvidia gpu" in field.help.lower()


def test_engine_apply_settings_saves_the_comment_level(tmp_path: Path) -> None:
    """Same shape as ``test_engine_apply_settings_reacts_and_saves``, for the new field: applying
    it through the engine must not raise (no ``set_*`` reaction is expected for this one - the
    assistant just reads ``cfg.assist.comments`` fresh on its next prompt) and must persist."""
    from exocortex import app as app_mod

    cfg = Config()
    cfg.path = tmp_path / "config.toml"
    eng = app_mod.Engine(cfg, EventBus())

    class _Assistant:
        def set_auto(self, enabled: bool) -> None:
            pass

        def set_explanations(self, enabled: bool) -> None:
            pass

        def set_model(self, model_id: str) -> None:
            pass

        def ensure_plan(self) -> None:
            pass

    class _Completer:
        def reset(self) -> None:
            pass

    eng.assistant = _Assistant()
    eng.completer = _Completer()
    cfg.assist.comments = "detailed"
    eng.apply_settings({"assist.comments": "detailed"})
    assert 'comments = "detailed"' in cfg.path.read_text(encoding="utf-8")
    saved = load_config(cfg.path)
    assert saved.assist.comments == "detailed"


# =====================================================================================
# FieldForm: the widget machinery SettingsDialog (and the Models window's Server tab)
# are both built on.
# =====================================================================================

def test_fieldform_load_changes_apply_and_restore_defaults(qapp: QApplication) -> None:
    fields = (
        Field("completion", "max_lines", "Lines predicted", "int", 1, 40),
        Field("explain", "enabled", "Explanations on", "bool"),
        Field("ui", "opacity", "Opacity", "float", 0.4, 1.0, 0.05),
        Field("assist", "style", "Answer style", "choice", choices=("bullets", "short", "detailed")),
        Field("assist", "persona", "Persona", "text"),
    )
    cfg = Config()
    cfg.assist.persona = "backend engineer"
    form = FieldForm(fields, cfg)
    try:
        form.load()
        assert form.widgets["completion.max_lines"].value() == cfg.completion.max_lines
        assert form.widgets["assist.persona"].text() == "backend engineer"
        assert form.changes() == {}

        form.widgets["completion.max_lines"].setValue(cfg.completion.max_lines + 3)
        form.widgets["explain.enabled"].setChecked(not cfg.explain.enabled)
        form.widgets["ui.opacity"].setValue(0.55)
        form.widgets["assist.style"].setCurrentText("detailed")
        expected = {
            "completion.max_lines": cfg.completion.max_lines + 3,
            "explain.enabled": not cfg.explain.enabled,
            "ui.opacity": 0.55,
            "assist.style": "detailed",
        }
        assert form.changes() == expected

        changed = form.apply()
        assert changed == expected
        assert cfg.completion.max_lines == expected["completion.max_lines"]
        assert cfg.explain.enabled == expected["explain.enabled"]
        assert cfg.ui.opacity == pytest.approx(0.55)
        assert cfg.assist.style == "detailed"
        assert form.changes() == {} and form.apply() == {}          # nothing left to apply

        form.restore_defaults()
        defaults = Config()
        assert form.widgets["completion.max_lines"].value() == defaults.completion.max_lines
        assert form.widgets["assist.style"].currentText() == defaults.assist.style
        assert form.widgets["assist.persona"].text() == defaults.assist.persona
        assert "assist.persona" in form.changes()                   # "backend engineer" was overwritten
    finally:
        form.widget.deleteLater()
        qapp.processEvents()


def test_fieldform_restore_defaults_can_skip_fields(qapp: QApplication) -> None:
    fields = (
        Field("llm", "model", "Model", "text"),
        Field("completion", "max_lines", "Lines predicted", "int", 1, 40),
    )
    cfg = Config()
    form = FieldForm(fields, cfg)
    try:
        form.widgets["llm.model"].setText("kept-model")
        form.widgets["completion.max_lines"].setValue(cfg.completion.max_lines + 5)
        form.restore_defaults(skip={"llm.model"})
        assert form.widgets["llm.model"].text() == "kept-model"
        assert form.widgets["completion.max_lines"].value() == Config().completion.max_lines
    finally:
        form.widget.deleteLater()
        qapp.processEvents()

"""Offscreen tests for the Models dialog's "Context" tab (2026-10-06, context library).

Same shape as ``tests/test_models_dialog.py``: a :class:`FakeActions` records every call as
``(name, args, kwargs)`` and returns canned data (``None`` by default, like an idle engine); the
file picker and the add-text dialog are monkeypatched on the instance so nothing opens a native
window.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import dataclasses
from typing import Any

import pytest
from PySide6.QtWidgets import QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QLineEdit, QMessageBox, QSpinBox

from exocortex.bus import (
    ComponentStatusEvent,
    ContextLibraryEvent,
    ContextUsageEvent,
    ModelLibraryEvent,
)
from exocortex.config import Config, ContextConfig
from exocortex.ui.models import ModelsDialog


class FakeActions:
    """Records every call as ``(name, args, kwargs)``; returns canned data the test set up."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        self.model_library_result: list[dict[str, object]] | None = None
        self.hardware_report_result: dict[str, object] | None = None
        self.model_profile_result: dict[str, object] | None = None
        self.auto_tune_result: dict[str, object] | None = None
        self.context_documents_result: list[dict[str, object]] | None = None
        self.context_usage_result: dict[str, object] | None = None
        self.raise_on: set[str] = set()

    def _record(self, name: str, args: tuple[Any, ...] = (), kwargs: dict[str, Any] | None = None) -> None:
        self.calls.append((name, args, dict(kwargs or {})))
        if name in self.raise_on:
            raise RuntimeError(f"{name} blew up")

    def calls_named(self, name: str) -> list[tuple[tuple[Any, ...], dict[str, Any]]]:
        return [(a, kw) for (n, a, kw) in self.calls if n == name]

    # ---- model manager surface (only what load() touches) ----
    def model_library(self) -> list[dict[str, object]] | None:
        self._record("model_library")
        return self.model_library_result

    def hardware_report(self, refresh: bool = False) -> dict[str, object] | None:
        self._record("hardware_report", (), {"refresh": refresh})
        return self.hardware_report_result

    def model_profile(self, key: str) -> dict[str, object] | None:
        self._record("model_profile", (key,))
        return self.model_profile_result

    # ---- context surface (exocortex.actions.EngineActions, 2026-10-06) ----
    def context_documents(self) -> list[dict[str, object]] | None:
        self._record("context_documents")
        return self.context_documents_result

    def add_context_document(self, path: str, *, mode: str | None = None) -> None:
        self._record("add_context_document", (path,), {"mode": mode})

    def add_context_text(self, name: str, text: str, *, mode: str | None = None) -> None:
        self._record("add_context_text", (name, text), {"mode": mode})

    def remove_context_document(self, doc_id: str) -> None:
        self._record("remove_context_document", (doc_id,))

    def set_context_document(self, doc_id: str, *, enabled: bool | None = None, mode: str | None = None) -> None:
        self._record("set_context_document", (doc_id,), {"enabled": enabled, "mode": mode})

    def refresh_context_digest(self, doc_id: str) -> None:
        self._record("refresh_context_digest", (doc_id,))

    def context_usage(self) -> dict[str, object] | None:
        self._record("context_usage")
        return self.context_usage_result

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)

        def f(*args: Any, **kwargs: Any) -> None:
            self._record(name, args, kwargs)
            return None

        return f


@pytest.fixture(scope="session")
def qapp() -> QApplication:
    return QApplication.instance() or QApplication(["exocortex-tests"])


@pytest.fixture()
def cfg() -> Config:
    return Config()


@pytest.fixture()
def actions() -> FakeActions:
    return FakeActions()


@pytest.fixture()
def dialog(qapp: QApplication, cfg: Config, actions: FakeActions):
    d = ModelsDialog(cfg, actions)
    d.show()
    actions.calls.clear()
    yield d
    d.close()


def _doc(doc_id: str, **overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "id": doc_id,
        "name": f"{doc_id}.md",
        "kind": "markdown",
        "chars": 3600,
        "tokens": 900,
        "enabled": True,
        "mode": "reference",
        "digest": "",
        "digest_state": "n/a",
        "source_path": f"C:/docs/{doc_id}.md",
        "added_at": "2026-10-06T10:00:00",
    }
    base.update(overrides)
    return base


def _usage_event(**overrides: Any) -> ContextUsageEvent:
    base: dict[str, Any] = dict(
        job_id=7, kind="assist", slot_tokens=4096, budget_tokens=2900, used_tokens=2400,
        transcript_tokens=1500, reference_tokens=400, lookup_tokens=120, earlier_tokens=80,
        screen_tokens=200, system_tokens=520, prompt_tokens=2350, cached_tokens=2000,
        trimmed=False, anchor_moved=False, chars_per_token=3.7, documents=2,
    )
    base.update(overrides)
    return ContextUsageEvent(**base)


# ============================================================================================ tab
def test_context_tab_exists_and_show_tab_selects_it(dialog: ModelsDialog) -> None:
    assert "Context" in ModelsDialog.TABS
    dialog.show_tab("Context")
    assert dialog.tabs.tabText(dialog.tabs.currentIndex()) == "Context"
    assert ModelsDialog.TABS.index("Context") == ModelsDialog.TABS.index("Server") + 1


def test_load_reads_documents_and_usage(dialog: ModelsDialog, actions: FakeActions) -> None:
    dialog.load()
    assert len(actions.calls_named("context_documents")) == 1
    assert len(actions.calls_named("context_usage")) == 1


# ======================================================================================== documents
def test_document_rows_render(dialog: ModelsDialog, actions: FakeActions) -> None:
    actions.context_documents_result = [
        _doc("aaa111", name="README.md", kind="markdown", tokens=1234, mode="inject", digest_state="n/a"),
        _doc("bbb222", name="design.pdf", kind="pdf", tokens=5678, mode="reference", enabled=False,
             digest="A design doc about threads.", digest_state="ready"),
    ]
    dialog._refresh_context()

    t = dialog.context_table
    assert t.rowCount() == 2
    assert [t.horizontalHeaderItem(c).text() for c in range(t.columnCount())] == [
        "Name", "Kind", "Tokens", "Mode", "On", "Digest"]
    assert [t.item(0, c).text() for c in range(6)] == ["README.md", "markdown", "1234", "inject", "on", "n/a"]
    assert [t.item(1, c).text() for c in range(6)] == ["design.pdf", "pdf", "5678", "reference", "off", "ready"]
    # the digest text is the row's tool tip, on every cell of the row
    assert t.item(1, 5).toolTip() == "A design doc about threads."
    assert t.item(1, 0).toolTip() == "A design doc about threads."
    assert t.item(0, 0).toolTip() == ""


def test_tokens_column_falls_back_to_a_chars_estimate(dialog: ModelsDialog, actions: FakeActions) -> None:
    d = _doc("ccc333", chars=3600)
    del d["tokens"]
    actions.context_documents_result = [d]
    dialog._refresh_context()
    # ceil(3600 / 3.6) = 1000 with the default chars_per_token
    assert dialog.context_table.item(0, 2).text() == "1000"


def test_none_and_empty_document_lists_render_an_empty_table(dialog: ModelsDialog, actions: FakeActions) -> None:
    actions.context_documents_result = None
    dialog._refresh_context()
    assert dialog.context_table.rowCount() == 0
    actions.context_documents_result = []
    dialog._refresh_context()
    assert dialog.context_table.rowCount() == 0


def test_selection_is_kept_by_id_across_refreshes(dialog: ModelsDialog, actions: FakeActions) -> None:
    actions.context_documents_result = [_doc("aaa"), _doc("bbb"), _doc("ccc")]
    dialog._refresh_context()
    dialog.context_table.setCurrentCell(1, 0)
    assert dialog._selected_context_row()["id"] == "bbb"

    # a new document lands at the front: "bbb" moves to row 2 and stays selected
    actions.context_documents_result = [_doc("zzz"), _doc("aaa"), _doc("bbb"), _doc("ccc")]
    dialog.on_event(ContextLibraryEvent(reason="added", doc_id="zzz"))
    assert dialog.context_table.currentRow() == 2
    assert dialog._selected_context_row()["id"] == "bbb"


def test_buttons_follow_the_selection(dialog: ModelsDialog, actions: FakeActions) -> None:
    actions.context_documents_result = [
        _doc("aaa", mode="reference", enabled=True),
        _doc("bbb", mode="inject", enabled=False),
    ]
    dialog._refresh_context()
    # nothing selected: row-bound buttons are off, adding is always possible
    dialog.context_table.setCurrentCell(-1, -1)
    for b in (dialog.context_remove_btn, dialog.context_toggle_btn, dialog.context_mode_btn,
              dialog.context_redigest_btn):
        assert b.isEnabled() is False
    assert dialog.context_add_files_btn.isEnabled() and dialog.context_add_text_btn.isEnabled()

    dialog.context_table.setCurrentCell(0, 0)
    assert dialog.context_toggle_btn.text() == "Disable"
    assert "inject" in dialog.context_mode_btn.text()
    assert dialog.context_redigest_btn.isEnabled() is True

    dialog.context_table.setCurrentCell(1, 0)
    assert dialog.context_toggle_btn.text() == "Enable"
    assert "reference" in dialog.context_mode_btn.text()
    assert dialog.context_redigest_btn.isEnabled() is False  # an inject document has no digest


# ============================================================================== add / remove / edit
def test_add_text_calls_add_context_text(
    dialog: ModelsDialog, actions: FakeActions, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(dialog, "_prompt_context_text", lambda: ("Talk notes", "# Notes\nhello"))
    dialog.context_add_text_btn.click()
    assert actions.calls_named("add_context_text") == [(("Talk notes", "# Notes\nhello"), {"mode": None})]


def test_add_text_cancelled_does_nothing(
    dialog: ModelsDialog, actions: FakeActions, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(dialog, "_prompt_context_text", lambda: None)
    dialog.context_add_text_btn.click()
    assert actions.calls_named("add_context_text") == []


def test_add_files_calls_add_context_document_per_path(
    dialog: ModelsDialog, actions: FakeActions, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(dialog, "_pick_context_files", lambda: ["C:/docs/a.md", "C:/docs/b.pdf"])
    dialog.context_add_files_btn.click()
    assert actions.calls_named("add_context_document") == [
        (("C:/docs/a.md",), {"mode": None}),
        (("C:/docs/b.pdf",), {"mode": None}),
    ]


def test_add_files_cancelled_does_nothing(
    dialog: ModelsDialog, actions: FakeActions, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(dialog, "_pick_context_files", lambda: [])
    dialog.context_add_files_btn.click()
    assert actions.calls_named("add_context_document") == []


def test_one_failing_add_does_not_stop_the_rest(
    dialog: ModelsDialog, actions: FakeActions, monkeypatch: pytest.MonkeyPatch
) -> None:
    actions.raise_on = {"add_context_document"}
    monkeypatch.setattr(dialog, "_pick_context_files", lambda: ["C:/docs/a.md", "C:/docs/b.md"])
    dialog.context_add_files_btn.click()
    assert len(actions.calls_named("add_context_document")) == 2
    assert "failed" in dialog.context_status.text()


def test_remove_asks_and_calls_only_on_yes(
    dialog: ModelsDialog, actions: FakeActions, monkeypatch: pytest.MonkeyPatch
) -> None:
    actions.context_documents_result = [_doc("aaa"), _doc("bbb")]
    dialog._refresh_context()
    dialog.context_table.setCurrentCell(1, 0)

    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **kw: QMessageBox.StandardButton.No))
    dialog.context_remove_btn.click()
    assert actions.calls_named("remove_context_document") == []

    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **kw: QMessageBox.StandardButton.Yes))
    dialog.context_remove_btn.click()
    assert actions.calls_named("remove_context_document") == [(("bbb",), {})]


def test_enable_toggle_flips_the_selected_row(dialog: ModelsDialog, actions: FakeActions) -> None:
    actions.context_documents_result = [_doc("aaa", enabled=True), _doc("bbb", enabled=False)]
    dialog._refresh_context()

    dialog.context_table.setCurrentCell(0, 0)
    dialog.context_toggle_btn.click()
    dialog.context_table.setCurrentCell(1, 0)
    dialog.context_toggle_btn.click()
    assert actions.calls_named("set_context_document") == [
        (("aaa",), {"enabled": False, "mode": None}),
        (("bbb",), {"enabled": True, "mode": None}),
    ]


def test_mode_switch_flips_between_inject_and_reference(dialog: ModelsDialog, actions: FakeActions) -> None:
    actions.context_documents_result = [_doc("aaa", mode="reference"), _doc("bbb", mode="inject")]
    dialog._refresh_context()

    dialog.context_table.setCurrentCell(0, 0)
    dialog.context_mode_btn.click()
    dialog.context_table.setCurrentCell(1, 0)
    dialog.context_mode_btn.click()
    assert actions.calls_named("set_context_document") == [
        (("aaa",), {"enabled": None, "mode": "inject"}),
        (("bbb",), {"enabled": None, "mode": "reference"}),
    ]


def test_redigest_calls_refresh_context_digest(dialog: ModelsDialog, actions: FakeActions) -> None:
    actions.context_documents_result = [_doc("aaa", mode="reference", digest_state="failed")]
    dialog._refresh_context()
    dialog.context_table.setCurrentCell(0, 0)
    dialog.context_redigest_btn.click()
    assert actions.calls_named("refresh_context_digest") == [(("aaa",), {})]


def test_row_buttons_without_a_selection_never_call_actions(dialog: ModelsDialog, actions: FakeActions) -> None:
    actions.context_documents_result = [_doc("aaa")]
    dialog._refresh_context()
    dialog.context_table.setCurrentCell(-1, -1)
    dialog._on_context_remove()
    dialog._on_context_toggle()
    dialog._on_context_mode()
    dialog._on_context_redigest()
    for name in ("remove_context_document", "set_context_document", "refresh_context_digest"):
        assert actions.calls_named(name) == []


def test_failing_row_actions_are_swallowed(
    dialog: ModelsDialog, actions: FakeActions, monkeypatch: pytest.MonkeyPatch
) -> None:
    actions.context_documents_result = [_doc("aaa", mode="reference")]
    dialog._refresh_context()
    dialog.context_table.setCurrentCell(0, 0)
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **kw: QMessageBox.StandardButton.Yes))
    monkeypatch.setattr(dialog, "_prompt_context_text", lambda: ("n", "t"))
    actions.raise_on = {"remove_context_document", "set_context_document", "refresh_context_digest",
                        "add_context_text"}
    dialog._on_context_remove()
    dialog._on_context_toggle()
    dialog._on_context_mode()
    dialog._on_context_redigest()
    dialog._on_context_add_text()  # none of these raise
    assert "failed" in dialog.context_status.text()


def test_picker_helpers_return_the_documented_types(dialog: ModelsDialog, monkeypatch: pytest.MonkeyPatch) -> None:
    from PySide6.QtWidgets import QFileDialog

    monkeypatch.setattr(QFileDialog, "getOpenFileNames",
                        staticmethod(lambda *a, **kw: (["C:/x/a.md", "C:/x/b.txt"], "")))
    assert dialog._pick_context_files() == ["C:/x/a.md", "C:/x/b.txt"]

    monkeypatch.setattr(QFileDialog, "getOpenFileNames", staticmethod(lambda *a, **kw: ([], "")))
    assert dialog._pick_context_files() == []


def test_open_folder_without_a_folder_says_so(
    dialog: ModelsDialog, cfg: Config, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg.context.dir = str(tmp_path / "missing")
    dialog._on_context_open_folder()
    assert "does not exist" in dialog.context_status.text()

    opened: list[object] = []
    from exocortex.ui import models as models_mod

    monkeypatch.setattr(models_mod.QDesktopServices, "openUrl", staticmethod(lambda url: opened.append(url)))
    cfg.context.dir = str(tmp_path)
    dialog._on_context_open_folder()
    assert len(opened) == 1


# ========================================================================================== events
def test_library_event_refreshes_and_shows_error_reasons(dialog: ModelsDialog, actions: FakeActions) -> None:
    actions.context_documents_result = [_doc("aaa")]
    dialog.on_event(ContextLibraryEvent(reason="added", doc_id="aaa"))
    assert dialog.context_table.rowCount() == 1
    assert dialog.context_status.text() == ""

    dialog.on_event(ContextLibraryEvent(reason="error: unreadable pdf"))
    assert "unreadable pdf" in dialog.context_status.text()

    dialog.on_event(ContextLibraryEvent(reason="digest", doc_id="aaa"))
    assert "unreadable pdf" in dialog.context_status.text()   # an unrelated event never wipes an unread error
    dialog._context_action_begins()                           # ... the user's next action does
    assert dialog.context_status.text() == ""


def test_events_never_raise_with_a_none_returning_actions(dialog: ModelsDialog) -> None:
    # FakeActions returns None for every context call here
    dialog.on_event(ContextLibraryEvent(reason="added", doc_id="x"))
    dialog.on_event(_usage_event())
    dialog.on_event(ModelLibraryEvent(reason="loaded"))
    dialog.on_event(ComponentStatusEvent(component="llm", state="ready"))
    dialog.load()
    assert dialog.context_table.rowCount() == 0


def test_events_never_raise_when_actions_raise(dialog: ModelsDialog, actions: FakeActions) -> None:
    actions.raise_on = {"context_documents", "context_usage"}
    dialog.on_event(ContextLibraryEvent(reason="added"))
    dialog.load()
    assert dialog.context_table.rowCount() == 0


# ======================================================================================= usage panel
def test_usage_panel_renders_an_event(dialog: ModelsDialog) -> None:
    dialog.on_event(_usage_event())
    text = dialog.context_usage_label.text()
    assert "Server context per request: 4096 tokens" in text
    assert "last answer (assist)" in text
    assert "used 2400 of 2900 tokens" in text
    for piece in ("transcript 1500", "reference 400", "lookup 120", "earlier 80", "screen 200"):
        assert piece in text
    assert "server counted 2350 (2000 cached)" in text
    assert "ok" in text and "trimmed" not in text
    assert "2 documents" in text
    assert "3.70 chars/token" in text


def test_usage_panel_marks_trimmed_and_missing_server_counts(dialog: ModelsDialog) -> None:
    dialog.on_event(_usage_event(trimmed=True, anchor_moved=True, prompt_tokens=None, cached_tokens=None,
                                 documents=1))
    text = dialog.context_usage_label.text()
    assert "trimmed" in text
    assert "window advanced" in text
    assert "server counted" not in text
    assert "1 document" in text and "1 documents" not in text


def test_usage_panel_without_cached_count(dialog: ModelsDialog) -> None:
    dialog.on_event(_usage_event(cached_tokens=None))
    text = dialog.context_usage_label.text()
    assert "server counted 2350" in text and "cached" not in text


def test_usage_panel_renders_the_actions_snapshot(dialog: ModelsDialog, actions: FakeActions) -> None:
    actions.context_usage_result = {
        "has_usage": True, "job_id": 3, "kind": "ask", "slot_tokens": 8192, "budget_tokens": 6000,
        "used_tokens": 1000, "transcript_tokens": 700, "reference_tokens": 0, "lookup_tokens": 0,
        "earlier_tokens": 0, "screen_tokens": 100, "system_tokens": 200, "prompt_tokens": 950,
        "cached_tokens": 0, "trimmed": False, "anchor_moved": False, "documents_used": 0, "documents": 4,
        "chars_per_token": 3.6, "earlier_summary": "They discussed the roadmap.",
    }
    dialog.load()
    text = dialog.context_usage_label.text()
    assert "8192 tokens" in text and "last answer (ask)" in text and "used 1000 of 6000" in text
    assert "server counted 950 (0 cached)" in text
    assert "0 documents" in text
    assert "roadmap" in dialog.context_usage_label.toolTip()


@pytest.mark.parametrize("payload", [
    None,
    {},
    {"slot_tokens": 4096},
    {"has_usage": False, "slot_tokens": 4096, "chars_per_token": 3.6, "documents": 2, "kind": None},
    {"has_usage": True, "kind": "assist"},                      # missing nearly every key
    {"has_usage": True, "kind": "assist", "used_tokens": "junk", "slot_tokens": "junk",
     "chars_per_token": "junk", "transcript_tokens": None},      # junk values
])
def test_usage_panel_tolerates_empty_and_partial_data(dialog: ModelsDialog, payload: object) -> None:
    dialog._render_usage(payload)
    text = dialog.context_usage_label.text()
    assert text.startswith("Server context per request:")


def test_usage_panel_before_any_job(dialog: ModelsDialog) -> None:
    dialog._render_usage({"has_usage": False, "slot_tokens": 4096, "chars_per_token": 3.6, "documents": 2})
    text = dialog.context_usage_label.text()
    assert "4096 tokens" in text
    assert "no answer yet" in text
    assert "2 enabled documents" in text
    assert "last answer" not in text

    dialog._render_usage(None)
    assert "unknown" in dialog.context_usage_label.text()
    assert "no answer yet" in dialog.context_usage_label.text()


# ============================================================================================ form
def test_context_fields_cover_every_context_config_field() -> None:
    declared = {f.key for f in ModelsDialog._CONTEXT_FIELDS}
    assert declared == {f.name for f in dataclasses.fields(ContextConfig)}
    assert all(f.section == "context" for f in ModelsDialog._CONTEXT_FIELDS)
    assert len({f.path for f in ModelsDialog._CONTEXT_FIELDS}) == len(ModelsDialog._CONTEXT_FIELDS)


def test_context_field_defaults_fit_their_ranges() -> None:
    defaults = ContextConfig()
    for f in ModelsDialog._CONTEXT_FIELDS:
        value = getattr(defaults, f.key)
        if f.kind in ("int", "float"):
            assert f.lo <= value <= f.hi, f.path
        elif f.kind == "choice":
            assert value in f.choices, f.path


def test_dir_is_a_restart_marked_text_field() -> None:
    by_key = {f.key: f for f in ModelsDialog._CONTEXT_FIELDS}
    assert by_key["dir"].kind == "text"
    assert by_key["dir"].restart is True
    assert [f.key for f in ModelsDialog._CONTEXT_FIELDS if f.restart] == ["dir"]


def test_form_loads_the_config_and_widget_types(dialog: ModelsDialog, cfg: Config) -> None:
    w = dialog._context_form.widgets
    assert isinstance(w["context.dir"], QLineEdit)
    assert isinstance(w["context.default_mode"], QComboBox)
    assert [w["context.include_in"].itemText(i) for i in range(w["context.include_in"].count())] == [
        "all", "questions", "none"]
    assert isinstance(w["context.reference_tokens"], QSpinBox)
    assert isinstance(w["context.lookup_min_score"], QDoubleSpinBox)
    assert isinstance(w["context.digest"], QCheckBox)
    assert w["context.reference_tokens"].value() == cfg.context.reference_tokens
    assert w["context.trim_step"].value() == pytest.approx(cfg.context.trim_step)
    assert w["context.budget_mode"].currentText() == "auto"
    assert w["context.compaction"].currentText() == "summary"


def test_every_field_has_a_help_text_or_is_self_explanatory(dialog: ModelsDialog) -> None:
    # the long-form knobs (dir, modes, budgets, ...) all carry a tool tip
    for key in ("dir", "default_mode", "include_in", "reference_tokens", "lookup_tokens", "budget_mode",
                "lane_share", "trim_step", "compaction", "chars_per_token"):
        assert dialog._context_form.widgets[f"context.{key}"].toolTip(), key


def test_context_apply_emits_only_the_changed_keys(dialog: ModelsDialog, cfg: Config) -> None:
    seen: list[dict[str, object]] = []
    dialog.applied.connect(seen.append)

    w = dialog._context_form.widgets
    w["context.include_in"].setCurrentText("questions")
    w["context.reference_tokens"].setValue(2000)
    w["context.trim_step"].setValue(0.5)

    dialog.context_apply_btn.click()

    assert len(seen) == 1
    assert seen[0] == {"context.include_in": "questions", "context.reference_tokens": 2000,
                       "context.trim_step": 0.5}
    assert cfg.context.include_in == "questions"
    assert cfg.context.reference_tokens == 2000
    assert cfg.context.trim_step == pytest.approx(0.5)
    assert dialog.context_note.text() == "Applied."


def test_context_apply_with_no_changes_emits_nothing(dialog: ModelsDialog) -> None:
    seen: list[dict[str, object]] = []
    dialog.applied.connect(seen.append)
    dialog.context_apply_btn.click()
    assert seen == []
    assert dialog.context_note.text() == ""


def test_changing_dir_sets_the_restart_note(dialog: ModelsDialog, cfg: Config) -> None:
    seen: list[dict[str, object]] = []
    dialog.applied.connect(seen.append)
    dialog._context_form.widgets["context.dir"].setText("D:/my-docs")
    dialog.context_apply_btn.click()
    assert seen == [{"context.dir": "D:/my-docs"}]
    assert cfg.context.dir == "D:/my-docs"
    assert "next launch" in dialog.context_note.text()
    assert "Library folder" in dialog.context_note.text()


def test_context_apply_does_not_touch_server_form(dialog: ModelsDialog) -> None:
    seen: list[dict[str, object]] = []
    dialog.applied.connect(seen.append)
    dialog._server_form.widgets["llm.backend"].setCurrentText("ollama")
    dialog.context_apply_btn.click()
    assert seen == []  # the Server tab's pending edit is its own Apply's business


def test_restore_defaults_resets_widgets_but_writes_nothing_until_apply(dialog: ModelsDialog, cfg: Config) -> None:
    cfg.context.reference_tokens = 3000
    cfg.context.compaction = "off"
    dialog.load()
    w = dialog._context_form.widgets
    assert w["context.reference_tokens"].value() == 3000
    assert w["context.compaction"].currentText() == "off"

    dialog.context_restore_btn.click()
    assert w["context.reference_tokens"].value() == ContextConfig().reference_tokens
    assert w["context.compaction"].currentText() == "summary"
    assert cfg.context.reference_tokens == 3000  # still the old value until Apply

    seen: list[dict[str, object]] = []
    dialog.applied.connect(seen.append)
    dialog.context_apply_btn.click()
    assert seen == [{"context.reference_tokens": ContextConfig().reference_tokens,
                     "context.compaction": "summary"}]


def test_load_resets_notes_and_reloads_the_form(dialog: ModelsDialog, cfg: Config) -> None:
    dialog.context_note.setText("stale")
    dialog.context_status.setText("stale")
    cfg.context.lookup_top_k = 5
    dialog.load()
    assert dialog.context_note.text() == ""
    assert dialog.context_status.text() == ""
    assert dialog._context_form.widgets["context.lookup_top_k"].value() == 5


def test_an_added_event_does_not_wipe_a_pending_error_status(dialog: ModelsDialog) -> None:
    """Review 2026-10-06: two adds in one go (a bad path, then a good file) drained both events
    in one pass and the second wiped the first's error before anyone read it."""
    dialog.on_event(ContextLibraryEvent(reason="error: C:/nope.md is not a file"))
    assert "nope.md" in dialog.context_status.text()
    dialog.on_event(ContextLibraryEvent(reason="added", doc_id="abc"))
    assert "nope.md" in dialog.context_status.text()


def test_a_failed_digest_is_shown_in_the_status_line(dialog: ModelsDialog, actions: FakeActions) -> None:
    actions.context_documents_result = [_doc("abc", name="design.pdf", digest_state="failed")]
    dialog.on_event(ContextLibraryEvent(reason="digest failed", doc_id="abc"))
    assert "design.pdf" in dialog.context_status.text() and "failed" in dialog.context_status.text()

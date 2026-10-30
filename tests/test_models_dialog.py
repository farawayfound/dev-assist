"""Offscreen tests for the Models... dialog (Library / Discover / Tuning / Server / Hardware).

A purpose-built :class:`FakeActions` records every call it receives (name, args, kwargs) and
returns whatever canned value the test wired up (``None`` by default, matching the real engine
when a backend is idle/unreachable) -- the dialog must tolerate that everywhere, per its own
module docstring.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from typing import Any

import pytest
from PySide6.QtWidgets import QApplication, QCheckBox, QComboBox, QLineEdit, QMessageBox

from exocortex.bus import ComponentStatusEvent, ModelInfoEvent, ModelLibraryEvent, ModelPullEvent, ModelSearchEvent
from exocortex.config import Config
from exocortex.ui.models import ModelsDialog


class FakeActions:
    """Records every call as ``(name, args, kwargs)``; returns canned data the test set up."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        self.model_library_result: list[dict[str, object]] | None = None
        self.hardware_report_result: dict[str, object] | None = None
        self.model_profile_result: dict[str, object] | None = None
        self.auto_tune_result: dict[str, object] | None = None

    def _record(self, name: str, args: tuple[Any, ...] = (), kwargs: dict[str, Any] | None = None) -> None:
        self.calls.append((name, args, dict(kwargs or {})))

    def calls_named(self, name: str) -> list[tuple[tuple[Any, ...], dict[str, Any]]]:
        return [(a, kw) for (n, a, kw) in self.calls if n == name]

    # ---- model manager surface (exocortex.actions.EngineActions) ----
    def model_library(self) -> list[dict[str, object]] | None:
        self._record("model_library")
        return self.model_library_result

    def loaded_model(self) -> str:
        self._record("loaded_model")
        return ""

    def load_model(self, key: str) -> None:
        self._record("load_model", (key,))

    def unload_model(self) -> None:
        self._record("unload_model")

    def set_default_model(self, key: str) -> None:
        self._record("set_default_model", (key,))

    def delete_model(self, key: str) -> None:
        self._record("delete_model", (key,))

    def pull_model(self, repo_id: str, quant: str = "auto", *, include_mmproj: bool = False) -> None:
        self._record("pull_model", (repo_id, quant), {"include_mmproj": include_mmproj})

    def cancel_pull(self) -> None:
        self._record("cancel_pull")

    def search_models(self, query: str, request_id: int) -> None:
        self._record("search_models", (query, request_id))

    def inspect_model(self, repo_id: str, request_id: int) -> None:
        self._record("inspect_model", (repo_id, request_id))

    def hardware_report(self, refresh: bool = False) -> dict[str, object] | None:
        self._record("hardware_report", (), {"refresh": refresh})
        return self.hardware_report_result

    def model_profile(self, key: str) -> dict[str, object] | None:
        self._record("model_profile", (key,))
        return self.model_profile_result

    def set_model_profile(self, key: str, values: dict[str, object] | None) -> None:
        self._record("set_model_profile", (key, values))

    def auto_tune_model(self, key: str) -> dict[str, object] | None:
        self._record("auto_tune_model", (key,))
        return self.auto_tune_result

    def apply_settings(self, changed: dict[str, object]) -> None:
        self._record("apply_settings", (changed,))

    # ---- safety net: any other EngineActions method the dialog might call ----
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
    d.show()  # isVisible()/isVisibleTo() on children needs a shown top-level, even offscreen
    actions.calls.clear()  # drop the construction-time load() calls; tests assert their own
    yield d
    d.close()


def _row(key: str, **overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "key": key,
        "name": key.split("/")[-1],
        "size_bytes": 4_000_000_000,
        "source": "library",
        "quant": "Q4_K_M",
        "path": f"C:/models/llm/{key}",
        "context_length": 8192,
        "loaded": False,
        "is_default": False,
        "has_profile": False,
    }
    base.update(overrides)
    return base


# ============================================================================================ tabs
def test_seven_tabs_in_order(dialog: ModelsDialog) -> None:
    assert ModelsDialog.TABS == ("Library", "Discover", "Tuning", "Server", "Context", "Capabilities", "Hardware")
    assert dialog.tabs.count() == 7
    for i, name in enumerate(ModelsDialog.TABS):
        assert dialog.tabs.tabText(i) == name
    for name in ModelsDialog.TABS:
        dialog.show_tab(name)
        assert dialog.tabs.currentIndex() == ModelsDialog.TABS.index(name)


# ========================================================================================= library
def test_library_rows_render_dicts(dialog: ModelsDialog, actions: FakeActions) -> None:
    rows = [
        _row("acme/small", name="Small One", quant="Q4_K_M", size_bytes=4_000_000_000,
             context_length=8192, loaded=True, is_default=False),
        _row("acme/big", name="Big One", source="library", is_default=True, loaded=False),
        _row("lmstudio-user/other", name="LM Studio Model", source="lmstudio"),
    ]
    actions.model_library_result = rows
    dialog._refresh_library()

    table = dialog.library_table
    assert table.rowCount() == 3
    assert table.item(0, 0).text() == "Small One"
    assert table.item(0, 1).text() == "Q4_K_M"
    assert table.item(0, 3).text() == "library"
    assert table.item(0, 4).text() == "8192"
    assert table.item(0, 5).text() == "loaded"
    assert table.item(1, 5).text() == "default"
    assert table.item(2, 3).text() == "lmstudio"
    assert table.item(2, 5).text() == ""


def test_load_button_and_double_click_load_the_selected_key(dialog: ModelsDialog, actions: FakeActions) -> None:
    rows = [_row("acme/small"), _row("acme/big")]
    actions.model_library_result = rows
    dialog._refresh_library()

    dialog.library_table.setCurrentCell(1, 0)
    dialog.load_btn.click()
    assert actions.calls_named("load_model") == [(("acme/big",), {})]

    actions.calls.clear()
    dialog.library_table.setCurrentCell(0, 0)
    dialog.library_table.itemDoubleClicked.emit(dialog.library_table.item(0, 0))
    assert actions.calls_named("load_model") == [(("acme/small",), {})]


def test_unload_calls_unload_model(dialog: ModelsDialog, actions: FakeActions) -> None:
    actions.model_library_result = [_row("acme/small", loaded=True)]
    dialog._refresh_library()
    dialog.unload_btn.click()
    assert actions.calls_named("unload_model") == [((), {})]


def test_set_default_calls_the_selected_key(dialog: ModelsDialog, actions: FakeActions) -> None:
    actions.model_library_result = [_row("acme/small"), _row("acme/big")]
    dialog._refresh_library()
    dialog.library_table.setCurrentCell(1, 0)
    dialog.set_default_btn.click()
    assert actions.calls_named("set_default_model") == [(("acme/big",), {})]


def test_delete_disabled_for_lmstudio_and_loaded_rows(dialog: ModelsDialog, actions: FakeActions) -> None:
    actions.model_library_result = [
        _row("acme/loaded", loaded=True),
        _row("lmstudio-user/other", source="lmstudio"),
        _row("acme/plain"),
    ]
    dialog._refresh_library()

    dialog.library_table.setCurrentCell(0, 0)
    assert dialog.delete_btn.isEnabled() is False

    dialog.library_table.setCurrentCell(1, 0)
    assert dialog.delete_btn.isEnabled() is False

    dialog.library_table.setCurrentCell(2, 0)
    assert dialog.delete_btn.isEnabled() is True


def test_delete_asks_and_calls_delete_model_only_on_yes(
    dialog: ModelsDialog, actions: FakeActions, monkeypatch: pytest.MonkeyPatch
) -> None:
    actions.model_library_result = [_row("acme/plain")]
    dialog._refresh_library()
    dialog.library_table.setCurrentCell(0, 0)

    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **kw: QMessageBox.StandardButton.No))
    dialog.delete_btn.click()
    assert actions.calls_named("delete_model") == []

    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **kw: QMessageBox.StandardButton.Yes))
    dialog.delete_btn.click()
    assert actions.calls_named("delete_model") == [(("acme/plain",), {})]


def test_component_status_event_updates_the_status_line(dialog: ModelsDialog) -> None:
    dialog.on_event(ComponentStatusEvent(component="llm", state="error", detail="server crashed"))
    assert dialog.library_status.text() == "server crashed"
    dialog.on_event(ComponentStatusEvent(component="llm", state="ready", detail=""))
    assert dialog.library_status.text() == "ready"


def test_model_library_event_triggers_a_refresh(dialog: ModelsDialog, actions: FakeActions) -> None:
    actions.model_library_result = [_row("acme/small")]
    dialog.on_event(ModelLibraryEvent(reason="pull finished"))
    assert dialog.library_table.rowCount() == 1
    assert actions.calls_named("model_library")


# ========================================================================================= discover
def test_stale_search_event_ignored_fresh_one_fills_table(dialog: ModelsDialog, actions: FakeActions) -> None:
    dialog.search_edit.setText("gemma")
    dialog._do_search()
    stale_rid = dialog._search_request_id
    dialog.search_edit.setText("gemma 4")
    dialog._do_search()
    fresh_rid = dialog._search_request_id
    assert fresh_rid != stale_rid

    dialog.on_event(ModelSearchEvent(request_id=stale_rid, query="gemma", results=({"id": "old/repo"},)))
    assert dialog.search_table.rowCount() == 0

    dialog.on_event(ModelSearchEvent(request_id=fresh_rid, query="gemma 4",
                                      results=({"id": "unsloth/gemma-4-12b-it-GGUF", "downloads": 100},)))
    assert dialog.search_table.rowCount() == 1
    assert dialog.search_table.item(0, 0).text() == "unsloth/gemma-4-12b-it-GGUF"


def test_model_info_event_fills_quant_combo_and_mmproj_checkbox(dialog: ModelsDialog) -> None:
    dialog._inspect_request_id = 7
    dialog.on_event(ModelInfoEvent(
        request_id=7, repo_id="unsloth/gemma-4-12b-it-GGUF",
        quants=({"name": "Q4_K_M", "size_bytes": 4_000_000_000}, {"name": "Q8_0", "size_bytes": 8_000_000_000}),
        chosen="Q4_K_M", architecture="gemma", params=12_000_000_000, context_length=32768, mmproj=True,
    ))
    assert dialog.quant_combo.itemData(0) == "auto"
    assert "auto" in dialog.quant_combo.itemText(0)
    assert "Q4_K_M" in dialog.quant_combo.itemText(0)
    assert dialog.quant_combo.itemData(1) == "Q4_K_M"
    assert dialog.quant_combo.itemData(2) == "Q8_0"
    assert dialog.mmproj_cb.isEnabled() is True

    dialog.on_event(ModelInfoEvent(request_id=7, repo_id="other/repo", mmproj=False))
    assert dialog.mmproj_cb.isEnabled() is False

    # a stale answer (old request id) must not touch the combo
    dialog._inspect_request_id = 9
    dialog.on_event(ModelInfoEvent(request_id=7, repo_id="stale/repo", chosen="Q2_K"))
    assert "Q2_K" not in dialog.quant_combo.itemText(0)


def test_pull_calls_pull_model_with_auto_quant_and_no_mmproj(dialog: ModelsDialog, actions: FakeActions) -> None:
    dialog._search_rows = [{"id": "unsloth/gemma-4-12b-it-GGUF"}]
    dialog._populate_search_table()
    dialog.search_table.setCurrentCell(0, 0)

    dialog.pull_btn.click()
    assert actions.calls_named("pull_model") == [
        (("unsloth/gemma-4-12b-it-GGUF", "auto"), {"include_mmproj": False})
    ]


def test_pull_progress_and_load_now(dialog: ModelsDialog, actions: FakeActions) -> None:
    dialog.show_tab("Discover")  # load_now_btn's isVisible() needs its tab page to be the current one
    dialog._search_rows = [{"id": "unsloth/gemma-4-12b-it-GGUF"}]
    dialog._populate_search_table()
    dialog.search_table.setCurrentCell(0, 0)
    dialog.pull_btn.click()
    assert dialog.cancel_pull_btn.isEnabled() is True

    dialog.on_event(ModelPullEvent(repo_id="unsloth/gemma-4-12b-it-GGUF", state="queued", file="model.gguf"))
    assert "queued" in dialog.pull_label.text()
    assert dialog.load_now_btn.isVisible() is False

    dialog.on_event(ModelPullEvent(repo_id="unsloth/gemma-4-12b-it-GGUF", state="downloading",
                                    file="model-00001-of-00002.gguf", done_bytes=50, total_bytes=100))
    assert dialog.pull_progress.value() == 50
    assert dialog.cancel_pull_btn.isEnabled() is True

    actions.model_library_result = [_row("unsloth/gemma-4-12b-it")]
    dialog.on_event(ModelPullEvent(repo_id="unsloth/gemma-4-12b-it-GGUF", state="done",
                                    model_key="unsloth/gemma-4-12b-it"))
    assert dialog.load_now_btn.isVisible() is True
    assert dialog.cancel_pull_btn.isEnabled() is False

    dialog.load_now_btn.click()
    assert actions.calls_named("load_model") == [(("unsloth/gemma-4-12b-it",), {})]
    assert dialog.load_now_btn.isVisible() is False


def test_cancel_pull_calls_cancel_pull(dialog: ModelsDialog, actions: FakeActions) -> None:
    dialog._search_rows = [{"id": "unsloth/gemma-4-12b-it-GGUF"}]
    dialog._populate_search_table()
    dialog.search_table.setCurrentCell(0, 0)
    assert dialog.cancel_pull_btn.isEnabled() is False

    dialog.pull_btn.click()
    assert dialog.cancel_pull_btn.isEnabled() is True
    dialog.cancel_pull_btn.click()
    assert actions.calls_named("cancel_pull") == [((), {})]


def test_pull_error_event_shows_detail(dialog: ModelsDialog) -> None:
    dialog.on_event(ModelPullEvent(repo_id="x/y", state="error", detail="disk full"))
    assert "disk full" in dialog.pull_label.text()


# ========================================================================================== tuning
def test_tuning_overrides_off_saves_none(dialog: ModelsDialog, actions: FakeActions) -> None:
    actions.model_library_result = [_row("acme/small")]
    dialog._refresh_library()
    dialog.tuning_apply_btn.click()
    assert actions.calls_named("set_model_profile") == [(("acme/small", None), {})]


def test_tuning_ticked_override_saves_only_that_field(dialog: ModelsDialog, actions: FakeActions) -> None:
    actions.model_library_result = [_row("acme/small")]
    dialog._refresh_library()
    cb, widget = dialog._tuning_widgets["ctx_size"]
    cb.setChecked(True)
    widget.setValue(12345)

    dialog.tuning_apply_btn.click()
    assert actions.calls_named("set_model_profile") == [(("acme/small", {"ctx_size": 12345}), {})]


def test_auto_tune_fills_fields_from_returned_dict(dialog: ModelsDialog, actions: FakeActions) -> None:
    actions.model_library_result = [_row("acme/small")]
    dialog._refresh_library()
    actions.auto_tune_result = {"ctx_size": 8192, "kv_cache_type": "q4_0"}

    dialog.autotune_btn.click()

    ctx_cb, ctx_widget = dialog._tuning_widgets["ctx_size"]
    kv_cb, kv_widget = dialog._tuning_widgets["kv_cache_type"]
    assert ctx_cb.isChecked() is True
    assert ctx_widget.value() == 8192
    assert kv_cb.isChecked() is True
    assert kv_widget.currentText() == "q4_0"
    assert actions.calls_named("auto_tune_model") == [(("acme/small",), {})]


def test_presets_change_the_right_fields(dialog: ModelsDialog, actions: FakeActions) -> None:
    actions.model_library_result = [_row("acme/small", context_length=32768)]
    dialog._refresh_library()

    dialog.preset_speed_btn.click()
    assert dialog._tuning_widgets["ctx_size"][1].value() == 4096
    assert dialog._tuning_widgets["kv_cache_type"][1].currentText() == "q8_0"
    assert dialog._tuning_widgets["flash_attn"][1].isChecked() is True
    assert dialog._tuning_widgets["temperature"][1].value() == pytest.approx(0.3)
    for key in ("ctx_size", "kv_cache_type", "flash_attn", "temperature"):
        assert dialog._tuning_widgets[key][0].isChecked() is True

    dialog.preset_quality_btn.click()
    assert dialog._tuning_widgets["ctx_size"][1].value() == 32768
    assert dialog._tuning_widgets["kv_cache_type"][1].currentText() == "f16"

    dialog.preset_low_memory_btn.click()
    assert dialog._tuning_widgets["ctx_size"][1].value() == 4096
    assert dialog._tuning_widgets["kv_cache_type"][1].currentText() == "q4_0"

    dialog.use_global_defaults_btn.click()
    for cb, _widget in dialog._tuning_widgets.values():
        assert cb.isChecked() is False


# =========================================================================================== server
def test_server_apply_emits_applied_with_exactly_the_changed_keys(dialog: ModelsDialog, cfg: Config) -> None:
    seen: list[dict[str, object]] = []
    dialog.applied.connect(seen.append)

    backend_widget = dialog._server_form.widgets["llm.backend"]
    url_widget = dialog._server_form.widgets["llm.server_url"]
    assert isinstance(backend_widget, QComboBox)
    assert isinstance(url_widget, QLineEdit)

    backend_widget.setCurrentText("ollama")
    url_widget.setText("http://127.0.0.1:9999")

    dialog.server_apply_btn.click()

    assert len(seen) == 1
    assert seen[0] == {"llm.backend": "ollama", "llm.server_url": "http://127.0.0.1:9999"}
    assert cfg.llm.backend == "ollama"
    assert cfg.llm.server_url == "http://127.0.0.1:9999"
    assert "Backend" in dialog.server_note.text()
    assert "Server URL" in dialog.server_note.text()


def test_server_apply_with_no_changes_emits_nothing(dialog: ModelsDialog) -> None:
    seen: list[dict[str, object]] = []
    dialog.applied.connect(seen.append)
    dialog.server_apply_btn.click()
    assert seen == []


# ========================================================================================= hardware
def test_hardware_renders_report_numbers(dialog: ModelsDialog, actions: FakeActions) -> None:
    actions.hardware_report_result = {
        "os": "Windows-11", "cpu_name": "Fake CPU", "physical_cores": 8,
        "ram_total_mb": 32768, "vendor": "nvidia",
        "vram_total_mb": 16003, "vram_free_mb": 15000, "budget_mb": 12002,
        "recommended_policy": "auto", "generated_at": "2026-09-27T00:00:00",
    }
    dialog._refresh_hardware()

    texts = [dialog.hw_form.itemAt(i).widget().text() for i in range(dialog.hw_form.count())
             if dialog.hw_form.itemAt(i).widget() is not None]
    joined = " | ".join(texts)
    assert "16003 MiB" in joined
    assert "32768 MiB" in joined


def test_hardware_refresh_calls_hardware_report_with_refresh_true(dialog: ModelsDialog, actions: FakeActions) -> None:
    dialog.hw_refresh_btn.click()
    assert actions.calls_named("hardware_report") == [((), {"refresh": True})]


# ================================================================================ review round two
# 2026-09-27: D1-D4, a second adversarial review of the same day's dialog code.

def test_unrelated_library_event_does_not_wipe_in_progress_tuning_edits(
    dialog: ModelsDialog, actions: FakeActions
) -> None:
    """D1: on_event -> _refresh_library -> _reload_tuning_model_combo used to end with an
    unconditional _load_tuning_profile(...), wiping unsaved ticks/values on ANY ModelLibraryEvent
    -- even one about a different model pull finishing, with the tuned model's own key unchanged."""
    actions.model_library_result = [_row("acme/small"), _row("acme/big")]
    dialog._refresh_library()
    assert dialog.tuning_model_combo.currentData() == "acme/small"

    cb, widget = dialog._tuning_widgets["ctx_size"]
    cb.setChecked(True)
    widget.setValue(55555)

    # same rows, same resolved model -- e.g. an unrelated pull-finished / default-changed event.
    dialog.on_event(ModelLibraryEvent(reason="unrelated"))

    assert dialog.tuning_model_combo.currentData() == "acme/small"
    assert cb.isChecked() is True
    assert widget.value() == 55555


def test_library_event_that_changes_the_resolved_model_still_reloads(
    dialog: ModelsDialog, actions: FakeActions
) -> None:
    """The other half of D1's fix: when the resolved model key genuinely changes (here: the
    previously-selected model is gone from the list), the profile form must still reload."""
    actions.model_library_result = [_row("acme/small")]
    dialog._refresh_library()
    cb, widget = dialog._tuning_widgets["ctx_size"]
    cb.setChecked(True)
    widget.setValue(55555)

    actions.model_library_result = [_row("acme/big")]
    actions.model_profile_result = None
    dialog.on_event(ModelLibraryEvent(reason="acme/small deleted"))

    assert dialog.tuning_model_combo.currentData() == "acme/big"
    assert cb.isChecked() is False
    assert widget.value() != 55555


def test_second_pull_click_disables_pull_and_refusal_event_is_ignored(
    dialog: ModelsDialog, actions: FakeActions
) -> None:
    """D2: pull_btn must disable while a pull is active, and a refusal ModelPullEvent for a
    second click ("another pull is still running") must not touch progress/label/Cancel, nor
    the tracked in-flight repo -- it is ignored entirely."""
    dialog._search_rows = [{"id": "acme/one"}]
    dialog._populate_search_table()
    dialog.search_table.setCurrentCell(0, 0)

    dialog.pull_btn.click()
    assert dialog.pull_btn.isEnabled() is False
    assert dialog._active_pull_repo == "acme/one"

    dialog.on_event(ModelPullEvent(repo_id="acme/one", state="downloading",
                                    file="model.gguf", done_bytes=10, total_bytes=100))
    assert dialog.pull_progress.value() == 10
    assert dialog.cancel_pull_btn.isEnabled() is True

    # a second click's refusal, however the engine tags it -- must change nothing.
    dialog.on_event(ModelPullEvent(repo_id="acme/two", state="error",
                                    detail="another pull is still running"))
    assert dialog.pull_progress.value() == 10
    assert dialog.cancel_pull_btn.isEnabled() is True
    assert dialog.pull_btn.isEnabled() is False
    assert dialog._active_pull_repo == "acme/one"
    assert "another pull" not in dialog.pull_label.text()

    # the tracked pull's own terminal event clears tracking and re-enables Pull.
    actions.model_library_result = [_row("acme/one")]
    dialog.on_event(ModelPullEvent(repo_id="acme/one", state="done", model_key="acme/one"))
    assert dialog.pull_btn.isEnabled() is True
    assert dialog._active_pull_repo is None


def test_extra_args_empty_list_loads_unticked_and_empty_not_the_string_brackets(
    dialog: ModelsDialog, actions: FakeActions
) -> None:
    """D3: ModelProfile.extra_args defaults to [] (unlike every other profile field's None), so
    treating it like the rest ticked the override box and showed the literal text "[]"."""
    actions.model_library_result = [_row("acme/small")]
    actions.model_profile_result = {"ctx_size": 4096, "extra_args": []}
    dialog._refresh_library()

    cb, widget = dialog._tuning_widgets["extra_args"]
    assert cb.isChecked() is False
    assert widget.text() == ""
    assert widget.text() != "[]"


def test_extra_args_round_trips_as_a_list_not_a_string(dialog: ModelsDialog, actions: FakeActions) -> None:
    """D3: a real override loads as a space-joined string and saves back as a list, never str()."""
    actions.model_library_result = [_row("acme/small")]
    actions.model_profile_result = {"extra_args": ["--foo", "--bar=1"]}
    dialog._refresh_library()

    cb, widget = dialog._tuning_widgets["extra_args"]
    assert cb.isChecked() is True
    assert widget.text() == "--foo --bar=1"

    widget.setText("--baz --qux=2")
    dialog.tuning_apply_btn.click()
    assert actions.calls_named("set_model_profile") == [
        (("acme/small", {"extra_args": ["--baz", "--qux=2"]}), {})
    ]


def test_auto_tune_extra_args_fill_path_also_treats_empty_list_as_no_override(
    dialog: ModelsDialog, actions: FakeActions
) -> None:
    """D3: the same trap in the auto-tune fill path (_on_auto_tune_clicked)."""
    actions.model_library_result = [_row("acme/small")]
    dialog._refresh_library()
    actions.auto_tune_result = {"ctx_size": 8192, "extra_args": []}

    dialog.autotune_btn.click()

    cb, widget = dialog._tuning_widgets["extra_args"]
    assert cb.isChecked() is False
    assert widget.text() == ""


def test_busy_set_on_loading_and_only_cleared_on_ready_or_error_never_on_info(dialog: ModelsDialog) -> None:
    """D4(a): an "info" status (the engine's own busy-refusal notice) must never clear busy."""
    dialog.on_event(ComponentStatusEvent(component="llm", state="loading", detail="loading acme/small"))
    assert dialog._llm_busy is True

    dialog.on_event(ComponentStatusEvent(component="llm", state="info", detail="busy: still loading"))
    assert dialog._llm_busy is True

    dialog.on_event(ComponentStatusEvent(component="llm", state="ready", detail="loaded"))
    assert dialog._llm_busy is False


def test_unload_start_sets_busy_and_disables_unload(dialog: ModelsDialog, actions: FakeActions) -> None:
    """D4(b): the engine publishes state "loading" with detail "unloading <key>" at unload start;
    that must set busy exactly like a load does, disabling Unload while it runs."""
    actions.model_library_result = [_row("acme/small", loaded=True)]
    dialog._refresh_library()
    assert dialog.unload_btn.isEnabled() is True

    dialog.on_event(ComponentStatusEvent(component="llm", state="loading", detail="unloading acme/small"))
    assert dialog._llm_busy is True
    assert dialog.unload_btn.isEnabled() is False


def test_model_library_event_own_completion_clears_busy(dialog: ModelsDialog, actions: FakeActions) -> None:
    """D4(c, event half): busy also clears on the operation's own ModelLibraryEvent completion."""
    dialog._llm_busy = True
    actions.model_library_result = [_row("acme/small", loaded=True)]
    dialog.on_event(ModelLibraryEvent(reason="unloaded"))
    assert dialog._llm_busy is False


def test_load_click_guarded_while_busy_bypassing_the_disabled_button(
    dialog: ModelsDialog, actions: FakeActions
) -> None:
    """D4(c): itemDoubleClicked calls _on_load_clicked with no guard of its own, bypassing the
    disabled Load button -- the handler itself must refuse while busy."""
    actions.model_library_result = [_row("acme/small")]
    dialog._refresh_library()
    dialog.library_table.setCurrentCell(0, 0)
    dialog._llm_busy = True

    dialog.library_table.itemDoubleClicked.emit(dialog.library_table.item(0, 0))
    assert actions.calls_named("load_model") == []
    assert "busy" in dialog.library_status.text()


def test_unload_click_guarded_while_busy(dialog: ModelsDialog, actions: FakeActions) -> None:
    actions.model_library_result = [_row("acme/small", loaded=True)]
    dialog._refresh_library()
    dialog._llm_busy = True

    dialog._on_unload_clicked()
    assert actions.calls_named("unload_model") == []


def test_delete_click_guarded_while_loaded_never_even_asks(
    dialog: ModelsDialog, actions: FakeActions, monkeypatch: pytest.MonkeyPatch
) -> None:
    actions.model_library_result = [_row("acme/small", loaded=True)]
    dialog._refresh_library()
    dialog.library_table.setCurrentCell(0, 0)

    def _fail_if_asked(*_a: object, **_kw: object) -> None:
        raise AssertionError("must not confirm a delete of a loaded model")

    monkeypatch.setattr(QMessageBox, "question", staticmethod(_fail_if_asked))
    dialog._on_delete_clicked()
    assert actions.calls_named("delete_model") == []


# ============================================================================================ None
def test_on_event_never_raises_with_a_none_returning_actions(dialog: ModelsDialog, actions: FakeActions) -> None:
    # every canned result stays None (the default) -- model_library/hardware_report/model_profile/
    # auto_tune_model all answer None, as a real backend does while idle/unreachable.
    dialog.on_event(ModelLibraryEvent(reason="x"))
    dialog.on_event(ModelPullEvent(repo_id="x/y", state="queued"))
    dialog.on_event(ModelSearchEvent(request_id=dialog._search_request_id, query="x"))
    dialog.on_event(ModelInfoEvent(request_id=dialog._inspect_request_id, repo_id="x/y"))
    dialog.on_event(ComponentStatusEvent(component="llm", state="info", detail=""))
    dialog.on_event(object())  # not one of the known event types either

    dialog._refresh_library()
    dialog._refresh_hardware()
    dialog.autotune_btn.click()
    dialog.tuning_apply_btn.click()

"""Offscreen tests for Settings > Optimize (2026-10-07): the dialog hooks in ``exocortex.ui.settings``
and the page built by ``exocortex.ui.optimize_panel.OptimizePanel``.

A recording :class:`FakeOptimizeActions` DEFINES the six ``optimize_*`` methods (plus ``list_models``)
as real methods -- the dialog only accepts an object whose CLASS defines ``optimize_snapshot`` -- and
returns canned snapshots. The plan dicts inside them are genuine ``Plan.to_dict()`` output of
``exocortex.optimize.recommend`` for a synthetic copy of the owner's machine (RTX 5090 Laptop 24463 MiB,
Qwen3.8-27B hybrid header, Whisper on the CPU today), so the panel is exercised on the shapes the real
engine will send. Nothing here touches hardware, the network or the developer's ``config.toml``
(``Config()`` objects with no path; the overlay test's ``apply_settings`` is the catch-all fake's no-op).
"""
from __future__ import annotations

import copy
import inspect
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QComboBox, QSpinBox

from exocortex import optimize as opt
from exocortex.bus import EventBus
from exocortex.config import Config
from exocortex.ui import optimize_panel as panel_mod
from exocortex.ui.optimize_panel import DEGRADED_TEXT, GLYPHS, OptimizePanel
from exocortex.ui.overlay import OverlayWindow
from exocortex.ui.settings import FIELDS, FieldForm, SettingsDialog, _resolve_actions, all_fields, sections_covered
from exocortex.ui.theme import COLORS
from tests.test_overlay_completion import FakeActions as CatchAllActions
from tests.test_settings import qapp  # noqa: F401 -- reused session fixture

KEY = "lmstudio-community/Qwen3.8-27B"
OTHER_KEY = "deepgrove/maple-preview"
GOAL_ORDER = ("latency", "balanced", "context", "quality")


# ---------------------------------------------------------------------------------------------
# fixtures: genuine plans for a synthetic copy of the owner's machine
# ---------------------------------------------------------------------------------------------
def _plan_inputs(profile: dict | None = None) -> opt.PlanInputs:
    machine = opt.Machine(
        cpu="Intel(R) Core(TM) Ultra 9 285H", physical_cores=16, ram_total_mb=64957, ram_free_mb=41000, mem_mts=7467,
        gpu=opt.GpuFacts(name="NVIDIA GeForce RTX 5090 Laptop GPU", total_mb=24463, free_mb=22400, baseline_mb=2281,
                         power_capped=True, power_w=115.0, power_limit_w=130.0, sm_mhz=700, sm_max_mhz=3090),
        igpu="Intel(R) Arc(TM) 140T GPU", npu="Intel(R) AI Boost", devices_known=True, ct2_cuda=True,
        sherpa_ready=True, runtime="nvidia-cuda12 2.44.0", guard_floor_mb=1500)
    model = opt.ModelFacts(
        key=KEY, size_bytes=16_810_000_000, arch="qwen35", n_layer=65, n_ctx_train=262144, n_head_kv=4, head_dim=256,
        kv_heads_total=260, full_attention_interval=4, nextn_layers=1, ssm_conv=4, ssm_state=128, ssm_inner=6144,
        ssm_groups=16)
    launch = opt.LaunchConfig(ctx=81920, kv="q8_0", parallel=2, ubatch=512, threads=14, flash=True, gpu_layers=999999)
    return opt.PlanInputs(settings=opt.Settings(), machine=machine, model=model, launch=launch, running=True,
                          backend="llamacpp", profile=profile, cfg=opt.CfgFacts(asr_now="whisper-cpu-int8", ocr_now="igpu"),
                          evidence=opt.Evidence(), now_epoch=1000.0)


def _plans() -> dict[str, dict[str, Any]]:
    """{goal: Plan.to_dict()} for an UNPINNED profile, so the language-model row is ticked."""
    return {g: p.to_dict() for g, p in opt.plan_all(_plan_inputs(profile=None)).items()}


def _snapshot(plans: dict[str, Any] | None = None, **over: Any) -> dict[str, Any]:
    snap: dict[str, Any] = {
        "engine": {"ready": True, "backend": "llamacpp", "model": KEY, "swap_resident": False},
        "facts": {
            "cpu": "Intel(R) Core(TM) Ultra 9 285H", "cores": 16, "ram_total_mb": 64957, "ram_free_mb": 41000,
            "mem_mts": 7467, "gpu": "NVIDIA GeForce RTX 5090 Laptop GPU",
            "vram": {"total_mb": 24463, "free_mb": 22400, "others_mb": 2281},
            "igpu": "Intel(R) Arc(TM) 140T GPU", "npu": "Intel(R) AI Boost", "runtime": "nvidia-cuda12 2.44.0",
            "power": {"power_capped": True, "power_w": 115.0, "power_limit_w": 130.0, "sm_mhz": 700, "sm_max_mhz": 3090},
            "guard_floor_mb": 1500,
        },
        "models": [{"key": KEY, "size_gb": 16.8, "ctx_train": 262144}],
        "plans": _plans() if plans is None else plans,
        "selected": "balanced",
        "gates": {"llm": [True, ""], "speech": [True, ""], "apply": [True, ""]},
        "analysis": {"state": "done", "message": "", "age_s": 1.0},
        "run": {"state": "idle", "run_id": 0, "step": "", "fraction": 0.0, "message": ""},
        "measured": {"llm": None, "asr": [], "ocr": [], "at": "", "stale": False},
        "apply": {"state": "idle", "message": "", "digest": "", "undo": False},
    }
    snap.update(over)
    return snap


class FakeOptimizeActions:
    """Records every call as ``(name, args, kwargs)`` and answers with ``self.snapshot`` (a copy, like
    the real engine). The six optimize methods and ``list_models`` are REAL methods on purpose."""

    def __init__(self, snapshot: Any = None) -> None:
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        self.snapshot: Any = _snapshot() if snapshot is None else snapshot
        self.snapshot_error: Exception | None = None
        self.benchmark_result: Any = True
        self.apply_result: Any = True
        self.undo_result: Any = True
        self.fail_with: Exception | None = None       # raised by every other optimize_* method

    def _rec(self, name: str, args: tuple[Any, ...] = (), kwargs: dict[str, Any] | None = None) -> None:
        self.calls.append((name, args, dict(kwargs or {})))

    def named(self, name: str) -> list[tuple[tuple[Any, ...], dict[str, Any]]]:
        return [(a, kw) for (n, a, kw) in self.calls if n == name]

    def optimize_snapshot(self) -> Any:
        self._rec("optimize_snapshot")
        if self.snapshot_error is not None:
            raise self.snapshot_error
        return copy.deepcopy(self.snapshot)

    def optimize_analyze(self, goal: str = "", options: dict[str, Any] | None = None, model_key: str = "") -> None:
        self._rec("optimize_analyze", (goal, options, model_key))
        if self.fail_with is not None:
            raise self.fail_with

    def optimize_benchmark(self, scopes: list[str] | None = None, force: bool = False) -> Any:
        self._rec("optimize_benchmark", (scopes,), {"force": force})
        if self.fail_with is not None:
            raise self.fail_with
        return self.benchmark_result

    def optimize_cancel(self) -> None:
        self._rec("optimize_cancel")
        if self.fail_with is not None:
            raise self.fail_with

    def optimize_apply(self, digest: str) -> Any:
        self._rec("optimize_apply", (digest,))
        if self.fail_with is not None:
            raise self.fail_with
        return self.apply_result

    def optimize_undo(self) -> Any:
        self._rec("optimize_undo")
        if self.fail_with is not None:
            raise self.fail_with
        return self.undo_result

    def list_models(self) -> list[str]:
        self._rec("list_models")
        return [KEY]


@pytest.fixture()
def make_dialog(qapp: QApplication):
    """``make(actions=None, cfg=None, show=True)`` -> ``(dialog, panel_or_None, applied_list)``. The
    dialog is built the way ``OverlayWindow.open_settings`` builds it (``actions.list_models`` as the
    second argument), the Optimize tab is selected and the dialog shown, so the panel's first
    ``showEvent`` runs."""
    made: list[SettingsDialog] = []

    def make(actions: Any = None, cfg: Config | None = None, *, show: bool = True) -> tuple[SettingsDialog, Any, list[dict]]:
        cfg = cfg if cfg is not None else Config()
        dlg = SettingsDialog(cfg, actions.list_models if actions is not None else None)
        made.append(dlg)
        applied: list[dict] = []
        dlg.applied.connect(applied.append)
        dlg.tabs.setCurrentIndex(dlg.tabs.count() - 1)
        if show:
            dlg.show()
            QTest.qWait(30)
        return dlg, dlg._optimize_panel, applied

    yield make
    for d in made:
        d.close()
        d.deleteLater()
    qapp.processEvents()


def _row(panel: OptimizePanel, label_start: str) -> int:
    for r in range(panel.plan_table.rowCount()):
        if panel.plan_table.item(r, 1).text().startswith(label_start):
            return r
    raise AssertionError(f"no plan row starting with {label_start!r}")


def _tick(panel: OptimizePanel, label_start: str, on: bool) -> None:
    panel.plan_table.item(_row(panel, label_start), 0).setCheckState(Qt.CheckState.Checked if on else Qt.CheckState.Unchecked)


def _item(plan: dict[str, Any], item_id: str) -> dict[str, Any]:
    return next(i for i in plan["items"] if i["id"] == item_id)


# ---------------------------------------------------------------------------------------------
# the tab and its fields
# ---------------------------------------------------------------------------------------------
def test_optimize_is_the_last_tab_and_completion_stays_first(make_dialog: Callable) -> None:
    dlg, panel, _ = make_dialog(None, show=False)
    assert list(FIELDS)[0] == "Completion" and list(FIELDS)[-1] == "Optimize"
    assert dlg.tabs.count() == len(FIELDS)
    assert dlg.tabs.tabText(dlg.tabs.count() - 1) == "Optimize"
    assert isinstance(panel, OptimizePanel) and dlg.tabs.widget(dlg.tabs.count() - 1) is panel
    assert "llm" not in sections_covered() and {"optimize", "asr"} <= sections_covered()
    assert (dlg.width(), dlg.height()) == (680, 640) and (dlg.minimumWidth(), dlg.minimumHeight()) == (520, 420)


def test_optimize_fields_match_the_plan() -> None:
    fields = {f.path: f for f in FIELDS["Optimize"]}
    assert list(fields) == ["optimize.goal", "optimize.reserve_vram_mb", "optimize.ctx_cap", "optimize.asr_gpu",
                            "asr.engine", "asr.device", "asr.compute_type", "asr.cpu_threads"]
    assert fields["optimize.goal"].choices == GOAL_ORDER
    assert (fields["optimize.reserve_vram_mb"].lo, fields["optimize.reserve_vram_mb"].hi, fields["optimize.reserve_vram_mb"].step) == (0, 12000, 256)
    assert (fields["optimize.ctx_cap"].lo, fields["optimize.ctx_cap"].hi, fields["optimize.ctx_cap"].step) == (0, 262144, 2048)
    assert fields["optimize.asr_gpu"].choices == ("auto", "never")
    assert fields["asr.engine"].choices == ("auto", "faster-whisper", "sherpa")
    assert fields["asr.device"].choices == ("auto", "cuda", "cpu")
    assert fields["asr.compute_type"].choices == ("auto", "float16", "int8_float16", "int8")
    assert (fields["asr.cpu_threads"].lo, fields["asr.cpu_threads"].hi) == (0, 32)
    assert [p for p, f in fields.items() if f.restart] == ["asr.engine", "asr.device", "asr.compute_type", "asr.cpu_threads"]
    assert not any(p.startswith("optimize.") and f.restart for p, f in fields.items())
    # the measured numbers the owner asked to see are in the engine's help
    helptext = fields["asr.engine"].help
    assert "6.2 s" in helptext and "0.15 s" in helptext and "0.11 s" in helptext and "0.9 GB" in helptext
    # each path is unique across the whole table (asr.* appears nowhere else)
    seen = [f.path for f in all_fields()]
    assert len(seen) == len(set(seen))


# ---------------------------------------------------------------------------------------------
# the degraded page (no engine)
# ---------------------------------------------------------------------------------------------
def test_degraded_page_disables_the_actions_but_keeps_the_fields_editable(make_dialog: Callable) -> None:
    dlg, panel, _ = make_dialog(None)
    assert dlg._actions is None
    assert panel.why_label.text() == DEGRADED_TEXT == "Open Settings from the running app to benchmark and optimise."
    for btn in (panel.bench_btn, panel.cancel_btn, panel.apply_btn, panel.undo_btn, panel.analyze_btn):
        assert not btn.isEnabled()
    assert panel.plan_table.isHidden() and panel.compare_table.isHidden()
    assert panel.goal_combo.isEnabled() and dlg._widgets["optimize.goal"] is panel.goal_combo
    panel.goal_combo.setCurrentText("latency")
    dlg._widgets["optimize.ctx_cap"].setValue(32768)
    assert dlg.changes() == {"optimize.goal": "latency", "optimize.ctx_cap": 32768}
    assert panel.goal_text.text() == panel_mod.GOAL_TEXT["latency"]
    assert panel._poll_timer.isActive() is False


@pytest.mark.parametrize("answer", [None, {}, "error"])
def test_none_empty_and_raising_snapshots_read_as_not_connected(make_dialog: Callable, answer: Any) -> None:
    fake = FakeOptimizeActions()
    if answer == "error":
        fake.snapshot_error = RuntimeError("controller exploded")
    else:
        fake.snapshot = answer
    dlg, panel, _ = make_dialog(fake)
    assert dlg._actions is fake                              # the class defines optimize_snapshot
    assert panel.why_label.text() == DEGRADED_TEXT
    assert not panel.bench_btn.isEnabled() and not panel.apply_btn.isEnabled()
    assert fake.named("optimize_analyze") == []              # nothing to analyse against a dead controller
    # the engine comes back: Settings is reopened (load()) and the page fills in
    fake.snapshot_error = None
    fake.snapshot = _snapshot()
    dlg.load()
    assert panel.plan_table.rowCount() > 0 and panel.bench_btn.isEnabled()


def test_a_panel_that_cannot_be_built_leaves_a_plain_working_tab(qapp: QApplication, monkeypatch: pytest.MonkeyPatch) -> None:
    class Boom:
        def __init__(self, *a: Any, **k: Any) -> None:
            raise RuntimeError("panel defect")

    monkeypatch.setattr(panel_mod, "OptimizePanel", Boom)
    dlg = SettingsDialog(Config(), None)
    try:
        assert dlg._optimize_panel is None and dlg.tabs.count() == len(FIELDS)
        assert dlg.tabs.tabText(dlg.tabs.count() - 1) == "Optimize"
        dlg._widgets["asr.cpu_threads"].setValue(6)
        assert dlg.changes() == {"asr.cpu_threads": 6}
        dlg.load()                                           # load() copes with no panel
    finally:
        dlg.close()
        qapp.processEvents()


# ---------------------------------------------------------------------------------------------
# how the dialog finds the engine
# ---------------------------------------------------------------------------------------------
def test_resolve_actions_order_and_rejection_of_catch_all_fakes() -> None:
    explicit, bound, from_parent = FakeOptimizeActions(), FakeOptimizeActions(), FakeOptimizeActions()
    parent = SimpleNamespace(_actions=from_parent)
    assert _resolve_actions(explicit, bound.list_models, parent) is explicit
    assert _resolve_actions(None, bound.list_models, parent) is bound            # list_models.__self__
    assert _resolve_actions(None, None, parent) is from_parent                   # parent._actions
    assert _resolve_actions(None, lambda: [], SimpleNamespace()) is None
    assert _resolve_actions(None, None, None) is None

    catch = CatchAllActions()
    assert hasattr(catch, "optimize_snapshot")                                   # the trap: hasattr is useless
    assert _resolve_actions(catch, catch.list_models, SimpleNamespace(_actions=catch)) is None
    assert _resolve_actions(catch, bound.list_models, None) is bound             # a rejected one falls through
    assert _resolve_actions(object(), None, None) is None
    instance_only = SimpleNamespace(optimize_snapshot=lambda: {})                # not on the CLASS
    assert _resolve_actions(instance_only, None, None) is None


def test_constructor_is_positionally_compatible_and_actions_is_keyword_only(qapp: QApplication) -> None:
    fake = FakeOptimizeActions()
    made = [SettingsDialog(Config(), None), SettingsDialog(Config(), fake.list_models, None),
            SettingsDialog(Config(), None, None, actions=fake)]
    try:
        assert [d._actions for d in made] == [None, fake, fake]
        with pytest.raises(TypeError):
            SettingsDialog(Config(), None, None, fake)       # type: ignore[misc]  # 4th positional is not allowed
    finally:
        for d in made:
            d.close()
        qapp.processEvents()


# ---------------------------------------------------------------------------------------------
# dialog hooks: apply() without arguments, apply_paths, set_values, widget_values
# ---------------------------------------------------------------------------------------------
def test_apply_button_calls_apply_without_any_argument(qapp: QApplication) -> None:
    """PySide would feed clicked(bool) into an optional parameter: apply() has none, and the click
    reaches the shared body as ``only=None``."""
    assert list(inspect.signature(SettingsDialog.apply).parameters) == ["self"]
    seen: list[Any] = []

    class Recording(SettingsDialog):
        def _apply(self, only: Any) -> dict[str, Any]:
            seen.append(only)
            return super()._apply(only)

    cfg = Config()
    dlg = Recording(cfg, None)
    try:
        dlg._widgets["completion.max_lines"].setValue(cfg.completion.max_lines + 1)
        dlg.apply_btn.click()
        assert seen == [None]
        assert cfg.completion.max_lines == Config().completion.max_lines + 1
        dlg.ok_btn.click()
        assert seen == [None, None]
    finally:
        dlg.close()
        qapp.processEvents()


def test_apply_paths_filters_and_leaves_other_edits_pending(qapp: QApplication) -> None:
    cfg = Config()
    dlg = SettingsDialog(cfg, None)
    got: list[dict[str, Any]] = []
    dlg.applied.connect(got.append)
    try:
        dlg._widgets["completion.max_lines"].setValue(cfg.completion.max_lines + 2)
        dlg._widgets["optimize.ctx_cap"].setValue(32768)
        dlg._widgets["asr.engine"].setCurrentText("sherpa")
        changed = dlg.apply_paths({"optimize.ctx_cap", "asr.engine", "ui.opacity"})   # ui.opacity: unchanged, ignored
        assert changed == {"optimize.ctx_cap": 32768, "asr.engine": "sherpa"}
        assert got == [changed]
        assert cfg.optimize.ctx_cap == 32768 and cfg.asr.engine == "sherpa"
        assert cfg.completion.max_lines == Config().completion.max_lines            # still only a pending edit
        assert dlg.changes() == {"completion.max_lines": Config().completion.max_lines + 2}
        assert "next launch" in dlg.note.text() and "Speech engine" in dlg.note.text()
        assert dlg.apply_paths([]) == {} and dlg.apply_paths({"nope.nothing"}) == {} and got == [changed]
    finally:
        dlg.close()
        qapp.processEvents()


def test_fieldform_apply_only_is_optional_and_filters(qapp: QApplication) -> None:
    cfg = Config()
    form = FieldForm(FIELDS["Optimize"], cfg, None)
    try:
        form.load()
        form.widgets["optimize.ctx_cap"].setValue(4096)
        form.widgets["asr.cpu_threads"].setValue(3)
        assert form.apply(only={"asr.cpu_threads"}) == {"asr.cpu_threads": 3}
        assert cfg.optimize.ctx_cap == 0 and cfg.asr.cpu_threads == 3
        assert form.apply() == {"optimize.ctx_cap": 4096}                   # default: everything
    finally:
        form.widget.deleteLater()
        qapp.processEvents()


def test_set_values_reports_unknown_paths_and_widget_values_reads_back(qapp: QApplication) -> None:
    dlg = SettingsDialog(Config(), None)
    try:
        failed = dlg.set_values({"asr.engine": "sherpa", "screen.max_text_chars": 1500, "no.such": 1, "ui.opacity": "abc"})
        assert sorted(failed) == ["no.such", "ui.opacity"]
        assert dlg.widget_values(["asr.engine", "screen.max_text_chars", "no.such"]) == {
            "asr.engine": "sherpa", "screen.max_text_chars": 1500}
    finally:
        dlg.close()
        qapp.processEvents()


def test_restore_defaults_covers_optimize_and_asr(make_dialog: Callable) -> None:
    cfg = Config()
    cfg.optimize.goal, cfg.optimize.reserve_vram_mb, cfg.optimize.asr_gpu = "context", 4096, "never"
    cfg.asr.engine, cfg.asr.device, cfg.asr.compute_type, cfg.asr.cpu_threads = "sherpa", "cpu", "int8", 8
    dlg, panel, _ = make_dialog(None, cfg, show=False)
    assert dlg.changes() == {}
    assert dlg._widgets["optimize.goal"].currentText() == "context" and dlg._widgets["asr.cpu_threads"].value() == 8
    dlg.restore_defaults()
    d = Config()
    assert dlg._widgets["optimize.goal"].currentText() == d.optimize.goal
    assert dlg._widgets["optimize.reserve_vram_mb"].value() == d.optimize.reserve_vram_mb
    assert dlg._widgets["optimize.ctx_cap"].value() == d.optimize.ctx_cap
    assert dlg._widgets["optimize.asr_gpu"].currentText() == d.optimize.asr_gpu
    assert dlg._widgets["asr.engine"].currentText() == d.asr.engine
    assert dlg._widgets["asr.device"].currentText() == d.asr.device
    assert dlg._widgets["asr.compute_type"].currentText() == d.asr.compute_type
    assert dlg._widgets["asr.cpu_threads"].value() == d.asr.cpu_threads
    assert set(dlg.changes()) == {"optimize.goal", "optimize.reserve_vram_mb", "optimize.asr_gpu",
                                  "asr.engine", "asr.device", "asr.compute_type", "asr.cpu_threads"}
    assert panel.goal_combo.currentText() == "balanced"


def test_form_widgets_live_in_the_panel_and_the_limits_box_collapses(make_dialog: Callable) -> None:
    dlg, panel, _ = make_dialog(None, show=False)
    assert panel.isAncestorOf(panel.goal_combo) and not panel._limits_body.isAncestorOf(panel.goal_combo)
    for path in ("optimize.reserve_vram_mb", "optimize.ctx_cap", "optimize.asr_gpu", "asr.engine", "asr.device",
                 "asr.compute_type", "asr.cpu_threads"):
        assert panel._limits_body.isAncestorOf(dlg._widgets[path]), path
    assert panel._limits_body.isHidden() and not panel.limits_box.isChecked()      # collapsed by default
    panel.limits_box.setChecked(True)
    assert not panel._limits_body.isHidden()
    panel.limits_box.setChecked(False)
    assert panel._limits_body.isHidden()
    assert isinstance(dlg._widgets["optimize.reserve_vram_mb"], QSpinBox)
    assert isinstance(dlg._widgets["asr.engine"], QComboBox)


# ---------------------------------------------------------------------------------------------
# with an engine: first show, rendering, selection, compare strip
# ---------------------------------------------------------------------------------------------
def test_first_show_analyses_once_with_the_unsaved_options(make_dialog: Callable) -> None:
    fake = FakeOptimizeActions()
    cfg = Config()
    cfg.optimize.goal = "context"
    dlg, panel, _ = make_dialog(fake, cfg, show=False)
    assert fake.named("optimize_analyze") == []                       # nothing before the page is shown
    dlg._widgets["optimize.reserve_vram_mb"].setValue(2048)           # an UNSAVED edit
    dlg.show()
    QTest.qWait(30)
    assert fake.named("optimize_analyze") == [(("context", {"reserve_vram_mb": 2048, "ctx_cap": 0, "asr_gpu": "auto"}, ""), {})]
    dlg.hide()
    dlg.show()
    panel.hide()
    panel.show()
    QTest.qWait(30)
    assert len(fake.named("optimize_analyze")) == 1                   # once, not on every show


def test_plan_table_renders_rows_with_glyphs_and_default_ticks(make_dialog: Callable) -> None:
    plans = _plans()
    bal = plans["balanced"]
    _item(bal, "speech")["evidence_kind"] = "recorded"
    _item(bal, "llm")["evidence_kind"] = "measured"
    fake = FakeOptimizeActions(_snapshot(plans))
    dlg, panel, _ = make_dialog(fake)
    t = panel.plan_table
    assert [t.horizontalHeaderItem(c).text() for c in range(t.columnCount())] == ["", "Job", "Now", "Recommended", "Takes effect", "Evidence"]
    assert t.rowCount() == len(bal["items"])
    llm, speech, ocr = _row(panel, "Answers"), _row(panel, "Speech"), _row(panel, "Accurate OCR")
    assert (GLYPHS["measured"], GLYPHS["recorded"], GLYPHS["estimated"]) == ("●", "○", "◌")
    assert t.item(llm, 5).text().startswith("●")               # measured
    assert t.item(speech, 5).text().startswith("○")             # recorded
    assert t.item(ocr, 5).text().startswith("◌")                # estimated
    assert t.item(speech, 2).text() == "whisper-cpu-int8" and t.item(speech, 3).text() == "sherpa-cpu"
    assert t.item(speech, 4).text() == "at the next launch" and t.item(llm, 4).text() == "after the model reloads (~15 s)"
    assert t.item(llm, 3).text() == _item(bal, "llm")["proposed"]
    # checkboxes: changed + enabled rows follow their default tick, unchanged rows have none
    assert t.item(llm, 0).checkState() == Qt.CheckState.Checked and t.item(speech, 0).checkState() == Qt.CheckState.Checked
    assert t.item(ocr, 0).data(Qt.ItemDataRole.CheckStateRole) is None
    assert panel.warnings_label.text().count("!") >= 1                # the planner's "cannot keep up" warning
    assert panel.goal_combo.currentText() == "balanced"


def test_selecting_a_row_fills_the_why_text(make_dialog: Callable) -> None:
    fake = FakeOptimizeActions()
    dlg, panel, _ = make_dialog(fake)
    plan = fake.snapshot["plans"]["balanced"]
    assert panel.why_label.text() == "\n".join(plan["summary"])       # nothing selected: the plan's summary
    speech = _row(panel, "Speech")
    panel.plan_table.selectRow(speech)
    text = panel.why_label.text()
    assert _item(plan, "speech")["why"] in text and text.startswith("Speech recognition:")
    assert "Evidence: 0.15 s/utterance" in text
    panel.plan_table.selectRow(_row(panel, "NPU"))
    assert _item(plan, "npu")["why"] in panel.why_label.text()


def test_compare_table_has_all_four_goals_and_a_click_selects_one(make_dialog: Callable) -> None:
    fake = FakeOptimizeActions()
    dlg, panel, _ = make_dialog(fake)
    c = panel.compare_table
    assert c.rowCount() == 4 and c.columnCount() == 6
    assert [c.horizontalHeaderItem(i).text() for i in range(6)] == ["Goal", "Window", "KV", "Speech", "Free VRAM", "Cold re-read"]
    assert [c.item(r, 0).text() for r in range(4)] == list(GOAL_ORDER)
    plans = fake.snapshot["plans"]
    for r, goal in enumerate(GOAL_ORDER):
        assert c.item(r, 1).text() == f"{plans[goal]['predicted']['ctx']:,}"
        assert c.item(r, 3).text() == "Parakeet CPU"
        assert c.item(r, 4).text().endswith(" GB") and c.item(r, 5).text().endswith(" s")
    assert c.item(GOAL_ORDER.index("quality"), 2).text() == "f16" and c.item(0, 2).text() == "q8_0"
    assert c.item(0, 5).text().startswith("~")                        # typical prompt speed, not measured
    # the quality row's window really is smaller than context's
    assert plans["quality"]["predicted"]["ctx"] < plans["context"]["predicted"]["ctx"]
    before = len(fake.calls)
    c.cellClicked.emit(GOAL_ORDER.index("context"), 0)
    assert panel.goal_combo.currentText() == "context" and panel._row_digest == plans["context"]["digest"]
    assert c.currentRow() == GOAL_ORDER.index("context")
    assert panel.plan_table.item(_row(panel, "Screen text"), 3).text() == _item(plans["context"], "screen")["proposed"]
    assert [n for (n, _a, _k) in fake.calls[before:] if n != "optimize_snapshot"] == []   # no engine work


def test_changing_the_goal_only_re_renders_from_the_snapshot(make_dialog: Callable) -> None:
    fake = FakeOptimizeActions()
    dlg, panel, _ = make_dialog(fake)
    for _ in range(panel_mod._GRACE_POLLS + 1):                       # let the first-show grace polling run out
        panel._poll()
    analyses = len(fake.named("optimize_analyze"))
    before = len(fake.calls)
    panel.goal_combo.setCurrentText("latency")
    assert panel._row_digest == fake.snapshot["plans"]["latency"]["digest"]
    assert panel.plan_table.item(_row(panel, "Answers"), 3).text() == _item(fake.snapshot["plans"]["latency"], "llm")["proposed"]
    assert panel.goal_text.text() == panel_mod.GOAL_TEXT["latency"]
    QTest.qWait(panel_mod.REANALYZE_MS + 150)
    assert len(fake.named("optimize_analyze")) == analyses
    assert [n for (n, _a, _k) in fake.calls[before:]] == []           # not even a snapshot read
    assert dlg.changes() == {"optimize.goal": "latency"}              # the goal is an ordinary pending setting


def test_apply_button_label_follows_the_ticks(make_dialog: Callable) -> None:
    dlg, panel, _ = make_dialog(FakeOptimizeActions())
    assert panel.apply_btn.text() == "Apply 2 changes (reloads the model, answers pause about 15 s)"
    _tick(panel, "Answers", False)
    assert panel.apply_btn.text() == "Apply 1 change" and panel.apply_btn.isEnabled()
    _tick(panel, "Speech", False)
    assert not panel.apply_btn.isEnabled() and "nothing selected" in panel.apply_btn.text()
    _tick(panel, "Answers", True)
    assert panel.apply_btn.text() == "Apply 1 change (reloads the model, answers pause about 15 s)"
    panel.goal_combo.setCurrentText("context")                        # context: speech + llm + two live rows
    assert panel.apply_btn.text().startswith("Apply 4 changes") and "reloads the model" in panel.apply_btn.text()


def test_machine_label_vram_bars_and_devices(make_dialog: Callable) -> None:
    fake = FakeOptimizeActions()
    dlg, panel, _ = make_dialog(fake)
    m = panel.machine_label.text()
    for needle in ("Intel(R) Core(TM) Ultra 9 285H, 16 cores", "63.4 GB RAM (7467 MT/s), 40.0 GB free",
                   "NVIDIA GeForce RTX 5090 Laptop GPU", "24463 MiB, 22400 free, 2281 held by others",
                   "iGPU Intel(R) Arc(TM) 140T GPU", "NPU Intel(R) AI Boost", "runtime nvidia-cuda12 2.44.0",
                   "power-capped: 115 W of 130 W, 700 MHz of 3090"):
        assert needle in m, needle
    # no power line when the GPU is not capped
    fake.snapshot["facts"]["power"] = {"power_capped": False}
    panel.refresh()
    assert "power-capped" not in panel.machine_label.text()

    plain = panel.vram_plain.splitlines()
    assert plain[0].startswith("now   ") and plain[1].startswith("plan  ")
    assert len(plain[0]) - 6 == len(plain[1]) - 6 == panel_mod._BAR_WIDTH
    assert "weights" in plain[3] and "KV" in plain[3] and "speech 0" in plain[3] and "buffers" in plain[3]
    assert COLORS["err"] not in panel.vram_label.text()                # 22400 MiB free is above the floor
    # free memory below the guard floor is drawn in the error colour
    fake.snapshot["facts"]["vram"]["free_mb"] = 900
    panel.refresh()
    assert COLORS["err"] in panel.vram_label.text()
    dev = panel.devices_label.text()
    assert "RTX 5090" in dev and "weights" in dev and "Intel(R) Arc(TM) 140T GPU: accurate OCR (igpu)" in dev
    assert "Intel(R) AI Boost: idle" in dev


def test_results_label_summarises_the_measurements(make_dialog: Callable) -> None:
    fake = FakeOptimizeActions()
    dlg, panel, _ = make_dialog(fake)
    assert "typical numbers" in panel.results_label.text()
    fake.snapshot["measured"] = {
        "llm": {"pp": 960.0, "tg": 23.8, "cached_ttft_ms": 210.0, "spread": 0.04, "rounds": 2},
        "asr": [{"label": "whisper-cpu-int8", "median_s": 6.2}, {"label": "sherpa-cpu", "median_s": 0.15}],
        "ocr": [], "at": "2026-10-07T21:14:09", "stale": False}
    panel.refresh()
    assert panel.results_label.text() == (
        "Answers: 960 tok/s read, 23.8 tok/s written, 210 ms to first token cached (2 rounds, spread 4%)"
        " · Speech: Whisper CPU 6.2 s, Parakeet CPU 0.15 s · measured 2026-10-07 21:14")


# ---------------------------------------------------------------------------------------------
# re-analysis, polling, run / cancel
# ---------------------------------------------------------------------------------------------
def test_limit_edits_re_analyse_after_a_debounce_with_unsaved_values(make_dialog: Callable) -> None:
    fake = FakeOptimizeActions()
    dlg, panel, _ = make_dialog(fake)
    assert len(fake.named("optimize_analyze")) == 1
    for value in (1024, 2048, 3072):                                  # three quick edits -> one analysis
        dlg._widgets["optimize.reserve_vram_mb"].setValue(value)
    assert panel._reanalyze_timer.isActive()
    QTest.qWait(panel_mod.REANALYZE_MS + 200)
    calls = fake.named("optimize_analyze")
    assert len(calls) == 2
    assert calls[1][0] == ("balanced", {"reserve_vram_mb": 3072, "ctx_cap": 0, "asr_gpu": "auto"}, "")
    dlg._widgets["optimize.ctx_cap"].setValue(32768)
    dlg._widgets["optimize.asr_gpu"].setCurrentText("never")
    QTest.qWait(panel_mod.REANALYZE_MS + 200)
    assert fake.named("optimize_analyze")[2][0][1] == {"reserve_vram_mb": 3072, "ctx_cap": 32768, "asr_gpu": "never"}
    # setting the same values again (what load() does) costs no round trip
    dlg._widgets["optimize.ctx_cap"].setValue(0)
    dlg._widgets["optimize.ctx_cap"].setValue(32768)
    QTest.qWait(panel_mod.REANALYZE_MS + 200)
    assert len(fake.named("optimize_analyze")) == 3
    assert fake.named("optimize_benchmark") == [] and fake.named("optimize_apply") == []


def test_choosing_another_model_re_analyses_for_it(make_dialog: Callable) -> None:
    snap = _snapshot(models=[{"key": KEY, "size_gb": 16.8, "ctx_train": 262144}, {"key": OTHER_KEY, "size_gb": 5.0, "ctx_train": 8192}])
    fake = FakeOptimizeActions(snap)
    dlg, panel, _ = make_dialog(fake)
    assert not panel.model_combo.isHidden() and panel.model_combo.count() == 2
    assert panel.model_combo.currentData() == KEY                     # defaults to the loaded model
    assert fake.named("optimize_analyze")[0][0][2] == ""
    panel.model_combo.setCurrentIndex(1)
    QTest.qWait(panel_mod.REANALYZE_MS + 200)
    assert fake.named("optimize_analyze")[1][0] == ("balanced", {"reserve_vram_mb": 0, "ctx_cap": 0, "asr_gpu": "auto"}, OTHER_KEY)
    # a single model: no picker
    fake.snapshot = _snapshot()
    panel.refresh()
    assert panel.model_combo.isHidden() and panel.model_label.isHidden()


@pytest.mark.parametrize("block", ["run", "analysis", "apply"])
def test_polling_runs_only_while_something_is_running_and_stops_at_a_terminal_state(make_dialog: Callable, block: str) -> None:
    fake = FakeOptimizeActions()
    dlg, panel, _ = make_dialog(fake)
    for _ in range(panel_mod._GRACE_POLLS + 1):                       # the first-show grace runs out
        panel._poll()
    assert not panel._poll_timer.isActive()
    fake.snapshot[block] = {**fake.snapshot[block], "state": "running", "run_id": 1, "fraction": 0.4, "step": "llm"}
    panel.refresh()
    assert panel._poll_timer.isActive() and panel._poll_timer.interval() == 500
    panel._poll()
    assert panel._poll_timer.isActive()                               # still running
    fake.snapshot[block] = {**fake.snapshot[block], "state": "done" if block != "apply" else "verified"}
    panel._poll()
    assert not panel._poll_timer.isActive()
    dlg.hide()
    assert not panel._poll_timer.isActive()


def test_run_benchmark_sends_the_scope_and_follows_progress(make_dialog: Callable) -> None:
    fake = FakeOptimizeActions()
    dlg, panel, _ = make_dialog(fake)
    assert panel.bench_btn.isEnabled() and not panel.cancel_btn.isEnabled()
    assert panel.bench_bar.isHidden()
    panel.bench_btn.click()
    assert fake.named("optimize_benchmark") == [((["llm", "speech"],), {"force": False})]   # a list, never the clicked flag
    panel.scope_combo.setCurrentText("Speech only")
    panel.bench_btn.click()
    panel.scope_combo.setCurrentText("Language model only")
    panel.bench_btn.click()
    assert [a[0] for a, _k in fake.named("optimize_benchmark")] == [["llm", "speech"], ["speech"], ["llm"]]

    fake.snapshot["run"] = {"state": "running", "run_id": 4, "step": "llm", "fraction": 0.5, "message": "round 2 of 2"}
    panel.refresh()
    assert panel.cancel_btn.isEnabled() and not panel.bench_btn.isEnabled() and not panel.apply_btn.isEnabled()
    assert not panel.bench_bar.isHidden() and panel.bench_bar.value() == 50
    assert panel.bench_status.text() == "Measuring the language model: round 2 of 2"
    panel.cancel_btn.click()
    assert len(fake.named("optimize_cancel")) == 1
    fake.snapshot["run"] = {"state": "cancelled", "run_id": 4, "step": "llm", "fraction": 0.5, "message": ""}
    panel.refresh()
    assert panel.bench_status.text() == "Cancelled." and panel.bench_bar.isHidden() and panel.bench_btn.isEnabled()


def test_refused_benchmark_shows_the_engines_reason(make_dialog: Callable) -> None:
    fake = FakeOptimizeActions()
    fake.benchmark_result = False
    fake.snapshot["run"]["message"] = "an answer is being generated"
    dlg, panel, _ = make_dialog(fake)
    panel.bench_btn.click()
    assert panel.bench_status.text() == "an answer is being generated"
    fake.snapshot["run"]["message"] = ""
    panel.bench_btn.click()
    assert panel.bench_status.text() == "refused"
    fake.benchmark_result = None                                       # None reads as a refusal too
    panel.bench_btn.click()
    assert panel.bench_status.text() == "refused"
    fake.fail_with = RuntimeError("boom")                              # a raising call never escapes
    panel.bench_btn.click()
    panel.cancel_btn.setEnabled(True)
    panel.cancel_btn.click()
    panel.analyze_btn.click()
    assert panel.bench_status.text() == "refused"


def test_a_closed_gate_disables_run_and_says_why(make_dialog: Callable) -> None:
    fake = FakeOptimizeActions(_snapshot(gates={"llm": [False, "someone is speaking; wait for a quiet moment"],
                                                "speech": [True, ""], "apply": [True, ""]}))
    dlg, panel, _ = make_dialog(fake)
    assert not panel.bench_btn.isEnabled()
    assert panel.bench_status.text() == "Not now: someone is speaking; wait for a quiet moment"
    panel.scope_combo.setCurrentText("Speech only")                    # the speech gate is open
    assert panel.bench_btn.isEnabled() and panel.bench_status.text() == ""


def test_results_from_an_older_run_are_ignored(make_dialog: Callable) -> None:
    fake = FakeOptimizeActions()
    fake.snapshot["run"] = {"state": "done", "run_id": 5, "step": "speech", "fraction": 1.0, "message": "finished run five"}
    dlg, panel, _ = make_dialog(fake)
    assert panel.bench_status.text() == "finished run five"
    fake.snapshot["run"] = {"state": "running", "run_id": 3, "step": "llm", "fraction": 0.2, "message": "stale"}
    panel.refresh()
    assert panel.bench_status.text() == "finished run five" and panel.bench_bar.isHidden()
    fake.snapshot["run"] = {"state": "running", "run_id": 6, "step": "llm", "fraction": 0.2, "message": "new run"}
    panel.refresh()
    assert panel.bench_status.text() == "Measuring the language model: new run"


# ---------------------------------------------------------------------------------------------
# apply / undo
# ---------------------------------------------------------------------------------------------
def test_apply_settings_rows_write_exactly_their_paths_and_leave_other_edits_pending(make_dialog: Callable) -> None:
    fake = FakeOptimizeActions()
    cfg = Config()
    dlg, panel, applied = make_dialog(fake, cfg)
    dlg._widgets["completion.max_lines"].setValue(cfg.completion.max_lines + 4)     # an unrelated pending edit
    _tick(panel, "Answers", False)                                                   # settings only
    analyses = len(fake.named("optimize_analyze"))
    panel.apply_btn.click()
    assert applied == [{"asr.engine": "sherpa", "asr.cpu_threads": 4}]
    assert dlg._widgets["asr.engine"].currentText() == "sherpa" and dlg._widgets["asr.cpu_threads"].value() == 4
    assert cfg.asr.engine == "sherpa" and cfg.asr.cpu_threads == 4
    assert cfg.completion.max_lines == Config().completion.max_lines                 # not written
    assert dlg.changes() == {"completion.max_lines": Config().completion.max_lines + 4}
    assert "next launch" in dlg.note.text() and "Speech engine" in dlg.note.text() and "Speech CPU threads" in dlg.note.text()
    assert "Applied 2 settings." in panel.apply_result.text() and "Takes effect at the next launch" in panel.apply_result.text()
    assert fake.named("optimize_apply") == []                                        # no model change was ticked
    assert len(fake.named("optimize_analyze")) == analyses + 1                       # the config moved: planned again
    assert panel.undo_btn.isEnabled()


def test_apply_model_row_calls_the_engine_once_with_the_plan_digest(make_dialog: Callable) -> None:
    fake = FakeOptimizeActions()
    dlg, panel, applied = make_dialog(fake)
    digest = fake.snapshot["plans"]["balanced"]["digest"]
    fake.snapshot["apply"] = {"state": "running", "message": "relaunching", "digest": digest, "undo": False}
    panel.apply_btn.click()
    assert fake.named("optimize_apply") == [((digest,), {})]
    assert applied == [{"asr.engine": "sherpa", "asr.cpu_threads": 4}]               # the settings row went through the dialog
    assert panel._pending is not None and not panel.apply_btn.isEnabled() and not panel.undo_btn.isEnabled()
    assert panel._poll_timer.isActive()
    assert panel.apply_result.text().startswith("Reloading the model...")
    # the engine verifies, then reports a terminal state
    fake.snapshot["apply"] = {"state": "verified", "message": "context window confirmed", "digest": digest, "undo": True}
    panel._poll()
    assert panel._pending is None and "applied and verified" in panel.apply_result.text()
    assert panel.undo_btn.isEnabled()
    for _ in range(panel_mod._GRACE_POLLS + 1):
        panel._poll()
    assert not panel._poll_timer.isActive()


def test_unticked_rows_are_skipped(make_dialog: Callable) -> None:
    fake = FakeOptimizeActions()
    cfg = Config()
    dlg, panel, applied = make_dialog(fake, cfg)
    _tick(panel, "Speech", False)                                                    # model row only
    panel.apply_btn.click()
    assert applied == [] and cfg.asr.engine == "auto" and dlg._widgets["asr.engine"].currentText() == "auto"
    assert len(fake.named("optimize_apply")) == 1
    # nothing ticked: the button is off and a forced click does nothing
    fake2 = FakeOptimizeActions()
    dlg2, panel2, applied2 = make_dialog(fake2)
    _tick(panel2, "Speech", False)
    _tick(panel2, "Answers", False)
    assert not panel2.apply_btn.isEnabled()
    panel2._on_apply()
    assert fake2.named("optimize_apply") == [] and applied2 == []


def test_ticks_survive_a_poll_but_not_a_new_plan(make_dialog: Callable) -> None:
    fake = FakeOptimizeActions()
    dlg, panel, _ = make_dialog(fake)
    _tick(panel, "Speech", False)
    panel.refresh()
    panel._poll()
    assert panel.plan_table.item(_row(panel, "Speech"), 0).checkState() == Qt.CheckState.Unchecked
    # a new analysis changes the digest: the defaults come back
    fake.snapshot["plans"]["balanced"]["digest"] = "ffffffffffff"
    panel.refresh()
    assert panel.plan_table.item(_row(panel, "Speech"), 0).checkState() == Qt.CheckState.Checked


@pytest.mark.parametrize("answer", [False, None, "raise"])
def test_refused_apply_shows_the_engines_reason(make_dialog: Callable, answer: Any) -> None:
    fake = FakeOptimizeActions()
    fake.snapshot["apply"] = {"state": "refused", "message": "an answer is being generated", "digest": "", "undo": False}
    if answer == "raise":
        fake.fail_with = RuntimeError("controller down")
    else:
        fake.apply_result = answer
    dlg, panel, _ = make_dialog(fake)
    _tick(panel, "Speech", False)
    panel.apply_btn.click()
    assert panel._pending is None
    assert "The model change was refused: an answer is being generated" in panel.apply_result.text()
    assert panel.apply_btn.isEnabled()                                               # nothing pending: can try again


def test_a_rollback_is_reported_as_an_error_and_clears_undo(make_dialog: Callable) -> None:
    fake = FakeOptimizeActions()
    dlg, panel, _ = make_dialog(fake)
    digest = fake.snapshot["plans"]["balanced"]["digest"]
    fake.snapshot["apply"] = {"state": "running", "message": "", "digest": digest, "undo": False}
    _tick(panel, "Speech", False)
    panel.apply_btn.click()
    fake.snapshot["apply"] = {"state": "rolled-back", "message": "VRAM below the guard floor", "digest": digest, "undo": False}
    panel._poll()
    assert panel._pending is None and not panel._model_verified
    assert panel.apply_result.text() == "Rolled back to the previous setup: VRAM below the guard floor"
    assert COLORS["err"] in panel.apply_result.styleSheet()
    assert not panel.undo_btn.isEnabled()


def test_a_stale_apply_state_for_another_digest_is_not_mistaken_for_ours(make_dialog: Callable) -> None:
    fake = FakeOptimizeActions()
    fake.snapshot["apply"] = {"state": "verified", "message": "old", "digest": "0123456789ab", "undo": True}
    dlg, panel, _ = make_dialog(fake)
    _tick(panel, "Speech", False)
    panel.apply_btn.click()
    assert panel._pending is not None                                                # the old "verified" was not ours
    digest = fake.snapshot["plans"]["balanced"]["digest"]
    fake.snapshot["apply"] = {"state": "verified", "message": "ours", "digest": digest, "undo": True}
    panel._poll()
    assert panel._pending is None and "ours" in panel.apply_result.text()


def test_undo_restores_the_widgets_and_the_saved_values(make_dialog: Callable) -> None:
    fake = FakeOptimizeActions()
    cfg = Config()
    dlg, panel, applied = make_dialog(fake, cfg)
    _tick(panel, "Answers", False)
    assert not panel.undo_btn.isEnabled()
    panel.apply_btn.click()
    assert cfg.asr.engine == "sherpa" and panel.undo_btn.isEnabled()
    panel.undo_btn.click()
    assert dlg._widgets["asr.engine"].currentText() == "auto" and dlg._widgets["asr.cpu_threads"].value() == 0
    assert cfg.asr.engine == "auto" and cfg.asr.cpu_threads == 0
    assert applied[-1] == {"asr.engine": "auto", "asr.cpu_threads": 0}
    assert fake.named("optimize_undo") == []                                         # no model change had been verified
    assert "Restored the previous settings." in panel.apply_result.text()
    assert not panel.undo_btn.isEnabled()


def test_undo_also_asks_the_engine_when_a_model_change_was_verified(make_dialog: Callable) -> None:
    fake = FakeOptimizeActions()
    dlg, panel, _ = make_dialog(fake)
    digest = fake.snapshot["plans"]["balanced"]["digest"]
    fake.snapshot["apply"] = {"state": "running", "message": "", "digest": digest, "undo": False}
    panel.apply_btn.click()
    fake.snapshot["apply"] = {"state": "verified", "message": "ok", "digest": digest, "undo": True}
    panel._poll()
    assert panel.undo_btn.isEnabled()
    fake.snapshot["apply"] = {"state": "running", "message": "restoring", "digest": digest, "undo": True}
    panel.undo_btn.click()
    assert len(fake.named("optimize_undo")) == 1
    assert dlg._widgets["asr.engine"].currentText() == "auto"                         # the settings part came back too
    assert panel._pending == {"kind": "undo", "digest": ""} and not panel.undo_btn.isEnabled()
    fake.snapshot["apply"] = {"state": "verified", "message": "previous profile back", "digest": digest, "undo": False}
    panel._poll()
    assert panel._pending is None and "Previous model setup restored" in panel.apply_result.text()


def test_a_pending_apply_gives_up_after_the_timeout(make_dialog: Callable, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeOptimizeActions()
    dlg, panel, _ = make_dialog(fake)
    fake.snapshot["apply"] = {"state": "running", "message": "", "digest": fake.snapshot["plans"]["balanced"]["digest"], "undo": False}
    _tick(panel, "Speech", False)
    panel.apply_btn.click()
    assert panel._pending is not None
    panel._pending_since -= panel_mod._PENDING_TIMEOUT_S + 1
    fake.snapshot["apply"] = {"state": "idle", "message": "", "digest": "", "undo": False}
    panel._poll()
    assert panel._pending is None and "did not report a result in time" in panel.apply_result.text()


# ---------------------------------------------------------------------------------------------
# through a real OverlayWindow
# ---------------------------------------------------------------------------------------------
class OverlayFake(CatchAllActions):
    """The overlay tests' catch-all fake, plus real optimize methods (so the overlay finds it through
    ``parent._actions``: its ``list_models`` is a plain function with no ``__self__``)."""

    def __init__(self, snapshot: dict[str, Any]) -> None:
        super().__init__()
        self.snapshot = snapshot

    def optimize_snapshot(self) -> dict[str, Any]:
        return copy.deepcopy(self.snapshot)

    def optimize_analyze(self, goal: str = "", options: dict[str, Any] | None = None, model_key: str = "") -> None:
        self.calls.append(("optimize_analyze", (goal, options, model_key)))


@pytest.fixture()
def overlay_factory(qapp: QApplication):
    made: list[OverlayWindow] = []

    def make(actions: Any) -> OverlayWindow:
        win = OverlayWindow(Config(), EventBus(), actions)
        made.append(win)
        win.show()
        QTest.qWait(30)
        return win

    yield make
    for w in made:
        if w._settings is not None:
            w._settings.close()
        w.close_silently()
    qapp.processEvents()


def test_overlay_with_the_catch_all_fake_opens_a_degraded_optimize_tab(overlay_factory: Callable) -> None:
    actions = CatchAllActions()
    win = overlay_factory(actions)
    win.open_settings()                                                               # must not raise
    dlg = win._settings
    assert dlg is not None and dlg._actions is None
    panel = dlg._optimize_panel
    assert dlg.tabs.count() == len(FIELDS) and panel.why_label.text() == DEGRADED_TEXT
    assert not panel.bench_btn.isEnabled()
    assert not [c for c in actions.calls if c[0].startswith("optimize_")]             # nothing was asked of the fake
    win.open_settings()                                                               # reused: load() -> refresh()
    assert win._settings is dlg


def test_overlay_with_an_optimize_capable_engine_finds_it_through_the_parent(overlay_factory: Callable) -> None:
    actions = OverlayFake(_snapshot())
    win = overlay_factory(actions)
    win.open_settings()
    dlg = win._settings
    assert dlg is not None and dlg._actions is actions
    dlg.tabs.setCurrentIndex(dlg.tabs.count() - 1)
    QTest.qWait(40)
    panel = dlg._optimize_panel
    assert panel.plan_table.rowCount() > 0 and panel.bench_btn.isEnabled()
    assert [c for c in actions.calls if c[0] == "optimize_analyze"][0][1][0] == "balanced"
    # a settings row applied through the dialog reaches Engine.apply_settings with exactly its paths
    _tick(panel, "Answers", False)
    panel.apply_btn.click()
    QTest.qWait(30)
    assert ("apply_settings", ({"asr.engine": "sherpa", "asr.cpu_threads": 4},)) in actions.calls

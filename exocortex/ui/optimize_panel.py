"""The Settings > Optimize page (2026-10-07): benchmark this machine, read the plan, apply it, undo it.

Why a panel of its own. The tab is more than a form: it measures (a benchmark on the engine), shows
a plan per goal (``latency`` / ``balanced`` / ``context`` / ``quality``) with the evidence behind
every row, and applies the ticked rows -- some live, some at the next launch, one by reloading the
language model. All the decisions are made elsewhere (:mod:`exocortex.optimize` is the pure planner,
:mod:`exocortex.optimize_run` drives it on the engine); this file only draws what
``actions.optimize_snapshot()`` returns and forwards button presses. Three rules shaped it:

* **The form stays the source of truth.** ``FIELDS["Optimize"]`` is an ordinary tab: the dialog builds
  its :class:`~exocortex.ui.settings.FieldForm` as usual and hands it here; the panel only moves that
  form's widgets into its own layout (as the very last step of construction, so a failure earlier
  leaves the plain form usable as the fallback page). ``load`` / ``changes`` / ``apply`` /
  ``restore_defaults`` therefore keep working untouched, and OK / Apply on the dialog persist
  ``optimize.goal`` and the limits like any other value.
* **Poll, never push.** The overlay forwards no new event kinds to a Settings dialog (and may not be
  edited for this), so the panel reads a cached, copy-only ``optimize_snapshot()``. A 500 ms timer
  runs only while an analysis, benchmark run or apply is ``running`` (plus a short grace right after
  a button press, because the worker thread flips the state a moment after the call returns) and
  stops by itself. No worker thread ever calls into a widget, so closing the dialog mid-run cannot
  touch a dead object.
* **Degrade, never raise.** With no engine (tests, a headless dialog, the overlay tests' catch-all
  fakes) or an engine that answers ``None`` / ``{}`` / an exception, the page says "Open Settings
  from the running app to benchmark and optimise", disables Run / Analyze / Apply / Undo and leaves
  the goal and limit fields editable. Every ``actions.*`` and ``host.*`` call is wrapped.

Applying splits by where the value lives. Settings rows (speech, OCR device, prompt budgets) go
through the dialog's own widgets -- ``host.set_values`` then ``host.apply_paths`` -- so they travel the
usual ``applied`` -> ``Engine.apply_settings`` -> ``save_config`` path, other tabs' pending edits stay
pending, and the dialog's note lists the "(restart)" fields. The model row goes to
``actions.optimize_apply(digest)``: the ENGINE applies the plan it computed itself, verifies and rolls
back on its own; the panel then follows ``snapshot["apply"]`` until it is terminal. Undo restores the
widget values it saved before applying and asks the engine to undo its last verified change.

Qt trap noted once: ``clicked(bool)`` must never be connected straight to a method with an optional
argument (PySide would feed the ``checked`` flag into it), so every button goes through
``lambda _=False: ...``.
"""
from __future__ import annotations

import copy
import html
import logging
import re
import time
from typing import Any

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QFormLayout,
    QWidget,
)

from exocortex.config import Config
from exocortex.ui.settings import Field, get_field_value
from exocortex.ui.theme import COLORS

log = logging.getLogger(__name__)

#: Shown in the plan area when there is no engine to ask (or it answered nothing).
DEGRADED_TEXT = "Open Settings from the running app to benchmark and optimise."

GOALS: tuple[str, ...] = ("latency", "balanced", "context", "quality")

#: One plain sentence per goal, under the goal picker.
GOAL_TEXT: dict[str, str] = {
    "latency": "Fastest answers: a modest window and about 3 GB of GPU memory kept free for the desktop.",
    "balanced": "Today's behaviour plus the safe fixes: a window that leaves room for the display and the shared model server.",
    "context": "The longest conversation window this GPU can hold: every spare GB becomes context.",
    "quality": "A 16-bit KV cache where at least 32k tokens still fit; otherwise the same as balanced.",
}

#: Evidence glyphs: measured on this machine, recorded earlier / imported, or estimated / typical.
GLYPHS: dict[str, str] = {"measured": "●", "recorded": "○", "estimated": "◌", "typical": "◌"}
_GLYPH_COLORS: dict[str, str] = {"measured": COLORS["ok"], "recorded": COLORS["accent"],
                                 "estimated": COLORS["text_dim"], "typical": COLORS["text_dim"]}

#: Benchmark scopes behind ``scope_combo`` (label -> scopes handed to ``optimize_benchmark``).
SCOPES: dict[str, list[str]] = {
    "Language model and speech": ["llm", "speech"],
    "Language model only": ["llm"],
    "Speech only": ["speech"],
}

APPLIES_TEXT: dict[str, str] = {
    "model-restart": "after the model reloads (~15 s)",
    "live": "right away",
    "next-launch": "at the next launch",
    "none": "",
}

STEP_TEXT: dict[str, str] = {
    "facts": "Reading the machine",
    "llm": "Measuring the language model",
    "speech": "Measuring speech recognition",
    "apply": "Applying",
    "verify": "Verifying",
}

PLAN_COLUMNS: tuple[str, ...] = ("", "Job", "Now", "Recommended", "Takes effect", "Evidence")
COMPARE_COLUMNS: tuple[str, ...] = ("Goal", "Window", "KV", "Speech", "Free VRAM", "Cold re-read")

POLL_MS = 500            # snapshot poll while something is running
REANALYZE_MS = 400       # debounce for limit / model edits
_GRACE_POLLS = 3         # polls kept after a button press even if nothing reads "running" yet
_PENDING_TIMEOUT_S = 180.0
_APPLY_TERMINAL = ("verified", "rolled-back", "refused", "error")
_BAR_WIDTH = 56
_MONO = "font-family:Consolas,'Cascadia Mono',monospace"

_PANEL_QSS = f"""
QLabel#section {{ color: {COLORS['accent']}; font-weight: 600; padding-top: 4px; }}
QLabel#hint {{ color: {COLORS['text_faint']}; }}
QTableWidget {{ background: {COLORS['panel']}; border: 1px solid {COLORS['border']}; border-radius: 4px;
                gridline-color: {COLORS['border']}; }}
QHeaderView::section {{ background: {COLORS['bg_solid']}; color: {COLORS['text_dim']};
                        border: none; padding: 4px 6px; }}
QGroupBox {{ border: 1px solid {COLORS['border']}; border-radius: 6px; margin-top: 10px; padding-top: 8px; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px; color: {COLORS['text_dim']}; }}
"""


# ---------------------------------------------------------------------------------------------
# tolerant readers: the snapshot is plain data from another thread and another author, so every
# access goes through these instead of trusting its shape
# ---------------------------------------------------------------------------------------------
def _d(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _l(value: object) -> list[Any]:
    return list(value) if isinstance(value, (list, tuple)) else []


def _num(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _int(value: object, default: int = 0) -> int:
    n = _num(value)
    return int(n) if n is not None else default


def _esc(text: object) -> str:
    return html.escape(str(text), quote=False)


def _fit_table(table: QTableWidget) -> None:
    """Size ``table`` to all its rows (the page scrolls, the tables do not)."""
    table.resizeRowsToContents()
    rows = sum(table.rowHeight(r) for r in range(table.rowCount()))
    table.setFixedHeight(table.horizontalHeader().sizeHint().height() + rows + 2 * table.frameWidth() + 4)


def _bar_cells(values: list[float], total: float, width: int) -> list[int]:
    """Character counts for a text bar: proportional, at least one for any non-zero value, summing
    to ``width`` (the largest segment absorbs the rounding)."""
    if total <= 0 or not values:
        return [0] * len(values)
    cells = [int(round(v / total * width)) if v > 0 else 0 for v in values]
    cells = [1 if (v > 0 and c == 0) else c for v, c in zip(values, cells)]
    big = max(range(len(cells)), key=lambda i: cells[i])
    cells[big] = max(0, cells[big] + (width - sum(cells)))
    return cells


def _fmt_s(seconds: float) -> str:
    """6.2 / 0.15 / 12.5 (no trailing zeros)."""
    return f"{seconds:.2f}".rstrip("0").rstrip(".")


def _asr_name(label: str) -> str:
    """A friendly name for an ASR variant label (``whisper-cpu-int8`` -> ``Whisper CPU``)."""
    low = label.lower()
    if low.startswith("sherpa"):
        return "Parakeet CPU"
    if low.startswith("whisper"):
        if "cuda" in low:
            precision = low.rsplit("-", 1)[-1]
            return f"Whisper GPU {precision}" if precision not in ("cuda", "whisper") else "Whisper GPU"
        return "Whisper CPU"
    return label


def _item_foreground(item: QTableWidgetItem, color: str) -> None:
    item.setForeground(QColor(color))


class OptimizePanel(QWidget):
    """The Optimize tab's page. ``form`` is the dialog's :class:`~exocortex.ui.settings.FieldForm`
    for ``FIELDS["Optimize"]`` (its widgets are moved in here); ``host`` is the dialog
    (``set_values`` / ``widget_values`` / ``apply_paths`` / ``note``); ``actions`` is the engine
    or ``None``. See the module docstring for the contract."""

    def __init__(self, cfg: Config, actions: Any, form: Any, host: Any, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._cfg = cfg
        self._actions = actions
        self._form = form
        self._host = host
        self._fields: dict[str, Field] = {f.path: f for f in form.fields}

        # ---- state (all replaced, never shared with the engine) ----
        self._snap: dict[str, Any] = {}
        self._connected = False
        self._last_plans: dict[str, Any] | None = None
        self._last_goal = ""
        self._row_items: list[dict[str, Any]] = []
        self._row_digest = ""
        self._user_ticks: dict[tuple[str, str], bool] = {}
        self._selected_id = ""
        self._run_id = 0
        self._refusal = ""
        self._analyzed_once = False
        self._last_analysis_args: tuple[Any, ...] | None = None
        self._grace = 0
        self._undo_values: dict[str, Any] = {}
        self._pending: dict[str, str] | None = None
        self._pending_since = 0.0
        self._model_verified = False
        self._models_sig: tuple[str, ...] = ()
        self._model_user = False
        self._populating = False
        self._rendering = False
        self._ever_shown = False
        self.vram_plain = ""

        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(POLL_MS)
        self._poll_timer.timeout.connect(self._poll)
        self._reanalyze_timer = QTimer(self)
        self._reanalyze_timer.setSingleShot(True)
        self._reanalyze_timer.setInterval(REANALYZE_MS)
        self._reanalyze_timer.timeout.connect(self._flush_reanalysis)

        self.setStyleSheet(_PANEL_QSS)
        self._build()
        self._place_form_widgets()          # last structural step: see the module docstring
        self._wire()
        self.refresh()

    # =========================================================================================
    # construction
    # =========================================================================================
    @staticmethod
    def _label(text: str = "", *, name: str = "", wrap: bool = True, parent: QWidget | None = None) -> QLabel:
        lab = QLabel(text, parent)
        lab.setWordWrap(wrap)
        lab.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        if name:
            lab.setObjectName(name)
        return lab

    def _table(self, columns: tuple[str, ...]) -> QTableWidget:
        t = QTableWidget(0, len(columns), self)
        t.setHorizontalHeaderLabels(list(columns))
        t.verticalHeader().setVisible(False)
        t.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        t.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        t.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        t.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        t.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        t.setWordWrap(True)
        t.setTextElideMode(Qt.TextElideMode.ElideNone)
        t.horizontalHeader().setStretchLastSection(False)
        return t

    def _build(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self.scroll = QScrollArea(self)
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        outer.addWidget(self.scroll)
        body = QWidget(self.scroll)
        lay = QVBoxLayout(body)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(6)
        self.scroll.setWidget(body)

        # 1. goal ------------------------------------------------------------------------------
        self._goal_row = QHBoxLayout()
        self._goal_row.addWidget(QLabel("Optimise for", body))      # goal_combo is inserted at index 1
        self._goal_row.addStretch(1)
        lay.addLayout(self._goal_row)
        self.goal_text = self._label("", name="hint", parent=body)
        lay.addWidget(self.goal_text)

        # 2. machine ---------------------------------------------------------------------------
        lay.addWidget(self._label("This machine", name="section", wrap=False, parent=body))
        self.machine_label = self._label("", parent=body)
        lay.addWidget(self.machine_label)
        model_row = QHBoxLayout()
        self.model_label = QLabel("Model", body)
        self.model_combo = QComboBox(body)
        model_row.addWidget(self.model_label)
        model_row.addWidget(self.model_combo, 1)
        lay.addLayout(model_row)

        # 3. measure ---------------------------------------------------------------------------
        lay.addWidget(self._label("Measure", name="section", wrap=False, parent=body))
        run_row = QHBoxLayout()
        self.scope_combo = QComboBox(body)
        for text in SCOPES:
            self.scope_combo.addItem(text)
        self.bench_btn = QPushButton("Run benchmark", body)
        self.bench_btn.setObjectName("primary")
        self.cancel_btn = QPushButton("Cancel", body)
        run_row.addWidget(self.scope_combo)
        run_row.addWidget(self.bench_btn)
        run_row.addWidget(self.cancel_btn)
        run_row.addStretch(1)
        lay.addLayout(run_row)
        self.bench_hint = self._label(
            "About a minute and a half, run when nobody is speaking. It measures the running model and speech "
            "engine; it does not start a second server.", name="hint", parent=body)
        lay.addWidget(self.bench_hint)
        self.bench_bar = QProgressBar(body)
        self.bench_bar.setRange(0, 100)
        self.bench_bar.setTextVisible(False)
        self.bench_bar.setVisible(False)
        lay.addWidget(self.bench_bar)
        self.bench_status = self._label("", parent=body)
        lay.addWidget(self.bench_status)
        self.results_label = self._label("", parent=body)
        lay.addWidget(self.results_label)

        # 4. compare ---------------------------------------------------------------------------
        self.compare_title = self._label("Compare the goals", name="section", wrap=False, parent=body)
        lay.addWidget(self.compare_title)
        self.compare_table = self._table(COMPARE_COLUMNS)
        ch = self.compare_table.horizontalHeader()
        for col in range(len(COMPARE_COLUMNS)):
            ch.setSectionResizeMode(col, QHeaderView.ResizeMode.Stretch)
        lay.addWidget(self.compare_table)

        # 5. where each job runs ----------------------------------------------------------------
        self.plan_title = self._label("Where each job runs", name="section", wrap=False, parent=body)
        lay.addWidget(self.plan_title)
        self.plan_table = self._table(PLAN_COLUMNS)
        ph = self.plan_table.horizontalHeader()
        modes = (QHeaderView.ResizeMode.Fixed, QHeaderView.ResizeMode.Interactive, QHeaderView.ResizeMode.Stretch,
                 QHeaderView.ResizeMode.Stretch, QHeaderView.ResizeMode.Interactive, QHeaderView.ResizeMode.Stretch)
        for col, mode in enumerate(modes):
            ph.setSectionResizeMode(col, mode)
        self.plan_table.setColumnWidth(0, 30)
        self.plan_table.setColumnWidth(1, 130)
        self.plan_table.setColumnWidth(4, 110)
        lay.addWidget(self.plan_table)
        self.why_label = self._label("", parent=body)
        lay.addWidget(self.why_label)
        self.warnings_label = self._label("", parent=body)
        lay.addWidget(self.warnings_label)

        # 6. devices and GPU memory ---------------------------------------------------------------
        self.devices_title = self._label("Devices and GPU memory", name="section", wrap=False, parent=body)
        lay.addWidget(self.devices_title)
        self.devices_label = self._label("", parent=body)
        lay.addWidget(self.devices_label)
        self.vram_label = self._label("", parent=body)
        self.vram_label.setTextFormat(Qt.TextFormat.RichText)
        lay.addWidget(self.vram_label)

        # 7. apply ----------------------------------------------------------------------------
        self.apply_btn = QPushButton("Apply", body)
        self.apply_btn.setObjectName("primary")
        lay.addWidget(self.apply_btn)               # its label can be long ("... reloads the model ..."): own row
        undo_row = QHBoxLayout()
        self.undo_btn = QPushButton("Undo", body)
        self.analyze_btn = QPushButton("Analyze again", body)
        undo_row.addWidget(self.undo_btn)
        undo_row.addWidget(self.analyze_btn)
        undo_row.addStretch(1)
        lay.addLayout(undo_row)
        self.apply_result = self._label("", parent=body)
        lay.addWidget(self.apply_result)

        # 8. limits and speech engine (collapsible) -------------------------------------------------
        self.limits_box = QGroupBox("Limits and speech engine", body)
        self.limits_box.setCheckable(True)
        box_lay = QVBoxLayout(self.limits_box)
        box_lay.setContentsMargins(8, 4, 8, 8)
        self._limits_body = QWidget(self.limits_box)
        self._limits_layout = QFormLayout(self._limits_body)
        self._limits_layout.setContentsMargins(0, 0, 0, 0)
        self._limits_layout.setSpacing(6)
        self._limits_layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        box_lay.addWidget(self._limits_body)
        lay.addWidget(self.limits_box)
        lay.addStretch(1)

    def _place_form_widgets(self) -> None:
        """Move the form's widgets into this page: the goal picker next to its label, everything
        else into the collapsible box. Done last so a failure while building leaves ``form`` whole."""
        widgets = self._form.widgets
        self.goal_combo: QComboBox = widgets["optimize.goal"]
        self._goal_row.insertWidget(1, self.goal_combo)
        for f in self._form.fields:
            if f.path == "optimize.goal":
                continue
            w = widgets[f.path]
            label = QLabel(f.label + (" (restart)" if f.restart else ""), self._limits_body)
            label.setWordWrap(True)
            if f.help:
                label.setToolTip(f.help)
            self._limits_layout.addRow(label, w)
            if f.help:
                hint = self._label(f.help, name="help", parent=self._limits_body)
                self._limits_layout.addRow(hint)
        self.limits_box.setChecked(False)               # collapsed until asked for
        self._limits_body.setVisible(False)
        self.limits_box.toggled.connect(self._limits_body.setVisible)

    def _wire(self) -> None:
        # PySide trap: every clicked(bool) goes through a lambda with a defaulted parameter
        self.bench_btn.clicked.connect(lambda _=False: self._on_run())
        self.cancel_btn.clicked.connect(lambda _=False: self._on_cancel())
        self.apply_btn.clicked.connect(lambda _=False: self._on_apply())
        self.undo_btn.clicked.connect(lambda _=False: self._on_undo())
        self.analyze_btn.clicked.connect(lambda _=False: self._on_analyze())
        self.scope_combo.currentIndexChanged.connect(lambda _=0: self._on_scope_changed())
        self.goal_combo.currentTextChanged.connect(lambda _t="": self._on_goal_changed())
        self.model_combo.currentIndexChanged.connect(lambda _=0: self._on_model_changed())
        for path in ("optimize.reserve_vram_mb", "optimize.ctx_cap", "optimize.asr_gpu"):
            w = self._form.widgets.get(path)
            if isinstance(w, QSpinBox):
                w.valueChanged.connect(lambda _v=0: self._schedule_reanalysis())
            elif isinstance(w, QComboBox):
                w.currentTextChanged.connect(lambda _t="": self._schedule_reanalysis())
        self.plan_table.itemChanged.connect(self._on_item_changed)
        self.plan_table.itemSelectionChanged.connect(self._on_row_selected)
        self.compare_table.cellClicked.connect(lambda r=0, _c=0: self._on_compare_clicked(r))
        self.compare_table.currentCellChanged.connect(lambda r=-1, _c=0, _pr=-1, _pc=0: self._on_compare_clicked(r))

    # =========================================================================================
    # engine access (every call wrapped; a failure reads as "no answer")
    # =========================================================================================
    def _call(self, name: str, *args: Any, **kwargs: Any) -> tuple[bool, Any]:
        if self._actions is None:
            return False, None
        try:
            return True, getattr(self._actions, name)(*args, **kwargs)
        except Exception:  # noqa: BLE001 - an engine fault must never take the Settings dialog down
            log.exception("actions.%s() failed", name)
            return False, None

    def _host_call(self, name: str, *args: Any) -> tuple[bool, Any]:
        try:
            return True, getattr(self._host, name)(*args)
        except Exception:  # noqa: BLE001 - same rule for the dialog hooks
            log.exception("host.%s() failed", name)
            return False, None

    def _read_snapshot(self) -> dict[str, Any] | None:
        ok, snap = self._call("optimize_snapshot")
        if not ok or not isinstance(snap, dict) or not snap:
            return None                         # None / {} / raising = "engine not connected"
        return snap

    # =========================================================================================
    # public: refresh / polling
    # =========================================================================================
    def refresh(self) -> None:
        """Re-read the engine's snapshot and redraw. Never starts any work."""
        self._take_snapshot(self._read_snapshot())

    def _take_snapshot(self, snap: dict[str, Any] | None) -> None:
        self._connected = snap is not None
        self._snap = snap or {}
        self._render()
        self._update_polling()

    def _any_running(self) -> bool:
        s = self._snap
        return any(_d(s.get(k)).get("state") == "running" for k in ("analysis", "run", "apply"))

    def _update_polling(self) -> None:
        """Poll while something runs (or just got started, or an apply/undo is awaited); stop otherwise."""
        want = self._connected and (self._any_running() or self._grace > 0 or self._pending is not None)
        if want and self._ever_shown and not self.isVisible():
            want = False                        # the dialog was hidden: showEvent refreshes and resumes
        if want:
            if not self._poll_timer.isActive():
                self._poll_timer.start()
        elif self._poll_timer.isActive():
            self._poll_timer.stop()

    def _poll(self) -> None:
        if self._grace > 0:
            self._grace -= 1
        if self._pending is not None and time.monotonic() - self._pending_since > _PENDING_TIMEOUT_S:
            self._pending = None
            self._set_result("The engine did not report a result in time; open Models to check the model.", "error")
        self._take_snapshot(self._read_snapshot())

    # =========================================================================================
    # Qt events
    # =========================================================================================
    def showEvent(self, event: Any) -> None:  # noqa: N802 - Qt override
        super().showEvent(event)
        self._ever_shown = True
        self.refresh()
        QTimer.singleShot(0, self._fit_tables)
        if not self._analyzed_once and self._actions is not None and self._connected:
            self._start_analysis()

    def hideEvent(self, event: Any) -> None:  # noqa: N802 - Qt override
        super().hideEvent(event)
        self._poll_timer.stop()

    def resizeEvent(self, event: Any) -> None:  # noqa: N802 - Qt override
        super().resizeEvent(event)
        self._fit_tables()

    def _fit_tables(self) -> None:
        for t in (self.compare_table, self.plan_table):
            if t.rowCount():
                _fit_table(t)

    # =========================================================================================
    # reading the widgets
    # =========================================================================================
    def _goal(self) -> str:
        try:
            g = str(get_field_value(self.goal_combo, self._fields["optimize.goal"]))
        except Exception:  # noqa: BLE001 - a widget that cannot be read counts as the default goal
            g = "balanced"
        return g if g in GOALS else "balanced"

    def _options(self) -> dict[str, Any]:
        """The UNSAVED limit widgets, as the planner's options (what Analyze should use)."""
        out: dict[str, Any] = {}
        for key in ("reserve_vram_mb", "ctx_cap", "asr_gpu"):
            path = f"optimize.{key}"
            w = self._form.widgets.get(path)
            if w is not None and path in self._fields:
                out[key] = get_field_value(w, self._fields[path])
        return out

    def _model_key(self) -> str:
        """"" (= the loaded model) unless the user picked one in the model combo."""
        if self._model_user and self.model_combo.count():
            return str(self.model_combo.currentData() or "")
        return ""

    def _scopes(self) -> list[str]:
        return list(SCOPES.get(self.scope_combo.currentText(), ["llm", "speech"]))

    def _plans(self) -> dict[str, Any]:
        return _d(self._snap.get("plans"))

    def _current_plan(self) -> dict[str, Any]:
        return _d(self._plans().get(self._goal()))

    # =========================================================================================
    # rendering
    # =========================================================================================
    def _render(self) -> None:
        for step in (self._render_machine, self._render_models, self._render_plans, self._render_run,
                     self._render_apply_state, self._render_buttons):
            try:
                step()
            except Exception:  # noqa: BLE001 - a malformed snapshot must not break the rest of the page
                log.exception("Optimize panel: %s failed", step.__name__)

    # ---- machine ----------------------------------------------------------------------------
    @staticmethod
    def _gpu_name(facts: dict[str, Any]) -> str:
        gpu = facts.get("gpu")
        return str(_d(gpu).get("name", "")) if isinstance(gpu, dict) else str(gpu or "")

    @staticmethod
    def _facts_vram(facts: dict[str, Any]) -> tuple[float | None, float | None, float | None]:
        """(total, free, held by others) in MiB, from ``facts["vram"]`` or the ``facts["gpu"]`` dict."""
        v = _d(facts.get("vram"))
        g = _d(facts.get("gpu"))
        total = _num(v.get("total_mb", g.get("total_mb")))
        free = _num(v.get("free_mb", g.get("free_mb")))
        others = _num(v.get("others_mb", g.get("baseline_mb")))
        return total, free, others

    @staticmethod
    def _power(facts: dict[str, Any]) -> dict[str, Any]:
        p = _d(facts.get("power"))
        return p if p else _d(facts.get("gpu"))

    def _machine_text(self) -> str:
        facts = _d(self._snap.get("facts"))
        if not facts:
            return ""
        lines: list[str] = []
        cpu = str(facts.get("cpu") or "")
        cores = _int(facts.get("cores", facts.get("physical_cores")))
        head = (cpu + (f", {cores} cores" if cores else "")) if cpu else str(facts.get("machine") or "")
        ram_total, ram_free, mts = _num(facts.get("ram_total_mb")), _num(facts.get("ram_free_mb")), _num(facts.get("mem_mts"))
        if ram_total:
            ram = f"{ram_total / 1024:.1f} GB RAM"
            if mts:
                ram += f" ({int(mts)} MT/s)"
            if ram_free:
                ram += f", {ram_free / 1024:.1f} GB free"
            head = f"{head} · {ram}" if head else ram
        if head:
            lines.append(head)
        total, free, others = self._facts_vram(facts)
        gpu = self._gpu_name(facts)
        if gpu or total:
            parts = [gpu] if gpu else []
            if total:
                mem = f"{int(total)} MiB"
                if free is not None:
                    mem += f", {int(free)} free"
                if others is not None:
                    mem += f", {int(others)} held by others"
                parts.append(mem)
            lines.append(" · ".join(parts))
        extras: list[str] = []
        if facts.get("igpu"):
            extras.append(f"iGPU {facts['igpu']}")
        if facts.get("npu"):
            extras.append(f"NPU {facts['npu']}")
        if not extras:
            extras.extend(str(d) for d in _l(facts.get("devices")))
        if facts.get("runtime"):
            extras.append(f"runtime {facts['runtime']}")
        if extras:
            lines.append(" · ".join(extras))
        power = self._power(facts)
        if power.get("power_capped"):
            pieces: list[str] = []
            w, lim = _num(power.get("power_w")), _num(power.get("power_limit_w"))
            if w and lim:
                pieces.append(f"{w:.0f} W of {lim:.0f} W")
            mhz, mhz_max = _num(power.get("sm_mhz")), _num(power.get("sm_max_mhz"))
            if mhz and mhz_max:
                pieces.append(f"{mhz:.0f} MHz of {mhz_max:.0f}")
            lines.append("power-capped" + (": " + ", ".join(pieces) if pieces else ""))
        return "\n".join(lines)

    def _render_machine(self) -> None:
        self.machine_label.setText(self._machine_text())
        self.goal_text.setText(GOAL_TEXT.get(self._goal(), ""))

    def _render_models(self) -> None:
        models = [m for m in _l(self._snap.get("models")) if isinstance(m, dict) and m.get("key")]
        loaded = str(_d(self._snap.get("engine")).get("model") or "")
        sig = tuple(str(m["key"]) for m in models)
        if sig != self._models_sig:
            previous = str(self.model_combo.currentData() or "")
            self._populating = True
            try:
                self.model_combo.clear()
                for m in models:
                    size = _num(m.get("size_gb"))
                    text = str(m["key"]) + (f"  ({size:.1f} GB)" if size else "")
                    self.model_combo.addItem(text, str(m["key"]))
                want = previous if (previous and previous in sig and self._model_user) else loaded
                idx = self.model_combo.findData(want) if want else -1
                self.model_combo.setCurrentIndex(idx if idx >= 0 else (0 if models else -1))
            finally:
                self._populating = False
            self._models_sig = sig
        show = len(models) > 1
        self.model_label.setVisible(show)
        self.model_combo.setVisible(show)

    # ---- plans ------------------------------------------------------------------------------
    def _set_plan_area_visible(self, on: bool) -> None:
        for w in (self.compare_title, self.compare_table, self.plan_title, self.plan_table,
                  self.devices_title, self.devices_label, self.vram_label):
            w.setVisible(on)

    def _render_plans(self) -> None:
        goal = self._goal()
        plans = self._plans()
        if not self._connected:
            self._clear_plan_tables()
            self._set_plan_area_visible(False)
            self.why_label.setText(DEGRADED_TEXT)
            self.warnings_label.setText("")
            return
        plan = _d(plans.get(goal))
        if not plan:
            self._clear_plan_tables()
            self._set_plan_area_visible(False)
            analysis = _d(self._snap.get("analysis"))
            if analysis.get("state") == "error":
                self.why_label.setText(str(analysis.get("message") or "The analysis failed."))
            else:
                self.why_label.setText(str(analysis.get("message") or "Analysing this machine..."))
            self.warnings_label.setText("")
            return
        self._set_plan_area_visible(True)
        changed = plans != self._last_plans or goal != self._last_goal
        if changed:
            self._last_plans = copy.deepcopy(plans)
            self._last_goal = goal
            self._fill_compare(plans, goal)
            self._fill_plan_table(plan)
            self._fit_tables()
        self._update_why(plan)
        self._render_warnings(plan)
        self._render_devices(plan)
        self._render_vram(plan)

    def _clear_plan_tables(self) -> None:
        self._rendering = True
        try:
            self.plan_table.setRowCount(0)
            self.compare_table.setRowCount(0)
        finally:
            self._rendering = False
        self._row_items = []
        self._last_plans = None
        self._last_goal = ""
        self.devices_label.setText("")
        self.vram_label.setText("")
        self.vram_plain = ""

    @staticmethod
    def _plan_kv(plan: dict[str, Any]) -> str:
        for it in _l(plan.get("items")):
            if _d(it).get("id") == "llm":
                kv = _d(_d(it).get("profile")).get("kv_cache_type")
                if kv:
                    return str(kv)
                m = re.search(r"(\w+) KV", str(_d(it).get("proposed", "")))
                return m.group(1) if m else ""
        return ""

    @staticmethod
    def _plan_speech(plan: dict[str, Any]) -> str:
        for it in _l(plan.get("items")):
            if _d(it).get("id") == "speech":
                return _asr_name(str(_d(it).get("proposed") or _d(it).get("now") or ""))
        return ""

    def _fill_compare(self, plans: dict[str, Any], current: str) -> None:
        t = self.compare_table
        self._rendering = True
        try:
            t.setRowCount(len(GOALS))
            for r, goal in enumerate(GOALS):
                plan = _d(plans.get(goal))
                pred, vram = _d(plan.get("predicted")), _d(plan.get("vram"))
                ctx = _int(pred.get("ctx"))
                window = f"{ctx:,}" if ctx else ("advice only" if plan else "")
                free = _num(vram.get("free_after_mb"))
                reread = _num(pred.get("cold_reread_s"))
                reread_txt = ""
                if reread is not None:
                    reread_txt = ("~" if pred.get("pp_basis") == "typical" else "") + f"{reread:.1f} s"
                cells = (goal, window, self._plan_kv(plan), self._plan_speech(plan),
                         f"{free / 1024:.1f} GB" if free is not None else "", reread_txt)
                for c, text in enumerate(cells):
                    item = QTableWidgetItem(text)
                    item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
                    if goal == current:
                        _item_foreground(item, COLORS["accent"])
                        font = item.font()
                        font.setBold(True)
                        item.setFont(font)
                    if c == 4 and free is not None and free < self._guard_floor():
                        _item_foreground(item, COLORS["err"])
                    if c == 5 and pred.get("pp_basis") == "typical":
                        item.setToolTip("At the typical prompt speed; run the benchmark for a measured figure.")
                    t.setItem(r, c, item)
            t.setCurrentCell(GOALS.index(current) if current in GOALS else 0, 0)
        finally:
            self._rendering = False

    def _tick_state(self, digest: str, it: dict[str, Any]) -> bool:
        if not (it.get("changed") and it.get("enabled", True)):
            return False
        return self._user_ticks.get((digest, str(it.get("id"))), bool(it.get("tick")))

    def _fill_plan_table(self, plan: dict[str, Any]) -> None:
        items = [_d(i) for i in _l(plan.get("items"))]
        digest = str(plan.get("digest") or "")
        self._row_items = items
        self._row_digest = digest
        t = self.plan_table
        self._rendering = True
        try:
            t.clearContents()
            t.setRowCount(len(items))
            for r, it in enumerate(items):
                changed, enabled = bool(it.get("changed")), bool(it.get("enabled", True))
                kind = str(it.get("evidence_kind") or "estimated")
                glyph = GLYPHS.get(kind, GLYPHS["estimated"])
                ev_text = str(it.get("evidence_text") or "")
                cells = (
                    "",
                    str(it.get("label") or it.get("id") or ""),
                    str(it.get("now") or ""),
                    str(it.get("proposed") or ""),
                    APPLIES_TEXT.get(str(it.get("applies") or "none"), str(it.get("applies") or "")) if changed else "",
                    f"{glyph} {ev_text or kind}",
                )
                for c, text in enumerate(cells):
                    cell = QTableWidgetItem(text)
                    flags = Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled
                    if c == 0:
                        if changed:
                            if enabled:
                                flags |= Qt.ItemFlag.ItemIsUserCheckable
                            else:
                                flags = Qt.ItemFlag.ItemIsSelectable
                            cell.setCheckState(Qt.CheckState.Checked if self._tick_state(digest, it)
                                               else Qt.CheckState.Unchecked)
                            if not enabled:
                                cell.setToolTip("Blocked: " + "; ".join(str(b) for b in _l(plan.get("blocked"))))
                    else:
                        if not changed:
                            _item_foreground(cell, COLORS["text_dim"])
                        if c == 3 and changed:
                            _item_foreground(cell, COLORS["accent"])
                        if c == 5:
                            _item_foreground(cell, _GLYPH_COLORS.get(kind, COLORS["text_dim"]))
                        if text:
                            cell.setToolTip(text)
                    cell.setFlags(flags)
                    t.setItem(r, c, cell)
            keep = next((i for i, it in enumerate(items) if str(it.get("id")) == self._selected_id), -1)
            if keep >= 0:
                t.selectRow(keep)
            else:
                t.clearSelection()
        finally:
            self._rendering = False

    def _update_why(self, plan: dict[str, Any]) -> None:
        row = self._selected_row()
        if row is not None:
            it = self._row_items[row]
            text = f"{it.get('label', '')}: {it.get('why', '')}".strip()
            ev = str(it.get("evidence_text") or "")
            if ev:
                text += f"\nEvidence: {ev}"
            self.why_label.setText(text)
        else:
            self.why_label.setText("\n".join(str(s) for s in _l(plan.get("summary"))))

    def _selected_row(self) -> int | None:
        rows = sorted({i.row() for i in self.plan_table.selectedIndexes()})
        if rows and 0 <= rows[0] < len(self._row_items):
            return rows[0]
        return None

    def _render_warnings(self, plan: dict[str, Any]) -> None:
        blocked = [str(b) for b in _l(plan.get("blocked"))]
        warns = [str(w) for w in _l(plan.get("warnings"))]
        parts = [f"<span style='color:{COLORS['err']}'>Blocked: {_esc(b)}</span>" for b in blocked]
        parts += [f"<span style='color:{COLORS['warn']}'>! {_esc(w)}</span>" for w in warns]
        self.warnings_label.setTextFormat(Qt.TextFormat.RichText)
        self.warnings_label.setText("<br>".join(parts))

    def _render_devices(self, plan: dict[str, Any]) -> None:
        lines = []
        for dev, what in _d(plan.get("devices")).items():
            lines.append(f"{dev}: " + "; ".join(str(w) for w in _l(what)))
        self.devices_label.setText("\n".join(lines))

    def _guard_floor(self) -> float:
        facts = _d(self._snap.get("facts"))
        floor = _num(facts.get("guard_floor_mb"))
        if floor is None:
            floor = _num(getattr(getattr(self._cfg, "llm", None), "vram_guard_free_mb", None))
        return floor if floor is not None else 1500.0

    def _render_vram(self, plan: dict[str, Any]) -> None:
        """Two text bars, "now" over "plan": others | weights | KV | buffers | speech | free. No
        custom painting; free memory under the guard floor is drawn in the error colour."""
        facts = _d(self._snap.get("facts"))
        vram = _d(plan.get("vram"))
        total, free_now, others_now = self._facts_vram(facts)
        total = total or _num(vram.get("total_mb"))
        if not total:
            self.vram_label.setText("")
            self.vram_plain = ""
            return
        floor = self._guard_floor()
        ok_c, err_c = COLORS["ok"], COLORS["err"]

        def seg(char: str, n: int, color: str) -> str:
            return f"<span style='color:{color}'>{_esc(char * n)}</span>" if n else ""

        plain_rows: list[str] = []
        html_rows: list[str] = []
        legend: list[str] = []
        # now: others | this app | free
        if free_now is not None:
            others = max(0.0, others_now or 0.0) if others_now is not None else 0.0
            app = max(0.0, total - free_now - others)
            cells = _bar_cells([others, app, max(0.0, free_now)], total, _BAR_WIDTH)
            free_c = err_c if free_now < floor else ok_c
            html_rows.append("now   " + seg("░", cells[0], COLORS["text_faint"]) + seg("█", cells[1], COLORS["accent"])
                             + seg("·", cells[2], free_c))
            plain_rows.append("now   " + "░" * cells[0] + "█" * cells[1] + "·" * cells[2])
            legend.append(f"now: others {int(others)} | this app {int(app)} | free {int(free_now)} MiB of {int(total)}")
        # plan: others | weights | KV | buffers | speech | free
        if vram:
            others = _num(vram.get("others_mb")) or 0.0
            weights, kv = _num(vram.get("weights_mb")) or 0.0, _num(vram.get("kv_mb")) or 0.0
            buffers = sum(_num(vram.get(k)) or 0.0 for k in ("state_mb", "overhead_mb", "extra_mb"))
            speech = _num(vram.get("asr_mb")) or 0.0
            free = max(0.0, total - others - weights - kv - buffers - speech)
            cells = _bar_cells([others, weights, kv, buffers, speech, free], total, _BAR_WIDTH)
            free_c = err_c if free < floor else ok_c
            html_rows.append("plan  " + seg("░", cells[0], COLORS["text_faint"]) + seg("█", cells[1], COLORS["accent"])
                             + seg("▓", cells[2], "#c4b5fd") + seg("▒", cells[3], COLORS["text_dim"])
                             + seg("▄", cells[4], "#f9a8d4") + seg("·", cells[5], free_c))
            plain_rows.append("plan  " + "░" * cells[0] + "█" * cells[1] + "▓" * cells[2]
                              + "▒" * cells[3] + "▄" * cells[4] + "·" * cells[5])
            tail = f"free {int(free)}" + (f" (below the {int(floor)} MiB floor)" if free < floor else "")
            legend.append(f"plan: others {int(others)} | weights {int(weights)} | KV {int(kv)} | buffers {int(buffers)} | "
                          f"speech {int(speech)} | {tail} MiB")
        self.vram_plain = "\n".join(plain_rows + legend)
        legend_html = "<br>".join(_esc(x) for x in legend)
        self.vram_label.setText(
            f"<pre style=\"margin:0; {_MONO}\">" + "\n".join(html_rows) + "</pre>"
            f"<span style='color:{COLORS['text_dim']}'>{legend_html}</span>")
        self.vram_label.setToolTip(self.vram_plain)

    # ---- run / results ----------------------------------------------------------------------
    def _set_status(self, text: str, state: str = "") -> None:
        self.bench_status.setText(text)
        self.bench_status.setVisible(bool(text))        # no empty gap on a quiet page
        color = {"ok": COLORS["ok"], "warn": COLORS["warn"], "error": COLORS["err"]}.get(state, COLORS["text_dim"])
        self.bench_status.setStyleSheet(f"color: {color};")

    def _set_result(self, text: str, state: str = "") -> None:
        self.apply_result.setText(text)
        self.apply_result.setVisible(bool(text))
        color = {"ok": COLORS["ok"], "warn": COLORS["warn"], "error": COLORS["err"]}.get(state, COLORS["text_dim"])
        self.apply_result.setStyleSheet(f"color: {color};")

    def _gate_reason(self) -> str:
        """Why the selected benchmark scopes cannot start right now ("" = they can)."""
        gates = _d(self._snap.get("gates"))
        for scope in self._scopes():
            g = gates.get(scope)
            if isinstance(g, (list, tuple)) and len(g) >= 1 and not g[0]:
                return str(g[1]) if len(g) > 1 and g[1] else "not available right now"
        return ""

    def _render_run(self) -> None:
        if not self._connected:
            self.bench_bar.setVisible(False)
            self._set_status("")
            self.results_label.setText("")
            return
        run = _d(self._snap.get("run"))
        rid = _int(run.get("run_id"))
        if rid < self._run_id:
            return                              # a result from an older run: ignore it
        self._run_id = rid
        state = str(run.get("state") or "idle")
        message = str(run.get("message") or "")
        fraction = min(1.0, max(0.0, _num(run.get("fraction")) or 0.0))
        self.bench_bar.setVisible(state == "running")
        self.bench_bar.setValue(int(round(fraction * 100)))
        analysis = _d(self._snap.get("analysis"))
        if state == "running":
            self._refusal = ""
            step = STEP_TEXT.get(str(run.get("step") or ""), str(run.get("step") or "Working"))
            self._set_status(f"{step}{': ' + message if message else '...'}", "warn")
        elif state == "done":
            self._refusal = ""
            self._set_status(message or "Done.", "ok")
        elif state == "error":
            self._refusal = ""
            self._set_status(message or "The benchmark failed.", "error")
        elif state == "cancelled":
            self._refusal = ""
            self._set_status(message or "Cancelled.", "")
        elif self._refusal:
            self._set_status(self._refusal, "error")
        elif analysis.get("state") == "running":
            self._set_status("Analysing this machine...", "warn")
        elif analysis.get("state") == "error":
            self._set_status(str(analysis.get("message") or "The analysis failed."), "error")
        else:
            reason = self._gate_reason()
            self._set_status(f"Not now: {reason}" if reason else "", "")
        self.results_label.setText(self._results_text())

    def _results_text(self) -> str:
        m = _d(self._snap.get("measured"))
        llm, asr = _d(m.get("llm")), [a for a in _l(m.get("asr")) if isinstance(a, dict)]
        if not llm and not asr:
            return "No benchmark has been run on this machine yet: the plan uses typical numbers (◌)."
        parts: list[str] = []
        if llm:
            bits = []
            pp, tg, ttft = _num(llm.get("pp")), _num(llm.get("tg")), _num(llm.get("cached_ttft_ms"))
            if pp:
                bits.append(f"{pp:.0f} tok/s read")
            if tg:
                bits.append(f"{tg:.1f} tok/s written")
            if ttft:
                bits.append(f"{ttft:.0f} ms to first token cached")
            extra = []
            if llm.get("rounds"):
                extra.append(f"{_int(llm.get('rounds'))} rounds")
            spread = _num(llm.get("spread"))
            if spread is not None:
                extra.append(f"spread {spread * 100:.0f}%")
            if bits:
                parts.append("Answers: " + ", ".join(bits) + (f" ({', '.join(extra)})" if extra else ""))
        if asr:
            rows = []
            for a in asr:
                sec = _num(a.get("median_s"))
                if sec is not None:
                    rows.append(f"{_asr_name(str(a.get('label') or ''))} {_fmt_s(sec)} s")
            if rows:
                parts.append("Speech: " + ", ".join(rows))
        at = str(m.get("at") or "").replace("T", " ")[:16]
        if at:
            parts.append(f"measured {at}" + (" (older)" if m.get("stale") else ""))
        return " · ".join(parts)

    # ---- apply state / buttons --------------------------------------------------------------
    def _render_apply_state(self) -> None:
        if self._pending is None:
            return
        a = _d(self._snap.get("apply"))
        state = str(a.get("state") or "idle")
        message = str(a.get("message") or "")
        kind = self._pending["kind"]
        digest_ok = kind == "undo" or not a.get("digest") or a.get("digest") == self._pending["digest"]
        if state == "running":
            self._set_result("Reloading the model... " + message if kind == "apply" else "Restoring the model... " + message, "warn")
        elif state in _APPLY_TERMINAL and digest_ok:
            self._pending = None
            if state == "verified":
                self._model_verified = kind == "apply"
                self._set_result(("Model change applied and verified. " if kind == "apply" else "Previous model setup restored. ")
                                 + message, "ok")
            elif state == "rolled-back":
                self._model_verified = False
                self._set_result("Rolled back to the previous setup: " + (message or "the check failed."), "error")
            else:
                self._set_result(f"Model change {state}: " + (message or "no reason given."), "error")
            if state in ("verified", "rolled-back"):
                self._start_analysis()

    def _ticked(self, plan: dict[str, Any]) -> list[dict[str, Any]]:
        digest = str(plan.get("digest") or "")
        return [it for it in (_d(i) for i in _l(plan.get("items"))) if self._tick_state(digest, it)]

    @staticmethod
    def _apply_label(n: int, has_model: bool) -> str:
        if n <= 0:
            return "Apply (nothing selected)"
        text = f"Apply {n} change{'s' if n != 1 else ''}"
        if has_model:
            text += " (reloads the model, answers pause about 15 s)"
        return text

    def _render_buttons(self) -> None:
        snap = self._snap
        connected = self._connected
        run, apply_, analysis = _d(snap.get("run")), _d(snap.get("apply")), _d(snap.get("analysis"))
        run_running = run.get("state") == "running"
        apply_running = apply_.get("state") == "running" or self._pending is not None
        plan = self._current_plan()
        ticked = self._ticked(plan) if plan else []
        has_model = any(_d(it.get("profile")) or it.get("applies") == "model-restart" for it in ticked)
        gate_reason = self._gate_reason() if connected else ""

        self.bench_btn.setEnabled(connected and not run_running and not apply_running and not gate_reason)
        self.bench_btn.setToolTip(gate_reason)
        self.cancel_btn.setEnabled(connected and run_running)
        self.analyze_btn.setEnabled(connected and analysis.get("state") != "running")
        self.apply_btn.setText(self._apply_label(len(ticked), has_model))
        self.apply_btn.setEnabled(connected and bool(ticked) and not run_running and not apply_running)
        apply_gate = _d(snap.get("gates")).get("apply")
        if has_model and isinstance(apply_gate, (list, tuple)) and apply_gate and not apply_gate[0]:
            self.apply_btn.setToolTip("The model part cannot be applied now: " + str(apply_gate[1] if len(apply_gate) > 1 else ""))
        else:
            self.apply_btn.setToolTip("")
        undoable = bool(self._undo_values) or bool(apply_.get("undo")) or self._model_verified
        self.undo_btn.setEnabled(connected and undoable and not run_running and not apply_running)
        self.scope_combo.setEnabled(connected and not run_running)
        if not connected:
            self.bench_btn.setToolTip("Open Settings from the running app to benchmark.")

    # =========================================================================================
    # actions
    # =========================================================================================
    def _start_analysis(self) -> None:
        if self._actions is None:
            return
        opts, key = self._options(), self._model_key()
        self._last_analysis_args = (tuple(sorted(opts.items())), key)
        self._analyzed_once = True
        self._call("optimize_analyze", self._goal(), opts, key)
        self._grace = _GRACE_POLLS
        self.refresh()

    def _on_analyze(self) -> None:
        self._start_analysis()

    def _schedule_reanalysis(self) -> None:
        if self._actions is None or not self._connected:
            return
        self._reanalyze_timer.start()

    def _flush_reanalysis(self) -> None:
        """The debounce fired: re-analyse only if the options or model really differ from the last
        analysis (a ``load()`` that re-sets the same values must not cost an engine round trip)."""
        if not self._analyzed_once or self._actions is None:
            return                              # the first show will analyse with the current values
        opts, key = self._options(), self._model_key()
        if (tuple(sorted(opts.items())), key) == self._last_analysis_args:
            return
        self._start_analysis()

    def _on_scope_changed(self) -> None:
        """A different benchmark scope has a different gate: redraw the status line and the buttons."""
        self._render_run()
        self._render_buttons()

    def _on_goal_changed(self) -> None:
        """Only re-render: all four goals' plans are already in the snapshot."""
        self._render_machine()
        self._render_plans()
        self._render_buttons()

    def _on_model_changed(self) -> None:
        if self._populating:
            return
        self._model_user = True
        self._schedule_reanalysis()

    def _on_compare_clicked(self, row: int) -> None:
        if self._rendering or not (0 <= row < len(GOALS)):
            return
        goal = GOALS[row]
        if goal != self._goal():
            idx = self.goal_combo.findText(goal)
            if idx >= 0:
                self.goal_combo.setCurrentIndex(idx)

    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        if self._rendering or item.column() != 0:
            return
        row = item.row()
        if 0 <= row < len(self._row_items):
            self._user_ticks[(self._row_digest, str(self._row_items[row].get("id")))] = (
                item.checkState() == Qt.CheckState.Checked)
            if len(self._user_ticks) > 64:
                self._user_ticks.pop(next(iter(self._user_ticks)))
        self._render_buttons()

    def _on_row_selected(self) -> None:
        if self._rendering:
            return
        row = self._selected_row()
        self._selected_id = str(self._row_items[row].get("id")) if row is not None else ""
        self._update_why(self._current_plan())

    def _on_run(self) -> None:
        if self._actions is None:
            return
        ok, res = self._call("optimize_benchmark", self._scopes(), force=False)
        if not ok or not res:
            self.refresh()
            msg = str(_d(self._snap.get("run")).get("message") or "")
            self._refusal = msg or "refused"
            self._set_status(self._refusal, "error")
            return
        self._refusal = ""
        self._grace = _GRACE_POLLS
        self.refresh()

    def _on_cancel(self) -> None:
        self._call("optimize_cancel")
        self.refresh()

    def _on_apply(self) -> None:
        """Settings rows through the dialog's widgets, the model row through the engine."""
        plan = self._current_plan()
        if not plan or self._pending is not None:
            return
        ticked = self._ticked(plan)
        if not ticked:
            return
        values: dict[str, Any] = {}
        for it in ticked:
            values.update(_d(it.get("settings")))
        model_row = next((it for it in ticked if _d(it.get("profile"))), None)
        notes: list[str] = []
        level = "ok"
        self._model_verified = False

        if values:
            paths = set(values)
            ok_prev, previous = self._host_call("widget_values", paths)
            self._host_call("set_values", values)
            ok_apply, changed = self._host_call("apply_paths", paths)
            if ok_prev and isinstance(previous, dict):
                # remember what a REAL change replaced; a path that was already at its planned value
                # keeps the older undo value (pressing Apply twice must not lose the original)
                for path, before in previous.items():
                    if before != values.get(path) or path not in self._undo_values:
                        self._undo_values[path] = before
            n = len(changed) if isinstance(changed, dict) else 0
            if not ok_apply:
                notes.append("The settings could not be applied.")
                level = "error"
            elif n:
                note = self._host_note()
                notes.append(f"Applied {n} setting{'s' if n != 1 else ''}." + (f" {note}" if note and note != "Applied." else ""))
            else:
                notes.append("The settings were already in place.")

        if model_row is not None:
            digest = str(plan.get("digest") or "")
            ok, res = self._call("optimize_apply", digest)
            if ok and res:
                self._pending = {"kind": "apply", "digest": digest}
                self._pending_since = time.monotonic()
                notes.append("Reloading the model; answers pause for about 15 s.")
                level = "warn" if level == "ok" else level
                self._grace = _GRACE_POLLS
            else:
                self.refresh()
                reason = str(_d(self._snap.get("apply")).get("message") or "") or "refused"
                notes.append(f"The model change was refused: {reason}")
                level = "error"
        self._set_result(" ".join(notes), level)
        if self._pending is None and values:
            self._start_analysis()              # the config moved: plan again (refreshes the page)
        else:
            self.refresh()                      # a pending apply that already finished replaces the text above

    def _host_note(self) -> str:
        """The dialog's own note after ``apply_paths`` (it lists the "(restart)" fields)."""
        try:
            return str(self._host.note.text())
        except Exception:  # noqa: BLE001 - a host without a note label simply adds nothing
            return ""

    def _on_undo(self) -> None:
        if self._pending is not None:
            return
        notes: list[str] = []
        level = "ok"
        if self._undo_values:
            vals = dict(self._undo_values)
            self._host_call("set_values", vals)
            self._host_call("apply_paths", set(vals))
            self._undo_values = {}
            notes.append("Restored the previous settings.")
        if _d(self._snap.get("apply")).get("undo") or self._model_verified:
            ok, res = self._call("optimize_undo")
            if ok and res:
                self._pending = {"kind": "undo", "digest": ""}
                self._pending_since = time.monotonic()
                notes.append("Restoring the previous model setup...")
                level = "warn"
                self._grace = _GRACE_POLLS
            else:
                self.refresh()
                reason = str(_d(self._snap.get("apply")).get("message") or "") or "refused"
                notes.append(f"The model undo was refused: {reason}")
                level = "error"
        self._model_verified = False
        self._set_result(" ".join(notes) or "Nothing to undo.", level)
        if self._pending is None and notes:
            self._start_analysis()
        else:
            self.refresh()

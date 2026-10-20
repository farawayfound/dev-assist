"""Models dialog: the app's own GGUF library, Hugging Face discovery, per-model tuning, the
global LLM server settings and a read-only hardware report.

Opened from the overlay's ``...`` menu ("Models...", right after "Settings...") and the slim
bar's Context menu, replacing the old Settings dialog's "Model" tab and the "Choose model"
submenu (2026-09-27, portable-install). Like every other top-level window of the app it is
capture-excluded and kept above the overlay by ``OverlayWindow``'s application-wide show filter.

This module talks to the engine ONLY through an ``EngineActions`` implementation handed in at
construction time (never ``exocortex.llm.library`` / ``hardware`` / ``server`` directly) and
listens for ``ModelPullEvent`` / ``ModelLibraryEvent`` / ``ModelSearchEvent`` / ``ModelInfoEvent``
/ ``ComponentStatusEvent("llm", ...)`` through :meth:`ModelsDialog.on_event`, which the overlay
forwards. Every ``actions`` call is wrapped in try/except: a slow network, a missing backend or a
test double whose methods return ``None`` must never crash this dialog.
"""
from __future__ import annotations

import logging
import shlex
from pathlib import Path
from typing import TYPE_CHECKING, Any

from PySide6.QtCore import QTimer, QUrl, Qt, Signal
from PySide6.QtGui import QColor, QDesktopServices
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from exocortex.bus import (
    ComponentStatusEvent,
    ModelInfoEvent,
    ModelLibraryEvent,
    ModelPullEvent,
    ModelSearchEvent,
)
from exocortex.config import Config
from exocortex.ui.theme import COLORS, build_qss

log = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover - import cycle guard, actions.py has no Qt deps
    from exocortex.actions import EngineActions


# ---------------------------------------------------------------------------------------------
# Field / FieldForm helpers, shared with exocortex.ui.settings (agent U2 is landing them there
# concurrently -- see the contract in the task). Imported when present; a local, behaviourally
# identical fallback keeps this module working either way.
# ---------------------------------------------------------------------------------------------
try:
    from exocortex.ui.settings import Field, FieldForm, get_field_value, make_field_widget, set_field_value
except ImportError:  # pragma: no cover - only exercised until exocortex.ui.settings lands these
    from dataclasses import dataclass
    from collections.abc import Callable

    from PySide6.QtWidgets import QDoubleSpinBox, QSpinBox

    from exocortex.ui.settings import Field  # already there: only the helpers below are missing

    def make_field_widget(f: "Field", parent: QWidget | None = None, *,
                           refresh_models: "Callable[[], None] | None" = None,
                           list_models: "Callable[[], list[str]] | None" = None) -> QWidget:
        if f.kind == "bool":
            return QCheckBox(parent)
        if f.kind == "int":
            w = QSpinBox(parent)
            w.setRange(int(f.lo), int(f.hi))
            w.setSingleStep(max(1, int(f.step)))
            return w
        if f.kind == "float":
            w = QDoubleSpinBox(parent)
            w.setRange(float(f.lo), float(f.hi))
            w.setSingleStep(float(f.step))
            w.setDecimals(2)
            return w
        if f.kind == "choice":
            w = QComboBox(parent)
            w.addItems(list(f.choices))
            return w
        return QLineEdit(parent)

    def set_field_value(widget: QWidget, f: "Field", value: Any) -> None:
        if isinstance(widget, QCheckBox):
            widget.setChecked(bool(value))
        elif isinstance(widget, QDoubleSpinBox):
            widget.setValue(float(value))
        elif isinstance(widget, QSpinBox):
            widget.setValue(int(value))
        elif isinstance(widget, QComboBox):
            text = str(value)
            idx = widget.findText(text)
            if idx < 0:
                widget.addItem(text)
                idx = widget.findText(text)
            widget.setCurrentIndex(idx)
        elif isinstance(widget, QLineEdit):
            widget.setText(str(value))

    def get_field_value(widget: QWidget, f: "Field") -> Any:
        if isinstance(widget, QCheckBox):
            return widget.isChecked()
        if isinstance(widget, QDoubleSpinBox):
            return round(widget.value(), 3)
        if isinstance(widget, QSpinBox):
            return int(widget.value())
        if isinstance(widget, QComboBox):
            return widget.currentText().strip()
        if isinstance(widget, QLineEdit):
            return widget.text().strip()
        raise TypeError(type(widget))

    class FieldForm:
        """``FieldForm``-alike: builds one widget per field and reads/writes them into ``cfg``."""

        def __init__(self, fields: tuple["Field", ...], cfg: Config, parent: QWidget | None = None) -> None:
            self.fields = fields
            self._cfg = cfg
            self._parent = parent
            self.widgets: dict[str, QWidget] = {f.path: make_field_widget(f, parent) for f in fields}
            self._widget: QWidget | None = None

        @property
        def widget(self) -> QWidget:
            """The ``QScrollArea`` holding a label + widget row per field (built once, lazily)."""
            if self._widget is None:
                from PySide6.QtWidgets import QFormLayout as _QFormLayout

                scroll = QScrollArea(self._parent)
                scroll.setWidgetResizable(True)
                inner = QWidget(scroll)
                form = _QFormLayout(inner)
                for f in self.fields:
                    form.addRow(f.label, self.widgets[f.path])
                scroll.setWidget(inner)
                self._widget = scroll
            return self._widget

        def load(self) -> None:
            for f in self.fields:
                set_field_value(self.widgets[f.path], f, getattr(getattr(self._cfg, f.section), f.key))

        def changes(self) -> dict[str, Any]:
            out: dict[str, Any] = {}
            for f in self.fields:
                new = get_field_value(self.widgets[f.path], f)
                cur = getattr(getattr(self._cfg, f.section), f.key)
                if isinstance(cur, float):
                    if abs(float(new) - cur) > 1e-9:
                        out[f.path] = float(new)
                elif new != cur:
                    out[f.path] = new
            return out

        def apply(self) -> dict[str, Any]:
            changed = self.changes()
            by_path = {f.path: f for f in self.fields}
            for path, value in changed.items():
                f = by_path[path]
                setattr(getattr(self._cfg, f.section), f.key, value)
            return changed

        def restore_defaults(self, skip: tuple[str, ...] = ()) -> None:
            defaults = Config()
            for f in self.fields:
                if f.path in skip:
                    continue
                set_field_value(self.widgets[f.path], f, getattr(getattr(defaults, f.section), f.key))


_DIALOG_QSS = f"""
QDialog {{ background: {COLORS['bg_solid']}; }}
QTabWidget::pane {{ border: 1px solid {COLORS['border']}; border-radius: 6px; top: -1px; }}
QTabBar::tab {{ background: transparent; color: {COLORS['text_dim']}; padding: 5px 10px;
                border-bottom: 2px solid transparent; }}
QTabBar::tab:selected {{ color: {COLORS['accent']}; border-bottom: 2px solid {COLORS['accent']}; }}
QSpinBox, QDoubleSpinBox, QComboBox {{ background: {COLORS['panel']}; border: 1px solid {COLORS['border']};
                                        border-radius: 4px; padding: 2px 6px; min-width: 90px; }}
QComboBox QAbstractItemView {{ background: {COLORS['bg_solid']}; color: {COLORS['text']};
                               selection-background-color: {COLORS['accent_dim']}; }}
QCheckBox {{ spacing: 6px; }}
QTableWidget {{ background: {COLORS['panel']}; border: 1px solid {COLORS['border']}; border-radius: 4px;
                gridline-color: {COLORS['border']}; }}
QHeaderView::section {{ background: {COLORS['bg_solid']}; color: {COLORS['text_dim']};
                        border: none; padding: 4px 6px; }}
"""


# ---------------------------------------------------------------------------------------------
# formatting helpers
# ---------------------------------------------------------------------------------------------
def _fmt_size(num_bytes: object) -> str:
    """``7.1 GB`` / ``480 MB``; "" when unknown."""
    if not num_bytes:
        return ""
    try:
        n = float(num_bytes)
    except (TypeError, ValueError):
        return ""
    gb = n / 1e9
    return f"{gb:.1f} GB" if gb >= 1 else f"{n / 1e6:.0f} MB"


def _fmt_params(params: object) -> str:
    """``12B`` / ``350M``; "-" when unknown."""
    if not params:
        return "-"
    try:
        n = float(params)
    except (TypeError, ValueError):
        return str(params)
    if n >= 1e9:
        return f"{n / 1e9:.0f}B"
    if n >= 1e6:
        return f"{n / 1e6:.0f}M"
    return str(int(n))


def _fmt_mb(value: object) -> str:
    """``16003 MiB`` for realistic RAM/VRAM sizes (matches ``hardware.py``'s own "16003 MiB"
    formatting), ``2048.0 GB`` once it is a disk-sized number that is worth switching units for.

    2026-09-27 (portable-install): the threshold used to be 4096, which converted every real
    VRAM/RAM total (always >= a few thousand MiB) to GB and could never print "16003 MiB" --
    contradicting this very docstring and ``tests/test_setup.py``'s "16003 MiB" expectation for
    the same hardware value. 1 TiB (in MiB) is comfortably above any RAM/VRAM total.
    """
    if value in (None, ""):
        return ""
    try:
        n = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{int(n)} MiB" if abs(n) < 1_048_576 else f"{n / 1024:.1f} GB"


def _fmt_runtime(runtime: object) -> str:
    if not isinstance(runtime, dict):
        return str(runtime) if runtime else ""
    text = " ".join(str(runtime[k]) for k in ("flavor", "version") if runtime.get(k))
    if runtime.get("source"):
        text += f" ({runtime['source']})"
    if runtime.get("exe"):
        text += f" - {runtime['exe']}"
    return text.strip()


def _fmt_gpus(gpus: object) -> str:
    if not isinstance(gpus, (list, tuple)):
        return str(gpus) if gpus else ""
    names = [str(g.get("name", "")) if isinstance(g, dict) else str(g) for g in gpus]
    return ", ".join(n for n in names if n)


def _hardware_summary(report: dict[str, object]) -> str:
    vram_total = report.get("vram_total_mb")
    if not vram_total:
        return ""
    try:
        vram_total = float(vram_total)
    except (TypeError, ValueError):
        return ""
    budget = report.get("budget_mb")
    try:
        budget = float(budget) if budget not in (None, "") else 0.75 * vram_total
    except (TypeError, ValueError):
        budget = 0.75 * vram_total
    policy = report.get("recommended_policy", "auto")
    return f"Models up to ~{budget / 1024:.0f} GB fit (0.75 x {int(vram_total)} MiB); policy: {policy}"


def _status_color(state: str) -> str:
    if state in ("ready", "done"):
        return COLORS["ok"]
    if state in ("error", "cancelled"):
        return COLORS["err"]
    if state in ("loading", "busy", "queued", "downloading", "verifying"):
        return COLORS["warn"]
    return COLORS["text_dim"]


_PULL_ACTIVE_STATES = ("queued", "downloading", "verifying")


class ModelsDialog(QDialog):
    """Tabbed model manager: Library / Discover / Tuning / Server / Hardware.

    ``applied`` carries Server-tab changes as ``{"section.key": new_value}``, the same shape as
    ``SettingsDialog.applied``; the overlay routes it to ``actions.apply_settings`` exactly like
    the settings dialog does.
    """

    TABS: tuple[str, ...] = ("Library", "Discover", "Tuning", "Server", "Hardware")
    applied = Signal(dict)

    _LIBRARY_COLUMNS: tuple[str, ...] = ("Model", "Quant", "Size", "Source", "Context", "Status")

    #: Per-model tuning overrides (Tuning tab). ``section`` is unused (these live in a per-model
    #: profile dict, not on ``Config``) -- only ``key`` / ``label`` / ``kind`` matter here.
    _TUNING_FIELDS: tuple[Field, ...] = (
        Field("profile", "ctx_size", "Context size", "int", 1024, 262144, 1024),
        Field("profile", "gpu_layers", "GPU layers", "int", 0, 999999, 1),
        Field("profile", "kv_cache_type", "KV cache type", "choice", choices=("f16", "q8_0", "q4_0")),
        Field("profile", "flash_attn", "Flash attention", "bool"),
        Field("profile", "threads", "CPU threads", "int", 0, 64, 1),
        Field("profile", "parallel", "Parallel slots", "int", 1, 8, 1),
        Field("profile", "temperature", "Temperature", "float", 0.0, 1.5, 0.05),
        Field("profile", "top_p", "Top-p", "float", 0.0, 1.0, 0.05),
        Field("profile", "top_k", "Top-k", "int", 0, 200, 1),
        Field("profile", "min_p", "Min-p", "float", 0.0, 1.0, 0.05),
        Field("profile", "thinking", "Thinking", "choice", choices=("off", "on")),
        Field("profile", "extra_args", "Extra args (space-separated)", "text"),
    )

    #: Global LLM server settings (Server tab). One ``FieldForm`` over ``cfg.llm``.
    _SERVER_FIELDS: tuple[Field, ...] = (
        Field("llm", "backend", "Backend", "choice", choices=("auto", "llamacpp", "ollama", "openai"),
              restart=True,
              help="auto = route at startup: llamacpp when a runtime pack for this GPU and at "
                   "least one GGUF are found; else ollama when a server answers at the address "
                   "below; else openai when a server answers there; else a clear error."),
        Field("llm", "server_url", "Server URL (ollama / openai backend; empty = the standard address)",
              "text", restart=True),
        Field("llm", "llama_backend", "llama.cpp GPU framework", "choice",
              choices=("auto", "cuda", "rocm", "vulkan", "cpu", "sycl", "openvino"), restart=True,
              help="Which llama.cpp runtime pack to use (llamacpp backend only)."),
        Field("llm", "llama_source", "llama-server binary source", "choice",
              choices=("auto", "lmstudio", "upstream"), restart=True),
        Field("llm", "llama_device", "GPU device (--device)", "text", restart=True),
        Field("llm", "llama_upstream_build", "Upstream llama.cpp build tag", "text", restart=True),
        Field("llm", "library_dir", "Model library folder", "text", restart=True),
        Field("llm", "runtime_dir", "Runtime pack cache folder", "text", restart=True),
        Field("llm", "llama_port", "llama-server port (0 = pick a free one)", "int", 0, 65535, 1, restart=True),
        Field("llm", "load_mode", "Load mode", "choice", choices=("mmap", "mmap+mlock"), restart=True),
        Field("llm", "thinking", "Thinking", "choice", choices=("off", "on")),
        Field("llm", "temperature", "Temperature", "float", 0.0, 1.5, 0.05),
        Field("llm", "top_p", "Top-p", "float", 0.0, 1.0, 0.05),
        Field("llm", "context_tokens", "Transcript context (tokens)", "int", 500, 12000, 250),
        Field("llm", "assist_max_tokens", "Answer budget (tokens)", "int", 50, 1500, 10),
        Field("llm", "ask_max_tokens", "Ask-box budget (tokens)", "int", 50, 2000, 10),
        Field("llm", "solve_max_tokens", "Solve budget (tokens)", "int", 100, 3000, 50),
        Field("llm", "text_max_tokens", "Text walkthrough budget (tokens)", "int", 100, 2000, 50),
        Field("llm", "notes_max_tokens", "Notes budget (tokens)", "int", 100, 2000, 50),
        Field("llm", "insights_max_tokens", "Insights budget (tokens)", "int", 100, 1500, 20),
        Field("llm", "recap_max_tokens", "Recap budget (tokens)", "int", 100, 1500, 20),
        Field("llm", "ctx_size", "Context size", "int", 2048, 65536, 1024, restart=True),
        Field("llm", "threads", "CPU threads (0 = auto)", "int", 0, 64, 1, restart=True),
        Field("llm", "parallel", "Parallel slots", "int", 1, 8, 1, restart=True),
        Field("llm", "kv_cache_type", "KV cache type", "choice", choices=("f16", "q8_0", "q4_0"), restart=True),
        Field("llm", "flash_attn", "Flash attention", "bool", restart=True),
        Field("llm", "request_timeout_s", "Request timeout (s)", "float", 10.0, 600.0, 5.0),
        Field("llm", "swap_unload_url", "llama-swap unload URL", "text",
              help='The shared llama-swap unload endpoint; "" disables.'),
        Field("llm", "vram_guard", "GPU pressure guard", "bool"),
        Field("llm", "vram_guard_free_mb", "GPU guard: free VRAM floor (MB)", "int", 0, 16000, 100),
    )

    def __init__(self, cfg: Config, actions: "EngineActions", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._cfg = cfg
        self._actions = actions
        self._library_rows: list[dict[str, object]] = []
        self._search_rows: list[dict[str, object]] = []
        self._search_request_id = 0
        self._inspect_request_id = 0
        self._pull_done_key = ""
        self._active_pull_repo: str | None = None  # review round two, 2026-09-27 (D2)
        self._tuning_key: str | None = None
        self._any_loaded = False
        self._llm_busy = False  # review round two, 2026-09-27 (D4): explicit, not a getattr default

        self.setWindowTitle("Models")
        self.setWindowFlags(Qt.WindowType.Dialog | Qt.WindowType.Tool | Qt.WindowType.WindowStaysOnTopHint)
        self.setStyleSheet(build_qss(cfg.ui.font_size) + _DIALOG_QSS)
        self.setMinimumSize(640, 460)
        self.resize(760, 560)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(8)
        self.tabs = QTabWidget(self)
        outer.addWidget(self.tabs, 1)

        self.tabs.addTab(self._build_library_tab(), "Library")
        self.tabs.addTab(self._build_discover_tab(), "Discover")
        self.tabs.addTab(self._build_tuning_tab(), "Tuning")
        self.tabs.addTab(self._build_server_tab(), "Server")
        self.tabs.addTab(self._build_hardware_tab(), "Hardware")

        self.load()

    # ---------------------------------------------------------------------------------- public
    def load(self) -> None:
        """Re-read everything from ``actions`` / ``cfg`` (called on every open)."""
        self._refresh_library()
        self._refresh_hardware()
        self._server_form.load()
        self.server_note.setText("")

    def show_tab(self, name: str) -> None:
        if name in self.TABS:
            self.tabs.setCurrentIndex(self.TABS.index(name))

    def on_event(self, ev: object) -> None:
        """Forwarded by the overlay: the five model events, plus ComponentStatusEvent("llm")."""
        try:
            if isinstance(ev, ModelPullEvent):
                self._on_model_pull_event(ev)
            elif isinstance(ev, ModelLibraryEvent):
                # review round two, 2026-09-27 (D4): busy clears only on the operation's OWN
                # completion (a "loaded" / "unloaded" reason), never on an unrelated library
                # change (a pull finishing, a delete, a default change) that happens to land
                # while a load/unload is still in flight.
                if ev.reason in ("loaded", "unloaded"):
                    self._llm_busy = False
                self._refresh_library()
            elif isinstance(ev, ModelSearchEvent):
                self._on_model_search_event(ev)
            elif isinstance(ev, ModelInfoEvent):
                self._on_model_info_event(ev)
            elif isinstance(ev, ComponentStatusEvent) and ev.component == "llm":
                self._set_library_status(ev.detail or ev.state, ev.state)
                # review round two, 2026-09-27 (D4): busy is driven by "loading" / "ready" /
                # "error" only -- an "info" status (e.g. the engine's own "busy: ..." refusal
                # notice for a second click) must never clear it while the first op still runs.
                if ev.state == "loading":
                    self._llm_busy = True
                elif ev.state in ("ready", "error"):
                    self._llm_busy = False
                self._update_library_buttons()
        except Exception:  # noqa: BLE001 - never let a bad event take the dialog down
            log.exception("ModelsDialog.on_event failed for %r", ev)

    # ================================================================================ Library
    def _build_library_tab(self) -> QWidget:
        w = QWidget(self)
        outer = QVBoxLayout(w)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(8)

        self.library_table = QTableWidget(0, len(self._LIBRARY_COLUMNS), w)
        self.library_table.setHorizontalHeaderLabels(list(self._LIBRARY_COLUMNS))
        self.library_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.library_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.library_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.library_table.itemSelectionChanged.connect(self._update_library_buttons)
        self.library_table.itemDoubleClicked.connect(lambda _item: self._on_load_clicked())
        outer.addWidget(self.library_table, 1)

        btn_row = QHBoxLayout()
        self.load_btn = QPushButton("Load", w)
        self.load_btn.clicked.connect(self._on_load_clicked)
        self.unload_btn = QPushButton("Unload", w)
        self.unload_btn.clicked.connect(self._on_unload_clicked)
        self.set_default_btn = QPushButton("Set default", w)
        self.set_default_btn.clicked.connect(self._on_set_default_clicked)
        self.delete_btn = QPushButton("Delete", w)
        self.delete_btn.clicked.connect(self._on_delete_clicked)
        self.refresh_btn = QPushButton("Refresh", w)
        self.refresh_btn.clicked.connect(self._on_refresh_clicked)
        self.open_folder_btn = QPushButton("Open folder", w)
        self.open_folder_btn.clicked.connect(self._on_open_folder_clicked)
        for b in (self.load_btn, self.unload_btn, self.set_default_btn, self.delete_btn,
                  self.refresh_btn, self.open_folder_btn):
            btn_row.addWidget(b)
        btn_row.addStretch(1)
        outer.addLayout(btn_row)

        self.library_status = QLabel("", w)
        self.library_status.setWordWrap(True)
        outer.addWidget(self.library_status)
        return w

    def _refresh_library(self) -> None:
        selected = self._selected_library_row()
        selected_key = selected.get("key") if selected else None

        try:
            result = self._actions.model_library()
        except Exception:  # noqa: BLE001
            log.exception("actions.model_library() failed")
            self._set_library_status("failed to read the model library", "error")
            result = None
        rows = list(result) if result else []
        self._library_rows = rows

        self.library_table.setRowCount(len(rows))
        for i, r in enumerate(rows):
            self._set_library_row(i, r)
        self.library_table.resizeColumnsToContents()
        self.library_table.horizontalHeader().setStretchLastSection(True)
        if selected_key is not None:
            for i, r in enumerate(rows):
                if r.get("key") == selected_key:
                    self.library_table.setCurrentCell(i, 0)
                    break
        self._any_loaded = any(r.get("loaded") for r in rows)
        self._update_library_buttons()
        self._reload_tuning_model_combo()

    def _set_library_row(self, i: int, r: dict[str, object]) -> None:
        loaded = bool(r.get("loaded"))
        is_default = bool(r.get("is_default"))
        status = "loaded" if loaded else ("default" if is_default else "")
        values = (
            str(r.get("name") or r.get("key") or ""),
            str(r.get("quant") or ""),
            _fmt_size(r.get("size_bytes")),
            str(r.get("source") or ""),
            str(r.get("context_length") or ""),
            status,
        )
        for col, value in enumerate(values):
            item = QTableWidgetItem(value)
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            if col == len(values) - 1 and loaded:
                item.setForeground(QColor(COLORS["accent"]))
            self.library_table.setItem(i, col, item)

    def _selected_library_row(self) -> dict[str, object] | None:
        idx = self.library_table.currentRow()
        if 0 <= idx < len(self._library_rows):
            return self._library_rows[idx]
        return None

    def _update_library_buttons(self) -> None:
        row = self._selected_library_row()
        has_row = row is not None
        loaded = bool(row.get("loaded")) if row else False
        # While llama-server is switching models (ComponentStatusEvent "loading"), Load / Unload /
        # Delete are off: the engine refuses a second change anyway, and a click that lands during
        # a minutes-long cold load must not look like it did something.
        busy = self._llm_busy
        self.load_btn.setEnabled(has_row and not loaded and not busy)
        self.unload_btn.setEnabled(self._any_loaded and not busy)
        self.set_default_btn.setEnabled(has_row)
        self.open_folder_btn.setEnabled(has_row and bool(row.get("path")) if row else False)
        self.delete_btn.setEnabled(has_row and row.get("source") == "library" and not loaded and not busy)

    def _set_library_status(self, text: str, state: str = "") -> None:
        self.library_status.setText(text)
        self.library_status.setStyleSheet(f"color: {_status_color(state)};")

    def _on_load_clicked(self) -> None:
        # review round two, 2026-09-27 (D4): mirrors load_btn's own enablement (see
        # _update_library_buttons) so a bypass of the disabled button -- itemDoubleClicked calls
        # this directly with no guard of its own -- can't reach the engine either.
        row = self._selected_library_row()
        if not row:
            return
        if self._llm_busy:
            self._set_library_status("busy: wait for the current load/unload to finish", "loading")
            return
        if row.get("loaded"):
            self._set_library_status(f"{row.get('name') or row.get('key')} is already loaded", "info")
            return
        key = row.get("key", "")
        try:
            self._actions.load_model(key)
        except Exception:  # noqa: BLE001
            log.exception("actions.load_model(%r) failed", key)
            self._set_library_status(f"failed to load {key}", "error")

    def _on_unload_clicked(self) -> None:
        # review round two, 2026-09-27 (D4): mirrors unload_btn's own enablement.
        if self._llm_busy:
            self._set_library_status("busy: wait for the current load/unload to finish", "loading")
            return
        if not self._any_loaded:
            self._set_library_status("no model is loaded", "info")
            return
        try:
            self._actions.unload_model()
        except Exception:  # noqa: BLE001
            log.exception("actions.unload_model() failed")
            self._set_library_status("failed to unload the model", "error")

    def _on_set_default_clicked(self) -> None:
        row = self._selected_library_row()
        if not row:
            return
        key = row.get("key", "")
        try:
            self._actions.set_default_model(key)
        except Exception:  # noqa: BLE001
            log.exception("actions.set_default_model(%r) failed", key)
            self._set_library_status(f"failed to set {key} as default", "error")

    def _on_delete_clicked(self) -> None:
        row = self._selected_library_row()
        if not row:
            return
        # review round two, 2026-09-27 (D4): mirrors delete_btn's own enablement -- never even
        # asks to confirm a delete the engine (or the disabled button) would refuse anyway.
        if self._llm_busy:
            self._set_library_status("busy: wait for the current load/unload to finish", "loading")
            return
        if row.get("loaded"):
            self._set_library_status(f"unload {row.get('name') or row.get('key')} before deleting it", "info")
            return
        key = row.get("key", "")
        name = row.get("name") or key
        answer = QMessageBox.question(
            self, "Delete model", f"Delete '{name}' from the library? This cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            self._actions.delete_model(key)
        except Exception:  # noqa: BLE001
            log.exception("actions.delete_model(%r) failed", key)
            self._set_library_status(f"failed to delete {key}", "error")

    def _on_refresh_clicked(self) -> None:
        self._refresh_library()

    def _on_open_folder_clicked(self) -> None:
        row = self._selected_library_row()
        if not row or not row.get("path"):
            return
        try:
            p = Path(str(row["path"]))
            folder = p if p.is_dir() else p.parent
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))
        except Exception:  # noqa: BLE001
            log.exception("open folder for %r failed", row.get("key"))

    # ================================================================================ Discover
    def _build_discover_tab(self) -> QWidget:
        w = QWidget(self)
        layout = QVBoxLayout(w)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        self.search_edit = QLineEdit(w)
        self.search_edit.setPlaceholderText("Search the Hugging Face Hub for a GGUF model...")
        self.search_edit.returnPressed.connect(self._do_search)
        self.search_edit.textChanged.connect(lambda _t: self._search_timer.start())
        layout.addWidget(self.search_edit)

        self._search_timer = QTimer(self)
        self._search_timer.setSingleShot(True)
        self._search_timer.setInterval(400)
        self._search_timer.timeout.connect(self._do_search)

        self.search_table = QTableWidget(0, 5, w)
        self.search_table.setHorizontalHeaderLabels(["Repo", "Downloads", "Likes", "Updated", "Gated"])
        self.search_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.search_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.search_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.search_table.itemSelectionChanged.connect(self._on_search_row_selected)
        layout.addWidget(self.search_table, 1)

        details = QFormLayout()
        self.detail_architecture = QLabel("-", w)
        self.detail_params = QLabel("-", w)
        self.detail_context = QLabel("-", w)
        details.addRow("Architecture", self.detail_architecture)
        details.addRow("Parameters", self.detail_params)
        details.addRow("Context length", self.detail_context)
        layout.addLayout(details)

        quant_row = QHBoxLayout()
        quant_row.addWidget(QLabel("Quant", w))
        self.quant_combo = QComboBox(w)
        quant_row.addWidget(self.quant_combo, 1)
        layout.addLayout(quant_row)

        self.mmproj_cb = QCheckBox("Include vision projector (mmproj)", w)
        self.mmproj_cb.setEnabled(False)
        layout.addWidget(self.mmproj_cb)

        pull_row = QHBoxLayout()
        self.pull_btn = QPushButton("Pull", w)
        self.pull_btn.setObjectName("primary")
        self.pull_btn.clicked.connect(self._on_pull_clicked)
        self.cancel_pull_btn = QPushButton("Cancel", w)
        self.cancel_pull_btn.setEnabled(False)
        self.cancel_pull_btn.clicked.connect(self._on_cancel_pull_clicked)
        self.load_now_btn = QPushButton("Load now", w)
        self.load_now_btn.hide()
        self.load_now_btn.clicked.connect(self._on_load_now_clicked)
        pull_row.addWidget(self.pull_btn)
        pull_row.addWidget(self.cancel_pull_btn)
        pull_row.addWidget(self.load_now_btn)
        pull_row.addStretch(1)
        layout.addLayout(pull_row)

        self.pull_progress = QProgressBar(w)
        self.pull_progress.setRange(0, 100)
        layout.addWidget(self.pull_progress)
        self.pull_label = QLabel("", w)
        self.pull_label.setWordWrap(True)
        layout.addWidget(self.pull_label)

        self.discover_status = QLabel("", w)
        self.discover_status.setWordWrap(True)
        layout.addWidget(self.discover_status)
        # No trailing stretch: the results table (stretch 1 above) takes every spare row, so a
        # tall window shows more of the search hits instead of an empty band under the progress bar.
        return w

    def _do_search(self) -> None:
        self._search_timer.stop()
        query = self.search_edit.text().strip()
        if not query:
            return
        self._search_request_id += 1
        rid = self._search_request_id
        try:
            self._actions.search_models(query, rid)
        except Exception:  # noqa: BLE001
            log.exception("actions.search_models(%r) failed", query)
            self.discover_status.setText("search failed")

    def _on_search_row_selected(self) -> None:
        idx = self.search_table.currentRow()
        if not (0 <= idx < len(self._search_rows)):
            return
        repo_id = str(self._search_rows[idx].get("id", ""))
        if not repo_id:
            return
        self._inspect_request_id += 1
        rid = self._inspect_request_id
        try:
            self._actions.inspect_model(repo_id, rid)
        except Exception:  # noqa: BLE001
            log.exception("actions.inspect_model(%r) failed", repo_id)

    def _populate_search_table(self) -> None:
        rows = self._search_rows
        self.search_table.setRowCount(len(rows))
        for i, r in enumerate(rows):
            values = (
                str(r.get("id", "")),
                str(r.get("downloads", "")),
                str(r.get("likes", "")),
                str(r.get("modified", "")),
                "gated" if r.get("gated") else "",
            )
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.search_table.setItem(i, col, item)
        self.search_table.resizeColumnsToContents()
        self.search_table.horizontalHeader().setStretchLastSection(True)

    def _on_model_search_event(self, ev: ModelSearchEvent) -> None:
        if ev.request_id != self._search_request_id:
            return
        self._search_rows = [dict(r) for r in ev.results]
        self._populate_search_table()
        self.discover_status.setText(ev.error if ev.error else f"{len(self._search_rows)} result(s)")

    def _on_model_info_event(self, ev: ModelInfoEvent) -> None:
        if ev.request_id != self._inspect_request_id:
            return
        quants = [dict(q) for q in ev.quants]
        chosen = ev.chosen or ""
        chosen_size = next(
            (q.get("size_bytes") for q in quants if str(q.get("name") or q.get("quant") or "") == chosen),
            None,
        )
        self.quant_combo.clear()
        auto_label = f"auto ({chosen}, {_fmt_size(chosen_size)} - fits this machine)" if chosen else "auto"
        self.quant_combo.addItem(auto_label, "auto")
        for q in quants:
            name = str(q.get("name") or q.get("quant") or "")
            self.quant_combo.addItem(f"{name}  {_fmt_size(q.get('size_bytes'))}".strip(), name)
        self.quant_combo.setCurrentIndex(0)
        self.mmproj_cb.setEnabled(bool(ev.mmproj))
        self.mmproj_cb.setChecked(False)
        self.detail_architecture.setText(ev.architecture or "-")
        self.detail_params.setText(_fmt_params(ev.params))
        self.detail_context.setText(str(ev.context_length) if ev.context_length else "-")
        if ev.error:
            self.discover_status.setText(ev.error)

    def _on_pull_clicked(self) -> None:
        idx = self.search_table.currentRow()
        if not (0 <= idx < len(self._search_rows)):
            self.discover_status.setText("select a model to pull")
            return
        repo_id = str(self._search_rows[idx].get("id", ""))
        quant = self.quant_combo.currentData() or "auto"
        include_mmproj = self.mmproj_cb.isChecked()
        try:
            self._actions.pull_model(repo_id, quant, include_mmproj=include_mmproj)
        except Exception:  # noqa: BLE001
            log.exception("actions.pull_model(%r) failed", repo_id)
            self.discover_status.setText("pull failed to start")
            return
        # review round two, 2026-09-27 (D2): track the in-flight pull's repo and disable Pull
        # immediately, so a second click can't reach the engine at all while this one runs.
        self._active_pull_repo = repo_id
        self.pull_btn.setEnabled(False)
        self.cancel_pull_btn.setEnabled(True)
        self.load_now_btn.hide()
        self.pull_progress.setValue(0)
        self.pull_label.setText("queued...")

    def _on_cancel_pull_clicked(self) -> None:
        try:
            self._actions.cancel_pull()
        except Exception:  # noqa: BLE001
            log.exception("actions.cancel_pull() failed")

    def _on_load_now_clicked(self) -> None:
        key = self._pull_done_key
        if not key:
            return
        try:
            self._actions.load_model(key)
        except Exception:  # noqa: BLE001
            log.exception("actions.load_model(%r) failed", key)
        self.load_now_btn.hide()

    def _on_model_pull_event(self, ev: ModelPullEvent) -> None:
        # review round two, 2026-09-27 (D2): a refusal notice for a click that landed while a
        # pull is already running ("another pull is still running") is not a status update for
        # ANY pull -- it must never touch progress, the label or Cancel. Ignore it outright.
        if ev.state == "error" and ev.detail.startswith("another pull"):
            return
        # ...and once a pull is tracked, events for any other repo (e.g. that same refused
        # second click, if the engine ever tags it with the repo it refused) are none of this
        # pull's business either.
        if self._active_pull_repo and ev.repo_id != self._active_pull_repo:
            return

        total = ev.total_bytes or 0
        done = ev.done_bytes or 0
        pct = int(done * 100 / total) if total else 0
        self.pull_progress.setValue(max(0, min(100, pct)))
        done_mb, total_mb = done / 1e6, total / 1e6
        detail = f" - {ev.detail}" if ev.detail else ""
        self.pull_label.setText(f"{ev.file}  {done_mb:.1f}/{total_mb:.1f} MB  {ev.state}{detail}".strip())
        self.cancel_pull_btn.setEnabled(ev.state in _PULL_ACTIVE_STATES)
        if ev.state in _PULL_ACTIVE_STATES:
            self._active_pull_repo = ev.repo_id
            self.pull_btn.setEnabled(False)
        elif ev.state in ("done", "error", "cancelled"):
            # a genuine terminal event for the pull we're tracking (or an unsolicited one that
            # arrived with no pull tracked at all): the operation is over, re-enable Pull.
            self._active_pull_repo = None
            self.pull_btn.setEnabled(True)
        if ev.state == "done":
            self._pull_done_key = ev.model_key
            self.load_now_btn.setVisible(bool(ev.model_key))
            self._refresh_library()
        else:
            self.load_now_btn.hide()

    # ================================================================================== Tuning
    def _build_tuning_tab(self) -> QWidget:
        w = QWidget(self)
        outer = QVBoxLayout(w)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(8)

        top_row = QHBoxLayout()
        top_row.addWidget(QLabel("Model", w))
        self.tuning_model_combo = QComboBox(w)
        self.tuning_model_combo.currentIndexChanged.connect(self._on_tuning_model_changed)
        top_row.addWidget(self.tuning_model_combo, 1)
        outer.addLayout(top_row)

        self.tuning_auto_cb = QCheckBox("Use auto-tuned profile for this model", w)
        self.tuning_auto_cb.toggled.connect(lambda checked: self._on_auto_tune_clicked() if checked else None)
        outer.addWidget(self.tuning_auto_cb)

        form = QFormLayout()
        self._tuning_widgets: dict[str, tuple[QCheckBox, QWidget]] = {}
        for f in self._TUNING_FIELDS:
            value_widget = make_field_widget(f, w)
            value_widget.setEnabled(False)
            override_cb = QCheckBox("override", w)
            override_cb.toggled.connect(value_widget.setEnabled)
            row = QWidget(w)
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.addWidget(override_cb)
            row_layout.addWidget(value_widget, 1)
            form.addRow(f.label, row)
            self._tuning_widgets[f.key] = (override_cb, value_widget)
        outer.addLayout(form)

        preset_row = QHBoxLayout()
        self.autotune_btn = QPushButton("Auto-tune for this machine", w)
        self.autotune_btn.clicked.connect(self._on_auto_tune_clicked)
        self.preset_speed_btn = QPushButton("Speed", w)
        self.preset_speed_btn.clicked.connect(self._apply_preset_speed)
        self.preset_quality_btn = QPushButton("Quality", w)
        self.preset_quality_btn.clicked.connect(self._apply_preset_quality)
        self.preset_low_memory_btn = QPushButton("Low memory", w)
        self.preset_low_memory_btn.clicked.connect(self._apply_preset_low_memory)
        self.use_global_defaults_btn = QPushButton("Use global defaults", w)
        self.use_global_defaults_btn.clicked.connect(self._use_global_defaults)
        for b in (self.autotune_btn, self.preset_speed_btn, self.preset_quality_btn,
                  self.preset_low_memory_btn, self.use_global_defaults_btn):
            preset_row.addWidget(b)
        preset_row.addStretch(1)
        outer.addLayout(preset_row)

        apply_row = QHBoxLayout()
        self.tuning_apply_btn = QPushButton("Apply", w)
        self.tuning_apply_btn.setObjectName("primary")
        self.tuning_apply_btn.clicked.connect(self._on_tuning_apply_clicked)
        apply_row.addWidget(self.tuning_apply_btn)
        apply_row.addStretch(1)
        outer.addLayout(apply_row)

        self.tuning_note = QLabel("", w)
        self.tuning_note.setWordWrap(True)
        outer.addWidget(self.tuning_note)
        outer.addStretch(1)
        return w

    def _reload_tuning_model_combo(self) -> None:
        combo = self.tuning_model_combo
        previous = combo.currentData()
        combo.blockSignals(True)
        combo.clear()
        default_index = 0
        for i, r in enumerate(self._library_rows):
            key = str(r.get("key", ""))
            combo.addItem(str(r.get("name") or key), key)
            if r.get("loaded"):
                default_index = i
        combo.blockSignals(False)
        resolved_key: str | None = None
        if combo.count():
            idx = combo.findData(previous) if previous else -1
            combo.setCurrentIndex(idx if idx >= 0 else default_index)
            resolved_key = combo.currentData()
        # review round two, 2026-09-27 (D1): this runs on EVERY ModelLibraryEvent (via
        # _refresh_library), including ones unrelated to the model being tuned (a pull for a
        # different repo finishing, a default change...). Reloading the profile form
        # unconditionally wiped whatever ticks/values the user had in progress. Only reload when
        # the resolved model actually changed; otherwise keep the form exactly as the user left it.
        if resolved_key != self._tuning_key:
            self._load_tuning_profile(resolved_key)

    def _on_tuning_model_changed(self, _index: int) -> None:
        self._load_tuning_profile(self.tuning_model_combo.currentData())

    def _tuning_default(self, key: str) -> Any:
        llm = self._cfg.llm
        if key == "top_k":
            return llm.extra_body.get("top_k", 20)
        if key == "min_p":
            return llm.extra_body.get("min_p", 0.0)
        if key == "extra_args":
            return " ".join(llm.llama_extra_args)
        return getattr(llm, key, "")

    # review round two, 2026-09-27 (D3): ModelProfile.extra_args is a list (default_factory=list,
    # so "no override" is `[]`, never `None` like every other profile field). Round-tripping it
    # through str()/QLineEdit as if it meant the same as `None` ticked the box for every
    # auto-generated profile (showing the literal text "[]") and saved the STRING "[]" back on
    # Apply / Auto-tune. These two helpers are the only place that needs to know that.
    @staticmethod
    def _profile_field_present(f: Field, value: object) -> bool:
        """Whether `value` (as read from a saved profile dict) means "this field is overridden"."""
        if f.key == "extra_args":
            return bool(value)  # [] (or missing) == no override, same as None for every other field
        return value is not None

    @staticmethod
    def _profile_field_display(f: Field, value: object) -> object:
        """`value` converted for `set_field_value` -- extra_args' list becomes the space-joined
        text its QLineEdit shows; every other field passes through unchanged."""
        if f.key == "extra_args" and isinstance(value, (list, tuple)):
            return " ".join(str(v) for v in value)
        return value

    def _profile_field_from_widget(self, f: Field) -> object:
        """The value to SAVE for `f`, read back from its widget -- extra_args' text is split into
        a list (shlex-style, so quoting works), never saved as a string."""
        cb, widget = self._tuning_widgets[f.key]
        value = get_field_value(widget, f)
        if f.key == "extra_args":
            return shlex.split(str(value))
        return value

    def _load_tuning_profile(self, key: str | None) -> None:
        self._tuning_key = key or None
        profile: dict[str, object] | None = None
        if key:
            try:
                profile = self._actions.model_profile(key)
            except Exception:  # noqa: BLE001
                log.exception("actions.model_profile(%r) failed", key)
                profile = None
        for f in self._TUNING_FIELDS:
            cb, widget = self._tuning_widgets[f.key]
            value = profile.get(f.key) if profile else None
            if profile is not None and self._profile_field_present(f, value):
                cb.setChecked(True)
                set_field_value(widget, f, self._profile_field_display(f, value))
            else:
                cb.setChecked(False)
                set_field_value(widget, f, self._tuning_default(f.key))
        self.tuning_note.setText(str(profile.get("note") or profile.get("generated_from") or "") if profile else "")

    def _on_auto_tune_clicked(self) -> None:
        key = self._tuning_key
        if not key:
            return
        try:
            profile = self._actions.auto_tune_model(key)
        except Exception:  # noqa: BLE001
            log.exception("actions.auto_tune_model(%r) failed", key)
            profile = None
        if not profile:
            return
        for f in self._TUNING_FIELDS:
            value = profile.get(f.key)
            if self._profile_field_present(f, value):
                cb, widget = self._tuning_widgets[f.key]
                cb.setChecked(True)
                set_field_value(widget, f, self._profile_field_display(f, value))
        self.tuning_note.setText(str(profile.get("note") or profile.get("generated_from") or ""))

    def _on_tuning_apply_clicked(self) -> None:
        key = self._tuning_key
        if not key:
            return
        values: dict[str, object] = {}
        for f in self._TUNING_FIELDS:
            cb, _widget = self._tuning_widgets[f.key]
            if cb.isChecked():
                values[f.key] = self._profile_field_from_widget(f)
        try:
            self._actions.set_model_profile(key, values or None)
        except Exception:  # noqa: BLE001
            log.exception("actions.set_model_profile(%r, ...) failed", key)

    def _set_override(self, key: str, value: object) -> None:
        f = next(f for f in self._TUNING_FIELDS if f.key == key)
        cb, widget = self._tuning_widgets[key]
        cb.setChecked(True)
        set_field_value(widget, f, value)

    def _apply_preset_speed(self) -> None:
        self._set_override("ctx_size", 4096)
        self._set_override("kv_cache_type", "q8_0")
        self._set_override("flash_attn", True)
        self._set_override("temperature", 0.3)

    def _apply_preset_quality(self) -> None:
        row = next((r for r in self._library_rows if str(r.get("key")) == self._tuning_key), None)
        context_length = int(row.get("context_length") or 0) if row else 0
        self._set_override("ctx_size", max(context_length, 16384))
        self._set_override("kv_cache_type", "f16")
        self._set_override("temperature", 0.6)

    def _apply_preset_low_memory(self) -> None:
        self._set_override("ctx_size", 4096)
        self._set_override("kv_cache_type", "q4_0")
        self._set_override("gpu_layers", max(0, self._cfg.llm.gpu_layers // 2))

    def _use_global_defaults(self) -> None:
        for cb, _widget in self._tuning_widgets.values():
            cb.setChecked(False)

    # =================================================================================== Server
    def _build_server_tab(self) -> QWidget:
        w = QWidget(self)
        outer = QVBoxLayout(w)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(8)

        # FieldForm builds the whole scroll area + form (labels with their "(restart)" suffix,
        # tool tips from Field.help) -- use that widget directly, exactly like SettingsDialog's
        # tabs; building a second form around its widgets left an orphan QScrollArea floating
        # over the Backend row (a review finding of 2026-09-27).
        self._server_form = FieldForm(self._SERVER_FIELDS, self._cfg, w)
        outer.addWidget(self._server_form.widget, 1)

        self.server_note = QLabel("", w)
        self.server_note.setWordWrap(True)
        outer.addWidget(self.server_note)

        btn_row = QHBoxLayout()
        self.server_restore_btn = QPushButton("Restore defaults", w)
        # A plain lambda (not the bound method) so Qt's automatic connection never hands the
        # button's "checked" bool into `skip=()` (PySide passes as many args as the slot accepts,
        # including optional ones -- see the fallback FieldForm.restore_defaults contract above).
        self.server_restore_btn.clicked.connect(lambda: self._server_form.restore_defaults())
        self.server_apply_btn = QPushButton("Apply", w)
        self.server_apply_btn.setObjectName("primary")
        self.server_apply_btn.clicked.connect(self._on_server_apply)
        btn_row.addWidget(self.server_restore_btn)
        btn_row.addStretch(1)
        btn_row.addWidget(self.server_apply_btn)
        outer.addLayout(btn_row)
        return w

    def _on_server_apply(self) -> None:
        changed = self._server_form.apply()
        restart = [f.label for f in self._SERVER_FIELDS if f.restart and f.path in changed]
        if restart:
            self.server_note.setText("Takes effect at the next launch: " + ", ".join(restart))
        elif changed:
            self.server_note.setText("Applied.")
        else:
            self.server_note.setText("")
        if changed:
            log.info("model server settings changed: %s", ", ".join(f"{k}={v!r}" for k, v in changed.items()))
            self.applied.emit(changed)

    # ================================================================================= Hardware
    def _build_hardware_tab(self) -> QWidget:
        w = QWidget(self)
        outer = QVBoxLayout(w)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(8)

        self.hw_summary = QLabel("", w)
        self.hw_summary.setWordWrap(True)
        outer.addWidget(self.hw_summary)

        self._hw_form_container = QWidget(w)
        self.hw_form = QFormLayout(self._hw_form_container)
        outer.addWidget(self._hw_form_container)

        self.hw_refresh_btn = QPushButton("Refresh", w)
        self.hw_refresh_btn.clicked.connect(lambda: self._refresh_hardware(refresh=True))
        outer.addWidget(self.hw_refresh_btn)
        outer.addStretch(1)
        return w

    def _refresh_hardware(self, refresh: bool = False) -> None:
        try:
            report = self._actions.hardware_report(refresh=refresh)
        except Exception:  # noqa: BLE001
            log.exception("actions.hardware_report(refresh=%r) failed", refresh)
            report = None
        self._populate_hardware(report or {})

    def _populate_hardware(self, report: dict[str, object]) -> None:
        while self.hw_form.rowCount():
            self.hw_form.removeRow(0)
        rows: list[tuple[str, object]] = [
            ("OS", report.get("os")),
            ("CPU", report.get("cpu_name")),
            ("Physical cores", report.get("physical_cores")),
            ("RAM", _fmt_mb(report.get("ram_total_mb"))),
            ("GPU vendor", report.get("vendor")),
        ]
        if report.get("gpus"):
            rows.append(("GPUs", _fmt_gpus(report.get("gpus"))))
        rows += [
            ("VRAM total", _fmt_mb(report.get("vram_total_mb"))),
            ("VRAM free", _fmt_mb(report.get("vram_free_mb"))),
            ("Budget", _fmt_mb(report.get("budget_mb"))),
            ("NVIDIA driver", report.get("nvidia_driver")),
            ("Library dir", report.get("library_dir")),
            ("Library free", _fmt_mb(report.get("library_free_mb"))),
            ("Runtime", _fmt_runtime(report.get("runtime"))),
            ("Policy", report.get("recommended_policy")),
            ("Generated", report.get("generated_at")),
        ]
        if report.get("notes"):
            rows.append(("Notes", "\n".join(str(n) for n in report["notes"])))
        for label, value in rows:
            if value in (None, ""):
                continue
            val_label = QLabel(str(value), self._hw_form_container)
            val_label.setWordWrap(True)
            self.hw_form.addRow(label, val_label)
        self.hw_summary.setText(_hardware_summary(report))

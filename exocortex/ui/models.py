"""Models dialog: the app's own GGUF library, Hugging Face discovery, per-model tuning, the
global LLM server settings, the context library and a read-only hardware report.

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

2026-10-06 (context library): a sixth tab, "Context", sits between Server and Hardware. It lists
the reference documents the user uploaded (``actions.context_documents()``) with buttons to add
files / pasted text, remove, enable or disable, switch a document between ``inject`` (its text in
every prompt) and ``reference`` (digest + on-demand passages), re-generate a digest and open the
library folder; a read-only panel shows how the last answer's prompt filled the server's context
window (``actions.context_usage()`` on open, every ``ContextUsageEvent`` afterwards); and a
``FieldForm`` over the whole ``[context]`` section applies like the Server tab does (``applied``
carries only the changed ``context.*`` keys). ``ContextLibraryEvent`` re-reads the document list.

2026-10-07 (capabilities): a seventh tab, "Capabilities", sits between Context and Hardware. It is
where the owner decides what the model may do on its own to gather context: five switches (read
files, edit files, execute commands, fetch web content, MCP servers), the scope of each (readable /
editable folders and their exclusions, the allowed commands, the allowed / blocked web domains, the
MCP server list), the scalar limits (a ``FieldForm`` over ``[capabilities]``) and the project maps
(``Pre-index scoped directories...``, load / re-index / delete). ``Apply`` writes the switches, the
lists and the form into ``cfg.capabilities`` and emits ``applied`` with the changed
``capabilities.*`` keys; ``CapabilityMapEvent`` re-reads the map list and ``McpToolsEvent`` fills
the MCP table's Tools column. Nothing here touches the filesystem or the network: the engine does.
"""
from __future__ import annotations

import copy
import logging
import math
import os
import re
import shlex
from pathlib import Path
from typing import TYPE_CHECKING, Any

from PySide6.QtCore import QEvent, QObject, QPoint, QRect, QSize, QTimer, QUrl, Qt, Signal
from PySide6.QtGui import QColor, QDesktopServices, QFontDatabase
from PySide6.QtWidgets import (
    QAbstractItemView,
    QAbstractSpinBox,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLayout,
    QLayoutItem,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
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
    CapabilityMapEvent,
    ComponentStatusEvent,
    ContextLibraryEvent,
    ContextUsageEvent,
    McpToolsEvent,
    ModelInfoEvent,
    ModelLibraryEvent,
    ModelPullEvent,
    ModelSearchEvent,
)
from exocortex.config import SAFE_COMMANDS, Config
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
QGroupBox {{ border: 1px solid {COLORS['border']}; border-radius: 6px; margin-top: 12px; padding: 10px 8px 8px 8px; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px; color: {COLORS['text_dim']}; }}
QListWidget {{ background: {COLORS['panel']}; border: 1px solid {COLORS['border']}; border-radius: 4px; }}
QListWidget::item:selected {{ background: {COLORS['accent_dim']}; color: {COLORS['text']}; }}
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


def _as_int(value: object) -> int | None:
    """``int(value)`` for a real number, ``None`` for anything else (None, "", junk, bool)."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(float(value))  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        return None


def _usage_dict(usage: object) -> dict[str, object]:
    """Normalise whatever the usage panel is handed (a ``ContextUsageEvent``, the dict
    ``actions.context_usage()`` returns, ``None`` before any job) into a plain dict.

    An event's ``documents`` is the number of documents that contributed to THAT prompt, while the
    snapshot's ``documents`` is how many are enabled (its contributing count is ``documents_used``);
    the event is therefore normalised to the snapshot's ``documents_used`` spelling.
    """
    if isinstance(usage, dict):
        return dict(usage)
    if isinstance(usage, ContextUsageEvent):
        out: dict[str, object] = {
            name: getattr(usage, name)
            for name in ("job_id", "kind", "slot_tokens", "budget_tokens", "used_tokens",
                         "transcript_tokens", "reference_tokens", "lookup_tokens", "earlier_tokens",
                         "screen_tokens", "system_tokens", "prompt_tokens", "cached_tokens",
                         "trimmed", "anchor_moved", "chars_per_token")
        }
        out["documents_used"] = usage.documents
        out["has_usage"] = True
        return out
    return {}


def _format_usage(usage: object) -> str:
    """The one-paragraph "how full was the context window" text of the Context tab.

    Tolerant by design: ``None`` / ``{}`` (no answer yet) and dicts missing any key still render
    something sensible instead of raising -- the panel is shown on open, before any job ran.
    """
    d = _usage_dict(usage)
    sep = " \u00b7 "
    parts: list[str] = []
    slot = _as_int(d.get("slot_tokens"))
    parts.append(f"Server context per request: {slot} tokens" if slot else "Server context per request: unknown")

    kind = d.get("kind")
    used = _as_int(d.get("used_tokens"))
    answered = bool(d.get("has_usage", bool(kind) or bool(used))) and bool(kind or used)
    if not answered:
        parts.append("no answer yet")
        docs = _as_int(d.get("documents"))
        if docs is not None:
            parts.append(f"{docs} enabled document{'s' if docs != 1 else ''}")
    else:
        head = f"last answer ({kind})" if kind else "last answer"
        budget = _as_int(d.get("budget_tokens"))
        if used is not None and budget:
            head += f": used {used} of {budget} tokens"
        elif used is not None:
            head += f": used {used} tokens"
        blocks = [
            f"{label} {_as_int(d.get(key)) or 0}"
            for label, key in (("transcript", "transcript_tokens"), ("reference", "reference_tokens"),
                               ("lookup", "lookup_tokens"), ("earlier", "earlier_tokens"),
                               ("screen", "screen_tokens"), ("system", "system_tokens"))
        ]
        parts.append(head + ": " + sep.join(blocks))
        prompt = _as_int(d.get("prompt_tokens"))
        if prompt is not None:
            cached = _as_int(d.get("cached_tokens"))
            parts.append(f"server counted {prompt}" + (f" ({cached} cached)" if cached is not None else ""))
        parts.append("trimmed" if d.get("trimmed") else "ok")
        if d.get("anchor_moved"):
            parts.append("window advanced")
        docs = _as_int(d.get("documents_used"))
        if docs is not None:
            parts.append(f"{docs} document{'s' if docs != 1 else ''}")
    try:
        cpt = float(d.get("chars_per_token") or 0.0)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        cpt = 0.0
    if cpt > 0:
        parts.append(f"{cpt:.2f} chars/token")
    return sep.join(parts)


def _status_color(state: str) -> str:
    if state in ("ready", "done"):
        return COLORS["ok"]
    if state in ("error", "cancelled"):
        return COLORS["err"]
    if state in ("loading", "busy", "queued", "downloading", "verifying"):
        return COLORS["warn"]
    return COLORS["text_dim"]


# ---------------------------------------------------------------------------------------------
# Capabilities tab helpers (2026-10-07)
# ---------------------------------------------------------------------------------------------
#: A valid MCP server name: the same shape the engine's config reader accepts (the runner names a
#: server's tools ``mcp:<server>:<tool>`` and splits on the first two colons).
_MCP_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")

#: ``token=`` / ``key=`` style query values of an MCP url, hidden in the table (the same shape the
#: engine redacts in ``session.json``): the table is on screen, possibly shared.
_SECRET_QUERY_RE = re.compile(r"(?i)([?&][^=&#]*(?:token|key)[^=&#]*=)[^&#]*")


def _redact_url(url: str) -> str:
    """``url`` with every ``token=`` / ``key=`` query value replaced by ``***``."""
    return _SECRET_QUERY_RE.sub(r"\1***", url)


def _split_lines(text: str) -> list[str]:
    """The non-blank lines of a plain-text box, whitespace stripped (a list compares as a list)."""
    return [line.strip() for line in text.splitlines() if line.strip()]


def _folder_name(path: str) -> str:
    """The last component of a folder path ("C:/work/app" -> "app"); the drive letter for a drive root."""
    p = Path(path)
    return p.name or p.drive.rstrip(":") or "project"


def _mcp_kind(spec: dict[str, Any]) -> str:
    """The transport of an MCP entry, read exactly as the runner reads it (``McpServer.__init__``):
    an explicit ``transport`` (stripped, lower-cased), else "http" for an entry with a ``url`` and
    no ``command``, else "stdio". Kept in step with ``exocortex.mcp`` by a test."""
    return str(spec.get("transport") or ("http" if spec.get("url") and not spec.get("command") else "stdio")
               ).strip().lower()


def _mcp_enabled(spec: dict[str, Any]) -> bool:
    """Whether the runner would use an MCP entry (``capabilities._spec_enabled``): ``enabled`` absent,
    or a bool / int that is true. Anything else ("false" the string, a float) counts as disabled."""
    val = spec.get("enabled", True)
    return bool(val) if isinstance(val, (bool, int)) else False


#: The status line names at most this many tools; the built-in ones always come first and are never cut.
_STATUS_MAX_TOOLS = 8


def _format_capability_status(status: object) -> tuple[str, str]:
    """The Capabilities tab's one-paragraph status line from ``actions.capability_status()``'s dict:
    the tools the ask box would get right now (at most ``_STATUS_MAX_TOOLS`` names, then "+N more":
    the built-in tools first and in full, a big MCP server's ``mcp:`` tools fill what is left),
    the MCP servers' states, the active map and the engine's own notes (one per line). Returns
    ``(text, state)``; ``None`` / ``{}`` -> ``("", "")``."""
    if not isinstance(status, dict) or not status:
        return "", ""
    tools_raw = status.get("tools")
    tools = [str(t) for t in tools_raw] if isinstance(tools_raw, (list, tuple)) else []
    builtin = [t for t in tools if not t.startswith("mcp:")]
    external = [t for t in tools if t.startswith("mcp:")]
    room = max(0, _STATUS_MAX_TOOLS - len(builtin))
    shown = builtin + external[:room]
    more = len(external) - room
    listing = ", ".join(shown) + (f" +{more} more" if more > 0 else "")
    parts = ["Offered to the ask box: " + listing if tools else "No tools are offered right now"]
    mcp = status.get("mcp")
    if isinstance(mcp, dict) and mcp:
        parts.append("MCP: " + ", ".join(f"{name} {state}" for name, state in mcp.items()))
    active = str(status.get("active_map") or "")
    if active:
        parts.append(f"Active map: {active}")
    lines = [" · ".join(parts)]
    notes = status.get("notes")
    if isinstance(notes, (list, tuple)):
        lines += [str(n) for n in notes if str(n).strip()]
    return "\n".join(lines), ("" if tools else "loading")


class _FlowLayout(QLayout):
    """Left-to-right layout that wraps onto a new line when the width runs out (Qt's "Flow Layout"
    example): the Capabilities tab's row of five switches stays one row at the window's usual width
    and wraps instead of being clipped at its 640 px minimum."""

    def __init__(self, parent: QWidget | None = None, *, h_spacing: int = 14, v_spacing: int = 4) -> None:
        super().__init__(parent)
        self._items: list[QLayoutItem] = []
        self._h = h_spacing
        self._v = v_spacing

    def addItem(self, item: QLayoutItem) -> None:  # noqa: N802 - Qt override
        self._items.append(item)

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index: int) -> QLayoutItem | None:  # noqa: N802 - Qt override
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index: int) -> QLayoutItem | None:  # noqa: N802 - Qt override
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def expandingDirections(self) -> Qt.Orientation:  # noqa: N802 - Qt override
        return Qt.Orientation(0)

    def hasHeightForWidth(self) -> bool:  # noqa: N802 - Qt override
        return True

    def heightForWidth(self, width: int) -> int:  # noqa: N802 - Qt override
        return self._arrange(QRect(0, 0, width, 0), apply=False)

    def setGeometry(self, rect: QRect) -> None:  # noqa: N802 - Qt override
        super().setGeometry(rect)
        self._arrange(rect, apply=True)

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt override
        return self.minimumSize()

    def minimumSize(self) -> QSize:  # noqa: N802 - Qt override
        size = QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        margins = self.contentsMargins()
        return size + QSize(margins.left() + margins.right(), margins.top() + margins.bottom())

    def _arrange(self, rect: QRect, *, apply: bool) -> int:
        """Lay the items out inside ``rect`` (or just measure); returns the height used."""
        margins = self.contentsMargins()
        area = rect.adjusted(margins.left(), margins.top(), -margins.right(), -margins.bottom())
        x, y, line_h = area.x(), area.y(), 0
        for item in self._items:
            hint = item.sizeHint()
            if line_h and x + hint.width() > area.right() + 1:
                x, y, line_h = area.x(), y + line_h + self._v, 0
            if apply:
                item.setGeometry(QRect(QPoint(x, y), hint))
            x += hint.width() + self._h
            line_h = max(line_h, hint.height())
        return y + line_h - rect.y() + margins.bottom()


class _WheelGuard(QObject):
    """Event filter for the Capabilities tab's spin boxes and combos: a wheel turn over one that
    does not have the keyboard focus scrolls the page instead of silently changing the value (the
    tab is a long scroll area full of them)."""

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:  # noqa: N802 - Qt override
        if event.type() == QEvent.Type.Wheel and isinstance(obj, QWidget) and not obj.hasFocus():
            event.ignore()
            return True
        return False


_PULL_ACTIVE_STATES = ("queued", "downloading", "verifying")


class ModelsDialog(QDialog):
    """Tabbed model manager: Library / Discover / Tuning / Server / Context / Capabilities / Hardware.

    ``applied`` carries Server-tab changes as ``{"section.key": new_value}``, the same shape as
    ``SettingsDialog.applied``; the overlay routes it to ``actions.apply_settings`` exactly like
    the settings dialog does. The Context and Capabilities tabs emit it the same way (only their
    own ``context.*`` / ``capabilities.*`` keys).
    """

    TABS: tuple[str, ...] = ("Library", "Discover", "Tuning", "Server", "Context", "Capabilities", "Hardware")
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

    #: The ``[context]`` section (Context tab; 2026-10-06). One ``FieldForm`` over ``cfg.context``;
    #: the help texts are the comments of ``exocortex.config.ContextConfig``, condensed.
    _CONTEXT_FIELDS: tuple[Field, ...] = (
        Field("context", "dir", "Library folder (empty = <models>/context)", "text", restart=True,
              help="Where uploaded documents and their manifest live. Empty = the context folder "
                   "next to the model library; documents persist across sessions."),
        Field("context", "default_mode", "Mode of a new document", "choice", choices=("inject", "reference"),
              help='"reference" = digest + passages pulled in only when the question matches; '
                   '"inject" = the whole text in every prompt. Each document can be switched afterwards.'),
        Field("context", "include_in", "Documents reach", "choice", choices=("all", "questions", "none"),
              help='"all" = every answer, solve, code, text, suggestion and completion; "questions" = '
                   'spoken / typed / on-screen questions only; "none" = keep the library, inject nothing.'),
        Field("context", "reference_tokens", "Reference block (tokens)", "int", 0, 8000, 50,
              help="Budget of the stable <reference> block: injected documents in full plus one "
                   "catalogue line per reference-mode document. Trimmed first when the server's context is short."),
        Field("context", "lookup_tokens", "Lookup block (tokens)", "int", 0, 4000, 50,
              help="Budget of the volatile <lookup> block: passages pulled from reference-mode "
                   "documents for this one question. Trimmed first when the server's context is short."),
        Field("context", "lookup_top_k", "Passages per answer", "int", 0, 10, 1,
              help="At most this many passages are pulled in per answer."),
        Field("context", "lookup_min_score", "Passage match threshold", "float", 0.0, 20.0, 0.5,
              help="BM25 score a passage needs to be pulled in at all; higher = only strong matches."),
        Field("context", "chunk_chars", "Passage size (chars)", "int", 200, 6000, 100,
              help="A document is indexed in passages of about this size (split on headings / paragraphs)."),
        Field("context", "max_doc_chars", "Longest document (chars)", "int", 1000, 2000000, 1000,
              help="A longer document is cut here when it is added (with a warning in the log)."),
        Field("context", "digest", "Write a digest for reference documents", "bool",
              help="Pre-process each reference-mode document with the model into a 2-4 line digest "
                   "(what it is, what it covers) on the notes lane. Off: the catalogue shows the file "
                   "name and its first heading only."),
        Field("context", "digest_tokens", "Digest length (tokens)", "int", 32, 400, 8),
        Field("context", "budget_mode", "Prompt sizing", "choice", choices=("auto", "fixed"),
              help='"auto" = fit every prompt into the server\'s real per-request context, trimming '
                   'lookup, reference, summary, screen, then the oldest transcript; "fixed" = the '
                   'no fitting: the transcript gets [llm] transcript tokens as before (the window '
                   'still anchors, the documents still apply).'),
        Field("context", "reserve_tokens", "Safety margin (tokens)", "int", 0, 2048, 32,
              help="Kept free under the server's context."),
        Field("context", "lane_share", "Notes-lane share of the server context", "float", 0.1, 0.6, 0.05,
              help="llama-server's parallel slots share ONE KV pool, so an answer and a notes-lane job "
                   "(notes, summary, digest) running together must fit it together. This share is "
                   "the notes lane's; answers get the rest. Ignored with one slot."),
        Field("context", "transcript_min_tokens", "Newest speech keeps at least (tokens)", "int", 0, 4000, 50,
              help="The newest speech always keeps at least this much, whatever else has to go."),
        Field("context", "trim_step", "Window advance step", "float", 0.0, 0.9, 0.05,
              help="When the transcript window overflows, drop at least this share of it at once "
                   "(whole lines, oldest first) instead of sliding one line at a time, so the "
                   "prompt prefix stays identical and llama.cpp's cache keeps hitting. 0 = slide."),
        Field("context", "compaction", "Earlier transcript", "choice", choices=("summary", "off"),
              help='"summary" = the model folds what fell out of the window into a short running '
                   'summary (one small notes-lane job per trim); "off" = it is simply gone.'),
        Field("context", "summary_tokens", "Summary length (tokens)", "int", 50, 1000, 10),
        Field("context", "chars_per_token", "Characters per token", "float", 2.0, 6.0, 0.1,
              help="Used for every budget estimate until the server's own prompt-token counts have "
                   "calibrated it; 3.6 is conservative for prose with timestamps and code mixed in."),
        Field("context", "auto_calibrate", "Calibrate from the server's counts", "bool",
              help="Off pins characters per token to the value above."),
    )

    #: Document-table columns of the Context tab.
    _CONTEXT_COLUMNS: tuple[str, ...] = ("Name", "Kind", "Tokens", "Mode", "On", "Digest")

    #: File-picker filter of "Add files..." (the formats ``exocortex.context.extract_text`` reads).
    _CONTEXT_FILE_FILTER: str = (
        "Documents (*.md *.markdown *.txt *.rst *.pdf *.docx *.pptx *.html *.htm *.log *.csv *.json "
        "*.toml *.yaml *.yml *.ini *.cfg *.py *.js *.ts *.tsx *.jsx *.java *.go *.rs *.c *.cc *.cpp "
        "*.h *.hpp *.cs *.sql *.sh *.ps1 *.bat *.rb *.php *.kt *.swift *.m);;All files (*)"
    )

    #: The scalar part of ``[capabilities]`` (Capabilities tab; 2026-10-07). One ``FieldForm`` over
    #: ``cfg.capabilities``; the five switches, the folder / command / domain lists and the MCP
    #: server table are custom widgets (``_CAPABILITY_LIST_KEYS`` and ``cap_*_cb`` below) that
    #: Apply treats exactly like form fields. The ranges sit inside the ones a map's settings are
    #: validated against (``exocortex.capabilities.apply_snapshot``), so a value the tab can write
    #: is a value a map can carry.
    _CAPABILITY_FIELDS: tuple[Field, ...] = (
        Field("capabilities", "apply_to", "Tools reach", "choice", choices=("questions", "ask", "all"),
              help='"questions" = the other speaker (auto / manual), screen questions and the ask box '
                   '(read tools for the first three, everything for the ask box); "ask" = the ask box '
                   'only; "all" = also suggestions, solves, code and text (read tools). Never inline '
                   'completions or explanations.'),
        Field("capabilities", "max_tool_calls", "Ask box: tool calls per answer", "int", 0, 20, 1,
              help="The ask box carries text you typed, so a longer wait is acceptable."),
        Field("capabilities", "tool_result_chars", "Ask box: one result is cut at (chars)", "int", 200, 100000, 500),
        Field("capabilities", "tool_time_s", "Ask box: time for the whole loop (s)", "float", 1.0, 600.0, 5.0,
              help="Tools and model hops together, per answer; after it the model must answer."),
        Field("capabilities", "live_max_tool_calls", "Live answers: tool calls per answer", "int", 0, 20, 1,
              help="Auto / manual / screen answers, suggestions, solves, code and text: every hop "
                   "delays an answer someone is waiting for."),
        Field("capabilities", "live_tool_result_chars", "Live answers: one result is cut at (chars)", "int",
              200, 100000, 250),
        Field("capabilities", "live_tool_time_s", "Live answers: time for the whole loop (s)", "float", 1.0, 600.0, 1.0),
        Field("capabilities", "read_max_file_chars", "Longest file read at once (chars)", "int", 200, 1000000, 1000,
              help="A longer file is cut here (the model can ask for a line range)."),
        Field("capabilities", "edit_backup", "Back up a file before editing it", "bool",
              help="The original goes to <session>/edits/ before every change."),
        Field("capabilities", "allow_all_commands", "All commands -- runs arbitrary code: lifts every other scope",
              "bool", help="Lifts the allowed-command list, the chaining filter, the path check and the git "
                           "rewrite. Timeouts and output caps stay. A map never turns this on."),
        Field("capabilities", "command_timeout_s", "Command timeout (s)", "float", 1.0, 600.0, 1.0),
        Field("capabilities", "command_output_chars", "Command output kept (chars)", "int", 200, 100000, 500),
        Field("capabilities", "command_cwd", "Command working folder (empty = the first readable folder)", "text",
              help="Must sit inside a readable folder."),
        Field("capabilities", "web_allow_local", "Allow local / private network addresses", "bool",
              help="Off: loopback, private, link-local and CGNAT addresses are refused. A map never turns this on."),
        Field("capabilities", "web_max_fetches", "Web fetches + searches per answer", "int", 0, 10, 1),
        Field("capabilities", "web_page_chars", "Web page text kept (chars)", "int", 200, 100000, 500),
        Field("capabilities", "web_timeout_s", "Web timeout (s)", "float", 1.0, 120.0, 1.0),
        Field("capabilities", "web_search_url", "Web search URL ({query} is replaced)", "text",
              help="GET request; DuckDuckGo's HTML results page by default."),
        Field("capabilities", "web_search_results", "Web search results", "int", 1, 20, 1),
        Field("capabilities", "mcp_timeout_s", "MCP call timeout (s)", "float", 1.0, 300.0, 1.0),
        Field("capabilities", "maps_dir", "Maps folder (empty = <models>/maps)", "text",
              help="Where the project maps are saved; they persist across sessions. Applies at once; "
                   "maps saved in the old folder are not moved."),
        Field("capabilities", "map_max_files", "Map: most files listed", "int", 100, 50000, 500,
              help="A bigger tree is cut here (the rest is counted, not listed)."),
        Field("capabilities", "map_tokens", "Map: size in the prompt (tokens)", "int", 100, 4000, 50,
              help="Budget of the <map> block every tool-enabled prompt carries."),
    )

    #: The ``[capabilities]`` list values the tab edits through a list / text box, by config key.
    _CAPABILITY_LIST_KEYS: tuple[str, ...] = (
        "read_roots", "read_exclude", "edit_roots", "edit_exclude", "allowed_commands",
        "web_allowed_domains", "web_blocked_domains",
    )

    #: Label of each switch, in the order of the switch row (also the Load confirmation's wording).
    _CAPABILITY_SWITCHES: tuple[tuple[str, str], ...] = (
        ("read_files", "Read files"),
        ("edit_files", "Edit files"),
        ("execute_commands", "Execute commands"),
        ("fetch_web", "Fetch web content"),
        ("mcp_servers", "MCP servers"),
    )

    #: How often (ms) the status line re-reads the MCP states while the Capabilities tab is visible.
    _CAP_POLL_MS = 2000

    _MCP_COLUMNS: tuple[str, ...] = ("Name", "Transport", "Command or URL", "Enabled", "Tools")
    _MAP_COLUMNS: tuple[str, ...] = ("Name", "Roots", "Files", "Indexed", "Active")

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
        self._context_rows: list[dict[str, object]] = []  # 2026-10-06: the Context tab's documents
        # 2026-10-07 (capabilities): the Capabilities tab's working copies and request bookkeeping.
        self._mcp_specs: list[dict[str, Any]] = []        # the MCP table's rows; written to cfg on Apply only
        self._mcp_tools_text: dict[str, str] = {}         # server name -> the Tools column's text
        self._mcp_requests: dict[str, int] = {}           # server name -> the id of its latest Test
        self._mcp_request_id = 0                          # the counter handed to mcp_server_tools()
        self._map_rows: list[dict[str, object]] = []      # actions.capability_maps(), as shown
        self._cap_status_state: str | None = None         # the status line's colour state as last drawn
        # the status line's MCP states change on the engine's own threads (a connect pass ends, a
        # server dies) without any event: while the Capabilities tab is on screen, re-read them
        self._cap_poll_timer = QTimer(self)
        self._cap_poll_timer.setInterval(self._CAP_POLL_MS)
        self._cap_poll_timer.timeout.connect(self._poll_capability_status)
        self._wheel_guard = _WheelGuard(self)

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
        self.tabs.addTab(self._build_context_tab(), "Context")
        self.tabs.addTab(self._build_capabilities_tab(), "Capabilities")
        self.tabs.addTab(self._build_hardware_tab(), "Hardware")
        self.tabs.currentChanged.connect(lambda _i: self._sync_capability_poll())

        self.load()

    # ---------------------------------------------------------------------------------- public
    def load(self) -> None:
        """Re-read everything from ``actions`` / ``cfg`` (called on every open)."""
        self._refresh_library()
        self._refresh_hardware()
        self._server_form.load()
        self.server_note.setText("")
        self._context_form.load()
        self.context_note.setText("")
        self.context_status.setText("")
        self._refresh_context()
        self._render_usage(self._fetch_context_usage())
        self._load_capabilities()

    def show_tab(self, name: str) -> None:
        if name in self.TABS:
            self.tabs.setCurrentIndex(self.TABS.index(name))

    def showEvent(self, event: Any) -> None:  # noqa: N802 - Qt override
        super().showEvent(event)
        self._sync_capability_poll()

    def hideEvent(self, event: Any) -> None:  # noqa: N802 - Qt override
        super().hideEvent(event)
        self._sync_capability_poll()

    def on_event(self, ev: object) -> None:
        """Forwarded by the overlay: the five model events, ComponentStatusEvent("llm"), (2026-10-06)
        the two context events and (2026-10-07) the two capability events."""
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
            elif isinstance(ev, ContextLibraryEvent):
                self._on_context_library_event(ev)
            elif isinstance(ev, ContextUsageEvent):
                self._render_usage(ev, snapshot=self._fetch_context_usage())
            elif isinstance(ev, CapabilityMapEvent):
                self._on_capability_map_event(ev)
            elif isinstance(ev, McpToolsEvent):
                self._on_mcp_tools_event(ev)
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

    # ================================================================================== Context
    def _build_context_tab(self) -> QWidget:
        """Documents table, action buttons, usage panel and the ``[context]`` form (2026-10-06)."""
        w = QWidget(self)
        outer = QVBoxLayout(w)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(8)

        self.context_table = QTableWidget(0, len(self._CONTEXT_COLUMNS), w)
        self.context_table.setHorizontalHeaderLabels(list(self._CONTEXT_COLUMNS))
        self.context_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.context_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.context_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.context_table.setMinimumHeight(110)
        self.context_table.itemSelectionChanged.connect(self._update_context_buttons)
        outer.addWidget(self.context_table, 3)

        btn_row = QHBoxLayout()
        self.context_add_files_btn = QPushButton("Add files\u2026", w)
        self.context_add_files_btn.clicked.connect(self._on_context_add_files)
        self.context_add_text_btn = QPushButton("Add text\u2026", w)
        self.context_add_text_btn.clicked.connect(self._on_context_add_text)
        self.context_remove_btn = QPushButton("Remove", w)
        self.context_remove_btn.clicked.connect(self._on_context_remove)
        self.context_toggle_btn = QPushButton("Enable/Disable", w)
        self.context_toggle_btn.clicked.connect(self._on_context_toggle)
        self.context_mode_btn = QPushButton("Mode: inject/reference", w)
        self.context_mode_btn.clicked.connect(self._on_context_mode)
        self.context_redigest_btn = QPushButton("Re-digest", w)
        self.context_redigest_btn.clicked.connect(self._on_context_redigest)
        self.context_open_folder_btn = QPushButton("Open folder", w)
        self.context_open_folder_btn.clicked.connect(self._on_context_open_folder)
        for b in (self.context_add_files_btn, self.context_add_text_btn, self.context_remove_btn,
                  self.context_toggle_btn, self.context_mode_btn, self.context_redigest_btn,
                  self.context_open_folder_btn):
            btn_row.addWidget(b)
        btn_row.addStretch(1)
        outer.addLayout(btn_row)

        self.context_status = QLabel("", w)
        self.context_status.setWordWrap(True)
        outer.addWidget(self.context_status)

        self.context_usage_label = QLabel("", w)
        self.context_usage_label.setWordWrap(True)
        self.context_usage_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.context_usage_label.setStyleSheet(f"color: {COLORS['text_dim']};")
        outer.addWidget(self.context_usage_label)

        self._context_form = FieldForm(self._CONTEXT_FIELDS, self._cfg, w)
        outer.addWidget(self._context_form.widget, 4)

        self.context_note = QLabel("", w)
        self.context_note.setWordWrap(True)
        outer.addWidget(self.context_note)

        apply_row = QHBoxLayout()
        self.context_restore_btn = QPushButton("Restore defaults", w)
        # A plain lambda for the same reason as the Server tab's (see _build_server_tab).
        self.context_restore_btn.clicked.connect(lambda: self._context_form.restore_defaults())
        self.context_apply_btn = QPushButton("Apply", w)
        self.context_apply_btn.setObjectName("primary")
        self.context_apply_btn.clicked.connect(self._on_context_apply)
        apply_row.addWidget(self.context_restore_btn)
        apply_row.addStretch(1)
        apply_row.addWidget(self.context_apply_btn)
        outer.addLayout(apply_row)
        return w

    # ---- documents table
    def _refresh_context(self) -> None:
        """Re-read ``actions.context_documents()`` into the table, keeping the selected row by id."""
        selected = self._selected_context_row()
        selected_id = selected.get("id") if selected else None
        try:
            result = self._actions.context_documents()
        except Exception:  # noqa: BLE001
            log.exception("actions.context_documents() failed")
            self._set_context_status("failed to read the context library", "error")
            result = None
        rows = [dict(r) for r in result if isinstance(r, dict)] if result else []
        self._context_rows = rows

        self.context_table.setRowCount(len(rows))
        for i, r in enumerate(rows):
            self._set_context_row(i, r)
        self.context_table.resizeColumnsToContents()
        self.context_table.horizontalHeader().setStretchLastSection(True)
        if selected_id is not None:
            for i, r in enumerate(rows):
                if r.get("id") == selected_id:
                    self.context_table.setCurrentCell(i, 0)
                    break
        self._update_context_buttons()

    def _set_context_row(self, i: int, r: dict[str, object]) -> None:
        """One document row. "On" is plain text ("on" / "off": the toggle is the button); the
        digest text itself is the row's tool tip, its state ("pending" / "ready" / ...) the column."""
        tokens = _as_int(r.get("tokens"))
        if tokens is None:
            chars = _as_int(r.get("chars")) or 0
            cpt = self._cfg.context.chars_per_token or 3.6
            tokens = math.ceil(chars / cpt) if chars > 0 else 0
        enabled = bool(r.get("enabled", True))
        values = (
            str(r.get("name") or r.get("id") or ""),
            str(r.get("kind") or ""),
            str(tokens),
            str(r.get("mode") or ""),
            "on" if enabled else "off",
            str(r.get("digest_state") or ""),
        )
        digest = str(r.get("digest") or "")
        for col, value in enumerate(values):
            item = QTableWidgetItem(value)
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            if digest:
                item.setToolTip(digest)
            if not enabled:
                item.setForeground(QColor(COLORS["text_dim"]))
            elif col == len(values) - 1 and value == "failed":
                item.setForeground(QColor(COLORS["err"]))
            self.context_table.setItem(i, col, item)

    def _selected_context_row(self) -> dict[str, object] | None:
        idx = self.context_table.currentRow()
        if 0 <= idx < len(self._context_rows):
            return self._context_rows[idx]
        return None

    def _update_context_buttons(self) -> None:
        row = self._selected_context_row()
        has_row = row is not None
        enabled = bool(row.get("enabled", True)) if row else True
        mode = str(row.get("mode") or "") if row else ""
        self.context_remove_btn.setEnabled(has_row)
        self.context_toggle_btn.setEnabled(has_row)
        self.context_toggle_btn.setText("Disable" if enabled else "Enable")
        self.context_mode_btn.setEnabled(has_row)
        self.context_mode_btn.setText(
            "Mode: switch to inject" if mode == "reference"
            else "Mode: switch to reference" if mode == "inject"
            else "Mode: inject/reference"
        )
        self.context_redigest_btn.setEnabled(has_row and mode == "reference")

    def _set_context_status(self, text: str, state: str = "") -> None:
        self.context_status.setText(text)
        self.context_status.setStyleSheet(f"color: {_status_color(state)};")

    def _on_context_library_event(self, ev: ContextLibraryEvent) -> None:
        """A document changed (added / removed / toggled / digest ready): re-read the list. A
        failure reason ("error: ...") from the engine is shown under the buttons."""
        self._refresh_context()
        reason = ev.reason or ""
        if reason.startswith("error"):
            self._set_context_status(reason, "error")
        elif reason.startswith("digest failed"):
            doc = next((d for d in self._context_rows if str(d.get("id")) == ev.doc_id), None)
            name = str(doc.get("name") or ev.doc_id) if doc else ev.doc_id
            self._set_context_status(f"digest of {name} failed -- the model is unavailable or answered nothing; "
                                     "Re-digest retries", "error")
        # Any other reason leaves the status line alone: an "added" that follows a failed add
        # must not wipe the failure the user has not read yet (review, 2026-10-06); the user's
        # next action resets it. The usage panel's enabled-document count moves with the library.
        self._render_usage(self._fetch_context_usage())

    # ---- document actions
    def _pick_context_files(self) -> list[str]:
        """The "Add files..." picker; factored out so tests replace it on the instance."""
        try:
            paths, _flt = QFileDialog.getOpenFileNames(
                self, "Add reference documents", "", self._CONTEXT_FILE_FILTER)
        except Exception:  # noqa: BLE001
            log.exception("file picker failed")
            return []
        return [str(p) for p in paths]

    def _prompt_context_text(self) -> tuple[str, str] | None:
        """The "Add text..." dialog: a name and the pasted text. ``None`` when cancelled or empty;
        factored out so tests replace it on the instance."""
        dlg = QDialog(self)
        dlg.setWindowTitle("Add text to the context library")
        layout = QVBoxLayout(dlg)
        layout.addWidget(QLabel("Name", dlg))
        name_edit = QLineEdit(dlg)
        name_edit.setPlaceholderText("Pasted text")
        layout.addWidget(name_edit)
        layout.addWidget(QLabel("Text (markdown is fine)", dlg))
        text_edit = QPlainTextEdit(dlg)
        text_edit.setMinimumSize(420, 220)
        layout.addWidget(text_edit, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, dlg)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        layout.addWidget(buttons)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return None
        text = text_edit.toPlainText()
        if not text.strip():
            return None
        return (name_edit.text().strip() or "Pasted text", text)

    def _context_action_begins(self) -> None:
        """Every user action on the tab starts with a clean status line: the failure it may
        have been showing has now been seen (review, 2026-10-06)."""
        self._set_context_status("", "")

    def _on_context_add_files(self) -> None:
        self._context_action_begins()
        try:
            paths = list(self._pick_context_files() or [])
        except Exception:  # noqa: BLE001
            log.exception("_pick_context_files failed")
            paths = []
        for path in paths:
            try:
                self._actions.add_context_document(str(path), mode=None)
            except Exception:  # noqa: BLE001
                log.exception("actions.add_context_document(%r) failed", path)
                self._set_context_status(f"failed to add {Path(str(path)).name}", "error")

    def _on_context_add_text(self) -> None:
        self._context_action_begins()
        try:
            picked = self._prompt_context_text()
        except Exception:  # noqa: BLE001
            log.exception("_prompt_context_text failed")
            picked = None
        if not picked:
            return
        name, text = picked
        try:
            self._actions.add_context_text(name, text, mode=None)
        except Exception:  # noqa: BLE001
            log.exception("actions.add_context_text(%r) failed", name)
            self._set_context_status(f"failed to add {name}", "error")

    def _on_context_remove(self) -> None:
        self._context_action_begins()
        row = self._selected_context_row()
        if not row:
            return
        doc_id = str(row.get("id") or "")
        name = row.get("name") or doc_id
        answer = QMessageBox.question(
            self, "Remove document", f"Remove '{name}' from the context library? This cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            self._actions.remove_context_document(doc_id)
        except Exception:  # noqa: BLE001
            log.exception("actions.remove_context_document(%r) failed", doc_id)
            self._set_context_status(f"failed to remove {name}", "error")

    def _on_context_toggle(self) -> None:
        self._context_action_begins()
        row = self._selected_context_row()
        if not row:
            return
        doc_id = str(row.get("id") or "")
        try:
            self._actions.set_context_document(doc_id, enabled=not bool(row.get("enabled", True)))
        except Exception:  # noqa: BLE001
            log.exception("actions.set_context_document(%r, enabled=...) failed", doc_id)
            self._set_context_status("failed to change the document", "error")

    def _on_context_mode(self) -> None:
        self._context_action_begins()
        row = self._selected_context_row()
        if not row:
            return
        doc_id = str(row.get("id") or "")
        new_mode = "inject" if str(row.get("mode") or "") == "reference" else "reference"
        try:
            self._actions.set_context_document(doc_id, mode=new_mode)
        except Exception:  # noqa: BLE001
            log.exception("actions.set_context_document(%r, mode=%r) failed", doc_id, new_mode)
            self._set_context_status("failed to change the document", "error")

    def _on_context_redigest(self) -> None:
        self._context_action_begins()
        row = self._selected_context_row()
        if not row:
            return
        doc_id = str(row.get("id") or "")
        try:
            self._actions.refresh_context_digest(doc_id)
        except Exception:  # noqa: BLE001
            log.exception("actions.refresh_context_digest(%r) failed", doc_id)
            self._set_context_status("failed to queue the digest", "error")

    def _on_context_open_folder(self) -> None:
        try:
            folder = Path(self._cfg.context_dir())
            if not folder.is_dir():
                self._set_context_status("the context folder does not exist yet (add a document first)", "info")
                return
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))
        except Exception:  # noqa: BLE001
            log.exception("open the context folder failed")

    # ---- usage panel
    def _fetch_context_usage(self) -> dict[str, object] | None:
        try:
            return self._actions.context_usage()
        except Exception:  # noqa: BLE001
            log.exception("actions.context_usage() failed")
            return None

    def _render_usage(self, usage: object, *, snapshot: object = None) -> None:
        """Fill the usage panel from a ``ContextUsageEvent`` or ``actions.context_usage()``'s dict
        (``None`` / ``{}`` before any job renders as "no answer yet"). `snapshot` (the engine's
        dict, when the panel is fed an event) supplies what an event does not carry: the running
        earlier-summary tool tip."""
        self.context_usage_label.setText(_format_usage(usage))
        summary = _usage_dict(usage).get("earlier_summary") or _usage_dict(snapshot).get("earlier_summary")
        self.context_usage_label.setToolTip(f"Earlier in this meeting (summary):\n{summary}" if summary else "")

    # ---- settings
    def _on_context_apply(self) -> None:
        changed = self._context_form.apply()
        restart = [f.label for f in self._CONTEXT_FIELDS if f.restart and f.path in changed]
        if restart:
            self.context_note.setText("Takes effect at the next launch: " + ", ".join(restart))
        elif changed:
            self.context_note.setText("Applied.")
        else:
            self.context_note.setText("")
        if changed:
            log.info("context settings changed: %s", ", ".join(f"{k}={v!r}" for k, v in changed.items()))
            self.applied.emit(changed)

    # ============================================================================== Capabilities
    def _build_capabilities_tab(self) -> QWidget:
        """The Capabilities tab (2026-10-07): one scroll area with the groups Switches (+ the scalar
        limits), Read scope, Edit scope, Commands, Web, MCP servers and Maps, top to bottom; the
        status line and the Apply / Restore defaults row stay pinned under it so Apply is never a
        scroll away."""
        w = QWidget(self)
        outer = QVBoxLayout(w)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(8)

        scroll = QScrollArea(w)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        body = QWidget(scroll)
        col = QVBoxLayout(body)
        col.setContentsMargins(2, 2, 8, 2)
        col.setSpacing(8)
        scroll.setWidget(body)
        outer.addWidget(scroll, 1)
        self.cap_scroll = scroll

        # config key -> the widget holding that list (QListWidget for folders, QPlainTextEdit for
        # one-entry-per-line boxes); read and written through _cap_read_list / _cap_write_list.
        self._cap_list_widgets: dict[str, QListWidget | QPlainTextEdit] = {}

        col.addWidget(self._build_cap_switches_group(body))
        col.addWidget(self._build_cap_read_group(body))
        col.addWidget(self._build_cap_edit_group(body))
        col.addWidget(self._build_cap_commands_group(body))
        col.addWidget(self._build_cap_web_group(body))
        col.addWidget(self._build_cap_mcp_group(body))
        col.addWidget(self._build_cap_maps_group(body))
        col.addStretch(1)

        self.cap_status = QLabel("", w)
        self.cap_status.setWordWrap(True)
        self.cap_status.setTextFormat(Qt.TextFormat.PlainText)
        self.cap_status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.cap_status.setStyleSheet(f"color: {COLORS['text_dim']};")
        outer.addWidget(self.cap_status)

        self.cap_note = QLabel("", w)
        self.cap_note.setWordWrap(True)
        self.cap_note.setTextFormat(Qt.TextFormat.PlainText)
        outer.addWidget(self.cap_note)

        apply_row = QHBoxLayout()
        self.cap_restore_btn = QPushButton("Restore defaults", w)
        self.cap_restore_btn.clicked.connect(self._on_capabilities_restore)
        self.cap_apply_btn = QPushButton("Apply", w)
        self.cap_apply_btn.setObjectName("primary")
        self.cap_apply_btn.clicked.connect(self._on_capabilities_apply)
        apply_row.addWidget(self.cap_restore_btn)
        apply_row.addStretch(1)
        apply_row.addWidget(self.cap_apply_btn)
        outer.addLayout(apply_row)
        return w

    # ---- tab construction helpers
    @staticmethod
    def _cap_group(title: str, parent: QWidget) -> tuple[QGroupBox, QVBoxLayout]:
        """A titled group box with a vertical layout, the unit the tab is stacked from."""
        group = QGroupBox(title, parent)
        lay = QVBoxLayout(group)
        lay.setSpacing(6)
        return group, lay

    @staticmethod
    def _cap_hint(text: str, parent: QWidget) -> QLabel:
        """A dim, wrapped explanatory line."""
        label = QLabel(text, parent)
        label.setWordWrap(True)
        label.setTextFormat(Qt.TextFormat.PlainText)
        label.setStyleSheet(f"color: {COLORS['text_dim']};")
        return label

    @staticmethod
    def _cap_lines_edit(parent: QWidget, height: int, placeholder: str = "") -> QPlainTextEdit:
        """A plain-text box for one entry per line (patterns, commands, domains)."""
        edit = QPlainTextEdit(parent)
        edit.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
        edit.setTabChangesFocus(True)
        edit.setMinimumHeight(height)
        edit.setMaximumHeight(max(height, 220))
        edit.setPlaceholderText(placeholder)
        return edit

    def _build_cap_switches_group(self, parent: QWidget) -> QGroupBox:
        group, lay = self._cap_group("Switches", parent)
        tips = {
            "read_files": "The model may read, list and search the readable folders below (recursively).",
            "edit_files": "The ask box may create and edit files in the editable folders below.",
            "execute_commands": "The ask box may run the allowed commands (below) in a readable folder.",
            "fetch_web": "The ask box may search the web and fetch pages within the domain lists below.",
            "mcp_servers": "The ask box may call the tools of the enabled MCP servers below.",
        }
        row = _FlowLayout()          # one row; wraps only when the window is narrower than the five
        self._cap_switches: dict[str, QCheckBox] = {}
        for key, label in self._CAPABILITY_SWITCHES:
            cb = QCheckBox(label, group)
            cb.setToolTip(tips[key])
            row.addWidget(cb)
            self._cap_switches[key] = cb
        lay.addLayout(row)
        self.cap_read_cb = self._cap_switches["read_files"]
        self.cap_edit_cb = self._cap_switches["edit_files"]
        self.cap_exec_cb = self._cap_switches["execute_commands"]
        self.cap_web_cb = self._cap_switches["fetch_web"]
        self.cap_mcp_cb = self._cap_switches["mcp_servers"]

        lay.addWidget(self._cap_hint(
            "Only the ask box can run commands, edit files or reach the web; answers to the other "
            "speaker or the screen get read-only tools. With no readable folder no tool is offered at all.",
            group))

        # The FieldForm builds its own scroll area; this group sits inside the tab's one scroll
        # area, so only the form's inner widget is kept (a nested scroll area would trap the wheel).
        self._capabilities_form = FieldForm(self._CAPABILITY_FIELDS, self._cfg)
        form_scroll = self._capabilities_form.widget
        inner = form_scroll.takeWidget()
        lay.addWidget(inner)
        form_scroll.deleteLater()
        form_layout = inner.layout()
        if isinstance(form_layout, QFormLayout):
            form_layout.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        for widget in self._capabilities_form.widgets.values():
            if isinstance(widget, (QAbstractSpinBox, QComboBox)):
                widget.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
                widget.installEventFilter(self._wheel_guard)
        return group

    def _build_cap_scope_group(
        self, parent: QWidget, title: str, hint: str, roots_key: str, exclude_key: str,
    ) -> tuple[QGroupBox, QListWidget, QPushButton, QPushButton, QPlainTextEdit]:
        """A folder list with Add folder... / Remove and an exclusion box; shared by Read / Edit scope."""
        group, lay = self._cap_group(title, parent)
        lay.addWidget(self._cap_hint(hint, group))
        row = QHBoxLayout()
        roots = QListWidget(group)
        roots.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        roots.setMinimumHeight(64)
        roots.setMaximumHeight(120)
        row.addWidget(roots, 1)
        buttons = QVBoxLayout()
        add_btn = QPushButton("Add folder…", group)
        remove_btn = QPushButton("Remove", group)
        remove_btn.setEnabled(False)
        buttons.addWidget(add_btn)
        buttons.addWidget(remove_btn)
        buttons.addStretch(1)
        row.addLayout(buttons)
        lay.addLayout(row)
        lay.addWidget(QLabel("Excluded (one pattern per line)", group))
        exclude = self._cap_lines_edit(group, 70, ".git\nnode_modules\n*.pem")
        lay.addWidget(exclude)

        add_btn.clicked.connect(lambda: self._on_cap_add_root(roots))
        remove_btn.clicked.connect(lambda: self._on_cap_remove_roots(roots))
        roots.itemSelectionChanged.connect(
            lambda: remove_btn.setEnabled(bool(roots.selectedItems()) or roots.currentRow() >= 0))
        roots.currentRowChanged.connect(
            lambda _row: remove_btn.setEnabled(bool(roots.selectedItems()) or roots.currentRow() >= 0))
        self._cap_list_widgets[roots_key] = roots
        self._cap_list_widgets[exclude_key] = exclude
        return group, roots, add_btn, remove_btn, exclude

    def _build_cap_read_group(self, parent: QWidget) -> QGroupBox:
        group, self.cap_read_roots, self.cap_read_add_btn, self.cap_read_remove_btn, self.cap_read_exclude = (
            self._build_cap_scope_group(
                parent, "Read scope",
                "The folders the model may read, list and search, recursively. Without one, no tool is "
                "offered at all. Names matching an exclusion (any path component) are never read, "
                "listed, searched or indexed; the app's own config, models, sessions and maps folders "
                "are always off limits.",
                "read_roots", "read_exclude"))
        return group

    def _build_cap_edit_group(self, parent: QWidget) -> QGroupBox:
        group, self.cap_edit_roots, self.cap_edit_add_btn, self.cap_edit_remove_btn, self.cap_edit_exclude = (
            self._build_cap_scope_group(
                parent, "Edit scope",
                "The folders the ask box may write to (needs the Edit files switch). Only files that "
                "match no exclusion; each change is atomic and, with the backup option, the original "
                "is kept in the session folder. There is no delete tool.",
                "edit_roots", "edit_exclude"))
        return group

    def _build_cap_commands_group(self, parent: QWidget) -> QGroupBox:
        group, lay = self._cap_group("Commands", parent)
        lay.addWidget(self._cap_hint(
            "One allowed command per line; an entry allows that command and its arguments "
            "(\"git log\" allows \"git log --oneline\"). Needs the Execute commands switch and a "
            "readable folder to run in.", group))
        self.cap_commands = self._cap_lines_edit(group, 130)
        lay.addWidget(self.cap_commands)
        row = QHBoxLayout()
        self.cap_commands_restore_btn = QPushButton("Restore safe list", group)
        self.cap_commands_restore_btn.clicked.connect(self._on_cap_restore_safe_commands)
        row.addWidget(self.cap_commands_restore_btn)
        row.addStretch(1)
        lay.addLayout(row)
        lay.addWidget(self._cap_hint(
            "This list is a guard against accidents, not a sandbox: an allowed command can still do "
            "whatever its own arguments let it. File readers and listers only see the readable "
            "folders. \"All commands\" (above) lifts the list and every other check.", group))
        self._cap_list_widgets["allowed_commands"] = self.cap_commands
        return group

    def _build_cap_web_group(self, parent: QWidget) -> QGroupBox:
        group, lay = self._cap_group("Web", parent)
        lay.addWidget(self._cap_hint(
            "Needs the Fetch web content switch. Allowed domains: one per line, empty = any public "
            "site; \"example.com\" also allows its subdomains. Blocked domains win. Local and "
            "private addresses are refused unless allowed above.", group))
        lay.addWidget(QLabel("Allowed domains", group))
        self.cap_web_allowed = self._cap_lines_edit(group, 60, "docs.python.org")
        lay.addWidget(self.cap_web_allowed)
        lay.addWidget(QLabel("Blocked domains", group))
        self.cap_web_blocked = self._cap_lines_edit(group, 60)
        lay.addWidget(self.cap_web_blocked)
        self._cap_list_widgets["web_allowed_domains"] = self.cap_web_allowed
        self._cap_list_widgets["web_blocked_domains"] = self.cap_web_blocked
        return group

    def _build_cap_mcp_group(self, parent: QWidget) -> QGroupBox:
        group, lay = self._cap_group("MCP servers", parent)
        lay.addWidget(self._cap_hint(
            "Needs the MCP servers switch. A stdio server is started as a child process with a "
            "minimal environment plus the variables you give it; an http server is called with POST. "
            "Its tools are offered to the ask box as mcp:<server>:<tool>. Apply, then Test.", group))
        self.cap_mcp_table = QTableWidget(0, len(self._MCP_COLUMNS), group)
        self.cap_mcp_table.setHorizontalHeaderLabels(list(self._MCP_COLUMNS))
        self.cap_mcp_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.cap_mcp_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.cap_mcp_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.cap_mcp_table.setMinimumHeight(100)
        self.cap_mcp_table.itemSelectionChanged.connect(self._update_mcp_buttons)
        lay.addWidget(self.cap_mcp_table)
        row = QHBoxLayout()
        self.cap_mcp_add_btn = QPushButton("Add…", group)
        self.cap_mcp_add_btn.clicked.connect(self._on_mcp_add)
        self.cap_mcp_edit_btn = QPushButton("Edit…", group)
        self.cap_mcp_edit_btn.clicked.connect(self._on_mcp_edit)
        self.cap_mcp_remove_btn = QPushButton("Remove", group)
        self.cap_mcp_remove_btn.clicked.connect(self._on_mcp_remove)
        self.cap_mcp_test_btn = QPushButton("Test", group)
        self.cap_mcp_test_btn.clicked.connect(self._on_mcp_test)
        for b in (self.cap_mcp_add_btn, self.cap_mcp_edit_btn, self.cap_mcp_remove_btn, self.cap_mcp_test_btn):
            row.addWidget(b)
        row.addStretch(1)
        lay.addLayout(row)
        return group

    def _build_cap_maps_group(self, parent: QWidget) -> QGroupBox:
        group, lay = self._cap_group("Maps", parent)
        lay.addWidget(self._cap_hint(
            "A map is a short overview of the readable folders (what each file is, on a basic "
            "understanding: no model call). It is saved with the settings it was made with and kept "
            "in the project; the active map rides along in every tool-enabled prompt, so the model "
            "knows which paths exist. Loading a map re-applies its settings; Re-index re-scans a map's "
            "own folders, Pre-index maps the folders listed above.", group))
        self.cap_maps_table = QTableWidget(0, len(self._MAP_COLUMNS), group)
        self.cap_maps_table.setHorizontalHeaderLabels(list(self._MAP_COLUMNS))
        self.cap_maps_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.cap_maps_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.cap_maps_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.cap_maps_table.setMinimumHeight(90)
        self.cap_maps_table.itemSelectionChanged.connect(self._update_map_buttons)
        lay.addWidget(self.cap_maps_table)
        row = QHBoxLayout()
        self.cap_index_btn = QPushButton("Pre-index scoped directories…", group)
        self.cap_index_btn.clicked.connect(self._on_map_index)
        self.cap_map_load_btn = QPushButton("Load", group)
        self.cap_map_load_btn.clicked.connect(self._on_map_load)
        self.cap_map_reindex_btn = QPushButton("Re-index", group)
        self.cap_map_reindex_btn.setToolTip(
            "Re-scan the folders this map was made from, with the settings it was saved with. "
            "What the tab shows now is not used.")
        self.cap_map_reindex_btn.clicked.connect(self._on_map_reindex)
        self.cap_map_delete_btn = QPushButton("Delete", group)
        self.cap_map_delete_btn.clicked.connect(self._on_map_delete)
        self.cap_map_open_btn = QPushButton("Open folder", group)
        self.cap_map_open_btn.clicked.connect(self._on_map_open_folder)
        for b in (self.cap_index_btn, self.cap_map_load_btn, self.cap_map_reindex_btn,
                  self.cap_map_delete_btn, self.cap_map_open_btn):
            row.addWidget(b)
        row.addStretch(1)
        lay.addLayout(row)
        self.cap_map_status = QLabel("", group)
        self.cap_map_status.setWordWrap(True)
        self.cap_map_status.setTextFormat(Qt.TextFormat.PlainText)
        lay.addWidget(self.cap_map_status)
        return group

    # ---- load / read back
    def _load_capabilities(self) -> None:
        """Fill the whole tab from ``cfg.capabilities`` and re-read the maps and the status (every
        open; ``actions`` may return ``None`` or raise for any of them)."""
        self._mcp_tools_text.clear()
        self._mcp_requests.clear()
        self._fill_capabilities_from_cfg()
        self.cap_note.setText("")
        self.cap_map_status.setText("")
        self._refresh_maps()
        self._refresh_capability_status()

    def _fill_capabilities_from_cfg(self) -> None:
        """The form, the five switches, the lists and the MCP table from ``cfg.capabilities`` (also
        after a map was loaded: the engine rewrote the settings it carries)."""
        cc = self._cfg.capabilities
        self._capabilities_form.load()
        for key, cb in self._cap_switches.items():
            cb.setChecked(bool(getattr(cc, key)))
        for key in self._CAPABILITY_LIST_KEYS:
            self._cap_write_list(key, getattr(cc, key))
        self._mcp_specs = [copy.deepcopy(s) for s in cc.mcp if isinstance(s, dict)]
        self._refresh_mcp_table()

    def _cap_read_list(self, key: str) -> list[str]:
        """The entries of one list as the tab shows them: blank lines dropped, whitespace stripped."""
        widget = self._cap_list_widgets[key]
        if isinstance(widget, QListWidget):
            return [t for t in (widget.item(i).text().strip() for i in range(widget.count())) if t]
        return _split_lines(widget.toPlainText())

    def _cap_write_list(self, key: str, values: object) -> None:
        widget = self._cap_list_widgets[key]
        items = [str(v) for v in values] if isinstance(values, (list, tuple)) else []
        if isinstance(widget, QListWidget):
            widget.clear()
            widget.addItems(items)
        else:
            widget.setPlainText("\n".join(items))

    # ---- folders
    def _pick_directory(self) -> str:
        """The "Add folder..." picker ("" when cancelled); factored out so tests replace it on the instance."""
        try:
            return str(QFileDialog.getExistingDirectory(self, "Add a folder") or "")
        except Exception:  # noqa: BLE001
            log.exception("folder picker failed")
            return ""

    def _on_cap_add_root(self, widget: QListWidget) -> None:
        try:
            path = str(self._pick_directory() or "").strip()
        except Exception:  # noqa: BLE001
            log.exception("_pick_directory failed")
            path = ""
        if not path:
            return
        wanted = os.path.normcase(os.path.normpath(path))
        for i in range(widget.count()):
            if os.path.normcase(os.path.normpath(widget.item(i).text())) == wanted:
                widget.setCurrentRow(i)   # already listed
                return
        widget.addItem(QListWidgetItem(path))
        widget.setCurrentRow(widget.count() - 1)

    @staticmethod
    def _on_cap_remove_roots(widget: QListWidget) -> None:
        items = widget.selectedItems() or ([widget.currentItem()] if widget.currentItem() is not None else [])
        for item in items:
            widget.takeItem(widget.row(item))

    def _on_cap_restore_safe_commands(self) -> None:
        """Fill the commands box with ``SAFE_COMMANDS`` (nothing is applied until Apply)."""
        self.cap_commands.setPlainText("\n".join(SAFE_COMMANDS))

    # ---- apply / restore
    def _set_cap_note(self, text: str, state: str = "") -> None:
        self.cap_note.setText(text)
        self.cap_note.setStyleSheet(f"color: {_status_color(state)};" if state else "")

    def _capabilities_changes(self) -> dict[str, Any]:
        """``{"capabilities.key": new_value}`` for everything on the tab that differs from the
        config: the form's scalars, the five switches and the lists (compared as lists)."""
        cc = self._cfg.capabilities
        changes: dict[str, Any] = dict(self._capabilities_form.changes())
        for key, cb in self._cap_switches.items():
            if cb.isChecked() != bool(getattr(cc, key)):
                changes[f"capabilities.{key}"] = cb.isChecked()
        for key in self._CAPABILITY_LIST_KEYS:
            new = self._cap_read_list(key)
            if new != [str(v) for v in getattr(cc, key)]:
                changes[f"capabilities.{key}"] = new
        if self._mcp_specs != [s for s in cc.mcp if isinstance(s, dict)]:
            changes["capabilities.mcp"] = copy.deepcopy(self._mcp_specs)
        return changes

    def _apply_capabilities(self) -> dict[str, Any]:
        """Write what differs into ``cfg.capabilities`` and emit ``applied`` with those keys; returns them."""
        changed = self._capabilities_changes()
        cc = self._cfg.capabilities
        for path, value in changed.items():
            setattr(cc, path.split(".", 1)[1], copy.deepcopy(value))
        restart = [f.label for f in self._CAPABILITY_FIELDS if f.restart and f.path in changed]
        if restart:
            self._set_cap_note("Takes effect at the next launch: " + ", ".join(restart))
        elif changed:
            self._set_cap_note("Applied.")
        else:
            self._set_cap_note("")
        if changed:
            # keys only: an MCP server's env may hold a token, which never goes into the log
            log.info("capability settings changed: %s", ", ".join(sorted(changed)))
            self.applied.emit(changed)
            if "capabilities.maps_dir" in changed:
                # applied is a direct signal: the engine already switched folders, so the table must
                # list the new one's maps (the dialog otherwise re-reads them on events only)
                self._refresh_maps()
        self._refresh_capability_status()
        return changed

    def _on_capabilities_apply(self) -> None:
        self._apply_capabilities()

    def _on_capabilities_restore(self) -> None:
        """Reset the form, the switches and every list to ``Config().capabilities`` -- on screen
        only; nothing is written or applied until Apply."""
        defaults = Config().capabilities
        self._capabilities_form.restore_defaults()
        for key, cb in self._cap_switches.items():
            cb.setChecked(bool(getattr(defaults, key)))
        for key in self._CAPABILITY_LIST_KEYS:
            self._cap_write_list(key, getattr(defaults, key))
        self._mcp_specs = [copy.deepcopy(s) for s in defaults.mcp]
        self._refresh_mcp_table()
        self._set_cap_note("Defaults restored -- Apply to keep them.")

    # ---- status line
    def _fetch_capability_status(self) -> object:
        try:
            return self._actions.capability_status()
        except Exception:  # noqa: BLE001
            log.exception("actions.capability_status() failed")
            return None

    def _refresh_capability_status(self) -> None:
        text, state = _format_capability_status(self._fetch_capability_status())
        if text != self.cap_status.text():       # a poll that finds nothing new must not drop a text selection
            self.cap_status.setText(text)
        if state != self._cap_status_state:
            self._cap_status_state = state
            self.cap_status.setStyleSheet(f"color: {_status_color(state)};")

    def _cap_tab_on_screen(self) -> bool:
        return self.isVisible() and self.tabs.currentIndex() == self.TABS.index("Capabilities")

    def _sync_capability_poll(self) -> None:
        """Run the 2 s status poll exactly while the Capabilities tab is on screen."""
        if self._cap_tab_on_screen():
            if not self._cap_poll_timer.isActive():
                self._cap_poll_timer.start()
        else:
            self._cap_poll_timer.stop()

    def _poll_capability_status(self) -> None:
        if not self._cap_tab_on_screen():
            self._cap_poll_timer.stop()
            return
        self._refresh_capability_status()

    # ---- MCP servers
    @staticmethod
    def _mcp_target(spec: dict[str, Any]) -> str:
        """The "Command or URL" cell: the url (secret query values hidden) or ``command args``."""
        if _mcp_kind(spec) == "http":
            return _redact_url(str(spec.get("url") or ""))
        args = spec.get("args")
        parts = [str(spec.get("command") or "")] + ([str(a) for a in args] if isinstance(args, (list, tuple)) else [])
        return " ".join(p for p in parts if p)

    def _refresh_mcp_table(self, select: int | None = None) -> None:
        """Re-draw the MCP table from the working copy, keeping the selected row (or ``select``)."""
        keep = self.cap_mcp_table.currentRow() if select is None else select
        specs = self._mcp_specs
        self.cap_mcp_table.setRowCount(len(specs))
        for i, spec in enumerate(specs):
            name = str(spec.get("name") or "")
            enabled = _mcp_enabled(spec)
            tools_text = self._mcp_tools_text.get(name, "")
            values = (name, _mcp_kind(spec), self._mcp_target(spec),
                      "yes" if enabled else "no", tools_text)
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                item.setToolTip(value)
                if not enabled:
                    item.setForeground(QColor(COLORS["text_dim"]))
                elif col == 4 and tools_text.startswith("error"):
                    item.setForeground(QColor(COLORS["err"]))
                self.cap_mcp_table.setItem(i, col, item)
        self.cap_mcp_table.resizeColumnsToContents()
        self.cap_mcp_table.horizontalHeader().setStretchLastSection(True)
        if specs and 0 <= keep < len(specs):
            self.cap_mcp_table.setCurrentCell(keep, 0)
        else:
            self.cap_mcp_table.setCurrentCell(-1, -1)
        self._update_mcp_buttons()

    def _selected_mcp_index(self) -> int | None:
        idx = self.cap_mcp_table.currentRow()
        return idx if 0 <= idx < len(self._mcp_specs) else None

    def _update_mcp_buttons(self) -> None:
        has_row = self._selected_mcp_index() is not None
        for b in (self.cap_mcp_edit_btn, self.cap_mcp_remove_btn, self.cap_mcp_test_btn):
            b.setEnabled(has_row)

    def _prompt_mcp_server(self, existing: dict[str, Any] | None) -> dict[str, Any] | None:
        """The Add / Edit dialog of one MCP server: name, transport, command + arguments (stdio),
        url (http), environment as ``KEY=VALUE`` lines and the enabled flag. Returns the server's
        spec, or ``None`` when cancelled; factored out so tests replace it on the instance."""
        old: dict[str, Any] = dict(existing) if isinstance(existing, dict) else {}
        dlg = QDialog(self)
        dlg.setWindowTitle("Edit MCP server" if existing else "Add MCP server")
        form = QFormLayout(dlg)
        name_edit = QLineEdit(str(old.get("name") or ""), dlg)
        name_edit.setPlaceholderText("filesystem")
        transport_box = QComboBox(dlg)
        transport_box.addItems(["stdio", "http"])
        transport_box.setCurrentText("http" if _mcp_kind(old) == "http" else "stdio")
        command_edit = QLineEdit(str(old.get("command") or ""), dlg)
        command_edit.setPlaceholderText("npx")
        args = old.get("args")
        args_edit = self._cap_lines_edit(dlg, 70, "-y\n@modelcontextprotocol/server-filesystem\nD:/work")
        args_edit.setPlainText("\n".join(str(a) for a in args) if isinstance(args, (list, tuple)) else "")
        url_edit = QLineEdit(str(old.get("url") or ""), dlg)
        url_edit.setPlaceholderText("https://host/mcp")
        env = old.get("env")
        env_edit = self._cap_lines_edit(dlg, 60, "API_KEY=...")
        env_edit.setPlainText("\n".join(f"{k}={v}" for k, v in env.items()) if isinstance(env, dict) else "")
        enabled_cb = QCheckBox("Enabled", dlg)
        enabled_cb.setChecked(_mcp_enabled(old))
        form.addRow("Name", name_edit)
        form.addRow("Transport", transport_box)
        form.addRow("Command (stdio)", command_edit)
        form.addRow("Arguments (stdio, one per line)", args_edit)
        form.addRow("URL (http)", url_edit)
        form.addRow("Environment (KEY=VALUE per line)", env_edit)
        form.addRow("", enabled_cb)

        def sync_transport() -> None:
            stdio = transport_box.currentText() == "stdio"
            command_edit.setEnabled(stdio)
            args_edit.setEnabled(stdio)
            url_edit.setEnabled(not stdio)

        transport_box.currentTextChanged.connect(lambda _t: sync_transport())
        sync_transport()

        result: list[dict[str, Any]] = []

        def refuse(message: str) -> None:
            QMessageBox.warning(dlg, "MCP server", message)

        def accept() -> None:
            name = name_edit.text().strip()
            if not _MCP_NAME_RE.match(name):
                refuse("The name must be 1 to 64 letters, digits, '_', '.' or '-'.")
                return
            transport = transport_box.currentText()
            env_out: dict[str, str] = {}
            for line in _split_lines(env_edit.toPlainText()):
                key, sep, value = line.partition("=")
                if not sep or not key.strip():
                    refuse(f"'{line}' is not KEY=VALUE.")
                    return
                env_out[key.strip()] = value.strip()
            spec: dict[str, Any] = {"name": name}
            args_out: list[str] = []
            if transport == "stdio":
                command = command_edit.text().strip()
                if not command:
                    refuse("A stdio server needs a command.")
                    return
                spec["command"] = command
                args_out = _split_lines(args_edit.toPlainText())
            else:
                url = url_edit.text().strip()
                if not re.match(r"(?i)^https?://\S+$", url):
                    refuse("An http server needs an http:// or https:// URL.")
                    return
                spec["url"] = url
            enabled = enabled_cb.isChecked()
            if not old:                 # a new server gets the full shape
                spec["transport"], spec["env"], spec["enabled"] = transport, env_out, enabled
                if transport == "stdio":
                    spec["args"] = args_out
            else:
                # An edit leaves a hand-written entry as it was wherever the meaning is unchanged: a key
                # the entry never had stays out (unless it now differs from the default), and a value
                # that reads the same ("HTTP", "false", 1) is not rewritten. Otherwise opening Edit and
                # pressing OK would mark capabilities.mcp changed -- or flip a server the runner had
                # fail-closed to enabled.
                if "transport" in old and _mcp_kind(old) == transport:
                    spec["transport"] = old["transport"]
                elif "transport" in old or transport != _mcp_kind(old):
                    spec["transport"] = transport
                if transport == "stdio":
                    old_args = old.get("args")
                    if "args" in old and isinstance(old_args, (list, tuple)) and [str(a) for a in old_args] == args_out:
                        spec["args"] = old["args"]
                    elif "args" in old or args_out:
                        spec["args"] = args_out
                old_env = old.get("env")
                if "env" in old and isinstance(old_env, dict) and {str(k): str(v) for k, v in old_env.items()} == env_out:
                    spec["env"] = old["env"]
                elif "env" in old or env_out:
                    spec["env"] = env_out
                if "enabled" in old and _mcp_enabled(old) == enabled:
                    spec["enabled"] = old["enabled"]
                elif "enabled" in old or not enabled:
                    spec["enabled"] = enabled
            for k, v in old.items():   # a key this dialog does not edit is kept as the config had it
                if k not in ("name", "transport", "command", "args", "url", "env", "enabled"):
                    spec[k] = v
            result.append(spec)
            dlg.accept()

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, dlg)
        buttons.accepted.connect(accept)
        buttons.rejected.connect(dlg.reject)
        form.addRow(buttons)
        dlg.setMinimumWidth(480)
        if dlg.exec() != QDialog.DialogCode.Accepted or not result:
            return None
        return result[0]

    def _ask_mcp_server(self, existing: dict[str, Any] | None) -> dict[str, Any] | None:
        try:
            spec = self._prompt_mcp_server(existing)
        except Exception:  # noqa: BLE001
            log.exception("_prompt_mcp_server failed")
            return None
        return dict(spec) if isinstance(spec, dict) else None

    def _mcp_name_taken(self, name: str, *, except_index: int | None = None) -> bool:
        return any(i != except_index and str(s.get("name") or "") == name for i, s in enumerate(self._mcp_specs))

    def _on_mcp_add(self) -> None:
        spec = self._ask_mcp_server(None)
        if not spec:
            return
        name = str(spec.get("name") or "")
        if self._mcp_name_taken(name):
            self._set_cap_note(f"A server named '{name}' is already listed.", "error")
            return
        self._mcp_specs.append(spec)
        self._set_cap_note("")
        self._refresh_mcp_table(select=len(self._mcp_specs) - 1)

    def _on_mcp_edit(self) -> None:
        idx = self._selected_mcp_index()
        if idx is None:
            return
        before = self._mcp_specs[idx]
        old_name = str(before.get("name") or "")
        spec = self._ask_mcp_server(copy.deepcopy(before))
        if not spec:
            return
        name = str(spec.get("name") or "")
        if self._mcp_name_taken(name, except_index=idx):
            self._set_cap_note(f"A server named '{name}' is already listed.", "error")
            return
        self._mcp_specs[idx] = spec
        if spec != before:
            self._forget_mcp_test(old_name)     # a tool list of the old definition no longer applies
        self._set_cap_note("")
        self._refresh_mcp_table(select=idx)

    def _cfg_mcp_spec(self, name: str) -> dict[str, Any] | None:
        return next((s for s in self._cfg.capabilities.mcp if isinstance(s, dict) and s.get("name") == name), None)

    def _forget_mcp_test(self, name: str) -> None:
        self._mcp_tools_text.pop(name, None)
        self._mcp_requests.pop(name, None)

    def _on_mcp_remove(self) -> None:
        idx = self._selected_mcp_index()
        if idx is None:
            return
        self._forget_mcp_test(str(self._mcp_specs[idx].get("name") or ""))
        del self._mcp_specs[idx]
        self._refresh_mcp_table(select=min(idx, len(self._mcp_specs) - 1))

    def _on_mcp_test(self) -> None:
        """Ask the engine for the selected server's tools; the answer comes back as ``McpToolsEvent``
        and fills the row's Tools column (an answer to an older Test of the same row is ignored)."""
        idx = self._selected_mcp_index()
        if idx is None:
            return
        name = str(self._mcp_specs[idx].get("name") or "")
        if not name:
            return
        self._mcp_request_id += 1
        request_id = self._mcp_request_id
        self._mcp_requests[name] = request_id
        self._mcp_tools_text[name] = "testing…"
        self._refresh_mcp_table(select=idx)
        try:
            self._actions.mcp_server_tools(name, request_id)
        except Exception:  # noqa: BLE001
            log.exception("actions.mcp_server_tools(%r) failed", name)
            self._mcp_requests.pop(name, None)
            self._mcp_tools_text[name] = "error: could not start the test"
            self._refresh_mcp_table(select=idx)

    def _on_mcp_tools_event(self, ev: McpToolsEvent) -> None:
        try:
            self._record_mcp_tools(ev)
        finally:
            # a Test connects (and caches) the server, so its state changed even for an answer the
            # table ignores; the status line is the one place that shows it
            self._refresh_capability_status()

    def _record_mcp_tools(self, ev: McpToolsEvent) -> None:
        if self._mcp_requests.get(ev.server) != ev.request_id:
            return   # a stale answer (the row was tested again, edited or removed since)
        del self._mcp_requests[ev.server]
        tools = [t for t in ev.tools if isinstance(t, dict)]
        if ev.error:
            text = f"error: {ev.error}"
            spec = next((s for s in self._mcp_specs if s.get("name") == ev.server), None)
            if spec is not None and spec != self._cfg_mcp_spec(ev.server):
                text += " (Apply first: the engine tests the applied settings)"
        else:
            names = [str(t.get("name") or "") for t in tools]
            text = f"{len(tools)} tool{'' if len(tools) == 1 else 's'}"
            if names:
                text += ": " + ", ".join(names[:8]) + (", …" if len(names) > 8 else "")
        self._mcp_tools_text[ev.server] = text
        self._refresh_mcp_table()

    # ---- maps
    def _refresh_maps(self) -> None:
        """Re-read ``actions.capability_maps()`` into the table, keeping the selected row by name."""
        selected = self._selected_map_row()
        selected_name = selected.get("name") if selected else None
        try:
            result = self._actions.capability_maps()
        except Exception:  # noqa: BLE001
            log.exception("actions.capability_maps() failed")
            self._set_map_status("failed to read the maps", "error")
            result = None
        rows = [dict(r) for r in result if isinstance(r, dict)] if isinstance(result, (list, tuple)) else []
        self._map_rows = rows
        self.cap_maps_table.setRowCount(len(rows))
        for i, r in enumerate(rows):
            self._set_map_row(i, r)
        self.cap_maps_table.resizeColumnsToContents()
        self.cap_maps_table.horizontalHeader().setStretchLastSection(True)
        if selected_name is not None:
            for i, r in enumerate(rows):
                if r.get("name") == selected_name:
                    self.cap_maps_table.setCurrentCell(i, 0)
                    break
        self._update_map_buttons()

    def _set_map_row(self, i: int, r: dict[str, object]) -> None:
        """One map row: name, roots, file count ("(cut)" when the tree was cut at the limit), when
        it was indexed and whether it is the active one; the folder count is in the tool tip."""
        roots = r.get("roots")
        root_list = [str(x) for x in roots] if isinstance(roots, (list, tuple)) else ([str(roots)] if roots else [])
        files = _as_int(r.get("files")) or 0
        dirs = _as_int(r.get("dirs")) or 0
        truncated = bool(r.get("truncated"))
        active = bool(r.get("active"))
        values = (
            str(r.get("name") or ""),
            "; ".join(root_list),
            f"{files} (cut)" if truncated else str(files),
            str(r.get("indexed_at") or "")[:16].replace("T", " "),
            "active" if active else "",
        )
        tip = "\n".join(root_list) + f"\n{files} files in {dirs} folders" + ("\ncut at the file limit" if truncated else "")
        for col, value in enumerate(values):
            item = QTableWidgetItem(value)
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            item.setToolTip(tip.strip())
            if active:
                item.setForeground(QColor(COLORS["accent"]))
            self.cap_maps_table.setItem(i, col, item)

    def _selected_map_row(self) -> dict[str, object] | None:
        idx = self.cap_maps_table.currentRow()
        if 0 <= idx < len(self._map_rows):
            return self._map_rows[idx]
        return None

    def _update_map_buttons(self) -> None:
        has_row = self._selected_map_row() is not None
        for b in (self.cap_map_load_btn, self.cap_map_reindex_btn, self.cap_map_delete_btn):
            b.setEnabled(has_row)

    def _set_map_status(self, text: str, state: str = "") -> None:
        self.cap_map_status.setText(text)
        self.cap_map_status.setStyleSheet(f"color: {_status_color(state)};")

    def _on_capability_map_event(self, ev: CapabilityMapEvent) -> None:
        """A map was indexed / loaded / deleted (or the engine refused): re-read the list and say
        what happened. A loaded map rewrote the settings it carries, so the tab is re-filled from
        the config -- otherwise its next Apply would put the old values back."""
        reason = ev.reason or ""
        name = ev.name
        if reason == "loaded":
            self._fill_capabilities_from_cfg()
            self._set_cap_note("")
        self._refresh_maps()
        if reason == "indexing":
            self._set_map_status(f"Indexing {name}…", "loading")
        elif reason == "indexed":
            text = f"Map '{name}': {ev.files} files, {ev.dirs} folders"
            self._set_map_status(text + (f" ({ev.detail})" if ev.detail else ""), "done")
        elif reason == "loaded":
            self._set_map_status(f"Loaded '{name}'", "done")
        elif reason == "deleted":
            self._set_map_status(f"Deleted '{name}'", "done")
        elif reason.startswith("error"):
            self._set_map_status(reason, "error")
        else:
            self._set_map_status(reason, "")
        self._refresh_capability_status()

    def _prompt_map_name(self, default: str) -> str | None:
        """Ask for the new map's name (``None`` when cancelled or empty); factored out so tests
        replace it on the instance."""
        text, ok = QInputDialog.getText(
            self, "Pre-index scoped directories", "Name of the map:", QLineEdit.EchoMode.Normal, default)
        text = str(text).strip()
        return text if ok and text else None

    def _confirm(self, title: str, text: str) -> bool:
        """A Yes / No question (default No); factored out so tests replace it on the instance."""
        answer = QMessageBox.question(
            self, title, text, QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No)
        return answer == QMessageBox.StandardButton.Yes

    def _call_index(self, name: str) -> None:
        try:
            self._actions.index_capability_map(name)
        except Exception:  # noqa: BLE001
            log.exception("actions.index_capability_map(%r) failed", name)
            self._set_map_status(f"failed to index {name}", "error")

    def _on_map_index(self) -> None:
        """"Pre-index scoped directories...": name the map, apply the tab (so the map carries what
        the tab shows), then have the engine index the readable folders in the background. A name
        that already exists asks first and, on Yes, deletes that map before indexing (the engine
        re-indexes an existing map from its own saved settings, which would ignore the tab)."""
        self._set_map_status("", "")
        roots = self._cap_read_list("read_roots")
        if not roots:
            self._set_map_status("List at least one readable folder first (Read scope), then pre-index.", "info")
            return
        try:
            name = self._prompt_map_name(_folder_name(roots[0]))
        except Exception:  # noqa: BLE001
            log.exception("_prompt_map_name failed")
            name = None
        name = str(name or "").strip()
        if not name:
            return
        existing = next((r for r in self._map_rows if str(r.get("name") or "") == name), None)
        if existing is not None and not self._confirm_replace_map(name, existing, roots):
            return
        self._apply_capabilities()
        if existing is not None:
            # the engine re-indexes an existing map from the settings it was saved with, so a map
            # the tab is meant to replace has to go first
            try:
                self._actions.delete_capability_map(name)
            except Exception:  # noqa: BLE001
                log.exception("actions.delete_capability_map(%r) failed", name)
                self._set_map_status(f"failed to replace {name}", "error")
                return
        self._call_index(name)

    def _confirm_replace_map(self, name: str, existing: dict[str, object], roots: list[str]) -> bool:
        """Ask before Pre-index overwrites the map ``name`` (a failing confirm counts as No)."""
        raw = existing.get("roots")
        old_roots = [str(x) for x in raw] if isinstance(raw, (list, tuple)) else ([str(raw)] if raw else [])
        try:
            return self._confirm(
                "Replace map",
                f"A map named '{name}' already exists. It covers:\n  " + ("\n  ".join(old_roots) or "no folder")
                + "\n\nReplace it with the folders in the tab?\n  " + "\n  ".join(roots))
        except Exception:  # noqa: BLE001
            log.exception("_confirm failed")
            return False

    def _on_map_reindex(self) -> None:
        """Re-scan the selected map's own folders. Nothing on the tab is applied first: the engine
        re-indexes an existing map from the settings it was saved with (its folders, exclusions and
        file limit), so the map keeps its scope whatever the tab shows now."""
        row = self._selected_map_row()
        if not row:
            return
        self._set_map_status("", "")
        self._call_index(str(row.get("name") or ""))

    def _load_map_question(self, row: dict[str, object]) -> str:
        """What loading ``row`` will change, in words: the switches it flips and the folders it
        sets, read from the row's ``settings`` snapshot against the current configuration."""
        cc = self._cfg.capabilities
        name = str(row.get("name") or "")
        settings = row.get("settings")
        lines: list[str] = []
        if isinstance(settings, dict):
            for key, label in self._CAPABILITY_SWITCHES:
                new = settings.get(key)
                if isinstance(new, bool) and new != bool(getattr(cc, key)):
                    lines.append(f"- {label}: {'on' if new else 'off'} (now {'on' if getattr(cc, key) else 'off'})")
            for key, label in (("read_roots", "Readable folders"), ("edit_roots", "Editable folders")):
                new_roots = settings.get(key)
                if isinstance(new_roots, (list, tuple)) and [str(x) for x in new_roots] != list(getattr(cc, key)):
                    lines.append(f"- {label}: " + (", ".join(str(x) for x in new_roots) or "none"))
            enabled_names = settings.get("mcp_enabled")
            if isinstance(enabled_names, (list, tuple)):
                # only the servers the config already has can be flipped; a map never adds one
                wanted = {str(n) for n in enabled_names}
                for spec in cc.mcp:
                    if isinstance(spec, dict) and spec.get("name"):
                        now = _mcp_enabled(spec)
                        if (str(spec["name"]) in wanted) != now:
                            lines.append(f"- MCP server '{spec['name']}': {'off' if now else 'on'} (now {'on' if now else 'off'})")
            body = ("It will change:\n" + "\n".join(lines)) if lines else "It leaves the switches and folders as they are."
        else:
            body = "It re-applies the settings it was saved with."
        text = (f"Load the map '{name}'?\n\n{body}\n\nIts saved exclusions, commands, domains and limits are "
                "re-applied as well. A map never turns on \"All commands\" or local web access and never "
                "touches the MCP server definitions.")
        if self._capabilities_changes():
            text += "\n\nEdits on this tab that you have not applied are replaced."
        return text

    def _on_map_load(self) -> None:
        row = self._selected_map_row()
        if not row:
            return
        name = str(row.get("name") or "")
        self._set_map_status("", "")
        try:
            ok = self._confirm("Load map", self._load_map_question(row))
        except Exception:  # noqa: BLE001
            log.exception("_confirm failed")
            ok = False
        if not ok:
            return
        try:
            self._actions.load_capability_map(name)
        except Exception:  # noqa: BLE001
            log.exception("actions.load_capability_map(%r) failed", name)
            self._set_map_status(f"failed to load {name}", "error")

    def _on_map_delete(self) -> None:
        row = self._selected_map_row()
        if not row:
            return
        name = str(row.get("name") or "")
        self._set_map_status("", "")
        try:
            ok = self._confirm("Delete map", f"Delete the map '{name}'? The next index makes a new one.")
        except Exception:  # noqa: BLE001
            log.exception("_confirm failed")
            ok = False
        if not ok:
            return
        try:
            self._actions.delete_capability_map(name)
        except Exception:  # noqa: BLE001
            log.exception("actions.delete_capability_map(%r) failed", name)
            self._set_map_status(f"failed to delete {name}", "error")

    def _on_map_open_folder(self) -> None:
        try:
            folder = Path(self._cfg.maps_dir())
            if not folder.is_dir():
                self._set_map_status("the maps folder does not exist yet (pre-index a directory first)", "info")
                return
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))
        except Exception:  # noqa: BLE001
            log.exception("open the maps folder failed")

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

"""DashboardView: the overlay's dashboard mode -- the tracked session board as columns.

Four columns, each one a card list (``_Column``) or the running notes (``_NotesColumn``), in a
horizontal ``QSplitter``: **Session notes**, **Questions to ask**, **Blind spots & follow-ups**
and **Next steps & prompts** (next steps plus the prompts to type into Claude Code). Open items
come first; answered / done items sit under a collapsible "Answered (n)" toggle with their
answers. Every card offers its actions on hover buttons and a right-click menu: mark answered
(with an optional answer), dismiss, reopen, edit, copy. Column headers add an item by hand and
refresh that column; the header row refreshes everything.

Like every widget in :mod:`exocortex.ui.widgets`, this one never touches the bus: the overlay
hands it snapshots (:meth:`set_snapshot`, from ``DashboardUpdatedEvent``), the notes
(:meth:`set_notes`) and the refresh job's lifecycle (:meth:`begin_update` /
:meth:`finish_update` / :meth:`update_error`), and the view talks back through Qt signals the
overlay forwards to ``EngineActions``. Nothing here blocks: a manual action is one engine call.
"""
from __future__ import annotations

import logging
from collections.abc import Sequence

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QContextMenuEvent, QGuiApplication
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMenu,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTextBrowser,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from exocortex.bus import DashboardItem
from exocortex.config import Config
from exocortex.ui.markdown_lite import md_to_html
from exocortex.ui.theme import COLORS
from exocortex.ui.widgets import fmt_mmss

log = logging.getLogger(__name__)

#: (column key, title), in display order.
COLUMNS: tuple[tuple[str, str], ...] = (
    ("notes", "Session notes"),
    ("questions", "Questions to ask"),
    ("blindspots", "Blind spots & follow-ups"),
    ("nextsteps", "Next steps & prompts"),
)
#: Which item kinds each card column shows, in order.
KINDS_OF_COLUMN: dict[str, tuple[str, ...]] = {
    "questions": ("question",),
    "blindspots": ("blindspot",),
    "nextsteps": ("nextstep", "prompt"),
}
_COLUMN_FLAG: dict[str, str] = {
    "notes": "show_notes", "questions": "show_questions", "blindspots": "show_blindspots", "nextsteps": "show_nextsteps",
}
_ADD_KIND: dict[str, str] = {"questions": "question", "blindspots": "blindspot", "nextsteps": "nextstep"}
_EMPTY: dict[str, str] = {
    "questions": "No open questions yet. They arrive with each refresh, or add one with +.",
    "blindspots": "No blind spots or follow-ups yet.",
    "nextsteps": "No next steps yet.",
}
_COLUMN_MIN_WIDTH = 170
_ACTION_LABELS: dict[str, str] = {
    "answer": "Mark answered…", "done": "Mark done…", "dismiss": "Dismiss", "reopen": "Reopen",
    "edit": "Edit…", "copy": "Copy",
}


def _is_prompt_text(text: str) -> bool:
    """Whether a hand-added "next step" reads as a prompt for Claude Code (typed as one)."""
    t = (text or "").strip().casefold()
    return t.startswith("/") or t.startswith(("use the ", "ask the ", "run the "))


# ------------------------------------------------------------------ one item ----------
class ItemCard(QFrame):
    """One tracked item: its text, its status line (and answer), and its actions."""

    action = Signal(int, str)   # item id, "answer" | "dismiss" | "reopen" | "edit" | "copy"

    def __init__(self, item: DashboardItem, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.item = item
        self.setObjectName("card")
        is_prompt = item.kind == "prompt"
        resolved = item.status == "answered"
        self.setStyleSheet(
            f"QFrame#card {{ background: {COLORS['panel']}; border: 1px solid {COLORS['border']}; border-radius: 6px; }}"
            "QPushButton#icon { padding: 0px 4px; font-size: 9pt; }"
        )
        v = QVBoxLayout(self)
        v.setContentsMargins(8, 5, 6, 5)
        v.setSpacing(3)

        self.text_label = QLabel(self)
        self.text_label.setWordWrap(True)
        self.text_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.text_label.setText(item.text)
        style = ""
        if is_prompt:
            style += "font-family: Consolas, 'Cascadia Mono', 'Courier New', monospace;"
        if resolved:
            style += f"color: {COLORS['text_dim']};"
        if style:
            self.text_label.setStyleSheet(style)
        v.addWidget(self.text_label)

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(2)
        self.status_label = QLabel(self._status_text(), self)
        self.status_label.setObjectName("subtitle")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet(f"color: {COLORS['ok'] if resolved else COLORS['text_faint']};")
        row.addWidget(self.status_label, 1)
        self._buttons: dict[str, QPushButton] = {}
        actions = (("copy",) if is_prompt else ()) + (("reopen",) if resolved else ("answer", "dismiss"))
        glyphs = {"answer": "✓", "dismiss": "✕", "reopen": "↺", "copy": "⎘"}
        tips = {
            "answer": "Mark done" if item.kind in ("nextstep", "prompt") else "Mark answered (and note the answer)",
            "dismiss": "Dismiss (no longer relevant)", "reopen": "Reopen", "copy": "Copy the prompt to the clipboard",
        }
        for name in actions:
            btn = QPushButton(glyphs[name], self)
            btn.setObjectName("icon")
            btn.setToolTip(tips[name])
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.clicked.connect(lambda _checked=False, n=name: self.action.emit(self.item.id, n))
            self._buttons[name] = btn
            row.addWidget(btn)
        v.addLayout(row)

    def _status_text(self) -> str:
        i = self.item
        if i.status == "answered":
            verb = "done" if i.kind in ("nextstep", "prompt") else "answered"
            head = f"{verb} {fmt_mmss(i.updated_at)}"
            return f"{head} — {i.answer}" if i.answer else head
        head = f"since {fmt_mmss(i.created_at)}"
        return head + (" · added by you" if i.source == "user" else "")

    def contextMenuEvent(self, event: QContextMenuEvent) -> None:  # noqa: N802 - Qt override
        menu = QMenu(self)
        menu.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        resolved = self.item.status == "answered"
        names = (("reopen",) if resolved else (("done" if self.item.kind in ("nextstep", "prompt") else "answer"), "dismiss")) \
            + ("edit", "copy")
        for name in names:
            act = menu.addAction(_ACTION_LABELS[name])
            act.triggered.connect(lambda _checked=False, n=("answer" if name == "done" else name): self.action.emit(self.item.id, n))
        menu.popup(event.globalPos())
        event.accept()


# ------------------------------------------------------------------ a card column -----
class _Column(QFrame):
    add_requested = Signal(str)        # column key
    refresh_requested = Signal(str)    # column key
    item_action = Signal(int, str)     # forwarded from the cards

    def __init__(self, key: str, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.key = key
        self.setObjectName("column")
        self.setMinimumWidth(_COLUMN_MIN_WIDTH)
        self.setStyleSheet("QPushButton#icon { padding: 1px 5px; }")
        self._resolved_open = False
        v = QVBoxLayout(self)
        v.setContentsMargins(2, 0, 2, 0)
        v.setSpacing(4)

        head = QHBoxLayout()
        head.setSpacing(2)
        self.title_label = QLabel(title, self)
        self.title_label.setObjectName("subtitle")
        self.title_label.setStyleSheet(f"color: {COLORS['text']}; font-weight: 600;")
        self.title_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        head.addWidget(self.title_label, 1)
        self.count_label = QLabel("", self)
        self.count_label.setObjectName("subtitle")
        head.addWidget(self.count_label)
        self.add_btn = QPushButton("+", self)
        self.add_btn.setObjectName("icon")
        self.add_btn.setToolTip("Add an item by hand")
        self.add_btn.clicked.connect(lambda: self.add_requested.emit(self.key))
        head.addWidget(self.add_btn)
        self.refresh_btn = QPushButton("↻", self)
        self.refresh_btn.setObjectName("icon")
        self.refresh_btn.setToolTip("Refresh this column now (one model call)")
        self.refresh_btn.clicked.connect(lambda: self.refresh_requested.emit(self.key))
        head.addWidget(self.refresh_btn)
        v.addLayout(head)

        self.scroll = QScrollArea(self)
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._container = QWidget(self.scroll)
        self._vbox = QVBoxLayout(self._container)
        self._vbox.setContentsMargins(0, 0, 4, 0)
        self._vbox.setSpacing(4)
        self.open_layout = QVBoxLayout()
        self.open_layout.setSpacing(4)
        self._vbox.addLayout(self.open_layout)
        self.empty_label = QLabel(_EMPTY.get(key, ""), self._container)
        self.empty_label.setObjectName("subtitle")
        self.empty_label.setWordWrap(True)
        self._vbox.addWidget(self.empty_label)
        self.resolved_toggle = QToolButton(self._container)
        self.resolved_toggle.setCheckable(True)
        self.resolved_toggle.setArrowType(Qt.ArrowType.RightArrow)
        self.resolved_toggle.setText("Answered (0)")
        self.resolved_toggle.toggled.connect(self._on_resolved_toggled)
        self.resolved_toggle.hide()
        self._vbox.addWidget(self.resolved_toggle)
        self.resolved_container = QWidget(self._container)
        self.resolved_layout = QVBoxLayout(self.resolved_container)
        self.resolved_layout.setContentsMargins(0, 0, 0, 0)
        self.resolved_layout.setSpacing(4)
        self.resolved_container.hide()
        self._vbox.addWidget(self.resolved_container)
        self._vbox.addStretch(1)
        self.scroll.setWidget(self._container)
        v.addWidget(self.scroll, 1)

        self.status_label = QLabel("", self)
        self.status_label.setObjectName("subtitle")
        self.status_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        v.addWidget(self.status_label)

    @staticmethod
    def _clear(layout: QVBoxLayout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

    def _on_resolved_toggled(self, checked: bool) -> None:
        self._resolved_open = checked
        self.resolved_toggle.setArrowType(Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow)
        self.resolved_container.setVisible(checked and self.resolved_layout.count() > 0)

    def set_items(self, open_items: Sequence[DashboardItem], resolved_items: Sequence[DashboardItem]) -> None:
        self._clear(self.open_layout)
        self._clear(self.resolved_layout)
        for item in open_items:
            card = ItemCard(item, self._container)
            card.action.connect(self.item_action.emit)
            self.open_layout.addWidget(card)
        for item in resolved_items:
            card = ItemCard(item, self.resolved_container)
            card.action.connect(self.item_action.emit)
            self.resolved_layout.addWidget(card)
        self.empty_label.setVisible(not open_items)
        n_open, n_done = len(open_items), len(resolved_items)
        self.count_label.setText(f"{n_open} open" + (f" · {n_done} done" if n_done else ""))
        verb = "Done" if self.key == "nextsteps" else "Answered"
        self.resolved_toggle.setText(f"{verb} ({n_done})")
        self.resolved_toggle.setVisible(n_done > 0)
        self.resolved_container.setVisible(self._resolved_open and n_done > 0)

    def cards(self) -> list[ItemCard]:
        """The open cards, top to bottom (tests)."""
        out: list[ItemCard] = []
        for i in range(self.open_layout.count()):
            w = self.open_layout.itemAt(i).widget()
            if isinstance(w, ItemCard):
                out.append(w)
        return out

    def resolved_cards(self) -> list[ItemCard]:
        out: list[ItemCard] = []
        for i in range(self.resolved_layout.count()):
            w = self.resolved_layout.itemAt(i).widget()
            if isinstance(w, ItemCard):
                out.append(w)
        return out

    def set_status(self, text: str) -> None:
        self.status_label.setText(text)


# ------------------------------------------------------------------ the notes column --
class _NotesColumn(QFrame):
    refresh_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.key = "notes"
        self.setMinimumWidth(_COLUMN_MIN_WIDTH)
        self.setStyleSheet("QPushButton#icon { padding: 1px 5px; }")
        v = QVBoxLayout(self)
        v.setContentsMargins(2, 0, 2, 0)
        v.setSpacing(4)
        head = QHBoxLayout()
        head.setSpacing(2)
        self.title_label = QLabel("Session notes", self)
        self.title_label.setObjectName("subtitle")
        self.title_label.setStyleSheet(f"color: {COLORS['text']}; font-weight: 600;")
        self.title_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        head.addWidget(self.title_label, 1)
        self.updated_label = QLabel("", self)
        self.updated_label.setObjectName("subtitle")
        head.addWidget(self.updated_label)
        self.refresh_btn = QPushButton("↻", self)
        self.refresh_btn.setObjectName("icon")
        self.refresh_btn.setToolTip("Regenerate the notes now (Ctrl+Alt+N)")
        self.refresh_btn.clicked.connect(self.refresh_requested.emit)
        head.addWidget(self.refresh_btn)
        v.addLayout(head)
        self.browser = QTextBrowser(self)
        self.browser.setOpenExternalLinks(False)
        v.addWidget(self.browser, 1)
        self.set_notes("", None)

    def set_notes(self, markdown: str, at: float | None) -> None:
        self._markdown = markdown or ""
        self.browser.setHtml(md_to_html(self._markdown) if self._markdown.strip() else md_to_html(
            "*No notes yet. They refresh every few minutes once enough was said (the Notes settings), "
            "or press ↻.*"))
        self.updated_label.setText(f"updated {fmt_mmss(at)}" if at is not None else "")

    @property
    def markdown(self) -> str:
        return self._markdown


# ------------------------------------------------------------------ the view ----------
class DashboardView(QWidget):
    """The board. Signals, forwarded by the overlay to the engine: ``item_action(item id,
    action, text)``, ``item_added(kind, text)``, ``refresh_requested(sections)`` (an empty tuple
    = everything) and ``notes_requested()``."""

    item_action = Signal(int, str, str)
    item_added = Signal(str, str)
    refresh_requested = Signal(tuple)
    notes_requested = Signal()

    def __init__(self, cfg: Config, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._cfg = cfg
        self._items: tuple[DashboardItem, ...] = ()
        self._generated_at: float | None = None
        self._detail = ""
        self._job: int | None = None
        self._job_label = ""
        self._error = ""

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(4)

        head = QHBoxLayout()
        head.setSpacing(6)
        self.title_label = QLabel("Dashboard", self)
        self.title_label.setObjectName("title")
        head.addWidget(self.title_label)
        self.status_label = QLabel("", self)
        self.status_label.setObjectName("subtitle")
        self.status_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        head.addWidget(self.status_label, 1)
        self.refresh_all_btn = QPushButton("Refresh all", self)
        self.refresh_all_btn.setObjectName("primary")
        self.refresh_all_btn.setToolTip("Refresh the questions, blind spots and next steps (one model call) and the notes")
        self.refresh_all_btn.clicked.connect(lambda: self.refresh_requested.emit(()))
        head.addWidget(self.refresh_all_btn)
        layout.addLayout(head)

        self.splitter = QSplitter(Qt.Orientation.Horizontal, self)
        self.splitter.setChildrenCollapsible(False)
        self.notes_col = _NotesColumn(self.splitter)
        self.notes_col.refresh_requested.connect(self.notes_requested.emit)
        self.splitter.addWidget(self.notes_col)
        self.columns: dict[str, _Column] = {}
        for key, title in COLUMNS[1:]:
            col = _Column(key, title, self.splitter)
            col.item_action.connect(self._on_card_action)
            col.add_requested.connect(self._on_add)
            col.refresh_requested.connect(lambda k: self.refresh_requested.emit((k,)))
            self.columns[key] = col
            self.splitter.addWidget(col)
        for i in range(self.splitter.count()):
            self.splitter.setStretchFactor(i, 1)
        layout.addWidget(self.splitter, 1)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._render_status)
        self.apply_config()
        self._render()

    # ---- configuration ---------------------------------------------------------------------
    def apply_config(self) -> None:
        """Re-read ``[dashboard]``: which columns show, and the status re-render cadence."""
        d = self._cfg.dashboard
        for key, flag in _COLUMN_FLAG.items():
            widget = self.notes_col if key == "notes" else self.columns[key]
            widget.setVisible(bool(getattr(d, flag, True)))
        try:
            interval = max(1, int(d.view_refresh_s))
        except (TypeError, ValueError):
            interval = 5
        self._timer.setInterval(interval * 1000)
        if not self._timer.isActive():
            self._timer.start()

    def visible_columns(self) -> list[str]:
        out = []
        for key, _title in COLUMNS:
            widget = self.notes_col if key == "notes" else self.columns[key]
            if not widget.isHidden():
                out.append(key)
        return out

    def shutdown(self) -> None:
        self._timer.stop()

    # ---- data in -----------------------------------------------------------------------------
    def set_snapshot(
        self, items: Sequence[DashboardItem], notes: str, notes_at: float | None, generated_at: float, detail: str = "",
    ) -> None:
        self._items = tuple(items)
        self._generated_at = generated_at
        self._detail = detail or ""
        self._error = ""
        self.notes_col.set_notes(notes, notes_at)
        self._render()

    def set_notes(self, markdown: str, at: float | None) -> None:
        self.notes_col.set_notes(markdown, at)

    def begin_update(self, job_id: int, label: str = "") -> None:
        self._job = job_id
        self._job_label = label or "Dashboard refresh"
        self._error = ""
        self._render_status()

    def finish_update(self, job_id: int, cancelled: bool = False) -> None:
        if job_id != self._job:
            return
        self._job = None
        if cancelled:
            self._detail = "refresh cancelled"
        self._render_status()

    def update_error(self, job_id: int, message: str) -> None:
        if job_id != self._job:
            return
        self._job = None
        self._error = message
        self._render_status()

    def clear(self) -> None:
        self._items = ()
        self._generated_at = None
        self._detail = ""
        self._job = None
        self._error = ""
        self.notes_col.set_notes("", None)
        self._render()

    @property
    def items(self) -> tuple[DashboardItem, ...]:
        return self._items

    @property
    def updating(self) -> bool:
        return self._job is not None

    # ---- rendering ---------------------------------------------------------------------------
    def _render(self) -> None:
        for key, col in self.columns.items():
            kinds = KINDS_OF_COLUMN[key]
            open_items = [i for i in self._items if i.kind in kinds and i.status == "open"]
            done_items = [i for i in self._items if i.kind in kinds and i.status == "answered"]
            open_items.sort(key=lambda i: (kinds.index(i.kind), i.created_at, i.id))
            done_items.sort(key=lambda i: -i.updated_at)
            col.set_items(open_items, done_items)
        self._render_status()

    def _render_status(self) -> None:
        if self._job is not None:
            text = f"{self._job_label.lower()}…"
            color = COLORS["warn"]
        elif self._error:
            text = f"error: {self._error}"
            color = COLORS["err"]
        else:
            parts = []
            if self._generated_at is not None:
                parts.append(f"updated {fmt_mmss(self._generated_at)}")
            if self._detail:
                parts.append(self._detail)
            text = " · ".join(parts)
            color = COLORS["text_dim"]
        self.status_label.setText(text)
        self.status_label.setStyleSheet(f"color: {color};")

    # ---- actions out -------------------------------------------------------------------------
    def _item(self, item_id: int) -> DashboardItem | None:
        return next((i for i in self._items if i.id == item_id), None)

    def _on_card_action(self, item_id: int, action: str) -> None:
        item = self._item(item_id)
        if action == "copy":
            if item is not None:
                clipboard = QGuiApplication.clipboard()
                if clipboard is not None:
                    clipboard.setText(item.text)
            return
        if action == "answer":
            is_step = item is not None and item.kind in ("nextstep", "prompt")
            text, ok = QInputDialog.getText(
                self, "Mark done" if is_step else "Mark answered",
                "What happened? (optional)" if is_step else "What was the answer? (optional)",
            )
            if not ok:
                return
            self.item_action.emit(item_id, "answer", text.strip())
            return
        if action == "edit":
            text, ok = QInputDialog.getText(self, "Edit item", "Text:", text=item.text if item is not None else "")
            if not ok or not text.strip():
                return
            self.item_action.emit(item_id, "edit", text.strip())
            return
        self.item_action.emit(item_id, action, "")

    def _on_add(self, column_key: str) -> None:
        kind = _ADD_KIND.get(column_key)
        if kind is None:
            return
        prompt = {
            "question": "A question to ask:",
            "blindspot": "A blind spot or follow-up:",
            "nextstep": "A next step (or a prompt for Claude Code, starting with /):",
        }[kind]
        text, ok = QInputDialog.getText(self, "Add item", prompt)
        if not ok or not text.strip():
            return
        if kind == "nextstep" and _is_prompt_text(text):
            kind = "prompt"
        self.item_added.emit(kind, text.strip())

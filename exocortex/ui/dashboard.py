"""DashboardView: the overlay's dashboard mode -- the tracked session board as columns.

Six columns in a horizontal ``QSplitter``, left to right: **Session notes** (the running notes,
narrative only), **Action items** (primary tasks with their sub-tasks indented underneath, and the
suggested prompts for Claude Code under the item they serve), **Goals** (the acceptance criteria,
grouped under the action item they are for), **Constraints** (the rules and guardrails in force),
**Questions to ask** and **Blind spots & follow-ups**. Any column can be hidden (the View menu
flips ``[dashboard].show_*``; :meth:`apply_config` re-reads them). Open items come first;
answered / done / met items sit under a collapsible toggle with their answers. Every card offers
its actions on hover buttons and a right-click menu: mark answered / done / met (with an optional
note), dismiss, reopen, edit, copy -- and, on a task, add a sub-task, an acceptance criterion or a
prompt. Column headers add an item by hand and refresh that column; the header row refreshes
everything.

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

#: (column key, title), in display order. The keys are also the refresh sections.
COLUMNS: tuple[tuple[str, str], ...] = (
    ("notes", "Session notes"),
    ("nextsteps", "Action items"),
    ("goals", "Goals"),
    ("constraints", "Constraints"),
    ("questions", "Questions to ask"),
    ("blindspots", "Blind spots & follow-ups"),
)
#: Which item kinds each card column shows, in order.
KINDS_OF_COLUMN: dict[str, tuple[str, ...]] = {
    "nextsteps": ("nextstep", "prompt"),
    "goals": ("goal",),
    "constraints": ("constraint",),
    "questions": ("question",),
    "blindspots": ("blindspot",),
}
#: ``[dashboard]`` flag that shows / hides each column.
COLUMN_FLAGS: dict[str, str] = {
    "notes": "show_notes", "nextsteps": "show_nextsteps", "goals": "show_goals",
    "constraints": "show_constraints", "questions": "show_questions", "blindspots": "show_blindspots",
}
_ADD_KIND: dict[str, str] = {
    "nextsteps": "nextstep", "goals": "goal", "constraints": "constraint", "questions": "question",
    "blindspots": "blindspot",
}
_EMPTY: dict[str, str] = {
    "nextsteps": "No action items yet. The tasks from the brief and their steps show up here, or add one with +.",
    "goals": "No acceptance criteria yet: how each task will be judged done.",
    "constraints": "No constraints or guardrails yet: the rules the work must respect.",
    "questions": "No open questions yet. They arrive with each refresh, or add one with +.",
    "blindspots": "No blind spots or follow-ups yet.",
}
#: The toggle under a column's open items: its resolved items' name.
_RESOLVED_VERB: dict[str, str] = {"nextsteps": "Done", "goals": "Met", "questions": "Answered", "blindspots": "Resolved"}
_COLUMN_MIN_WIDTH = 120     # six columns must fit a small screen; cards wrap, the splitter widens any column
_INDENT_PX = 14
_ACTION_LABELS: dict[str, str] = {
    "answer": "Mark answered…", "done": "Mark done…", "met": "Mark met…", "resolve": "Mark resolved…",
    "dismiss": "Dismiss", "lift": "Lift (no longer applies)", "reopen": "Reopen",
    "edit": "Edit…", "copy": "Copy", "add_sub": "Add a sub-task…", "add_goal": "Add an acceptance criterion…",
    "add_prompt": "Add a prompt…",
}
#: A card's button / menu names that the engine knows under another one: finishing anything is
#: "answer" (the note is the answer), and lifting a constraint is a dismissal.
_ENGINE_ACTION: dict[str, str] = {"done": "answer", "met": "answer", "resolve": "answer", "lift": "dismiss"}
#: What a finished item of each kind is called.
_DONE_WORD: dict[str, str] = {"nextstep": "done", "prompt": "done", "goal": "met", "blindspot": "resolved", "question": "answered"}
_DONE_ACTION: dict[str, str] = {"nextstep": "done", "prompt": "done", "goal": "met", "blindspot": "resolve", "question": "answer"}


def _is_prompt_text(text: str) -> bool:
    """Whether a hand-added "action item" reads as a prompt for Claude Code (typed as one)."""
    t = (text or "").strip().casefold()
    return t.startswith("/") or t.startswith(("use the ", "ask the ", "run the "))


def _clip(text: str, n: int = 60) -> str:
    one = " ".join((text or "").split())
    return one if len(one) <= n else one[: n - 1] + "…"


# ------------------------------------------------------------------ one item ----------
class ItemCard(QFrame):
    """One tracked item: its text, its status line (and answer), and its actions. `note` is
    extra status text (a task's "2/3 steps done")."""

    action = Signal(int, str)   # item id, an action name (see _ACTION_LABELS)

    def __init__(self, item: DashboardItem, parent: QWidget | None = None, *, note: str = "") -> None:
        super().__init__(parent)
        self.item = item
        self.note = note
        self.setObjectName("card")
        kind = item.kind
        resolved = item.status == "answered"
        is_prompt = kind == "prompt"
        is_task = kind == "nextstep" and item.parent_id is None
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
        if is_task:
            style += "font-weight: 600;"
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
        done_name = _DONE_ACTION.get(kind, "answer")
        if resolved:
            actions: tuple[str, ...] = ("reopen",)
        elif kind == "constraint":
            actions = ("lift",)
        else:
            actions = (("copy",) if is_prompt else ()) + (("add_sub",) if is_task else ()) + (done_name, "dismiss")
        glyphs = {"answer": "✓", "done": "✓", "met": "✓", "resolve": "✓", "dismiss": "✕", "lift": "✕",
                  "reopen": "↺", "copy": "⎘", "add_sub": "＋"}
        tips = {
            "answer": "Mark answered (and note the answer)", "done": "Mark done", "met": "Mark met (and note the evidence)",
            "resolve": "Mark resolved", "dismiss": "Dismiss (no longer relevant)",
            "lift": "Lift: this constraint no longer applies", "reopen": "Reopen",
            "copy": "Copy the prompt to the clipboard", "add_sub": "Add a sub-task under this task",
        }
        for name in actions:
            btn = QPushButton(glyphs[name], self)
            btn.setObjectName("icon")
            btn.setToolTip(tips[name])
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.clicked.connect(lambda _checked=False, n=name: self.action.emit(self.item.id, _ENGINE_ACTION.get(n, n)))
            self._buttons[name] = btn
            row.addWidget(btn)
        v.addLayout(row)

    def _status_text(self) -> str:
        i = self.item
        if i.status == "answered":
            head = f"{_DONE_WORD.get(i.kind, 'answered')} {fmt_mmss(i.updated_at)}"
            return f"{head} — {i.answer}" if i.answer else head
        head = f"since {fmt_mmss(i.created_at)}"
        if i.source == "user":
            head += " · added by you"
        return f"{head} · {self.note}" if self.note else head

    def contextMenuEvent(self, event: QContextMenuEvent) -> None:  # noqa: N802 - Qt override
        menu = QMenu(self)
        menu.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        kind = self.item.kind
        if self.item.status == "answered":
            names: tuple[str, ...] = ("reopen", "edit", "copy")
        elif kind == "constraint":
            names = ("lift", "edit", "copy")
        else:
            names = (_DONE_ACTION.get(kind, "answer"), "dismiss", "edit", "copy")
            if kind == "nextstep":
                names += (("add_sub",) if self.item.parent_id is None else ()) + ("add_goal", "add_prompt")
        for name in names:
            act = menu.addAction(_ACTION_LABELS[name])
            act.triggered.connect(lambda _checked=False, n=name: self.action.emit(self.item.id, _ENGINE_ACTION.get(n, n)))
        menu.popup(event.globalPos())
        event.accept()


def _indented(widget: QWidget, levels: int, parent: QWidget) -> QWidget:
    """`widget` pushed right by `levels` indents (the card itself keeps its full border)."""
    if levels <= 0:
        widget.setParent(parent)
        return widget
    holder = QWidget(parent)
    h = QHBoxLayout(holder)
    h.setContentsMargins(levels * _INDENT_PX, 0, 0, 0)
    h.setSpacing(0)
    widget.setParent(holder)
    h.addWidget(widget)
    return holder


# ------------------------------------------------------------------ a card column -----
class _Column(QFrame):
    """A column of cards: open ones, then a collapsible list of the resolved ones. Subclasses
    lay the open cards out (a tree, groups); this one lists them oldest first."""

    add_requested = Signal(str)        # column key
    refresh_requested = Signal(str)    # column key
    item_action = Signal(int, str)     # forwarded from the cards

    def __init__(self, key: str, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.key = key
        self.kinds = KINDS_OF_COLUMN[key]
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
        self.open_container = QWidget(self._container)
        self.open_layout = QVBoxLayout(self.open_container)
        self.open_layout.setContentsMargins(0, 0, 0, 0)
        self.open_layout.setSpacing(4)
        self._vbox.addWidget(self.open_container)
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
                w.setParent(None)      # out of the tree at once, not only when the event loop deletes it
                w.deleteLater()

    @staticmethod
    def _cards_in(layout: QVBoxLayout) -> list[ItemCard]:
        """The cards a layout holds, top to bottom, looking inside the indent holders."""
        out: list[ItemCard] = []
        for i in range(layout.count()):
            w = layout.itemAt(i).widget()
            if isinstance(w, ItemCard):
                out.append(w)
            elif w is not None:
                out.extend(c for c in w.findChildren(ItemCard) if c.parent() is w)
        return out

    def _on_resolved_toggled(self, checked: bool) -> None:
        self._resolved_open = checked
        self.resolved_toggle.setArrowType(Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow)
        self.resolved_container.setVisible(checked and self.resolved_layout.count() > 0)

    def _card(self, item: DashboardItem, parent: QWidget, note: str = "") -> ItemCard:
        card = ItemCard(item, parent, note=note)
        card.action.connect(self.item_action.emit)
        return card

    def build_open(self, open_items: Sequence[DashboardItem], everything: Sequence[DashboardItem]) -> list[QWidget]:
        """The widgets for the open items, top to bottom (oldest first)."""
        ordered = sorted(open_items, key=lambda i: (self.kinds.index(i.kind), i.created_at, i.id))
        return [self._card(i, self.open_container) for i in ordered]

    def count_text(self, open_items: Sequence[DashboardItem], done_items: Sequence[DashboardItem]) -> str:
        n_open, n_done = len(open_items), len(done_items)
        return f"{n_open} open" + (f" · {n_done} done" if n_done else "")

    def set_items(self, items: Sequence[DashboardItem]) -> None:
        """Show `items` (every item of this column's kinds, open and resolved)."""
        open_items = [i for i in items if i.status == "open"]
        done_items = sorted((i for i in items if i.status == "answered"), key=lambda i: -i.updated_at)
        self._clear(self.open_layout)
        self._clear(self.resolved_layout)
        for widget in self.build_open(open_items, items):
            self.open_layout.addWidget(widget)
        for item in done_items:
            self.resolved_layout.addWidget(self._card(item, self.resolved_container))
        self.empty_label.setVisible(not open_items)
        self.count_label.setText(self.count_text(open_items, done_items))
        n_done = len(done_items)
        verb = _RESOLVED_VERB.get(self.key)
        self.resolved_toggle.setText(f"{verb or 'Done'} ({n_done})")
        self.resolved_toggle.setVisible(bool(verb) and n_done > 0)
        self.resolved_container.setVisible(bool(verb) and self._resolved_open and n_done > 0)

    def cards(self) -> list[ItemCard]:
        """The open cards, top to bottom (tests)."""
        return self._cards_in(self.open_layout)

    def resolved_cards(self) -> list[ItemCard]:
        return self._cards_in(self.resolved_layout)

    def set_status(self, text: str) -> None:
        self.status_label.setText(text)


class _ActionsColumn(_Column):
    """Primary tasks with their sub-tasks indented underneath; each item's suggested prompts
    under it (when `show_prompts`); prompts that serve no task, last."""

    def __init__(self, key: str, title: str, parent: QWidget | None = None) -> None:
        super().__init__(key, title, parent)
        self.show_prompts = True
        self.add_btn.setToolTip("Add a primary task (or a prompt for Claude Code, starting with /)")

    def build_open(self, open_items: Sequence[DashboardItem], everything: Sequence[DashboardItem]) -> list[QWidget]:
        steps = [i for i in open_items if i.kind == "nextstep"]
        prompts = [i for i in open_items if i.kind == "prompt"] if self.show_prompts else []
        ids = {i.id for i in steps}
        any_task = {i.id: i for i in everything if i.kind == "nextstep"}
        order = lambda i: (i.created_at, i.id)  # noqa: E731
        kids: dict[int, list[DashboardItem]] = {}
        for i in steps:
            if i.parent_id is not None and i.parent_id in ids:
                kids.setdefault(i.parent_id, []).append(i)
        roots = sorted((i for i in steps if i.parent_id is None or i.parent_id not in ids), key=order)
        by_parent: dict[int, list[DashboardItem]] = {}
        for p in prompts:
            if p.parent_id is not None and p.parent_id in ids:
                by_parent.setdefault(p.parent_id, []).append(p)
        progress: dict[int, str] = {}
        for i in everything:
            if i.kind == "nextstep" and i.parent_id is not None and i.status in ("open", "answered"):
                done, total = progress.get(i.parent_id, "0/0").split("/")
                progress[i.parent_id] = f"{int(done) + (i.status == 'answered')}/{int(total) + 1}"
        widgets: list[QWidget] = []

        def add(item: DashboardItem, level: int, note: str = "") -> None:
            card = self._card(item, self.open_container, note)
            widgets.append(_indented(card, level, self.open_container))
            for p in sorted(by_parent.get(item.id, ()), key=order):
                widgets.append(_indented(self._card(p, self.open_container), level + 1, self.open_container))

        for root in roots:
            if root.parent_id is not None and root.parent_id in any_task:
                note = f"step of “{_clip(any_task[root.parent_id].text, 40)}”"   # its task is done (or reopened later)
            else:
                note = f"{progress[root.id]} steps done" if root.id in progress else ""
            add(root, 0, note)
            for sub in sorted(kids.get(root.id, ()), key=order):
                add(sub, 1)
        loose = [p for p in prompts if p.parent_id is None or p.parent_id not in ids]
        if loose:
            heading = QLabel("Prompts to run", self.open_container)
            heading.setObjectName("subtitle")
            widgets.append(heading)
            widgets.extend(self._card(p, self.open_container) for p in sorted(loose, key=order))
        return widgets

    def count_text(self, open_items: Sequence[DashboardItem], done_items: Sequence[DashboardItem]) -> str:
        tasks = [i for i in open_items if i.kind == "nextstep"]
        ids = {i.id for i in tasks}
        primary = sum(1 for i in tasks if i.parent_id is None or i.parent_id not in ids)   # shown at the top
        done = sum(1 for i in done_items if i.kind == "nextstep")
        text = f"{primary} task{'s' if primary != 1 else ''}"
        if len(tasks) > primary:
            text += f" · {len(tasks) - primary} sub"
        return text + (f" · {done} done" if done else "")


class _GoalsColumn(_Column):
    """Acceptance criteria grouped under the action item each is for ("Whole session" first)."""

    def __init__(self, key: str, title: str, parent: QWidget | None = None) -> None:
        super().__init__(key, title, parent)
        self.task_text: dict[int, str] = {}
        self.task_rank: dict[int, int] = {}
        self.add_btn.setToolTip("Add an acceptance criterion (for a task, or for the whole session)")

    def set_tasks(self, tasks: Sequence[DashboardItem]) -> None:
        """The action items goals can be for (any status), in the order the board lists them."""
        ordered = sorted(tasks, key=lambda i: (i.created_at, i.id))
        self.task_text = {t.id: t.text for t in ordered}
        self.task_rank = {t.id: n for n, t in enumerate(ordered)}

    def build_open(self, open_items: Sequence[DashboardItem], everything: Sequence[DashboardItem]) -> list[QWidget]:
        order = lambda i: (i.created_at, i.id)  # noqa: E731
        groups: dict[int | None, list[DashboardItem]] = {}
        for g in open_items:
            key = g.parent_id if g.parent_id in self.task_text else None
            groups.setdefault(key, []).append(g)
        widgets: list[QWidget] = []
        keys = ([None] if None in groups else []) + sorted((k for k in groups if k is not None), key=lambda k: self.task_rank[k])
        for key in keys:
            if len(keys) > 1 or key is not None:
                heading = QLabel("Whole session" if key is None else f"For: {_clip(self.task_text[key])}", self.open_container)
                heading.setObjectName("subtitle")
                heading.setWordWrap(True)
                heading.setToolTip("" if key is None else self.task_text[key])
                heading.setStyleSheet(f"color: {COLORS['text_dim']}; font-weight: 600;")
                widgets.append(heading)
            widgets.extend(self._card(g, self.open_container) for g in sorted(groups[key], key=order))
        return widgets


class _ConstraintsColumn(_Column):
    """The rules in force: cards with a "lift" button, never "done"."""

    def count_text(self, open_items: Sequence[DashboardItem], done_items: Sequence[DashboardItem]) -> str:
        return f"{len(open_items)} in force"


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
    action, text)``, ``item_added(kind, text, parent id)`` (parent -1 = none),
    ``refresh_requested(sections)`` (an empty tuple = everything) and ``notes_requested()``."""

    item_action = Signal(int, str, str)
    item_added = Signal(str, str, int)
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
        self.refresh_all_btn.setToolTip("Refresh every column (one model call) and the notes")
        self.refresh_all_btn.clicked.connect(lambda: self.refresh_requested.emit(()))
        head.addWidget(self.refresh_all_btn)
        layout.addLayout(head)

        self.splitter = QSplitter(Qt.Orientation.Horizontal, self)
        self.splitter.setChildrenCollapsible(False)
        self.notes_col = _NotesColumn(self.splitter)
        self.notes_col.refresh_requested.connect(self.notes_requested.emit)
        self.splitter.addWidget(self.notes_col)
        self.columns: dict[str, _Column] = {}
        factories = {"nextsteps": _ActionsColumn, "goals": _GoalsColumn, "constraints": _ConstraintsColumn}
        for key, title in COLUMNS[1:]:
            col = factories.get(key, _Column)(key, title, self.splitter)
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
        """Re-read ``[dashboard]``: which columns show (and whether prompts show under the action
        items), and the status re-render cadence."""
        d = self._cfg.dashboard
        for key, flag in COLUMN_FLAGS.items():
            widget = self.notes_col if key == "notes" else self.columns[key]
            widget.setVisible(bool(getattr(d, flag, True)))
        actions = self.columns["nextsteps"]
        if isinstance(actions, _ActionsColumn) and actions.show_prompts != bool(getattr(d, "show_prompts", True)):
            actions.show_prompts = bool(getattr(d, "show_prompts", True))
            self._render()
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
        goals = self.columns["goals"]
        if isinstance(goals, _GoalsColumn):
            goals.set_tasks([i for i in self._items if i.kind == "nextstep"])
        for key, col in self.columns.items():
            col.set_items([i for i in self._items if i.kind in col.kinds])
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

    def _emit_add(self, kind: str, text: str, parent_id: int | None) -> None:
        self.item_added.emit(kind, text.strip(), -1 if parent_id is None else int(parent_id))

    def _on_card_action(self, item_id: int, action: str) -> None:
        item = self._item(item_id)
        if action == "copy":
            if item is not None:
                clipboard = QGuiApplication.clipboard()
                if clipboard is not None:
                    clipboard.setText(item.text)
            return
        if action == "answer" or action in ("done", "met", "resolve"):
            kind = item.kind if item is not None else "question"
            title = {"nextstep": "Mark done", "prompt": "Mark done", "goal": "Mark met"}.get(kind, "Mark answered")
            label = {"nextstep": "What happened? (optional)", "prompt": "What happened? (optional)",
                     "goal": "What showed it holds? (optional)"}.get(kind, "What was the answer? (optional)")
            text, ok = QInputDialog.getText(self, title, label)
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
        if action in ("add_sub", "add_goal", "add_prompt"):
            if item is None:
                return
            kind, title, label = {
                "add_sub": ("nextstep", "Add a sub-task", f"A sub-task of “{_clip(item.text)}”:"),
                "add_goal": ("goal", "Add an acceptance criterion", f"How will we know “{_clip(item.text)}” is done?"),
                "add_prompt": ("prompt", "Add a prompt", f"A prompt for Claude Code for “{_clip(item.text)}”:"),
            }[action]
            text, ok = QInputDialog.getText(self, title, label)
            if ok and text.strip():
                self._emit_add(kind, text, item_id)
            return
        self.item_action.emit(item_id, action, "")

    def _on_add(self, column_key: str) -> None:
        kind = _ADD_KIND.get(column_key)
        if kind is None:
            return
        parent_id: int | None = None
        if kind == "goal":
            tasks = [i for i in self._items if i.kind == "nextstep" and i.status == "open"]
            if tasks:
                choices = ["(the whole session)"] + [_clip(t.text, 80) for t in tasks]
                picked, ok = QInputDialog.getItem(self, "Add an acceptance criterion", "For which action item?", choices, 0, False)
                if not ok:
                    return
                if picked != choices[0] and picked in choices:
                    parent_id = tasks[choices.index(picked) - 1].id
        prompt = {
            "nextstep": "A primary task (or a prompt for Claude Code, starting with /):",
            "goal": "An acceptance criterion (how we will know it is done):",
            "constraint": "A constraint or guardrail (a rule the work must respect):",
            "question": "A question to ask:",
            "blindspot": "A blind spot or follow-up:",
        }[kind]
        text, ok = QInputDialog.getText(self, "Add item", prompt)
        if not ok or not text.strip():
            return
        if kind == "nextstep" and _is_prompt_text(text):
            kind = "prompt"
        self._emit_add(kind, text, parent_id)

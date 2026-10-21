"""The dashboard's database: tracked items and the running notes, in SQLite.

One :class:`DashboardStore` per session, backed by ``<session folder>/dashboard.sqlite`` (or
memory when persistence is off / in tests). It is the source of truth for the dashboard view
and the only thing a refresh reads from and writes to; the model sees a rendering of the OPEN
items and answers with changes, never the whole table. Thread-safe: one connection behind an
RLock, called from the assist-worker thread (a refresh), the Qt thread via the engine (a manual
edit) and the dashboard timer (notes).

Items carry a ``kind``:

- ``nextstep`` -- an action item: a primary task, or (with ``parent_id``) a sub-task of one. The
  tree is one level deep: a sub-task of a sub-task is attached to the task itself.
- ``goal`` -- an acceptance criterion; ``parent_id`` is the action item it is for, none = the
  whole session.
- ``constraint`` -- a rule or guardrail that must be respected (open = in force, dropped = lifted).
- ``question`` / ``blindspot`` -- to ask / to watch; ``prompt`` -- to type into Claude Code, with
  ``parent_id`` = the action item it serves (or none).

and a ``status`` (``open`` / ``answered`` -- answered, done, met or resolved, with ``answer`` saying
how -- / ``dropped``) and a normalised copy of their text used to fold duplicates *within the same
parent*: adding the wording of an item that is already open returns that item, of one already
answered is ignored, of one dropped reopens it. Dismissing a task dismisses its open sub-tasks,
goals and prompts with it; finishing a task finishes its open sub-tasks and retires their pending
prompts (its goals are left for a separate "met").
"""
from __future__ import annotations

import logging
import re
import sqlite3
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from exocortex.bus import DashboardItem

log = logging.getLogger(__name__)

#: In the order the board lists them and a refresh renders them: tasks first, because goals and
#: prompts refer to them by id.
KINDS: tuple[str, ...] = ("nextstep", "goal", "constraint", "question", "blindspot", "prompt")
STATUSES: tuple[str, ...] = ("open", "answered", "dropped")
KIND_TITLES: dict[str, str] = {
    "nextstep": "Action items",
    "goal": "Goals",
    "constraint": "Constraints",
    "question": "Questions to ask",
    "blindspot": "Blind spots & follow-ups",
    "prompt": "Suggested prompts",
}
#: Kinds that may carry a ``parent_id`` (an action item).
PARENTED_KINDS: tuple[str, ...] = ("nextstep", "goal", "prompt")

_PUNCT_RE = re.compile(r"[^\w\s]")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL,
    text TEXT NOT NULL,
    norm TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'open',
    answer TEXT NOT NULL DEFAULT '',
    source TEXT NOT NULL DEFAULT 'model',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    parent_id INTEGER
);
CREATE INDEX IF NOT EXISTS items_kind_status ON items (kind, status);
CREATE TABLE IF NOT EXISTS notes (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    markdown TEXT NOT NULL DEFAULT '',
    updated_at REAL
);
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def normalize_text(text: str) -> str:
    """Casefold, punctuation to spaces, collapse whitespace: the duplicate-folding key
    ("per-tenant?" and "per tenant" fold together)."""
    return " ".join(_PUNCT_RE.sub(" ", text or "").split()).casefold()


@dataclass(frozen=True)
class DashboardSnapshot:
    """What the view shows: every item it should render (open ones first, then the resolved
    ones kept on screen) and the running notes."""

    items: tuple[DashboardItem, ...] = ()
    notes: str = ""
    notes_at: float | None = None

    def of(self, kind: str, status: str | None = None) -> list[DashboardItem]:
        return [i for i in self.items if i.kind == kind and (status is None or i.status == status)]


@dataclass(frozen=True)
class ApplyResult:
    added: int = 0
    resolved: int = 0
    dropped: int = 0
    skipped: int = 0

    @property
    def changed(self) -> bool:
        return bool(self.added or self.resolved or self.dropped)

    def describe(self) -> str:
        parts = []
        if self.added:
            parts.append(f"+{self.added} new")
        if self.resolved:
            parts.append(f"{self.resolved} answered / done")
        if self.dropped:
            parts.append(f"{self.dropped} dropped")
        return ", ".join(parts) if parts else "nothing new"


def _row_item(row: sqlite3.Row) -> DashboardItem:
    parent = row["parent_id"]
    return DashboardItem(
        id=int(row["id"]), kind=str(row["kind"]), text=str(row["text"]), status=str(row["status"]),
        answer=str(row["answer"]), created_at=float(row["created_at"]), updated_at=float(row["updated_at"]),
        source=str(row["source"]), parent_id=int(parent) if parent is not None else None,
    )


class DashboardStore:
    def __init__(self, path: Path | str | None = None) -> None:
        self._path = Path(path) if path else None
        self._lock = threading.RLock()
        if self._path is not None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path) if self._path else ":memory:", check_same_thread=False,
                                     isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            if self._path is not None:
                try:
                    self._conn.execute("PRAGMA journal_mode=WAL")
                except sqlite3.DatabaseError:
                    log.debug("dashboard store: WAL not available", exc_info=True)
            self._conn.executescript(_SCHEMA)
            self._migrate()

    def _migrate(self) -> None:
        """A database written before tasks had sub-tasks lacks ``parent_id``: add it in place."""
        cols = {str(r["name"]) for r in self._conn.execute("PRAGMA table_info(items)")}
        if "parent_id" not in cols:
            self._conn.execute("ALTER TABLE items ADD COLUMN parent_id INTEGER")
        self._conn.execute("CREATE INDEX IF NOT EXISTS items_parent ON items (parent_id)")

    @property
    def path(self) -> Path | None:
        return self._path

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except sqlite3.Error:
                pass

    # ---- items ----------------------------------------------------------------------------
    def _valid_parent(self, kind: str, parent_id: int | None) -> int | None:
        """The parent a new `kind` item may really have: an existing, not dismissed action item.
        A sub-task of a sub-task hangs off the task itself (the tree is one level deep)."""
        if parent_id is None or kind not in PARENTED_KINDS:
            return None
        row = self._conn.execute("SELECT id, kind, status, parent_id FROM items WHERE id = ?", (int(parent_id),)).fetchone()
        if row is None or row["kind"] != "nextstep" or row["status"] == "dropped":
            return None
        if kind == "nextstep" and row["parent_id"] is not None:
            return self._valid_parent(kind, int(row["parent_id"]))
        return int(row["id"])

    def add_item(
        self, kind: str, text: str, *, now: float, source: str = "model", answer: str = "", parent_id: int | None = None,
    ) -> DashboardItem | None:
        """Add an open item; folds duplicates (see the module docstring). None for an empty text
        or an unknown kind, or when the same wording was already answered. `parent_id` is honoured
        only for tasks (-> a sub-task), goals and prompts, and only when it names an action item."""
        text = " ".join((text or "").split())
        if not text or kind not in KINDS:
            return None
        norm = normalize_text(text)
        if not norm:
            return None
        with self._lock:
            parent = self._valid_parent(kind, parent_id)
            row = self._conn.execute(
                "SELECT * FROM items WHERE kind = ? AND norm = ? AND COALESCE(parent_id, 0) = ? "
                "ORDER BY updated_at DESC LIMIT 1", (kind, norm, parent or 0)
            ).fetchone()
            if row is not None:
                existing = _row_item(row)
                if existing.status == "open":
                    return existing
                if existing.status == "answered":
                    return None
                self._conn.execute("UPDATE items SET status = 'open', answer = '', updated_at = ? WHERE id = ?",
                                   (float(now), existing.id))
                return self.get(existing.id)
            cur = self._conn.execute(
                "INSERT INTO items (kind, text, norm, status, answer, source, created_at, updated_at, parent_id) "
                "VALUES (?, ?, ?, 'open', ?, ?, ?, ?, ?)",
                (kind, text, norm, answer or "", source or "model", float(now), float(now), parent),
            )
            return self.get(int(cur.lastrowid))

    def get(self, item_id: int) -> DashboardItem | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM items WHERE id = ?", (int(item_id),)).fetchone()
        return _row_item(row) if row is not None else None

    def _set(self, item_id: int, *, now: float, **fields: object) -> DashboardItem | None:
        if not fields:
            return self.get(item_id)
        cols = ", ".join(f"{k} = ?" for k in fields)
        with self._lock:
            cur = self._conn.execute(f"UPDATE items SET {cols}, updated_at = ? WHERE id = ?",
                                     (*fields.values(), float(now), int(item_id)))
            if cur.rowcount == 0:
                return None
            return self.get(item_id)

    def resolve(self, item_id: int, answer: str = "", *, now: float) -> DashboardItem | None:
        """Mark answered / done / met / resolved, with `answer` saying how (may be empty). A
        finished primary task finishes its open sub-tasks with it and retires the prompts still
        pending under them; its goals stay as they are (met is a separate claim)."""
        with self._lock:
            item = self._set(item_id, now=now, status="answered", answer=" ".join((answer or "").split()))
            if item is None or item.kind != "nextstep" or item.parent_id is not None:
                return item
            steps = [i.id for i in self.children(item.id, "open") if i.kind == "nextstep"]
            for step in steps:
                self._set(step, now=now, status="answered", answer="with the task")
            for owner in (item.id, *steps):
                for p in self.children(owner, "open"):
                    if p.kind == "prompt":
                        self._set(p.id, now=now, status="dropped")
            return item

    def drop(self, item_id: int, *, now: float) -> DashboardItem | None:
        """Dismiss an item -- and, with it, everything still open that hangs off it (a task's
        sub-tasks, goals and prompts)."""
        with self._lock:
            item = self._set(item_id, now=now, status="dropped")
            if item is None:
                return None
            pending = [item.id]
            while pending:
                marks = ",".join("?" * len(pending))
                rows = self._conn.execute(
                    f"SELECT id FROM items WHERE parent_id IN ({marks}) AND status = 'open'", pending).fetchall()
                pending = [int(r["id"]) for r in rows]
                if pending:
                    marks = ",".join("?" * len(pending))
                    self._conn.execute(f"UPDATE items SET status = 'dropped', updated_at = ? WHERE id IN ({marks})",
                                       (float(now), *pending))
            return item

    def reopen(self, item_id: int, *, now: float) -> DashboardItem | None:
        return self._set(item_id, now=now, status="open", answer="")

    def edit(self, item_id: int, text: str, *, now: float) -> DashboardItem | None:
        text = " ".join((text or "").split())
        if not text:
            return self.get(item_id)
        return self._set(item_id, now=now, text=text, norm=normalize_text(text))

    def items(self, kind: str | None = None, status: str | None = None) -> list[DashboardItem]:
        """Open items oldest first (the order they came up), everything else newest first."""
        clauses, params = [], []
        if kind is not None:
            clauses.append("kind = ?")
            params.append(kind)
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM items{where} ORDER BY CASE status WHEN 'open' THEN 0 ELSE 1 END, "
                "CASE status WHEN 'open' THEN created_at ELSE -updated_at END, id", params,
            ).fetchall()
        return [_row_item(r) for r in rows]

    def open_items(self, kind: str, limit: int | None = None) -> list[DashboardItem]:
        out = self.items(kind, "open")
        return out[:limit] if limit is not None else out

    def children(self, parent_id: int, status: str | None = None) -> list[DashboardItem]:
        """Everything attached to an action item (sub-tasks, goals, prompts), oldest first."""
        sql = "SELECT * FROM items WHERE parent_id = ?"
        params: list[object] = [int(parent_id)]
        if status is not None:
            sql += " AND status = ?"
            params.append(status)
        with self._lock:
            rows = self._conn.execute(sql + " ORDER BY created_at, id", params).fetchall()
        return [_row_item(r) for r in rows]

    def counts(self) -> dict[str, tuple[int, int]]:
        """kind -> (open, answered) counts."""
        out = {k: (0, 0) for k in KINDS}
        with self._lock:
            rows = self._conn.execute(
                "SELECT kind, status, COUNT(*) AS n FROM items WHERE status != 'dropped' GROUP BY kind, status"
            ).fetchall()
        for r in rows:
            kind, status, n = str(r["kind"]), str(r["status"]), int(r["n"])
            if kind not in out:
                continue
            o, a = out[kind]
            out[kind] = (o + n, a) if status == "open" else (o, a + n)
        return out

    def apply(
        self,
        adds: Sequence[tuple],
        resolves: Sequence[tuple[int, str]],
        drops: Sequence[int],
        *,
        now: float,
        max_new_per_kind: int = 4,
        source: str = "model",
    ) -> ApplyResult:
        """Apply one refresh's changes: `resolves` (item id, answer) and `drops` first (they
        refer to items shown to the model), then `adds` -- ``(kind, text)``, ``(kind, text,
        parent)`` or ``(kind, text, parent, ref)`` -- at most `max_new_per_kind` genuinely new
        items per kind and parent (duplicates do not count against it). `parent` is an item id, or
        the `ref` (a label such as "N4") of an item added earlier in this same batch: the model
        writes a task and its sub-tasks together, before any of them has an id."""
        added = resolved = dropped = skipped = 0
        with self._lock:
            for item_id, answer in resolves:
                item = self.get(item_id)
                if item is None or item.status == "answered":
                    skipped += 1
                    continue
                self.resolve(item_id, answer, now=now)
                resolved += 1
            for item_id in drops:
                item = self.get(item_id)
                if item is None or item.status == "dropped":
                    skipped += 1
                    continue
                self.drop(item_id, now=now)
                dropped += 1
            per_bucket: dict[tuple[str, int | None], int] = {}
            refs: dict[str, int] = {}
            for entry in adds:
                kind, text = entry[0], entry[1]
                parent = entry[2] if len(entry) > 2 else None
                ref = entry[3] if len(entry) > 3 else None
                if isinstance(parent, str):
                    parent = refs.get(parent)
                if kind not in KINDS:
                    skipped += 1
                    continue
                parent = self._valid_parent(kind, parent)
                if per_bucket.get((kind, parent), 0) >= max(0, max_new_per_kind):
                    skipped += 1
                    continue
                before = self._conn.execute("SELECT COUNT(*) FROM items").fetchone()[0]
                item = self.add_item(kind, text, now=now, source=source, parent_id=parent)
                after = self._conn.execute("SELECT COUNT(*) FROM items").fetchone()[0]
                if item is not None and ref:
                    refs[str(ref)] = item.id        # also when it folded into an open duplicate
                if item is None:
                    skipped += 1
                elif after > before or (item.status == "open" and abs(item.updated_at - float(now)) < 1e-9
                                        and item.created_at != item.updated_at):
                    added += 1
                    per_bucket[(kind, parent)] = per_bucket.get((kind, parent), 0) + 1
                else:
                    skipped += 1
        return ApplyResult(added=added, resolved=resolved, dropped=dropped, skipped=skipped)

    # ---- notes ----------------------------------------------------------------------------
    def set_notes(self, markdown: str, at: float | None) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO notes (id, markdown, updated_at) VALUES (1, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET markdown = excluded.markdown, updated_at = excluded.updated_at",
                (markdown or "", at),
            )

    def notes(self) -> tuple[str, float | None]:
        with self._lock:
            row = self._conn.execute("SELECT markdown, updated_at FROM notes WHERE id = 1").fetchone()
        if row is None:
            return "", None
        at = row["updated_at"]
        return str(row["markdown"]), (float(at) if at is not None else None)

    # ---- meta -----------------------------------------------------------------------------
    def get_meta(self, key: str, default: str | None = None) -> str | None:
        with self._lock:
            row = self._conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return str(row["value"]) if row is not None else default

    def set_meta(self, key: str, value: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO meta (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, str(value)),
            )

    # ---- views ----------------------------------------------------------------------------
    def snapshot(self, *, resolved_keep: int = 12, show_resolved: bool = True) -> DashboardSnapshot:
        """Every open item, plus (with `show_resolved`) the `resolved_keep` most recently
        answered items of each kind; dropped items never show."""
        items: list[DashboardItem] = []
        for kind in KINDS:
            items.extend(self.items(kind, "open"))
            if show_resolved and resolved_keep > 0:
                items.extend(self.items(kind, "answered")[:resolved_keep])
        notes, at = self.notes()
        return DashboardSnapshot(items=tuple(items), notes=notes, notes_at=at)

    def to_markdown(self, title: str = "Dashboard") -> str:
        """The whole board as markdown (the session folder's dashboard.md): the notes, then the
        action items as a tree (sub-tasks indented, each task's prompts under it), the goals
        grouped as "for <task>", the constraints and the two question lists. ``- [ ]`` is open,
        ``- [x]`` answered / done / met (with its answer)."""
        notes, _at = self.notes()
        live = [i for i in self.items() if i.status != "dropped"]
        live.sort(key=lambda i: i.id)
        by_id = {i.id: i for i in live}
        kids: dict[int, list[DashboardItem]] = {}
        for i in live:
            if i.parent_id is not None and i.parent_id in by_id:
                kids.setdefault(i.parent_id, []).append(i)

        def mark(i: DashboardItem) -> str:
            return "[x]" if i.status == "answered" else "[ ]"

        def line(i: DashboardItem, indent: str = "") -> str:
            return f"{indent}- {mark(i)} {i.text}" + (f" — {i.answer}" if i.status == "answered" and i.answer else "")

        parts = [f"# {title}", "", "## Session notes", "", notes.strip() or "(no notes yet)", ""]
        parts += [f"## {KIND_TITLES['nextstep']}", ""]
        tasks = [i for i in live if i.kind == "nextstep" and (i.parent_id is None or i.parent_id not in by_id)]
        if not tasks:
            parts.append("- (none)")
        for task in tasks:
            parts.append(line(task))
            parts.extend(f"  - Prompt: `{p.text}`" for p in kids.get(task.id, []) if p.kind == "prompt")
            for sub in (k for k in kids.get(task.id, []) if k.kind == "nextstep"):
                parts.append(line(sub, "  "))
                parts.extend(f"    - Prompt: `{p.text}`" for p in kids.get(sub.id, []) if p.kind == "prompt")
        parts.append("")
        parts += [f"## {KIND_TITLES['goal']}", ""]
        goals = [i for i in live if i.kind == "goal"]
        if not goals:
            parts.append("- (none)")
        for g in goals:
            owner = by_id.get(g.parent_id) if g.parent_id is not None else None
            parts.append(line(g) + (f"  (for: {owner.text})" if owner is not None else ""))
        parts.append("")
        for kind in ("constraint", "question", "blindspot"):
            parts += [f"## {KIND_TITLES[kind]}", ""]
            rows = [i for i in live if i.kind == kind]
            if not rows:
                parts.append("- (none)")
            parts.extend(f"- {i.text}" if kind == "constraint" else line(i) for i in rows)
            parts.append("")
        loose = [i for i in live if i.kind == "prompt" and (i.parent_id is None or i.parent_id not in by_id)]
        if loose:
            parts += [f"## {KIND_TITLES['prompt']}", ""]
            parts.extend(f"- `{p.text}`" for p in loose)
            parts.append("")
        return "\n".join(parts).rstrip() + "\n"

    def clear(self) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM items")
            self._conn.execute("DELETE FROM notes")
            self._conn.execute("DELETE FROM meta")

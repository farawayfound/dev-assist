"""The dashboard's database: tracked items and the running notes, in SQLite.

One :class:`DashboardStore` per session, backed by ``<session folder>/dashboard.sqlite`` (or
memory when persistence is off / in tests). It is the source of truth for the dashboard view
and the only thing a refresh reads from and writes to; the model sees a rendering of the OPEN
items and answers with changes, never the whole table. Thread-safe: one connection behind an
RLock, called from the assist-worker thread (a refresh), the Qt thread via the engine (a manual
edit) and the dashboard timer (notes).

Items carry a ``kind`` (``question`` / ``blindspot`` / ``nextstep`` / ``prompt``), a ``status``
(``open`` / ``answered`` -- answered, done or resolved, with ``answer`` saying how -- / ``dropped``)
and a normalised copy of their text used to fold duplicates: adding the wording of an item that
is already open returns that item, of one already answered is ignored, of one dropped reopens it.
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

KINDS: tuple[str, ...] = ("question", "blindspot", "nextstep", "prompt")
STATUSES: tuple[str, ...] = ("open", "answered", "dropped")
KIND_TITLES: dict[str, str] = {
    "question": "Questions to ask",
    "blindspot": "Blind spots & follow-ups",
    "nextstep": "Next steps",
    "prompt": "Suggested prompts",
}

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
    updated_at REAL NOT NULL
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
    return DashboardItem(
        id=int(row["id"]), kind=str(row["kind"]), text=str(row["text"]), status=str(row["status"]),
        answer=str(row["answer"]), created_at=float(row["created_at"]), updated_at=float(row["updated_at"]),
        source=str(row["source"]),
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
    def add_item(self, kind: str, text: str, *, now: float, source: str = "model", answer: str = "") -> DashboardItem | None:
        """Add an open item; folds duplicates (see the module docstring). None for an empty text
        or an unknown kind, or when the same wording was already answered."""
        text = " ".join((text or "").split())
        if not text or kind not in KINDS:
            return None
        norm = normalize_text(text)
        if not norm:
            return None
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM items WHERE kind = ? AND norm = ? ORDER BY updated_at DESC LIMIT 1", (kind, norm)
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
                "INSERT INTO items (kind, text, norm, status, answer, source, created_at, updated_at) "
                "VALUES (?, ?, ?, 'open', ?, ?, ?, ?)",
                (kind, text, norm, answer or "", source or "model", float(now), float(now)),
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
        """Mark answered / done / resolved, with `answer` saying how (may be empty)."""
        return self._set(item_id, now=now, status="answered", answer=" ".join((answer or "").split()))

    def drop(self, item_id: int, *, now: float) -> DashboardItem | None:
        return self._set(item_id, now=now, status="dropped")

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
        adds: Sequence[tuple[str, str]],
        resolves: Sequence[tuple[int, str]],
        drops: Sequence[int],
        *,
        now: float,
        max_new_per_kind: int = 4,
        source: str = "model",
    ) -> ApplyResult:
        """Apply one refresh's changes: `resolves` (item id, answer) and `drops` first (they
        refer to items shown to the model), then `adds` (kind, text), at most `max_new_per_kind`
        genuinely new items per kind (duplicates do not count against it)."""
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
            per_kind: dict[str, int] = {}
            for kind, text in adds:
                if kind not in KINDS:
                    skipped += 1
                    continue
                if per_kind.get(kind, 0) >= max(0, max_new_per_kind):
                    skipped += 1
                    continue
                before = self._conn.execute("SELECT COUNT(*) FROM items").fetchone()[0]
                item = self.add_item(kind, text, now=now, source=source)
                after = self._conn.execute("SELECT COUNT(*) FROM items").fetchone()[0]
                if item is None:
                    skipped += 1
                elif after > before or (item.status == "open" and abs(item.updated_at - float(now)) < 1e-9
                                        and item.created_at != item.updated_at):
                    added += 1
                    per_kind[kind] = per_kind.get(kind, 0) + 1
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
        """The whole board as markdown (the session folder's dashboard.md): notes, then each
        kind with ``- [ ]`` open and ``- [x]`` answered items (with their answers)."""
        notes, _at = self.notes()
        parts = [f"# {title}", "", "## Session notes", "", notes.strip() or "(no notes yet)", ""]
        for kind in KINDS:
            parts.append(f"## {KIND_TITLES[kind]}")
            parts.append("")
            open_items = self.items(kind, "open")
            done_items = self.items(kind, "answered")
            if not open_items and not done_items:
                parts.append("- (none)")
            for i in open_items:
                parts.append(f"- [ ] {i.text}")
            for i in done_items:
                parts.append(f"- [x] {i.text}" + (f" — {i.answer}" if i.answer else ""))
            parts.append("")
        return "\n".join(parts).rstrip() + "\n"

    def clear(self) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM items")
            self._conn.execute("DELETE FROM notes")
            self._conn.execute("DELETE FROM meta")

"""The task's workspace (added 2026-10-29): the files and panes seen on screen since the checkpoint.

A coding task rarely fits one screen or one file: the statement sits in a browser tab or a README, the
code spans an interface, its implementation and a test, and the user moves between them while the
screen is read every ``[screen].interval_s``. Until now a Code / Text job saw only the latest capture
(or, hands-free, the cumulative text of the current task) and the model wrote for whatever it took to
be the main object -- skipping the interface the user was actually in. The :class:`Workspace` keeps
one :class:`View` per file / pane seen since the last checkpoint and sorts every capture into them:

* **The same view** -- the window title names the same file, the window title is the same (a page
  scrolled), or :data:`SAME_VIEW` of the capture's distinctive lines (sentences and code, not the
  one-word labels of menus and file trees) were already in the view: the view is updated. Lines seen
  earlier and scrolled away are kept (an edited line keeps only its latest reading), so a view
  approximates the whole file, not one screen of it.
* **Another file of the task** -- a file named in a development window's title, or a capture sharing
  :data:`RELATED_LINES` of its lines (the project's file tree, the statement panel, a shared layout)
  or :data:`RELATED_IDENTIFIERS` identifiers (class, function, snake_case names) with a view: a new
  view beside the others.
* **Foreign** -- a capture unrelated to every view: a chat, an email, another project. A glance never
  changes the workspace. :data:`FOREIGN_RESET` consecutive foreign captures that belong together AND
  look like a task (code lines, a task statement) mean the user moved on to a new problem without a
  checkpoint -- "accidental non-clearing" -- and the workspace restarts from them.

A checkpoint (:meth:`Workspace.reset`) always starts over: it stays the explicit boundary. The
Assistant hands the OTHER views to a Code / Text job as the ``<workspace>`` block (most recently seen
first, each under a ``[name]`` header, views unseen for :data:`STALE_S` left out) while the current one
is the ``<screen>``. No Qt, no threads of its own; thread-safe (written from the bus listener, read by
the assist worker).
"""
from __future__ import annotations

import re
import threading
from collections.abc import Iterable
from dataclasses import dataclass, field

from exocortex.handsfree import _CODING_TASK_RE, is_code_line, is_edit, normalize_keys

#: A capture whose normalised lines are contained in a view's lines at this fraction or more is a
#: re-reading of that view (the same file scrolled or edited, the same page re-read).
SAME_VIEW = 0.55
#: A capture sharing this fraction of its lines with a view belongs to the same task (another file,
#: another pane of it): the project's file tree, the task statement beside the editor, a shared layout.
RELATED_LINES = 0.12
#: ... or this many identifiers (class / function / snake_case names, see :func:`identifiers`).
RELATED_IDENTIFIERS = 2
#: This many consecutive foreign, task-like, mutually related captures restart the workspace.
FOREIGN_RESET = 4
#: Views kept; beyond it the one unseen for longest goes (never the current one).
MAX_VIEWS = 12
#: A capture with fewer readable lines teaches nothing (a blank screen, a dialog).
MIN_LINES = 3
#: A view unseen for this long (seconds of the caller's clock) is left out of the rendered block.
STALE_S = 900.0
#: Code lines on a capture that make it task-like for the foreign-run rule.
_TASK_CODE_LINES = 2
#: Title segment separators ("file - project - app", "project – file").
_TITLE_SPLIT_RE = re.compile(r"\s+[-–—|·]\s+")

#: A file name in a window title ("● main.py - myproj - Visual Studio Code", "myproj – Main.java").
_FILE_IN_TITLE_RE = re.compile(
    r"(?<![\w.])([\w][\w.-]*\.(?:py|pyi|ipynb|js|mjs|cjs|ts|tsx|jsx|java|kt|kts|scala|cs|fs|vb|cpp|cc|cxx|"
    r"c|h|hpp|hh|go|rs|rb|php|swift|mm?|dart|lua|r|jl|exs?|erl|hs|ml|clj|sql|sh|bash|zsh|ps1|psm1|bat|cmd|"
    r"html?|css|scss|less|vue|svelte|json|ya?ml|toml|xml|ini|cfg|conf|md|rst|txt|csv|proto|graphql|tf|"
    r"gradle|lock))(?![\w.])",
    re.IGNORECASE,
)
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
#: Words that read like identifiers but mean nothing about which task is on screen.
_STOP_WORDS = frozenset(
    """
    function functions variable variables argument arguments parameter parameters constraint constraints
    complexity solution solutions example examples explanation explanations description descriptions
    submission submissions editorial problems problem difficulty accepted runtime language languages
    python3 javascript typescript integer integers boolean character characters substring substrings
    sequence sequences element elements position positions possible following returns contains consider
    determine whether between without greater smaller maximum minimum operation operations approach
    implement implementation implementations interface interfaces abstract override exception exceptions
    readme license todo listnode treenode optional required default defaults import imports package
    namespace public private protected internal static final const string strings number numbers object
    objects function_name class_name file_name user_name first_name last_name
    """.split()
)


def file_in_title(title: str) -> str:
    """The file a window title names ("main.py" from "● main.py - myproj - Visual Studio Code"),
    lower-cased, or "" when it names none."""
    m = _FILE_IN_TITLE_RE.search(title or "")
    return m.group(1).lower() if m else ""


def project_in_title(title: str, file: str = "") -> str:
    """The project / solution a development window's title names next to its file -- "myproj" from
    "main.py - myproj - Visual Studio Code" or "myproj – Main.java" -- lower-cased; "" when the
    title names no file or no project (a browser tab, a terminal)."""
    file = file or file_in_title(title)
    if not file:
        return ""
    parts = [p.strip() for p in _TITLE_SPLIT_RE.split(" ".join((title or "").split())) if p.strip()]
    parts = [p.lstrip("●*• ").strip() for p in parts]
    if len(parts) >= 3:
        return parts[-2].lower()
    if len(parts) == 2 and file_in_title(parts[1]) == file and not file_in_title(parts[0]):
        return parts[0].lower()
    return ""


def identifiers(text: str) -> frozenset[str]:
    """The identifier-like words of ``text``: CamelCase or snake_case tokens and long plain words,
    minus :data:`_STOP_WORDS` -- what two files of one project, or a statement and its code, share."""
    out: set[str] = set()
    for tok in _IDENT_RE.findall(text or ""):
        if len(tok) < 4 or tok.lower() in _STOP_WORDS:
            continue
        inner = tok.strip("_")
        camel = any(c.isupper() for c in inner[1:]) and any(c.islower() for c in inner)
        snake = "_" in inner
        if camel or snake or len(tok) >= 8:
            out.add(tok)
    return frozenset(out)


def containment(keys: set[str], within: set[str]) -> float:
    """The share of ``keys`` found in ``within`` (0.0 for an empty ``keys``)."""
    return len(keys & within) / len(keys) if keys else 0.0


def looks_like_task(lines: Iterable[str], text: str) -> bool:
    """Whether a capture shows a coding task (code lines or a task statement) rather than a chat,
    an email or a news page -- what a foreign run must be made of to restart the workspace."""
    code = sum(1 for line in lines if is_code_line(line))
    return code >= _TASK_CODE_LINES or bool(_CODING_TASK_RE.search(text or ""))


@dataclass
class _Capture:
    """One screen reading, normalised once."""

    text: str
    lines: tuple[str, ...]
    keys: list[str]
    idents: frozenset[str]
    file: str
    project: str
    title: str
    process: str
    at: float

    @property
    def key_set(self) -> set[str]:
        return set(self.keys)

    @property
    def content_keys(self) -> set[str]:
        """The distinctive lines: sentences (3+ words) and code, not the one- or two-word labels of
        menus, file trees and tabs that every screen of an app shares. All keys when too few."""
        keys = {k for k, line in zip(self.keys, self.lines) if len(k.split()) >= 3 or is_code_line(line)}
        return keys if len(keys) >= 2 else set(self.keys)


@dataclass
class View:
    """One file or pane of the task, as every reading of it so far has shown it."""

    file: str = ""                 # the file named in the window title (lower-cased), "" when none
    project: str = ""              # the project named beside it (lower-cased), "" when none
    title: str = ""                # the window title at the latest reading
    process: str = ""              # the window's executable at the latest reading
    keys: set[str] = field(default_factory=set)        # every normalised line ever seen in it
    idents: set[str] = field(default_factory=set)
    lines: dict[str, str] = field(default_factory=dict)  # normalised line -> its reading, first-seen order
    first_seen: float = 0.0
    last_seen: float = 0.0
    captures: int = 0

    @property
    def name(self) -> str:
        """What the <workspace> header calls this view: its file, else its window title, else "pane"."""
        if self.file:
            return self.file
        title = " ".join((self.title or "").split())
        return title[:60] if title else "pane"

    def text(self, max_chars: int) -> str:
        """The view's lines, first-seen order, the latest reading of each, cut at ``max_chars``."""
        body = "\n".join(self.lines.values())
        if max_chars >= 0 and len(body) > max_chars:
            body = body[:max_chars].rsplit("\n", 1)[0] if "\n" in body[:max_chars] else body[:max_chars]
        return body

    def merge(self, cap: _Capture) -> None:
        """Fold a reading in: new lines are appended in reading order; a kept line that is gone
        from this reading while a near-identical one (``is_edit``: a few characters changed) is on
        it was edited, and only the new reading stays. Lines merely scrolled away are kept."""
        self.keys.update(cap.keys)
        self.idents.update(cap.idents)
        new_keys = set(cap.keys)
        fresh = [k for k in cap.keys if k not in self.lines]
        if fresh:
            for key in [k for k in self.lines if k not in new_keys]:
                if any(is_edit(key, k) for k in fresh):
                    del self.lines[key]
        for key, line in zip(cap.keys, cap.lines):
            self.lines.setdefault(key, line)
        if cap.file:
            self.file = cap.file
        if cap.project:
            self.project = cap.project
        if cap.title:
            self.title = cap.title
        if cap.process:
            self.process = cap.process
        self.last_seen = cap.at
        if not self.captures:
            self.first_seen = cap.at
        self.captures += 1

    def related_to(self, cap: _Capture) -> bool:
        """Whether ``cap`` belongs to the same task as this view (shared lines or identifiers)."""
        return (containment(cap.key_set, self.keys) >= RELATED_LINES
                or len(cap.idents & self.idents) >= RELATED_IDENTIFIERS)


class Workspace:
    """The views of the current task -- see the module docstring. Every method is thread-safe."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._views: list[View] = []
        self._current: View | None = None
        self._foreign: list[_Capture] = []
        self.last_outcome = ""          # "same" | "new" | "foreign" | "reset" | "empty" (tests, logs)

    # ---- feeding --------------------------------------------------------------------------

    def observe(self, text: str, *, title: str = "", process: str = "", now: float = 0.0) -> View | None:
        """Sort one screen reading (its reading-order ``text``; the window ``title`` / ``process`` at
        the time) into the views. Returns the view it landed in -- the current view from now on --
        or None for an empty or foreign capture. Idempotent for a repeated identical reading."""
        cap = self._capture(text, title, process, now)
        with self._lock:
            if cap is None:
                self.last_outcome = "empty"
                return None
            view = self._place(cap, allow_foreign=True)
            return view

    def reset(self) -> None:
        """A checkpoint: forget every view (the next capture starts the next task's workspace)."""
        with self._lock:
            self._views = []
            self._current = None
            self._foreign = []
            self.last_outcome = ""

    # ---- reading --------------------------------------------------------------------------

    @property
    def current(self) -> View | None:
        with self._lock:
            return self._current

    def views(self) -> list[View]:
        """Every view, oldest first (a copy of the list; the views themselves are live)."""
        with self._lock:
            return list(self._views)

    def others_text(self, *, current: View | None = None, max_chars: int = 3000, now: float | None = None) -> str:
        """The <workspace> body: every view but ``current`` (the current view when None), most
        recently seen first, each as a ``[name]`` header and its lines, the whole cut to
        ``max_chars`` (shared evenly, at least 400 characters each while the budget allows). Views
        unseen for :data:`STALE_S` of the caller's clock (``now``) are left out. "" when there is
        nothing else to show."""
        with self._lock:
            cur = current if current is not None else self._current
            others = [v for v in self._views if v is not cur and v.captures]
            if now is not None:
                others = [v for v in others if now - v.last_seen <= STALE_S]
            others.sort(key=lambda v: v.last_seen, reverse=True)
            if not others or max_chars <= 0:
                return ""
            per_view = max(400, max_chars // len(others))
            parts: list[str] = []
            used = 0
            for view in others:
                header = f"[{view.name}]"
                room = min(per_view, max_chars - used - len(header) - 2)
                if room < 80:
                    break
                body = view.text(room)
                if not body.strip():
                    continue
                parts.append(f"{header}\n{body}")
                used += len(header) + len(body) + 2
            return "\n\n".join(parts)

    # ---- internals (under the lock) ---------------------------------------------------------

    @staticmethod
    def _capture(text: str, title: str, process: str, now: float) -> _Capture | None:
        raw = [line for line in (text or "").splitlines()]
        keys: list[str] = []
        lines: list[str] = []
        seen: set[str] = set()
        for line in raw:
            key_list = normalize_keys([line])
            if not key_list or key_list[0] in seen:
                continue
            seen.add(key_list[0])
            keys.append(key_list[0])
            lines.append(" ".join(line.split()))
        if len(keys) < MIN_LINES:
            return None
        file = file_in_title(title)
        return _Capture(
            text=text or "", lines=tuple(lines), keys=keys, idents=identifiers(text or ""),
            file=file, project=project_in_title(title, file), title=" ".join((title or "").split()),
            process=(process or "").strip(), at=float(now),
        )

    def _place(self, cap: _Capture, *, allow_foreign: bool) -> View | None:
        key_set = cap.key_set
        views = self._views
        # 1. the file named in the title
        if cap.file:
            for view in views:
                if view.file == cap.file:
                    return self._settle(view, cap, "same")
        # 2. the same window title (a page scrolled, a pane re-read) -- when the content agrees at all
        if cap.title:
            for view in views:
                if view.title == cap.title and containment(key_set, view.keys) >= RELATED_LINES:
                    return self._settle(view, cap, "same")
        # 3. a re-reading of a view, judged on its distinctive lines (the shared chrome of an app --
        #    menus, the file tree, the tabs -- must not make the next problem look like this one)
        content = cap.content_keys
        best: View | None = None
        best_score = 0.0
        for view in views:
            score = containment(content, view.keys)
            if score > best_score:
                best, best_score = view, score
        if best is not None and best_score >= SAME_VIEW and (not cap.file or not best.file):
            return self._settle(best, cap, "same")
        # 4. another file / pane of the task -- a development window's file counts as one unless its
        #    title names a DIFFERENT project than the files already here (a new project, no checkpoint)
        projects = {view.project for view in views if view.project}
        new_project = bool(cap.project) and bool(projects) and cap.project not in projects
        related = not views or (not new_project and (bool(cap.file) or any(view.related_to(cap) for view in views)))
        if related or not allow_foreign:
            view = View()
            views.append(view)
            self._evict(keep=view)
            return self._settle(view, cap, "new")
        # 5. foreign: a run of task-like, mutually related foreign captures restarts the workspace
        #    (judged on the raw lines: a one-character "}" is dropped from the normalised ones)
        if not looks_like_task(cap.text.splitlines(), cap.text):
            self._foreign = []
            self.last_outcome = "foreign"
            return None
        if self._foreign and not self._foreign_related(self._foreign[-1], cap):
            self._foreign = []
        self._foreign.append(cap)
        if len(self._foreign) < FOREIGN_RESET:
            self.last_outcome = "foreign"
            return None
        run, self._foreign = self._foreign, []
        self._views = []
        self._current = None
        placed: View | None = None
        for old in run:
            placed = self._place(old, allow_foreign=False)
        self.last_outcome = "reset"
        return placed

    @staticmethod
    def _foreign_related(prev: _Capture, cap: _Capture) -> bool:
        if cap.file and cap.file == prev.file:
            return True
        if cap.project and cap.project == prev.project:
            return True
        return (containment(cap.key_set, prev.key_set) >= RELATED_LINES
                or len(cap.idents & prev.idents) >= RELATED_IDENTIFIERS)

    def _settle(self, view: View, cap: _Capture, outcome: str) -> View:
        view.merge(cap)
        self._current = view
        self._foreign = []
        self.last_outcome = outcome
        return view

    def _evict(self, *, keep: View) -> None:
        while len(self._views) > MAX_VIEWS:
            victims = [v for v in self._views if v is not keep and v is not self._current]
            if not victims:
                return
            oldest = min(victims, key=lambda v: v.last_seen)
            self._views.remove(oldest)

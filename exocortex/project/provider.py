"""The one object the assistant and the dashboard ask for on-demand context.

:class:`ProjectContext` reads ``[project]`` live on every call: the toolkit catalog
(:mod:`exocortex.project.toolkit`, re-scanned on the configured cadence by :meth:`refresh`) and
the repository index (:mod:`exocortex.project.repo`, rebuilt when the repository changes --
:meth:`reconfigure`, and on every :meth:`refresh`). The repository is ``[project].repo_root`` when
set; otherwise, with ``repo_auto``, the directory Claude Code was most recently run in
(:mod:`exocortex.project.detect`), so a toolkit installed in the project being worked on is found
with no setup. Every method returns "" when its feature is off or has nothing to say, so a caller
can always just append what it gets.
"""
from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from pathlib import Path

from exocortex.config import Config
from exocortex.project.detect import detect_repo_root
from exocortex.project.repo import RepoIndex, extract_terms, render_hits
from exocortex.project.toolkit import ToolkitCatalog, ToolkitWatcher, check_prompt, lookup, render_catalog

log = logging.getLogger(__name__)

#: The auto-detected repository is looked up again at most this often (seconds).
_AUTO_TTL_S = 15.0


class ProjectContext:
    def __init__(self, cfg: Config, clock: Callable[[], float] | None = None, *, home: Path | None = None) -> None:
        self._cfg = cfg
        self._clock = clock or time.monotonic
        self._lock = threading.RLock()
        self._home = home
        c = cfg.project
        self._auto: str = ""
        self._auto_at: float | None = None
        self._watcher = ToolkitWatcher(
            self.effective_root, user_layer=lambda: self._cfg.project.toolkit_user_layer,
            toolkit_dir=lambda: self._cfg.project.toolkit_dir, rescan_s=lambda: self._cfg.project.rescan_s,
            clock=self._clock, home=home,
        )
        self._index: RepoIndex | None = None
        self._index_root = ""
        self._index_exclude: tuple[str, ...] = tuple(c.repo_exclude)
        self.reconfigure()

    # ---- which repository ------------------------------------------------------------------
    def effective_root(self) -> str:
        """``[project].repo_root`` when set, else (``repo_auto``) the directory Claude Code last
        ran in, else "". The auto-detected one is cached for a few seconds."""
        c = self._cfg.project
        explicit = (c.repo_root or "").strip()
        if explicit:
            return explicit
        return self._auto_root() if c.repo_auto else ""

    @property
    def repo_source(self) -> str:
        """"config" (an explicit repo_root), "auto" (detected from Claude Code) or "" (none)."""
        c = self._cfg.project
        if (c.repo_root or "").strip():
            return "config"
        return "auto" if c.repo_auto and self._auto_root() else ""

    def _auto_root(self) -> str:
        now = self._clock()
        with self._lock:
            if self._auto_at is not None and now - self._auto_at < _AUTO_TTL_S:
                return self._auto
        try:
            hours = max(0, int(self._cfg.project.repo_auto_max_age_h))
        except (TypeError, ValueError):
            hours = 12
        try:
            found = detect_repo_root(self._home, max_age_s=hours * 3600.0)
        except Exception:  # noqa: BLE001 - detection is best effort
            log.debug("repo auto-detection failed", exc_info=True)
            found = None
        with self._lock:
            self._auto = str(found) if found is not None else ""
            self._auto_at = now
            return self._auto

    # ---- configuration ------------------------------------------------------------------
    def reconfigure(self) -> None:
        """Apply a changed ``[project]``: forget the auto-detected repository, rebuild the repo
        index when the root (or the exclude list) changed, and force the next :meth:`refresh` to
        rescan the toolkit."""
        with self._lock:
            self._auto_at = None
            self._sync_index()
            self._watcher.refresh(force=True)

    def _sync_index(self) -> None:
        root = self.effective_root()
        exclude = tuple(self._cfg.project.repo_exclude)
        with self._lock:
            if root != self._index_root or exclude != self._index_exclude or (root and self._index is None):
                self._index = RepoIndex(root, exclude=exclude, clock=self._clock) if root else None
                self._index_root = root
                self._index_exclude = exclude

    def refresh(self, *, force: bool = False) -> bool:
        """Re-check the repository and rescan the toolkit when due; True when the catalog (or the
        repository it was found in) changed -- an install, an update, another project."""
        with self._lock:
            try:
                self._sync_index()
                return self._watcher.refresh(force=force)
            except Exception:  # noqa: BLE001 - a scan must never break a tick
                log.exception("toolkit rescan failed")
                return False

    # ---- toolkit --------------------------------------------------------------------------
    @property
    def catalog(self) -> ToolkitCatalog:
        with self._lock:
            return self._watcher.catalog

    @property
    def has_toolkit(self) -> bool:
        return bool(self._cfg.project.toolkit) and not self.catalog.is_empty

    def toolkit_summary(self) -> str:
        """"7 skills, 3 agents, 5 rules, 2 hooks (D:/repo)" for a toast / the doctor; "" when none."""
        cat = self.catalog
        if cat.is_empty:
            return ""
        parts = cat.summary() or "house rules only"
        return f"{parts} ({cat.root})" if cat.root else parts

    def context_summary(self) -> str:
        """One line for a toast: the repository and where it came from, then the toolkit found in
        it -- "Scenario1 (detected from Claude Code): toolkit 8 skills, 5 agents"."""
        root = self.repo_root
        source = {"config": "[project].repo_root", "auto": "detected from Claude Code"}.get(self.repo_source, "")
        if root is None:
            return "no repository (set one: View > Repository, or [project].repo_root)"
        cat = self.catalog
        toolkit = (cat.summary() or "house rules only") if not cat.is_empty else "no toolkit installed"
        return f"{root.name}" + (f" ({source})" if source else "") + f": {toolkit}"

    def toolkit_block(self) -> str:
        """The ``<toolkit>`` catalog for a system prompt, or ""."""
        if not self._cfg.project.toolkit:
            return ""
        cat = self.catalog
        if cat.is_empty:
            return ""
        body = render_catalog(cat, max_chars=max(300, int(self._cfg.project.toolkit_max_chars)))
        return f"<toolkit>\n{body}\n</toolkit>" if body else ""

    def toolkit_example(self) -> str:
        """A real installed command (``/recon [topic]``) to show the model a prompt's shape."""
        if not self._cfg.project.toolkit:
            return ""
        return self.catalog.example_command()

    def check_prompt(self, text: str) -> bool:
        """Whether a suggested prompt names only skills / agents that are installed (always True
        when the toolkit catalog is off or empty)."""
        if not self._cfg.project.toolkit:
            return True
        return check_prompt(self.catalog, text)

    def toolkit_lookup(self, *texts: str, max_chars: int = 3000) -> str:
        """The ``<toolkit_detail>`` block: the full text of the parts `texts` mention, or ""."""
        c = self._cfg.project
        if not (c.toolkit and c.toolkit_lookup):
            return ""
        cat = self.catalog
        if cat.is_empty:
            return ""
        body = lookup(cat, "\n".join(t for t in texts if t), max_chars=max_chars)
        return f"<toolkit_detail>\n{body}\n</toolkit_detail>" if body else ""

    # ---- repository -----------------------------------------------------------------------
    @property
    def repo_root(self) -> Path | None:
        with self._lock:
            return self._index.root if self._index is not None else None

    @property
    def repo_available(self) -> bool:
        with self._lock:
            return self._index is not None and self._index.available

    def repo_terms(self, *texts: str) -> list[str]:
        return extract_terms(*texts, limit=8)

    def repo_block(self, *texts: str, time_budget_s: float = 1.5) -> str:
        """The ``<repo>`` block: lines from the repository that match the identifiers / file
        names `texts` mention, within ``[project].repo_max_chars`` / ``repo_max_files``; "" when
        the lookup is off, no root is set, nothing identifier-like was mentioned, or nothing
        matched. Runs a bounded file scan (`time_budget_s`): call it off the UI thread."""
        c = self._cfg.project
        if not c.repo_search:
            return ""
        with self._lock:
            index = self._index
        if index is None or not index.available:
            return ""
        terms = extract_terms(*texts, limit=8)
        if not terms:
            return ""
        try:
            hits = index.search(terms, max_files=max(1, int(c.repo_max_files)), time_budget_s=time_budget_s)
        except Exception:  # noqa: BLE001 - a lookup must never break a prompt
            log.exception("repo lookup failed")
            return ""
        body = render_hits(hits, max_chars=max(100, int(c.repo_max_chars)))
        if not body:
            return ""
        return f"<repo root=\"{index.root.name}\">\n{body}\n</repo>"

"""The one object the assistant and the dashboard ask for on-demand context.

:class:`ContextProvider` reads ``[context]`` live on every call: the toolkit catalog
(:mod:`exocortex.context.toolkit`, re-scanned on the configured cadence by :meth:`refresh`) and
the repository index (:mod:`exocortex.context.repo`, rebuilt when ``repo_root`` changes --
:meth:`reconfigure`). Every method returns "" when its feature is off or has nothing to say, so
a caller can always just append what it gets.
"""
from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from pathlib import Path

from exocortex.config import Config
from exocortex.context.repo import RepoIndex, extract_terms, render_hits
from exocortex.context.toolkit import ToolkitCatalog, ToolkitWatcher, lookup, render_catalog

log = logging.getLogger(__name__)


class ContextProvider:
    def __init__(self, cfg: Config, clock: Callable[[], float] | None = None, *, home: Path | None = None) -> None:
        self._cfg = cfg
        self._clock = clock or time.monotonic
        self._lock = threading.RLock()
        self._home = home
        c = cfg.context
        self._watcher = ToolkitWatcher(
            lambda: self._cfg.context.repo_root, user_layer=lambda: self._cfg.context.toolkit_user_layer,
            toolkit_dir=lambda: self._cfg.context.toolkit_dir, rescan_s=lambda: self._cfg.context.rescan_s,
            clock=self._clock, home=home,
        )
        self._index: RepoIndex | None = None
        self._index_root = ""
        self._index_exclude: tuple[str, ...] = tuple(c.repo_exclude)
        self.reconfigure()

    # ---- configuration ------------------------------------------------------------------
    def reconfigure(self) -> None:
        """Apply a changed ``[context]``: rebuild the repo index when the root (or the exclude
        list) changed, and force the next :meth:`refresh` to rescan the toolkit."""
        c = self._cfg.context
        root = (c.repo_root or "").strip()
        with self._lock:
            exclude = tuple(c.repo_exclude)
            if root != self._index_root or exclude != self._index_exclude or (root and self._index is None):
                self._index = RepoIndex(root, exclude=exclude, clock=self._clock) if root else None
                self._index_root = root
                self._index_exclude = exclude
            self._watcher.refresh(force=True)

    def refresh(self, *, force: bool = False) -> bool:
        """Rescan the toolkit when due; True when the catalog changed (an install / update)."""
        with self._lock:
            try:
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
        return bool(self._cfg.context.toolkit) and not self.catalog.is_empty

    def toolkit_summary(self) -> str:
        """"7 skills, 3 agents, 5 rules, 2 hooks (D:/repo)" for a toast / the doctor; "" when none."""
        cat = self.catalog
        if cat.is_empty:
            return ""
        parts = cat.summary() or "house rules only"
        return f"{parts} ({cat.root})" if cat.root else parts

    def toolkit_block(self) -> str:
        """The ``<toolkit>`` catalog for a system prompt, or ""."""
        if not self._cfg.context.toolkit:
            return ""
        cat = self.catalog
        if cat.is_empty:
            return ""
        body = render_catalog(cat, max_chars=max(300, int(self._cfg.context.toolkit_max_chars)))
        return f"<toolkit>\n{body}\n</toolkit>" if body else ""

    def toolkit_lookup(self, *texts: str, max_chars: int = 3000) -> str:
        """The ``<toolkit_detail>`` block: the full text of the parts `texts` mention, or ""."""
        c = self._cfg.context
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
        names `texts` mention, within ``[context].repo_max_chars`` / ``repo_max_files``; "" when
        the lookup is off, no root is set, nothing identifier-like was mentioned, or nothing
        matched. Runs a bounded file scan (`time_budget_s`): call it off the UI thread."""
        c = self._cfg.context
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

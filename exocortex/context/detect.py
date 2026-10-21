"""Which repository is being worked on, found from Claude Code's own session logs.

Claude Code keeps one folder per project under ``~/.claude/projects`` with a ``.jsonl`` log per
session; every log records the ``cwd`` the session ran in. The most recently written log therefore
names the repository the user is working in *right now* -- which is where the toolkit is installed
and what the answers and the dashboard should know about, without the user typing a path.

Best effort and read-only: an unreadable folder, a log without a ``cwd`` or a directory that no
longer exists is skipped, and nothing here ever raises. Scratch sessions (a ``cwd`` under the
system temp directory) are ignored. ``[context].repo_root`` always wins over this.
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path

#: ``~/.claude/projects`` relative to the home directory.
PROJECTS_REL: tuple[str, ...] = (".claude", "projects")
#: How many lines of a log are read looking for its ``cwd`` (it is in the first few).
_HEAD_LINES = 80


def _cwd_of(log: Path) -> str | None:
    """The first ``cwd`` recorded in the head of the session log `log`."""
    try:
        with log.open("r", encoding="utf-8", errors="replace") as fh:
            for _ in range(_HEAD_LINES):
                line = fh.readline()
                if not line:
                    break
                if '"cwd"' not in line:
                    continue
                try:
                    obj = json.loads(line)
                except ValueError:
                    continue
                cwd = obj.get("cwd") if isinstance(obj, dict) else None
                if isinstance(cwd, str) and cwd.strip():
                    return cwd.strip()
    except OSError:
        return None
    return None


def _newest_log(folder: Path) -> tuple[float, Path] | None:
    """The most recently written ``*.jsonl`` directly in `folder`, with its mtime."""
    best: tuple[float, Path] | None = None
    try:
        with os.scandir(folder) as it:
            for entry in it:
                if not entry.name.endswith(".jsonl"):
                    continue
                try:
                    mtime = entry.stat().st_mtime
                except OSError:
                    continue
                if best is None or mtime > best[0]:
                    best = (mtime, Path(entry.path))
    except OSError:
        return None
    return best


def _is_scratch(path: Path) -> bool:
    try:
        temp = Path(tempfile.gettempdir()).resolve()
        resolved = path.resolve()
    except OSError:
        return False
    return resolved == temp or temp in resolved.parents


def recent_projects(
    home: Path | None = None, *, max_age_s: float = 12 * 3600.0, limit: int = 5, now: float | None = None,
) -> list[tuple[Path, float]]:
    """The directories Claude Code sessions ran in lately, newest first, as ``(directory,
    last activity)``: only ones that still exist, are not scratch space, and were active within
    `max_age_s` seconds (0 = no age limit)."""
    base = (home or Path.home()).joinpath(*PROJECTS_REL)
    scored: list[tuple[float, Path]] = []
    try:
        with os.scandir(base) as it:
            folders = [Path(e.path) for e in it if e.is_dir()]
    except OSError:
        return []
    for folder in folders:
        newest = _newest_log(folder)
        if newest is not None:
            scored.append(newest)
    scored.sort(key=lambda t: -t[0])
    clock = time.time() if now is None else now
    out: list[tuple[Path, float]] = []
    seen: set[str] = set()
    for mtime, log in scored:
        if max_age_s > 0 and clock - mtime > max_age_s:
            break
        cwd = _cwd_of(log)
        if cwd is None:
            continue
        path = Path(cwd)
        try:
            if not path.is_dir() or _is_scratch(path):
                continue
        except OSError:
            continue
        key = os.path.normcase(str(path))
        if key in seen:
            continue
        seen.add(key)
        out.append((path, mtime))
        if len(out) >= max(1, limit):
            break
    return out


def detect_repo_root(home: Path | None = None, *, max_age_s: float = 12 * 3600.0, now: float | None = None) -> Path | None:
    """The repository Claude Code was most recently run in, or None."""
    found = recent_projects(home, max_age_s=max_age_s, limit=1, now=now)
    return found[0][0] if found else None

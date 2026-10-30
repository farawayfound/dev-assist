"""Capabilities: what the model may do on its own to gather context (2026-10-07).

The policy, the tools and the project maps behind the Models window's "Capabilities" tab -- see
SPEC.md "Capabilities: tool use, scoped file / command / web / MCP access, project maps". Qt-free
and model-free (stdlib + the two transport helpers :mod:`exocortex.webfetch` and
:mod:`exocortex.mcp`, both imported lazily so a broken one degrades to an "unavailable" result
instead of breaking the app), so every piece is unit-testable on a ``tmp_path``.

Three things live here:

- **The policy and the tools** -- :class:`ToolRunner`, the ONE object the Assistant talks to:
  which tools a job of a given kind may call (:meth:`ToolRunner.specs`), the budget it gets
  (:meth:`ToolRunner.budget`) and the execution of one call under the ``[capabilities]`` policy
  (:meth:`ToolRunner.run`, which never raises). The read tools (``read_file`` / ``list_dir`` /
  ``find_files`` / ``search_text``) serve every covered kind; the active ones (``write_file`` /
  ``edit_file`` / ``run_command`` / ``fetch_url`` / ``web_search`` / ``mcp:<server>:<tool>``)
  serve the ask box only -- the one kind whose question the user typed (``ACTIVE_KINDS``).
- **The maps** -- :func:`build_map` walks the readable folders (excludes honoured, junctions never
  followed) and summarises every file on a basic understanding with no model call;
  :func:`render_map` turns one into the bounded ``<map>`` text every tool-enabled prompt carries;
  :func:`save_map` / :func:`load_map` keep it as JSON together with the NARROWING subset of the
  settings it was made with (:func:`settings_snapshot` / :func:`apply_snapshot`).
- **The dataclasses** the Assistant and the prompts module share (:class:`ToolSpec`,
  :class:`ToolCall`, :class:`ToolBudget`, :class:`ToolResult`).

Every result the model sees is TEXT that went through :func:`sanitize_untrusted` (a file, a page,
a server reply can contain ChatML special tokens or a forged ``<tool>`` call); a traceback never
reaches a result, only the message. Nothing here imports Qt.

Scope rules the walkers and the path resolver share: ``os.scandir`` only (``os.walk`` /
``Path.rglob`` follow junctions on Windows), a symlink / junction / cloud placeholder is never
followed, the excluded names are never listed, searched, read or indexed, and the app's own files
(``config.toml``, the model / context / map / session folders) are protected whatever the tab says.
The command allowlist is a guard against accidents, not a sandbox -- ``allow_all_commands`` is the
explicit opt-in that lifts it.
"""
from __future__ import annotations

import codecs
import copy
import fnmatch
import hashlib
import json
import logging
import math
import os
import re
import signal
import stat as stat_mod
import subprocess
import tempfile
import threading
import time
import unicodedata
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from exocortex.config import DEFAULT_READ_EXCLUDE, SAFE_COMMANDS, CapabilitiesConfig, Config

log = logging.getLogger(__name__)


class CapabilityError(Exception):
    """A readable failure of the maps / policy layer (a corrupt map file, a bad root)."""


#: The read tools: every covered kind may call them (the untrusted-text kinds included).
READ_TOOLS: frozenset[str] = frozenset({"read_file", "list_dir", "find_files", "search_text"})
#: The active tools (side effects, the network): the ask box only. MCP tools (``mcp:*``) count as
#: active too; they are named per server and are not enumerable here.
ACTIVE_TOOLS: frozenset[str] = frozenset({"write_file", "edit_file", "run_command", "fetch_url", "web_search"})
#: The kinds whose question text the USER typed -- the only ones that get the active tools.
ACTIVE_KINDS: frozenset[str] = frozenset({"ask"})
#: The kinds someone is waiting on live (an answer to the other speaker or to the screen): they
#: get the ``live_*`` limits of ``[capabilities]`` and read tools only.
LIVE_KINDS: frozenset[str] = frozenset({"auto", "manual", "screen", "suggest", "solve", "code", "text"})
#: Kinds that never get tools, whatever the config says.
NEVER_KINDS: frozenset[str] = frozenset({"complete", "explain"})
#: The prefix of an MCP tool's name: ``mcp:<server>:<tool>`` (split on the first two colons).
MCP_PREFIX = "mcp:"


@dataclass(frozen=True)
class ToolSpec:
    """One tool as the prompt lists it: ``name(arg: what, ...): description``."""

    name: str
    description: str
    args: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class ToolCall:
    """One parsed call: the tool's name, its arguments and the raw payload the model wrote."""

    name: str
    args: dict[str, Any]
    raw: str = ""


@dataclass
class ToolBudget:
    """What one answer may still spend (built by :meth:`ToolRunner.budget`, mutated by
    :meth:`ToolRunner.run` and the Assistant's loop). `allowed` is the exact set of tool names the
    prompt listed for this kind -- a call outside it is refused whatever the switches say;
    `chars` is the room ONE result may take; `deadline` a `clock()` value after which no more
    calls run; `web_left` the fetches + searches left; `read_done` flips once any local data
    (a file, a listing, a search hit, an MCP reply) entered the answer, after which the web tools
    are refused (the exfiltration guard); `seen` maps ``name + json(args)`` to the text already
    returned, so a repeated call is answered from memory, free of charge -- until a call with side
    effects (``write_file`` / ``edit_file`` / ``run_command`` / an MCP tool) runs: it empties
    `seen`, because every earlier read may be stale now."""

    kind: str
    allowed: frozenset[str]
    calls_left: int
    chars: int
    deadline: float
    web_left: int
    read_done: bool = False
    seen: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolResult:
    """What one call produced. `state` is "done" (the text is the tool's output), "error" (the
    tool ran and failed: the text says why) or "denied" (the policy refused it: the text says
    why); `detail` is the one-line size / reason the event shows, `summary` the one line that
    names what the call was about (the path, the command, the url, the query)."""

    text: str
    state: str
    detail: str = ""
    summary: str = ""


# ---------------------------------------------------------------------------------------------
# maps
# ---------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class FileEntry:
    """One indexed file: its path relative to its root (forward slashes), the root's index in
    `CapabilityMap.roots`, size in bytes, kind (a language / format name, "binary" or "other")
    and a one-line summary (<= 120 chars, sanitised; "" for binaries and oversized files)."""

    path: str
    root: int
    size: int
    kind: str
    summary: str


@dataclass(frozen=True)
class DirEntry:
    """One indexed directory: its relative path ("" = the root itself), the root's index, how many
    files it holds directly, their bytes, and a count per kind."""

    path: str
    root: int
    files: int
    bytes: int
    kinds: dict[str, int]


@dataclass
class CapabilityMap:
    """A project map: the roots it was built from, when, the narrowing settings snapshot, the
    entries, the rendered prompt text and whether the walk was cut at `map_max_files`."""

    name: str
    roots: list[str]
    indexed_at: str
    settings: dict[str, Any]
    files: list[FileEntry]
    dirs: list[DirEntry]
    text: str = ""
    truncated: bool = False


#: The ``[capabilities]`` keys a map stores and `load_capability_map` applies -- the ones that
#: can only NARROW or re-scope what the model may do. Never stored, never applied:
#: ``allow_all_commands``, ``web_allow_local``, ``mcp`` (the server specs: commands, urls, env),
#: ``command_cwd``, ``web_search_url``, ``maps_dir``, ``active_map`` and the ``*_timeout_s``.
SNAPSHOT_KEYS: tuple[str, ...] = (
    "read_files", "edit_files", "execute_commands", "fetch_web", "mcp_servers", "apply_to",
    "read_roots", "read_exclude", "read_max_file_chars", "edit_roots", "edit_exclude", "edit_backup",
    "allowed_commands", "web_allowed_domains", "web_blocked_domains",
    "max_tool_calls", "tool_result_chars", "tool_time_s",
    "live_max_tool_calls", "live_tool_result_chars", "live_tool_time_s",
    "web_max_fetches", "web_page_chars", "map_max_files", "map_tokens",
)



# ---------------------------------------------------------------------------------------------
# sanitising untrusted text (contract 3.8)
# ---------------------------------------------------------------------------------------------
#: The broken bar (U+00A6): visible, cheap and never part of a special token.
_BAR = "\u00a6"
_RE_PIPE = re.compile(r"<\|")
_RE_TAG = re.compile(r"<(/?)(start_of_turn|end_of_turn|think|tools?|map)>", re.IGNORECASE)
# `(?!bar)` keeps the function idempotent: the prompts module sanitises a second time, defensively.
_RE_RESULT = re.compile(r"<(/?)(result)(?!" + _BAR + ")", re.IGNORECASE)
_RE_CHOICE = re.compile(r"(?m)^([ \t]*)\[\[CHOICE\]\]")


def sanitize_untrusted(text: str) -> str:
    """Defuse the control markers in text the model did not write (a file, a page, a command's
    output, a server reply, a summary in a map): ``<|`` (every ChatML special token) becomes
    ``<`` + U+00A6, ``<start_of_turn>`` / ``<end_of_turn>`` / ``<think>`` / ``<tool>`` /
    ``<tools>`` / ``<map>`` (and their closing forms) get a U+00A6 before the ``>``, ``<result`` /
    ``</result>`` likewise, and a ``[[CHOICE]]`` at a line start becomes ``[CHOICE]``. Idempotent;
    case-insensitive for the tags. (``<tools>`` and ``<map>`` are the prompt's own block tags: a
    file, a server reply or a stored map must not be able to close them early.)"""
    if not text:
        return ""
    if not isinstance(text, str):
        text = str(text)
    if "<" not in text and "[[" not in text:
        return text
    out = _RE_PIPE.sub("<" + _BAR, text)
    out = _RE_TAG.sub(lambda m: f"<{m.group(1)}{m.group(2)}{_BAR}>", out)
    out = _RE_RESULT.sub(lambda m: f"<{m.group(1)}{m.group(2)}{_BAR}", out)
    out = _RE_CHOICE.sub(r"\1[CHOICE]", out)
    return out


#: The other block tags the prompt wraps its context in. Only the TEXT OF A MAP gets these defused
#: (a README heading ``</map></earlier>`` would otherwise close the block early); a file the model
#: reads with ``read_file`` keeps its HTML untouched.
_RE_BLOCK = re.compile(
    r"<(\s*/?\s*)(map|earlier|reference|lookup|transcript|screen|choices|previous|tools)(?=[\s>/])", re.IGNORECASE)


def sanitize_map_text(text: str) -> str:
    """:func:`sanitize_untrusted` plus the prompt's block tags (``<map>``, ``<earlier>``,
    ``<reference>``, ``<lookup>``, ``<transcript>``, ``<screen>``, ``<choices>``, ``<previous>``,
    ``<tools>``) -- for the text of a project map, which is built from file names and the first
    lines of files, and which a shared or hand-edited map file can fill with anything. Idempotent
    (the U+00A6 right after a tag name breaks the lookahead)."""
    return _RE_BLOCK.sub(lambda m: f"<{m.group(1)}{m.group(2)}{_BAR}", sanitize_untrusted(text))


# ---------------------------------------------------------------------------------------------
# small shared helpers
# ---------------------------------------------------------------------------------------------
class PathDenied(CapabilityError):
    """A path the policy refuses (outside the folders, excluded, protected, a link, ...). The
    message is the sentence the model sees and starts with ``denied:``."""


#: Windows file attributes that mean "not stored locally" (OFFLINE, RECALL_ON_OPEN,
#: RECALL_ON_DATA_ACCESS): opening such a placeholder would trigger a download.
_CLOUD_MASK = 0x1000 | 0x40000 | 0x400000
#: ``IO_REPARSE_TAG_MOUNT_POINT``: a junction. ``os.lstat`` does not call it a symlink.
_MOUNT_POINT_TAG = 0xA0000003
_PROBE_BYTES = 8192
_RESERVED_NAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL", "CLOCK$", "CONIN$", "CONOUT$"}
    | {f"COM{i}" for i in range(1, 10)} | {f"LPT{i}" for i in range(1, 10)}
)
_RE_SHORT_NAME = re.compile(r"^[^~]{1,8}~\d{1,7}(\.[^.~]*)?$")
_RE_DRIVE_RELATIVE = re.compile(r"^[A-Za-z]:(?![\\/])")
_RE_SEPARATORS = re.compile(r"[\\/]+")
_NAME_OK = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")

NUL_DEVICE = "NUL" if os.name == "nt" else "/dev/null"


def _is_link_stat(st: os.stat_result) -> bool:
    """A symlink or a junction, from an ``lstat`` result."""
    if stat_mod.S_ISLNK(st.st_mode):
        return True
    return getattr(st, "st_reparse_tag", 0) == _MOUNT_POINT_TAG


def _is_cloud_stat(st: os.stat_result) -> bool:
    """A cloud placeholder (Windows OneDrive & co.): opening it would download it."""
    return bool(getattr(st, "st_file_attributes", 0) & _CLOUD_MASK)


def _entry_is_link(entry: os.DirEntry[str]) -> bool:
    """`DirEntry` is a symlink or a junction (``is_junction`` exists from Python 3.12)."""
    try:
        if entry.is_symlink():
            return True
        junction = getattr(entry, "is_junction", None)
        return bool(junction()) if junction is not None else False
    except OSError:
        return True      # cannot tell: treat as a link (fail closed)


def _size_hint(n: int) -> str:
    """``812 B`` / ``3.2 KB`` / ``1.4 MB``."""
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n / (1024 * 1024):.1f} MB"


def _one_line(text: Any, limit: int = 120) -> str:
    """Collapse whitespace to one line and cut it to `limit` chars (with an ellipsis)."""
    line = " ".join(str(text).split())
    return line if len(line) <= limit else line[: max(0, limit - 1)] + "\u2026"


def _str_list(value: Any, *, limit: int = 500) -> list[str]:
    """`value` as a list of non-empty stripped strings; anything that is not a list -> []."""
    if not isinstance(value, (list, tuple)):
        return []
    out: list[str] = []
    for item in value:
        if isinstance(item, str) and item.strip():
            out.append(item.strip())
            if len(out) >= limit:
                break
    return out


def _clean_excludes(patterns: Iterable[str]) -> list[str]:
    """Casefolded, slash-normalised, non-empty exclude patterns."""
    out: list[str] = []
    for p in patterns:
        p = p.strip().replace("\\", "/").casefold().strip("/")
        if p:
            out.append(p)
    return out


def _parts_excluded(parts: Iterable[str], patterns: list[str]) -> str:
    """The first pattern excluding a path given as components relative to its folder ("" = none).
    A pattern without a slash is matched (fnmatch, casefolded) against every component; one with a
    slash against every leading part of the path (``docs/private`` hides that subtree)."""
    names = [p for p in parts if p not in ("", ".")]
    if not names or not patterns:
        return ""
    folded = [n.casefold() for n in names]
    for pat in patterns:
        if "/" in pat:
            for i in range(len(folded)):
                if fnmatch.fnmatchcase("/".join(folded[: i + 1]), pat):
                    return pat
        else:
            for n in folded:
                if fnmatch.fnmatchcase(n, pat):
                    return pat
    return ""


def _outermost(reals: list[Path]) -> list[int]:
    """The indexes of the entries of `reals` (real paths, identical ones already removed) that do
    not lie inside another entry, in their original order. Two listed folders where one holds the
    other would apply a slash-containing exclude (``docs/private``) relative to different roots, so
    only the outermost one is kept and its excludes govern the whole tree. ``Path`` comparison is
    case-insensitive on Windows."""
    return [i for i, p in enumerate(reals)
            if not any(j != i and p != q and p.is_relative_to(q) for j, q in enumerate(reals))]


def _glob_match(pattern: str, rel: str, name: str) -> bool:
    """Match a find_files / search_text glob. Without a slash it is matched against the name;
    with one against the relative posix path (``*`` crosses folders, ``**/`` may match nothing)."""
    pat = pattern.strip().replace("\\", "/").casefold()
    if not pat:
        return True
    if "/" not in pat:
        return fnmatch.fnmatchcase(name.casefold(), pat)
    r = rel.casefold()
    if fnmatch.fnmatchcase(r, pat):
        return True
    return pat.startswith("**/") and fnmatch.fnmatchcase(r, pat[3:])


def _oem_encoding() -> str:
    """The console code page cmd.exe's builtins write in (cp437 / cp850 / ...)."""
    if os.name == "nt":
        try:
            import ctypes

            return f"cp{ctypes.windll.kernel32.GetOEMCP()}"
        except Exception:      # noqa: BLE001 - any failure -> a safe default
            return "cp437"
    return "utf-8"


def _decode_output(data: bytes) -> str:
    """Decode a command's output: UTF-8 when it is valid (git, rg, python), else the OEM code
    page (cmd builtins); never raises. A UTF-8 BOM is dropped."""
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode(_oem_encoding(), errors="replace")


def _head_tail(text: str, limit: int) -> str:
    """`text` cut to about `limit` chars keeping its start and its end (the end of a command's
    output is where the error usually is)."""
    if len(text) <= limit:
        return text
    half = max(1, limit // 2)
    omitted = len(text) - 2 * half
    return text[:half] + f"\n(... {omitted} chars omitted ...)\n" + text[-half:]


def _atomic_write_bytes(path: Path, data: bytes, mode_from: os.stat_result | None = None) -> None:
    """Temp file in the same folder, fsync, then ``os.replace`` (all-or-nothing). Raw bytes: the
    text helpers elsewhere translate ``\\n`` to ``\\r\\n`` on Windows, which would rewrite every
    line ending of a file the model touched."""
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name[:40]}.", suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        if mode_from is not None:
            try:
                os.chmod(tmp, stat_mod.S_IMODE(mode_from.st_mode))
            except OSError:
                pass
        os.replace(tmp, path)
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------------------------
# the walkers (one scandir-based implementation for list_dir, find_files, search_text, the indexer)
# ---------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class _Ent:
    """One surviving directory entry: its name, its filesystem path, kind and size."""

    name: str
    path: str
    is_dir: bool
    size: int


def _scan_dir(dirpath: str, excl: list[str], rel_parts: tuple[str, ...], protected: frozenset[str]) -> list[_Ent]:
    """The entries of one folder that the policy lets through, sorted by casefolded name:
    excluded names, symlinks, junctions, cloud placeholders, protected paths and anything that
    is neither a file nor a folder are dropped; an unreadable folder is an empty one."""
    out: list[_Ent] = []
    try:
        it = os.scandir(dirpath)
    except OSError as exc:
        log.debug("scandir %s: %s", dirpath, exc)
        return out
    with it:
        while True:
            try:
                entry = next(it)
            except StopIteration:
                break
            except OSError:
                break
            try:
                name = entry.name
                if _parts_excluded(rel_parts + (name,), excl):
                    continue
                if _entry_is_link(entry):
                    continue
                is_dir = entry.is_dir(follow_symlinks=False)
                if not is_dir and not entry.is_file(follow_symlinks=False):
                    continue
                st = entry.stat(follow_symlinks=False)
                if _is_cloud_stat(st):
                    continue
                if protected and os.path.normcase(entry.path) in protected:
                    continue
                out.append(_Ent(name, entry.path, is_dir, 0 if is_dir else int(st.st_size)))
            except OSError:
                continue
    out.sort(key=lambda e: (e.name.casefold(), e.name))
    return out


class _Stop:
    """The "should the walk end?" callable: the answer's deadline (a ``clock()`` value) or a
    cancel event. `reason` says which one fired."""

    def __init__(self, deadline: float | None, clock: Callable[[], float], cancel: threading.Event | None) -> None:
        self._deadline = deadline
        self._clock = clock
        self._cancel = cancel
        self.reason = ""

    def __call__(self) -> bool:
        if self._cancel is not None and self._cancel.is_set():
            self.reason = "time limit" if getattr(self._cancel, "expired", False) else "cancelled"
            return True
        if self._deadline is not None and self._clock() > self._deadline:
            self.reason = "time limit"
            return True
        return False


def _walk(
    root: Path, start: Path, excl: list[str], protected: frozenset[str], stop: Callable[[], bool],
) -> Iterator[tuple[str, _Ent]]:
    """Walk below `start` (a folder inside `root`), yielding ``(path relative to root with forward
    slashes, entry)`` in a deterministic order: a folder's entries, then each sub-folder in turn.
    Explicit stack, no ``os.walk`` / ``rglob`` (they follow junctions on Windows); `stop` is polled
    every 64 entries. Lazy: the caller breaks out when its caps are reached."""
    try:
        base = start.relative_to(root).parts
    except ValueError:
        return
    stack: list[tuple[str, tuple[str, ...]]] = [(str(start), tuple(base))]
    n = 0
    while stack:
        dirpath, parts = stack.pop()
        pending: list[tuple[str, tuple[str, ...]]] = []
        for ent in _scan_dir(dirpath, excl, parts, protected):
            n += 1
            if n % 64 == 0 and stop():
                return
            sub = parts + (ent.name,)
            yield "/".join(sub), ent
            if ent.is_dir:
                pending.append((ent.path, sub))
        stack.extend(reversed(pending))


def _verify_entry(root: Path, ent_path: str, excl: list[str]) -> bool:
    """Defence in depth before a walked file is opened: its real path must still sit inside
    `root` and no component of the real path below the root may be excluded (a link swapped in
    while the walk ran, ``notes.txt -> .env``)."""
    try:
        real = Path(os.path.realpath(ent_path))
        rel = real.relative_to(root)
    except (OSError, ValueError):
        return False
    return not _parts_excluded(rel.parts, excl)


# ---------------------------------------------------------------------------------------------
# maps: slug, summaries, build, render, save / load, settings snapshot
# ---------------------------------------------------------------------------------------------
_SUMMARY_BYTES = 4096
_BIG_FILE = 256 * 1024
_SUMMARY_CHARS = 120

_EXT_KIND: dict[str, str] = {}
for _kind, _exts in {
    "python": ".py .pyw .pyi",
    "javascript": ".js .mjs .cjs .jsx",
    "typescript": ".ts .tsx .mts .cts",
    "markdown": ".md .markdown .mdx",
    "json": ".json .jsonc .jsonl .ipynb",
    "toml": ".toml",
    "yaml": ".yml .yaml",
    "text": ".txt .rst .log .csv .tsv .ini .cfg .conf .properties",
    "html": ".html .htm .xhtml .vue .svelte",
    "css": ".css .scss .sass .less",
    "shell": ".sh .bash .zsh .fish",
    "powershell": ".ps1 .psm1 .psd1",
    "batch": ".bat .cmd",
    "csharp": ".cs .csx",
    "cpp": ".cpp .cc .cxx .hpp .hh .hxx .ino",
    "c": ".c .h",
    "java": ".java .kt .kts .scala .groovy",
    "go": ".go",
    "rust": ".rs",
    "sql": ".sql",
    "binary": (".png .jpg .jpeg .gif .bmp .ico .webp .tif .tiff .psd .pdf .zip .7z .rar .gz .tgz .bz2 .xz "
               ".zst .tar .exe .dll .so .dylib .lib .obj .o .a .class .jar .pyd .whl .bin .dat .gguf .onnx "
               ".pt .pth .safetensors .npz .npy .mp3 .wav .flac .ogg .mp4 .mkv .avi .mov .webm .ttf .otf "
               ".woff .woff2 .sqlite .db .pkl .pickle .docx .xlsx .pptx .iso .img .msi .dmp"),
}.items():
    for _e in _exts.split():
        _EXT_KIND[_e] = _kind

#: A summary that reads like a credential is dropped (defence in depth: secret-bearing files are
#: excluded by name already, this catches a token pasted in a docstring or a comment).
_RE_SECRETISH = re.compile(
    r"(token|secret|passw|pwd|api[_-]?key|private[_-]?key|credential|authorization|bearer)\s*[=:]", re.IGNORECASE)
_RE_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


def _file_kind(name: str) -> str:
    """The kind of a file from its extension ("other" when unknown)."""
    low = name.casefold()
    if low.endswith(".env.example"):
        return "text"
    return _EXT_KIND.get(os.path.splitext(low)[1], "other")


def _first_line(text: str) -> str:
    """The first non-empty line that is not a shebang."""
    for line in text.splitlines():
        s = line.strip()
        if s and not s.startswith("#!"):
            return s
    return ""


def _json_top_keys(text: str, limit: int = 8) -> list[str]:
    """The top-level keys of a JSON object read from a (possibly cut) prefix: a tiny scanner that
    tracks string / depth state, so a document cut in the middle still yields its first keys."""
    s = text.lstrip("\ufeff \t\r\n")
    if not s.startswith("{"):
        return []
    keys: list[str] = []
    depth = 0
    i, n = 0, len(s)
    while i < n and len(keys) < limit:
        c = s[i]
        if c == '"':
            j = i + 1
            while j < n:
                if s[j] == "\\":
                    j += 2
                    continue
                if s[j] == '"':
                    break
                j += 1
            else:
                break                      # an unterminated string: the prefix ended inside it
            if depth == 1:
                k = j + 1
                while k < n and s[k] in " \t\r\n":
                    k += 1
                if k < n and s[k] == ":":
                    keys.append(s[i + 1:j])
            i = j + 1
            continue
        if c in "{[":
            depth += 1
        elif c in "}]":
            depth -= 1
            if depth <= 0:
                break
        i += 1
    return keys


def _toml_keys(text: str, limit: int = 8) -> list[str]:
    """Top-level keys (before the first table) then table names of a TOML prefix, in order."""
    out: list[str] = []
    in_table = False
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        m = re.match(r"^\[\[?\s*([^\]]+?)\s*\]\]?\s*(?:#.*)?$", s)
        if m:
            in_table = True
            if m.group(1) not in out:
                out.append(m.group(1))
        elif not in_table:
            k = re.match(r'^([A-Za-z0-9_.\-"]+)\s*=', s)
            if k and k.group(1) not in out:
                out.append(k.group(1))
        if len(out) >= limit:
            break
    return out


def _yaml_keys(text: str, limit: int = 8) -> list[str]:
    """Top-level (column 0) keys of a YAML prefix."""
    out: list[str] = []
    for line in text.splitlines():
        m = re.match(r"^([A-Za-z_][\w.\-]*):(?:\s|$)", line)
        if m and m.group(1) not in out:
            out.append(m.group(1))
            if len(out) >= limit:
                break
    return out


def _python_summary(text: str) -> str:
    """Module docstring's first line (else the first real comment) plus up to 6 top-level class /
    def names in brackets -- regexes on the first 4 KiB, no ``ast`` (the prefix may be cut)."""
    head = ""
    m = re.match(r'(?:[ \t]*(?:#[^\n]*)?\n)*[ \t]*[rRuU]?("""|\'\'\')(.*?)(?:\1|\Z)', text, re.S)
    if m:
        for line in m.group(2).splitlines():
            if line.strip():
                head = line.strip()
                break
    if not head:
        m1 = re.match(r'(?:[ \t]*(?:#[^\n]*)?\n)*[ \t]*[rRuU]?(["\'])([^\n]*?)\1', text)
        if m1:
            head = m1.group(2).strip()
    if not head:
        for line in text.splitlines():
            s = line.strip()
            if s.startswith("#") and not s.startswith("#!") and not re.match(
                    r"#\s*(-\*-|coding[:=]|type:|noqa|pylint|pragma|flake8|fmt:|isort:)", s, re.I):
                head = s.lstrip("#").strip()
                if head:
                    break
    names = re.findall(r"^(?:async\s+)?(?:class|def)\s+([A-Za-z_]\w*)", text, re.M)
    public = [n for n in names if not n.startswith("_")]
    shown = (public or names)[:6]
    tail = f" [{', '.join(shown)}]" if shown else ""
    if head or tail:
        return (head + tail).strip()
    return _first_line(text)


def _script_summary(text: str) -> str:
    """JS / TS: the first comment line, else the exported names, else the first line."""
    body = text
    for line in body.splitlines():
        s = line.strip()
        if not s or s in ("'use strict';", '"use strict";') or s.startswith("// @ts-") or s.startswith("#!"):
            continue
        if s.startswith("//"):
            c = s.lstrip("/").strip()
            if c:
                return c
        elif s.startswith("/*"):
            for c_line in body[body.find(s):].splitlines():
                c = c_line.strip().lstrip("/*").rstrip("*/").strip().lstrip("*").strip()
                if c:
                    return c
        break
    names = re.findall(
        r"^export\s+(?:default\s+)?(?:async\s+)?(?:class|function\*?|const|let|var|interface|type|enum)\s+"
        r"([A-Za-z_$][\w$]*)", text, re.M)
    if names:
        return "exports: " + ", ".join(dict.fromkeys(names[:6]))
    return _first_line(text)


def _summarize(kind: str, text: str) -> str:
    """The one-line heuristic summary of a file's first 4 KiB (see the contract, 3.6)."""
    try:
        if kind == "markdown":
            t = text
            if t.startswith("---"):
                end = t.find("\n---", 3)
                if end > 0:
                    t = t[end + 4:]
            m = re.search(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$", t, re.M)
            return m.group(1).strip() if m else _first_line(t)
        if kind == "python":
            return _python_summary(text)
        if kind in ("javascript", "typescript"):
            return _script_summary(text)
        if kind == "json":
            keys = _json_top_keys(text)
            return ("keys: " + ", ".join(keys)) if keys else _first_line(text)
        if kind == "toml":
            keys = _toml_keys(text)
            return ("keys: " + ", ".join(keys)) if keys else _first_line(text)
        if kind == "yaml":
            keys = _yaml_keys(text)
            return ("keys: " + ", ".join(keys)) if keys else _first_line(text)
    except Exception:      # noqa: BLE001 - a summary is best effort; never fail the walk
        log.debug("summary failed for kind %s", kind, exc_info=True)
    return _first_line(text)


def _clean_summary(raw: str) -> str:
    """A summary ready for the map: control characters out, one line, sanitised, <= 120 chars,
    "" when it reads like a credential."""
    s = _RE_CONTROL.sub(" ", raw or "")
    s = sanitize_untrusted(_one_line(s, _SUMMARY_CHARS * 2))
    if _RE_SECRETISH.search(s):
        return ""
    return _one_line(s, _SUMMARY_CHARS)


def map_slug(name: str) -> str:
    """The file-safe name of a map: casefolded ``[a-z0-9]`` runs joined by ``-``, plus ``-`` and
    the first 6 hex chars of the sha1 of the original name, so "My App" and "my-app" never share
    a file."""
    runs = re.findall(r"[a-z0-9]+", str(name).casefold())
    base = "-".join(runs)[:48].strip("-") or "map"
    return base + "-" + hashlib.sha1(str(name).encode("utf-8", "replace")).hexdigest()[:6]


def build_map(
    name: str, roots: list[str], *, excludes: list[str], max_files: int, settings: dict[str, Any],
    now: str | None = None, deadline: float | None = None, cancel: threading.Event | None = None,
    protected: Iterable[str | Path] = (),
) -> CapabilityMap:
    """Walk `roots` (``os.scandir``; excluded names, symlinks, junctions and cloud placeholders
    skipped, never descended into) and summarise every file on a basic understanding (the first
    4 KiB, heuristics per kind, no model call). Stops at `max_files` (``truncated``), at
    `deadline` (a ``time.monotonic()`` value) or when `cancel` is set; yields the GIL every 50
    files. `now` is the ISO timestamp to stamp (None = now). `protected` (optional, additive to
    the landed signature) lists paths that are never indexed -- the app's own folders; the engine
    passes ``ToolRunner.protected_paths()``. The returned map's ``text`` is empty: the caller
    renders it with :func:`render_map` for its budget. Raises :class:`CapabilityError` for a root
    that is not a folder."""
    excl = _clean_excludes(_str_list(list(excludes) if isinstance(excludes, (list, tuple)) else []))
    prot_paths = [Path(os.path.realpath(str(p))) for p in protected]
    prot = frozenset(os.path.normcase(str(p)) for p in prot_paths)
    stop = _Stop(deadline, time.monotonic, cancel)
    files: list[FileEntry] = []
    dirs: dict[tuple[int, str], list[Any]] = {}      # (root, rel) -> [files, bytes, kinds]
    truncated = False
    # Check every root first, then keep the OUTERMOST ones: a listed folder inside another listed
    # folder is covered by it (and by its exclude patterns, which are relative to ITS root), so it
    # is neither walked twice nor listed as a root of its own.
    given = [str(r) for r in roots]
    reals: list[Path] = []
    for raw_root in given:
        rp = Path(raw_root)
        if not rp.is_dir():
            raise CapabilityError(f"folder not found: {raw_root}")
        reals.append(Path(os.path.realpath(rp)))
    firsts = [i for i, p in enumerate(reals) if p not in reals[:i]]
    inner = _outermost([reals[i] for i in firsts])
    kept = [firsts[i] for i in inner]
    root_list = [given[i] for i in kept]
    for ri, gi in enumerate(kept):
        real = reals[gi]
        # the same test ToolRunner._roots applies: a root EQUAL to or INSIDE a protected path is out
        if any(real == q or real.is_relative_to(q) for q in prot_paths):
            continue
        dirs.setdefault((ri, ""), [0, 0, {}])
        for rel, ent in _walk(real, real, excl, prot, stop):
            if ent.is_dir:
                dirs.setdefault((ri, rel), [0, 0, {}])
                continue
            if len(files) >= max(1, int(max_files)):
                truncated = True
                break
            kind = _file_kind(ent.name)
            summary = ""
            # A big file or a known binary extension is listed with its size only (never opened).
            if ent.size <= _BIG_FILE and kind != "binary":
                if not _verify_entry(real, ent.path, excl):
                    continue                 # swapped for a link / escaped the root while we walked
                try:
                    with open(ent.path, "rb") as fh:
                        head = fh.read(_SUMMARY_BYTES)
                except OSError:
                    head = b""
                if b"\x00" in head:
                    kind = "binary"
                elif head:
                    text = head.decode("utf-8", errors="replace").replace("\r\n", "\n").lstrip("\ufeff")
                    summary = _clean_summary(_summarize(kind, text))
            files.append(FileEntry(rel, ri, ent.size, kind, summary))
            parent = rel.rsplit("/", 1)[0] if "/" in rel else ""
            slot = dirs.setdefault((ri, parent), [0, 0, {}])
            slot[0] += 1
            slot[1] += ent.size
            slot[2][kind] = slot[2].get(kind, 0) + 1
            if len(files) % 50 == 0:
                time.sleep(0)                # let the UI / audio threads in on a long walk
                if stop():
                    truncated = True
                    break
        if truncated:
            break
    if stop.reason:                          # the walker itself stopped on the deadline / cancel
        truncated = True
    dir_entries = [DirEntry(path, ri, v[0], v[1], dict(v[2])) for (ri, path), v in sorted(dirs.items())]
    return CapabilityMap(
        name=name, roots=root_list, indexed_at=now or datetime.now().isoformat(timespec="seconds"),
        settings=copy.deepcopy(settings) if isinstance(settings, dict) else {}, files=files,
        dirs=dir_entries, text="", truncated=truncated,
    )


_PRIORITY_NAMES = frozenset({
    "pyproject.toml", "package.json", "cargo.toml", "setup.py", "setup.cfg", "requirements.txt", "go.mod",
    "pom.xml", "build.gradle", "makefile", "dockerfile", "docker-compose.yml", "tsconfig.json",
})


def _file_priority(name: str, at_root: bool, kind: str = "") -> int:
    """README / SPEC first, then the root's other docs, then the root's project config, then entry
    points; binaries (screenshots, archives) last -- they say little and cost a line each."""
    low = name.casefold()
    if low.startswith(("readme", "spec")):
        return 0
    if at_root and low.endswith((".md", ".markdown", ".rst")):
        return 1
    if at_root and (low in _PRIORITY_NAMES
                    or (low.startswith("config") and low.endswith((".toml", ".json", ".yml", ".yaml", ".ini")))):
        return 2
    if low == "__main__.py" or re.match(r"^(main|app|index)\.", low):
        return 3
    return 5 if kind == "binary" else 4


def _dir_priority(name: str) -> int:
    return 0 if name.casefold() in ("docs", "doc", "documentation") else 1


def _kinds_text(kinds: dict[str, int]) -> str:
    """``python 12, markdown 3`` -- the biggest four kinds, more as an ellipsis."""
    ordered = sorted(kinds.items(), key=lambda kv: (-kv[1], kv[0]))
    shown = ", ".join(f"{k} {v}" for k, v in ordered[:4])
    return shown + (", ..." if len(ordered) > 4 else "")


def _file_line(f: FileEntry) -> str:
    base = f.path.rsplit("/", 1)[-1]
    if len(base) > 80:
        base = base[:79] + "\u2026"
    if f.summary:
        return f"  {base} -- {f.summary}"
    if f.kind == "binary":
        return f"  {base} (binary, {_size_hint(f.size)})"
    if f.size > _BIG_FILE:
        return f"  {base} ({_size_hint(f.size)})"
    return f"  {base}"


def render_map(m: CapabilityMap, *, max_chars: int) -> str:
    """The bounded ``<map>`` text: the root line(s), then a depth-first tree with directories as
    ``dir/ (N files: python 12, markdown 3)`` and files as ``  name -- summary``; README / docs /
    config / entry points first; when the budget runs out deeper directories collapse to their
    count line and the text ends with ``(... N more files; use list_dir / find_files /
    search_text)``. Never longer than `max_chars`; the same map renders to the same text."""
    max_chars = max(0, int(max_chars))
    roots = list(m.roots)
    multi = len(roots) > 1
    # ---- the tree: dir key -> counts, children, files
    counts: dict[tuple[int, str], DirEntry] = {(d.root, d.path): d for d in m.dirs}
    by_dir: dict[tuple[int, str], list[FileEntry]] = {}
    for f in m.files:
        parent = f.path.rsplit("/", 1)[0] if "/" in f.path else ""
        by_dir.setdefault((f.root, parent), []).append(f)
    keys: set[tuple[int, str]] = set(counts) | set(by_dir)
    for ri in range(len(roots)):
        keys.add((ri, ""))
    for ri, path in list(keys):                      # make sure every ancestor exists
        while "/" in path:
            path = path.rsplit("/", 1)[0]
            keys.add((ri, path))
    children: dict[tuple[int, str], list[str]] = {}
    for ri, path in keys:
        if path:
            parent = path.rsplit("/", 1)[0] if "/" in path else ""
            children.setdefault((ri, parent), []).append(path)
    order: list[tuple[int, str, int]] = []           # (root, path, depth) depth-first

    def visit(ri: int, path: str, depth: int) -> None:
        order.append((ri, path, depth))
        subs = sorted(children.get((ri, path), []),
                      key=lambda p: (_dir_priority(p.rsplit("/", 1)[-1]), p.rsplit("/", 1)[-1].casefold(), p))
        for sub in subs:
            visit(ri, sub, depth + 1)

    for ri in range(len(roots)):
        visit(ri, "", 0)

    def label(ri: int, path: str) -> str:
        if ri == 0:
            return (path + "/") if path else "./"
        base = roots[ri].replace("\\", "/").rstrip("/")
        return f"{base}/{path}/" if path else f"{base}/"

    def dir_line(ri: int, path: str) -> str:
        d = counts.get((ri, path))
        if d is None:
            fl = by_dir.get((ri, path), [])
            kinds: dict[str, int] = {}
            for f in fl:
                kinds[f.kind] = kinds.get(f.kind, 0) + 1
            n_files, kinds_map = len(fl), kinds
        else:
            n_files, kinds_map = d.files, d.kinds
        detail = f"{n_files} files" + (f": {_kinds_text(kinds_map)}" if kinds_map else "")
        return f"{label(ri, path)} ({detail})"

    total_files = len(m.files)
    when = (m.indexed_at or "")[:10]
    head = [f'Project map "{m.name}": {total_files} files in {len(keys)} folders'
            + (f", indexed {when}" if when else "") + (" (index cut: more files exist)" if m.truncated else "")]
    for i, r in enumerate(roots):
        head.append(f"Root{' ' + str(i + 1) if multi else ''}: {r.replace(chr(92), '/')}")
    if multi:
        head.append("Paths are relative to Root 1; the other roots are shown with their full path.")
    footer_more = "(... {n} more files; use list_dir / find_files / search_text)"
    footer_cut = "(... more files were not indexed; use list_dir / find_files / search_text)"

    used = sum(len(h) + 1 for h in head)
    reserve = len(footer_more.format(n=total_files)) + 1
    budget = max(0, max_chars - reserve)

    shown_dirs: list[tuple[int, str, int]] = []
    shown_set: set[tuple[int, str]] = set()
    index_of = {(ri, p): i for i, (ri, p, _d) in enumerate(order)}
    # The folders' count lines may take at most about half of what is left, so the files of the
    # top folders (README, entry points) always get room too.
    dir_budget = used + max(0, budget - used) * 0.55
    for ri, path, depth in sorted(order, key=lambda t: (t[2], index_of[(t[0], t[1])])):
        parent = (ri, path.rsplit("/", 1)[0] if "/" in path else "")
        if path and parent not in shown_set:
            continue
        cost = len(dir_line(ri, path)) + 1
        if used + cost > (dir_budget if path else budget) and shown_dirs:
            continue
        shown_dirs.append((ri, path, depth))
        shown_set.add((ri, path))
        used += cost

    chosen: dict[tuple[int, str], list[FileEntry]] = {}
    shown_files = 0
    for ri, path, _depth in sorted(shown_dirs, key=lambda t: (t[2], index_of[(t[0], t[1])])):
        fl = sorted(by_dir.get((ri, path), []),
                    key=lambda f: (_file_priority(f.path.rsplit("/", 1)[-1], path == "" and ri == 0, f.kind),
                                   f.path.rsplit("/", 1)[-1].casefold(), f.path))
        for f in fl:
            line = _file_line(f)
            if used + len(line) + 1 > budget:
                break
            chosen.setdefault((ri, path), []).append(f)
            used += len(line) + 1
            shown_files += 1

    lines = list(head)
    for ri, path, _depth in order:
        if (ri, path) not in shown_set:
            continue
        lines.append(dir_line(ri, path))
        lines.extend(_file_line(f) for f in chosen.get((ri, path), []))
    hidden = total_files - shown_files
    if hidden > 0:
        lines.append(footer_more.format(n=hidden))
    elif m.truncated:
        lines.append(footer_cut)
    text = sanitize_map_text("\n".join(lines))
    if len(text) > max_chars:
        text = text[: max(0, max_chars - 1)].rstrip() + ("\u2026" if max_chars > 0 else "")
    return text


def _atomic_write_text_utf8(path: Path, text: str) -> None:
    _atomic_write_bytes(path, text.encode("utf-8"))


def save_map(m: CapabilityMap, path: Path) -> Path:
    """Write `m` as JSON (``{"version": 1, "name", "roots", "indexed_at", "settings", "text",
    "truncated", "files": [...], "dirs": [...]}``), atomically; returns `path`."""
    path = Path(path)
    doc = {
        "version": 1, "name": m.name, "roots": list(m.roots), "indexed_at": m.indexed_at,
        "settings": m.settings, "text": m.text, "truncated": bool(m.truncated),
        "files": [{"path": f.path, "root": f.root, "size": f.size, "kind": f.kind, "summary": f.summary}
                  for f in m.files],
        "dirs": [{"path": d.path, "root": d.root, "files": d.files, "bytes": d.bytes, "kinds": d.kinds}
                 for d in m.dirs],
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_text_utf8(path, json.dumps(doc, ensure_ascii=False, separators=(",", ":")))
    except (OSError, TypeError, ValueError) as exc:
        raise CapabilityError(f"could not save the map {m.name!r}: {exc}") from exc
    return path


def _read_map_doc(path: Path) -> dict[str, Any]:
    """The parsed JSON of a map file, structurally checked. Raises :class:`CapabilityError`."""
    path = Path(path)
    try:
        raw = path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise CapabilityError(f"could not read the map {path.name}: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise CapabilityError(f"the map {path.name} is not valid UTF-8: {exc}") from exc
    try:
        doc = json.loads(raw)
    except ValueError as exc:
        raise CapabilityError(f"the map {path.name} is corrupt (not JSON): {exc}") from exc
    if not isinstance(doc, dict):
        raise CapabilityError(f"the map {path.name} is corrupt (not an object)")
    if doc.get("version") != 1:
        raise CapabilityError(f"the map {path.name} has an unsupported version: {doc.get('version')!r}")
    if not isinstance(doc.get("name"), str) or not doc["name"]:
        raise CapabilityError(f"the map {path.name} is corrupt (no name)")
    roots = doc.get("roots")
    if not isinstance(roots, list) or not all(isinstance(r, str) for r in roots):
        raise CapabilityError(f"the map {path.name} is corrupt (bad roots)")
    for key in ("files", "dirs"):
        if not isinstance(doc.get(key, []), list):
            raise CapabilityError(f"the map {path.name} is corrupt (bad {key})")
    return doc


def load_map(path: Path) -> CapabilityMap:
    """Read a map written by :func:`save_map`. Raises :class:`CapabilityError` (a readable
    message) for a missing / corrupt / wrong-version file; unknown keys are ignored."""
    doc = _read_map_doc(path)
    files: list[FileEntry] = []
    for e in doc.get("files", []):
        try:
            files.append(FileEntry(str(e["path"]), int(e["root"]), int(e["size"]), str(e["kind"]),
                                   str(e.get("summary", ""))))
        except (KeyError, TypeError, ValueError):
            raise CapabilityError(f"the map {Path(path).name} is corrupt (a bad file entry)") from None
    dirs: list[DirEntry] = []
    for e in doc.get("dirs", []):
        try:
            kinds = e.get("kinds", {})
            dirs.append(DirEntry(str(e["path"]), int(e["root"]), int(e["files"]), int(e["bytes"]),
                                 {str(k): int(v) for k, v in dict(kinds).items()}))
        except (KeyError, TypeError, ValueError, AttributeError):
            raise CapabilityError(f"the map {Path(path).name} is corrupt (a bad folder entry)") from None
    settings = doc.get("settings", {})
    return CapabilityMap(
        name=doc["name"], roots=list(doc["roots"]), indexed_at=str(doc.get("indexed_at", "")),
        settings=dict(settings) if isinstance(settings, dict) else {}, files=files, dirs=dirs,
        text=sanitize_map_text(str(doc.get("text", ""))), truncated=bool(doc.get("truncated", False)),
    )


def map_header(path: Path) -> dict[str, Any]:
    """The listing row of one map file without its entries: ``{"name", "path", "roots", "files",
    "dirs", "indexed_at", "truncated", "settings"}``. Raises :class:`CapabilityError` like
    :func:`load_map`."""
    doc = _read_map_doc(path)
    settings = doc.get("settings", {})
    return {
        "name": doc["name"], "path": str(path), "roots": list(doc["roots"]),
        "files": len(doc.get("files", [])), "dirs": len(doc.get("dirs", [])),
        "indexed_at": str(doc.get("indexed_at", "")), "truncated": bool(doc.get("truncated", False)),
        "settings": dict(settings) if isinstance(settings, dict) else {},
    }


# ---- the settings a map carries
_BOOL_KEYS = ("read_files", "edit_files", "execute_commands", "fetch_web", "mcp_servers", "edit_backup")
_LIST_KEYS = ("read_roots", "read_exclude", "edit_roots", "edit_exclude", "allowed_commands",
              "web_allowed_domains", "web_blocked_domains")
#: key -> (low, high) for the ints the tab edits; out of range = ignored, never clamped.
_INT_RANGES: dict[str, tuple[int, int]] = {
    "read_max_file_chars": (200, 1_000_000), "max_tool_calls": (0, 20), "tool_result_chars": (200, 100_000),
    "live_max_tool_calls": (0, 20), "live_tool_result_chars": (200, 100_000), "web_max_fetches": (0, 10),
    "web_page_chars": (200, 100_000), "map_max_files": (100, 50_000), "map_tokens": (100, 4000),
}
_FLOAT_RANGES: dict[str, tuple[float, float]] = {"tool_time_s": (1.0, 600.0), "live_tool_time_s": (1.0, 600.0)}
_APPLY_TO_VALUES = ("questions", "ask", "all")


def _server_entries(value: Any) -> list[dict[str, Any]]:
    """The dict entries of a ``mcp`` config value (anything else -> [])."""
    return [s for s in value if isinstance(s, dict)] if isinstance(value, list) else []


def settings_snapshot(cc: CapabilitiesConfig) -> dict[str, Any]:
    """The `SNAPSHOT_KEYS` of `cc` as plain data, plus ``"mcp_enabled"``: the NAMES of the MCP
    servers that are enabled (never their specs)."""
    snap: dict[str, Any] = {}
    for key in SNAPSHOT_KEYS:
        value = getattr(cc, key, None)
        snap[key] = list(value) if isinstance(value, (list, tuple)) else value
    enabled = [str(s.get("name")) for s in _server_entries(getattr(cc, "mcp", []))
               if isinstance(s.get("name"), str) and _spec_enabled(s)]
    snap["mcp_enabled"] = enabled
    return snap


def apply_snapshot(cc: CapabilitiesConfig, data: dict[str, Any]) -> list[str]:
    """Apply a map's snapshot into `cc`, validated and narrowing-only: a bool from a real bool,
    a list of non-empty strings, an int / float within sane ranges -- anything else is ignored
    with a warning; ``allowed_commands`` keeps only entries already in `cc.allowed_commands` or
    in ``SAFE_COMMANDS``; the two exclude lists and the blocked domains can only GROW (a map
    never un-hides a file or un-blocks a host); an empty ``web_allowed_domains`` does not widen a
    non-empty one; ``mcp_enabled`` only flips the ``enabled`` flag of servers that exist in
    `cc.mcp` by name; keys outside `SNAPSHOT_KEYS` (``allow_all_commands``, ``mcp``,
    ``command_cwd``, ...) are ignored. Returns the keys that changed, as ``"capabilities.<key>"``
    paths (what ``apply_settings`` expects)."""
    changed: list[str] = []
    if not isinstance(data, dict):
        log.warning("map settings ignored: not an object")
        return changed

    def set_(key: str, value: Any) -> None:
        old = getattr(cc, key, None)
        if old != value or type(old) is not type(value):
            setattr(cc, key, value)
            if old != value:
                changed.append(f"capabilities.{key}")

    for key in SNAPSHOT_KEYS:
        if key not in data:
            continue
        value = data[key]
        if key in _BOOL_KEYS:
            if isinstance(value, bool):
                set_(key, value)
            else:
                log.warning("map setting %s ignored: not a bool", key)
        elif key == "apply_to":
            if isinstance(value, str) and value in _APPLY_TO_VALUES:
                set_(key, value)
            else:
                log.warning("map setting apply_to ignored: %r", value)
        elif key in _LIST_KEYS:
            if not isinstance(value, list) or not all(isinstance(v, str) and v.strip() for v in value) \
                    or len(value) > 500:
                log.warning("map setting %s ignored: not a list of non-empty strings", key)
                continue
            new = [v.strip() for v in value]
            cur = list(getattr(cc, key, []) or [])
            if key == "allowed_commands":
                allowed_pool = {c.casefold() for c in cur} | {c.casefold() for c in SAFE_COMMANDS}
                kept = [c for c in new if c.casefold() in allowed_pool]
                if len(kept) != len(new):
                    log.warning("map setting allowed_commands: %d entries not in the current list or the "
                                "safe list were dropped", len(new) - len(kept))
                new = kept
            elif key in ("read_exclude", "edit_exclude", "web_blocked_domains"):
                new = cur + [v for v in new if v not in cur]
            elif key == "web_allowed_domains" and not new and cur:
                log.warning("map setting web_allowed_domains ignored: an empty list would widen the scope")
                continue
            set_(key, new)
        elif key in _INT_RANGES:
            lo, hi = _INT_RANGES[key]
            if isinstance(value, float) and value.is_integer():
                value = int(value)
            if isinstance(value, int) and not isinstance(value, bool) and lo <= value <= hi:
                set_(key, value)
            else:
                log.warning("map setting %s ignored: %r is not an integer in %d..%d", key, value, lo, hi)
        elif key in _FLOAT_RANGES:
            lo_f, hi_f = _FLOAT_RANGES[key]
            if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) \
                    and lo_f <= value <= hi_f:
                set_(key, float(value))
            else:
                log.warning("map setting %s ignored: %r", key, value)
    if "mcp_enabled" in data:
        names = data["mcp_enabled"]
        if isinstance(names, list) and all(isinstance(n, str) for n in names):
            flipped = False
            for spec in _server_entries(getattr(cc, "mcp", [])):
                nm = spec.get("name")
                if not isinstance(nm, str):
                    continue
                want = nm in names
                if _spec_enabled(spec) != want:
                    spec["enabled"] = want
                    flipped = True
            if flipped:
                changed.append("capabilities.mcp")
        else:
            log.warning("map setting mcp_enabled ignored: not a list of names")
    return changed


# ---------------------------------------------------------------------------------------------
# the runner
# ---------------------------------------------------------------------------------------------
class _Refusal(Exception):
    """The policy says no: the message is the ``denied: ...`` sentence the model reads."""


class _Failure(Exception):
    """An allowed call failed: the message is the ``error: ...`` sentence the model reads."""


@dataclass(frozen=True)
class _Root:
    """One configured folder: as the user spelled it, and with links resolved."""

    configured: Path
    real: Path


@dataclass(frozen=True)
class _Res:
    """A resolved path: the real path, the root it sits in, that root's index and the path below
    the root with forward slashes ("" = the root itself)."""

    path: Path
    root: _Root
    index: int
    rel: str


class _CallCancel(threading.Event):
    """The ONE cancel event of one web / MCP call: set when the job's cancel is set OR the
    answer's deadline passes. Nothing runs in the background -- ``is_set`` (what the fetcher and
    the MCP client poll every ~100 ms) and ``wait`` look at the clock themselves -- so there is no
    watcher thread to leak. `expired` says it was the deadline, not the job, that fired."""

    def __init__(self, parent: threading.Event | None, deadline: float | None, clock: Callable[[], float]) -> None:
        super().__init__()
        self._parent = parent
        self._deadline = deadline
        self._clock = clock
        self.expired = False

    def is_set(self) -> bool:
        if super().is_set():
            return True
        if self._parent is not None and self._parent.is_set():
            self.set()
            return True
        if self._deadline is not None and self._clock() > self._deadline:
            self.expired = True
            self.set()
            return True
        return False

    def wait(self, timeout: float | None = None) -> bool:
        end = None if timeout is None else time.monotonic() + max(0.0, timeout)
        while not self.is_set():
            left = None if end is None else end - time.monotonic()
            if left is not None and left <= 0:
                return False
            super().wait(0.05 if left is None else min(0.05, left))
        return True


@dataclass
class _Ctx:
    """What one read-tool call works with: the live roots / excludes / protected set and the
    stop test (deadline + cancel)."""

    tool: str
    roots: list[_Root]
    excl: list[str]
    prot: frozenset[str]
    stop: _Stop
    cancel: threading.Event | None
    chars: int = 6000

    @property
    def multi(self) -> bool:
        return len(self.roots) > 1


#: apply_to -> the kinds that may use tools at all (read tools; the active ones only for "ask").
_APPLY_KINDS: dict[str, frozenset[str]] = {
    "ask": frozenset({"ask"}),
    "questions": frozenset({"auto", "manual", "ask", "screen"}),
    "all": frozenset({"auto", "manual", "ask", "screen", "suggest", "solve", "code", "text"}),
}
_WEB_TOOLS = frozenset({"fetch_url", "web_search"})
#: the tools that can change the world: a call to one of them (or to an MCP tool) empties the
#: repeated-call cache of its answer
_MUTATING_TOOLS = frozenset({"write_file", "edit_file", "run_command"})
_LIST_CAP = 200
_FIND_CAP = 200
_HIT_CAP = 60
_SEARCH_FILE_BYTES = 1024 * 1024
_SEARCH_TOTAL_BYTES = 64 * 1024 * 1024
_SEARCH_MAX_FILES = 20000
_SCAN_CAP = 200_000
_CLOSE_JOIN_S = 3.0           # ToolRunner.close() waits this long, in total, for the MCP clients to close
_MAX_EDIT_BYTES = 2 * 1024 * 1024
_MAX_WRITE_CHARS = 1_000_000
_BOM = codecs.BOM_UTF8
#: The exfiltration limits on what the model may put in a web call: a bandwidth limit on what an
#: injected instruction can send out (not a guarantee -- the docs say so).
_WEB_URL_MAX = 256           # a longer url is refused
_WEB_QUERY_MAX = 100         # ... and so is one whose query string is longer than this
_WEB_SEARCH_MAX = 120        # a search query is cut to this

_USAGE: dict[str, str] = {
    "read_file": 'read_file(path, start_line?, end_line?)',
    "list_dir": 'list_dir(path?, depth?)',
    "find_files": 'find_files(pattern, path?)',
    "search_text": 'search_text(query, path?, glob?)',
    "write_file": 'write_file(path, content)',
    "edit_file": 'edit_file(path, old, new)',
    "run_command": 'run_command(command, cwd?)',
    "fetch_url": 'fetch_url(url)',
    "web_search": 'web_search(query)',
}

_BUILTIN_SPECS: dict[str, ToolSpec] = {
    "read_file": ToolSpec(
        "read_file", "read a text file (a long one is cut; start_line / end_line pick a range)",
        (("path", "file path, relative to a project folder or absolute"), ("start_line", "optional, 1-based"),
         ("end_line", "optional, inclusive"))),
    "list_dir": ToolSpec(
        "list_dir", "list the files and sub-folders of a folder",
        (("path", "folder; empty = the project folders"), ("depth", "optional, 1 to 3"))),
    "find_files": ToolSpec(
        "find_files", "find files and folders by a name pattern such as *.py or src/*.md",
        (("pattern", "glob pattern"), ("path", "optional folder to search in"))),
    "search_text": ToolSpec(
        "search_text", "find a literal text in the project's files (case-insensitive, not a regex)",
        (("query", "the text to find"), ("path", "optional folder or file"), ("glob", "optional file filter such as *.py"))),
    "write_file": ToolSpec(
        "write_file", "create a file or replace one completely (the original is backed up)",
        (("path", "file path inside an edit folder"), ("content", "the complete new text"))),
    "edit_file": ToolSpec(
        "edit_file", "replace one exact piece of text in a file (it must occur exactly once)",
        (("path", "file path inside an edit folder"), ("old", "the exact text to replace"), ("new", "the replacement text"))),
    "run_command": ToolSpec(
        "run_command", "run one allowed shell command (no pipes, redirects or chaining)",
        (("command", "the command line"), ("cwd", "optional folder to run it in"))),
    "fetch_url": ToolSpec("fetch_url", "fetch a web page as plain text", (("url", "http(s) address"),)),
    "web_search": ToolSpec("web_search", "search the web", (("query", "the search terms"),)),
}

#: Spellings small models use for the real argument names (built-in tools only, applied when the
#: real name is absent; MCP tools get the server's schema names verbatim).
_ALIASES: dict[str, dict[str, tuple[str, ...]]] = {
    "read_file": {"path": ("file", "file_path", "filepath", "filename", "name"),
                  "start_line": ("start", "from_line", "line_start"), "end_line": ("end", "to_line", "line_end")},
    "list_dir": {"path": ("dir", "directory", "folder", "file", "file_path")},
    "find_files": {"pattern": ("glob", "name", "query", "filename", "file"), "path": ("dir", "directory", "folder")},
    "search_text": {"query": ("pattern", "regex", "text", "q", "search", "term"),
                    "path": ("dir", "directory", "folder", "file"), "glob": ("include", "files", "filter")},
    "write_file": {"path": ("file", "file_path", "filepath", "filename"), "content": ("text", "contents", "data", "body")},
    "edit_file": {"path": ("file", "file_path", "filepath", "filename"), "old": ("old_string", "old_text", "find", "search"),
                  "new": ("new_string", "new_text", "replace", "replacement")},
    "run_command": {"command": ("cmd", "commandline", "command_line", "shell"), "cwd": ("dir", "directory", "workdir")},
    "fetch_url": {"url": ("uri", "link", "address", "href")},
    "web_search": {"query": ("q", "search", "text", "terms")},
}
_INT_ARGS = ("start_line", "end_line", "depth")

_CHAIN_CHARS = "&|;<>^`%"
_GIT_DIFFY = frozenset({"diff", "log", "show", "blame"})
#: git subcommands whose output covers the whole repository unless a pathspec narrows it: they
#: get ``-- .`` (the folder the command runs in) and one exclude pathspec per exclude pattern
#: (``status`` and ``ls-files`` only show names, but names outside the folder are the same leak)
_GIT_SCOPED = frozenset({"diff", "log", "show", "status", "ls-files"})
_DENY_FLAGS = frozenset({
    "--output", "--no-index", "--ext-diff", "--textconv", "--pre", "--pre-glob", "--hostname-bin", "-exec",
    "-execdir", "-ok", "-okdir", "-delete", "-fprint", "-fprint0", "-fprintf", "-fls", "--exec-path",
    "--git-dir", "--work-tree", "--open-files-in-pager", "--follow", "--search-zip", "--config-env",
    "--pathspec-from-file", "--pathspec-file-nul",
})
_RECURSE_FLAGS = frozenset({"-r", "--recursive", "/r"})
_RECURSE_OK = frozenset({"dir", "where", "tree", "findstr"})
_SLASH_S_OK = frozenset({"dir"})
_OUT_EXEMPT: dict[str, frozenset[str]] = {"-o": frozenset({"rg", "grep"}), "/o": frozenset({"dir"})}
_FOLLOW_CMDS = frozenset({"rg", "ls", "grep"})
#: commands that only list names: a wildcard in their arguments cannot reveal an excluded file's content
_LISTERS = frozenset({"dir", "ls", "tree", "where", "which", "pwd", "hostname", "whoami", "ver", "date",
                      "tasklist", "systeminfo"})
#: commands cmd.exe runs itself (not as a program): it splits their arguments on `,` and `=` too
_CMD_BUILTINS = frozenset({
    "dir", "type", "ver", "date", "time", "echo", "copy", "move", "del", "erase", "ren", "rename", "md", "mkdir", "rd",
    "rmdir", "cd", "chdir", "set", "path", "vol", "title", "mklink", "assoc", "ftype", "start", "call", "pushd",
    "popd", "cls", "color", "prompt", "verify", "label",
})
#: commands that read a ``/`` as the start of a switch: a forward-slash PATH argument is turned into
#: a backslash one before they see it (the map and the listings print forward slashes)
_SLASH_SWITCH_CMDS = _CMD_BUILTINS | frozenset({"more", "findstr", "where", "sort"})
#: commands whose first non-option argument is the SEARCH PATTERN, not a file (a wildcard there is
#: a regex ``*``, not a mask)
_PATTERN_CMDS = frozenset({"grep", "rg", "findstr"})
#: tasklist / systeminfo switches that connect to ANOTHER MACHINE (``/s host`` ``/u user`` ``/p pass``,
#: in any spelling and case): a resolver lookup or an RPC / NTLM authentication carrying model text
_NET_CMDS = frozenset({"tasklist", "systeminfo"})
_RE_REMOTE_SWITCH = re.compile(r"[-/][sup]([:=].*)?", re.IGNORECASE)
#: pip / npm may only carry these options: everything else (``--outdated``, ``--index-url``,
#: ``--proxy``, ``--registry``, an abbreviation of any of them) can open a connection
_PKG_CMDS = frozenset({"pip", "npm"})
_PKG_FLAGS = frozenset({"--format", "--json", "--version", "-v", "--depth", "--all", "-a", "--long", "-l",
                        "--not-required", "--user", "--local"})
#: the long options grep and ls may carry. GNU getopt accepts any unique PREFIX of a long option
#: (``--recurs``), so a deny-list of exact names cannot work: these are an allow-list.
_GREP_LONG = frozenset({
    "--ignore-case", "--no-ignore-case", "--invert-match", "--word-regexp", "--line-regexp", "--count",
    "--files-with-matches", "--files-without-match", "--max-count", "--line-number", "--with-filename",
    "--no-filename", "--only-matching", "--quiet", "--silent", "--no-messages", "--text", "--after-context",
    "--before-context", "--context", "--fixed-strings", "--extended-regexp", "--basic-regexp", "--perl-regexp",
    "--regexp", "--file", "--null", "--null-data", "--byte-offset", "--initial-tab", "--color", "--colour",
    "--binary-files", "--help", "--version",
})
_LS_LONG = frozenset({
    "--all", "--almost-all", "--classify", "--human-readable", "--size", "--reverse", "--sort", "--time",
    "--color", "--colour", "--directory", "--group-directories-first", "--format", "--inode", "--numeric-uid-gid",
    "--full-time", "--author", "--escape", "--literal", "--quoting-style", "--tabsize", "--width",
    "--time-style", "--indicator-style", "--file-type", "--si", "--block-size", "--kibibytes", "--no-group",
    "--help", "--version",
})
_LONG_OK: dict[str, frozenset[str]] = {"grep": _GREP_LONG, "ls": _LS_LONG}
#: options whose value is a glob / pattern / number, never a path: ``--glob=*.py`` and ``-g *.py``
_GLOB_VALUE_FLAGS = frozenset({"--glob", "--iglob", "--regexp"})
#: grep / rg: the short-option letters that take a value (the first of them in a cluster takes the REST of
#: the cluster, or the next argument when it is the last letter), and the long options whose value may
#: be the next argument. A letter listed here wrongly would hide a real file operand from the path
#: gate, so only options that really take a separate value are in.
_ARG_LETTERS: dict[str, frozenset[str]] = {"grep": frozenset("ABCmefdDX"), "rg": frozenset("ABCmMjtTdEefgr")}
_SEPARATE_VALUE_LONG: dict[str, frozenset[str]] = {
    "grep": frozenset({"--regexp", "--max-count", "--after-context", "--before-context", "--context"}),
    "rg": frozenset({"--regexp", "--max-count", "--glob", "--iglob", "--type", "--type-not", "--max-columns",
                     "--threads", "--max-depth", "--encoding", "--context", "--after-context", "--before-context"}),
}
_RE_SLASH_FLAG = re.compile(r"^/[A-Za-z?]{1,2}(:.*)?$")
_RE_DRIVE_ANY = re.compile(r"^[A-Za-z]:")
#: getopt-style tools: their short options cluster (`-rniHs`), of any length, digits included
_GETOPT_CMDS = frozenset({"grep", "rg", "ls", "cat", "head", "tail", "wc", "sort", "which"})
_RE_CLUSTER = re.compile(r"-[A-Za-z0-9]{2,}")
_RE_CLUSTER_SHORT = re.compile(r"-[A-Za-z]{2,4}")
_RE_PATHSPEC_MAGIC = re.compile(r"^:[(/!^]")
#: an exclude pattern that can be handed to rg / git as it is, or inside double quotes
_RE_GLOB_PLAIN = re.compile(r"^[A-Za-z0-9_.*?/\-]+$")
_RE_GLOB_QUOTABLE = re.compile(r"^[^\x00-\x1f\"&|<>^%!`$\\]+$")
#: ``scheme://user:secret@host`` -> ``scheme://***@host`` (git remote -v prints the credentials)
_RE_URL_CRED = re.compile(r"(?<=://)[^/@\s]+@")
#: the longest command line a gated rewrite may reach (cmd.exe stops at 8191 characters)
_MAX_GATED_LINE = 7500


def _build_lookalikes() -> dict[int, str]:
    """Characters that a native console program's ANSI conversion of its command line ("best fit")
    turns into the ASCII punctuation the gate looks for -- verified with findstr, which reads an en
    dash as ``-s`` (recurse), a fullwidth asterisk as a wildcard and a fullwidth backslash as a path
    separator, while cmd.exe's own parser (and ours) saw ordinary letters."""
    table: dict[int, str] = {}
    for chars, ascii_char in (
            ("\u2010\u2011\u2012\u2013\u2014\u2015\u2212\ufe58\ufe63\uff0d", "-"),
            ("\u2044\u2215\u29f8\uff0f", "/"),
            ("\u2216\u29f5\u29f9\ufe68\uff3c", "\\"),
            ("\u201c\u201d\u201e\u201f\u2033\u2036\u00ab\u00bb\uff02", '"'),
            ("\u2236\ua789\uff1a", ":"),
            ("\u2217\u204e\u2731\uff0a", "*"),
            ("\uff1f", "?")):
        for ch in chars:
            table[ord(ch)] = ascii_char
    return table


_LOOKALIKES = _build_lookalikes()


def _has_lookalikes(text: str) -> bool:
    """Does `text` hold a character that reads as a different ASCII character to a program that
    converts its command line to the console code page: a compatibility form (NFKC changes it: the
    fullwidth asterisk, solidus and colon, a no-break space) or one of the dash / slash / quote variants?"""
    if text.isascii():
        return False
    return any(ord(ch) in _LOOKALIKES for ch in text) or unicodedata.normalize("NFKC", text) != text


def _split_command(cmd: str) -> list[str]:
    """Split a command line the way the shell will: whitespace separates words except inside
    double quotes (single quotes too off Windows); the quote characters stay in the token and a
    quote may open in the middle of a word (``/c:"a b"``, ``--since="2 weeks ago"``) --
    ``shlex.split(posix=False)`` only honours a quote at the start of a word, so the gate would
    see different words than the shell. Raises ValueError on an unbalanced quote."""
    toks: list[str] = []
    cur: list[str] = []
    quote = ""
    for ch in cmd:
        if quote:
            cur.append(ch)
            if ch == quote:
                quote = ""
        elif ch == '"' or (ch == "'" and os.name != "nt"):
            quote = ch
            cur.append(ch)
        elif ch.isspace():
            if cur:
                toks.append("".join(cur))
                cur = []
        else:
            cur.append(ch)
    if quote:
        raise ValueError("unbalanced quote")
    if cur:
        toks.append("".join(cur))
    return toks


def _unquote(tok: str) -> str:
    """Strip one pair of surrounding quotes from a token (the splitter keeps them)."""
    t = tok.strip()
    if len(t) >= 2 and t[0] == t[-1] and t[0] in "\"'":
        return t[1:-1]
    return t


def _findstr_letters(bare: str) -> str:
    """The casefolded ASCII letters of a findstr switch group before its colon (``/i/sd:x`` -> ``isd``)."""
    return "".join(re.findall(r"[A-Za-z]", bare.partition(":")[0])).casefold()


def _backslash_token(tok: str) -> str:
    """A path argument with its forward slashes turned into backslashes, for the cmd.exe commands
    that would read ``/ui`` as a switch. A double-quoted token keeps its quotes and loses a trailing
    backslash (findstr reads ``\\"`` as an escaped quote)."""
    t = tok.strip()
    quoted = len(t) >= 2 and t[0] == t[-1] == '"'
    body = (t[1:-1] if quoted else t).replace("/", "\\")
    if quoted:
        return '"' + (body.rstrip("\\") or body) + '"'
    return body


def _norm_nl(text: str, nl: str) -> str:
    """`text` with every line ending turned into `nl`."""
    return text.replace("\r\n", "\n").replace("\n", nl)


def _kill_tree(proc: "subprocess.Popen[bytes]", env: dict[str, str]) -> None:
    """Kill `proc` and everything it started (``taskkill /T /F`` on Windows, the process group
    elsewhere). Never raises."""
    try:
        if os.name == "nt":
            root = os.environ.get("SystemRoot") or os.environ.get("SYSTEMROOT") or r"C:\Windows"
            exe = os.path.join(root, "System32", "taskkill.exe")
            subprocess.run(
                [exe, "/T", "/F", "/PID", str(proc.pid)], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL, env=env, timeout=10, check=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:      # noqa: BLE001 - fall through to the plain kill
        log.debug("process-tree kill failed", exc_info=True)
    try:
        proc.kill()
    except OSError:
        pass
    try:
        proc.wait(timeout=3)
    except Exception:      # noqa: BLE001
        pass


def _child_env() -> dict[str, str]:
    """The process environment plus ``NoDefaultCurrentDirectoryInExePath=1``: cmd.exe then does not
    search the working directory for the command, so a ``git.bat`` planted in a project folder is
    never run. Any differently-cased copy of the variable is replaced (Windows env names are
    case-insensitive; two spellings in one block are undefined)."""
    env = {k: v for k, v in os.environ.items() if k.casefold() != "nodefaultcurrentdirectoryinexepath"}
    env["NoDefaultCurrentDirectoryInExePath"] = "1"
    env.setdefault("GIT_TERMINAL_PROMPT", "0")
    return env


def _decode_stream(head: bytes, tail: bytes) -> tuple[str, str]:
    """Decode the kept head and tail of a command's output with ONE encoding (UTF-8 when the head
    is valid UTF-8, else the OEM code page), so a tail that starts mid-character is not read as
    the other encoding."""
    enc = "utf-8"
    try:
        codecs.getincrementaldecoder("utf-8")().decode(head, final=False)
    except UnicodeDecodeError:
        enc = _oem_encoding()
    h = head.decode("utf-8-sig" if enc == "utf-8" else enc, errors="replace")
    t = tail.decode(enc, errors="replace") if tail else ""
    return h, t


def _run_process(
    command: str, cwd: str, *, timeout_s: float, out_chars: int, stop: Callable[[], bool],
    cancel: threading.Event | None, env: dict[str, str],
) -> tuple[str, str, int | None]:
    """Run `command` through the shell in `cwd`: stdout and stderr merged, stdin closed, a reader
    thread draining the pipe into a bounded head + tail buffer, a 100 ms poll for the timeout, the
    answer's deadline (`stop`) and `cancel`, and a process-TREE kill when any of them fires.
    Returns ``(output text, outcome, return code)`` with outcome "ok" / "timeout" / "cancelled" /
    "deadline". Raises :class:`OSError` when the process cannot start."""
    flags = 0
    extra: dict[str, Any] = {}
    if os.name == "nt":
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    else:
        extra["start_new_session"] = True
    proc = subprocess.Popen(
        command, shell=True, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL, env=env, creationflags=flags, bufsize=0, **extra)
    cap = max(512, int(out_chars) * 2)
    head = bytearray()
    tail = bytearray()
    total = [0]

    def drain() -> None:
        try:
            while True:
                chunk = proc.stdout.read(65536)      # type: ignore[union-attr]
                if not chunk:
                    return
                total[0] += len(chunk)
                room = cap - len(head)
                if room > 0:
                    head.extend(chunk[:room])
                    chunk = chunk[room:]
                if chunk:
                    tail.extend(chunk)
                    if len(tail) > 2 * cap:
                        del tail[: len(tail) - cap]
        except (OSError, ValueError):
            return

    reader = threading.Thread(target=drain, name="command-output", daemon=True)
    reader.start()
    outcome = "ok"
    started = time.monotonic()
    while True:
        try:
            proc.wait(timeout=0.1)
            break
        except subprocess.TimeoutExpired:
            pass
        if cancel is not None and cancel.is_set():
            outcome = "cancelled"
        elif time.monotonic() - started > timeout_s:
            outcome = "timeout"
        elif stop():
            outcome = "deadline"
        if outcome != "ok":
            _kill_tree(proc, env)
            break
    reader.join(2.0)
    if not reader.is_alive():          # a grandchild that kept the pipe open would block a close
        try:
            proc.stdout.close()        # type: ignore[union-attr]
        except OSError:
            pass
    kept_tail = bytes(tail[-cap:])
    h, t = _decode_stream(bytes(head), kept_tail)
    omitted = total[0] - len(head) - len(kept_tail)
    marker = f"\n(... {omitted} bytes omitted ...)\n" if omitted > 0 else ""
    text = h + marker + t
    return _head_tail(text.replace("\r\n", "\n"), out_chars), outcome, proc.returncode


def _cut(text: str, limit: int) -> str:
    """`text` cut to `limit` chars with the "(cut: shows N of M chars)" note."""
    if limit > 0 and len(text) > limit:
        return text[:limit].rstrip() + f"\n(cut: shows {limit} of {len(text)} chars)"
    return text


def _spec_enabled(spec: dict[str, Any]) -> bool:
    """An MCP entry is enabled when ``enabled`` is absent or a true bool / int (anything else,
    "no" the string included, counts as disabled: fail closed)."""
    val = spec.get("enabled", True)
    return bool(val) if isinstance(val, (bool, int)) else False


class ToolRunner:
    """The policy and the tools (see the module docstring). One per engine, shared by every
    assist job; thread-safe (a job runs its calls on the assist worker while the UI thread reads
    `status()` and the engine's threads call `refresh()` / `connect_mcp()`).

    `session_folder` returns the current session folder (for edit backups) or None;
    `web` (a :class:`exocortex.webfetch.WebFetcher`-alike) and `mcp_factory` (``spec ->``
    :class:`exocortex.mcp.McpServer`-alike) exist so tests can pass fakes; the defaults are the
    real ones, constructed lazily. `clock` is what budgets and deadlines are measured with.

    The config is read LIVE on every call (the user applies the tab at any moment) and treated
    defensively: a list field that is not a list means "none", a root that is empty, relative or
    not a folder is skipped with one warning per :meth:`refresh`.
    """

    def __init__(
        self, cfg: Config, *, session_folder: Callable[[], Path | None] | None = None,
        web: Any = None, mcp_factory: Callable[[dict[str, Any]], Any] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._cfg = cfg
        self._session_folder = session_folder or (lambda: None)
        self._web = web
        self._web_injected = web is not None
        self._mcp_factory = mcp_factory
        self._clock = clock
        self._lock = threading.RLock()
        self._connect_lock = threading.Lock()
        self._map_text = ""
        self._warned: set[str] = set()
        # MCP: client per server name, the spec snapshot it was built from, its cached tool
        # listing (what specs() serves) and its state.
        self._mcp_clients: dict[str, Any] = {}
        self._mcp_snapshots: dict[str, dict[str, Any]] = {}
        self._mcp_listing: dict[str, list[Any]] = {}
        self._mcp_state: dict[str, str] = {}
        # servers whose last call raised an MCP error: the next connect pass rebuilds them
        self._mcp_suspect: set[str] = set()
        self._closers: list[threading.Thread] = []      # the daemon threads closing dropped clients
        self._closed = False                            # close() ran: nothing is connected any more

    # ---- config accessors (live, defensive)
    @property
    def _cc(self) -> CapabilitiesConfig:
        return self._cfg.capabilities

    def _on(self, name: str) -> bool:
        v = getattr(self._cc, name, False)
        return bool(v) if isinstance(v, (bool, int)) else False

    def _int(self, name: str, default: int, lo: int, hi: int) -> int:
        v = getattr(self._cc, name, default)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or (isinstance(v, float) and not math.isfinite(v)):
            v = default
        return int(min(hi, max(lo, v)))

    def _float(self, name: str, default: float, lo: float, hi: float) -> float:
        v = getattr(self._cc, name, default)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            v = default
        return float(min(hi, max(lo, v)))

    def _warn_once(self, key: str, message: str) -> None:
        with self._lock:
            if key in self._warned:
                return
            self._warned.add(key)
        log.warning("%s", message)

    # ---- what a job may do
    def _covered(self, kind: str) -> bool:
        """Does `apply_to` cover `kind` at all (never the NEVER_KINDS, an unknown mode = "ask")?"""
        if kind in NEVER_KINDS:
            return False
        mode = getattr(self._cc, "apply_to", "questions")
        return kind in _APPLY_KINDS.get(mode if isinstance(mode, str) else "", _APPLY_KINDS["ask"])

    def wants(self, kind: str) -> bool:
        """Does a job of `kind` get a tools block and a tool loop? False unless `apply_to` covers
        the kind, some switch is on AND :meth:`specs` lists at least one tool (so with no readable
        folder configured nothing changes about any prompt)."""
        return bool(self.specs(kind))

    def specs(self, kind: str) -> list[ToolSpec]:
        """The tools a job of `kind` may call right now, sorted by name (a deterministic, stable
        string for the system prompt). Read tools need `read_files` and a configured root;
        `run_command` needs `execute_commands` and a root (its cwd); the web tools need
        `fetch_web`; the MCP tools need `mcp_servers` and a CONNECTED server (the cached listing
        taken by :meth:`connect_mcp` -- never a connection attempt here). Active tools only for
        `ACTIVE_KINDS`; nothing for `NEVER_KINDS`."""
        try:
            return self._specs(kind)
        except Exception:      # noqa: BLE001 - a broken config must never break a prompt
            log.warning("could not build the tool list", exc_info=True)
            return []

    def _specs(self, kind: str) -> list[ToolSpec]:
        if not self._covered(kind):
            return []
        cc = self._cc
        if not _str_list(getattr(cc, "read_roots", [])):
            return []                  # no readable folder: no tools at all, whatever the switches say
        out: list[ToolSpec] = []
        if self._on("read_files"):
            out += [_BUILTIN_SPECS[n] for n in READ_TOOLS]
        if kind in ACTIVE_KINDS:
            if self._on("edit_files") and _str_list(getattr(cc, "edit_roots", [])):
                out += [_BUILTIN_SPECS["write_file"], _BUILTIN_SPECS["edit_file"]]
            if self._on("execute_commands"):
                out.append(_BUILTIN_SPECS["run_command"])
            if self._on("fetch_web"):
                out += [_BUILTIN_SPECS["fetch_url"], _BUILTIN_SPECS["web_search"]]
            if self._on("mcp_servers"):
                out += self._mcp_tool_specs()
        out.sort(key=lambda s: s.name)
        return out

    def budget(self, kind: str) -> ToolBudget:
        """A fresh :class:`ToolBudget` for one answer of `kind`: the ``live_*`` limits for
        `LIVE_KINDS`, the plain ones for the ask box; `allowed` = the names :meth:`specs` lists;
        `deadline` = ``clock() + tool_time_s``."""
        if kind in LIVE_KINDS:
            calls = self._int("live_max_tool_calls", 2, 0, 50)
            chars = self._int("live_tool_result_chars", 2500, 200, 200_000)
            secs = self._float("live_tool_time_s", 10.0, 0.5, 3600.0)
        else:
            calls = self._int("max_tool_calls", 4, 0, 50)
            chars = self._int("tool_result_chars", 6000, 200, 200_000)
            secs = self._float("tool_time_s", 30.0, 0.5, 3600.0)
        allowed = frozenset(s.name for s in self.specs(kind))
        web = self._int("web_max_fetches", 3, 0, 50) if allowed & _WEB_TOOLS else 0
        return ToolBudget(kind=kind, allowed=allowed, calls_left=calls, chars=chars,
                          deadline=self._clock() + secs, web_left=web)

    # ---- the folders: roots, excludes, protected paths
    def protected_paths(self) -> list[Path]:
        """The app's own paths the model never reads or edits, whatever the tab says: the
        ``config.toml`` in use and the models / context / maps / sessions folders (real paths).
        The engine passes these to :func:`build_map` so the indexer skips them too."""
        cfg = self._cfg
        cands: list[Any] = []
        if getattr(cfg, "path", None):
            cands.append(cfg.path)
        for getter in ("models_dir", "context_dir", "maps_dir", "sessions_dir"):
            try:
                cands.append(getattr(cfg, getter)())
            except Exception:      # noqa: BLE001 - a broken path config protects nothing, loudly
                log.debug("protected path %s unavailable", getter, exc_info=True)
        out: list[Path] = []
        for c in cands:
            try:
                out.append(Path(os.path.realpath(str(c))))
            except (OSError, ValueError):
                continue
        return out

    def _protected_norm(self) -> frozenset[str]:
        return frozenset(os.path.normcase(str(p)) for p in self.protected_paths())

    def _roots(self, *, edit: bool = False) -> list[_Root]:
        """The usable roots right now (absolute, an existing folder, not inside a protected
        path). Every skipped entry is warned about once per :meth:`refresh`."""
        attr = "edit_roots" if edit else "read_roots"
        raw = getattr(self._cc, attr, [])
        if not isinstance(raw, (list, tuple)):
            self._warn_once(f"{attr}:type", f"capabilities.{attr} is not a list: no folders")
            return []
        prot = self.protected_paths()
        out: list[_Root] = []
        seen: set[str] = set()
        for item in raw:
            if not isinstance(item, str) or not item.strip():
                self._warn_once(f"{attr}:empty", f"capabilities.{attr}: an empty or non-text entry was skipped")
                continue
            s = item.strip()
            try:
                if not os.path.isabs(s):
                    self._warn_once(f"{attr}:{s}", f"capabilities.{attr}: {s!r} is not an absolute path; skipped")
                    continue
                if not Path(s).is_dir():
                    self._warn_once(f"{attr}:{s}", f"capabilities.{attr}: {s!r} is not a folder; skipped")
                    continue
                real = Path(os.path.realpath(s))
            except (OSError, ValueError):
                self._warn_once(f"{attr}:{s}", f"capabilities.{attr}: {s!r} cannot be used; skipped")
                continue
            if any(real == q or real.is_relative_to(q) for q in prot):
                self._warn_once(f"{attr}:{s}", f"capabilities.{attr}: {s!r} is inside the app's own files; skipped")
                continue
            key = os.path.normcase(str(real))
            if key in seen:
                continue
            seen.add(key)
            out.append(_Root(Path(os.path.normpath(s)), real))
        keep = _outermost([r.real for r in out])
        if len(keep) != len(out):
            for i, r in enumerate(out):
                if i not in keep:
                    self._warn_once(
                        f"{attr}:nested:{r.configured}",
                        f"capabilities.{attr}: {str(r.configured)!r} is inside another listed folder; it is covered "
                        "by that folder and that folder's exclude patterns apply (a folder inside an excluded "
                        "sub-folder of another one is not readable)")
            out = [out[i] for i in keep]
        return out

    def _excludes(self, *, edit: bool = False) -> list[str]:
        """The exclude patterns for reads (``read_exclude``) or edits (that plus ``edit_exclude``).
        A malformed ``read_exclude`` falls back to the defaults, never to "nothing excluded"."""
        raw = getattr(self._cc, "read_exclude", None)
        pats = _str_list(raw) if isinstance(raw, (list, tuple)) else list(DEFAULT_READ_EXCLUDE)
        if edit:
            pats += _str_list(getattr(self._cc, "edit_exclude", []))
        return _clean_excludes(pats)

    @staticmethod
    def _path_text(path: Any) -> str:
        """Validate the TEXT of a path argument (no filesystem access)."""
        if isinstance(path, bool) or not isinstance(path, (str, int, float)):
            raise PathDenied("denied: the path must be text")
        s = _unquote(str(path))
        if not s:
            raise PathDenied("denied: the path is empty")
        if any(ord(c) < 32 for c in s):
            raise PathDenied("denied: control characters in a path")
        if s.startswith(("\\\\", "//")):
            raise PathDenied("denied: network (UNC) paths are not allowed")
        if s.startswith("~"):
            raise PathDenied("denied: '~' paths are not allowed")
        if _RE_DRIVE_RELATIVE.match(s):
            raise PathDenied("denied: drive-relative paths (C:name) are not allowed")
        if any(c in s for c in '*?"<>|'):
            raise PathDenied("denied: wildcard or reserved characters in a path")
        _drive, rest = os.path.splitdrive(s)
        for comp in _RE_SEPARATORS.split(rest):
            if comp in ("", "."):
                continue
            if comp == "..":
                raise PathDenied("denied: '..' is not allowed in a path; use the full path")
            if comp[-1] in ". ":
                raise PathDenied(f"denied: '{comp}' ends with a dot or a space")
            if ":" in comp:
                raise PathDenied(f"denied: '{comp}' names an alternate data stream")
            if _RE_SHORT_NAME.match(comp):
                raise PathDenied(f"denied: '{comp}' is an 8.3 short name; use the long name")
            if comp.split(".")[0].rstrip(" ").upper() in _RESERVED_NAMES:
                raise PathDenied(f"denied: '{comp}' is a reserved device name")
        return s

    def _resolve(
        self, path: Any, *, edit: bool = False, base: Path | None = None, roots: list[_Root] | None = None,
    ) -> _Res:
        """Resolve a path argument to a real path inside one of the roots, or raise
        :class:`PathDenied`. An absolute path is used as is; a relative one is tried against each
        root in order (`root / path` exists wins, else the first root) or against `base` (a
        command's cwd). Refused: UNC, ``~``, drive-relative, ``..``, components ending in a dot or
        space, 8.3 short names, streams, device names, anything outside the roots, any component
        that is a symlink / junction, anything excluded (the requested AND the resolved path) and
        the app's own protected paths."""
        use = roots if roots is not None else self._roots(edit=edit)
        if not use:
            raise PathDenied(f"denied: no {'editable' if edit else 'readable'} folder is configured")
        s = self._path_text(path)
        excl = self._excludes(edit=edit)
        drive, rest = os.path.splitdrive(s)
        rooted = rest.startswith(("/", "\\"))
        if rooted:
            cand = Path(os.path.normpath(s if (drive or os.name != "nt") else os.path.abspath(s)))
        elif base is not None:
            cand = Path(os.path.normpath(os.path.join(str(base), s)))
        else:
            cand = None            # type: ignore[assignment]
            for r in use:
                c = Path(os.path.normpath(os.path.join(str(r.real), s)))
                if os.path.lexists(c):
                    cand = c
                    break
            if cand is None:
                cand = Path(os.path.normpath(os.path.join(str(use[0].real), s)))
        index, rel = -1, None
        for i, r in enumerate(use):
            for anchor in (r.real, r.configured):
                try:
                    rel = cand.relative_to(anchor)
                    index = i
                    break
                except ValueError:
                    continue
            if index >= 0:
                break
        if index < 0 or rel is None:
            listed = ", ".join(r.configured.as_posix() for r in use)
            raise PathDenied(f"denied: {cand.as_posix()} is outside the {'editable' if edit else 'readable'} "
                             f"folders ({listed})")
        root = use[index]
        cur = root.real
        for part in rel.parts:                    # no link anywhere below the root
            cur = cur / part
            try:
                st = os.lstat(cur)
            except FileNotFoundError:
                break
            except OSError as exc:
                raise PathDenied(f"denied: cannot inspect {part}: {exc.strerror or exc}") from None
            if _is_link_stat(st):
                raise PathDenied(f"denied: '{part}' is a symbolic link or junction; links are never followed")
        hit = _parts_excluded(rel.parts, excl)
        if hit:
            raise PathDenied(f"denied: {rel.as_posix() or s} is excluded from the {'edit' if edit else 'read'} scope")
        try:
            real = Path(os.path.realpath(cand))
            real_rel = real.relative_to(root.real)
        except (OSError, ValueError):
            raise PathDenied(f"denied: {cand.as_posix()} resolves outside the folder") from None
        if _parts_excluded(real_rel.parts, excl):
            raise PathDenied(f"denied: {real_rel.as_posix()} is excluded from the {'edit' if edit else 'read'} scope")
        for q in self.protected_paths():
            if real == q or real.is_relative_to(q):
                raise PathDenied(f"denied: {real_rel.as_posix() or s} belongs to the app and is never exposed")
        return _Res(real, root, index, real_rel.as_posix() if real_rel.parts else "")

    def resolve_read(self, path: str, *, base: Path | None = None) -> Path:
        """The real path of `path` if the model may read it, else :class:`PathDenied`."""
        return self._resolve(path, base=base).path

    def resolve_edit(self, path: str) -> Path:
        """The real path of `path` if the model may write it (an edit root, not excluded or
        protected), else :class:`PathDenied`."""
        return self._resolve(path, edit=True).path

    @staticmethod
    def _show(root: _Root, rel: str, multi: bool) -> str:
        """How a path is written back to the model: relative to its root with one root, absolute
        (forward slashes) with several."""
        if multi:
            return (root.real / rel).as_posix() if rel else root.real.as_posix()
        return rel or "."

    # ---- running one call
    def run(self, call: ToolCall, *, budget: ToolBudget, cancel: threading.Event | None = None) -> ToolResult:
        """Execute one call under the policy. NEVER raises: "denied" for a name outside
        `budget.allowed`, a switch that is off, a path outside the readable / editable folders or
        excluded / protected, a refused command, a refused url, a web tool after local data was
        read; "error" when an allowed call fails. The text is sanitised and cut to
        `budget.chars`; `budget.calls_left` / `web_left` / `read_done` / `seen` are updated;
        `cancel` is checked before and polled during the call."""
        try:
            return self._run(call, budget, cancel)
        except Exception as exc:      # noqa: BLE001 - run() must never raise into the tool loop
            log.warning("tool call %r crashed", getattr(call, "name", "?"), exc_info=True)
            msg = f"error: the tool failed: {exc}" if str(exc) else "error: the tool failed"
            return ToolResult(sanitize_untrusted(msg), "error", _one_line(msg[7:]), "")

    @staticmethod
    def _normalise_args(name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Built-in tools: map the usual wrong spellings of an argument name to the real one and
        turn digit strings into ints for the numeric arguments. MCP tools are left verbatim."""
        out = {k: v for k, v in args.items() if isinstance(k, str) and v is not None}
        aliases = _ALIASES.get(name)
        if aliases:
            for canon, alts in aliases.items():
                if canon not in out:
                    for alt in alts:
                        if alt in out:
                            out[canon] = out[alt]
                            break
            for k in _INT_ARGS:
                v = out.get(k)
                if isinstance(v, str) and re.fullmatch(r"\s*-?\d+\s*", v):
                    out[k] = int(v)
        return out

    @staticmethod
    def _summary_for(name: str, a: dict[str, Any]) -> str:
        """The one line the tool-call event shows: the path / pattern / command / url / query."""
        key = {"read_file": "path", "list_dir": "path", "find_files": "pattern", "search_text": "query",
               "write_file": "path", "edit_file": "path", "run_command": "command", "fetch_url": "url",
               "web_search": "query"}.get(name)
        if key is not None:
            v = a.get(key)
            if v in (None, ""):
                return "." if name == "list_dir" else "(" + key + " missing)"
            return _one_line(v)
        if a:
            first = next(iter(a.values()))
            return _one_line(first if isinstance(first, str) else json.dumps(a, default=str, ensure_ascii=False))
        return ""

    def _permitted(self, name: str, budget: ToolBudget) -> bool:
        """Is `name` offered to this answer? The prompt's list (`allowed`) AND the trust tier: an
        active tool is refused for any kind outside `ACTIVE_KINDS` even if a hand-built budget
        lists it."""
        if name not in budget.allowed:
            return False
        if budget.kind in NEVER_KINDS:
            return False
        if name in READ_TOOLS:
            return True
        if name in ACTIVE_TOOLS or name.startswith(MCP_PREFIX):
            return budget.kind in ACTIVE_KINDS
        return False

    def _run(self, call: ToolCall, budget: ToolBudget, cancel: threading.Event | None) -> ToolResult:
        name = str(getattr(call, "name", "") or "").strip()
        raw_args = getattr(call, "args", None)
        args = self._normalise_args(name, raw_args if isinstance(raw_args, dict) else {})
        summary = self._summary_for(name, args)
        if cancel is not None and cancel.is_set():
            return ToolResult("error: cancelled", "error", "cancelled", summary)
        key = name + "\x00" + json.dumps(args, sort_keys=True, default=str, ensure_ascii=False)
        prev = budget.seen.get(key)
        if prev is not None:            # a repeated call is answered from memory, free of charge
            state = "denied" if prev.startswith("denied:") else "error" if prev.startswith("error:") else "done"
            return ToolResult("(same call as before)\n" + prev, state, "same call as before", summary)
        budget.calls_left = max(0, budget.calls_left - 1)     # every other call costs one, refused ones too
        if not self._permitted(name, budget):
            return self._finish(budget, key, name, "denied", "denied: tool not available for this kind of answer", summary)
        if budget.deadline > 0 and self._clock() > budget.deadline:     # a deadline of 0 means "no limit"
            return self._finish(budget, key, name, "denied",
                                "denied: the time for tool calls in this answer is spent; write the answer now", summary)
        call_cancel = _CallCancel(cancel, budget.deadline if budget.deadline > 0 else None, self._clock)
        try:
            text = self._dispatch(name, args, budget, cancel, call_cancel)
            state = "done"
        except (PathDenied, _Refusal) as exc:
            text, state = str(exc), "denied"
        except _Failure as exc:
            text, state = str(exc), "error"
        except CapabilityError as exc:
            text, state = str(exc), "error"
        except Exception as exc:      # noqa: BLE001 - the model reads the message, never a traceback
            log.warning("tool %s failed", name, exc_info=True)
            text, state = f"{exc}" if str(exc) else type(exc).__name__, "error"
        if (state == "error" and call_cancel.expired and "ran out" not in text
                and not (cancel is not None and cancel.is_set())):
            text = "error: stopped: the time for this answer ran out"      # a call the deadline cut short
        if state != "done" and not text.startswith(state + ":"):
            text = f"{state}: {text}"
        return self._finish(budget, key, name, state, text, summary)

    def _finish(self, budget: ToolBudget, key: str, name: str, state: str, text: str, summary: str) -> ToolResult:
        """Sanitise and cut the text, remember it for the duplicate check, flip `read_done` and
        build the result with its one-line detail."""
        text = _cut(sanitize_untrusted(text), budget.chars)
        if not text.strip():
            text = "(empty result)"
        if state != "denied" and (name in _MUTATING_TOOLS or name.startswith(MCP_PREFIX)):
            budget.seen.clear()          # the world may have changed (even a call that failed half way): every earlier read is stale
        budget.seen[key] = text
        if state == "done" and (name in READ_TOOLS or name.startswith(MCP_PREFIX) or name == "run_command"):
            budget.read_done = True
        if state == "done":
            detail = _size_hint(len(text.encode("utf-8", "replace")))
        else:
            detail = _one_line(text.split(":", 1)[1].strip() if ":" in text else text)
        return ToolResult(text, state, detail, summary)

    def _ctx(self, tool: str, budget: ToolBudget, cancel: threading.Event | None) -> _Ctx:
        return _Ctx(tool, self._roots(), self._excludes(), self._protected_norm(),
                    _Stop(budget.deadline if budget.deadline > 0 else None, self._clock, cancel), cancel,
                    int(budget.chars))

    def _dispatch(
        self, name: str, a: dict[str, Any], budget: ToolBudget, cancel: threading.Event | None,
        call_cancel: threading.Event | None = None,
    ) -> str:
        """Check the live switch for `name`, then run it. Raises :class:`_Refusal`,
        :class:`PathDenied` or :class:`_Failure` with the sentence for the model. Every tool gets
        ONE cancel event for its call (`call_cancel`: set when the job's `cancel` is set OR the
        answer's deadline passes), which the walkers, the command runner, the fetcher and the MCP
        client poll."""
        web_cancel = call_cancel if call_cancel is not None else cancel
        if name in READ_TOOLS:
            if not self._on("read_files"):
                raise _Refusal("denied: Read files is switched off")
            if not _str_list(getattr(self._cc, "read_roots", [])):
                raise _Refusal("denied: no readable folder is configured")
            return getattr(self, "_t_" + name)(a, self._ctx(name, budget, web_cancel))
        if name in ("write_file", "edit_file"):
            if not self._on("edit_files"):
                raise _Refusal("denied: Edit files is switched off")
            return getattr(self, "_t_" + name)(a)
        if name == "run_command":
            if not self._on("execute_commands"):
                raise _Refusal("denied: Execute commands is switched off")
            return self._t_run_command(a, self._ctx(name, budget, web_cancel))
        if name in _WEB_TOOLS:
            if not self._on("fetch_web"):
                raise _Refusal("denied: Fetch web content is switched off")
            if budget.read_done:
                raise _Refusal("denied: no web access after reading local data in the same answer; "
                               "search first, read after")
            if budget.web_left <= 0:
                raise _Refusal("denied: no web fetches are left for this answer")
            if name == "fetch_url":
                self._fetch_target(a)            # a refused address does not spend a fetch
            budget.web_left -= 1
            return self._t_fetch_url(a, web_cancel) if name == "fetch_url" else self._t_web_search(a, web_cancel)
        if name.startswith(MCP_PREFIX):
            if not self._on("mcp_servers"):
                raise _Refusal("denied: MCP servers are switched off")
            return self._t_mcp(name, a, web_cancel)
        raise _Refusal("denied: tool not available for this kind of answer")

    # ---- argument helpers
    @staticmethod
    def _text(a: dict[str, Any], key: str, tool: str, *, required: bool = True, allow_empty: bool = False,
              limit: int = 200_000) -> str:
        v = a.get(key)
        if v is None:
            if required:
                raise _Failure(f"error: missing argument '{key}'. Usage: {_USAGE.get(tool, tool)}")
            return ""
        if isinstance(v, (dict, list, bool)):
            raise _Failure(f"error: argument '{key}' must be text. Usage: {_USAGE.get(tool, tool)}")
        s = str(v)
        if not allow_empty and not s.strip():
            if required:
                raise _Failure(f"error: argument '{key}' is empty. Usage: {_USAGE.get(tool, tool)}")
            return ""
        if len(s) > limit:
            raise _Failure(f"error: argument '{key}' is too long ({len(s)} chars, limit {limit})")
        return s

    @staticmethod
    def _opt_int(a: dict[str, Any], key: str) -> int | None:
        v = a.get(key)
        if v is None or v == "":
            return None
        if isinstance(v, bool) or not isinstance(v, (int, float)) or (isinstance(v, float) and not v.is_integer()):
            raise _Failure(f"error: argument '{key}' must be a whole number")
        return int(v)

    # ---- read tools
    def _targets(
        self, c: _Ctx, raw: Any, *, allow_file: bool = False,
    ) -> list[tuple[_Root, Path, _Res | None]]:
        """Where a list / find / search call looks: every root when `raw` is empty, else the one
        resolved folder (or file, for a search)."""
        text = "" if raw is None else str(raw).strip()
        if text in ("", ".", "./", ".\\"):
            if not c.roots:
                raise _Refusal("denied: no readable folder is configured (or none of them exists)")
            return [(r, r.real, None) for r in c.roots]
        res = self._resolve(text, roots=c.roots)
        shown = self._show(res.root, res.rel, c.multi)
        if res.path.is_file():
            if allow_file:
                return [(res.root, res.path, res)]
            raise _Failure(f"error: {shown} is a file; use read_file")
        if not res.path.is_dir():
            raise _Failure(f"error: {shown} does not exist.{self._suggest(c, res.path.name, want_dir=True)}")
        return [(res.root, res.path, res)]

    #: A "does not exist" error names at most this many same-named entries found elsewhere, after
    #: a walk of at most this many entries (bounded like `find_files`; the deadline stops it too).
    _SUGGEST_CAP = 3
    _SUGGEST_SCAN = 20_000

    def _suggest(self, c: _Ctx, name: str, *, want_dir: bool) -> str:
        """The tail of a "does not exist" error: files (or folders, `want_dir`) of the same name
        elsewhere in the readable folders, so a model that trusted a stale path recovers in one
        call instead of guessing. 2026-10-08, the owner's first live session: the project map had
        been made in the morning, the folders were renamed in the afternoon, and the evening's
        first question asked for ``Interview3_Prep/Research/R3_...md`` when the file had become
        ``Research/R3_...md``. Falls back to the usual hint when nothing matches."""
        hint = " Use list_dir to see the folders." if want_dir else " Use list_dir or find_files to see the files."
        key = (name or "").casefold()
        if not key:
            return hint
        found: list[str] = []
        scanned = 0
        for root in c.roots:
            for rel, ent in _walk(root.real, root.real, c.excl, c.prot, c.stop):
                scanned += 1
                if scanned > self._SUGGEST_SCAN:
                    break
                if ent.is_dir == want_dir and ent.name.casefold() == key:
                    found.append(self._show(root, rel, c.multi) + ("/" if ent.is_dir else ""))
                    if len(found) >= self._SUGGEST_CAP:
                        break
            if len(found) >= self._SUGGEST_CAP or scanned > self._SUGGEST_SCAN:
                break
        if not found:
            return hint
        return (" Did you mean: " + ", ".join(found) + "? (the folders may have changed since the project "
                "map was made; the map's path is out of date)")

    def _t_read_file(self, a: dict[str, Any], c: _Ctx) -> str:
        res = self._resolve(self._text(a, "path", "read_file"), roots=c.roots)
        shown = self._show(res.root, res.rel, c.multi)
        try:
            st = os.stat(res.path)
        except FileNotFoundError:
            raise _Failure(f"error: {shown} does not exist.{self._suggest(c, res.path.name, want_dir=False)}") from None
        except OSError as exc:
            raise _Failure(f"error: cannot read {shown}: {exc.strerror or exc}") from None
        if stat_mod.S_ISDIR(st.st_mode):
            raise _Failure(f"error: {shown} is a folder; use list_dir")
        if not stat_mod.S_ISREG(st.st_mode):
            raise _Failure(f"error: {shown} is not a regular file")
        if _is_cloud_stat(st):
            raise _Refusal(f"denied: {shown} is an online-only cloud file (not stored on this machine)")
        maxc = min(self._int("read_max_file_chars", 20000, 200, 1_000_000), max(200, c.chars - 160))
        start = self._opt_int(a, "start_line")
        end = self._opt_int(a, "end_line")
        if start is not None and start < 1:
            start = 1
        if end is not None and end < (start or 1):
            raise _Failure("error: end_line is before start_line")
        try:
            with open(res.path, "rb") as fh:
                if b"\x00" in fh.read(_PROBE_BYTES):
                    raise _Failure(f"error: {shown} looks like a binary file ({_size_hint(st.st_size)}); not shown")
                fh.seek(0)
                if start is None and end is None:
                    return self._read_whole(fh, st.st_size, maxc)
                return self._read_range(fh, start or 1, end, maxc, c.stop)
        except OSError as exc:
            raise _Failure(f"error: cannot read {shown}: {exc.strerror or exc}") from None

    @staticmethod
    def _read_whole(fh: Any, size: int, maxc: int) -> str:
        """The start of a file: at most `maxc * 4` bytes are read (never the whole of a huge file),
        decoded as UTF-8 (a BOM dropped, bad bytes replaced), CRLF shown as LF, cut to `maxc`."""
        limit = maxc * 4
        data = fh.read(limit + 1)
        more = len(data) > limit
        text = codecs.getincrementaldecoder("utf-8-sig")(errors="replace").decode(
            data[:limit], final=not more).replace("\r\n", "\n")
        hint = "pass start_line / end_line to read further"
        if len(text) > maxc:
            if more:
                return text[:maxc] + f"\n(cut: shows {maxc} chars of a {_size_hint(size)} file; {hint})"
            return text[:maxc] + f"\n(cut: shows {maxc} of {len(text)} chars; {hint})"
        if more:
            return text + f"\n(cut: shows the first {len(text)} chars of a {_size_hint(size)} file; {hint})"
        return text if text else "(empty file)"

    @staticmethod
    def _read_range(fh: Any, start: int, end: int | None, maxc: int, stop: Callable[[], bool]) -> str:
        """Lines `start`..`end` (1-based), streamed in binary mode so a huge log is never loaded."""
        out: list[str] = []
        chars = 0
        lineno = 0
        consumed = 0
        reason = ""
        while True:
            raw = fh.readline(maxc * 4 + 1)
            if not raw:
                reason = "eof"
                break
            lineno += 1
            consumed += len(raw)
            if consumed > _SEARCH_TOTAL_BYTES:
                reason = "scan"
                break
            if lineno % 512 == 0 and stop():
                reason = "stop"
                break
            if lineno < start:
                continue
            if end is not None and lineno > end:
                reason = "end"
                break
            line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
            if lineno == 1:
                line = line.lstrip("\ufeff")
            if len(line) > maxc:
                line = line[:maxc] + "..."
            out.append(line)
            chars += len(line) + 1
            if chars >= maxc:
                reason = "cap"
                break
        if not out and reason == "eof":
            return f"(the file has only {lineno} lines)"
        text = "\n".join(out)
        if reason == "end":
            text += f"\n(more lines follow; continue with start_line={(end or 0) + 1})"
        elif reason == "cap":
            text += f"\n(cut at {maxc} chars; continue with start_line={lineno + 1})"
        elif reason == "scan":
            text += "\n(stopped: scan limit reached; the file is very large)"
        elif reason == "stop":
            text += f"\n(stopped: {getattr(stop, 'reason', '') or 'time limit'})"
        return text or "(no lines)"

    def _t_list_dir(self, a: dict[str, Any], c: _Ctx) -> str:
        depth = max(1, min(3, self._opt_int(a, "depth") or 1))
        lines: list[str] = []
        state = {"count": 0, "capped": False}

        def emit(dirpath: str, dparts: tuple[str, ...], level: int) -> None:
            ents = _scan_dir(dirpath, c.excl, dparts, c.prot)
            ents.sort(key=lambda e: (not e.is_dir, e.name.casefold()))
            for e in ents:
                if state["count"] >= _LIST_CAP:
                    state["capped"] = True
                    return
                state["count"] += 1
                pad = "  " * level
                if e.is_dir:
                    lines.append(f"{pad}{e.name}/")
                    if level < depth and not c.stop():
                        emit(e.path, dparts + (e.name,), level + 1)
                else:
                    lines.append(f"{pad}{e.name}  ({_size_hint(e.size)})")

        for root, start, res in self._targets(c, a.get("path")):
            parts = tuple(res.rel.split("/")) if (res is not None and res.rel) else ()
            lines.append(self._show(root, "/".join(parts), c.multi).rstrip("/") + "/")
            before = len(lines)
            emit(str(start), parts, 1)
            if len(lines) == before:
                lines.append("  (empty)")
        if state["capped"]:
            lines.append(f"(stopped: {_LIST_CAP} entries shown; list a sub-folder for more)")
        if c.stop.reason:
            lines.append(f"(stopped: {c.stop.reason})")
        return "\n".join(lines)

    def _t_find_files(self, a: dict[str, Any], c: _Ctx) -> str:
        pattern = self._text(a, "pattern", "find_files", limit=300).strip()
        hits: list[str] = []
        scanned = 0
        why = ""
        for root, start, _res in self._targets(c, a.get("path")):
            for rel, ent in _walk(root.real, start, c.excl, c.prot, c.stop):
                scanned += 1
                if scanned > _SCAN_CAP:
                    why = "scan limit"
                    break
                if _glob_match(pattern, rel, ent.name):
                    hits.append(self._show(root, rel, c.multi) + ("/" if ent.is_dir else ""))
                    if len(hits) >= _FIND_CAP:
                        why = f"{_FIND_CAP} matches"
                        break
            if why:
                break
        why = why or c.stop.reason
        text = "\n".join(hits) if hits else f"(no file or folder matches {pattern!r})"
        if why:
            text += f"\n(stopped: {why} after {scanned} entries; narrow the pattern or path)"
        return text

    def _t_search_text(self, a: dict[str, Any], c: _Ctx) -> str:
        query = self._text(a, "query", "search_text", limit=500)
        needle = query.strip("\r\n").casefold()
        glob = self._text(a, "glob", "search_text", required=False, limit=200).strip()
        targets = self._targets(c, a.get("path"), allow_file=True)

        def candidates() -> Iterator[tuple[_Root, str, _Ent]]:
            for root, start, res in targets:
                if res is not None and res.path.is_file():
                    try:
                        size = res.path.stat().st_size
                    except OSError:
                        continue
                    yield root, res.rel, _Ent(res.path.name, str(res.path), False, size)
                else:
                    for rel, ent in _walk(root.real, start, c.excl, c.prot, c.stop):
                        if not ent.is_dir:
                            yield root, rel, ent

        hits: list[str] = []
        files = 0
        skipped_big = 0
        scanned = 0
        why = ""
        for root, rel, ent in candidates():
            if glob and not _glob_match(glob, rel, ent.name):
                continue
            files += 1
            if files > _SEARCH_MAX_FILES:
                why = "file limit"
                break
            if ent.size > _SEARCH_FILE_BYTES:
                skipped_big += 1
                continue
            if not _verify_entry(root.real, ent.path, c.excl):
                continue
            try:
                with open(ent.path, "rb") as fh:
                    data = fh.read(_SEARCH_FILE_BYTES + 1)
            except OSError:
                continue
            if b"\x00" in data[:_PROBE_BYTES]:
                continue
            scanned += len(data)
            if scanned > _SEARCH_TOTAL_BYTES:
                why = "scan limit"
                break
            text = data.decode("utf-8", errors="replace")
            if needle not in text.casefold():
                continue
            shown = self._show(root, rel, c.multi)
            for n, line in enumerate(text.splitlines(), 1):
                if len(line) > 2000:
                    continue
                if needle in line.casefold():
                    hits.append(f"{shown}:{n}: {line.strip()[:200]}")
                    if len(hits) >= _HIT_CAP:
                        why = f"{_HIT_CAP} matches"
                        break
            if why or c.stop():
                break
        why = why or c.stop.reason
        out = "\n".join(hits) if hits else f"(no matches for {query!r} in {files} files)"
        if skipped_big and not hits:
            out += f"\n({skipped_big} files over 1 MiB were skipped)"
        if why:
            out += f"\n(stopped: {why}; narrow the query, path or glob)"
        return out

    # ---- edit tools
    def _show_edit(self, res: _Res) -> str:
        return self._show(res.root, res.rel, len(self._roots(edit=True)) > 1)

    @staticmethod
    def _load_text_file(p: Path, shown: str) -> tuple[bool, str, str, bytes, os.stat_result]:
        """A file to be edited, read as BYTES: ``(has BOM, text, line ending, raw bytes, stat)``.
        Refuses a folder, a file over 2 MiB, a binary file and anything that is not valid UTF-8."""
        try:
            st = os.stat(p)
            if stat_mod.S_ISDIR(st.st_mode):
                raise _Failure(f"error: {shown} is a folder")
            if st.st_size > _MAX_EDIT_BYTES:
                raise _Failure(f"error: {shown} is larger than 2 MiB; not edited")
            data = p.read_bytes()
        except FileNotFoundError:
            raise _Failure(f"error: {shown} does not exist") from None
        except OSError as exc:
            raise _Failure(f"error: cannot read {shown}: {exc.strerror or exc}") from None
        if b"\x00" in data[:_PROBE_BYTES]:
            raise _Refusal(f"denied: {shown} is a binary file")
        bom = data.startswith(_BOM)
        try:
            text = (data[len(_BOM):] if bom else data).decode("utf-8")
        except UnicodeDecodeError:
            raise _Failure(f"error: {shown} is not valid UTF-8 text; not edited") from None
        crlf = text.count("\r\n")
        lf = text.count("\n") - crlf
        return bom, text, "\r\n" if crlf > lf else "\n", data, st

    def _backup(self, original: Path, data: bytes) -> bool:
        """Copy the original bytes to ``<session>/edits/<HHMMSS>_<n>_<name>`` (the counter makes a
        second backup of the same file in the same second a new file, never an overwrite).
        Returns whether a backup was made; raises :class:`_Refusal` when backups are on and there
        is no session folder."""
        flag = getattr(self._cc, "edit_backup", True)
        if isinstance(flag, (bool, int)) and not flag:
            return False
        try:
            folder = self._session_folder()
        except Exception:      # noqa: BLE001 - a recorder that has not started yet
            folder = None
        if not folder:
            raise _Refusal("denied: no session folder for backups")
        try:
            target_dir = Path(folder) / "edits"
            target_dir.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%H%M%S")
            n = 1
            while True:
                target = target_dir / f"{stamp}_{n}_{original.name}"
                try:
                    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0))
                except FileExistsError:
                    n += 1
                    continue
                with os.fdopen(fd, "wb") as fh:
                    fh.write(data)
                return True
        except OSError as exc:
            raise _Failure(f"error: could not save the backup: {exc.strerror or exc}; the file was not changed") from None

    def _t_write_file(self, a: dict[str, Any]) -> str:
        content = self._text(a, "content", "write_file", allow_empty=True, limit=_MAX_WRITE_CHARS)
        res = self._resolve(self._text(a, "path", "write_file"), edit=True)
        shown = self._show_edit(res)
        p = res.path
        if not res.rel:
            raise _Failure(f"error: {shown} is a folder")
        exists = os.path.lexists(p)
        backed = False
        st: os.stat_result | None = None
        if exists:
            bom, _old, nl, raw, st = self._load_text_file(p, shown)
            data = (_BOM if bom else b"") + _norm_nl(content, nl).encode("utf-8")
            backed = self._backup(p, raw)
        else:
            try:
                p.parent.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise _Failure(f"error: cannot create the folder for {shown}: {exc.strerror or exc}") from None
            data = content.replace("\r\n", "\n").encode("utf-8")
        try:
            _atomic_write_bytes(p, data, st)
        except OSError as exc:
            raise _Failure(f"error: could not write {shown}: {exc.strerror or exc}") from None
        return f"{'wrote' if exists else 'created'} {shown} ({len(content)} chars{'; the original is backed up' if backed else ''})"

    def _t_edit_file(self, a: dict[str, Any]) -> str:
        old_in = self._text(a, "old", "edit_file")
        new_in = self._text(a, "new", "edit_file", allow_empty=True)
        res = self._resolve(self._text(a, "path", "edit_file"), edit=True)
        shown = self._show_edit(res)
        if not res.rel or not os.path.lexists(res.path):
            raise _Failure(f"error: {shown} does not exist; use write_file to create it")
        bom, text, nl, raw, st = self._load_text_file(res.path, shown)
        old, new = _norm_nl(old_in, nl), _norm_nl(new_in, nl)
        if old == new:
            raise _Failure("error: old and new are identical; nothing to change")
        count = text.count(old)
        if count == 0:
            raise _Failure(f"error: the text to replace was not found in {shown}. Read the file again and copy "
                           "the lines exactly (whitespace counts)")
        if count > 1:
            raise _Failure(f"error: the text to replace occurs {count} times in {shown}; add surrounding lines "
                           "so that it matches exactly once")
        data = (_BOM if bom else b"") + text.replace(old, new, 1).encode("utf-8")
        backed = self._backup(res.path, raw)
        try:
            _atomic_write_bytes(res.path, data, st)
        except OSError as exc:
            raise _Failure(f"error: could not write {shown}: {exc.strerror or exc}") from None
        return (f"edited {shown}: 1 replacement ({old.count(chr(10)) + 1} line(s) -> {new.count(chr(10)) + 1} line(s))"
                + ("; the original is backed up" if backed else ""))

    # ---- commands
    def _exists_under(self, cwd: Path, token: str) -> bool:
        try:
            return os.path.lexists(os.path.join(str(cwd), token))
        except (OSError, ValueError):
            return False

    def _command_cwd(self, a: dict[str, Any], c: _Ctx) -> Path:
        """The folder a command runs in: the call's ``cwd`` (resolved under a readable folder),
        else ``command_cwd`` (must sit under one), else the first readable folder."""
        raw = a.get("cwd")
        if raw is not None and str(raw).strip():
            res = self._resolve(str(raw), roots=c.roots)
            if not res.path.is_dir():
                raise _Failure(f"error: {self._show(res.root, res.rel, c.multi)} is not a folder")
            return res.path
        cfg_cwd = getattr(self._cc, "command_cwd", "")
        if isinstance(cfg_cwd, str) and cfg_cwd.strip():
            try:
                res = self._resolve(cfg_cwd, roots=c.roots)
                if res.path.is_dir():
                    return res.path
            except PathDenied as exc:
                self._warn_once("command_cwd", f"capabilities.command_cwd ignored: {exc}")
        if not c.roots:
            raise _Refusal("denied: no readable folder to run the command in")
        return c.roots[0].real

    def _t_run_command(self, a: dict[str, Any], c: _Ctx) -> str:
        command = self._text(a, "command", "run_command", limit=2000)
        cwd = self._command_cwd(a, c)
        gated = not self._on("allow_all_commands")
        final = self._gate_command(command, cwd, c) if gated else command.strip()
        timeout = self._float("command_timeout_s", 15.0, 0.1, 600.0)
        limit = self._int("command_output_chars", 6000, 200, 200_000)
        try:
            out, outcome, rc = _run_process(
                final, str(cwd), timeout_s=timeout, out_chars=limit, stop=c.stop, cancel=c.cancel,
                env=_child_env())
        except (OSError, ValueError) as exc:
            raise _Failure(f"error: could not start the command: {exc}") from None
        if gated and final[:4].casefold() == "git ":
            out = _RE_URL_CRED.sub("***@", out)      # `git remote -v` prints the credentials in a url
        partial = ("\n" + out) if out.strip() else ""
        if outcome == "timeout":
            raise _Failure(f"error: timed out after {timeout:g} s; the command was stopped{partial}")
        if outcome == "cancelled" and getattr(c.cancel, "expired", False):
            outcome = "deadline"                 # the call's own event fired on the answer's deadline
        if outcome == "cancelled":
            raise _Failure(f"error: cancelled{partial}")
        if outcome == "deadline":
            raise _Failure(f"error: stopped: the time for this answer ran out{partial}")
        text = out if out.strip() else "(no output)"
        if rc not in (0, None):
            text += f"\n(exit code {rc})"
        return text

    @staticmethod
    def _gate_arity(cmd0: str, rest: list[str]) -> None:
        """``hostname`` and ``date`` take no argument here: ``hostname NAME`` renames the machine
        (POSIX) and ``date VALUE`` sets the clock when the process may. Windows ``date`` may carry
        ``/t``; POSIX ``date`` only ``-u`` and a ``+FORMAT``."""
        if cmd0 == "hostname" and rest:
            raise _Refusal("denied: hostname takes no arguments here")
        if cmd0 == "date":
            low = [_unquote(t).casefold() for t in rest]
            ok = (not low or low == ["/t"]) if os.name == "nt" else all(t == "-u" or t.startswith("+") for t in low)
            if not ok:
                raise _Refusal("denied: date only shows the date here (date /t); it could set the clock")

    @staticmethod
    def _pattern_in_flag(cmd0: str, rest: list[str]) -> bool:
        """Does an option supply the search pattern (``-e`` / ``-f`` / ``--regexp`` / ``--file``,
        findstr's ``/c:`` ``/g:``)? Then every positional argument is a FILE, none is the pattern.
        Errs towards True: a cluster holding an ``e`` or ``f`` counts."""
        for tok in rest:
            bare = _unquote(tok)
            if cmd0 == "findstr":
                if os.name == "nt" and len(bare) > 1 and bare[0] in "/-" and ":" in bare:
                    if any(ch in _findstr_letters(bare) for ch in "cg"):
                        return True
                continue
            if bare == "--":
                break
            if bare.startswith("--"):
                if bare.casefold().startswith(("--regexp", "--file")):
                    return True
            elif len(bare) > 1 and bare.startswith("-") and re.fullmatch(r"-[A-Za-z0-9]*[ef][A-Za-z0-9]*", bare):
                return True
        return False

    @staticmethod
    def _value_follows(cmd0: str, bare: str) -> bool:
        """Is the NEXT argument the value of the option `bare` (a pattern, a glob, a number), never a
        path or a mask? Only for grep and rg, and only for the separate-token spelling: a long option
        from the table, or a short cluster whose FIRST value-taking letter is its last (``-A``,
        ``-nA``, ``-ie``; in ``-Ae`` the ``e`` is A's value). ``-f`` takes a file, which is gated
        like any path, so it never counts."""
        letters = _ARG_LETTERS.get(cmd0)
        if letters is None:
            return False
        if bare.startswith("--"):
            return bare in _SEPARATE_VALUE_LONG[cmd0]
        m = re.fullmatch(r"-([A-Za-z0-9]+)", bare)
        if m is None:
            return False
        cluster = m.group(1)
        for i, ch in enumerate(cluster):
            if ch in letters:
                return i == len(cluster) - 1 and ch not in "fr"
        return False

    def _gate_command(self, command: str, cwd: Path, c: _Ctx) -> str:
        """The argv gate (everything but ``allow_all_commands``): returns the command line to run,
        or raises :class:`_Refusal`. Order: control / chaining characters, tokenise, allowlist
        prefix, git rules, then every argument (flags against the deny-list, path-like tokens
        through the path resolver), then the rewrites: git runs with hooks, pagers, external diff
        and optional locks off and, for diff / log / show, inside the folder it runs in and without
        the excluded names; rg gets the excludes as globs that come last; pip is kept off the
        network; a forward-slash path becomes a backslash one for the cmd.exe commands."""
        cmd = command.strip()
        if not cmd:
            raise _Failure("error: the command is empty")
        if len(cmd) > 1000:
            raise _Refusal("denied: the command is longer than 1000 characters")
        if (any(ch in cmd for ch in _CHAIN_CHARS) or "$(" in cmd or "${" in cmd
                or any(ord(ch) < 32 and ch != "\t" for ch in cmd) or (os.name != "nt" and "$" in cmd)):
            raise _Refusal("denied: chaining / redirection (& | ; < > ^ ` $( ${ %) is not allowed in a command")
        if _has_lookalikes(cmd):
            raise _Refusal("denied: look-alike characters (fullwidth, dash or quote variants) are not allowed in a command")
        try:
            toks = _split_command(cmd)
        except ValueError:
            raise _Refusal("denied: unbalanced quotes in the command") from None
        if not toks:
            raise _Failure("error: the command is empty")
        ntoks = [t.casefold() for t in toks]
        allowed = _str_list(getattr(self._cc, "allowed_commands", []))
        if not any(ntoks[: len(e.split())] == e.casefold().split() for e in allowed if e.split()):
            shown = ", ".join(allowed[:14]) + (", ..." if len(allowed) > 14 else "")
            raise _Refusal(f"denied: '{' '.join(toks[:2])}' is not an allowed command. Allowed: {shown}")
        cmd0 = ntoks[0]
        git = cmd0 == "git"
        rest = toks[2:] if git else toks[1:]
        sub = ntoks[1] if git and len(ntoks) > 1 else ""
        if cmd0 == "findstr" and not cmd.isascii():
            raise _Refusal("denied: findstr takes ASCII only here (it reads other characters as switches); "
                           "use search_text")
        self._gate_arity(cmd0, rest)
        if git:
            nrest = [t.casefold() for t in rest]
            if sub == "remote" and nrest != ["-v"]:
                raise _Refusal("denied: only 'git remote -v' is allowed")
            if sub == "stash" and nrest[:1] != ["list"]:
                raise _Refusal("denied: only 'git stash list' is allowed")
            if sub == "branch":
                list_mode = "--list" in nrest
                for t in nrest:
                    if t not in ("-a", "-v", "-vv", "--list", "--all", "--verbose") and not (list_mode and not t.startswith("-")):
                        raise _Refusal("denied: git branch is allowed bare, or with -a, -v or --list [pattern]")
        pat_flag = cmd0 in _PATTERN_CMDS and self._pattern_in_flag(cmd0, rest)
        pattern_pending = cmd0 in _PATTERN_CMDS and not pat_flag
        dd_cmd = git or cmd0 == "rg"          # `--` ends the options (git: and starts the pathspecs)
        after_dd = False
        skip_value = False
        new_rest: list[str] = []
        kinds: list[str] = []                  # per argument: "flag" / "path" / "rev" / "word"
        for tok in rest:
            bare = _unquote(tok)
            if dd_cmd and bare == "--" and not after_dd:
                after_dd = True
                new_rest.append(tok)
                kinds.append("flag")
                continue
            if skip_value:                      # the value of -A / -e / -g ...: a number, a pattern, a glob
                skip_value = False
                new_rest.append(tok)
                kinds.append("word")
                continue
            new_tok, kind = self._gate_token(tok, cmd0=cmd0, git=git, after_dd=after_dd, cwd=cwd, c=c,
                                             pattern_ok=pattern_pending)
            if kind != "flag":
                pattern_pending = False
            elif not after_dd and self._value_follows(cmd0, bare):
                skip_value = True
            new_rest.append(new_tok)
            kinds.append(kind)
        if git:
            return self._git_line(sub, new_rest, kinds, cwd, c)
        if cmd0 == "rg":
            return self._rg_line(toks[0], new_rest, cwd, c)
        if cmd0 == "pip" and new_rest and new_rest[0].casefold() in ("list", "show"):
            extra = ["--disable-pip-version-check"] + (["--no-index"] if new_rest[0].casefold() == "list" else [])
            return " ".join([toks[0], new_rest[0], *extra, *new_rest[1:]])
        if os.name == "nt" and cmd0 in _SLASH_SWITCH_CMDS:
            skip = -1
            if cmd0 == "findstr" and not pat_flag:
                skip = next((i for i, k in enumerate(kinds) if k != "flag"), -1)      # the search string
            out_rest = [_backslash_token(t) if k == "path" and i != skip and "/" in t else t
                        for i, (t, k) in enumerate(zip(new_rest, kinds))]
            if out_rest != new_rest:
                return " ".join([toks[0], *out_rest])
        return cmd

    # ---- the rewrites of a gated command
    def _exclude_forms(self, cwd: Path, c: _Ctx, *, deep: str, cmd0: str) -> list[str]:
        """Every exclude pattern (and every protected path below `cwd`) as a glob relative to `cwd`,
        for rg and git, which both match relative to the folder they run in. A pattern without a
        slash matches a name at any depth (`deep` is its prefix: "" for rg, whose globs already
        work that way, ``**/`` for git); a pattern with a slash is anchored at its ROOT, so it is
        emitted as ``**/<pattern>`` (over-excluding is the safe direction) and, when `cwd` lies
        below the root on its way, also as what is left of it below `cwd`. A pattern that cannot be
        written safely on a command line refuses the command (never skipped silently)."""
        cwd_parts: tuple[str, ...] = ()
        for r in c.roots:
            try:
                cwd_parts = tuple(cwd.relative_to(r.real).parts)
                break
            except ValueError:
                continue
        forms: list[str] = []
        for pat in c.excl:
            if not _RE_GLOB_QUOTABLE.match(pat):
                raise _Refusal(f"denied: the exclude pattern {_one_line(pat, 40)!r} cannot be passed to {cmd0}; "
                               "use read_file / search_text")
            if "/" not in pat:
                forms.append(deep + pat)
                continue
            forms.append("**/" + pat)
            parts = pat.split("/")
            if (cwd_parts and len(parts) > len(cwd_parts)
                    and all(fnmatch.fnmatchcase(cw.casefold(), parts[i]) for i, cw in enumerate(cwd_parts))):
                forms.append("/".join(parts[len(cwd_parts):]))
        for p in self.protected_paths():
            try:
                rel = p.relative_to(cwd).as_posix()
            except ValueError:
                continue
            if rel and rel != "." and _RE_GLOB_QUOTABLE.match(rel):
                forms.append(rel)
        return list(dict.fromkeys(forms))

    def _git_line(self, sub: str, rest: list[str], kinds: list[str], cwd: Path, c: _Ctx) -> str:
        """The git command line: hooks, pager, fsmonitor, external diff and optional locks off. For
        diff / log / show the output is limited to the folder the command runs in (``-- .``, unless
        the model named its own paths) and the excluded names are removed with exclude pathspecs, so
        the whole repository's history never reaches the model. A path given before any ``--`` moves
        behind it."""
        head = ["git", "--no-pager", "--no-optional-locks", "-c", "core.fsmonitor=false",
                "-c", f"core.hooksPath={NUL_DEVICE}"]
        if sub == "diff":
            head += ["-c", "diff.autoRefreshIndex=false"]      # `git diff` would take .git/index.lock to refresh
        head.append(sub)
        if sub in _GIT_DIFFY:
            head += ["--no-ext-diff", "--no-textconv"]
        if sub not in _GIT_SCOPED:
            return " ".join(head + rest)
        dd = next((i for i, t in enumerate(rest) if _unquote(t) == "--"), -1)
        pre, post = (rest, []) if dd < 0 else (rest[:dd], rest[dd + 1:])
        opts = [t for t, k in zip(pre, kinds) if k != "path"]
        paths = [t for t, k in zip(pre, kinds) if k == "path"] + post
        specs: list[str] = []
        for form in self._exclude_forms(cwd, c, deep="**/", cmd0="git"):
            specs.append(f'":(exclude,icase,glob){form}"')
            specs.append(f'":(exclude,icase,glob){form}/**"')
        line = " ".join([*head, *opts, "--", *(paths or ["."]), *specs])
        if len(line) > _MAX_GATED_LINE:
            raise _Refusal("denied: too many exclude patterns to pass to git; use read_file / search_text")
        return line

    def _rg_line(self, prog: str, rest: list[str], cwd: Path, c: _Ctx) -> str:
        """The rg command line: no config file, the model's options, then the exclude globs LAST
        (ripgrep lets a later glob win, so a user ``-g '*.json'`` cannot re-include
        ``credentials.json``) and, behind the model's own ``--``, its operands."""
        dd = next((i for i, t in enumerate(rest) if _unquote(t) == "--"), -1)
        pre, post = (rest, []) if dd < 0 else (rest[:dd], rest[dd + 1:])
        globs: list[str] = []
        for form in self._exclude_forms(cwd, c, deep="", cmd0="rg"):
            globs.append(f"--iglob=!{form}" if _RE_GLOB_PLAIN.match(form) else f'"--iglob=!{form}"')
        parts = [prog, "--no-config", *pre, *globs]
        if dd >= 0:
            parts += ["--", *post]
        line = " ".join(parts)
        if len(line) > _MAX_GATED_LINE:
            raise _Refusal("denied: too many exclude patterns to pass to rg; use read_file / search_text")
        return line

    # ---- the per-argument gate
    def _gate_token(
        self, tok: str, *, cmd0: str, git: bool, after_dd: bool, cwd: Path, c: _Ctx, pattern_ok: bool = False,
    ) -> tuple[str, str]:
        """Check one argument. Returns ``(token to use, kind)``: kind is "flag" (an option), "path"
        (it names, or may name, a file), "rev" (a git ``<rev>:<path>``, rewritten to be relative to
        the folder) or "word"."""
        bare = _unquote(tok)
        if not bare:
            return tok, "word"
        # cmd.exe's own commands treat `,` `;` and `=` outside double quotes as argument
        # separators: `type README.md,\Windows\win.ini` reads both files, and no single token of
        # ours would be a path outside the folders. (`;` is refused with the chaining characters.)
        if (os.name == "nt" and cmd0 in _CMD_BUILTINS and ("," in tok or "=" in tok)
                and not (len(tok) >= 2 and tok[0] == tok[-1] == '"')):
            raise _Refusal(f"denied: ',' and '=' separate arguments for {cmd0}; put the argument in double quotes")
        if not after_dd and self._gate_flag(bare, cmd0, git, cwd, c):
            return tok, "flag"
        if cmd0 == "where":
            # where.exe reads `$ENV:pattern` and `dir:pattern`: names outside every folder
            tail = bare[2:] if re.match(r"^[A-Za-z]:[\\/]", bare) else bare
            if "$" in bare or ":" in tail:
                raise _Refusal("denied: where's $VAR:pattern and path:pattern forms are not allowed")
        new, is_path = self._gate_path(bare, cmd0, git, after_dd, cwd, c, pattern_ok=pattern_ok)
        if new != bare:
            return (f'"{new}"' if re.search(r"\s", new) else new), "rev"
        return tok, ("path" if is_path else "word")

    def _gate_flag(self, bare: str, cmd0: str, git: bool, cwd: Path, c: _Ctx) -> bool:
        """If `bare` is an option: check it against the deny-lists and return True. A value glued
        on with ``=`` (or ``/f:`` ``/g:``) is checked as a path."""
        if cmd0 in _NET_CMDS and _RE_REMOTE_SWITCH.fullmatch(bare):
            raise _Refusal(f"denied: the option {bare.partition(':')[0]} connects to another machine")
        if os.name == "nt" and cmd0 == "findstr" and len(bare) > 1 and bare[0] in "/-":
            self._gate_findstr_switch(bare, cmd0, cwd, c)
            return True
        slash = os.name == "nt" and not git and _RE_SLASH_FLAG.match(bare) is not None
        if not slash and not (bare.startswith("-") and len(bare) > 1):
            return False
        if slash:
            flag, sep, value = bare.partition(":")
        else:
            flag, sep, value = bare.partition("=")
        f = flag.casefold()
        if cmd0 in _PKG_CMDS and flag not in _PKG_FLAGS:
            raise _Refusal(f"denied: the option {flag} is not allowed for {cmd0}")
        if f in _DENY_FLAGS:
            raise _Refusal(f"denied: the option {flag} is not allowed")
        if f in _OUT_EXEMPT and cmd0 not in _OUT_EXEMPT[f]:
            raise _Refusal(f"denied: the option {flag} can write files and is not allowed for {cmd0}")
        if not git:
            if f in _RECURSE_FLAGS and cmd0 not in _RECURSE_OK:
                raise _Refusal(f"denied: the recursive option {flag} is not allowed for {cmd0}; "
                               "use find_files or search_text")
            if f == "/s" and cmd0 not in _SLASH_S_OK:
                raise _Refusal(f"denied: the recursive option {flag} is not allowed for {cmd0}")
            if not slash:
                self._gate_option_letters(flag, f, cmd0)
        if flag == "-L" and cmd0 in _FOLLOW_CMDS:
            raise _Refusal("denied: the option -L follows links and is not allowed")
        if value and f not in _GLOB_VALUE_FLAGS and ((not slash and sep == "=") or (slash and f in ("/f", "/g"))):
            self._gate_path(_unquote(value), cmd0, git, False, cwd, c)
        return True

    @staticmethod
    def _gate_option_letters(flag: str, f: str, cmd0: str) -> None:
        """The dash-options of GNU-style tools. A long option is checked against an allow-list for
        grep and ls (getopt accepts unique prefixes, ``--recurs`` is ``--recursive``); a cluster of
        short letters (any length, digits included) is scanned for the letters that recurse (r R),
        write (o), follow links (L) or, for grep, walk folders (d D) and, for rg, read archives (z)."""
        if flag.startswith("--"):
            allowed = _LONG_OK.get(cmd0)
            if allowed is not None and f not in allowed:
                raise _Refusal(f"denied: the option {flag} is not allowed for {cmd0}")
            return
        if cmd0 == "grep" and flag in ("-d", "-D"):
            raise _Refusal(f"denied: the option {flag} (directories / devices) is not allowed for grep; "
                           "use search_text")
        if cmd0 == "rg" and flag == "-z":
            raise _Refusal("denied: the option -z (search inside archives) is not allowed")
        # getopt tools take clusters of any length (`-rniHs`); other programs (java -version) have
        # single-dash words that only the short, historical pattern may read as a cluster
        if (_RE_CLUSTER if cmd0 in _GETOPT_CMDS else _RE_CLUSTER_SHORT).fullmatch(flag) is None:
            return
        letters = flag[1:]
        if any(ch in "rR" for ch in letters) and cmd0 not in _RECURSE_OK:
            raise _Refusal(f"denied: the recursive option {flag} is not allowed for {cmd0}; "
                           "use find_files or search_text")
        if "o" in letters and cmd0 not in _OUT_EXEMPT["-o"]:
            raise _Refusal(f"denied: the option {flag} can write files and is not allowed for {cmd0}")
        if "L" in letters and cmd0 in _FOLLOW_CMDS:
            raise _Refusal(f"denied: the option {flag} follows links and is not allowed")
        if cmd0 == "grep" and any(ch in "dD" for ch in letters):
            raise _Refusal(f"denied: the option {flag} (directories / devices) is not allowed for grep; "
                           "use search_text")
        if cmd0 == "rg" and "z" in letters:
            raise _Refusal(f"denied: the option {flag} (search inside archives) is not allowed")

    def _gate_findstr_switch(self, bare: str, cmd0: str, cwd: Path, c: _Ctx) -> None:
        """A findstr switch group, read by the LETTERS before its colon: findstr takes ``/`` and ``-``
        alike, lets ``/``, ``-`` and ``--`` start further switches inside one token (``/i/s``,
        ``-i/s``, ``/-s``) and reads ``/si`` as ``/s /i``. ``s`` (recurse), ``d`` (other folders) and
        ``f`` (a file naming the files to search) reach files the path gate never sees, in any
        cluster and any case; ``g`` reads its search strings from a file, which must itself be a
        readable path. (Switches only count before the search string; a token after it is a file.)"""
        letters = _findstr_letters(bare)
        if any(ch in letters for ch in "sdf"):
            raise _Refusal(f"denied: the findstr option {bare.partition(':')[0]} reaches other folders or "
                           "files than the ones named; use search_text")
        value = bare.partition(":")[2]
        if value and "g" in letters:
            self._gate_path(_unquote(value), cmd0, False, False, cwd, c)

    def _gate_rev_path(self, bare: str, cwd: Path, c: _Ctx) -> str:
        """``<rev>:<path>`` (also ``:<path>``, ``:<n>:<path>``): git resolves the path from the
        REPOSITORY root, so a ``..`` or a path of a sibling folder would read outside the read
        root. The part after the last colon is resolved like any path argument (excluded, protected,
        outside, ``..`` refused) and written back relative to the folder (``./``)."""
        segs = [s for s in re.split(r"[\\/:]+", bare[bare.index(":") + 1:]) if s]
        if ".." in segs:
            raise _Refusal("denied: '..' is not allowed in a path; use the full path")
        last = bare.rindex(":")
        norm = bare[last + 1:].replace("\\", "/")
        if norm in ("", "."):
            norm = ""
        self._resolve(norm or ".", base=cwd, roots=c.roots)
        tail = norm if norm.startswith("./") else "./" + norm
        return bare[: last + 1] + tail

    def _gate_path(
        self, bare: str, cmd0: str, git: bool, after_dd: bool, cwd: Path, c: _Ctx, *, pattern_ok: bool = False,
    ) -> tuple[str, bool]:
        """A non-option argument: if it is (or could be) a path it must resolve inside a readable
        folder, not excluded or protected; a wildcard is refused for every command that can show
        file CONTENT (the listers show names only; git filters by pathspec). Returns ``(argument to
        use, was it a path)``; a git ``<rev>:<path>`` comes back rewritten."""
        wild = any(ch in bare for ch in "*?")
        if wild and pattern_ok:
            return bare, False                 # the search pattern of grep / rg / findstr: a regex, not a mask
        if git and _RE_PATHSPEC_MAGIC.match(bare):
            # `:(top)x` and `:/x` name files from the REPOSITORY root, `:!x` re-includes: never allowed
            raise _Refusal("denied: pathspec magic (:(...) :/ :! :^) is not allowed; name the path")
        if git and not after_dd:
            if ":" in bare and not _RE_DRIVE_ANY.match(bare):          # <rev>:<path>
                return self._gate_rev_path(bare, cwd, c), False
            if _parts_excluded([x for x in _RE_SEPARATORS.split(bare) if x], c.excl):
                raise _Refusal("denied: that path is excluded from the read scope")
            path_like = (os.path.isabs(bare) or bare.startswith(("~", "\\\\", "//")) or self._exists_under(cwd, bare))
        else:
            path_like = (wild or "/" in bare or "\\" in bare or bare.startswith((".", "~")) or os.path.isabs(bare)
                         or _RE_DRIVE_ANY.match(bare) is not None or self._exists_under(cwd, bare)
                         or bare.split(".")[0].upper() in _RESERVED_NAMES)
        if not path_like:
            return bare, False
        if wild:
            cut = max(bare.rfind("/"), bare.rfind("\\"))
            folder = bare[:cut] if cut >= 0 else ""
            if any(ch in folder for ch in "*?"):
                raise _Refusal("denied: wildcards in folder names are not allowed")
            if ":" in (bare[2:] if _RE_DRIVE_ANY.match(bare) else bare):
                raise _Refusal("denied: ':' in a wildcard pattern is not allowed (a stream or a $VAR:pattern form)")
            if folder:
                self._resolve(folder, base=cwd, roots=c.roots)
            if cmd0 not in _LISTERS and not git:
                raise _Refusal(f"denied: wildcards are not allowed for {cmd0}; name the file or use search_text")
            return bare, True
        self._resolve(bare, base=cwd, roots=c.roots)
        return bare, True

    # ---- web tools (exocortex.webfetch, imported lazily)
    def _web_client(self) -> Any:
        """The injected fetcher, else a lazily built :class:`exocortex.webfetch.WebFetcher`."""
        with self._lock:
            if self._web is not None:
                return self._web
        try:
            from exocortex.webfetch import WebFetcher
        except ImportError:
            raise _Failure("error: exocortex.webfetch unavailable") from None
        client = WebFetcher(self._cfg)
        with self._lock:
            if self._web is None:
                self._web = client
            return self._web

    @staticmethod
    def _web_error(exc: BaseException) -> Exception:
        """A fetcher failure as the sentence for the model ("denied: ..." stays a refusal)."""
        msg = " ".join(str(exc).split()) or type(exc).__name__
        if msg.casefold().startswith("denied"):
            return _Refusal(msg)
        return _Failure(msg if msg.casefold().startswith("error") else f"error: {msg}")

    def _fetch_target(self, a: dict[str, Any]) -> str:
        """The address of a ``fetch_url`` call, or a refusal: an address longer than 256 characters
        or with a query string longer than 100 is how a prompt-injected instruction would carry
        text out of the prompt to a host of its choosing. A bandwidth limit, not a guarantee."""
        url = self._text(a, "url", "fetch_url", limit=20_000).strip()
        if len(url) > _WEB_URL_MAX:
            raise _Refusal(f"denied: an address longer than {_WEB_URL_MAX} characters is not fetched")
        query = url.partition("#")[0].partition("?")[2]
        if len(query) > _WEB_QUERY_MAX:
            raise _Refusal(f"denied: an address whose query string is longer than {_WEB_QUERY_MAX} characters "
                           "is not fetched")
        return url

    def _t_fetch_url(self, a: dict[str, Any], cancel: threading.Event | None) -> str:
        url = self._fetch_target(a)
        client = self._web_client()
        try:
            res = client.fetch(url, cancel=cancel)
        except Exception as exc:      # noqa: BLE001 - WebError and anything a transport raises
            raise self._web_error(exc) from None
        title = " ".join(str(getattr(res, "title", "") or "").split())
        final = str(getattr(res, "final_url", "") or url)
        body = str(getattr(res, "text", "") or "")
        head = (f"Title: {title}\n" if title else "") + f"URL: {final}\n\n"
        return head + (body if body.strip() else "(the page has no text)")

    def _t_web_search(self, a: dict[str, Any], cancel: threading.Event | None) -> str:
        query = self._text(a, "query", "web_search", limit=20_000).strip()[:_WEB_SEARCH_MAX]
        client = self._web_client()
        try:
            hits = list(client.search(query, cancel=cancel) or [])
        except Exception as exc:      # noqa: BLE001
            raise self._web_error(exc) from None
        if not hits:
            return f"(no results for {query!r})"
        lines: list[str] = []
        for i, h in enumerate(hits[: self._int("web_search_results", 5, 1, 20)], 1):
            title = " ".join(str(getattr(h, "title", "") or "").split()) or "(untitled)"
            url = str(getattr(h, "url", "") or "")
            snippet = " ".join(str(getattr(h, "snippet", "") or "").split())
            lines.append(f"{i}. {title}\n   {url}" + (f"\n   {snippet}" if snippet else ""))
        return "\n".join(lines)

    # ---- MCP (exocortex.mcp, imported lazily)
    def _mcp_configured(self) -> dict[str, dict[str, Any]]:
        """Every well-formed server entry of ``capabilities.mcp`` by name (enabled or not). An
        entry without a valid name (``[A-Za-z0-9_.-]{1,64}``) or a repeated one is skipped."""
        raw = getattr(self._cc, "mcp", [])
        out: dict[str, dict[str, Any]] = {}
        if not isinstance(raw, list):
            return out
        for spec in raw:
            if not isinstance(spec, dict):
                continue
            name = spec.get("name")
            if not isinstance(name, str) or not _NAME_OK.match(name):
                self._warn_once(f"mcp:name:{name!r}", f"capabilities.mcp: a server named {name!r} was skipped "
                                "(names are 1-64 letters, digits, '_', '.', '-')")
                continue
            if name in out:
                self._warn_once(f"mcp:dup:{name}", f"capabilities.mcp: the server {name!r} is listed twice; the first wins")
                continue
            out[name] = spec
        return out

    def _mcp_wanted(self) -> dict[str, dict[str, Any]]:
        """The servers that should be connected right now: the switch is on and the entry enabled."""
        if not self._on("mcp_servers"):
            return {}
        return {n: s for n, s in self._mcp_configured().items() if _spec_enabled(s)}

    @staticmethod
    def _mcp_args(schema: Any) -> tuple[tuple[str, str], ...]:
        """The first 8 properties of a tool's input schema as ``(name, "type, required: text")``."""
        props = schema.get("properties") if isinstance(schema, dict) else None
        if not isinstance(props, dict):
            return ()
        required = {r for r in (schema.get("required") or []) if isinstance(r, str)}
        out: list[tuple[str, str]] = []
        for pname, pdef in list(props.items())[:8]:
            if not isinstance(pname, str):
                continue
            ptype = pdef.get("type") if isinstance(pdef, dict) else ""
            desc = pdef.get("description") if isinstance(pdef, dict) else ""
            what = ", ".join(x for x in (str(ptype) if ptype else "", "required" if pname in required else "optional") if x)
            if isinstance(desc, str) and desc.strip():
                what += ": " + desc
            out.append((sanitize_untrusted(_one_line(pname, 40)), sanitize_untrusted(_one_line(what, 80))))
        return tuple(out)

    def _mcp_tool_specs(self) -> list[ToolSpec]:
        """The MCP tools of the CONNECTED, enabled servers, from the cached listing only (no
        connection attempt, no I/O)."""
        wanted = self._mcp_wanted()
        if not wanted:
            return []
        with self._lock:
            listing = {n: list(v) for n, v in self._mcp_listing.items()}
            states = {n: self._client_state(n) for n in self._mcp_state}
        out: list[ToolSpec] = []
        for server, tools in listing.items():
            if server not in wanted or states.get(server) != "connected":
                continue
            for t in tools:
                tname = str(getattr(t, "name", "") or "")
                if not _NAME_OK.match(tname):
                    continue
                desc = sanitize_untrusted(_one_line(getattr(t, "description", "") or "", 160))
                out.append(ToolSpec(f"{MCP_PREFIX}{server}:{tname}", f"[{server}] {desc}".strip(),
                                    self._mcp_args(getattr(t, "input_schema", None))))
        return out

    def _client_state(self, name: str) -> str:
        """The state of server `name` as the runner reports and offers it: its own bookkeeping, but
        a "connected" server whose client has since died (the stdio process exited, the reader hit
        EOF, an http server kept answering 5xx) reads as the client's own state, so its tools stop
        being offered and the next connect pass rebuilds it. A plain attribute read: no I/O. Call
        with ``self._lock`` held."""
        state = self._mcp_state.get(name, "idle")
        if state == "connected":
            live = getattr(self._mcp_clients.get(name), "state", "connected")
            if isinstance(live, str) and live != "connected":
                return live
        return state

    def _mark_suspect(self, server: str, exc: BaseException, cancel: threading.Event | None) -> None:
        """A call to `server` raised an MCP error: have the next :meth:`connect_mcp` rebuild its
        connection (a timeout, an expired http session, a wedged process). A call that was merely
        stopped by the user or the deadline leaves the connection usable and is not marked."""
        if cancel is not None and cancel.is_set():
            return
        try:
            from exocortex.mcp import McpError
        except ImportError:
            return
        if isinstance(exc, McpError):
            with self._lock:
                if server in self._mcp_clients:
                    self._mcp_suspect.add(server)

    def _make_mcp(self, spec: dict[str, Any]) -> Any:
        if self._mcp_factory is not None:
            return self._mcp_factory(spec)
        try:
            from exocortex.mcp import McpServer
        except ImportError:
            raise _Failure("error: exocortex.mcp unavailable") from None
        return McpServer(spec, timeout_s=self._float("mcp_timeout_s", 20.0, 1.0, 300.0))

    @staticmethod
    def _close_quiet(obj: Any) -> None:
        close = getattr(obj, "close", None)
        if close is None:
            return
        try:
            close()
        except Exception:      # noqa: BLE001 - closing is best effort
            log.debug("close failed", exc_info=True)

    def _drop_mcp(self, keep: dict[str, dict[str, Any]]) -> None:
        """Forget the MCP clients that are no longer wanted or whose spec changed; their process
        trees are killed on a daemon thread (taskkill can take a second)."""
        dead: list[Any] = []
        with self._lock:
            for name in list(self._mcp_clients):
                if name not in keep or self._mcp_snapshots.get(name) != keep[name]:
                    dead.append(self._mcp_clients.pop(name))
                    self._mcp_snapshots.pop(name, None)
                    self._mcp_listing.pop(name, None)
                    self._mcp_state.pop(name, None)
                    self._mcp_suspect.discard(name)
            for name in list(self._mcp_state):
                if name not in keep and name not in self._mcp_clients:
                    self._mcp_state.pop(name, None)
                    self._mcp_listing.pop(name, None)
                    self._mcp_suspect.discard(name)
        if dead:
            closer = threading.Thread(target=lambda: [self._close_quiet(c) for c in dead], name="mcp-close",
                                      daemon=True)
            with self._lock:
                self._closers = [t for t in self._closers if t.is_alive()] + [closer]
            closer.start()

    def _t_mcp(self, name: str, a: dict[str, Any], cancel: threading.Event | None) -> str:
        parts = name.split(":", 2)
        if len(parts) != 3 or not parts[1] or not parts[2]:
            raise _Refusal("denied: a malformed MCP tool name; it is mcp:<server>:<tool>")
        server, tool = parts[1], parts[2]
        if server not in self._mcp_wanted():
            raise _Refusal(f"denied: the MCP server {server!r} is not enabled")
        with self._lock:
            client = self._mcp_clients.get(server)
            connected = self._client_state(server) == "connected"
            listed = {str(getattr(t, "name", "")) for t in self._mcp_listing.get(server, [])}
        if client is None or not connected:
            raise _Refusal(f"denied: the MCP server {server!r} is not connected")
        if tool not in listed:
            raise _Refusal(f"denied: the MCP server {server!r} has no tool {tool!r}")
        try:
            out = client.call(tool, dict(a), cancel=cancel)
        except Exception as exc:      # noqa: BLE001 - McpError and anything a transport raises
            msg = " ".join(str(exc).split()) or type(exc).__name__
            self._mark_suspect(server, exc, cancel)
            raise _Failure(f"error: {server}:{tool}: {msg}") from None
        return "" if out is None else str(out)

    # ---- the active map
    def map_text(self) -> str:
        """The rendered text of the active project map ("" when none): the ``<map>`` block."""
        with self._lock:
            return self._map_text

    def set_map_text(self, text: str) -> None:
        """Replace the active map's rendered text (the engine calls this on index / load / delete).
        The text is sanitised on the way in (a map file can be shared or edited by hand)."""
        clean = sanitize_map_text(text or "")
        with self._lock:
            self._map_text = clean

    # ---- lifecycle
    def refresh(self) -> None:
        """The config changed: re-read it; drop the MCP clients whose spec changed or vanished
        (the engine then calls :meth:`connect_mcp` on a background thread when needed)."""
        try:
            with self._lock:
                self._warned.clear()
                if not self._web_injected:
                    self._web = None          # rebuilt on next use (an in-flight fetch keeps its own reference)
            self._drop_mcp(self._mcp_wanted())
        except Exception:      # noqa: BLE001 - refresh is called from the UI thread
            log.warning("capabilities refresh failed", exc_info=True)

    def connect_mcp(self) -> None:
        """Connect every enabled MCP server that is not connected yet -- or whose client has died
        or failed a call -- and cache their tool listings (what :meth:`specs` serves). Blocking (a
        stdio server may take seconds to start): the engine runs it on a daemon thread, never the
        assist worker. The policy is re-read before each server, so a switch turned off meanwhile
        stops the queue, and a last drop removes whatever connected after the switch went off."""
        try:
            with self._connect_lock:
                if self._closed:
                    return
                wanted = self._mcp_wanted()
                self._drop_mcp(wanted)
                for name, spec in wanted.items():
                    if self._closed:
                        break
                    if self._mcp_wanted().get(name) != spec:
                        continue                  # switched off or edited while an earlier server connected
                    with self._lock:
                        have = (name in self._mcp_clients and self._mcp_snapshots.get(name) == spec
                                and self._client_state(name) == "connected" and name not in self._mcp_suspect)
                    if not have:
                        self._connect_one(name, spec)
                if not self._closed:
                    self._drop_mcp(self._mcp_wanted())
        except Exception:      # noqa: BLE001
            log.warning("MCP connect failed", exc_info=True)

    def _connect_one(self, name: str, spec: dict[str, Any]) -> None:
        snapshot = copy.deepcopy(spec)
        with self._lock:
            self._mcp_state[name] = "connecting"
            old = self._mcp_clients.pop(name, None)
            self._mcp_listing.pop(name, None)
            self._mcp_snapshots.pop(name, None)
            self._mcp_suspect.discard(name)
        self._close_quiet(old)
        client: Any = None
        try:
            client = self._make_mcp(snapshot)
            client.connect()
            tools = list(client.tools())
        except Exception as exc:      # noqa: BLE001 - one broken server must not stop the others
            self._close_quiet(client)
            msg = _one_line(f"error: {exc}" if str(exc) else f"error: {type(exc).__name__}", 160)
            log.warning("MCP server %s did not connect: %s", name, msg)
            with self._lock:
                self._mcp_state[name] = msg
            return
        with self._lock:
            # the connect took seconds: keep the client only if the policy still wants exactly it
            keep = not self._closed and self._mcp_wanted().get(name) == snapshot
            if keep:
                self._mcp_clients[name] = client
                self._mcp_snapshots[name] = snapshot
                self._mcp_listing[name] = tools
                self._mcp_state[name] = "connected"
            elif name not in self._mcp_clients:
                self._mcp_state.pop(name, None)
        if not keep:
            log.info("MCP server %s connected but is no longer wanted; closing it", name)
            self._close_quiet(client)
            return
        log.info("MCP server %s connected: %d tool(s)", name, len(tools))

    def mcp_tools(self, name: str) -> list[Any]:
        """The tools of the configured server `name` (connecting when needed -- a dead or suspect
        connection is rebuilt; the tab's Test button, on a background thread). Raises
        :class:`exocortex.mcp.McpError`."""
        spec = self._mcp_configured().get(name)
        if spec is None:
            try:
                from exocortex.mcp import McpError
                raise McpError(f"no MCP server named {name!r} is configured")
            except ImportError:
                raise CapabilityError(f"no MCP server named {name!r} is configured") from None
        with self._lock:
            if (self._client_state(name) == "connected" and name not in self._mcp_suspect
                    and self._mcp_snapshots.get(name) == spec and name in self._mcp_listing):
                return list(self._mcp_listing[name])
        snapshot = copy.deepcopy(spec)
        client = self._make_mcp(snapshot)
        try:
            client.connect()
            tools = list(client.tools())
        except Exception:
            self._close_quiet(client)
            raise
        old: Any = None
        with self._connect_lock:
            with self._lock:
                cur = self._mcp_configured().get(name)
                keep = (not self._closed and self._on("mcp_servers") and cur is not None
                        and _spec_enabled(cur) and cur == snapshot)
                if keep:
                    old = self._mcp_clients.pop(name, None)
                    self._mcp_clients[name] = client
                    self._mcp_snapshots[name] = snapshot
                    self._mcp_listing[name] = tools
                    self._mcp_state[name] = "connected"
                    self._mcp_suspect.discard(name)
        self._close_quiet(old if keep else client)
        return tools

    def mcp_states(self) -> dict[str, str]:
        """``{server name: "idle" | "connecting" | "connected" | "error: ..."}`` -- a server whose
        client has died reads as its error, not as "connected"."""
        names = list(self._mcp_configured())
        with self._lock:
            return {n: self._client_state(n) for n in names}

    def status(self) -> dict[str, object]:
        """The body of ``EngineActions.capability_status()``: ``{"tools": [...], "read_roots":
        [...], "mcp": {...}, "active_map": ..., "notes": [...]}``; never touches the filesystem
        or the network."""
        try:
            return self._status()
        except Exception:      # noqa: BLE001
            log.warning("status failed", exc_info=True)
            return {"tools": [], "read_roots": [], "mcp": {}, "active_map": "", "notes": ["status unavailable"]}

    def _status(self) -> dict[str, object]:
        cc = self._cc
        roots = _str_list(getattr(cc, "read_roots", []))
        notes: list[str] = []
        if not roots:
            notes.append("no read folder: the model gets no tools")
        else:
            if not self._on("read_files"):
                notes.append("Read files is off: the model cannot look at your files")
            for r in roots:
                if not os.path.isabs(r):
                    notes.append(f"read folder {r!r} is not an absolute path and is ignored")
        edit_roots = _str_list(getattr(cc, "edit_roots", []))
        if self._on("edit_files") and not edit_roots:
            notes.append("Edit files is on but no edit folder is set: the model cannot edit anything")
        if self._on("execute_commands") and roots and self._on("allow_all_commands"):
            notes.append("All commands is on: the model can run arbitrary code (every other scope is lifted)")
        if self._on("fetch_web"):
            domains = _str_list(getattr(cc, "web_allowed_domains", []))
            notes.append(f"web access is limited to {len(domains)} domain(s)" if domains
                         else "web access: any public site (set allowed domains to narrow it)")
        states = self.mcp_states()
        if self._on("mcp_servers"):
            for n, s in states.items():
                if s.startswith("error"):
                    notes.append(f"MCP server {n}: {s}")
        apply_to = getattr(cc, "apply_to", "questions")
        if roots:
            notes.append({"ask": "tools serve the ask box only", "questions": "tools serve the ask box and live answers "
                          "(read-only for the other speaker and the screen)",
                          "all": "tools serve every answer kind (read-only outside the ask box)"}.get(
                              apply_to if isinstance(apply_to, str) else "", "tools serve the ask box only"))
        return {
            "tools": [s.name for s in self.specs("ask")], "read_roots": roots, "mcp": states,
            "active_map": str(getattr(cc, "active_map", "") or ""), "notes": notes,
        }

    def close(self) -> None:
        """Close every MCP client (kill their process trees) and the web client. The clients close
        in parallel threads, joined against ONE deadline (a server that ignores stdin EOF takes
        2+ s to die; one after another they would stall the quit by that much each); a client still
        closing when the deadline passes is named in the log. After this nothing connects again."""
        with self._lock:
            self._closed = True
            clients = list(self._mcp_clients.items())
            closers = [t for t in self._closers if t.is_alive()]
            web = self._web
            self._mcp_clients.clear()
            self._mcp_snapshots.clear()
            self._mcp_listing.clear()
            self._mcp_state.clear()
            self._mcp_suspect.clear()
            if not self._web_injected:
                self._web = None
        threads = [(n, threading.Thread(target=self._close_quiet, args=(c,), name="mcp-close", daemon=True))
                   for n, c in clients]
        for _n, t in threads:
            t.start()
        end = time.monotonic() + _CLOSE_JOIN_S
        for t in [t for _n, t in threads] + closers:
            t.join(max(0.0, end - time.monotonic()))
        slow = [n for n, t in threads if t.is_alive()]
        if slow:
            log.warning("MCP server(s) %s did not close within %g s; their processes may be left running",
                        ", ".join(slow), _CLOSE_JOIN_S)
        if web is not None:
            self._close_quiet(web)

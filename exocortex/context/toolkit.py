"""Auto-discovery of a Claude Code toolkit installed in the repository being worked on.

claude-toolkit (``install.sh`` / ``dropin.mjs``) places plain files into a checkout: skills under
``.claude/skills/<name>/SKILL.md``, agents under ``.claude/agents/<name>.md``, path rules under
``.claude/rules/<name>.md``, workflows under ``.claude/workflows/``, hooks under
``.claude/hooks/tk/``, and two marker blocks in ``CLAUDE.local.md`` (the house rules, whose
"## Loop" section is the working loop, and the detected repo facts -- the test / lint commands).
Hand-copied parts land in the same places. This module reads exactly those files and builds a
:class:`ToolkitCatalog` from their own frontmatter (``name``, ``description``, ``argument-hint``,
``tools``, ``model``, ``paths``...), so nothing about any particular skill is hard-coded here and
an updated toolkit describes itself. :func:`render_catalog` turns it into the compact
``<toolkit>`` text a system prompt carries; :func:`lookup` returns the full text of the parts a
piece of text mentions ("/recon", "the reviewer agent") for the on-demand ``<toolkit_detail>``
block. :class:`ToolkitWatcher` re-scans on a cadence, so a toolkit installed mid-session appears.

The toolkit's own clone (``modules/index.json``), when configured or found in the usual places,
only ENRICHES the descriptions of parts that are installed; it never lists parts that are not,
so the model is never steered toward a skill the repo does not have.
"""
from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

PART_KINDS: tuple[str, ...] = ("skill", "agent", "workflow", "rule", "hook")
HOUSE_RULES_MARKERS = ("<!-- dev-toolkit:begin -->", "<!-- dev-toolkit:end -->")
FACTS_MARKERS = ("<!-- dev-toolkit:facts:begin -->", "<!-- dev-toolkit:facts:end -->")
#: Where the toolkit clone usually lives, relative to the home directory (Quickstart / install.mjs).
CLONE_DIRS: tuple[str, ...] = ("claude-toolkit", ".claude/dev-toolkit")
#: A part's full text is capped at this many characters when handed over on mention.
MAX_BODY_CHARS = 6000
#: Files larger than this are not read (a stray binary in .claude/).
_MAX_FILE_BYTES = 512 * 1024

_FM_RE = re.compile(r"\A﻿?---[ \t]*\r?\n(.*?)\r?\n---[ \t]*(?:\r?\n|\Z)", re.DOTALL)
_KIND_LABELS = {"skill": "Skills", "agent": "Agents", "workflow": "Workflows", "rule": "Rules", "hook": "Hooks"}


@dataclass(frozen=True)
class ToolkitPart:
    """One installed part. ``command`` is how it is typed / named."""

    kind: str
    name: str
    description: str = ""
    argument_hint: str = ""
    path: str = ""
    model: str = ""
    tools: str = ""
    user_invoked_only: bool = False
    paths: tuple[str, ...] = ()        # rules: the globs they load for
    body: str = ""                     # the text after the frontmatter, capped at MAX_BODY_CHARS
    origin: str = "repo"               # repo | user

    @property
    def command(self) -> str:
        if self.kind in ("skill", "workflow"):
            return f"/{self.name}" + (f" {self.argument_hint}" if self.argument_hint else "")
        return self.name


@dataclass
class ToolkitCatalog:
    """Everything one scan found. ``signature`` identifies the files seen (path, mtime, size)
    so a watcher can tell an install / update from a no-op rescan."""

    root: str = ""
    parts: list[ToolkitPart] = field(default_factory=list)
    loop: tuple[str, ...] = ()                       # the house rules' "## Loop" bullets
    commands: tuple[tuple[str, str], ...] = ()       # (purpose, command) from the facts block
    facts: tuple[str, ...] = ()                      # the facts block's other lines (CI, services...)
    hooks_registered: tuple[str, ...] = ()           # hook events wired in .claude/settings.local.json
    manifest_parts: tuple[str, ...] = ()             # parts the toolkit's own manifest says it placed
    signature: tuple = ()

    @property
    def is_empty(self) -> bool:
        return not self.parts and not self.loop and not self.commands

    def by_kind(self, kind: str) -> list[ToolkitPart]:
        return [p for p in self.parts if p.kind == kind]

    def find(self, name: str, kind: str | None = None) -> ToolkitPart | None:
        key = name.strip().lstrip("/").casefold()
        for p in self.parts:
            if p.name.casefold() == key and (kind is None or p.kind == kind):
                return p
        return None

    def summary(self) -> str:
        """"7 skills, 3 agents, 5 rules, 1 workflow, 2 hooks" ("" when nothing was found)."""
        counts = []
        for kind in PART_KINDS:
            n = len(self.by_kind(kind))
            if n:
                counts.append(f"{n} {kind}{'s' if n != 1 else ''}")
        return ", ".join(counts)


# ---------------------------------------------------------------------- frontmatter --------
def _unquote(value: str) -> str:
    v = value.strip()
    if len(v) >= 2 and v[0] == v[-1] and v[0] in ("'", '"'):
        inner = v[1:-1]
        return inner.replace('\\"', '"').replace("\\'", "'") if v[0] == '"' else inner
    return v


def parse_frontmatter(text: str) -> tuple[dict[str, object], str]:
    """``(meta, body)`` for a Markdown file with a ``---`` YAML-ish frontmatter block. Only the
    subset the toolkit uses is understood -- ``key: value`` lines (quoted or bare), ``key: [a,
    b]`` inline lists and ``key:`` followed by ``- item`` lines -- which keeps this free of a
    YAML dependency. Keys are lower-cased. A file without frontmatter gives ``({}, text)``."""
    m = _FM_RE.match(text)
    if m is None:
        return {}, text
    meta: dict[str, object] = {}
    list_key: str | None = None
    for raw in m.group(1).splitlines():
        line = raw.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if list_key is not None and stripped.startswith("- "):
            items = meta.get(list_key)
            if isinstance(items, list):
                items.append(_unquote(stripped[2:]))
            continue
        list_key = None
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip().casefold()
        value = value.strip()
        if not value:
            meta[key] = []
            list_key = key
        elif value.startswith("[") and value.endswith("]"):
            meta[key] = [_unquote(v) for v in value[1:-1].split(",") if v.strip()]
        elif value.casefold() in ("true", "false"):
            meta[key] = value.casefold() == "true"
        else:
            meta[key] = _unquote(value)
    return meta, text[m.end():]


# ---------------------------------------------------------------------- discovery ----------
def _read(path: Path) -> str | None:
    try:
        if path.stat().st_size > _MAX_FILE_BYTES:
            return None
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def _stat_sig(path: Path, base: Path) -> tuple[str, int, int] | None:
    try:
        st = path.stat()
    except OSError:
        return None
    try:
        rel = str(path.relative_to(base))
    except ValueError:
        rel = str(path)
    return (rel.replace("\\", "/"), int(st.st_mtime_ns), int(st.st_size))


def _first_sentence(text: str, limit: int = 240) -> str:
    for raw in text.splitlines():
        line = raw.strip().lstrip("#").strip()
        if not line or line.startswith(("---", "```", "<!--", "|", ">")):
            continue
        line = re.sub(r"\s+", " ", line)
        return line[:limit]
    return ""


def _js_description(text: str) -> str:
    """A workflow / hook script's one-liner: its ``meta`` description, else the first ``//``
    comment line that is not a usage line."""
    m = re.search(r"description\s*:\s*(['\"])(.+?)\1", text)
    if m:
        return m.group(2).strip()[:240]
    for raw in text.splitlines()[:12]:
        line = raw.strip()
        if line.startswith("//"):
            body = line.lstrip("/").strip()
            if body and not body.startswith(("node ", "Usage", "usage")) and ":" in body:
                return re.sub(r"\s+", " ", body)[:240]
            if body and ":" not in body:
                return re.sub(r"\s+", " ", body)[:240]
    return ""


def _as_str(value: object) -> str:
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    return "" if value is None else str(value)


def _scan_claude_dir(base: Path, cat: ToolkitCatalog, sig: list, *, origin: str) -> None:
    """Parts under one ``.claude`` directory (the repo's, or the user layer's)."""
    known = {(p.kind, p.name.casefold()) for p in cat.parts}

    def add(part: ToolkitPart) -> None:
        if (part.kind, part.name.casefold()) in known:
            return
        known.add((part.kind, part.name.casefold()))
        cat.parts.append(part)

    skills = base / "skills"
    if skills.is_dir():
        for d in sorted(p for p in skills.iterdir() if p.is_dir()):
            f = d / "SKILL.md"
            text = _read(f) if f.is_file() else None
            if text is None:
                continue
            if (s := _stat_sig(f, base.parent)) is not None:
                sig.append(s)
            meta, body = parse_frontmatter(text)
            add(ToolkitPart(
                kind="skill", name=_as_str(meta.get("name")) or d.name,
                description=_as_str(meta.get("description")) or _first_sentence(body),
                argument_hint=_as_str(meta.get("argument-hint")), path=str(f),
                user_invoked_only=bool(meta.get("disable-model-invocation", False)),
                body=body.strip()[:MAX_BODY_CHARS], origin=origin,
            ))
    agents = base / "agents"
    if agents.is_dir():
        for f in sorted(p for p in agents.iterdir() if p.is_file() and p.suffix.casefold() == ".md"):
            text = _read(f)
            if text is None:
                continue
            if (s := _stat_sig(f, base.parent)) is not None:
                sig.append(s)
            meta, body = parse_frontmatter(text)
            add(ToolkitPart(
                kind="agent", name=_as_str(meta.get("name")) or f.stem,
                description=_as_str(meta.get("description")) or _first_sentence(body),
                path=str(f), model=_as_str(meta.get("model")), tools=_as_str(meta.get("tools")),
                body=body.strip()[:MAX_BODY_CHARS], origin=origin,
            ))
    rules = base / "rules"
    if rules.is_dir():
        for f in sorted(p for p in rules.iterdir() if p.is_file() and p.suffix.casefold() == ".md"):
            text = _read(f)
            if text is None:
                continue
            if (s := _stat_sig(f, base.parent)) is not None:
                sig.append(s)
            meta, body = parse_frontmatter(text)
            paths = meta.get("paths")
            add(ToolkitPart(
                kind="rule", name=_as_str(meta.get("name")) or f.stem,
                description=_as_str(meta.get("description")) or _first_sentence(body),
                path=str(f), paths=tuple(str(p) for p in paths) if isinstance(paths, list) else (),
                body=body.strip()[:MAX_BODY_CHARS], origin=origin,
            ))
    workflows = base / "workflows"
    if workflows.is_dir():
        for f in sorted(p for p in workflows.iterdir() if p.is_file() and p.suffix.casefold() in (".js", ".mjs", ".md")):
            text = _read(f)
            if text is None:
                continue
            if (s := _stat_sig(f, base.parent)) is not None:
                sig.append(s)
            if f.suffix.casefold() == ".md":
                meta, body = parse_frontmatter(text)
                desc = _as_str(meta.get("description")) or _first_sentence(body)
                hint = _as_str(meta.get("argument-hint"))
                name = _as_str(meta.get("name")) or f.stem
            else:
                desc, hint, name, body = _js_description(text), "", f.stem, ""
                m = re.search(r"argument-hint\s*:\s*(['\"])(.+?)\1", text) or re.search(r"\[base\]", desc)
                if m is not None and m.lastindex:
                    hint = m.group(2)
            add(ToolkitPart(kind="workflow", name=name, description=desc, argument_hint=hint, path=str(f),
                            body=body.strip()[:MAX_BODY_CHARS], origin=origin))
    for hooks in (base / "hooks" / "tk", base / "hooks"):
        if not hooks.is_dir():
            continue
        for f in sorted(p for p in hooks.iterdir() if p.is_file() and p.suffix.casefold() in (".mjs", ".js", ".py", ".sh")):
            text = _read(f)
            if text is None:
                continue
            if (s := _stat_sig(f, base.parent)) is not None:
                sig.append(s)
            add(ToolkitPart(kind="hook", name=f.stem, description=_js_description(text), path=str(f), origin=origin))
    settings = base / "settings.local.json"
    text = _read(settings) if settings.is_file() else None
    if text is not None:
        if (s := _stat_sig(settings, base.parent)) is not None:
            sig.append(s)
        try:
            data = json.loads(text.lstrip("﻿"))
        except ValueError:
            data = None
        if isinstance(data, dict) and isinstance(data.get("hooks"), dict):
            cat.hooks_registered = tuple(sorted(set(cat.hooks_registered) | set(str(k) for k in data["hooks"])))


def _block(text: str, markers: tuple[str, str]) -> str:
    start = text.find(markers[0])
    end = text.find(markers[1], start + len(markers[0])) if start >= 0 else -1
    if start < 0 or end < 0:
        return ""
    return text[start + len(markers[0]):end]


def _loop_bullets(house_rules: str) -> tuple[str, ...]:
    out: list[str] = []
    in_loop = False
    for raw in house_rules.splitlines():
        line = raw.strip()
        if line.startswith("## "):
            in_loop = line[3:].strip().casefold().startswith("loop")
            continue
        if in_loop and line.startswith("- "):
            out.append(re.sub(r"\s+", " ", line[2:].strip()))
    return tuple(out)


_FACTS_ROW_RE = re.compile(r"^\|\s*([^|]+?)\s*\|\s*`([^`]+)`\s*\|")


def _parse_facts(facts: str) -> tuple[tuple[tuple[str, str], ...], tuple[str, ...]]:
    commands: list[tuple[str, str]] = []
    other: list[str] = []
    for raw in facts.splitlines():
        line = raw.strip()
        if not line or line.startswith(("Generated by", "|---", "| Purpose")):
            continue
        m = _FACTS_ROW_RE.match(line)
        if m:
            commands.append((m.group(1).strip(), m.group(2).strip()))
        elif not line.startswith("|"):
            other.append(line)
    return tuple(commands), tuple(other)


def _scan_claude_local(path: Path, cat: ToolkitCatalog, sig: list, base: Path) -> None:
    text = _read(path) if path.is_file() else None
    if text is None:
        return
    if (s := _stat_sig(path, base)) is not None:
        sig.append(s)
    cat.loop = _loop_bullets(_block(text, HOUSE_RULES_MARKERS))
    cat.commands, cat.facts = _parse_facts(_block(text, FACTS_MARKERS))


def _scan_manifest(repo_root: Path, cat: ToolkitCatalog, sig: list) -> None:
    """The toolkit's own manifest (``.git/dev-toolkit.json``): which parts it placed."""
    for candidate in (repo_root / ".git" / "dev-toolkit.json",):
        text = _read(candidate) if candidate.is_file() else None
        if text is None:
            continue
        if (s := _stat_sig(candidate, repo_root)) is not None:
            sig.append(s)
        try:
            data = json.loads(text.lstrip("﻿"))
        except ValueError:
            return
        files = data.get("files") if isinstance(data, dict) else None
        if isinstance(files, dict):
            parts = {str(v.get("part")) for v in files.values() if isinstance(v, dict) and v.get("part")}
            cat.manifest_parts = tuple(sorted(parts))
        return


def find_toolkit_dir(explicit: str = "", home: Path | None = None) -> Path | None:
    """The toolkit clone: `explicit` when it holds ``modules/index.json``, else the first of
    ``~/claude-toolkit`` / ``~/.claude/dev-toolkit`` that does; None when none exists."""
    candidates: list[Path] = []
    if explicit.strip():
        candidates.append(Path(explicit.strip()).expanduser())
    else:
        base = home or Path.home()
        candidates.extend(base / rel for rel in CLONE_DIRS)
    for c in candidates:
        if (c / "modules" / "index.json").is_file():
            return c
    return None


def _enrich_from_clone(clone: Path, cat: ToolkitCatalog) -> None:
    """The clone's ``modules/index.json`` "what" is the catalog one-liner the toolkit itself
    lists a part under: when the clone is available, installed parts take it (minus a leading
    ``/name [args]:`` the catalog renders anyway). A part the clone does not know keeps its own
    frontmatter description; a part the clone knows but the repo lacks is never added."""
    text = _read(clone / "modules" / "index.json")
    if text is None:
        return
    try:
        index = json.loads(text.lstrip("﻿"))
    except ValueError:
        return
    if not isinstance(index, dict):
        return
    enriched: list[ToolkitPart] = []
    for p in cat.parts:
        entry = index.get(p.name)
        what = entry.get("what") if isinstance(entry, dict) else None
        if isinstance(what, str) and what.strip():
            desc = re.sub(rf"^/{re.escape(p.name)}\b[^:]*:\s*", "", what.strip())
            enriched.append(ToolkitPart(**{**p.__dict__, "description": desc or p.description}))
        else:
            enriched.append(p)
    cat.parts = enriched


def discover(
    repo_root: Path | str | None,
    *,
    user_layer: bool = False,
    toolkit_dir: str = "",
    home: Path | None = None,
) -> ToolkitCatalog:
    """Scan `repo_root` (its ``.claude`` directory, ``CLAUDE.local.md`` and the toolkit manifest)
    and, with `user_layer`, ``~/.claude`` as well; parts installed in the repo win over the
    user layer's of the same kind and name. Never raises: an unreadable file is skipped."""
    root = Path(repo_root).expanduser() if repo_root else None
    cat = ToolkitCatalog(root=str(root) if root is not None else "")
    sig: list = []
    try:
        if root is not None and root.is_dir():
            _scan_claude_dir(root / ".claude", cat, sig, origin="repo")
            _scan_claude_local(root / "CLAUDE.local.md", cat, sig, root)
            _scan_manifest(root, cat, sig)
        if user_layer:
            user_base = (home or Path.home()) / ".claude"
            if user_base.is_dir():
                _scan_claude_dir(user_base, cat, sig, origin="user")
        clone = find_toolkit_dir(toolkit_dir, home=home)
        if clone is not None and cat.parts:
            _enrich_from_clone(clone, cat)
    except Exception:  # noqa: BLE001 - discovery is best effort, never a reason to fail a prompt
        log.exception("toolkit discovery failed under %s", root)
    cat.signature = tuple(sorted(sig))
    return cat


# ---------------------------------------------------------------------- rendering ----------
def _short(text: str, n: int) -> str:
    t = re.sub(r"\s+", " ", (text or "").strip())
    if len(t) <= n:
        return t
    cut = t[: max(1, n - 1)]
    if " " in cut[n // 2:]:
        cut = cut[: cut.rfind(" ")]
    return cut.rstrip(" ,;:") + "…"


_INTRO = (
    "Installed in the user's repository (.claude/, CLAUDE.local.md). Skills are typed at the Claude "
    "Code prompt as /name [argument]; agents are used by name (\"Use the reviewer agent on the diff, "
    "one lens: bypasses\"); rules load by themselves for matching files."
)
#: Loop bullets kept, and the length each is cut to, when the catalog carries the working loop.
_LOOP_BULLETS = 2
_LOOP_CHARS = 260


def render_catalog(cat: ToolkitCatalog, max_chars: int = 3000) -> str:
    """The compact catalog for a system prompt: one line per kind (``Skills: /recon [topic]: ...;
    /scope ...``), the repo's detected commands and the working loop. Shrunk to `max_chars` by
    shortening descriptions first, then dropping rules and hooks, then the loop, then
    truncating."""
    if cat.is_empty:
        return ""

    def build(desc_len: int, with_rules: bool, with_hooks: bool, with_loop: bool) -> str:
        lines = [_INTRO]
        for kind in PART_KINDS:
            parts = cat.by_kind(kind)
            if not parts or (kind == "rule" and not with_rules) or (kind == "hook" and not with_hooks):
                continue
            entries = []
            for p in parts:
                label = p.command
                extras = []
                if kind == "agent":
                    if p.model:
                        extras.append(p.model)
                    if p.tools and not any(t in p.tools for t in ("Edit", "Write")):
                        extras.append("read-only")
                if kind == "rule" and p.paths:
                    extras.append(", ".join(p.paths[:3]))
                if kind == "skill" and p.user_invoked_only:
                    extras.append("user-invoked")
                if extras:
                    label += f" ({'; '.join(extras)})"
                desc = _short(p.description, desc_len)
                entries.append(f"{label}: {desc}" if desc else label)
            lines.append(f"{_KIND_LABELS[kind]}: " + "; ".join(entries) + ".")
        if cat.commands:
            lines.append("Repo commands (CLAUDE.local.md): " + "; ".join(f"{k}: `{c}`" for k, c in cat.commands) + ".")
        if with_loop and cat.loop:
            lines.append("Working loop (house rules): " + " ".join(_short(b, _LOOP_CHARS) for b in cat.loop[:_LOOP_BULLETS]))
        return "\n".join(lines)

    attempts = (
        (160, True, True, True), (120, True, True, True), (90, True, True, True), (70, True, True, True),
        (60, False, True, True), (60, False, False, True), (45, False, False, True), (45, False, False, False),
        (35, False, False, False),
    )
    text = ""
    for desc_len, rules, hooks, loop in attempts:
        text = build(desc_len, rules, hooks, loop)
        if len(text) <= max_chars:
            return text
    return text[: max(0, max_chars - 1)].rstrip() + "…"


# ---------------------------------------------------------------------- lookup -------------
_KIND_WORDS = r"(?:skill|agent|workflow|rule|hook|command|subagent)"


def mentioned_parts(cat: ToolkitCatalog, text: str) -> list[ToolkitPart]:
    """The parts `text` names: ``/name`` anywhere (skills and workflows), or the bare name next
    to its kind's word ("the reviewer agent", "agent reviewer", "recon skill"). In catalog
    order, each part once."""
    if cat.is_empty or not (text or "").strip():
        return []
    low = text.casefold()
    out: list[ToolkitPart] = []
    for p in cat.parts:
        name = re.escape(p.name.casefold())
        if p.kind in ("skill", "workflow") and re.search(rf"(?<![\w/])/{name}(?![\w-])", low):
            out.append(p)
            continue
        if re.search(rf"\b{name}\b\s+{_KIND_WORDS}\b", low) or re.search(rf"\b{_KIND_WORDS}\s+(?:called\s+|named\s+)?{name}\b", low):
            out.append(p)
    return out


def lookup(cat: ToolkitCatalog, text: str, max_chars: int = 3000) -> str:
    """The full text of every part `text` mentions (``### /recon [topic] (skill)`` + body),
    within `max_chars` -- "" when nothing is mentioned."""
    parts = mentioned_parts(cat, text)
    if not parts:
        return ""
    chunks: list[str] = []
    used = 0
    for p in parts:
        body = p.body.strip() or p.description
        head = f"### {p.command} ({p.kind})"
        room = max_chars - used - len(head) - 2
        if room <= 80:
            break
        if len(body) > room:
            body = body[: room - 1].rstrip() + "…"
        chunk = f"{head}\n{body}"
        chunks.append(chunk)
        used += len(chunk) + 2
    return "\n\n".join(chunks)


# ---------------------------------------------------------------------- watcher ------------
class ToolkitWatcher:
    """Re-runs :func:`discover` on a cadence and reports whether what it found changed (an
    install, an update, a removal). Every input is a callable so a config edit takes effect at
    the next refresh without a restart."""

    def __init__(
        self,
        repo_root: Callable[[], str],
        *,
        user_layer: Callable[[], bool] = lambda: False,
        toolkit_dir: Callable[[], str] = lambda: "",
        rescan_s: Callable[[], float] = lambda: 30.0,
        clock: Callable[[], float] | None = None,
        home: Path | None = None,
    ) -> None:
        import time

        self._repo_root = repo_root
        self._user_layer = user_layer
        self._toolkit_dir = toolkit_dir
        self._rescan_s = rescan_s
        self._clock = clock or time.monotonic
        self._home = home
        self._last = float("-inf")
        self._root_seen = ""
        self.catalog = ToolkitCatalog()

    def refresh(self, *, force: bool = False) -> bool:
        """Rescan when `force` or the cadence allows; True when the catalog changed."""
        now = self._clock()
        root = str(self._repo_root() or "")
        try:
            interval = float(self._rescan_s())
        except (TypeError, ValueError):
            interval = 30.0
        if not force and root == self._root_seen and now - self._last < max(1.0, interval):
            return False
        self._last = now
        self._root_seen = root
        new = discover(root or None, user_layer=bool(self._user_layer()), toolkit_dir=str(self._toolkit_dir() or ""),
                       home=self._home)
        old = self.catalog
        # A different root with nothing installed on either side is not a change worth a toast.
        changed = new.signature != old.signature or (new.root != old.root and not (new.is_empty and old.is_empty))
        self.catalog = new
        return changed

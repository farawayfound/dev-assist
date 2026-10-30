"""A lightweight, search-on-demand view of the repository being worked on.

The model never gets the repository wholesale. :func:`extract_terms` pulls the identifier-shaped
tokens out of what was said / typed / seen on screen (``snake_case``, ``CamelCase``, dotted
names, file names, paths, back-ticked words); :class:`RepoIndex` keeps a cached list of the text
files under the root and :meth:`RepoIndex.search` greps them for those terms within a time
budget; :func:`render_hits` turns the matches into the small ``<repo>`` block a prompt carries.
A lookup that matches nothing hands over nothing.
"""
from __future__ import annotations

import logging
import os
import re
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

TEXT_EXTENSIONS: frozenset[str] = frozenset({
    ".py", ".pyi", ".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx", ".json", ".md", ".txt", ".toml", ".yaml",
    ".yml", ".ini", ".cfg", ".html", ".htm", ".css", ".scss", ".sql", ".sh", ".ps1", ".bat", ".cmd", ".go",
    ".rs", ".java", ".kt", ".kts", ".cs", ".c", ".h", ".cpp", ".hpp", ".cc", ".rb", ".php", ".swift", ".m",
    ".vue", ".svelte", ".graphql", ".proto", ".xml", ".csv", ".env.example", ".dockerfile", ".tf", ".lua",
    ".r", ".jl", ".ex", ".exs", ".erl", ".scala", ".clj", ".dart",
})
TEXT_NAMES: frozenset[str] = frozenset({"Makefile", "Dockerfile", "Justfile", "CMakeLists.txt", "README", "LICENSE"})
DEFAULT_EXCLUDE: tuple[str, ...] = (
    ".git", "node_modules", ".venv", "venv", "dist", "build", "__pycache__", ".tools", "models",
    ".pytest_cache", ".mypy_cache", ".ruff_cache", ".idea", ".vscode", ".claude",
)

_STOP_WORDS: frozenset[str] = frozenset("""
the this that with from into onto over under about after before while where which what when why how
and but for nor yet are was were been being have has had does did doing will would could should might
must can shall may just also only very really there their they them then than these those your you
our ours mine his her its it's i'm we're they're let's don't doesn't didn't isn't aren't wasn't
function method class module import return value values object objects string number list array
file files line lines code test tests testing error errors issue problem thing things something
maybe okay yeah right like some more most less least first last next new old same other another
here now today later again always never every each both either neither each much many few
""".split())

_BACKTICK_RE = re.compile(r"`([^`\n]{2,80})`")
_PATH_RE = re.compile(r"(?<![\w./-])((?:[\w.-]+/)+[\w.-]+)(?![\w/])")
_FILE_RE = re.compile(r"(?<![\w./-])([\w-]+\.(?:py|pyi|js|mjs|ts|tsx|jsx|json|md|toml|yaml|yml|go|rs|java|kt|cs|c|h|cpp|hpp|rb|php|sql|sh|ps1|html|css|vue|proto))(?![\w.])", re.I)
_SNAKE_RE = re.compile(r"\b([A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+)\b")
_CAMEL_RE = re.compile(r"\b([A-Za-z][a-z0-9]+(?:[A-Z][a-z0-9]+)+[A-Za-z0-9]*)\b")
_DOTTED_RE = re.compile(r"\b([A-Za-z_][\w]*(?:\.[A-Za-z_][\w]*)+)\b")
_CONST_RE = re.compile(r"\b([A-Z][A-Z0-9]{3,}(?:_[A-Z0-9]+)*)\b")
_CALL_RE = re.compile(r"\b([A-Za-z_][\w]{2,})\(")


def extract_terms(*texts: str, limit: int = 8) -> list[str]:
    """Identifier-shaped tokens worth searching the repo for, most frequent first, at most
    `limit`. Plain English words never qualify, so a conversation with no code names in it
    yields nothing and no search runs."""
    counts: dict[str, int] = {}
    order: dict[str, int] = {}

    def note(term: str, weight: int = 1) -> None:
        t = term.strip().strip(".,;:()[]{}'\"")
        if len(t) < 3 or len(t) > 80 or t.casefold() in _STOP_WORDS:
            return
        if t.replace(".", "").replace("_", "").isdigit():
            return
        counts[t] = counts.get(t, 0) + weight
        order.setdefault(t, len(order))

    for text in texts:
        if not text:
            continue
        for m in _BACKTICK_RE.finditer(text):
            note(m.group(1), 3)
        for m in _PATH_RE.finditer(text):
            if "/" in m.group(1) and not m.group(1).startswith(("http", "www.")):
                note(m.group(1), 2)
                note(m.group(1).rsplit("/", 1)[-1], 1)
        for m in _FILE_RE.finditer(text):
            note(m.group(1), 2)
        for m in _SNAKE_RE.finditer(text):
            note(m.group(1), 2)
        for m in _CAMEL_RE.finditer(text):
            note(m.group(1), 2)
        for m in _DOTTED_RE.finditer(text):
            tok = m.group(1)
            if not re.fullmatch(r"[\d.]+", tok) and not tok.casefold().endswith((".com", ".org", ".net", ".io")):
                note(tok, 1)
        for m in _CONST_RE.finditer(text):
            note(m.group(1), 1)
        for m in _CALL_RE.finditer(text):
            if m.group(1).casefold() not in _STOP_WORDS:
                note(m.group(1), 1)
    ranked = sorted(counts, key=lambda t: (-counts[t], order[t]))
    out: list[str] = []
    seen: set[str] = set()
    for t in ranked:
        key = t.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
        if len(out) >= max(0, limit):
            break
    return out


@dataclass(frozen=True)
class RepoHit:
    path: str          # relative to the root, forward slashes
    line: int          # 1-based; 0 = the file name itself matched
    text: str
    term: str


def _is_text_file(path: Path) -> bool:
    name = path.name
    if name in TEXT_NAMES:
        return True
    suffix = path.suffix.casefold()
    if suffix in TEXT_EXTENSIONS:
        return True
    lowered = name.casefold()
    return any(lowered.endswith(ext) for ext in (".env.example", ".dockerfile"))


class RepoIndex:
    """A cached list of the text files under `root` and a budgeted search over them.

    The file list is rebuilt at most every `refresh_s` seconds (and on demand with
    ``files(force=True)``); directories named in `exclude` are pruned, files over
    `max_file_bytes` are skipped, and the walk stops at `max_files`. ``search`` scans file
    names first, then file contents, and stops when `time_budget_s` is spent -- so a huge
    repository costs at most that much per lookup.
    """

    def __init__(
        self,
        root: Path | str,
        *,
        exclude: Iterable[str] = DEFAULT_EXCLUDE,
        max_files: int = 8000,
        max_file_bytes: int = 256 * 1024,
        refresh_s: float = 60.0,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._root = Path(root).expanduser()
        self._exclude = {e.strip().casefold() for e in exclude if e and e.strip()}
        self._max_files = max(1, int(max_files))
        self._max_file_bytes = max(1024, int(max_file_bytes))
        self._refresh_s = max(0.0, float(refresh_s))
        self._clock = clock or time.monotonic
        self._files: list[Path] = []
        self._listed_at = float("-inf")
        self.truncated = False

    @property
    def root(self) -> Path:
        return self._root

    @property
    def available(self) -> bool:
        return self._root.is_dir()

    def files(self, *, force: bool = False) -> list[Path]:
        """Relative paths of the text files under the root (cached)."""
        now = self._clock()
        if not force and now - self._listed_at < self._refresh_s and self._listed_at != float("-inf"):
            return list(self._files)
        out: list[Path] = []
        self.truncated = False
        if self._root.is_dir():
            for dirpath, dirnames, filenames in os.walk(self._root):
                dirnames[:] = sorted(d for d in dirnames if d.casefold() not in self._exclude and not d.startswith(".git"))
                base = Path(dirpath)
                for name in sorted(filenames):
                    p = base / name
                    if not _is_text_file(p):
                        continue
                    try:
                        if p.stat().st_size > self._max_file_bytes:
                            continue
                    except OSError:
                        continue
                    try:
                        out.append(p.relative_to(self._root))
                    except ValueError:
                        continue
                    if len(out) >= self._max_files:
                        self.truncated = True
                        break
                if self.truncated:
                    break
        self._files = out
        self._listed_at = now
        return list(out)

    def search(
        self,
        terms: Sequence[str],
        *,
        max_files: int = 6,
        max_hits_per_file: int = 3,
        time_budget_s: float = 1.5,
        max_scan_files: int = 1500,
    ) -> list[RepoHit]:
        """Files whose NAME contains a term, then lines that contain a term, best files first
        (most distinct terms matched); at most `max_files` files. A term with an upper-case
        letter or an underscore is matched case-sensitively (an identifier), the rest
        case-insensitively."""
        terms = [t for t in dict.fromkeys(t.strip() for t in terms) if len(t) >= 3]
        if not terms or not self.available:
            return []
        deadline = time.monotonic() + max(0.05, float(time_budget_s))
        files = self.files()
        folded = [(t, t.casefold(), any(c.isupper() for c in t) or "_" in t) for t in terms]

        name_hits: dict[Path, set[str]] = {}
        for p in files:
            low = p.name.casefold()
            for t, tl, _cs in folded:
                if tl in low or (("/" in tl) and tl in str(p).replace("\\", "/").casefold()):
                    name_hits.setdefault(p, set()).add(t)
        # Scan the name matches first, then the rest newest-modified first.
        ordered = list(name_hits)
        rest = [p for p in files if p not in name_hits]
        try:
            rest.sort(key=lambda p: (self._root / p).stat().st_mtime, reverse=True)
        except OSError:
            pass
        ordered.extend(rest[: max(0, max_scan_files - len(ordered))])

        hits_by_file: dict[Path, list[RepoHit]] = {}
        terms_by_file: dict[Path, set[str]] = {p: set(ts) for p, ts in name_hits.items()}
        for p in ordered:
            if time.monotonic() > deadline:
                break
            try:
                data = (self._root / p).read_bytes()
            except OSError:
                continue
            if b"\x00" in data[:4096]:
                continue
            text = data.decode("utf-8", errors="ignore")
            rel = str(p).replace("\\", "/")
            found: list[RepoHit] = []
            for n, line in enumerate(text.splitlines(), 1):
                if len(found) >= max_hits_per_file:
                    break
                low = line.casefold()
                for t, tl, cs in folded:
                    if (t in line) if cs else (tl in low):
                        found.append(RepoHit(path=rel, line=n, text=line.strip()[:160], term=t))
                        terms_by_file.setdefault(p, set()).add(t)
                        break
            if found:
                hits_by_file[p] = found
            elif p in name_hits:
                hits_by_file[p] = [RepoHit(path=rel, line=0, text="", term=next(iter(name_hits[p])))]
            if len([f for f in hits_by_file if f in name_hits or hits_by_file[f]]) >= max_files * 3:
                break
        ranked = sorted(hits_by_file, key=lambda p: (-len(terms_by_file.get(p, ())), -(p in name_hits), str(p)))
        out: list[RepoHit] = []
        for p in ranked[: max(0, max_files)]:
            out.extend(hits_by_file[p])
        return out


def render_hits(hits: Sequence[RepoHit], max_chars: int = 1800) -> str:
    """Grouped by file: ``path`` then ``  L12: text`` lines; "" without hits. Cut at
    `max_chars` on a whole line."""
    if not hits:
        return ""
    chunks: list[list[str]] = []
    current = ""
    for h in hits:
        if h.path != current:
            current = h.path
            chunks.append([h.path])
        if h.line > 0:
            chunks[-1].append(f"  L{h.line}: {h.text}")
    out: list[str] = []
    used = 0
    for chunk in chunks:
        # A file goes in whole (its path plus as many of its lines as fit, at least one), or not
        # at all: a bare path with its lines cut off says nothing.
        head, body = chunk[0], chunk[1:]
        needed = len(head) + 1 + (len(body[0]) + 1 if body else 0)
        if used + needed > max_chars:
            break
        out.append(head)
        used += len(head) + 1
        for line in body:
            if used + len(line) + 1 > max_chars:
                break
            out.append(line)
            used += len(line) + 1
    return "\n".join(out)

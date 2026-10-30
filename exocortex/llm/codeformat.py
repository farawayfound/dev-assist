"""Code-only answers (added 2026-10-29): a Code answer is one fenced code block and nothing else.

The model is told so (``prompts.system_code(code_only=True)``) and, because a small model slips
now and then -- a ``**Approach:**`` line, a ``### Step 1`` heading or a bullet list where a comment
belongs, prose or the complexity line outside the fence -- the finished answer is normalised here as
well: text outside the block becomes comment lines inside it, markdown that landed inside the block
becomes comments in the language's own syntax, a fence-less answer that is clearly code is fenced.
With explanations on the walkthrough and the complexity line stay where the plan parser expects them
(:func:`exocortex.plan.split_plan`); only the inside of the block is cleaned. Pure string functions.
"""
from __future__ import annotations

import re

#: A fence line (the same shape `exocortex.plan` parses): the marker and an optional language tag.
_FENCE_RE = re.compile(r"^[ \t]*(```|~~~)[ \t]*([\w+#.-]*)[^\n]*$")
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+(\S.*)$")
_BOLD_LINE_RE = re.compile(r"^\s*(?:\*\*|__)([^*_].*?)(?:\*\*|__)\s*:?\s*$")
_BULLET_RE = re.compile(r"^\s*(?:[-*•]|\d{1,3}[.)])\s+(\S.*)$")
_QUOTE_RE = re.compile(r"^\s*>\s?")
_INLINE_MARK_RE = re.compile(r"\*\*(.+?)\*\*|__(.+?)__|`([^`]+)`")
_RULE_RE = re.compile(r"^\s*(?:-{3,}|\*{3,}|_{3,})\s*$")
#: What makes a line code rather than prose: brackets, an assignment or comparison, a statement
#: terminator, or a leading keyword (a superset of the hands-free heuristic, comments included).
_CODE_RE = re.compile(
    r"[{}();=<>\[\]]|^\s*(?:def|class|return|import|from|for|while|if|elif|else|try|except|finally|"
    r"const|let|var|function|public|private|protected|static|void|int|float|double|bool|string|char|"
    r"#include|#define|using|namespace|package|fn|pub|struct|enum|impl|select|insert|update|delete|with|"
    r"print|raise|yield|async|await|match|case|switch|break|continue|pass|end|do|then|begin|local|"
    r"export|type|interface|lambda|new|throw|throws|catch|override|abstract|final|module|require)\b",
    re.IGNORECASE,
)
_COMMENT_RE = re.compile(r"^\s*(?:#(?!include|define|pragma|if|ifdef|ifndef|endif|else|undef)|//|--\s|/\*|\*\s|\*/|<!--|;|%|')")

#: Fence tags whose line comment is ``#`` / ``--`` / an HTML comment; everything else gets ``//``.
_HASH_TAGS = frozenset(
    "python py python3 ruby rb sh bash zsh shell fish powershell ps1 pwsh yaml yml toml r perl pl makefile "
    "make dockerfile elixir ex exs nim julia jl cmake conf ini cfg tcl awk coffee coffeescript gnuplot".split()
)
_DASH_TAGS = frozenset("sql mysql postgresql postgres pgsql sqlite plsql tsql lua haskell hs ada vhdl elm".split())
_HTML_TAGS = frozenset("html htm xml svg vue xhtml xaml".split())
#: `[assist].primary_language` names -> fence tags.
_LANGUAGE_TAGS = {
    "python": "python", "python3": "python", "py": "python", "javascript": "javascript", "js": "javascript",
    "typescript": "typescript", "ts": "typescript", "java": "java", "c#": "csharp", "csharp": "csharp",
    "c++": "cpp", "cpp": "cpp", "c": "c", "go": "go", "golang": "go", "rust": "rust", "ruby": "ruby",
    "php": "php", "swift": "swift", "kotlin": "kotlin", "scala": "scala", "sql": "sql", "bash": "bash",
    "shell": "bash", "powershell": "powershell", "r": "r", "lua": "lua", "haskell": "haskell",
    "dart": "dart", "html": "html", "perl": "perl", "elixir": "elixir", "julia": "julia", "matlab": "matlab",
}


def language_tag(language: str) -> str:
    """The fence tag for a `[assist].primary_language` value ("" when it is not a single word)."""
    name = (language or "").strip().lower()
    if name in _LANGUAGE_TAGS:
        return _LANGUAGE_TAGS[name]
    return name if re.fullmatch(r"[a-z][a-z0-9+#.-]*", name) else ""


def comment_syntax(tag: str, body: list[str] | tuple[str, ...] = ()) -> tuple[str, str]:
    """``(prefix, suffix)`` of a line comment for the fence ``tag`` -- inferred from the code itself
    when the tag is unknown or missing (``#`` lines vs ``//`` lines vs ``-- `` lines; ``//`` wins a tie)."""
    name = (tag or "").strip().lower()
    if name in _HASH_TAGS:
        return "# ", ""
    if name in _DASH_TAGS:
        return "-- ", ""
    if name in _HTML_TAGS:
        return "<!-- ", " -->"
    if name:
        return "// ", ""
    hashes = slashes = dashes = 0
    pythonish = 0
    for line in body:
        s = line.strip()
        if s.startswith("#") and not re.match(r"#\s*(?:include|define|pragma|if|ifdef|ifndef|endif|else|undef)\b", s):
            hashes += 1
        elif s.startswith("//"):
            slashes += 1
        elif s.startswith("-- "):
            dashes += 1
        if re.match(r"(?:def|class|elif|import|from)\s", s) or s.endswith(":"):
            pythonish += 1
    if hashes > slashes and hashes >= dashes:
        return "# ", ""
    if dashes > slashes and dashes > hashes:
        return "-- ", ""
    if slashes == 0 and pythonish >= 2:
        return "# ", ""
    return "// ", ""


def _fence_spans(lines: list[str]) -> list[tuple[int, int | None, str]]:
    spans: list[tuple[int, int | None, str]] = []
    open_at: int | None = None
    tag = ""
    for i, line in enumerate(lines):
        m = _FENCE_RE.match(line)
        if m is None:
            continue
        if open_at is None:
            open_at, tag = i, m.group(2).strip().lower()
        else:
            spans.append((open_at, i, tag))
            open_at = None
    if open_at is not None:
        spans.append((open_at, None, tag))
    return spans


def _main_span(spans: list[tuple[int, int | None, str]]) -> tuple[int, int | None, str]:
    """The block that is the code: the one still open (streaming cut it), else the largest."""
    if spans[-1][1] is None:
        return spans[-1]
    return max(spans, key=lambda s: ((s[1] or 0) - s[0], -s[0]))


def is_code_line(line: str) -> bool:
    """Whether ``line`` reads as code (or a code comment) rather than prose."""
    s = line.strip()
    if not s:
        return False
    if _COMMENT_RE.match(s):
        return True
    return bool(_CODE_RE.search(s))


def unmark(line: str) -> str:
    """A markdown line as plain text: heading hashes, bullet / quote markers, bold / code marks gone."""
    s = line.strip()
    if not s or _RULE_RE.match(s) or _FENCE_RE.match(s):
        return ""
    m = _HEADING_RE.match(s)
    if m:
        s = m.group(1)
    s = _QUOTE_RE.sub("", s)
    m = _BULLET_RE.match(s)
    if m:
        s = m.group(1)
    m = _BOLD_LINE_RE.match(s)
    if m:
        s = m.group(1)
    s = _INLINE_MARK_RE.sub(lambda mm: mm.group(1) or mm.group(2) or mm.group(3) or "", s)
    return " ".join(s.split())


def _comment(text: str, prefix: str, suffix: str) -> str:
    return f"{prefix}{text}{suffix}"


def clean_code_line(line: str, prefix: str, suffix: str) -> str:
    """One line INSIDE the block: markdown that is not code in this language becomes a comment --
    a heading (unless ``#`` is the comment marker, where it already is one), a bold-only line, a
    bullet whose text is prose. Code and real comments pass unchanged."""
    s = line.rstrip()
    if not s.strip():
        return s
    if _HEADING_RE.match(s):
        if prefix.startswith("#"):
            return s
        return _comment(unmark(s), prefix, suffix)
    if _BOLD_LINE_RE.match(s):
        return _comment(unmark(s), prefix, suffix)
    m = _BULLET_RE.match(s)
    if m and not is_code_line(m.group(1)):
        return _comment(unmark(s), prefix, suffix)
    if not is_code_line(s) and _INLINE_MARK_RE.search(s) and s.count("**") >= 2:
        return _comment(unmark(s), prefix, suffix)
    return s


def _mostly_code(lines: list[str]) -> bool:
    text = [line for line in lines if line.strip()]
    if len(text) < 2:
        return False
    return sum(1 for line in text if is_code_line(line)) / len(text) >= 0.5


def _trim_blank(lines: list[str]) -> list[str]:
    start, end = 0, len(lines)
    while start < end and not lines[start].strip():
        start += 1
    while end > start and not lines[end - 1].strip():
        end -= 1
    return lines[start:end]


def code_only(text: str, *, explain: bool = False, language: str = "") -> str:
    """Normalise a finished Code answer to code only (module docstring).

    Without ``explain``: one fenced block (the largest, or the one still open) whose inside is cleaned
    with :func:`clean_code_line`; every non-blank line outside it -- prose, the complexity line, a
    secondary snippet's lines -- becomes a comment line appended at the end of the block, after one
    blank line. A fence-less answer that is mostly code is fenced (tagged with ``language``'s tag);
    fence-less prose (a one-line answer, a refusal) is returned unchanged. With ``explain`` only the
    inside of the main block is cleaned and everything else stays. Blank / whitespace text is
    returned as is."""
    if not text or not text.strip():
        return text
    src = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = src.split("\n")
    spans = _fence_spans(lines)
    if not spans:
        if explain or not _mostly_code(lines):
            return text
        tag = language_tag(language)
        body = _trim_blank(lines)
        prefix, suffix = comment_syntax(tag, body)
        return "\n".join([f"```{tag}", *(clean_code_line(l, prefix, suffix) for l in body), "```"])
    open_at, close_at, tag = _main_span(spans)
    end = close_at if close_at is not None else len(lines)
    body = lines[open_at + 1:end]
    prefix, suffix = comment_syntax(tag or language_tag(language), body)
    body = [clean_code_line(l, prefix, suffix) for l in body]
    if explain:
        tail = lines[end:] if close_at is not None else ["```"]
        return "\n".join([*lines[:open_at + 1], *body, *tail])
    extra: list[str] = []
    for i, line in enumerate(lines):
        if open_at <= i <= end:
            continue
        if _FENCE_RE.match(line):
            continue
        plain = unmark(line)
        if plain:
            extra.append(_comment(plain, prefix, suffix))
    body = _trim_blank(body)
    if extra:
        body = [*body, "", *extra]
    return "\n".join([f"```{tag}", *body, "```"])

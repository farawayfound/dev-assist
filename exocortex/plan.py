"""Explanations: the solution *plan* shared by the Assist tab and completion mode.

With ``[explain].enabled`` on, an autonomous "solve" answer is structured as a **walkthrough** (the
thought process, streamed *before* the solution), then the solution -- code in a fenced block with
minimal but important comments -- then the complexity line. This module is the pure logic around
that structure:

* :func:`split_plan` parses the (possibly still streaming) answer into walkthrough / solution.
* :func:`consume_suggestion` is the whitespace-tolerant "what remains of a suggestion after the
  user typed this" matcher used by completion mode; with ``skip_comments`` a comment line the user
  types past is simply dropped from the suggestion (comments are optional to type).
* :func:`align_plan` / :func:`plan_ghost` locate the typed text inside the plan's solution and
  produce the next lines as ghost text, so completion mode can follow the plan without a model
  call per line, and notice when the last line of code has been typed out.
* :func:`typed_fraction` measures how much of the solution is visible on screen (OCR lines), the
  cue -- in the full overlay, where nothing tracks the caret -- for the explanation that follows
  once the code has been written out.

No threads, no Qt, no model calls: everything here is deterministic and unit-tested.
"""
from __future__ import annotations

import difflib
import re
from collections.abc import Iterable
from dataclasses import dataclass

#: A line-starting comment marker. ``#`` counts unless it opens a C preprocessor directive; ``--``
#: only when followed by a space (SQL/Lua/Haskell), so ``x--`` never does.
_COMMENT_START_RE = re.compile(
    r"(?:#(?!\s*(?:include|define|pragma|if|ifdef|ifndef|elif|else|endif|undef|error|import|line)\b)"
    r"|//|/\*|<!--|--(?=\s|$))"
)
_COMMENT_LINE_RE = re.compile(r"^\s*(?:" + _COMMENT_START_RE.pattern + r"|\*\s|\*/)")
_FENCE_LINE_RE = re.compile(r"^[ \t]*(```|~~~)[ \t]*([\w+#.-]*)[^\n]*$")
_WALKTHROUGH_LABEL_RE = re.compile(
    r"^\s*(?:\*\*|__)?\s*(?:walk-?through|thought process|thoughts|approach)\s*(?:\*\*|__)?\s*:?\s*(?:\*\*|__)?[ \t]*",
    re.IGNORECASE,
)

#: A solution line must be this similar (difflib ratio) to a screen/typed line to count as the same
#: line when OCR noise makes an exact comparison fail.
FUZZY_RATIO = 0.85
#: How many of the last typed lines anchor an alignment against the plan.
_ANCHOR_LINES = 3


def squash(text: str) -> str:
    """``text`` with every whitespace character removed (indentation- and spacing-insensitive key)."""
    return "".join(text.split())


def is_comment_line(line: str) -> bool:
    """True when ``line`` is a comment line (``#``, ``//``, ``/* … */``, ``* …``, ``-- …``, ``<!--``)."""
    return bool(_COMMENT_LINE_RE.match(line))


def _comment_end(suggestion: str, i: int) -> int | None:
    """If a comment starts at ``i`` (after optional blanks), the index of that line's newline (or the
    end of the text); ``None`` when no comment starts there."""
    k = i
    n = len(suggestion)
    while k < n and suggestion[k] in " \t":
        k += 1
    if not _COMMENT_START_RE.match(suggestion, k):
        return None
    end = suggestion.find("\n", k)
    return n if end < 0 else end


def consume_suggestion(suggestion: str, typed: str, skip_comments: bool = False) -> str | None:
    """What remains of ``suggestion`` after the user typed ``typed`` over it.

    Whitespace-tolerant: runs of spaces/tabs/newlines on either side match loosely (an editor's
    auto-indent, a skipped space, a doubled space are not deviations). Returns ``""`` when the
    suggestion was typed out completely (extra trailing whitespace allowed), ``None`` on the
    first real mismatch. Comparison is otherwise exact (case-sensitive).

    With ``skip_comments``, a mismatch that lands on a comment in the suggestion (``# …``, ``// …``,
    a trailing ``# note``) skips that comment to the end of its line and carries on, so a user who
    types past a comment straight into the code that follows keeps the rest of the suggestion --
    the comment is dropped, nothing is regenerated.
    """
    i, j = 0, 0
    n, m = len(suggestion), len(typed)
    while j < m:
        if i >= n:
            return "" if typed[j:].isspace() or typed[j:] == "" else None
        s, t = suggestion[i], typed[j]
        if s == t:
            i += 1
            j += 1
            continue
        s_ws, t_ws = s.isspace(), t.isspace()
        if s_ws and t_ws:
            while i < n and suggestion[i].isspace():
                i += 1
            while j < m and typed[j].isspace():
                j += 1
            continue
        if t_ws:
            j += 1        # user typed extra whitespace
            continue
        if s_ws:
            i += 1        # user skipped whitespace the suggestion had
            continue
        if skip_comments:
            end = _comment_end(suggestion, i)
            if end is not None and end > i:
                i = end   # skip the comment; the newline that follows is handled as whitespace
                continue
        return None
    return suggestion[i:]


# --------------------------------------------------------------------------- the plan text ----
@dataclass(frozen=True)
class PlanParts:
    """A solve answer split into its explanation-mode parts (all may be partial while streaming)."""

    walkthrough: str      # the thought process written before the solution ("Walkthrough:" label stripped)
    solution: str         # the fenced code (without fences), or the answer text after the walkthrough
    is_code: bool         # the solution came from a fenced code block
    language: str         # the fence's language tag ("" when none)
    after: str            # text after the closing fence (the complexity line, notes)
    code_open: bool       # a fence was opened but not closed yet (still streaming)


def split_plan(text: str) -> PlanParts:
    """Parse a (possibly streaming) explanation-mode solve answer.

    The walkthrough is everything before the first code fence. Inside an unclosed fence the last,
    possibly half-generated line is left out, so what is returned as ``solution`` only ever holds
    complete lines. Without any fence the answer is prose: the walkthrough is its first block (up
    to the first blank line) and ``solution`` is the rest (the direct answer / text to enter).
    """
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    # Every fenced block: (open line, close line or None while still open, language). The solution
    # is the block being written while streaming, else the largest one -- a short illustrative
    # snippet inside the walkthrough must not be mistaken for it.
    blocks: list[tuple[int, int | None, str]] = []
    cur_open: int | None = None
    cur_lang = ""
    for i, line in enumerate(lines):
        m = _FENCE_LINE_RE.match(line)
        if m is None:
            continue
        if cur_open is None:
            cur_open, cur_lang = i, m.group(2).strip().lower()
        else:
            blocks.append((cur_open, i, cur_lang))
            cur_open = None
    if cur_open is not None:
        blocks.append((cur_open, None, cur_lang))
    open_idx: int | None = None
    close_idx: int | None = None
    language = ""
    if blocks:
        if blocks[-1][1] is None:
            open_idx, close_idx, language = blocks[-1]
        else:
            open_idx, close_idx, language = max(blocks, key=lambda b: (b[1] - b[0], -b[0]))  # type: ignore[operator]
    if open_idx is None:
        walk_lines: list[str] = []
        rest: list[str] = []
        seen_text = False
        for k, line in enumerate(lines):
            if seen_text and not line.strip():
                rest = lines[k + 1:]
                break
            if line.strip():
                seen_text = True
            walk_lines.append(line)
        solution = "\n".join(rest).strip()
        return PlanParts(_strip_label("\n".join(walk_lines)), solution, False, "", "", False)
    walkthrough = _strip_label("\n".join(lines[:open_idx]))
    if close_idx is None:
        body = lines[open_idx + 1:]
        if body and not text.endswith("\n"):
            body = body[:-1]                      # a half-generated last line is not usable yet
        return PlanParts(walkthrough, "\n".join(body), True, language, "", True)
    body = lines[open_idx + 1:close_idx]
    after = "\n".join(lines[close_idx + 1:]).strip()
    return PlanParts(walkthrough, "\n".join(body), True, language, after, False)


def _strip_label(text: str) -> str:
    return _WALKTHROUGH_LABEL_RE.sub("", text, count=1).strip()


# --------------------------------------------------------------------------- alignment --------
@dataclass(frozen=True)
class PlanView:
    """What completion mode needs to know about the current plan (a snapshot, safe to share)."""

    id: int
    solution: str
    is_code: bool
    done: bool


@dataclass(frozen=True)
class PlanCursor:
    """Where the typed text sits inside a plan's solution."""

    line: int         # index of the solution line the caret is on
    rest: str         # what remains of that line after the typed part ("" when fully typed)
    at_end: bool      # the last code line is typed out; only comments/blank lines follow
    score: int        # anchor strength: 2 per exactly matching preceding typed line, 1 per fuzzy one


#: Two statements sharing this many leading characters (whitespace removed) and a call/block
#: head are the same line for anchoring purposes: ``def twoSum(self, nums, target):`` in the plan
#: vs the editor's ``def twoSum(self, nums: List[int], target: int) -> List[int]:``.
_HEAD_CHARS = 12


def _similar(a: str, b: str) -> bool:
    """Same line, allowing OCR-grade noise or a differently annotated statement head."""
    if a == b:
        return True
    if len(a) >= _HEAD_CHARS and len(b) >= _HEAD_CHARS and a[:_HEAD_CHARS] == b[:_HEAD_CHARS] and ("(" in a or ":" in a):
        return True
    if len(a) < 6 or len(b) < 6 or abs(len(a) - len(b)) > max(3, len(a) // 4):
        return False
    return difflib.SequenceMatcher(None, a, b).ratio() >= FUZZY_RATIO


def _match_back(squashed: list[str], comments: list[bool], i: int, prev: list[str]) -> int:
    """How many of ``prev`` (the last typed lines, squashed, oldest first) match the solution lines
    before ``i``, walking backwards and skipping the solution's comment/blank lines. A typed line
    that is itself a comment (the user did type one of the plan's comments) is ignored."""
    j = i - 1
    score = 0
    for typed in reversed(prev):
        nt = squash(typed)
        while j >= 0 and (not squashed[j] or comments[j]):
            j -= 1
        if j < 0:
            break
        if squashed[j] == nt:
            score += 2            # an exact anchor outranks a fuzzy one (near-duplicate lines)
            j -= 1
        elif _similar(squashed[j], nt):
            score += 1
            j -= 1
        elif is_comment_line(typed):
            continue
        else:
            break
    return score


def _rest_after(line: str, count: int) -> str:
    """``line`` minus its first ``count`` non-whitespace characters (and the blanks between them)."""
    seen = 0
    for k, ch in enumerate(line):
        if seen >= count:
            return line[k:]
        if not ch.isspace():
            seen += 1
    return ""


def _last_code_index(lines: list[str]) -> int:
    for k in range(len(lines) - 1, -1, -1):
        if lines[k].strip() and not is_comment_line(lines[k]):
            return k
    return -1


def align_plan(solution: str, prefix: str, hint: int = 0) -> PlanCursor | None:
    """Locate the end of ``prefix`` (the text before the caret) inside ``solution``.

    The caret's own line must be a (whitespace-insensitive) prefix of a solution line; the last
    few typed lines before it must match the solution lines before that one, comments in the
    solution being optional (skipping one is not a deviation). OCR-grade noise is tolerated in the
    anchoring lines (``FUZZY_RATIO``) and, once anchored, in the caret line. Nothing typed at all
    puts the cursor at the solution's first line. ``hint`` (a previous cursor line) breaks ties in
    favour of forward progress. ``None`` when the typed text does not follow the plan.
    """
    lines = solution.split("\n")
    if not solution.strip():
        return None
    last_code = _last_code_index(lines)
    if not prefix.strip():
        for i, line in enumerate(lines):
            if line.strip():
                return PlanCursor(i, line.lstrip(), False, 0)
        return None
    plines = prefix.split("\n")
    cur = plines[-1]
    prev = [ln for ln in plines[:-1] if ln.strip()][-_ANCHOR_LINES:]
    ncur = squash(cur)
    if ncur == "" and not prev:
        return None
    squashed = [squash(ln) for ln in lines]
    comments = [is_comment_line(ln) for ln in lines]
    best: tuple[tuple[int, int, int], PlanCursor] | None = None
    for i, line in enumerate(lines):
        nl = squashed[i]
        if not nl:
            continue
        if ncur:
            exact = nl.startswith(ncur)
            if comments[i] and not exact:
                continue      # a comment is only "the caret's line" while the user is typing it
            fuzzy = not exact and len(ncur) >= 8 and _similar(nl[: len(ncur)], ncur)
            if not exact and not fuzzy:
                continue      # cheap rejection before the anchor walk
        score = _match_back(squashed, comments, i, prev)
        if prev and score == 0:
            continue
        if ncur:
            if not exact and score < 1:
                continue      # a noisy caret line needs an anchor above it
            rest = _rest_after(line, len(ncur))
            if cur.endswith((" ", "\t")):
                rest = rest.lstrip(" \t")
            at_end = not rest.strip() and i >= last_code
        else:
            rest = line.lstrip()
            at_end = False
        rank = (score, 1 if i >= hint else 0, -i)   # strongest anchor, then forward progress, then earliest
        if best is None or rank > best[0]:
            best = (rank, PlanCursor(i, rest, at_end, score))
    return best[1] if best is not None else None


def plan_ghost(solution: str, cursor: PlanCursor, max_lines: int) -> str:
    """The ghost text for ``cursor``: the rest of its line, then the following solution lines (with
    their own indentation) up to ``max_lines`` lines in total; trailing blank lines dropped."""
    lines = solution.split("\n")
    out = [cursor.rest]
    k = cursor.line + 1
    while (max_lines <= 0 or len(out) < max_lines) and k < len(lines):
        out.append(lines[k])
        k += 1
    while out and not out[-1].strip():
        out.pop()
    text = "\n".join(out)
    return text if text.strip() else ""


# --------------------------------------------------------------------------- on-screen check -
def typed_fraction(solution: str, screen_lines: Iterable[str]) -> float:
    """Share of the solution's code lines (non-blank, non-comment, at least 4 characters) that
    appear among ``screen_lines`` -- exactly, as the tail of a gutter-merged OCR line, or fuzzily
    (``FUZZY_RATIO``). 0.0 when the solution has no such lines."""
    want = [squash(ln) for ln in solution.split("\n") if ln.strip() and not is_comment_line(ln)]
    want = [w for w in want if len(w) >= 4]
    if not want:
        return 0.0
    have = [squash(ln) for ln in screen_lines if ln.strip()]
    have_set = set(have)
    hits = 0
    for w in want:
        if w in have_set or any(h.endswith(w) and len(h) - len(w) <= 4 for h in have):
            hits += 1
            continue
        if any(_similar(w, h) for h in have):
            hits += 1
    return hits / len(want)


def containment(old: set[str], new: set[str]) -> float:
    """Share of ``old`` still present in ``new`` (1.0 when nothing was lost; 0.0 when ``old`` is empty)."""
    if not old:
        return 0.0
    return len(old & new) / len(old)

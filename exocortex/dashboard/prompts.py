"""The dashboard refresh prompt and the parser for the model's changes.

A refresh shows the model the OPEN items of each tracked list under a short id (``Q1``, ``B2``,
``N1``, ``P1``), what was said since the last refresh, the screen and -- on demand -- lines from
the repository and the full text of a toolkit part that was mentioned, and asks for the CHANGES
only::

    ## Questions to ask
    - NEW: Should export() paginate past 1000 rows?
    - ANSWERED Q2: yes, the cache is keyed by tenant id
    ## Blind spots & follow-ups
    - DROP B1
    ## Next steps
    - DONE N1: the failing test is in
    ## Suggested prompts
    - NEW: /repro-fix export() returns 500 on an empty tenant

Unlisted open items stay open (a truncated answer never loses anything); duplicates of open
items are folded by the store. The system prompt is built only from the config and the toolkit
catalog, so it stays byte-identical across refreshes for the KV prefix cache
(:mod:`exocortex.llm.prompts`'s rule); the user turn starts with the current items, then the
transcript, then the volatile screen / repo blocks.
"""
from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from exocortex.bus import DashboardItem
from exocortex.llm import prompts

KINDS: tuple[str, ...] = ("question", "blindspot", "nextstep", "prompt")
#: The dashboard's refreshable sections (the columns a refresh can be limited to) and the kinds
#: each one covers. "insights" section names ("questions" / "blindspots") are the same words.
SECTIONS: tuple[str, ...] = ("questions", "blindspots", "nextsteps")
KINDS_OF_SECTION: dict[str, tuple[str, ...]] = {
    "questions": ("question",),
    "blindspots": ("blindspot",),
    "nextsteps": ("nextstep", "prompt"),
}
HEADINGS: dict[str, str] = {
    "question": "## Questions to ask",
    "blindspot": "## Blind spots & follow-ups",
    "nextstep": "## Next steps",
    "prompt": "## Suggested prompts",
}
LABELS: dict[str, str] = {
    "question": "Questions to ask",
    "blindspot": "Blind spots & follow-ups",
    "nextstep": "Next steps",
    "prompt": "Suggested prompts",
}
ALIAS: dict[str, str] = {"question": "Q", "blindspot": "B", "nextstep": "N", "prompt": "P"}
MAX_NEW_ITEMS = 8


def normalize_sections(sections: object) -> tuple[str, ...]:
    """The known sections `sections` names (section names or kind names), in order; () = all."""
    if isinstance(sections, str):
        sections = (sections,)
    try:
        wanted = {str(s).strip().casefold() for s in (sections or ())}
    except TypeError:
        return ()
    out: list[str] = []
    for sec in SECTIONS:
        if sec in wanted or any(k in wanted for k in KINDS_OF_SECTION[sec]):
            out.append(sec)
    return tuple(out)


def kinds_for(sections: object, *, suggest_prompts: bool = True) -> tuple[str, ...]:
    """The kinds one refresh covers: every kind of the named sections (all sections when none
    is named), minus "prompt" when prompt suggestions are off."""
    secs = normalize_sections(sections) or SECTIONS
    kinds = [k for sec in secs for k in KINDS_OF_SECTION[sec]]
    if not suggest_prompts:
        kinds = [k for k in kinds if k != "prompt"]
    return tuple(k for k in KINDS if k in kinds)


# ---------------------------------------------------------------------- rendering ----------
def render_current(open_by_kind: dict[str, Sequence[DashboardItem]], kinds: Sequence[str] = KINDS) -> tuple[str, dict[str, int]]:
    """The ``<current>`` body and the alias -> item id map it introduced."""
    lines: list[str] = []
    aliases: dict[str, int] = {}
    for kind in kinds:
        lines.append(f"{LABELS[kind]}:")
        items = list(open_by_kind.get(kind, ()))
        if not items:
            lines.append("- (none yet)")
        for n, item in enumerate(items, 1):
            alias = f"{ALIAS[kind]}{n}"
            aliases[alias] = item.id
            lines.append(f"- {alias}: {item.text}")
    return "\n".join(lines), aliases


def render_open_markdown(items: Sequence[DashboardItem]) -> str:
    """``- text`` lines for the Notes tab's Insights pane / insights.md ("" when empty)."""
    return "\n".join(f"- {i.text}" for i in items if i.status == "open")


def _kind_rule(kind: str, role: str, max_new: int, has_toolkit: bool) -> str:
    audience = prompts.QUESTION_AUDIENCE[prompts.normalize_role(role)]
    if kind == "question":
        return (
            f"{LABELS[kind]}: questions the user should put to {audience} right now -- to clarify a "
            "requirement, pin down an assumption, probe a design choice or an edge case, or settle "
            "something said but not decided. Each exactly as the user would say it: first person, at "
            "most 20 words, specific (the real function, file, service and variable names, the numbers); "
            "never generic (\"any concerns?\") and never one the transcript already answers."
        )
    if kind == "blindspot":
        return (
            f"{LABELS[kind]}: risks or gaps nobody has raised -- an unhandled edge case or error path, "
            "a missing test, a concurrency, performance, security or data-loss concern, a breaking change "
            "for callers, an unverified assumption -- and follow-ups not to forget (a TODO, someone to loop "
            "in, a ticket to file). Concern first, then in a few words why it matters, at most 20 words; "
            "only what the transcript, the screen or <repo> supports."
        )
    if kind == "nextstep":
        return (
            f"{LABELS[kind]}: the concrete next actions for the user -- the next edit, the test to write or "
            "run, the command to run, the thing to check -- in the order to do them, at most 20 words each, "
            "naming the real files, functions and commands."
        )
    tail = (
        "Use ONLY the skills, agents and workflows listed in <toolkit>, named exactly as listed, with real "
        "arguments (a /skill with its topic; \"Use the <agent> agent on <what>, one lens: <lens>\"); follow "
        "the working loop's order when it names one."
        if has_toolkit else
        "Write a plain, self-contained request for a coding agent (goal, files, constraints, done-when) in "
        "one or two sentences."
    )
    return (
        f"{LABELS[kind]}: what to type into Claude Code next, exactly as it would be typed, when a next "
        f"step calls for the coding agent. {tail} At most {max_new} per update, only when a step calls "
        "for one; DROP a prompt once it was run or is no longer the right next move."
    )


def system_dashboard(
    *, persona: str = "", role: str = "driver", context: str = "", max_new: int = 4,
    kinds: Sequence[str] = KINDS, toolkit: str = "",
) -> str:
    """System prompt for kind "dashboard". `toolkit` is the ``<toolkit>`` catalog block (or "");
    `kinds` the lists this refresh covers, in :data:`KINDS` order."""
    role = prompts.normalize_role(role)
    kinds = tuple(k for k in KINDS if k in set(kinds)) or KINDS
    try:
        n = max(1, min(int(max_new), MAX_NEW_ITEMS))
    except (TypeError, ValueError):
        n = 4
    has_toolkit = bool((toolkit or "").strip())
    id_list = ", ".join(f"{LABELS[k]} ({ALIAS[k]}1, {ALIAS[k]}2, ...)" for k in kinds)
    headings = ", ".join(f"\"{HEADINGS[k]}\"" for k in kinds)
    lines = [
        "You are the user's co-pilot in a live pair-programming / team coding session, keeping a small "
        "DASHBOARD current. " + prompts.ROLE_FRAMES[role],
        f"The dashboard tracks short lists with ids: {id_list}. <current> shows the OPEN items as they "
        "stand. <transcript_new> is what was said since the dashboard was last updated and "
        "<transcript_earlier> what came before it (\"Me\" is the user); <screen> is the user's screen "
        "now; <repo> (if present) holds lines from the project's own files that match names mentioned; "
        "<toolkit_detail> (if present) is the full text of a toolkit part that was mentioned.",
        "Report ONLY changes, under the headings below, one change per line; write nothing for an "
        "item that merely stays open:",
        f"- NEW: <text> -- a new item worth tracking (at most {n} per heading, most important first; "
        "never one that duplicates an open item).",
        "- ANSWERED Q3: <the answer in a few words> -- an open question the discussion has now answered. "
        "Likewise DONE N2: <what happened> for a step that was taken, and RESOLVED B1: <how> for a blind "
        "spot that was handled or settled. Always the id of the item, as shown in <current>.",
        "- DROP B2 -- an item that is no longer relevant (the plan changed, it is covered elsewhere).",
        *(_kind_rule(k, role, n, has_toolkit) for k in kinds),
        f"Output exactly these headings, in this order, each alone on its own line: {headings}. Under "
        "each, one change per line starting with \"- \"; a single \"- (no change)\" line when nothing "
        "changed there. No other headings, no preamble, no closing remarks, no code fences; never "
        "mention being an AI, a model, or a co-pilot. OCR text can be garbled: ignore noise.",
        prompts.FOCUS_RULE,
        "If there is no technical discussion and no code to work from at all, output exactly "
        f"\"{prompts.SKIP_MARKER}\" and nothing else.",
    ]
    if has_toolkit:
        lines.append(prompts.TOOLKIT_RULE)
        lines.append(toolkit.strip())
    lines.extend(prompts.context_lines(context))
    if persona:
        lines.append(f"About me: {persona}")
    return "\n".join(lines)


def user_dashboard(
    *, current: str, new: str, earlier: str = "", screen: str = "", repo: str = "", toolkit_detail: str = "",
) -> str:
    """User turn for :func:`system_dashboard`: current items (changes each refresh, but first so
    the stable system prompt stays a prefix), the transcript, then the volatile blocks."""
    parts = [f"<current>\n{current.strip() or '(nothing tracked yet)'}\n</current>"]
    if (earlier or "").strip():
        parts.append(f"<transcript_earlier>\n{earlier.strip()}\n</transcript_earlier>")
    parts.append(f"<transcript_new>\n{new.strip() or '(nothing new was said)'}\n</transcript_new>")
    screen_body = (screen or "").strip()
    parts.append(f"<screen>\n{screen_body or '(nothing readable on screen)'}\n</screen>")
    if (repo or "").strip():
        parts.append(repo.strip())
    if (toolkit_detail or "").strip():
        parts.append(toolkit_detail.strip())
    parts.append("Update the dashboard now: list only the changes under each heading, as instructed.")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------- parsing ------------
@dataclass
class DashboardUpdate:
    """The changes one answer asks for. `resolves` / `drops` / `keeps` carry the aliases as the
    model wrote them (upper-cased); the service maps them to item ids."""

    adds: list[tuple[str, str]] = field(default_factory=list)        # (kind, text)
    resolves: list[tuple[str, str]] = field(default_factory=list)    # (alias, answer)
    drops: list[str] = field(default_factory=list)
    keeps: list[str] = field(default_factory=list)
    unparsed: list[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not (self.adds or self.resolves or self.drops)


_HEADING_RE = re.compile(r"^#{1,6}\s*(.*?)\s*:?\s*$")
_BOLD_HEADING_RE = re.compile(r"^\*\*([^*]+?):?\*\*:?$")
_BULLET_RE = re.compile(r"^(?:[-*•]|\d+[.)])\s+(.*)$")
_CHECK_RE = re.compile(r"^\[(?P<x>[ xX])\]\s*(?P<rest>.*)$")
_VERB_RE = re.compile(
    r"^(?P<verb>new|add(?:ed)?|answered|done|resolved|fixed|settled|drop(?:ped)?|remove[d]?|delete[d]?|keep|kept|unchanged|open)"
    r"\b[\s:：-]*(?P<rest>.*)$",
    re.IGNORECASE,
)
_ALIAS_RE = re.compile(r"^(?P<alias>[QBNP]\s?\d{1,3})\b[\s:：\-–—]*(?P<rest>.*)$", re.IGNORECASE)
_ALIAS_ANYWHERE_RE = re.compile(r"\b([QBNP]\d{1,3})\b", re.IGNORECASE)
_NONE_ITEMS = frozenset({"(no change)", "no change", "(none)", "none", "(none yet)", "(nothing)", "-", "n/a", "(n/a)",
                         "(nothing to do)", "(nothing to add)", "no changes", "(no changes)", "unchanged"})
_RESOLVE_VERBS = frozenset({"answered", "done", "resolved", "fixed", "settled"})
_DROP_VERBS = frozenset({"drop", "dropped", "remove", "removed", "delete", "deleted"})
_KEEP_VERBS = frozenset({"keep", "kept", "unchanged", "open"})
_NEW_VERBS = frozenset({"new", "add", "added"})


def section_of_heading(title: str) -> str | None:
    """The kind a heading names (loosely: "question"; "blind"/"follow"/"risk"/"gap"; "next"/
    "step"/"action"; "prompt"/"claude"/"command"), or None."""
    t = (title or "").casefold()
    if "prompt" in t or "claude" in t or "command" in t:
        return "prompt"
    if "question" in t:
        return "question"
    if any(w in t for w in ("blind", "follow", "risk", "gap")):
        return "blindspot"
    if any(w in t for w in ("next", "step", "action", "todo", "to do")):
        return "nextstep"
    return None


def _clean_alias(alias: str) -> str:
    return alias.replace(" ", "").upper()


def _parse_body(body: str, kind: str, update: DashboardUpdate) -> None:
    text = body.strip()
    if not text or text.casefold().rstrip(". ") in _NONE_ITEMS:
        return
    check = _CHECK_RE.match(text)
    if check:
        text = check.group("rest").strip()
        if check.group("x").lower() == "x":
            m = _ALIAS_RE.match(text)
            if m:
                update.resolves.append((_clean_alias(m.group("alias")), m.group("rest").strip()))
                return
    verb_match = _VERB_RE.match(text)
    if verb_match:
        verb = verb_match.group("verb").casefold()
        rest = verb_match.group("rest").strip()
        if verb in _NEW_VERBS:
            if rest and rest.casefold().rstrip(". ") not in _NONE_ITEMS:
                update.adds.append((kind, rest))
            return
        if verb in _RESOLVE_VERBS or verb in _DROP_VERBS or verb in _KEEP_VERBS:
            m = _ALIAS_RE.match(rest)
            if m is None:
                found = _ALIAS_ANYWHERE_RE.findall(rest)
                if not found:
                    update.unparsed.append(text)
                    return
                aliases = [_clean_alias(a) for a in found]
                answer = ""
            else:
                aliases = [_clean_alias(m.group("alias"))]
                answer = m.group("rest").strip()
                # "DROP Q2, Q4" / "DROP Q2 and Q4": every id named, when the rest is only ids.
                more = _ALIAS_ANYWHERE_RE.findall(answer)
                if more and not re.sub(r"[QBNP]\d{1,3}|[,\s]|and", "", answer, flags=re.IGNORECASE).strip():
                    aliases.extend(_clean_alias(a) for a in more)
                    answer = ""
            for alias in aliases:
                if verb in _RESOLVE_VERBS:
                    update.resolves.append((alias, answer))
                elif verb in _DROP_VERBS:
                    update.drops.append(alias)
                else:
                    update.keeps.append(alias)
            return
    m = _ALIAS_RE.match(text)
    if m:
        alias = _clean_alias(m.group("alias"))
        rest = m.group("rest").strip()
        inner = _VERB_RE.match(rest)
        if inner:
            verb = inner.group("verb").casefold()
            tail = inner.group("rest").strip()
            if verb in _RESOLVE_VERBS:
                update.resolves.append((alias, tail))
            elif verb in _DROP_VERBS:
                update.drops.append(alias)
            elif verb in _KEEP_VERBS or verb in _NEW_VERBS:
                update.keeps.append(alias)
            return
        update.keeps.append(alias)   # a restated item: still open
        return
    update.adds.append((kind, text))


def parse_update(text: str, kinds: Sequence[str] = KINDS, *, max_new: int = 4) -> DashboardUpdate:
    """Split an answer into :class:`DashboardUpdate`. Lines under a heading naming one of
    `kinds` belong to it; other headings' lines are dropped; lines before the first heading
    go to the first kind. A bullet (or any non-empty line) is one change: a verb
    (``NEW`` / ``ANSWERED`` / ``DONE`` / ``RESOLVED`` / ``DROP`` / ``KEEP``) optionally followed
    by an id and text, an id followed by such a verb, or -- without any verb -- a new item.
    Adds are de-duplicated and capped at `max_new` per kind."""
    wanted = tuple(k for k in KINDS if k in set(kinds)) or KINDS
    update = DashboardUpdate()
    current: str | None = wanted[0]
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        heading = _HEADING_RE.match(line) if line.startswith("#") else _BOLD_HEADING_RE.match(line)
        if heading:
            kind = section_of_heading(heading.group(1))
            current = kind if kind in wanted else None
            continue
        if current is None:
            continue
        bullet = _BULLET_RE.match(line)
        _parse_body(bullet.group(1) if bullet else line, current, update)
    try:
        cap = max(1, min(int(max_new), MAX_NEW_ITEMS))
    except (TypeError, ValueError):
        cap = 4
    seen: set[tuple[str, str]] = set()
    per_kind: dict[str, int] = {}
    adds: list[tuple[str, str]] = []
    for kind, body in update.adds:
        key = (kind, " ".join(body.split()).casefold())
        if key in seen or per_kind.get(kind, 0) >= cap:
            continue
        seen.add(key)
        per_kind[kind] = per_kind.get(kind, 0) + 1
        adds.append((kind, " ".join(body.split())))
    update.adds = adds
    return update

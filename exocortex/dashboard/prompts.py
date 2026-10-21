"""The dashboard refresh prompt and the parser for the model's changes.

A refresh shows the model the OPEN items of each tracked list under a short id (``N1`` action
items, ``G1`` goals, ``C1`` constraints, ``Q1`` questions, ``B1`` blind spots, ``P1`` prompts),
what was said since the last refresh, the screen and -- on demand -- lines from the repository and
the full text of a toolkit part that was mentioned, and asks for the CHANGES only::

    ## Action items
    - NEW: Task 1: segment evaluation endpoint
    - NEW under N1: parse the filter expression
    - DONE N2: the parser is in
    ## Goals
    - NEW for N1: attribute filters return the matching customer ids
    - MET G1: the unit test passes
    ## Constraints
    - NEW: TypeScript on Node 22, no new dependencies
    ## Questions to ask
    - ANSWERED Q2: yes, the cache is keyed by tenant id
    ## Blind spots & follow-ups
    - DROP B1
    ## Suggested prompts
    - NEW for N1: /recon segment evaluation: filters, customers, the in-memory store

The action items are a tree one level deep (primary tasks, with sub-tasks indented under them in
``<current>``); goals and prompts hang off the action item they name. Unlisted open items stay open
(a truncated answer never loses anything); duplicates of open items are folded by the store. The
system prompt is built only from the config and the toolkit catalog, so it stays byte-identical
across refreshes for the KV prefix cache (:mod:`exocortex.llm.prompts`'s rule); the user turn
starts with the current items, then the transcript, then the volatile screen / repo blocks.
"""
from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import NamedTuple

from exocortex.bus import DashboardItem
from exocortex.llm import prompts

#: Tasks first: goals and prompts refer to them by id, so a refresh lists them before those.
KINDS: tuple[str, ...] = ("nextstep", "goal", "constraint", "question", "blindspot", "prompt")
#: The dashboard's refreshable sections (the columns a refresh can be limited to) and the kinds
#: each one covers. "insights" section names ("questions" / "blindspots") are the same words.
SECTIONS: tuple[str, ...] = ("nextsteps", "goals", "constraints", "questions", "blindspots")
KINDS_OF_SECTION: dict[str, tuple[str, ...]] = {
    "nextsteps": ("nextstep", "prompt"),
    "goals": ("goal",),
    "constraints": ("constraint",),
    "questions": ("question",),
    "blindspots": ("blindspot",),
}
HEADINGS: dict[str, str] = {
    "nextstep": "## Action items",
    "goal": "## Goals",
    "constraint": "## Constraints",
    "question": "## Questions to ask",
    "blindspot": "## Blind spots & follow-ups",
    "prompt": "## Suggested prompts",
}
LABELS: dict[str, str] = {
    "nextstep": "Action items",
    "goal": "Goals",
    "constraint": "Constraints",
    "question": "Questions to ask",
    "blindspot": "Blind spots & follow-ups",
    "prompt": "Suggested prompts",
}
ALIAS: dict[str, str] = {"nextstep": "N", "goal": "G", "constraint": "C", "question": "Q", "blindspot": "B", "prompt": "P"}
#: Kinds whose new items may name the action item they belong to (``NEW under N1:`` / ``NEW for N1:``).
PARENTED_KINDS: tuple[str, ...] = ("nextstep", "goal", "prompt")
#: How many open items of a kind a refresh is shown, as a multiple of ``[dashboard].max_open_items``
#: (a task list has sub-tasks, a task has several criteria).
_CAP_FACTOR: dict[str, int] = {"nextstep": 2, "goal": 2}
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
    is named), minus "prompt" when prompt suggestions are off. Goals and prompts name action
    items, so those are shown (and may be updated) whenever either of them is covered."""
    secs = normalize_sections(sections) or SECTIONS
    kinds = {k for sec in secs for k in KINDS_OF_SECTION[sec]}
    if not suggest_prompts:
        kinds.discard("prompt")
    if kinds & {"goal", "prompt"}:
        kinds.add("nextstep")
    return tuple(k for k in KINDS if k in kinds)


# ---------------------------------------------------------------------- rendering ----------
def tree_order(items: Sequence[DashboardItem]) -> list[DashboardItem]:
    """Action items as a tree: each primary task (oldest first) followed by its sub-tasks. A
    sub-task whose task is not among `items` is treated as a task of its own."""
    ids = {i.id for i in items}
    roots = [i for i in items if i.parent_id is None or i.parent_id not in ids]
    kids: dict[int, list[DashboardItem]] = {}
    for i in items:
        if i.parent_id is not None and i.parent_id in ids:
            kids.setdefault(i.parent_id, []).append(i)
    out: list[DashboardItem] = []
    for root in sorted(roots, key=lambda i: (i.created_at, i.id)):
        out.append(root)
        out.extend(sorted(kids.get(root.id, ()), key=lambda i: (i.created_at, i.id)))
    return out


def render_current(
    open_by_kind: dict[str, Sequence[DashboardItem]], kinds: Sequence[str] = KINDS, *, limit: int = 0,
) -> tuple[str, dict[str, int]]:
    """The ``<current>`` body and the alias -> item id map it introduced. Action items appear as
    a tree (sub-tasks indented); a goal or prompt says which action item it is for. With `limit`
    > 0 a kind shows at most that many open items (x2 for action items and goals), the earliest
    first; a task is shown with all its sub-tasks or not at all, so the cap on action items is
    soft by the last task's steps."""
    lines: list[str] = []
    aliases: dict[str, int] = {}
    alias_of: dict[int, str] = {}
    for kind in kinds:
        lines.append(f"{LABELS[kind]}:")
        items = list(open_by_kind.get(kind, ()))
        if kind == "nextstep":
            items = tree_order(items)
        cap = limit * _CAP_FACTOR.get(kind, 1) if limit > 0 else 0
        if cap and len(items) > cap:
            if kind == "nextstep":
                shown_ids = {i.id for i in items}
                kept: list[DashboardItem] = []
                for item in items:
                    is_sub = item.parent_id is not None and item.parent_id in shown_ids
                    if len(kept) >= cap and not is_sub:
                        break
                    kept.append(item)
                items = kept
            else:
                items = items[:cap]
        if not items:
            lines.append("- (none yet)")
        top_ids = {i.id for i in items}
        for n, item in enumerate(items, 1):
            alias = f"{ALIAS[kind]}{n}"
            aliases[alias] = item.id
            alias_of[item.id] = alias
            indent = "  " if kind == "nextstep" and item.parent_id is not None and item.parent_id in top_ids else ""
            owner = alias_of.get(item.parent_id) if kind in ("goal", "prompt") and item.parent_id is not None else None
            tag = f" (for {owner})" if owner else ""
            lines.append(f"{indent}- {alias}{tag}: {item.text}")
    return "\n".join(lines), aliases


def render_open_markdown(items: Sequence[DashboardItem]) -> str:
    """``- text`` lines for the Notes tab's Insights pane / insights.md ("" when empty)."""
    return "\n".join(f"- {i.text}" for i in items if i.status == "open")


def _kind_rule(kind: str, role: str, max_new: int, has_toolkit: bool, example: str) -> str:
    audience = prompts.QUESTION_AUDIENCE[prompts.normalize_role(role)]
    if kind == "nextstep":
        return (
            f"{LABELS[kind]}: the work to do, as a tree. A PRIMARY TASK is one thing the brief, the interviewer "
            "or the user asked for (\"Task 1: ...\", one line, at most 15 words); under it go its SUB-TASKS, "
            "the concrete steps in the order to do them (at most 15 words each, naming the real files, "
            "functions and commands). New primary task: \"NEW: <task>\". New sub-task: \"NEW under N2: "
            "<step>\" (N2 = the primary task it belongs to). Mark an item DONE only when it was actually "
            "finished, never because it was discussed."
        )
    if kind == "goal":
        return (
            f"{LABELS[kind]}: the acceptance criteria -- how we will know an action item is done: the "
            "behaviour, output, status code, test or number that must hold, one criterion per line, at "
            "most 20 words, as stated in the brief or the discussion (never invented). Attach each to its "
            "action item: \"NEW for N2: <criterion>\"; a criterion for the whole session has no id: "
            "\"NEW: <criterion>\". Report \"MET G3: <the evidence>\" when it was shown to hold (a passing "
            "test, observed output)."
        )
    if kind == "constraint":
        return (
            f"{LABELS[kind]}: the rules and guardrails that MUST be respected -- stack and versions, the "
            "time box, \"no new dependencies\", diff size or style limits, data and safety rules, what not "
            "to touch or to ask first -- whether the brief, the interviewer or the user stated them. Each "
            "one line phrased as a rule, at most 20 words; only what was actually said or is written on "
            "screen, never invented. DROP C2 when it no longer applies (lifted or superseded)."
        )
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
    if has_toolkit:
        tail = (
            "Start every prompt with a slash command from <toolkit>"
            + (f" (like {example})" if example else "")
            + " or with \"Use the <agent> agent on <what>\", named exactly as listed there -- never "
            "invent one -- and follow the working loop's order for the stage the task is at. Give it the "
            "task's real details as arguments (the files, endpoints, what done means) and carry in the "
            "constraints and acceptance criteria that matter. For each open primary action item that has "
            "no prompt yet, suggest the prompt for its next step."
        )
    else:
        tail = (
            "Write a plain, self-contained request for a coding agent (goal, files, constraints, done-when) in "
            "one or two sentences."
        )
    return (
        f"{LABELS[kind]}: what to type into Claude Code next, exactly as it would be typed. {tail} Attach "
        f"each to the action item it serves: \"NEW for N2: <prompt>\". At most {max_new} per update; DROP a "
        "prompt once it was run or is no longer the right next move."
    )


def system_dashboard(
    *, persona: str = "", role: str = "driver", context: str = "", max_new: int = 4,
    kinds: Sequence[str] = KINDS, toolkit: str = "", example: str = "",
) -> str:
    """System prompt for kind "dashboard". `toolkit` is the ``<toolkit>`` catalog block (or "");
    `example` a real command from it to show the prompt format; `kinds` the lists this refresh
    covers, in :data:`KINDS` order."""
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
        "never one that duplicates an open item). A new item takes the next free number of its list, in "
        "the order you write them (with N1..N3 shown, the next new action item is N4), so a sub-task, "
        "goal or prompt written after it can name it.",
        "- ANSWERED Q3: <the answer in a few words> -- an open question the discussion has now answered. "
        "Likewise DONE N2: <what happened> for an action item that was finished, MET G1: <evidence> for a goal "
        "that was shown to hold, and RESOLVED B1: <how> for a blind spot that was handled or settled. Always "
        "the id of the item, as shown in <current>.",
        "- DROP B2 -- an item that is no longer relevant (the plan changed, it is covered elsewhere).",
        *(_kind_rule(k, role, n, has_toolkit, example) for k in kinds),
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
class Add(NamedTuple):
    """One new item the model asked for: its kind, its text and -- for a sub-task, a goal or a
    prompt -- the alias of the action item it belongs to ("" = none)."""

    kind: str
    text: str
    parent: str = ""


@dataclass
class DashboardUpdate:
    """The changes one answer asks for. `resolves` / `drops` / `keeps` carry the aliases as the
    model wrote them (upper-cased); the service maps them to item ids."""

    adds: list[Add] = field(default_factory=list)
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
    r"^(?P<verb>new|add(?:ed)?|answered|done|resolved|fixed|settled|met"
    r"|drop(?:ped)?|remove[d]?|delete[d]?|keep|kept|unchanged|open)"
    r"\b[\s:：-]*(?P<rest>.*)$",
    re.IGNORECASE,
)
_ALIAS_RE = re.compile(r"^(?P<alias>[QBNPCG]\s?\d{1,3})\b[\s:：\-–—]*(?P<rest>.*)$", re.IGNORECASE)
_ALIAS_ANYWHERE_RE = re.compile(r"\b([QBNPCG]\d{1,3})\b", re.IGNORECASE)
#: "under N2: step" / "for N2: criterion" / "(for N2) text" -- the action item a new item belongs to.
_TARGET_RE = re.compile(
    r"^\(?\s*(?:(?P<prep>under|for|of|in|on|to)\s+)?(?P<alias>N\s?\d{1,3})\s*\)?\s*[:\-–—]?\s*(?P<rest>.+)$",
    re.IGNORECASE,
)
_NONE_ITEMS = frozenset({"(no change)", "no change", "(none)", "none", "(none yet)", "(nothing)", "-", "n/a", "(n/a)",
                         "(nothing to do)", "(nothing to add)", "no changes", "(no changes)", "unchanged"})
_RESOLVE_VERBS = frozenset({"answered", "done", "resolved", "fixed", "settled", "met"})
_DROP_VERBS = frozenset({"drop", "dropped", "remove", "removed", "delete", "deleted"})
_KEEP_VERBS = frozenset({"keep", "kept", "unchanged", "open"})
_NEW_VERBS = frozenset({"new", "add", "added"})


def section_of_heading(title: str) -> str | None:
    """The kind a heading names (loosely: "prompt"/"claude"/"command"; "question"; "constraint"/
    "guardrail"/"rule"; "goal"/"acceptance"/"criteri"; "blind"/"follow"/"risk"/"gap"; "action"/
    "task"/"next"/"step"/"todo"), or None."""
    t = (title or "").casefold()
    if "prompt" in t or "claude" in t or "command" in t:
        return "prompt"
    if "question" in t:
        return "question"
    if any(w in t for w in ("constraint", "guardrail", "rule", "limit")):
        return "constraint"
    if any(w in t for w in ("goal", "acceptance", "criteri")):
        return "goal"
    if any(w in t for w in ("blind", "follow", "risk", "gap")):
        return "blindspot"
    if any(w in t for w in ("action", "task", "next", "step", "todo", "to do")):
        return "nextstep"
    return None


def _clean_alias(alias: str) -> str:
    return alias.replace(" ", "").upper()


def _new_item(kind: str, rest: str, update: DashboardUpdate, *, bare: bool = True) -> None:
    """A NEW item: `rest` may start with the action item it belongs to (``under N2: ...``; for a
    task, goal or prompt only). A bare alias (``N2: ...``) counts only after an explicit NEW;
    without a verb, only with a preposition (a bare ``N2: text`` there is a restated item)."""
    rest = rest.strip()
    if not rest or rest.casefold().rstrip(". ") in _NONE_ITEMS:
        return
    if kind in PARENTED_KINDS:
        m = _TARGET_RE.match(rest)
        if m is not None and (bare or m.group("prep")):
            update.adds.append(Add(kind, m.group("rest").strip(), _clean_alias(m.group("alias"))))
            return
    update.adds.append(Add(kind, rest))


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
            _new_item(kind, rest, update)
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
                if more and not re.sub(r"[QBNPCG]\d{1,3}|[,\s]|and", "", answer, flags=re.IGNORECASE).strip():
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
    _new_item(kind, text, update, bare=False)


def parse_update(text: str, kinds: Sequence[str] = KINDS, *, max_new: int = 4) -> DashboardUpdate:
    """Split an answer into :class:`DashboardUpdate`. Lines under a heading naming one of
    `kinds` belong to it; other headings' lines are dropped; lines before the first heading
    go to the first kind. A bullet (or any non-empty line) is one change: a verb
    (``NEW`` / ``ANSWERED`` / ``DONE`` / ``MET`` / ``RESOLVED`` / ``DROP`` / ``KEEP``) optionally followed
    by an id and text, an id followed by such a verb, or -- without any verb -- a new item. A new
    task, goal or prompt may name the action item it belongs to (``NEW under N2: ...``). Adds are
    de-duplicated and capped at `max_new` per kind (and per action item)."""
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
    seen: set[tuple[str, str, str]] = set()
    per_bucket: dict[tuple[str, str], int] = {}
    adds: list[Add] = []
    for add in update.adds:
        body = " ".join(add.text.split())
        key = (add.kind, add.parent, body.casefold())
        bucket = (add.kind, add.parent)
        if key in seen or per_bucket.get(bucket, 0) >= cap or not body:
            continue
        seen.add(key)
        per_bucket[bucket] = per_bucket.get(bucket, 0) + 1
        adds.append(Add(add.kind, body, add.parent))
    update.adds = adds
    return update

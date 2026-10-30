"""Assistant: auto/manual question answering and running meeting notes, backed by LM Studio.

Owns five daemon threads:

- ``assist-worker`` -- runs one job at a time, drawn from seven latest-wins single-slot mailboxes,
  in priority order: the main lane (``kind`` "auto"/"manual"/"ask"/"screen"), the side lane
  ("suggest"/"solve"/"code"), the text lane ("text"), the recap lane ("recap"), the explain lane
  ("explain"), the insight lane ("insights") and the completion lane ("complete").
  The main lane always has priority: submitting a main-lane job cancels whatever job is currently
  running (of any lane) and replaces whatever main-lane job was queued; a lower lane's job never
  cancels a running higher lane's job -- it simply waits in its own slot (a newer job of the same
  lane before it runs replaces it) and runs once the lanes above are idle. Within a lane, a new
  job cancels a running job of that lane (a new solve also cancels a running explanation of the
  previous task, and "suggest"/"solve"/"code" all cancel a running "code") -- except "text", which
  only ever cancels an older "text": a walkthrough neither preempts nor is preempted by a running
  solve/code/explain, they simply queue behind each other. That is also why "text" has a mailbox
  of its own rather than sharing the side lane's: a mailbox holds ONE queued job, so a queued
  walkthrough would otherwise be silently overwritten by the next auto-solve (and a queued Code by
  a Text press) with no event ever published. See :meth:`Assistant._next_assist_job` and
  ``_CANCELS``.
- A **checkpoint** (:meth:`Assistant.checkpoint`) is a soft reset of the CODING context only:
  from that point on, "solve", "code", "text" and inline completions ("complete") see only
  transcript segments and answers produced AFTER the checkpoint, and the current plan (if any) is
  dropped.
  Everything meeting-wide -- the transcript store, the answer history/export, the ask box, the
  screen-question de-dup memory, the auto-suggest throttle state, ... -- is left untouched. See
  :meth:`Assistant.checkpoint` for the exact contract.
- "code" (:meth:`Assistant.code`, the Code button) writes the code for the task discussed/shown
  since the last checkpoint. It runs on the same side lane as "suggest"/"solve" (never cancels a
  main-lane job) but, unlike ``solve()``, is MANUAL: it is never refused while paused.
- "text" (:meth:`Assistant.text`, the Text button) is the prose twin of "code": a plain-English
  walkthrough of that same task (what is asked, the key insight, the approach step by step, the
  edge cases, the complexity) -- the way an engineer talks a problem through before coding. Its
  own lane (dispatched after the side lane, before explanations), same checkpoint-scoped
  transcript, same <previous> code block, manual like "code"; stored as an answer but never as
  code (``_code_answers``), never part of the plan, and never recorded in the auto-solve throttle
  (a prose request must not delay the automatic solve of the next screen).
- How much the generated code is commented -- "solve", "code", and code continued by inline
  completions -- follows ``cfg.assist.comments`` (``prompts.COMMENT_LEVELS``; "detailed" also
  opens the code with a requirements + pseudo-code comment block); its budget headroom is
  ``COMMENT_TOKEN_HEADROOM``.
- With explanations on (``cfg.explain.enabled``) a "solve" or "code" answer is the session's
  **plan** (walkthrough, commented solution, complexity); the completer follows it
  (:meth:`Assistant.plan`) and, once the solution's code has been typed out -- reported by the
  completer (:meth:`plan_typed`) or seen on screen (:meth:`_maybe_explain_from_screen`) -- an
  "explain" job explains the finished solution.
- ``notes-worker`` -- runs notes jobs the same way (at most one queued; the running one is not
  cancellable). 2026-10-06 (context library): it is also the **notes lane** for the two small
  background jobs the context library adds -- a *compaction* (fold the transcript that fell out of
  the anchored window into the running ``<earlier>`` summary) and a *digest* (a 2-4 line catalogue
  entry for an uploaded reference document). Each has its own latest-wins slot; notes always go
  first, then compactions, then digests (:meth:`Assistant._next_notes_job`). Keeping them off the
  assist lane means a compaction or a digest can never delay a live answer.
- Every assist job's prompt is assembled by :class:`exocortex.context.ContextManager` (the
  anchored transcript window, the reference/lookup blocks, the earlier summary, all sized to the
  server's real context -- see ``exocortex/context.py``); the Assistant publishes how that went as
  two :class:`~exocortex.bus.ContextUsageEvent` per job and feeds the server's own prompt-token
  count back so the estimate calibrates itself.
- 2026-10-07 (capabilities): with a :class:`exocortex.capabilities.ToolRunner` (``tools=``) a job
  whose kind it covers runs a **tool loop** on the assist worker (:meth:`Assistant._stream_answer`):
  the system prompt ends with the ``<tools>`` list, the user turn carries the project ``<map>``,
  and when the model writes ``<tool>{...}</tool>`` (on a line of its own, or after a line of
  prose) the server stops the pass at the closing tag, the runner executes the call under the
  ``[capabilities]`` policy, and the conversation is continued: the call closes the assistant
  turn, its ``<result>`` block comes back as a USER turn and a fresh assistant turn is opened
  (2026-10-08: every request a byte-for-byte extension of the previous one plus what the model
  generated, so the KV prefix cache stays intact; the chat path appends two messages). Only the
  visible text reaches the deltas, the history, the plan and ``AssistDoneEvent``; every call is
  published as a :class:`~exocortex.bus.ToolCallEvent`. Never for "complete" / "explain", nor for
  the notes lane's jobs.
- ``bus-listener`` -- drains this instance's own bus subscription, watching for
  :class:`~exocortex.bus.FinalTranscriptEvent` to drive auto-answer mode and for
  :class:`~exocortex.bus.ScreenTextEvent` to drive screen-triggered questions and auto-suggest.
- ``notes-auto-timer`` -- calls :meth:`Assistant.generate_notes` with ``force=False`` every
  ``cfg.notes.auto_interval_s`` seconds (a no-op thread when that interval is ``<= 0``).
- ``insight-timer`` -- every ``INSIGHT_TICK_S`` seconds, starts a background **co-pilot insights**
  pass (kind "insights": questions worth asking + blind spots / follow-ups, per
  ``[assist].suggest_questions`` / ``flag_blind_spots``) once ``insight_interval_s`` has passed
  since the last pass AND either ``insight_min_new_words`` of new speech arrived or the screen
  changed substantially (:meth:`Assistant._maybe_auto_insights`). A background pass is the
  lowest-priority work there is: any other job submitted while it runs cancels it (it is retried
  ``INSIGHT_RETRY_S`` after it started), and it never replaces a queued on-demand request
  (:meth:`Assistant.insights_now`). The result is a living document (:attr:`Assistant.insights`)
  published as :class:`~exocortex.bus.InsightsUpdatedEvent` -- never an answer, never shown in
  the Assist tab. "recap" (:meth:`Assistant.recap`) is the on-demand "where are we" summary; it
  is stored as an answer like "text".

Auto mode additionally uses a ``threading.Timer`` ("coalesce timer") that is (re)started every
time a new SYSTEM question segment arrives, so a multi-sentence question spoken over a couple of
seconds is answered once, combined, instead of once per sentence.

All cross-thread state (the running/queued job, the notes document, the answer history, the
checkpoint boundary and its own answer list, the coalesce timer, the auto-answer flag, the cached
model list, the screen-reading state) is protected by small dedicated locks so any public method
can be called from any thread (the Qt UI thread, hotkey callbacks, the bus-listener thread, ...).
"""
from __future__ import annotations

import dataclasses
import itertools
import logging
import queue
import re
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Generic, NamedTuple, TypeVar

from exocortex.bus import (
    AssistDeltaEvent,
    AssistDoneEvent,
    AssistErrorEvent,
    AssistKind,
    AssistStartedEvent,
    AutoAssistChangedEvent,
    CheckpointEvent,
    ComponentStatusEvent,
    ContextLibraryEvent,
    ContextUsageEvent,
    EventBus,
    FinalTranscriptEvent,
    InsightsUpdatedEvent,
    NotesErrorEvent,
    NotesUpdatedEvent,
    ScreenTextEvent,
    Source,
    ToolCallEvent,
    TranscriptSegment,
)
from exocortex.config import Config
from exocortex.context import ContextBundle, ContextManager, ContextStore
from exocortex.llm import prompts, questions
from exocortex.llm.choices import ChoiceSet, annotate_choices, choices_block, finalize_answer, find_choices
from exocortex.llm.client import LLMError, LMStudioClient, StreamStats
from exocortex.plan import PlanView, containment, split_plan, typed_fraction
from exocortex.transcript import Transcript

log = logging.getLogger(__name__)

#: How far back (in seconds, by segment `t_end`) auto-mode looks when coalescing consecutive
#: SYSTEM question segments into one question.
_AUTO_COALESCE_WINDOW_S = 15.0

#: How many trailing SYSTEM segments `answer_last()`/auto-mode ever consider joining.
_MAX_CONTEXT_SEGMENTS = 3

#: Minimum seconds between two answers triggered by a NEW question appearing on screen (kind
#: "screen"). A module attribute (read fresh on every check, never cached) so tests can shrink it
#: with `monkeypatch.setattr(exocortex.llm.assistant, "SCREEN_QUESTION_MIN_INTERVAL_S", ...)`.
SCREEN_QUESTION_MIN_INTERVAL_S: float = 15.0

#: How long the suggest-lane worker polls the main-lane slot's condition while waiting for either
#: lane to receive a job (see `Assistant._next_assist_job`). Short enough that a suggest job that
#: became ready while the main lane was empty starts promptly.
_DISPATCH_POLL_S = 0.05

#: Non-word, non-space characters stripped when normalising an on-screen question for the
#: recently-answered de-duplication set.
_PUNCT_RE = re.compile(r"[^\w\s]")

#: Extra tokens a job gets when a multiple-choice option set was detected: room for the mandatory
#: "Answer: …" line after an answer that already fills its style's shape (8 bullets + "Bottom
#: line:" in the "detailed" style can use up `assist_max_tokens` on their own).
CHOICES_TOKEN_HEADROOM = 64

#: A spoken / typed question that refers to on-screen answer options ("which one is it?"), the
#: only case in which such a question borrows the option set found on the screen.
_REFERS_TO_OPTIONS_RE = re.compile(
    r"\b(which|options?|choices?|answer|correct|right one|pick|choose|select|these|this one|"
    r"true or false|multiple[- ]choice|a or b|a, b)\b",
    re.IGNORECASE,
)

#: While a plan exists, a screen change is still "the same task" (no new solve) as long as this
#: share of the lines the plan was made from is still on screen. Typing the solution into an editor
#: only ADDS lines (the problem statement stays), switching to another problem removes them.
PLAN_KEEP_FRACTION: float = 0.4

#: Extra tokens a "solve" / "code" answer gets per `cfg.assist.comments` level: the comments are
#: tokens on top of the same code (a comment line per logical step for "normal"; a requirements +
#: pseudo-code block and a comment above most lines for "detailed", which roughly doubles the
#: fenced block), and `solve_max_tokens` (700) was sized for barely commented code.
COMMENT_TOKEN_HEADROOM: dict[str, int] = {"none": 0, "minimal": 0, "normal": 100, "detailed": 350}

#: How often (seconds) the insight-timer thread checks whether a background insights pass is due.
#: Module attributes, read fresh on every tick / check, so tests can shrink them.
INSIGHT_TICK_S: float = 5.0
#: After a background insights pass was cancelled (another job preempted it) or failed, the next
#: one may start this many seconds after it started, instead of a whole interval later.
INSIGHT_RETRY_S: float = 30.0
#: Without enough new speech, a background pass still runs when the screen (the code being
#: written) changed substantially since the last pass: its normalised lines are less similar than
#: this (Jaccard) to the ones that pass saw, and there are at least `INSIGHT_MIN_SCREEN_LINES`.
INSIGHT_SCREEN_JACCARD: float = 0.6
INSIGHT_MIN_SCREEN_LINES: int = 5

#: Which lane's running job a newly submitted job of each kind cancels (latest wins within a lane;
#: a new task supersedes the previous task's explanation; nothing below ever touches the main lane).
#: "text" is deliberately the odd one out: it cancels only an older walkthrough and is cancelled by
#: nothing but a newer one (plus checkpoint / pause / reset) -- a Text press must never throw away
#: a streaming solve (the plan) and an automatic re-solve must never silently kill a manual request.
#: The same guarantee for QUEUED jobs comes from "text" having its own `_text_slot`.
_CANCELS: dict[str, tuple[str, ...]] = {
    "suggest": ("suggest", "solve", "code", "explain"),
    "solve": ("suggest", "solve", "code", "explain"),
    "code": ("suggest", "solve", "code", "explain"),
    "text": ("text",),
    "explain": ("explain",),
    "complete": ("complete",),
    "recap": ("recap",),
    "insights": ("insights",),
    "dashboard": ("dashboard",),
}

#: Kinds whose SYSTEM prompt carries the installed Claude Code toolkit's catalog (<toolkit>,
#: exocortex.project) when `[project].toolkit_in_answers` is on; "dashboard" builds its own.
_TOOLKIT_KINDS: frozenset[str] = frozenset({"ask", "auto", "manual", "screen", "recap", "text", "code", "solve", "suggest", "insights"})
#: Kinds whose USER turn gets the on-demand <repo> / <toolkit_detail> lookups (the question, the
#: tail of the transcript and the screen name the identifiers searched for).
_LOOKUP_KINDS: frozenset[str] = frozenset({"ask", "auto", "manual", "screen", "recap", "text", "code", "solve"})
#: File-scan budget of one repo lookup made for an answer (a spoken question must stay snappy);
#: the dashboard refresh uses its own, longer budget.
_LOOKUP_BUDGET_S = 0.8
#: How much of the newest transcript the repo / toolkit lookups read for the names to search
#: (`_project_extra`); taken before the context manager sizes the window, so the lookups' own
#: length can be budgeted.
_LOOKUP_TAIL_CHARS = 600

#: An `LLMError` message that means "the prompt did not fit the server's context" (llama-server:
#: ``request (N tokens) exceeds the available context size (4096 tokens)``; LM Studio / OpenAI
#: style servers word it "too many / too long" or name ``n_ctx``). The one case in which an assist
#: job is rebuilt smaller and streamed once more (see `Assistant._run_assist_job`).
_OVERFLOW_RE = re.compile(r"exceed|context size|too (?:many|long)|n_ctx", re.IGNORECASE)

#: A document digest is made from at most this many leading characters of the document (and fewer
#: when the notes lane has less room: `ContextManager.available_chars`).
_DIGEST_EXCERPT_CHARS = 6000

#: The notes job's transcript chunk never shrinks below this, whatever the notes lane's room says
#: (`head_text` returns nothing for a zero budget, which would stall the notes forever).
_MIN_NOTES_CHUNK_CHARS = 200
#: 2026-10-06: a notes pass keeps the configured answer budget as long as the lane leaves at least
#: this many tokens of new transcript beside it; below that the document scales down (see
#: `Assistant._notes_budget`), but never under `_NOTES_MIN_DOC_TOKENS`.
_NOTES_MIN_CHUNK_TOKENS = 400
_NOTES_MIN_DOC_TOKENS = 200
#: 2026-10-06: a compaction / digest steps aside for a live answer this many times before it runs
#: to completion regardless, and waits at most this long for the answer lane to go idle first.
_LANE_YIELD_ATTEMPTS = 2
_LANE_IDLE_WAIT_S = 20.0


def _head_lines(text: str, max_chars: int) -> str:
    """The first `max_chars` characters of `text`, ending on a line boundary when there is one
    (the oldest headings of a notes document survive a cut; the tail is what the rewrite adds back)."""
    if len(text) <= max_chars:
        return text
    head = text[:max(0, max_chars)]
    cut = head.rfind("\n")
    return head[:cut] if cut > 0 else head

_T = TypeVar("_T")


def _jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard similarity of two sets of lines; 0.0 when exactly one of them is empty."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _normalize_question(text: str) -> str:
    """Casefold, strip punctuation, and collapse whitespace (de-duplicates screen questions)."""
    return " ".join(_PUNCT_RE.sub("", text).split()).casefold()


def _normalize_line(text: str) -> str:
    """Casefold and collapse whitespace (matches `exocortex.screen.ocr.new_lines`'s own
    normalisation), so two OCR passes of the *same* on-screen content -- which commonly differ
    only in case or incidental whitespace -- compare as identical for auto-suggest's Jaccard
    similarity check instead of as entirely unrelated lines."""
    return " ".join(text.split()).casefold()


def _normalized_line_set(lines: Sequence[str]) -> set[str]:
    """`lines`, normalised via `_normalize_line` and with blanks dropped, as a set."""
    return {norm for line in lines if (norm := _normalize_line(line))}


# ---- the tool loop's pieces (2026-10-07, capabilities) -------------------------------------

#: Kinds that never get tools, whatever the runner says (inline completions are ghost text typed
#: into another app; an explanation answers a finished plan). dev-assist's own background kinds
#: join them: their system prompts carry no <tools> block, and the dashboard builds its own.
_NO_TOOL_KINDS = frozenset({"complete", "explain", "insights", "recap", "dashboard"})

#: Appended to the result of the call that used the last one (or after the time budget ran out):
#: the model's cue, repeated by `prompts._TOOL_RULE`, to write the answer now.
_NO_CALLS_LEFT = "(no calls left: write the answer now)"

#: What a result becomes when the server says the continuation did not fit its context.
_RESULT_DROPPED = "(result dropped to fit the context)"

#: A short narration that announces a lookup but makes none ("Let me check the config file."):
#: the first stream of an answer that matches gets ONE retry with the call opened for it. The
#: announcement must OPEN the text (after optional filler such as "Okay,") with a first-person
#: lookup verb ("let me check", "I'll start by reading", "I need to look"), or be a bare
#: "Checking / Reading / Looking at ..." clause that is the whole text -- so a complete answer
#: that merely says "Looking at the transcript, Atlas ships Friday." or "Let me know if ..." is not
#: one.
_INTENT_RE = re.compile(
    r"^\W*(?:(?:okay|ok|sure|alright|right|so|now|first|well)\b\W*)*"
    r"(?:(?:let me|let['\u2019]s|i(?:'ll|\u2019ll| will)|i need to|i['\u2019]m going to)\s+"
    r"(?:(?:first|just|now|quickly)\s+)*"
    r"(?:check|look|read|open|search|find|verify|run|list|fetch|grep|pull|take a look|examine|inspect|review|browse"
    r"|scan|load|start by|see (?:if|what|whether|how|which|where|the))\b"
    r"|(?:checking|reading|searching|looking (?:up|at|in|through|for))\b"
    r"(?:(?![,;:]|[.!?]\s+\S)[^\n]){0,80}$)",
    re.IGNORECASE,
)
_INTENT_MAX_CHARS = 240
#: ... and it only counts when it is unfinished (no full stop at its end) or names what it will
#: look at -- a file, a folder, a command, a url -- because a finished sentence that names nothing
#: ("Let me check that.") is as likely an answer as an announcement.
_SENTENCE_END_RE = re.compile(r"[.!?\u2026][\"')\]\u201d]*\s*$")
_INTENT_TARGET_RE = re.compile(
    r"[`\"/\\]|\bhttps?:|\bwww\.|\b[\w-]+\.[A-Za-z]{1,6}\b"
    r"|\b(?:files?|folders?|directory|directories|dir|repo|repository|project|codebase|command|commands|url|link|page|site"
    r"|website|docs?|documentation|readme|config|configuration|settings|script|logs?|source|code|module|package|tests?)\b"
    r"|\b(?:git|pip|npm|ls|tree|grep|rg|findstr|pytest|python|where|which|cat|type)\s+\S",
    re.IGNORECASE,
)


def _announces_lookup(text: str) -> bool:
    """Is `text` (the first stream of an answer) only an announcement of a lookup it never made?"""
    if not text.strip() or len(text) >= _INTENT_MAX_CHARS or not _INTENT_RE.search(text):
        return False
    return _SENTENCE_END_RE.search(text) is None or _INTENT_TARGET_RE.search(text) is not None


#: The chat path cannot prefill an assistant turn, so the two retries below ask in words instead.
_CHAT_FORCE_CALL = '(make the tool call now: <tool>{"name": "...", "args": {...}}</tool>)'
_CHAT_WRITE_ANSWER = "(write the answer now)"

#: A call whose JSON object is complete but whose closing tag has not come after this many more
#: characters was written by a model that forgot the tag: `_send_hop` stops reading there (the one
#: client-side cut left, 2026-10-08) instead of paying for a made-up continuation.
_CLOSE_GRACE_CHARS = 80


def _forgot_closing_tag(held: str) -> bool:
    """True when `held` (the text from a call's opening marker on) holds a complete JSON object
    followed by `_CLOSE_GRACE_CHARS` characters or more without a closing tag."""
    payload = prompts.find_tool_call(held)
    if payload is None:
        return False
    at = held.find(prompts.TOOL_OPEN)
    tail = held[at + len(prompts.TOOL_OPEN) + len(payload):]
    return prompts.TOOL_CLOSE not in tail and len(tail) >= _CLOSE_GRACE_CHARS

#: A call that may carry a file's content (ask with `write_file` / `edit_file`) needs more room
#: than a one-line lookup: `max_tokens` of such hops is raised to at least this.
_WRITE_HOP_MIN_TOKENS = 1200
_WRITE_TOOLS = frozenset({"write_file", "edit_file"})

#: Argument names that best say what a call is about, in order (the event's one-line summary).
_SUMMARY_KEYS = ("query", "pattern", "command", "url", "path", "file", "name")


class _NoAnswerError(Exception):
    """The tool loop ended without an answer: the model called tools and then wrote nothing, even
    when asked once more. Reported as an `AssistErrorEvent` (not as an LLM failure)."""


class _Outcome(NamedTuple):
    """What the loop needs of a runner's result (`exocortex.capabilities.ToolResult`, or any
    object with the same four attributes)."""

    text: str
    state: str
    detail: str
    summary: str


def _outcome(result: Any) -> _Outcome:
    """`result` as an `_Outcome`: missing attributes read as empty, an unknown state as "done"."""
    state = str(getattr(result, "state", "") or "done")
    if state not in ("done", "error", "denied"):
        state = "done"
    return _Outcome(
        str(getattr(result, "text", "") or ""), state,
        str(getattr(result, "detail", "") or ""), str(getattr(result, "summary", "") or ""),
    )


def _call_summary(args: dict[str, Any]) -> str:
    """One line (<= 120 chars) naming what a call is about, from its arguments: the query, the
    pattern, the command, the url or the path, else the first scalar value."""
    text = ""
    for key in _SUMMARY_KEYS:
        value = args.get(key)
        if isinstance(value, (str, int, float)) and not isinstance(value, bool) and str(value).strip():
            text = str(value)
            break
    else:
        for value in args.values():
            if isinstance(value, (str, int, float)) and not isinstance(value, bool) and str(value).strip():
                text = str(value)
                break
    return " ".join(text.split())[:120]


def _one_line(text: str, limit: int = 160) -> str:
    """The first non-empty line of `text`, cut to `limit` characters."""
    for line in (text or "").splitlines():
        if line.strip():
            return line.strip()[:limit]
    return ""


@dataclass
class _JobTools:
    """Everything the tool loop needs of ONE job, resolved once (`Assistant._job_tools`): the
    runner's per-answer budget (its `calls_left` is what the system prompt announced), the specs it
    listed, the rendered ``<tools>`` block and the project map cut to `[capabilities].map_tokens`."""

    budget: Any
    specs: list[Any]
    block: str
    map_text: str


@dataclass
class _ToolStep:
    """One call of the loop as the continuation carries it: the visible text the model wrote
    before it (`text`), the call's payload, the tool's name and its result. `note` is the loop's
    own trailing line inside the result block; `sanitize` is False for the loop's own notices
    (a malformed call's, which names ``<tool>`` on purpose); `dropped` replaces the result's body
    by a placeholder when the server ran out of context."""

    text: str
    payload: str
    name: str
    result: str
    note: str = ""
    sanitize: bool = True
    dropped: bool = False

    def block(self, *, nudge: str = "") -> str:
        """The ``<result>`` block of this step; `nudge` is the answer retry's trailing line."""
        body = _RESULT_DROPPED if self.dropped else self.result
        if self.note:
            body = f"{body.rstrip()}\n{self.note}"
        if nudge:
            body = f"{body.rstrip()}\n{nudge}"
        return prompts.tool_result_block(self.name, body, sanitize=self.sanitize)

    def call_text(self) -> str:
        """What the assistant wrote: the visible lead and the call. A ChatML special token the
        model managed to write (``<|...``) is defused, as in a result: this text goes back into the
        prompt verbatim."""
        text = f"{self.text}{prompts.TOOL_OPEN}{self.payload}{prompts.TOOL_CLOSE}"
        return text.replace("<|", "<¦")


class _ToolSplitter:
    """Splits the stream of one tool-loop request into the visible answer and the held-back call
    (the shape of ``client._ThinkSplitter``: feed arbitrary chunks, tags may straddle them).

    Text goes to `emit` (the job's ``on_delta``) unless it starts a call: a ``<tool>`` that opens a
    LINE -- after indentation and an optional bullet / quote marker (``prompts.CALL_LEAD_RE``) --
    switches to holding everything from there on, in `held`. A ``<tool>`` in the middle of a line
    of prose is a call too when what follows it is a complete JSON object that spells out its
    arguments (`prompts.is_explicit_call`): the prose before it stays visible, the call is held
    and the raw JSON is never published; any other mid-line ``<tool>`` (a name-only object, ``{}``,
    plain words) is ordinary text, and one whose object is still being written waits for its
    end. A line that might still turn into a call (``<too``, ``- ``, ``1.``) waits
    for its next characters, and a code-fence line is withheld until the line after it shows
    whether a call follows (then the fence is swallowed). Decided text is not published at once
    either: it waits for a newline or `PUBLISH_CHARS` characters, so a call at the very start of the
    answer never leaks a partial ``<too`` and a tiny stream does not become a flurry of deltas.

    `visible` is every character decided visible; `held` is None while no call has started. A
    splitter made with `held=TOOL_OPEN` is already inside a call (the loop forced one open).
    """

    PUBLISH_CHARS = 120
    #: How much of an unfinished mid-line ``<tool>{...`` is waited for before it is called text.
    MID_CALL_WAIT_CHARS = 20_000

    #: A line that is (or may become) a code-fence line: ```` ``` ```` / ``~~~`` plus an info string.
    _FENCE_LINE = re.compile(r"[ \t]*(?:`{3,}|~{3,})[^`\n]*")
    _FENCE_START = re.compile(r"[ \t]*(?:`{1,2}|~{1,2})?$|[ \t]*(?:`{3,}|~{3,})[^\n]*$")

    def __init__(self, emit: Callable[[str], None], *, held: str | None = None) -> None:
        self._emit = emit
        self.visible = ""
        self.held: str | None = held
        #: True once a delta went out (a request that failed before that may be retried).
        self.published = False
        self._pending = ""        # decided visible, not yet published
        self._buf = ""            # the line being read, not decided yet
        self._fence = ""          # a fence line withheld in case a call follows it
        self._mid_line = False    # the current line is prose: pass it on, watching for a mid-line <tool>

    def feed(self, text: str) -> None:
        """Take the next chunk of the stream."""
        if not text:
            return
        if self.held is not None:
            self.held += text
            return
        self._buf += text
        self._drain(final=False)

    def finish(self) -> None:
        """The stream ended: decide whatever is still ambiguous and publish the rest."""
        if self.held is None:
            self._drain(final=True)
        self._release_fence()
        self._flush()

    # ---- internals
    def _visible(self, text: str) -> None:
        if not text:
            return
        self.visible += text
        self._pending += text
        if "\n" in self._pending or len(self._pending) >= self.PUBLISH_CHARS:
            self._flush()

    def _flush(self) -> None:
        if self._pending:
            out, self._pending = self._pending, ""
            self.published = True
            self._emit(out)

    def _release_fence(self) -> None:
        if self._fence:
            fence, self._fence = self._fence, ""
            self._visible(fence)

    def _begin_call(self, at: int) -> None:
        """A call starts at index `at` of the buffer: hold it and everything after; swallow a
        fence line that was withheld right before it."""
        self._fence = ""
        self.held = self._buf[at:]
        self._buf = ""

    @staticmethod
    def _open_prefix_len(text: str) -> int:
        """Length of the longest suffix of `text` that is a proper prefix of ``<tool>`` (0 = none)."""
        for k in range(min(len(prompts.TOOL_OPEN) - 1, len(text)), 0, -1):
            if prompts.TOOL_OPEN.startswith(text[-k:]):
                return k
        return 0

    def _judge_mid_call(self, tail: str) -> str:
        """What the text after a mid-line ``<tool>`` makes of it: "call" (a complete JSON object
        that spells out its arguments), "text" (it cannot become one, or it is complete but is not a
        call: ``{}``, a name with no arguments) or "wait" (nothing yet, an object still open, or a
        code-fence opener), the last for at most `MID_CALL_WAIT_CHARS` characters."""
        stripped = tail.lstrip()
        if not stripped:
            return "wait"
        end = prompts._json_object_end(tail)
        if end is None:
            if stripped.startswith("{") or prompts._FENCE_PREFIX_RE.fullmatch(stripped):
                return "wait" if len(tail) < self.MID_CALL_WAIT_CHARS else "text"
            return "text"
        return "call" if prompts.is_explicit_call(tail[:end]) else "text"

    def _scan_prose(self, nl: int, final: bool) -> bool:
        """Pass the prose line being read on to `_visible`, looking for a ``<tool>`` that opens a
        call in the middle of it (`nl` = index of the line's newline in the buffer, -1 for none yet).
        True = stop draining: a call began, or the next characters decide what is held back (a
        partial ``<tool>`` at the end of the buffer, or a ``<tool>`` whose object is not complete)."""
        buf = self._buf
        at = buf.find(prompts.TOOL_OPEN)
        if at >= 0 and (nl < 0 or at < nl):
            if at:
                self._visible(buf[:at])
                self._buf = buf = buf[at:]
            verdict = self._judge_mid_call(buf[len(prompts.TOOL_OPEN):])
            if verdict == "call":
                self._begin_call(0)
                return True
            if verdict == "wait" and not final:
                return True
            self._visible(prompts.TOOL_OPEN)                    # ordinary text: go on after the marker
            self._buf = buf[len(prompts.TOOL_OPEN):]
            return False
        end = len(buf) if nl < 0 else nl + 1
        keep = self._open_prefix_len(buf[:end]) if nl < 0 and not final else 0
        if end > keep:
            self._visible(buf[:end - keep])
            self._buf = buf[end - keep:]
        if nl >= 0:
            self._mid_line = False
            return False
        return keep > 0

    def _drain(self, *, final: bool) -> None:
        while self._buf and self.held is None:
            nl = self._buf.find("\n")
            if self._mid_line:
                if self._scan_prose(nl, final):
                    return
                continue
            complete = nl >= 0
            line = self._buf[:nl] if complete else self._buf
            lead = prompts.CALL_LEAD_RE.match(line).end()      # always matches (possibly empty)
            rest = line[lead:]
            if rest.startswith(prompts.TOOL_OPEN):
                self._begin_call(lead)
                return
            partial_tag = bool(rest) and prompts.TOOL_OPEN.startswith(rest)
            ambiguous = (
                partial_tag or prompts.CALL_LEAD_PARTIAL_RE.fullmatch(line) is not None
                or self._FENCE_START.fullmatch(line) is not None
            )
            if ambiguous and not complete and not final:
                return                                          # wait for the next characters
            if complete and self._FENCE_LINE.fullmatch(line):
                fence, self._buf = self._buf[:nl + 1], self._buf[nl + 1:]
                self._release_fence()
                self._fence = fence                             # withheld: a call may follow
                continue
            if final and not complete and partial_tag:
                self._begin_call(lead)                          # the stream stopped inside "<too"
                return
            # An ordinary line (or the start of one): prose. `_scan_prose` passes it on, a long one
            # flowing through, and watches it for a ``<tool>`` in the middle.
            self._release_fence()
            self._mid_line = True


class _JobSlot(Generic[_T]):
    """A one-item mailbox: putting a new job replaces whatever was queued and not yet claimed.

    This is what gives ``assist-worker``/``notes-worker`` their "at most one job running, at most
    one job queued" behaviour without a full queue: a job that is still waiting when a newer one
    arrives is simply overwritten (superseded), never run.

    ``close()`` is used instead of a ``put(None)`` sentinel: closing wakes a blocked `get()` with
    `None` (the worker loop's stop signal) *and* makes every `put()` after it a permanent no-op.
    Without that, a job submitted concurrently with `stop()` -- e.g. the notes-auto-timer thread
    firing `generate_notes()` at the exact moment `stop()` runs -- could land its `put()` after
    `stop()`'s closing put, silently resurrecting the slot: the worker would run that one extra
    job and then block in `get()` forever, since no further sentinel is ever sent (a real thread
    leak, not just a delayed shutdown). `reopen()` undoes `close()` so the same instance can be
    `start()`-ed again after `stop()`.
    """

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._has_item = False
        self._item: _T | None = None
        self._closed = False

    def put(self, item: _T) -> None:
        """Store `item`, replacing anything unclaimed, and wake one waiting `get()`.

        A no-op once `close()` has been called and `reopen()` has not yet undone it.
        """
        with self._cond:
            if self._closed:
                return
            self._item = item
            self._has_item = True
            self._cond.notify_all()

    def put_if(self, item: _T, pred: Callable[[_T], bool]) -> bool:
        """`put(item)` only when the slot is empty or `pred(queued)` is true for what it holds;
        True when stored. Atomic (a check + `put()` would let a concurrent `put()` slip in between):
        a background insights pass uses it so it never replaces a queued on-demand request."""
        with self._cond:
            if self._closed:
                return False
            if self._has_item and self._item is not None and not pred(self._item):
                return False
            self._item = item
            self._has_item = True
            self._cond.notify_all()
            return True

    def get(self) -> _T | None:
        """Block until an item is available or the slot is closed, then return and clear it.

        Returns `None` when the slot is (or becomes) closed with nothing queued -- the worker
        loop's cue to stop.
        """
        with self._cond:
            while not self._has_item and not self._closed:
                self._cond.wait()
            item, self._item, self._has_item = self._item, None, False
            return item

    def try_get(self) -> _T | None:
        """Return and clear the queued item without blocking; `None` if there isn't one.

        Unlike `get()`, a closed-and-empty slot is indistinguishable from an open-and-empty one
        here (both return `None`) -- callers that need to detect "closed" use `closed` instead.
        """
        with self._cond:
            if not self._has_item:
                return None
            item, self._item, self._has_item = self._item, None, False
            return item

    def remove_if(self, pred: Callable[[_T], bool]) -> bool:
        """Atomically drop the queued item if `pred(item)` is true; True when something was dropped.
        (A `try_get()` + `put()`-back would let a concurrent `put()` be overwritten.)"""
        with self._cond:
            if self._has_item and self._item is not None and pred(self._item):
                self._item = None
                self._has_item = False
                return True
            return False

    def wait(self, timeout: float | None = None) -> None:
        """Block until an item is queued or the slot is closed, or `timeout` seconds elapse.

        Used by a dispatcher that needs to watch this slot without necessarily claiming what
        wakes it (see `Assistant._next_assist_job`), so this never consumes the item itself.
        """
        with self._cond:
            if timeout is None:
                while not self._has_item and not self._closed:
                    self._cond.wait()
                return
            deadline = time.monotonic() + timeout
            while not self._has_item and not self._closed:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return
                self._cond.wait(remaining)

    @property
    def closed(self) -> bool:
        """Whether `close()` has been called (and `reopen()` has not since undone it)."""
        with self._cond:
            return self._closed

    def clear(self) -> None:
        """Drop whatever is queued, without waking a waiting `get()`."""
        with self._cond:
            self._item = None
            self._has_item = False

    def close(self) -> None:
        """Wake a blocked `get()` with `None` and reject every `put()` from now on."""
        with self._cond:
            self._closed = True
            self._item = None
            self._has_item = False
            self._cond.notify_all()

    def reopen(self) -> None:
        """Undo `close()` so the slot can be used again after a stop()+start() cycle."""
        with self._cond:
            self._closed = False
            self._item = None
            self._has_item = False


@dataclass
class _AssistJob:
    """One queued or running assist-worker job.

    `focus` and `screen_lines` are only meaningful for `kind == "suggest"`: `focus` is the raw
    (unstripped) text passed to `suggest_now()` (used to build the prompt via
    `prompts.user_suggest`, which applies its own default when it is blank); `screen_lines` is
    the set of OCR lines that triggered an auto-suggest job (`None` for a manual `suggest_now()`
    call, which instead falls back to the screen text fetched for the job itself -- see
    `Assistant._record_suggest_start`).
    """

    id: int
    kind: AssistKind
    question: str
    cancel: threading.Event
    focus: str = ""
    screen_lines: tuple[str, ...] | None = None
    #: kind "complete" only: the text before/after the caret and a description of the field.
    prefix: str = ""
    suffix: str = ""
    field: str = ""
    #: kind "explain" only: the plan this explanation is for (None = whatever plan is current).
    plan_id: int | None = None
    #: kind "insights" only: the sections this job refreshes (`prompts.INSIGHT_SECTIONS` subset),
    #: whether it is a background pass (`auto`: preempted by any other job, retried later) rather
    #: than an on-demand request, and when it started running (session clock).
    sections: tuple[str, ...] = ()
    auto: bool = False
    started: float = 0.0
    #: 2026-10-29, kind "code" only: the manual-advance "Code from scratch" gesture -- the prompt
    #: tells the model to ignore the code already on screen, whatever `[assist].consider_given_code`
    #: says (`Assistant._consider_given_code`).
    fresh: bool = False
    #: 2026-10-07 (capabilities): the tool loop's state for this job, resolved ONCE by
    #: `Assistant._job_tools` (None = no tools for it) so the system prompt, the user turn, the
    #: prompt rebuilt by the overflow retry and the loop itself all agree on one budget.
    tool_ctx: "_JobTools | None" = None
    tool_ctx_resolved: bool = False
    #: Calls handed to the runner so far: once any ran, the outer overflow retry stays off (a tool
    #: must never execute twice for one answer).
    tools_used: int = 0


@dataclass
class _NotesJob:
    """One queued or running notes-worker job."""

    id: int
    force: bool


@dataclass
class _CompactJob:
    """One queued compaction (2026-10-06): fold the text that fell out of `scope`'s transcript
    window into that scope's running summary. `scope` is the checkpoint boundary the window was
    cut for (None = the whole meeting, a perfectly valid scope). `attempts` counts the times this
    scope's compaction yielded to a live answer and was re-queued (see `_lane_cancel`)."""

    id: int
    scope: int | None
    attempts: int = 0
    #: A user-driven ``compact()`` (not an overflow): its summary is installed ACTIVELY the moment
    #: it finishes (one prefix re-evaluation), because the forced drop already moved the anchor, so
    #: there is no imminent natural anchor move to fold a merely *staged* summary into.
    force: bool = False


@dataclass
class _DigestJob:
    """One queued document digest (2026-10-06): the catalogue entry for a reference document.
    `attempts` as for `_CompactJob`."""

    id: int
    doc_id: str
    attempts: int = 0


@dataclass
class _Prepared:
    """Everything one assist attempt needs: the prompt, its budget, and (for every kind but
    "explain", whose prompt carries no transcript) the context bundle it was assembled from."""

    system: str
    user: str
    max_tokens: int
    assistant_prefix: str | None
    temperature: float | None
    bundle: ContextBundle | None


@dataclass
class _Plan:
    """The current explanations-mode plan: a "solve" answer (streaming or finished) and what has
    happened to it since. `screen_lines` are the normalised OCR lines it was made from (see
    `PLAN_KEEP_FRACTION`); `text` is re-parsed lazily by `Assistant.plan()`."""

    id: int
    screen_lines: set[str]
    created: float
    text: str = ""
    done: bool = False
    explaining: bool = False
    explained: bool = False
    explanation: str = ""


class Assistant:
    """Runs auto/manual question answering and running meeting notes on its own threads."""

    def __init__(
        self,
        cfg: Config,
        bus: EventBus,
        client: LMStudioClient,
        transcript: Transcript,
        clock: Callable[[], float],
        context: ContextManager | None = None,
        tools: Any = None,
    ) -> None:
        self._cfg = cfg
        self._bus = bus
        self._client = client
        self._transcript = transcript
        self._clock = clock
        #: 2026-10-07 (capabilities): the `exocortex.capabilities.ToolRunner` (duck-typed: `wants`,
        #: `specs`, `budget`, `run`, `map_text`). `None` -> no job ever gets tools, and every prompt
        #: and request is exactly what it was before the capabilities existed.
        self._tools = tools
        #: 2026-10-06: the context library + dynamic window. `None` -> a private, in-memory
        #: manager (no documents, the anchored window only), so every pre-existing caller and test
        #: that builds an `Assistant` with the old five arguments keeps working.
        self._context: ContextManager = (
            context if context is not None else ContextManager(cfg, ContextStore(None), transcript)
        )
        #: Bumped by `reset()`: a compaction that was already running when the session was reset
        #: must not write its summary into the NEW session (see `_run_compact_job`).
        self._session_epoch = 0
        self._epoch_lock = threading.Lock()
        #: The document id a digest job is running for (`_kick_digests` skips it).
        self._digest_running: str | None = None
        self._digest_lock = threading.Lock()

        self._job_ids = itertools.count(1)
        self._assist_slot: _JobSlot[_AssistJob] = _JobSlot()      # main lane: auto/manual/ask/screen
        self._suggest_slot: _JobSlot[_AssistJob] = _JobSlot()     # side lane: suggest/solve/code, never cancels the main lane
        self._text_slot: _JobSlot[_AssistJob] = _JobSlot()        # text lane: the Text button's walkthrough (see _CANCELS)
        self._recap_slot: _JobSlot[_AssistJob] = _JobSlot()       # recap lane: the Recap action ("where are we")
        self._explain_slot: _JobSlot[_AssistJob] = _JobSlot()     # explain lane: explanation of the finished solution
        self._insight_slot: _JobSlot[_AssistJob] = _JobSlot()     # insight lane: questions to ask / blind spots
        self._dashboard_slot: _JobSlot[_AssistJob] = _JobSlot()   # dashboard lane: a refresh of the tracked board (exocortex.dashboard)
        self._complete_slot: _JobSlot[_AssistJob] = _JobSlot()    # completion lane: inline completions (lowest priority)
        self._notes_slot: _JobSlot[_NotesJob] = _JobSlot()
        #: Notes lane, below notes: the compaction of what fell out of the transcript window and
        #: the digest of an uploaded document. One queued job each, latest wins -- what a replaced
        #: job would have done is never lost (the window keeps the dropped text pending; an
        #: undigested document stays "pending" in the store and `_kick_digests` finds it again).
        self._compact_slot: _JobSlot[_CompactJob] = _JobSlot()
        self._digest_slot: _JobSlot[_DigestJob] = _JobSlot()

        self._current_job: _AssistJob | None = None
        self._current_lock = threading.Lock()

        #: Explanations mode: the current plan (see `_Plan`) and its parsed-text cache.
        self._plan: _Plan | None = None
        self._plan_parsed: tuple[int, int, Any] | None = None     # (plan id, text length, PlanParts)
        self._plan_lock = threading.Lock()

        self._answers: list[tuple[float, str, str]] = []
        self._answers_lock = threading.Lock()

        #: Checkpoint boundary (see `checkpoint()`): the transcript segment id and `_answers`
        #: index it was set at (`None`/0 = no checkpoint yet, i.e. solve/code/complete see the
        #: whole session), and the solve/code answers produced since then -- the <previous> a
        #: later "code" job in the same checkpoint builds on (`_previous_code_text`), capped at 10.
        self._checkpoint_seg_id: int | None = None
        self._checkpoint_answers: int = 0
        self._checkpoint_time: float | None = None
        self._code_answers: list[tuple[float, str, str]] = []
        self._checkpoint_lock = threading.Lock()

        self._suggestions: list[tuple[float, str]] = []
        self._suggestions_lock = threading.Lock()

        # ---- screen reading integration ----
        self._screen_provider: Callable[[], str] | None = None
        self._last_screen_text = ""
        self._screen_text_lock = threading.Lock()

        #: Auto-suggest throttle state, updated when a suggest job STARTS (see
        #: `_record_suggest_start`), not merely when one is submitted/queued.
        self._last_suggest_time = float("-inf")
        self._last_suggested_lines: set[str] = set()
        self._suggest_state_lock = threading.Lock()

        #: Screen-question state: the OCR lines last seen (to compute `new_lines`), a bounded
        #: FIFO of normalised questions already answered, and the last time one was answered.
        self._screen_prev_lines: tuple[str, ...] = ()
        self._screen_answered_questions: list[str] = []
        self._last_screen_answer_time = float("-inf")
        self._screen_question_lock = threading.Lock()

        self._notes_markdown = ""
        self._notes_last_id: int | None = None
        self._notes_lock = threading.Lock()

        #: Co-pilot insights: the living document (section -> markdown bullets, see
        #: `prompts.parse_insights`) and the background-pass bookkeeping -- when the last pass
        #: started, and the transcript segment / screen lines the last COMPLETED pass covered.
        self._insights: dict[str, str] = {}
        self._insights_last_start = float("-inf")
        self._insights_seg_id: int | None = None
        self._insights_screen_lines: set[str] = set()
        self._insights_lock = threading.Lock()

        #: Dashboard mode (exocortex.dashboard, 2026-10-20): the DashboardService's hooks that
        #: BUILD a "dashboard" job's prompt from its store and SETTLE the result (None = no
        #: dashboard wired: `dashboard_now` refuses, and the Notes-tab insights passes run as
        #: before). And the project context (exocortex.project.provider) that adds the
        #: installed toolkit's catalog and repository lookups to the other kinds' prompts.
        self._dashboard_build: Callable[..., tuple[str, str, int]] | None = None
        self._dashboard_settle: Callable[..., None] | None = None
        self._project: Any = None

        self._auto_enabled = cfg.assist.auto_answer_questions
        #: Global pause for all *automatic* assist activity (auto-answer, screen questions,
        #: auto-suggest/solve). Read on every trigger, never cached. Manual answer_last()/ask()
        #: are intentionally not gated by it.
        self._paused = False
        #: Completion mode: automatic answers/solves are suppressed (they would compete with the
        #: inline completions for the model and have no panel to show in); manual ones still work.
        self._auto_suppressed = False
        #: Hands-free mode: its read -> checkpoint -> text -> code cycle replaces the automatic
        #: solve / on-screen question answer on a screen change (both are gated off by it);
        #: spoken-question auto-answer and everything manual keep working. `_hands_free_context`
        #: is the controller's cumulative screen text (what scrolled out of view + the current
        #: screen), handed to "text" / "code" jobs instead of the plain screen while it is on.
        self._hands_free = False
        self._hands_free_context: Callable[[], str] | None = None
        # Manual advance (`[handsfree].manual`) is read live through `_manual_advance()` on every
        # automatic trigger, like the hands-free thresholds: no set_* call, no cached copy.
        self._last_fired_auto_seg_id: int | None = None
        self._coalesce_timer: threading.Timer | None = None
        self._coalesce_lock = threading.Lock()
        #: Bumped every time the coalesce timer is (re)armed or the slot is closed; a fired
        #: timer callback that no longer matches this value is stale and no-ops (see _fire_auto).
        self._coalesce_generation = 0

        self._cached_models: list[str] = []
        self._models_lock = threading.Lock()

        self._sub_queue: "queue.Queue[Any] | None" = None
        self._stop_event = threading.Event()
        #: 2026-10-06: set while an assist job runs (and by stop()); a compaction / digest streams
        #: with it as its cancel token so a live answer never shares the GPU with background work
        #: -- measured: a compaction running beside an answer took its first token from 0.3 s to
        #: 4.9 s. The yielded job hands its text back and re-queues itself; after
        #: `_LANE_YIELD_ATTEMPTS` yields it runs to completion regardless (completion mode fires a
        #: job every second or two, so an unbounded yield would starve the lane).
        self._lane_yield = threading.Event()
        self._threads: list[threading.Thread] = []

    # ---- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        """Start all worker threads and run an initial LM Studio health check."""
        if self._threads:
            log.warning("assistant: start() called while already running; stopping first")
            self.stop()
        self._stop_event.clear()
        self._lane_yield.clear()
        for slot in self._assist_slots():
            slot.reopen()
        self._notes_slot.reopen()
        self._compact_slot.reopen()
        self._digest_slot.reopen()
        with self._insights_lock:
            # The first background insights pass waits one full interval: a few opening sentences
            # are not worth a model call, and on-demand insights work from the first second.
            self._insights_last_start = self._clock()
        self._sub_queue = self._bus.subscribe()
        self._threads = [
            threading.Thread(target=self._assist_worker_loop, name="assist-worker", daemon=True),
            threading.Thread(target=self._notes_worker_loop, name="notes-worker", daemon=True),
            threading.Thread(target=self._bus_listener_loop, name="bus-listener", daemon=True),
            threading.Thread(target=self._notes_auto_loop, name="notes-auto-timer", daemon=True),
            threading.Thread(target=self._insight_timer_loop, name="insight-timer", daemon=True),
        ]
        for t in self._threads:
            t.start()
        self._run_health_check()
        log.info("assistant: started")

    def stop(self) -> None:
        """Cancel any running job, stop every thread and timer, and unsubscribe from the bus."""
        self.cancel()
        self._stop_event.set()
        self._lane_yield.set()          # aborts a streaming compaction / digest too
        with self._coalesce_lock:
            # Bump the generation first so a coalesce callback that is already past its own
            # generation check (the threading.Timer.cancel() race -- see _fire_auto) still
            # can't submit a new auto job after this point.
            self._coalesce_generation += 1
            timer, self._coalesce_timer = self._coalesce_timer, None
        if timer is not None:
            timer.cancel()
            timer.join(timeout=1.0)
            if timer.is_alive():
                log.warning("assistant: assist-coalesce timer did not stop within timeout")
        for slot in self._assist_slots():
            slot.close()
        self._notes_slot.close()
        self._compact_slot.close()
        self._digest_slot.close()
        for t in self._threads:
            t.join(timeout=5.0)
            if t.is_alive():
                log.warning("assistant: thread %s did not stop within timeout", t.name)
        self._threads = []
        if self._sub_queue is not None:
            self._bus.unsubscribe(self._sub_queue)
            self._sub_queue = None
        log.info("assistant: stopped")

    # ---- public actions ------------------------------------------------------

    def answer_last(self) -> None:
        """Answer the most recent SYSTEM question(s) right now (kind="manual")."""
        segs = self._transcript.last(Source.SYSTEM, _MAX_CONTEXT_SEGMENTS)
        if not segs:
            log.info("assistant: answer_last() called with no SYSTEM segments in the transcript yet")
            return
        self._submit_assist("manual", self._build_question(segs))

    def ask(self, text: str) -> None:
        """Answer a question the user typed directly (kind="ask", larger token budget)."""
        self._submit_assist("ask", text)

    def generate_notes(self, force: bool = True) -> None:
        """Enqueue a notes regeneration job. With `force=False` the worker may skip it (too
        little new speech since the last pass); `force=True` always regenerates."""
        self._notes_slot.put(_NotesJob(id=next(self._job_ids), force=force))

    @property
    def context(self) -> ContextManager:
        """The context manager this assistant builds its prompts with (2026-10-06): the document
        store, the anchored transcript window and the token estimator."""
        return self._context

    def digest_document(self, doc_id: str) -> None:
        """Queue the model-written digest of a reference document (2026-10-06).

        Runs on the notes lane (:meth:`_next_notes_job`), never on the answer lane, one document at
        a time: the slot holds one queued job and a newer call replaces it, but a replaced
        document is still "pending" in the store, so when the running digest finishes the worker
        queues the next pending one itself (`_kick_digests`) -- adding five files at once digests
        all five, in turn. The job stores the digest with `ContextStore.set_digest` and publishes
        `ContextLibraryEvent(reason="digest", doc_id=...)`. A no-op for an unknown document.
        """
        self._digest_slot.put(_DigestJob(id=next(self._job_ids), doc_id=doc_id))

    def cancel(self) -> None:
        """Cancel the running assist job (if any) and drop any queued assist job."""
        with self._current_lock:
            if self._current_job is not None:
                self._current_job.cancel.set()
        self._assist_slot.clear()

    def set_auto(self, enabled: bool) -> None:
        """Enable/disable auto-answering of detected SYSTEM questions."""
        self._auto_enabled = enabled
        self._bus.publish(AutoAssistChangedEvent(enabled=enabled))
        log.info("assistant: auto-answer %s", "enabled" if enabled else "disabled")

    def set_model(self, model_id: str) -> None:
        """Switch the configured/active model and re-run the health check."""
        self._cfg.llm.model = model_id
        self._client.cfg.model = model_id
        log.info("assistant: model set to %r", model_id)
        self._run_health_check()

    def list_models(self) -> list[str]:
        """Model ids from the most recent health check's background refresh."""
        with self._models_lock:
            return list(self._cached_models)

    @property
    def auto_enabled(self) -> bool:
        """Whether auto-answering of detected SYSTEM questions is currently on."""
        return self._auto_enabled

    @property
    def notes_markdown(self) -> str:
        """The current running notes document (markdown), empty until the first pass succeeds."""
        with self._notes_lock:
            return self._notes_markdown

    @property
    def answers(self) -> list[tuple[float, str, str]]:
        """(t, question, answer) history of every successful, non-cancelled answer, for export."""
        with self._answers_lock:
            return list(self._answers)

    @property
    def suggestions(self) -> list[tuple[float, str]]:
        """(t, text) history of every successful, non-cancelled suggestion, capped at 50.

        Suggestions are not answers to a question, so they never appear in `answers`.
        """
        with self._suggestions_lock:
            return list(self._suggestions)

    @property
    def last_screen_text(self) -> str:
        """The (possibly truncated) screen text used by the most recently STARTED job.

        Empty when no job has started yet, or the most recent one used none (no screen provider
        set, or `cfg.screen.use_in_answers` was off for an "auto"/"manual"/"ask" job).
        """
        with self._screen_text_lock:
            return self._last_screen_text

    def set_screen_provider(self, fn: Callable[[], str] | None) -> None:
        """Register (or, with `None`, clear) the function used to fetch the latest screen text.

        Called on the assist-worker thread once per job (see `_screen_text_for_job`), never from
        here -- registering a provider does not itself trigger a call.
        """
        self._screen_provider = fn

    def suggest_now(self, focus: str = "") -> None:
        """Generate proactive suggestions from the current screen text + recent transcript now.

        Kind "suggest". Never cancels a running auto/manual/ask/screen job: if one is running,
        this job waits in its own single slot (a newer `suggest_now()` call before it runs
        replaces it, latest wins) and runs once the main lane goes idle. A new auto/manual/ask/
        screen job, in turn, cancels a running suggest job (see `_submit_assist`).
        """
        self._enqueue_side_job("suggest", focus=focus, screen_lines=None)

    def solve(self, focus: str = "") -> None:
        """Autonomously SOLVE whatever task/problem is currently on screen (kind "solve").

        Runs on the same non-cancelling side lane as `suggest`, so it never stomps a spoken
        question's answer, but its result is shown in the main Assist panel (not the Screen tab)
        and, unlike a suggestion, is stored as an answer. `focus` carries a direction the user
        picked from a `[[CHOICE]]` prompt (blank = "solve whatever is shown"). Refused while
        paused.
        """
        if self._paused:
            return
        self._enqueue_side_job("solve", focus=focus, screen_lines=None)

    def code(self, focus: str = "", label: str = "", fresh: bool = False) -> int:
        """Write the code for the task discussed (transcript since the last checkpoint) and/or
        shown on screen, right now (kind "code", the Code button). Returns the job id.
        `label`, when given, is what the Assist tab / history show as the question instead of
        `focus` (hands-free mode passes a long instruction as `focus`). `fresh` (2026-10-29, the
        "Code from scratch" gesture) makes the prompt ignore the code already written on screen,
        whatever `[assist].consider_given_code` says.

        Runs on the same side lane as `suggest`/`solve` -- it never cancels a running main-lane
        (auto/manual/ask/screen) job -- but a new "suggest"/"solve"/"code" job cancels a running
        "code" and vice versa (`_CANCELS`). Unlike `solve()`, this is MANUAL: it is NOT refused
        while paused (see `exocortex.actions.EngineActions.code`). `focus` is an optional extra
        instruction; blank falls back to `prompts.user_code`'s default ask.
        """
        return self._enqueue_side_job("code", focus=focus, screen_lines=None, label=label, fresh=fresh)

    def text(self, focus: str = "", label: str = "") -> int:
        """Walk through, in plain English, the task discussed (transcript since the last checkpoint)
        and/or shown on screen, right now (kind "text", the Text button): what is asked, the key
        insight, the approach step by step, the edge cases and the complexity -- or, for a
        non-coding question, the reasoning to the answer. Returns the job id.

        The prose twin of `code()`: never cancels a running main-lane job, same checkpoint-scoped
        transcript and <previous> code block, MANUAL like "code" (not refused while paused). Unlike
        "code" it runs from its own lane (`_text_slot`, dispatched after the side lane), cancels
        only an older "text" job and is cancelled by no other kind (`_CANCELS`): a walkthrough
        waits behind a streaming solve rather than dropping the plan, a running walkthrough is
        never killed by an automatic re-solve, and a queued one is never overwritten by one.
        Stored as an answer, never as code (`_code_answers`), never the plan, and not recorded in
        the auto-solve throttle. `focus` is an optional extra instruction; blank falls back to
        `prompts.user_text`'s default ask; `label` as for `code()`.
        """
        return self._enqueue_side_job("text", focus=focus, screen_lines=None, label=label)

    def recap(self, focus: str = "") -> int:
        """Recap the session right now (kind "recap", the Recap action): the goal, what is done,
        what is being worked on now, what is next -- from the running notes (the whole session),
        the recent transcript and the screen. Returns the job id.

        MANUAL like "code" / "text": not refused while paused. Runs from its own lane (right after
        the text lane), cancels only an older "recap" and never a running main-lane / solve / code
        / walkthrough job; a running background insights pass yields to it. Stored as an answer
        (Assist tab, history, export). `focus` is an optional extra instruction."""
        return self._enqueue_side_job("recap", focus=focus, screen_lines=None)

    def insights_now(self, sections: Sequence[str] | None = None) -> int:
        """Refresh the co-pilot insights right now (kind "insights", on demand). `sections` is a
        subset of `prompts.INSIGHT_SECTIONS` ("questions", "blindspots"); None / empty = both.
        Returns the job id.

        MANUAL: works while paused, in manual advance and in completion mode, whatever the
        `[assist]` toggles say (those only govern the background passes). Runs on the insight lane:
        it supersedes a queued or running insights job (latest wins) but never cancels any other
        job; the main lane still preempts it like everything else. The result updates the living
        document (`insights`) and is published as an `InsightsUpdatedEvent`."""
        secs = prompts.normalize_sections(sections or prompts.INSIGHT_SECTIONS) or prompts.INSIGHT_SECTIONS
        job_id = self._enqueue_insights(secs, auto=False)
        return job_id if job_id is not None else 0

    @property
    def insights(self) -> dict[str, str]:
        """The co-pilot insights document: section ("questions" / "blindspots") -> its markdown
        bullet list ("" = no open items). Empty until the first pass completed."""
        with self._insights_lock:
            return dict(self._insights)

    @property
    def insights_markdown(self) -> str:
        """The insights document as markdown for the session folder (insights.md): every section's
        heading, "- (none)" for an empty one; "" until the first pass completed."""
        with self._insights_lock:
            current = dict(self._insights)
        return prompts.render_insights(current, skip_empty=False) if current else ""

    def replace_insights(self, sections: dict[str, str], *, auto: bool = False) -> bool:
        """Set the insights document from outside -- the dashboard mirrors its open questions and
        blind spots here while it tracks them (`exocortex.dashboard.service`), so the Notes tab's
        Insights pane and the session's insights.md keep working. Publishes an
        `InsightsUpdatedEvent` when a section actually changed; returns whether one did."""
        changed = False
        with self._insights_lock:
            for section in prompts.INSIGHT_SECTIONS:
                if section not in sections:
                    continue
                body = (sections.get(section) or "").strip()
                if self._insights.get(section, "") != body:
                    self._insights[section] = body
                    changed = True
            snapshot = dict(self._insights)
        if changed:
            self._bus.publish(InsightsUpdatedEvent(
                questions=snapshot.get("questions", ""), blindspots=snapshot.get("blindspots", ""),
                generated_at=self._clock(), auto=auto,
            ))
        return changed

    # ---- the dashboard lane (exocortex.dashboard, 2026-10-20) --------------------------------

    @property
    def is_running(self) -> bool:
        """Whether `start()` has run (and `stop()` has not): the workers drain the lanes."""
        return bool(self._threads)

    def set_dashboard_hooks(
        self,
        build: Callable[..., tuple[str, str, int]] | None,
        settle: Callable[..., None] | None,
    ) -> None:
        """Wire the dashboard service: `build(job, screen_text, until_id)` returns the (system,
        user, max_tokens) of a "dashboard" job; `settle(job, ok=..., text=..., until_id=...,
        screen_text=...)` is told how it ended (`ok` False: cancelled / failed, nothing to apply).
        Both run on the assist-worker thread."""
        self._dashboard_build = build
        self._dashboard_settle = settle

    def dashboard_tracking(self) -> bool:
        """Whether the dashboard's background refreshes replace the insights passes: a service
        is wired and `[dashboard].track` is on (read live)."""
        dashboard = getattr(self._cfg, "dashboard", None)
        return self._dashboard_build is not None and bool(getattr(dashboard, "track", False))

    def dashboard_now(self, *, auto: bool = False, sections: Sequence[str] = ()) -> int | None:
        """Queue one dashboard refresh (kind "dashboard") on its own lane; the job id, or None
        when no dashboard is wired or a background pass found an on-demand request queued.

        On demand (`auto=False`): supersedes a running or queued dashboard job (latest wins) but
        never cancels any other job; works while paused. Background (`auto=True`): only takes an
        empty slot or one holding another background refresh, and is cancelled by any other job
        submitted while it runs (`_preempt_auto_insights`); the service retries it later."""
        if self._dashboard_build is None:
            return None
        secs = tuple(str(s) for s in (sections or ()) if str(s).strip())
        label = "Dashboard refresh" + (f": {', '.join(secs)}" if secs else "")
        job = _AssistJob(id=next(self._job_ids), kind="dashboard", question=label, cancel=threading.Event(),
                         sections=secs, auto=auto)
        if auto:
            if not self._dashboard_slot.put_if(job, lambda queued: queued.auto):
                return None
        else:
            self._cancel_running_of(_CANCELS["dashboard"])
            self._dashboard_slot.put(job)
        log.debug("assistant: submitted %s dashboard job %d: %s", "auto" if auto else "manual", job.id, secs or "all")
        return job.id

    def _dashboard_prompt(self, job: _AssistJob, screen_text: str, until_id: int | None) -> tuple[str, str, int]:
        build = self._dashboard_build
        if build is None:
            raise LLMError("the dashboard is not wired (no DashboardService)")
        return build(job, screen_text, until_id)

    def _settle_dashboard(
        self, job: _AssistJob, *, ok: bool, text: str = "", until_id: int | None = None, screen_text: str = "",
    ) -> None:
        """Tell the dashboard service how a "dashboard" job ended (a no-op for every other kind)."""
        if job.kind != "dashboard":
            return
        settle = self._dashboard_settle
        if settle is None:
            return
        try:
            settle(job, ok=ok, text=text, until_id=until_id, screen_text=screen_text)
        except Exception:  # noqa: BLE001 - the service's own bug must never kill the worker
            log.exception("assistant: dashboard settle hook failed for job %d", job.id)

    # ---- on-demand context (exocortex.project, 2026-10-20) ------------------------------------

    def set_project_context(self, provider: Any) -> None:
        """Register (or with None, clear) the `exocortex.project.provider.ProjectContext` whose
        `toolkit_block()` joins the system prompt of answers and `repo_block()` /
        `toolkit_lookup()` the user turn (`_TOOLKIT_KINDS` / `_LOOKUP_KINDS`). Called on the
        assist-worker thread once per job, never from here."""
        self._project = provider

    def _toolkit_system_suffix(self, kind: str) -> str:
        """The <toolkit> catalog (with its standing rule) to append to a system prompt, or ""."""
        provider = self._project
        project_cfg = getattr(self._cfg, "project", None)
        if provider is None or kind not in _TOOLKIT_KINDS or not getattr(project_cfg, "toolkit_in_answers", False):
            return ""
        try:
            block = provider.toolkit_block() or ""
        except Exception:  # noqa: BLE001 - context is optional, never a reason to fail an answer
            log.exception("assistant: toolkit_block() failed")
            return ""
        return f"\n{prompts.TOOLKIT_RULE}\n{block}" if block.strip() else ""

    def _project_extra(self, job: _AssistJob, window: str, screen_text: str) -> str:
        """The on-demand blocks for a user turn: lines from the repository matching the names
        the question / recent transcript / screen mention, and the full text of a toolkit part
        that was mentioned. "" when there is no provider, nothing to look up or nothing found."""
        provider = self._project
        if provider is None or job.kind not in _LOOKUP_KINDS:
            return ""
        tail = window[-_LOOKUP_TAIL_CHARS:] if window else ""
        project_cfg = getattr(self._cfg, "project", None)
        try:
            repo = provider.repo_block(job.question, job.focus, tail, screen_text, time_budget_s=_LOOKUP_BUDGET_S) or ""
            detail = ""
            if getattr(project_cfg, "toolkit_in_answers", False):
                detail = provider.toolkit_lookup(job.question, job.focus, tail) or ""
        except Exception:  # noqa: BLE001
            log.exception("assistant: project lookup failed for job %d", job.id)
            return ""
        return "\n\n".join(part for part in (repo.strip(), detail.strip()) if part)

    def complete(self, prefix: str, suffix: str, field: str) -> int:
        """Queue an inline completion (kind "complete") for text ending at the caret; returns the
        job id. Runs on the completion lane (lowest priority: it never delays, cancels or replaces
        a queued solve -- the plan the completions may be waiting for), replacing any queued or
        running completion (latest wins); results are ephemeral (never stored as answers)."""
        job = _AssistJob(
            id=next(self._job_ids), kind="complete", question="Completion", cancel=threading.Event(),
            prefix=prefix, suffix=suffix, field=field,
        )
        self._cancel_running_of(_CANCELS["complete"])
        self._preempt_auto_insights()
        self._complete_slot.put(job)
        log.debug("assistant: submitted complete job %d (%d chars before caret)", job.id, len(prefix))
        return job.id

    def cancel_job(self, job_id: int) -> None:
        """Cancel one specific job: the running one if it matches, or drop it from a slot."""
        with self._current_lock:
            if self._current_job is not None and self._current_job.id == job_id:
                self._current_job.cancel.set()
                return
        for slot in self._assist_slots():
            if slot.remove_if(lambda queued: queued.id == job_id):
                return

    # ---- checkpoints --------------------------------------------------------------------

    def checkpoint(self) -> None:
        """Soft reset of the CODING context only (module docstring / SPEC.md): a running solve/
        code/text/explain/complete job is cancelled and any queued one in the suggest/explain/
        complete lanes is dropped (they belong to the old coding task) -- a running or queued
        main-lane (auto/manual/ask/screen) job is never touched. From this point on, `solve()`,
        `code()`, `text()` and inline completions see only transcript segments and answers produced
        AFTER this moment,
        and the current plan (if any) is dropped along with it. Everything meeting-wide -- the
        transcript store, the Live tab, notes, the answer history/export, the ask box (`ask()`
        keeps the full transcript), speaker labels, the screen-question de-dup memory, and the
        auto-suggest throttle state (`_last_suggested_lines`/`_last_suggest_time`) -- is left
        untouched: clearing the throttle state here would make the very next OCR jitter re-solve
        the OLD problem that is still on screen. Publishes a `CheckpointEvent`.
        """
        self._cancel_running_of(("suggest", "solve", "code", "text", "explain", "complete"))
        self._suggest_slot.clear()
        self._text_slot.clear()
        self._explain_slot.clear()
        self._complete_slot.clear()
        seg_id = self._transcript.last_id()
        t = self._clock()
        with self._answers_lock:
            answers = len(self._answers)
        with self._checkpoint_lock:
            old_seg_id = self._checkpoint_seg_id
            self._checkpoint_seg_id = seg_id
            self._checkpoint_answers = answers
            self._checkpoint_time = t
            self._code_answers = []
        if old_seg_id is not None and old_seg_id != seg_id:
            # The window's anchor / pending compaction of the OLD checkpoint scope will never be
            # used again (the whole-meeting scope, None, is the ask box's and stays).
            try:
                self._context.window.forget_scope(old_seg_id)
            except Exception:  # noqa: BLE001 - context bookkeeping must never break a checkpoint
                log.exception("assistant: forgetting the old checkpoint's context scope failed")
        with self._plan_lock:
            self._plan = None
            self._plan_parsed = None
        self._bus.publish(CheckpointEvent(t=t, segment_id=seg_id))
        log.info("assistant: checkpoint set (segment %s, %d answer(s) so far)", seg_id, answers)

    def compact(self) -> int:
        """Compact the context window *now* (the overlay's "Compact" control, like the
        ``/compact`` command): fold everything but the newest couple of lines of the whole-meeting
        transcript window into its running summary straight away, instead of waiting for the
        window to overflow.

        The whole-meeting scope (``None``) is used on purpose: it is what the spoken/typed/on-screen
        answers (auto, manual, ask, screen) build on, and what the Monitor's summary readout shows,
        and it is the scope that grows across a long meeting. The checkpoint-scoped coding context
        (``code`` / ``solve`` / ``text``) is already bounded to the post-checkpoint transcript, and
        Checkpoint is its reset; compacting it here would leave the broad transcript untouched,
        which is the opposite of what Compact is for.

        The drop is immediate (the next prompt is already shorter); the summary of the dropped text
        is produced by a compaction job on the notes lane (so it never competes with a live answer)
        and, because the drop already moved the anchor, that job installs its summary ACTIVELY the
        moment it finishes rather than staging it for a future anchor move -- otherwise the dropped
        speech would be missing from every prompt AND its summary withheld until the next natural
        overflow. Returns the transcript characters folded (0 = nothing to compact, or
        ``[context].compaction`` is off -- with compaction off the window is left untouched rather
        than silently discarding the transcript)."""
        if self._cfg.context.compaction != "summary":
            log.info("assistant: compact requested but [context].compaction is off; nothing folded")
            return 0
        scope = None   # the whole meeting (see the docstring)
        try:
            dropped = self._context.window.compact_now(scope)
        except Exception:  # noqa: BLE001 - context bookkeeping must never break a user action
            log.exception("assistant: forced compaction of scope %s failed", scope)
            return 0
        if dropped > 0:
            self._compact_slot.put(_CompactJob(id=next(self._job_ids), scope=scope, force=True))
            log.info("assistant: forced compaction queued (scope %s, %d chars folded)", scope, dropped)
        return dropped

    def _checkpoint_boundary(self) -> int | None:
        """Transcript segment id the current checkpoint starts after (`None` = no checkpoint yet:
        solve/code/complete then see the whole transcript, same as before checkpoints existed)."""
        with self._checkpoint_lock:
            return self._checkpoint_seg_id

    def _checkpoint_answers_index(self) -> int:
        """Index into `_answers` the current checkpoint starts after (0 = no checkpoint yet)."""
        with self._checkpoint_lock:
            return self._checkpoint_answers

    def _append_code_answer(self, job: _AssistJob, entry: tuple[float, str, str]) -> None:
        """Record `entry` (t, question, answer text) into `_code_answers`, capped at 10 -- the
        <previous> block a later "code" job in this same checkpoint builds on (see
        `_previous_code_text`). A no-op if `job` is cancelled: `_run_assist_job` reads
        `job.cancel` early (well before this call) to decide whether to record an answer at all,
        so a checkpoint landing in between -- which cancels this job AND clears `_code_answers` --
        would otherwise re-populate the very list it just emptied with a stale, pre-checkpoint
        entry. Re-checking here, under the same lock `checkpoint()` clears `_code_answers` with,
        closes that race the way `_begin_plan`/`_finish_plan` do for the plan."""
        with self._checkpoint_lock:
            if job.cancel.is_set():
                return
            self._code_answers.append(entry)
            del self._code_answers[:-10]

    def _previous_code_text(self) -> str:
        """<previous> for `prompts.user_code`: the most recent solve/code answer since the current
        checkpoint (`_code_answers`), its solution part (`exocortex.plan.split_plan`) when that
        parses out something non-empty, else the whole answer text -- capped at 2400 chars. "" when
        there is none yet (a fresh checkpoint, or nothing solved/coded since)."""
        with self._checkpoint_lock:
            recent = self._code_answers[-1] if self._code_answers else None
        if recent is None:
            return ""
        text = recent[2]
        solution = split_plan(text).solution
        return (solution if solution.strip() else text)[:2400]

    # ---- explanations (the plan) ------------------------------------------------------------

    def plan(self) -> PlanView | None:
        """A snapshot of the current plan for the completer: its (possibly partial) solution.
        `None` when explanations are off or no solve has started since the last reset."""
        if not self._cfg.explain.enabled:
            return None
        with self._plan_lock:
            plan = self._plan
            if plan is None:
                return None
            parts = self._parsed_plan_locked(plan)
            return PlanView(id=plan.id, solution=parts.solution, is_code=parts.is_code, done=plan.done)

    @property
    def plan_walkthrough(self) -> str:
        """The current plan's walkthrough text ("" when there is no plan)."""
        with self._plan_lock:
            return self._parsed_plan_locked(self._plan).walkthrough if self._plan is not None else ""

    @property
    def plan_explanation(self) -> str:
        """The explanation of the current plan's solution ("" until one has been generated)."""
        with self._plan_lock:
            return self._plan.explanation if self._plan is not None else ""

    def plan_typed(self, plan_id: int) -> None:
        """The completer saw the user type out the last line of plan `plan_id`'s code: explain it.
        A no-op for a stale, unfinished or already explained plan, or while paused / in manual
        advance (the explanation is automatic inference; `explain_now` still works)."""
        if self._paused or self._manual_advance():
            return
        self._request_explain(plan_id, "typed out in completion mode")

    def explain_now(self) -> None:
        """Explain the current plan's solution right now (manual; works even before the code has
        been typed out). With no plan, explain whatever code is on screen."""
        with self._plan_lock:
            plan = self._plan
            plan_id = plan.id if plan is not None else None
            if plan is not None:
                plan.explaining = True
                plan.explained = False
        self._enqueue_side_job("explain", focus="", screen_lines=None, plan_id=plan_id)

    def set_explanations(self, enabled: bool) -> None:
        """Explanations toggled at runtime (the config flag is read live everywhere else). Turning
        them off drops a pending or streaming explanation -- nobody is waiting for it any more."""
        if not enabled:
            self._explain_slot.clear()
            self._cancel_running_of(("explain",))
            with self._plan_lock:
                if self._plan is not None:
                    self._plan.explaining = False
        log.info("assistant: explanations %s", "on" if enabled else "off")

    def ensure_plan(self) -> None:
        """Make sure a plan exists for what is on screen (entering completion mode / turning
        explanations on): starts a solve unless the current plan still matches the screen."""
        if self._paused or not self._cfg.explain.enabled:
            return
        with self._plan_lock:
            plan = self._plan
        if plan is not None and self._plan_matches_screen(plan, self._current_screen_lines()):
            return
        self.solve("")

    def _current_screen_lines(self) -> set[str]:
        provider = self._screen_provider
        if provider is None:
            return set()
        try:
            return _normalized_line_set((provider() or "").splitlines())
        except Exception:  # noqa: BLE001
            return set()

    @staticmethod
    def _plan_matches_screen(plan: _Plan, lines: set[str]) -> bool:
        return containment(plan.screen_lines, lines) >= PLAN_KEEP_FRACTION

    def _parsed_plan_locked(self, plan: _Plan) -> Any:
        cached = self._plan_parsed
        if cached is not None and cached[0] == plan.id and cached[1] == len(plan.text):
            return cached[2]
        parts = split_plan(plan.text)
        self._plan_parsed = (plan.id, len(plan.text), parts)
        return parts

    def _request_explain(self, plan_id: int, reason: str) -> None:
        with self._plan_lock:
            plan = self._plan
            if plan is None or plan.id != plan_id or not plan.done or plan.explaining or plan.explained:
                return
            plan.explaining = True
        log.info("assistant: explaining plan %d (%s)", plan_id, reason)
        self._enqueue_side_job("explain", focus="", screen_lines=None, plan_id=plan_id)

    def _begin_plan(self, job: _AssistJob, screen_text: str) -> None:
        lines = job.screen_lines if job.screen_lines is not None else tuple(screen_text.splitlines())
        with self._plan_lock:
            if job.cancel.is_set():
                # Re-checked here, under the same lock `checkpoint()` uses to drop the plan: the
                # caller's own check (a few statements earlier, unlocked) can be stale if a
                # checkpoint cancels this job and clears `self._plan` in between -- without this,
                # this call would silently resurrect a plan for the checkpoint's now-invalid job.
                return
            self._plan = _Plan(id=job.id, screen_lines=_normalized_line_set(lines), created=self._clock())
            self._plan_parsed = None

    def _plan_delta(self, job_id: int, delta: str) -> None:
        with self._plan_lock:
            if self._plan is not None and self._plan.id == job_id:
                self._plan.text += delta

    def _finish_plan(self, job_id: int, text: str, keep: bool) -> None:
        with self._plan_lock:
            plan = self._plan
            if plan is None or plan.id != job_id:
                return
            if not keep:
                self._plan = None
                self._plan_parsed = None
                return
            plan.text = text
            plan.done = True

    def _finish_explanation(self, plan_id: int | None, text: str, keep: bool) -> None:
        with self._plan_lock:
            plan = self._plan
            if plan is None or (plan_id is not None and plan.id != plan_id):
                return                 # the plan this explanation was for has been replaced
            plan.explaining = False
            if keep:
                plan.explained = True
                plan.explanation = text

    def _maybe_explain_from_screen(self, lines: tuple[str, ...]) -> None:
        """Full-overlay counterpart of `plan_typed`: once (nearly) every code line of the finished
        plan is visible on screen -- the user typed the solution out -- explain it."""
        if self._paused or self._manual_advance() or not self._cfg.explain.enabled:
            return
        with self._plan_lock:
            plan = self._plan
            if plan is None or not plan.done or plan.explaining or plan.explained:
                return
            parts = self._parsed_plan_locked(plan)
            plan_id = plan.id
        if not parts.is_code or not parts.solution.strip():
            return
        if typed_fraction(parts.solution, lines) >= self._cfg.explain.typed_fraction:
            self._request_explain(plan_id, "solution visible on screen")

    def set_auto_suppressed(self, suppressed: bool) -> None:
        """Completion mode on/off: suppress automatic answers, screen questions and auto-solve."""
        self._auto_suppressed = suppressed
        if suppressed:
            # Completion mode: no Insights pane on screen, and the model belongs to the completer.
            self._drop_auto_insights()
        log.info("assistant: automatic lanes %s", "suppressed" if suppressed else "active")

    def _manual_advance(self) -> bool:
        """Manual advance (``[handsfree].manual``, read live): while on, NOTHING starts a model
        call by itself -- spoken-question auto-answers, on-screen question answers, automatic
        solves / suggestions, automatic notes and automatic explanations all wait for the user's
        press-and-hold gesture or a manual action. Every manual entry point (`answer_last`,
        `ask`, `code`, `text`, `suggest_now`, `explain_now`, `generate_notes`) stays ungated, and
        so do the deliberate mode activations (`ensure_plan`, the completer's own requests)."""
        return bool(self._cfg.handsfree.manual)

    def set_hands_free(self, enabled: bool, context_provider: Callable[[], str] | None = None) -> None:
        """Hands-free mode on/off: while on, a screen change never starts an automatic solve /
        suggestion or answers a new on-screen question -- the hands-free controller's own
        read -> checkpoint -> text -> code cycle handles the screen (`exocortex.handsfree`).
        `context_provider` (the controller's cumulative screen text) feeds "text" / "code" jobs
        in place of the plain screen text, with `cfg.handsfree.context_chars` of room, so what
        scrolled out of view is still there; it falls back to the screen provider when it
        returns "". Spoken-question auto-answer, explanations and every manual action are
        unaffected."""
        self._hands_free = enabled
        self._hands_free_context = context_provider if enabled else None
        log.info("assistant: hands-free %s", "on" if enabled else "off")

    def recent_code(self) -> str:
        """Every solve / code answer produced since the current checkpoint, newest last, joined
        by blank lines ("" when none). Hands-free mode matches OCR lines against it so the user
        typing the app's own solution into the editor never reads as a new task."""
        with self._checkpoint_lock:
            return "\n\n".join(entry[2] for entry in self._code_answers)

    def set_paused(self, paused: bool) -> None:
        """Pause/resume all *automatic* assist activity (auto-answer, screen questions,
        auto-suggest/solve). Pausing also cancels any job currently running. Manual actions
        (`answer_last`, `ask`) still work while paused; `solve()` is refused (it is the
        autonomous path). Idempotent."""
        self._paused = paused
        if paused:
            self.cancel()
            self._drop_auto_insights()
        log.info("assistant: %s", "paused" if paused else "resumed")

    @property
    def paused(self) -> bool:
        """Whether automatic assist activity is currently paused."""
        return self._paused

    def reset(self) -> None:
        """Clear per-session assist state: cancel any running job, drop every queued job (a
        suggest/solve/code/text/explain/complete job queued behind the running one belongs to the
        old session -- `cancel()` alone only clears the main lane's slot, so without this a queued
        walkthrough would run against the new, empty session and land in its answers), the
        answer/suggestion history, the screen-question de-dup memory, the notes document, the
        co-pilot insights (the first background pass of the new session again waits a full
        interval), the auto-answer bookkeeping, the plan, and the checkpoint state. Called by
        `Engine.new_session()` so a fresh session starts clean."""
        self.cancel()
        for slot in self._assist_slots():
            slot.clear()
        self._compact_slot.clear()
        with self._epoch_lock:
            # The bump and the window reset happen under ONE lock, and a compaction job reads the
            # epoch and claims its text under the same lock: a job either saw the old epoch with
            # the old window (and is dropped as stale when it finishes) or the new epoch with the
            # cleared window (nothing to claim) -- never the new epoch with the old session's text
            # (review, 2026-10-06).
            self._session_epoch += 1
            try:
                self._context.reset_session()   # anchors, summaries, pending compactions; documents stay
            except Exception:  # noqa: BLE001 - context bookkeeping must never break a reset
                log.exception("assistant: resetting the context window failed")
        with self._coalesce_lock:
            self._coalesce_generation += 1
            timer, self._coalesce_timer = self._coalesce_timer, None
        if timer is not None:
            timer.cancel()
        self._last_fired_auto_seg_id = None
        with self._answers_lock:
            self._answers.clear()
        with self._suggestions_lock:
            self._suggestions.clear()
        with self._screen_question_lock:
            self._screen_prev_lines = ()
            self._screen_answered_questions.clear()
            self._last_screen_answer_time = float("-inf")
        with self._suggest_state_lock:
            self._last_suggest_time = float("-inf")
            self._last_suggested_lines = set()
        with self._notes_lock:
            self._notes_markdown = ""
            self._notes_last_id = None
        with self._insights_lock:
            self._insights = {}
            self._insights_last_start = self._clock()
            self._insights_seg_id = None
            self._insights_screen_lines = set()
        with self._plan_lock:
            self._plan = None
            self._plan_parsed = None
        with self._checkpoint_lock:
            self._checkpoint_seg_id = None
            self._checkpoint_answers = 0
            self._checkpoint_time = None
            self._code_answers = []
        log.info("assistant: reset")

    # ---- assist-worker ---------------------------------------------------------

    def _submit_assist(self, kind: AssistKind, question: str, *, cancel_running: bool = True) -> None:
        """Enqueue a main-lane job, superseding whatever main-lane job was queued. By default the
        running job (either lane) is cancelled so the new one starts at once; a "screen" job
        passes `cancel_running=False` and simply waits its turn, so an on-screen question never
        throws away a solve/answer that is mid-stream."""
        job = _AssistJob(id=next(self._job_ids), kind=kind, question=question, cancel=threading.Event())
        if cancel_running:
            with self._current_lock:
                if self._current_job is not None:
                    self._current_job.cancel.set()
        else:
            self._preempt_auto_insights()
        self._assist_slot.put(job)
        log.debug("assistant: submitted %s job %d: %r", kind, job.id, question)

    def _enqueue_side_job(
        self, kind: AssistKind, *, focus: str, screen_lines: tuple[str, ...] | None,
        plan_id: int | None = None, label: str = "", fresh: bool = False,
    ) -> int:
        """Queue a side-lane ("suggest"/"solve"/"code"), text-lane ("text") or explain-lane
        ("explain") job. Cancels only a running job that the new kind supersedes (`_CANCELS`); a
        running main-lane job is never touched -- see `suggest_now`/`solve`/`code`/`text`.
        Returns the job's id (the one its Assist* events will carry)."""
        default_label = {
            "solve": "what is on screen", "code": "the task being discussed",
            "text": "the task being discussed", "explain": "Explanation",
            "recap": "Recap of the session so far",
        }.get(kind, "Suggestions")
        if fresh and kind == "code":
            default_label += " \u00b7 from scratch"
        display_question = label.strip() or focus.strip() or default_label
        job = _AssistJob(
            id=next(self._job_ids), kind=kind, question=display_question,
            cancel=threading.Event(), focus=focus, screen_lines=screen_lines, plan_id=plan_id,
            fresh=bool(fresh and kind == "code"),
        )
        self._cancel_running_of(_CANCELS.get(kind, ()))
        self._preempt_auto_insights()
        slot = {
            "explain": self._explain_slot, "text": self._text_slot, "recap": self._recap_slot,
        }.get(kind, self._suggest_slot)
        slot.put(job)
        log.debug("assistant: submitted %s job %d: focus=%r", kind, job.id, focus)
        return job.id

    def _cancel_running_of(self, kinds: tuple[str, ...]) -> None:
        """Cancel the running job if its kind is one of `kinds`."""
        with self._current_lock:
            if self._current_job is not None and self._current_job.kind in kinds:
                self._current_job.cancel.set()

    def _assist_slots(self) -> tuple["_JobSlot[_AssistJob]", ...]:
        """Every assist-worker lane, highest priority first."""
        return (
            self._assist_slot, self._suggest_slot, self._text_slot, self._recap_slot, self._explain_slot,
            self._insight_slot, self._dashboard_slot, self._complete_slot,
        )

    def _assist_worker_loop(self) -> None:
        log.info("assist-worker: started")
        while True:
            job = self._next_assist_job()
            if job is None:
                log.info("assist-worker: stopping")
                return
            with self._current_lock:
                self._current_job = job
            self._lane_yield.set()          # background lane jobs step aside for an answer
            try:
                self._run_assist_job(job)
            except Exception:  # pragma: no cover - safety net; _run_assist_job handles its own errors
                log.exception("assist-worker: unhandled error running job %d", job.id)
            finally:
                with self._current_lock:
                    if self._current_job is job:
                        self._current_job = None
                if not self._stop_event.is_set():
                    self._lane_yield.clear()

    def _next_assist_job(self) -> _AssistJob | None:
        """Block until a job is ready, taking the highest-priority lane that has one (main, then
        side, then text, then recap, then explain, then insights, then completion).

        Each lane is its own latest-wins single slot (`_JobSlot`), so this never buffers more
        than one job per lane. Polls `_assist_slot` (with a short timeout) rather than blocking
        on it forever so a lower lane's job queued while the main lane is empty still gets picked
        up promptly; `_assist_slot.put()`/`close()` wake it immediately (no polling delay) since
        those `notify_all()` the very condition this waits on.
        """
        while True:
            for slot in self._assist_slots():
                job = slot.try_get()
                if job is not None:
                    return job
            if self._assist_slot.closed:
                return None
            self._assist_slot.wait(timeout=_DISPATCH_POLL_S)

    def _run_assist_job(self, job: _AssistJob) -> None:
        """Build the prompt once, stream the answer, and publish Started/Delta/Done/Error."""
        self._bus.publish(AssistStartedEvent(job_id=job.id, kind=job.kind, question=job.question))
        if job.kind in ("insights", "dashboard"):
            job.started = self._clock()
        if job.kind == "insights":
            with self._insights_lock:
                self._insights_last_start = job.started

        screen_text = self._screen_text_for_job(job)
        if job.kind in ("suggest", "solve", "code"):
            # Not for "text": a walkthrough solves nothing, and bumping `_last_suggest_time` here
            # would make `_maybe_auto_suggest` skip the automatic solve of the NEXT screen for
            # `suggest_min_interval_s`; the same screen needs no guard (auto-solve only fires on a
            # changed ScreenTextEvent).
            self._record_suggest_start(job, screen_text)
        is_plan = job.kind in ("solve", "code") and self._cfg.explain.enabled
        if job.kind == "explain" and job.plan_id is not None:
            with self._plan_lock:
                current = self._plan.id if self._plan is not None else None
            if current != job.plan_id:
                job.cancel.set()       # the plan it was for has been replaced meanwhile: pointless now

        if job.cancel.is_set():
            # Superseded (cancel(), or a newer job submitted while this one was still being
            # claimed/prepared) before it ever started streaming: skip the LLM request entirely
            # -- there's no point spending a slot on the real (or fake) server for output nobody
            # will see -- and publish the Done every submitted job is owed.
            log.debug("assist job %d (%s) cancelled before streaming started", job.id, job.kind)
            self._settle_plan_state(job, is_plan, "", keep=False)
            self._settle_insights(job, ok=False)
            self._settle_dashboard(job, ok=False)
            self._bus.publish(
                AssistDoneEvent(job_id=job.id, text="", ttft_s=0.0, tokens=0, tok_per_s=0.0, cancelled=True)
            )
            return

        if is_plan:
            self._begin_plan(job, screen_text)   # only now: a superseded job must not wipe a good plan

        until_id = self._transcript.last_id()
        # "solve"/"code"/"text"/"complete" see only the transcript since the last checkpoint (or
        # the whole session with none set yet); every other kind (notably "ask") keeps the full window.
        since_id = self._checkpoint_boundary() if job.kind in ("solve", "code", "text", "complete") else None
        choices, choices_on_screen = self._detect_choices(job, screen_text)
        try:
            prepared = self._prepare(job, screen_text, choices, choices_on_screen, since_id, until_id)
        except Exception as exc:  # noqa: BLE001 - a prompt that cannot be built (a dashboard / project
            # hook raising) is this job's error, never a dead worker or a job without its Done
            log.error("assist job %d (%s): building the prompt failed: %s", job.id, job.kind, exc, exc_info=True)
            self._settle_plan_state(job, is_plan, "", keep=False)
            self._settle_insights(job, ok=False)
            self._settle_dashboard(job, ok=False)
            self._bus.publish(AssistErrorEvent(job_id=job.id, message=str(exc)))
            return
        self._publish_usage(job, prepared.bundle)

        start_wall = time.monotonic()
        delta_count = 0
        first_delta_wall: float | None = None

        def on_delta(text: str) -> None:
            nonlocal delta_count, first_delta_wall
            delta_count += 1
            if first_delta_wall is None:
                first_delta_wall = time.monotonic()
            if is_plan:
                self._plan_delta(job.id, text)   # the completer follows the plan while it streams
            self._bus.publish(AssistDeltaEvent(job_id=job.id, delta=text))

        retried = False
        while True:
            try:
                text, sent_chars, stats = self._stream_answer(job, prepared, on_delta)
                break
            except LLMError as exc:
                if (
                    not retried and prepared.bundle is not None and delta_count == 0
                    and job.tools_used == 0      # a tool that already ran must never run twice
                    and not job.cancel.is_set() and _OVERFLOW_RE.search(str(exc))
                ):
                    # 2026-10-06: the server says the prompt did not fit its context -- the
                    # estimate was too optimistic. Shrink the estimator (note_overflow), rebuild
                    # the prompt with the smaller budgets and stream ONCE more; Started was
                    # already published and is not repeated, a second failure is reported below
                    # exactly as before. Only before the first delta: a stream that already
                    # produced text must not restart under the reader.
                    retried = True
                    log.warning("assist job %d (%s): prompt overflowed the server's context (%s); "
                                "rebuilding smaller and retrying once", job.id, job.kind, exc)
                    self._note_overflow()
                    try:
                        prepared = self._prepare(job, screen_text, choices, choices_on_screen, since_id, until_id)
                    except Exception as prep_exc:  # noqa: BLE001 - the retry must end in Done/Error like any job
                        log.error("assist job %d (%s): rebuilding the prompt failed: %s", job.id, job.kind,
                                  prep_exc, exc_info=True)
                        self._settle_plan_state(job, is_plan, "", keep=False)
                        self._settle_insights(job, ok=False)
                        self._settle_dashboard(job, ok=False)
                        self._bus.publish(AssistErrorEvent(job_id=job.id, message=str(prep_exc)))
                        return
                    self._publish_usage(job, prepared.bundle)
                    continue
                log.error("assist job %d (%s) failed: %s", job.id, job.kind, exc, exc_info=True)
                self._settle_plan_state(job, is_plan, "", keep=False)
                self._settle_insights(job, ok=False)
                self._settle_dashboard(job, ok=False)
                self._bus.publish(AssistErrorEvent(job_id=job.id, message=str(exc)))
                self._bus.publish(ComponentStatusEvent(component="llm", state="error", detail=str(exc)))
                return
            except _NoAnswerError as exc:
                # The tool loop's own verdict (not a server failure): the calls ran, nothing was written.
                log.warning("assist job %d (%s): %s", job.id, job.kind, exc)
                self._settle_plan_state(job, is_plan, "", keep=False)
                self._bus.publish(AssistErrorEvent(job_id=job.id, message=str(exc)))
                return
            except Exception as exc:  # noqa: BLE001 - surface any bug as a job error, never kill the worker
                log.error("assist job %d (%s) failed unexpectedly: %s", job.id, job.kind, exc, exc_info=True)
                self._settle_plan_state(job, is_plan, "", keep=False)
                self._settle_insights(job, ok=False)
                self._settle_dashboard(job, ok=False)
                self._bus.publish(AssistErrorEvent(job_id=job.id, message=str(exc)))
                return

        self._finish_usage(job, prepared, stats, sent_chars)
        self._queue_compaction(prepared.bundle)

        if not job.cancel.is_set() and prompts.is_skip(text):
            # The model found nothing technical to do (a poll, chrome, small talk): drop it quietly.
            log.info("assist job %d (%s): model skipped (%s)", job.id, job.kind, prompts.SKIP_MARKER)
            self._settle_plan_state(job, is_plan, "", keep=False)
            # Nothing to add is still a completed pass: the same material is not re-sent next tick.
            self._settle_insights(job, ok=True, until_id=until_id, screen_text=screen_text)
            self._settle_dashboard(job, ok=True, text="", until_id=until_id, screen_text=screen_text)
            self._bus.publish(AssistDoneEvent(job_id=job.id, text="", ttft_s=0.0, tokens=0, tok_per_s=0.0,
                                              cancelled=True, skipped=True))
            return

        cancelled = job.cancel.is_set()
        if stats is not None:
            ttft_s, tokens, tok_per_s, cancelled = stats.ttft_s, stats.tokens, stats.tok_per_s, cancelled or stats.cancelled
        else:
            elapsed = max(time.monotonic() - start_wall, 1e-9)
            ttft_s = (first_delta_wall - start_wall) if first_delta_wall is not None else elapsed
            tokens = delta_count
            tok_per_s = tokens / elapsed

        # The multiple-choice verdict line goes to the bus and the history only; the plan (what
        # the completer offers as ghost text) keeps the raw answer.
        final_text = text
        if not cancelled and choices is not None and text.strip():
            final_text = self._finalize_choices(job, text, choices)

        self._settle_plan_state(job, is_plan, text, keep=not cancelled and bool(text.strip()))
        # Before the Done event, so the Insights pane already holds the new lists when its
        # streaming preview ends.
        self._settle_insights(job, ok=not cancelled, text=text, until_id=until_id, screen_text=screen_text)
        self._settle_dashboard(job, ok=not cancelled, text=text, until_id=until_id, screen_text=screen_text)
        self._bus.publish(
            AssistDoneEvent(
                job_id=job.id, text=final_text, ttft_s=ttft_s, tokens=tokens, tok_per_s=tok_per_s, cancelled=cancelled
            )
        )
        if not cancelled and final_text.strip() and job.kind not in ("complete", "insights", "dashboard"):
            if job.kind == "suggest":
                with self._suggestions_lock:
                    self._suggestions.append((self._clock(), final_text))
                    del self._suggestions[:-50]
            else:
                # "solve", "code", "text" and "explain" answers count as answers (they appear in
                # the Assist panel and export), not suggestions; only "solve" and "code" also feed
                # `_code_answers` (the <previous> block a later "code"/"text" job builds on) -- a
                # walkthrough is prose, never something to build code on.
                entry = (self._clock(), job.question, final_text)
                with self._answers_lock:
                    self._answers.append(entry)
                if job.kind in ("solve", "code"):
                    self._append_code_answer(job, entry)

    # ---- context bookkeeping around a job (2026-10-06) -----------------------------------

    def _publish_usage(self, job: _AssistJob, bundle: ContextBundle | None) -> None:
        """Publish the estimate half of a job's `ContextUsageEvent` (the bundle's own, stamped with
        the job id) right after its prompt was built. Nothing for "explain" (no bundle)."""
        if bundle is None:
            return
        self._bus.publish(dataclasses.replace(bundle.usage, job_id=job.id))

    def _sent_chars(self, system: str, user: str, assistant_prefix: str | None) -> int:
        """The characters actually sent for a prompt, mirroring `_stream_text`'s choice of path:
        the rendered ChatML for a raw completion, system + user for a chat request. This is the
        text the server counted `prompt_tokens` for, so it is what the estimator calibrates on."""
        if assistant_prefix is not None:
            return len(prompts.render_chatml(system, user, assistant_prefix))
        if self._cfg.llm.thinking == "on":
            return len(system) + len(user)
        return len(prompts.render_chatml(system, user, prompts.NO_THINK_PREFIX))

    def _observe_prompt(self, sent_chars: int, stats: Any) -> None:
        """Calibrate the context manager's chars-per-token estimate from the server's own count
        for the request that just finished (a no-op when the client reported none)."""
        tokens = getattr(stats, "prompt_tokens", None)
        if tokens is None:
            return
        try:
            self._context.observe_prompt(sent_chars, tokens)
        except Exception:  # noqa: BLE001 - calibration must never break a job
            log.exception("assistant: calibrating the token estimate failed")

    def _finish_usage(
        self, job: _AssistJob, prepared: _Prepared, stats: Any, sent_chars: int | None = None,
    ) -> None:
        """After the stream: calibrate from the server's count and publish the second
        `ContextUsageEvent` of the job -- the same estimates plus `prompt_tokens` / `cached_tokens`
        as the server reported them (None when it did not) and the post-calibration
        chars-per-token -- and hand it to the manager (`record_usage`, what the Models window's
        usage panel reads back). `sent_chars` (the tool loop's) is the size of the LAST request
        sent -- the one `stats.prompt_tokens` counts; None = the prepared prompt, as before."""
        if sent_chars is None:
            sent_chars = self._sent_chars(prepared.system, prepared.user, prepared.assistant_prefix)
        self._observe_prompt(sent_chars, stats)
        bundle = prepared.bundle
        if bundle is None:
            return
        try:
            cpt = float(self._context.estimator.chars_per_token)
        except Exception:  # noqa: BLE001
            cpt = bundle.usage.chars_per_token
        final = dataclasses.replace(
            bundle.usage, job_id=job.id,
            prompt_tokens=getattr(stats, "prompt_tokens", None),
            cached_tokens=getattr(stats, "cached_tokens", None),
            chars_per_token=cpt,
        )
        self._bus.publish(final)
        try:
            self._context.record_usage(final)
        except Exception:  # noqa: BLE001
            log.exception("assistant: recording the context usage failed")

    def _note_overflow(self) -> None:
        """Tell the estimator the server found a prompt too long (guarded)."""
        try:
            self._context.note_overflow()
        except Exception:  # noqa: BLE001
            log.exception("assistant: noting a context overflow failed")

    def _queue_compaction(self, bundle: ContextBundle | None) -> None:
        """Queue the compaction of what fell out of the transcript window, when the bundle says so
        (`compact_wanted`, which already accounts for `[context].compaction`). Queued AFTER the
        answer is done so the summary job never competes with the answer for the model; latest
        wins in its slot, and the dropped text stays pending in the window until a job claims it."""
        if bundle is None or not bundle.compact_wanted or self._cfg.context.compaction != "summary":
            return
        self._compact_slot.put(_CompactJob(id=next(self._job_ids), scope=bundle.compact_scope))

    def _settle_plan_state(self, job: _AssistJob, is_plan: bool, text: str, *, keep: bool) -> None:
        """Record how a plan-related job ended: a solve that is the plan is finished (or, when
        cancelled / failed / skipped, discarded); an explanation is attached to the plan. `is_plan`
        is what the job was when it STARTED -- explanations toggled off mid-stream must not leave
        the plan stuck unfinished."""
        if is_plan:
            self._finish_plan(job.id, text, keep)
        elif job.kind == "explain":
            self._finish_explanation(job.plan_id, text, keep)

    @staticmethod
    def _detect_choices(job: _AssistJob, screen_text: str) -> tuple[ChoiceSet | None, bool]:
        """The multiple-choice option set this job is about, and whether it came from the screen.

        Looked for in the question itself first (a typed "Which is prime? A) 4 B) 6 C) 7"), then in
        the screen text handed to the prompt: always for a "solve" or "screen" job (the screen IS
        the task), for a spoken or typed question only when it refers to options ("which one is
        it?") -- a quiz open in another tab must not force a verdict onto "what does NAT stand
        for". Never for suggestions, explanations, inline completions, "code" (writing code is
        never a multiple-choice answer; the language default resolves the one thing that could
        otherwise look like a fork -- see `prompts.system_code`) or "text" (a walkthrough reasons
        its way to an answer in prose; the two-sentence verdict shape would cut it short).
        """
        if job.kind in ("suggest", "explain", "complete", "code", "text", "insights", "recap", "dashboard"):
            return None, False
        if job.question and job.kind != "solve":
            cs = find_choices(job.question, typed=True)
            if cs is not None:
                log.info("assist job %d (%s): multiple-choice question in the text (%d options)",
                         job.id, job.kind, len(cs.choices))
                return cs, False
        if not screen_text:
            return None, False
        if job.kind in ("ask", "auto", "manual") and not _REFERS_TO_OPTIONS_RE.search(job.question):
            return None, False
        cs = find_choices(screen_text, question=job.question if job.kind != "solve" else "")
        if cs is not None:
            log.info("assist job %d (%s): multiple-choice question on screen (%d options)",
                     job.id, job.kind, len(cs.choices))
            return cs, True
        return None, False

    def _finalize_choices(self, job: _AssistJob, text: str, choices: ChoiceSet) -> str:
        """`text` with the multiple-choice verdict as its last line (`choices.finalize_answer`).

        The appended line is also published as one more `AssistDeltaEvent`, since the headless
        console only ever prints deltas; the Done event and the answer history carry the
        finalised text. A fork answer (`CHOICE_PREFIX` lines) is left untouched.
        """
        if prompts.split_choices(text)[1]:
            return text
        final = finalize_answer(text, choices)
        if final == text:
            return text
        last = final.rsplit("\n", 1)[-1]
        if not text.rstrip().endswith(last):
            # Stream the verdict too (bold, exactly as the Done text ends), unless the model
            # already produced that very line; a plain "Answer: 1 — 4" is not enough, since the
            # console shows only what was streamed.
            self._bus.publish(AssistDeltaEvent(job_id=job.id, delta="\n\n" + last))
        log.info("assist job %d (%s): multiple-choice verdict %r", job.id, job.kind, last)
        return final

    # ---- prompt assembly (2026-10-06: the context manager sizes every block) ------------

    def _prepare(
        self, job: _AssistJob, screen_text: str, choices: ChoiceSet | None, choices_on_screen: bool,
        since_id: int | None, until_id: int | None,
    ) -> _Prepared:
        """Assemble one job's prompt: the system prompt first (its size is part of the budget),
        then the context bundle (`ContextManager.build`: anchored transcript window, reference /
        lookup blocks, earlier summary, screen text, all fitted to the server's context for this
        job's answer budget), then the user turn. Called a second time, with a smaller estimate,
        by the overflow retry. "explain" prompts carry no transcript and skip the manager.

        A "dashboard" job's prompt comes from the dashboard service's hook (`_dashboard_prompt`:
        the board's open items, the transcript up to `until_id`, the project lookups) and skips
        the manager as well. For the other kinds the project context (2026-10-20) appends the
        installed toolkit's catalog to the system prompt (`_toolkit_system_suffix`, config-stable,
        so it stays a cached prefix) and puts the on-demand repository / toolkit-detail lookups
        in the user turn (`_project_extra`), which the manager budgets as fixed text."""
        if job.kind == "complete":
            return self._prepare_complete(job, screen_text, since_id, until_id)
        if job.kind == "dashboard":
            system, user, max_tokens = self._dashboard_prompt(job, screen_text, until_id)
            return _Prepared(system, user, max_tokens, None, None, None)
        # The annotated copies exist only for the prompt: every screen-line fingerprint (plan
        # matching, suggest throttling, screen-question de-dup) keeps comparing raw OCR text.
        prompt_screen, prompt_question = screen_text, job.question
        if choices is not None:
            if choices_on_screen:
                prompt_screen = annotate_choices(screen_text, choices)
            else:
                prompt_question = annotate_choices(job.question, choices)
        system = self._system_for(job, choices) + self._toolkit_system_suffix(job.kind)
        max_tokens = self._max_tokens_for(job, choices)
        if job.kind == "explain":
            user = self._user_for(job, "", prompt_screen, choices, prompt_question)
            return _Prepared(system, user, max_tokens, None, None, None)
        previous = self._previous_for(job)
        block = choices_block(choices) if choices is not None else ""
        spoken = job.kind in ("auto", "manual", "ask", "screen")
        tail = self._transcript.window_text(_LOOKUP_TAIL_CHARS, until_id=until_id, since_id=since_id)
        extra = self._project_extra(job, tail, prompt_screen)
        # The characters of the user turn that are not a context block (the manager budgets the
        # rest): the ask / question / focus, the <choices> block, the <previous> code (or the
        # insights' lists, the recap's notes), the project lookups, the tags.
        fixed = 200 + len(block) + len(previous) + len(prompt_question if spoken else job.focus) + len(extra)
        tool_ctx = self._job_tools(job)
        if tool_ctx is not None:
            # The room one tool result may take, and the <map> block: the transcript window is cut
            # to leave them, so a result appended mid-answer still fits the server's context.
            fixed += int(tool_ctx.budget.chars) + len(tool_ctx.map_text)
        bundle = self._build_context(
            job, question=self._lookup_question(job), screen_text=prompt_screen, since_id=since_id,
            until_id=until_id, max_tokens=max_tokens, system=system, fixed_chars=fixed,
        )
        user = self._user_for(
            job, bundle.transcript, bundle.screen, choices, prompt_question, previous=previous,
            reference=bundle.reference, earlier=bundle.earlier, lookup=bundle.lookup, extra=extra,
        )
        return _Prepared(system, user, max_tokens, None, None, bundle)

    def _prepare_complete(
        self, job: _AssistJob, screen_text: str, since_id: int | None, until_id: int | None
    ) -> _Prepared:
        """`_prepare` for an inline completion: its extra blocks (answers, notes, the plan's
        solution) and the typed prefix / suffix / field count against the budget as fixed text."""
        cc = self._cfg.completion
        inputs = self._complete_inputs(job)
        system = self._complete_system()
        fixed = 300 + len(job.prefix) + len(job.suffix) + len(job.field) + sum(len(v) for v in inputs.values())
        bundle = self._build_context(
            job, question=f"{job.field}\n{job.prefix[-300:]}", screen_text=screen_text, since_id=since_id,
            until_id=until_id, max_tokens=cc.max_tokens, system=system, fixed_chars=fixed,
        )
        user = self._complete_user(job, bundle, inputs)
        return _Prepared(system, user, cc.max_tokens, prompts.NO_THINK_PREFIX + job.prefix, cc.temperature, bundle)

    def _build_context(
        self, job: _AssistJob, *, question: str, screen_text: str, since_id: int | None,
        until_id: int | None, max_tokens: int, system: str, fixed_chars: int,
    ) -> ContextBundle:
        """`ContextManager.build`, guarded: a bug there degrades to the pre-context behaviour (the
        plain transcript window, no reference material, no compaction) instead of killing the
        worker, and is logged."""
        try:
            return self._context.build(
                kind=job.kind, question=question, screen_text=screen_text, since_id=since_id,
                until_id=until_id, max_tokens=max_tokens, system_text=system, fixed_chars=fixed_chars,
            )
        except Exception:  # noqa: BLE001 - the worker must always get a prompt
            log.exception("assist job %d (%s): context build failed; using the plain transcript window",
                          job.id, job.kind)
        window = self._transcript.window_text(self._cfg.llm.context_tokens * 4, until_id=until_id, since_id=since_id)
        usage = ContextUsageEvent(job_id=job.id, kind=job.kind)
        return ContextBundle(window, "", "", "", screen_text, usage, since_id, False, 0)

    @staticmethod
    def _lookup_question(job: _AssistJob) -> str:
        """What the reference index searches for: the spoken / typed / on-screen question itself;
        for the screen-driven kinds (whose `question` is only a display label) the user's own
        focus text, which may be blank -- the manager adds the recent transcript and screen terms."""
        if job.kind in ("suggest", "solve", "code", "text", "explain", "insights", "recap"):
            return job.focus
        return job.question

    def _consider_given_code(self, job: _AssistJob) -> bool:
        """2026-10-29: whether this job's prompt tells the model to work with the code already on
        screen (`[assist].consider_given_code`, read live) -- never for a "fresh" code job, the
        manual-advance "Code from scratch" gesture (the flag means nothing on any other kind)."""
        fresh = bool(job.fresh) and job.kind == "code"
        return bool(self._cfg.assist.consider_given_code) and not fresh

    def _system_for(self, job: _AssistJob, choices: ChoiceSet | None = None) -> str:
        """The system prompt for `job`. `choices` is accepted for symmetry with the budget but does
        not change it: the multiple-choice rule is worded conditionally ("If a <choices> block is
        present ..."), so a system prompt never differs between turns (a stable KV-cache prefix).
        `[assist].session_context` joins every one of them (config-only text, just as stable)."""
        persona = self._cfg.assist.persona
        context = self._cfg.assist.session_context
        # 2026-10-07 (capabilities): with tools for this job the rule + the <tools> block end the
        # prompt; without, no keyword is passed and the builders render exactly what they always did.
        tool_ctx = self._job_tools(job)
        extra: dict[str, str] = {"tools": tool_ctx.block} if tool_ctx is not None else {}
        if job.kind == "ask":
            return prompts.system_ask(persona, context, **extra)
        if job.kind in ("solve", "code"):
            ex = self._cfg.explain
            level = prompts.normalize_comment_level(self._cfg.assist.comments)
            builder = prompts.system_solve if job.kind == "solve" else prompts.system_code
            return builder(
                persona, explain=ex.enabled, comments=level,
                language=self._cfg.assist.primary_language, libraries=self._cfg.assist.libraries,
                context=context, consider_given_code=self._consider_given_code(job), **extra,
            )
        if job.kind == "text":
            return prompts.system_text(persona, context, consider_given_code=self._consider_given_code(job),
                                       **extra)
        if job.kind == "explain":
            return prompts.system_explain(persona, context)
        if job.kind == "suggest":
            return prompts.system_suggest(persona, context, **extra)
        if job.kind == "insights":
            a = self._cfg.assist
            secs = job.sections or prompts.INSIGHT_SECTIONS
            return prompts.system_insights(persona, a.role, secs, a.insight_max_items, context)
        if job.kind == "recap":
            return prompts.system_recap(persona, context)
        # "auto", "manual", "screen"
        return prompts.system_assist(persona, self._cfg.assist.style, context, **extra)

    def _job_tools(self, job: _AssistJob) -> _JobTools | None:
        """The tool loop's state for `job`, resolved on first use and kept on the job (None = this
        job gets no tools): the runner must `wants` the kind, list at least one tool and allow at
        least one call. The budget is taken ONCE, so the "At most N calls" line of the system
        prompt, the prompt rebuilt by the overflow retry and the loop all see the same one. Never
        for "complete" / "explain". A runner that raises means no tools, not a failed job."""
        if job.tool_ctx_resolved:
            return job.tool_ctx
        job.tool_ctx_resolved = True
        runner = self._tools
        if runner is None or job.kind in _NO_TOOL_KINDS:
            return None
        try:
            if not runner.wants(job.kind):
                return None
            specs = list(runner.specs(job.kind))
            budget = runner.budget(job.kind)
            block = prompts.tools_block(specs, max_calls=budget.calls_left)
            if not block or budget.calls_left <= 0:
                return None
            map_text = self._map_for_prompt(runner.map_text())
        except Exception:  # noqa: BLE001 - tools are an extra: a broken runner must not break an answer
            log.exception("assist job %d (%s): the tool runner failed; answering without tools", job.id, job.kind)
            return None
        job.tool_ctx = _JobTools(budget=budget, specs=specs, block=block, map_text=map_text)
        return job.tool_ctx

    def _map_for_prompt(self, text: str) -> str:
        """The runner's map text for the ``<map>`` block, "" for no map (or `map_tokens` = 0).

        The Engine already cut the text ONCE when it rendered the map, to
        `[capabilities].map_tokens` x `[context].chars_per_token` characters -- the CONFIGURED
        ratio, never the calibrated estimator -- so it is used verbatim here and the block is
        byte-identical across jobs (the cached prefix of the user turn reaches into the transcript).
        The cut below is only a safety net, at four times that figure, for a map file indexed under
        a much larger `map_tokens`; it too uses the configured ratio, so it is just as stable."""
        text = (text or "").strip()
        if not text:
            return ""
        tokens = max(0, int(self._cfg.capabilities.map_tokens))
        if tokens <= 0:
            return ""
        try:
            ratio = float(self._cfg.context.chars_per_token)
        except (TypeError, ValueError):
            ratio = 4.0
        limit = 4 * max(1, int(tokens * ratio))
        return _head_lines(text, limit).strip() if len(text) > limit else text

    def _max_tokens_for(self, job: _AssistJob, choices: ChoiceSet | None = None) -> int:
        """The answer budget (`max_tokens`) of `job`; `choices` adds `CHOICES_TOKEN_HEADROOM` to the
        question-answering kinds so the verdict line never competes with the answer's own shape."""
        headroom = CHOICES_TOKEN_HEADROOM if choices is not None else 0
        llm = self._cfg.llm
        if job.kind == "ask":
            return llm.ask_max_tokens + headroom
        if job.kind in ("solve", "code"):
            level = prompts.normalize_comment_level(self._cfg.assist.comments)
            return self._code_budget(self._cfg.explain.enabled, level) + headroom
        if job.kind == "text":
            return llm.text_max_tokens
        if job.kind == "explain":
            return self._cfg.explain.explain_max_tokens
        if job.kind == "suggest":
            return llm.assist_max_tokens
        if job.kind == "insights":
            return llm.insights_max_tokens
        if job.kind == "recap":
            return llm.recap_max_tokens
        return llm.assist_max_tokens + headroom     # "auto", "manual", "screen"

    def _previous_for(self, job: _AssistJob) -> str:
        """The block a user turn carries besides the transcript, the screen and the context
        blocks, which the context manager budgets as fixed text: the <previous> code of a "code" /
        "text" job, the current lists of an "insights" pass, the running notes of a "recap"; ""
        for every other kind."""
        if job.kind in ("code", "text"):
            return self._previous_code_text()
        if job.kind == "insights":
            secs = job.sections or prompts.INSIGHT_SECTIONS
            with self._insights_lock:
                return prompts.render_insights({s: self._insights.get(s, "") for s in secs})
        if job.kind == "recap":
            with self._notes_lock:
                return self._notes_markdown
        return ""

    def _user_for(
        self, job: _AssistJob, window: str, screen_text: str, choices: ChoiceSet | None = None,
        question: str | None = None, *, previous: str | None = None,
        reference: str = "", earlier: str = "", lookup: str = "", extra: str = "",
    ) -> str:
        """The user turn for `job`. `choices` (a detected multiple-choice option set) adds the
        <choices> block to the question-answering prompts; `question` replaces `job.question` in
        the prompt (its annotated copy) and defaults to it; `previous` is the kind's fixed block
        (`_previous_for`: the <previous> code of a "code" / "text" job, the lists of an "insights"
        pass, the notes of a "recap"; None = look it up). `reference` / `earlier` / `lookup` are the
        context manager's blocks (2026-10-06), "" when absent -- which renders as before they
        existed; `extra` is the project context's on-demand lookups (`_project_extra`)."""
        block = choices_block(choices) if choices is not None else ""
        q = job.question if question is None else question
        ctx = {"reference": reference, "earlier": earlier, "lookup": lookup}
        tool_ctx = self._job_tools(job)
        if tool_ctx is not None and tool_ctx.map_text:
            ctx["map_text"] = tool_ctx.map_text         # 2026-10-07: the project map, after <earlier>
        if job.kind == "ask":
            return prompts.user_ask(window, q, screen_text, block, extra=extra, **ctx)
        if job.kind == "solve":
            return prompts.user_solve(window, screen_text, job.focus, block, extra=extra,
                                      consider_given_code=self._consider_given_code(job), **ctx)
        if job.kind in ("code", "text"):
            prev = self._previous_code_text() if previous is None else previous
            builder = prompts.user_code if job.kind == "code" else prompts.user_text
            return builder(window, screen_text, prev, job.focus, extra=extra,
                           consider_given_code=self._consider_given_code(job), **ctx)
        if job.kind == "explain":
            with self._plan_lock:
                plan = self._plan
                parts = self._parsed_plan_locked(plan) if plan is not None else None
            walkthrough = parts.walkthrough if parts is not None else ""
            solution = parts.solution if parts is not None else ""
            return prompts.user_explain(walkthrough, solution, screen_text)
        if job.kind == "suggest":
            return prompts.user_suggest(window, screen_text, job.focus, **ctx)
        if job.kind == "insights":
            prev = self._previous_for(job) if previous is None else previous
            return prompts.user_insights(window, screen_text, prev, job.sections or prompts.INSIGHT_SECTIONS, **ctx)
        if job.kind == "recap":
            notes = self._previous_for(job) if previous is None else previous
            return prompts.user_recap(window, notes, screen_text, job.focus, extra=extra, **ctx)
        # "auto", "manual", "screen"
        return prompts.user_assist(window, q, screen_text, block, extra=extra, **ctx)

    def _build_prompt_parts(
        self, job: _AssistJob, window: str, screen_text: str, choices: ChoiceSet | None = None,
        question: str | None = None, *, reference: str = "", earlier: str = "", lookup: str = "",
        extra: str = "",
    ) -> tuple[str, str, int]:
        """System/user prompt text and the token budget for one job, around an explicit `window`
        (the pre-context-library entry point; `_prepare` is what the job runner uses now)."""
        return (
            self._system_for(job, choices) + self._toolkit_system_suffix(job.kind),
            self._user_for(job, window, screen_text, choices, question,
                           reference=reference, earlier=earlier, lookup=lookup, extra=extra),
            self._max_tokens_for(job, choices),
        )

    def _code_budget(self, explain: bool, level: str) -> int:
        """Token budget for a "solve" / "code" answer: `solve_max_tokens`, plus the walkthrough's
        share with explanations on, plus `COMMENT_TOKEN_HEADROOM` for the comment `level` (one of
        `prompts.COMMENT_LEVELS`)."""
        extra = self._cfg.explain.walkthrough_max_tokens if explain else 0
        return self._cfg.llm.solve_max_tokens + extra + COMMENT_TOKEN_HEADROOM.get(level, 0)

    def _complete_inputs(self, job: _AssistJob) -> dict[str, str]:
        """The inline completion's own context blocks that are not the transcript / screen: the
        recent answers, the notes and the plan's solution, keyed by their prompt tag and present
        only when their `[completion].use_*` switch (or explanations) asks for them. They are
        scoped to the current checkpoint like the transcript window."""
        cc = self._cfg.completion
        inputs: dict[str, str] = {}
        if cc.use_answers:
            checkpoint_answers = self._checkpoint_answers_index()
            with self._answers_lock:
                recent = self._answers[checkpoint_answers:][-3:]
            inputs["answers"] = "\n\n".join(f"Q: {q[:200]}\nA: {a[:600]}" for _t, q, a in recent)
        if cc.use_notes:
            with self._notes_lock:
                inputs["notes"] = self._notes_markdown[:1200]
        if self._cfg.explain.enabled:
            view = self.plan()
            if view is not None and view.solution.strip():
                inputs["solution"] = view.solution    # the plan: the continuation must follow it
        return inputs

    def _complete_system(self) -> str:
        cc = self._cfg.completion
        return prompts.system_complete(
            self._cfg.assist.persona, cc.hint, explain=self._cfg.explain.enabled,
            language=self._cfg.assist.primary_language,
            comments=prompts.normalize_comment_level(self._cfg.assist.comments),
        )

    def _complete_user(self, job: _AssistJob, bundle: ContextBundle, inputs: dict[str, str]) -> str:
        """The user turn of a kind "complete" job.

        The session context goes in stable-first (reference, earlier summary, transcript, answers,
        notes, screen, the plan's solution, then the volatile lookup passages); the typed prefix
        becomes the start of the assistant turn so the model continues it verbatim (see
        `prompts.system_complete`). The transcript (like the answers) is already scoped to the
        current checkpoint -- see `_run_assist_job`.
        """
        cc = self._cfg.completion
        blocks: list[tuple[str, str]] = []
        if bundle.reference:
            blocks.append(("reference", bundle.reference))
        if bundle.earlier:
            blocks.append(("earlier", bundle.earlier))
        if cc.use_transcript:
            blocks.append(("transcript", bundle.transcript))
        if cc.use_answers:
            blocks.append(("answers", inputs.get("answers", "")))
        if cc.use_notes:
            blocks.append(("notes", inputs.get("notes", "")))
        if cc.use_screen:
            blocks.append(("screen", bundle.screen))
        if "solution" in inputs:
            blocks.append(("solution", inputs["solution"]))
        if bundle.lookup:
            blocks.append(("lookup", bundle.lookup))
        return prompts.user_complete(blocks, job.field, job.suffix)

    def _screen_text_for_job(self, job: _AssistJob) -> str:
        """Fetch (and cache as `last_screen_text`) the screen text to hand this job's prompt.

        Called once per job, on the assist-worker thread, at job start -- even for a job that
        turns out to already be cancelled (its `AssistStartedEvent` was still published, so it
        "started" per the module docstring). Honors `cfg.screen.use_in_answers` for "auto",
        "manual" and "ask" (SPEC.md: `ScreenConfig.use_in_answers` is documented as gating
        "assist/ask prompts"). "suggest", "screen", "solve", "code", "text" and "explain" always
        use the screen text when a provider is set: a suggestion has nothing to propose without it,
        "code" and "text" need it for the same reason `solve` does (part of what "the task being
        discussed / shown" means), and a "screen" job exists specifically to answer a question the OCR just found on
        screen -- without the surrounding screen text, the model would see only the bare extracted
        question line (e.g. "how would you implement this?" with no antecedent for "this"), so
        gating it the same way as a normal spoken-question answer would defeat the feature's own
        purpose. Any exception from the provider is swallowed and treated as no screen text.
        """
        provider = self._screen_provider
        text = ""
        if job.kind == "complete":
            wants_screen = self._cfg.completion.use_screen
        else:
            wants_screen = (
                job.kind in ("suggest", "screen", "solve", "code", "text", "explain", "insights", "recap", "dashboard")
                or self._cfg.screen.use_in_answers
            )
        max_chars = self._cfg.screen.max_text_chars
        context = self._hands_free_context if self._hands_free and job.kind in ("text", "code") else None
        if context is not None and wants_screen:
            # Hands-free: the cumulative screen of the task (what scrolled out of view + the
            # current screen), with its own, larger budget; "" means nothing read yet.
            try:
                text = context() or ""
            except Exception:  # noqa: BLE001 - never break answers over the context provider
                log.exception("assistant: hands-free context provider raised for job %d", job.id)
                text = ""
            if text:
                max_chars = max(max_chars, int(self._cfg.handsfree.context_chars))
        if not text and provider is not None and wants_screen:
            try:
                text = provider() or ""
            except Exception:  # noqa: BLE001 - a broken screen provider must never break answers
                log.exception("assistant: screen_provider() raised for job %d; using no screen text", job.id)
                text = ""
        if max_chars >= 0:
            text = text[:max_chars]
        with self._screen_text_lock:
            self._last_screen_text = text
        return text

    def _record_suggest_start(self, job: _AssistJob, screen_text: str) -> None:
        """Record the auto-suggest throttle state when a suggest job starts (see the module
        docstring's "record ... when a suggest job STARTS", not when it is merely submitted).

        Uses the OCR lines that triggered an auto-suggest job (`job.screen_lines`) when present;
        a manually triggered `suggest_now()` call (`screen_lines is None`) falls back to the
        screen text fetched for this same job. Lines are normalised (see `_normalize_line`) so
        the next Jaccard comparison in `_maybe_auto_suggest` isn't thrown off by incidental OCR
        noise (case, extra whitespace) in the lines that triggered THIS suggestion.
        """
        lines = job.screen_lines if job.screen_lines is not None else tuple(screen_text.splitlines())
        with self._suggest_state_lock:
            self._last_suggest_time = self._clock()
            self._last_suggested_lines = _normalized_line_set(lines)

    # ---- notes-worker ---------------------------------------------------------

    def _notes_worker_loop(self) -> None:
        log.info("notes-worker: started")
        while True:
            job = self._next_notes_job()
            if job is None:
                log.info("notes-worker: stopping")
                return
            try:
                if isinstance(job, _CompactJob):
                    self._run_compact_job(job)
                elif isinstance(job, _DigestJob):
                    self._run_digest_job(job)
                else:
                    self._run_notes_job(job)
            except Exception:  # pragma: no cover - safety net; the _run_* methods handle their own errors
                log.exception("notes-worker: unhandled error running job %d", job.id)

    def _next_notes_job(self) -> "_NotesJob | _CompactJob | _DigestJob | None":
        """Block until a notes-lane job is ready: notes first, then a compaction, then a digest
        (2026-10-06); `None` once the notes slot is closed (the worker's stop signal).

        Same shape as `_next_assist_job`: the lanes' slots are polled with `try_get`, and the
        notes slot's condition is waited on with a short timeout, so a notes job (or `close()`)
        wakes the worker at once while a compaction / digest queued on its own slot is picked up
        within `_DISPATCH_POLL_S`. A job that would run while a notes job waits never delays it by
        more than the one job already running (the lane is not cancellable).
        """
        while True:
            for slot in (self._notes_slot, self._compact_slot, self._digest_slot):
                job = slot.try_get()
                if job is not None:
                    return job
            if self._notes_slot.closed:
                return None
            self._notes_slot.wait(timeout=_DISPATCH_POLL_S)

    def _run_notes_job(self, job: _NotesJob) -> None:
        with self._notes_lock:
            since_id = self._notes_last_id
            previous = self._notes_markdown
        if not job.force:
            new_words = self._transcript.word_count(since_id=since_id)
            if new_words < self._cfg.notes.min_new_words:
                log.info(
                    "notes job %d: skipped (%d new word(s) < %d required)",
                    job.id, new_words, self._cfg.notes.min_new_words,
                )
                return

        # upstream finding 3 (ported from mac-cortex, adjusted for Windows): this used to be
        # `text_since(since_id)` with no cap at all, so a long meeting -- or one failed cycle,
        # which froze `since_id` -- could send a prompt exceeding `ctx_size`, and keep doing so
        # every cycle. The slice is now bounded to the same char budget `window_text` gives every
        # other job. Unlike mac-cortex (which keeps the NEWEST part of an oversized slice and
        # advances the checkpoint on failure -- both silently drop transcript from the notes), the
        # slice here is the OLDEST part that fits and the checkpoint advances exactly to the last
        # segment it contained (`Transcript.head_text`): a backlog (an LLM outage, auto-notes
        # paused for a stretch, one very long gap between cycles) is processed in order, chunk by
        # chunk, with nothing skipped, and every prompt stays bounded no matter how stale
        # `since_id` is. `tip` is the newest segment at this point; a chunk that stops short of it
        # was cut by the budget, and the next chunk is queued right after this one succeeds.
        tip = self._transcript.last_id()
        # Dashboard mode keeps the tasks, goals, constraints and questions in its own database:
        # the notes are then narrative only (the notes prompt's `tracked` form).
        dash = self._cfg.dashboard
        system = prompts.system_notes(self._cfg.notes.template, tracked=bool(dash.enabled and dash.track))
        chunk_chars, max_tokens, previous = self._notes_budget(previous, system)
        new_text, last_included = self._transcript.head_text(chunk_chars, since_id=since_id)
        until_id = last_included if last_included is not None else since_id   # nothing new: stay put
        max_words = prompts.words_for_tokens(max_tokens) if max_tokens < int(self._cfg.llm.notes_max_tokens) else None
        user = prompts.user_notes(previous, new_text, max_words)
        try:
            text = self._stream_text(system, user, max_tokens=max_tokens)
        except LLMError as exc:
            if _OVERFLOW_RE.search(str(exc)):
                self._note_overflow()   # 2026-10-06: the next (re-sent) chunk is cut smaller
            log.error("notes job %d failed: %s", job.id, exc, exc_info=True)
            self._bus.publish(NotesErrorEvent(message=str(exc)))
            self._bus.publish(ComponentStatusEvent(component="llm", state="error", detail=str(exc)))
            # upstream finding 3 (Windows fix): leave `_notes_last_id` untouched on failure. The
            # bounded chunk above already keeps this job's prompt within `context_tokens * 4`
            # chars no matter how stale `since_id` is, so the "unbounded slice keeps overflowing
            # ctx_size forever" case finding 3 describes cannot recur -- the next cycle resends
            # the same bounded chunk and, once the failure was transient (backend briefly down, a
            # network timeout, a malformed streaming response), succeeds and catches up. Advancing
            # here instead (what mac-cortex does) would silently and permanently drop whatever was
            # said during this window from the notes, which is a worse outcome than a retry.
            return
        except Exception as exc:  # noqa: BLE001 - surface any bug as a job error, never kill the worker
            log.error("notes job %d failed unexpectedly: %s", job.id, exc, exc_info=True)
            self._bus.publish(NotesErrorEvent(message=str(exc)))
            # upstream finding 3 (Windows fix): same reasoning as the LLMError branch above -- an
            # arbitrary bug caught here is even less likely to be caused by the (now-bounded)
            # prompt size, so it should not permanently discard this window from the notes either.
            return

        self._observe_prompt(self._sent_chars(system, user, None), self._client.last_stats)
        markdown = text.strip()
        with self._notes_lock:
            self._notes_markdown = markdown
            self._notes_last_id = until_id
        self._bus.publish(NotesUpdatedEvent(markdown=markdown, generated_at=self._clock()))
        if last_included is not None and tip is not None and last_included < tip:
            # upstream finding 3 (Windows fix): the budget cut this chunk short of the transcript
            # as it stood when the job began -- run the next chunk now rather than after a full
            # `auto_interval_s`, so notes catch up at LLM speed. Every chunk advances the
            # checkpoint by at least one segment, so this always terminates; speech that arrived
            # only *during* the call is not a backlog and waits for the normal cadence.
            log.info("notes job %d: backlog remains past segment %d (tip %d); queuing the next chunk",
                     job.id, last_included, tip)
            self.generate_notes(force=True)

    def _notes_budget(self, previous: str, system: str) -> tuple[int, int, str]:
        """``(chunk_chars, max_tokens, previous)`` for one notes job (2026-10-06, context window).

        A notes pass rewrites the WHOLE document: previous notes + the new chunk in, a document of
        up to `[llm].notes_max_tokens` out, and all of it inside the notes lane's share of the
        server's pool (`ContextManager.job_limit("notes")`, beside a live answer). Measured on the
        owner's 4096-token profile the lane is ~1228 tokens: the configured 900-token document
        cannot even fit once, let alone twice (previous + new), and the old `context_tokens * 4`
        chunk alone was 2500 tokens -- so the whole thing scales to the lane instead:
          - the chunk is `context_tokens * 4` chars (the old cap) when the lane has room for it
            beside the previous notes and the full answer, else what is left -- never under
            `_MIN_NOTES_CHUNK_CHARS`, so the notes always advance;
          - when the full answer would leave the chunk under `_NOTES_MIN_CHUNK_TOKENS`, the answer
            budget AND the previous notes shrink together to an equal share of what is left after
            that chunk (`user_notes(..., max_words)` tells the model the new length), so the
            document stays complete, just compact; a previous document longer than that share is
            cut on a line boundary (once -- the rewritten one fits from then on).
        Any sizing failure falls back to the old cap and the configured answer budget."""
        cap = max(_MIN_NOTES_CHUNK_CHARS, int(self._cfg.llm.context_tokens) * 4)
        notes_max = max(16, int(self._cfg.llm.notes_max_tokens))
        try:
            est = self._context.estimator
            template = est.tokens(prompts.user_notes("", "", notes_max)) + 48   # its tags + the ChatML frame
            room = self._context.job_limit("notes") - est.tokens(system) - template
            prev_t = est.tokens(previous) if previous else 0
            if room - prev_t - notes_max >= _NOTES_MIN_CHUNK_TOKENS:
                max_tokens = notes_max
            else:
                doc_t = max(_NOTES_MIN_DOC_TOKENS, (room - _NOTES_MIN_CHUNK_TOKENS) // 2)
                max_tokens = min(notes_max, doc_t)
                if prev_t > doc_t:
                    previous = _head_lines(previous, est.chars(doc_t))
                    prev_t = est.tokens(previous)
                    log.info("notes: previous notes cut to ~%d tokens to fit the notes lane (%d tokens)",
                             doc_t, self._context.job_limit("notes"))
            chunk_t = room - prev_t - max_tokens
            chunk_chars = min(cap, max(_MIN_NOTES_CHUNK_CHARS, est.chars(max(0, chunk_t))))
            return chunk_chars, max_tokens, previous
        except Exception:  # noqa: BLE001 - never stall the notes over a sizing bug
            log.exception("notes: sizing the prompt failed; using the configured cap and budget")
            return cap, notes_max, previous

    def _run_compact_job(self, job: _CompactJob) -> None:
        """Fold the transcript that fell out of `job.scope`'s window into that scope's running
        summary (the next prompts' <earlier> block). Claims the oldest part of the pending dropped
        text that fits the notes lane (the manager keeps the rest pending and this job re-queues
        itself while any is left), asks the model for a summary of at most
        `[context].summary_tokens`, and stores it; on ANY failure the claimed text is handed back
        so nothing is lost and the next answer's build asks for the compaction again."""
        c = self._cfg.context
        if c.compaction != "summary":
            return
        summary_tokens = max(16, int(c.summary_tokens))
        max_tokens = summary_tokens + 32
        system = prompts.system_compact()
        try:
            with self._epoch_lock:          # atomic with reset(): see the note there
                epoch = self._session_epoch
                previous = self._context.window.staged_summary(job.scope) or self._context.window.earlier_summary(job.scope)
                room = self._context.available_chars(
                    "notes", max_tokens=max_tokens, system_text=system, other_text=previous,
                )
                claimed = self._context.claim_compaction(job.scope, max_chars=max(room, _MIN_NOTES_CHUNK_CHARS))
        except Exception:  # noqa: BLE001 - context bookkeeping must never kill the notes worker
            log.exception("compact job %d: claiming the dropped transcript failed", job.id)
            return
        if claimed is None:
            return
        previous, dropped = claimed
        user = prompts.user_compact(previous, dropped, prompts.words_for_tokens(summary_tokens))
        summary = ""
        cancel = self._lane_cancel(job.attempts)
        try:
            summary = prompts.strip_think(
                self._stream_text(system, user, max_tokens=max_tokens, cancel=cancel)
            )
        except LLMError as exc:
            if _OVERFLOW_RE.search(str(exc)):
                self._note_overflow()
            log.warning("compact job %d failed: %s", job.id, exc)
        except Exception:  # noqa: BLE001 - surface any bug in the log, never kill the worker
            log.exception("compact job %d failed unexpectedly", job.id)
        with self._epoch_lock:
            stale = epoch != self._session_epoch
        if stale:
            return                         # the session was reset while this ran: drop everything
        yielded = self._yielded(cancel)
        try:
            if not summary or self._stop_event.is_set() or yielded:
                self._context.restore_compaction(job.scope, dropped)
                if yielded and not self._stop_event.is_set():
                    log.info("compact job %d: stepped aside for an answer; re-queued", job.id)
                    self._compact_slot.put(_CompactJob(id=next(self._job_ids), scope=job.scope,
                                                       attempts=job.attempts + 1, force=job.force))
                return
            self._observe_prompt(self._sent_chars(system, user, None), self._client.last_stats)
            if job.force:
                # A user-driven Compact already moved the anchor; there is no imminent natural
                # anchor move to fold a staged summary into, so install it active now (one prefix
                # re-evaluation) -- otherwise the dropped speech is absent AND unsummarised until
                # the window next overflows on its own.
                self._context.set_summary(job.scope, summary)
            else:
                self._context.stage_summary(job.scope, summary)    # enters the prompt at the next anchor move
            more = self._context.window.has_pending(job.scope)
        except Exception:  # noqa: BLE001
            log.exception("compact job %d: storing the summary failed", job.id)
            return
        log.info("compact job %d: scope %s summary %s (%d chars)", job.id, job.scope,
                 "installed" if job.force else "staged", len(summary))
        if more:
            # A forced compaction with a long backlog keeps folding actively (each chunk installed)
            # until the pending buffer is drained, so the whole dropped span is summarised at once.
            self._compact_slot.put(_CompactJob(id=next(self._job_ids), scope=job.scope, force=job.force))

    # ---- yielding to a live answer (2026-10-06) ---------------------------------------------

    def _lane_cancel(self, attempts: int) -> threading.Event:
        """The cancel token for a notes-lane background job: `_lane_yield` (set while an answer
        runs, and by stop()) for the first `_LANE_YIELD_ATTEMPTS` tries -- after first waiting
        for the answer lane to go idle -- then the plain stop event, so the job cannot be starved
        forever by a stream of inline completions."""
        if attempts >= _LANE_YIELD_ATTEMPTS:
            return self._stop_event
        self._wait_answer_idle()
        return self._lane_yield

    def _wait_answer_idle(self, timeout: float = _LANE_IDLE_WAIT_S) -> None:
        deadline = time.monotonic() + timeout
        while not self._stop_event.is_set() and time.monotonic() < deadline:
            with self._current_lock:
                busy = self._current_job is not None
            if not busy and not self._lane_yield.is_set():
                return
            time.sleep(_DISPATCH_POLL_S)

    def _yielded(self, cancel: threading.Event) -> bool:
        """Whether a lane job streamed with `cancel` was cut short to make room for an answer:
        the token is still set (the answer is running), or the client reports the stream as
        cancelled and this was not a shutdown."""
        if cancel is self._stop_event:
            return False
        if self._stop_event.is_set():
            return False
        if cancel.is_set():
            return True
        stats = self._client.last_stats
        return bool(stats is not None and getattr(stats, "cancelled", False))

    def _kick_digests(self, exclude: str | None = None) -> None:
        """Queue the next document still waiting for its digest (`store.pending_digests()`),
        skipping the one running and `exclude`. Called when the assistant starts / the model
        changes (a library added while no model was loaded gets digested now) and by the digest
        job itself when it finishes, which is how a burst of added files is digested one by one
        although the slot holds a single job."""
        try:
            pending = self._context.store.pending_digests()
        except Exception:  # noqa: BLE001
            log.exception("assistant: listing pending digests failed")
            return
        with self._digest_lock:
            running = self._digest_running
        for doc_id in pending:
            if doc_id != running and doc_id != exclude:
                self._digest_slot.put(_DigestJob(id=next(self._job_ids), doc_id=doc_id))
                return

    def _run_digest_job(self, job: _DigestJob) -> None:
        """Write a document's catalogue entry: the first ~6000 characters (fewer when the notes
        lane has less room) go to the model, which answers in 2-4 lines (`[context].digest_tokens`).
        Stores it ("ready", or "failed" when the model errors / says nothing), publishes a
        `ContextLibraryEvent`, and queues the next pending document."""
        doc_id = job.doc_id
        with self._digest_lock:
            self._digest_running = doc_id
        try:
            self._digest_document_now(job)
        finally:
            with self._digest_lock:
                self._digest_running = None
            self._kick_digests(exclude=doc_id)

    def _digest_document_now(self, job: _DigestJob) -> None:
        doc_id = job.doc_id
        store = self._context.store
        try:
            doc = store.get(doc_id)
            if doc is None:
                return
            max_tokens = max(16, int(self._cfg.context.digest_tokens)) + 32
            system = prompts.system_digest()
            room = self._context.available_chars(
                "notes", max_tokens=max_tokens, system_text=system, other_text=doc.name,
            )
            excerpt = store.text(doc_id)[: max(_MIN_NOTES_CHUNK_CHARS, min(_DIGEST_EXCERPT_CHARS, room))]
        except Exception:  # noqa: BLE001 - never kill the notes worker over a library problem
            log.exception("digest job %d: preparing document %s failed", job.id, doc_id)
            # Marked failed rather than left "pending": two documents that both keep failing here
            # would otherwise re-queue each other forever (review, 2026-10-06). Re-digest retries.
            try:
                store.set_digest(doc_id, "", "failed")
            except Exception:  # noqa: BLE001
                log.debug("digest job %d: could not mark %s failed", job.id, doc_id, exc_info=True)
            self._bus.publish(ContextLibraryEvent(reason="digest failed", doc_id=doc_id))
            return
        digest = ""
        cancel = self._lane_cancel(job.attempts)
        if excerpt.strip():
            user = prompts.user_digest(doc.name, excerpt)
            try:
                digest = prompts.strip_think(
                    self._stream_text(system, user, max_tokens=max_tokens, cancel=cancel)
                )
            except LLMError as exc:
                if _OVERFLOW_RE.search(str(exc)):
                    self._note_overflow()
                log.warning("digest job %d (%s) failed: %s", job.id, doc.name, exc)
            except Exception:  # noqa: BLE001
                log.exception("digest job %d (%s) failed unexpectedly", job.id, doc.name)
            else:
                self._observe_prompt(self._sent_chars(system, user, None), self._client.last_stats)
        if self._stop_event.is_set():
            return                         # shutting down: leave it pending for the next launch
        if self._yielded(cancel):
            log.info("digest job %d (%s): stepped aside for an answer; re-queued", job.id, doc.name)
            self._digest_slot.put(_DigestJob(id=next(self._job_ids), doc_id=doc_id, attempts=job.attempts + 1))
            return
        try:
            store.set_digest(doc_id, digest, "ready" if digest else "failed")
        except Exception:  # noqa: BLE001
            log.exception("digest job %d: storing the digest of %s failed", job.id, doc_id)
            return
        self._bus.publish(ContextLibraryEvent(reason="digest" if digest else "digest failed", doc_id=doc_id))

    # ---- shared streaming --------------------------------------------------------

    def _stream_text(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int,
        cancel: threading.Event | None = None,
        on_delta: Callable[[str], None] | None = None,
        assistant_prefix: str | None = None,
        temperature: float | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> str:
        """Run one completion against the LLM honoring `cfg.llm.thinking`; return the full text.

        `on_delta` (if given) is called with each non-empty piece of visible text as it arrives.
        Reasoning text (`thinking == "on"`) is consumed and discarded -- it is never passed to
        `on_delta` and never otherwise published. An explicit `assistant_prefix` (inline
        completions: the text the user typed) forces the raw-completion path, since a chat API
        cannot start the assistant turn with given text.
        """
        parts: list[str] = []
        if assistant_prefix is not None:
            prompt = prompts.render_chatml(system, user, assistant_prefix)
            for piece in self._client.stream_completion(
                prompt, max_tokens=max_tokens, temperature=temperature, stop=prompts.STOP, cancel=cancel,
                extra_body=extra_body,
            ):
                if piece:
                    parts.append(piece)
                    if on_delta is not None:
                        on_delta(piece)
            return "".join(parts)
        if self._cfg.llm.thinking == "on":
            messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
            for delta in self._client.stream_chat(messages, max_tokens=max_tokens, cancel=cancel):
                if delta.content:
                    parts.append(delta.content)
                    if on_delta is not None:
                        on_delta(delta.content)
        else:
            prompt = prompts.render_chatml(system, user, prompts.NO_THINK_PREFIX)
            for piece in self._client.stream_completion(
                prompt, max_tokens=max_tokens, stop=prompts.STOP, cancel=cancel
            ):
                if piece:
                    parts.append(piece)
                    if on_delta is not None:
                        on_delta(piece)
        return "".join(parts)

    # ---- the tool loop (2026-10-07, capabilities) --------------------------------------

    def _stream_answer(
        self, job: _AssistJob, prepared: _Prepared, on_delta: Callable[[str], None],
    ) -> tuple[str, int, StreamStats | None]:
        """Stream one assist job's answer: `(visible text, characters of the LAST request sent,
        stats of the stream)` -- the last request is what `stats.prompt_tokens` counts, so it is
        what `_finish_usage` / `_observe_prompt` pair it with.

        A job without tools (no runner, a kind the runner does not cover, "complete" / "explain")
        gets the plain `_stream_text` call, the very request it always sent. A job with tools runs
        the loop (`_run_tool_loop`): the visible text is all that reaches `on_delta`, the history,
        the plan and `AssistDoneEvent`; the calls and their results live in the continuation only."""
        tool_ctx = self._job_tools(job)
        if tool_ctx is None or prepared.assistant_prefix is not None:
            extra = None
            if job.kind == "complete" and self._cfg.completion.repeat_penalty and self._cfg.completion.repeat_penalty != 1.0:
                extra = {"repeat_penalty": float(self._cfg.completion.repeat_penalty)}
            text = self._stream_text(
                prepared.system, prepared.user, max_tokens=prepared.max_tokens, cancel=job.cancel,
                on_delta=on_delta, assistant_prefix=prepared.assistant_prefix,
                temperature=prepared.temperature, extra_body=extra,
            )
            sent = self._sent_chars(prepared.system, prepared.user, prepared.assistant_prefix)
            return text, sent, self._client.last_stats
        return self._run_tool_loop(job, prepared, tool_ctx, on_delta)

    def _run_tool_loop(
        self, job: _AssistJob, prepared: _Prepared, tool_ctx: _JobTools, on_delta: Callable[[str], None],
    ) -> tuple[str, int, StreamStats | None]:
        """The tool loop. Each pass streams one request: a `_ToolSplitter` publishes the visible
        text and holds back a call; when the stream ends with a call (`<tool>` opening a line, or
        following prose on its line with a complete JSON object; the server stops the pass at
        the closing tag, `</tool>` being a stop string of every call pass) it is parsed, run by the
        runner under the policy and appended to the conversation as a USER turn carrying its
        `<result>` block, after which a fresh assistant turn is opened and the model continues
        (raw path: `_raw_prompt`, every request a byte-for-byte extension of the previous one plus
        the text the model generated, so the server's KV cache holds all but the new turn; chat
        path: two messages appended). It ends when a pass writes no call:

        - a malformed call is told so (the first is free, later ones cost a call);
        - the call that spends the last one (or the time budget) gets "(no calls left ...)" and the
          next pass runs without the call stop, so a call it still writes is cut there;
        - a first pass that only SAYS it will look something up ("Let me check ...", `_announces_lookup`)
          is re-streamed once with the call opened for it, the narration dropped from the answer;
        - a pass that writes nothing after tools ran (whatever lead-in came before its call) gets one
          more pass, then an error."""
        from exocortex.capabilities import ToolCall      # lazy: the loop is the only user

        runner = self._tools
        budget = tool_ctx.budget
        raw = self._cfg.llm.thinking != "on"
        initial_calls = max(1, int(budget.calls_left))
        hard_cap = 2 * initial_calls + 2        # a net under a runner that never charges a repeated call
        write_hops = job.kind == "ask" and bool(_WRITE_TOOLS & set(budget.allowed))
        steps: list[_ToolStep] = []
        answer = ""
        all_stats: list[StreamStats] = []
        sent_chars = 0
        calls = 0                               # ToolCallEvents published (their 1-based index)
        malformed = 0
        forced_used = blank_used = False
        forced_lead: str | None = None          # next pass: the narration the call is forced after
        blank = False                           # next pass: the "write the answer" retry
        final = False                           # no more calls: the next pass has no call stop
        hop = 0
        while not job.cancel.is_set():
            hop += 1
            if not final and (budget.calls_left <= 0 or time.monotonic() >= budget.deadline):
                final = True
            # The SERVER stops a call pass at the closing tag: the tag's tokens stay in the slot's
            # cache, the request completes normally, and the continuation (an extension of this
            # very text) lands on the same slot. The answer pass, which may make no call, stops at
            # an opening tag instead. 2026-10-08, from the owner's first live session: the client-
            # side cut this replaces turned every hop into a cancelled request -- the server
            # noticed the dropped connection only after the NEXT request had been handed the other
            # slot, which re-evaluated the whole prompt (2998 tokens, 3 s on the 27B) every hop.
            # A `</tool>` the model writes in its own prose now cuts the answer there: accepted,
            # the rule tells it never to mention the tools.
            stop = list(prompts.STOP) + [prompts.TOOL_OPEN if final else prompts.TOOL_CLOSE]
            max_tokens = prepared.max_tokens
            if write_hops and not final:
                max_tokens = max(max_tokens, _WRITE_HOP_MIN_TOKENS)      # a write_file call carries a body
            forced, forced_lead = forced_lead, None
            was_blank, blank = blank, False
            splitter, stats, sent = self._stream_hop(
                job, prepared, steps, on_delta, raw=raw, stop=stop, max_tokens=max_tokens,
                cut_on_call=final, forced_lead=forced, blank=was_blank,
            )
            sent_chars = sent
            if stats is not None:
                all_stats.append(stats)
            visible, held = splitter.visible, splitter.held
            reason = str(getattr(stats, "finish_reason", "") or "")
            if job.cancel.is_set():
                answer += visible if forced is None else ""
                break

            if held is None or final:
                # No call (or one when none may be made: the stream was cut at it, its visible part kept).
                if (
                    held is None and hop == 1 and not forced_used and not steps and not final
                    and _announces_lookup(visible)
                ):
                    forced_used = True
                    forced_lead = visible if visible.endswith("\n") else visible + "\n"
                    log.info("assist job %d (%s): narration without a tool call; forcing one", job.id, job.kind)
                    continue
                if steps and not visible.strip():      # judge THIS pass: `answer` already holds the lead-ins
                    if blank_used:
                        raise _NoAnswerError(f"the model made {calls} tool call(s) but wrote no answer")
                    blank_used = blank = final = True      # (a blank pass's text is dropped: the retry lacks it)
                    log.info("assist job %d (%s): blank answer after %d call(s); asking once more", job.id, job.kind, calls)
                    continue
                answer += visible
                if forced is not None and not answer.strip():
                    answer = forced.rstrip("\n")       # the loop forced a call it could not get: keep the narration
                break

            # ---- the pass ended with a call
            if forced is not None and not held[len(prompts.TOOL_OPEN):].strip():
                answer += forced.rstrip("\n")          # asked for a call, the model declined: the narration is its answer
                break
            lead = forced if forced is not None else visible
            if forced is None:
                answer += visible
            payload: str | None
            parsed: tuple[str, dict[str, Any]] | str
            if reason == "length":
                payload, parsed = None, "the call was cut off by the length limit"
            else:
                # The stream was abandoned at the closing tag or at the end of the JSON object; a
                # call with no complete object is read as written: whatever follows the marker.
                payload = prompts.find_tool_call(held)
                if payload is None:
                    payload = held[len(prompts.TOOL_OPEN):]
                parsed = prompts.parse_tool_call(payload)
            if isinstance(parsed, str):
                malformed += 1
                shown = (payload if payload is not None else held[len(prompts.TOOL_OPEN):]).strip()[:300]
                step = _ToolStep(
                    lead, shown, "tool",
                    f'(the call could not be parsed: {parsed}. Write exactly: '
                    f'<tool>{{"name": ..., "args": {{...}}}}</tool>)',
                    sanitize=False,
                )
                if malformed > 1:
                    budget.calls_left -= 1             # the first malformed call is free, the others count
                if budget.calls_left <= 0:
                    step.note = _NO_CALLS_LEFT
                    final = True
                steps.append(step)
                log.info("assist job %d (%s): malformed tool call #%d: %s", job.id, job.kind, malformed, parsed)
                continue

            name, args = parsed
            calls += 1
            summary = _call_summary(args)
            self._bus.publish(ToolCallEvent(job_id=job.id, index=calls, name=name, summary=summary, state="running"))
            if job.cancel.is_set():                    # last look before anything runs
                self._bus.publish(ToolCallEvent(
                    job_id=job.id, index=calls, name=name, summary=summary, state="error", detail="cancelled"))
                break
            started = time.monotonic()
            if started >= budget.deadline:
                out = _Outcome("denied: the time for tool calls in this answer is used up", "denied",
                               "time budget used up", summary)
            else:
                job.tools_used += 1
                try:
                    out = _outcome(runner.run(
                        ToolCall(name=name, args=args, raw=payload or ""), budget=budget, cancel=job.cancel))
                except Exception as exc:  # noqa: BLE001 - `run` never raises; a fake or a bug must not kill the answer
                    log.exception("assist job %d (%s): tool %s raised", job.id, job.kind, name)
                    out = _Outcome(f"error: {exc}", "error", str(exc)[:120], summary)
            elapsed = time.monotonic() - started
            detail = out.detail or (_one_line(out.text) if out.state != "done" else "")
            self._bus.publish(ToolCallEvent(
                job_id=job.id, index=calls, name=name, summary=_one_line(out.summary, 120) or summary,
                state=out.state, detail=detail[:200], elapsed_s=elapsed))
            log.info("assist job %d (%s): tool %s %s (%s) in %.2fs", job.id, job.kind, name, out.state, summary, elapsed)
            step = _ToolStep(lead, (payload or "").strip(), name, out.text)
            if budget.calls_left <= 0 or time.monotonic() >= budget.deadline or calls >= hard_cap:
                step.note = _NO_CALLS_LEFT
                final = True
            steps.append(step)
        return answer, sent_chars, self._merge_stats(all_stats)

    def _stream_hop(
        self, job: _AssistJob, prepared: _Prepared, steps: list[_ToolStep], on_delta: Callable[[str], None], *,
        raw: bool, stop: list[str], max_tokens: int, cut_on_call: bool, forced_lead: str | None, blank: bool,
    ) -> tuple[_ToolSplitter, StreamStats | None, int]:
        """One request of the loop, streamed through a fresh `_ToolSplitter`: returns the splitter
        (`visible` / `held`), the stream's stats and the characters sent. A pass after a tool ran
        whose request the server rejects as too long is retried ONCE with the oldest result's body
        dropped (`_RESULT_DROPPED`); the first pass has no result to drop -- its overflow goes to
        `_run_assist_job`'s own retry, which only ever runs when no tool has."""
        retried = False
        while True:
            splitter = _ToolSplitter(on_delta, held=prompts.TOOL_OPEN if forced_lead is not None else None)
            try:
                stats, sent = self._send_hop(
                    job, prepared, steps, splitter, raw=raw, stop=stop, max_tokens=max_tokens,
                    cut_on_call=cut_on_call, forced_lead=forced_lead, blank=blank,
                )
                return splitter, stats, sent
            except LLMError as exc:
                droppable = next((s for s in steps if not s.dropped), None)
                if (
                    steps and droppable is not None and not retried and not splitter.published
                    and not job.cancel.is_set() and _OVERFLOW_RE.search(str(exc))
                ):
                    droppable.dropped = True
                    retried = True
                    log.warning("assist job %d (%s): the continuation overflowed the server's context (%s); "
                                "dropping the oldest tool result and retrying the pass once", job.id, job.kind, exc)
                    self._note_overflow()
                    continue
                raise

    def _send_hop(
        self, job: _AssistJob, prepared: _Prepared, steps: list[_ToolStep], splitter: _ToolSplitter, *,
        raw: bool, stop: list[str], max_tokens: int, cut_on_call: bool, forced_lead: str | None, blank: bool,
    ) -> tuple[StreamStats | None, int]:
        """Send the continuation as it stands and feed the splitter; `(stats, characters sent)`.
        The server's stop strings end a pass (`_run_tool_loop`); the stream is abandoned client-side
        only as a fallback: a server that ignored the stop string (`</tool>` seen in the held text),
        a model that forgot the closing tag and wrote on past its JSON object
        (`_forgot_closing_tag`), or, with `cut_on_call` (no calls left), a call that starts anyway."""
        if raw:
            prompt = self._raw_prompt(prepared, steps, forced_lead, blank)
            sent = len(prompt)
            stream = self._client.stream_completion(prompt, max_tokens=max_tokens, stop=stop, cancel=job.cancel)
        else:
            messages = self._chat_messages(prepared, steps, forced_lead, blank)
            sent = sum(len(m["content"]) for m in messages)
            stream = self._client.stream_chat(messages, max_tokens=max_tokens, cancel=job.cancel, stop=stop)
        try:
            for item in stream:
                piece = item if raw else item.content
                if not piece:
                    continue
                splitter.feed(piece)
                if splitter.held is not None and (
                    cut_on_call or prompts.TOOL_CLOSE in splitter.held or _forgot_closing_tag(splitter.held)
                ):
                    break
        finally:
            close = getattr(stream, "close", None)
            if callable(close):
                close()                    # lets the client's stream finish (and record its stats) now
        splitter.finish()
        return self._client.last_stats, sent

    @staticmethod
    def _raw_prompt(prepared: _Prepared, steps: list[_ToolStep], forced_lead: str | None, blank: bool) -> str:
        """The raw-path prompt of a pass. The first assistant turn carries `NO_THINK_PREFIX` and
        the text + call the model wrote; each result then comes back as a USER turn (its <result>
        block) and a fresh assistant turn, again with `NO_THINK_PREFIX`, is opened for the model to
        continue in -- the shape chat models are trained on: a tool's reply is a message TO the
        model, never text inside its own turn. 2026-10-08, the owner's first live session: results
        appended inside the assistant turn made the Qwen model end its turn right after them (one
        token, end of text, on the continuation AND on the retry) -- the "made 1 tool call(s) but
        wrote no answer" failure. Every prompt is a byte-for-byte extension of the previous request
        plus what the model generated, so the server's KV cache holds all but the new turn.
        `forced_lead` opens a call after the narration it follows (a first pass without a call);
        `blank` is the answer retry: a nudge inside the last result block."""
        prompt = prompts.render_chatml(prepared.system, prepared.user, prompts.NO_THINK_PREFIX)
        last = len(steps) - 1
        for i, s in enumerate(steps):
            block = s.block(nudge=_CHAT_WRITE_ANSWER if blank and i == last else "")
            prompt += (
                f"{s.call_text()}{prompts.IM_END}\n"
                f"{prompts.IM_START}user\n{block.strip(chr(10))}{prompts.IM_END}\n"
                f"{prompts.IM_START}assistant\n{prompts.NO_THINK_PREFIX}"
            )
        if forced_lead is not None:
            prompt += forced_lead + prompts.TOOL_OPEN
        return prompt

    @staticmethod
    def _chat_messages(
        prepared: _Prepared, steps: list[_ToolStep], forced_lead: str | None, blank: bool,
    ) -> list[dict[str, str]]:
        """The chat-path messages of a pass: system, user, then per call an assistant message (the
        visible lead and the call) and a user message (the result block) -- the same turns the raw
        path renders. A chat API cannot prefill an assistant turn, so the two retries ask in words:
        the forced call as a user message of its own, the answer nudge inside the last result block
        (roles must keep alternating for strict templates)."""
        messages = [
            {"role": "system", "content": prepared.system},
            {"role": "user", "content": prepared.user},
        ]
        last = len(steps) - 1
        for i, s in enumerate(steps):
            messages.append({"role": "assistant", "content": s.call_text()})
            block = s.block(nudge=_CHAT_WRITE_ANSWER if blank and i == last else "")
            messages.append({"role": "user", "content": block.strip("\n")})
        if forced_lead is not None:
            messages.append({"role": "assistant", "content": forced_lead.rstrip()})
            messages.append({"role": "user", "content": _CHAT_FORCE_CALL})
        return messages

    @staticmethod
    def _merge_stats(stats: list[StreamStats]) -> StreamStats | None:
        """One `StreamStats` for a whole loop: the FIRST stream's time to first token (what the
        reader waited for), the tokens and times of every stream summed, and the LAST stream's
        prompt / cached tokens, finish reason and cancelled flag. One stream is returned as is."""
        if not stats:
            return None
        if len(stats) == 1:
            return stats[0]
        first, last = stats[0], stats[-1]
        tokens = sum(s.tokens for s in stats)
        elapsed = sum(s.elapsed_s for s in stats)
        gen_time = sum(max(s.elapsed_s - s.ttft_s, 0.0) if s.tokens > 1 else s.elapsed_s for s in stats)
        return dataclasses.replace(
            last, ttft_s=first.ttft_s, tokens=tokens, elapsed_s=elapsed,
            tok_per_s=tokens / gen_time if gen_time > 0 else 0.0,
        )

    # ---- auto mode (bus-listener + coalesce timer) --------------------------------

    def _bus_listener_loop(self) -> None:
        log.info("bus-listener: started")
        assert self._sub_queue is not None
        sub = self._sub_queue
        while not self._stop_event.is_set():
            try:
                event = sub.get(timeout=0.2)
            except queue.Empty:
                continue
            self._handle_bus_event(event)
        log.info("bus-listener: stopping")

    def _handle_bus_event(self, event: Any) -> None:
        if isinstance(event, ScreenTextEvent):
            self._handle_screen_event(event)
            return
        if not isinstance(event, FinalTranscriptEvent):
            return
        seg = event.segment
        if seg.source != Source.SYSTEM:
            return
        if self._last_fired_auto_seg_id is not None and seg.id <= self._last_fired_auto_seg_id:
            return  # already covered by a previous auto answer
        if self._paused or self._auto_suppressed or self._manual_advance() or not self._auto_enabled:
            return
        is_q = seg.is_question or questions.is_question(seg.text, self._cfg.assist.min_question_words)
        if not is_q:
            return
        self._arm_coalesce_timer()

    # ---- screen reading (bus-listener) ---------------------------------------------

    def _handle_screen_event(self, event: ScreenTextEvent) -> None:
        """React to an OCR update: a new on-screen question, and/or auto-suggest eligibility.

        `prev_lines` is updated on every changed event regardless of either feature's config
        flag or outcome, so a later toggle sees only genuinely new lines from that point on.
        """
        if not event.changed:
            return
        prev = self._screen_prev_lines
        current = tuple(event.lines)
        new = self._compute_new_lines(prev, current)
        self._screen_prev_lines = current
        self._maybe_answer_screen_question(new)
        self._maybe_auto_suggest(event)
        self._maybe_explain_from_screen(current)

    @staticmethod
    def _compute_new_lines(prev: tuple[str, ...], current: tuple[str, ...]) -> list[str]:
        """`exocortex.screen.ocr.new_lines(prev, current)`, imported lazily; a plain
        (unnormalised) set difference if that module is unavailable."""
        try:
            from exocortex.screen.ocr import new_lines as _new_lines
        except Exception:
            log.debug("assistant: exocortex.screen.ocr unavailable; using a plain set difference")
            seen = set(prev)
            return [line for line in current if line not in seen]
        return _new_lines(prev, current)

    def _maybe_answer_screen_question(self, new_lines: list[str]) -> None:
        """Submit a kind "screen" job for a new question line on screen, subject to
        `cfg.screen.answer_screen_questions`, a bounded recently-answered de-dup set, and
        `SCREEN_QUESTION_MIN_INTERVAL_S` between screen-triggered answers."""
        if self._paused or self._auto_suppressed or self._hands_free or self._manual_advance() \
                or not self._cfg.screen.answer_screen_questions:
            return
        question = self._first_new_question(new_lines)
        if not question:
            return
        if not questions.is_relevant(question, self._cfg.assist.focus, self._cfg.assist.ignore_patterns):
            log.info("assistant: ignoring on-screen question (not %s): %r", self._cfg.assist.focus, question[:80])
            return
        norm = _normalize_question(question)
        now = self._clock()
        with self._screen_question_lock:
            if norm in self._screen_answered_questions:
                return
            if now - self._last_screen_answer_time < SCREEN_QUESTION_MIN_INTERVAL_S:
                return
            self._last_screen_answer_time = now
            self._screen_answered_questions.append(norm)
            del self._screen_answered_questions[:-50]
        self._submit_assist("screen", question, cancel_running=False)

    @staticmethod
    def _first_new_question(new_lines: list[str]) -> str:
        """The first new OCR line reading as a question (`is_question(..., min_words=6)`),
        joined with the line right after it if that one is also a new question line (so a
        question split across two OCR lines is answered as one). "" if there is none."""
        for i, line in enumerate(new_lines):
            if not questions.is_question(line, min_words=6):
                continue
            if i + 1 < len(new_lines) and questions.is_question(new_lines[i + 1], min_words=6):
                return f"{line} {new_lines[i + 1]}"
            return line
        return ""

    def _maybe_auto_suggest(self, event: ScreenTextEvent) -> None:
        """React to a screen change: in autonomous mode SOLVE what is now on screen, otherwise
        propose talking points (`suggest`). Fires only when `cfg.screen.auto_suggest` is on,
        enough time passed since the last side-lane job started, and the new lines aren't
        near-identical (Jaccard >= 0.6) to the ones that triggered the last one.

        The similarity check itself compares normalised lines (`_normalize_line`): two OCR
        passes over unchanged on-screen content routinely differ in case or incidental
        whitespace, and comparing the raw strings would treat that noise as entirely different
        content -- 0.0 Jaccard similarity for a single-line screen -- re-triggering a spurious
        job for content the user has already been helped with.
        """
        if self._paused or self._hands_free or self._manual_advance() or not self._cfg.screen.auto_suggest:
            return
        kind: AssistKind = "solve" if self._cfg.assist.autonomous else "suggest"
        explain = self._cfg.explain.enabled
        if self._auto_suppressed and not (kind == "solve" and explain):
            # Completion mode: no panel for suggestions -- but with explanations on the solve IS
            # the plan the completions follow, so it still runs.
            return
        with self._suggest_state_lock:
            elapsed = self._clock() - self._last_suggest_time
            prev_lines = self._last_suggested_lines
        if elapsed < self._cfg.screen.suggest_min_interval_s:
            return
        current = _normalized_line_set(event.lines)
        if _jaccard(current, prev_lines) >= 0.6:
            return
        if kind == "solve" and explain:
            with self._plan_lock:
                plan = self._plan
            if plan is not None and self._plan_matches_screen(plan, current):
                return   # still the same task (the user is typing the solution): keep the plan
        self._enqueue_side_job(kind, focus="", screen_lines=tuple(event.lines))

    def _arm_coalesce_timer(self) -> None:
        """(Re)start the coalesce timer so a follow-on sentence extends the wait, not doubles it."""
        interval_s = max(self._cfg.assist.coalesce_ms, 0) / 1000.0
        with self._coalesce_lock:
            if self._coalesce_timer is not None:
                self._coalesce_timer.cancel()
            self._coalesce_generation += 1
            generation = self._coalesce_generation
            timer = threading.Timer(interval_s, self._fire_auto, args=(generation,))
            timer.daemon = True
            timer.name = "assist-coalesce"
            self._coalesce_timer = timer
            timer.start()

    def _fire_auto(self, generation: int) -> None:
        """Run the coalesced auto-answer, unless a newer timer has since superseded this one.

        `threading.Timer.cancel()` only prevents a timer whose wait has not yet elapsed from
        firing: if `_arm_coalesce_timer` re-arms (or `stop()` closes) at almost the exact moment
        an older timer's wait expires, `cancel()` can return successfully while that older
        timer's callback still runs. Tagging each armed timer with a generation number and
        checking it here makes that race harmless instead of firing a second, redundant (or
        post-shutdown) auto-answer.
        """
        with self._coalesce_lock:
            if generation != self._coalesce_generation:
                log.debug("bus-listener: stale coalesce timer (gen %d) superseded, ignoring", generation)
                return
            self._coalesce_timer = None
        if self._paused or self._auto_suppressed or self._manual_advance() or not self._auto_enabled:
            return
        segs = self._recent_system_segments(_AUTO_COALESCE_WINDOW_S)
        if not segs:
            return
        self._last_fired_auto_seg_id = segs[-1].id
        self._submit_assist("auto", self._build_question(segs))

    def _recent_system_segments(self, window_s: float) -> list[TranscriptSegment]:
        """Up to the last `_MAX_CONTEXT_SEGMENTS` SYSTEM segments, restricted to `window_s` of
        each other (by `t_end`), oldest -> newest."""
        segs = self._transcript.last(Source.SYSTEM, _MAX_CONTEXT_SEGMENTS)
        if not segs:
            return []
        newest_end = segs[-1].t_end
        return [s for s in segs if newest_end - s.t_end <= window_s]

    @staticmethod
    def _build_question(segs: list[TranscriptSegment]) -> str:
        """Join `segs`' text into one string and extract the question to answer from it.

        `questions.extract_question` only looks at the last element of the list it is given, so
        the segments are joined into a single string first to get cross-segment context (see
        SPEC.md llm/assistant.py "Integration facts").
        """
        joined = " ".join(s.text for s in segs)
        return questions.extract_question([joined]) or segs[-1].text

    # ---- co-pilot insights (insight-timer + bookkeeping) ------------------------------

    def _enqueue_insights(self, sections: tuple[str, ...], *, auto: bool) -> int | None:
        """Queue an insights job for `sections` on the insight lane; its id, or None when a
        background pass was refused because an on-demand request is already queued.

        On demand (`auto=False`): supersedes a running insights job of either kind and replaces a
        queued one. Background (`auto=True`): cancels nothing and only takes an empty slot or one
        holding another background pass (`_JobSlot.put_if`)."""
        label = {
            ("questions",): "Questions to ask", ("blindspots",): "Blind spots & follow-ups",
        }.get(sections, "Questions & blind spots")
        job = _AssistJob(
            id=next(self._job_ids), kind="insights", question=label, cancel=threading.Event(),
            sections=sections, auto=auto,
        )
        if auto:
            if not self._insight_slot.put_if(job, lambda queued: queued.auto):
                return None
            with self._insights_lock:
                # Throttle from the moment it is queued (and again when it starts running), so the
                # next tick does not queue a duplicate while this one waits behind other work.
                self._insights_last_start = self._clock()
        else:
            self._cancel_running_of(_CANCELS["insights"])
            self._insight_slot.put(job)
        log.debug("assistant: submitted %s insights job %d: %s", "auto" if auto else "manual", job.id, sections)
        return job.id

    def _preempt_auto_insights(self) -> None:
        """Cancel a RUNNING background insights pass: whatever else was just submitted matters
        more. It is retried `INSIGHT_RETRY_S` after it started (`_settle_insights`); an on-demand
        insights job is left alone."""
        with self._current_lock:
            job = self._current_job
            if job is not None and job.kind in ("insights", "dashboard") and job.auto:
                job.cancel.set()

    def _drop_auto_insights(self) -> None:
        """Pause / completion mode: cancel a running background pass (insights or a dashboard
        refresh) and drop a queued one."""
        self._preempt_auto_insights()
        self._insight_slot.remove_if(lambda queued: queued.auto)
        self._dashboard_slot.remove_if(lambda queued: queued.auto)

    def _auto_insight_sections(self) -> tuple[str, ...]:
        """The sections background passes refresh, per `[assist].suggest_questions` /
        `flag_blind_spots` (read live); () when both are off."""
        a = self._cfg.assist
        wanted = [s for s, on in (("questions", a.suggest_questions), ("blindspots", a.flag_blind_spots)) if on]
        return prompts.normalize_sections(wanted)

    def _maybe_auto_insights(self) -> bool:
        """One insight-timer tick: queue a background insights pass when one is due; True when it
        queued one.

        Due means: a section is enabled, `[assist].insight_interval_s` is positive and has passed
        since the last pass started, nothing automatic is held back (paused, completion mode,
        manual advance), and there is something new to look at -- `insight_min_new_words` of
        speech since the last completed pass, or a screen that changed substantially since then
        (`INSIGHT_SCREEN_JACCARD`, at least `INSIGHT_MIN_SCREEN_LINES` lines): the driver writing
        code in silence still gets blind spots on it.

        Never while the dashboard tracks the session (`dashboard_tracking`): its refreshes cover
        the same questions and blind spots and mirror them into the insights document."""
        if self.dashboard_tracking():
            return False
        sections = self._auto_insight_sections()
        if not sections:
            return False
        a = self._cfg.assist
        try:
            interval = float(a.insight_interval_s)
            min_words = max(0, int(a.insight_min_new_words))
        except (TypeError, ValueError):
            return False
        if interval <= 0 or self._paused or self._auto_suppressed or self._manual_advance():
            return False
        now = self._clock()
        with self._insights_lock:
            if now - self._insights_last_start < interval:
                return False
            since_id = self._insights_seg_id
            seen_lines = set(self._insights_screen_lines)
        new_words = self._transcript.word_count(since_id=since_id)
        due = new_words > 0 and new_words >= min_words
        if not due:
            current = self._current_screen_lines()
            due = len(current) >= INSIGHT_MIN_SCREEN_LINES and _jaccard(current, seen_lines) < INSIGHT_SCREEN_JACCARD
        if not due:
            return False
        return self._enqueue_insights(sections, auto=True) is not None

    def _settle_insights(
        self, job: _AssistJob, *, ok: bool, text: str = "", until_id: int | None = None, screen_text: str = "",
    ) -> None:
        """Record how an "insights" job ended (a no-op for every other kind).

        `ok` (it ran to the end, including a model that had nothing to add): the refreshed
        sections replace theirs in the document, the background-pass watermark moves to what this
        pass saw (the transcript tip when it started, its screen lines), and an
        `InsightsUpdatedEvent` goes out when a section actually changed. Not `ok` (cancelled,
        failed): nothing changes, and a background pass is retried `INSIGHT_RETRY_S` after it
        started instead of a whole interval later -- unless a newer pass has started since."""
        if job.kind != "insights":
            return
        if not ok:
            if job.auto:
                try:
                    interval = float(self._cfg.assist.insight_interval_s)
                except (TypeError, ValueError):
                    interval = 0.0
                with self._insights_lock:
                    if self._insights_last_start == job.started:
                        self._insights_last_start = job.started - max(0.0, interval - INSIGHT_RETRY_S)
            return
        secs = job.sections or prompts.INSIGHT_SECTIONS
        parsed: dict[str, str] | None = None
        if text.strip() and not prompts.is_skip(text):
            parsed = prompts.parse_insights(text, secs, max_items=self._cfg.assist.insight_max_items)
        changed = False
        with self._insights_lock:
            if parsed is not None:
                for section, body in parsed.items():
                    if self._insights.get(section) != body:
                        self._insights[section] = body
                        changed = True
            if until_id is not None:
                self._insights_seg_id = until_id
            self._insights_screen_lines = _normalized_line_set(screen_text.splitlines())
            snapshot = dict(self._insights)
        if changed:
            self._bus.publish(InsightsUpdatedEvent(
                questions=snapshot.get("questions", ""), blindspots=snapshot.get("blindspots", ""),
                generated_at=self._clock(), auto=job.auto,
            ))
            log.info("assistant: insights updated (%s, %s)", "auto" if job.auto else "on demand", ", ".join(secs))

    def _insight_timer_loop(self) -> None:
        log.info("insight-timer: started")
        while not self._stop_event.wait(INSIGHT_TICK_S):
            try:
                self._maybe_auto_insights()
            except Exception:  # noqa: BLE001 - one bad tick must never kill the timer
                log.exception("insight-timer: tick failed")
        log.info("insight-timer: stopping")

    # ---- notes auto timer -------------------------------------------------------

    def _notes_auto_loop(self) -> None:
        interval = self._cfg.notes.auto_interval_s
        if interval <= 0:
            return
        log.info("notes-auto-timer: started (every %ds)", interval)
        while not self._stop_event.wait(interval):
            if self._manual_advance():
                continue   # manual advance: notes regenerate only on demand (the button / hotkey)
            self.generate_notes(force=False)
        log.info("notes-auto-timer: stopping")

    # ---- LM Studio health / model list -----------------------------------------------

    def _run_health_check(self) -> None:
        """Check LM Studio reachability/model state and publish ComponentStatusEvent("llm", ...)."""
        try:
            health = self._client.health()
        except LLMError as exc:
            log.error("assistant: health check failed: %s", exc)
            self._bus.publish(ComponentStatusEvent(component="llm", state="error", detail=str(exc)))
            return

        if not health.reachable:
            detail = f"LM Studio not reachable at {self._cfg.llm.base_url}: {health.detail}"
            log.warning("assistant: %s", detail)
            self._bus.publish(ComponentStatusEvent(component="llm", state="error", detail=detail))
            return

        if health.model_loaded is not True and self._cfg.llm.auto_load_model:
            self._bus.publish(
                ComponentStatusEvent(component="llm", state="loading", detail=f"loading {self._cfg.llm.model}...")
            )
            try:
                self._client.ensure_loaded()
            except LLMError as exc:
                log.error("assistant: failed to load model %r: %s", self._cfg.llm.model, exc)
                self._bus.publish(ComponentStatusEvent(component="llm", state="error", detail=str(exc)))
                return

        self._read_context_window()
        self._bus.publish(ComponentStatusEvent(component="llm", state="ready", detail=self._cfg.llm.model))
        self._refresh_models_async()
        self._kick_digests()

    def _read_context_window(self) -> None:
        """Size every later prompt to what the server really gives one request (2026-10-06):
        `client.context_window()` (llama-server's /props, LM Studio's model entry; None when it
        cannot say -- then the manager falls back to `[llm].ctx_size`). Guarded: a client without
        the method, or one that raises, just means "unknown"."""
        n: int | None = None
        try:
            reader = getattr(self._client, "context_window", None)
            value = reader() if callable(reader) else None
            n = int(value) if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None
        except Exception:  # noqa: BLE001 - any failure means "unknown"
            log.debug("assistant: reading the server's context window failed", exc_info=True)
        try:
            self._context.set_slot_tokens(n)
        except Exception:  # noqa: BLE001
            log.exception("assistant: setting the context slot size failed")
            return
        if n is None:
            log.info("assistant: server context window unknown; using [llm].ctx_size (%d tokens)",
                     self._context.slot_tokens)
        else:
            log.info("assistant: server context window is %d tokens per request", n)

    def _refresh_models_async(self) -> None:
        """Refresh the cached model list in the background so health checks stay fast."""

        def _work() -> None:
            try:
                models = self._client.list_models()
            except LLMError as exc:
                log.debug("assistant: model list refresh failed: %s", exc)
                return
            with self._models_lock:
                self._cached_models = [m.id for m in models]

        threading.Thread(target=_work, name="assist-models-refresh", daemon=True).start()

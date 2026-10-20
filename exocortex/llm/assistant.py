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
  cancellable).
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

import itertools
import logging
import queue
import re
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from exocortex.bus import (
    AssistDeltaEvent,
    AssistDoneEvent,
    AssistErrorEvent,
    AssistKind,
    AssistStartedEvent,
    AutoAssistChangedEvent,
    CheckpointEvent,
    ComponentStatusEvent,
    EventBus,
    FinalTranscriptEvent,
    InsightsUpdatedEvent,
    NotesErrorEvent,
    NotesUpdatedEvent,
    ScreenTextEvent,
    Source,
    TranscriptSegment,
)
from exocortex.config import Config
from exocortex.llm import prompts, questions
from exocortex.llm.choices import ChoiceSet, annotate_choices, choices_block, finalize_answer, find_choices
from exocortex.llm.client import LLMError, LMStudioClient
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
}

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


@dataclass
class _NotesJob:
    """One queued or running notes-worker job."""

    id: int
    force: bool


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
    ) -> None:
        self._cfg = cfg
        self._bus = bus
        self._client = client
        self._transcript = transcript
        self._clock = clock

        self._job_ids = itertools.count(1)
        self._assist_slot: _JobSlot[_AssistJob] = _JobSlot()      # main lane: auto/manual/ask/screen
        self._suggest_slot: _JobSlot[_AssistJob] = _JobSlot()     # side lane: suggest/solve/code, never cancels the main lane
        self._text_slot: _JobSlot[_AssistJob] = _JobSlot()        # text lane: the Text button's walkthrough (see _CANCELS)
        self._recap_slot: _JobSlot[_AssistJob] = _JobSlot()       # recap lane: the Recap action ("where are we")
        self._explain_slot: _JobSlot[_AssistJob] = _JobSlot()     # explain lane: explanation of the finished solution
        self._insight_slot: _JobSlot[_AssistJob] = _JobSlot()     # insight lane: questions to ask / blind spots
        self._complete_slot: _JobSlot[_AssistJob] = _JobSlot()    # completion lane: inline completions (lowest priority)
        self._notes_slot: _JobSlot[_NotesJob] = _JobSlot()

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
        self._threads: list[threading.Thread] = []

    # ---- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        """Start all worker threads and run an initial LM Studio health check."""
        if self._threads:
            log.warning("assistant: start() called while already running; stopping first")
            self.stop()
        self._stop_event.clear()
        for slot in self._assist_slots():
            slot.reopen()
        self._notes_slot.reopen()
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

    def code(self, focus: str = "", label: str = "") -> int:
        """Write the code for the task discussed (transcript since the last checkpoint) and/or
        shown on screen, right now (kind "code", the Code button). Returns the job id.
        `label`, when given, is what the Assist tab / history show as the question instead of
        `focus` (hands-free mode passes a long instruction as `focus`).

        Runs on the same side lane as `suggest`/`solve` -- it never cancels a running main-lane
        (auto/manual/ask/screen) job -- but a new "suggest"/"solve"/"code" job cancels a running
        "code" and vice versa (`_CANCELS`). Unlike `solve()`, this is MANUAL: it is NOT refused
        while paused (see `exocortex.actions.EngineActions.code`). `focus` is an optional extra
        instruction; blank falls back to `prompts.user_code`'s default ask.
        """
        return self._enqueue_side_job("code", focus=focus, screen_lines=None, label=label)

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
            self._checkpoint_seg_id = seg_id
            self._checkpoint_answers = answers
            self._checkpoint_time = t
            self._code_answers = []
        with self._plan_lock:
            self._plan = None
            self._plan_parsed = None
        self._bus.publish(CheckpointEvent(t=t, segment_id=seg_id))
        log.info("assistant: checkpoint set (segment %s, %d answer(s) so far)", seg_id, answers)

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
        plan_id: int | None = None, label: str = "",
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
        display_question = label.strip() or focus.strip() or default_label
        job = _AssistJob(
            id=next(self._job_ids), kind=kind, question=display_question,
            cancel=threading.Event(), focus=focus, screen_lines=screen_lines, plan_id=plan_id,
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
            self._insight_slot, self._complete_slot,
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
            try:
                self._run_assist_job(job)
            except Exception:  # pragma: no cover - safety net; _run_assist_job handles its own errors
                log.exception("assist-worker: unhandled error running job %d", job.id)
            finally:
                with self._current_lock:
                    if self._current_job is job:
                        self._current_job = None

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
        if job.kind == "insights":
            job.started = self._clock()
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
        window = self._transcript.window_text(self._cfg.llm.context_tokens * 4, until_id=until_id, since_id=since_id)
        assistant_prefix: str | None = None
        temperature: float | None = None
        choices, choices_on_screen = self._detect_choices(job, screen_text)
        if job.kind == "complete":
            system, user, max_tokens, assistant_prefix, temperature = self._complete_prompt_parts(job, window, screen_text)
        else:
            # The annotated copies exist only for the prompt: every screen-line fingerprint (plan
            # matching, suggest throttling, screen-question de-dup) keeps comparing raw OCR text.
            prompt_screen, prompt_question = screen_text, job.question
            if choices is not None:
                if choices_on_screen:
                    prompt_screen = annotate_choices(screen_text, choices)
                else:
                    prompt_question = annotate_choices(job.question, choices)
            system, user, max_tokens = self._build_prompt_parts(job, window, prompt_screen, choices, prompt_question)

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

        try:
            extra = None
            if job.kind == "complete" and self._cfg.completion.repeat_penalty and self._cfg.completion.repeat_penalty != 1.0:
                extra = {"repeat_penalty": float(self._cfg.completion.repeat_penalty)}
            text = self._stream_text(
                system, user, max_tokens=max_tokens, cancel=job.cancel, on_delta=on_delta,
                assistant_prefix=assistant_prefix, temperature=temperature, extra_body=extra,
            )
        except LLMError as exc:
            log.error("assist job %d (%s) failed: %s", job.id, job.kind, exc, exc_info=True)
            self._settle_plan_state(job, is_plan, "", keep=False)
            self._settle_insights(job, ok=False)
            self._bus.publish(AssistErrorEvent(job_id=job.id, message=str(exc)))
            self._bus.publish(ComponentStatusEvent(component="llm", state="error", detail=str(exc)))
            return
        except Exception as exc:  # noqa: BLE001 - surface any bug as a job error, never kill the worker
            log.error("assist job %d (%s) failed unexpectedly: %s", job.id, job.kind, exc, exc_info=True)
            self._settle_plan_state(job, is_plan, "", keep=False)
            self._settle_insights(job, ok=False)
            self._bus.publish(AssistErrorEvent(job_id=job.id, message=str(exc)))
            return

        if not job.cancel.is_set() and prompts.is_skip(text):
            # The model found nothing technical to do (a poll, chrome, small talk): drop it quietly.
            log.info("assist job %d (%s): model skipped (%s)", job.id, job.kind, prompts.SKIP_MARKER)
            self._settle_plan_state(job, is_plan, "", keep=False)
            # Nothing to add is still a completed pass: the same material is not re-sent next tick.
            self._settle_insights(job, ok=True, until_id=until_id, screen_text=screen_text)
            self._bus.publish(AssistDoneEvent(job_id=job.id, text="", ttft_s=0.0, tokens=0, tok_per_s=0.0,
                                              cancelled=True, skipped=True))
            return

        stats = self._client.last_stats
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
        self._bus.publish(
            AssistDoneEvent(
                job_id=job.id, text=final_text, ttft_s=ttft_s, tokens=tokens, tok_per_s=tok_per_s, cancelled=cancelled
            )
        )
        if not cancelled and final_text.strip() and job.kind not in ("complete", "insights"):
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
        if job.kind in ("suggest", "explain", "complete", "code", "text", "insights", "recap"):
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

    def _build_prompt_parts(
        self, job: _AssistJob, window: str, screen_text: str, choices: ChoiceSet | None = None,
        question: str | None = None,
    ) -> tuple[str, str, int]:
        """System/user prompt text and the token budget for one job.

        `choices` (a detected multiple-choice option set) adds the <choices> block to the
        question-answering prompts and `CHOICES_TOKEN_HEADROOM` to their budget, so the verdict
        line never has to compete with the answer's own shape for tokens; `question` replaces
        `job.question` in the prompt (its annotated copy) and defaults to it.
        """
        block = choices_block(choices) if choices is not None else ""
        headroom = CHOICES_TOKEN_HEADROOM if choices is not None else 0
        q = job.question if question is None else question
        context = self._cfg.assist.session_context
        if job.kind == "ask":
            system = prompts.system_ask(self._cfg.assist.persona, context)
            user = prompts.user_ask(window, q, screen_text, block)
            return system, user, self._cfg.llm.ask_max_tokens + headroom
        if job.kind == "solve":
            ex = self._cfg.explain
            level = prompts.normalize_comment_level(self._cfg.assist.comments)
            system = prompts.system_solve(
                self._cfg.assist.persona, explain=ex.enabled, comments=level,
                language=self._cfg.assist.primary_language,
                libraries=self._cfg.assist.libraries, context=context,
            )
            user = prompts.user_solve(window, screen_text, job.focus, block)
            return system, user, self._code_budget(ex.enabled, level) + headroom
        if job.kind == "code":
            ex = self._cfg.explain
            level = prompts.normalize_comment_level(self._cfg.assist.comments)
            system = prompts.system_code(
                self._cfg.assist.persona, explain=ex.enabled, comments=level,
                language=self._cfg.assist.primary_language,
                libraries=self._cfg.assist.libraries, context=context,
            )
            user = prompts.user_code(window, screen_text, self._previous_code_text(), job.focus)
            return system, user, self._code_budget(ex.enabled, level) + headroom
        if job.kind == "text":
            system = prompts.system_text(self._cfg.assist.persona, context)
            user = prompts.user_text(window, screen_text, self._previous_code_text(), job.focus)
            return system, user, self._cfg.llm.text_max_tokens
        if job.kind == "explain":
            with self._plan_lock:
                plan = self._plan
                parts = self._parsed_plan_locked(plan) if plan is not None else None
            walkthrough = parts.walkthrough if parts is not None else ""
            solution = parts.solution if parts is not None else ""
            system = prompts.system_explain(self._cfg.assist.persona, context)
            user = prompts.user_explain(walkthrough, solution, screen_text)
            return system, user, self._cfg.explain.explain_max_tokens
        if job.kind == "suggest":
            system = prompts.system_suggest(self._cfg.assist.persona, context)
            user = prompts.user_suggest(window, screen_text, job.focus)
            return system, user, self._cfg.llm.assist_max_tokens
        if job.kind == "insights":
            a = self._cfg.assist
            secs = job.sections or prompts.INSIGHT_SECTIONS
            with self._insights_lock:
                previous = prompts.render_insights({s: self._insights.get(s, "") for s in secs})
            system = prompts.system_insights(a.persona, a.role, secs, a.insight_max_items, context)
            user = prompts.user_insights(window, screen_text, previous, secs)
            return system, user, self._cfg.llm.insights_max_tokens
        if job.kind == "recap":
            with self._notes_lock:
                notes = self._notes_markdown
            system = prompts.system_recap(self._cfg.assist.persona, context)
            user = prompts.user_recap(window, notes, screen_text, job.focus)
            return system, user, self._cfg.llm.recap_max_tokens
        # "auto", "manual", "screen"
        system = prompts.system_assist(self._cfg.assist.persona, self._cfg.assist.style, context)
        user = prompts.user_assist(window, q, screen_text, block)
        return system, user, self._cfg.llm.assist_max_tokens + headroom

    def _code_budget(self, explain: bool, level: str) -> int:
        """Token budget for a "solve" / "code" answer: `solve_max_tokens`, plus the walkthrough's
        share with explanations on, plus `COMMENT_TOKEN_HEADROOM` for the comment `level` (one of
        `prompts.COMMENT_LEVELS`)."""
        extra = self._cfg.explain.walkthrough_max_tokens if explain else 0
        return self._cfg.llm.solve_max_tokens + extra + COMMENT_TOKEN_HEADROOM.get(level, 0)

    def _complete_prompt_parts(
        self, job: _AssistJob, window: str, screen_text: str
    ) -> tuple[str, str, int, str, float]:
        """(system, user, max_tokens, assistant_prefix, temperature) for a kind "complete" job.

        The session context goes into the user turn stable-first (transcript, answers, notes,
        screen); the typed prefix becomes the start of the assistant turn so the model continues
        it verbatim (see `prompts.system_complete`). `window` (like the answers below) is already
        scoped to the current checkpoint -- see `_run_assist_job`.
        """
        cc = self._cfg.completion
        blocks: list[tuple[str, str]] = []
        if cc.use_transcript:
            blocks.append(("transcript", window))
        if cc.use_answers:
            checkpoint_answers = self._checkpoint_answers_index()
            with self._answers_lock:
                recent = self._answers[checkpoint_answers:][-3:]
            joined = "\n\n".join(f"Q: {q[:200]}\nA: {a[:600]}" for _t, q, a in recent)
            blocks.append(("answers", joined))
        if cc.use_notes:
            with self._notes_lock:
                notes = self._notes_markdown[:1200]
            blocks.append(("notes", notes))
        if cc.use_screen:
            blocks.append(("screen", screen_text))
        explain = self._cfg.explain.enabled
        if explain:
            view = self.plan()
            if view is not None and view.solution.strip():
                blocks.append(("solution", view.solution))   # the plan: the continuation must follow it
        system = prompts.system_complete(
            self._cfg.assist.persona, cc.hint, explain=explain, language=self._cfg.assist.primary_language,
            comments=prompts.normalize_comment_level(self._cfg.assist.comments),
        )
        user = prompts.user_complete(blocks, job.field, job.suffix)
        return system, user, cc.max_tokens, prompts.NO_THINK_PREFIX + job.prefix, cc.temperature

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
                job.kind in ("suggest", "screen", "solve", "code", "text", "explain", "insights", "recap")
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
            job = self._notes_slot.get()
            if job is None:
                log.info("notes-worker: stopping")
                return
            try:
                self._run_notes_job(job)
            except Exception:  # pragma: no cover - safety net; _run_notes_job handles its own errors
                log.exception("notes-worker: unhandled error running job %d", job.id)

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
        new_text, last_included = self._transcript.head_text(self._cfg.llm.context_tokens * 4, since_id=since_id)
        until_id = last_included if last_included is not None else since_id   # nothing new: stay put
        system = prompts.system_notes(self._cfg.notes.template)
        user = prompts.user_notes(previous, new_text)
        try:
            text = self._stream_text(system, user, max_tokens=self._cfg.llm.notes_max_tokens)
        except LLMError as exc:
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
            if job is not None and job.kind == "insights" and job.auto:
                job.cancel.set()

    def _drop_auto_insights(self) -> None:
        """Pause / completion mode: cancel a running background pass and drop a queued one."""
        self._preempt_auto_insights()
        self._insight_slot.remove_if(lambda queued: queued.auto)

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
        code in silence still gets blind spots on it."""
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

        self._bus.publish(ComponentStatusEvent(component="llm", state="ready", detail=self._cfg.llm.model))
        self._refresh_models_async()

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

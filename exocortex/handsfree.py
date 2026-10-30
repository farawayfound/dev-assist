"""Hands-free mode: watch the screen and, when the task on it changes, do the whole cycle alone.

While hands-free is on (the switch at the right of the tab strip, ``Ctrl+Alt+H``,
``[handsfree].enabled``) the :class:`HandsFreeController` watches the screen reader's samples
(taken every ``[screen].interval_s``, the ordinary capture interval). What starts a cycle is a
NEW TASK between one sample and the next, decided by :func:`is_new_task` on a
:func:`classify_change` report -- any one of:

* **new text**: at least ``[handsfree].min_new_lines`` lines of prose (3+ words) that were not on
  screen before, are not an *edit* of a line that was (:func:`is_edit`: the line being typed,
  OCR noise on a re-read, a counter, a typo fix) and were not *typed* progressively (a line the
  controller watched grow, sample after sample, since the last task) -- a new question on a slide,
  however short, even under a static header; a new question line (ends in ``?`` / opens with
  what-how-why...) always counts;
* **most of the view changed at once**: the share of lines that differ (1 - Jaccard of the two
  line sets, :func:`sample_diff`) reaches ``change_threshold`` and at least ``min_changed_lines``
  lines are new or gone (edits and typed lines never count) -- a big scroll through the code, a
  completely different screen.

Typing never gets there: the line being typed is an edit of itself from one sample to the next,
however different the screen becomes over time. Lines matching ``[assist].ignore_patterns`` (and
the built-in list of site polls / feedback prompts) never count. ``[assist].focus`` gates
hands-free too, least to most permissive: ``"technical"`` wants a technical cue in the new text
or code on screen; ``"auto"`` (the default, code-biased) takes a question or a line that reads
as a task (``questions.is_task_line``: "Implement...", "Given an array...", a technical term)
and, for a big change of view, wants at least one such line among the new ones -- so a chat or
an email drifting into view does not distract it; ``"all"`` takes any new text and any big change.

Once such a change has *settled* (no further change for ``settle_s``) the settled screen is
re-read with the accurate OCR engine and, unless :func:`is_new_task` says it is still the screen
the last cycle acted on (a popup that came and went, an alt-tab and back), the app does by itself
what the user does by hand today:

    Read (accurate OCR)  ->  Checkpoint  ->  Text (walkthrough)  ->  Code (solution)

:func:`task_kind` looks at the settled screen first: code lines, an editor's line-number gutter,
a development window in front (``ScreenTextEvent.window_title`` / ``window_process``: VS Code,
PyCharm, a terminal, LeetCode...) or a statement that asks for code make it a **code task**, and
``[handsfree].on_code`` then decides the cycle: ``"code"`` (default) runs Code only, with an ask
(:data:`HANDS_FREE_CODE_FOCUS`) that also covers questions ABOUT the code shown (its output, a
bug, which option is right) so the model never answers "(nothing to do)"; ``"code_first"`` runs
Code then Text; ``"ignore"`` keeps the ordinary order. Text and Code are the app's ordinary
``"text"`` / ``"code"`` jobs (they land in the Assist tab, count as answers, become the plan with
explanations on); they run one after the other -- the controller waits for the first one's
``AssistDoneEvent`` / ``AssistErrorEvent`` before asking for the second -- so the walkthrough
streams first and can be read aloud while the code is written (``code_first`` swaps them;
``run_text`` / ``run_code`` drop one). Text is skipped by itself while explanations are on (a
"code" answer then opens with its own walkthrough) and in completion mode (the slim bar has
nowhere to show prose). A new task mid-cycle checkpoints again (which cancels the running job)
and starts over: the screen dictates, there is no minimum interval. A job of the cycle that a
spoken question preempted (``AssistDoneEvent(cancelled=True)``: a main-lane answer cancels
whatever runs) is re-requested, at most :data:`MAX_STEP_RETRIES` times, instead of leaving a
blank answer; a job the model *skipped* (``skipped=True``, it found nothing to do) is simply
done.

Two more things happen on every settled screen. **Cumulative context** (:class:`ScreenContext`):
the accurately read lines are merged into a running document for the current task, so what has
scrolled out of view -- the statement, the code typed further up -- stays in the ``<screen>``
block handed to hands-free Text / Code jobs (``context_chars`` budget, the current screen always
intact at the end; an earlier reading of a line since edited is dropped). **Deviation**: when
the app has produced code for this task and the code on screen is mostly NOT that code
(:func:`deviation` >= ``deviation_threshold``, and the user's code grew by ``min_changed_lines``
since the last time), Code is regenerated -- no checkpoint, the walkthrough stays -- so the
solution builds on what was actually typed. Pause pauses the watcher too; Reset forgets the last
task; a checkpoint the USER sets while hands-free is on means "redo this screen".

**Manual advance** (``[handsfree].manual``, read live like the thresholds): everything above --
sampling, settle detection, the accurate re-read, task detection, the checkpoint and the
cumulative context -- keeps running, but a new task ARMS instead of running: no Text / Code step
starts by itself (and code is never regenerated on deviation). The state is published as
``"armed"`` and the steps wait for :meth:`~HandsFreeController.trigger_step` -- the overlay's
press-and-hold mouse gesture (left button: Text, middle button: Code) -- which runs that one step
with the task's kind (a code task keeps :data:`HANDS_FREE_CODE_FOCUS`), the cumulative context
and the ordinary retry machinery.

Runs on its own daemon thread ("hands-free") draining a bus subscription; the engine calls it
only through :meth:`start` / :meth:`stop` / :meth:`set_paused` / :meth:`reset`, which hand a
command to that thread (so every state change happens there); the ``[handsfree]`` thresholds are
read live on every evaluation, so a settings change needs no call at all. The engine side is
reached through :class:`HandsFreeHooks`; :meth:`context_text` is read from the assistant's
worker thread. Every state change is published as a :class:`~exocortex.bus.HandsFreeEvent` for
the UI (``detail`` carries the reason a settled screen started nothing). The decision helpers at
the top are pure.
"""
from __future__ import annotations

import difflib
import logging
import queue
import re
import threading
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from exocortex.bindings import manual_gesture_summary
from exocortex.bus import (
    AssistDoneEvent,
    AssistErrorEvent,
    AssistStartedEvent,
    CheckpointEvent,
    EventBus,
    HandsFreeEvent,
    HandsFreeState,
    ScreenTextEvent,
)
from exocortex.config import Config, HandsFreeConfig
from exocortex.llm import questions
from exocortex.plan import _similar, is_comment_line, squash

log = logging.getLogger(__name__)

#: How long the controller waits for the accurate re-read it asked for before evaluating the
#: sampled lines instead (the reader answers within one tick; this only covers a dead reader).
ACCURATE_READ_TIMEOUT_S = 15.0
#: How long one Text / Code step may take before the cycle moves on without it (a hung model).
STEP_TIMEOUT_S = 240.0
#: How many times a step of the cycle is re-requested after being preempted (cancelled) by a
#: higher-priority job such as a spoken question's answer.
MAX_STEP_RETRIES = 2
#: The controller thread's wake-up cadence for its timers (settle / timeouts).
_TICK_S = 0.1
#: Bounded memory of assist job ids -> kinds (from AssistStartedEvents).
_MAX_JOB_MEMORY = 64
#: Bounded memory of lines the user typed progressively since the last task.
_MAX_TYPED_MEMORY = 400
#: Two lines whose squashed forms share this many leading characters are the same line edited
#: (the cumulative context keeps only the current reading).
_SAME_LINE_HEAD = 12
#: A fresh prose line needs this many words to count as "new text" (a heading such as "Two Sum"
#: does not make a task by itself; a two-word question, "Why recursion?", still does).
MIN_WORDS_NEW_TEXT = 3
#: This many code-looking lines on the settled screen make it a code task.
MIN_CODE_LINES = 2
#: This many line-number-gutter lines ("12  def f():") make it a code task (an editor is open).
MIN_GUTTER_LINES = 3
#: Two (almost) equally long lines are the same line at two moments when at most this many
#: characters differ -- or one in ten for long lines (OCR noise on a re-read, a counter that
#: ticked, a typo fix).
EDIT_MAX_DIFF = 2
#: Headers used inside the cumulative context handed to the model.
CONTEXT_EARLIER_HEADER = "[earlier on screen, now scrolled out of view]"
CONTEXT_NOW_HEADER = "[on screen now]"

#: The ask handed to the Code step of a code task (``[handsfree].on_code``): the ordinary Code
#: ask makes the model answer "(nothing to do)" to a question ABOUT the code shown (a quiz slide
#: with a snippet, "what does this print?"), which is exactly what a coding slide often is.
HANDS_FREE_CODE_FOCUS = (
    "The screen shows code and/or a coding question. If it asks to write, complete or fix code, "
    "output the finished code. If it asks ABOUT the code shown -- its output, a bug, its "
    "complexity, which option is correct -- give that answer in one line first, then the code "
    "that demonstrates it. There is always something to write here: never answer that there is "
    "nothing to do."
)
#: What the Assist tab / history show as the question for that step.
HANDS_FREE_CODE_LABEL = "the coding task on screen"
#: Status detail while a task is armed in manual advance (the title-bar subtitle), for the DEFAULT
#: gestures and hold settings; :func:`armed_detail` renders the configured ones (2026-10-29).
ARMED_DETAIL = manual_gesture_summary(Config())


def armed_detail(cfg: Config) -> str:
    """The armed detail for ``cfg``'s gestures -- "hold left = Text, click middle = Code, click
    mouse 4 = Code from scratch" with the defaults -- following ``[hotkeys].manual_*`` and the
    ``[handsfree].manual_hold_*`` switches; :data:`ARMED_DETAIL` when every gesture is unbound."""
    return manual_gesture_summary(cfg) or ARMED_DETAIL

_CODE_LINE_RE = re.compile(
    r"[{}();=<>\[\]]|^\s*(?:def|class|return|import|from|for|while|if|elif|else|try|except|finally|"
    r"const|let|var|function|public|private|static|void|int|float|double|bool|string|#include|using|"
    r"namespace|package|fn|pub|struct|enum|impl|select|insert|update|delete|with)\b",
    re.IGNORECASE,
)
#: A URL or a query string: never code (an address bar reads "…/models?pipeline_tag=text&n=64").
_URL_RE = re.compile(r"https?://|\bwww\.|\b[a-z0-9-]+\.(?:com|org|net|io|co|ai|dev|edu|gov)\b|\?[a-z_]+=[^\s]*&", re.IGNORECASE)
#: An editor's line-number gutter read together with the line ("12  def f():", "3 return x") --
#: not a numbered list ("1. Foo", "2) Bar") and not a percentage / time ("10:41", "50%").
_GUTTER_RE = re.compile(r"^\s*\d{1,4}(?![.):%,\d])\s+\S")
#: Window titles of editors, IDEs, terminals and coding sites.
_DEV_WINDOW_RE = re.compile(
    r"visual studio|vs ?code|pycharm|intellij|webstorm|phpstorm|clion|goland|rubymine|datagrip|"
    r"\brider\b|android studio|xcode|eclipse|netbeans|sublime text|notepad\+\+|\bg?vim\b|neovim|"
    r"emacs|jupyter|spyder|rstudio|\bcursor\b|\bzed\b|windsurf|leetcode|hackerrank|codility|"
    r"codesignal|coderpad|codepen|replit|codeforces|codewars|codesandbox|stackblitz|github|gitlab|"
    r"bitbucket|terminal|powershell|command prompt|\bcmd\b|\bbash\b|\bssh\b|jetbrains",
    re.IGNORECASE,
)
#: Executable names (without ".exe") of the same.
_DEV_PROCESS_RE = re.compile(
    r"^(?:code|cursor|windsurf|devenv|pycharm(?:64)?|idea(?:64)?|webstorm(?:64)?|phpstorm(?:64)?|"
    r"clion(?:64)?|rider(?:64)?|goland(?:64)?|rubymine(?:64)?|datagrip(?:64)?|studio64|sublime_text|"
    r"notepad\+\+|g?vim|nvim(?:-qt)?|emacs|windowsterminal|powershell|pwsh|cmd|wt|mintty|conhost|"
    r"alacritty|wezterm(?:-gui)?|zed|rstudio|spyder|jupyter(?:-lab)?)$",
    re.IGNORECASE,
)
#: A statement that asks for code even when no code is on screen yet.
_CODING_TASK_RE = re.compile(
    r"\b(?:write|implement|code|program|create|build|design)\b[^.\n?]{0,60}"
    r"\b(?:function|method|class|program|algorithm|solution|script|query|code|api|routine|procedure|regex)\b"
    r"|\bgiven\b[^.\n]{0,80}\b(?:array|string|list|tree|graph|matrix|integer|number|nums|linked list|binary tree|input)\b"
    r"|\b(?:input|output|example ?\d*|constraints?)\s*:"
    r"|\btime complexity\b|\bspace complexity\b|\bo\((?:n|1|log)"
    r"|\bleetcode\b|\bhackerrank\b|\bcodility\b|\bcodesignal\b|\bcoderpad\b"
    r"|\bunit tests?\b|\bcompile[sd]?\b|\bsql\b|\bregex\b|\bpseudo-?code\b",
    re.IGNORECASE,
)


class HandsFreeHooks(Protocol):
    """What the controller needs from the engine (see ``app.Engine``)."""

    def capture_now(self, accurate: bool = False) -> None:
        """Force a screen capture + OCR now (``accurate=True``: with the accurate engine)."""

    def checkpoint(self) -> None:
        """Soft-reset the coding context (publishes a CheckpointEvent)."""

    def text(self, focus: str = "", label: str = "") -> int | None:
        """Start a "text" (walkthrough) job for the task on screen / discussed since the
        checkpoint; returns its job id (None when the engine deferred it: LLM not ready).
        ``focus`` is an extra instruction (blank: the default ask), ``label`` what the UI shows."""

    def code(self, focus: str = "", label: str = "", fresh: bool = False) -> int | None:
        """Start a "code" job for the same task; returns its job id (None when deferred). ``fresh``
        (2026-10-29): the "Code from scratch" gesture -- the job ignores the code on screen."""

    def recent_code(self) -> str:
        """The code the app itself produced since the current checkpoint ("" when none)."""


# ------------------------------------------------------------------------------------------ #
# pure decision helpers
# ------------------------------------------------------------------------------------------ #


def normalize_line(line: str) -> str:
    """Case- and whitespace-insensitive key of one OCR line ("" for a line under 2 chars)."""
    key = " ".join(str(line).split()).casefold()
    return key if len(key) >= 2 else ""


def normalize_lines(lines: Iterable[str]) -> set[str]:
    """Case- and whitespace-insensitive line set (OCR noise between two passes over the same
    screen is mostly case and spacing); lines shorter than 2 characters are dropped."""
    out: set[str] = set()
    for line in lines:
        key = normalize_line(line)
        if key:
            out.add(key)
    return out


def normalize_keys(lines: Iterable[str]) -> list[str]:
    """:func:`normalize_line` of every line, in screen order, duplicates and empties dropped."""
    seen: dict[str, None] = {}
    for line in lines:
        key = normalize_line(line)
        if key:
            seen.setdefault(key, None)
    return list(seen)


def sample_diff(previous: set[str] | None, current: set[str]) -> tuple[float, int]:
    """How much the screen changed between two consecutive samples: ``(score, changed_lines)``
    with ``score = 1 - |A & B| / |A | B|`` (0.0 for two empty screens, 1.0 against no previous
    sample) and ``changed_lines`` the number of lines in one set but not the other."""
    if previous is None:
        return (1.0 if current else 0.0), len(current)
    union = previous | current
    if not union:
        return 0.0, 0
    common = len(previous & current)
    return 1.0 - common / len(union), len(union) - common


def is_code_line(line: str) -> bool:
    """Whether an OCR ``line`` looks like code rather than prose: a comment line is prose; a
    sentence (ends in ``.``/``?``/``!`` with several words) is prose; a URL / query string (a
    browser's address bar, a link) is prose however many ``=`` and ``?`` it carries; otherwise
    brackets, an assignment, a comparison or a leading keyword make it code."""
    text = line.strip()
    if not text or is_comment_line(text):
        return False
    words = text.split()
    if text[-1] in ".?!" and len(words) >= 4:
        return False
    if _URL_RE.search(text):
        return False
    return bool(_CODE_LINE_RE.search(text))


def is_edit(a: str, b: str) -> bool:
    """Whether two squashed line keys are the SAME line at two moments rather than two lines:
    one extends the other (typing, backspacing; one trailing character of OCR noise allowed,
    at least 4 characters in common), or they are (almost) equally long (within 2) and differ in
    at most :data:`EDIT_MAX_DIFF` characters, one in ten for long lines (OCR noise on a re-read,
    a counter that ticked -- "Ln 12" / "Ln 13" -- a typo fix). Two different questions of similar
    wording ("what is a hash map?" / "what is a heap?", "what is a stack?" / "what is a queue?")
    are NOT an edit: they neither extend one another nor differ in so few characters."""
    if a == b:
        return bool(a)
    if not a or not b:
        return False
    short, long_ = (a, b) if len(a) <= len(b) else (b, a)
    common = 0
    for x, y in zip(short, long_):
        if x != y:
            break
        common += 1
    if common >= max(4, len(short) - 1):
        return True
    if abs(len(a) - len(b)) <= 2 and len(short) >= 6:
        matching = sum(block.size for block in difflib.SequenceMatcher(None, a, b).get_matching_blocks())
        return len(long_) - matching <= max(EDIT_MAX_DIFF, len(long_) // 10)
    return False


@dataclass(frozen=True)
class ChangeReport:
    """What changed from one screen to another (:func:`classify_change`). Every line is a
    :func:`normalize_line` key."""

    score: float                             #: 1 - Jaccard of the two line sets (sample_diff)
    changed: int                             #: lines in one set but not the other
    fresh: tuple[str, ...]                   #: lines new on screen: no earlier line they are an edit of, not typed
    fresh_prose: tuple[str, ...]             #: ... of which prose (not is_code_line)
    fresh_code: tuple[str, ...]              #: ... of which code
    questions: tuple[str, ...]               #: ... of the prose, the ones that read as a question
    edits: tuple[tuple[str, str], ...]       #: (before, after): the same line at two moments
    typed: tuple[str, ...]                   #: would-be fresh lines the user typed progressively
    gone: int                                #: earlier lines no longer on screen (not edited)

    @property
    def new_text(self) -> tuple[str, ...]:
        """The fresh prose lines with at least :data:`MIN_WORDS_NEW_TEXT` words."""
        return tuple(line for line in self.fresh_prose if len(line.split()) >= MIN_WORDS_NEW_TEXT)


def classify_change(
    previous: Sequence[str] | None, current: Sequence[str], typed: Iterable[str] = ()
) -> ChangeReport:
    """Classify the lines of ``current`` against ``previous`` (raw OCR lines or keys; ``None`` =
    no earlier screen, everything is fresh). A line of ``current`` that is not in ``previous`` is
    an *edit* when :func:`is_edit` pairs it with a line that disappeared, *typed* when it pairs
    with (or equals) one of ``typed`` (squashed keys of lines watched growing since the last
    task), and *fresh* otherwise."""
    cur_keys = normalize_keys(current)
    cur_set = set(cur_keys)
    if previous is None:
        prev_keys: list[str] = []
        score, changed = sample_diff(None, cur_set)
    else:
        prev_keys = normalize_keys(previous)
        score, changed = sample_diff(set(prev_keys), cur_set)
    prev_set = set(prev_keys)
    added = [k for k in cur_keys if k not in prev_set]
    removed = [k for k in prev_keys if k not in cur_set]
    removed_sq = [(k, squash(k)) for k in removed]
    typed_sq = [squash(t) for t in typed if squash(t)]
    edits: list[tuple[str, str]] = []
    fresh: list[str] = []
    typed_out: list[str] = []
    paired: set[str] = set()
    for key in added:
        sq = squash(key)
        partner = next((rk for rk, rsq in removed_sq if rk not in paired and is_edit(rsq, sq)), None)
        if partner is None:
            partner = next((rk for rk, rsq in removed_sq if is_edit(rsq, sq)), None)
        if partner is not None:
            edits.append((partner, key))
            paired.add(partner)
            continue
        if any(t == sq or is_edit(t, sq) for t in typed_sq):
            typed_out.append(key)
            continue
        fresh.append(key)
    gone = sum(1 for k in removed if k not in paired)
    fresh_prose = tuple(k for k in fresh if not is_code_line(k))
    fresh_code = tuple(k for k in fresh if is_code_line(k))
    qs = tuple(k for k in fresh_prose if questions.is_question(k, min_words=2))
    return ChangeReport(
        score=score, changed=changed, fresh=tuple(fresh), fresh_prose=fresh_prose,
        fresh_code=fresh_code, questions=qs, edits=tuple(edits), typed=tuple(typed_out), gone=gone,
    )


def is_new_task(
    report: ChangeReport, hf: HandsFreeConfig, *, focus: str = "auto",
    ignore_patterns: Sequence[str] | None = None,
) -> tuple[bool, str]:
    """Whether a :class:`ChangeReport` means a NEW TASK is on screen (module docstring), and
    why / why not (a short reason for the log and the status line). ``focus`` is
    ``[assist].focus`` (``questions.focus_level``): ``"technical"`` wants a technical cue in the
    new text or code on screen; ``"auto"`` takes questions and task-like lines
    (``questions.is_task_line``) and wants one of those among the new lines of a big change of
    view; ``"all"`` takes any new text and any big change. ``ignore_patterns`` are
    ``[assist].ignore_patterns`` (the built-in poll / feedback patterns always apply)."""
    threshold = min(1.0, max(0.0, float(hf.change_threshold)))
    min_changed = max(1, int(hf.min_changed_lines))
    min_new = max(1, int(hf.min_new_lines))
    level = questions.focus_level(focus)
    patterns = list(ignore_patterns or ())
    code_seen = bool(report.fresh_code)

    def relevant(line: str) -> bool:
        # Ignore patterns at every level; the personal-question filter at "auto" and "technical";
        # the technical cue at "technical" -- unless code came with the change.
        if level == "technical" and code_seen:
            return questions.is_relevant(line, "auto", patterns)
        return questions.is_relevant(line, level, patterns)

    def task_like(line: str) -> bool:
        return level == "all" or code_seen or questions.is_task_line(line)

    new_questions = [q for q in report.questions if relevant(q)]
    if new_questions:
        return True, f"a new question: {new_questions[0][:48]!r}"
    new_text = [line for line in report.new_text if relevant(line) and task_like(line)]
    if len(new_text) >= min_new:
        return True, f"{len(new_text)} new line(s) of text"
    if report.score >= threshold and len(report.fresh) + report.gone >= min_changed:
        if level == "all" or code_seen or any(relevant(line) and task_like(line) for line in report.fresh_prose):
            return True, f"{report.score:.0%} of the view changed ({len(report.fresh)} new, {report.gone} gone)"
        return False, f"most of the view changed but nothing to answer on it ({level} focus)"
    if not report.fresh and not report.edits and not report.typed:
        return False, "nothing changed"
    if not report.fresh:
        return False, f"only edits / typing ({len(report.edits) + len(report.typed)} line(s))"
    if new_text:
        return False, f"{len(new_text)} new line(s) of text < {min_new}"
    if report.new_text or report.questions:
        return False, f"new text but nothing to answer in it ({level} focus)"
    return False, f"{len(report.fresh)} new line(s), none a task (score {report.score:.2f})"


def task_kind(
    lines: Sequence[str], text: str = "", window_title: str = "", window_process: str = "",
    mode: str = "",
) -> tuple[str, str]:
    """``("code", why)`` when the settled screen is a coding task -- :data:`MIN_CODE_LINES`
    code-looking lines, an editor's line-number gutter (:data:`MIN_GUTTER_LINES` lines), a
    development window in front (``window_title`` / ``window_process`` from the
    ``ScreenTextEvent``; not in ``"region"`` mode, where the window in front and the region read
    are unrelated) or a statement that asks for code -- else ``("text", why)``."""
    code_lines = [line for line in lines if is_code_line(line)]
    if len(code_lines) >= MIN_CODE_LINES:
        return "code", f"{len(code_lines)} code line(s) on screen"
    gutter = sum(1 for line in lines if _GUTTER_RE.match(str(line)))
    if gutter >= MIN_GUTTER_LINES:
        return "code", f"an editor gutter on screen ({gutter} numbered lines)"
    if mode != "region":
        proc = str(window_process or "").rsplit(".", 1)[0] if str(window_process or "").lower().endswith(".exe") else str(window_process or "")
        if window_title and _DEV_WINDOW_RE.search(str(window_title)):
            return "code", f"a development window in front ({str(window_title)[:40]!r})"
        if proc and _DEV_PROCESS_RE.match(proc):
            return "code", f"a development window in front ({proc})"
    blob = text if text else "\n".join(str(line) for line in lines)
    m = _CODING_TASK_RE.search(blob)
    if m:
        return "code", f"the task asks for code ({m.group(0)[:40]!r})"
    return "text", "no code on screen"


def own_code_lines(code_text: str) -> list[str]:
    """Squashed (whitespace-free) code lines of ``code_text`` worth matching: non-blank,
    non-comment, at least 4 characters. Fences are skipped."""
    out: list[str] = []
    for raw in code_text.splitlines():
        if not raw.strip() or is_comment_line(raw) or raw.lstrip().startswith(("```", "~~~")):
            continue
        key = squash(raw)
        if len(key) >= 4:
            out.append(key)
    return out


def is_own_code_line(line: str, own: Sequence[str]) -> bool:
    """Whether an OCR ``line`` is (a noisy reading of) one of the app's own code lines ``own``
    (:func:`own_code_lines`): exact after squashing, the tail of a gutter-merged line
    (``"12def f():"``), or fuzzy (``plan._similar``)."""
    if not own:
        return False
    key = squash(line)
    if len(key) < 4:
        return False
    for want in own:
        if key == want or (key.endswith(want) and len(key) - len(want) <= 4) or _similar(key, want):
            return True
    return False


def deviation(
    lines: Iterable[str], own: Sequence[str], exclude: set[str] | None = None,
    typed: Sequence[str] | None = None,
) -> tuple[float, set[str]]:
    """How far the code on screen has drifted from the app's own code: ``(fraction, user_lines)``
    where ``user_lines`` are the (normalised) code lines on screen that are NOT the app's and
    ``fraction`` is their share of all code lines on screen. Lines in ``exclude`` (the screen
    as it was when the task was picked up: code-looking lines of the statement such as
    ``Input: nums = [2,7]``) are ignored. ``(0.0, set())`` when the app has produced no code
    (``own`` empty) or there is no code on screen.

    ``typed`` (upstream finding 10, ported from mac-cortex) are squashed keys of lines the
    sample-to-sample tracker already knows are still being typed (``HandsFreeController._typed``,
    the same bookkeeping :func:`classify_change` uses to keep a growing line from counting as
    fresh text). A line that fuzzy-matches one of them (:func:`is_edit`, equal squashed form) is
    excluded from ``user`` too: a line the user has not finished typing yet routinely fails the
    exact/gutter-tail/fuzzy match in :func:`is_own_code_line` (too short, no early paren/colon)
    purely because it is incomplete, not because it is a different line than the app's own --
    without this it would count as drift and could wrongly trigger a Code regeneration
    mid-keystroke.
    """
    if not own:
        return 0.0, set()
    typed_keys = [t for t in (typed or ()) if t]
    own_seen = 0
    user: set[str] = set()
    for line in lines:
        if not is_code_line(line):
            continue
        key = normalize_line(line)
        if exclude and key in exclude:
            continue
        # upstream finding 10 (ported from mac-cortex): a line still growing sample-to-sample
        # should never count as drifted-away-from-own code, even mid-keystroke.
        if typed_keys:
            # upstream finding 10 (Windows fix): casefold before squashing, same as the keys
            # self._typed actually holds (squash(normalize_line(...)), via normalize_keys() in
            # _track_sample) -- squash() alone does not casefold, so an uppercase character
            # anywhere in the line (class names, True/False/None, PascalCase/camelCase) used to
            # make this comparison miss every match and silently defeat the exclusion.
            sq = squash(normalize_line(line))
            if any(sq == t or is_edit(sq, t) for t in typed_keys):
                continue
        if is_own_code_line(line, own):
            own_seen += 1
        elif key:
            user.add(key)
    total = own_seen + len(user)
    if total == 0:
        return 0.0, set()
    return len(user) / total, user


class ScreenContext:
    """The cumulative screen of the current task: every accurately read screen merged in first-
    seen order, rendered as what has scrolled out of view followed by the current screen intact.
    Thread-safe (written by the controller thread, read by the assistant's worker)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._seen: dict[str, str] = {}        # normalised key -> original text, first-seen order
        self._current_keys: list[str] = []
        self._current_text = ""

    def reset(self, lines: Sequence[str] = (), text: str = "") -> None:
        with self._lock:
            self._seen = {}
            self._current_keys = []
            self._current_text = ""
        if lines or text:
            self.merge(lines, text)

    def merge(self, lines: Sequence[str], text: str) -> None:
        """Record the latest screen (its OCR ``lines`` and reading-order ``text``)."""
        with self._lock:
            keys: list[str] = []
            for line in lines:
                key = normalize_line(line)
                if not key:
                    continue
                keys.append(key)
                if key not in self._seen:
                    self._seen[key] = " ".join(str(line).split())
            self._current_keys = keys
            self._current_text = text.strip()

    @property
    def empty(self) -> bool:
        with self._lock:
            return not self._seen and not self._current_text

    def text(self, max_chars: int) -> str:
        """The cumulative context, at most ``max_chars``: earlier lines that are no longer on
        screen (an earlier reading of a line since edited is dropped -- same squashed head), the
        oldest trimmed first, then the current screen verbatim."""
        with self._lock:
            current = set(self._current_keys)
            heads = {squash(k)[:_SAME_LINE_HEAD] for k in current if len(squash(k)) >= _SAME_LINE_HEAD}
            earlier: list[str] = []
            for key, text in self._seen.items():
                if key in current:
                    continue
                sq = squash(key)
                if len(sq) >= _SAME_LINE_HEAD and sq[:_SAME_LINE_HEAD] in heads:
                    continue
                earlier.append(text)
            now = self._current_text
        if not earlier:
            return now[:max_chars]
        budget = max_chars - len(now) - len(CONTEXT_EARLIER_HEADER) - len(CONTEXT_NOW_HEADER) - 4
        if budget <= 0:
            return now[:max_chars]
        kept: list[str] = []
        used = 0
        for line in reversed(earlier):          # newest earlier lines first; the oldest go
            if used + len(line) + 1 > budget:
                break
            kept.append(line)
            used += len(line) + 1
        if not kept:
            return now[:max_chars]
        kept.reverse()
        return f"{CONTEXT_EARLIER_HEADER}\n" + "\n".join(kept) + f"\n\n{CONTEXT_NOW_HEADER}\n{now}"


# ------------------------------------------------------------------------------------------ #
# the controller
# ------------------------------------------------------------------------------------------ #


@dataclass(frozen=True)
class _Screen:
    """A screen as the controller keeps it between events."""

    lines: tuple[str, ...] = ()
    text: str = ""
    window_title: str = ""
    window_process: str = ""
    mode: str = ""

    @classmethod
    def of(cls, ev: ScreenTextEvent) -> "_Screen":
        return cls(
            lines=tuple(ev.lines), text=ev.text, window_title=str(getattr(ev, "window_title", "") or ""),
            window_process=str(getattr(ev, "window_process", "") or ""), mode=str(getattr(ev, "mode", "") or ""),
        )


class HandsFreeController:
    """The hands-free watcher (module docstring). One instance per engine; ``start()``/``stop()``
    may be called repeatedly (the switch)."""

    def __init__(self, cfg: Config, bus: EventBus, hooks: HandsFreeHooks, clock: Callable[[], float]) -> None:
        self._cfg = cfg
        self._bus = bus
        self._hooks = hooks
        self._clock = clock
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._queue: "queue.Queue[Any] | None" = None
        self._lock = threading.Lock()
        self._enabled = False
        self._published: tuple[bool, str, str] | None = None
        self.context = ScreenContext()
        # ---- controller-thread state ----
        self._paused = False
        self._watch: HandsFreeState = "watching"
        self._watch_detail = ""
        self._phase: str | None = None           # "checkpoint" | "text" | "code" while a cycle runs
        self._steps: list[str] = []
        self._task = "text"                      # task_kind of the cycle being run
        self._await: str | None = None           # "change" | "accurate": what the next ScreenTextEvent is
        self._await_deadline: float | None = None
        self._prev_sample: list[str] | None = None
        self._typed: dict[str, None] = {}        # squashed keys of lines watched growing since the last task
        self._big_change = False
        self._pending = _Screen()
        self._last_change: float | None = None
        self._last_trigger_keys: list[str] = []
        self._last_deviation: set[str] = set()
        self._awaiting_kind: str | None = None
        self._awaiting_job_id: int | None = None
        self._step_started: float | None = None
        self._step_retries = 0
        self._armed = False                      # manual advance: a task waits for trigger_step
        self._own_checkpoints_pending = 0
        self._job_kinds: dict[int, str] = {}
        self.triggers = 0                        # cycles started (tests / status)
        self.regenerations = 0                   # deviation-driven Code re-runs (tests / status)
        self.last_task_kind = ""                 # task_kind of the last cycle (tests / status)
        self.last_reason = ""                    # why the last settled screen did / did not start a cycle

    # ---- lifecycle (any thread) -------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        with self._lock:
            return self._enabled

    @property
    def state(self) -> HandsFreeState:
        with self._lock:
            if not self._enabled:
                return "off"
            if self._paused:
                return "paused"
            return self._phase if self._phase else self._watch  # type: ignore[return-value]

    def context_text(self) -> str:
        """The cumulative screen context for the current task (``[handsfree].context_chars``),
        "" while hands-free is off or nothing has been read yet. Any thread."""
        if not self.enabled or self.context.empty:
            return ""
        return self.context.text(max(500, int(self._cfg.handsfree.context_chars)))

    def start(self) -> None:
        """Turn hands-free on: the current screen is treated as a (possibly new) task right
        away. A no-op while already on."""
        with self._lock:
            if self._enabled:
                return
            self._enabled = True
            self._stop_event.clear()
            self._queue = self._bus.subscribe()
            self._thread = threading.Thread(target=self._run, name="hands-free", daemon=True)
            self._thread.start()
        self._post(("start",))

    def stop(self) -> None:
        """Turn hands-free off. A running Text / Code job is left to finish."""
        with self._lock:
            if not self._enabled:
                return
            self._enabled = False
            q, thread = self._queue, self._thread
            self._queue, self._thread = None, None
        self._stop_event.set()
        if q is not None:
            try:
                q.put_nowait(("stop",))
            except queue.Full:  # pragma: no cover - unbounded queue
                pass
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=5.0)
            if thread.is_alive():
                log.warning("hands-free: thread did not stop within timeout")
        if q is not None:
            self._bus.unsubscribe(q)
        self._publish(False, "off", "")

    def set_paused(self, paused: bool) -> None:
        """Pause / resume watching (the app's Pause button). Resuming re-evaluates the screen."""
        self._post(("pause", bool(paused)))

    def reset(self) -> None:
        """Forget the last task (new session): the next settled screen is a new task."""
        self._post(("reset",))

    def trigger_step(self, kind: str) -> None:
        """Manual advance (the mouse gesture): run one ``"text"`` / ``"code"`` / ``"code_fresh"``
        (2026-10-29: Code from scratch, the given code ignored) step for the current task right
        now, whatever the armed steps were. Any thread; ignored while paused (the engine falls
        back to a plain text/code then) or off."""
        self._post(("manual", kind if kind in ("code", "code_fresh") else "text"))

    def resume_auto(self) -> None:
        """Manual advance was switched OFF: if a task is armed and nothing is running, run its
        planned steps now (no new checkpoint -- arming already set one), the way Auto would have.
        Any thread; a no-op while paused, mid-cycle, or with nothing armed."""
        self._post(("resume_auto",))

    def _post(self, command: tuple[Any, ...]) -> None:
        with self._lock:
            q = self._queue
        if q is None:
            if command[0] == "pause":       # remembered for the next start()
                self._paused = bool(command[1])
            return
        q.put_nowait(command)

    # ---- the controller thread --------------------------------------------------------------

    def _run(self) -> None:
        log.info("hands-free: started")
        q = self._queue
        assert q is not None
        while not self._stop_event.is_set():
            try:
                item = q.get(timeout=_TICK_S)
            except queue.Empty:
                item = None
            if self._stop_event.is_set():
                break
            try:
                if item is not None:
                    self._handle(item)
                self._tick()
            except Exception:  # noqa: BLE001 - the watcher must survive any single bad event
                log.exception("hands-free: error handling %r", item)
        log.info("hands-free: stopped")

    def _handle(self, item: Any) -> None:
        if isinstance(item, tuple):
            self._command(item)
        elif isinstance(item, ScreenTextEvent):
            self._on_screen(item)
        elif isinstance(item, AssistStartedEvent):
            self._job_kinds[item.job_id] = item.kind
            if len(self._job_kinds) > _MAX_JOB_MEMORY:
                for old in sorted(self._job_kinds)[: len(self._job_kinds) - _MAX_JOB_MEMORY]:
                    self._job_kinds.pop(old, None)
            if self._awaiting_kind == item.kind and self._awaiting_job_id is None:
                self._awaiting_job_id = item.job_id     # a deferred request: the id was not known yet
        elif isinstance(item, (AssistDoneEvent, AssistErrorEvent)):
            self._job_kinds.pop(item.job_id, None)
            if self._awaiting_job_id is None or item.job_id != self._awaiting_job_id:
                return
            if isinstance(item, AssistDoneEvent) and item.cancelled:
                if getattr(item, "skipped", False):
                    log.info("hands-free: the model found nothing to do for the %s step; moving on", self._awaiting_kind)
                elif self._step_retries < MAX_STEP_RETRIES:
                    self._step_retries += 1
                    log.info("hands-free: %s step was preempted; retrying (%d/%d)", self._awaiting_kind,
                             self._step_retries, MAX_STEP_RETRIES)
                    self._request_step(str(self._awaiting_kind), "interrupted - retrying")
                    return
            self._next_step()
        elif isinstance(item, CheckpointEvent):
            if self._own_checkpoints_pending > 0:
                self._own_checkpoints_pending -= 1
            elif not self._paused:
                log.info("hands-free: checkpoint set by the user -- redoing the current screen")
                self._abort_cycle()
                self._forget_task()
                self._request_change_read()

    def _command(self, command: tuple[Any, ...]) -> None:
        name = command[0]
        if name == "start":
            self._abort_cycle()
            self._forget_task()
            self._prev_sample = None
            if self._paused:
                self._set_watch("paused")
            else:
                self._set_watch("watching")
                self._request_change_read()
        elif name == "pause":
            paused = bool(command[1])
            if paused == self._paused:
                return
            self._paused = paused
            if paused:
                self._abort_cycle()
                self._await = None
                self._big_change = False
                self._set_watch("paused")
            else:
                self._set_watch("watching")
                self._request_change_read()
        elif name == "reset":
            self._abort_cycle()
            self._forget_task()
            self._prev_sample = None
            if not self._paused:
                self._set_watch("watching")
                self._request_change_read()
        elif name == "manual":
            if self._paused:
                return
            self._abort_cycle()
            self._request_step(str(command[1]))
        elif name == "resume_auto":
            if self._paused or self._manual() or not self._armed or self._phase is not None:
                return
            self._armed = False
            steps = self._planned_steps(self._task)
            log.info("hands-free: manual advance off -- running the armed task (%s)",
                     " -> ".join(steps) or "nothing (both steps off)")
            if steps:
                # Not _start_cycle(checkpoint=False): that is the deviation path and would
                # overwrite self._task ("regen"), losing a code task's ask for its Code step.
                self._abort_cycle()
                self._steps = steps
                self._next_step()
            else:
                self._set_watch("watching")

    def _forget_task(self) -> None:
        self._last_trigger_keys = []
        self._last_deviation = set()
        self._typed = {}
        self._big_change = False
        self._armed = False
        self.context.reset()

    def _manual(self) -> bool:
        """``[handsfree].manual``, read live: arm new tasks instead of running them."""
        return bool(self._cfg.handsfree.manual)

    # ---- screen events / settling -----------------------------------------------------------

    def _request_change_read(self) -> None:
        """Force a capture and treat whatever comes back as a new task (enable / resume /
        reset / a user checkpoint)."""
        self._await = "change"
        self._await_deadline = self._clock() + ACCURATE_READ_TIMEOUT_S
        try:
            self._hooks.capture_now(False)
        except Exception:  # noqa: BLE001
            log.exception("hands-free: capture_now failed")

    def _on_screen(self, ev: ScreenTextEvent) -> None:
        if self._paused:
            return
        if self._await == "accurate":
            if ev.accurate:
                self._await = None
                self._await_deadline = None
                self._evaluate(_Screen.of(ev))
                return
            # A sample whose OCR was already in flight when the accurate read was requested (it
            # is older than the accurate frame): keep it in the sample-to-sample bookkeeping,
            # keep waiting for the accurate answer.
            if ev.changed:
                self._track_sample(ev)
            return
        if self._await == "change":
            self._await = None
            self._await_deadline = None
            self._big_change = True
            self._track_sample(ev, score_it=False)
        elif not ev.changed or ev.accurate:
            return
        else:
            self._track_sample(ev)
        self._last_change = self._clock()
        self._set_watch("settling")

    def _track_sample(self, ev: ScreenTextEvent, *, score_it: bool = True) -> None:
        """Fold a sample into the sample-to-sample bookkeeping: classify it against the previous
        sample (a new task arms the next settle to start a cycle; lines that grew are remembered
        as typed) and make it the previous sample and the pending screen."""
        keys = normalize_keys(ev.lines)
        if score_it:
            report = classify_change(self._prev_sample, keys, typed=self._typed)
            for before, after in report.edits:
                self._typed.pop(squash(before), None)
                self._typed[squash(after)] = None
            if len(self._typed) > _MAX_TYPED_MEMORY:
                for old in list(self._typed)[: len(self._typed) - _MAX_TYPED_MEMORY]:
                    self._typed.pop(old, None)
            new, reason = self._is_new_task(report)
            if new:
                if not self._big_change:
                    log.info("hands-free: new task between samples (%s)", reason)
                self._big_change = True
            else:
                log.debug("hands-free: sample changed but no new task (%s)", reason)
        self._prev_sample = keys
        self._pending = _Screen.of(ev)

    def _is_new_task(self, report: ChangeReport) -> tuple[bool, str]:
        return is_new_task(report, self._cfg.handsfree, focus=str(self._cfg.assist.focus or "auto"),
                           ignore_patterns=list(self._cfg.assist.ignore_patterns or ()))

    def _tick(self) -> None:
        if self._paused:
            return
        now = self._clock()
        if self._await is not None and self._await_deadline is not None and now >= self._await_deadline:
            waited, self._await, self._await_deadline = self._await, None, None
            log.warning("hands-free: the screen reader did not answer a %s read in time", waited)
            if waited == "accurate":
                self._evaluate(self._pending)
        if self._await is None and self._watch == "settling" and self._last_change is not None:
            if now - self._last_change >= max(0.0, float(self._cfg.handsfree.settle_s)):
                self._last_change = None
                self._evaluate_pending()
        if self._awaiting_kind is not None and self._step_started is not None and now - self._step_started > STEP_TIMEOUT_S:
            log.warning("hands-free: %s step timed out; moving on", self._awaiting_kind)
            self._next_step()

    def _evaluate_pending(self) -> None:
        """The screen settled: re-read it accurately (when configured) and then evaluate."""
        if self._cfg.handsfree.accurate_ocr:
            self._await = "accurate"
            self._await_deadline = self._clock() + ACCURATE_READ_TIMEOUT_S
            self._set_watch("reading")
            try:
                self._hooks.capture_now(True)
            except Exception:  # noqa: BLE001
                log.exception("hands-free: accurate capture_now failed")
                self._await = None
                self._await_deadline = None
                self._evaluate(self._pending)
            return
        self._evaluate(self._pending)

    def _evaluate(self, screen: _Screen) -> None:
        """Act on the settled, accurately read screen: a new task when a new task led here
        (unless, against the screen already acted on, it is not one after all); otherwise keep
        the cumulative context current and regenerate Code when the typed code has drifted."""
        hf = self._cfg.handsfree
        lines, text = screen.lines, screen.text
        keys = normalize_keys(lines)
        try:
            own = own_code_lines(self._hooks.recent_code() or "")
        except Exception:  # noqa: BLE001
            own = []
        detail = ""
        if self._big_change:
            self._big_change = False
            report = classify_change(self._last_trigger_keys or None, keys, typed=self._typed)
            if not keys:
                new, reason = False, "nothing readable on screen"
            else:
                new, reason = self._is_new_task(report)
            self.last_reason = reason
            if new:
                kind, why = task_kind(lines, text, screen.window_title, screen.window_process, screen.mode)
                steps = self._planned_steps(kind)
                log.info("hands-free: new task (%s) -- %s task (%s): checkpoint -> %s", reason, kind, why,
                         "armed (manual advance)" if self._manual() else (" -> ".join(steps) or "nothing (both steps off)"))
                self._trigger(keys, lines, text, kind, steps)
                return
            log.info("hands-free: settled screen starts nothing (%s)", reason)
            detail = "same task" if self._last_trigger_keys else reason[:40]
        self.context.merge(lines, text)
        if self._phase is None and not self._manual() and own and float(hf.deviation_threshold) > 0:
            # upstream finding 10 (ported from mac-cortex): exclude lines still being typed (per
            # the sample-to-sample tracker) from counting as drift.
            fraction, user_lines = deviation(lines, own, set(self._last_trigger_keys), typed=list(self._typed))
            grew = len(user_lines - self._last_deviation)
            if (fraction >= float(hf.deviation_threshold) and len(user_lines) >= max(1, int(hf.min_changed_lines))
                    and grew >= max(1, int(hf.min_changed_lines))):
                self._last_deviation = user_lines
                self.regenerations += 1
                log.info("hands-free: the code on screen drifted from the app's (%.0f%% not ours, %d line(s)); regenerating Code",
                         fraction * 100, len(user_lines))
                self._start_cycle(["code"], checkpoint=False)
                return
        self._settle_done(detail)

    def _settle_done(self, detail: str = "") -> None:
        """Nothing to do on the settled screen: back to watching (unless a cycle is running) --
        or back to armed, in manual advance with a task still waiting for the gesture."""
        if self._manual() and self._armed:
            self._watch = "armed"
            self._watch_detail = armed_detail(self._cfg)
        else:
            self._watch = "watching"
            self._watch_detail = detail
        self._publish_state()

    def _planned_steps(self, task: str = "text") -> list[str]:
        """The steps of a new cycle for a ``task`` of :func:`task_kind`: Text (unless explanations
        or completion mode make it pointless) and Code, in ``code_first`` order; a code task
        under ``on_code = "code"`` is Code only, under ``"code_first"`` Code then Text."""
        hf = self._cfg.handsfree
        run_text = bool(hf.run_text) and not self._cfg.explain.enabled and not self._cfg.completion.enabled
        run_code = bool(hf.run_code)
        on_code = str(getattr(hf, "on_code", "code") or "code").strip().lower()
        code_task = task == "code" and run_code and on_code in ("code", "code_first")
        if code_task and on_code == "code":
            run_text = False
        steps = [kind for kind, on in (("text", run_text), ("code", run_code)) if on]
        if (hf.code_first or code_task) and steps[:1] == ["text"]:
            steps.reverse()
        return steps

    # ---- the cycle ---------------------------------------------------------------------------

    def _trigger(self, keys: list[str], lines: tuple[str, ...], text: str, kind: str, steps: list[str]) -> None:
        self._last_trigger_keys = list(keys)
        self._last_deviation = set()
        self._typed = {}
        self.triggers += 1
        self.last_task_kind = kind
        self._task = kind
        self.context.reset(lines, text)
        if self._manual():
            # Manual advance: checkpoint and remember the task, but run nothing -- the steps wait
            # for the press-and-hold gesture (trigger_step).
            self._armed = True
            steps = []
        self._start_cycle(steps, checkpoint=True)

    def _start_cycle(self, steps: list[str], *, checkpoint: bool) -> None:
        self._abort_cycle()
        self._steps = list(steps)
        if checkpoint:
            self._phase = "checkpoint"
            self._publish_state()
            self._own_checkpoints_pending += 1
            try:
                self._hooks.checkpoint()
            except Exception:  # noqa: BLE001
                log.exception("hands-free: checkpoint failed")
                self._own_checkpoints_pending -= 1
        else:
            self._task = "regen"
        self._next_step()

    def _next_step(self) -> None:
        self._awaiting_kind = None
        self._awaiting_job_id = None
        self._step_started = None
        self._step_retries = 0
        if self._steps and self._manual():
            # Manual advance was switched on while this cycle's remaining steps were already
            # queued (its list was built under Auto): stop auto-firing them and arm instead,
            # exactly as a new task would under manual.
            log.info("hands-free: manual advance on mid-cycle -- %s left armed instead of run",
                     " -> ".join(self._steps))
            self._steps = []
            self._armed = True
        if not self._steps:
            self._phase = None
            self._watch_detail = ""
            if self._manual() and self._armed:
                self._set_watch("armed", armed_detail(self._cfg))
            else:
                self._set_watch("watching")
            return
        self._request_step(self._steps.pop(0))

    def _request_step(self, kind: str, detail: str = "") -> None:
        # 2026-10-29: "code_fresh" (the Code-from-scratch gesture) is a "code" step whose job ignores
        # the code on screen; the phase and the awaited kind stay "code" (the events carry that kind).
        fresh = kind == "code_fresh"
        if fresh:
            kind = "code"
        self._phase = kind
        self._awaiting_kind = kind
        self._awaiting_job_id = None
        self._step_started = self._clock()
        self._publish_state(detail)
        focus = label = ""
        if kind == "code" and self._task == "code":
            focus, label = HANDS_FREE_CODE_FOCUS, HANDS_FREE_CODE_LABEL
        try:
            if kind == "text":
                job_id = self._hooks.text(focus, label) if focus else self._hooks.text()
            elif fresh:
                job_id = self._hooks.code(focus, label, fresh=True)
            else:
                job_id = self._hooks.code(focus, label) if focus else self._hooks.code()
        except Exception:  # noqa: BLE001
            log.exception("hands-free: %s step failed to start", kind)
            self._next_step()
            return
        if job_id is not None:
            self._awaiting_job_id = int(job_id)

    def _abort_cycle(self) -> None:
        self._steps = []
        self._phase = None
        self._awaiting_kind = None
        self._awaiting_job_id = None
        self._step_started = None
        self._step_retries = 0

    # ---- status ------------------------------------------------------------------------------

    def _set_watch(self, state: HandsFreeState, detail: str = "") -> None:
        self._watch = state
        self._watch_detail = detail
        self._publish_state(detail)

    def _publish_state(self, detail: str = "") -> None:
        state: HandsFreeState = "paused" if self._paused else (self._phase or self._watch)  # type: ignore[assignment]
        if not detail and self._phase is None and state in ("watching", "armed"):
            detail = self._watch_detail
        self._publish(True, state, detail)

    def _publish(self, enabled: bool, state: HandsFreeState, detail: str) -> None:
        key = (enabled, state, detail)
        if key == self._published:
            return
        self._published = key
        self._bus.publish(HandsFreeEvent(enabled=enabled, state=state, detail=detail))

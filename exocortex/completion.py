"""Completion mode: inline ghost-text completions at the caret of whatever has focus.

One thread (``completer``) owns a :class:`~exocortex.screen.caret.CaretTracker` and ticks every
``cfg.completion.poll_ms``. Each tick it observes the focused text field (position, text before
the caret) and runs a small state machine:

* **A suggestion is showing.** If the user has typed exactly what the suggestion said (compared
  whitespace-tolerantly, see :func:`consume_suggestion`), the ghost text simply shrinks by what
  was typed and follows the caret -- no model call. Backspacing into the typed text grows it back.
  Typing something else is a *deviation*: the ghost text disappears.
* **Nothing is showing.** Only once the text has been stable for ``debounce_ms`` (typos get
  fixed, indentation settles) *and* at least ``min_interval_ms`` passed since the last request,
  a new completion is requested for the current text. Never for a prefix that was already asked.
* **A request is in flight.** Streamed deltas are shown as soon as they arrive, reconciled
  against whatever the user typed meanwhile; if the user moved away from the text the request
  was made for, it is cancelled.

The model is asked through the :class:`~exocortex.llm.assistant.Assistant` (job kind
``"complete"``); results come back over the bus as ordinary Assist events, drained on this same
thread. The ghost window is driven by :class:`~exocortex.bus.CompletionEvent`s.

**Blind editors.** Monaco (VS Code, LeetCode, most web IDEs) exposes no text through
accessibility unless screen-reader mode is on -- only the caret. For such a field the text
before/after the caret is rebuilt from the screen reader's latest OCR frame around the caret
(:func:`exocortex.screen.context.context_at`); between OCR frames, typing is followed from the
caret's horizontal movement (one character width per character, monospace), assuming the user is
typing the suggestion, and the next frame (forced, at most once a second) re-synchronises with
what was really typed. Comparisons in this mode are fuzzy about OCR noise.

**Explanations on** (``cfg.explain.enabled``). The Assistant's "solve" answer is a *plan* whose
solution the ghost text follows: each tick the typed text is aligned against it
(:func:`exocortex.plan.align_plan`) and the next lines are shown straight from the plan -- no model
call per line, instant, and comments in it are optional to type (typing past one just drops it,
see :func:`consume_suggestion`). While the plan is still being written the completer waits for it
instead of asking the model; when the typed text leaves the plan it falls back to model
completions (which then carry the plan as context). Typing out the plan's last line of code
reports ``backend.plan_typed`` so the explanation of the finished solution can follow.

Nothing here injects keystrokes: the user types the suggestion themselves (the ghost text is
positioned and sized so that typing over it matches).
"""
from __future__ import annotations

import difflib
import logging
import queue
import threading
import time
# upstream finding 11 (ported from mac-cortex): OrderedDict lets _asked double as a
# recency-ordered set so the >500 prune below evicts the truly oldest prefixes.
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from exocortex.bus import (
    AssistDeltaEvent,
    AssistDoneEvent,
    AssistErrorEvent,
    CompletionEvent,
    ComponentStatusEvent,
    EventBus,
    drain,
)
from exocortex.config import Config
from exocortex.llm.prompts import clean_completion, normalize_comment_level
from exocortex.plan import PlanView, align_plan, consume_suggestion, plan_ghost
from exocortex.screen.caret import CaretInfo
from exocortex.screen.context import CaretContext, ScreenSnapshot, context_at

__all__ = ["Completer", "CompletionBackend", "TrackerLike", "consume_suggestion", "is_monospace", "reconcile_fuzzy"]

log = logging.getLogger(__name__)

#: How far the user may backspace into a shown suggestion's anchor before it counts as leaving.
MAX_BACKSPACE_CHARS = 40

#: A request that has produced nothing for this long is abandoned (model stalled / job dropped).
INFLIGHT_TIMEOUT_S = 30.0
#: Blind editors: never force a screen capture more often than this.
BLIND_CAPTURE_MIN_S = 1.0
#: Blind editors: how alike (difflib ratio) the OCR'd lines above the caret must be to count as
#: the same context as when the suggestion was made.
BLIND_ABOVE_SIMILARITY = 0.8

#: Font families that are monospace (case-insensitive substring match on the family name).
_MONO_HINTS = ("consolas", "cascadia", "courier", "mono", "menlo", "monaco", "fira code", "source code",
               "jetbrains", "hack", "inconsolata", "ubuntu mono", "dejavu sans mono", "lucida console")
#: Apps whose text fields are code editors when the font is unknown.
_CODE_APPS = ("code.exe", "code - insiders.exe", "cursor.exe", "devenv.exe", "notepad++.exe", "idea64.exe",
              "pycharm64.exe", "sublime_text.exe", "windowsterminal.exe", "rider64.exe", "clion64.exe")


class TrackerLike(Protocol):
    available: bool
    detail: str

    def attach(self) -> bool: ...
    def detach(self) -> None: ...
    def probe(self) -> CaretInfo | None: ...


class CompletionBackend(Protocol):
    """What the Completer needs from the Assistant. ``plan``/``plan_typed`` are only used with
    explanations on (a backend without them simply never serves a plan)."""

    def complete(self, prefix: str, suffix: str, field: str) -> int: ...
    def cancel_job(self, job_id: int) -> None: ...
    def plan(self) -> PlanView | None: ...
    def plan_typed(self, plan_id: int) -> None: ...


def _norm_key(text: str) -> str:
    """Whitespace-collapsed, case-folded key for de-duplicating asked prefixes (OCR noise)."""
    return " ".join(text.split()).casefold()


def reconcile_fuzzy(anchor: str, prefix: str, suggestion: str, skip_comments: bool = False) -> str | None:
    """Blind-mode `_reconcile`: tolerate OCR noise in the lines above the caret.

    The caret's own line must still extend `anchor`'s last line exactly (whitespace-tolerantly);
    the lines above only need to be similar. Returns the remaining suggestion, "" when consumed,
    None on deviation or when the context changed too much.
    """
    a_above, _, a_line = anchor.rpartition("\n")
    p_above, _, p_line = prefix.rpartition("\n")
    if a_above != p_above:
        if difflib.SequenceMatcher(None, a_above, p_above).ratio() < BLIND_ABOVE_SIMILARITY:
            return None
    if p_line.startswith(a_line):
        return consume_suggestion(suggestion, p_line[len(a_line):], skip_comments)
    if a_line.startswith(p_line) and len(a_line) - len(p_line) <= MAX_BACKSPACE_CHARS:
        return a_line[len(p_line):] + suggestion
    # whitespace-only differences in the typed line (OCR drops/adds spaces)
    if _norm_key(p_line).startswith(_norm_key(a_line)):
        typed = p_line.strip()[len(a_line.strip()):]
        return consume_suggestion(suggestion, typed, skip_comments)
    return None


def is_monospace(font_name: str, app: str) -> bool:
    """Best guess whether the target renders in a monospace font."""
    name = (font_name or "").casefold()
    if name:
        return any(h in name for h in _MONO_HINTS)
    return (app or "").casefold() in _CODE_APPS


@dataclass
class _Shown:
    anchor: str      # text before the caret when the suggestion was produced
    text: str        # the (cleaned) suggestion for that anchor


@dataclass
class _InFlight:
    anchor: str
    job_id: int
    raw: str = ""
    started: float = 0.0


class Completer:
    """Owns the caret-polling thread and the completion state machine."""

    def __init__(
        self,
        cfg: Config,
        bus: EventBus,
        backend: CompletionBackend,
        tracker: TrackerLike,
        clock: Callable[[], float] = time.monotonic,
        screen_source: Callable[[], ScreenSnapshot | None] | None = None,
        capture_now: Callable[[], None] | None = None,
    ) -> None:
        self._cfg = cfg
        self._bus = bus
        self._backend = backend
        self._tracker = tracker
        self._clock = clock
        self._screen_source = screen_source
        self._capture_now = capture_now
        # blind-editor state
        self._blind_ts: float | None = None
        self._blind_ctx: CaretContext | None = None
        self._blind_anchor: tuple[int, int] = (0, 0)
        self._last_capture_req = float("-inf")
        self._blind_active = False
        # explanations: plan-following state
        self._plan_line = 0                    # last solution line the caret was aligned to (tie-breaker)
        self._plan_id: int | None = None       # plan the ghost text currently follows
        self._plan_typed_ids: set[int] = set() # plans whose last code line was reported typed out
        self._plan_cache: tuple[tuple[int, int, bool, str], Any] | None = None   # last alignment, keyed by what it saw
        self.plan_served = 0                   # ticks that served ghost text from the plan (tests/logs)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._sub: "queue.Queue[Any] | None" = None
        self._paused = False
        self._reset_requested = False
        self._lock = threading.Lock()

        # state (touched only on the completer thread, except via the lock-protected flags)
        self._key: tuple[Any, ...] | None = None
        self._last_prefix: str | None = None
        self._last_change = float("-inf")
        self._last_request = float("-inf")
        # upstream finding 11 (ported from mac-cortex): an OrderedDict used as an
        # insertion/recency-ordered set of prefixes already asked for the current field, so the
        # >500 prune below drops the truly oldest entries, not a hash-order-dependent (effectively
        # random) subset of a plain set.
        self._asked: "OrderedDict[str, None]" = OrderedDict()
        self._shown: _Shown | None = None
        self._inflight: _InFlight | None = None
        self._last_event: CompletionEvent | None = None
        self._last_info: CaretInfo | None = None
        self._last_suffix = ""
        self._last_reject = ""
        self.requests = 0
        self.shown_count = 0

    # ---- lifecycle -------------------------------------------------------------------------
    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._sub = self._bus.subscribe()
        self._thread = threading.Thread(target=self._run, name="completer", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        t, self._thread = self._thread, None
        if t is not None:
            t.join(timeout=5.0)
            if t.is_alive():
                log.warning("completer: thread did not stop within timeout")
        if self._sub is not None:
            self._bus.unsubscribe(self._sub)
            self._sub = None
        self._publish_hidden()

    def set_paused(self, paused: bool) -> None:
        with self._lock:
            self._paused = paused

    @property
    def paused(self) -> bool:
        with self._lock:
            return self._paused

    def reset(self) -> None:
        """Forget the current suggestion / asked prefixes (session reset)."""
        with self._lock:
            self._reset_requested = True

    # ---- thread ------------------------------------------------------------------------------
    def _run(self) -> None:
        ok = False
        try:
            ok = self._tracker.attach()
        except Exception:  # noqa: BLE001
            log.exception("completer: tracker attach failed")
        if not ok:
            self._bus.publish(ComponentStatusEvent(component="completion", state="error", detail=self._tracker.detail))
            log.warning("completer: caret tracking unavailable (%s)", self._tracker.detail)
            return
        self._bus.publish(ComponentStatusEvent(component="completion", state="ready", detail=self._tracker.detail))
        log.info("completer: started (%s)", self._tracker.detail)
        period = max(0.03, self._cfg.completion.poll_ms / 1000.0)
        self._reset_requested = False
        try:
            while not self._stop.is_set():
                t0 = self._clock()
                try:
                    self._drain_bus()
                    self._tick(t0)
                except Exception:  # noqa: BLE001
                    log.exception("completer: tick failed")
                elapsed = self._clock() - t0
                self._stop.wait(max(0.0, period - elapsed))
        finally:
            self._cancel_inflight()
            try:
                self._tracker.detach()
            except Exception:  # noqa: BLE001
                pass
            log.info("completer: stopped")

    # ---- state machine ---------------------------------------------------------------------
    def _tick(self, now: float) -> None:
        with self._lock:
            paused = self._paused
            reset = self._reset_requested
            self._reset_requested = False
        if reset:
            self._clear_all()
        if paused:
            self._clear_all()
            return
        info = self._tracker.probe()
        if info is None:
            reject = getattr(self._tracker, "last_reject", "")
            if reject != self._last_reject:
                self._last_reject = reject
                log.info("completer: no target (%s)", reject or "nothing focused")
            self._clear_all()
            return
        self._last_reject = ""
        if info.key != self._key:
            self._clear_all()
            self._key = info.key
            self._last_change = now
        self._last_info = info
        self._blind_active = bool(info.blind)
        if info.blind:
            blind = self._blind_text(info, now)
            if blind is None:
                self._publish_hidden()
                return
            prefix, suffix = blind
        else:
            prefix, suffix = info.text_before, info.text_after
        self._last_suffix = suffix
        if prefix != self._last_prefix:
            self._last_prefix = prefix
            self._last_change = now

        if self._cfg.explain.enabled and self._serve_plan(prefix, info):
            return

        cfg = self._cfg.completion
        shown = self._shown
        if shown is not None:
            remaining = self._reconcile(shown, prefix)
            if remaining:
                self._show(remaining, info)
                return
            self._shown = None
            self._publish_hidden()
            if remaining is None:
                # Deviation: wait for the text to settle before asking again.
                self._last_change = now

        inflight = self._inflight
        if inflight is not None:
            if not (prefix.startswith(inflight.anchor) or inflight.anchor.startswith(prefix)):
                self._cancel_inflight()
            elif now - inflight.started > INFLIGHT_TIMEOUT_S and not inflight.raw:
                log.warning("completer: job %d produced nothing in %.0f s; abandoning it", inflight.job_id, INFLIGHT_TIMEOUT_S)
                self._cancel_inflight()
                # upstream finding 6 (ported from mac-cortex): _asked stores the NORMALISED key
                # (see _request below), so discarding the raw inflight.anchor here was a silent
                # no-op whenever normalisation actually changes the string (whitespace/case),
                # permanently blacklisting that prefix. Discard the normalised key instead.
                self._asked.pop(_norm_key(inflight.anchor), None)
            else:
                self._show_partial(inflight, prefix, info)
                return

        if (now - self._last_change) * 1000.0 < cfg.debounce_ms:
            return
        if (now - self._last_request) * 1000.0 < cfg.min_interval_ms:
            return
        if _norm_key(prefix) in self._asked:
            return
        if len(prefix.strip()) < cfg.min_prefix_chars and not info.field_hint:
            return
        self._request(info, prefix, suffix, now)

    def _reconcile(self, shown: _Shown, prefix: str) -> str | None:
        """Remaining ghost text for the current prefix; "" when consumed; None on deviation.

        A comment line the user types past is dropped, not a deviation, whenever the suggestion
        can carry comments: with explanations on (the plan's solution does) or at any
        `cfg.assist.comments` level but "none" (`prompts.system_complete` asks model completions
        for that level's comments even without a plan -- treating a skipped one as a deviation
        would throw the suggestion away and cost a model round-trip for nothing).
        """
        skip = self._cfg.explain.enabled or normalize_comment_level(self._cfg.assist.comments) != "none"
        if prefix.startswith(shown.anchor):
            return consume_suggestion(shown.text, prefix[len(shown.anchor):], skip)
        if shown.anchor.startswith(prefix) and len(shown.anchor) - len(prefix) <= MAX_BACKSPACE_CHARS:
            return shown.anchor[len(prefix):] + shown.text   # backspaced into the anchor: grow back
        if self._blind_active:
            return reconcile_fuzzy(shown.anchor, prefix, shown.text, skip)
        return None

    # ---- explanations: follow the plan -----------------------------------------------------------
    def _serve_plan(self, prefix: str, info: CaretInfo) -> bool:
        """Serve the ghost text from the Assistant's plan when the typed text follows it.

        Returns True when this tick is handled (ghost text shown from the plan, or deliberately
        nothing while the plan is still being written / has been typed out), False to fall back
        to model completions (no plan, or the typed text has left it).
        """
        getter = getattr(self._backend, "plan", None)
        if getter is None:
            return False
        try:
            plan = getter()
        except Exception:  # noqa: BLE001
            log.debug("completer: plan() failed", exc_info=True)
            return False
        if plan is None:
            return False
        if plan.id != self._plan_id:
            self._plan_id = plan.id
            self._plan_line = 0
        if not plan.solution.strip():
            if plan.done:
                return False                 # nothing usable in it (a prose answer with no body)
            self._cancel_inflight()          # the walkthrough is streaming: wait for the solution
            self._shown = None
            self._publish_hidden()
            return True
        key = (plan.id, len(plan.solution), plan.done, prefix)
        if self._plan_cache is not None and self._plan_cache[0] == key:
            cursor = self._plan_cache[1]              # nothing changed since the last tick
        else:
            cursor = align_plan(plan.solution, prefix, hint=self._plan_line)
            self._plan_cache = (key, cursor)
        if cursor is None:
            return False
        self._plan_line = cursor.line
        ghost = plan_ghost(plan.solution, cursor, self._cfg.completion.max_lines)
        if cursor.at_end and plan.done and plan.is_code and plan.id not in self._plan_typed_ids:
            self._plan_typed_ids.add(plan.id)
            log.info("completer: plan %d typed out", plan.id)
            try:
                self._backend.plan_typed(plan.id)
            except Exception:  # noqa: BLE001
                log.debug("completer: plan_typed failed", exc_info=True)
        self._cancel_inflight()
        if not ghost:
            self._shown = None
            self._publish_hidden()           # caught up with what has been generated, or all typed
            return True
        # Kept as the "shown" suggestion too: blind editors follow typing between OCR frames by
        # assuming the shown text is what is being typed (see _blind_text).
        self._shown = _Shown(anchor=prefix, text=ghost)
        self.plan_served += 1
        self._show(ghost, info)
        return True

    def _request(self, info: CaretInfo, prefix: str, suffix: str, now: float) -> None:
        field = f"{info.app} - {info.field_name[:60] or 'text field'}"
        if info.field_hint:
            field += f" (placeholder: {info.field_hint})"
        if info.blind:
            field = (f"{info.app} - code editor (its text is not accessible; the typed text and the "
                     f"text after the cursor were read from the screen and may contain OCR mistakes)")
        try:
            job_id = self._backend.complete(prefix, suffix, field)
        except Exception:  # noqa: BLE001
            log.exception("completer: request failed")
            return
        self._inflight = _InFlight(anchor=prefix, job_id=job_id, started=now)
        log.info("completer: requested job %d for %s (%d chars before caret, tail=%r)",
                 job_id, field[:40], len(prefix), prefix[-40:])
        # upstream finding 11 (ported from mac-cortex): move-to-end on every ask so _asked stays
        # ordered oldest -> newest by recency (re-asking a known prefix refreshes it), and the
        # prune below reliably evicts the oldest entries instead of a random ~80%.
        key = _norm_key(prefix)
        self._asked[key] = None
        self._asked.move_to_end(key)
        if len(self._asked) > 500:
            while len(self._asked) > 100:
                self._asked.popitem(last=False)
        self._last_request = now
        self.requests += 1

    # ---- blind editors (text from the screen) ---------------------------------------------------
    def _blind_text(self, info: CaretInfo, now: float) -> tuple[str, str] | None:
        """Prefix/suffix for an editor that exposes no text, from the latest OCR frame plus the
        caret's movement since that frame. None = nothing usable yet (a capture was requested)."""
        snap = self._screen_source() if self._screen_source is not None else None
        if snap is not None and snap.captured_at != self._blind_ts:
            ctx = context_at(snap, info.x, info.y, info.h, info.font_px)
            if ctx is not None:
                self._blind_ctx = ctx
                self._blind_ts = snap.captured_at
                self._blind_anchor = (info.x, info.y)
        ctx = self._blind_ctx
        if ctx is None or self._blind_ctx is None:
            self._request_capture(now)
            return None
        dx = info.x - self._blind_anchor[0]
        dy = info.y - self._blind_anchor[1]
        line_tol = max(4.0, (info.h or 16) * 0.5)
        base = ctx.prefix
        if abs(dy) > line_tol:
            # The caret changed line since the frame: what was typed is unknown until the next OCR.
            self._request_capture(now)
            return None
        moved = round(dx / ctx.char_w) if ctx.char_w > 0 else 0
        if moved > 0:
            self._request_capture(now)
            shown = self._shown
            text = shown.text if shown is not None else (self._inflight.raw if self._inflight is not None else "")
            anchor = shown.anchor if shown is not None else (self._inflight.anchor if self._inflight is not None else None)
            if anchor is not None and _norm_key(anchor) == _norm_key(base) and text:
                first_line = text.split("\n", 1)[0]
                if moved <= len(first_line):
                    return base + first_line[:moved], ctx.suffix   # optimistic: typing the suggestion
            return None                                          # typed something we cannot see yet
        if moved < 0:
            base = base[:moved] if -moved <= len(ctx.current_left) else base
            self._request_capture(now)
        return base, ctx.suffix

    def _request_capture(self, now: float) -> None:
        if self._capture_now is None or now - self._last_capture_req < BLIND_CAPTURE_MIN_S:
            return
        self._last_capture_req = now
        try:
            self._capture_now()
        except Exception:  # noqa: BLE001
            log.debug("completer: capture_now failed", exc_info=True)

    def _cancel_inflight(self) -> None:
        inflight, self._inflight = self._inflight, None
        if inflight is not None:
            try:
                self._backend.cancel_job(inflight.job_id)
            except Exception:  # noqa: BLE001
                log.debug("completer: cancel failed", exc_info=True)

    def _clear_all(self) -> None:
        self._cancel_inflight()
        self._shown = None
        self._key = None
        self._last_prefix = None
        self._last_suffix = ""
        self._asked.clear()
        self._blind_ctx = None
        self._blind_ts = None
        self._plan_line = 0
        self._plan_cache = None
        if len(self._plan_typed_ids) > 100:
            self._plan_typed_ids = set(sorted(self._plan_typed_ids)[-20:])
        self._publish_hidden()

    # ---- results from the assistant -----------------------------------------------------------
    def _drain_bus(self) -> None:
        if self._sub is None:
            return
        for ev in drain(self._sub):
            inflight = self._inflight
            if inflight is None:
                continue
            if isinstance(ev, AssistDeltaEvent) and ev.job_id == inflight.job_id:
                inflight.raw += ev.delta
            elif isinstance(ev, AssistDoneEvent) and ev.job_id == inflight.job_id:
                self._inflight = None
                if ev.cancelled:
                    continue
                text = clean_completion(inflight.anchor, ev.text or inflight.raw, self._cfg.completion.max_lines)
                if text:
                    self._shown = _Shown(anchor=inflight.anchor, text=text)
                    self.shown_count += 1
                    # Shown on the next tick with a fresh caret position; show now with the last one.
                    if self._last_info is not None and self._last_prefix is not None:
                        remaining = self._reconcile(self._shown, self._last_prefix)
                        if remaining:
                            self._show(remaining, self._last_info)
                        else:
                            self._shown = None
            elif isinstance(ev, AssistErrorEvent) and ev.job_id == inflight.job_id:
                self._inflight = None
                log.warning("completer: completion failed: %s", ev.message)

    def _show_partial(self, inflight: _InFlight, prefix: str, info: CaretInfo) -> None:
        if not inflight.raw:
            return
        text = clean_completion(inflight.anchor, inflight.raw, self._cfg.completion.max_lines)
        if not text:
            return
        remaining = self._reconcile(_Shown(anchor=inflight.anchor, text=text), prefix)
        if remaining:
            self._show(remaining, info)
        else:
            self._publish_hidden()

    # ---- ghost events ---------------------------------------------------------------------------
    def _show(self, text: str, info: CaretInfo) -> None:
        font_px = info.font_px * self._cfg.completion.font_scale if info.font_px else 0.0
        current_line = (self._last_prefix or "").rsplit("\n", 1)[-1] if info.blind else info.current_line
        ev = CompletionEvent(
            visible=True, text=text, x=info.x, y=info.y, line_h=info.h, font_px=font_px,
            font_name=info.font_name, mono=is_monospace(info.font_name, info.app) or info.blind,
            current_line=current_line, app=info.app,
        )
        last = self._last_event
        if last is not None and last.visible and (last.text, last.x, last.y, last.line_h, last.font_px) == (
            ev.text, ev.x, ev.y, ev.line_h, ev.font_px
        ):
            return
        self._last_event = ev
        self._bus.publish(ev)

    def _publish_hidden(self) -> None:
        last = self._last_event
        if last is not None and not last.visible:
            return
        self._last_event = CompletionEvent(visible=False)
        self._bus.publish(self._last_event)

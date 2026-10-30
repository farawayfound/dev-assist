"""DashboardService: keeps the dashboard store current and publishes its snapshots.

Owns a "dashboard-timer" daemon thread that, every ``TICK_S`` seconds, drains its own bus
subscription (the running notes land in the store as soon as the notes worker publishes them),
re-checks the repository's toolkit install through the project context, and starts a background
refresh when one is due: ``[dashboard].refresh_interval_s`` has passed since the last one AND
either ``min_new_words`` of new speech arrived since the last refresh or the screen changed
substantially (the same two triggers as the Notes tab's insights passes, which these replace while
``[dashboard].track`` is on).

A refresh is a job of kind "dashboard" on the assistant's own lane system (lowest priority but
one: any other job preempts a background refresh, which is retried ``RETRY_S`` later; an on-demand
refresh is never dropped). The assistant calls back into this service to BUILD the prompt
(:meth:`_build`: the open items with ids, the transcript since the watermark, the screen, the
repo / toolkit blocks) and to SETTLE the result (:meth:`_settle`: parse the changes, apply them to
the store, move the watermark, publish a ``DashboardUpdatedEvent`` and mirror the open questions /
blind spots into the Notes tab's Insights pane). Manual edits from the view (``item_action`` /
``add_item``) go straight to the store and publish the same event.
"""
from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from exocortex.bus import ComponentStatusEvent, DashboardUpdatedEvent, EventBus, NotesUpdatedEvent, drain
from exocortex.config import Config
from exocortex.dashboard import prompts as dprompts
from exocortex.dashboard.store import KINDS, DashboardSnapshot, DashboardStore
from exocortex.llm import prompts as lprompts

log = logging.getLogger(__name__)

#: Module attributes (read fresh on every use) so tests can shrink them.
TICK_S: float = 5.0
RETRY_S: float = 30.0
SCREEN_JACCARD: float = 0.6
MIN_SCREEN_LINES: int = 5
#: Characters of transcript BEFORE the watermark handed over as context for the new part.
EARLIER_CHARS: int = 1200
DB_NAME = "dashboard.sqlite"
MD_NAME = "dashboard.md"


def _normalized_line_set(text: str) -> set[str]:
    return {" ".join(line.split()).casefold() for line in (text or "").splitlines() if line.strip()}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class DashboardService:
    def __init__(
        self,
        cfg: Config,
        bus: EventBus,
        transcript: Any,
        assistant: Any,
        clock: Callable[[], float],
        *,
        project: Any | None = None,
        screen_provider: Callable[[], str] | None = None,
        folder: Path | None = None,
    ) -> None:
        self._cfg = cfg
        self._bus = bus
        self._transcript = transcript
        self._assistant = assistant
        self._clock = clock
        self._project = project
        self._screen_provider = screen_provider
        self._folder = folder
        self._lock = threading.RLock()
        self._store = DashboardStore(self._db_path(folder))
        self._last_start = clock()          # the first background refresh waits a full interval
        self._seg_id: int | None = None
        self._screen_lines: set[str] = set()
        self._jobs: dict[int, tuple[dict[str, int], tuple[str, ...]]] = {}   # job id -> (aliases, kinds)
        self._paused = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._sub: Any = None
        if hasattr(assistant, "set_dashboard_hooks"):
            assistant.set_dashboard_hooks(self._build, self._settle)

    @staticmethod
    def _db_path(folder: Path | None) -> Path | None:
        return (folder / DB_NAME) if folder is not None else None

    # ---- lifecycle --------------------------------------------------------------------------
    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._sub = self._bus.subscribe()
        self._thread = threading.Thread(target=self._timer_loop, name="dashboard-timer", daemon=True)
        self._thread.start()
        self.publish(auto=False, detail="")

    def stop(self) -> None:
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None and thread.is_alive():
            thread.join(timeout=5.0)
        if self._sub is not None:
            self._bus.unsubscribe(self._sub)
            self._sub = None
        with self._lock:
            self._store.close()

    def reset(self, folder: Path | None = None) -> None:
        """A new session: a fresh store (in `folder`, or memory), fresh watermarks."""
        with self._lock:
            self._store.close()
            self._folder = folder
            self._store = DashboardStore(self._db_path(folder))
            self._seg_id = None
            self._screen_lines = set()
            self._jobs.clear()
            self._last_start = self._clock()
        self.publish(auto=False, detail="new session")

    def set_paused(self, paused: bool) -> None:
        self._paused = bool(paused)

    @property
    def store(self) -> DashboardStore:
        with self._lock:
            return self._store

    @property
    def tracking(self) -> bool:
        return bool(self._cfg.dashboard.track)

    @property
    def folder(self) -> Path | None:
        return self._folder

    # ---- public actions ---------------------------------------------------------------------
    def refresh_now(self, sections: Sequence[str] = ()) -> int | None:
        """One refresh now (on demand: never dropped, works while paused); the job id."""
        secs = dprompts.normalize_sections(sections)
        if not hasattr(self._assistant, "dashboard_now"):
            return None
        return self._assistant.dashboard_now(auto=False, sections=secs)

    def item_action(self, item_id: int, action: str, text: str = "") -> bool:
        now = self._clock()
        store = self.store
        action = (action or "").strip().casefold()
        current = store.get(int(item_id))
        if action in ("answer", "answered", "done", "resolve", "resolved") and current is not None \
                and current.kind == "constraint":
            action = "dismiss"      # a rule is in force or lifted, never "answered"
        if action in ("answer", "answered", "done", "resolve", "resolved"):
            item = store.resolve(int(item_id), text, now=now)
            detail = "marked answered"
        elif action in ("dismiss", "drop", "remove"):
            item = store.drop(int(item_id), now=now)
            detail = "dismissed"
        elif action == "reopen":
            item = store.reopen(int(item_id), now=now)
            detail = "reopened"
        elif action == "edit":
            item = store.edit(int(item_id), text, now=now)
            detail = "edited"
        else:
            log.warning("dashboard: unknown item action %r", action)
            return False
        if item is None:
            return False
        self.publish(auto=False, detail=detail)
        return True

    def add_item(self, kind: str, text: str, parent_id: int | None = None) -> bool:
        """Add an item by hand; `parent_id` makes a task a sub-task, or attaches a goal / prompt
        to an action item."""
        kind = (kind or "").strip().casefold()
        if kind not in KINDS:
            return False
        item = self.store.add_item(kind, text, now=self._clock(), source="user",
                                   parent_id=int(parent_id) if parent_id is not None else None)
        if item is None:
            return False
        self.publish(auto=False, detail="added")
        return True

    def snapshot(self) -> DashboardSnapshot:
        d = self._cfg.dashboard
        return self.store.snapshot(resolved_keep=max(0, int(d.resolved_keep)), show_resolved=bool(d.show_resolved))

    def markdown(self) -> str:
        return self.store.to_markdown()

    def publish(self, *, auto: bool, detail: str) -> None:
        """A `DashboardUpdatedEvent` with the current snapshot; the open questions / blind spots
        also go to the assistant's insights document (the Notes tab's pane, insights.md); the
        board is written to the session folder as markdown."""
        snap = self.snapshot()
        self._bus.publish(DashboardUpdatedEvent(items=snap.items, notes=snap.notes, notes_at=snap.notes_at,
                                                generated_at=self._clock(), auto=auto, detail=detail))
        replace = getattr(self._assistant, "replace_insights", None)
        if callable(replace) and self.tracking:
            try:
                replace({
                    "questions": dprompts.render_open_markdown(snap.of("question")),
                    "blindspots": dprompts.render_open_markdown(snap.of("blindspot")),
                }, auto=auto)
            except Exception:  # noqa: BLE001
                log.exception("dashboard: mirroring the insights failed")
        self._write_markdown()

    def _write_markdown(self) -> None:
        folder = self._folder
        if folder is None:
            return
        try:
            from exocortex.session import _atomic_write_text

            _atomic_write_text(folder / MD_NAME, self.markdown())
        except Exception:  # noqa: BLE001 - the file is a convenience, never a reason to fail
            log.debug("dashboard: could not write %s", MD_NAME, exc_info=True)

    # ---- the timer ---------------------------------------------------------------------------
    def _timer_loop(self) -> None:
        log.info("dashboard-timer: started")
        while not self._stop.wait(TICK_S):
            try:
                self.tick()
            except Exception:  # noqa: BLE001 - one bad tick must never kill the timer
                log.exception("dashboard-timer: tick failed")
        log.info("dashboard-timer: stopping")

    def tick(self) -> bool:
        """One timer tick (public for tests): notes from the bus, a toolkit rescan, and a
        background refresh when due. True when a refresh was queued."""
        if self._sub is not None:
            for ev in drain(self._sub):
                if isinstance(ev, NotesUpdatedEvent):
                    self.store.set_notes(ev.markdown, ev.generated_at)
                    self.publish(auto=True, detail="notes updated")
        if self._project is not None:
            try:
                if self._project.refresh():
                    detail = f"Project: {self._project.context_summary()}"
                    log.info("dashboard: %s", detail)
                    self._bus.publish(ComponentStatusEvent(component="project", state="info", detail=detail))
            except Exception:  # noqa: BLE001
                log.exception("dashboard: project refresh failed")
        return self._maybe_auto()

    def _maybe_auto(self) -> bool:
        d = self._cfg.dashboard
        if not self.tracking or self._paused:
            return False
        if not getattr(self._assistant, "is_running", True) or not hasattr(self._assistant, "dashboard_now"):
            return False
        manual = bool(getattr(getattr(self._cfg, "handsfree", None), "manual", False))
        if manual:
            return False
        try:
            interval = float(d.refresh_interval_s)
            min_words = max(0, int(d.min_new_words))
        except (TypeError, ValueError):
            return False
        if interval <= 0:
            return False
        now = self._clock()
        with self._lock:
            if now - self._last_start < interval:
                return False
            since_id = self._seg_id
            seen = set(self._screen_lines)
        new_words = self._transcript.word_count(since_id=since_id)
        due = new_words > 0 and new_words >= min_words
        if not due:
            current = self._current_screen_lines()
            due = len(current) >= MIN_SCREEN_LINES and _jaccard(current, seen) < SCREEN_JACCARD
        if not due:
            return False
        job_id = self._assistant.dashboard_now(auto=True, sections=())
        if job_id is None:
            return False
        with self._lock:
            self._last_start = now
        return True

    def _current_screen_lines(self) -> set[str]:
        provider = self._screen_provider
        if provider is None:
            return set()
        try:
            return _normalized_line_set(provider() or "")
        except Exception:  # noqa: BLE001
            return set()

    # ---- the assistant's hooks (assist-worker thread) ------------------------------------------
    def _build(self, job: Any, screen_text: str, until_id: int | None) -> tuple[str, str, int]:
        cfg = self._cfg
        d, a = cfg.dashboard, cfg.assist
        kinds = dprompts.kinds_for(getattr(job, "sections", ()), suggest_prompts=bool(d.suggest_prompts))
        store = self.store
        current, aliases = dprompts.render_current(
            {k: store.open_items(k) for k in kinds}, kinds, limit=max(1, int(d.max_open_items)))
        with self._lock:
            since_id = self._seg_id
            self._jobs[job.id] = (aliases, kinds)
        budget = max(400, int(cfg.llm.context_tokens) * 4)
        new = self._transcript.window_text(budget, until_id=until_id, since_id=since_id)
        earlier = self._transcript.window_text(EARLIER_CHARS, until_id=since_id) if since_id is not None else ""
        repo = detail = toolkit = example = ""
        if self._project is not None:
            try:
                toolkit = self._project.toolkit_block()
                example = self._project.toolkit_example()
                detail = self._project.toolkit_lookup(new, screen_text)
                repo = self._project.repo_block(new, screen_text)
            except Exception:  # noqa: BLE001 - project context is optional, never a reason to fail a refresh
                log.exception("dashboard: project lookup failed")
        system = dprompts.system_dashboard(
            persona=a.persona, role=a.role, context=a.session_context, max_new=int(d.max_new_items),
            kinds=kinds, toolkit=toolkit, example=example,
        )
        user = dprompts.user_dashboard(current=current, new=new, earlier=earlier, screen=screen_text,
                                       repo=repo, toolkit_detail=detail)
        return system, user, int(cfg.llm.dashboard_max_tokens)

    @staticmethod
    def _clean_prompt(text: str) -> str:
        """A suggested prompt as it would be typed: without the backticks or quotes a model wraps it in."""
        return (text or "").strip().strip("`\"'").strip()

    def _prompt_installed(self, text: str) -> bool:
        """False for a prompt that opens with a slash command / agent the installed toolkit does not
        have (a hallucinated one would only fail when typed). Always True without a toolkit."""
        if self._project is None:
            return True
        try:
            return bool(self._project.check_prompt(text))
        except Exception:  # noqa: BLE001
            return True

    def _settle(self, job: Any, *, ok: bool, text: str = "", until_id: int | None = None, screen_text: str = "") -> None:
        with self._lock:
            aliases, kinds = self._jobs.pop(job.id, ({}, KINDS))
        auto = bool(getattr(job, "auto", False))
        if not ok:
            if auto:
                try:
                    interval = float(self._cfg.dashboard.refresh_interval_s)
                except (TypeError, ValueError):
                    interval = 0.0
                with self._lock:
                    started = float(getattr(job, "started", 0.0) or 0.0)
                    if self._last_start <= started:
                        self._last_start = started - max(0.0, interval - RETRY_S)
            self.publish(auto=auto, detail="refresh cancelled")
            return
        now = self._clock()
        detail = "nothing new"
        if (text or "").strip() and not lprompts.is_skip(text):
            update = dprompts.parse_update(text, kinds, max_new=int(self._cfg.dashboard.max_new_items))
            # A rule is never "done": a constraint the model resolves is lifted (dropped).
            resolves = [(aliases[a], ans) for a, ans in update.resolves if a in aliases and not a.startswith("C")]
            drops = [aliases[a] for a in update.drops if a in aliases]
            drops += [aliases[a] for a, _ in update.resolves if a in aliases and a.startswith("C")]
            unknown = [a for a, _ in update.resolves if a not in aliases] + [a for a in update.drops if a not in aliases]
            if unknown:
                log.info("dashboard: the model named ids that were not shown: %s", ", ".join(unknown))
            # A new action item takes the next free N number, in the order the model wrote them (the
            # prompt says so): a sub-task, goal or prompt written after it can name it before it has
            # an id. The store resolves such a label against the items added earlier in the batch.
            next_n = 1 + max((int(a[1:]) for a in aliases if a.startswith("N") and a[1:].isdigit()), default=0)
            adds: list[tuple] = []
            for add in update.adds:
                body = add.text
                if add.kind == "prompt":
                    body = self._clean_prompt(body)
                    if not body or not self._prompt_installed(body):
                        log.info("dashboard: dropped a suggested prompt that names something not installed: %r", add.text[:80])
                        continue
                ref = None
                if add.kind == "nextstep":
                    ref, next_n = f"N{next_n}", next_n + 1
                parent = aliases[add.parent] if add.parent in aliases else (add.parent or None)
                adds.append((add.kind, body, parent, ref))
            result = self.store.apply(adds, resolves, drops, now=now,
                                      max_new_per_kind=int(self._cfg.dashboard.max_new_items))
            detail = "updated: " + result.describe() if result.changed else "nothing new"
            if update.unparsed:
                log.debug("dashboard: unparsed lines: %r", update.unparsed[:5])
        with self._lock:
            if until_id is not None:
                self._seg_id = until_id
            self._screen_lines = _normalized_line_set(screen_text)
        self.publish(auto=auto, detail=detail)
        log.info("dashboard: %s (%s)", detail, "auto" if auto else "on demand")

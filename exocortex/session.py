"""SessionRecorder: owns a per-session folder and periodically snapshots it to disk.

On `start()` it creates ``<sessions_dir>/<YYYY-MM-DD_HH-MM-SS>/`` (a numbered suffix is added if
that exact folder already exists), subscribes to the bus, and starts a daemon autosave thread
that saves whenever a transcript/notes/assist event has been seen since the last save.
`stop()`/`export()` save immediately; `stop()` is always safe to call, including before `start()`
(a no-op then). `export()` needs a session to save, so it raises like the `folder` property does
if called before `start()`. Calling `start()` again while already running stops the previous
session first, so no subscription or thread is ever leaked.

Depends structurally (via ``Protocol``, not a concrete import) on a transcript store shaped
like ``exocortex.transcript.Transcript`` and an assistant shaped like
``exocortex.llm.assistant.Assistant``, so this module has no import-time dependency on either
(both are owned/written elsewhere) and can be exercised in tests with simple fakes.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

from exocortex.bus import (
    AssistDoneEvent,
    EventBus,
    FinalTranscriptEvent,
    InsightsUpdatedEvent,
    NotesUpdatedEvent,
    SessionEvent,
    SpeakerRenamedEvent,
    TranscriptSegment,
    drain,
)
from exocortex.config import Config, config_as_dict

log = logging.getLogger(__name__)

_FOLDER_FMT = "%Y-%m-%d_%H-%M-%S"


def _atomic_write_text(path: Path, text: str) -> None:
    """Write `text` to `path` atomically: fully written to a sibling `<name>.tmp` file, then
    moved onto `path` with `os.replace`.

    upstream finding 5 (ported from mac-cortex): session files were previously written with a
    plain `Path.write_text`, so a crash, force-quit or power loss mid-write could leave
    transcript.md/transcript.jsonl/notes.md/answers.md/session.json truncated or holding invalid
    JSON -- exactly the durable record SessionRecorder exists to produce. `os.replace` is a
    single filesystem rename, so each file's save is all-or-nothing: a crash before it completes
    leaves that file's previous good version on disk untouched. (The five files are still
    written one after another, so a crash between two of them can leave them from different
    ticks -- as before -- but each one is now internally consistent, never truncated or
    half-written.)

    Windows note: on NTFS `os.replace` is atomic, but it raises `PermissionError` while another
    process (AV, OneDrive, an editor) is transiently holding the destination file open -- exactly
    the class of transient failure the `_run()` guard below (finding 2) now survives, since the
    next autosave tick just retries.
    """
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


@runtime_checkable
class Transcript(Protocol):
    """Structural contract this module needs from `exocortex.transcript.Transcript`."""

    def to_markdown(self, title: str) -> str: ...
    def to_jsonl(self) -> str: ...
    def segments(self, since_id: int | None = None) -> list[TranscriptSegment]: ...


@runtime_checkable
class Assistant(Protocol):
    """Structural contract this module needs from `exocortex.llm.assistant.Assistant`."""

    notes_markdown: str
    answers: list[tuple[float, str, str]]


def _format_mmss(t: float) -> str:
    """Render seconds-since-session-start as "mm:ss", or "h:mm:ss" once an hour has passed.

    Matches `exocortex.transcript._format_timestamp` so timestamps in answers.md line up
    with the ones in transcript.md for meetings that run past an hour.
    """
    total_seconds = max(0, round(t))
    hours, rem = divmod(total_seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


class SessionRecorder:
    """Creates the session folder, autosaves on change, and writes a final snapshot on stop.

    Files written on every save: transcript.md, transcript.jsonl, notes.md, answers.md,
    insights.md (the co-pilot's questions to ask + blind spots / follow-ups, when the assistant
    keeps them), session.json (config snapshot + speaker labels collected from
    SpeakerRenamedEvent + the session start time).
    """

    def __init__(self, cfg: Config, bus: EventBus, transcript: Transcript, assistant: Assistant) -> None:
        self._cfg = cfg
        self._bus = bus
        self._transcript = transcript
        self._assistant = assistant
        self._folder: Path | None = None
        self._started_at: datetime | None = None
        self._speaker_labels: dict[str, str] = {}
        self._queue: Any = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._dirty = threading.Event()
        self._state_lock = threading.Lock()
        self._save_lock = threading.Lock()

    @property
    def folder(self) -> Path:
        """The session folder. Raises RuntimeError until `start()` has been called."""
        if self._folder is None:
            raise RuntimeError("SessionRecorder.start() must be called before accessing folder")
        return self._folder

    def start(self) -> Path:
        """Create the session folder, subscribe to the bus, and start the autosave thread.

        If this recorder is already running, the previous session is stopped first (final
        snapshot written, bus subscription released) so calling `start()` again never leaks
        a subscription or an orphaned autosave thread.
        """
        if self._thread is not None and self._thread.is_alive():
            log.warning("SessionRecorder.start() called while already running; stopping previous session first")
            self.stop()
        self._started_at = datetime.now()
        folder = self._reserve_folder(self._started_at)
        folder.mkdir(parents=True, exist_ok=True)
        self._folder = folder
        self._speaker_labels = {}
        self._stop_event.clear()
        self._dirty.clear()
        self._queue = self._bus.subscribe()
        self._thread = threading.Thread(target=self._run, name="SessionRecorder", daemon=True)
        self._thread.start()
        self._bus.publish(SessionEvent(kind="new", path=str(folder)))
        log.info("session started: %s", folder)
        return folder

    def _reserve_folder(self, started_at: datetime) -> Path:
        """Pick `<sessions_dir>/<started_at>`, or a `_2`, `_3`, ... suffixed sibling if a folder
        with that exact timestamp already exists (two sessions started within the same second),
        so a fast restart never silently merges into (and overwrites) a previous session."""
        base = self._cfg.sessions_dir()
        name = started_at.strftime(_FOLDER_FMT)
        folder = base / name
        n = 2
        while folder.exists():
            folder = base / f"{name}_{n}"
            n += 1
        return folder

    def stop(self) -> None:
        """Stop the autosave thread and write a final snapshot.

        Safe to call even if `start()` was never called (a no-op in that case) or if the
        session was already stopped.
        """
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        if self._folder is None:
            log.debug("SessionRecorder.stop() called before start(); nothing to save")
            return
        try:
            # upstream finding 2 (ported from mac-cortex): the final save is wrapped exactly like
            # the autosave tick in _run() below -- this method's own docstring promises it is
            # "always safe to call", so the same transient OSError (full disk, an AV/OneDrive-style
            # momentary file lock) that the periodic tick now survives must not be allowed to
            # propagate out of here either. The bus unsubscribe below still runs even when this
            # save fails, so a failed stop() never leaks the subscription.
            self._drain_pending()
            self._save(kind="saved")
        except Exception:  # noqa: BLE001 - stop() must always be safe to call
            log.exception("session: final save on stop() failed; session folder may be stale: %s", self._folder)
        if self._queue is not None:
            self._bus.unsubscribe(self._queue)
            self._queue = None
        log.info("session stopped: %s", self._folder)

    def export(self) -> Path:
        """Save immediately (picking up any not-yet-processed bus events) and return the folder.

        upstream finding 2 (ported from mac-cortex): unlike `stop()`, a save failure here is
        deliberately re-raised, not swallowed. `export()` is the user-initiated "save now" action
        (wired to the Export menu item); its caller (`exocortex.ui.overlay._on_export`) catches
        the exception and reports it back to the user as a toast, so swallowing it here would make
        a failed export silently look like a successful one.
        """
        self._drain_pending()
        self._save(kind="exported")
        return self.folder

    def _run(self) -> None:
        """Autosave loop: wake every `autosave_interval_s`, save only if something changed.

        upstream finding 2 (ported from mac-cortex): the tick body is wrapped in try/except so one
        bad save -- a transient OSError from a full disk, an AV/OneDrive-style process briefly
        holding a file open, or any other I/O hiccup -- is logged and retried next interval instead
        of silently killing this daemon thread for the rest of the session (README's "keeps what
        it last autosaved" promise otherwise stops being true the moment autosave dies).
        """
        interval = max(1, self._cfg.paths.autosave_interval_s)
        while not self._stop_event.is_set():
            try:
                self._drain_pending()
                if self._dirty.is_set():
                    self._save(kind="saved")
            except Exception:  # noqa: BLE001 - keep autosaving even after one bad tick
                log.exception("session: autosave tick failed; will retry in %ds", interval)
            self._stop_event.wait(interval)

    def _drain_pending(self) -> None:
        """Process any events queued on the bus subscription without blocking."""
        if self._queue is None:
            return
        for event in drain(self._queue):
            self._handle_event(event)

    def _handle_event(self, event: Any) -> None:
        with self._state_lock:
            if isinstance(event, (FinalTranscriptEvent, NotesUpdatedEvent, AssistDoneEvent, InsightsUpdatedEvent)):
                self._dirty.set()
            elif isinstance(event, SpeakerRenamedEvent):
                self._speaker_labels[event.speaker_id] = event.new_label
                self._dirty.set()

    def _save(self, *, kind: Literal["saved", "exported"]) -> None:
        folder = self.folder
        with self._state_lock:
            speakers_snapshot = dict(self._speaker_labels)
        with self._save_lock:
            title = folder.name
            # upstream finding 5 (ported from mac-cortex): write each file atomically (see
            # _atomic_write_text) so a crash mid-save cannot corrupt the durable record.
            _atomic_write_text(folder / "transcript.md", self._transcript.to_markdown(title))
            _atomic_write_text(folder / "transcript.jsonl", self._transcript.to_jsonl())
            _atomic_write_text(folder / "notes.md", self._assistant.notes_markdown or "")
            _atomic_write_text(folder / "answers.md", self._render_answers())
            # getattr: the Protocol above predates insights, and test fakes may not carry them.
            insights = getattr(self._assistant, "insights_markdown", "") or ""
            if insights or (folder / "insights.md").exists():
                _atomic_write_text(folder / "insights.md", insights)
            _atomic_write_text(folder / "session.json", self._render_session_json(speakers_snapshot))
            self._dirty.clear()
        self._bus.publish(SessionEvent(kind=kind, path=str(folder)))
        log.debug("session %s: %s", kind, folder)

    def _render_answers(self) -> str:
        blocks = [f"## [{_format_mmss(t)}] {question}\n\n{answer}\n" for t, question, answer in self._assistant.answers]
        return "\n".join(blocks)

    def _render_session_json(self, speakers: dict[str, str]) -> str:
        payload = {
            "config": config_as_dict(self._cfg),
            "speakers": speakers,
            "started_at": (self._started_at or datetime.now()).isoformat(),
        }
        return json.dumps(payload, indent=2, default=str)

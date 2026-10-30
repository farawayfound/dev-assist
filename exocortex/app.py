"""Composition root: wires audio → speech → transcript → assistant → session → UI.

`python -m exocortex` runs the overlay. `--no-ui` runs headless and prints to stdout.
`--doctor` prints environment diagnostics and exits.
"""
from __future__ import annotations

import argparse
import dataclasses
import logging
import shlex
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

from exocortex import cuda_env
from exocortex.actions import EngineActions
from exocortex.bus import (
    CompletionModeEvent,
    AssistDeltaEvent,
    AssistDoneEvent,
    AssistErrorEvent,
    AssistStartedEvent,
    CheckpointEvent,
    ComponentStatusEvent,
    ContextLibraryEvent,
    DashboardModeEvent,
    DashboardUpdatedEvent,
    Event,
    EventBus,
    FinalTranscriptEvent,
    HandsFreeEvent,
    InsightsUpdatedEvent,
    ManualAdvanceEvent,
    ModelInfoEvent,
    ModelLibraryEvent,
    ModelPullEvent,
    ModelSearchEvent,
    NotesUpdatedEvent,
    ScreenStatusEvent,
    ScreenTextEvent,
    Source,
    SpeakerRenamedEvent,
    drain,
)
from exocortex.config import Config, load_config, project_root, save_config
from exocortex.context import ContextManager, ContextStore
from exocortex.llm.client import make_client
from exocortex.llm.route import resolve_route
from exocortex.transcript import Transcript

log = logging.getLogger(__name__)

#: Cadence of the gpu-guard sampler (module-level so tests can shrink it).
GPU_GUARD_INTERVAL_S = 5.0

# review round two, 2026-09-27 (E4): exocortex.llm.library.ModelProfile's numeric fields, so
# set_model_profile can coerce whatever the settings dialog handed it (its widgets round-trip
# through plain text) instead of storing a str that later blows up `list(profile.extra_args)` or
# an int-typed llama-server flag built from a non-numeric value.
_PROFILE_INT_FIELDS = frozenset({"ctx_size", "gpu_layers", "threads", "parallel", "top_k"})
_PROFILE_FLOAT_FIELDS = frozenset({"temperature", "top_p", "min_p"})
#: Sentinel: this value could not be coerced -- drop the key so ModelProfile's own default (not
#: necessarily None) applies, rather than storing something list(...)/int(...) chokes on later.
_PROFILE_DROP = object()


def _coerce_extra_args(value: object) -> list[str]:
    """`ModelProfile.extra_args` the way `LlamaServerManager` uses it: `list(profile.extra_args)`
    exploded straight into llama-server's argv. The Tuning tab's free-text field hands this back
    as a str -- e.g. the placeholder text "[]" for "none", or "--foo bar --baz" -- so split it the
    way a shell would (``shlex.split``; a stray unmatched quote falls back to ``str.split``) rather
    than storing the str itself, which iterates character by character."""
    if isinstance(value, str):
        text = value.strip()
        if text in ("", "[]"):
            return []
        try:
            return shlex.split(text, posix=False)
        except ValueError:
            return text.split()
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return []


def _coerce_profile_field(name: str, value: object) -> object:
    """One `set_model_profile` value, coerced to what `ModelProfile.__init__` expects; returns
    `_PROFILE_DROP` when it cannot be parsed (the caller logs and omits the key)."""
    if name == "extra_args":
        return _coerce_extra_args(value)
    if value is None or name not in _PROFILE_INT_FIELDS and name not in _PROFILE_FLOAT_FIELDS:
        return value   # str/bool fields (kv_cache_type, flash_attn, thinking, note, ...): as given
    if isinstance(value, bool):
        return _PROFILE_DROP    # bool is an int subclass -- never silently accepted as ctx_size etc.
    caster = int if name in _PROFILE_INT_FIELDS else float
    if isinstance(value, (int, float)):
        return caster(value)
    try:
        return caster(str(value).strip())
    except (TypeError, ValueError):
        return _PROFILE_DROP


class _RecordingBus:
    """Bus facade handed to the SpeechPipeline: records finals/renames into the Transcript
    *before* fan-out, so every subscriber (Assistant, UI, recorder) sees a transcript that
    already contains the segment it is being told about."""

    def __init__(self, bus: EventBus, transcript: Transcript) -> None:
        self._bus = bus
        self._transcript = transcript

    def publish(self, event: Event) -> None:
        if isinstance(event, FinalTranscriptEvent):
            self._transcript.add(event.segment)
        elif isinstance(event, SpeakerRenamedEvent):
            self._transcript.rename_speaker(event.speaker_id, event.new_label)
        self._bus.publish(event)

    def subscribe(self, maxsize: int = 0):  # noqa: ANN201 - mirrors EventBus
        return self._bus.subscribe(maxsize)

    def unsubscribe(self, q: Any) -> None:
        self._bus.unsubscribe(q)


class Engine(EngineActions):
    """Owns every component and exposes the EngineActions surface used by the UI/hotkeys.

    start() returns quickly; the heavy parts (model download, Whisper load, audio capture)
    run on an "engine-init" thread and report progress through ComponentStatusEvents.
    """

    def __init__(
        self, cfg: Config, bus: EventBus | None = None, *, on_quit: Callable[[], None] | None = None,
        route_resolver: Callable[..., Any] | None = None,
    ) -> None:
        self.cfg = cfg
        self.bus = bus or EventBus()
        # None = use the module-level `resolve_route` at call time (so `monkeypatch.setattr(app_mod,
        # "resolve_route", ...)` in tests is honoured even though this default is captured here);
        # an explicit callable (tests only) always wins.
        self._route_resolver = route_resolver
        self.transcript = Transcript()
        # 2026-10-06 (context library): the on-disk reference documents and the ONE object the
        # Assistant sizes every prompt with. Built here (not in start()) so the Models window's
        # Context tab works before the LLM is up and a document added early is already in the
        # manifest when the first answer is built. Constructing the store never touches the disk.
        self.context_store = ContextStore(cfg.context_dir())
        self.context = ContextManager(cfg, self.context_store, self.transcript)
        self._t0 = time.monotonic()
        self._next_id = 0
        self._id_lock = threading.Lock()
        self._on_quit = on_quit
        self._stopped = threading.Event()
        self._init_thread: threading.Thread | None = None
        self._llm_thread: threading.Thread | None = None
        self._llm_ready = threading.Event()
        self._pending: list[tuple[str, str, str]] = []   # (kind, text, label) requests made before the LLM was ready
        self._pending_lock = threading.Lock()
        # upstream finding 9 (Windows fix): the reason the LLM never came up, set by _drop_pending
        # (under _pending_lock) so a request made after that failure is reported, not queued
        # behind a "still starting" notice that is no longer true (see _defer).
        self._llm_failed: str | None = None
        self._paused = False                         # global pause (screen + automatic assist)
        self._completion_wanted = False              # completion mode requested before the LLM was ready
        self._gpu_thread: threading.Thread | None = None   # the gpu-guard sampler (see _gpu_guard_loop)

        # components (created in start())
        self.route: Any = None       # exocortex.llm.route.Route -- which backend start() picked, and why
        self.client: Any = None
        self.assistant: Any = None
        self.recorder: Any = None
        self.transcriber: Any = None
        self.clusterer: Any = None
        self.pipeline: Any = None
        self.server: Any = None          # LlamaServerManager when backend == "llamacpp"
        self.screen: Any = None          # ScreenReader
        self.completer: Any = None       # Completer (completion mode)
        self.handsfree: Any = None       # HandsFreeController (hands-free mode), created on first use
        self.project: Any = None         # exocortex.project.provider.ProjectContext (toolkit catalog, repo lookups)
        self.dashboard: Any = None       # exocortex.dashboard.service.DashboardService (the tracked board)
        self._models_cache: list[str] = []
        # 2026-09-27 (portable-install): the model manager's one-at-a-time Hub pull ("model-pull"
        # thread) and its cancel flag; stop() sets the flag and joins briefly so a quit mid-pull
        # leaves resumable .part files behind instead of a half-written shard.
        self._pull_thread: threading.Thread | None = None
        self._pull_cancel = threading.Event()
        # One model change at a time: set_model / unload_model run on this "llm-swap" thread and
        # refuse a second request while it is alive (two concurrent swap_model()/stop() calls on
        # the one LlamaServerManager would race on its process handle and orphan a llama-server).
        self._swap_thread: threading.Thread | None = None
        # Whether Assistant.start() has run: on a fresh machine the route resolves to "none" (no
        # model yet) and _init_llm never starts it, so the first load_model() has to (see
        # _mark_llm_ready).
        self._assistant_started = False

    # ---- clock / ids -----------------------------------------------------------------
    def clock(self) -> float:
        """Seconds since the engine (session) started."""
        return time.monotonic() - self._t0

    def next_id(self) -> int:
        with self._id_lock:
            self._next_id += 1
            return self._next_id

    # ---- lifecycle --------------------------------------------------------------------
    def start(self) -> None:
        cfg = self.cfg
        cuda_env.setup_cuda_dlls()
        self._preload_native()
        self._status("app", "loading", "starting")

        from exocortex.llm.assistant import Assistant
        from exocortex.session import SessionRecorder

        # Fast probes only (filesystem + connect-refused, ~1.5s worst case): keep start() quick.
        # The Ollama-autostart probe (up to 20s) only runs later, from _init_llm, off this thread.
        resolver = self._route_resolver or resolve_route
        self.route = resolver(cfg.llm, start_ollama=False)
        log.info("llm route: %s (%s)", self.route.backend, self.route.reason)
        self._status("llm", "loading", self.route.reason)
        # 2026-09-27 (portable-install): a fresh machine has a runtime but no model yet -- say so
        # through the bus so the overlay opens the Models window on its Discover tab, instead of
        # leaving a red status dot whose detail the user has to read.
        if self.route.backend in ("llamacpp", "none") and getattr(self.route, "model_count", 0) == 0 \
                and not self._library_has_models():
            self.bus.publish(ModelLibraryEvent(reason="empty-library"))

        self.client = make_client(cfg.llm, self.route.backend, base_url=self.route.base_url or None)
        self.assistant = Assistant(cfg, self.bus, self.client, self.transcript, self.clock, context=self.context)
        self.recorder = SessionRecorder(cfg, self.bus, self.transcript, self.assistant)
        folder = self.recorder.start()
        log.info("session folder: %s", folder)
        self._init_project()
        self._init_dashboard(folder)

        self._llm_thread = threading.Thread(target=self._init_llm, name="llm-init", daemon=True)
        self._llm_thread.start()
        self._init_thread = threading.Thread(target=self._init_speech, name="engine-init", daemon=True)
        self._init_thread.start()
        self._init_screen()
        self._start_gpu_guard()
        if cfg.completion.enabled:
            self.set_completion_mode(True)
        if cfg.handsfree.enabled:
            self.set_hands_free(True)
        if cfg.dashboard.enabled:
            self.bus.publish(DashboardModeEvent(enabled=True))   # the overlay opens on the board

    def _init_project(self) -> None:
        """The project context (2026-10-20): the Claude Code toolkit discovered in
        [project].repo_root and the repository lookups, handed to the assistant (answers, ask,
        recap) and to the dashboard service. Never blocks start(): a failure just leaves prompts
        without it."""
        try:
            from exocortex.project.provider import ProjectContext

            self.project = ProjectContext(self.cfg, self.clock)
            if self.assistant is not None and hasattr(self.assistant, "set_project_context"):
                self.assistant.set_project_context(self.project)
            detail = f"Project: {self.project.context_summary()}"
            log.info("%s", detail)
            if self.project.repo_root is not None:      # a repository was found: say which (a toast)
                self._status("project", "info", detail)
        except Exception as e:  # noqa: BLE001
            log.exception("project context failed to start")
            self.project = None
            self._status("app", "info", f"repository / toolkit lookups unavailable: {e}")

    def _init_dashboard(self, folder: Path | None) -> None:
        """The dashboard service (2026-10-20): the tracked board's store (in the session folder
        while [dashboard].persist is on) and its refresh timer. Its background refreshes replace
        the assistant's insights passes while [dashboard].track is on."""
        try:
            from exocortex.dashboard.service import DashboardService

            self.dashboard = DashboardService(
                self.cfg, self.bus, self.transcript, self.assistant, self.clock, project=self.project,
                screen_provider=lambda: self.screen.latest_text if self.screen is not None else "",
                folder=folder if self.cfg.dashboard.persist else None,
            )
            self.dashboard.set_paused(self._paused)
            self.dashboard.start()
        except Exception as e:  # noqa: BLE001
            log.exception("dashboard service failed to start")
            self.dashboard = None
            self._status("dashboard", "error", f"dashboard unavailable: {e}")

    def _start_gpu_guard(self) -> None:
        """The GPU pressure guard (exocortex.gpu): turns a sustained free-VRAM collapse -- the
        signature of a second model landing on the display GPU mid-meeting -- into an immediate
        toast instead of a post-mortem log find. Cheap, and never blocks start()."""
        if self.cfg.llm.vram_guard:
            self._gpu_thread = threading.Thread(target=self._gpu_guard_loop, name="gpu-guard", daemon=True)
            self._gpu_thread.start()

    def _gpu_guard_loop(self) -> None:
        from exocortex import gpu

        if gpu.vram_free_mb() is None:
            log.info("gpu-guard: no VRAM readings on this machine; guard off")
            return
        guard = gpu.VramGuard(gpu.vram_free_mb, self.cfg.llm.vram_guard_free_mb)
        log.info("gpu-guard: watching free VRAM (alert under %d MiB)", self.cfg.llm.vram_guard_free_mb)
        while not self._stopped.wait(GPU_GUARD_INTERVAL_S):
            message = guard.sample()
            if message is None:
                continue
            recovered = message.startswith("GPU memory recovered")
            (log.info if recovered else log.warning)("gpu-guard: %s", message)
            # "app" + "error" toasts in the overlay without touching the readiness dot.
            self._status("app", "info" if recovered else "error", message)

    def _init_screen(self) -> None:
        """Screen reading (capture + OCR) feeds answers/suggestions; failures never block the rest.

        The OCR engine factory binds ``[screen].ocr_device`` and the OpenVINO kernel cache
        (``<models_dir>/ov_cache``) for RapidOCR. With ``[handsfree].accurate_ocr`` on and a
        sampling engine other than RapidOCR, RapidOCR is registered as the accurate engine
        hands-free re-reads use (built first -- see the order rule in ``exocortex.screen.ocr``).
        """
        try:
            from exocortex.screen.reader import ScreenReader

            cfg = self.cfg
            cache_dir = cfg.models_dir() / "ov_cache"

            def _engine_factory(name: str) -> Any:
                from exocortex.screen.ocr import create_engine

                return create_engine(name, device=cfg.screen.ocr_device, cache_dir=cache_dir)

            self.screen = ScreenReader(cfg.screen, self.bus, self.clock, engine_factory=_engine_factory)
            if cfg.handsfree.accurate_ocr and cfg.screen.ocr_engine != "rapidocr" and hasattr(self.screen, "set_accurate_engine"):
                self.screen.set_accurate_engine("rapidocr")
            if self.recorder is not None and cfg.screen.save_text:
                self.screen.set_snapshot_dir(self.recorder.folder)
            if self.assistant is not None and hasattr(self.assistant, "set_screen_provider"):
                self.assistant.set_screen_provider(lambda: self.screen.latest_text if self.screen else "")
            self.screen.start()
        except Exception as e:  # noqa: BLE001
            log.exception("screen reader failed to start")
            self.screen = None
            self._status("app", "info", f"screen reading unavailable: {e}")

    def _init_llm(self) -> None:
        """Bring up the LLM backend routed by `start()` (`self.route.backend`), then start the
        assistant (which runs its own health check)."""
        cfg = self.cfg
        backend = self.route.backend if self.route is not None else cfg.llm.backend.lower()
        if backend == "llamacpp":
            try:
                from exocortex.llm.server import LlamaServerManager

                self._status("llm", "loading", f"starting llama-server for {cfg.llm.model}")
                log_dir = self.recorder.folder if self.recorder else None
                self.server = LlamaServerManager(cfg.llm, log_dir=log_dir)
                url = self.server.start(on_status=lambda d: self._status("llm", "loading", d))
                self.client.set_base_url(url)
                log.info("llama-server at %s (%s)", url, self.server.model.key if self.server.model else "?")
            except Exception as e:  # noqa: BLE001
                log.exception("llama-server failed to start")
                self._status("llm", "error", f"llama-server failed: {e}")
                # upstream finding 9 (ported from mac-cortex): a request _defer()red before this
                # failure surfaced would otherwise sit in self._pending forever (the only drain is
                # on the success tail below) -- report it as lost instead of dropping it silently.
                self._drop_pending("llama-server failed to start")
                return
        elif backend == "ollama":
            # start()'s route only ran the fast, no-autostart probe; if Ollama is not answering
            # now, this is where the up-to-20s autostart wait happens (off the main thread).
            health = self.client.health()
            if not health.reachable:
                self._status("llm", "loading", "Ollama not reachable yet; trying to start it...")
                retried = resolve_route(cfg.llm, start_ollama=True)
                self.route = retried
                log.info("llm route (retry): %s (%s)", retried.backend, retried.reason)
                self._status("llm", "loading", retried.reason)
                if retried.backend == "none":
                    self._status("llm", "error", retried.reason)
                    # upstream finding 9 (ported from mac-cortex): same lost-request risk as the
                    # llamacpp branch above -- nothing else will ever drain self._pending now.
                    self._drop_pending(retried.reason)
                    return
                if retried.backend != "ollama":
                    # Fell back to openai (or, in the unlikely case the same static filesystem
                    # probe now finds a llamacpp pack, that too): a plain client is enough either
                    # way -- the openai path needs nothing else, and a genuine llamacpp result
                    # here would mean re-running start()'s own probe, which the doctor/log already
                    # surfaces as worth investigating rather than silently spawning a server here.
                    old_client = self.client
                    self.client = make_client(cfg.llm, retried.backend, base_url=retried.base_url or None)
                    # self.assistant was already constructed in start() around the old (now-dead)
                    # Ollama client; without this, Assistant would keep talking to it forever (its
                    # own health check would fail even though a working client sits right here).
                    # Assistant has no public setter for this (out of scope for this change), so
                    # this rebinds its internal reference directly -- the same effect the llamacpp
                    # branch above gets "for free" by mutating the existing client in place via
                    # `set_base_url` instead of swapping the object.
                    if self.assistant is not None:
                        self.assistant._client = self.client
                    if old_client is not None and old_client is not self.client:
                        try:
                            old_client.close()
                        except Exception:  # noqa: BLE001
                            pass
        elif backend == "none":
            reason = self.route.reason if self.route is not None else "no LLM backend available"
            self._status("llm", "error", reason)
            # upstream finding 9 (ported from mac-cortex): same lost-request risk as above.
            self._drop_pending(reason)
            return
        # backend == "openai" (or "llamacpp"/"ollama" already handled above): nothing else to do,
        # the client already points at the configured/started server.
        if self._stopped.is_set():
            return
        try:
            self.assistant.start()
        except Exception as e:  # noqa: BLE001
            log.exception("assistant failed to start")
            self._status("llm", "error", f"assistant failed: {e}")
            # upstream finding 9 (ported from mac-cortex): same lost-request risk as above.
            self._drop_pending("assistant failed to start")
            return
        self._assistant_started = True
        self._llm_ready.set()
        self.bus.publish(ModelLibraryEvent(reason="loaded", loaded_key=self.loaded_model()))
        self._replay_pending()

    def _replay_pending(self) -> None:
        """Run every request `_defer` queued while the LLM was starting (or reloading). Called
        from `_init_llm`'s success tail and from a late first load (`_mark_llm_ready`)."""
        with self._pending_lock:
            pending, self._pending = self._pending, []
        for kind, text, label in pending:
            if kind == "ask":
                self.assistant.ask(text)
            elif kind == "answer":
                self.assistant.answer_last()
            elif kind == "notes":
                self.assistant.generate_notes(force=True)
            elif kind == "suggest":
                self.assistant.suggest_now()
            elif kind == "solve":
                self.assistant.solve(text)
            elif kind == "code":
                self.assistant.code(text, label=label) if label else self.assistant.code(text)
            elif kind == "text":
                self.assistant.text(text, label=label) if label else self.assistant.text(text)
            elif kind == "explain":
                self.assistant.explain_now()
            elif kind == "insights":
                self.assistant.insights_now(tuple(s for s in text.split(",") if s) or None)
            elif kind == "recap":
                self.assistant.recap()
            elif kind == "dashboard":
                if self.dashboard is not None:
                    self.dashboard.refresh_now(tuple(s for s in text.split(",") if s))
        if self._completion_wanted and self.completer is None and not self._stopped.is_set():
            self._completion_wanted = False
            self.set_completion_mode(True)

    @staticmethod
    def _preload_native() -> None:
        """Load the heavy native pieces on THIS thread before the init threads start. Importing
        numpy (its OpenBLAS DLL) on the screen-reader thread while ``llm-init`` opened its first
        socket and ``engine-init`` waited on the import lock deadlocked the Windows loader lock
        at startup (a py-spy dump on 2026-09-23: ``create_module`` of numpy's multiarray,
        ``socket.__init__``, ``importlib`` ``acquire`` -- the process sat idle until killed). One
        import and one throw-away socket here make both a no-op later."""
        try:
            import numpy  # noqa: F401
        except Exception:  # noqa: BLE001 - reported by the components that need it
            pass
        try:
            import socket

            socket.socket().close()
        except Exception:  # noqa: BLE001
            pass

    def _defer(self, kind: str, text: str = "", label: str = "") -> bool:
        """Queue a request while the LLM is still starting; returns True if it was deferred.

        # upstream finding 9 (ported from mac-cortex): `_init_llm()` does `self._llm_ready.set()`
        # and then, under `self._pending_lock`, swaps/drains `self._pending` exactly once -- there
        # is no later drain. Without the re-check below, a caller that reads `is_set() == False`
        # here just before `_init_llm` finishes that set()+drain could append to `self._pending`
        # *after* the one-time drain already ran, and the request would sit in a list nobody ever
        # looks at again. Re-checking `is_set()` a second time, now under the same lock `_init_llm`
        # drains under, closes the window: if it is now set, `_init_llm`'s drain has either already
        # happened (in which case the assistant is fully up -- `assistant.start()` runs strictly
        # before `set()` -- so running the request immediately is safe) or is about to no-op
        # waiting on this very lock, so returning False here (meaning "not deferred") and letting
        # the caller's normal immediate path run is correct either way.
        """
        if self._llm_ready.is_set() or self.assistant is None:
            return False
        with self._pending_lock:
            if self._llm_ready.is_set():
                return False
            failed = self._llm_failed
            if failed is None:
                self._pending = [p for p in self._pending if p[0] != kind] + [(kind, text, label)]
        if failed is not None:
            # upstream finding 9 (Windows fix): _init_llm already failed for good (nothing ever
            # sets _llm_ready after that, and nothing would drain _pending again), so queuing
            # would park this request forever. Consume it with a clear notice instead. Read
            # under the same lock _drop_pending sets it under, so a request racing the failure
            # is either drained by _drop_pending or refused here -- never stranded.
            log.warning("%s request dropped, the LLM is unavailable: %s", kind, failed)
            self._status("llm", "error", f"{failed} - cannot run {kind}")
            return True
        self._status("llm", "info", "still starting - your request will run as soon as the model is ready")
        return True

    def _drop_pending(self, reason: str) -> None:
        """Drain `self._pending` and tell the user what was lost when `_init_llm` fails.

        # upstream finding 9 (ported from mac-cortex): `_init_llm`'s only drain of `self._pending`
        # lives on its success tail, right after `self._llm_ready.set()` (see `_defer`'s
        # docstring). If the LLM backend or the assistant itself never comes up, that drain never
        # runs and anything a user did before the failure surfaced (ask/answer_last/
        # generate_notes/suggest_now/solve/code/text/explain_now, all routed through `_defer`)
        # would otherwise sit in `self._pending` forever with no further drain and no
        # request-specific notice. Called from every failure branch of `_init_llm` so the dropped
        # request(s) get an explicit, named notice instead of silent loss.
        """
        with self._pending_lock:
            self._llm_failed = reason   # later requests are refused by _defer, not queued forever
            pending, self._pending = self._pending, []
        if not pending:
            return
        kinds = ", ".join(sorted({kind for kind, _text, _label in pending}))
        log.warning("dropping %d request(s) queued before the LLM was ready (%s): %s", len(pending), kinds, reason)
        self._status("llm", "error", f"{reason} - dropped queued request(s): {kinds}")

    def _init_speech(self) -> None:
        cfg = self.cfg
        try:
            from exocortex.models_dl import ensure_models

            self._status("asr", "loading", "checking models")
            models = ensure_models(cfg.models_dir(), progress=self._download_progress)
        except Exception as e:  # noqa: BLE001
            log.exception("model download failed")
            self._status("asr", "error", f"model download failed: {e}")
            return
        if self._stopped.is_set():
            return

        from exocortex.speech.asr import Transcriber

        self._status("asr", "loading", f"loading whisper {cfg.asr.model}")
        try:
            self.transcriber = Transcriber(cfg.asr, on_status=lambda st, d: self._status("asr", st, d))  # type: ignore[arg-type]
            self.transcriber.load()
        except Exception as e:  # noqa: BLE001
            log.exception("whisper load failed")
            self._status("asr", "error", f"whisper failed: {e}")
            return
        self._status("asr", "ready", f"{cfg.asr.model} ({self.transcriber.device_used} {self.transcriber.compute_type_used})")
        if self._stopped.is_set():
            return

        if cfg.diarization.enabled:
            try:
                from exocortex.speech.diarize import SpeakerClusterer

                model_path = models.get(cfg.diarization.model) or (cfg.models_dir() / cfg.diarization.model)
                self.clusterer = SpeakerClusterer(
                    str(model_path),
                    threshold=cfg.diarization.threshold,
                    enroll_min_s=cfg.diarization.enroll_min_s,
                    max_speakers=cfg.diarization.max_speakers,
                    num_threads=cfg.diarization.num_threads,
                )
                self._status("diarization", "ready", Path(str(model_path)).name)
            except Exception as e:  # noqa: BLE001
                log.exception("diarization init failed")
                self.clusterer = None
                self._status("diarization", "error", f"disabled: {e}")
        else:
            self._status("diarization", "info", "disabled")

        try:
            from exocortex.speech.pipeline import SpeechPipeline

            self.pipeline = SpeechPipeline(
                cfg, _RecordingBus(self.bus, self.transcript), self.transcriber, self.clusterer,  # type: ignore[arg-type]
                self.clock, self.next_id,
            )
            if cfg.audio.mic_start_muted:
                self.pipeline.set_mic_muted(True)
            self.pipeline.start()
        except Exception as e:  # noqa: BLE001
            log.exception("speech pipeline failed to start")
            self._status("audio", "error", f"audio failed: {e}")
            return
        self._status("app", "ready", "listening")

    def stop(self) -> None:
        if self._stopped.is_set():
            return
        self._stopped.set()
        log.info("engine stopping")
        self._pull_cancel.set()
        if self._pull_thread is not None and self._pull_thread.is_alive():
            self._pull_thread.join(timeout=2.0)    # polled per 64 KiB chunk: returns within a chunk
        for name, comp in (("hands-free", self.handsfree), ("completer", self.completer), ("screen", self.screen),
                           ("pipeline", self.pipeline), ("dashboard", self.dashboard), ("assistant", self.assistant),
                           ("recorder", self.recorder), ("llama-server", self.server)):
            if comp is None:
                continue
            try:
                comp.stop()
            except Exception:  # noqa: BLE001
                log.exception("error stopping %s", name)
        # review round two, 2026-09-27 (E2): the loop above snapshotted self.server; on a fresh
        # machine (self.server None at start) a first Load racing this stop() can create and
        # start a LlamaServerManager (in _ensure_server, from the "llm-swap" thread) after that
        # snapshot was taken, so it was never in the tuple above and its thread never joined --
        # a llama-server could outlive the app. Stop whatever self.server is NOW (stop() is
        # idempotent, so stopping one already stopped above is harmless), then wait for that
        # thread (the _stopped.is_set() checks in set_model's _swap close the window going
        # forward) and stop the server once more in case the thread built one during the join.
        if self.server is not None:
            try:
                self.server.stop()
            except Exception:  # noqa: BLE001
                log.exception("error stopping llama-server")
        if self._swap_thread is not None and self._swap_thread.is_alive():
            self._swap_thread.join(timeout=5.0)
        if self.server is not None:
            try:
                self.server.stop()
            except Exception:  # noqa: BLE001
                log.exception("error stopping llama-server")
        if self.client is not None:
            try:
                self.client.close()
            except Exception:  # noqa: BLE001
                pass
        if self._gpu_thread is not None:
            self._gpu_thread.join(timeout=2.0)   # it watches self._stopped, set above
        # Prefer the resolved route once start() ran: "auto" only ever means llamacpp-or-something-
        # else, and by now we know which. Before start() (or if it never got that far), fall back
        # to the config test -- "auto" counts as llamacpp-like here too, since unloading the
        # owner's llama-swap when the route later turns out to be something else is harmless and
        # matches the explicit-llamacpp behaviour (same as the --replace gate in main() below).
        llamacpp_like = (
            self.route.backend == "llamacpp" if self.route is not None
            else self.cfg.llm.backend.lower() in ("llamacpp", "auto")
        )
        if llamacpp_like and self.cfg.llm.swap_unload_url:
            # Quit means quit: our own llama-server died with the components above; also ask the
            # shared llama-swap to drop whatever it holds so the GPU is empty when we are gone
            # (10 s budget -- a quit must stay snappy even when that server is wedged).
            from exocortex.instance import free_llama_swap

            freed = free_llama_swap(self.cfg.llm.swap_unload_url, timeout_s=10.0)
            if freed:
                log.info("quit: %s", freed)
        self.save_config()

    def save_config(self) -> None:
        """Persist the shared Config (on exit, and whenever the settings dialog applies changes)."""
        try:
            path = self.cfg.path or (project_root() / "config.toml")
            save_config(self.cfg, path)
            log.info("config saved to %s", path)
        except Exception:  # noqa: BLE001
            log.exception("could not save config")

    # ---- helpers --------------------------------------------------------------------
    def _status(self, component: str, state: str, detail: str = "") -> None:
        self.bus.publish(ComponentStatusEvent(component, state, detail))  # type: ignore[arg-type]

    def _download_progress(self, name: str, done: int, total: int) -> None:
        pct = int(done * 100 / total) if total else 0
        self._status("asr", "loading", f"downloading {name} {pct}%")

    # ---- EngineActions ----------------------------------------------------------------
    def answer_last(self) -> None:
        if self.assistant and not self._defer("answer"):
            self.assistant.answer_last()

    def ask(self, text: str) -> None:
        if self.assistant and text.strip() and not self._defer("ask", text.strip()):
            self.assistant.ask(text.strip())

    def generate_notes(self) -> None:
        if self.assistant and not self._defer("notes"):
            self.assistant.generate_notes(force=True)

    def cancel_assist(self) -> None:
        if self.assistant:
            self.assistant.cancel()

    def set_auto(self, enabled: bool) -> None:
        self.cfg.assist.auto_answer_questions = enabled
        if self.assistant:
            self.assistant.set_auto(enabled)

    def set_mic_muted(self, muted: bool) -> None:
        self.cfg.audio.mic_start_muted = muted
        if self.pipeline:
            self.pipeline.set_mic_muted(muted)

    def rename_speaker(self, speaker_id: str, label: str) -> None:
        label = label.strip()
        if not label:
            return
        if self.pipeline:
            self.pipeline.rename_speaker(speaker_id, label)  # publishes SpeakerRenamedEvent via the recording bus
        else:
            self.transcript.rename_speaker(speaker_id, label)
            self.bus.publish(SpeakerRenamedEvent(speaker_id, label))

    def export_session(self) -> Path:
        if self.recorder:
            return self.recorder.export()
        return self.cfg.sessions_dir()

    def new_session(self) -> None:
        if self.recorder:
            self.recorder.stop()
        self.transcript.clear()
        if self.clusterer is not None:
            self.clusterer.reset()
        if self.assistant is not None and hasattr(self.assistant, "reset"):
            self.assistant.reset()
        # 2026-10-06: the anchored transcript window and the running summary belong to the old
        # meeting; the uploaded documents persist across sessions (Assistant.reset does this too
        # -- the engine repeats it so a headless / not-yet-started assistant cannot skip it).
        try:
            self.context.reset_session()
        except Exception:  # noqa: BLE001
            log.exception("context window reset failed")
        if self.completer is not None:
            self.completer.reset()
        if self.handsfree is not None:
            self.handsfree.reset()
        self._t0 = time.monotonic()
        if self.recorder:
            self.recorder.start()
        if self.dashboard is not None:
            folder = None
            if self.recorder is not None and self.cfg.dashboard.persist:
                try:
                    folder = self.recorder.folder
                except Exception:  # noqa: BLE001 - a recorder that did not start: keep the board in memory
                    folder = None
            self.dashboard.reset(folder)
        self._status("session", "info", "new session")
        if self.completer is not None and self.cfg.explain.enabled and self._llm_ready.is_set()                 and hasattr(self.assistant, "ensure_plan"):
            self.assistant.ensure_plan()   # completion mode follows the plan: rebuild it for what is on screen

    # ---- screen reading ----
    def set_screen_mode(self, mode: str) -> None:
        self.cfg.screen.mode = mode
        if self.screen is not None:
            self.screen.set_mode(mode)

    def set_screen_region(self, x: int, y: int, w: int, h: int) -> None:
        self.cfg.screen.region = [int(x), int(y), int(w), int(h)]
        self.cfg.screen.mode = "region"
        if self.screen is not None:
            self.screen.set_region(int(x), int(y), int(w), int(h))

    def capture_screen_now(self) -> None:
        if self.screen is not None:
            self.screen.capture_now()

    def suggest_now(self) -> None:
        if self.assistant is not None and hasattr(self.assistant, "suggest_now") and not self._defer("suggest"):
            self.assistant.suggest_now()

    def solve(self, focus: str = "") -> None:
        if self.assistant is not None and hasattr(self.assistant, "solve") and not self._defer("solve", focus):
            self.assistant.solve(focus)

    # ---- checkpoints / code ----
    def checkpoint(self) -> None:
        """Checkpoint the coding context (`EngineActions.checkpoint`). Never deferred: the
        Assistant object exists from `start()` on, even while the LLM is still loading -- a
        checkpoint just records a boundary, it makes no model call."""
        if self.assistant is not None and hasattr(self.assistant, "checkpoint"):
            self.assistant.checkpoint()
        if self.completer is not None:
            self.completer.reset()

    def compact(self) -> int:
        """Compact the active context window now (`EngineActions.compact`). Never deferred: it is
        window bookkeeping plus a notes-lane summary job, no blocking model call. Returns the
        transcript characters folded (0 when there was nothing to fold / compaction is off)."""
        if self.assistant is not None and hasattr(self.assistant, "compact"):
            try:
                return int(self.assistant.compact())
            except Exception:  # noqa: BLE001 - a user action must never raise into the UI
                log.exception("engine: compact() failed")
        return 0

    def code(self, focus: str = "", label: str = "") -> int | None:
        """Returns the job id, or None when the request was deferred (LLM not ready yet).
        `label`: what the Assist tab shows as the question instead of `focus` (hands-free)."""
        if self.assistant is not None and hasattr(self.assistant, "code") and not self._defer("code", focus, label):
            if label:
                return self.assistant.code(focus, label=label)
            return self.assistant.code(focus)
        return None

    def text(self, focus: str = "", label: str = "") -> int | None:
        """Returns the job id, or None when the request was deferred (LLM not ready yet)."""
        if self.assistant is not None and hasattr(self.assistant, "text") and not self._defer("text", focus, label):
            if label:
                return self.assistant.text(focus, label=label)
            return self.assistant.text(focus)
        return None

    # ---- co-pilot insights / recap (pair programming, 2026-10-20) ----
    def insights_now(self, sections: tuple[str, ...] = ()) -> None:
        """`EngineActions.insights_now`; deferred (latest request wins) while the LLM starts.
        While the dashboard tracks the session ([dashboard].track) it refreshes the dashboard
        instead -- the same questions and blind spots, mirrored into the Insights pane."""
        secs = tuple(str(s) for s in (sections or ()) if str(s).strip())
        if self.dashboard is not None and self.cfg.dashboard.track:
            self.dashboard_refresh(secs)
            return
        if self.assistant is not None and hasattr(self.assistant, "insights_now") \
                and not self._defer("insights", ",".join(secs)):
            self.assistant.insights_now(secs or None)

    def recap(self) -> None:
        if self.assistant is not None and hasattr(self.assistant, "recap") and not self._defer("recap"):
            self.assistant.recap()

    def set_insight_options(self, *, questions: bool | None = None, blind_spots: bool | None = None) -> None:
        """Read live by the assistant's insight-timer on its next tick: nothing else to restart."""
        if questions is not None:
            self.cfg.assist.suggest_questions = bool(questions)
        if blind_spots is not None:
            self.cfg.assist.flag_blind_spots = bool(blind_spots)
        log.info("insights: questions %s, blind spots %s",
                 "on" if self.cfg.assist.suggest_questions else "off",
                 "on" if self.cfg.assist.flag_blind_spots else "off")

    def set_role(self, role: str) -> None:
        from exocortex.llm.prompts import normalize_role

        self.cfg.assist.role = normalize_role(role)
        log.info("role: %s", self.cfg.assist.role)

    def set_session_context(self, text: str) -> None:
        self.cfg.assist.session_context = " ".join(str(text or "").split())
        log.info("session context %s", "set" if self.cfg.assist.session_context else "cleared")

    # ---- dashboard mode (2026-10-20) ----
    def set_dashboard_mode(self, enabled: bool) -> None:
        """`EngineActions.set_dashboard_mode`: the view's switch ([dashboard].enabled, persisted
        with the config); the overlay follows the event. The tracking runs regardless."""
        self.cfg.dashboard.enabled = bool(enabled)
        self.bus.publish(DashboardModeEvent(enabled=bool(enabled)))
        log.info("dashboard mode %s", "on" if enabled else "off")

    def dashboard_refresh(self, sections: tuple[str, ...] = ()) -> None:
        """`EngineActions.dashboard_refresh`; deferred (latest request wins) while the LLM starts."""
        secs = tuple(str(s) for s in (sections or ()) if str(s).strip())
        if self.dashboard is None:
            self._status("dashboard", "error", "dashboard unavailable")
            return
        if self._defer("dashboard", ",".join(secs)):
            return
        self.dashboard.refresh_now(secs)

    def dashboard_item(self, item_id: int, action: str, text: str = "") -> None:
        if self.dashboard is not None:
            self.dashboard.item_action(int(item_id), str(action), str(text or ""))

    def dashboard_add(self, kind: str, text: str, parent_id: int | None = None) -> None:
        if self.dashboard is not None:
            self.dashboard.add_item(str(kind), str(text), int(parent_id) if parent_id is not None else None)

    def set_auto_suggest(self, enabled: bool) -> None:
        self.cfg.screen.auto_suggest = enabled

    def set_paused(self, paused: bool) -> None:
        """Pause/resume everything automatic: screen reader + the assistant's automatic lanes."""
        self._paused = paused
        if self.screen is not None:
            if paused:
                self.screen.pause()
            elif self.cfg.screen.mode != "off":
                self.screen.resume()
        if self.assistant is not None and hasattr(self.assistant, "set_paused"):
            self.assistant.set_paused(paused)
        if self.completer is not None:
            self.completer.set_paused(paused)
        if self.handsfree is not None:
            self.handsfree.set_paused(paused)
        if self.dashboard is not None:
            self.dashboard.set_paused(paused)
        self._status("app", "info", "paused" if paused else "listening")

    # ---- hands-free mode ----
    def set_hands_free(self, enabled: bool) -> None:
        """Hands-free on/off (`EngineActions.set_hands_free`). The controller is created on first
        use; it needs only the screen reader and this engine's checkpoint/text/code (which are
        deferred and replayed in order while the LLM is still starting), so it can run from
        `start()` on. The assistant's automatic solve / screen-question paths are gated off while
        it is on. `cfg.handsfree.enabled` mirrors the switch (persisted with the config)."""
        self.cfg.handsfree.enabled = enabled
        if enabled:
            if self.handsfree is None:
                try:
                    from exocortex.handsfree import HandsFreeController

                    self.handsfree = HandsFreeController(self.cfg, self.bus, _HandsFreeHooks(self), self.clock)
                except Exception as e:  # noqa: BLE001
                    log.exception("hands-free controller failed to start")
                    self._status("app", "error", f"hands-free failed: {e}")
                    return
            if self.assistant is not None and hasattr(self.assistant, "set_hands_free"):
                self.assistant.set_hands_free(True, self.handsfree.context_text)
            self.handsfree.set_paused(self._paused)
            self.handsfree.start()
        else:
            if self.handsfree is not None:
                self.handsfree.stop()
            if self.assistant is not None and hasattr(self.assistant, "set_hands_free"):
                self.assistant.set_hands_free(False)
        log.info("hands-free mode %s", "on" if enabled else "off")

    # ---- manual advance ----
    def set_manual_advance(self, enabled: bool) -> None:
        """Manual advance on/off (`EngineActions.set_manual_advance`). Mostly just the config
        flag (`cfg.handsfree.manual`, persisted with the config): the assistant's automatic lanes
        and the hands-free controller read it live on every trigger. Switching back to Auto also
        pokes the controller so a task armed and still waiting for the gesture runs now, the way
        Auto would have run it. The event keeps the UI switch, the slim bar's menu entry and the
        settings dialog in sync."""
        self.cfg.handsfree.manual = enabled
        if not enabled and self.handsfree is not None and getattr(self.handsfree, "enabled", False):
            try:
                self.handsfree.resume_auto()
            except Exception:  # noqa: BLE001
                log.exception("hands-free resume_auto failed")
        self.bus.publish(ManualAdvanceEvent(enabled=enabled))
        log.info("manual advance %s", "on" if enabled else "off")

    def manual_advance(self, kind: str = "text") -> None:
        """The press-and-hold gesture (`EngineActions.manual_advance`): run one generation now.
        With hands-free on (and not paused) the controller runs the step for the current task --
        the code-task ask, the armed state, the retry machinery; otherwise the ordinary manual
        text()/code(), which still gets the hands-free cumulative context while the controller
        runs (see `Assistant.set_hands_free`). Works while paused, like every manual action."""
        kind = "code" if kind == "code" else "text"
        hf = self.handsfree
        if hf is not None and getattr(hf, "enabled", False) and not self._paused:
            hf.trigger_step(kind)
        elif kind == "code":
            self.code()
        else:
            self.text()

    # ---- completion mode ----
    def set_completion_mode(self, enabled: bool) -> None:
        self.cfg.completion.enabled = enabled
        if enabled:
            if self.completer is None:
                if self.assistant is None:
                    self._status("completion", "error", "assistant not started")
                    return
                if not self._llm_ready.is_set():
                    # The assistant's worker is not running yet: a job queued now would be dropped
                    # by start(). Start the completer from _init_llm once the model is up.
                    self._completion_wanted = True
                    if hasattr(self.assistant, "set_auto_suppressed"):
                        self.assistant.set_auto_suppressed(True)
                    self.bus.publish(CompletionModeEvent(enabled=True, detail="waiting for the model"))
                    self._status("completion", "loading", "waiting for the model")
                    return
                try:
                    from exocortex.completion import Completer
                    from exocortex.screen.caret import CaretTracker

                    cc = self.cfg.completion
                    tracker = CaretTracker(prefix_chars=cc.prefix_chars, suffix_chars=cc.suffix_chars)
                    screen_source = (lambda: self.screen.latest_snapshot) if self.screen is not None else None
                    capture_now = (lambda: self.screen.capture_now()) if self.screen is not None else None
                    self.completer = Completer(self.cfg, self.bus, self.assistant, tracker, self.clock,
                                               screen_source=screen_source, capture_now=capture_now)
                except Exception as e:  # noqa: BLE001
                    log.exception("completion mode failed to start")
                    self._status("completion", "error", f"completion failed: {e}")
                    return
                self.completer.set_paused(self._paused)
                self.completer.start()
                if self.cfg.explain.enabled and hasattr(self.assistant, "ensure_plan"):
                    self.assistant.ensure_plan()   # the ghost text follows the plan: have one ready
        else:
            self._completion_wanted = False
            comp, self.completer = self.completer, None
            if comp is not None:
                comp.stop()
        if self.assistant is not None and hasattr(self.assistant, "set_auto_suppressed"):
            self.assistant.set_auto_suppressed(enabled)
        self.bus.publish(CompletionModeEvent(enabled=enabled))
        log.info("completion mode %s", "on" if enabled else "off")

    def set_completion_context(self, sources: dict[str, bool], hint: str | None = None) -> None:
        cc = self.cfg.completion
        for key, attr in (("transcript", "use_transcript"), ("screen", "use_screen"),
                          ("answers", "use_answers"), ("notes", "use_notes")):
            if key in sources:
                setattr(cc, attr, bool(sources[key]))
        # 2026-10-20: the same menu switches the on-demand lookups (exocortex.project) -- the
        # repository search and the toolkit catalog that answers and the dashboard get.
        if "repo" in sources:
            self.cfg.project.repo_search = bool(sources["repo"])
        if "toolkit" in sources:
            self.cfg.project.toolkit_in_answers = bool(sources["toolkit"])
        if hint is not None:
            cc.hint = hint.strip()
        if self.completer is not None:
            self.completer.reset()   # the context changed: forget the current suggestion

    # ---- explanations ----
    def set_explanations(self, enabled: bool) -> None:
        self.cfg.explain.enabled = enabled
        if self.assistant is not None and hasattr(self.assistant, "set_explanations"):
            self.assistant.set_explanations(enabled)
        if self.completer is not None:
            self.completer.reset()   # the ghost text's source changes (plan vs model): forget the current suggestion
        if enabled and self.assistant is not None and self._llm_ready.is_set() and hasattr(self.assistant, "ensure_plan"):
            self.assistant.ensure_plan()
        self._status("app", "info", "explanations on" if enabled else "explanations off")
        log.info("explanations %s", "on" if enabled else "off")

    def explain_now(self) -> None:
        if self.assistant is not None and hasattr(self.assistant, "explain_now") and not self._defer("explain"):
            self.assistant.explain_now()

    # ---- settings ----
    def apply_settings(self, changed: dict[str, object]) -> None:
        """The settings dialog wrote `changed` into cfg already; most components read it live.
        The rest is applied here, then the file is saved so nothing is lost on a crash."""
        if "llm.model" in changed:
            self.set_model(str(changed["llm.model"]))
        if "assist.auto_answer_questions" in changed and self.assistant is not None:
            self.assistant.set_auto(bool(changed["assist.auto_answer_questions"]))
        if "explain.enabled" in changed:
            self.set_explanations(bool(changed["explain.enabled"]))
        if "audio.mic_start_muted" in changed:
            self.set_mic_muted(bool(changed["audio.mic_start_muted"]))
        for key, source in (("audio.mic_device", Source.MIC), ("audio.system_device", Source.SYSTEM)):
            if key in changed and self.pipeline is not None:
                # reopening a WASAPI stream takes a moment: never on the UI thread
                threading.Thread(target=self.pipeline.restart_capture, args=(source,),
                                 name=f"capture-{source.value}-switch", daemon=True).start()
        if any(k.startswith("completion.") for k in changed) and self.completer is not None:
            self.completer.reset()          # lines / hint / timing changed: forget the current suggestion
        if "screen.interval_s" in changed and self.screen is not None:
            try:
                self.screen.capture_now()   # wakes the capture loop so the new cadence starts now
            except Exception:  # noqa: BLE001
                log.debug("capture_now after settings change failed", exc_info=True)
        if "screen.monitor" in changed and self.screen is not None:
            try:
                self.screen.set_monitor(int(changed["screen.monitor"]))   # the View menu's monitor choice
            except Exception:  # noqa: BLE001
                log.exception("set_monitor after settings change failed")
        if "handsfree.enabled" in changed:
            self.set_hands_free(bool(changed["handsfree.enabled"]))
        if "handsfree.manual" in changed:
            self.set_manual_advance(bool(changed["handsfree.manual"]))   # publishes the sync event
        # every other [handsfree] value is read live by the controller on its next evaluation
        if any(k.startswith("project.") for k in changed) and self.project is not None:
            try:
                self.project.reconfigure()      # a new repo root / toolkit dir: re-index, rescan now
                self._status("project", "info", f"Project: {self.project.context_summary()}")
            except Exception:  # noqa: BLE001
                log.exception("project reconfigure failed")
        if "dashboard.enabled" in changed:
            self.set_dashboard_mode(bool(changed["dashboard.enabled"]))   # publishes the sync event
        # every other [dashboard] value is read live by the service on its next tick / refresh
        if "context.dir" in changed:
            # 2026-10-06: the library's manifest is opened once, at construction -- a new folder
            # takes effect at the next launch (the Context tab marks the field restart=True).
            log.info("[context].dir changed to %r; the document library moves at the next launch",
                     changed["context.dir"])
        # every other context.* value is read live by the ContextManager on its next build()
        self.save_config()
        log.info("settings applied: %s", ", ".join(sorted(changed)))

    def toggle_screen_paused(self) -> bool:
        """Pause/resume screen reading; returns True when now paused."""
        if self.screen is None:
            return False
        if self.screen.paused:
            self.screen.resume()
            return False
        self.screen.pause()
        return True

    def set_model(self, model_id: str) -> None:
        if self.server is not None or self._can_spawn_server():
            def _swap() -> None:
                try:
                    # review round two, 2026-09-27 (E2): _start_swap only starts this thread when
                    # no other swap is running, but stop() can still race the start -- do nothing
                    # that creates or starts a server once self._stopped is set. Checked again
                    # below, right before _ensure_server and again right before swap_model.
                    if self._stopped.is_set():
                        return
                    self._status("llm", "loading", f"switching to {model_id}")
                    if self.assistant is not None and hasattr(self.assistant, "cancel"):
                        self.assistant.cancel()
                    # review round two, 2026-09-27 (E1): commit to cfg only now that this attempt
                    # was actually accepted (the swap thread is running) -- a request _start_swap
                    # refused as "still busy" must never change the default stop()'s save_config()
                    # persists.
                    self.cfg.llm.model = model_id
                    if self._stopped.is_set():
                        return
                    server = self._ensure_server()
                    if self._stopped.is_set():
                        return
                    url = server.swap_model(model_id, on_status=lambda d: self._status("llm", "loading", d))
                    self.client.set_base_url(url)
                    if self.assistant and hasattr(self.assistant, "set_model"):
                        self.assistant.set_model(model_id)
                    else:
                        self._status("llm", "ready", model_id)
                    self._mark_llm_ready()
                except Exception as e:  # noqa: BLE001
                    log.exception("model swap failed")
                    self._status("llm", "error", f"model swap failed: {e}")

            self._start_swap(_swap, f"loading {model_id}")
            return
        # ollama / openai (remote server, only the name changes) and pre-start (no route yet):
        # neither creates a server, so there is nothing to accept/refuse -- keep setting cfg here.
        self.cfg.llm.model = model_id
        if self.assistant and hasattr(self.assistant, "set_model"):
            self.assistant.set_model(model_id)
            self.bus.publish(ModelLibraryEvent(reason="loaded", loaded_key=model_id))

    # ---- model changes: one at a time, and a first load on a machine that started with none ----
    def _start_swap(self, work: Callable[[], None], what: str) -> bool:
        """Run `work` on the single "llm-swap" thread; refused (with a notice) while a previous
        model change is still running -- two concurrent swap_model()/stop() calls on the one
        LlamaServerManager would race on its process handle and orphan a llama-server."""
        current = self._swap_thread
        if current is not None and current.is_alive():
            # review round two, 2026-09-27 (E3): the stable "busy: " prefix lets a consumer that
            # tracks llm state (vs. just showing a toast) recognise and ignore this refusal --
            # nothing about the served model changed -- while the words "still busy" stay in the
            # text for anything (existing tests included) that still matches on them.
            self._status("llm", "info", f"busy: still busy with the previous model change; {what} skipped -- "
                                        "try again when it finishes")
            return False
        self._swap_thread = threading.Thread(target=work, name="llm-swap", daemon=True)
        self._swap_thread.start()
        return True

    def _can_spawn_server(self) -> bool:
        """True when this engine is the one that would run llama-server: start() resolved the
        route and no external backend was chosen (route "llamacpp", or "none" because the library
        was empty at start). Before start() (no route yet) nothing is spawned."""
        return self.route is not None and self.route.backend in ("llamacpp", "none")

    def _ensure_server(self) -> Any:
        """The LlamaServerManager, created on first use when start() found no model to serve
        (a fresh machine: the first pull + Load must not need a restart). Only called from the
        swap thread (see _start_swap), so the lazy creation cannot race."""
        if self.server is None:
            from exocortex.llm.server import LlamaServerManager

            log_dir = self.recorder.folder if self.recorder else None
            self.server = LlamaServerManager(self.cfg.llm, log_dir=log_dir)
            log.info("llama-server manager created on first load (the app started without a model)")
        return self.server

    def _mark_llm_ready(self) -> None:
        """A model is being served again: (re)start the assistant if it never ran, lift the
        "LLM unavailable" refusal `_drop_pending` may have set, and replay queued requests."""
        if self.assistant is not None and not self._assistant_started and hasattr(self.assistant, "start"):
            self.assistant.start()
            self._assistant_started = True
        with self._pending_lock:
            self._llm_failed = None
        self._llm_ready.set()
        self.bus.publish(ModelLibraryEvent(reason="loaded", loaded_key=self.loaded_model()))
        self._replay_pending()

    # ---- model manager (2026-09-27, portable-install: the "Models..." window) ------------
    # Same shape as set_model: return at once, do the work on a named daemon thread, report through
    # ModelLibraryEvent / ModelPullEvent / ModelSearchEvent / ModelInfoEvent / ComponentStatusEvent.
    # Everything that touches the library goes through exocortex.llm.library, imported lazily so
    # the engine still constructs (and every existing test still runs) if that module is absent.
    def _library_has_models(self) -> bool:
        try:
            from exocortex.llm import library

            return bool(library.scan_library(self.cfg.llm))
        except Exception:  # noqa: BLE001
            log.debug("library scan failed", exc_info=True)
            return False

    def model_library(self) -> list[dict[str, object]]:
        """Own library + LM Studio's folder, plus Ollama's list when that backend is active."""
        out: list[dict[str, object]] = []
        loaded = self.loaded_model()
        wanted = (self.cfg.llm.model or "").strip().lower()
        try:
            from exocortex.llm import library

            profiles = library.load_profiles(self.cfg.llm)
            for m in library.scan_library(self.cfg.llm, with_meta=True):
                ctx = m.meta.n_ctx_train if m.meta is not None else None
                out.append({
                    "key": m.key, "name": m.name, "size_bytes": m.size_bytes, "source": m.source,
                    "quant": m.quant, "path": str(m.path), "context_length": ctx,
                    "loaded": m.key == loaded, "is_default": wanted in (m.key.lower(), m.name.lower()),
                    "has_profile": m.key in profiles,
                })
        except Exception:  # noqa: BLE001
            log.exception("model_library: library scan failed")
        if self.route is not None and self.route.backend == "ollama" and self.client is not None:
            try:
                sizes = getattr(self.client, "model_sizes", {})
                for info in self.client.list_models():
                    out.append({"key": info.id, "name": info.id, "size_bytes": int(sizes.get(info.id, 0)),
                                "source": "ollama", "quant": info.quant or "", "path": "",
                                "context_length": info.context_length, "loaded": bool(info.loaded),
                                "is_default": wanted == info.id.lower(), "has_profile": False})
            except Exception:  # noqa: BLE001
                log.debug("model_library: ollama list failed", exc_info=True)
        self._models_cache = [str(d["key"]) for d in out]
        return out

    def loaded_model(self) -> str:
        if self.server is not None:
            model = getattr(self.server, "model", None)
            return str(model.key) if getattr(self.server, "running", False) and model is not None else ""
        if self.route is not None and self.route.backend in ("ollama", "openai") and self._llm_ready.is_set():
            return str(getattr(self.client, "resolved_model", None) or self.cfg.llm.model or "")
        return ""

    def load_model(self, key: str) -> None:
        self.set_model(key)

    def unload_model(self) -> None:
        if self.server is not None:
            def _unload() -> None:
                try:
                    # review round two, 2026-09-27 (E3): server.stop() can take ~10s; publish a
                    # "loading" status as soon as this thread actually runs (i.e. _start_swap
                    # accepted it) so the Models window knows an unload is in flight instead of
                    # only finding out once it is already done.
                    key = self.loaded_model()
                    self._status("llm", "loading", f"unloading {key}" if key else "unloading")
                    if self.assistant:
                        self.assistant.cancel()
                    self.server.stop()
                    self._llm_ready.clear()
                    self._status("llm", "info", "model unloaded -- pick one in Models to answer again")
                    self.bus.publish(ModelLibraryEvent(reason="unloaded", loaded_key=""))
                except Exception as e:  # noqa: BLE001
                    log.exception("unload failed")
                    self._status("llm", "error", f"unload failed: {e}")

            self._start_swap(_unload, "unloading")
            return
        if self.route is not None and self.route.backend == "ollama" and self.client is not None \
                and hasattr(self.client, "unload"):
            def _unload_ollama() -> None:
                try:
                    self.client.unload()
                    self._status("llm", "info", "model unloaded")
                    self.bus.publish(ModelLibraryEvent(reason="unloaded", loaded_key=""))
                except Exception as e:  # noqa: BLE001
                    self._status("llm", "error", f"unload failed: {e}")

            threading.Thread(target=_unload_ollama, name="llm-swap", daemon=True).start()

    def set_default_model(self, key: str) -> None:
        self.cfg.llm.model = key
        self.save_config()
        self.bus.publish(ModelLibraryEvent(reason=f"default {key}", loaded_key=self.loaded_model()))

    def delete_model(self, key: str) -> None:
        if key and key == self.loaded_model():
            self._status("llm", "error", f"{key} is loaded -- unload it before deleting")
            return
        try:
            from exocortex.llm import library

            matches = [m for m in library.scan_library(self.cfg.llm) if m.key == key]
            if not matches:
                raise FileNotFoundError(f"{key} is not in the library")
            removed = library.delete_model(matches[0])
            library.set_profile(self.cfg.llm, key, None)
            log.info("deleted %s (%d files)", key, len(removed))
            self.bus.publish(ModelLibraryEvent(reason=f"deleted {key}", loaded_key=self.loaded_model()))
        except PermissionError as e:
            self._status("llm", "error", str(e))
        except Exception as e:  # noqa: BLE001
            log.exception("delete failed")
            self._status("llm", "error", f"delete failed: {e}")

    def pull_model(self, repo_id: str, quant: str = "auto", *, include_mmproj: bool = False) -> None:
        if self._pull_thread is not None and self._pull_thread.is_alive():
            self.bus.publish(ModelPullEvent(repo_id=repo_id, state="error", detail="another pull is still running"))
            return
        self._pull_cancel.clear()
        self.bus.publish(ModelPullEvent(repo_id=repo_id, state="queued"))

        def _run() -> None:
            from exocortex import gpu
            from exocortex.llm import library

            def _progress(name: str, done: int, total: int) -> None:
                self.bus.publish(ModelPullEvent(repo_id=repo_id, state="downloading", file=name,
                                                done_bytes=done, total_bytes=total))

            try:
                budget = gpu.primary_budget_mb()
                model = library.pull_model(repo_id, quant, self.cfg.llm, include_mmproj=include_mmproj,
                                           budget_mb=budget, policy=self._recommended_policy(),
                                           progress=_progress, cancelled=self._pull_cancel.is_set)
            except library.PullCancelled as e:
                self.bus.publish(ModelPullEvent(repo_id=repo_id, state="cancelled", detail=str(e)))
                return
            except Exception as e:  # noqa: BLE001
                log.exception("pull failed")
                self.bus.publish(ModelPullEvent(repo_id=repo_id, state="error", detail=str(e)))
                return
            self.bus.publish(ModelPullEvent(repo_id=repo_id, state="done", model_key=model.key,
                                            done_bytes=model.size_bytes, total_bytes=model.size_bytes))
            self.bus.publish(ModelLibraryEvent(reason=f"pulled {model.key}", loaded_key=self.loaded_model()))
            wanted = (self.cfg.llm.model or "").strip().lower()
            if wanted in ("", "auto") or not self.loaded_model():
                # The first model on a fresh machine becomes the default without a second click.
                self.cfg.llm.model = model.key
                self.save_config()

        self._pull_thread = threading.Thread(target=_run, name="model-pull", daemon=True)
        self._pull_thread.start()

    def cancel_pull(self) -> None:
        self._pull_cancel.set()

    def search_models(self, query: str, request_id: int) -> None:
        def _run() -> None:
            from exocortex.llm import library

            try:
                found = library.hf_search(query, limit=25)
                results = tuple({"id": r.id, "downloads": r.downloads, "likes": r.likes,
                                 "modified": r.modified, "gated": r.gated} for r in found)
                self.bus.publish(ModelSearchEvent(request_id=request_id, query=query, results=results))
            except Exception as e:  # noqa: BLE001
                self.bus.publish(ModelSearchEvent(request_id=request_id, query=query, error=str(e)))

        threading.Thread(target=_run, name="model-search", daemon=True).start()

    def inspect_model(self, repo_id: str, request_id: int) -> None:
        def _run() -> None:
            from exocortex import gpu
            from exocortex.llm import library

            try:
                info = library.hf_repo_info(repo_id)
                budget = gpu.primary_budget_mb()
                chosen = library.choose_quant(info.quants, budget_mb=budget, policy=self._recommended_policy())
                limit = None if budget is None else 0.75 * budget * 2**20
                quants = tuple({"quant": q.quant, "size_bytes": q.size_bytes, "files": len(q.files),
                                "fits": limit is None or q.size_bytes <= limit} for q in info.quants)
                self.bus.publish(ModelInfoEvent(request_id=request_id, repo_id=repo_id, quants=quants,
                                                chosen=chosen.quant if chosen else "", architecture=info.architecture,
                                                params=info.params, context_length=info.context_length,
                                                mmproj=bool(info.mmproj)))
            except Exception as e:  # noqa: BLE001
                self.bus.publish(ModelInfoEvent(request_id=request_id, repo_id=repo_id, error=str(e)))

        threading.Thread(target=_run, name="model-search", daemon=True).start()

    def hardware_report(self, refresh: bool = False) -> dict[str, object]:
        from exocortex import hardware

        path = hardware.report_path(self.cfg.llm)
        report = None if refresh else hardware.load(path)
        if report is None:
            report = hardware.collect(self.cfg.llm, probe_runtime=True)
            try:
                hardware.save(report, path)
            except OSError:
                log.debug("hardware report not saved", exc_info=True)
        return hardware.to_dict(report)

    def _recommended_policy(self) -> str:
        try:
            return str(self.hardware_report().get("recommended_policy") or "auto")
        except Exception:  # noqa: BLE001
            return "auto"

    def model_profile(self, key: str) -> dict[str, object] | None:
        from exocortex.llm import library

        p = library.profile_for(self.cfg.llm, key)
        return None if p is None else dataclasses.asdict(p)

    def set_model_profile(self, key: str, values: dict[str, object] | None) -> None:
        from exocortex.llm import library

        profile = None
        if values is not None:
            known = library.ModelProfile.__dataclass_fields__
            coerced: dict[str, object] = {}
            for k, v in values.items():
                if k not in known:
                    continue
                cv = _coerce_profile_field(k, v)
                if cv is _PROFILE_DROP:
                    log.warning("set_model_profile(%s): %s=%r is not a number, dropped", key, k, v)
                    continue
                coerced[k] = cv
            profile = library.ModelProfile(**coerced)
            profile.auto = False
        library.set_profile(self.cfg.llm, key, profile)
        self.bus.publish(ModelLibraryEvent(reason=f"profile {key}", loaded_key=self.loaded_model()))
        if key and key == self.loaded_model():
            self.set_model(key)          # relaunch llama-server with the new flags

    def auto_tune_model(self, key: str) -> dict[str, object] | None:
        from exocortex import gpu
        from exocortex.llm import library

        matches = [m for m in library.scan_library(self.cfg.llm, with_meta=True) if m.key == key]
        if not matches:
            return None
        hw = self.hardware_report()
        runtime = hw.get("runtime") or {}
        framework = str(runtime.get("framework") or ("cpu" if hw.get("budget_mb") is None else "gpu"))
        profile = library.auto_profile(matches[0], budget_mb=gpu.primary_budget_mb(),
                                       ram_total_mb=hw.get("ram_total_mb"), physical_cores=hw.get("physical_cores"),
                                       gpu_framework=framework, base=self.cfg.llm)
        library.set_profile(self.cfg.llm, key, profile)
        self.bus.publish(ModelLibraryEvent(reason=f"auto-tuned {key}", loaded_key=self.loaded_model()))
        return dataclasses.asdict(profile)

    def list_models(self) -> list[str]:
        if self.server is not None:
            try:
                self._models_cache = [m.key for m in self.server.available_models()]
            except Exception:  # noqa: BLE001
                log.exception("scanning the model library failed")
            return list(self._models_cache)
        if self.assistant and hasattr(self.assistant, "list_models"):
            try:
                models = list(self.assistant.list_models())
                if models:
                    self._models_cache = models
            except Exception:  # noqa: BLE001
                log.exception("list_models failed")
        return list(self._models_cache)

    # ---- context library (2026-10-06: the Models window's "Context" tab) ---------------------
    def _context_digest_wanted(self) -> bool:
        return bool(self.cfg.context.digest)

    def _context_error(self, what: str, exc: BaseException) -> None:
        """A library operation failed: tell the Context tab (the event) and the toast (the status)
        -- never raise into the UI thread."""
        message = str(exc) or type(exc).__name__
        log.warning("context: %s failed: %s", what, message)
        self.bus.publish(ContextLibraryEvent(reason=f"error: {message}"))
        self._status("app", "error", f"context: {what} failed: {message}")

    def _queue_digest(self, doc_id: str) -> None:
        """Ask the Assistant (notes lane) to write the digest of `doc_id`. Before the assistant
        exists nothing is lost: its `start()` drains `store.pending_digests()` itself."""
        assistant = self.assistant
        if assistant is None or not hasattr(assistant, "digest_document"):
            return
        try:
            assistant.digest_document(doc_id)
        except Exception:  # noqa: BLE001
            log.exception("could not queue the digest of document %s", doc_id)

    def context_documents(self) -> list[dict[str, object]]:
        try:
            tokens_for_len = self.context.estimator.tokens_for_len
            rows: list[dict[str, object]] = []
            for doc in self.context_store.documents():
                row = doc.to_dict()
                row["tokens"] = tokens_for_len(doc.chars)
                rows.append(row)
            return rows
        except Exception:  # noqa: BLE001
            log.exception("could not list the context documents")
            return []

    def add_context_document(self, path: str, *, mode: str | None = None) -> None:
        name = str(path)
        try:
            # a path pasted from Explorer's "Copy as path" arrives quoted
            file = Path(str(path).strip().strip('"').strip("'")).expanduser()
            name = file.name or name
            c = self.cfg.context
            doc = self.context_store.add_file(
                file, mode=mode or c.default_mode, max_chars=c.max_doc_chars,
                digest_wanted=self._context_digest_wanted(),
            )
        except Exception as e:  # noqa: BLE001 - ContextError, OSError, a parser blowing up on a bad file
            self._context_error(f"adding {name}", e)
            return
        self.bus.publish(ContextLibraryEvent(reason="added", doc_id=doc.id))
        if doc.digest_state == "pending":
            self._queue_digest(doc.id)

    def add_context_text(self, name: str, text: str, *, mode: str | None = None) -> None:
        label = (name or "").strip() or "Untitled"
        try:
            c = self.cfg.context
            doc = self.context_store.add_text(
                label, text, mode=mode or c.default_mode, max_chars=c.max_doc_chars,
                digest_wanted=self._context_digest_wanted(),
            )
        except Exception as e:  # noqa: BLE001
            self._context_error(f"adding {label}", e)
            return
        self.bus.publish(ContextLibraryEvent(reason="added", doc_id=doc.id))
        if doc.digest_state == "pending":
            self._queue_digest(doc.id)

    def remove_context_document(self, doc_id: str) -> None:
        try:
            removed = self.context_store.remove(doc_id)
        except Exception as e:  # noqa: BLE001
            self._context_error("removing a document", e)
            return
        if not removed:
            self._context_error("removing a document", ValueError("no such document"))
            return
        self.bus.publish(ContextLibraryEvent(reason="removed", doc_id=doc_id))

    def set_context_document(self, doc_id: str, *, enabled: bool | None = None, mode: str | None = None) -> None:
        try:
            if self.context_store.get(doc_id) is None:
                raise ValueError("no such document")
            if mode is not None and mode not in ("inject", "reference"):
                raise ValueError(f"unknown mode {mode!r} (expected 'inject' or 'reference')")
            if enabled is not None:
                self.context_store.set_enabled(doc_id, bool(enabled))
            if mode is not None:
                self.context_store.set_mode(doc_id, mode, digest_wanted=self._context_digest_wanted())
            doc = self.context_store.get(doc_id)
        except Exception as e:  # noqa: BLE001
            self._context_error("updating a document", e)
            return
        self.bus.publish(ContextLibraryEvent(reason="updated", doc_id=doc_id))
        # Switching TO reference with no digest yet leaves the document "pending" (the catalogue
        # shows its first line meanwhile) and queues the model-written digest.
        if mode == "reference" and doc is not None and doc.digest_state == "pending":
            self._queue_digest(doc_id)

    def refresh_context_digest(self, doc_id: str) -> None:
        try:
            doc = self.context_store.get(doc_id)
            if doc is None:
                raise ValueError("no such document")
            if doc.mode != "reference":
                # An injected document goes into the prompt in full: it has no catalogue line.
                self._status("app", "info", f"context: {doc.name} is injected in full and has no digest")
                self.bus.publish(ContextLibraryEvent(reason="digest-skipped", doc_id=doc_id))
                return
            self.context_store.mark_digest_pending(doc_id)
        except Exception as e:  # noqa: BLE001
            self._context_error("refreshing a digest", e)
            return
        self.bus.publish(ContextLibraryEvent(reason="digest-queued", doc_id=doc_id))
        self._queue_digest(doc_id)

    def context_usage(self) -> dict[str, object]:
        try:
            return self.context.usage_snapshot()
        except Exception:  # noqa: BLE001
            log.exception("could not read the context usage")
            return {}

    def quit(self) -> None:
        if self._on_quit:
            self._on_quit()
        else:
            self.stop()


class _HandsFreeHooks:
    """`exocortex.handsfree.HandsFreeHooks` over an `Engine` (kept out of the Engine's public
    surface: `capture_now(accurate=...)` and the reader cadence are not `EngineActions`)."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def capture_now(self, accurate: bool = False) -> None:
        screen = self._engine.screen
        if screen is not None:
            screen.capture_now(accurate=accurate)

    def checkpoint(self) -> None:
        self._engine.checkpoint()

    def text(self, focus: str = "", label: str = "") -> int | None:
        return self._engine.text(focus, label)

    def code(self, focus: str = "", label: str = "") -> int | None:
        return self._engine.code(focus, label)

    def recent_code(self) -> str:
        assistant = self._engine.assistant
        if assistant is not None and hasattr(assistant, "recent_code"):
            return str(assistant.recent_code() or "")
        return ""


# ---------------------------------------------------------------------------------------
def _run_headless(cfg: Config, duration_s: float = 0.0) -> int:
    """Console mode: prints transcript lines, answers and notes to stdout until Ctrl+C (or duration_s)."""
    for stream in (sys.stdout, sys.stderr):
        # OCR text and model output carry characters a cp1252 console cannot encode (a "square"
        # glyph for an icon, an em dash); never let one of them end the run.
        try:
            stream.reconfigure(errors="replace")
        except Exception:  # noqa: BLE001 - a redirected / fake stream may not support it
            pass
    bus = EventBus()
    q = bus.subscribe()
    engine = Engine(cfg, bus)
    engine.start()
    deadline = time.monotonic() + duration_s if duration_s > 0 else None
    print("ExoCortex headless. Ctrl+C to stop. Type a question + Enter to ask; "
          "/questions, /blindspots, /insights, /recap, /notes, /code, /text, /refresh (the dashboard), "
          "/dashboard (print it).", flush=True)
    from exocortex.llm.prompts import render_insights

    def _stdin_loop() -> None:
        for line in sys.stdin:
            line = line.strip()
            if line == "/notes":
                engine.generate_notes()
            elif line == "/answer":
                engine.answer_last()
            elif line == "/checkpoint":
                engine.checkpoint()
            elif line == "/code":
                engine.code()
            elif line == "/text":
                engine.text()
            elif line == "/recap":
                engine.recap()
            elif line in ("/insights", "/questions", "/blindspots"):
                engine.insights_now({"/questions": ("questions",), "/blindspots": ("blindspots",)}.get(line, ()))
            elif line == "/refresh":
                engine.dashboard_refresh()
            elif line == "/dashboard":
                print(engine.dashboard.markdown() if engine.dashboard is not None else "(no dashboard)", flush=True)
            elif line in ("/handsfree", "/handsfree on", "/handsfree off"):
                engine.set_hands_free(line.endswith("on") or (line == "/handsfree" and not cfg.handsfree.enabled))
            elif line:
                engine.ask(line)

    threading.Thread(target=_stdin_loop, name="stdin", daemon=True).start()
    streaming = False
    try:
        while True:
            for ev in drain(q):
                if isinstance(ev, FinalTranscriptEvent):
                    s = ev.segment
                    flag = " ?" if s.is_question else ""
                    print(f"[{s.t_start:6.1f}s] {s.speaker_label}: {s.text}{flag}", flush=True)
                elif isinstance(ev, AssistStartedEvent):
                    print(f"\n>>> ({ev.kind}) Q: {ev.question}\n>>> ", end="", flush=True)
                    streaming = True
                elif isinstance(ev, AssistDeltaEvent):
                    print(ev.delta, end="", flush=True)
                elif isinstance(ev, AssistDoneEvent):
                    if streaming:
                        print(f"\n>>> done: ttft {ev.ttft_s:.2f}s, {ev.tok_per_s:.0f} tok/s{' (cancelled)' if ev.cancelled else ''}\n", flush=True)
                    streaming = False
                elif isinstance(ev, AssistErrorEvent):
                    print(f"\n>>> error: {ev.message}\n", flush=True)
                    streaming = False
                elif isinstance(ev, NotesUpdatedEvent):
                    print(f"\n===== NOTES =====\n{ev.markdown}\n=================\n", flush=True)
                elif isinstance(ev, InsightsUpdatedEvent):
                    body = render_insights({"questions": ev.questions, "blindspots": ev.blindspots}) or "(no open items)"
                    print(f"\n===== INSIGHTS =====\n{body}\n====================\n", flush=True)
                elif isinstance(ev, DashboardUpdatedEvent):
                    open_count = sum(1 for item in ev.items if item.status == "open")
                    print(f"[dashboard] {ev.detail or 'updated'}: {open_count} open item(s)", flush=True)
                elif isinstance(ev, CheckpointEvent):
                    print(f"[checkpoint] coding context reset (segment {ev.segment_id})", flush=True)
                elif isinstance(ev, HandsFreeEvent):
                    print(f"[hands-free] {'on' if ev.enabled else 'off'}: {ev.state}{' - ' + ev.detail if ev.detail else ''}", flush=True)
                elif isinstance(ev, ComponentStatusEvent):
                    print(f"[{ev.component}] {ev.state}: {ev.detail}", flush=True)
                elif isinstance(ev, ScreenTextEvent):
                    if ev.changed:
                        print(f"[screen] {ev.mode} {ev.source_name!r} {len(ev.lines)} lines, {ev.ocr_ms:.0f} ms: {ev.text[:80]!r}", flush=True)
                elif isinstance(ev, ScreenStatusEvent):
                    print(f"[screen] {ev.state}: {ev.detail}", flush=True)
            if deadline is not None and time.monotonic() > deadline:
                print("duration reached, stopping", flush=True)
                break
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        engine.stop()
    return 0


def _run_ui(cfg: Config, notice: str = "") -> int:
    """The overlay. `notice` (e.g. "Closed the previous ExoCortex session" from `--replace`) is
    shown as a toast shortly after start-up -- after the overlay's own start-up toast (a hotkey
    conflict, 2.5 s) so neither replaces the other."""
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtGui import QGuiApplication
    from PySide6.QtWidgets import QApplication

    from exocortex.ui.overlay import OverlayWindow

    QGuiApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication(sys.argv[:1])
    app.setApplicationName("ExoCortex")
    app.setQuitOnLastWindowClosed(False)

    bus = EventBus()
    holder: dict[str, Any] = {}

    def _quit() -> None:
        win = holder.get("overlay")
        if win is not None:
            try:
                win.close_silently()   # stops timers, saves geometry into cfg, no recursive quit
            except Exception:  # noqa: BLE001
                log.exception("overlay close failed")
        app.quit()

    engine = Engine(cfg, bus, on_quit=_quit)
    overlay = OverlayWindow(cfg, bus, engine)
    holder["overlay"] = overlay
    engine.start()
    if not cfg.ui.start_hidden:
        overlay.show()
    if notice:
        QTimer.singleShot(3000, overlay, lambda: overlay.show_toast(notice))
    app.aboutToQuit.connect(engine.stop)   # runs after the overlay wrote its geometry into cfg
    return app.exec()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="exocortex", description="Local, screen-shareable meeting & pair-programming co-pilot")
    parser.add_argument("--config", help="path to config.toml")
    parser.add_argument("--doctor", action="store_true", help="print environment diagnostics and exit")
    parser.add_argument("--no-ui", action="store_true", help="headless console mode")
    parser.add_argument("--duration", type=float, default=0.0, help="headless: stop after N seconds (0 = until Ctrl+C)")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument(
        "--replace", action="store_true",
        help="force-close every other running ExoCortex (and its model server) first, so this is the "
             "only session; run.bat and run.vbs always pass it",
    )
    args = parser.parse_args(argv)

    from exocortex.logging_setup import setup_logging

    setup_logging(args.log_level)
    cfg = load_config(args.config)
    log.info("config: %s", cfg.path or "(defaults)")

    notice = ""
    if args.replace:
        # Before anything heavy loads: the old session's model -- and the shared llama-swap's
        # copy, when one is resident -- must be gone from the GPU before this one's starts
        # loading (see exocortex/instance.py).
        from exocortex.instance import free_llama_swap, replace_previous

        notice = replace_previous(cfg.sessions_dir()).notice
        # The route is not resolved yet at this point (Engine.start() has not run); "auto" counts
        # as llamacpp-like here too -- unloading the owner's llama-swap when the route later turns
        # out to be Ollama/openai/none is harmless, and matches the explicit-llamacpp behaviour.
        if cfg.llm.backend.lower() in ("llamacpp", "auto") and cfg.llm.swap_unload_url:
            freed = free_llama_swap(cfg.llm.swap_unload_url)
            notice = "; ".join(part for part in (notice, freed) if part)

    if args.doctor:
        from exocortex.doctor import run_doctor

        return run_doctor(cfg)
    if args.no_ui:
        if notice:
            print(f"[replace] {notice}", flush=True)
        return _run_headless(cfg, args.duration)
    return _run_ui(cfg, notice)

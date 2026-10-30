"""Live screen reading: a background worker that grabs, OCRs and publishes screen text.

:class:`ScreenReader` owns a "screen-reader" daemon thread that periodically (or on demand,
via :meth:`ScreenReader.capture_now`) grabs the active window / a monitor / a user-picked
region (:mod:`exocortex.screen.capture`), skips the OCR pass when the image has not visibly
changed (a cheap thumbnail-diff "signature"), and otherwise OCRs it (:mod:`exocortex.screen.ocr`)
and publishes a :class:`~exocortex.bus.ScreenTextEvent` with the recognised text.

``exocortex.screen.capture`` and ``exocortex.screen.ocr`` are imported lazily, only once the
worker thread actually starts, for two reasons: neither module may be importable on a non-Windows
machine (``mss``/``winrt`` are Windows-only), and this keeps :mod:`exocortex.screen.reader` itself
trivially importable (and independently testable) regardless of their exact state. Tests inject
fakes via the ``grabber_factory``/``engine_factory`` constructor keywords instead.
"""
from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any, TYPE_CHECKING, Callable, Literal

from exocortex.bus import EventBus, ScreenStatusEvent, ScreenTextEvent
from exocortex.config import ScreenConfig

if TYPE_CHECKING:  # pragma: no cover - type checking only, never imported at runtime
    from exocortex.screen.capture import Frame, ScreenGrabber
    from exocortex.screen.ocr import OcrEngine

log = logging.getLogger(__name__)

_VALID_MODES = frozenset({"off", "auto", "monitor", "region"})

#: When the configured OCR engine can't be created, the one other named engine to try instead.
#: "auto" has no defined "other" (its own fallback logic already tried everything it can).
_OTHER_ENGINE: dict[str, str] = {"windows": "rapidocr", "rapidocr": "windows"}

#: Backoff after an OCR (or engine-creation) failure, before the next tick is attempted. A
#: module-level variable (read fresh on every use, not captured at import time) so tests can
#: shrink it, e.g. ``reader._ERROR_BACKOFF_S = 0.05``, to assert recovery without a real wait.
_ERROR_BACKOFF_S = 5.0


class ScreenReader:
    """Worker thread "screen-reader": grabs the screen, OCRs it and publishes the text.

    Every ``cfg.interval_s`` (or immediately after :meth:`capture_now`) the worker: grabs a
    frame for the current mode, computes a cheap change "signature" and skips the OCR pass
    when nothing changed (unless the capture was forced); OCRs the frame, reduces it to
    reading-order text via ``lines_to_text``, truncates it to ``cfg.max_text_chars`` (keeping
    the beginning) and publishes a :class:`ScreenTextEvent` (``changed`` reflects whether the
    text differs from the previous publish). It publishes :class:`ScreenStatusEvent` whenever
    the paused/off state changes, whenever the active mode/source changes, and on errors
    (grab failures and OCR failures each back off ``_ERROR_BACKOFF_S`` before the next try).

    The OCR engine is created lazily, on the worker thread, the first time it is needed: if
    the configured engine raises ``OcrUnavailable``, the one other named engine is tried once
    before giving up (reported as a status error); a later tick retries engine creation from
    scratch. When ``cfg.save_text`` and a snapshot directory is set (:meth:`set_snapshot_dir`),
    each *changed* capture's text (never its image) is written to
    ``<snapshot_dir>/screen/<HHMMSS>.txt``.

    All state that is read through a property or written from another thread is guarded by an
    internal lock; properties return copies or immutable values.
    """

    def __init__(
        self,
        cfg: ScreenConfig,
        bus: EventBus,
        clock: Callable[[], float],
        *,
        grabber_factory: Callable[[], "ScreenGrabber"] | None = None,
        engine_factory: Callable[[str], "OcrEngine"] | None = None,
    ) -> None:
        self._cfg = cfg
        self._bus = bus
        self._clock = clock
        self._grabber_factory = grabber_factory
        self._engine_factory = engine_factory

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None

        # ---- state protected by self._lock ----
        self._paused = False
        self._forced = False
        self._forced_accurate = False          # the forced capture should use the accurate engine
        self._accurate_name: str | None = None         # OCR engine name for accurate re-reads (None = off)
        self._snapshot_dir: Path | None = None
        self._last_text = ""
        self._last_lines: tuple[str, ...] = ()
        self._latest_event: ScreenTextEvent | None = None
        self._latest_snapshot: Any = None      # exocortex.screen.context.ScreenSnapshot (OCR lines + geometry)
        # 2026-10-29: `capture_and_wait` -- a request counter the worker echoes once the pass that
        # saw the request is over (text published, or nothing to publish: paused / off / an error).
        self._pass_requested = 0
        self._pass_started = 0
        self._pass_completed = 0
        self._pass_cv = threading.Condition(self._lock)

        # ---- state touched only by the worker thread itself ----
        self._engine: "OcrEngine | None" = None
        self._accurate_engine: "OcrEngine | None" = None
        self._accurate_attempted = False
        self._last_signature: object | None = None
        self._grab_error_active = False
        self._last_idle_state: str | None = None
        self._last_active_key: tuple[str, str] | None = None

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        """Start the "screen-reader" worker thread (a no-op while already running)."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._wake.clear()
        self._thread = threading.Thread(target=self._run, name="screen-reader", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Signal the worker to stop and wait up to 5 s for it to exit."""
        self._stop_event.set()
        self._wake.set()
        with self._pass_cv:
            self._pass_cv.notify_all()        # a capture_and_wait() caller stops waiting too
        thread, self._thread = self._thread, None
        if thread is not None and thread.is_alive():
            thread.join(timeout=5.0)

    # ------------------------------------------------------------------ #
    # controls (safe to call from any thread)
    # ------------------------------------------------------------------ #

    def set_mode(self, mode: str) -> None:
        """Switch capture mode: ``off`` | ``auto`` | ``monitor`` | ``region``.

        Raises ``ValueError`` for any other value. Forces the next tick to OCR regardless of
        whether the grabbed image looks unchanged from before: the source has just changed by
        definition, and SPEC.md requires a ScreenStatusEvent "on mode/source ... changes" --
        relying on the visual signature alone could coincidentally skip that (e.g. switching to
        a window whose thumbnail happens to resemble the previous one), leaving the status/text
        stale until the picture actually moves.
        """
        if mode not in _VALID_MODES:
            raise ValueError(f"invalid screen mode {mode!r}; expected one of {sorted(_VALID_MODES)}")
        with self._lock:
            self._cfg.mode = mode
            self._forced = True
        self._wake.set()

    def set_region(self, x: int, y: int, w: int, h: int) -> None:
        """Set the region to read (screen pixels) and switch to ``region`` mode.

        Raises ``ValueError`` unless ``w`` and ``h`` are both positive. Forces the next tick to
        OCR (see :meth:`set_mode`).
        """
        if w <= 0 or h <= 0:
            raise ValueError(f"region width/height must be > 0, got w={w} h={h}")
        with self._lock:
            self._cfg.region = [int(x), int(y), int(w), int(h)]
            self._cfg.mode = "region"
            self._forced = True
        self._wake.set()

    def set_monitor(self, index: int) -> None:
        """Set the monitor index used for ``monitor`` mode (0 = primary). Forces the next tick
        to OCR (see :meth:`set_mode`)."""
        with self._lock:
            self._cfg.monitor = int(index)
            self._forced = True
        self._wake.set()

    def capture_now(self, accurate: bool = False) -> None:
        """Wake the worker and force the next tick to OCR regardless of whether it changed.

        ``accurate=True`` (hands-free mode) runs that pass through the accurate engine set with
        :meth:`set_accurate_engine` when there is one (else the normal engine) and marks the
        event ``accurate=True`` either way -- the flag says "this answers an accurate request",
        ``ScreenTextEvent.engine`` says which engine actually produced it.
        """
        with self._lock:
            self._forced = True
            self._forced_accurate = self._forced_accurate or accurate
        self._wake.set()

    def capture_and_wait(self, timeout_s: float, *, accurate: bool = False) -> bool:
        """:meth:`capture_now`, then block -- at most ``timeout_s`` -- until the worker has run the
        pass that saw the request: its text published (``latest_text`` is then that fresh read), or
        nothing to publish because reading is paused / off or the grab failed. Returns whether the
        pass finished in time; False at once when the worker is not running, or from the worker's
        own thread. 2026-10-29: what a Code / Text job calls so it reads the screen the user is on
        at the moment of the shortcut, not the one up to ``interval_s`` ago. Any other thread."""
        thread = self._thread
        if thread is None or not thread.is_alive() or thread is threading.current_thread():
            return False
        with self._pass_cv:
            self._forced = True
            self._forced_accurate = self._forced_accurate or accurate
            self._pass_requested += 1
            wanted = self._pass_requested
        self._wake.set()
        with self._pass_cv:
            self._pass_cv.wait_for(
                lambda: self._pass_completed >= wanted or self._stop_event.is_set(),
                timeout=max(0.0, float(timeout_s)),
            )
            return self._pass_completed >= wanted

    def set_accurate_engine(self, name: str | None) -> None:
        """Name of the OCR engine to use for accurate re-reads (``capture_now(accurate=True)``),
        e.g. ``"rapidocr"``; ``None`` turns them off. Must be set before the worker creates its
        first engine to take effect: the accurate engine is built *before* the sampling one (see
        the process-order rule in :mod:`exocortex.screen.ocr`), and never later."""
        with self._lock:
            self._accurate_name = name or None

    @property
    def accurate_engine_ready(self) -> bool:
        """Whether an accurate engine was created (False until the worker built it, or forever
        when its creation failed / was refused)."""
        return self._accurate_engine is not None

    def pause(self) -> None:
        """Stop capturing until :meth:`resume`; the worker thread stays alive."""
        with self._lock:
            self._paused = True

    def resume(self) -> None:
        """Resume capturing after :meth:`pause`."""
        with self._lock:
            self._paused = False
        self._wake.set()

    def set_snapshot_dir(self, path: Path | None) -> None:
        """Set (or clear) the session folder used for ``cfg.save_text`` snapshots."""
        with self._lock:
            self._snapshot_dir = path

    # ------------------------------------------------------------------ #
    # read-only state
    # ------------------------------------------------------------------ #

    @property
    def latest_text(self) -> str:
        with self._lock:
            return self._last_text

    @property
    def latest_lines(self) -> tuple[str, ...]:
        with self._lock:
            return self._last_lines

    @property
    def latest_snapshot(self) -> Any:
        """The latest OCR result with line boxes and the frame's screen rect (a
        :class:`~exocortex.screen.context.ScreenSnapshot`), or ``None`` before the first OCR.
        Used by completion mode to read code around the caret in editors that expose no text."""
        with self._lock:
            return self._latest_snapshot

    @property
    def latest_event(self) -> ScreenTextEvent | None:
        with self._lock:
            return self._latest_event

    @property
    def mode(self) -> str:
        with self._lock:
            return self._cfg.mode

    @property
    def paused(self) -> bool:
        with self._lock:
            return self._paused

    # ------------------------------------------------------------------ #
    # worker thread
    # ------------------------------------------------------------------ #

    def _run(self) -> None:
        from exocortex.screen.capture import ScreenGrabber, is_changed, signature
        from exocortex.screen.ocr import OcrUnavailable, create_engine, lines_to_text

        grabber_factory = self._grabber_factory or ScreenGrabber
        engine_factory = self._engine_factory or create_engine

        try:
            grabber = grabber_factory()
        except Exception:
            log.exception("screen-reader: could not create the screen grabber")
            self._bus.publish(ScreenStatusEvent(state="error", detail="could not start screen capture"))
            return

        try:
            while not self._stop_event.is_set():
                try:
                    wait_s = self._tick(grabber, engine_factory, signature, is_changed, lines_to_text, OcrUnavailable)
                except Exception:
                    log.exception("screen-reader: unexpected error in worker tick")
                    wait_s = _ERROR_BACKOFF_S
                finally:
                    with self._pass_cv:          # the pass that saw request N is over (capture_and_wait)
                        self._pass_completed = max(self._pass_completed, self._pass_started)
                        self._pass_cv.notify_all()
                if self._stop_event.is_set():
                    break
                self._wake.wait(max(0.0, wait_s))
                self._wake.clear()
        finally:
            # Release the mss instance (a real screen-capture handle) and, if the OCR engine
            # holds a resource of its own (WindowsOcr's private asyncio loop), close that too.
            # Without this, repeatedly start()/stop()ing a ScreenReader in one process (or a
            # future "restart" feature) would leak one such handle per cycle.
            self._close_quietly(grabber)
            self._close_quietly(self._engine)
            self._close_quietly(self._accurate_engine)
            self._engine = None
            self._accurate_engine = None

    def _tick(
        self,
        grabber: "ScreenGrabber",
        engine_factory: Callable[[str], "OcrEngine"],
        signature_fn: Callable[..., object],
        is_changed_fn: Callable[..., bool],
        lines_to_text_fn: Callable[..., str],
        ocr_unavailable_cls: type[Exception],
    ) -> float:
        """Run one capture/OCR cycle; returns how long to wait before the next one."""
        with self._lock:
            mode = self._cfg.mode
            region = tuple(self._cfg.region)
            monitor = self._cfg.monitor
            paused = self._paused
            forced = self._forced
            accurate = self._forced_accurate and forced
            self._forced = False
            self._forced_accurate = False
            self._pass_started = self._pass_requested     # every request made so far is served by this pass
            interval = float(self._cfg.interval_s)
            exclude_own = bool(getattr(self._cfg, "exclude_own_windows", True))

        if paused or mode == "off":
            self._publish_idle("paused" if paused else "off")
            return interval
        self._last_idle_state = None

        region_arg = region if mode == "region" else None
        frame: "Frame | None"
        try:
            frame = grabber.grab(mode, region=region_arg, monitor=monitor)
        except Exception as exc:
            log.warning("screen-reader: grab failed: %s", exc)
            self._publish_grab_error(str(exc))
            return interval
        if frame is None:
            self._publish_grab_error("no capturable window/region")
            return interval
        self._grab_error_active = False
        if exclude_own:
            # Before the change signature: the overlay is capturable, and neither its text nor
            # its streaming answers may reach the OCR (or count as a screen change).
            self._mask_own_windows(frame)

        sig = signature_fn(frame.image)
        if not forced and not is_changed_fn(self._last_signature, sig):
            return interval
        window_title, window_process = self._foreground_info()

        try:
            engine = self._get_engine(engine_factory, ocr_unavailable_cls, frame)
            if accurate and self._accurate_engine is not None:
                engine = self._accurate_engine   # else: the sampling engine answers the request
            result = engine.recognize(frame.image)
        except Exception as exc:
            log.warning("screen-reader: OCR failed: %s", exc)
            self._bus.publish(ScreenStatusEvent(state="error", detail=str(exc)))
            return _ERROR_BACKOFF_S
        engine_name = self._engine_label(engine)

        # Only commit the signature once OCR actually succeeded: on a static screen, a
        # transient OCR failure must not permanently look "unchanged" (that would starve the
        # "a later tick recovers" behaviour -- with the same unchanging frame, the signature
        # check would skip every subsequent tick and OCR would never be retried).
        self._last_signature = sig

        with self._lock:
            max_chars = self._cfg.max_text_chars
            save_text = self._cfg.save_text
            snapshot_dir = self._snapshot_dir

        text = lines_to_text_fn(result.lines)[:max_chars]
        lines = tuple(line.text for line in result.lines)
        ocr_ms = result.elapsed_ms
        try:
            from exocortex.screen.context import ScreenSnapshot

            snapshot = ScreenSnapshot(lines=tuple(result.lines), rect=tuple(frame.rect),
                                      captured_at=self._clock(), source_name=frame.source_name)
        except Exception:  # noqa: BLE001 - geometry is optional
            snapshot = None

        with self._lock:
            self._latest_snapshot = snapshot
            changed = text != self._last_text
            self._last_text = text
            self._last_lines = lines
            event = ScreenTextEvent(
                mode=mode,
                source_name=frame.source_name,
                rect=frame.rect,
                text=text,
                lines=lines,
                changed=changed,
                ocr_ms=ocr_ms,
                captured_at=self._clock(),
                engine=engine_name,
                accurate=accurate,
                window_title=window_title,
                window_process=window_process,
            )
            self._latest_event = event
        self._bus.publish(event)

        if changed and save_text and snapshot_dir is not None:
            self._write_snapshot(snapshot_dir, text)

        active_key = (mode, frame.source_name)
        if active_key != self._last_active_key:
            self._last_active_key = active_key
            self._bus.publish(
                ScreenStatusEvent(state="active", detail=f"{mode}: {frame.source_name} ({ocr_ms:.0f} ms)")
            )

        return interval

    @staticmethod
    def _mask_own_windows(frame: "Frame") -> None:
        """Blank this app's own visible windows out of `frame` (``[screen].exclude_own_windows``).
        Best effort: a failure is logged and the frame is read as captured."""
        try:
            from exocortex.screen import foreground as _fg
            from exocortex.screen.capture import mask_rects

            rects = _fg.own_window_rects()
            if rects:
                mask_rects(frame, rects)
        except Exception:  # noqa: BLE001 - masking must never break screen reading
            log.debug("screen-reader: masking our own windows failed", exc_info=True)

    @staticmethod
    def _foreground_info() -> tuple[str, str]:
        """``(title, executable name)`` of the foreground window right now, for the event's
        ``window_title`` / ``window_process`` (hands-free mode's "a development window is in
        front" cue), whatever the capture mode. ``("", "")`` off Windows, when our own overlay is
        in front, or on any error -- a few microseconds, never a reason to skip the OCR."""
        try:
            from exocortex.screen.foreground import foreground_window

            win = foreground_window()
        except Exception:  # noqa: BLE001
            return "", ""
        if win is None:
            return "", ""
        return str(win.title or ""), str(win.process_name or "")

    @staticmethod
    def _engine_label(engine: object) -> str:
        """``"windows"`` / ``"rapidocr/igpu"`` for the event and the status line."""
        try:
            from exocortex.screen.ocr import engine_label

            return engine_label(engine)
        except Exception:  # noqa: BLE001 - a label must never break a tick
            return str(getattr(engine, "name", "") or "ocr")

    def _warm_quietly(self, engine: object, frame: "Frame") -> None:
        """Let an engine that supports it (RapidOCR on an accelerator) compile its kernels for
        this screen size now -- on a thread of its own, so sampling with the other engine starts
        at once; the engine serialises the warm-up and a read that arrives meanwhile (that read
        simply waits, ~7 s with a warm cache, ~25 s the first time on a machine)."""
        warm = getattr(engine, "warm", None)
        if not callable(warm):
            return
        h, w = frame.image.shape[:2]

        def _run() -> None:
            try:
                warm(int(w), int(h))
            except Exception:  # noqa: BLE001 - warm-up is an optimisation only
                log.debug("screen-reader: engine warm-up failed", exc_info=True)

        threading.Thread(target=_run, name="screen-ocr-warmup", daemon=True).start()

    @staticmethod
    def _close_quietly(obj: object | None) -> None:
        """Best-effort ``obj.close()``: not every grabber/engine has one, and this must never
        raise (it runs while the worker thread is shutting down)."""
        if obj is None:
            return
        close = getattr(obj, "close", None)
        if not callable(close):
            return
        try:
            close()
        except Exception:
            log.debug("screen-reader: close() failed for %r", obj, exc_info=True)

    def _get_engine(
        self, engine_factory: Callable[[str], "OcrEngine"], ocr_unavailable_cls: type[Exception],
        frame: "Frame | None" = None,
    ) -> "OcrEngine":
        """Return the cached OCR engine, creating it (with one fallback) if needed.

        When an accurate engine is wanted (:meth:`set_accurate_engine`) it is created FIRST, once,
        right here before the sampling engine -- :mod:`exocortex.screen.ocr`'s process-order rule:
        RapidOCR after Windows OCR crashes the interpreter, RapidOCR first is fine. A failure to
        create it is logged once and hands-free re-reads then use the sampling engine. If the
        sampling engine turns out to be the same kind, the accurate engine is simply reused as
        the sampling engine (one instance, no second model load).
        """
        if self._engine is not None:
            return self._engine
        with self._lock:
            requested = self._cfg.ocr_engine
            accurate_name = self._accurate_name
        if accurate_name and not self._accurate_attempted:
            self._accurate_attempted = True
            try:
                self._accurate_engine = engine_factory(accurate_name)
                log.info("screen-reader: accurate OCR engine %r ready", self._engine_label(self._accurate_engine))
                if frame is not None:
                    self._warm_quietly(self._accurate_engine, frame)
            except Exception as exc:  # noqa: BLE001 - the accurate pass is optional
                log.warning("screen-reader: accurate OCR engine %r unavailable (%s); re-reads use the sampling engine",
                            accurate_name, exc)
                self._accurate_engine = None
        if self._accurate_engine is not None and requested == accurate_name:
            self._engine = self._accurate_engine
            return self._engine
        try:
            engine = engine_factory(requested)
        except ocr_unavailable_cls as exc:
            other = _OTHER_ENGINE.get(requested)
            if other is None:
                raise
            if self._accurate_engine is not None and other == accurate_name:
                log.warning("screen-reader: OCR engine %r unavailable (%s); using the accurate %r engine for sampling too",
                            requested, exc, other)
                self._engine = self._accurate_engine
                return self._engine
            log.warning("screen-reader: OCR engine %r unavailable (%s); trying %r instead", requested, exc, other)
            engine = engine_factory(other)
        self._engine = engine
        if frame is not None and engine is not self._accurate_engine:
            self._warm_quietly(engine, frame)
        return engine

    def _publish_idle(self, state: Literal["paused", "off"]) -> None:
        """Publish an idle ScreenStatusEvent, but only once per (paused/off) transition."""
        if self._last_idle_state == state:
            return
        self._last_idle_state = state
        self._last_active_key = None  # force a fresh "active" status once capturing resumes
        self._bus.publish(ScreenStatusEvent(state=state))

    def _publish_grab_error(self, detail: str) -> None:
        """Publish a grab-failure ScreenStatusEvent, but only once until a grab succeeds again."""
        if self._grab_error_active:
            return
        self._grab_error_active = True
        self._bus.publish(ScreenStatusEvent(state="error", detail=detail))

    def _write_snapshot(self, snapshot_dir: Path, text: str) -> None:
        """Write a text-only snapshot (never an image) under ``<snapshot_dir>/screen/``."""
        try:
            out_dir = snapshot_dir / "screen"
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / f"{time.strftime('%H%M%S')}.txt").write_text(text, encoding="utf-8")
        except Exception:
            log.exception("screen-reader: failed to write screen text snapshot")

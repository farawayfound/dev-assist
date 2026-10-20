"""SpeechPipeline: the live transcription path from raw audio to `FinalTranscriptEvent`.

Owns up to two `AudioCapture` instances (MIC, SYSTEM; see `exocortex/audio/capture.py`), one
`VadSegmenter` per active source, a single ASR worker thread that owns every
`Transcriber.transcribe()` and `SpeakerClusterer.assign()` call, and publishes
`AudioLevelEvent` / `AudioStatusEvent` / `PartialTranscriptEvent` / `FinalTranscriptEvent` /
`ComponentStatusEvent` / `SpeakerRenamedEvent` to the bus.

Threading (see SPEC.md "Threading model"):

    capture consumer thread (owned by AudioCapture, one per source)
      --frames--> bounded per-source queue --> "vad-<source>" worker thread
                                                  (VAD segmentation + force-split)
      --VAD events--> priority queue (0=final, 1=partial) --> "asr-worker" thread
                                                  (owns the one Transcriber/SpeakerClusterer
                                                   call in flight at a time) --> EventBus

`AudioCapture` is imported lazily inside :meth:`SpeechPipeline.start` (never at module import
time), so this module stays importable without `pyaudiowpatch` installed; tests inject a fake
capture implementation via the `capture_factory` constructor argument.
"""
from __future__ import annotations

import itertools
import logging
import queue
import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Protocol, runtime_checkable

import numpy as np

from exocortex.bus import (
    AudioLevelEvent,
    AudioStatusEvent,
    ComponentStatusEvent,
    EventBus,
    FinalTranscriptEvent,
    PartialTranscriptEvent,
    SpeakerRenamedEvent,
    Source,
    TranscriptSegment,
)
from exocortex.config import Config
from exocortex.llm.questions import is_question
from exocortex.speech.asr import AsrResult, Transcriber
from exocortex.speech.diarize import SpeakerClusterer
from exocortex.speech.vad import SpeechEnd, SpeechOngoing, SpeechStart, VadEvent, VadSegmenter

log = logging.getLogger(__name__)

#: Frames are always 512 samples @ 16 kHz (SPEC.md / AudioCapture default).
_SAMPLE_RATE = 16000

#: Bounded per-source raw-frame queue: ~3.2s of 512-sample frames. Overflow drops the oldest
#: frame (with a warning counter) rather than blocking the capture's own consumer thread.
_FRAME_QUEUE_MAXSIZE = 100

#: AudioLevelEvent is throttled to at most this often per source (10 Hz).
_LEVEL_THROTTLE_S = 0.1

#: A frame-arrival gap longer than this (while not mid-utterance) triggers a clock realignment.
_GAP_REALIGN_S = 1.0

#: Echo suppression only compares finals of the other source that ended within this window.
_ECHO_WINDOW_S = 8.0
_ECHO_OVERLAP_THRESHOLD = 0.8

_QUEUE_POLL_S = 0.2
_VAD_JOIN_TIMEOUT_S = 3.0
_ASR_JOIN_TIMEOUT_S = 10.0

#: Safety cap on the ASR job queue. Finals are never dropped (their `put()` blocks if this is
#: ever reached, applying backpressure to the submitting vad-* thread) - this bound exists only
#: to stop the queue growing without limit if the ASR worker falls permanently behind (e.g. a
#: slow CPU fallback); it is far above anything a healthy pipeline should ever reach. Partial
#: jobs are best-effort and are dropped (never blocked) once the queue is at this size, since a
#: dropped partial never loses transcript data and the version bump that precedes it already
#: makes any older still-queued partial for that source stale.
_ASR_QUEUE_MAXSIZE = 256

_TOKEN_RE = re.compile(r"[a-z0-9']+")

_FINAL_PRIORITY = 0
_PARTIAL_PRIORITY = 1


@runtime_checkable
class _CaptureLike(Protocol):
    """Structural contract this module needs from `exocortex.audio.capture.AudioCapture`."""

    def start(self) -> Any: ...
    def stop(self) -> None: ...
    def set_muted(self, muted: bool) -> None: ...
    @property
    def device(self) -> Any: ...
    @property
    def running(self) -> bool: ...


#: Builds one capture instance for `source`; matches `AudioCapture.__init__`'s keyword-only
#: tail exactly (source and spec positional, everything else keyword) so a real
#: `exocortex.audio.capture.AudioCapture` and a test fake are interchangeable.
CaptureFactory = Callable[..., _CaptureLike]


def _default_capture_factory(
    source: Source,
    spec: str,
    *,
    target_rate: int,
    frame_samples: int,
    on_frames: Callable[[np.ndarray], None],
    on_level: Callable[[float, float], None] | None,
    on_status: Callable[[str, str, str], None] | None,
    keepalive: bool,
) -> _CaptureLike:
    """Build a real `exocortex.audio.capture.AudioCapture`.

    Imported lazily (only when actually called, i.e. from `SpeechPipeline.start`) so this
    module - and any test that never starts a pipeline - never needs `pyaudiowpatch`.
    """
    from exocortex.audio.capture import AudioCapture

    return AudioCapture(
        source,
        spec,
        target_rate=target_rate,
        frame_samples=frame_samples,
        on_frames=on_frames,
        on_level=on_level,
        on_status=on_status,
        keepalive=keepalive,
    )


def _token_set(text: str) -> set[str]:
    """Lower-cased, punctuation-stripped word tokens of `text`, for echo-overlap comparison."""
    return set(_TOKEN_RE.findall(text.lower()))


def _token_overlap(a: set[str], b: set[str]) -> float:
    """Normalised token-set overlap: `|a & b| / min(|a|, |b|)`, or 0.0 if either is empty."""
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


@dataclass(order=True)
class _AsrJob:
    """One unit of work for the ASR worker thread.

    Ordered by `(priority, seq)` only - `priority` 0 = final, 1 = partial (finals always run
    before any pending partial); `seq` is a monotonically increasing tiebreaker so jobs of the
    same priority run in submission order. `version` is the source's partial-version counter
    at submission time: a partial job is skipped (never transcribed) if a newer partial or a
    final for the same source has since been submitted (see `SpeechPipeline._bump_version`).
    """

    priority: int
    seq: int
    kind: Literal["final", "partial"] = field(compare=False)
    source: Source = field(compare=False)
    audio: np.ndarray = field(compare=False)
    t_start: float = field(compare=False)
    t_end: float = field(compare=False)
    version: int = field(compare=False)


class SpeechPipeline:
    """Owns AudioCapture x2 (per `AudioConfig` flags), VadSegmenter x2, one ASR worker thread,
    and the speaker clusterer.

    - MIC finals -> speaker `("me", "Me")`. SYSTEM finals -> `clusterer.assign()` if a
      clusterer was given, else `("spk1", "Speaker 1")`.
    - Echo suppression (`cfg.audio.echo_suppression`): drops a final whose normalised token set
      overlaps >= 0.8 with a final of the *other* source that ended within the last 8s (the
      earlier of the two, by processing order, is kept).
    - Publishes `AudioLevelEvent` (throttled to 10/s/source), `AudioStatusEvent`,
      `PartialTranscriptEvent`, `FinalTranscriptEvent`, `ComponentStatusEvent`
      (`"audio"`/`"diarization"`), and `SpeakerRenamedEvent`.
    - Segment ids come from the `next_id` callable supplied by the Engine (monotonically
      increasing).
    - `TranscriptSegment.is_question` is set via `llm.questions.is_question(text)` for SYSTEM
      segments only (MIC is always `False`: the user is never asked a question by themself).

    Integration facts this class handles (see SPEC.md "speech/pipeline.py"):
    - sherpa-onnx does not split continuous speech at `max_speech_duration` on its own: when a
      `SpeechOngoing`'s audio reaches `cfg.audio.max_utterance_s` worth of samples, this class
      calls `segmenter.flush()` to force a `SpeechEnd` for the audio so far (the VAD keeps
      running; the rest of the monologue becomes the next segment).
    - `VadSegmenter.process()` requires exactly 512 samples per call; frames are delivered at
      that size by `AudioCapture` and simply queued/dequeued whole.
    - If no frame arrives for `AudioCapture`'s at-most-512-sample cadence and the gap between two
      consecutive delivered frames exceeds 1s while the segmenter is not mid-utterance, the
      segmenter's time origin is realigned to `clock()` before the next frame is fed, so
      timestamps keep tracking the session clock across loopback idle periods / device changes.
    - Partials: only the newest `SpeechOngoing` per source is kept - a partial ASR job is
      skipped if a newer partial or a final for that source has since been submitted.
    """

    def __init__(
        self,
        cfg: Config,
        bus: EventBus,
        transcriber: Transcriber,
        clusterer: SpeakerClusterer | None,
        clock: Callable[[], float],
        next_id: Callable[[], int],
        *,
        capture_factory: CaptureFactory | None = None,
    ) -> None:
        """`transcriber` must already be loaded (`Transcriber.load()` called) before `start()`.
        `clusterer` may be `None` to disable diarization (SYSTEM finals then get a fixed
        `("spk1", "Speaker 1")`). `capture_factory` defaults to building a real
        `exocortex.audio.capture.AudioCapture`; tests inject a fake with the same call shape."""
        self._cfg = cfg
        self._bus = bus
        self._transcriber = transcriber
        self._clusterer = clusterer
        self._clock = clock
        self._next_id = next_id
        self._capture_factory: CaptureFactory = capture_factory or _default_capture_factory

        self._running = False
        self._stop_event = threading.Event()
        # Separate from `_stop_event`: only set once every vad-* thread has already joined (and
        # therefore finished flushing its segmenter), so the asr-worker can never exit early and
        # strand a flush-generated final job that was still in flight. See `stop()`.
        self._asr_stop_event = threading.Event()
        self._mic_muted = cfg.audio.mic_start_muted

        self._captures: dict[Source, _CaptureLike] = {}
        self._captures_lock = threading.Lock()
        self._segmenters: dict[Source, VadSegmenter] = {}
        self._frame_queues: dict[Source, "queue.Queue[np.ndarray]"] = {}
        self._vad_threads: dict[Source, threading.Thread] = {}
        self._frame_drop_counts: dict[Source, int] = {}
        self._drop_lock = threading.Lock()

        self._asr_queue: "queue.PriorityQueue[_AsrJob]" = queue.PriorityQueue(maxsize=_ASR_QUEUE_MAXSIZE)
        self._asr_thread: threading.Thread | None = None
        self._seq = itertools.count()
        # upstream finding 12 (ported from mac-cortex, mitigation only - see below): set for
        # the duration of any `_process_asr_job` call (final or partial). Checked by
        # `_enqueue_partial`, which stashes rather than queues a partial that arrives while
        # the worker is mid-transcribe (`_pending_partial` below), so several soon-to-be-
        # stale partials for the same source don't pile up on the queue while busy.
        #
        # This mitigates only the wasted-work half of finding 12; it does NOT touch the
        # finding's headline cost (each partial still re-transcribes the whole growing
        # utterance - no trailing-window or cadence back-off is implemented here) or its
        # "delays a final" scenario for a job *already* in flight: the worker still runs one
        # `_process_asr_job` to completion before it can look at the queue again, so a long
        # partial that has already started still fully blocks a final queued right behind it
        # (SPEC.md's "one transcribe at a time" contract). docs/UPSTREAM_SUGGESTIONS.md's own
        # audit records mac-cortex's status for #12 as "mitigated", not fixed, for exactly
        # this reason - matched here rather than overclaimed.
        self._asr_busy = threading.Event()
        # upstream finding 12 (ported from mac-cortex, adjusted for Windows): at most one
        # not-yet-queued partial job per source, stashed by `_enqueue_partial` while busy and
        # flushed onto `_asr_queue` by `_asr_worker_loop` the moment the worker frees up.
        # Always holds the *newest* job for a source (each `_enqueue_partial` call overwrites
        # any previous entry), so the still-current partial for a source is never silently
        # dropped just because the worker happened to be busy when it arrived - only a
        # genuinely superseded one is (still enforced by the pre/post `_is_current_partial`
        # checks in `_process_partial_job`, unchanged).
        self._pending_partial: dict[Source, _AsrJob] = {}
        self._pending_partial_lock = threading.Lock()

        self._partial_versions: dict[Source, int] = {}
        self._version_lock = threading.Lock()

        self._recent_finals: dict[Source, "deque[tuple[float, set[str]]]"] = {
            Source.MIC: deque(),
            Source.SYSTEM: deque(),
        }
        self._echo_lock = threading.Lock()

        self._last_level_ts: dict[Source, float] = {}
        self._level_lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        """Start the ASR worker, then one VAD worker and one AudioCapture per enabled source.

        A capture that fails to start publishes `ComponentStatusEvent("audio", "error", ...)`
        for that source; the other source (if enabled) still starts normally. Once every
        enabled capture has been attempted, `ComponentStatusEvent("audio", "ready", ...)` is
        published listing the devices that did start (`"<system> | <mic>"`), and
        `ComponentStatusEvent("diarization", ...)` reflects whether a clusterer is available.
        """
        if self._running:
            log.warning("SpeechPipeline.start() called while already running; ignoring")
            return
        self._stop_event.clear()
        self._asr_stop_event.clear()
        self._running = True

        self._asr_thread = threading.Thread(target=self._asr_worker_loop, name="asr-worker", daemon=True)
        self._asr_thread.start()

        sources = self._enabled_sources()
        started_names: dict[Source, str] = {}
        for source in sources:
            self._start_source(source, started_names)

        self._publish_audio_ready(started_names)
        self._publish_diarization_status()
        log.info("SpeechPipeline started: sources=%s", [s.value for s in sources])

    def _enabled_sources(self) -> list[Source]:
        sources: list[Source] = []
        if self._cfg.audio.capture_system:
            sources.append(Source.SYSTEM)
        if self._cfg.audio.capture_mic:
            sources.append(Source.MIC)
        return sources

    def _start_source(self, source: Source, started_names: dict[Source, str]) -> None:
        segmenter = self._build_segmenter()
        self._segmenters[source] = segmenter
        self._frame_queues[source] = queue.Queue(maxsize=_FRAME_QUEUE_MAXSIZE)
        thread = threading.Thread(target=self._vad_worker, args=(source,), name=f"vad-{source.value}", daemon=True)
        self._vad_threads[source] = thread
        thread.start()

        try:
            capture = self._capture_factory(
                source,
                self._spec_for(source),
                target_rate=_SAMPLE_RATE,
                frame_samples=512,
                on_frames=self._make_on_frames(source),
                on_level=self._make_on_level(source),
                on_status=self._make_on_status(source),
                keepalive=self._cfg.audio.loopback_keepalive,
            )
            if source is Source.MIC and self._mic_muted:
                capture.set_muted(True)
            device = capture.start()
        except Exception as exc:  # noqa: BLE001 - one failed capture must not stop the other
            log.exception("pipeline: failed to start %s capture", source.value)
            self._bus.publish(ComponentStatusEvent(component="audio", state="error", detail=f"{source.value}: {exc}"))
            return

        with self._captures_lock:
            self._captures[source] = capture
        started_names[source] = getattr(device, "name", "")

    def _spec_for(self, source: Source) -> str:
        return self._cfg.audio.system_device if source is Source.SYSTEM else self._cfg.audio.mic_device

    def _publish_audio_ready(self, started_names: dict[Source, str]) -> None:
        parts = [started_names[s] for s in (Source.SYSTEM, Source.MIC) if s in started_names]
        if parts:
            self._bus.publish(ComponentStatusEvent(component="audio", state="ready", detail=" | ".join(parts)))

    def _publish_diarization_status(self) -> None:
        if self._clusterer is not None:
            self._bus.publish(ComponentStatusEvent(component="diarization", state="ready"))
        elif self._cfg.diarization.enabled:
            self._bus.publish(
                ComponentStatusEvent(component="diarization", state="error", detail="no speaker clusterer available")
            )

    def stop(self) -> None:
        """Stop every capture and worker thread.

        Captures are stopped first (so no more frames arrive), then each `vad-*` thread is
        joined - which flushes its segmenter, publishing finals for any trailing speech before
        the thread exits - and finally the `asr-worker` thread is joined, which drains whatever
        those flushes queued before it exits. Safe to call multiple times / before `start()`.
        """
        if not self._running:
            return
        self._running = False
        self._stop_event.set()

        with self._captures_lock:
            captures = dict(self._captures)
            self._captures.clear()
        for source, capture in captures.items():
            try:
                capture.stop()
            except Exception:  # noqa: BLE001 - keep stopping the rest regardless
                log.exception("pipeline: error stopping %s capture", source.value)

        for source, thread in self._vad_threads.items():
            thread.join(timeout=_VAD_JOIN_TIMEOUT_S)
            if thread.is_alive():
                log.warning("pipeline: vad-%s thread did not stop within timeout", source.value)
        self._vad_threads.clear()

        # Only now: every vad-* thread has already flushed and queued its trailing final (if
        # any), so the asr-worker can safely be told to stop once it drains what is queued.
        self._asr_stop_event.set()
        if self._asr_thread is not None:
            self._asr_thread.join(timeout=_ASR_JOIN_TIMEOUT_S)
            if self._asr_thread.is_alive():
                log.warning("pipeline: asr-worker thread did not stop within timeout")
            self._asr_thread = None

        self._frame_queues.clear()
        self._segmenters.clear()
        log.info("SpeechPipeline stopped")

    def set_mic_muted(self, muted: bool) -> None:
        """Mute or unmute the MIC capture and publish `AudioStatusEvent(MIC, "muted"/"unmuted")`.

        Safe to call before `start()` (remembered and applied when the MIC capture opens) or
        while the MIC source is disabled (the status event still publishes; there is simply no
        capture to forward the mute to).
        """
        self._mic_muted = muted
        with self._captures_lock:
            capture = self._captures.get(Source.MIC)
        if capture is not None:
            try:
                capture.set_muted(muted)
            except Exception:  # noqa: BLE001 - the status event below is what the UI relies on
                log.exception("pipeline: capture.set_muted(%s) failed", muted)
        self._bus.publish(AudioStatusEvent(source=Source.MIC, state="muted" if muted else "unmuted"))

    @property
    def mic_muted(self) -> bool:
        """Whether the MIC capture is currently muted."""
        return self._mic_muted

    def rename_speaker(self, speaker_id: str, label: str) -> None:
        """Rename a speaker: remembered by the clusterer (if any) and broadcast on the bus."""
        if self._clusterer is not None:
            try:
                self._clusterer.rename(speaker_id, label)
            except Exception:  # noqa: BLE001 - the event below is what the UI/transcript rely on
                log.exception("pipeline: clusterer.rename(%r, %r) failed", speaker_id, label)
        self._bus.publish(SpeakerRenamedEvent(speaker_id=speaker_id, new_label=label))

    # ------------------------------------------------------------------ #
    # capture callbacks (called from the capture's own consumer thread)
    # ------------------------------------------------------------------ #

    def _make_on_frames(self, source: Source) -> Callable[[np.ndarray], None]:
        frame_queue = self._frame_queues[source]

        def _on_frames(frame: np.ndarray) -> None:
            try:
                frame_queue.put_nowait(frame)
            except queue.Full:
                self._drop_oldest_and_retry(source, frame_queue, frame)

        return _on_frames

    def _drop_oldest_and_retry(
        self, source: Source, frame_queue: "queue.Queue[np.ndarray]", frame: np.ndarray
    ) -> None:
        try:
            frame_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            frame_queue.put_nowait(frame)
        except queue.Full:
            return
        with self._drop_lock:
            self._frame_drop_counts[source] = self._frame_drop_counts.get(source, 0) + 1
            count = self._frame_drop_counts[source]
        log.warning("pipeline: %s frame queue overflow, dropped oldest frame (total drops=%d)", source.value, count)

    def _make_on_level(self, source: Source) -> Callable[[float, float], None]:
        def _on_level(rms: float, peak: float) -> None:
            now = time.monotonic()
            with self._level_lock:
                last = self._last_level_ts.get(source, 0.0)
                if now - last < _LEVEL_THROTTLE_S:
                    return
                self._last_level_ts[source] = now
            self._bus.publish(AudioLevelEvent(source=source, rms=rms, peak=peak))

        return _on_level

    def _make_on_status(self, source: Source) -> Callable[[str, str, str], None]:
        def _on_status(state: str, device_name: str, detail: str = "") -> None:
            if state in ("muted", "unmuted"):
                return  # set_mic_muted() publishes these explicitly (avoids double events)
            self._bus.publish(AudioStatusEvent(source=source, state=state, device_name=device_name, detail=detail))
            if state == "error":
                self._bus.publish(
                    ComponentStatusEvent(component="audio", state="error", detail=f"{source.value}: {detail}")
                )

        return _on_status

    # ------------------------------------------------------------------ #
    # VAD worker (one thread per source)
    # ------------------------------------------------------------------ #

    def _build_segmenter(self) -> VadSegmenter:
        a = self._cfg.audio
        model_path = str(self._cfg.models_dir() / "silero_vad.onnx")
        return VadSegmenter(
            model_path,
            sample_rate=_SAMPLE_RATE,
            threshold=a.vad_threshold,
            min_silence_s=a.min_silence_s,
            min_speech_s=a.min_speech_s,
            max_speech_s=a.max_utterance_s,
            pre_roll_s=a.pre_roll_s,
            ongoing_interval_s=self._cfg.asr.partial_interval_s,
        )

    def _vad_worker(self, source: Source) -> None:
        segmenter = self._segmenters[source]
        frame_queue = self._frame_queues[source]
        segmenter.set_time_origin(self._clock())
        last_frame_ts = time.monotonic()

        while True:
            try:
                frame = frame_queue.get(timeout=_QUEUE_POLL_S)
            except queue.Empty:
                if self._stop_event.is_set():
                    break
                continue
            now = time.monotonic()
            if now - last_frame_ts > _GAP_REALIGN_S and not segmenter.in_speech:
                segmenter.set_time_origin(self._clock())
            last_frame_ts = now
            events = segmenter.process(frame)
            self._handle_vad_events(source, events, segmenter)

        # Flush trailing in-progress speech so it is not silently lost on shutdown.
        self._handle_vad_events(source, segmenter.flush(), segmenter)
        log.debug("pipeline: vad-%s worker exiting", source.value)

    def _handle_vad_events(self, source: Source, events: list[VadEvent], segmenter: VadSegmenter) -> None:
        for event in events:
            if isinstance(event, SpeechStart):
                log.debug("pipeline: %s speech start at %.2f", source.value, event.t_start)
            elif isinstance(event, SpeechOngoing):
                self._handle_speech_ongoing(source, event, segmenter)
            elif isinstance(event, SpeechEnd):
                self._enqueue_final(source, event)

    def _handle_speech_ongoing(self, source: Source, event: SpeechOngoing, segmenter: VadSegmenter) -> None:
        max_samples = self._cfg.audio.max_utterance_s * _SAMPLE_RATE
        if len(event.audio) >= max_samples:
            log.info(
                "pipeline: force-splitting long %s monologue at %.1fs", source.value, len(event.audio) / _SAMPLE_RATE
            )
            self._handle_vad_events(source, segmenter.flush(), segmenter)
            return
        if self._cfg.asr.partials_enabled:
            self._enqueue_partial(source, event)

    # ------------------------------------------------------------------ #
    # ASR job queue
    # ------------------------------------------------------------------ #

    def _bump_version(self, source: Source) -> int:
        with self._version_lock:
            version = self._partial_versions.get(source, 0) + 1
            self._partial_versions[source] = version
            return version

    def _enqueue_final(self, source: Source, event: SpeechEnd) -> None:
        if len(event.audio) == 0:
            return
        version = self._bump_version(source)  # invalidates any still-pending partial
        job = _AsrJob(
            priority=_FINAL_PRIORITY, seq=next(self._seq), kind="final", source=source,
            audio=event.audio, t_start=event.t_start, t_end=event.t_end, version=version,
        )
        # Finals must never be silently dropped: block (applying backpressure to this vad-*
        # thread) in the extreme case the queue is already at `_ASR_QUEUE_MAXSIZE`.
        self._asr_queue.put(job)

    def _enqueue_partial(self, source: Source, event: SpeechOngoing) -> None:
        version = self._bump_version(source)  # makes any older still-queued/stashed partial stale
        t_end = event.t_start + len(event.audio) / _SAMPLE_RATE
        job = _AsrJob(
            priority=_PARTIAL_PRIORITY, seq=next(self._seq), kind="partial", source=source,
            audio=event.audio, t_start=event.t_start, t_end=t_end, version=version,
        )
        # upstream finding 12 (ported from mac-cortex, adjusted for Windows): when the
        # asr-worker is mid-transcribe, queuing this job would just sit behind it. Stash it as
        # the one pending partial for this source instead of putting it on the queue -
        # `_flush_pending_partials` queues whatever is still pending the moment the worker
        # frees up. Overwriting any previous entry here (rather than dropping this job
        # outright) matters: if we simply refused to enqueue, the *still-current* partial for
        # this source - the one nothing later ever supersedes - would never be queued at all
        # once the worker frees up, silently freezing that source's live caption for the rest
        # of the utterance. See test_enqueue_partial_skipped_while_asr_worker_busy /
        # test_pending_partial_flushed_once_asr_worker_frees_up in tests/test_pipeline.py, and
        # tests/test_pipeline_extra.py::
        # test_slow_transcribe_never_publishes_or_runs_a_superseded_partial, which pins exactly
        # this "the last partial still gets processed" contract end to end.
        # The busy check and the stash are one step under `_pending_partial_lock` - the same
        # lock `_flush_pending_partials` clears the flag and drains the stash under - so a
        # partial can never be stashed just after the worker's final flush and then sit there
        # unqueued until some unrelated later job passes through the worker.
        with self._pending_partial_lock:
            stashed = self._asr_busy.is_set()
            if stashed:
                self._pending_partial[source] = job
        if stashed:
            log.debug("pipeline: asr worker busy, stashed a %s partial job for later", source.value)
            return
        try:
            self._asr_queue.put_nowait(job)
        except queue.Full:
            # Best-effort: a dropped partial never loses transcript data, unlike a final.
            log.debug("pipeline: asr queue full, dropped a %s partial job", source.value)

    def _flush_pending_partials(self) -> None:
        # upstream finding 12 (ported from mac-cortex, adjusted for Windows): called once the
        # asr-worker finishes a job (see `_asr_worker_loop`): clears `_asr_busy` and queues
        # whatever partials `_enqueue_partial` stashed instead of queuing while busy - both
        # under `_pending_partial_lock`, atomically with `_enqueue_partial`'s own check-and-
        # stash, so nothing can slip into the stash between the clear and the drain.
        # `_is_current_partial`'s existing pre/post checks in `_process_partial_job` still
        # apply as normal, so a job that a *newer* partial or a final superseded in the
        # meantime is skipped there just like any other queued partial - this only
        # guarantees the job gets a chance to run.
        with self._pending_partial_lock:
            self._asr_busy.clear()
            if not self._pending_partial:
                return
            pending = list(self._pending_partial.values())
            self._pending_partial.clear()
        for job in pending:
            try:
                self._asr_queue.put_nowait(job)
            except queue.Full:
                log.debug("pipeline: asr queue full, dropped a stashed %s partial job", job.source.value)

    def _asr_worker_loop(self) -> None:
        while True:
            try:
                job = self._asr_queue.get(timeout=_QUEUE_POLL_S)
            except queue.Empty:
                if self._asr_stop_event.is_set():
                    break
                continue
            # upstream finding 12 (ported from mac-cortex): mark busy for the duration of the
            # job so `_enqueue_partial` stashes rather than queues a partial that would only
            # run once this job finishes.
            self._asr_busy.set()
            try:
                self._process_asr_job(job)
            except Exception:  # noqa: BLE001 - one bad job must not kill the worker thread
                log.exception("pipeline: asr-worker job failed (source=%s kind=%s)", job.source.value, job.kind)
            finally:
                # upstream finding 12 (ported from mac-cortex, adjusted for Windows): the worker
                # is free again - clear the busy flag and queue whatever partials were stashed
                # while it was busy (one atomic step, see `_flush_pending_partials`).
                self._flush_pending_partials()
        log.debug("pipeline: asr-worker exiting")

    def _process_asr_job(self, job: _AsrJob) -> None:
        if job.kind == "partial":
            self._process_partial_job(job)
        else:
            self._process_final_job(job)

    def _is_current_partial(self, job: _AsrJob) -> bool:
        with self._version_lock:
            return self._partial_versions.get(job.source) == job.version

    def _process_partial_job(self, job: _AsrJob) -> None:
        if not self._is_current_partial(job):
            return  # a newer partial or a final for this source already superseded it
        try:
            result = self._transcriber.transcribe(job.audio, partial=True)
        except Exception:  # noqa: BLE001 - a failed partial must not break the worker
            log.exception("pipeline: partial transcription failed for %s", job.source.value)
            if self._is_current_partial(job):
                # Clear the stale partial row rather than leaving old text on screen
                # indefinitely (mirrors the final job's error handling below).
                self._bus.publish(PartialTranscriptEvent(source=job.source, text="", t_start=job.t_start))
            return
        if not self._is_current_partial(job):
            return  # superseded while transcribing; never publish a stale partial
        text = "" if result.dropped else result.text
        self._bus.publish(PartialTranscriptEvent(source=job.source, text=text, t_start=job.t_start))

    def _process_final_job(self, job: _AsrJob) -> None:
        try:
            result = self._transcriber.transcribe(job.audio, partial=False)
        except Exception:  # noqa: BLE001 - still clear the partial row so the UI recovers
            log.exception("pipeline: final transcription failed for %s", job.source.value)
            self._bus.publish(PartialTranscriptEvent(source=job.source, text="", t_start=job.t_start))
            return

        # The utterance is finished either way: remove its partial row from the UI.
        self._bus.publish(PartialTranscriptEvent(source=job.source, text="", t_start=job.t_start))
        if result.dropped or not result.text:
            log.debug("pipeline: dropped %s final (%s)", job.source.value, result.reason)
            return

        segment = self._build_segment(job, result)
        if self._should_suppress_echo(segment):
            # upstream finding 7 (ported from mac-cortex): never log recognised text at INFO
            # (or any level) - only its length. This line lands in the persistent app log
            # (see logging_setup.py).
            log.info("pipeline: echo-suppressed %s final (%d chars)", job.source.value, len(segment.text))
            return
        self._record_final_for_echo(segment)
        self._bus.publish(FinalTranscriptEvent(segment=segment))

    def _build_segment(self, job: _AsrJob, result: AsrResult) -> TranscriptSegment:
        speaker_id, speaker_label = self._assign_speaker(job)
        confidence = float(np.clip(np.exp(result.avg_logprob), 0.0, 1.0))
        question = is_question(result.text) if job.source is Source.SYSTEM else False
        return TranscriptSegment(
            id=self._next_id(),
            source=job.source,
            speaker_id=speaker_id,
            speaker_label=speaker_label,
            text=result.text,
            t_start=job.t_start,
            t_end=job.t_end,
            confidence=confidence,
            is_question=question,
        )

    def _assign_speaker(self, job: _AsrJob) -> tuple[str, str]:
        if job.source is Source.MIC:
            return "me", "Me"
        if self._clusterer is not None:
            try:
                return self._clusterer.assign(job.audio)
            except Exception:  # noqa: BLE001 - diarization must never break transcription
                log.exception("pipeline: clusterer.assign() failed; falling back to spk1")
        return "spk1", "Speaker 1"

    # ------------------------------------------------------------------ #
    # echo suppression
    # ------------------------------------------------------------------ #

    def _record_final_for_echo(self, segment: TranscriptSegment) -> None:
        tokens = _token_set(segment.text)
        with self._echo_lock:
            dq = self._recent_finals[segment.source]
            dq.append((segment.t_end, tokens))
            while dq and segment.t_end - dq[0][0] > _ECHO_WINDOW_S:
                dq.popleft()

    def _should_suppress_echo(self, segment: TranscriptSegment) -> bool:
        if not self._cfg.audio.echo_suppression:
            return False
        tokens = _token_set(segment.text)
        if not tokens:
            return False
        other = Source.SYSTEM if segment.source is Source.MIC else Source.MIC
        with self._echo_lock:
            candidates = list(self._recent_finals[other])
        for t_end, other_tokens in candidates:
            if abs(segment.t_end - t_end) > _ECHO_WINDOW_S:
                continue
            if _token_overlap(tokens, other_tokens) >= _ECHO_OVERLAP_THRESHOLD:
                return True
        return False

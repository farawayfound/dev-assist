"""Thread-safe fan-out event bus and the event/data types shared by every module.

Design rules:
- Every producer thread calls `bus.publish(event)`; it never blocks.
- Every consumer calls `q = bus.subscribe()` once and drains `q` on its own thread
  (the Qt UI drains it from a QTimer; the Assistant drains it on a worker thread).
- Events are immutable dataclasses. `ts` is `time.monotonic()` at creation.
- Segment times (`t_start`, `t_end`) are seconds since session start (the Engine's clock).
"""
from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal


class Source(str, Enum):
    MIC = "mic"        # the user's microphone  -> speaker "me" / label "Me"
    SYSTEM = "system"  # WASAPI loopback of the default output -> remote participants


@dataclass(frozen=True)
class TranscriptSegment:
    id: int
    source: Source
    speaker_id: str      # "me" for MIC; "spk1", "spk2", ... for SYSTEM
    speaker_label: str   # "Me", "Speaker 1", or a user-assigned name
    text: str
    t_start: float       # seconds since session start
    t_end: float
    confidence: float    # 0..1, roughly exp(avg_logprob) clipped
    is_question: bool = False

    @property
    def duration(self) -> float:
        return max(0.0, self.t_end - self.t_start)


def _now() -> float:
    return time.monotonic()


@dataclass(frozen=True)
class Event:
    ts: float = field(default_factory=_now, init=False)


# ---- audio -----------------------------------------------------------------
@dataclass(frozen=True)
class AudioLevelEvent(Event):
    source: Source
    rms: float   # 0..1 linear
    peak: float  # 0..1 linear


@dataclass(frozen=True)
class AudioStatusEvent(Event):
    source: Source
    state: Literal["started", "stopped", "error", "device_changed", "muted", "unmuted"]
    device_name: str = ""
    detail: str = ""


# ---- transcript --------------------------------------------------------------
@dataclass(frozen=True)
class PartialTranscriptEvent(Event):
    """In-progress text for the utterance currently being spoken (replaces previous partial)."""
    source: Source
    text: str
    t_start: float


@dataclass(frozen=True)
class FinalTranscriptEvent(Event):
    segment: TranscriptSegment


@dataclass(frozen=True)
class SpeakerRenamedEvent(Event):
    speaker_id: str
    new_label: str


@dataclass(frozen=True)
class QuestionDetectedEvent(Event):
    segment_id: int
    question_text: str


# ---- screen reading ------------------------------------------------------------
@dataclass(frozen=True)
class ScreenTextEvent(Event):
    """OCR result of the latest screen capture (published only when the text changed, or forced)."""
    mode: str                          # auto | monitor | region
    source_name: str                   # window title, "monitor 1" or "region"
    rect: tuple[int, int, int, int]    # x, y, w, h in screen pixels
    text: str                          # lines joined with "\n"
    lines: tuple[str, ...]
    changed: bool                      # False when a forced snapshot found no change
    ocr_ms: float
    captured_at: float                 # seconds since session start
    engine: str = ""                   # OCR engine that produced it ("windows", "rapidocr/igpu", ...)
    # The answer to a `capture_now(accurate=True)` request (hands-free mode) -- set whether or not an
    # accurate engine existed to serve it (`engine` tells); never set on an ordinary sample.
    accurate: bool = False
    # The foreground window at capture time (title / executable name, e.g. "main.py - Visual Studio
    # Code" / "Code.exe"), whatever the capture mode -- hands-free mode uses it to tell a
    # development window. "" when unknown or when our own overlay was in front.
    window_title: str = ""
    window_process: str = ""


@dataclass(frozen=True)
class ScreenStatusEvent(Event):
    state: Literal["active", "paused", "off", "error"]
    detail: str = ""                   # e.g. "auto: Zoom Meeting" / "region 800x600" / error text


# ---- completion mode --------------------------------------------------------------
@dataclass(frozen=True)
class CompletionEvent(Event):
    """What the ghost-text window should show right now (published by the Completer).

    Coordinates are PHYSICAL screen pixels (as Win32/accessibility report them); ``x``/``y`` is
    the caret's top-left, ``line_h`` the caret height. ``font_px`` is the matched font size in
    physical pixels (0 = unknown, derive from ``line_h``). ``current_line`` is the text before
    the caret on the caret's own line, so the renderer can line up continuation lines with the
    start of the line. ``visible=False`` hides the ghost text (other fields then unused).
    """
    visible: bool = False
    text: str = ""
    x: int = 0
    y: int = 0
    line_h: int = 0
    font_px: float = 0.0
    font_name: str = ""
    mono: bool = False
    current_line: str = ""
    app: str = ""


@dataclass(frozen=True)
class CompletionModeEvent(Event):
    """Completion mode turned on/off (UI + engine stay in sync through this)."""
    enabled: bool = False
    detail: str = ""


# ---- hands-free mode ---------------------------------------------------------------
#: Hands-free controller states: off; watching (sampling, nothing new); settling (the screen just
#: changed, waiting for it to hold still); reading (an accurate OCR re-read is in flight);
#: checkpoint / text / code (that step of the cycle is running); armed (manual advance: the new
#: task was read and checkpointed, generation waits for the press-and-hold gesture); paused.
HandsFreeState = Literal["off", "watching", "settling", "reading", "checkpoint", "text", "code", "armed", "paused"]


@dataclass(frozen=True)
class HandsFreeEvent(Event):
    """Hands-free mode turned on/off, or its controller moved to another state (the switch, the
    slim bar's menu entry and the title-bar status all follow this)."""
    enabled: bool = False
    state: HandsFreeState = "off"
    detail: str = ""


@dataclass(frozen=True)
class ManualAdvanceEvent(Event):
    """Manual advance toggled (the Manual switch next to Hands-free, ``[handsfree].manual``).
    While on, nothing starts a model call by itself -- the transcript, screen reading, cumulative
    context and auto-checkpoints keep running, and generation happens only on the press-and-hold
    mouse gesture (left button: Text, middle button: Code) or a manual action. The switch, the
    slim bar's menu entry and the settings dialog all follow this."""
    enabled: bool = False


# ---- assistant ---------------------------------------------------------------
# auto: question heard in system audio; manual: hotkey/button; ask: typed question;
# screen: question that appeared on screen; suggest: proactive suggestions from screen + transcript;
# solve: autonomous "solve what is on screen"; complete: inline ghost-text completion at the caret;
# explain: explanation of the finished solution once its code has been typed out (explanations on);
# code: the "Code" button -- write the code for the task discussed / shown since the last checkpoint;
# text: the "Text" button -- a plain-English walkthrough of that same task (what is asked, the key
# insight, the approach, edge cases, complexity), the way an engineer talks it through before coding;
# insights: the co-pilot's proactive (or on-demand) read of the session -- questions worth asking
# right now and blind spots / follow-ups -- rendered in the Notes tab's Insights pane, never in the
# Assist tab (see InsightsUpdatedEvent); recap: an on-demand "where are we" summary of the session
AssistKind = Literal[
    "auto", "manual", "ask", "screen", "suggest", "solve", "complete", "explain", "code", "text",
    "insights", "recap", "dashboard",
]
# dashboard: one refresh of the tracked dashboard (exocortex.dashboard): the model is shown the open
# items and what is new since the last refresh and answers with the changes only; the result is
# applied to the store and published as a DashboardUpdatedEvent -- never shown as an answer.


@dataclass(frozen=True)
class CheckpointEvent(Event):
    """A checkpoint was set (the Checkpoint button / hotkey): the coding context -- what solve,
    code and inline completions are built from -- starts fresh from here. The transcript, notes
    and the ask box keep the whole meeting. `segment_id` is the last transcript segment BEFORE
    the boundary (None when the transcript was empty); `t` is the session clock."""
    t: float
    segment_id: int | None = None


@dataclass(frozen=True)
class AssistStartedEvent(Event):
    job_id: int
    kind: AssistKind
    question: str


@dataclass(frozen=True)
class AssistDeltaEvent(Event):
    job_id: int
    delta: str


@dataclass(frozen=True)
class AssistDoneEvent(Event):
    job_id: int
    text: str
    ttft_s: float
    tokens: int
    tok_per_s: float
    cancelled: bool = False
    # The model answered with the skip marker ("(nothing to do)" -- prompts.SKIP_MARKER): the job
    # finished, there was just nothing to show. Always paired with cancelled=True (nothing to
    # render) so older consumers keep treating it as "no answer".
    skipped: bool = False


@dataclass(frozen=True)
class AssistErrorEvent(Event):
    job_id: int
    message: str


@dataclass(frozen=True)
class NotesUpdatedEvent(Event):
    markdown: str
    generated_at: float  # seconds since session start


@dataclass(frozen=True)
class NotesErrorEvent(Event):
    message: str


@dataclass(frozen=True)
class InsightsUpdatedEvent(Event):
    """The co-pilot's living insights document changed (a proactive pass or an on-demand request
    finished): the current markdown bullet list of each section, "" when that section has no
    items yet. A pass that refreshed only some sections carries the others unchanged, so a
    consumer can always just replace what it shows. `auto` tells a background pass from one
    the user asked for; `generated_at` is the session clock."""
    questions: str = ""
    blindspots: str = ""
    generated_at: float = 0.0
    auto: bool = False


# ---- dashboard mode (2026-10-20) ------------------------------------------------------
#: One tracked item of the dashboard (exocortex.dashboard.store). `kind`: "nextstep" (an action
#: item: a primary task, or -- with `parent_id` -- a sub-task of one), "goal" (an acceptance
#: criterion; `parent_id` = the action item it is for, none = the whole session), "constraint" (a
#: rule or guardrail to respect), "question" (to ask), "blindspot" (blind spots & follow-ups) or
#: "prompt" (to type into Claude Code; `parent_id` = the action item it serves). `status`: "open",
#: "answered" (answered / done / met / resolved -- `answer` carries the answer or how it was
#: settled) or "dropped". `source`: "model" or "user". Times are the session clock.
@dataclass(frozen=True)
class DashboardItem:
    id: int
    kind: str
    text: str
    status: str = "open"
    answer: str = ""
    created_at: float = 0.0
    updated_at: float = 0.0
    source: str = "model"
    parent_id: int | None = None       # a sub-task's task; a goal's or prompt's action item


@dataclass(frozen=True)
class DashboardUpdatedEvent(Event):
    """The dashboard's data changed (a refresh finished, the user edited an item, the notes were
    regenerated): the whole current snapshot -- every item the view shows, open and resolved --
    plus the running notes, so a consumer always just replaces what it shows. `auto` tells a
    background refresh from one the user asked for; `detail` is a one-line status for the view
    ("updated: +2 new, 1 answered", "nothing new", "refresh cancelled", "error: ...")."""
    items: tuple[DashboardItem, ...] = ()
    notes: str = ""
    notes_at: float | None = None
    generated_at: float = 0.0
    auto: bool = False
    detail: str = ""


@dataclass(frozen=True)
class DashboardModeEvent(Event):
    """Dashboard mode (the view) turned on / off -- the title-bar button, the hotkey, the menu, the
    settings dialog and the engine's own switch all stay in sync through this."""
    enabled: bool = False


# ---- system ------------------------------------------------------------------
Component = Literal["audio", "asr", "diarization", "llm", "session", "app", "completion", "dashboard", "project"]
ComponentState = Literal["loading", "ready", "error", "info", "busy"]


@dataclass(frozen=True)
class ComponentStatusEvent(Event):
    component: Component
    state: ComponentState
    detail: str = ""


@dataclass(frozen=True)
class SessionEvent(Event):
    kind: Literal["new", "saved", "exported"]
    path: str = ""


@dataclass(frozen=True)
class AutoAssistChangedEvent(Event):
    enabled: bool


# ---- model manager (2026-09-27, portable-install) -----------------------------------------
# The "Models..." window (exocortex/ui/models.py) drives the engine through EngineActions'
# model_library / load_model / unload_model / delete_model / pull_model / search_models and hears
# back through these three events (plus ComponentStatusEvent("llm", ...) for load/unload progress,
# exactly like the status bar does). Every field is plain data so the dialog never touches the
# library module or the network itself.
PullState = Literal["queued", "downloading", "verifying", "done", "error", "cancelled"]


@dataclass(frozen=True)
class ModelPullEvent(Event):
    """Progress of one whole-model pull (`pull_model`): `file` is the shard / projector being
    streamed right now, `done_bytes` / `total_bytes` count the WHOLE pull (every file), `state`
    moves queued -> downloading -> verifying -> done, or error / cancelled with `detail`."""
    repo_id: str
    state: PullState
    file: str = ""
    done_bytes: int = 0
    total_bytes: int = 0
    detail: str = ""
    model_key: str = ""     # the library key once the pull is done (e.g. "unsloth/gemma-4-12b-it")


@dataclass(frozen=True)
class ModelLibraryEvent(Event):
    """The library changed (a pull finished, a model was deleted, a profile was saved, the
    loaded model changed): re-read `model_library()`. `reason` is for logs / a toast."""
    reason: str = ""
    loaded_key: str = ""    # the model llama-server currently serves ("" when nothing is loaded)


@dataclass(frozen=True)
class ModelSearchEvent(Event):
    """The answer to `search_models(query, request_id)`: `results` are plain dicts (id, downloads,
    likes, modified, gated) so a stale answer (an older `request_id`) can simply be dropped."""
    request_id: int
    query: str
    results: tuple[dict[str, Any], ...] = ()
    error: str = ""


@dataclass(frozen=True)
class ModelInfoEvent(Event):
    """The answer to `inspect_model(repo_id, request_id)`: the repo's quant options and the one
    `choose_quant` would pick for this machine, as plain dicts."""
    request_id: int
    repo_id: str
    quants: tuple[dict[str, Any], ...] = ()
    chosen: str = ""
    architecture: str = ""
    params: int | None = None
    context_length: int | None = None
    mmproj: bool = False
    error: str = ""


# ---- context library + the context window (2026-10-06) ------------------------------------
# The Models window's "Context" tab (exocortex/ui/models.py) drives the engine through
# EngineActions' context_documents / add_context_document / add_context_text /
# remove_context_document / set_context_document / refresh_context_digest and hears back through
# these two events. Plain data only, like the model-manager events above.
@dataclass(frozen=True)
class ContextLibraryEvent(Event):
    """The reference-document library changed (a document was added / removed / toggled / had its
    mode changed, or its digest was generated): re-read `context_documents()`. `reason` is for
    logs / a toast; `doc_id` names the document when one is concerned ("" otherwise)."""
    reason: str = ""
    doc_id: str = ""


@dataclass(frozen=True)
class ContextUsageEvent(Event):
    """How the context window was filled for one assist job (published by the Assistant right
    after the job's prompt was built, and once more when the server reported its own counts).
    Every `*_tokens` field is the app's own estimate (`[context].chars_per_token`) except
    `prompt_tokens` / `cached_tokens`, which are what the server reported for this request
    (llama-server `timings.prompt_n` / `cache_n`; None when the backend does not say).
    `slot_tokens` is the per-request context the budget was sized to; `budget_tokens` what was
    left for the user turn after the system prompt, the answer budget and the reserve.
    `trimmed` says something had to be cut to fit; `anchor_moved` that the transcript window's
    start advanced (a compaction may follow); `chars_per_token` is the calibrated estimate."""
    job_id: int
    kind: AssistKind
    slot_tokens: int = 0
    budget_tokens: int = 0
    used_tokens: int = 0
    transcript_tokens: int = 0
    reference_tokens: int = 0
    lookup_tokens: int = 0
    earlier_tokens: int = 0
    screen_tokens: int = 0
    system_tokens: int = 0
    prompt_tokens: int | None = None
    cached_tokens: int | None = None
    trimmed: bool = False
    anchor_moved: bool = False
    chars_per_token: float = 0.0
    documents: int = 0          # reference documents that contributed something to this prompt


@dataclass(frozen=True)
class OptimizeEvent(Event):
    """Progress of the Optimize tab's background work (2026-10-07, exocortex.optimize_run): an
    analysis, a benchmark run, or an apply/verify/rollback transaction. Published for the log and
    for anything that wants to follow along; the Optimize tab itself POLLS
    ``EngineActions.optimize_snapshot()`` (the overlay forwards no new event kinds to the
    settings dialog), so nothing depends on a subscriber seeing these."""
    kind: Literal["analysis", "started", "progress", "done", "error", "cancelled",
                  "applied", "rolled-back", "recovered"]
    run_id: int = 0
    step: str = ""            # facts | llm | speech | apply | verify
    fraction: float = 0.0     # 0..1 over the whole run
    detail: str = ""


@dataclass(frozen=True)
class ToolCallEvent(Event):
    """One tool call made by an assist job (2026-10-07, capabilities): published "running" right
    before the tool executes and once more with "done" / "error" / "denied" after. `index` is
    1-based within the job; `name` is the tool ("read_file", "run_command", "fetch_url",
    "mcp:<server>:<tool>", ...); `summary` the one line the call named (the path, the command,
    the url, the query; at most 120 chars); `detail` a size hint when done ("3.2 KB") or the
    reason when refused / failed. The Assist tab's footer shows the running one."""
    job_id: int
    index: int
    name: str
    summary: str
    state: str                 # "running" | "done" | "error" | "denied"
    detail: str = ""
    elapsed_s: float = 0.0


@dataclass(frozen=True)
class CapabilityMapEvent(Event):
    """A project map changed (2026-10-07, capabilities): re-read `capability_maps()`. `reason` is
    "indexing" (the walk started), "indexed" (saved: `files` / `dirs` counted, `truncated` in
    `detail` when the tree was cut), "loaded" (its settings were applied and it is the active
    map), "deleted", or "error: ..." (nothing changed). `automatic` marks the launch-time refresh
    of the active map (nobody asked for it); `changed`, with "indexed", says whether the set of
    files and folders differs from the map stored before the pass (sizes and summaries aside; a
    first index always counts as changed). 2026-10-29: the overlay keeps quiet about an automatic
    pass over an unchanged tree -- "Map 'X': 28 files" at every launch read as 28 documents loaded."""
    reason: str
    name: str = ""
    files: int = 0
    dirs: int = 0
    detail: str = ""
    automatic: bool = False
    changed: bool = True


@dataclass(frozen=True)
class McpToolsEvent(Event):
    """The answer to `mcp_server_tools(name, request_id)` (2026-10-07, capabilities): the tools
    the MCP server `server` offers, as plain dicts (name, description), or an `error`."""
    request_id: int
    server: str
    tools: tuple[dict[str, Any], ...] = ()
    error: str = ""


class EventBus:
    """Fan-out bus: each subscriber owns a queue; publish copies the reference to all."""

    def __init__(self) -> None:
        self._subs: list[queue.Queue[Any]] = []
        self._lock = threading.Lock()

    def subscribe(self, maxsize: int = 0) -> "queue.Queue[Any]":
        q: queue.Queue[Any] = queue.Queue(maxsize=maxsize)
        with self._lock:
            self._subs.append(q)
        return q

    def unsubscribe(self, q: "queue.Queue[Any]") -> None:
        with self._lock:
            if q in self._subs:
                self._subs.remove(q)

    def publish(self, event: Event) -> None:
        with self._lock:
            subs = list(self._subs)
        for q in subs:
            try:
                q.put_nowait(event)
            except queue.Full:
                # drop oldest to keep the consumer alive
                try:
                    q.get_nowait()
                    q.put_nowait(event)
                except queue.Empty:
                    pass
                except queue.Full:
                    pass


def drain(q: "queue.Queue[Any]", max_items: int = 200) -> list[Any]:
    """Non-blocking drain used by pollers (the UI timer)."""
    out: list[Any] = []
    for _ in range(max_items):
        try:
            out.append(q.get_nowait())
        except queue.Empty:
            break
    return out

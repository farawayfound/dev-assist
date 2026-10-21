"""The small command surface the UI (and hotkeys) use to drive the engine.

Kept Qt-free and import-light so `exocortex.ui` can depend on it without importing `exocortex.app`.
Every method must return immediately; results arrive later as events on the EventBus.
"""
from __future__ import annotations

from pathlib import Path
from typing import Protocol


class EngineActions(Protocol):
    def answer_last(self) -> None:
        """Answer the most recent thing the other participants said (manual assist)."""

    def ask(self, text: str) -> None:
        """Answer a free-form question typed by the user, with transcript context."""

    def generate_notes(self) -> None:
        """Regenerate the structured meeting notes now."""

    def cancel_assist(self) -> None:
        """Cancel the in-flight answer, if any."""

    def set_auto(self, enabled: bool) -> None:
        """Enable/disable automatic answering of detected questions."""

    def set_mic_muted(self, muted: bool) -> None:
        """Stop/resume transcribing the user's own microphone."""

    def rename_speaker(self, speaker_id: str, label: str) -> None:
        """Give a remote speaker a real name (rewrites past and future segments)."""

    def export_session(self) -> Path:
        """Save transcript/notes/answers now; returns the session folder."""

    def new_session(self) -> None:
        """Save and close the current session, then start a fresh one (clears transcript, notes, speakers)."""

    # ---- screen reading ----
    def set_screen_mode(self, mode: str) -> None:
        """off | auto (active window) | monitor | region."""

    def set_screen_region(self, x: int, y: int, w: int, h: int) -> None:
        """Set the region to read (screen pixels) and switch to region mode."""

    def capture_screen_now(self) -> None:
        """Force a capture + OCR right now (publishes ScreenTextEvent even if unchanged)."""

    def suggest_now(self) -> None:
        """Generate suggestions from the current screen text + recent transcript."""

    def solve(self, focus: str = "") -> None:
        """Autonomously solve whatever task/problem is on screen now; `focus` picks a direction
        (e.g. an option chosen from a `[[CHOICE]]` prompt). Blank = solve whatever is shown."""

    def set_paused(self, paused: bool) -> None:
        """Pause/resume all automatic activity: screen reading, auto-answer, auto-suggest/solve,
        and inline completions."""

    # ---- checkpoints / code ----
    def checkpoint(self) -> None:
        """Soft reset of the CODING context only: solve, the Code button and inline completions
        stop seeing the transcript, answers and plan from before this moment (a new coding
        problem starts fresh). The transcript, notes, answer history and the ask box keep the
        whole meeting. Publishes a `CheckpointEvent`."""

    def code(self, focus: str = "") -> None:
        """Write the code for the task discussed (transcript since the last checkpoint) and/or
        shown on screen, right now, in the primary language unless the task calls for another.
        Manual: works while paused. `focus` is an optional extra instruction."""

    def text(self, focus: str = "") -> None:
        """Walk through the task discussed (transcript since the last checkpoint) and/or shown on
        screen in plain English, right now: what is asked, the key insight, the approach step by
        step, the edge cases and the complexity -- the way an engineer talks a problem through
        before coding. Manual: works while paused. `focus` is an optional extra instruction."""

    # ---- pair programming: co-pilot insights / recap (2026-10-20) ----
    def insights_now(self, sections: tuple[str, ...] = ()) -> None:
        """Refresh the co-pilot insights right now: `sections` from ("questions", "blindspots"),
        empty = both. Manual: works while paused and whatever the [assist] toggles say (they only
        govern the background passes). Streams as AssistStarted/Delta/Done of kind "insights",
        then an `InsightsUpdatedEvent`; shown in the Notes tab's Insights pane."""

    def recap(self) -> None:
        """Recap the session so far -- goal, done, now, next -- into the Assist tab. Manual."""

    def set_insight_options(self, *, questions: bool | None = None, blind_spots: bool | None = None) -> None:
        """Background question suggestions / blind-spot flags on or off (None = leave as is):
        `[assist].suggest_questions` / `[assist].flag_blind_spots`."""

    def set_role(self, role: str) -> None:
        """Your role right now -- "driver" | "navigator" | "participant" (`[assist].role`) --
        which steers the suggested questions and blind spots."""

    def set_session_context(self, text: str) -> None:
        """What the session is about -- project, stack, goal (`[assist].session_context`) --
        handed to every answer, insight and recap as standing context. "" clears it."""

    # ---- dashboard mode (2026-10-20) ----
    def set_dashboard_mode(self, enabled: bool) -> None:
        """Dashboard mode (the multi-column view) on / off: `[dashboard].enabled`, persisted with
        the config. Publishes a `DashboardModeEvent`; the tracking itself (`[dashboard].track`)
        keeps running either way."""

    def dashboard_refresh(self, sections: tuple[str, ...] = ()) -> None:
        """Refresh the dashboard now: one model call that updates the tracked questions, blind
        spots & follow-ups and next steps / prompts (`sections` from ("questions", "blindspots",
        "nextsteps"); empty = all). Manual: works while paused. Deferred while the LLM starts.
        Streams as Assist events of kind "dashboard", then a `DashboardUpdatedEvent`."""

    def dashboard_item(self, item_id: int, action: str, text: str = "") -> None:
        """Change one tracked item by hand: `action` "answer" (mark answered / done, `text` = the
        answer, may be empty), "dismiss" (drop it), "reopen", or "edit" (`text` = the new
        wording). Publishes a `DashboardUpdatedEvent`."""

    def dashboard_add(self, kind: str, text: str, parent_id: int | None = None) -> None:
        """Add an item of `kind` ("nextstep" | "goal" | "constraint" | "question" | "blindspot" |
        "prompt") by hand. `parent_id` (an action item) makes a "nextstep" a sub-task and attaches
        a "goal" or "prompt" to it."""

    # ---- completion mode ----
    def set_completion_mode(self, enabled: bool) -> None:
        """Enter/leave completion mode: ghost-text completions at the caret of the focused app;
        automatic answers/solves are suppressed while it is on."""

    def set_completion_context(self, sources: dict[str, bool], hint: str | None = None) -> None:
        """Choose what feeds completions: keys "transcript", "screen", "answers", "notes" (missing
        keys unchanged); `hint` (if not None) is a standing instruction for the completer."""

    # ---- hands-free mode ----
    def set_hands_free(self, enabled: bool) -> None:
        """Hands-free mode on/off: watch the screen samples (`[screen].interval_s`) and, when one
        sample differs massively from the previous one (a new question, a big scroll) and the
        picture settles, automatically Read (accurate OCR) -> Checkpoint -> Text -> Code; regenerate
        Code when the typed code drifts from the app's own. Automatic solves and on-screen question
        answers are suppressed while it is on (the cycle replaces them). Publishes `HandsFreeEvent`s."""

    # ---- manual advance ----
    def set_manual_advance(self, enabled: bool) -> None:
        """Manual advance on/off (the Manual switch next to Hands-free, Ctrl+Alt+J): while on,
        nothing starts a model call by itself -- the transcript, screen scans (OCR), cumulative
        context and auto-checkpoints keep running, and generation happens only on the
        press-and-hold mouse gesture (left: Text, middle: Code) or a manual action. Works with
        hands-free mode on or off. Publishes a `ManualAdvanceEvent`."""

    def manual_advance(self, kind: str) -> None:
        """The press-and-hold gesture fired: generate now. `kind` is "text" (the left button) or
        "code" (the middle button / a quick middle-click). With hands-free on, runs that step for
        the task on screen (its kind's ask, the cumulative context); otherwise a plain
        `text()` / `code()`. Never brings the overlay to the foreground."""

    # ---- explanations ----
    def set_explanations(self, enabled: bool) -> None:
        """Explanations on/off: a walkthrough before every solution, minimal comments in the code,
        and an explanation once the code has been typed out (in completion mode the ghost text
        then follows the plan)."""

    def explain_now(self) -> None:
        """Explain the current solution right now (without waiting for the code to be typed out)."""

    # ---- settings ----
    def apply_settings(self, changed: dict[str, object]) -> None:
        """Values the settings dialog changed, as {"section.key": value}, already written into the
        shared Config: react to the ones that need it (model swap, explanations, auto-answer, ...)
        and persist the config file."""

    def set_auto_suggest(self, enabled: bool) -> None:
        """Enable/disable proactive suggestions when the screen changes."""

    def toggle_screen_paused(self) -> bool:
        """Pause/resume screen reading; returns True when reading is now paused."""

    def set_model(self, model_id: str) -> None:
        """Switch the model used for answers and notes (same as `load_model`)."""

    def list_models(self) -> list[str]:
        """Model keys in the library (may be cached; may be empty if unreachable)."""

    # ---- model manager (2026-09-27, portable-install: the "Models..." window) ----
    # Every method returns at once; results arrive as ModelLibraryEvent / ModelPullEvent /
    # ModelSearchEvent / ModelInfoEvent / ComponentStatusEvent("llm", ...) on the bus.
    def model_library(self) -> list[dict[str, object]]:
        """Every model the app can serve, as plain dicts: key, name, size_bytes, source
        ("library" | "lmstudio" | "ollama"), quant, path, context_length, loaded (bool),
        is_default (bool), has_profile (bool). Synchronous but cheap (a directory scan)."""

    def loaded_model(self) -> str:
        """The key of the model the backend currently serves ("" when nothing is loaded)."""

    def load_model(self, key: str) -> None:
        """Start serving `key` (llamacpp: restart the private llama-server with it; ollama: warm
        it up). Progress and the outcome come as ComponentStatusEvent("llm", ...)."""

    def unload_model(self) -> None:
        """Stop serving the current model and free its memory (llamacpp: stop llama-server;
        ollama: keep_alive 0). Answers are refused with a notice until a model is loaded again."""

    def set_default_model(self, key: str) -> None:
        """Make `key` the model the next launch loads ([llm].model), without loading it now."""

    def delete_model(self, key: str) -> None:
        """Delete a model the app's own library owns (never LM Studio's or Ollama's files); the
        loaded model must be unloaded first. Publishes ModelLibraryEvent (or an "llm" error)."""

    def pull_model(self, repo_id: str, quant: str = "auto", *, include_mmproj: bool = False) -> None:
        """Download a whole model from the Hugging Face Hub into the library: `quant` "auto" =
        the quant that fits this machine, else a quant name; every shard, sha256-verified,
        resumable. One pull at a time; progress as ModelPullEvents, then a ModelLibraryEvent."""

    def cancel_pull(self) -> None:
        """Cancel the running pull (partial files are kept for a later resume)."""

    def search_models(self, query: str, request_id: int) -> None:
        """Search the Hugging Face Hub for GGUF repos; answers with ModelSearchEvent(request_id)."""

    def inspect_model(self, repo_id: str, request_id: int) -> None:
        """List a Hub repo's quant options and the one that fits this machine; answers with
        ModelInfoEvent(request_id)."""

    def hardware_report(self, refresh: bool = False) -> dict[str, object]:
        """The hardware report the run script generated (exocortex.hardware), as a dict;
        `refresh` re-probes now."""

    def model_profile(self, key: str) -> dict[str, object] | None:
        """The per-model tuning profile for `key` (None = the global [llm] values apply)."""

    def set_model_profile(self, key: str, values: dict[str, object] | None) -> None:
        """Save (or with None: delete) the tuning profile for `key`; reloads the model when it is
        the one currently served. Publishes ModelLibraryEvent."""

    def auto_tune_model(self, key: str) -> dict[str, object] | None:
        """Compute (and save) the profile `exocortex.llm.library.auto_profile` recommends for
        `key` on this machine; returns it."""

    def quit(self) -> None:
        """Save the session and shut everything down."""

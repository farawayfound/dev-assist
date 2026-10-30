"""Settings dialog: the options from README's *Configuration* section, editable while running.

One declarative table (:data:`FIELDS`) drives the whole dialog: each :class:`Field` names a
``(section, key)`` of :class:`~exocortex.config.Config`, how to edit it (bool / int / float /
choice / text / model) and whether a restart is needed for it to take effect. The widget
machinery -- :func:`make_field_widget` / :func:`set_field_value` / :func:`get_field_value` and
:class:`FieldForm` (one per tab: builds the widgets, then loads / diffs / applies / restores
them) -- is module-level so :mod:`exocortex.ui.models` can build its own tabs (Server, Tuning) on
the same plumbing without importing this dialog. ``apply()`` writes the changed values into the
shared ``Config`` object (which every component reads live), reports them through ``applied`` so
the engine / overlay can react (``actions.apply_settings``), and the engine persists the file.
Nothing here knows about threads or the model: it is plain Qt.

Opened from the ``⋯`` menu and the slim bar's ``Context ▾`` menu ("Settings…"). Like every other
top-level window of the app it is kept above the overlay by
``OverlayWindow``'s application-wide show filter.

2026-09-27 (portable-install): the old "Model" tab (backend, GGUF/Ollama model, context size,
llama.cpp knobs...) moved out to the "Models…" window (:mod:`exocortex.ui.models`) alongside the
app's own GGUF library, Hugging Face discovery and per-model tuning; ``FIELDS`` no longer has a
"Model" entry and this dialog no longer touches the ``[llm]`` section at all. The ``model`` field
kind and ``_ModelCombo`` stay here (still built by :func:`make_field_widget`) since nothing else
needs a live, editable model picker yet.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFontMetrics
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from exocortex.config import Config
from exocortex.ui.audio_check import AudioCheckPanel
from exocortex.ui.theme import COLORS, build_qss

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Field:
    """One editable config value. ``kind``: bool | int | float | choice | text | model |
    mic_device | system_device."""

    section: str
    key: str
    label: str
    kind: str
    lo: float = 0.0
    hi: float = 0.0
    step: float = 1.0
    choices: tuple[str, ...] = ()
    restart: bool = False          # takes effect on the next launch only
    help: str = ""

    @property
    def path(self) -> str:
        return f"{self.section}.{self.key}"


#: Tab name -> fields, in display order. Mirrors README "Configuration" plus the completion /
#: explanation knobs that matter day to day. The model / server / tuning knobs live in the
#: "Models…" window (exocortex.ui.models) since 2026-09-27, not here.
FIELDS: dict[str, tuple[Field, ...]] = {
    "Completion": (
        Field("completion", "max_lines", "Lines predicted", "int", 1, 40,
              help="Ghost-text lines shown at once. With explanations on, that many lines of the plan's solution."),
        Field("completion", "max_tokens", "Completion length (tokens)", "int", 16, 600, 8,
              help="Model completions only; ~4 tokens per short line. Longer = slower to finish (streams as it goes)."),
        Field("completion", "debounce_ms", "Quiet time before a new suggestion (ms)", "int", 200, 5000, 100),
        Field("completion", "min_interval_ms", "Minimum time between requests (ms)", "int", 200, 5000, 100),
        Field("completion", "font_scale", "Ghost font scale", "float", 0.5, 2.0, 0.05),
        Field("completion", "default_font_pt", "Ghost font when unknown (pt)", "float", 6.0, 30.0, 0.5),
        Field("completion", "ghost_opacity", "Ghost opacity", "float", 0.1, 1.0, 0.05),
        Field("completion", "hint", "Standing hint", "text", help='e.g. "Python 3, terse" or "reply in French"'),
        Field("completion", "enabled", "Start in completion mode", "bool", help="At the next launch."),
    ),
    "Explanations": (
        Field("explain", "enabled", "Explanations on", "bool"),
        Field("explain", "typed_fraction", "Solution lines on screen that count as typed out", "float", 0.5, 1.0, 0.05),
        Field("explain", "walkthrough_max_tokens", "Walkthrough budget (tokens)", "int", 50, 600, 10),
        Field("explain", "explain_max_tokens", "Explanation budget (tokens)", "int", 100, 1200, 20),
    ),
    "Hands-free": (
        Field("handsfree", "min_new_lines", "New lines of text that make a new task", "int", 1, 20,
              help="Lines of prose (3+ words) that were not on screen before -- a new question on a slide, "
                   "however short, even under a static header. A new question line always counts. Lines you "
                   "type, OCR noise and counters never count. 1 = most sensitive."),
        Field("handsfree", "change_threshold", "Or: share of the view that changed at once", "float",
              0.1, 1.0, 0.05,
              help="Share of lines that differ from one screen sample to the next (the capture interval is on "
                   "the Screen tab). 0.6: most of the view changed (a big scroll, a different screen); 0.3 also "
                   "catches a moderate scroll; 0.9: only a completely different screen. Typing never gets there."),
        Field("handsfree", "min_changed_lines", "... and at least this many lines new or gone", "int", 1, 50),
        Field("handsfree", "on_code", "When code is on screen", "choice", choices=("code", "code_first", "ignore"),
              help="A code block, an editor gutter, a development window in front (VS Code, PyCharm, a terminal, "
                   "LeetCode...) or a task that asks for code. code: run Code only, with an ask that also covers "
                   "questions about the code shown. code_first: Code, then Text. ignore: the usual order."),
        Field("handsfree", "settle_s", "Wait for the screen to settle (s)", "float", 0.0, 5.0, 0.25,
              help="No further change for this long before the screen is read and acted on."),
        Field("handsfree", "deviation_threshold", "Regenerate Code when this share of the code on screen is not the app's",
              "float", 0.0, 1.0, 0.05, help="Code only, no checkpoint, so the solution builds on what you typed. 0 = never."),
        Field("handsfree", "context_chars", "Cumulative screen context handed to the model (chars)", "int", 1000, 20000, 500,
              help="What scrolled out of view plus the current screen, for hands-free Text / Code jobs."),
        Field("handsfree", "run_text", "Write the walkthrough (Text) for each new task", "bool",
              help="Skipped by itself while explanations are on (the code opens with a walkthrough) and in completion mode."),
        Field("handsfree", "run_code", "Write the code (Code) for each new task", "bool"),
        Field("handsfree", "code_first", "Code before Text", "bool"),
        Field("handsfree", "accurate_ocr", "Re-read the settled screen with RapidOCR before acting", "bool", restart=True,
              help="The accurate engine on the OCR device from the Screen tab; the cheap Windows OCR keeps sampling."),
        Field("handsfree", "manual", "Manual advance: generate only on press-and-hold", "bool",
              help="Same as the Manual switch next to Hands-free (Ctrl+Alt+J). Watching, screen scans, "
                   "checkpoints, the transcript and question detection keep running, but nothing is "
                   "generated -- no hands-free Text/Code, spoken answers, screen answers, solves or auto "
                   "notes -- until you press and hold the LEFT mouse button (Text) or the MIDDLE button "
                   "(Code), anywhere on screen. The overlay never comes to the front. Works with "
                   "hands-free on or off."),
        Field("handsfree", "manual_hold_s", "Press-and-hold time (s)", "float", 0.2, 3.0, 0.1,
              help="How long a held button must stay down before it generates. Moving the pointer while "
                   "the button is down (a drag, a text selection) always cancels, hold required or not."),
        Field("handsfree", "manual_hold_text", "Press and hold for Text (left button)", "bool",
              help="Checked (default): the left button must be held for the press-and-hold time before "
                   "Text is generated. UNCHECKING THIS MAKES EVERY ORDINARY LEFT CLICK GENERATE -- "
                   "leave it on unless you really mean that."),
        Field("handsfree", "manual_hold_code", "Press and hold for Code (middle button)", "bool",
              help="Unchecked (default): a quick press of the middle button / scroll wheel triggers Code "
                   "right away -- and so does a touchpad three-finger tap once Windows Settings > "
                   "Bluetooth & devices > Touchpad maps it to 'Middle mouse button' (a tap cannot be "
                   "held). Checked: the middle button must be held for the press-and-hold time too."),
        Field("handsfree", "enabled", "Hands-free mode on (also at the next launch)", "bool",
              help="Same as the switch at the right of the tab strip (Ctrl+Alt+H)."),
    ),
    "Assist": (
        Field("assist", "auto_answer_questions", "Auto-answer spoken questions", "bool"),
        Field("assist", "autonomous", "Autonomous: solve what is on screen", "bool"),
        Field("assist", "primary_language", "Primary language", "text",
              help="Solutions, the Code button and completions default to this language when the "
                   "task does not indicate one; a SQL question still gets SQL, code in the visible "
                   "editor keeps its language."),
        Field("assist", "comments", "Comments in generated code", "choice",
              choices=("none", "minimal", "normal", "detailed"),
              help="none: no comments. minimal: 2-3 comments before the important steps. normal: a "
                   "short comment above every logical step plus a one-line summary. detailed: the code "
                   "opens with a requirements + pseudo-code comment block, then a comment above most "
                   "lines (the way an engineer works a challenge through in the editor)."),
        Field("assist", "style", "Answer style", "choice", choices=("bullets", "short", "detailed")),
        Field("assist", "focus", "What is worth answering (screen questions, hands-free tasks)", "choice",
              choices=("auto", "technical", "all"),
              help="technical: only text with technical substance. auto (code-biased): questions and lines that "
                   "read as a task; personal-opinion questions without technical substance are dropped, and a big "
                   "change of view needs such a line to count. all: read everything, answer as much as possible. "
                   "Site polls / surveys / feedback prompts are always dropped."),
        Field("assist", "libraries", "Use libraries in generated code", "choice", choices=("none", "basic", "all"),
              help="A default, not a lock: 'without any libraries' / 'basic functions only' / 'use whatever "
                   "you like' in the conversation or on screen always wins. none: built-in constructs and basic "
                   "built-in functions only, no imports (helpers written out). basic: the standard library's "
                   "commonly known basics (collections, itertools, heapq, math, re; java.util; the STL). all: any "
                   "library or importable function, third-party packages included."),
        Field("assist", "persona", "Persona (about me)", "text",
              help='e.g. "I am a backend engineer interviewing for a platform role."'),
        # ---- pair programming / team coding (2026-10-20): also in the "⋯" > Assist menu ----
        Field("assist", "session_context", "Session context (project, stack, goal)", "text",
              help='e.g. "FastAPI + Postgres billing service; today: per-tenant rate limiting". Handed to '
                   "every answer, insight and recap, so they use the team's real names and constraints."),
        Field("assist", "role", "My role", "choice", choices=("driver", "navigator", "participant"),
              help="driver: you are at the keyboard -- questions to put to the navigator / team, what you "
                   "may be missing while heads-down. navigator: a teammate types and you review -- questions "
                   "for the driver, review points on the code on screen. participant: a design / team "
                   "discussion. Pairs swap often: the Assist menu switches it in one click."),
        Field("assist", "suggest_questions", "Suggest questions to ask (in the background)", "bool",
              help="Questions worth asking right now -- to clarify a requirement, pin down an assumption, "
                   "probe a design choice -- kept as a living list in the Notes tab's Insights pane. "
                   "Never switches tabs; on demand any time (Ctrl+Alt+W, Assist menu, Notes tab)."),
        Field("assist", "flag_blind_spots", "Flag blind spots & follow-ups (in the background)", "bool",
              help="Unhandled edge cases, missing tests, risky assumptions, breaking changes, and the "
                   "follow-ups not to forget -- in the same Insights pane."),
        Field("assist", "insight_interval_s", "Refresh insights at most every (s; 0 = on demand only)", "int",
              0, 1800, 15,
              help="Each background refresh is one local model call (the GPU cost knob). A refresh also "
                   "needs something new: the speech below, or the code on screen changing substantially."),
        Field("assist", "insight_min_new_words", "... once this many new words were said", "int", 0, 1000, 10),
        Field("assist", "insight_max_items", "Items per insight section", "int", 1, 8),
        Field("assist", "coalesce_ms", "Wait for a follow-on sentence (ms)", "int", 0, 3000, 100),
    ),
    "Notes": (
        Field("notes", "template", "Notes style", "choice", choices=("dev", "meeting"),
              help="dev: the same five headings, steered toward a working session -- code changed, bugs "
                   "and their causes, design choices and trade-offs, commands / config, TODOs. "
                   "meeting: the general-meeting wording."),
        Field("notes", "auto_interval_s", "Refresh the notes every (s; 0 = only on demand)", "int", 0, 1800, 30),
        Field("notes", "min_new_words", "... once this many new words were said", "int", 0, 2000, 10),
    ),
    # ---- dashboard mode (2026-10-20): the multi-column board and its background tracking ----
    "Dashboard": (
        Field("dashboard", "enabled", "Dashboard mode on (also at the next launch)", "bool",
              help="The overlay as a board: session notes, action items (tasks with their sub-tasks and the "
                   "prompts to type into Claude Code), goals (acceptance criteria), constraints, questions to "
                   "ask, blind spots & follow-ups, side by side. Same as the ▦ title-bar button (Ctrl+Alt+D). "
                   "While it is on, the running notes keep the narrative only: the tasks, goals, constraints "
                   "and questions live on the board."),
        Field("dashboard", "track", "Track questions, blind spots & next steps in the background", "bool",
              help="A small database (the session folder's dashboard.sqlite) holds the items; each refresh "
                   "hands the model only the open ones plus what is new, and applies the changes it reports. "
                   "While on, these refreshes replace the Notes tab's background insights passes (the "
                   "Insights pane shows the dashboard's open questions and blind spots)."),
        Field("dashboard", "refresh_interval_s", "Refresh at most every (s; 0 = on demand only)", "int", 0, 1800, 15,
              help="Each background refresh is one local model call (the GPU cost knob). It also needs "
                   "something new: the speech below, or the screen changing substantially. Screen capture "
                   "and OCR keep their own cadence (the Screen tab) in every view mode."),
        Field("dashboard", "min_new_words", "... once this many new words were said", "int", 0, 1000, 10),
        Field("dashboard", "max_open_items", "Open items per column handed to the model", "int", 2, 20,
              help="Twice that for action items (a task comes with its sub-tasks) and goals."),
        Field("dashboard", "max_new_items", "New items per column (and per task) per refresh", "int", 1, 8),
        Field("dashboard", "suggest_prompts", "Suggest prompts for Claude Code under the action items", "bool",
              help="Exactly what to type into Claude Code next, one per task that needs one. With a toolkit "
                   "installed in the repository (the Project tab) every prompt starts with one of its slash "
                   "commands or agents, named exactly as installed; a prompt naming something that is not "
                   "installed is thrown away."),
        Field("dashboard", "show_resolved", "Keep answered / done / met items visible, collapsed", "bool"),
        Field("dashboard", "resolved_keep", "... how many per column", "int", 0, 50),
        Field("dashboard", "show_notes", "Column: Session notes", "bool",
              help="The running notes (the Notes tab's cadence: every [notes].auto_interval_s). The ⋯ menu's "
                   "View entry switches every column too."),
        Field("dashboard", "show_nextsteps", "Column: Action items (tasks, sub-tasks, prompts)", "bool"),
        Field("dashboard", "show_goals", "Column: Goals (acceptance criteria)", "bool"),
        Field("dashboard", "show_constraints", "Column: Constraints (rules and guardrails)", "bool"),
        Field("dashboard", "show_questions", "Column: Questions to ask", "bool"),
        Field("dashboard", "show_blindspots", "Column: Blind spots & follow-ups", "bool"),
        Field("dashboard", "show_prompts", "Show the suggested prompts under their action items", "bool"),
        Field("dashboard", "width", "Window width in dashboard mode (px)", "int", 600, 4000, 20,
              help="The normal size is restored when you leave dashboard mode; resizing the board updates this."),
        Field("dashboard", "view_refresh_s", "Re-render the board every (s)", "int", 1, 60,
              help="Timestamps and relative times only; no model call."),
        Field("dashboard", "persist", "Keep the items in the session folder (dashboard.sqlite)", "bool",
              help="Applies to the next session (New session / the next launch); off = in memory only. "
                   "The board is also written as dashboard.md next to the notes either way."),
    ),
    # ---- the project (2026-10-20): what the model may look up beyond transcript + screen ----
    "Project": (
        Field("project", "repo_root", "Repository root (the project being worked on)", "text",
              help="A directory. Searched on demand for the file names and identifiers a prompt mentions; "
                   "never read wholesale. Also where the Claude Code toolkit is looked for (.claude/, "
                   "CLAUDE.local.md). Empty = the auto-detected one below, when that is on."),
        Field("project", "repo_auto", "Auto-detect the repository from Claude Code", "bool",
              help="With no root above: the directory Claude Code was most recently run in, read from its own "
                   "session logs (~/.claude/projects), re-checked every few seconds -- so the toolkit installed "
                   "in the project you are working in is found with no setup. The ⋯ menu's View > Repository "
                   "shows what was found."),
        Field("project", "repo_auto_max_age_h", "... only a session active within (hours; 0 = any)", "int", 0, 720),
        Field("project", "repo_search", "Look up files / identifiers mentioned (answers, dashboard)", "bool",
              help="Only the matching lines go to the model, only when something identifier-like was said, "
                   "typed or shown. Not for inline completions (latency)."),
        Field("project", "repo_max_files", "... files per lookup", "int", 1, 20),
        Field("project", "repo_max_chars", "... characters of matched lines", "int", 200, 8000, 100),
        Field("project", "toolkit", "Claude Code toolkit context (auto-discovered in the repository)", "bool",
              help="The installed skills, agents, rules, workflows and hooks, as a compact catalog built from "
                   "their own files -- nothing hard-coded -- so the model names them exactly and suggests "
                   "the right one. Re-checked every few seconds (below): an install mid-session is picked up."),
        Field("project", "toolkit_lookup", "... hand over the full text of a skill / agent that is mentioned", "bool"),
        Field("project", "toolkit_in_answers", "... also for answers, ask and recap (not only the dashboard)", "bool",
              help="The catalog sits at the end of the system prompt, so it is prefix-cached across turns."),
        Field("project", "toolkit_user_layer", "... also scan ~/.claude (the opt-in user layer)", "bool"),
        Field("project", "toolkit_dir", "Toolkit clone (optional)", "text",
              help="Richer one-line descriptions from its modules/index.json for the parts installed. Empty = "
                   "~/claude-toolkit or ~/.claude/dev-toolkit when present. Never lists parts that are not "
                   "installed."),
        Field("project", "toolkit_max_chars", "Toolkit catalog budget (chars)", "int", 300, 8000, 100),
        Field("project", "rescan_s", "Re-check the repository's .claude/ every (s)", "int", 5, 600, 5),
    ),
    "Screen": (
        Field("screen", "interval_s", "Capture interval (s)", "float", 0.5, 10.0, 0.5),
        Field("screen", "monitor", "Monitor to read in monitor mode (0 = primary)", "int", 0, 8,
              help="The ⋯ menu's View > Screen source lists the monitors by number with their size and position."),
        Field("screen", "auto_suggest", "Solve / suggest when the screen changes", "bool",
              help="Off by default while you drive: your editor changes all the time, and every change "
                   "would start an unrequested solve that takes over the Assist tab. On for an interview / "
                   "challenge screen."),
        Field("screen", "suggest_min_interval_s", "Minimum time between solves (s)", "int", 0, 300, 5),
        Field("screen", "answer_screen_questions", "Answer new questions that appear on screen", "bool",
              help="Off by default while you drive: in an editor or terminal a comment, docstring or log line "
                   "that reads as a question would take over the Assist tab. Spoken questions are unaffected."),
        Field("screen", "exclude_own_windows", "Leave this app's own windows out of the screen text", "bool",
              help="The overlay is an ordinary window screen capture sees; without this the co-pilot reads its "
                   "own answers and suggested questions back as screen content."),
        Field("screen", "max_text_chars", "Screen text handed to the model (chars)", "int", 500, 8000, 250,
              help="~1 s of prompt time per 1000 chars on this machine."),
        Field("screen", "show_text", "Show the OCR readout in the Screen tab", "bool"),
        Field("screen", "ocr_engine", "OCR engine", "choice", choices=("auto", "windows", "rapidocr"), restart=True,
              help="auto = Windows OCR (0.1 s, CPU-light) when available, else RapidOCR (more accurate)."),
        Field("screen", "ocr_device", "OCR device (RapidOCR)", "choice", choices=("auto", "igpu", "npu", "cpu"), restart=True,
              help="Where RapidOCR runs, also for hands-free's accurate re-reads: igpu = the integrated GPU: "
                   "Intel via OpenVINO (0.7 s, ~0 CPU), AMD via OpenVINO or DirectML (the 8060S / 780M iGPUs); "
                   "never the NVIDIA GPU. npu = Intel AI Boost via OpenVINO (1.5 s, ~0 CPU). cpu = OpenVINO CPU "
                   "(1.9 s) or onnxruntime (3.7 s, all cores). auto = igpu, else npu, else cpu."),
    ),
    "Window": (
        Field("ui", "opacity", "Opacity", "float", 0.4, 1.0, 0.05),
        Field("ui", "start_hidden", "Start hidden", "bool"),
        Field("ui", "font_size", "Font size (pt)", "int", 8, 18, restart=True),
    ),
    # ---- 2026-10-21: device pickers (applied at once) + the Check panel below them ----
    "Audio": (
        Field("audio", "mic_device", "Microphone (you)", "mic_device",
              help="Everything this device hears is transcribed as Me and never clustered. default = the "
                   "Windows default input, followed when it changes; or pick a device (part of a name, or a "
                   "device number, works too). Applied at once. Test devices below shows whether it hears you."),
        Field("audio", "system_device", "Output feed (everyone else)", "system_device",
              help="The [Loopback] of the device your speakers or headset play through: what it plays is "
                   "transcribed and clustered into Speaker 1, 2, ... default = the Windows default output, "
                   "followed when it changes. Applied at once."),
        Field("audio", "capture_mic", "Capture the microphone", "bool", restart=True),
        Field("audio", "capture_system", "Capture system audio", "bool", restart=True),
        Field("audio", "mic_start_muted", "Start with the mic muted", "bool"),
        Field("audio", "echo_suppression", "Echo suppression (drop mic text that echoes the other side)", "bool",
              help="For a mic that hears the speakers (laptop speakers, no headset): a line that repeats what "
                   "the other feed said in the last 8 s is dropped. With clean routing (Check speaker routing "
                   "below) there is no echo to catch, and it can only drop real lines -- both sides saying "
                   "\"okay\"."),
        Field("audio", "loopback_keepalive", "Loopback keep-alive tone", "bool", restart=True),
    ),
}


def all_fields() -> list[Field]:
    return [f for group in FIELDS.values() for f in group]


def _defaults() -> Config:
    return Config()


def sections_covered() -> set[str]:
    """Config sections the dialog edits (for tests / docs)."""
    return {f.section for f in all_fields()}


_DIALOG_QSS = f"""
QDialog {{ background: {COLORS['bg_solid']}; }}
QTabWidget::pane {{ border: 1px solid {COLORS['border']}; border-radius: 6px; top: -1px; }}
QTabBar::tab {{ background: transparent; color: {COLORS['text_dim']}; padding: 5px 10px;
                border-bottom: 2px solid transparent; }}
QTabBar::tab:selected {{ color: {COLORS['accent']}; border-bottom: 2px solid {COLORS['accent']}; }}
QSpinBox, QDoubleSpinBox, QComboBox {{ background: {COLORS['panel']}; border: 1px solid {COLORS['border']};
                                        border-radius: 4px; padding: 2px 6px; min-width: 90px; }}
QComboBox QAbstractItemView {{ background: {COLORS['bg_solid']}; color: {COLORS['text']};
                               selection-background-color: {COLORS['accent_dim']}; }}
QCheckBox {{ spacing: 6px; }}
QLabel#restart {{ color: {COLORS['warn']}; }}
QLabel#help {{ color: {COLORS['text_faint']}; }}
"""


#: Room for the list's frame, margins and scroll bar when it is widened to fit the longest id.
_LIST_CHROME_PX = 36


class _ModelCombo(QComboBox):
    """Editable model picker for a ``model``-kind field.

    Three things the plain combo does not do, all because model ids are long
    (``unsloth/NVIDIA-Nemotron-3.5-Lightning-30B-A3B``) and the field is narrow:

    * the library is re-read every time the list is dropped down (``showPopup`` calls the
      ``refresh`` callback handed to the constructor), so a model downloaded elsewhere while
      this widget sits on screen appears on the next click of the arrow instead of only after
      a restart;
    * the open list is widened to the longest id it holds (capped to the screen);
    * the full id is the tool tip - of the field, and of every row of the list - so hovering
      shows the whole name whenever what is painted has been elided.
    """

    def __init__(self, refresh: Callable[[], None], help_text: str = "",
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setEditable(True)
        self._refresh = refresh
        self._help = help_text
        self.currentTextChanged.connect(lambda _text: self.sync_tooltip())
        self.sync_tooltip()

    def set_models(self, models: list[str]) -> None:
        """Replace the list, giving every row its own full-id tool tip."""
        self.clear()
        for model in models:
            self.addItem(model)
            self.setItemData(self.count() - 1, model, Qt.ItemDataRole.ToolTipRole)

    def sync_tooltip(self) -> None:
        """Full id first, then the field's help - the combo paints an elided id, the tip does not."""
        self.setToolTip("\n".join(t for t in (self.currentText().strip(), self._help) if t))

    def showPopup(self) -> None:  # noqa: N802 - Qt override
        self._refresh()
        self._widen_list()
        super().showPopup()

    def _widen_list(self) -> None:
        _widen_popup(self)


def _widen_popup(combo: QComboBox) -> None:
    """Size ``combo``'s drop-down to its longest item, up to what fits on this screen."""
    view = combo.view()
    metrics = QFontMetrics(view.font())
    widest = max((metrics.horizontalAdvance(combo.itemText(i)) for i in range(combo.count())),
                 default=0)
    screen = combo.screen() or QApplication.primaryScreen()
    cap = screen.availableGeometry().width() - 80 if screen is not None else widest
    view.setMinimumWidth(max(combo.width(), min(widest + _LIST_CHROME_PX, cap)))


def _list_audio_devices() -> list[Any]:
    """Every WASAPI device PortAudio knows (imported lazily: pyaudiowpatch is Windows-only)."""
    from exocortex.audio.capture import list_devices

    return list_devices()


class _DeviceCombo(QComboBox):
    """Editable audio device picker for a ``mic_device`` / ``system_device`` field.

    The items are values the config takes: ``default`` (the Windows default, which capture follows
    when it changes), then every WASAPI device of that kind by its full name; part of a name or a
    PortAudio index can be typed too, exactly as in config.toml. Like :class:`_ModelCombo`, the list
    is re-read every time it drops down -- a headset plugged in while the dialog is open shows up on
    the next click -- and widened to the longest name; each row's tool tip has its rate, channels
    and whether Windows uses it by default. Nothing is enumerated until the first drop-down."""

    def __init__(self, kind: str, help_text: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setEditable(True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)   # device names are long
        self.kind = kind             # "mic" | "system"
        self._help = help_text
        self.addItem("default")
        self.currentTextChanged.connect(lambda _text: self.sync_tooltip())
        # a picked name shows from its start (the field is narrower than most names)
        self.currentIndexChanged.connect(lambda _i: self.lineEdit().setCursorPosition(0))
        self.sync_tooltip()

    def refresh(self) -> None:
        """Re-read the devices, keeping whatever is selected or typed."""
        current = self.currentText().strip()
        try:
            devices = _list_audio_devices()
        except Exception:  # noqa: BLE001 - keep the items there are
            log.warning("audio device list unavailable", exc_info=True)
            return
        if self.kind == "system":
            picks = [d for d in devices if d.is_loopback]
        else:
            picks = [d for d in devices if d.is_input and not d.is_loopback]
        default = next((d for d in picks if d.is_default), None)
        self.blockSignals(True)
        self.clear()
        self.addItem("default")
        self.setItemData(0, "The Windows default" + (f": {default.name}" if default else "")
                         + " -- followed when it changes", Qt.ItemDataRole.ToolTipRole)
        for d in picks:
            self.addItem(d.name)
            tip = f"{d.name}\n{d.sample_rate} Hz, {d.channels} ch" + ("\nthe Windows default" if d.is_default else "")
            self.setItemData(self.count() - 1, tip, Qt.ItemDataRole.ToolTipRole)
        idx = self.findText(current)
        if idx >= 0:
            self.setCurrentIndex(idx)
        else:
            self.setEditText(current)
        self.blockSignals(False)
        self.sync_tooltip()

    def sync_tooltip(self) -> None:
        self.setToolTip("\n".join(t for t in (self.currentText().strip(), self._help) if t))

    def showPopup(self) -> None:  # noqa: N802 - Qt override
        self.refresh()
        _widen_popup(self)
        super().showPopup()


# ---------------------------------------------------------------------------------------------
# widget machinery: one function per concern (build / write / read), shared by SettingsDialog's
# tabs and exocortex.ui.models.ModelsDialog's Server + Tuning tabs.
# ---------------------------------------------------------------------------------------------
def make_field_widget(f: Field, parent: QWidget | None = None, *,
                       refresh_models: Callable[[], None] | None = None,
                       list_models: Callable[[], list[str]] | None = None) -> QWidget:
    """Build the one widget ``f`` is edited with. ``refresh_models`` / ``list_models`` matter only
    for ``kind == "model"``: ``list_models`` seeds the initial list (best-effort: a failure here
    leaves it empty rather than raising), ``refresh_models`` is the no-argument callback the
    resulting :class:`_ModelCombo` calls every time its drop-down opens."""
    if f.kind == "bool":
        return QCheckBox(parent)
    if f.kind == "int":
        w = QSpinBox(parent)
        w.setRange(int(f.lo), int(f.hi))
        w.setSingleStep(max(1, int(f.step)))
        return w
    if f.kind == "float":
        w = QDoubleSpinBox(parent)
        w.setRange(float(f.lo), float(f.hi))
        w.setSingleStep(float(f.step))
        w.setDecimals(2)
        return w
    if f.kind == "choice":
        w = QComboBox(parent)
        w.addItems(list(f.choices))
        return w
    if f.kind == "model":
        w = _ModelCombo(refresh_models or (lambda: None), f.help, parent)
        if list_models is not None:
            try:
                w.set_models(list(list_models()))
            except Exception:  # noqa: BLE001
                log.exception("list_models() failed")
        return w
    if f.kind in ("mic_device", "system_device"):
        return _DeviceCombo("mic" if f.kind == "mic_device" else "system", f.help, parent)
    return QLineEdit(parent)


def set_field_value(widget: QWidget, f: Field, value: Any) -> None:
    """Write ``value`` into ``widget`` (the inverse of :func:`get_field_value`)."""
    if isinstance(widget, QCheckBox):
        widget.setChecked(bool(value))
    elif isinstance(widget, QDoubleSpinBox):
        widget.setValue(float(value))
    elif isinstance(widget, QSpinBox):
        widget.setValue(int(value))
    elif isinstance(widget, QComboBox):
        text = str(value)
        idx = widget.findText(text)
        if idx < 0:
            widget.addItem(text)
            idx = widget.findText(text)
        widget.setCurrentIndex(idx)
    elif isinstance(widget, QLineEdit):
        widget.setText(str(value))
    else:  # pragma: no cover - every kind above returns one of these
        raise TypeError(type(widget))


def get_field_value(widget: QWidget, f: Field) -> Any:
    """Read ``widget``'s current value back out, as the type ``f.kind`` implies."""
    if isinstance(widget, QCheckBox):
        return widget.isChecked()
    if isinstance(widget, QDoubleSpinBox):
        return round(widget.value(), 3)
    if isinstance(widget, QSpinBox):
        return int(widget.value())
    if isinstance(widget, QComboBox):
        return widget.currentText().strip()
    if isinstance(widget, QLineEdit):
        return widget.text().strip()
    raise TypeError(type(widget))  # pragma: no cover - every kind above returns one of these


class FieldForm:
    """One tab's worth of :class:`Field` widgets: builds them, then loads / diffs / applies /
    restores them against a :class:`Config`. ``widget`` is the ``QScrollArea`` (with the
    ``QFormLayout`` inside) the fields were laid out in -- everything :class:`SettingsDialog`
    used to build per tab, and what :mod:`exocortex.ui.models`'s Server tab is built from too.
    """

    def __init__(self, fields: tuple[Field, ...], cfg: Config, parent: QWidget | None = None) -> None:
        self.fields = fields
        self._cfg = cfg
        self.widgets: dict[str, QWidget] = {}

        scroll = QScrollArea(parent)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        inner = QWidget(scroll)
        form = QFormLayout(inner)
        form.setContentsMargins(8, 8, 8, 8)
        form.setSpacing(8)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        for f in fields:
            widget = make_field_widget(f, inner)
            self.widgets[f.path] = widget
            label = f.label + (" (restart)" if f.restart else "")
            if f.help and not widget.toolTip():
                widget.setToolTip(f.help)
            form.addRow(label, widget)
        scroll.setWidget(inner)
        self.widget = scroll
        self._layout = form

    def add_row(self, widget: QWidget) -> None:
        """Append a full-width widget below the fields (the Audio tab's Check panel)."""
        self._layout.addRow(widget)

    def load(self) -> None:
        """Fill every widget from the current config."""
        for f in self.fields:
            set_field_value(self.widgets[f.path], f, getattr(getattr(self._cfg, f.section), f.key))

    def changes(self) -> dict[str, Any]:
        """``{"section.key": new_value}`` for every widget whose value differs from the config."""
        out: dict[str, Any] = {}
        for f in self.fields:
            new = get_field_value(self.widgets[f.path], f)
            cur = getattr(getattr(self._cfg, f.section), f.key)
            if isinstance(cur, float):
                if abs(float(new) - cur) > 1e-9:
                    out[f.path] = float(new)
            elif new != cur:
                out[f.path] = new
        return out

    def apply(self) -> dict[str, Any]:
        """Write the changed values into the config; returns the changes."""
        changed = self.changes()
        for f in self.fields:
            if f.path in changed:
                setattr(getattr(self._cfg, f.section), f.key, changed[f.path])
        return changed

    def restore_defaults(self, skip: set[str] = frozenset()) -> None:
        """Fill every widget (but those in ``skip``) with the built-in defaults; nothing is
        written into the config until :meth:`apply`."""
        defaults = _defaults()
        for f in self.fields:
            if f.path in skip:
                continue
            set_field_value(self.widgets[f.path], f, getattr(getattr(defaults, f.section), f.key))


class SettingsDialog(QDialog):
    """Tabbed editor for :data:`FIELDS`, one :class:`FieldForm` per tab. ``applied`` carries
    ``{"section.key": new_value}`` for every value that changed on Apply / OK (empty dict when
    nothing changed)."""

    applied = Signal(dict)

    def __init__(self, cfg: Config, list_models: Callable[[], list[str]] | None = None,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._cfg = cfg
        self._list_models = list_models          # kept for callers built against the old surface;
        self._forms: dict[str, FieldForm] = {}    # no FIELDS entry is kind "model" any more.
        self.setWindowTitle("Settings")
        self.setWindowFlags(Qt.WindowType.Dialog | Qt.WindowType.Tool | Qt.WindowType.WindowStaysOnTopHint)
        self.setStyleSheet(build_qss(cfg.ui.font_size) + _DIALOG_QSS)
        self.setMinimumSize(520, 420)
        self.resize(560, 520)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(8)
        self.tabs = QTabWidget(self)
        outer.addWidget(self.tabs, 1)
        self.audio_check: AudioCheckPanel | None = None
        for name, group in FIELDS.items():
            form = FieldForm(group, cfg, self.tabs)
            self._forms[name] = form
            self.tabs.addTab(form.widget, name)
            if name == "Audio":
                self.audio_check = AudioCheckPanel(
                    form.widgets["audio.mic_device"], form.widgets["audio.system_device"],
                    form.widgets["audio.echo_suppression"], form.widget,
                )
                form.add_row(self.audio_check)

        self.note = QLabel("", self)
        self.note.setObjectName("restart")
        self.note.setWordWrap(True)
        outer.addWidget(self.note)

        row = QHBoxLayout()
        self.defaults_btn = QPushButton("Restore defaults", self)
        self.defaults_btn.clicked.connect(self.restore_defaults)
        self.apply_btn = QPushButton("Apply", self)
        self.apply_btn.clicked.connect(self.apply)
        self.ok_btn = QPushButton("OK", self)
        self.ok_btn.setObjectName("primary")
        self.ok_btn.clicked.connect(self._on_ok)
        self.cancel_btn = QPushButton("Cancel", self)
        self.cancel_btn.clicked.connect(self.reject)
        row.addWidget(self.defaults_btn)
        row.addStretch(1)
        for b in (self.apply_btn, self.ok_btn, self.cancel_btn):
            row.addWidget(b)
        outer.addLayout(row)
        self.load()

    # ---- merged view over every tab's FieldForm ------------------------------------------------
    @property
    def _widgets(self) -> dict[str, QWidget]:
        """``{"section.key": widget}`` across every tab (tests, and ``_on_settings_applied``-style
        callers, key off this exactly as they did when one dialog owned every widget)."""
        merged: dict[str, QWidget] = {}
        for form in self._forms.values():
            merged.update(form.widgets)
        return merged

    # ---- values --------------------------------------------------------------------------------
    def load(self) -> None:
        """Fill every widget from the current config."""
        for form in self._forms.values():
            form.load()
        self.note.setText("")

    def show_tab(self, name: str) -> bool:
        """Bring the tab called `name` to the front (the Assist menu's "Assist settings…"); False
        when there is no such tab."""
        for index in range(self.tabs.count()):
            if self.tabs.tabText(index) == name:
                self.tabs.setCurrentIndex(index)
                return True
        return False

    def refresh_models(self) -> None:
        """No-op: no ``FIELDS`` entry is kind "model" any more (that field now lives on the
        Models window's Server tab) -- kept only so callers built against the old surface do not
        break."""
        return

    def restore_defaults(self) -> None:
        """Fill every widget with the built-in defaults (nothing is written until Apply / OK)."""
        for form in self._forms.values():
            form.restore_defaults()

    def changes(self) -> dict[str, Any]:
        """``{"section.key": new_value}`` for every widget whose value differs from the config."""
        out: dict[str, Any] = {}
        for form in self._forms.values():
            out.update(form.changes())
        return out

    def apply(self) -> dict[str, Any]:
        """Write the changed values into the config and announce them; returns the changes."""
        changed: dict[str, Any] = {}
        for form in self._forms.values():
            changed.update(form.apply())
        by_path = {f.path: f for f in all_fields()}
        restart = [by_path[path].label for path in changed if by_path[path].restart]
        if restart:
            self.note.setText("Takes effect at the next launch: " + ", ".join(restart))
        elif changed:
            self.note.setText("Applied.")
        else:
            self.note.setText("")
        if changed:
            log.info("settings changed: %s", ", ".join(f"{k}={v!r}" for k, v in changed.items()))
            self.applied.emit(changed)
        return changed

    def _on_ok(self) -> None:
        self.apply()
        self.accept()

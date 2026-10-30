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

2026-10-07 (Optimize tab): ``FIELDS["Optimize"]`` is a normal tab (``[optimize]`` goal / limits and the
``[asr]`` engine knobs), but its PAGE is built by :class:`exocortex.ui.optimize_panel.OptimizePanel`, which
re-parents that form's widgets and adds the benchmark / plan / apply workflow on top. The dialog therefore
gained a few additive hooks and nothing else changed: ``actions=`` (the engine, found WITHOUT touching
``overlay.py`` -- see :func:`_resolve_actions`), :meth:`SettingsDialog.apply_paths` (apply only the values a
plan wrote, leaving unrelated pending edits on other tabs alone), :meth:`SettingsDialog.set_values` /
:meth:`SettingsDialog.widget_values` (fill / read widgets by path across tabs) and ``FieldForm.apply(only)``.
``apply()`` keeps its no-argument signature on purpose: it is connected straight to ``clicked(bool)`` and
PySide would otherwise feed the ``checked`` flag into an optional parameter.

2026-10-29 (Audio tab): a seventh field kind, ``device`` (:class:`_DeviceCombo`: an editable picker whose value
is a device *spec* as ``exocortex.audio.capture.resolve_device`` reads one -- "default", a name, an index), for
``[audio].system_device`` / ``mic_device``; ``FIELDS["Audio"]`` grew the device pickers and the voice-detection
knobs; and, like Optimize, the Audio tab's PAGE is built by :class:`exocortex.ui.audio_panel.AudioPanel`
(device list from a background scan, a Test button and level meter per source, what the running app has open)
around the ordinary ``FieldForm``. The page builders are listed in :data:`SettingsDialog._PAGE_BUILDERS`; a
panel that fails to build falls back to the plain form, so the tab always exists.

2026-10-29 (Shortcuts tab): ``FIELDS["Shortcuts"]`` lists every ``[hotkeys]`` entry as a ``binding`` field
(:class:`_BindingEdit`: the binding in words, **Set** -- listen for the next key(s) or mouse button -- and
**Default**), and its page is :class:`ShortcutsPage`, registered in ``SettingsDialog._PAGE_BUILDERS`` next to
Optimize. The grammar is :mod:`exocortex.bindings`; a row the grammar rejects is flagged and skipped by
``_apply()``. The overlay re-registers its shortcuts when any ``hotkeys.*`` value is applied. The Assist tab
gained **Consider given code** (``[assist].consider_given_code``) and the Hands-free tab the hold switch of
the third manual gesture (``manual_hold_code_fresh``).
"""
from __future__ import annotations

import logging
import sys
from collections.abc import Callable, Collection
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from PySide6.QtCore import QEvent, QObject, QPoint, Qt, QTimer, Signal
from PySide6.QtGui import QFontMetrics, QGuiApplication, QKeyEvent
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from exocortex.bindings import describe_binding, key_token_for_vk, normalize_binding
from exocortex.config import Config
from exocortex.ui.theme import COLORS, build_qss

if TYPE_CHECKING:  # the UI never imports the capture module at run time; the panel does
    from exocortex.audio.capture import DeviceInfo

try:
    from exocortex.ui.llhooks import CaptureHooks, ChordRecorder
except Exception:  # noqa: BLE001 - pure ctypes, but the dialog must open even if it cannot load
    CaptureHooks = None  # type: ignore[assignment,misc]
    ChordRecorder = None  # type: ignore[assignment,misc]

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Field:
    """One editable config value. ``kind``: bool | int | float | choice | text | model | device | binding
    (2026-10-29: ``device`` is an audio device picker, :class:`_DeviceCombo`; ``binding`` a ``[hotkeys]``
    shortcut, edited with :class:`_BindingEdit`)."""

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
        Field("handsfree", "manual_hold_code_fresh", "Press and hold for Code from scratch (mouse 4)", "bool",
              help="Unchecked (default): a click of the third gesture's button (mouse button 4 by default; "
                   "Shortcuts tab) writes the code from scratch, ignoring the code on screen. Checked: it "
                   "must be held for the press-and-hold time."),
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
        Field("assist", "consider_given_code", "Consider given code", "bool",
              help="On (default): solve, Code and Text work with the code already written in the development "
                   "window -- they complete it, or fix only what is wrong, keeping its structure, names and "
                   "approach -- instead of jumping to the optimal solution from scratch; a platform's starter "
                   "code is filled in. Off: the best solution from scratch, keeping only the entry point the "
                   "task dictates. The manual-advance 'Code from scratch' gesture (Shortcuts tab) always "
                   "ignores the given code."),
        Field("assist", "code_only", "Code answers as plain code only", "bool",
              help="On (default): a Code answer is one code block and nothing else -- a note or the complexity "
                   "goes inside it as comment lines in the language's own syntax, and markdown the model slips "
                   "in (a heading, bold, bullets) is turned into comments. With explanations on, the walkthrough "
                   "stays and the rule applies to the code block. Off: the answer as the model wrote it."),
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
        Field("screen", "read_on_shortcut", "Read the screen when Code / Text is pressed", "bool",
              help="On (default): the Code and Text shortcuts, buttons and gestures read the screen at that moment "
                   "and work on the file you are in, instead of the last pass (up to the capture interval old). "
                   "Waits at most the read timeout for the OCR."),
        Field("screen", "read_timeout_s", "Read timeout (s)", "float", 0.5, 15.0, 0.5),
        Field("screen", "workspace", "Workspace: hand the other files of the task to Code / Text", "bool",
              help="On (default): every file or pane seen since the checkpoint -- the task statement, a parent "
                   "class, an interface, a test -- is kept as the task's workspace and handed to Code / Text "
                   "next to the screen, so the answer is written for the file on screen in the context of the "
                   "others. A checkpoint starts a new workspace."),
        Field("screen", "workspace_chars", "Workspace text handed to the model (chars)", "int", 500, 12000, 250),
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
    "Monitor": (
        Field("monitor", "show_performance", "LLM performance (speed, tokens/second)", "bool"),
        Field("monitor", "show_context", "Context window and its contents", "bool"),
        Field("monitor", "show_cpu", "CPU utilisation", "bool"),
        Field("monitor", "show_ram", "System RAM utilisation", "bool"),
        Field("monitor", "show_gpu", "GPU utilisation and VRAM (dedicated GPU)", "bool"),
        Field("monitor", "show_igpu", "Integrated GPU (iGPU) utilisation", "bool",
              help="Hidden automatically when the machine has no integrated GPU."),
        Field("monitor", "show_npu", "NPU utilisation", "bool",
              help="The AI accelerator (e.g. Intel AI Boost). Hidden automatically when absent."),
    ),
    "Window": (
        Field("ui", "opacity", "Opacity", "float", 0.4, 1.0, 0.05),
        Field("ui", "start_hidden", "Start hidden", "bool"),
        Field("ui", "font_size", "Font size (pt)", "int", 8, 18, restart=True),
    ),
    # 2026-10-29: the page is built by exocortex.ui.audio_check.CheckedAudioPanel (exo-cortex's
    # AudioPanel -- device list, a Test per source, level meters -- plus this fork's Windows-mute and
    # speaker-routing checks) around this form. The on/off switches, the devices and the keep-alive tone
    # apply live (Engine.apply_settings -> SpeechPipeline.restart_source); the voice-detection knobs feed
    # a segmenter that is built once per source, so they stay restart-only.
    "Audio": (
        Field("audio", "capture_system", "Capture system audio", "bool",
              help="The other side of the call: a WASAPI loopback of an output device. Applies at once."),
        Field("audio", "system_device", "System audio device", "device",
              help="The [Loopback] of the device your speakers or headset play through: what it plays is "
                   "transcribed and clustered into Speaker 1, 2, ... Default = the loopback of the Windows "
                   "default output device, followed when it changes. Pick a device from the list (stored by "
                   "name), or type part of a name or a device index. Test opens it for a moment and reports "
                   "what arrives. Applies at once."),
        Field("audio", "loopback_keepalive", "Loopback keep-alive tone", "bool",
              help="A -80 dBFS tone on the default output keeps WASAPI loopback delivering audio while nothing "
                   "plays (Windows idles the endpoint otherwise). Applies at once."),
        Field("audio", "capture_mic", "Capture the microphone", "bool", help="You. Applies at once."),
        Field("audio", "mic_device", "Microphone", "device",
              help="Everything this device hears is transcribed as Me and never clustered. Default = the Windows "
                   "default input device, followed when it changes. Pick a device from the list (stored by "
                   "name), or type part of a name or a device index. Applies at once."),
        Field("audio", "mic_start_muted", "Start with the mic muted", "bool"),
        Field("audio", "echo_suppression", "Echo suppression (drop mic text that echoes the other side)", "bool",
              help="For a mic that hears the speakers (laptop speakers, no headset): a line that repeats what "
                   "the other feed said in the last 8 s is dropped. With clean routing (Check speaker routing "
                   "below) there is no echo to catch, and it can only drop real lines -- both sides saying "
                   "\"okay\"."),
        Field("audio", "vad_threshold", "Speech probability that counts as speech", "float", 0.1, 0.95, 0.05,
              restart=True, help="Silero VAD threshold. Lower catches quieter speech and more noise; higher misses "
                                 "soft starts."),
        Field("audio", "min_silence_s", "Silence that ends an utterance (s)", "float", 0.1, 2.0, 0.05, restart=True,
              help="Shorter gives finals sooner; 0.45 merges closely spoken sentences."),
        Field("audio", "min_speech_s", "Shortest speech kept (s)", "float", 0.05, 2.0, 0.05, restart=True),
        Field("audio", "max_utterance_s", "Longest utterance before a forced split (s)", "float", 5.0, 60.0, 1.0,
              restart=True),
        Field("audio", "pre_roll_s", "Audio kept from before speech starts (s)", "float", 0.0, 1.0, 0.05,
              restart=True),
    ),
    # 2026-10-29 (Shortcuts tab): every [hotkeys] entry, in the order the page lists them -- the global
    # shortcuts, then the manual-advance gestures -- as "binding" fields (one grammar for key combinations,
    # single-key taps and mouse buttons: exocortex.bindings), plus the thumb-button swallow switch. The page
    # is built by ShortcutsPage (see _PAGE_BUILDERS); tests pin that every HotkeysConfig field is here.
    "Shortcuts": (
        Field("hotkeys", "toggle_visible", "Show / hide the overlay", "binding"),
        Field("hotkeys", "ask_focus", "Show the overlay and focus the ask box", "binding"),
        Field("hotkeys", "answer_last", "Answer the last thing the other side said", "binding"),
        Field("hotkeys", "toggle_auto", "Auto-answer on / off", "binding"),
        Field("hotkeys", "suggest_now", "Suggestions from the screen and the conversation, now", "binding"),
        Field("hotkeys", "notes", "Regenerate the notes", "binding"),
        Field("hotkeys", "insights", "Suggested questions and blind spots / follow-ups, now", "binding"),
        Field("hotkeys", "recap", "Recap the session so far", "binding"),
        Field("hotkeys", "code", "Code: write the code for the task", "binding"),
        Field("hotkeys", "checkpoint", "Checkpoint: the coding context starts fresh", "binding"),
        Field("hotkeys", "toggle_explanations", "Explanations on / off", "binding"),
        Field("hotkeys", "toggle_completion", "Completion mode on / off", "binding"),
        Field("hotkeys", "toggle_handsfree", "Hands-free mode on / off", "binding"),
        Field("hotkeys", "toggle_manual", "Manual advance on / off", "binding"),
        Field("hotkeys", "toggle_dashboard", "Dashboard mode on / off", "binding"),
        Field("hotkeys", "select_region", "Drag a screen region to read", "binding"),
        Field("hotkeys", "toggle_screen", "Pause / resume screen reading", "binding"),
        Field("hotkeys", "toggle_mic", "Mute / unmute the mic", "binding"),
        Field("hotkeys", "toggle_pause", "Pause / resume everything automatic", "binding",
              help="A tap of a single key by default (Right Shift): pressed and released with no other key "
                   "in between, so typing capitals never trips it."),
        Field("hotkeys", "click_through", "Click-through on / off", "binding"),
        Field("hotkeys", "move_left", "Move the overlay left", "binding"),
        Field("hotkeys", "move_right", "Move the overlay right", "binding"),
        Field("hotkeys", "move_up", "Move the overlay up", "binding"),
        Field("hotkeys", "move_down", "Move the overlay down", "binding"),
        Field("hotkeys", "quit", "Quit (saves the session)", "binding"),
        Field("hotkeys", "manual_text", "Manual advance: Text", "binding",
              help="The left button, held, by default (the Hands-free tab decides which gestures need the hold)."),
        Field("hotkeys", "manual_code", "Manual advance: Code", "binding",
              help="A middle click by default. Works with the given code when 'Consider given code' (Assist "
                   "tab) is on."),
        Field("hotkeys", "manual_code_fresh", "Manual advance: Code from scratch", "binding",
              help="Mouse button 4 (the rear thumb button) by default. Always ignores the code on screen: "
                   "straight to the best solution."),
        Field("hotkeys", "swallow_thumb_buttons", "Keep thumb-button presses (mouse 4 / 5) from the app underneath",
              "bool",
              help="A bound thumb button would otherwise make a browser or an editor go Back / Forward on the "
                   "very press that asks for code. Modified clicks (Ctrl+left) are always kept, bare left / "
                   "middle / right presses never."),
    ),
    # 2026-10-07 (Optimize tab): appended LAST on purpose ("Completion" must stay first). This tab's page is built
    # by exocortex.ui.optimize_panel.OptimizePanel, which re-parents these widgets into its own layout; the
    # FieldForm underneath stays the source of truth for load / changes / apply / restore_defaults, so every
    # invariant tests/test_settings.py pins for the other tabs holds here too. Nothing under [llm] is edited.
    "Optimize": (
        Field("optimize", "goal", "Optimise for", "choice",
              choices=("latency", "balanced", "context", "quality"),
              help="latency: fastest answers and 3 GB of GPU memory kept free. balanced: today's behaviour plus the "
                   "safe fixes. context: every spare GB becomes conversation window. quality: 16-bit KV cache where "
                   ">= 32k fits."),
        Field("optimize", "reserve_vram_mb", "GPU memory kept free (MB, 0 = automatic)", "int", 0, 12000, 256,
              help="Never planned into: the desktop shares this GPU. 0 = the VRAM-guard floor + 256 plus a per-goal "
                   "allowance. Any value is raised to the guard floor + 256 if lower."),
        Field("optimize", "ctx_cap", "Longest window to plan (tokens, 0 = the model's maximum)", "int", 0, 262144, 2048),
        Field("optimize", "asr_gpu", "Speech recognition on the display GPU", "choice", choices=("auto", "never"),
              help="never = Whisper is not planned onto the GPU (the 2026-09-25 rule). auto = only when Parakeet "
                   "cannot be used (hotwords, an unsupported language) or measurably loses on accuracy."),
        Field("asr", "engine", "Speech engine", "choice", choices=("auto", "faster-whisper", "sherpa"), restart=True,
              help="auto = Whisper whenever CUDA exists, EVEN when the device below is cpu. Measured on this machine: "
                   "Whisper on the CPU 6.2 s per utterance, Parakeet (sherpa) on the CPU 0.15 s, Whisper on CUDA "
                   "0.11 s but it holds 0.9 GB of GPU memory. sherpa = Parakeet-TDT on the CPU: no GPU memory, "
                   "English or 25 European languages, no hotwords."),
        Field("asr", "device", "Whisper device", "choice", choices=("auto", "cuda", "cpu"), restart=True,
              help="Where Whisper runs. cpu keeps ~1-2.5 GB off the display GPU but costs seconds per utterance."),
        Field("asr", "compute_type", "Whisper precision", "choice",
              choices=("auto", "float16", "int8_float16", "int8"), restart=True,
              help="int8_float16 on CUDA holds about 0.9 GB, float16 about 2.4 GB; int8 is the CPU precision."),
        Field("asr", "cpu_threads", "Speech CPU threads (0 = automatic)", "int", 0, 32, 1, restart=True),
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
QLabel#section {{ color: {COLORS['accent']}; font-weight: 600; margin-top: 6px; }}
QLabel#fixed {{ color: {COLORS['text_dim']}; }}
QLabel#binding[state="invalid"] {{ color: {COLORS['err']}; }}
QLabel#binding[state="listening"] {{ color: {COLORS['accent']}; }}
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


class _DeviceCombo(QComboBox):
    """Editable device picker for a ``device``-kind field (``[audio].system_device`` / ``mic_device``).

    Its value is a *spec* exactly as :func:`exocortex.audio.capture.resolve_device` reads one:
    ``"default"``, a device name (matched as a case-insensitive substring) or a PortAudio index.
    The rows are "Default" (with the device it resolves to appended once a scan has said) plus
    every candidate device by its full name; a row's spec is its item data, so what is painted may
    say more than what is stored. Text typed that matches no row is stored as typed (a substring,
    an index), and a config value that matches no row is shown as typed rather than altered, so
    loading never produces a phantom change. :class:`exocortex.ui.audio_panel.AudioPanel` fills the
    rows from a background scan; the plain-form fallback leaves "Default" and the ability to type.
    No completer: what is typed is what is stored.
    """

    DEFAULT = "default"

    def __init__(self, help_text: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setEditable(True)
        self.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.setCompleter(None)  # type: ignore[arg-type]
        self.setMinimumContentsLength(18)
        if help_text:
            self.setToolTip(help_text)
        self.set_devices([], None)

    def set_devices(self, devices: "list[DeviceInfo]", default_name: str | None) -> None:
        """Replace the rows, keeping the current spec (chosen or typed). Emits no signals."""
        spec = self.spec()
        self.blockSignals(True)
        try:
            self.clear()
            self.addItem("Default" + (f"  ({default_name})" if default_name else ""), self.DEFAULT)
            self.setItemData(0, "Follows the Windows default device", Qt.ItemDataRole.ToolTipRole)
            for d in devices:
                self.addItem(d.name, d.name)
                tip = f"index {d.index} · {d.sample_rate} Hz · {d.channels} ch" + (" · Windows default" if d.is_default else "")
                self.setItemData(self.count() - 1, tip, Qt.ItemDataRole.ToolTipRole)
            self.set_spec(spec)
        finally:
            self.blockSignals(False)

    def spec(self) -> str:
        """The stored value: the chosen row's spec, else the typed text ("" and "Default" -> "default")."""
        text = self.currentText().strip()
        idx = self.currentIndex()
        if idx >= 0 and text == self.itemText(idx).strip():
            return str(self.itemData(idx) or self.DEFAULT)
        if not text or text.lower() == self.DEFAULT:
            return self.DEFAULT
        return text

    def set_spec(self, spec: object) -> None:
        """Select the row whose spec this is, or show it as typed text."""
        text = str(spec if spec is not None else "").strip() or self.DEFAULT
        if text.lower() == self.DEFAULT:
            text = self.DEFAULT
        for i in range(self.count()):
            if str(self.itemData(i)) == text:
                self.setCurrentIndex(i)
                return
        self.setCurrentIndex(-1)
        self.setEditText(text)


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
    if f.kind == "device":
        return _DeviceCombo(f.help, parent)
    if f.kind == "model":
        w = _ModelCombo(refresh_models or (lambda: None), f.help, parent)
        if list_models is not None:
            try:
                w.set_models(list(list_models()))
            except Exception:  # noqa: BLE001
                log.exception("list_models() failed")
        return w
    if f.kind == "binding":
        # 2026-10-29: the row's Default button needs the built-in value of this very field
        return _BindingEdit(str(getattr(getattr(_defaults(), f.section), f.key)), parent)
    return QLineEdit(parent)


def set_field_value(widget: QWidget, f: Field, value: Any) -> None:
    """Write ``value`` into ``widget`` (the inverse of :func:`get_field_value`)."""
    if isinstance(widget, _DeviceCombo):
        widget.set_spec(value)
    elif isinstance(widget, QCheckBox):
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
    elif isinstance(widget, _BindingEdit):
        widget.set_value(str(value))
    else:  # pragma: no cover - every kind above returns one of these
        raise TypeError(type(widget))


def get_field_value(widget: QWidget, f: Field) -> Any:
    """Read ``widget``'s current value back out, as the type ``f.kind`` implies."""
    if isinstance(widget, _DeviceCombo):
        return widget.spec()
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
    if isinstance(widget, _BindingEdit):
        return widget.value()
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

    def apply(self, only: Collection[str] | None = None) -> dict[str, Any]:
        """Write the changed values into the config; returns the changes. ``only`` (2026-10-07,
        Optimize tab) limits the write to those ``"section.key"`` paths: everything else stays a
        pending edit in its widget. Nothing connects this method to a Qt signal (it is called, never
        wired), so the optional parameter cannot receive a ``checked`` flag."""
        changed = {k: v for k, v in self.changes().items() if only is None or k in only}
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



# ---------------------------------------------------------------------------------------------
# Shortcuts tab (2026-10-29): every [hotkeys] binding in one list -- the current binding in words,
# a Set button that listens for the next key(s) / mouse button, a Default button per row -- plus
# the fixed in-overlay keys, so the tab shows every shortcut the app has.
# ---------------------------------------------------------------------------------------------
#: How long the Set button waits after the last press before taking the combination.
CAPTURE_QUIET_S = 0.5

#: (where, key or gesture, action): the shortcuts that are NOT rebindable, listed for completeness.
#: Mirrors docs/SHORTCUTS.md "Inside the overlay".
FIXED_SHORTCUTS: tuple[tuple[str, str, str], ...] = (
    ("Ask box", "Enter", "Send the question"),
    ("Ask box", "Esc", "Clear the box"),
    ("Region picker", "Drag", "Select the region"),
    ("Region picker", "Enter or double-click", "Accept the region"),
    ("Region picker", "Esc or right-click", "Cancel"),
    ("Live tab", "Double-click a speaker name", "Rename the speaker"),
    ("Title bar or slim bar", "Drag", "Move the overlay"),
    ("Window edges, bottom-right corner", "Drag", "Resize the overlay"),
    ("Panel, except over text", "Right-click", "The \u22ef menu"),
    ("Text panes and the ask box", "Right-click", "The text menu, Hide overlay, Quit"),
    ("Hands-free view, over the answer or the history", "Mouse wheel", "Scroll it (the view stays click-through)"),
    ("Hands-free view, over the expanded history", "Tap Up / Down", "Previous / next answer"),
    ("Shortcuts tab, while Set listens", "Esc", "Cancel"),
)


def _real_windows() -> bool:
    """True only on real Windows with the native "windows" Qt platform (never under the offscreen
    QPA the tests use): the one case the capture hooks may touch the real keyboard and mouse."""
    return sys.platform == "win32" and QGuiApplication.platformName() == "windows"


def _restyle(widget: QWidget, name: str) -> None:
    """Switch a widget's object name (its QSS role) and repolish it so the new style shows."""
    if widget.objectName() != name:
        widget.setObjectName(name)
        widget.style().unpolish(widget)
        widget.style().polish(widget)


_QT_KEY_TOKENS: dict[int, str] = {
    int(Qt.Key.Key_Space): "space", int(Qt.Key.Key_Left): "left", int(Qt.Key.Key_Up): "up",
    int(Qt.Key.Key_Right): "right", int(Qt.Key.Key_Down): "down", int(Qt.Key.Key_Return): "enter",
    int(Qt.Key.Key_Enter): "enter", int(Qt.Key.Key_Escape): "esc", int(Qt.Key.Key_Tab): "tab",
    int(Qt.Key.Key_Backspace): "backspace", int(Qt.Key.Key_Insert): "insert", int(Qt.Key.Key_Delete): "delete",
    int(Qt.Key.Key_Home): "home", int(Qt.Key.Key_End): "end", int(Qt.Key.Key_PageUp): "pageup",
    int(Qt.Key.Key_PageDown): "pagedown", int(Qt.Key.Key_QuoteLeft): "grave", int(Qt.Key.Key_Minus): "-",
    int(Qt.Key.Key_Equal): "=", int(Qt.Key.Key_BracketLeft): "[", int(Qt.Key.Key_BracketRight): "]",
    int(Qt.Key.Key_Semicolon): ";", int(Qt.Key.Key_Apostrophe): "'", int(Qt.Key.Key_Comma): ",",
    int(Qt.Key.Key_Period): ".", int(Qt.Key.Key_Slash): "/", int(Qt.Key.Key_Backslash): "backslash",
    int(Qt.Key.Key_Shift): "shift", int(Qt.Key.Key_Control): "ctrl", int(Qt.Key.Key_Alt): "alt",
    int(Qt.Key.Key_Meta): "win",
}
_KEY_A, _KEY_Z, _KEY_0, _KEY_9 = int(Qt.Key.Key_A), int(Qt.Key.Key_Z), int(Qt.Key.Key_0), int(Qt.Key.Key_9)
_KEY_F1, _KEY_F12 = int(Qt.Key.Key_F1), int(Qt.Key.Key_F12)


def _qt_key_token(event: QKeyEvent) -> str | None:
    """The binding token of a Qt key event (the capture's fallback path, when the low-level hooks
    are unavailable): the native virtual key when Qt reports one (side-specific modifiers), else
    the Qt key code. ``None`` for a key the grammar has no name for."""
    try:
        vk = int(event.nativeVirtualKey() or 0)
    except Exception:  # noqa: BLE001
        vk = 0
    token = key_token_for_vk(vk) if vk else None
    if token:
        return token
    key = int(event.key())
    if key in _QT_KEY_TOKENS:
        return _QT_KEY_TOKENS[key]
    if _KEY_A <= key <= _KEY_Z or _KEY_0 <= key <= _KEY_9:
        return chr(key).lower()
    if _KEY_F1 <= key <= _KEY_F12:
        return f"f{key - _KEY_F1 + 1}"
    return None


class _BindingEdit(QWidget):
    """The Shortcuts tab's editor for ONE binding (kind ``"binding"``): the current binding in
    words, **Set** -- listen for the next key combination or mouse button -- and **Default**.
    ``value()`` / ``set_value()`` carry the canonical binding text (``""`` = unbound); a text the
    grammar rejects is kept as typed and flagged, and the dialog refuses to apply it."""

    valueChanged = Signal(str)
    setRequested = Signal(object)        # this row: start listening
    cancelRequested = Signal(object)     # this row: stop listening

    def __init__(self, default: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._default = default
        self._value = ""
        self._valid = True
        self._listening = False
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        self.label = QLabel(self)
        self.label.setObjectName("binding")
        self.label.setMinimumWidth(170)
        self.set_btn = QPushButton("Set", self)
        self.set_btn.setCheckable(True)
        self.set_btn.setToolTip("Listen for the next key combination or mouse button, anywhere on screen; "
                                "a combination is taken half a second after the last key. Esc cancels.")
        self.default_btn = QPushButton("Default", self)
        self.default_btn.setToolTip(f"Back to the default: {describe_binding(default)}")
        row.addWidget(self.label, 1)
        row.addWidget(self.set_btn)
        row.addWidget(self.default_btn)
        self.set_btn.toggled.connect(self._on_set_toggled)
        self.default_btn.clicked.connect(lambda: self.set_value(self._default))
        self._refresh()

    @property
    def default(self) -> str:
        return self._default

    @property
    def listening(self) -> bool:
        return self._listening

    def value(self) -> str:
        return self._value

    def is_valid(self) -> bool:
        return self._valid

    def set_value(self, text: str) -> None:
        raw = str(text or "").strip()
        try:
            self._value, self._valid = normalize_binding(raw), True
        except ValueError:
            self._value, self._valid = raw.lower(), False
        self._refresh()
        self.valueChanged.emit(self._value)

    def set_listening(self, on: bool) -> None:
        self._listening = on
        self.set_btn.blockSignals(True)
        self.set_btn.setChecked(on)
        self.set_btn.blockSignals(False)
        self.set_btn.setText("Listening\u2026" if on else "Set")
        self._refresh()

    def _on_set_toggled(self, checked: bool) -> None:
        (self.setRequested if checked else self.cancelRequested).emit(self)

    def _refresh(self) -> None:
        if self._listening:
            text, state = "press keys or a mouse button\u2026", "listening"
        elif not self._valid:
            text, state = f"{self._value or '?'} (invalid)", "invalid"
        else:
            text, state = describe_binding(self._value), ""
        self.label.setText(text)
        self.label.setToolTip(self._value or "unbound")
        if self.label.property("state") != state:
            self.label.setProperty("state", state)
            self.label.style().unpolish(self.label)
            self.label.style().polish(self.label)


class ShortcutsPage(QWidget):
    """The Shortcuts tab's page, built around the tab's :class:`FieldForm` (which stays the source
    of truth for load / changes / apply / restore): one list of every ``[hotkeys]`` entry -- the
    global shortcuts, then the manual-advance gestures -- as :class:`_BindingEdit` rows, a status
    line (invalid entries, a binding used twice, a bare left / middle / right button on a global
    shortcut), and the fixed in-overlay keys (:data:`FIXED_SHORTCUTS`).

    **Set** listens through a :class:`~exocortex.ui.llhooks.ChordRecorder`: on the real Windows
    platform fed by the low-level :class:`~exocortex.ui.llhooks.CaptureHooks` (any key or mouse
    button, anywhere on screen, swallowed while listening -- so no hotkey fires and nothing is
    typed into the app underneath; a click on this dialog cancels instead), elsewhere by a Qt event
    filter (keys typed while the dialog has focus). The binding is taken ``CAPTURE_QUIET_S`` after
    the last press, once no key is held; Esc cancels. One row listens at a time."""

    def __init__(self, form: FieldForm, dialog: QWidget, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._form = form
        self._dialog = dialog
        self._capture_row: _BindingEdit | None = None
        self._recorder: Any = None
        self._capture_hooks: Any = None
        self._filter_installed = False

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        inner = QWidget(scroll)
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        intro = QLabel(
            "Click Set, then press the key combination or the mouse button -- anywhere on screen. A "
            "combination is taken half a second after the last key; Esc cancels. Default puts one "
            "shortcut back (Restore defaults below: every tab). Apply / OK re-register the shortcuts at "
            "once. A single key (F9, Right Shift) works as a tap; mouse buttons 4 and 5 are the thumb "
            "buttons.", inner)
        intro.setWordWrap(True)
        intro.setObjectName("help")
        self.status = QLabel("", inner)
        self.status.setWordWrap(True)
        self.status.setObjectName("help")
        layout.addWidget(intro)
        layout.addWidget(self.status)

        fields = list(form.fields)
        bindings = [f for f in fields if f.kind == "binding"]
        general = [f for f in bindings if not f.key.startswith("manual_")]
        manual = [f for f in bindings if f.key.startswith("manual_")]
        others = [f for f in fields if f.kind != "binding"]
        self._add_section(layout, inner, "Global shortcuts (work while any app has focus)", general)
        self._add_section(layout, inner, "Manual advance (while the Manual switch is on; whether a mouse "
                                         "gesture needs the press-and-hold is on the Hands-free tab)",
                          manual + others)
        self._add_fixed(layout, inner)
        layout.addStretch(1)
        scroll.setWidget(inner)
        outer.addWidget(scroll, 1)

        self.rows: dict[str, _BindingEdit] = {}
        for f in bindings:
            widget = form.widgets[f.path]
            if isinstance(widget, _BindingEdit):
                self.rows[f.path] = widget
                widget.valueChanged.connect(lambda _value: self.refresh_status())
                widget.setRequested.connect(self.start_capture)
                widget.cancelRequested.connect(lambda _row: self.cancel_capture())
        self.refresh_status()

    # ---- layout --------------------------------------------------------------------------------
    def _add_section(self, layout: QVBoxLayout, inner: QWidget, title: str, fields: list[Field]) -> None:
        head = QLabel(title, inner)
        head.setObjectName("section")
        head.setWordWrap(True)
        layout.addWidget(head)
        form = QFormLayout()
        form.setSpacing(6)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        for f in fields:
            widget = self._form.widgets[f.path]
            if f.help and not widget.toolTip():
                widget.setToolTip(f.help)
            label = QLabel(f.label, inner)
            label.setWordWrap(True)
            if f.help:
                label.setToolTip(f.help)
            form.addRow(label, widget)
        layout.addLayout(form)

    def _add_fixed(self, layout: QVBoxLayout, inner: QWidget) -> None:
        head = QLabel("Fixed, inside the overlay (not rebindable)", inner)
        head.setObjectName("section")
        layout.addWidget(head)
        grid = QGridLayout()
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(3)
        for r, (where, key, action) in enumerate(FIXED_SHORTCUTS):
            for c, text in enumerate((where, key, action)):
                cell = QLabel(text, inner)
                cell.setWordWrap(True)
                if c != 1:
                    cell.setObjectName("fixed")
                grid.addWidget(cell, r, c)
        grid.setColumnStretch(2, 1)
        layout.addLayout(grid)

    # ---- validation -----------------------------------------------------------------------------
    def _label_of(self, path: str) -> str:
        return next((f.label for f in self._form.fields if f.path == path), path)

    def invalid(self) -> dict[str, str]:
        """``{path: label}`` of the rows whose text the grammar rejects (the dialog skips them)."""
        return {path: self._label_of(path) for path, row in self.rows.items() if not row.is_valid()}

    def duplicates(self) -> list[tuple[str, list[str]]]:
        """``[(binding text, [labels])]`` for every non-blank binding used by more than one row."""
        users: dict[str, list[str]] = {}
        for path, row in self.rows.items():
            if row.is_valid() and row.value():
                users.setdefault(row.value(), []).append(self._label_of(path))
        return [(text, labels) for text, labels in users.items() if len(labels) > 1]

    def refresh_status(self) -> None:
        problems: list[str] = []
        invalid = self.invalid()
        if invalid:
            problems.append("Invalid: " + ", ".join(invalid.values()))
        for text, labels in self.duplicates():
            problems.append(f"{describe_binding(text)} is used twice: {' and '.join(labels)}")
        for path, row in self.rows.items():
            if path.rsplit(".", 1)[-1].startswith("manual_"):
                continue
            if row.value() in ("mouse_left", "mouse_middle", "mouse_right"):
                problems.append(f"{self._label_of(path)}: a bare {describe_binding(row.value()).lower()} "
                                "fires on every click -- add a modifier")
        self.status.setText("\n".join(problems) if problems else "Every shortcut has its own key.")
        _restyle(self.status, "restart" if problems else "help")

    # ---- the Set button: capture ----------------------------------------------------------------
    @property
    def capturing(self) -> bool:
        return self._capture_row is not None

    def start_capture(self, row: _BindingEdit) -> None:
        """Listen for ``row``'s next binding (cancelling a capture already in progress)."""
        if self._capture_row is not None:
            self._finish_capture(None, self._capture_row)
        if ChordRecorder is None:
            row.set_listening(False)
            self.status.setText("Shortcut capture is unavailable here.")
            return
        self._capture_row = row
        row.set_listening(True)
        self._recorder = ChordRecorder(
            lambda text, row=row: self._on_recorded(text, row), quiet_s=CAPTURE_QUIET_S,
            schedule=lambda delay_s, check: QTimer.singleShot(max(1, int(delay_s * 1000)), self, check),
        )
        self._recorder.start()
        self._capture_hooks = None
        if _real_windows() and CaptureHooks is not None:
            hooks = CaptureHooks(self._recorder, ignore_mouse_at=self._over_dialog)
            if hooks.install():
                self._capture_hooks = hooks
            else:
                log.warning("shortcut capture: the input hooks could not be installed; keys typed into "
                            "the dialog only")
        app = QApplication.instance()
        if app is not None and not self._filter_installed:
            app.installEventFilter(self)
            self._filter_installed = True
        label = next((self._label_of(p) for p, r in self.rows.items() if r is row), "the shortcut")
        self.status.setText(f"Listening for \u201c{label}\u201d: press the keys or the mouse button. Esc cancels.")
        _restyle(self.status, "help")

    def cancel_capture(self) -> None:
        """Stop listening without changing the row (the row's own Set button, Esc, a click on the
        dialog, the dialog closing)."""
        recorder = self._recorder
        if recorder is not None and recorder.listening:
            recorder.cancel()                       # reports None -> _on_recorded -> _finish_capture
        elif self._capture_row is not None:
            self._finish_capture(None, self._capture_row)

    def _on_recorded(self, text: str | None, row: _BindingEdit) -> None:
        # may run inside a hook callback: the teardown (unhooking) is deferred to the event loop
        QTimer.singleShot(0, self, lambda: self._finish_capture(text, row))

    def _finish_capture(self, text: str | None, row: _BindingEdit) -> None:
        if self._capture_row is row:
            self._capture_row = None
            hooks, self._capture_hooks = self._capture_hooks, None
            self._recorder = None
            if hooks is not None:
                try:
                    hooks.uninstall()
                except Exception:  # noqa: BLE001
                    log.exception("shortcut capture: unhooking failed")
            app = QApplication.instance()
            if app is not None and self._filter_installed:
                app.removeEventFilter(self)
                self._filter_installed = False
        row.set_listening(False)
        if text is not None:
            row.set_value(text)
        self.refresh_status()

    def _over_dialog(self, x: int, y: int) -> bool:
        try:
            return self._dialog.isVisible() and self._dialog.frameGeometry().contains(QPoint(x, y))
        except Exception:  # noqa: BLE001
            return False

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:  # noqa: N802 - Qt override
        recorder = self._recorder
        if recorder is None or not recorder.listening:
            return False
        kind = event.type()
        if kind in (QEvent.Type.KeyPress, QEvent.Type.KeyRelease):
            if self._capture_hooks is not None or not isinstance(event, QKeyEvent):
                return False                         # the hooks saw (and swallowed) the key first
            if event.isAutoRepeat():
                return True
            token = _qt_key_token(event)
            if token is not None:
                if kind == QEvent.Type.KeyPress:
                    recorder.press(token)
                else:
                    recorder.release(token)
            return True
        if kind == QEvent.Type.MouseButtonPress:
            row = self._capture_row
            if (row is not None and isinstance(obj, QWidget)
                    and (obj is row.set_btn or row.set_btn.isAncestorOf(obj))):
                return False                         # the row's own Set button: its toggle cancels
            self.cancel_capture()
        return False


def _resolve_actions(explicit: Any, list_models: Any, parent: Any) -> Any:
    """2026-10-07 (Optimize tab): the object that implements the optimizer actions, or ``None``.

    Order: the explicit ``actions=``; ``list_models.__self__`` (``OverlayWindow.open_settings`` passes
    ``actions.list_models``, so on a real Engine that bound method's ``__self__`` IS the engine, with no
    change to ``overlay.py``); ``parent._actions`` (an ``OverlayWindow``). A candidate counts only if
    its CLASS defines ``optimize_snapshot``: the test fakes use a ``__getattr__`` catch-all that makes
    ``hasattr`` true for every name, and such a fake must not look like an engine (the tab would then
    show a "connected" page full of ``None`` answers).
    """
    for cand in (explicit, getattr(list_models, "__self__", None), getattr(parent, "_actions", None)):
        if cand is not None and callable(getattr(type(cand), "optimize_snapshot", None)):
            return cand
    return None


class SettingsDialog(QDialog):
    """Tabbed editor for :data:`FIELDS`, one :class:`FieldForm` per tab. ``applied`` carries
    ``{"section.key": new_value}`` for every value that changed on Apply / OK (empty dict when
    nothing changed)."""

    applied = Signal(dict)

    #: Tab name -> the method that builds its page around the tab's FieldForm (2026-10-29: Optimize, then
    #: Audio and Shortcuts). Every other tab is the form's own scroll area; the form stays the source of truth
    #: for load / changes / apply / restore. Each builder falls back to the plain form on any failure.
    _PAGE_BUILDERS: dict[str, str] = {
        "Optimize": "_build_optimize_page", "Audio": "_build_audio_page", "Shortcuts": "_build_shortcuts_page",
    }

    def __init__(self, cfg: Config, list_models: Callable[[], list[str]] | None = None,
                 parent: QWidget | None = None, *, actions: Any = None) -> None:
        super().__init__(parent)
        self._cfg = cfg
        self._list_models = list_models          # kept for callers built against the old surface;
        self._forms: dict[str, FieldForm] = {}    # no FIELDS entry is kind "model" any more.
        self._actions = _resolve_actions(actions, list_models, parent)   # None = Optimize tab degraded
        self._optimize_panel: Any = None
        self._audio_panel: Any = None
        self._shortcuts_page: Any = None
        self.setWindowTitle("Settings")
        self.setWindowFlags(Qt.WindowType.Dialog | Qt.WindowType.Tool | Qt.WindowType.WindowStaysOnTopHint)
        self.setStyleSheet(build_qss(cfg.ui.font_size) + _DIALOG_QSS)
        self.setMinimumSize(520, 420)
        self.resize(680, 640)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(8)
        self.tabs = QTabWidget(self)
        outer.addWidget(self.tabs, 1)
        for name, group in FIELDS.items():
            builder = self._PAGE_BUILDERS.get(name)
            if builder is not None:
                # parent None: the panel re-parents the widgets into its own layout, and the (then
                # empty) scroll shell is never shown
                form = FieldForm(group, cfg, None)
                page = getattr(self, builder)(form)
            else:
                form = FieldForm(group, cfg, self.tabs)
                page = form.widget
            self._forms[name] = form
            self.tabs.addTab(page, name)

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

    @property
    def audio_check(self) -> Any:
        """The Audio page's Check panel (:class:`~exocortex.ui.audio_check.AudioCheckPanel`), or None
        when the page fell back to the plain form."""
        return getattr(self._audio_panel, "check", None)

    # ---- merged view over every tab's FieldForm ------------------------------------------------
    @property
    def _widgets(self) -> dict[str, QWidget]:
        """``{"section.key": widget}`` across every tab (tests, and ``_on_settings_applied``-style
        callers, key off this exactly as they did when one dialog owned every widget)."""
        merged: dict[str, QWidget] = {}
        for form in self._forms.values():
            merged.update(form.widgets)
        return merged

    def _build_optimize_page(self, form: FieldForm) -> QWidget:
        """The Optimize tab's page: an :class:`~exocortex.ui.optimize_panel.OptimizePanel` around
        ``form``'s widgets. Imported lazily and guarded so a defect in the panel can never take the
        whole Settings dialog down: on any failure the tab falls back to ``form``'s plain scrollable
        form (the panel moves the form's widgets as the LAST step of its construction, so a failure
        before that leaves the form intact)."""
        try:
            from exocortex.ui.optimize_panel import OptimizePanel

            panel = OptimizePanel(self._cfg, self._actions, form, self, self.tabs)
        except Exception:  # noqa: BLE001 - the tab must exist even if its panel cannot be built
            log.exception("Optimize panel unavailable; showing the plain form")
            return form.widget
        self._optimize_panel = panel
        return panel

    def _build_audio_page(self, form: FieldForm) -> QWidget:
        """The Audio tab's page (2026-10-29): an :class:`~exocortex.ui.audio_panel.AudioPanel` around
        ``form``'s widgets -- device pickers filled from a background scan, a Test button and level
        meter per source, what the running app has open -- with this fork's checks below it
        (:class:`~exocortex.ui.audio_check.CheckedAudioPanel`: the Windows mute of a silent mic, the
        speaker routing). Same guard as the Optimize page: on any failure the tab falls back to
        ``form``'s plain scrollable form."""
        try:
            from exocortex.ui.audio_check import CheckedAudioPanel

            panel = CheckedAudioPanel(self._cfg, self._actions, form, self, self.tabs)
        except Exception:  # noqa: BLE001 - the tab must exist even if its panel cannot be built
            log.exception("Audio panel unavailable; showing the plain form")
            return form.widget
        self._audio_panel = panel
        return panel

    def _build_shortcuts_page(self, form: FieldForm) -> QWidget:
        """The Shortcuts tab's page (2026-10-29): a :class:`ShortcutsPage` around ``form``'s widgets,
        with the same fallback to the plain form as the Optimize tab."""
        try:
            page = ShortcutsPage(form, self, self.tabs)
        except Exception:  # noqa: BLE001 - the tab must exist even if its page cannot be built
            log.exception("Shortcuts page unavailable; showing the plain form")
            return form.widget
        self._shortcuts_page = page
        return page

    def _invalid_shortcuts(self) -> dict[str, str]:
        """``{path: label}`` of the binding rows the grammar rejects -- never written to the config."""
        if self._shortcuts_page is None:
            return {}
        return dict(self._shortcuts_page.invalid())

    # ---- values --------------------------------------------------------------------------------
    def load(self) -> None:
        """Fill every widget from the current config."""
        for form in self._forms.values():
            form.load()
        self.note.setText("")
        if self._optimize_panel is not None:
            self._optimize_panel.refresh()
        if self._audio_panel is not None:
            self._audio_panel.refresh()

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
        """Write the changed values into the config and announce them; returns the changes.

        No parameters, deliberately: the Apply button's ``clicked(bool)`` is connected straight to
        this method, and PySide passes the ``checked`` flag to an optional first parameter. The
        filtered variant is :meth:`apply_paths`."""
        return self._apply(None)

    def apply_paths(self, paths: Collection[str]) -> dict[str, Any]:
        """Like :meth:`apply`, but only for the given ``"section.key"`` paths (2026-10-07, Optimize
        tab: a plan writes a handful of values; edits pending on other tabs stay pending). Emits
        ``applied`` once with exactly the paths that changed."""
        return self._apply(frozenset(paths))

    def set_values(self, values: dict[str, Any]) -> list[str]:
        """Fill widgets by ``"section.key"`` path across every tab (nothing is written to the config
        until :meth:`apply` / :meth:`apply_paths`). Returns the paths that could not be set: unknown
        paths, or a value the widget cannot take."""
        widgets = self._widgets
        by_path = {f.path: f for f in all_fields()}
        failed: list[str] = []
        for path, value in values.items():
            widget, field_ = widgets.get(path), by_path.get(path)
            if widget is None or field_ is None:
                failed.append(path)
                continue
            try:
                set_field_value(widget, field_, value)
            except (TypeError, ValueError, OverflowError):
                log.warning("set_values: %s cannot take %r", path, value)
                failed.append(path)
        return failed

    def widget_values(self, paths: Collection[str]) -> dict[str, Any]:
        """The current widget value of each known path (the inverse of :meth:`set_values`; the
        Optimize tab's Undo snapshot). Unknown paths are left out."""
        widgets = self._widgets
        by_path = {f.path: f for f in all_fields()}
        return {p: get_field_value(widgets[p], by_path[p]) for p in paths if p in widgets and p in by_path}

    def _apply(self, only: Collection[str] | None) -> dict[str, Any]:
        invalid = self._invalid_shortcuts()          # 2026-10-29: a rejected shortcut stays pending
        if invalid:
            allowed = set(self._widgets) if only is None else set(only)
            only = allowed - set(invalid)
        changed: dict[str, Any] = {}
        for form in self._forms.values():
            changed.update(form.apply(only))
        by_path = {f.path: f for f in all_fields()}
        restart = [by_path[path].label for path in changed if by_path[path].restart]
        if restart:
            self.note.setText("Takes effect at the next launch: " + ", ".join(restart))
        elif changed:
            self.note.setText("Applied.")
        else:
            self.note.setText("")
        if invalid:
            skipped = "Not applied, invalid shortcut: " + ", ".join(invalid.values())
            self.note.setText((self.note.text() + " " if self.note.text() else "") + skipped)
        if changed:
            log.info("settings changed: %s", ", ".join(f"{k}={v!r}" for k, v in changed.items()))
            self.applied.emit(changed)
        return changed

    def _on_ok(self) -> None:
        self.apply()
        self.accept()

    def hideEvent(self, event: Any) -> None:  # noqa: N802 - Qt override
        """A capture in progress must not outlive the dialog on screen (its hooks would keep
        swallowing input): Cancel / OK / the close button all end it."""
        if self._shortcuts_page is not None:
            try:
                self._shortcuts_page.cancel_capture()
            except Exception:  # noqa: BLE001
                log.exception("cancelling the shortcut capture failed")
        super().hideEvent(event)

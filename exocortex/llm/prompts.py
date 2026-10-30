"""ChatML rendering and prompt builders for the local LLM (LM Studio, model `maple-preview`).

`maple-preview` uses a Qwen3-style ChatML template that, in the LM Studio chat-completions API,
always opens a `<think>` block in the generation prompt. When we talk to the raw
`/v1/completions` endpoint (which does NOT apply a template) we render ChatML ourselves with
`render_chatml` and, for "no-think" mode, pass an `assistant_prefix` that immediately closes an
empty think block (`NO_THINK_PREFIX`) so the model starts answering right away.

Prompt-cache layout
--------------------
Prompt evaluation on this machine is slow (~150 tok/s) relative to generation (~55 tok/s), so every
prompt builder here is written to keep the *stable* parts first and the *volatile* parts last, so
llama.cpp's KV prefix cache can reuse work across turns:

    1. system prompt        -- identical for every turn of a given (persona, style) pair
    2. transcript window    -- only ever grows by appending new lines at the end
    3. the short instruction + the question itself -- changes every turn, so it goes last

Within `render_chatml`, this means: system message first (most stable), user message built as
`<transcript>...</transcript>` followed by the instruction and `<question>...</question>` (least
stable), and nothing volatile is ever inserted before something stable.
"""
from __future__ import annotations

import logging
import re
from collections.abc import Sequence

log = logging.getLogger(__name__)

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"

#: Assistant-prefix that closes an empty `<think>` block immediately, so the model skips
#: reasoning and starts answering right away (verified: ~1.3 s server-side for a 3-bullet answer).
NO_THINK_PREFIX = "<think>\n\n</think>\n\n"

#: Stop sequences for raw-completion requests: bail out as soon as the model tries to close the
#: assistant turn or open a new one (chat-formatted models sometimes keep generating turns).
STOP = ["<|im_end|>", "<|im_start|>"]

_STYLES = ("bullets", "short", "detailed")


def render_chatml(system: str, user: str, assistant_prefix: str = "") -> str:
    """Render a single-turn ChatML prompt for the raw `/v1/completions` endpoint.

    Matches the Qwen3/ChatML template `maple-preview` was trained on:
    `<|im_start|>system\\n{system}<|im_end|>\\n<|im_start|>user\\n{user}<|im_end|>\\n<|im_start|>assistant\\n{assistant_prefix}`.
    The result is meant to be used as-is as the `prompt` of a completion request; generation
    continues from the end of `assistant_prefix`.
    """
    return (
        f"{IM_START}system\n{system}{IM_END}\n"
        f"{IM_START}user\n{user}{IM_END}\n"
        f"{IM_START}assistant\n{assistant_prefix}"
    )


def system_assist(persona: str, style: str, context: str = "") -> str:
    """System prompt for answering a question the OTHER meeting participant just asked.

    Instructs the model to act as the user's silent, real-time co-pilot: it must answer as if it
    *were* the user speaking, first person, ready to say out loud immediately -- never as a
    separate assistant describing what the user could say.
    """
    lines = [
        "You are the user's silent, real-time co-pilot for a live meeting. Someone else in the "
        "meeting just asked a question. Answer it AS IF YOU WERE THE USER, in first person, in "
        "words the user could say out loud right now -- you are not a separate assistant "
        "describing what the user should say.",
        "Never say \"as an AI\", never mention being a language model, an assistant, or a co-pilot, "
        "and never add a preamble such as \"Sure,\" \"Great question,\" or \"Certainly\" -- start "
        "directly with the substance of the answer.",
        "Do not use markdown headers, code fences, or numbered lists.",
        "Ground every claim in the <transcript> below: reuse the names, numbers, decisions, and "
        "prior statements it actually contains instead of inventing new ones. If the transcript "
        "and your own knowledge don't give you a real answer, say \"I'd need to check\" rather "
        "than guessing or making something up.",
        _SCREEN_RULE,
        _FOCUS_RULE,
        _CHOICE_RULE,
    ]
    lines.extend(_context_lines(context))
    if persona:
        lines.append(f"About me: {persona}")
    lines.append(_style_instruction(style))
    return "\n".join(lines)


#: Shared rule explaining the optional <screen> block (text read from the user's screen by OCR).
_SCREEN_RULE = (
    "If a <screen> block is present, it is text read from what is currently visible on the user's "
    "screen (slides, a shared document, chat, code, a form). OCR is imperfect: ignore garbled "
    "fragments. When the question refers to something shown (\"this\", \"the slide\", \"the "
    "code\"), answer about that content specifically."
)

#: What the assistant is for. Keeps the small model from wandering into site polls, surveys,
#: feedback widgets, navigation chrome and small talk that OCR drags in from the screen.
_FOCUS_RULE = (
    "Focus strictly on technical substance: theory, concepts, system design, mathematics and "
    "coding. Ignore website chrome and anything that is not a real task: polls and surveys "
    "(\"Have you seen this in an interview before?\"), feedback or rating prompts, sign-up / "
    "premium / cookie banners, navigation, social prompts and small talk. Never answer those, "
    "never mention them."
)

#: Exact reply the model gives when there is nothing technical to do; callers drop such answers.
SKIP_MARKER = "(nothing to do)"

#: The standing instruction that goes with a <toolkit> block (exocortex.project: the Claude Code
#: skills / agents / rules installed in the user's repository, auto-discovered): name what is
#: listed exactly as listed, never invent one. Config-only text, so a system prompt carrying it
#: stays byte-identical across turns (the KV-cache rule in the module docstring).
TOOLKIT_RULE = (
    "A <toolkit> block lists the Claude Code skills, agents, rules and workflows installed in the "
    "user's repository, each with the exact way it is typed or named. When a next step, an answer or "
    "a suggested prompt calls for one, name it exactly as listed, with a real argument; never invent "
    "or guess a skill, agent or command that is not listed. A <toolkit_detail> block, when present, "
    "is the full text of a part that was mentioned -- use it to be precise about what that part does."
)
#: Public names for the pieces exocortex.dashboard.prompts builds its own system prompt from.
FOCUS_RULE = _FOCUS_RULE

#: Multiple-choice questions: how to read enumerated options, and how to answer when the user
#: turn carries a <choices> block. Always part of the system prompt, byte-identical every turn
#: (a system prompt that changes per turn would defeat the KV prefix cache) -- the behaviour is
#: gated by the wording ("if, and only if, ... a <choices> block"), not by toggling the text.
#: When an option set is actually detected, `exocortex.llm.choices.choices_block` restates the
#: options at the end of the user turn and `annotate_choices` rewrites "1. 4" as "Choice 1: 4"
#: in the <screen> block, so the decimal misreading has nothing left to latch on to; the
#: enumerator sentence here is the backstop for option sets the detector misses.
_CHOICE_RULE = (
    "Enumerated options: in screen text, a line such as \"1. 4\" or \"A) 8\" is an option label "
    "followed by that option's text (option 1 is \"4\"), never the decimal number 1.4. If, and only "
    "if, the user turn includes a <choices> block, the task is a multiple-choice question: commit "
    "to the single option that is most likely correct (do not default to the first one because it "
    "is listed first), explain in at most two short sentences, and make your very last line, with "
    "nothing after it, exactly the line the <choices> block asks for. Without a <choices> block, "
    "ignore this paragraph."
)


def _screen_block(screen_text: str) -> str:
    """Render the volatile <screen> block (empty string when there is no screen text)."""
    text = (screen_text or "").strip()
    return f"<screen>\n{text}\n</screen>\n\n" if text else ""


def _context_lines(context: str) -> list[str]:
    """The standing "what this session is about" line (`[assist].session_context`), or nothing.

    Config-only text, so a system prompt carrying it stays byte-identical from turn to turn for a
    fixed config (the module docstring's KV-cache note). Every system prompt puts it right before
    the persona line; whitespace (a pasted multi-line description) is collapsed to single spaces.
    """
    text = " ".join((context or "").split())
    return [f"Session context (what the team is working on): {text}"] if text else []


context_lines = _context_lines


def _extra_block(extra: str) -> str:
    """Render the optional on-demand context blocks a user turn carries right before its ask
    (``<repo>`` lines from the project, ``<toolkit_detail>``: exocortex.project) -- "" when
    there is none. Volatile, so always after the screen block."""
    text = (extra or "").strip()
    return f"{text}\n\n" if text else ""


def system_ask(persona: str, context: str = "") -> str:
    """System prompt for a question the USER typed directly to the co-pilot (the ask box).

    Unlike `system_assist`, the model answers the user themselves rather than speaking in their
    voice to a third party -- but the same grounding and no-preamble rules apply.
    """
    lines = [
        "You are the user's silent, real-time co-pilot for a live meeting. The user just typed you "
        "a question of their own -- answer them directly and concisely so they can read it or "
        "repeat it immediately.",
        "Never say \"as an AI\", never mention being a language model, an assistant, or a co-pilot, "
        "and never add a preamble such as \"Sure,\" \"Great question,\" or \"Certainly\" -- start "
        "directly with the substance of the answer.",
        "Do not use markdown headers or code fences.",
        "Ground every claim in the <transcript> below: reuse the names, numbers, decisions, and "
        "prior statements it actually contains instead of inventing new ones. If the transcript "
        "and your own knowledge don't give you a real answer, say \"I'd need to check\" rather "
        "than guessing or making something up.",
        _SCREEN_RULE,
        _CHOICE_RULE,
        "Answer in 2 to 5 tight bullet points or, if a single short sentence fully answers it, "
        "just that sentence.",
    ]
    lines.extend(_context_lines(context))
    if persona:
        lines.append(f"About me: {persona}")
    return "\n".join(lines)


def system_suggest(persona: str, context: str = "") -> str:
    """System prompt for proactive suggestions from the screen + the live transcript."""
    lines = [
        "You are the user's silent, real-time co-pilot for a live meeting. You are shown text read "
        "from the user's screen (<screen>) and the recent conversation (<transcript>). Propose what "
        "would help the user RIGHT NOW: a point worth raising, the answer to a question that is "
        "visible on screen or was just asked, a correction, or the next thing to say.",
        "Write at most 3 bullet points (each line starting with \"- \", at most 25 words, first "
        "person, ready to say out loud). If the screen shows a partially typed sentence, an empty "
        "input field with a prompt, or a form waiting for input, add one final line starting with "
        "\"Complete: \" followed only by the text that should be typed next.",
        "If nothing useful can be added, output exactly \"- (nothing to add)\". No preamble, no "
        "headers, never mention being an AI or a co-pilot. OCR text can be garbled: ignore noise.",
        _FOCUS_RULE,
    ]
    lines.extend(_context_lines(context))
    if persona:
        lines.append(f"About me: {persona}")
    return "\n".join(lines)


#: Marker the model prepends to each option line when a task genuinely forks and it needs the
#: user to pick a direction. The UI turns each such line into a click-to-choose button, so the
#: user never has to type. Kept deliberately unusual so it never collides with real answer text.
CHOICE_PREFIX = "[[CHOICE]]"

#: `_CHOICE_RULE` for `system_solve`, which also carries the `CHOICE_PREFIX` fork protocol: the
#: two must not be confused -- a detected option set is something to answer, never a fork.
_CHOICE_RULE_SOLVE = _CHOICE_RULE + (
    f" A <choices> block is never a fork: do not turn its options into \"{CHOICE_PREFIX}\" lines, "
    "pick one of them."
)


#: Fallback used whenever a configured `language` (`cfg.assist.primary_language`) turns out blank
#: -- see `_resolve_language`.
_DEFAULT_LANGUAGE = "Python"


def _resolve_language(language: str) -> str:
    """`language`, or `_DEFAULT_LANGUAGE` when it is blank.

    `cfg.assist.primary_language` is a free-text Settings field with no validation (a cleared
    field, or a hand-edited config.toml, saves as `""`); unlike `persona`/`hint`, which are only
    spliced into a prompt conditionally, `language` is always interpolated -- so a blank value
    must resolve to a real language name here, not produce a grammatically broken instruction
    with no fallback for exactly the "no language indicated" case this rule exists to cover.
    """
    return language.strip() or _DEFAULT_LANGUAGE


#: The comment levels `cfg.assist.comments` accepts, least to most commented. Anything else (a
#: hand-edited config.toml, a capitalised value) normalises to "normal" -- see
#: `normalize_comment_level`.
COMMENT_LEVELS = ("none", "minimal", "normal", "detailed")
_DEFAULT_COMMENT_LEVEL = "normal"


def normalize_comment_level(value: object) -> str:
    """`value` as one of `COMMENT_LEVELS`.

    `cfg.assist.comments` is a free-form string in the config file, so it is validated here, once,
    for every consumer (prompt builders, the token-budget headroom in `Assistant`): case and
    surrounding whitespace are ignored, an unknown value falls back to `_DEFAULT_COMMENT_LEVEL`.
    The old `[explain].comments` boolean maps onto the two levels it used to switch between
    (`True` -> "minimal", `False` -> "none") so a leftover caller can never silently get the
    default instead of "no comments".
    """
    if value is True:
        return "minimal"
    if value is False:
        return "none"
    key = str(value or "").strip().casefold()
    return key if key in COMMENT_LEVELS else _DEFAULT_COMMENT_LEVEL


def _comment_rule(level: str) -> str:
    """The comment instruction for a solve / Code answer at `level` (one of `COMMENT_LEVELS`).

    Every level that asks for comments also asks for them to be whole lines of their own, right
    above the code they describe, never trailing at the end of a code line: `exocortex.plan`
    (completion mode) skips whole comment lines when the user types past them and anchors the
    ghost text on the code lines, which a trailing comment would break. The "detailed" level is
    what an engineer does in an interview editor -- a comment block with the requirements and
    the pseudo-code first, then the code, densely commented -- and it lives INSIDE the fenced
    block so the ghost text can offer it too (unlike the explanations-mode walkthrough, which is
    prose before the fence). Each level's text carries a distinctive phrase (tests key on them):
    "no comments at all" / "minimal but important comments" / "well commented" / "pseudo-code".
    """
    key = normalize_comment_level(level)
    if key == "none":
        return (
            " The code must contain no comments at all: no comment lines, no trailing comments, no "
            "docstrings -- only code."
        )
    if key == "minimal":
        return (
            " The code MUST contain minimal but important comments: 2 or 3 comment lines (no more, no "
            "fewer), each alone on its own line right before an important step (the core idea or data "
            "structure, the key check, the key update), never trailing at the end of a code line -- a "
            "solution with no comments or with a comment on every line is wrong."
        )
    if key == "detailed":
        return (
            " Write the code the way an engineer works a coding challenge through in the editor. The "
            "fenced block MUST open with a comment block -- line comments in the language's own syntax "
            "(# or // ...), each alone on its own line, never a docstring -- that first states the "
            "requirements (inputs, outputs, constraints, the edge cases to handle) and then the "
            "pseudo-code of the approach, step by step; then ONE blank line; then the full code. The "
            "code itself MUST be densely commented: a comment line above most lines of code saying what "
            "that line does and why (roughly one comment per line or two of code), each alone on its own "
            "line right above the line it describes, never trailing at the end of a code line."
        )
    return (
        " The code MUST be well commented: a one-line comment at the top summarising the approach, then "
        "one short comment line above every logical step -- each function, loop and branch and every "
        "non-obvious line, roughly one comment per 3 to 5 lines of code -- and none above trivially "
        "obvious lines; every comment alone on its own line right above the code it describes, never "
        "trailing at the end of a code line."
    )


def _comment_rule_complete(level: str) -> str:
    """`_comment_rule`, rephrased for a continuation (`system_complete`): the density applies to
    the code being continued; the "detailed" pre-code block does not (a continuation starts
    mid-file, and with a plan the <solution> block already carries it)."""
    key = normalize_comment_level(level)
    if key == "none":
        return "When continuing code, never add comments of any kind."
    if key == "minimal":
        return (
            "When continuing code, add a short comment line only right before a genuinely important step "
            "(the core idea, the key check or update) and nowhere else, on its own line."
        )
    if key == "detailed":
        return (
            "When continuing code, put a comment line above most lines of code saying what the line does "
            "and why (roughly one per line or two), each on its own line, never at the end of a code line."
        )
    return (
        "When continuing code, put one short comment line above each logical step (a function, a loop, a "
        "branch, a non-obvious line -- roughly one per 3 to 5 lines) and none above trivially obvious "
        "lines; every comment on its own line, never at the end of a code line."
    )


#: The answer shape for a solve / Code answer. Without `explain`: the fenced solution and the
#: complexity line. With `explain` (explanations mode): the shape `split_plan` parses (walkthrough
#: block, blank line, fenced solution, complexity line), kept as one instruction so the model keeps
#: the order. `comments` (`cfg.assist.comments`, one of `COMMENT_LEVELS`) sets how the code is
#: commented in both cases -- see `_comment_rule`.
def _solve_shape(explain: bool, comments: str = "normal", language: str = "Python") -> str:
    language = _resolve_language(language)
    comment_rule = _comment_rule(comments)
    if not explain:
        return (
            "For a coding problem: give one correct, efficient solution as a single fenced code block "
            f"in the language shown ({language} if none is indicated), then a single line stating the "
            f"time and space complexity -- no essay.{comment_rule} For a question: give the direct "
            "answer. For a form or a half-typed message: give the exact text to enter, nothing else."
        )
    return (
        "Structure the answer in this order. FIRST a walkthrough: a line \"Walkthrough:\" followed by "
        "one block of 3 to 6 short lines (no blank lines inside the block) that restates what is "
        "asked, names the key insight, the approach and the edge cases to handle -- your thought "
        "process, written before the solution -- then ONE blank line. THEN, for a coding problem, one "
        "correct, efficient solution as a single fenced code block in the language shown "
        f"({language} if none is indicated), then a single line stating the time and space "
        f"complexity.{comment_rule} For a question: after the walkthrough, the direct answer. For a "
        "form or a half-typed message: after the walkthrough, the exact text to enter."
    )


#: Shared by `system_solve`, `system_code` and (in a rephrased form) `system_complete`: a task
#: that indicates its own language wins; `language` (`cfg.assist.primary_language`) is only the
#: fallback for an open or ambiguous choice. Never volatile (depends only on the config), so the
#: system prompt it is part of stays byte-identical across turns for a fixed config -- see the
#: module docstring's KV-cache note.
#: `[assist].libraries` levels (see `_library_rule`); anything else is treated as "basic".
LIBRARY_LEVELS: tuple[str, ...] = ("none", "basic", "all")


def normalize_library_level(level: str | None) -> str:
    """A `[assist].libraries` value folded to one of `LIBRARY_LEVELS` ("basic" for anything else)."""
    value = str(level or "").strip().lower()
    return value if value in LIBRARY_LEVELS else "basic"


#: What each `[assist].libraries` level means when the task itself says nothing about libraries.
_LIBRARY_DEFAULTS: dict[str, str] = {
    "none": (
        "use only the language's built-in constructs and its basic built-in functions (len, range, "
        "sorted, min/max, string and list methods...): no import statements and no library calls -- "
        "no collections, itertools, heapq, math, re, numpy or the like; write any helper you need "
        "(a heap, a deque, a counter) yourself"
    ),
    "basic": (
        "the standard library's commonly known basics are fine (collections, itertools, heapq, math, "
        "re, functools, bisect and the like; java.util; the STL; the language's built-in objects), "
        "with the imports written out, but no third-party package"
    ),
    "all": (
        "use whatever libraries and importable functions make the solution shortest and clearest, "
        "third-party packages included (numpy, pandas, requests, lodash...), with the imports "
        "written out"
    ),
}


def _library_rule(libraries: str = "basic") -> str:
    """The one line telling the model which libraries generated code may use. Like
    `_language_rule` it is a DEFAULT, never a lock: what the task, the conversation or the code
    on screen says about libraries always wins ("without any libraries", "basic functions only",
    "no imports", "use whatever you like", the imports already in the editor), and a change of
    ask mid-task means the code is redone accordingly. `[assist].libraries` only decides the
    open case: "none" = built-in constructs and basic built-in functions only, "basic" (default)
    = the standard library's commonly known basics, "all" = anything importable, third-party
    packages included."""
    default = _LIBRARY_DEFAULTS[normalize_library_level(libraries)]
    return (
        "Libraries: follow what the task says about them -- the problem statement, a remark in the "
        "conversation (\"without any libraries\", \"basic functions only\", \"no imports\", \"you "
        "may use any library\") or the libraries the code on screen already imports -- and when that "
        "ask changes mid-task, redo the code accordingly. Only when nothing is indicated: "
        f"{default}."
    )


def _language_rule(language: str) -> str:
    language = _resolve_language(language)
    return (
        "Write the code in the language the task calls for -- the one named in the conversation or "
        "the problem statement, the one the visible editor / existing code is in, or the one the task "
        "implies (a SQL question gets SQL, a C# discussion gets C#). Only if none is indicated, the "
        f"choice is open, or it is ambiguous, use {language}. Never ask which language to use."
    )


def system_solve(persona: str, explain: bool = False, comments: str = "normal", language: str = "Python",
                 libraries: str = "basic", context: str = "") -> str:
    """System prompt for autonomous mode: SOLVE whatever task is on the user's screen.

    This is the hands-off path. Where `system_suggest` proposes talking points, this one is told
    to fully work out the problem shown on screen (a coding problem, an interview question, a
    form, a math/logic problem, a chat message awaiting a reply) and output the finished result,
    without asking the user how to proceed. Its one escape hatch is the `CHOICE_PREFIX` protocol:
    only when the task truly forks does it emit option lines instead of an answer -- a language
    choice is never one of those forks any more (`_language_rule`, `language`, resolves it).

    With `explain` (explanations mode) the answer is a *plan*: a walkthrough of the problem and the
    thought process first, then the solution, then the complexity line -- the shape
    `exocortex.plan.split_plan` parses. `comments` (`cfg.assist.comments`, one of
    `COMMENT_LEVELS`) sets how the solution code is commented, with or without `explain`.
    """
    lines = [
        "You are the user's silent, real-time co-pilot. The <screen> block is text read from what "
        "is on the user's screen right now: a task, problem, question, or prompt the user needs "
        "handled -- a coding problem, an interview or exam question, a form, a math or logic "
        "problem, a chat message awaiting a reply, and so on. SOLVE IT COMPLETELY AND "
        "INDEPENDENTLY and output the finished result the user can use or say immediately.",
        "Work it out yourself. Do NOT ask the user how they want to proceed, do NOT ask clarifying "
        "questions, and do NOT describe what you could do -- just do it. When a detail is missing, "
        "make the most reasonable assumption and proceed; note it in at most one short line only if "
        "it materially changes the result.",
        _solve_shape(explain, comments, language),
        _language_rule(language),
        _library_rule(libraries),
        "ONLY when the task genuinely forks into distinct valid directions that would each produce "
        "a different result (for example two different outputs the prompt could be asking for) and "
        "you truly cannot pick, output NOTHING except up to three option lines, each on its own line "
        f"starting with \"{CHOICE_PREFIX} \" followed by a short label naming that direction. Never "
        "use this to avoid solving something you can solve.",
        "No preamble, no \"Sure\" or \"Here is\", no markdown headers, never mention being an AI, a "
        "model, or a co-pilot. OCR text can be garbled: infer the intent and ignore obvious noise.",
        _FOCUS_RULE,
        _CHOICE_RULE_SOLVE,
        f"If the screen shows no technical task to solve, output exactly \"{SKIP_MARKER}\" and nothing else.",
    ]
    lines.extend(_context_lines(context))
    if persona:
        lines.append(f"About me: {persona}")
    return "\n".join(lines)


def user_solve(transcript_window: str, screen_text: str, focus: str = "", choices: str = "", extra: str = "") -> str:
    """User turn for `system_solve`: transcript (stable) then screen (volatile) then the ask.

    `focus` carries a direction the user picked from a previous `CHOICE_PREFIX` prompt (or a
    manual instruction); blank means "solve whatever is on screen". `choices` is the rendered
    <choices> block (`exocortex.llm.choices.choices_block`) when the screen shows a
    multiple-choice question, else ""; it sits after the screen text, right before the ask.
    `extra` (on-demand context: `_extra_block`) goes between the screen and the choices.
    """
    ask = focus.strip() or (
        "Solve whatever task, problem, or question is shown on the screen above. "
        "Output the finished solution now."
    )
    return (
        f"<transcript>\n{transcript_window.strip() or '(no speech yet)'}\n</transcript>\n\n"
        f"{_screen_block(screen_text) or '<screen>\n(nothing readable on screen)\n</screen>\n\n'}"
        f"{_extra_block(extra)}{choices}{ask}"
    )


def system_code(persona: str, explain: bool = False, comments: str = "normal", language: str = "Python",
                libraries: str = "basic", context: str = "") -> str:
    """System prompt for kind "code" (the Code button): write the code for the coding task being
    discussed / shown, right now.

    Same solve machinery as `system_solve` (`_solve_shape`, `_language_rule`, the no-questions
    rule, no preamble, `_FOCUS_RULE`, `SKIP_MARKER`) but framed around the checkpoint-scoped
    <transcript>/<screen>/<previous> a `code` job's user turn carries (`user_code`) instead of a
    single <screen> snapshot. Unlike `system_solve`, the `CHOICE_PREFIX` fork protocol is left out
    entirely: a Code press never forks (the language default, via `_language_rule`, always
    resolves what would otherwise be the one common fork -- which language to use) and
    `Assistant._detect_choices` never looks for a <choices> block on a "code" job either.
    """
    lines = [
        "You are the user's silent, real-time co-pilot. The user just pressed \"Code\": write the "
        "code for the coding task being discussed and/or shown, right now. The task is described by "
        "<transcript> (what has been said since the current checkpoint) and/or <screen>; <previous> "
        "(if present) is code produced earlier for this same task -- when the conversation asks for "
        "a change, a fix, or an extension, build on it instead of starting over.",
        "Work it out yourself. Do NOT ask the user how they want to proceed, do NOT ask clarifying "
        "questions, and do NOT describe what you could do -- just do it. When a detail is missing, "
        "make the most reasonable assumption and proceed; note it in at most one short line only if "
        "it materially changes the result.",
        _solve_shape(explain, comments, language),
        _language_rule(language),
        _library_rule(libraries),
        "No preamble, no \"Sure\" or \"Here is\", no markdown headers, never mention being an AI, a "
        "model, or a co-pilot. OCR text can be garbled: infer the intent and ignore obvious noise.",
        _FOCUS_RULE,
        f"If there is genuinely no coding task to write, output exactly \"{SKIP_MARKER}\" and nothing else.",
    ]
    lines.extend(_context_lines(context))
    if persona:
        lines.append(f"About me: {persona}")
    return "\n".join(lines)


def user_code(transcript_window: str, screen_text: str, previous: str = "", focus: str = "", extra: str = "") -> str:
    """User turn for `system_code`: transcript (stable) -> previous code (only when present) ->
    screen (volatile) -> on-demand context (`extra`, only when present) -> the ask.

    `transcript_window` is already scoped to the segments since the last checkpoint (see
    `Assistant.checkpoint`); "(nothing said since the checkpoint)" when there is none yet.
    `previous` is the most recent solve/code answer since the checkpoint (its solution part when
    that answer is a plan -- see `exocortex.plan.split_plan`), and is left out of the prompt
    entirely when blank, which is most turns for a brand new task. `focus` is an optional extra
    instruction; blank falls back to the default ask below.
    """
    ask = focus.strip() or (
        "Write the code for the coding task described in the transcript above and/or shown on the "
        "screen. If the transcript asks for a change to the previous code, apply it to that code. "
        "Output the finished code now."
    )
    previous_block = f"<previous>\n{previous.strip()}\n</previous>\n\n" if previous.strip() else ""
    return (
        f"<transcript>\n{transcript_window.strip() or '(nothing said since the checkpoint)'}\n</transcript>\n\n"
        f"{previous_block}"
        f"{_screen_block(screen_text) or '<screen>\n(nothing readable on screen)\n</screen>\n\n'}"
        f"{_extra_block(extra)}{ask}"
    )


def system_text(persona: str, context: str = "") -> str:
    """System prompt for kind "text" (the Text button): a plain-English walkthrough of the task
    being discussed / shown, right now -- the way an engineer talks a problem through out loud
    before writing any code.

    Same framing and context as `system_code` (checkpoint-scoped <transcript>, <screen>, an
    optional <previous> with code produced earlier for the task, no questions back, no preamble,
    `_FOCUS_RULE`, `SKIP_MARKER`) but the output is prose: what is asked, the key insight, the
    approach step by step, the edge cases and the complexity -- never a code block. It picks the
    most recent question or task that matters when several are around, and for a non-coding
    question it walks through the reasoning to the answer. Like `system_code` it carries no
    `CHOICE_PREFIX` fork protocol and `Assistant._detect_choices` never looks for a <choices>
    block on a "text" job.
    """
    lines = [
        "You are the user's silent, real-time co-pilot. The user just pressed \"Text\": explain, in "
        "plain English, how to approach the task or question being discussed and/or shown, right now "
        "-- the way an engineer talks a problem through out loud before writing any code. The task "
        "is described by <transcript> (what has been said since the current checkpoint) and/or "
        "<screen>; <previous> (if present) is code produced earlier for this same task -- then walk "
        "through the approach that code takes. When several questions or tasks are around, pick the "
        "most recent one that matters (the latest coding problem, interview question, or the thing "
        "being asked about) and walk through that one.",
        "Cover, in this order: what is being asked, restated in your own words (inputs, outputs, "
        "constraints); the key insight or observation that unlocks it; the approach, step by step, in "
        "words -- the data structures, the main loop or recursion, what is tracked and how it is "
        "updated; the edge cases to handle; the time and space complexity. For a non-coding question, "
        "walk through the reasoning to the answer and end with the answer.",
        "Prose only: short plain-English paragraphs or a compact list of lines starting with \"- \", "
        "6 to 12 lines in total, first person is fine (\"I'd start by ...\"), ready to say out loud. "
        "No code fences and no code -- at most a few tokens of code inline (a variable or function "
        "name).",
        "Work it out yourself. Do NOT ask the user how they want to proceed, do NOT ask clarifying "
        "questions, and do NOT describe what you could do -- just do it. When a detail is missing, "
        "make the most reasonable assumption and say so in at most one short line.",
        "No preamble, no \"Sure\" or \"Here is\", no markdown headers, never mention being an AI, a "
        "model, or a co-pilot. OCR text can be garbled: infer the intent and ignore obvious noise.",
        _FOCUS_RULE,
        f"If there is genuinely no task or question to walk through, output exactly \"{SKIP_MARKER}\" and nothing else.",
    ]
    lines.extend(_context_lines(context))
    if persona:
        lines.append(f"About me: {persona}")
    return "\n".join(lines)


def user_text(transcript_window: str, screen_text: str, previous: str = "", focus: str = "", extra: str = "") -> str:
    """User turn for `system_text`: the same checkpoint-scoped transcript -> previous code (only
    when present) -> screen -> on-demand context -> ask layout as `user_code`, with the ask asking
    for the walkthrough. `focus` is an optional extra instruction; blank falls back to the default
    ask below."""
    ask = focus.strip() or (
        "Walk me through, in plain English, the task or question described in the transcript above "
        "and/or shown on the screen: what is asked, the key insight, the approach step by step, the "
        "edge cases and the complexity. If code was produced earlier, walk through that approach. "
        "Output the walkthrough now."
    )
    previous_block = f"<previous>\n{previous.strip()}\n</previous>\n\n" if previous.strip() else ""
    return (
        f"<transcript>\n{transcript_window.strip() or '(nothing said since the checkpoint)'}\n</transcript>\n\n"
        f"{previous_block}"
        f"{_screen_block(screen_text) or '<screen>\n(nothing readable on screen)\n</screen>\n\n'}"
        f"{_extra_block(extra)}{ask}"
    )


def system_explain(persona: str, context: str = "") -> str:
    """System prompt for kind "explain": explain the solution the user has just written out.

    Runs once the plan's code is on screen (explanations mode). The user turn carries the
    walkthrough and the solution from the plan plus the current screen text, so the explanation is
    about the code the user actually typed.
    """
    lines = [
        "You are the user's silent, real-time co-pilot. The user has just finished writing out the "
        "solution below: <walkthrough> is the thought process written before it, <solution> is the "
        "planned solution, and <screen> (if present) shows what is on the user's screen now, "
        "including the code as they typed it. Explain the finished solution so the user can present "
        "or defend it right away: what each part does, why it is correct, the time and space "
        "complexity, and one likely follow-up question with its answer.",
        "Write 5 to 9 short bullet points (each line starting with \"- \"), first person is fine, "
        "no preamble, no headers, no code fences -- quote at most a few tokens of code inline. Never "
        "mention being an AI, a model, or a co-pilot. OCR text can be garbled: ignore noise.",
        _FOCUS_RULE,
    ]
    lines.extend(_context_lines(context))
    if persona:
        lines.append(f"About me: {persona}")
    return "\n".join(lines)


def user_explain(walkthrough: str, solution: str, screen_text: str = "") -> str:
    """User turn for `system_explain`: walkthrough, solution, then the (volatile) screen text."""
    return (
        f"<walkthrough>\n{walkthrough.strip() or '(none)'}\n</walkthrough>\n\n"
        f"<solution>\n{solution.strip() or '(the code on the screen)'}\n</solution>\n\n"
        f"{_screen_block(screen_text)}"
        "Explain the solution above as instructed."
    )


def is_skip(text: str) -> bool:
    """True when a model answer is (only) the `SKIP_MARKER`, allowing for stray punctuation."""
    body = (text or "").strip().strip("\"'`.").strip().casefold()
    return body in {SKIP_MARKER.casefold(), SKIP_MARKER.strip("()").casefold(), "- (nothing to add)", "(nothing to add)"}


def split_choices(text: str) -> tuple[str, list[str]]:
    """Split a model answer into (body, choices).

    Lines beginning with `CHOICE_PREFIX` are pulled out as option labels (the UI renders them as
    click-to-choose buttons); everything else is the answer body. Returns the body with those
    lines removed and the list of labels (in order, de-duplicated, blanks dropped).
    """
    body: list[str] = []
    choices: list[str] = []
    seen: set[str] = set()
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(CHOICE_PREFIX):
            label = stripped[len(CHOICE_PREFIX):].strip(" :\t-—")
            if label and label.casefold() not in seen:
                seen.add(label.casefold())
                choices.append(label)
        else:
            body.append(line)
    return "\n".join(body).strip(), choices


def system_complete(
    persona: str, hint: str = "", explain: bool = False, language: str = "Python", comments: str = "normal",
) -> str:
    """System prompt for completion mode (inline ghost-text completion at the caret).

    The text the user has typed so far is NOT in the user turn: it is rendered as the beginning
    of the assistant turn (see `render_chatml(..., assistant_prefix=NO_THINK_PREFIX + prefix)`),
    so the model literally continues it -- no echo, same language, same indentation -- and the
    prompt only ever grows at its end while the user types, which keeps llama.cpp's KV prefix
    cache warm. The user turn carries the session context and describes the field.

    With `explain` (explanations mode) the context may carry a <solution> block -- the planned
    solution -- which the continuation must follow. `comments` (`cfg.assist.comments`) sets how
    densely code that is continued WITHOUT such a block is commented (`_comment_rule_complete`; a
    plan's solution already carries its comments). `language` (`cfg.assist.primary_language`) is
    the same fallback `_language_rule` gives `system_solve`/`system_code`, rephrased for a
    continuation rather than a fresh answer: most of the time the text already typed settles the
    language, so the config value only ever matters at the very start of a field.
    """
    language = _resolve_language(language)
    lines = [
        "You are an inline autocomplete engine. The user is typing into a text field or a code "
        "editor; your assistant turn already begins with EXACTLY what they have typed so far, and "
        "you continue it from the cursor onward. Output ONLY the continuation: no quotes, no "
        "explanation, no restating of what is already typed, no markdown fences.",
        "Continue in the same language, tone, register and formatting as the typed text (code stays "
        "code with the same indentation and style; prose stays prose). Finish the current word, "
        "sentence or statement first, then at most a couple more lines. Stop at a natural boundary.",
        "Use the <context> (the meeting transcript, prior answers, notes, what is on screen) to make "
        "the continuation specific and correct: reuse the real names, numbers, decisions and code "
        "it contains. <field> describes where the user is typing (its label/placeholder and the "
        "text after the cursor); the continuation must fit before that text.",
        "If the typed text already ends at a natural end and nothing useful can be added, output "
        "nothing.",
        "The <context> may contain website chrome, polls, surveys or ads that OCR picked up; they "
        "are noise -- only the technical content (code, problem statements, conversation) matters.",
        "When the typed text and the field do not make the language evident (an empty or brand-new "
        f"code field), continue in {language}; when they do -- an existing file, a language named in "
        "the conversation, a SQL prompt -- keep continuing in that language instead. Never ask which "
        "language to use.",
    ]
    if explain:
        lines.append(
            "If <context> holds a <solution> block, that is the planned solution for what the user is "
            "working on: continue the typed text so that it follows that solution exactly (same code, "
            "same order, same names, same comments); the user may skip typing comments."
        )
    lines.append(_comment_rule_complete(comments))
    if hint.strip():
        lines.append(f"Standing instruction from the user: {hint.strip()}")
    if persona:
        lines.append(f"About the user: {persona}")
    return "\n".join(lines)


def user_complete(context_blocks: Sequence[tuple[str, str]], field: str, suffix: str) -> str:
    """User turn for `system_complete`.

    `context_blocks` is an ordered list of (tag, text) pairs -- e.g. ("transcript", ...),
    ("answers", ...), ("notes", ...), ("screen", ...) -- rendered STABLE-FIRST inside one
    <context> element; blank texts are skipped. `field` describes the target (app, label,
    placeholder), `suffix` is the text after the cursor (may be empty). The typed prefix itself is
    NOT here (it is the assistant prefix, see `system_complete`).
    """
    parts: list[str] = ["<context>"]
    for tag, text in context_blocks:
        body = (text or "").strip()
        if body:
            parts.append(f"<{tag}>\n{body}\n</{tag}>")
    if len(parts) == 1:
        parts.append("(no session context yet)")
    parts.append("</context>\n")
    parts.append(f"<field>\n{field.strip() or '(unknown field)'}\n</field>\n")
    after = (suffix or "").strip("\n")
    parts.append(f"<after_cursor>\n{after}\n</after_cursor>\n" if after.strip() else "")
    parts.append("Continue the text I am typing from the cursor. Output only the continuation.")
    return "\n".join(p for p in parts if p != "")


_FENCE_LINE_RE = re.compile(r"^\s*(```|~~~)[^\n]*\n?")


def clean_completion(prefix: str, text: str, max_lines: int) -> str:
    """Normalise a raw completion for display as ghost text.

    - drops a leading markdown fence line and a trailing one (models sometimes wrap code),
    - removes a leading repeat of the tail of `prefix` (echo), whitespace-insensitively,
    - cuts to `max_lines` lines (a partial trailing line is kept),
    - never returns text that is only whitespace.
    """
    out = text.replace("\r\n", "\n").replace("\r", "\n")
    out = _FENCE_LINE_RE.sub("", out, count=1)
    stripped = out.rstrip()
    for fence in ("```", "~~~"):
        if stripped.endswith(fence):
            stripped = stripped[: -len(fence)].rstrip()
            out = stripped
            break
    out = _strip_echo(prefix, out)
    out = trim_repetition(out)
    if max_lines > 0:
        lines = out.split("\n")
        if len(lines) > max_lines:
            out = "\n".join(lines[:max_lines])
    return out if out.strip() else ""


_ENUM_RUN_RE = re.compile(r"(?:[ \t]*\d+[.)][ \t]*){3,}(?:\d+)?$")
_WORD_RUN_RE = re.compile(r"\b(\w+)(?:[ \t]+\1\b){2,}")
_PHRASE_RUN_RE = re.compile(r"(.{4,40}?)(?:\1){2,}", re.DOTALL)


def trim_repetition(text: str) -> str:
    """Cut a completion at the point where it degenerates into repetition.

    Small models fed odd input loop: "2. 3. 4. 5. ..." (bare enumerators with no content), the
    same word three times in a row, or a phrase repeated back to back. The text is cut before the
    run starts (a bare enumerator run at the end is dropped entirely).
    """
    if not text:
        return text
    original = text
    m = _PHRASE_RUN_RE.search(text)
    if m:
        text = text[: m.start() + len(m.group(1))]
    m = _WORD_RUN_RE.search(text)
    if m:
        text = text[: m.start() + len(m.group(1))]
    text = _ENUM_RUN_RE.sub("", text)
    return text.rstrip() if text != original else text


def _strip_echo(prefix: str, text: str) -> str:
    """If `text` starts by repeating the last line of `prefix` (or a tail of it, >= 3 chars),
    drop that repeat. Whitespace around the echoed part is tolerated: when the typed line already
    ends in whitespace, the same leading whitespace is removed from the continuation."""
    if not prefix or not text:
        return text
    last_line = prefix.rsplit(chr(10), 1)[-1]
    tail = last_line.strip()
    if not tail:
        return text
    if text.startswith(last_line):
        return text[len(last_line):]
    lead = text.lstrip()
    trailing_ws = last_line.endswith((" ", chr(9)))
    if lead.startswith(tail):
        rest = lead[len(tail):]
        return rest.lstrip(" " + chr(9)) if trailing_ws else rest
    for n in range(min(len(tail), 40), 2, -1):
        piece = tail[-n:]
        if lead.startswith(piece):
            rest = lead[n:]
            return rest.lstrip(" " + chr(9)) if trailing_ws else rest
    return text


def user_suggest(transcript_window: str, screen_text: str, focus: str = "") -> str:
    """User turn for `system_suggest`: transcript (stable) then screen (volatile) then the ask."""
    ask = focus.strip() or "What should I say or do next? Keep it to the most useful 1-3 points."
    return (
        f"<transcript>\n{transcript_window.strip() or '(no speech yet)'}\n</transcript>\n\n"
        f"{_screen_block(screen_text) or '<screen>\n(nothing readable on screen)\n</screen>\n\n'}"
        f"{ask}"
    )


# ---- co-pilot insights + recap (pair programming / team coding, 2026-10-20) ------------------

#: Co-pilot insight sections (`[assist].suggest_questions` / `[assist].flag_blind_spots`), in
#: display order, and the exact heading each is written under -- by the model, by
#: `render_insights`, and so in the Notes tab's Insights pane and the session's insights.md.
INSIGHT_SECTIONS: tuple[str, ...] = ("questions", "blindspots")
INSIGHT_HEADINGS: dict[str, str] = {
    "questions": "## Questions to ask",
    "blindspots": "## Blind spots & follow-ups",
}
#: Hard cap on the bullets per section, whatever `[assist].insight_max_items` says.
MAX_INSIGHT_ITEMS = 8

#: `[assist].role` values; anything else normalises to "driver".
ROLES: tuple[str, ...] = ("driver", "navigator", "participant")


def normalize_role(value: object) -> str:
    """`value` as one of `ROLES` ("driver", the default, for anything unknown)."""
    key = str(value or "").strip().casefold()
    return key if key in ROLES else "driver"


def normalize_sections(sections: object) -> tuple[str, ...]:
    """The known insight sections named by `sections` (an iterable of names, or a single name),
    in display order and de-duplicated; () when none of them is known."""
    if isinstance(sections, str):
        sections = (sections,)
    try:
        wanted = {str(s).strip().casefold() for s in (sections or ())}
    except TypeError:
        return ()
    return tuple(s for s in INSIGHT_SECTIONS if s in wanted)


#: Who is at the keyboard and what the co-pilot is for, per `[assist].role`.
_ROLE_FRAMES: dict[str, str] = {
    "driver": (
        "The user is the DRIVER: they are at the keyboard -- <screen> is their editor, terminal or "
        "browser -- while teammates talk it through with them (<transcript>: \"Me\" is the user, "
        "every other label is a teammate). Heads-down at the keyboard, a driver misses things; you "
        "keep track of them."
    ),
    "navigator": (
        "The user is the NAVIGATOR: a teammate is at the keyboard -- <screen> is the shared editor "
        "or terminal -- and the user reviews and steers (<transcript>: \"Me\" is the user, every "
        "other label is a teammate). You help the user review sharply."
    ),
    "participant": (
        "The user is one participant in a technical discussion -- a design review, a debugging or "
        "planning session (<transcript>: \"Me\" is the user, every other label is someone else; "
        "<screen> is what is being shown). You help the user contribute."
    ),
}

#: Whom a suggested question is addressed to, per role.
_QUESTION_AUDIENCE: dict[str, str] = {
    "driver": "the navigator or the team",
    "navigator": "the driver",
    "participant": "the group",
}
#: Public names (exocortex.dashboard.prompts frames its refresh prompt with the same role text).
ROLE_FRAMES = _ROLE_FRAMES
QUESTION_AUDIENCE = _QUESTION_AUDIENCE


def _insight_rule(section: str, role: str, max_items: int) -> str:
    """What goes under one insight heading (see `system_insights`)."""
    if section == "questions":
        return (
            f"Under \"{INSIGHT_HEADINGS['questions']}\": up to {max_items} questions the user should "
            f"ask {_QUESTION_AUDIENCE[role]} right now -- to clarify a requirement, pin down an "
            "assumption, probe a design choice or an edge case, or settle something that was said "
            "but not decided. Write each exactly as the user would say it out loud: first person, at "
            "most 20 words, specific to what was said or shown (reuse the real function, file, "
            "service and variable names and the numbers). Never a generic question (\"any "
            "concerns?\", \"does that make sense?\") and never one the transcript already answers."
        )
    return (
        f"Under \"{INSIGHT_HEADINGS['blindspots']}\": up to {max_items} risks or gaps nobody has "
        "raised yet -- an unhandled edge case or error path, a missing test, a concurrency, "
        "performance, security or data-loss concern, a breaking change for callers, an unverified "
        "assumption -- and follow-ups that must not be forgotten (a TODO, someone to loop in, a "
        "ticket to file). Each line names the concern first, then in a few words why it matters, "
        "at most 20 words. Only what the transcript or the code on screen supports."
    )


def system_insights(
    persona: str = "", role: str = "driver", sections: object = INSIGHT_SECTIONS, max_items: int = 4,
    context: str = "",
) -> str:
    """System prompt for kind "insights": the co-pilot's living list of questions worth asking and
    blind spots / follow-ups in a pair-programming or team coding session.

    `role` (`[assist].role`, one of `ROLES`) frames who is at the keyboard and whom a question is
    for; `sections` (a subset of `INSIGHT_SECTIONS`) picks the headings the answer must carry, in
    display order; `max_items` caps the bullets per section (clamped to 1..`MAX_INSIGHT_ITEMS`).
    The user turn's <previous> carries the list as it stands, so a refresh keeps what is still
    open and drops what was answered (`user_insights`). Config-only: byte-identical across turns
    for a fixed config, like every system prompt here.
    """
    role = normalize_role(role)
    secs = normalize_sections(sections) or INSIGHT_SECTIONS
    try:
        n = int(max_items)
    except (TypeError, ValueError):
        n = 4
    n = max(1, min(n, MAX_INSIGHT_ITEMS))
    headings = ", ".join(f"\"{INSIGHT_HEADINGS[s]}\"" for s in secs)
    lines = [
        "You are the user's co-pilot in a live pair-programming / team coding session. " + _ROLE_FRAMES[role],
        "Keep a short, living list. <previous> (if present) is the list as it stands: keep an item "
        "only while it is still open and still relevant, drop whatever the conversation has since "
        "answered, settled or made moot, and add what is new -- most important first.",
        *(_insight_rule(s, role, n) for s in secs),
        f"Output exactly these headings, in this order, each alone on its own line: {headings}. "
        "Under each, one item per line starting with \"- \"; a single \"- (none)\" line when a "
        "heading has nothing worth saying. No other headings, no preamble, no closing remarks, no "
        "code fences; never mention being an AI, a model, or a co-pilot. OCR text can be garbled: "
        "ignore noise.",
        _FOCUS_RULE,
        "If there is no technical discussion and no code to work from at all, output exactly "
        f"\"{SKIP_MARKER}\" and nothing else.",
    ]
    lines.extend(_context_lines(context))
    if persona:
        lines.append(f"About me: {persona}")
    return "\n".join(lines)


#: What `user_insights`' closing ask calls each section.
_INSIGHT_ASK_NAMES: dict[str, str] = {
    "questions": "the questions to ask",
    "blindspots": "the blind spots & follow-ups",
}


def user_insights(
    transcript_window: str, screen_text: str, previous: str = "", sections: object = INSIGHT_SECTIONS,
) -> str:
    """User turn for `system_insights`: transcript (stable) -> the current list (only when there is
    one: `render_insights` of the sections being refreshed) -> screen (volatile) -> the ask -- the
    same layout as `user_code`."""
    secs = normalize_sections(sections) or INSIGHT_SECTIONS
    ask = f"Update {' and '.join(_INSIGHT_ASK_NAMES[s] for s in secs)} now, as instructed."
    previous_block = f"<previous>\n{previous.strip()}\n</previous>\n\n" if (previous or "").strip() else ""
    return (
        f"<transcript>\n{transcript_window.strip() or '(no speech yet)'}\n</transcript>\n\n"
        f"{previous_block}"
        f"{_screen_block(screen_text) or '<screen>\n(nothing readable on screen)\n</screen>\n\n'}"
        f"{ask}"
    )


_INSIGHT_HEADING_RE = re.compile(r"^#{1,6}\s*(.*?)\s*:?\s*$")
_INSIGHT_BOLD_HEADING_RE = re.compile(r"^\*\*([^*]+?):?\*\*:?$")
_INSIGHT_BULLET_RE = re.compile(r"^(?:[-*•]|\d+[.)])\s+(.*)$")
#: Placeholder items a model writes for an empty section (compared casefolded, without trailing dots).
_NONE_ITEMS = frozenset({"(none)", "none", "(none yet)", "n/a", "(n/a)", "-", "(nothing to do)", "(nothing to add)"})


def _section_of(title: str) -> str | None:
    """The insight section a heading names, matched loosely; None for any other heading."""
    t = title.casefold()
    if "question" in t:
        return "questions"
    if any(word in t for word in ("blind", "follow", "risk", "gap")):
        return "blindspots"
    return None


def parse_insights(text: str, sections: object = INSIGHT_SECTIONS, max_items: int = MAX_INSIGHT_ITEMS) -> dict[str, str]:
    """Split an insights answer into ``{section: markdown bullet list}`` for the requested
    `sections` (`normalize_sections`; all of them when none is known), "" for a section the model
    listed nothing under.

    A line under a heading -- ``## ...`` or a line that is entirely ``**bold**`` -- naming a section
    (matched loosely: "question"; "blind" / "follow" / "risk" / "gap") belongs to it; lines under
    any other heading are dropped; lines before the first heading go to the first requested
    section (a single-section answer that left its heading out). Bullets ("- ", "* ", "1. ")
    become "- " items, a non-bullet line continues the item above it, "(none)"-style placeholders
    and blank lines are dropped, and a section keeps at most `max_items` items (clamped to
    1..`MAX_INSIGHT_ITEMS`)."""
    secs = normalize_sections(sections) or INSIGHT_SECTIONS
    try:
        cap = max(1, min(int(max_items), MAX_INSIGHT_ITEMS))
    except (TypeError, ValueError):
        cap = MAX_INSIGHT_ITEMS
    items: dict[str, list[str]] = {s: [] for s in secs}
    current: str | None = secs[0]
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        heading = _INSIGHT_HEADING_RE.match(line) or _INSIGHT_BOLD_HEADING_RE.match(line)
        if heading:
            section = _section_of(heading.group(1))
            current = section if section in items else None
            continue
        if current is None:
            continue
        bullet = _INSIGHT_BULLET_RE.match(line)
        body = (bullet.group(1) if bullet else line).strip()
        if not body or body.casefold().rstrip(". ") in _NONE_ITEMS:
            continue
        if bullet or not items[current]:
            items[current].append(body)
        else:
            items[current][-1] = f"{items[current][-1]} {body}"
    return {s: "\n".join(f"- {item}" for item in items[s][:cap]) for s in secs}


def insight_sections_in(text: str) -> tuple[str, ...]:
    """The insight sections whose heading already appears in `text` -- a streaming answer, so the
    Insights pane can show the new text for those and keep the rest as they were. Display order."""
    found: set[str] = set()
    for raw in (text or "").splitlines():
        line = raw.strip()
        heading = _INSIGHT_HEADING_RE.match(line) or _INSIGHT_BOLD_HEADING_RE.match(line)
        if heading:
            section = _section_of(heading.group(1))
            if section is not None:
                found.add(section)
    return tuple(s for s in INSIGHT_SECTIONS if s in found)


def render_insights(sections: dict[str, str], *, skip_empty: bool = True) -> str:
    """Markdown for an insights document: each section's heading (`INSIGHT_HEADINGS`) and bullets,
    in display order. A section that is missing or "" is left out with `skip_empty` (the Insights
    pane, a <previous> block) or written as "- (none)" without it (the session's insights.md)."""
    parts: list[str] = []
    for s in INSIGHT_SECTIONS:
        body = (sections.get(s) or "").strip()
        if not body and skip_empty:
            continue
        parts.append(f"{INSIGHT_HEADINGS[s]}\n{body or '- (none)'}")
    return "\n\n".join(parts)


def system_recap(persona: str = "", context: str = "") -> str:
    """System prompt for kind "recap" (the Recap action): where the session stands right now --
    goal, done, now, next -- short enough to read out to the team or to catch up on after a
    distraction. The user turn carries the recent transcript, the running notes (the whole
    session's memory) and the screen (`user_recap`)."""
    lines = [
        "You are the user's co-pilot in a live pair-programming / team coding session. The user "
        "asked for a RECAP of where the session stands, to catch up or to read out to the team. "
        "<transcript> is the most recent discussion (\"Me\" is the user), <notes> (if present) are "
        "the running notes covering the whole session so far, <screen> is what is on the user's "
        "screen now.",
        "Write exactly four parts, each starting on a new line with its bold label: \"**Goal:**\" "
        "-- one line, what the team is trying to do; \"**Done:**\" -- 1 to 3 \"- \" bullets, what "
        "has been built, fixed or decided; \"**Now:**\" -- 1 or 2 \"- \" bullets, what is being "
        "worked on right now and what blocks it, if anything; \"**Next:**\" -- 1 to 3 \"- \" "
        "bullets, the next steps and the questions still open.",
        "At most 130 words in total. Be specific (the real function, file and service names, the "
        "numbers) and invent nothing: when a part is unknown, say so in a few words. No preamble, "
        "no other headings, no code fences; never mention being an AI, a model, or a co-pilot.",
        _FOCUS_RULE,
        f"If nothing has happened yet that could be recapped, output exactly \"{SKIP_MARKER}\" and nothing else.",
    ]
    lines.extend(_context_lines(context))
    if persona:
        lines.append(f"About me: {persona}")
    return "\n".join(lines)


def user_recap(transcript_window: str, notes: str = "", screen_text: str = "", focus: str = "", extra: str = "") -> str:
    """User turn for `system_recap`: transcript (stable) -> running notes (only when there are
    any) -> screen (volatile, only when there is any) -> on-demand context (only when there is
    any) -> the ask. `focus` is an optional extra instruction; blank falls back to the default ask."""
    ask = focus.strip() or "Recap the session now, as instructed."
    notes_block = f"<notes>\n{notes.strip()}\n</notes>\n\n" if (notes or "").strip() else ""
    return (
        f"<transcript>\n{transcript_window.strip() or '(no speech yet)'}\n</transcript>\n\n"
        f"{notes_block}"
        f"{_screen_block(screen_text)}"
        f"{_extra_block(extra)}{ask}"
    )


def _style_instruction(style: str) -> str:
    """Return the style-specific formatting rule for `system_assist`."""
    key = style if style in _STYLES else "bullets"
    if key == "short":
        return "Answer in 1 to 2 sentences, plain prose, no bullets, no line breaks."
    if key == "detailed":
        return (
            "Answer with up to 8 concise bullet points (each line starting with \"- \"), each "
            "starting with its key point, then one final line starting with \"Bottom line: \" that "
            "sums up the answer in a single sentence."
        )
    return (
        "Answer with 2 to 4 tight bullet points (each line starting with \"- \"), each at most 20 "
        "words, each starting with its key point first."
    )


#: `[notes].template` values: "dev" (a working session of engineers, the default) or "meeting"
#: (the original general-meeting wording). Both keep the same five headings in the same order.
NOTES_TEMPLATES: tuple[str, ...] = ("dev", "meeting")

#: Added to the notes prompt for `[notes].template = "dev"`: what each heading holds in a
#: pair-programming / team coding session.
_DEV_NOTES_RULE = (
    "This is a working session of software engineers (pair programming, a team coding or debugging "
    "session): under \"## Key points\" record the technical substance -- what was changed in the "
    "code (files, functions, modules by name), bugs found and their root cause, design choices and "
    "the trade-offs behind them, commands, configuration or environment details that matter; under "
    "\"## Decisions\" each technical decision with its reason in a few words; under \"## Action "
    "items\" also the TODOs left in the code and the follow-ups for after the session; under \"## "
    "Open questions\" the technical questions still unresolved. Use the exact names (functions, "
    "files, services, tickets, numbers) from the transcript."
)

#: The same, when the dashboard keeps the action items (tasks and sub-tasks), goals, constraints
#: and questions in its own database: the notes hold only the narrative.
_DEV_NOTES_RULE_TRACKED = (
    "This is a working session of software engineers (pair programming, a team coding or debugging "
    "session): under \"## Key points\" record the technical substance -- the scope of the brief, what "
    "was changed in the code (files, functions, modules by name), bugs found and their root cause, "
    "design choices and the trade-offs behind them, commands, configuration or environment details "
    "that matter; under \"## Decisions\" each technical decision with its reason in a few words; "
    "under \"## Open questions\" the technical questions still unresolved. Use the exact names "
    "(functions, files, services, tickets, numbers) from the transcript."
)

_NOTES_OUTPUT_RULE = "Output only the markdown document itself, nothing before or after it."


def normalize_notes_template(value: object) -> str:
    """`value` as one of `NOTES_TEMPLATES` ("dev", the default, for anything unknown)."""
    key = str(value or "").strip().casefold()
    return key if key in NOTES_TEMPLATES else "dev"


def system_notes(template: str = "meeting", *, tracked: bool = False) -> str:
    """System prompt for maintaining the running notes document.

    `template` (`[notes].template`, `NOTES_TEMPLATES`): "meeting" is the original general-meeting
    prompt, unchanged; "dev" adds `_DEV_NOTES_RULE` right before the closing "Output only ..."
    line, steering the same five headings toward a working session. `tracked` (dashboard mode is
    on): the action items, goals, constraints and questions live in the dashboard's database, so
    the document is narrative only -- four headings, no "## Action items". Config-only text.
    """
    dev = normalize_notes_template(template) == "dev"
    if tracked:
        return "\n".join((_notes_body(tracked=True), *((_DEV_NOTES_RULE_TRACKED,) if dev else ()), _NOTES_OUTPUT_RULE))
    if dev:
        return f"{_notes_body()}\n{_DEV_NOTES_RULE}\n{_NOTES_OUTPUT_RULE}"
    return f"{_notes_body()}\n{_NOTES_OUTPUT_RULE}"


def _notes_body(*, tracked: bool = False) -> str:
    """Everything in the notes prompt before its closing output rule (shared by both templates)."""
    if tracked:
        return (
            "You are maintaining running notes for a live meeting. Rewrite the full notes document in "
            "markdown using EXACTLY these four headings, in this order, each alone on its own line: "
            "\"## Summary\", \"## Key points\", \"## Decisions\", \"## Open questions\". Never add, remove, "
            "rename, reorder, or duplicate a heading.\n"
            "The tasks and sub-tasks, the goals (acceptance criteria), the constraints and the questions to "
            "ask are tracked elsewhere: do not list them here and do not write a heading for them.\n"
            "Keep every still-true fact from the <previous_notes> (if any) and merge in whatever new "
            "information <new_transcript> adds; drop only what the new transcript has clearly "
            "superseded or corrected. Be concise: short bullet points, no long paragraphs, no repeated "
            "information, no invented facts.\n"
            "\"## Summary\" is 1-3 sentences of plain prose: what the session is about and where it stands. "
            "\"## Key points\", \"## Decisions\" and \"## Open questions\" are each a \"- \" bullet list "
            "(write a single \"- (none)\" line if empty)."
        )
    return (
        "You are maintaining running notes for a live meeting. Rewrite the full notes document in "
        "markdown using EXACTLY these five headings, in this order, each alone on its own line: "
        "\"## Summary\", \"## Key points\", \"## Decisions\", \"## Action items\", "
        "\"## Open questions\". Never add, remove, rename, reorder, or duplicate a heading.\n"
        "Keep every still-true fact from the <previous_notes> (if any) and merge in whatever new "
        "information <new_transcript> adds; drop only what the new transcript has clearly "
        "superseded or corrected. Be concise: short bullet points, no long paragraphs, no repeated "
        "information, no invented facts.\n"
        "\"## Summary\" is 1-3 sentences of plain prose. \"## Key points\", \"## Decisions\" and "
        "\"## Open questions\" are each a \"- \" bullet list (write a single \"- (none)\" line if "
        "empty). \"## Action items\" lists each item as \"- [ ] owner: task (due)\" when an owner "
        "and/or due date were mentioned in the transcript, or \"- [ ] task\" when they were not "
        "(write a single \"- (none)\" line if there are no action items)."
    )


def user_assist(transcript_window: str, question: str, screen_text: str = "", choices: str = "", extra: str = "") -> str:
    """User turn for `system_assist`: transcript context, optional screen text, then the question.

    STABLE-PREFIX ORDER: `transcript_window` only ever grows by appending new lines at its end, so
    placing it before the instruction+question keeps the first N tokens of this user turn
    byte-identical across consecutive assist turns, which is what lets the KV prefix cache be
    reused instead of re-evaluating the whole prompt on every question (see module docstring).
    The <screen> block changes often, so it goes AFTER the transcript and before the question;
    `extra` (on-demand context, `_extra_block`) and `choices` (the rendered <choices> block of a
    multiple-choice question, else "") follow it.
    """
    return (
        f"<transcript>\n{transcript_window}\n</transcript>\n\n"
        f"{_screen_block(screen_text)}{_extra_block(extra)}{choices}"
        "The other participant just asked the question below. Answer it as instructed in the "
        "system prompt.\n"
        f"<question>\n{question}\n</question>"
    )


def user_ask(transcript_window: str, question: str, screen_text: str = "", choices: str = "", extra: str = "") -> str:
    """User turn for `system_ask`: transcript context, optional screen text, then the question.

    Same stable-prefix ordering as `user_assist` -- see module docstring.
    """
    return (
        f"<transcript>\n{transcript_window}\n</transcript>\n\n"
        f"{_screen_block(screen_text)}{_extra_block(extra)}{choices}"
        "Answer the question below, using the transcript above (and the screen text, if any) for "
        "context where it's relevant.\n"
        f"<question>\n{question}\n</question>"
    )


def user_notes(previous_notes: str, new_transcript: str) -> str:
    """User turn for `system_notes`: previous notes (stable-ish) then the new transcript to merge."""
    prev = previous_notes.strip() or "(none yet -- this is the first notes pass)"
    return (
        f"<previous_notes>\n{prev}\n</previous_notes>\n\n"
        f"<new_transcript>\n{new_transcript}\n</new_transcript>\n\n"
        "Update the notes: keep everything from <previous_notes> that is still true, merge in "
        "whatever <new_transcript> adds, and follow the heading and formatting rules from the "
        "system prompt exactly."
    )


def approx_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token) used for trimming transcript windows to a budget."""
    return len(text) // 4 + 1

"""Tests for the `[assist].focus` gradient added to exocortex.llm.questions:
`FOCUS_LEVELS` / `focus_level` / `is_technical_focus` / `is_task_line` / the `is_relevant`
gradient, plus the matching config default (`AssistConfig.focus`) and that `load_config` never
normalises a stored value. Fully offline, no `live` marker -- see
tests/test_prompts_questions.py for the pre-existing `is_relevant` default-focus table this
extends with the full three-level gradient, and tests/test_settings.py for the config-loading
pattern reused for the last two tests.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from exocortex.config import AssistConfig, Config, load_config
from exocortex.llm import questions

# --------------------------------------------------------------------------------------------
# FOCUS_LEVELS / focus_level / is_technical_focus
# --------------------------------------------------------------------------------------------


def test_focus_levels_value() -> None:
    assert questions.FOCUS_LEVELS == ("technical", "auto", "all")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("technical", "technical"),
        ("all", "all"),
        ("auto", "auto"),
        ("TECHNICAL", "technical"),
        ("All", "all"),
        (" technical ", "technical"),
        (None, "auto"),
        ("", "auto"),
        ("weird", "auto"),
        ("strict", "auto"),
    ],
)
def test_focus_level(value: object, expected: str) -> None:
    assert questions.focus_level(value) == expected


def test_is_technical_focus() -> None:
    assert questions.is_technical_focus("technical") is True
    assert questions.is_technical_focus("Technical") is True
    assert questions.is_technical_focus("auto") is False
    assert questions.is_technical_focus("all") is False
    assert questions.is_technical_focus(None) is False
    assert questions.is_technical_focus("") is False


# --------------------------------------------------------------------------------------------
# is_task_line
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "What is the time complexity of this approach?",     # a question
        "Reverse a linked list",                              # opening task verb
        "Q3: explain the CAP theorem",                        # numbered/lettered prefix + verb
        "Given an array of integers, nums, return the two indices that sum to target.",
        "Implement a rate limiter for the public API.",
        "Task 2. Design a URL shortener.",
        "Kafka keeps messages ordered within a partition, so consumers replay them in order.",   # a sentence with a technical cue
    ],
)
def test_is_task_line_true_cases(text: str) -> None:
    assert questions.is_task_line(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "lunch at noon works for me",
        "We should ship this on Friday.",
        "Thanks for joining today.",
        "",
        "   ",
        "right?",       # too short to be a question (is_question's min_words=2 default here)
    ],
)
def test_is_task_line_false_cases(text: str) -> None:
    assert questions.is_task_line(text) is False


# --------------------------------------------------------------------------------------------
# is_relevant gradient (technical / auto / all)
# --------------------------------------------------------------------------------------------


def test_is_relevant_ignore_pattern_fails_at_every_level() -> None:
    poll = "Did you like this problem? Rate it now"
    for focus in ("technical", "auto", "all"):
        assert questions.is_relevant(poll, focus=focus) is False


def test_is_relevant_why_did_you_choose_that_passes_auto_and_all_fails_technical() -> None:
    text = "Why did you choose that?"
    assert questions.is_relevant(text, focus="all") is True
    assert questions.is_relevant(text, focus="auto") is True
    assert questions.is_relevant(text, focus="technical") is False


def test_is_relevant_have_you_used_kafka_passes_every_level() -> None:
    text = "Have you used Kafka?"
    for focus in ("technical", "auto", "all"):
        assert questions.is_relevant(text, focus=focus) is True


def test_is_relevant_do_you_like_our_platform_fails_every_level() -> None:
    """An ignore-pattern poll ("did/do you like ... our") is dropped regardless of focus."""
    text = "Do you like our platform?"
    for focus in ("technical", "auto", "all"):
        assert questions.is_relevant(text, focus=focus) is False


def test_is_relevant_personal_question_without_cue_passes_all_fails_auto_and_technical() -> None:
    text = "Have you ever been to Paris?"
    assert questions.is_relevant(text, focus="all") is True
    assert questions.is_relevant(text, focus="auto") is False
    assert questions.is_relevant(text, focus="technical") is False


# --------------------------------------------------------------------------------------------
# config: default + load_config never normalises a stored value
# --------------------------------------------------------------------------------------------


def test_assist_config_default_focus_is_auto() -> None:
    assert AssistConfig().focus == "auto"
    assert Config().assist.focus == "auto"


def test_load_config_keeps_a_technical_value(tmp_path: Path) -> None:
    """`load_config` (`exocortex.config._apply`) writes the raw string straight through -- it is
    `questions.focus_level`/`is_relevant` that fold an unknown or old value to "auto", not the
    config loader itself, so a `[assist] focus = "technical"` in config.toml must round-trip
    unchanged."""
    path = tmp_path / "config.toml"
    path.write_text('[assist]\nfocus = "technical"\n', encoding="utf-8")
    cfg = load_config(path)
    assert cfg.assist.focus == "technical"

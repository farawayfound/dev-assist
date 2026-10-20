"""Tests for the "text" prompt builders (`prompts.system_text` / `prompts.user_text`) added for
the Text button. Fully offline, no `live` marker -- see tests/test_prompts_questions.py and
tests/test_prompts_code.py for the rest of prompts.py."""
from __future__ import annotations

from exocortex.llm import prompts

# --------------------------------------------------------------------------------------------
# system_text -- framing
# --------------------------------------------------------------------------------------------


def test_system_text_frames_the_text_button_press() -> None:
    out = prompts.system_text("")
    assert 'pressed "Text"' in out


def test_system_text_asks_for_plain_english() -> None:
    out = prompts.system_text("").lower()
    assert "plain english" in out


def test_system_text_covers_insight_approach_edge_cases_complexity() -> None:
    """The walkthrough must cover the same beats a real engineer talks through out loud."""
    out = prompts.system_text("").lower()
    assert "key insight" in out
    assert "approach" in out
    assert "edge case" in out
    assert "complexity" in out


def test_system_text_includes_tags() -> None:
    out = prompts.system_text("")
    assert "<transcript>" in out
    assert "<screen>" in out
    assert "<previous>" in out


def test_system_text_prose_only_no_code_fences() -> None:
    out = prompts.system_text("")
    assert "Prose only" in out
    assert "no code fences" in out.lower()
    assert "```" not in out


def test_system_text_do_not_ask_case_insensitive() -> None:
    out = prompts.system_text("").lower()
    assert "do not ask" in out


def test_system_text_never_mentions_being_an_ai() -> None:
    out = prompts.system_text("").lower()
    assert "never mention being an ai" in out


def test_system_text_has_skip_marker() -> None:
    out = prompts.system_text("")
    assert prompts.SKIP_MARKER in out


def test_system_text_has_focus_rule() -> None:
    out = prompts.system_text("")
    assert "Focus strictly on technical substance" in out


def test_system_text_no_choice_prefix_or_choices_tag() -> None:
    """A walkthrough never forks and never reads a <choices> block -- unlike system_solve, it
    carries neither the CHOICE_PREFIX fork protocol nor any mention of <choices>."""
    out = prompts.system_text("")
    assert prompts.CHOICE_PREFIX not in out
    assert "<choices>" not in out


def test_system_text_includes_persona_when_present() -> None:
    assert "backend engineer" in prompts.system_text("I am a backend engineer.")


def test_system_text_omits_persona_when_empty() -> None:
    out = prompts.system_text("")
    assert "About me" not in out


def test_system_text_is_byte_identical_across_calls() -> None:
    a = prompts.system_text("persona")
    b = prompts.system_text("persona")
    assert a == b


def test_system_text_differs_from_system_code_but_both_carry_skip_marker() -> None:
    """Distinct prompts (prose walkthrough vs. code answer) that nonetheless share the same
    "nothing to do" escape hatch."""
    text_out = prompts.system_text("")
    code_out = prompts.system_code("")
    assert text_out != code_out
    assert prompts.SKIP_MARKER in text_out
    assert prompts.SKIP_MARKER in code_out


# --------------------------------------------------------------------------------------------
# user_text -- ordering / placeholders
# --------------------------------------------------------------------------------------------


def test_user_text_orders_transcript_then_previous_then_screen_then_ask() -> None:
    out = prompts.user_text("spoke earlier", "def foo(): pass", previous="def old(): pass", focus="")
    assert (
        out.index("<transcript>") < out.index("<previous>") < out.index("<screen>")
        < out.index("Walk me through")
    )
    assert "spoke earlier" in out and "def old(): pass" in out and "def foo" in out


def test_user_text_omits_previous_block_when_blank() -> None:
    out = prompts.user_text("spoke earlier", "def foo(): pass", previous="", focus="")
    assert "<previous>" not in out


def test_user_text_omits_previous_block_when_only_whitespace() -> None:
    out = prompts.user_text("spoke earlier", "", previous="   \n  ", focus="")
    assert "<previous>" not in out


def test_user_text_blank_transcript_placeholder() -> None:
    out = prompts.user_text("", "", previous="", focus="")
    assert "(nothing said since the checkpoint)" in out


def test_user_text_blank_screen_placeholder() -> None:
    out = prompts.user_text("something", "", previous="", focus="")
    assert "(nothing readable on screen)" in out


def test_user_text_default_ask_ends_with_output_the_walkthrough_now() -> None:
    out = prompts.user_text("", "", focus="")
    assert out.rstrip().endswith("Output the walkthrough now.")


def test_user_text_uses_focus_when_given_and_it_ends_the_prompt() -> None:
    out = prompts.user_text("", "code on screen", focus="Focus on the recursion only")
    assert out.rstrip().endswith("Focus on the recursion only")


def test_user_text_screen_placed_after_previous() -> None:
    out = prompts.user_text("transcript", "screen text here", previous="prior code", focus="")
    assert out.index("<previous>") < out.index("<screen>")
    assert out.index("prior code") < out.index("screen text here")

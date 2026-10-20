"""Prompt builders for the pair-programming co-pilot (2026-10-20): the insights (questions to ask,
blind spots & follow-ups) prompt, its parser / renderer, the recap prompt, the dev-session notes
template and the session-context line every system prompt can carry."""
from __future__ import annotations

import pytest

from exocortex.llm import prompts


# --------------------------------------------------------------------------------------------
# normalisation
# --------------------------------------------------------------------------------------------


def test_normalize_role_and_sections() -> None:
    assert prompts.normalize_role("Navigator ") == "navigator"
    assert prompts.normalize_role("pilot") == "driver"
    assert prompts.normalize_role(None) == "driver"
    assert prompts.normalize_sections(["blindspots", "questions", "questions", "bogus"]) == ("questions", "blindspots")
    assert prompts.normalize_sections("blindspots") == ("blindspots",)
    assert prompts.normalize_sections(None) == ()
    assert prompts.normalize_sections(42) == ()


# --------------------------------------------------------------------------------------------
# system_insights / user_insights
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "role, frame, audience",
    [
        ("driver", "The user is the DRIVER", "the navigator or the team"),
        ("navigator", "The user is the NAVIGATOR", "the driver"),
        ("participant", "The user is one participant", "the group"),
    ],
)
def test_system_insights_frames_the_role_and_addresses_its_questions(role: str, frame: str, audience: str) -> None:
    out = prompts.system_insights("", role)
    assert frame in out
    assert f"ask {audience} right now" in out


def test_system_insights_lists_only_the_requested_headings_in_display_order() -> None:
    both = prompts.system_insights("", "driver", ("blindspots", "questions"))
    assert both.index("## Questions to ask") < both.index("## Blind spots & follow-ups")
    only = prompts.system_insights("", "driver", ("blindspots",))
    assert "## Blind spots & follow-ups" in only and "## Questions to ask" not in only
    assert "## Questions to ask" in prompts.system_insights("", "driver", ("nope",))   # unknown: both


def test_system_insights_clamps_the_items_per_section() -> None:
    assert "up to 8 questions" in prompts.system_insights("", "driver", ("questions",), 99)
    assert "up to 1 questions" in prompts.system_insights("", "driver", ("questions",), 0)
    assert "up to 4 questions" in prompts.system_insights("", "driver", ("questions",), "lots")


def test_system_insights_carries_the_skip_marker_focus_rule_and_context_then_persona_last() -> None:
    out = prompts.system_insights("I am the tech lead", "driver", context="  FastAPI  billing\nservice ")
    assert prompts.SKIP_MARKER in out
    assert "Focus strictly on technical substance" in out
    lines = out.splitlines()
    assert lines[-2] == "Session context (what the team is working on): FastAPI billing service"
    assert lines[-1] == "About me: I am the tech lead"


def test_system_insights_is_byte_identical_for_a_fixed_config() -> None:
    """A system prompt that changed from turn to turn would defeat llama.cpp's KV prefix cache."""
    args = ("p", "navigator", ("questions",), 3, "ctx")
    assert prompts.system_insights(*args) == prompts.system_insights(*args)


def test_user_insights_layout_is_stable_first() -> None:
    out = prompts.user_insights(
        "[00:01] Sam: rate limit per tenant", "def limit():", "## Questions to ask\n- old?", ("questions",)
    )
    assert out.startswith("<transcript>\n[00:01] Sam: rate limit per tenant\n</transcript>")
    assert (out.index("<transcript>") < out.index("<previous>") < out.index("<screen>")
            < out.index("Update the questions to ask now"))
    assert "<previous>\n## Questions to ask\n- old?\n</previous>" in out


def test_user_insights_omits_an_empty_previous_and_placeholders_missing_input() -> None:
    out = prompts.user_insights("", "", "", ("questions", "blindspots"))
    assert "<previous>" not in out
    assert "(no speech yet)" in out and "(nothing readable on screen)" in out
    assert "Update the questions to ask and the blind spots & follow-ups now" in out


# --------------------------------------------------------------------------------------------
# parse_insights / render_insights / insight_sections_in
# --------------------------------------------------------------------------------------------


def test_parse_insights_splits_by_heading_and_cleans_the_items() -> None:
    text = (
        "## Questions to ask\n"
        "- Should `retry()` back off on 429s?\n"
        "* What's the SLA for the export job?\n"
        "- (none)\n"
        "## Notes to self\n"
        "- dropped: not a known section\n"
        "**Blind spots & follow-ups:**\n"
        "1. No test for an empty payload\n"
        "   which the parser crashes on\n"
        "2) TODO: loop in Sam about the migration\n"
    )
    assert prompts.parse_insights(text) == {
        "questions": "- Should `retry()` back off on 429s?\n- What's the SLA for the export job?",
        "blindspots": "- No test for an empty payload which the parser crashes on\n"
                      "- TODO: loop in Sam about the migration",
    }


def test_parse_insights_takes_a_headingless_single_section_answer_and_caps_it() -> None:
    text = "\n".join(f"- item {i}" for i in range(12))
    assert prompts.parse_insights(text, ("blindspots",), max_items=3) == {"blindspots": "- item 0\n- item 1\n- item 2"}


def test_parse_insights_placeholder_only_sections_come_back_empty() -> None:
    out = prompts.parse_insights("## Questions to ask\n- (none)\n## Blind spots & follow-ups\n- None.")
    assert out == {"questions": "", "blindspots": ""}


def test_parse_insights_returns_only_the_requested_sections() -> None:
    text = "## Questions to ask\n- q\n## Blind spots & follow-ups\n- b"
    assert prompts.parse_insights(text, ("questions",)) == {"questions": "- q"}


def test_render_insights_skips_empty_sections_or_writes_none() -> None:
    assert prompts.render_insights({"questions": "", "blindspots": "- b"}) == "## Blind spots & follow-ups\n- b"
    assert prompts.render_insights({}) == ""
    assert prompts.render_insights({"questions": "- q"}, skip_empty=False) == (
        "## Questions to ask\n- q\n\n## Blind spots & follow-ups\n- (none)"
    )


def test_render_then_parse_round_trips() -> None:
    doc = {"questions": "- a?\n- b?", "blindspots": "- c"}
    assert prompts.parse_insights(prompts.render_insights(doc)) == doc


def test_insight_sections_in_detects_the_headings_streamed_so_far() -> None:
    assert prompts.insight_sections_in("") == ()
    assert prompts.insight_sections_in("## Questions to ask\n- a") == ("questions",)
    assert prompts.insight_sections_in("**Follow-ups**\n- x\n## Questions\n") == ("questions", "blindspots")


# --------------------------------------------------------------------------------------------
# recap
# --------------------------------------------------------------------------------------------


def test_system_recap_asks_for_the_four_parts_and_carries_the_context() -> None:
    out = prompts.system_recap("", "billing service")
    for label in ("**Goal:**", "**Done:**", "**Now:**", "**Next:**"):
        assert label in out
    assert prompts.SKIP_MARKER in out
    assert out.splitlines()[-1] == "Session context (what the team is working on): billing service"


def test_user_recap_layout_with_and_without_notes() -> None:
    out = prompts.user_recap("[00:05] Me: done with the parser", "## Summary\nParser rewrite.", "parser.py")
    assert out.index("<transcript>") < out.index("<notes>") < out.index("<screen>") < out.index("Recap the session now")
    bare = prompts.user_recap("", "", "")
    assert "<notes>" not in bare and "<screen>" not in bare and "(no speech yet)" in bare
    assert prompts.user_recap("t", focus="Just the decisions.").endswith("Just the decisions.")


# --------------------------------------------------------------------------------------------
# notes templates
# --------------------------------------------------------------------------------------------


def test_notes_templates_share_the_five_headings_and_meeting_is_the_original() -> None:
    headings = ["## Summary", "## Key points", "## Decisions", "## Action items", "## Open questions"]
    meeting, dev = prompts.system_notes("meeting"), prompts.system_notes("dev")
    for out in (meeting, dev):
        positions = [out.index(h) for h in headings]
        assert positions == sorted(positions)
        assert "- [ ] owner: task (due)" in out
        assert out.endswith("Output only the markdown document itself, nothing before or after it.")
    assert prompts.system_notes() == meeting            # the no-argument call keeps the original prompt
    assert "working session of software engineers" in dev
    assert "working session of software engineers" not in meeting
    assert prompts.normalize_notes_template("DEV") == "dev" and prompts.normalize_notes_template("x") == "dev"


# --------------------------------------------------------------------------------------------
# the session context in the existing system prompts
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "build",
    [
        lambda c: prompts.system_assist("", "bullets", c),
        lambda c: prompts.system_ask("", c),
        lambda c: prompts.system_suggest("", c),
        lambda c: prompts.system_solve("", context=c),
        lambda c: prompts.system_code("", context=c),
        lambda c: prompts.system_text("", c),
        lambda c: prompts.system_explain("", c),
    ],
)
def test_session_context_line_appears_only_when_set(build) -> None:
    assert build("") == build("   ")
    assert "Session context" not in build("")
    assert "Session context (what the team is working on): payments API" in build("payments API")

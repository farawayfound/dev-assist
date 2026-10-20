"""Tests for exocortex.llm.prompts and exocortex.llm.questions. Fully offline, no `live` marker."""
from __future__ import annotations

import pytest

from exocortex.llm import prompts, questions

# --------------------------------------------------------------------------------------------
# prompts.py -- ChatML rendering
# --------------------------------------------------------------------------------------------


def test_render_chatml_exact_format() -> None:
    result = prompts.render_chatml("SYS", "USR")
    assert result == (
        "<|im_start|>system\nSYS<|im_end|>\n"
        "<|im_start|>user\nUSR<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def test_render_chatml_with_assistant_prefix() -> None:
    result = prompts.render_chatml("SYS", "USR", "partial answer")
    assert result == (
        "<|im_start|>system\nSYS<|im_end|>\n"
        "<|im_start|>user\nUSR<|im_end|>\n"
        "<|im_start|>assistant\npartial answer"
    )


def test_render_chatml_default_assistant_prefix_is_empty() -> None:
    result = prompts.render_chatml("SYS", "USR")
    assert result.endswith("<|im_start|>assistant\n")


def test_no_think_prefix_value() -> None:
    assert prompts.NO_THINK_PREFIX == "<think>\n\n</think>\n\n"


def test_no_think_prefix_yields_documented_suffix() -> None:
    result = prompts.render_chatml("SYS", "USR", prompts.NO_THINK_PREFIX)
    assert result.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")


def test_stop_list() -> None:
    assert prompts.STOP == ["<|im_end|>", "<|im_start|>"]


def test_im_markers() -> None:
    assert prompts.IM_START == "<|im_start|>"
    assert prompts.IM_END == "<|im_end|>"


# --------------------------------------------------------------------------------------------
# prompts.py -- approx_tokens
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("", 1),
        ("abc", 1),
        ("abcd", 2),
        ("a" * 8, 3),
        ("a" * 400, 101),
    ],
)
def test_approx_tokens(text: str, expected: int) -> None:
    assert prompts.approx_tokens(text) == expected


# --------------------------------------------------------------------------------------------
# prompts.py -- system / user prompt builders
# --------------------------------------------------------------------------------------------


def test_system_assist_includes_persona_when_present() -> None:
    out = prompts.system_assist("I am a backend engineer.", "bullets")
    assert "About me: I am a backend engineer." in out


def test_system_assist_omits_persona_when_empty() -> None:
    out = prompts.system_assist("", "bullets")
    assert "About me:" not in out


def test_system_assist_instructs_against_mentioning_ai() -> None:
    for style in ("bullets", "short", "detailed"):
        out = prompts.system_assist("", style).lower()
        assert "never say" in out and "as an ai" in out


def test_system_assist_forbids_markdown_headers() -> None:
    out = prompts.system_assist("", "bullets")
    assert "markdown headers" in out.lower()


def test_system_assist_first_person_copilot_framing() -> None:
    out = prompts.system_assist("", "bullets").lower()
    assert "as if you were the user" in out or "first person" in out


def test_system_assist_style_bullets() -> None:
    out = prompts.system_assist("", "bullets")
    assert "2 to 4" in out
    assert "20 words" in out


def test_system_assist_style_short() -> None:
    out = prompts.system_assist("", "short")
    assert "1 to 2 sentences" in out


def test_system_assist_style_detailed() -> None:
    out = prompts.system_assist("", "detailed")
    assert "up to 8" in out
    assert "Bottom line" in out


def test_system_assist_unknown_style_falls_back_to_bullets() -> None:
    out = prompts.system_assist("", "nonsense-style")
    assert "2 to 4" in out


def test_system_assist_mentions_checking_instead_of_inventing() -> None:
    out = prompts.system_assist("", "bullets")
    assert "I'd need to check" in out


def test_system_ask_includes_persona_when_present() -> None:
    out = prompts.system_ask("I lead the platform team.")
    assert "About me: I lead the platform team." in out


def test_system_ask_omits_persona_when_empty() -> None:
    out = prompts.system_ask("")
    assert "About me:" not in out


def test_system_ask_instructs_against_mentioning_ai() -> None:
    out = prompts.system_ask("").lower()
    assert "never say" in out and "as an ai" in out


def test_system_notes_has_exact_headings_in_order() -> None:
    out = prompts.system_notes()
    headings = ["## Summary", "## Key points", "## Decisions", "## Action items", "## Open questions"]
    positions = [out.index(h) for h in headings]
    assert positions == sorted(positions)
    for h in headings:
        assert h in out


def test_system_notes_mentions_action_item_format() -> None:
    out = prompts.system_notes()
    assert "- [ ] owner: task (due)" in out


def test_user_assist_wraps_transcript_and_question() -> None:
    out = prompts.user_assist("Alice: hi\nBob: hello", "What's the plan?")
    assert "<transcript>\nAlice: hi\nBob: hello\n</transcript>" in out
    assert "<question>\nWhat's the plan?\n</question>" in out
    # transcript (stable) must come before the question (volatile) for KV-cache reuse.
    assert out.index("<transcript>") < out.index("<question>")


def test_user_ask_wraps_transcript_and_question() -> None:
    out = prompts.user_ask("Alice: hi", "What time is it?")
    assert "<transcript>\nAlice: hi\n</transcript>" in out
    assert "<question>\nWhat time is it?\n</question>" in out
    assert out.index("<transcript>") < out.index("<question>")


def test_user_assist_transcript_prefix_stable_across_growth() -> None:
    """Appending to the transcript must not change the bytes that came before it (KV-cache reuse)."""
    turn1 = prompts.user_assist("line1", "Q1?")
    turn2 = prompts.user_assist("line1\nline2", "Q2?")
    common = "<transcript>\nline1"
    assert turn1.startswith(common)
    assert turn2.startswith(common)


def test_user_notes_merges_previous_and_new() -> None:
    out = prompts.user_notes("## Summary\nold stuff", "Alice: new info")
    assert "<previous_notes>\n## Summary\nold stuff\n</previous_notes>" in out
    assert "<new_transcript>\nAlice: new info\n</new_transcript>" in out


def test_user_notes_placeholder_when_no_previous_notes() -> None:
    out = prompts.user_notes("", "Alice: hi")
    assert "<previous_notes>" in out
    assert "(none yet" in out


# --------------------------------------------------------------------------------------------
# questions.py -- is_question (table-driven, >= 25 cases)
# --------------------------------------------------------------------------------------------

IS_QUESTION_CASES: list[tuple[str, int, bool]] = [
    # ends with "?" -> True regardless of phrasing, as long as min_words is met
    ("What do you think about Kafka?", 3, True),
    ("Do you have the numbers ready?", 3, True),
    ("Is that going to work for us?", 3, True),
    ("Can we push the deadline?", 3, True),
    ("How would you design a rate limiter?", 3, True),
    ("Can you walk me through how you'd design a rate limiter for the public API?", 3, True),
    # interrogative start, no trailing "?"
    ("What do you think about this approach", 3, True),
    ("So what do you think about it", 3, True),
    ("Can you walk me through how you would design a rate limiter", 3, True),
    ("Tell me about your experience with Kafka", 3, True),
    ("Could we get this done by Friday", 3, True),
    ("Would that work for the team", 3, True),
    ("Should we consider RabbitMQ instead", 3, True),
    ("Do you have the latest numbers", 3, True),
    ("Is there a plan for the migration", 3, True),
    ("Explain how the retry logic works", 3, True),
    ("Describe the failure mode you saw", 3, True),
    ("Walk me through the deployment process", 3, True),
    ("Okay so what's the timeline looking like", 3, True),
    ("Um, how long will that take", 3, True),
    ("And why did that happen", 3, True),
    # rhetorical / backchannel tags -> False (filtered by min_words)
    ("right?", 3, False),
    ("Right?", 3, False),
    ("you know?", 3, False),
    ("You know?", 3, False),
    # plain statements -> False
    ("We should use Kafka for this service", 3, False),
    ("I wonder if we should just go with Kafka", 3, False),
    ("Thanks for joining", 3, False),
    ("That sounds right to me", 3, False),
    ("I think we're aligned on this", 3, False),
    ("Let's also make sure someone owns the migration plan", 3, False),
    # backchannels -> False (too short)
    ("Yeah", 3, False),
    ("Sure", 3, False),
    ("Mm-hmm", 3, False),
    ("Okay", 3, False),
    # empty / whitespace -> False
    ("", 3, False),
    ("   ", 3, False),
    # min_words parameter changes the outcome
    ("What now", 3, False),
    ("What now", 2, True),
    ("What do you mean by that exactly?", 10, False),
    ("hi there friend", 3, False),
]


@pytest.mark.parametrize(("text", "min_words", "expected"), IS_QUESTION_CASES)
def test_is_question(text: str, min_words: int, expected: bool) -> None:
    assert questions.is_question(text, min_words) is expected


def test_is_question_default_min_words_is_three() -> None:
    assert questions.is_question("What now") is False
    assert questions.is_question("What is now") is True


# --------------------------------------------------------------------------------------------
# questions.py -- is_question: punctuation-only leading tokens (ASR false-start artifacts)
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "min_words", "expected"),
    [
        ("... what now", 2, True),
        ("-- what do you think about it", 3, True),
        ("— how long will that take", 3, True),  # em dash
        ("um, -- so what now", 2, True),
        ("well-known issue right now", 3, False),  # internal hyphen must NOT be stripped
        ("-- can co-pilot help with this", 3, True),  # leading dash stripped, internal hyphen kept
    ],
)
def test_is_question_ignores_punctuation_only_leading_tokens(
    text: str, min_words: int, expected: bool
) -> None:
    assert questions.is_question(text, min_words) is expected


# --------------------------------------------------------------------------------------------
# questions.py -- extract_question (>= 6 cases)
# --------------------------------------------------------------------------------------------


def test_extract_question_empty_list() -> None:
    assert questions.extract_question([]) == ""


def test_extract_question_single_sentence_segment() -> None:
    q = "What would you say is the main trade-off between Kafka and RabbitMQ for our use case?"
    assert questions.extract_question([q]) == q


def test_extract_question_includes_preceding_sentence_for_context() -> None:
    text = "That's a great point. What would you say is the main trade-off between Kafka and RabbitMQ?"
    assert questions.extract_question([text]) == text


def test_extract_question_ignores_older_segments() -> None:
    q = "What would you say is the main trade-off between Kafka and RabbitMQ for our use case?"
    segments = ["Today we need to decide on the message queue.", "Sure.", q]
    assert questions.extract_question(segments) == q


def test_extract_question_falls_back_to_newest_text_when_no_question() -> None:
    segments = [
        "What would you say is the main trade-off between Kafka and RabbitMQ?",
        "I can take that action item",
    ]
    assert questions.extract_question(segments) == "I can take that action item"


def test_extract_question_drops_trailing_non_question_sentence() -> None:
    text = "Quick question. What's the timeline? Anyway let's continue."
    assert questions.extract_question([text]) == "Quick question. What's the timeline?"


def test_extract_question_caps_at_400_chars() -> None:
    context = "Just so you have some background before I ask this important thing. "
    question = "What do you think about " + ("the extremely long trade-off analysis " * 15) + "?"
    combined = context + question
    result = questions.extract_question([combined])
    assert len(result) <= 400
    assert result == combined.strip()[:400]


def test_extract_question_last_of_conftest_truth_data() -> None:
    segments = [
        "I can take that action item",
        "Can you walk me through how you'd design a rate limiter for the public API?",
    ]
    assert questions.extract_question(segments) == segments[-1]


def test_extract_question_no_sentences_returns_empty_or_stripped() -> None:
    assert questions.extract_question([""]) == ""
    assert questions.extract_question(["   "]) == ""


def test_extract_question_question_is_first_sentence_no_wraparound() -> None:
    """q_idx == 0 must not wrap around to the LAST sentence via a negative index."""
    text = "What's the plan? Anyway, let's begin."
    assert questions.extract_question([text]) == "What's the plan?"


def test_extract_question_merges_two_consecutive_question_sentences() -> None:
    text = "Quick context. Are we done? What's next?"
    assert questions.extract_question([text]) == "Are we done? What's next?"


def test_extract_question_whitespace_newest_with_real_older_segment() -> None:
    """Only the newest element is examined; a blank newest segment yields "" even if an
    earlier segment in the list contains a real question (matches the documented,
    same-segment-only scoping of this function)."""
    assert questions.extract_question(["real question?", "   "]) == ""


# --------------------------------------------------------------------- solve / choices ----
def test_system_solve_is_autonomous_and_mentions_choice_marker() -> None:
    out = prompts.system_solve("")
    low = out.lower()
    assert "solve" in low
    assert "do not ask" in low
    assert prompts.CHOICE_PREFIX in out
    # it must not tell the model to punt with "I'd need to check" (that is the meeting-answer rule)
    assert "i'd need to check" not in low


def test_system_solve_includes_persona_when_present() -> None:
    assert "backend engineer" in prompts.system_solve("I am a backend engineer.")


def test_user_solve_orders_transcript_then_screen_then_ask() -> None:
    out = prompts.user_solve("spoke earlier", "def foo(): pass", focus="")
    assert out.index("<transcript>") < out.index("<screen>") < out.index("Solve whatever")
    assert "spoke earlier" in out and "def foo" in out


def test_user_solve_uses_focus_when_given() -> None:
    out = prompts.user_solve("", "code on screen", focus="Do it in Rust")
    assert out.rstrip().endswith("Do it in Rust")


def test_split_choices_extracts_choice_lines() -> None:
    body, choices = prompts.split_choices(
        "Here is one option or the other.\n"
        f"{prompts.CHOICE_PREFIX} Iterative\n"
        f"{prompts.CHOICE_PREFIX}: Recursive\n"
    )
    assert body == "Here is one option or the other."
    assert choices == ["Iterative", "Recursive"]


def test_split_choices_no_markers_returns_text_and_empty_list() -> None:
    body, choices = prompts.split_choices("just a full answer\nsecond line")
    assert body == "just a full answer\nsecond line"
    assert choices == []


def test_split_choices_dedupes_case_insensitively() -> None:
    _body, choices = prompts.split_choices(
        f"{prompts.CHOICE_PREFIX} Python\n{prompts.CHOICE_PREFIX} python\n{prompts.CHOICE_PREFIX} C++"
    )
    assert choices == ["Python", "C++"]


# --------------------------------------------------------------------- relevance (focus) ----
@pytest.mark.parametrize(
    "text, relevant",
    [
        ("Q - Have you seen this in an interview before?", False),
        ("Have you seen this question in a real interview before?", False),
        ("Did you like this problem? Rate it now", False),
        ("Was this helpful?", False),
        ("Want to see more? Sign up for Premium", False),
        ("Do you like working from home?", False),
        ("Have you worked with Kafka before?", True),
        ("How would you design a URL shortener at scale?", True),
        ("What is the time complexity of this approach?", True),
        ("Given an array of integers, return indices of the two numbers that add up to target?", True),
        ("Explain the CAP theorem.", True),
    ],
)
def test_is_relevant_default_focus(text: str, relevant: bool) -> None:
    assert questions.is_relevant(text) is relevant


def test_is_relevant_focus_all_and_custom_patterns() -> None:
    assert questions.is_relevant("Do you like working from home?", focus="all")
    assert not questions.is_relevant("Have you seen this in an interview before?", focus="all")   # polls are always out
    assert not questions.is_relevant("Tell me about your weekend plans", ignore_patterns=["weekend"])
    assert questions.is_relevant("What is 2+2", ignore_patterns=["(unclosed"])                    # bad regex is skipped


def test_skip_marker_helpers() -> None:
    assert prompts.is_skip("(nothing to do)") and prompts.is_skip(' "(Nothing to do)." ')
    assert not prompts.is_skip("nothing to do here: refactor x")
    assert "polls and surveys" in prompts.system_solve("") and prompts.SKIP_MARKER in prompts.system_solve("")


# ------------------------------------------------------------------- explanations prompts ----
def test_system_solve_explain_shape_and_explain_prompts() -> None:
    plain = prompts.system_solve("")
    assert "Walkthrough:" not in plain and "single fenced code block" in plain
    shaped = prompts.system_solve("I am a backend engineer.", explain=True, comments="minimal")
    assert "FIRST a walkthrough" in shaped and "\"Walkthrough:\"" in shaped
    assert "minimal but important comments" in shaped and "ONE blank line" in shaped
    assert "I am a backend engineer." in shaped and prompts.CHOICE_PREFIX in shaped
    assert "no comments at all" in prompts.system_solve("", explain=True, comments="none")
    system = prompts.system_explain("persona here")
    assert "follow-up question" in system and "persona here" in system and "never mention" in system.casefold()
    user = prompts.user_explain("- walk", "def f(): pass", "screen text")
    assert user.index("<walkthrough>") < user.index("<solution>") < user.index("<screen>")
    assert "def f(): pass" in user and user.endswith("Explain the solution above as instructed.")
    assert "(none)" in prompts.user_explain("", "code") and "(the code on the screen)" in prompts.user_explain("w", "")
    assert "<screen>" not in prompts.user_explain("w", "c", "")
    complete = prompts.system_complete("", "", explain=True)
    assert "<solution> block" in complete and "may skip typing comments" in complete
    assert "<solution>" not in prompts.system_complete("", "")

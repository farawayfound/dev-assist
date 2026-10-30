"""Tests for the 2026-10-06 context-library additions to exocortex.llm.prompts: the ranking rule in
every system prompt, the <reference>/<earlier>/<lookup> blocks in every user builder, the
byte-identical-when-empty guarantee, and the compaction / digest prompts. Pure string tests."""
from __future__ import annotations

import pytest

from exocortex.llm import prompts
from exocortex.llm.prompts import (
    _REFERENCE_RULE,
    context_blocks,
    lookup_block,
    system_ask,
    system_assist,
    system_code,
    system_compact,
    system_complete,
    system_digest,
    system_solve,
    system_suggest,
    system_text,
    user_ask,
    user_assist,
    user_code,
    user_compact,
    user_complete,
    user_digest,
    user_solve,
    user_suggest,
    user_text,
    words_for_tokens,
)

# ---------------------------------------------------------------------------
# the rule in the system prompts
# ---------------------------------------------------------------------------

_SYSTEM_BUILDERS = {
    "assist": lambda: system_assist("a persona", "bullets"),
    "ask": lambda: system_ask("a persona"),
    "suggest": lambda: system_suggest("a persona"),
    "solve": lambda: system_solve("a persona"),
    "solve_explain": lambda: system_solve("a persona", explain=True, comments="detailed", language="Go"),
    "code": lambda: system_code("a persona"),
    "text": lambda: system_text("a persona"),
}


@pytest.mark.parametrize("name", sorted(_SYSTEM_BUILDERS))
def test_system_prompt_carries_the_reference_rule_verbatim(name: str) -> None:
    assert _REFERENCE_RULE in _SYSTEM_BUILDERS[name]()


@pytest.mark.parametrize("name", sorted(_SYSTEM_BUILDERS))
def test_system_prompt_is_byte_identical_across_calls(name: str) -> None:
    # a system prompt that changed per turn would defeat the KV prefix cache
    assert _SYSTEM_BUILDERS[name]() == _SYSTEM_BUILDERS[name]()


def test_reference_rule_text_is_gated_by_wording_and_ranks_the_sources() -> None:
    rule = _REFERENCE_RULE
    assert "If a <reference> or <lookup> block is present" in rule
    assert "<transcript>" in rule and "<earlier>" in rule and "<lookup>" in rule
    assert "LAST lines" in rule and "most recent speech" in rule
    assert "reference only" in rule
    assert "never mention that it was provided" in rule
    assert "unprompted" in rule
    assert "override" in rule


def test_reference_rule_appears_exactly_once_per_system_prompt() -> None:
    for name, build in _SYSTEM_BUILDERS.items():
        assert build().count(_REFERENCE_RULE) == 1, name


def test_reference_rule_sits_right_after_the_screen_rule_in_assist_and_ask() -> None:
    for text in (system_assist("", "bullets"), system_ask("")):
        assert text.index(prompts._SCREEN_RULE) < text.index(_REFERENCE_RULE)
        assert text.index(_REFERENCE_RULE) == text.index(prompts._SCREEN_RULE) + len(prompts._SCREEN_RULE) + 1


def test_reference_rule_comes_before_the_focus_rule_elsewhere() -> None:
    for text in (system_suggest(""), system_solve(""), system_code(""), system_text("")):
        assert text.index(_REFERENCE_RULE) < text.index(prompts._FOCUS_RULE)


def test_system_complete_mentions_the_background_blocks_and_stays_stable() -> None:
    text = system_complete("p", hint="h", explain=True, language="Python", comments="normal")
    assert "<reference>" in text and "<earlier>" in text
    assert "ranks below the meeting transcript" in text
    assert "<transcript>" not in text      # the system prompt never names a block the user turn may lack
    assert text == system_complete("p", hint="h", explain=True, language="Python", comments="normal")


def test_system_prompts_do_not_depend_on_whether_a_library_is_loaded() -> None:
    # the builders take no library argument at all: nothing about the library can leak in
    import inspect

    for build in (system_assist, system_ask, system_suggest, system_solve, system_code, system_text):
        assert not [n for n in inspect.signature(build).parameters if "reference" in n or "lookup" in n]


# ---------------------------------------------------------------------------
# block renderers
# ---------------------------------------------------------------------------


def test_context_blocks_empty_is_empty() -> None:
    assert context_blocks() == ""
    assert context_blocks("", "") == ""
    assert context_blocks("  \n ", "\n") == ""


def test_context_blocks_renders_reference_then_earlier() -> None:
    assert context_blocks("R1", "E1") == "<reference>\nR1\n</reference>\n\n<earlier>\nE1\n</earlier>\n\n"


def test_context_blocks_omits_the_empty_one_and_strips() -> None:
    assert context_blocks("  R1 \n", "") == "<reference>\nR1\n</reference>\n\n"
    assert context_blocks("", "\nE1\n") == "<earlier>\nE1\n</earlier>\n\n"


def test_lookup_block_renders_strips_and_omits() -> None:
    assert lookup_block() == ""
    assert lookup_block("   ") == ""
    assert lookup_block("\n[doc > title]\ntext\n") == "<lookup>\n[doc > title]\ntext\n</lookup>\n\n"


# ---------------------------------------------------------------------------
# byte-identical when empty (the old renderings, spelled out by hand)
# ---------------------------------------------------------------------------

_SCREEN = "<screen>\nslide text\n</screen>\n\n"
_NO_SCREEN = "<screen>\n(nothing readable on screen)\n</screen>\n\n"


def test_user_assist_empty_blocks_are_byte_identical_to_the_old_output() -> None:
    old = (
        "<transcript>\nT\n</transcript>\n\n" + _SCREEN + "<choices>C</choices>\n"
        "The other participant just asked the question below. Answer it as instructed in the "
        "system prompt.\n<question>\nQ?\n</question>"
    )
    assert user_assist("T", "Q?", "slide text", "<choices>C</choices>\n") == old
    assert user_assist("T", "Q?", "slide text", "<choices>C</choices>\n", reference="", earlier="", lookup="") == old


def test_user_assist_without_screen_or_choices_is_byte_identical() -> None:
    old = (
        "<transcript>\nT\n</transcript>\n\n"
        "The other participant just asked the question below. Answer it as instructed in the "
        "system prompt.\n<question>\nQ?\n</question>"
    )
    assert user_assist("T", "Q?") == old


def test_user_ask_empty_blocks_are_byte_identical_to_the_old_output() -> None:
    old = (
        "<transcript>\nT\n</transcript>\n\n" + _SCREEN +
        "Answer the question below, using the transcript above (and the screen text, if any) for "
        "context where it's relevant.\n<question>\nQ?\n</question>"
    )
    assert user_ask("T", "Q?", "slide text") == old
    assert user_ask("T", "Q?", "slide text", reference="", earlier="", lookup="") == old


def test_user_suggest_empty_blocks_are_byte_identical_to_the_old_output() -> None:
    old_default = (
        "<transcript>\nT\n</transcript>\n\n" + _NO_SCREEN +
        "What should I say or do next? Keep it to the most useful 1-3 points."
    )
    assert user_suggest("T", "") == old_default
    assert user_suggest("T", "", reference="", earlier="", lookup="") == old_default
    assert user_suggest("", "slide text", "do X") == "<transcript>\n(no speech yet)\n</transcript>\n\n" + _SCREEN + "do X"


def test_user_solve_empty_blocks_are_byte_identical_to_the_old_output() -> None:
    old = "<transcript>\nT\n</transcript>\n\n" + _SCREEN + "<choices>C</choices>\nfocus"
    assert user_solve("T", "slide text", "focus", "<choices>C</choices>\n") == old
    assert user_solve("T", "slide text", "focus", "<choices>C</choices>\n", reference="", earlier="", lookup="") == old
    default_ask = (
        "<transcript>\n(no speech yet)\n</transcript>\n\n" + _NO_SCREEN +
        "Solve whatever task, problem, or question is shown on the screen above. "
        "Output the finished solution now."
    )
    assert user_solve("", "") == default_ask


def test_user_code_empty_blocks_are_byte_identical_to_the_old_output() -> None:
    old = (
        "<transcript>\nT\n</transcript>\n\n<previous>\nprev\n</previous>\n\n" + _SCREEN + "do it"
    )
    assert user_code("T", "slide text", "prev", "do it") == old
    assert user_code("T", "slide text", "prev", "do it", reference="", earlier="", lookup="") == old
    old_empty = (
        "<transcript>\n(nothing said since the checkpoint)\n</transcript>\n\n" + _NO_SCREEN + "do it"
    )
    assert user_code("", "", focus="do it") == old_empty


def test_user_text_empty_blocks_are_byte_identical_to_the_old_output() -> None:
    old = "<transcript>\nT\n</transcript>\n\n<previous>\nprev\n</previous>\n\n" + _SCREEN + "explain"
    assert user_text("T", "slide text", "prev", "explain") == old
    assert user_text("T", "slide text", "prev", "explain", reference="", earlier="", lookup="") == old
    assert user_text("T", "", focus="explain") == "<transcript>\nT\n</transcript>\n\n" + _NO_SCREEN + "explain"


def test_blank_whitespace_blocks_count_as_empty() -> None:
    plain = user_assist("T", "Q?", "slide text")
    assert user_assist("T", "Q?", "slide text", reference="  \n", earlier="\n\n", lookup="   ") == plain


# ---------------------------------------------------------------------------
# block order with every builder
# ---------------------------------------------------------------------------


def _positions(text: str, *markers: str) -> list[int]:
    out = []
    for marker in markers:
        assert text.count(marker) == 1, (marker, text)
        out.append(text.index(marker))
    return out


def _assert_increasing(text: str, *markers: str) -> None:
    pos = _positions(text, *markers)
    assert pos == sorted(pos), (markers, pos)


def test_user_assist_block_order() -> None:
    text = user_assist("T-LINE", "QUESTION", "SCREEN-TEXT", "<choices>\nopts\n</choices>\n\n",
                       reference="REF-TEXT", earlier="EARLIER-TEXT", lookup="LOOKUP-TEXT")
    _assert_increasing(text, "<reference>", "<earlier>", "<transcript>", "<screen>", "<lookup>", "<choices>",
                       "<question>")
    assert text.startswith("<reference>\nREF-TEXT\n</reference>\n\n<earlier>\nEARLIER-TEXT\n</earlier>\n\n<transcript>")


def test_user_ask_block_order() -> None:
    text = user_ask("T-LINE", "QUESTION", "SCREEN-TEXT", "<choices>\nopts\n</choices>\n\n",
                    reference="REF-TEXT", earlier="EARLIER-TEXT", lookup="LOOKUP-TEXT")
    _assert_increasing(text, "<reference>", "<earlier>", "<transcript>", "<screen>", "<lookup>", "<choices>",
                       "<question>")


def test_user_solve_block_order() -> None:
    text = user_solve("T-LINE", "SCREEN-TEXT", "THE-FOCUS", "<choices>\nopts\n</choices>\n\n",
                      reference="REF-TEXT", earlier="EARLIER-TEXT", lookup="LOOKUP-TEXT")
    _assert_increasing(text, "<reference>", "<earlier>", "<transcript>", "<screen>", "<lookup>", "<choices>",
                       "THE-FOCUS")


def test_user_suggest_block_order() -> None:
    text = user_suggest("T-LINE", "SCREEN-TEXT", "THE-FOCUS",
                        reference="REF-TEXT", earlier="EARLIER-TEXT", lookup="LOOKUP-TEXT")
    _assert_increasing(text, "<reference>", "<earlier>", "<transcript>", "<screen>", "<lookup>", "THE-FOCUS")


@pytest.mark.parametrize("builder", [user_code, user_text])
def test_user_code_and_text_block_order(builder) -> None:
    text = builder("T-LINE", "SCREEN-TEXT", "PREVIOUS-CODE", "THE-FOCUS",
                   reference="REF-TEXT", earlier="EARLIER-TEXT", lookup="LOOKUP-TEXT")
    _assert_increasing(text, "<reference>", "<earlier>", "<transcript>", "<previous>", "<screen>", "<lookup>",
                       "THE-FOCUS")


def test_lookup_only_leaves_the_stable_prefix_untouched() -> None:
    # the lookup is volatile: adding it must not change anything that precedes the screen block
    base = user_assist("T", "Q?", "S", reference="R", earlier="E")
    with_lookup = user_assist("T", "Q?", "S", reference="R", earlier="E", lookup="L")
    cut = base.index("<screen>")
    assert with_lookup[:cut] == base[:cut]


def test_reference_and_earlier_alone_do_not_move_the_transcript_text() -> None:
    plain = user_ask("T", "Q?", "S")
    with_blocks = user_ask("T", "Q?", "S", reference="R", earlier="E")
    assert with_blocks.endswith(plain)


def test_user_complete_is_unchanged_and_accepts_the_new_tags_as_ordinary_blocks() -> None:
    text = user_complete([("reference", "R"), ("earlier", "E"), ("transcript", "T"), ("lookup", "L")], "f", "")
    _assert_increasing(text, "<reference>", "<earlier>", "<transcript>", "<lookup>", "<field>")
    assert user_complete([], "f", "").count("(no session context yet)") == 1


# ---------------------------------------------------------------------------
# compact / digest
# ---------------------------------------------------------------------------


def test_words_for_tokens() -> None:
    assert prompts.SUMMARY_WORDS_PER_TOKEN == 0.75
    assert words_for_tokens(400) == 300
    assert words_for_tokens(1) == 1
    assert words_for_tokens(0) == 1


def test_user_compact_mentions_the_word_limit_and_carries_both_inputs() -> None:
    text = user_compact("PREV-SUMMARY", "[00:10] Ann: we ship Friday", 240)
    assert "240 words" in text
    assert "PREV-SUMMARY" in text and "we ship Friday" in text
    assert text.index("<previous_summary>") < text.index("<dropped_transcript>") < text.index("240 words")


def test_user_compact_without_a_previous_summary_says_so() -> None:
    text = user_compact("", "dropped", 100)
    assert "none yet" in text
    assert "100 words" in text


def test_system_compact_is_constant_and_states_the_rules() -> None:
    text = system_compact()
    assert text == system_compact()
    lowered = text.casefold()
    for needle in ("names", "numbers", "decisions", "open questions", "never invent", "only the summary"):
        assert needle in lowered, needle
    assert "word limit" in lowered


def test_user_digest_names_the_document_and_the_line_count() -> None:
    text = user_digest("architecture.md", "  # Architecture\nThreads...  ")
    assert 'name="architecture.md"' in text
    assert "# Architecture\nThreads..." in text
    assert "2 to 4 line" in text


def test_user_digest_blank_name_falls_back() -> None:
    assert 'name="untitled"' in user_digest("  ", "x")


def test_system_digest_is_constant_and_asks_for_two_to_four_lines() -> None:
    text = system_digest()
    assert text == system_digest()
    assert "2 to 4" in text
    assert "only the entry" in text.casefold()
    assert "topics" in text and "names" in text



def test_user_notes_word_limit_is_opt_in_and_appended_last() -> None:
    """2026-10-06: the notes job states a document length only when the notes lane is too small
    for the configured answer; without it the builder is byte-identical to the old output."""
    plain = prompts.user_notes("## Summary\nold", "[00:01] A: new")
    assert "Keep the whole document" not in plain
    assert plain == prompts.user_notes("## Summary\nold", "[00:01] A: new", None)
    limited = prompts.user_notes("## Summary\nold", "[00:01] A: new", 180)
    assert limited.startswith(plain)
    assert limited.endswith(" Keep the whole document under 180 words.")
    assert prompts.user_notes("", "", 0) == prompts.user_notes("", "")     # 0 = no limit


@pytest.mark.parametrize("raw,expected", [
    ("", ""),
    ("plain digest text", "plain digest text"),
    ("<think>\nmusing\n</think>\n\nthe answer", "the answer"),
    ("draft bullets\n</think>\n\nfinal bullets", "final bullets"),         # a lone close tag: the answer follows it
    ("the whole digest\n</think>\n", "the whole digest"),                  # ... unless nothing follows it
    ("<think>a</think>b<think>c</think>", "b"),
])
def test_strip_think_removes_reasoning_residue(raw: str, expected: str) -> None:
    """Measured live 2026-10-06 on gemma-4-E4B-it under the no-think prefix: digests and
    summaries came back with a stray </think>; stored verbatim they would pollute every prompt."""
    assert prompts.strip_think(raw) == expected

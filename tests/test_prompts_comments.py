"""Tests for the comment-level prompt layer added for the "comments in generated code" Assist
setting: `prompts.COMMENT_LEVELS` / `normalize_comment_level` / `_comment_rule` /
`_comment_rule_complete`, how they splice into `system_solve` / `system_code` / `system_complete`,
and the matching config plumbing (`AssistConfig.comments`, the removed `ExplainConfig.comments`
boolean, `load_config` / `to_toml`). Fully offline, no `live` marker -- see
tests/test_prompts_code.py for the sibling "code" prompt tests these were modelled on."""
from __future__ import annotations

import logging

import pytest

from exocortex.config import Config, ExplainConfig, load_config, to_toml
from exocortex.llm import prompts

# One level -> the phrase that uniquely identifies it in `_comment_rule`'s output (see that
# function's docstring: "Each level's text carries a distinctive phrase (tests key on them)").
_RULE_PHRASE = {
    "none": "no comments at all",
    "minimal": "minimal but important comments",
    "normal": "well commented",
    "detailed": "pseudo-code",
}

# Same idea for `_comment_rule_complete` (system_complete's rephrased version).
_COMPLETE_PHRASE = {
    "none": "never add comments",
    "minimal": "genuinely important step",
    "normal": "each logical step",
    "detailed": "most lines of code",
}

# Levels that ask for comments at all (i.e. everything except "none").
_COMMENTED_LEVELS = ("minimal", "normal", "detailed")


# --------------------------------------------------------------------------------------------
# COMMENT_LEVELS / normalize_comment_level
# --------------------------------------------------------------------------------------------


def test_comment_levels_order() -> None:
    assert prompts.COMMENT_LEVELS == ("none", "minimal", "normal", "detailed")


@pytest.mark.parametrize("level", prompts.COMMENT_LEVELS)
def test_normalize_comment_level_is_idempotent_for_every_level(level: str) -> None:
    assert prompts.normalize_comment_level(level) == level


def test_normalize_comment_level_ignores_case_and_whitespace() -> None:
    assert prompts.normalize_comment_level("Detailed") == "detailed"
    assert prompts.normalize_comment_level(" none ") == "none"
    assert prompts.normalize_comment_level("NORMAL") == "normal"
    assert prompts.normalize_comment_level("  Minimal\t") == "minimal"


def test_normalize_comment_level_maps_the_old_explain_comments_bool() -> None:
    """The removed `[explain].comments` boolean maps onto the two levels it used to switch
    between, so a leftover caller (or a hand-edited config) can never silently get the default
    instead of "no comments" / "minimal comments"."""
    assert prompts.normalize_comment_level(True) == "minimal"
    assert prompts.normalize_comment_level(False) == "none"


def test_normalize_comment_level_unknown_or_blank_falls_back_to_normal() -> None:
    assert prompts.normalize_comment_level("") == "normal"
    assert prompts.normalize_comment_level("bogus") == "normal"
    assert prompts.normalize_comment_level(None) == "normal"


# --------------------------------------------------------------------------------------------
# _comment_rule: distinctive phrase per level
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("level", prompts.COMMENT_LEVELS)
def test_comment_rule_distinctive_phrase(level: str) -> None:
    assert _RULE_PHRASE[level] in prompts._comment_rule(level)


def test_comment_rule_detailed_asks_for_requirements_and_a_comment_block() -> None:
    """The "detailed" level additionally asks for a pre-code block of requirements + pseudo-code,
    the way a real engineer tackles a coding challenge in the editor -- see the feature request."""
    rule = prompts._comment_rule("detailed")
    assert "pseudo-code" in rule
    assert "requirements" in rule
    assert "comment block" in rule


def test_comment_rule_none_forbids_comments_outright() -> None:
    rule = prompts._comment_rule("none")
    assert "no comments at all" in rule
    assert "own line" not in rule  # nothing to anchor -- there are no comments to place


@pytest.mark.parametrize("level", _COMMENTED_LEVELS)
def test_comment_rule_commented_levels_put_each_comment_on_its_own_line(level: str) -> None:
    assert "own line" in prompts._comment_rule(level)


def test_comment_rule_every_commented_level_forbids_trailing_comments() -> None:
    """`_comment_rule`'s own docstring says every commented level demands comments "never
    trailing at the end of a code line" (so exocortex.plan's completion-mode comment skipping /
    anchoring keeps working). "minimal" used to say only "alone on its own line" and was caught
    by the review: the phrase is load-bearing, so all three commented levels are checked."""
    for level in ("minimal", "normal", "detailed"):
        assert "never trailing at the end of a code line" in prompts._comment_rule(level), level
    assert "never trailing" not in prompts._comment_rule("none")   # nothing to keep off the line


# --------------------------------------------------------------------------------------------
# every level reaching system_solve / system_code, explain False and True
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("build", [prompts.system_solve, prompts.system_code])
@pytest.mark.parametrize("level", prompts.COMMENT_LEVELS)
@pytest.mark.parametrize("explain", [False, True])
def test_comment_level_phrase_reaches_solve_and_code(build, level: str, explain: bool) -> None:
    out = build("", explain=explain, comments=level)
    assert _RULE_PHRASE[level] in out


@pytest.mark.parametrize("build", [prompts.system_solve, prompts.system_code])
def test_explain_shape_still_has_walkthrough_and_blank_line_markers(build) -> None:
    """Turning on explanations must not disturb the plan shape `exocortex.plan.split_plan`
    parses, for every comment level."""
    for level in prompts.COMMENT_LEVELS:
        out = build("", explain=True, comments=level)
        assert "FIRST a walkthrough" in out
        assert "ONE blank line" in out


@pytest.mark.parametrize("build", [prompts.system_solve, prompts.system_code])
def test_no_explain_shape_has_no_walkthrough_markers(build) -> None:
    """Without `explain`, none of the walkthrough-shape wording should appear -- note "detailed"
    has its own, unrelated "ONE blank line" between its pre-code comment block and the code, so
    that phrase alone is not a safe marker; "Walkthrough:" is unique to the explain shape."""
    for level in prompts.COMMENT_LEVELS:
        out = build("", explain=False, comments=level)
        assert "FIRST a walkthrough" not in out
        assert "\"Walkthrough:\"" not in out


# --------------------------------------------------------------------------------------------
# system_complete: _comment_rule_complete phrase per level, explain False and True
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("level", prompts.COMMENT_LEVELS)
@pytest.mark.parametrize("explain", [False, True])
def test_comment_rule_complete_phrase_reaches_system_complete(level: str, explain: bool) -> None:
    out = prompts.system_complete("", explain=explain, comments=level)
    assert _COMPLETE_PHRASE[level] in out


def test_may_skip_typing_comments_only_with_explain_true() -> None:
    without = prompts.system_complete("", explain=False)
    with_explain = prompts.system_complete("", explain=True)
    assert "may skip typing comments" not in without
    assert "may skip typing comments" in with_explain


def test_same_comments_in_the_plan_sentence_only_with_explain_true() -> None:
    """The explain-only sentence tells the continuation to follow the plan's <solution> exactly,
    "same code, same order, same names, same comments" -- absent when explanations are off."""
    without = prompts.system_complete("", explain=False)
    with_explain = prompts.system_complete("", explain=True)
    assert "same comments" not in without
    assert "same comments" in with_explain


def test_comment_rule_complete_always_present_regardless_of_explain() -> None:
    """Unlike the explain-only sentence, `_comment_rule_complete` is appended to
    system_complete unconditionally -- it governs comments in code typed OUTSIDE of a plan too."""
    for level in prompts.COMMENT_LEVELS:
        for explain in (False, True):
            out = prompts.system_complete("", explain=explain, comments=level)
            assert prompts._comment_rule_complete(level) in out


# --------------------------------------------------------------------------------------------
# default level, byte-identical calls, KV-cache stability across levels
# --------------------------------------------------------------------------------------------


def test_default_comment_level_is_normal() -> None:
    assert prompts.system_solve("") == prompts.system_solve("", comments="normal")
    assert prompts.system_code("") == prompts.system_code("", comments="normal")
    assert prompts.system_complete("") == prompts.system_complete("", comments="normal")


@pytest.mark.parametrize("build", [prompts.system_solve, prompts.system_code])
def test_system_prompt_byte_identical_across_calls_same_level(build) -> None:
    a = build("persona", explain=True, comments="detailed", language="Go")
    b = build("persona", explain=True, comments="detailed", language="Go")
    assert a == b


@pytest.mark.parametrize("build", [prompts.system_solve, prompts.system_code])
def test_system_prompt_differs_across_comment_levels(build) -> None:
    """A stable config must reuse the KV cache turn to turn, but a DIFFERENT `comments` setting
    (a real config change) must produce a genuinely different prompt."""
    outputs = {level: build("persona", explain=False, comments=level) for level in prompts.COMMENT_LEVELS}
    assert len(set(outputs.values())) == len(prompts.COMMENT_LEVELS)


def test_system_complete_differs_across_comment_levels() -> None:
    outputs = {level: prompts.system_complete("persona", comments=level) for level in prompts.COMMENT_LEVELS}
    assert len(set(outputs.values())) == len(prompts.COMMENT_LEVELS)


# --------------------------------------------------------------------------------------------
# config plumbing: AssistConfig.comments, the removed ExplainConfig.comments, load/save
# --------------------------------------------------------------------------------------------


def test_config_assist_comments_defaults_to_normal_and_is_a_known_level() -> None:
    cfg = Config()
    assert cfg.assist.comments == "normal"
    assert cfg.assist.comments in prompts.COMMENT_LEVELS


def test_explain_config_has_no_comments_attribute() -> None:
    """The old `ExplainConfig.comments` boolean was removed in favour of `AssistConfig.comments`
    (one setting shared by solve/code/completions, not a separate explain-only toggle)."""
    import dataclasses

    field_names = {f.name for f in dataclasses.fields(ExplainConfig)}
    assert "comments" not in field_names
    assert not hasattr(ExplainConfig(), "comments")


def test_load_config_warns_on_removed_explain_comments_key_but_still_loads(tmp_path, caplog) -> None:
    """A config.toml left over from before the [explain].comments -> [assist].comments move must
    not crash load_config -- unknown keys are dropped with a warning, same as any other stale key."""
    p = tmp_path / "config.toml"
    p.write_text('[explain]\ncomments = true\n', encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        cfg = load_config(p)
    assert "unknown key" in caplog.text.lower()
    assert "[explain].comments" in caplog.text
    # the rest of [explain] still loads with its defaults -- only the stale key was dropped
    assert cfg.explain == ExplainConfig()


def test_load_config_assist_comments_loads_raw_string_for_normalize_to_map(tmp_path) -> None:
    """config.py stores whatever raw string is in the TOML file (it does no validation of its
    own); `prompts.normalize_comment_level` is the single place that maps a hand-typed value like
    "Detailed" onto a real level."""
    p = tmp_path / "config.toml"
    p.write_text('[assist]\ncomments = "Detailed"\n', encoding="utf-8")
    cfg = load_config(p)
    assert cfg.assist.comments == "Detailed"
    assert prompts.normalize_comment_level(cfg.assist.comments) == "detailed"


def test_to_toml_default_config_has_comments_and_text_max_tokens() -> None:
    out = to_toml(Config())
    assert 'comments = "normal"' in out
    assert "text_max_tokens = 400" in out

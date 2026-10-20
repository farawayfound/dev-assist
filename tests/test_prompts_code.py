"""Tests for the primary-language rule (`prompts._language_rule`) and the "code" prompt builders
(`prompts.system_code` / `prompts.user_code`) added for the Code button + checkpoints. Fully
offline, no `live` marker -- see tests/test_prompts_questions.py for the rest of prompts.py."""
from __future__ import annotations

from exocortex.llm import prompts

# --------------------------------------------------------------------------------------------
# _language_rule / language plumbing shared by system_solve, system_code, system_complete
# --------------------------------------------------------------------------------------------


def test_language_rule_names_the_fallback_and_forbids_asking() -> None:
    out = prompts._language_rule("Python")
    assert "Python" in out
    assert "never ask which language" in out.lower()


def test_language_rule_uses_the_given_language_as_the_only_fallback() -> None:
    out = prompts._language_rule("Rust")
    assert "Rust" in out and "Python" not in out


def test_language_rule_blank_falls_back_to_python_not_a_broken_sentence() -> None:
    """A cleared `assist.primary_language` Settings field (or a hand-edited config.toml) saves as
    `""`; `_language_rule` must not splice that straight into the sentence ("...use . Never
    ask...") -- it needs a real fallback language for exactly the case ("none is indicated") this
    rule exists to resolve."""
    for blank in ("", "   ", "\t\n"):
        out = prompts._language_rule(blank)
        assert "use ." not in out
        assert "use  " not in out
        assert "use Python" in out


def test_system_solve_default_language_is_python_and_includes_the_rule() -> None:
    out = prompts.system_solve("")
    assert "Python if none is indicated" in out
    assert "never ask which language" in out.lower()


def test_system_solve_custom_language_replaces_the_default_everywhere() -> None:
    out = prompts.system_solve("", language="Go")
    assert "Go if none is indicated" in out
    assert "use Go" in out
    assert "Python" not in out


def test_system_solve_fork_example_no_longer_mentions_language() -> None:
    """The language choice is resolved by `_language_rule`/`language` now, never a fork -- see the
    design note: "the fork example ... must drop the language case"."""
    out = prompts.system_solve("")
    assert "target language" not in out.lower()
    assert prompts.CHOICE_PREFIX in out          # the fork protocol itself is still there


def test_solve_shape_language_default_and_override() -> None:
    assert "Python if none is indicated" in prompts._solve_shape(False, True)
    assert "Rust if none is indicated" in prompts._solve_shape(False, True, "Rust")
    assert "Rust if none is indicated" in prompts._solve_shape(True, True, "Rust")


def test_solve_shape_blank_language_falls_back_to_python() -> None:
    assert "Python if none is indicated" in prompts._solve_shape(False, True, "")
    assert "( if none is indicated)" not in prompts._solve_shape(False, True, "")
    assert "Python if none is indicated" in prompts._solve_shape(True, True, "  ")


def test_system_complete_language_rule_phrased_for_completions() -> None:
    out = prompts.system_complete("")
    assert "Python" in out
    assert "never ask which language" in out.lower()
    custom = prompts.system_complete("", language="TypeScript")
    assert "TypeScript" in custom and "Python" not in custom


def test_system_complete_blank_language_falls_back_to_python() -> None:
    out = prompts.system_complete("", language="")
    assert "continue in Python" in out
    assert "continue in ;" not in out


# --------------------------------------------------------------------------------------------
# system_code
# --------------------------------------------------------------------------------------------


def test_system_code_frames_the_code_button_press() -> None:
    out = prompts.system_code("")
    low = out.lower()
    assert "\"code\"" in low
    assert "<transcript>" in out and "<screen>" in out and "<previous>" in out
    assert "do not ask" in low


def test_system_code_never_forks_no_choice_prefix() -> None:
    """Unlike system_solve, system_code must never carry the CHOICE_PREFIX fork protocol -- a
    Code press never forks (the language default always resolves it)."""
    for explain in (False, True):
        out = prompts.system_code("", explain=explain)
        assert prompts.CHOICE_PREFIX not in out


def test_system_code_includes_language_rule_and_skip_marker() -> None:
    out = prompts.system_code("", language="Kotlin")
    assert "Kotlin" in out
    assert prompts.SKIP_MARKER in out


def test_system_solve_and_system_code_blank_language_fall_back_to_python() -> None:
    """`system_solve`/`system_code` pass `cfg.assist.primary_language` straight through to
    `_solve_shape`/`_language_rule` with no guard of their own -- verify the shared fallback
    actually reaches them, not just the two helpers in isolation."""
    for build in (prompts.system_solve, prompts.system_code):
        out = build("", language="")
        assert "use ." not in out
        assert "( if none is indicated)" not in out
        assert "Python" in out


def test_system_code_includes_persona_when_present() -> None:
    assert "backend engineer" in prompts.system_code("I am a backend engineer.")


def test_system_code_explain_shape_matches_solve() -> None:
    plain = prompts.system_code("")
    assert "Walkthrough:" not in plain and "single fenced code block" in plain
    shaped = prompts.system_code("", explain=True, comments="minimal")
    assert "FIRST a walkthrough" in shaped and "\"Walkthrough:\"" in shaped
    assert "minimal but important comments" in shaped
    assert "no comments at all" in prompts.system_code("", explain=True, comments="none")


def test_system_code_never_mentions_being_an_ai() -> None:
    out = prompts.system_code("").lower()
    assert "never mention being an ai" in out


# --------------------------------------------------------------------------------------------
# user_code ordering / placeholders
# --------------------------------------------------------------------------------------------


def test_user_code_orders_transcript_then_previous_then_screen_then_ask() -> None:
    out = prompts.user_code("spoke earlier", "def foo(): pass", previous="def old(): pass", focus="")
    assert (
        out.index("<transcript>") < out.index("<previous>") < out.index("<screen>")
        < out.index("Write the code")
    )
    assert "spoke earlier" in out and "def old(): pass" in out and "def foo" in out


def test_user_code_omits_previous_block_when_blank() -> None:
    out = prompts.user_code("spoke earlier", "def foo(): pass", previous="", focus="")
    assert "<previous>" not in out


def test_user_code_omits_previous_block_when_only_whitespace() -> None:
    out = prompts.user_code("spoke earlier", "", previous="   \n  ", focus="")
    assert "<previous>" not in out


def test_user_code_blank_transcript_placeholder() -> None:
    out = prompts.user_code("", "", previous="", focus="")
    assert "(nothing said since the checkpoint)" in out


def test_user_code_blank_screen_placeholder() -> None:
    out = prompts.user_code("something", "", previous="", focus="")
    assert "(nothing readable on screen)" in out


def test_user_code_default_ask_mentions_previous_code_and_change() -> None:
    out = prompts.user_code("", "", focus="")
    assert out.rstrip().endswith("Output the finished code now.")
    assert "change to the previous code" in out


def test_user_code_uses_focus_when_given() -> None:
    out = prompts.user_code("", "code on screen", focus="Now handle duplicates")
    assert out.rstrip().endswith("Now handle duplicates")


# --------------------------------------------------------------------------------------------
# KV-cache stability: the system prompt must be byte-identical across calls for a fixed config
# --------------------------------------------------------------------------------------------


def test_system_solve_is_byte_identical_across_calls_for_the_same_config() -> None:
    a = prompts.system_solve("persona", explain=True, comments="normal", language="Python")
    b = prompts.system_solve("persona", explain=True, comments="normal", language="Python")
    assert a == b


def test_system_code_is_byte_identical_across_calls_for_the_same_config() -> None:
    a = prompts.system_code("persona", explain=False, comments="detailed", language="Java")
    b = prompts.system_code("persona", explain=False, comments="detailed", language="Java")
    assert a == b


def test_system_complete_is_byte_identical_across_calls_for_the_same_config() -> None:
    a = prompts.system_complete("persona", "hint", explain=True, language="C++")
    b = prompts.system_complete("persona", "hint", explain=True, language="C++")
    assert a == b


def test_system_solve_changes_only_when_the_language_setting_changes() -> None:
    a = prompts.system_solve("persona", language="Python")
    b = prompts.system_solve("persona", language="Python")
    c = prompts.system_solve("persona", language="Rust")
    assert a == b
    assert a != c

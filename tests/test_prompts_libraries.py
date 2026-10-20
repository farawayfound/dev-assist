"""Tests for the library-level prompt layer added for the "use libraries in generated code"
Assist setting: `prompts.LIBRARY_LEVELS` / `normalize_library_level` / `_library_rule`, and how
they splice into `system_solve` / `system_code`. Fully offline, no `live` marker -- modelled on
tests/test_prompts_comments.py (the sibling "comments" prompt tests) and
tests/test_prompts_code.py (the language-rule tests, for the "no Python leaks in" pattern).
"""
from __future__ import annotations

import pytest

from exocortex.llm import prompts

# --------------------------------------------------------------------------------------------
# LIBRARY_LEVELS / normalize_library_level
# --------------------------------------------------------------------------------------------


def test_library_levels_value() -> None:
    assert prompts.LIBRARY_LEVELS == ("none", "basic", "all")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("none", "none"),
        ("basic", "basic"),
        ("all", "all"),
        ("BASIC", "basic"),
        ("All", "all"),
        ("NONE", "none"),
        (" basic ", "basic"),
        (None, "basic"),
        ("", "basic"),
        ("weird", "basic"),
        ("everything", "basic"),
    ],
)
def test_normalize_library_level(value: object, expected: str) -> None:
    assert prompts.normalize_library_level(value) == expected


# --------------------------------------------------------------------------------------------
# _library_rule: per-level content
# --------------------------------------------------------------------------------------------


def test_library_rule_none_forbids_imports_and_names_no_library() -> None:
    out = prompts._library_rule("none")
    low = out.lower()
    assert "no import" in low
    assert "no library call" in low
    # third-party packages (the "all" level's own vocabulary) must not appear in the "none" rule
    for lib in ("third-party", "requests", "lodash"):
        assert lib not in low


def test_library_rule_basic_names_standard_library_only() -> None:
    out = prompts._library_rule("basic")
    low = out.lower()
    assert "standard library" in low
    assert "third-party" in low  # "... but no third-party package"
    assert "no third-party package" in low


def test_library_rule_all_allows_third_party_packages() -> None:
    out = prompts._library_rule("all")
    low = out.lower()
    assert "third-party" in low
    assert "third-party packages included" in low
    assert "numpy" in low or "pandas" in low


def test_library_rule_unknown_value_falls_back_to_basic() -> None:
    assert prompts._library_rule("weird") == prompts._library_rule("basic")
    assert prompts._library_rule(None) == prompts._library_rule("basic")


@pytest.mark.parametrize("level", ["none", "basic", "all"])
def test_library_rule_every_level_says_follow_the_task_and_redo_on_change(level: str) -> None:
    """Every level is a SOFT default: the stable phrases the design calls out must appear
    regardless of which default is configured (assert on the phrases, not the example lists)."""
    out = prompts._library_rule(level)
    assert "follow what the task says" in out
    assert "Only when nothing is indicated" in out
    assert "ask changes" in out.lower()


# --------------------------------------------------------------------------------------------
# _library_rule is present in system_solve and system_code, for every level
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("build", [prompts.system_solve, prompts.system_code])
@pytest.mark.parametrize("level", ["none", "basic", "all"])
def test_library_rule_present_in_solve_and_code(build, level: str) -> None:
    out = build("", libraries=level)
    assert "follow what the task says" in out
    assert "Only when nothing is indicated" in out


def test_system_solve_default_libraries_is_basic() -> None:
    out = prompts.system_solve("")
    assert out == prompts.system_solve("", libraries="basic")
    assert "standard library" in out.lower()


def test_system_code_default_libraries_is_basic() -> None:
    out = prompts.system_code("")
    assert out == prompts.system_code("", libraries="basic")
    assert "standard library" in out.lower()


def test_system_solve_libraries_none_forbids_imports() -> None:
    out = prompts.system_solve("", libraries="none")
    assert "no import" in out.lower()


def test_system_code_libraries_all_allows_third_party() -> None:
    out = prompts.system_code("", libraries="all")
    assert "third-party packages included" in out.lower()


# --------------------------------------------------------------------------------------------
# a custom language must never make "Python" appear anywhere in the prompt -- the "basic"
# library default names no language, so this is a clean, direct check of that combination.
# --------------------------------------------------------------------------------------------


def test_system_solve_custom_language_basic_libraries_no_python_leak() -> None:
    out = prompts.system_solve("", language="Go", libraries="basic")
    assert "Python" not in out
    assert "Go" in out


def test_system_code_custom_language_every_library_level_no_python_leak() -> None:
    for level in prompts.LIBRARY_LEVELS:
        out = prompts.system_code("", language="Go", libraries=level)
        assert "Python" not in out


def test_system_solve_custom_language_every_library_level_no_python_leak() -> None:
    for level in prompts.LIBRARY_LEVELS:
        out = prompts.system_solve("", language="Rust", libraries=level)
        assert "Python" not in out


# --------------------------------------------------------------------------------------------
# byte-identical across calls (KV-cache stability), same shape as the language/comments tests
# --------------------------------------------------------------------------------------------


def test_system_solve_libraries_byte_identical_across_calls() -> None:
    a = prompts.system_solve("persona", libraries="all")
    b = prompts.system_solve("persona", libraries="all")
    assert a == b


def test_system_solve_changes_only_when_the_library_setting_changes() -> None:
    a = prompts.system_solve("persona", libraries="basic")
    b = prompts.system_solve("persona", libraries="basic")
    c = prompts.system_solve("persona", libraries="none")
    assert a == b
    assert a != c

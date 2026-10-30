"""The 2026-10-29 multi-file additions to exocortex.llm.prompts: the <workspace> block and its rule
in the Code / Text builders, the code-only answer shape of `system_code`, and the byte-identical
defaults (no keyword = the prompt as it was). Pure string tests."""
from __future__ import annotations

from exocortex.llm import prompts
from exocortex.llm.prompts import system_code, system_text, user_code, user_text, workspace_block

WS = "[models.py]\nclass User:\n    user_id: int\n\n[repo.py]\nclass UserRepository:\n    pass"


def test_defaults_render_as_before() -> None:
    assert system_code("p") == system_code("p", workspace=None, code_only=None)
    assert system_text("p") == system_text("p", workspace=None)
    assert "<workspace>" not in system_code("p") and "<workspace>" not in system_text("p")
    assert "Output ONLY code" not in system_code("p")
    assert user_code("T", "s", "prev", "do it") == user_code("T", "s", "prev", "do it", workspace="")
    assert user_text("T", "s", "", "") == user_text("T", "s", "", "", workspace="")
    assert "<workspace>" not in user_code("T", "s", "prev", "do it")


def test_workspace_block_helper() -> None:
    assert workspace_block("") == "" and workspace_block("   \n") == ""
    assert workspace_block("  x\ny ") == "<workspace>\nx\ny\n</workspace>\n\n"


def test_the_workspace_rule_in_the_code_and_text_system_prompts() -> None:
    code = system_code("", workspace=True)
    assert "<workspace> (when present) holds the OTHER files" in code
    assert "RIGHT NOW" in code and "never the parent or main object instead of it" in code
    # placed with the code rules: after the given-code lines, before the no-preamble line
    given = system_code("", workspace=True, consider_given_code=True)
    assert given.index("Given code:") < given.index("<workspace> (when present)") < given.index("No preamble")
    text = system_text("", workspace=True)
    assert "<workspace> (when present) holds the OTHER files" in text
    assert "rather than the main object" in text
    assert text.index("<workspace> (when present)") < text.index("Prose only")


def test_system_prompts_with_the_new_keywords_are_stable_across_calls() -> None:
    for build in (lambda: system_code("p", workspace=True, code_only=True),
                  lambda: system_code("p", explain=True, workspace=True, code_only=True),
                  lambda: system_text("p", workspace=True)):
        assert build() == build()


def test_code_only_shape_replaces_the_fenced_block_plus_complexity_line_shape() -> None:
    out = system_code("", code_only=True)
    assert "Output ONLY code: one fenced code block" in out
    assert "no complexity line outside the block" in out
    assert "comment lines in the language's own comment syntax" in out
    assert "no essay" not in out                                  # the old shape is gone
    assert "well commented" in out                                # the comment level still applies
    assert "Python if none is indicated" in out
    assert "Go if none is indicated" in system_code("", code_only=True, language="Go")
    assert "no comments at all" in system_code("", code_only=True, comments="none")


def test_code_only_with_explanations_keeps_the_plan_shape_and_bans_markdown_inside_the_block() -> None:
    out = system_code("", explain=True, code_only=True)
    assert "FIRST a walkthrough" in out and "Walkthrough:" in out       # the shape split_plan parses
    assert "never markdown (headings, **bold**, bullets)" in out
    assert "Output ONLY code" not in out


def test_user_code_renders_the_workspace_before_the_screen_and_extends_the_ask() -> None:
    out = user_code("T", "the screen", "", "", workspace=WS)
    assert f"<workspace>\n{WS}\n</workspace>\n\n<screen>\nthe screen\n</screen>\n\n" in out
    assert out.endswith("Output the finished code now. The screen shows the file I am in right now; the workspace "
                        "holds the other files of the same task. Write the code for the file on screen, consistent "
                        "with them.")
    with_prev = user_code("T", "the screen", "old code", "", workspace=WS)
    assert with_prev.index("<previous>") < with_prev.index("<workspace>") < with_prev.index("<screen>")
    assert with_prev.index("</transcript>") < with_prev.index("<previous>")


def test_user_code_workspace_sentence_follows_the_given_code_sentence_and_a_focus() -> None:
    out = user_code("T", "s", "", "fix the loop", workspace=WS, consider_given_code=True)
    given = prompts._given_code_ask("code", True)
    assert out.endswith(f"fix the loop{given}{prompts._WORKSPACE_ASKS['code']}")


def test_user_text_renders_the_workspace_the_same_way() -> None:
    out = user_text("T", "the screen", "", "", workspace=WS)
    assert f"<workspace>\n{WS}\n</workspace>\n\n<screen>\nthe screen\n</screen>\n\n" in out
    assert out.endswith("Walk through what the file on screen needs, in the context of them.")
    assert user_text("T", "s", "", "", workspace="  ") == user_text("T", "s", "", "")


def test_workspace_with_no_screen_text_still_renders_the_placeholder_screen() -> None:
    out = user_code("T", "", "", "", workspace=WS)
    assert "</workspace>\n\n<screen>\n(nothing readable on screen)\n</screen>" in out

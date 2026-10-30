"""Offline tests for exocortex.llm.codeformat (2026-10-29): a Code answer normalised to code only --
prose and the complexity line outside the fence become comments inside it, markdown inside the fence
becomes comments in the language's own syntax, fence-less code is fenced, prose is left alone, and with
explanations on only the inside of the block changes."""
from __future__ import annotations

from exocortex.llm import codeformat
from exocortex.llm.codeformat import clean_code_line, code_only, comment_syntax, language_tag, unmark

PY = "```python\ndef two_sum(nums, target):\n    seen = {}\n    for i, n in enumerate(nums):\n        if target - n in seen:\n            return [seen[target - n], i]\n        seen[n] = i\n```"


def test_clean_code_is_returned_unchanged() -> None:
    assert code_only(PY) == PY
    assert code_only("") == "" and code_only("   \n") == "   \n"


def test_text_outside_the_fence_becomes_comments_at_the_end_of_the_block() -> None:
    answer = "Here is the solution:\n\n" + PY + "\n\n**Time:** O(n), **Space:** O(n)\n"
    out = code_only(answer)
    assert out.startswith("```python\ndef two_sum") and out.endswith("```")
    assert out.count("```") == 2
    body = out.split("\n")
    assert body[-3:] == ["# Here is the solution:", "# Time: O(n), Space: O(n)", "```"]
    assert "**" not in out


def test_markdown_inside_the_block_becomes_comments_in_the_languages_syntax() -> None:
    answer = (
        "```java\n### Approach\n**Step 1:** use a map\n- iterate once\n"
        "public int[] twoSum(int[] nums, int target) {\n    Map<Integer, Integer> seen = new HashMap<>();\n"
        "    return new int[0];\n}\n```"
    )
    out = code_only(answer)
    lines = out.split("\n")
    assert lines[1] == "// Approach"
    assert lines[2] == "// Step 1: use a map"
    assert lines[3] == "// iterate once"
    assert lines[4].startswith("public int[] twoSum")          # code untouched
    assert "Map<Integer, Integer> seen = new HashMap<>();" in out


def test_python_headings_are_already_comments_and_stay() -> None:
    answer = "```python\n# Approach: one pass with a dict\ndef f(x):\n    return x\n```"
    assert code_only(answer) == answer
    assert clean_code_line("### Approach", "# ", "") == "### Approach"
    assert clean_code_line("### Approach", "// ", "") == "// Approach"


def test_bullets_that_are_code_are_kept() -> None:
    # a C-style "- x" line is rare; a bullet whose text is code-like must not be commented
    assert clean_code_line("- x = 1", "// ", "") == "- x = 1"
    assert clean_code_line("- use a hash map here", "// ", "") == "// use a hash map here"
    assert clean_code_line("**Note:** the input is sorted", "# ", "") == "# Note: the input is sorted"


def test_fence_less_code_is_fenced_with_the_primary_language_tag() -> None:
    raw = "def f(x):\n    return x * 2\n\nprint(f(2))"
    out = code_only(raw, language="Python")
    assert out == "```python\ndef f(x):\n    return x * 2\n\nprint(f(2))\n```"
    assert code_only("int main() {\n    return 0;\n}", language="C++").startswith("```cpp\n")


def test_fence_less_prose_is_left_alone() -> None:
    prose = "There is no code to write here: the question asks for a definition.\nA stack is LIFO."
    assert code_only(prose) == prose
    assert code_only("(nothing to do)") == "(nothing to do)"


def test_the_largest_block_is_the_code_and_a_snippet_outside_it_is_commented() -> None:
    answer = "```python\nprint(f(2))\n```\n\nFull solution:\n\n" + PY + "\n"
    out = code_only(answer)
    assert out.startswith("```python\ndef two_sum")
    assert "# print(f(2))" in out and "# Full solution:" in out
    assert out.count("```") == 2


def test_an_unclosed_block_is_closed() -> None:
    answer = "```python\ndef f():\n    return 1\n"
    out = code_only(answer)
    assert out == "```python\ndef f():\n    return 1\n```"


def test_explain_mode_keeps_the_walkthrough_and_the_complexity_line() -> None:
    answer = "Walkthrough:\nUse a dict.\nOne pass.\n\n```java\n**Code**\npublic int f() {\n    return 1;\n}\n```\nTime O(1), space O(1)"
    out = code_only(answer, explain=True)
    assert out.startswith("Walkthrough:\nUse a dict.\nOne pass.\n\n```java\n")
    assert "// Code\npublic int f() {" in out
    assert out.endswith("```\nTime O(1), space O(1)")


def test_explain_mode_without_a_fence_is_untouched() -> None:
    prose = "Walkthrough:\nreasoning\n\nThe answer is 42."
    assert code_only(prose, explain=True) == prose


def test_comment_syntax_by_tag_and_by_inference() -> None:
    assert comment_syntax("python") == ("# ", "")
    assert comment_syntax("sql") == ("-- ", "")
    assert comment_syntax("html") == ("<!-- ", " -->")
    assert comment_syntax("java") == ("// ", "")
    assert comment_syntax("", ["# a comment", "x = 1"]) == ("# ", "")
    assert comment_syntax("", ["// a comment", "int x = 1;"]) == ("// ", "")
    assert comment_syntax("", ["-- a comment", "SELECT 1"]) == ("-- ", "")
    assert comment_syntax("", ["#include <stdio.h>", "int main() {", "}"]) == ("// ", "")
    assert comment_syntax("", ["def f(x):", "    return x", "class A:"]) == ("# ", "")


def test_language_tag_and_unmark() -> None:
    assert language_tag("Python") == "python" and language_tag("C#") == "csharp" and language_tag("Go") == "go"
    assert language_tag("") == "" and language_tag("Visual Basic .NET") == ""
    assert unmark("### **Approach**: use `a map`") == "Approach: use a map"
    assert unmark("- item") == "item" and unmark("1. first") == "first" and unmark("---") == "" and unmark("> quoted") == "quoted"


def test_windows_line_endings_are_normalised() -> None:
    out = code_only("```python\r\nx = 1\r\n```\r\nTime O(1)\r\n")
    assert out == "```python\nx = 1\n\n# Time O(1)\n```"


def test_module_constants_cover_the_common_tags() -> None:
    assert "python" in codeformat._HASH_TAGS and "sql" in codeformat._DASH_TAGS and "html" in codeformat._HTML_TAGS

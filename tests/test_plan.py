"""Offline tests for exocortex.plan: parsing an explanations-mode solve answer, the comment-
skipping matcher, aligning typed text against the plan's solution, and the on-screen check."""
from __future__ import annotations

import pytest

from exocortex.plan import (
    PlanCursor,
    align_plan,
    consume_suggestion,
    containment,
    is_comment_line,
    plan_ghost,
    split_plan,
    squash,
    typed_fraction,
)

SOLUTION = (
    "class Solution:\n"
    "    def twoSum(self, nums, target):\n"
    "        # map each value to its index for O(1) lookups\n"
    "        seen = {}\n"
    "        for i, n in enumerate(nums):\n"
    "            # the partner we need for this element\n"
    "            need = target - n\n"
    "            if need in seen:\n"
    "                return [seen[need], i]\n"
    "            seen[n] = i\n"
    "        return []"
)
ANSWER = (
    "Walkthrough:\n"
    "- We need two indices whose values add up to target.\n"
    "- Key insight: one pass with a hash map of value -> index.\n"
    "- Edge cases: duplicates, no answer.\n"
    "\n"
    "```python\n" + SOLUTION + "\n```\n"
    "Time O(n), space O(n)."
)


# ------------------------------------------------------------------------- split_plan --------
def test_split_plan_complete_answer() -> None:
    parts = split_plan(ANSWER)
    assert parts.walkthrough.startswith("- We need two indices") and "Walkthrough" not in parts.walkthrough
    assert parts.solution == SOLUTION and parts.is_code and parts.language == "python"
    assert parts.after == "Time O(n), space O(n)." and not parts.code_open


def test_split_plan_while_streaming_keeps_only_complete_code_lines() -> None:
    cut = ANSWER.index("seen = {}") + 4                       # mid-way through a code line
    parts = split_plan(ANSWER[:cut])
    assert parts.code_open and parts.is_code
    assert parts.solution.endswith("# map each value to its index for O(1) lookups")
    assert "seen" not in parts.solution.split("\n")[-1]
    # the walkthrough alone (no fence yet) is the whole text so far
    early = split_plan(ANSWER[: ANSWER.index("```")])
    assert early.walkthrough.startswith("- We need") and early.solution == "" and not early.is_code


def test_split_plan_prose_answer_splits_at_first_blank_line() -> None:
    parts = split_plan("**Walkthrough:** the question asks for the CAP trade-off.\nPick two of three.\n\nAnswer: consistency and partition tolerance.")
    assert parts.walkthrough == "the question asks for the CAP trade-off.\nPick two of three."
    assert parts.solution == "Answer: consistency and partition tolerance." and not parts.is_code
    assert split_plan("").walkthrough == "" and split_plan("").solution == ""


# ------------------------------------------------------------------------- comments ----------
@pytest.mark.parametrize(
    "line, expected",
    [
        ("    # a comment", True), ("// c", True), ("  /* block */", True), (" * inside", True),
        ("-- sql comment", True), ("<!-- html -->", True),
        ("#include <stdio.h>", False), ("x = 1", False), ("a--", False), ("*args", False), ("", False),
    ],
)
def test_is_comment_line(line: str, expected: bool) -> None:
    assert is_comment_line(line) is expected


def test_consume_suggestion_skips_a_comment_the_user_types_past() -> None:
    sug = "# map values to indices\n        seen = {}\n        for i, n in enumerate(nums):"
    assert consume_suggestion(sug, "seen") is None                                  # default: a deviation
    assert consume_suggestion(sug, "seen", skip_comments=True) == " = {}\n        for i, n in enumerate(nums):"
    assert consume_suggestion(sug, "seen = {}\n", skip_comments=True) == "        for i, n in enumerate(nums):"
    assert consume_suggestion(sug, "# map values", skip_comments=True) == " to indices\n        seen = {}\n        for i, n in enumerate(nums):"
    assert consume_suggestion(sug, "sean", skip_comments=True) is None              # a real deviation still is one
    # a trailing comment on the same line is skipped too; a C preprocessor line is not a comment
    assert consume_suggestion("x = 1  # note\ny = 2", "x = 1\ny", skip_comments=True) == " = 2"
    assert consume_suggestion("#include <a>\nint x;", "int", skip_comments=True) is None


# ------------------------------------------------------------------------- alignment ---------
def test_align_nothing_typed_starts_at_the_top() -> None:
    cur = align_plan(SOLUTION, "")
    assert cur == PlanCursor(0, "class Solution:", False, 0)
    assert align_plan(SOLUTION, "   \n  ") == cur
    assert plan_ghost(SOLUTION, cur, 2) == "class Solution:\n    def twoSum(self, nums, target):"


def test_align_after_editor_header_shows_comment_then_code() -> None:
    prefix = "class Solution:\n    def twoSum(self, nums, target):\n        "
    cur = align_plan(SOLUTION, prefix)
    assert cur is not None and cur.line == 2 and cur.rest == "# map each value to its index for O(1) lookups"
    assert cur.score == 4 and not cur.at_end               # two exact anchors
    assert plan_ghost(SOLUTION, cur, 3) == (
        "# map each value to its index for O(1) lookups\n        seen = {}\n        for i, n in enumerate(nums):"
    )


def test_align_skipping_the_comment_and_mid_line() -> None:
    prefix = "class Solution:\n    def twoSum(self, nums, target):\n        seen ="
    cur = align_plan(SOLUTION, prefix)
    assert cur is not None and cur.line == 3 and cur.rest == " {}"
    cur2 = align_plan(SOLUTION, prefix + " {}")
    assert cur2 is not None and cur2.rest == "" and not cur2.at_end
    assert plan_ghost(SOLUTION, cur2, 3) == "\n        for i, n in enumerate(nums):\n            # the partner we need for this element"
    # a trailing space typed after "=" is not shown again in the ghost text
    cur3 = align_plan(SOLUTION, prefix + " ")
    assert cur3 is not None and cur3.rest == "{}"


def test_align_tolerates_an_annotated_signature_and_ocr_noise() -> None:
    prefix = "class Solution:\n    def twoSum(self, nums: List[int], target: int) -> List[int]:\n        seen = {}\n        for i, n in "
    cur = align_plan(SOLUTION, prefix)
    assert cur is not None and cur.line == 4 and cur.rest == "enumerate(nums):"
    noisy = "class So1ution:\n    def twoSum(self, nums, target):\n        seen = {}\n        for i, n in enumerate(nums):\n            need = target - n\n            if need in "
    cur = align_plan(SOLUTION, noisy)
    assert cur is not None and cur.line == 7 and cur.rest == "seen:"


def test_align_deviation_and_end_of_code() -> None:
    assert align_plan(SOLUTION, "class Solution:\n    def twoSum(self, nums, target):\n        print(") is None
    assert align_plan(SOLUTION, "def something_else():\n    x") is None
    assert align_plan("", "x") is None
    done = "\n".join(ln for ln in SOLUTION.split("\n") if not is_comment_line(ln))   # typed without the comments
    cur = align_plan(SOLUTION, done)
    assert cur is not None and cur.at_end and cur.rest == "" and cur.line == 10
    assert plan_ghost(SOLUTION, cur, 3) == ""
    # the last code line half typed is not the end yet
    cur = align_plan(SOLUTION, done[:-2])
    assert cur is not None and not cur.at_end and cur.rest == "[]"


def test_align_prefers_forward_progress_on_repeated_lines() -> None:
    sol = "a = 1\nx = 0\nb = 2\nx = 0\nc = 3"
    assert align_plan(sol, "a = 1\nx = 0\n").line == 2                     # after the first x = 0 -> b
    assert align_plan(sol, "b = 2\nx = 0\n", hint=3).line == 4              # after the second -> c


# ------------------------------------------------------------------------- on-screen check ---
def test_typed_fraction_counts_code_lines_on_screen() -> None:
    lines = tuple(ln for ln in SOLUTION.split("\n") if not is_comment_line(ln))
    assert typed_fraction(SOLUTION, lines) == 1.0
    assert typed_fraction(SOLUTION, lines[:5]) == pytest.approx(5 / 9)
    gutter = tuple(f"{i + 1} {ln}" for i, ln in enumerate(lines))            # OCR merged the line numbers
    assert typed_fraction(SOLUTION, gutter) == 1.0
    noisy = tuple(ln.replace("seen", "sean") for ln in lines)               # OCR-grade noise still counts
    assert typed_fraction(SOLUTION, noisy) == 1.0
    assert typed_fraction(SOLUTION, ("Two Sum", "Given an array")) == 0.0
    assert typed_fraction("# only a comment", ("x",)) == 0.0


def test_containment_and_squash() -> None:
    assert containment({"a", "b", "c", "d"}, {"a", "b", "x"}) == 0.5
    assert containment(set(), {"a"}) == 0.0
    assert squash("  a b\tc\n") == "abc"


def test_split_plan_prefers_the_streaming_block_then_the_largest() -> None:
    aside = "Walkthrough:\n- consider `[1, 2]`\n\n```python\nx = [1, 2]\n```\nNow the solution:\n\n```python\n" + SOLUTION + "\n```\nO(n)"
    parts = split_plan(aside)
    assert parts.solution == SOLUTION and parts.after == "O(n)"
    assert "x = [1, 2]" in parts.walkthrough                       # the aside stays with the walkthrough
    streaming = aside[: aside.index("seen = {}")]                  # the big block is still open: it wins
    parts = split_plan(streaming)
    assert parts.code_open and parts.solution.startswith("class Solution:")
    early = split_plan(aside[: aside.index("Now the solution")])   # only the aside so far
    assert early.solution == "x = [1, 2]" and not early.code_open
    crlf = split_plan(ANSWER.replace("\n", "\r\n"))
    assert crlf.solution == SOLUTION


def test_align_plan_is_fast_enough_for_the_tick_rate() -> None:
    import time
    big = "\n".join(f"    value_{i} = compute(step_{i}, {i})  # note {i}" if i % 3 else f"    # comment {i}" for i in range(200))
    prefix = "\n".join(big.split("\n")[:150]) + "\n    value_"
    t0 = time.perf_counter()
    for _ in range(20):
        cur = align_plan(big, prefix, hint=140)
    per_call = (time.perf_counter() - t0) / 20
    assert cur is not None and cur.line == 151
    assert per_call < 0.02, per_call                               # 150 ms tick: must stay far below

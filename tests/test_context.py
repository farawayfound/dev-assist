"""Offline tests for exocortex.screen.context: code context around the caret from OCR boxes."""
from __future__ import annotations

from exocortex.screen.context import ScreenSnapshot, context_at, estimate_char_width
from exocortex.screen.ocr import OcrLine

CW = 8      # synthetic character width (px)
LH = 20     # synthetic line height


def L(text: str, x: int, y: int) -> OcrLine:
    return OcrLine(text=text, box=(x, y, CW * len(text), 18), confidence=0.9)


def leetcode_frame(merged_first_line: bool = False) -> ScreenSnapshot:
    lines = [
        L("1. Two Sum", 40, 100),
        L("Given an array of integers nums and an integer target,", 40, 130),
        L("return indices of the two numbers such that they add up", 40, 150),
        L("Q - Have you seen this in an interview before?", 40, 400),
        L("2", 880, 120), L("def twoSum(self, nums, target):", 932, 120),
        L("3", 880, 140), L("seen = {}", 964, 140),
        L("4", 880, 160), L("for i, n in enumerate(nums):", 964, 160),
        L("5", 880, 180),
        L("6", 880, 200), L("return []", 964, 200),
    ]
    if merged_first_line:
        lines.append(L("1 class Solution:", 880, 100))
    else:
        lines += [L("1", 880, 100), L("class Solution:", 900, 100)]
    return ScreenSnapshot(lines=tuple(lines), rect=(0, 0, 1600, 900), captured_at=1.0, source_name="LeetCode")


def test_code_pane_isolated_gutter_stripped_indentation_restored() -> None:
    ctx = context_at(leetcode_frame(), x=900 + 12 * CW, y=180, line_h=LH, font_px=13.3)
    assert ctx is not None
    assert ctx.above == (
        "class Solution:\n    def twoSum(self, nums, target):\n        seen = {}\n        for i, n in enumerate(nums):"
    )
    assert ctx.current_left == " " * 12 and not ctx.has_line
    assert ctx.below == "        return []"
    assert "Two Sum" not in ctx.prefix and "interview" not in ctx.prefix   # the description pane is another pane
    assert ctx.prefix.endswith("\n" + " " * 12)


def test_caret_inside_an_existing_line_splits_it() -> None:
    ctx = context_at(leetcode_frame(), x=964 + 4 * CW, y=140, line_h=LH, font_px=13.3)
    assert ctx is not None and ctx.has_line
    assert ctx.current_left == "        seen"
    assert ctx.suffix.startswith(" = {}")
    assert ctx.prefix.endswith("        seen")


def test_merged_gutter_number_keeps_relative_indentation() -> None:
    ctx = context_at(leetcode_frame(merged_first_line=True), x=900 + 12 * CW, y=180, line_h=LH, font_px=13.3)
    assert ctx is not None
    lines = ctx.above.split("\n")
    assert lines[0].strip() == "class Solution:" and lines[1].strip().startswith("def twoSum")
    indents = [len(l) - len(l.lstrip(" ")) for l in lines]
    assert indents[2] == indents[3] == indents[1] + 4     # body one level deeper than the def


def test_caret_outside_frame_or_no_snapshot() -> None:
    assert context_at(None, 10, 10, LH, 13.3) is None
    assert context_at(leetcode_frame(), x=5000, y=5000, line_h=LH, font_px=13.3) is None


def test_char_width_estimates() -> None:
    assert estimate_char_width(20.0, 0) == 12.0
    assert estimate_char_width(0.0, 26) == 26 * 0.46
    assert estimate_char_width(0.0, 0) == 8.0


def test_above_is_capped() -> None:
    lines = [L(f"line{i} = {i}", 100, 100 + i * LH) for i in range(80)]
    snap = ScreenSnapshot(lines=tuple(lines), rect=(0, 0, 800, 3000), captured_at=1.0)
    ctx = context_at(snap, x=100, y=100 + 79 * LH, line_h=LH, font_px=13.3, max_above_lines=10, max_above_chars=60)
    assert ctx is not None
    assert len(ctx.above) <= 60 and ctx.above.endswith("line78 = 78")
    assert not ctx.above.startswith("\n")

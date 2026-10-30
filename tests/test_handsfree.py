"""Offline tests for exocortex.handsfree (hands-free mode): the pure decision helpers at the top
of the module and the HandsFreeController end to end.

Pure helpers (see their docstrings in exocortex/handsfree.py for the contract): normalize_lines,
normalize_keys, sample_diff, is_code_line, is_edit, classify_change (ChangeReport), is_new_task,
task_kind, own_code_lines, is_own_code_line, deviation, ScreenContext.

The controller is driven through a FakeHooks (records every call; text()/code() take
`focus="", label=""` -- as `HandsFreeHooks.text`/`.code` do -- and record what they were called
with; they return incrementing job ids, or None when told to defer, matching "the engine deferred
it: LLM not ready"), a real EventBus, and a FakeClock the test advances explicitly. The
controller's background thread only *notices* an advanced clock on its next real-time tick (its
loop polls every ~0.1 s), so every helper here combines a clock advance with `wait_until` rather
than sleeping for the real settle_s / timeout duration -- this is also what lets these tests run
in a fraction of a second even though a real cycle "takes ~20-30 s".

TRIGGER_LINES and BASELINE_LINES both carry code, so `exocortex.handsfree.task_kind` reads them as
CODE tasks (a coding statement / >= MIN_CODE_LINES code-looking lines) -- which, under the real
default `[handsfree].on_code = "code"`, would run Code only and drop the Text step these tests
exist to exercise the order/lifecycle of. The shared `cfg` fixture below sets
`cfg.handsfree.on_code = "ignore"` (the ordinary checkpoint -> text -> code order regardless of
task kind) so those scenarios stay about what they are testing -- change detection, retries,
pause/reset, deviation -- and not about code-task step selection. `on_code = "code"` /
`"code_first"` / `"ignore"` themselves, and `task_kind` picking a code task in the first place, are
tested directly (their own config, prose-plus-code or dev-window screens) in the "controller:
code-task step selection" section further down.

Central to several tests here (and to the user-facing concern that prompted this suite): there is
no minimum interval between cycles anywhere in this module -- the screen (a diff between one
SAMPLE and the very next one, not against when a cycle last ran) is the only thing that starts a
cycle. `test_massive_changes_back_to_back_are_not_rate_limited` and
`test_progressive_typing_never_triggers_even_though_cumulative_drift_is_total` exercise exactly
that. The deviation (code-regeneration) tests exercise the "if I deviate substantially from the
AI output code, regenerate" behaviour end to end.
"""
from __future__ import annotations

import time
from typing import Any, Callable, Sequence

import pytest

from exocortex.bus import (
    AssistDoneEvent,
    AssistErrorEvent,
    AssistStartedEvent,
    CheckpointEvent,
    EventBus,
    HandsFreeEvent,
    ScreenTextEvent,
)
from exocortex.config import Config, HandsFreeConfig
from exocortex.handsfree import (
    CONTEXT_EARLIER_HEADER,
    CONTEXT_NOW_HEADER,
    HANDS_FREE_CODE_FOCUS,
    HANDS_FREE_CODE_LABEL,
    MAX_STEP_RETRIES,
    HandsFreeController,
    ScreenContext,
    classify_change,
    deviation,
    is_code_line,
    is_edit,
    is_new_task,
    is_own_code_line,
    normalize_keys,
    normalize_lines,
    own_code_lines,
    sample_diff,
    task_kind,
)

# ==================================================================================== #
# fakes / helpers
# ==================================================================================== #


class FakeClock:
    """A controllable session clock: `advance()` moves it forward. The controller thread only
    observes the new value at its next real-time tick (its loop polls ~every 0.1 s), so tests
    pair this with `wait_until` instead of sleeping for the real duration being simulated."""

    def __init__(self, start: float = 1_000.0) -> None:
        self._t = start

    def __call__(self) -> float:
        return self._t

    def advance(self, dt: float) -> None:
        self._t += dt


class FakeHooks:
    """Fake HandsFreeHooks. Records every call in order; `text()`/`code()` return incrementing
    job ids (also recorded per-kind in `.text_ids`/`.code_ids`), or None ("the engine deferred
    it") while `.defer_text`/`.defer_code` is set."""

    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []
        self.capture_calls: list[bool] = []
        self.checkpoint_calls = 0
        self.text_calls = 0
        self.code_calls = 0
        self.text_ids: list[int | None] = []
        self.code_ids: list[int | None] = []
        self.text_focus_labels: list[tuple[str, str]] = []
        self.code_focus_labels: list[tuple[str, str]] = []
        self.defer_text = False
        self.defer_code = False
        self.recent_code_text = ""
        self._next_id = 1

    def capture_now(self, accurate: bool = False) -> None:
        self.calls.append(("capture_now", accurate))
        self.capture_calls.append(accurate)

    def checkpoint(self) -> None:
        self.calls.append(("checkpoint",))
        self.checkpoint_calls += 1

    def text(self, focus: str = "", label: str = "") -> int | None:
        self.calls.append(("text", focus, label))
        self.text_calls += 1
        self.text_focus_labels.append((focus, label))
        jid = None if self.defer_text else self._alloc()
        self.text_ids.append(jid)
        return jid

    def code(self, focus: str = "", label: str = "", fresh: bool = False) -> int | None:
        # 2026-10-29: a "code_fresh" step calls with fresh=True; recorded as a 4th element so the
        # older 3-tuple comparisons stay exact for the ordinary steps
        self.calls.append(("code", focus, label, "fresh") if fresh else ("code", focus, label))
        self.code_calls += 1
        self.code_focus_labels.append((focus, label))
        jid = None if self.defer_code else self._alloc()
        self.code_ids.append(jid)
        return jid

    def recent_code(self) -> str:
        return self.recent_code_text

    def _alloc(self) -> int:
        jid = self._next_id
        self._next_id += 1
        return jid


def wait_until(predicate: Callable[[], bool], timeout: float = 3.0, interval: float = 0.01) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


def publish_screen(
    bus: EventBus,
    *,
    lines: Sequence[str],
    changed: bool = True,
    accurate: bool = False,
    text: str | None = None,
    window_title: str = "",
    window_process: str = "",
    mode: str = "auto",
) -> None:
    bus.publish(
        ScreenTextEvent(
            mode=mode,
            source_name="Fake Window",
            rect=(0, 0, 800, 600),
            text=text if text is not None else "\n".join(lines),
            lines=tuple(lines),
            changed=changed,
            ocr_ms=5.0,
            captured_at=0.0,
            engine="fake",
            accurate=accurate,
            window_title=window_title,
            window_process=window_process,
        )
    )


def settle(
    bus: EventBus,
    clock: FakeClock,
    controller: HandsFreeController,
    lines: Sequence[str],
    *,
    changed: bool = True,
    text: str | None = None,
    window_title: str = "",
    window_process: str = "",
    mode: str = "auto",
    extra_advance: float = 5.0,
    drain: float = 0.2,
) -> None:
    """Publish a sample and advance the clock well past settle_s.

    `drain` gives the controller's background thread real wall-clock time to dequeue and process
    the event -- stamping its internal "last changed at" with the clock's value as it is NOW --
    before the clock is advanced; skipping this would let a slow-to-be-dequeued event stamp
    itself with the ALREADY advanced value, which would never look "settled" until a further
    advance. (Note: while a cycle is running, `controller.state` reports the running phase, not
    "settling" -- the sample is still tracked internally, just not visible through `.state`.)
    """
    publish_screen(bus, lines=lines, changed=changed, text=text, window_title=window_title,
                    window_process=window_process, mode=mode)
    time.sleep(drain)
    clock.advance(extra_advance)


def start_and_settle(
    bus: EventBus,
    clock: FakeClock,
    hooks: FakeHooks,
    controller: HandsFreeController,
    lines: Sequence[str],
    **kw: Any,
) -> None:
    controller.start()
    assert wait_until(lambda: hooks.capture_calls == [False]), "start() must force a capture(False)"
    settle(bus, clock, controller, lines, **kw)


def expect_checkpoint(bus: EventBus, hooks: FakeHooks, count: int) -> None:
    """Waits for the count'th checkpoint() call, then publishes the CheckpointEvent the real
    engine's checkpoint() would raise as a side effect (so `_own_checkpoints_pending` stays
    accurate for later tests of the user-checkpoint-forces-a-redo behaviour)."""
    assert wait_until(lambda: hooks.checkpoint_calls == count), f"expected {count} checkpoint(s), got {hooks.checkpoint_calls}"
    bus.publish(CheckpointEvent(t=0.0, segment_id=None))


def complete_step(bus: EventBus, hooks: FakeHooks, kind: str, expect_count: int, *, cancelled: bool = False, error: bool = False) -> int:
    """Waits for the `expect_count`'th `kind` job request, then publishes its outcome."""
    calls_attr, ids_attr = (("text_calls", "text_ids") if kind == "text" else ("code_calls", "code_ids"))
    assert wait_until(lambda: getattr(hooks, calls_attr) >= expect_count), f"never requested {kind} #{expect_count}"
    jid = getattr(hooks, ids_attr)[expect_count - 1]
    assert jid is not None, "job was deferred (None); publish an AssistStartedEvent instead"
    if error:
        bus.publish(AssistErrorEvent(job_id=jid, message="boom"))
    else:
        bus.publish(AssistDoneEvent(job_id=jid, text="answer", ttft_s=0.05, tokens=10, tok_per_s=100.0, cancelled=cancelled))
    return jid


TRIGGER_LINES = [
    "Two Sum: given nums and target, return indices of the two numbers that add to target.",
    "def two_sum(nums, target):",
    "    pass",
]

BASELINE_LINES = ["def a(): pass", "def b(): pass", "def c(): pass"]

LUNCH_LINES = ["hey how's it going", "want to grab lunch later", "sure sounds good, noon works"]


@pytest.fixture
def bus() -> EventBus:
    return EventBus()


@pytest.fixture
def hooks() -> FakeHooks:
    return FakeHooks()


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def cfg() -> Config:
    c = Config()
    c.handsfree.settle_s = 0.05
    c.handsfree.accurate_ocr = False
    # TRIGGER_LINES / BASELINE_LINES read as code tasks (module docstring); "ignore" keeps the
    # ordinary checkpoint -> text -> code order so the tests below stay about what they test.
    c.handsfree.on_code = "ignore"
    return c


@pytest.fixture
def make_controller():
    created: list[HandsFreeController] = []

    def _make(cfg: Config, bus: EventBus, hooks: FakeHooks, clock: FakeClock) -> HandsFreeController:
        c = HandsFreeController(cfg, bus, hooks, clock)
        created.append(c)
        return c

    yield _make
    for c in created:
        try:
            c.stop()
        except Exception:
            pass


def make_baseline(bus: EventBus, hooks: FakeHooks, clock: FakeClock, cfg: Config, make_controller) -> HandsFreeController:
    """A controller with one completed cycle (checkpoint 1, text 1, code 1) on BASELINE_LINES."""
    c = make_controller(cfg, bus, hooks, clock)
    start_and_settle(bus, clock, hooks, c, BASELINE_LINES)
    expect_checkpoint(bus, hooks, 1)
    complete_step(bus, hooks, "text", 1)
    complete_step(bus, hooks, "code", 1)
    assert wait_until(lambda: c.state == "watching")
    return c


# ==================================================================================== #
# pure helpers: normalize_lines / sample_diff
# ==================================================================================== #


def test_normalize_lines_dedupes_case_and_whitespace_and_drops_short_lines() -> None:
    lines = ["  Hello   World  ", "hello world", "Hi", "ok", "a"]
    assert normalize_lines(lines) == {"hello world", "hi", "ok"}


def test_sample_diff_no_previous_sample_is_maximally_different() -> None:
    assert sample_diff(None, {"a", "b"}) == (1.0, 2)


def test_sample_diff_both_empty_is_zero() -> None:
    assert sample_diff(None, set()) == (0.0, 0)
    assert sample_diff(set(), set()) == (0.0, 0)


def test_sample_diff_identical_screens_is_zero() -> None:
    assert sample_diff({"a", "b"}, {"a", "b"}) == (0.0, 0)


def test_sample_diff_a_question_change() -> None:
    score, changed = sample_diff({"q1", "q2", "q3"}, {"q1", "q4", "q5"})
    assert score == pytest.approx(0.8)
    assert changed == 4


def test_sample_diff_typing_one_line_stays_low() -> None:
    previous = {f"line{i}" for i in range(12)} | {"partial answ"}
    current = {f"line{i}" for i in range(12)} | {"partial answer"}
    score, changed = sample_diff(previous, current)
    assert score == pytest.approx(0.1428571, abs=0.001)
    assert changed == 2


def test_sample_diff_a_big_scroll_is_high() -> None:
    previous = {f"old{i}" for i in range(8)}
    current = {f"old{i}" for i in range(2)} | {f"new{i}" for i in range(6)}
    score, changed = sample_diff(previous, current)
    assert score == pytest.approx(0.8571428, abs=0.001)
    assert changed == 12


# ==================================================================================== #
# pure helpers: is_code_line
# ==================================================================================== #


@pytest.mark.parametrize(
    "line, expected",
    [
        ("# a comment explaining this", False),
        ("// also a comment", False),
        ("This function returns the total sum.", False),  # a sentence: prose
        ("def total(nums):", True),  # keyword + brackets
        ("x = 5", True),  # assignment
        ("if x > 3:", True),  # keyword + comparison
        ("return [1, 2, 3]", True),  # keyword + brackets
        ("hello world", False),  # plain prose, no code cue
        ("", False),
    ],
)
def test_is_code_line(line: str, expected: bool) -> None:
    assert is_code_line(line) is expected


# ==================================================================================== #
# pure helpers: is_edit
# ==================================================================================== #


def test_is_edit_extension_typing_forward() -> None:
    assert is_edit("implement a fu", "implement a func") is True


def test_is_edit_backspace_typing_backward() -> None:
    assert is_edit("implement a function", "implement a functio") is True


def test_is_edit_tolerates_one_trailing_noise_character() -> None:
    """The extension branch allows the shorter string's LAST character to be OCR noise (an
    unrelated mismatch) as long as everything before it still matches."""
    assert is_edit("hash mapx", "hash map for keys") is True


def test_is_edit_a_ticking_counter_is_an_edit() -> None:
    assert is_edit("Ln 12", "Ln 13") is True


def test_is_edit_a_page_counter_is_an_edit() -> None:
    assert is_edit("page3of12", "page4of12") is True


def test_is_edit_two_different_short_questions_are_not_an_edit() -> None:
    assert is_edit("what is a hash map?", "what is a heap?") is False


def test_is_edit_two_different_equal_length_questions_are_not_an_edit() -> None:
    assert is_edit("what is a stack?", "what is a queue?") is False


def test_is_edit_empty_strings() -> None:
    assert is_edit("", "") is False
    assert is_edit("", "hello") is False
    assert is_edit("hello", "") is False


def test_is_edit_identical_non_empty_strings() -> None:
    assert is_edit("same line", "same line") is True


# ==================================================================================== #
# pure helpers: normalize_keys
# ==================================================================================== #


def test_normalize_keys_dedupes_case_and_whitespace_preserves_first_seen_order() -> None:
    lines = ["  Hello   World  ", "hi", "HELLO WORLD", "ok", "a", "Second Line", "hi"]
    # "a" is dropped (< 2 chars); "HELLO WORLD" and the second "hi" are dupes of earlier keys and
    # do not appear again or move the earlier occurrence's position.
    assert normalize_keys(lines) == ["hello world", "hi", "ok", "second line"]


def test_normalize_keys_empty_input_is_empty() -> None:
    assert normalize_keys([]) == []


# ==================================================================================== #
# pure helpers: classify_change
# ==================================================================================== #


def test_classify_change_categorizes_fresh_prose_code_edits_gone_and_questions() -> None:
    previous = ["Two Sum", "def two_sum(nums, target):", "page 3 of 12"]
    current = [
        "def two_sum(nums, target):",   # unchanged: neither fresh, edited, nor gone
        "page 4 of 12",                 # a ticking counter: an edit of "page 3 of 12"
        "What is a hash map?",          # fresh, prose, a question, new_text (3+ words)
        "x = 5",                        # fresh, code (not prose, not a "question")
        "a brand new prose line here",  # fresh, prose, new_text (3+ words), not a question
    ]
    report = classify_change(previous, current)

    assert set(report.fresh) == {"what is a hash map?", "x = 5", "a brand new prose line here"}
    assert report.fresh_code == ("x = 5",)
    assert set(report.fresh_prose) == {"what is a hash map?", "a brand new prose line here"}
    assert report.questions == ("what is a hash map?",)
    assert set(report.new_text) == {"what is a hash map?", "a brand new prose line here"}
    assert report.edits == (("page 3 of 12", "page 4 of 12"),)
    assert report.typed == ()
    assert report.gone == 1  # "two sum": no longer present, and not paired with any added line


def test_classify_change_new_text_excludes_short_fresh_prose_lines() -> None:
    """A fresh prose line under MIN_WORDS_NEW_TEXT words (a heading) is `fresh`/`fresh_prose` but
    not `new_text`."""
    report = classify_change(None, ["Two Sum"])
    assert report.fresh_prose == ("two sum",)
    assert report.new_text == ()


def test_classify_change_typed_memory_absorbs_a_line_that_extends_a_typed_key() -> None:
    """A would-be fresh line that extends a key already recorded as `typed` (the controller's
    memory of lines it watched grow, sample after sample, since the last task) is folded into
    `typed`, not `fresh` -- typing never looks like new text however far it has drifted."""
    report = classify_change(["some unrelated old line"], ["partial answer now here"], typed=["partial ans"])
    assert report.typed == ("partial answer now here",)
    assert report.fresh == ()


def test_classify_change_no_previous_screen_everything_is_fresh() -> None:
    report = classify_change(None, ["one line here", "def f(): pass"])
    assert set(report.fresh) == {"one line here", "def f(): pass"}
    assert report.gone == 0
    assert report.score == 1.0


# ==================================================================================== #
# pure helpers: is_new_task
# ==================================================================================== #


def test_is_new_task_short_question_under_a_static_header_triggers() -> None:
    header = ["Header line one", "Header line two", "Header line three", "Header line four"]
    report = classify_change(header, header + ["Why recursion?"])
    new, reason = is_new_task(report, HandsFreeConfig(), focus="auto")
    assert new is True
    assert reason  # non-empty
    assert "why recursion?" in reason


def test_is_new_task_same_short_question_under_a_much_longer_static_header_still_triggers() -> None:
    """The same short question, but under 25 static lines instead of 4: the Jaccard score drops
    far below `change_threshold` (well under 0.6) -- the new-question rule is score-independent,
    so it still triggers."""
    header = [f"Static filler line number {i}" for i in range(25)]
    report = classify_change(header, header + ["Why recursion?"])
    assert report.score < 0.1  # far below change_threshold -- the score-based rule alone would not fire
    new, reason = is_new_task(report, HandsFreeConfig(), focus="auto")
    assert new is True
    assert reason


def test_is_new_task_typing_a_line_never_triggers() -> None:
    report = classify_change(["partial answ"], ["partial answer"])
    new, reason = is_new_task(report, HandsFreeConfig(), focus="all")  # even the most permissive level
    assert new is False
    assert reason  # non-empty


def test_is_new_task_a_poll_line_never_triggers_at_any_focus_level() -> None:
    header = ["Header line one", "Header line two"]
    report = classify_change(header, header + ["Have you seen this in an interview before?"])
    assert report.questions  # it does read as a question ...
    for level in ("technical", "auto", "all"):
        new, reason = is_new_task(report, HandsFreeConfig(), focus=level)
        assert new is False, f"the built-in ignore pattern must drop it at {level!r} too"
        assert reason


def test_is_new_task_a_big_scroll_with_fresh_code_triggers_under_auto() -> None:
    """A big scroll (score >= change_threshold) whose fresh lines include code: "auto" takes it
    even though none of the fresh lines is itself a question or an explicit task statement --
    code on screen is enough (the module docstring's "code-biased")."""
    previous = [f"old_{i} = 1" for i in range(6)]
    current = [f"new_{i} = compute({i})" for i in range(6)]
    report = classify_change(previous, current)
    assert report.score >= 0.6
    assert len(report.fresh_code) >= 3
    new, reason = is_new_task(report, HandsFreeConfig(), focus="auto")
    assert new is True
    assert reason


def test_is_new_task_a_chat_drifting_into_view_triggers_under_all_but_not_auto() -> None:
    """A big change of view that is only casual, non-task prose (no question, no code, no task
    verb): "all" reads everything and triggers; "auto" (code-biased) wants something task-like
    among the new lines and does not."""
    current = ["hey how is it going", "want to grab lunch later", "sure sounds good noon works"]
    report = classify_change(None, current)
    assert report.score == 1.0
    new_all, reason_all = is_new_task(report, HandsFreeConfig(), focus="all")
    assert new_all is True
    assert reason_all
    new_auto, reason_auto = is_new_task(report, HandsFreeConfig(), focus="auto")
    assert new_auto is False
    assert reason_auto


def test_is_new_task_min_new_lines_needs_that_many_new_task_like_lines() -> None:
    """Non-question task statements (so the question rule does not short-circuit): with
    `min_new_lines = 2`, one new task-like line is not enough; two are."""
    hf = HandsFreeConfig(min_new_lines=2)
    previous = ["static one", "static two"]
    one_line = classify_change(previous, previous + ["Implement a cache with eviction"])
    two_lines = classify_change(previous, previous + ["Implement a cache with eviction", "Write a function that sorts values"])

    new1, reason1 = is_new_task(one_line, hf, focus="auto")
    assert new1 is False
    assert reason1

    new2, reason2 = is_new_task(two_lines, hf, focus="auto")
    assert new2 is True
    assert reason2


# ==================================================================================== #
# pure helpers: task_kind
# ==================================================================================== #


def test_task_kind_code_lines_on_screen() -> None:
    kind, why = task_kind(["def f(x):", "return x + 1", "a plain prose sentence here."], "")
    assert kind == "code"
    assert why


def test_task_kind_editor_gutter_on_screen() -> None:
    gutter_lines = ["12  Walk through the problem", "13  Explain the approach", "14  Describe edge cases"]
    kind, why = task_kind(gutter_lines, "")
    assert kind == "code"
    assert "gutter" in why


def test_task_kind_dev_window_title_outside_region_mode() -> None:
    lines = ["just some prose on screen", "nothing special here"]
    kind, why = task_kind(lines, "", window_title="main.py - Visual Studio Code", mode="monitor")
    assert kind == "code"
    assert why


def test_task_kind_dev_window_process_outside_region_mode() -> None:
    lines = ["just some prose on screen", "nothing special here"]
    kind, why = task_kind(lines, "", window_process="Code.exe", mode="monitor")
    assert kind == "code"
    assert why


def test_task_kind_dev_window_is_ignored_in_region_mode() -> None:
    """In `mode="region"` the window in front and the region read are unrelated -- the window
    title / process must NOT make it a code task."""
    lines = ["just some prose on screen", "nothing special here"]
    kind, why = task_kind(lines, "", window_title="main.py - Visual Studio Code", window_process="Code.exe", mode="region")
    assert kind == "text"
    assert why


def test_task_kind_coding_task_cue_with_no_code_on_screen() -> None:
    text = "Given an array of integers nums and a target, return the indices."
    kind, why = task_kind(["Given an array of integers nums and a target,"], text, mode="monitor")
    assert kind == "code"
    assert why


def test_task_kind_else_text() -> None:
    kind, why = task_kind(["just chatting about lunch plans today"], "just chatting about lunch plans today", mode="monitor")
    assert kind == "text"
    assert why


# ==================================================================================== #
# pure helpers: own_code_lines / is_own_code_line
# ==================================================================================== #


def test_own_code_lines_skips_fences_comments_and_short_lines() -> None:
    code = "```python\n# a comment explaining the loop\ndef f(x):\n    total = 1\n    total += x\n~~~\n"
    assert own_code_lines(code) == ["deff(x):", "total=1", "total+=x"]


def test_is_own_code_line_exact_after_squash() -> None:
    own = own_code_lines("def two_sum(nums, target):\n    seen = {}\n    return [seen[n], i]")
    assert is_own_code_line("  def two_sum(nums, target):  ", own) is True


def test_is_own_code_line_gutter_merged_tail() -> None:
    own = own_code_lines("def two_sum(nums, target):\n    seen = {}\n    return [seen[n], i]")
    assert is_own_code_line("12    return [seen[n], i]", own) is True


def test_is_own_code_line_fuzzy_differently_annotated_head() -> None:
    own = own_code_lines("def two_sum(nums, target):\n    seen = {}\n    return [seen[n], i]")
    assert is_own_code_line("def two_sum(nums: List[int], target: int):", own) is True


def test_is_own_code_line_unrelated_line_is_false() -> None:
    own = own_code_lines("def two_sum(nums, target):\n    seen = {}\n    return [seen[n], i]")
    assert is_own_code_line("    print('done')", own) is False


def test_is_own_code_line_empty_own_is_always_false() -> None:
    assert is_own_code_line("    return []", []) is False


# ==================================================================================== #
# pure helpers: deviation
# ==================================================================================== #


def test_deviation_no_own_code_is_zero_and_empty() -> None:
    assert deviation(["for i in range(10): pass"], own=[]) == (0.0, set())


def test_deviation_typing_the_apps_code_is_low() -> None:
    own = own_code_lines("def solve(nums):\n    total = 0\n    for n in nums:\n        total += n\n    return total")
    lines = ["def solve(nums):", "    total = 0", "    for n in nums:"]
    fraction, user = deviation(lines, own)
    assert fraction == 0.0
    assert user == set()


def test_deviation_a_different_approach_is_high() -> None:
    own = own_code_lines("def solve(nums):\n    total = 0\n    for n in nums:\n        total += n\n    return total")
    lines = ["def solve(nums):", "    result = functools.reduce(lambda a, b: a + b, nums)", "    return result"]
    fraction, user = deviation(lines, own)
    assert fraction >= 0.5
    assert "result = functools.reduce(lambda a, b: a + b, nums)" in user


def test_deviation_exclude_drops_statement_lines() -> None:
    own = own_code_lines("def two_sum(nums, target):\n    seen = {}\n    return []")
    statement_line = "Input: nums = [2, 7, 11, 15], target = 9"
    assert is_code_line(statement_line)  # looks like code -- would count without `exclude`
    lines = [statement_line, "def two_sum(nums, target):"]

    without_exclude = deviation(lines, own)
    assert without_exclude == (0.5, {"input: nums = [2, 7, 11, 15], target = 9"})

    with_exclude = deviation(lines, own, exclude=normalize_lines([statement_line]))
    assert with_exclude == (0.0, set())


def test_deviation_typed_param_drops_lines_still_being_typed() -> None:
    """upstream finding 10 (ported from mac-cortex): a line the sample-to-sample tracker already
    knows is still being typed must not count as user drift just because it does not (yet)
    fuzzy-match the app's own code."""
    from exocortex.plan import squash

    own = own_code_lines("def two_sum(nums, target):\n    seen = {}\n    return [seen[n], i]")
    partial_line = "    return [se"  # midway through typing the app's own "return" line
    lines = ["def two_sum(nums, target):", partial_line]

    without_typed = deviation(lines, own)
    assert without_typed[0] > 0.0
    assert partial_line.strip() != own[-1], "sanity: the partial line genuinely doesn't match yet"

    with_typed = deviation(lines, own, typed=[squash(partial_line)])
    assert with_typed == (0.0, set())


def test_deviation_typed_param_does_not_hide_a_genuinely_different_line() -> None:
    """The `typed` exclusion is fuzzy-matched against the given keys, not "anything in progress":
    a line unrelated to what is being tracked as typed must still count as drift."""
    own = own_code_lines("def two_sum(nums, target):\n    seen = {}\n    return [seen[n], i]")
    lines = ["def two_sum(nums, target):", "    result = brute_force(nums, target)"]

    with_unrelated_typed = deviation(lines, own, typed=["x=1"])  # pre-squashed, unrelated key
    assert with_unrelated_typed[0] > 0.0
    assert "result = brute_force(nums, target)" in with_unrelated_typed[1]


def test_deviation_typed_param_matches_mixed_case_lines() -> None:
    """upstream finding 10 (Windows fix): HandsFreeController._typed stores keys squashed from
    normalize_keys() output, which is already casefolded (see _track_sample: `keys =
    normalize_keys(ev.lines)`, then `self._typed[squash(after)] = None`). deviation()'s own
    squash of the candidate line must therefore also be casefolded before comparing, or the
    `typed` exclusion silently never fires for any line containing an uppercase character
    (class names, True/False/None, PascalCase/camelCase identifiers -- i.e. most real code)."""
    from exocortex.handsfree import normalize_line
    from exocortex.plan import squash

    own = own_code_lines("class HashMap:\n    def __init__(self):\n        self.Result = HashMap()")
    partial_line = "    Result = HashM"  # midway through typing the app's own line, mixed case
    lines = ["class HashMap:", partial_line]

    without_typed = deviation(lines, own)
    assert without_typed[0] > 0.0, "sanity: the partial line genuinely doesn't match yet"

    # the real value self._typed would hold for this line, per _track_sample's own bookkeeping.
    typed_key = squash(normalize_line(partial_line))

    with_typed = deviation(lines, own, typed=[typed_key])
    assert with_typed == (0.0, set())


# ==================================================================================== #
# pure helpers: ScreenContext
# ==================================================================================== #


def test_screen_context_starts_empty() -> None:
    ctx = ScreenContext()
    assert ctx.empty
    assert ctx.text(1000) == ""


def test_screen_context_merge_puts_current_screen_intact_after_earlier_lines() -> None:
    ctx = ScreenContext()
    ctx.reset(["line one", "line two", "line three"], "line one\nline two\nline three")
    ctx.merge(["line two", "line three", "line four"], "line two\nline three\nline four")

    assert not ctx.empty
    text = ctx.text(2000)
    assert text == (
        f"{CONTEXT_EARLIER_HEADER}\nline one\n\n{CONTEXT_NOW_HEADER}\nline two\nline three\nline four"
    )


def test_screen_context_edited_line_drops_its_earlier_reading() -> None:
    """A statement head that is still recognisably the same line (squashed 12-char head match)
    must not appear twice -- once as "earlier" and once (edited) as current."""
    ctx = ScreenContext()
    ctx.reset(["def two_sum(nums, target):"], "def two_sum(nums, target):")
    ctx.merge(
        ["def two_sum(nums: List[int], target: int) -> List[int]:"],
        "def two_sum(nums: List[int], target: int) -> List[int]:",
    )
    text = ctx.text(2000)
    assert CONTEXT_EARLIER_HEADER not in text  # nothing earlier survives: it was the edited line
    assert text == "def two_sum(nums: List[int], target: int) -> List[int]:"


def test_screen_context_budget_trims_oldest_earlier_lines_never_current() -> None:
    ctx = ScreenContext()
    oldest = "OLDEST_" + "x" * 80
    middle = "MIDDLE_" + "x" * 80
    newest = "newest and short"
    ctx.reset([oldest, middle, newest], "\n".join([oldest, middle, newest]))
    current_text = "the current screen, short and simple"
    ctx.merge([current_text], current_text)

    text = ctx.text(150)  # room for the current screen + one short earlier line, not two long ones
    assert current_text in text
    assert newest in text
    assert oldest not in text
    assert middle not in text


def test_screen_context_reset_clears_everything() -> None:
    ctx = ScreenContext()
    ctx.reset(["old task line"], "old task line")
    assert not ctx.empty
    ctx.reset()
    assert ctx.empty
    assert ctx.text(1000) == ""


# ==================================================================================== #
# controller: lifecycle / state
# ==================================================================================== #


def test_state_is_off_before_start_and_after_stop(bus, hooks, clock, cfg, make_controller) -> None:
    c = make_controller(cfg, bus, hooks, clock)
    assert c.state == "off"
    assert c.enabled is False

    c.start()
    assert wait_until(lambda: c.enabled is True)

    c.stop()
    assert c.state == "off"
    assert c.enabled is False


def test_stop_publishes_off_event_and_unsubscribes(bus, hooks, clock, cfg, make_controller) -> None:
    c = make_controller(cfg, bus, hooks, clock)
    q = bus.subscribe()
    c.start()
    assert wait_until(lambda: hooks.capture_calls == [False])

    c.stop()
    events = []
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        while not q.empty():
            events.append(q.get_nowait())
        if any(isinstance(e, HandsFreeEvent) and e.enabled is False and e.state == "off" for e in events):
            break
        time.sleep(0.02)
    assert any(isinstance(e, HandsFreeEvent) and e.enabled is False and e.state == "off" for e in events)

    # publishing after stop() must not error and must not reach a dead controller
    publish_screen(bus, lines=["should be ignored"])
    time.sleep(0.1)


def test_handsfree_events_are_published_only_on_change(bus, hooks, clock, cfg, make_controller) -> None:
    c = make_controller(cfg, bus, hooks, clock)
    q = bus.subscribe()
    c.set_paused(True)  # remembered even before start() (queued as a pending pause)
    c.start()
    assert wait_until(lambda: c.state == "paused")

    # a second, redundant pause request must not publish a duplicate "paused" event
    c.set_paused(True)
    time.sleep(0.2)

    paused_events = [e for e in list(q.queue) if isinstance(e, HandsFreeEvent) and e.state == "paused"]
    assert len(paused_events) == 1


# ==================================================================================== #
# controller: the default cycle (order, code_first, run_text/run_code, explain/completion)
# ==================================================================================== #


def test_default_cycle_runs_checkpoint_text_then_code_in_order(bus, hooks, clock, cfg, make_controller) -> None:
    c = make_controller(cfg, bus, hooks, clock)
    c.start()
    assert wait_until(lambda: hooks.capture_calls == [False])
    # even an UNCHANGED reading counts as the massive change start() forces
    settle(bus, clock, c, TRIGGER_LINES, changed=False)

    expect_checkpoint(bus, hooks, 1)
    assert wait_until(lambda: hooks.text_calls == 1)
    time.sleep(0.2)
    assert hooks.code_calls == 0, "code must wait for text's AssistDoneEvent"
    assert [call[0] for call in hooks.calls] == ["capture_now", "checkpoint", "text"]

    complete_step(bus, hooks, "text", 1)
    assert wait_until(lambda: hooks.code_calls == 1)
    complete_step(bus, hooks, "code", 1)
    assert wait_until(lambda: c.state == "watching")


def test_code_first_reverses_step_order(bus, hooks, clock, cfg, make_controller) -> None:
    cfg.handsfree.code_first = True
    c = make_controller(cfg, bus, hooks, clock)
    start_and_settle(bus, clock, hooks, c, TRIGGER_LINES)
    expect_checkpoint(bus, hooks, 1)

    assert wait_until(lambda: hooks.code_calls == 1)
    time.sleep(0.2)
    assert hooks.text_calls == 0, "text must wait for code's AssistDoneEvent when code_first"

    complete_step(bus, hooks, "code", 1)
    assert wait_until(lambda: hooks.text_calls == 1)
    complete_step(bus, hooks, "text", 1)
    assert wait_until(lambda: c.state == "watching")


@pytest.mark.parametrize("flag, other_kind", [("run_text", "code"), ("run_code", "text")])
def test_run_text_or_run_code_false_skips_its_step(bus, hooks, clock, cfg, make_controller, flag, other_kind) -> None:
    setattr(cfg.handsfree, flag, False)
    c = make_controller(cfg, bus, hooks, clock)
    start_and_settle(bus, clock, hooks, c, TRIGGER_LINES)
    expect_checkpoint(bus, hooks, 1)
    complete_step(bus, hooks, other_kind, 1)
    assert wait_until(lambda: c.state == "watching")
    skipped = "text" if flag == "run_text" else "code"
    assert (hooks.text_calls if skipped == "text" else hooks.code_calls) == 0


@pytest.mark.parametrize("section", ["explain", "completion"])
def test_explain_or_completion_mode_skips_text_automatically(bus, hooks, clock, cfg, make_controller, section) -> None:
    getattr(cfg, section).enabled = True
    c = make_controller(cfg, bus, hooks, clock)
    start_and_settle(bus, clock, hooks, c, TRIGGER_LINES)
    expect_checkpoint(bus, hooks, 1)
    complete_step(bus, hooks, "code", 1)
    assert wait_until(lambda: c.state == "watching")
    assert hooks.text_calls == 0


# ==================================================================================== #
# controller: code-task step selection ([handsfree].on_code / task_kind)
# ==================================================================================== #

#: A question ABOUT a code snippet -- 2 code lines (MIN_CODE_LINES) -- task_kind reads this as a
#: CODE task (module docstring / test_task_kind_code_lines_on_screen above).
CODE_TASK_LINES = [
    "What does this function return for nums = [2, 7, 11, 15], target = 9?",
    "def two_sum(nums, target):",
    "    return []",
]


def test_code_task_runs_checkpoint_then_code_only_with_the_code_focus_and_label(bus, hooks, clock, cfg, make_controller) -> None:
    cfg.handsfree.on_code = "code"  # the real default; this file's `cfg` fixture uses "ignore"
    c = make_controller(cfg, bus, hooks, clock)
    start_and_settle(bus, clock, hooks, c, CODE_TASK_LINES)
    expect_checkpoint(bus, hooks, 1)

    assert wait_until(lambda: hooks.code_calls == 1)
    time.sleep(0.2)
    assert hooks.text_calls == 0, "on_code='code' (default) runs Code only for a code task"
    assert hooks.code_focus_labels[-1] == (HANDS_FREE_CODE_FOCUS, HANDS_FREE_CODE_LABEL)

    complete_step(bus, hooks, "code", 1)
    assert wait_until(lambda: c.state == "watching")
    assert c.last_task_kind == "code"


def test_code_task_with_on_code_code_first_runs_code_then_text(bus, hooks, clock, cfg, make_controller) -> None:
    cfg.handsfree.on_code = "code_first"
    c = make_controller(cfg, bus, hooks, clock)
    start_and_settle(bus, clock, hooks, c, CODE_TASK_LINES)
    expect_checkpoint(bus, hooks, 1)

    assert wait_until(lambda: hooks.code_calls == 1)
    time.sleep(0.2)
    assert hooks.text_calls == 0, "code must run before text when code_first"
    assert hooks.code_focus_labels[-1] == (HANDS_FREE_CODE_FOCUS, HANDS_FREE_CODE_LABEL)

    complete_step(bus, hooks, "code", 1)
    assert wait_until(lambda: hooks.text_calls == 1)
    complete_step(bus, hooks, "text", 1)
    assert wait_until(lambda: c.state == "watching")


def test_code_task_with_on_code_ignore_runs_the_ordinary_text_then_code_order(bus, hooks, clock, cfg, make_controller) -> None:
    cfg.handsfree.on_code = "ignore"
    c = make_controller(cfg, bus, hooks, clock)
    start_and_settle(bus, clock, hooks, c, CODE_TASK_LINES)
    expect_checkpoint(bus, hooks, 1)

    assert wait_until(lambda: hooks.text_calls == 1)
    time.sleep(0.2)
    assert hooks.code_calls == 0, "code must wait for text's AssistDoneEvent under on_code='ignore'"

    complete_step(bus, hooks, "text", 1)
    assert wait_until(lambda: hooks.code_calls == 1)
    # the Code step still gets the "also covers questions ABOUT the code" ask: on_code only
    # decides ORDER (and, for "code", whether Text runs at all), not whether this ask is used.
    assert hooks.code_focus_labels[-1] == (HANDS_FREE_CODE_FOCUS, HANDS_FREE_CODE_LABEL)
    complete_step(bus, hooks, "code", 1)
    assert wait_until(lambda: c.state == "watching")


def test_dev_window_title_makes_a_text_looking_screen_a_code_task_outside_region_mode(bus, hooks, clock, cfg, make_controller) -> None:
    cfg.handsfree.on_code = "code"
    c = make_controller(cfg, bus, hooks, clock)
    prose_lines = ["What should I say when they ask about my biggest weakness?"]
    start_and_settle(bus, clock, hooks, c, prose_lines, window_title="main.py - Visual Studio Code", mode="monitor")
    expect_checkpoint(bus, hooks, 1)

    assert wait_until(lambda: hooks.code_calls == 1)
    time.sleep(0.2)
    assert hooks.text_calls == 0, "a development window in front makes this a code task outside region mode"
    assert c.last_task_kind == "code"

    complete_step(bus, hooks, "code", 1)
    assert wait_until(lambda: c.state == "watching")


def test_dev_window_title_is_ignored_for_task_kind_in_region_mode(bus, hooks, clock, cfg, make_controller) -> None:
    c = make_controller(cfg, bus, hooks, clock)
    prose_lines = ["What should I say when they ask about my biggest weakness?"]
    start_and_settle(bus, clock, hooks, c, prose_lines, window_title="main.py - Visual Studio Code", mode="region")
    expect_checkpoint(bus, hooks, 1)
    complete_step(bus, hooks, "text", 1)
    complete_step(bus, hooks, "code", 1)
    assert wait_until(lambda: c.state == "watching")
    assert c.last_task_kind == "text"


# ==================================================================================== #
# controller: change detection (threshold / min_changed_lines / no rate limit)
# ==================================================================================== #


def test_progressive_typing_never_triggers_even_though_cumulative_drift_is_total(
    bus, hooks, clock, cfg, make_controller
) -> None:
    """Coding progressively diverges from where it started, but each SAMPLE differs from the
    previous SAMPLE by only one line (a one-line scroll) -- this must never trigger, however far
    the screen has drifted from the first sample by the time many such samples have gone by."""
    all_lines = [f"line_{i:02d} = compute({i})" for i in range(14)]
    window = 6
    c = make_controller(cfg, bus, hooks, clock)
    start_and_settle(bus, clock, hooks, c, all_lines[0:window])
    expect_checkpoint(bus, hooks, 1)
    complete_step(bus, hooks, "text", 1)
    complete_step(bus, hooks, "code", 1)
    assert wait_until(lambda: c.state == "watching")

    for k in range(1, 9):
        settle(bus, clock, c, all_lines[k : k + window])
        assert wait_until(lambda: c.state == "watching")
        assert hooks.checkpoint_calls == 1, f"a one-line scroll (step {k}) must never re-trigger"

    # proof this really was a massive cumulative drift: the first and last windows share nothing
    score, _ = sample_diff(normalize_lines(all_lines[0:window]), normalize_lines(all_lines[8 : 8 + window]))
    assert score == 1.0


def test_min_changed_lines_gates_a_high_score_small_change(bus, hooks, clock, cfg, make_controller) -> None:
    c = make_baseline(bus, hooks, clock, cfg, make_controller)
    # vs BASELINE_LINES this scores 0.67 (>= the 0.6 default threshold) but only 2 lines changed
    settle(bus, clock, c, ["def a(): pass"])
    assert wait_until(lambda: c.state == "watching")
    assert hooks.checkpoint_calls == 1


def test_change_at_or_above_threshold_triggers_a_new_cycle(bus, hooks, clock, cfg, make_controller) -> None:
    c = make_baseline(bus, hooks, clock, cfg, make_controller)
    new_lines = [
        "def solve_completely_different(x, y, z): return x + y + z",
        "print(solve_completely_different(1, 2, 3))",
    ]
    settle(bus, clock, c, new_lines)
    expect_checkpoint(bus, hooks, 2)
    complete_step(bus, hooks, "text", 2)
    complete_step(bus, hooks, "code", 2)
    assert wait_until(lambda: c.state == "watching")


def test_massive_changes_back_to_back_are_not_rate_limited(bus, hooks, clock, cfg, make_controller) -> None:
    """The user's concern this test answers directly: nothing in the controller enforces a
    minimum interval between cycles (there is no such setting) -- only the screen's own diff
    against the previous sample decides. Two unrelated massive changes trigger two full cycles
    back to back."""
    c = make_baseline(bus, hooks, clock, cfg, make_controller)
    assert hooks.checkpoint_calls == 1

    second_task = ["def gcd(a, b):", "    while b:", "        a, b = b, a % b", "    return a"]
    settle(bus, clock, c, second_task)
    expect_checkpoint(bus, hooks, 2)
    complete_step(bus, hooks, "text", 2)
    complete_step(bus, hooks, "code", 2)
    assert wait_until(lambda: c.state == "watching")

    third_task = ["class Stack:", "    def __init__(self):", "        self.items = []", "    def push(self, x): pass"]
    settle(bus, clock, c, third_task)
    expect_checkpoint(bus, hooks, 3)
    complete_step(bus, hooks, "text", 3)
    complete_step(bus, hooks, "code", 3)
    assert wait_until(lambda: c.state == "watching")
    assert c.triggers == 3


def test_settled_screen_matching_last_trigger_does_not_retrigger(bus, hooks, clock, cfg, make_controller) -> None:
    """An alt-tab and back: the SAMPLE-level diff sees a massive change en route, but the screen
    it finally settles on is the one already handled, so no new cycle starts."""
    c = make_baseline(bus, hooks, clock, cfg, make_controller)
    other_app = ["Unrelated Chat App", "someone: hey are you free", "someone: lunch at noon?"]
    publish_screen(bus, lines=other_app, changed=True)
    time.sleep(0.15)
    publish_screen(bus, lines=BASELINE_LINES, changed=True)  # back to the screen already handled
    time.sleep(0.2)  # let both samples be dequeued (with the clock still unadvanced) before jumping it
    clock.advance(5.0)

    assert wait_until(lambda: c.state == "watching")
    assert hooks.checkpoint_calls == 1


def test_settled_screen_matching_last_trigger_publishes_same_task_detail(bus, hooks, clock, cfg, make_controller) -> None:
    c = make_baseline(bus, hooks, clock, cfg, make_controller)
    q = bus.subscribe()
    other_app = ["Unrelated Chat App", "someone: hey are you free", "someone: lunch at noon?"]
    publish_screen(bus, lines=other_app, changed=True)
    time.sleep(0.15)
    publish_screen(bus, lines=BASELINE_LINES, changed=True)  # back to the screen already handled
    time.sleep(0.2)
    clock.advance(5.0)
    assert wait_until(lambda: c.state == "watching")
    assert hooks.checkpoint_calls == 1

    def saw_same_task() -> bool:
        return any(isinstance(e, HandsFreeEvent) and e.state == "watching" and e.detail == "same task"
                   for e in list(q.queue))

    assert wait_until(saw_same_task)


def test_one_line_question_change_under_a_static_header_starts_a_new_cycle(bus, hooks, clock, cfg, make_controller) -> None:
    """The user's bug (today's root-cause fixes, module docstring): a single new question line
    under an otherwise-static slide starts a cycle even though only 1 line changed -- well under
    `min_changed_lines` (3 by default) -- and the Jaccard score stays low. The header itself is
    task-like ("Problem: Implement...") so the FIRST cycle (below) also has something to trigger
    on; the point under test is the SECOND, one-line change."""
    header = [
        "Problem: Implement a rate limiter for the API.",
        "Constraints: array size up to 10000.",
        "Time limit: 30 minutes.",
        "Good luck!",
    ]
    c = make_controller(cfg, bus, hooks, clock)
    start_and_settle(bus, clock, hooks, c, header)
    expect_checkpoint(bus, hooks, 1)
    complete_step(bus, hooks, "text", 1)
    complete_step(bus, hooks, "code", 1)
    assert wait_until(lambda: c.state == "watching")

    settle(bus, clock, c, header + ["Why does this approach work?"])
    expect_checkpoint(bus, hooks, 2)
    assert c.last_reason.startswith("a new question")
    complete_step(bus, hooks, "text", 2)
    complete_step(bus, hooks, "code", 2)
    assert wait_until(lambda: c.state == "watching")


def test_last_task_kind_and_last_reason_are_set_after_a_trigger(bus, hooks, clock, cfg, make_controller) -> None:
    c = make_baseline(bus, hooks, clock, cfg, make_controller)
    assert c.last_task_kind == "code"  # BASELINE_LINES has >= MIN_CODE_LINES code-looking lines
    assert c.last_reason  # non-empty


@pytest.mark.parametrize("focus, expect_trigger", [("technical", False), ("auto", False), ("all", True)])
def test_focus_level_gates_non_technical_screens(bus, hooks, clock, cfg, make_controller, focus, expect_trigger) -> None:
    """LUNCH_LINES are casual, not personal-opinion ("have you", "do you like") and carry no
    technical cue and no task-like line -- "technical" wants a cue, "auto" (code-biased) still
    wants *something* task-like among the new lines (`questions.is_task_line`), only "all" reads
    everything."""
    cfg.assist.focus = focus
    c = make_controller(cfg, bus, hooks, clock)
    start_and_settle(bus, clock, hooks, c, LUNCH_LINES)
    if expect_trigger:
        expect_checkpoint(bus, hooks, 1)
    else:
        time.sleep(0.3)
        assert hooks.checkpoint_calls == 0
        assert wait_until(lambda: c.state == "watching")


# ==================================================================================== #
# controller: accurate OCR re-read
# ==================================================================================== #


def test_accurate_ocr_requests_capture_and_acts_on_its_content(bus, hooks, clock, cfg, make_controller) -> None:
    cfg.handsfree.accurate_ocr = True
    c = make_controller(cfg, bus, hooks, clock)
    c.start()
    assert wait_until(lambda: hooks.capture_calls == [False])

    # the cheap sampling engine's noisy reading settles first
    settle(bus, clock, c, ["Reverse a Iinked Iist", "def reverse(head):", "    prev = None"])
    assert wait_until(lambda: c.state == "reading")
    assert wait_until(lambda: hooks.capture_calls == [False, True])

    # the accurate re-read corrects the OCR noise -- the controller must act on THIS content
    accurate_lines = ["Reverse a linked list", "def reverse(head):", "    prev = None"]
    publish_screen(bus, lines=accurate_lines, changed=True, accurate=True)

    expect_checkpoint(bus, hooks, 1)
    assert "linked list" in c.context_text()
    complete_step(bus, hooks, "text", 1)
    complete_step(bus, hooks, "code", 1)


def test_accurate_ocr_ignores_stray_non_accurate_event_while_awaiting_it(bus, hooks, clock, cfg, make_controller) -> None:
    """While the controller is in "reading" (it asked for an accurate re-read via
    capture_now(True) and is waiting for the response), an ordinary sample published in the
    meantime (the normal [screen].interval_s cadence keeps running concurrently; nothing stops it
    from delivering one more ordinary event before the accurate one comes back) must be ignored,
    not evaluated as if it were the accurate read: `HandsFreeController._on_screen` checks
    `ev.accurate` in its `self._await == "accurate"` branch and only folds a non-accurate stray
    into the ordinary sample-to-sample bookkeeping (`_track_sample`), never treats it as the
    answer. This test demonstrates that with a stray non-accurate event carrying unrelated
    content that must not be acted on."""
    cfg.handsfree.accurate_ocr = True
    c = make_controller(cfg, bus, hooks, clock)
    c.start()
    assert wait_until(lambda: hooks.capture_calls == [False])

    settle(bus, clock, c, ["def two_sum(nums, target):", "    seen = {}", "    return []"])
    assert wait_until(lambda: c.state == "reading")
    assert wait_until(lambda: hooks.capture_calls == [False, True])

    # an ordinary (non-accurate) sample sneaks in before the real accurate read comes back
    publish_screen(bus, lines=["a totally unrelated screen appeared here now"], changed=True, accurate=False)
    time.sleep(0.3)
    assert c.state == "reading", "must keep waiting for the accurate read, not act on a stray non-accurate sample"
    assert hooks.checkpoint_calls == 0

    # now the real accurate read arrives
    publish_screen(bus, lines=["def two_sum(nums, target):", "    seen = {}", "    return []"], changed=True, accurate=True)
    expect_checkpoint(bus, hooks, 1)


# ==================================================================================== #
# controller: job lifecycle (retries, errors, wrong id, deferred, mid-cycle abort)
# ==================================================================================== #


def test_preempted_step_is_retried_then_the_cycle_moves_on(bus, hooks, clock, cfg, make_controller) -> None:
    c = make_controller(cfg, bus, hooks, clock)
    start_and_settle(bus, clock, hooks, c, TRIGGER_LINES)
    expect_checkpoint(bus, hooks, 1)
    assert wait_until(lambda: hooks.text_calls == 1)

    for n in range(1, MAX_STEP_RETRIES + 1):
        jid = hooks.text_ids[-1]
        bus.publish(AssistDoneEvent(job_id=jid, text="", ttft_s=0.0, tokens=0, tok_per_s=0.0, cancelled=True))
        assert wait_until(lambda: hooks.text_calls == n + 1), f"retry {n} never re-requested text"

    # one more cancellation beyond MAX_STEP_RETRIES: no further retry, the cycle moves on to code
    jid = hooks.text_ids[-1]
    bus.publish(AssistDoneEvent(job_id=jid, text="", ttft_s=0.0, tokens=0, tok_per_s=0.0, cancelled=True))
    assert wait_until(lambda: hooks.code_calls == 1)
    assert hooks.text_calls == MAX_STEP_RETRIES + 1
    complete_step(bus, hooks, "code", 1)


def test_skipped_assist_done_event_moves_on_without_retrying(bus, hooks, clock, cfg, make_controller) -> None:
    """`AssistDoneEvent(cancelled=True, skipped=True)` (the model answered "(nothing to do)") is
    "step done", not a preemption -- it must never be retried, unlike a plain `cancelled=True`
    (test_preempted_step_is_retried_then_the_cycle_moves_on)."""
    c = make_controller(cfg, bus, hooks, clock)
    start_and_settle(bus, clock, hooks, c, TRIGGER_LINES)
    expect_checkpoint(bus, hooks, 1)
    assert wait_until(lambda: hooks.text_calls == 1)

    jid = hooks.text_ids[-1]
    bus.publish(AssistDoneEvent(job_id=jid, text="", ttft_s=0.0, tokens=0, tok_per_s=0.0, cancelled=True, skipped=True))
    assert wait_until(lambda: hooks.code_calls == 1)
    assert hooks.text_calls == 1, "a skipped step must not be retried"
    complete_step(bus, hooks, "code", 1)
    assert wait_until(lambda: c.state == "watching")


def test_assist_error_event_advances_without_retrying(bus, hooks, clock, cfg, make_controller) -> None:
    c = make_controller(cfg, bus, hooks, clock)
    start_and_settle(bus, clock, hooks, c, TRIGGER_LINES)
    expect_checkpoint(bus, hooks, 1)
    assert wait_until(lambda: hooks.text_calls == 1)

    bus.publish(AssistErrorEvent(job_id=hooks.text_ids[-1], message="model error"))
    assert wait_until(lambda: hooks.code_calls == 1)
    assert hooks.text_calls == 1, "an error must not be retried"
    complete_step(bus, hooks, "code", 1)


def test_done_for_a_different_job_id_is_ignored(bus, hooks, clock, cfg, make_controller) -> None:
    c = make_controller(cfg, bus, hooks, clock)
    start_and_settle(bus, clock, hooks, c, TRIGGER_LINES)
    expect_checkpoint(bus, hooks, 1)
    assert wait_until(lambda: hooks.text_calls == 1)

    real_id = hooks.text_ids[-1]
    bus.publish(AssistDoneEvent(job_id=real_id + 999, text="wrong job", ttft_s=0.0, tokens=0, tok_per_s=0.0))
    time.sleep(0.2)
    assert hooks.code_calls == 0, "a Done for an unrelated job id must not advance the cycle"
    assert c.state == "text"

    bus.publish(AssistDoneEvent(job_id=real_id, text="real answer", ttft_s=0.1, tokens=5, tok_per_s=50.0))
    assert wait_until(lambda: hooks.code_calls == 1)
    complete_step(bus, hooks, "code", 1)


def test_deferred_request_is_matched_to_the_first_assist_started_event_of_its_kind(bus, hooks, clock, cfg, make_controller) -> None:
    hooks.defer_text = True
    c = make_controller(cfg, bus, hooks, clock)
    start_and_settle(bus, clock, hooks, c, TRIGGER_LINES)
    expect_checkpoint(bus, hooks, 1)
    assert wait_until(lambda: hooks.text_calls == 1)
    assert hooks.text_ids[-1] is None
    assert c.state == "text"

    # an unrelated kind must not be mistaken for the awaited text job
    bus.publish(AssistStartedEvent(job_id=111, kind="code", question="unrelated"))
    time.sleep(0.1)

    # the real text job finally starts
    bus.publish(AssistStartedEvent(job_id=777, kind="text", question="the task"))
    # a second Started of the same kind (e.g. a stray requeue) must not steal the match
    bus.publish(AssistStartedEvent(job_id=888, kind="text", question="the task"))

    # a Done for the second (never matched) id is ignored
    bus.publish(AssistDoneEvent(job_id=888, text="wrong", ttft_s=0.0, tokens=0, tok_per_s=0.0))
    time.sleep(0.2)
    assert hooks.code_calls == 0

    bus.publish(AssistDoneEvent(job_id=777, text="right", ttft_s=0.1, tokens=5, tok_per_s=50.0))
    assert wait_until(lambda: hooks.code_calls == 1)
    complete_step(bus, hooks, "code", 1)


def test_new_massive_change_mid_cycle_aborts_and_restarts(bus, hooks, clock, cfg, make_controller) -> None:
    c = make_controller(cfg, bus, hooks, clock)
    start_and_settle(bus, clock, hooks, c, TRIGGER_LINES)
    expect_checkpoint(bus, hooks, 1)
    assert wait_until(lambda: hooks.text_calls == 1)
    assert c.triggers == 1

    unrelated_task = ["class Node:", "    def __init__(self, val):", "        self.val = val", "        self.next = None"]
    settle(bus, clock, c, unrelated_task)

    assert wait_until(lambda: hooks.checkpoint_calls == 2), "a new massive change mid-cycle must checkpoint again"
    assert c.triggers == 2
    assert wait_until(lambda: hooks.text_calls == 2), "the new cycle must (re)request text"
    bus.publish(CheckpointEvent(t=0.0, segment_id=None))  # the engine's own side effect

    # the stale (aborted) first text job's late Done must be ignored (no longer awaited)
    stale_id = hooks.text_ids[0]
    bus.publish(AssistDoneEvent(job_id=stale_id, text="stale", ttft_s=0.0, tokens=0, tok_per_s=0.0))
    time.sleep(0.2)
    assert hooks.code_calls == 0

    complete_step(bus, hooks, "text", 2)
    complete_step(bus, hooks, "code", 1)  # the FIRST code() call: cycle 1 was aborted before reaching it
    assert wait_until(lambda: c.state == "watching")


# ==================================================================================== #
# controller: deviation -> Code-only regeneration
# ==================================================================================== #


def test_deviation_regenerates_code_only_and_does_not_repeat_until_grown(bus, hooks, clock, cfg, make_controller) -> None:
    own_solution = (
        "def two_sum(nums, target):\n"
        "    seen = {}\n"
        "    for i, n in enumerate(nums):\n"
        "        if target - n in seen:\n"
        "            return [seen[n], i]\n"
        "        seen[n] = i\n"
        "    return []"
    )
    baseline = [
        "Problem: return indices of two numbers in nums that add up to target.",
        "def two_sum(nums, target):",
        "    pass",
    ]
    c = make_controller(cfg, bus, hooks, clock)
    start_and_settle(bus, clock, hooks, c, baseline)
    expect_checkpoint(bus, hooks, 1)
    complete_step(bus, hooks, "text", 1)
    complete_step(bus, hooks, "code", 1)
    assert wait_until(lambda: c.state == "watching")
    hooks.recent_code_text = own_solution  # "the app produced this code for the task"

    # the user writes a completely different (brute-force) approach instead, appended one line
    # at a time (each step a small, non-triggering change vs the previous SAMPLE)
    step1 = baseline[:2] + ["    result = []"]
    step2 = step1 + ["    for i in range(len(nums)):"]
    step3 = step2 + ["        for j in range(i + 1, len(nums)):"]

    settle(bus, clock, c, step1)
    assert wait_until(lambda: c.state == "watching")
    assert c.regenerations == 0, "only 1 deviating line so far: below min_changed_lines"

    settle(bus, clock, c, step2)
    assert wait_until(lambda: c.state == "watching")
    assert c.regenerations == 0, "only 2 deviating lines so far: below min_changed_lines"

    settle(bus, clock, c, step3)  # now 3 deviating lines: regenerate
    assert wait_until(lambda: c.regenerations == 1)
    assert hooks.checkpoint_calls == 1, "a deviation regeneration must not checkpoint"
    assert wait_until(lambda: hooks.code_calls == 2)
    complete_step(bus, hooks, "code", 2)
    assert wait_until(lambda: c.state == "watching")

    # republishing the SAME (still-deviating) code must not regenerate again
    settle(bus, clock, c, step3)
    assert wait_until(lambda: c.state == "watching")
    assert c.regenerations == 1

    # growing by 3 more deviating lines regenerates again
    step4 = step3 + [
        "            if nums[i] + nums[j] == target:",
        "                return [i, j]",
        '    print("no match found")',
    ]
    settle(bus, clock, c, step4)
    assert wait_until(lambda: c.regenerations == 2)
    assert hooks.checkpoint_calls == 1
    assert wait_until(lambda: hooks.code_calls == 3)
    complete_step(bus, hooks, "code", 3)
    assert wait_until(lambda: c.state == "watching")
    assert c.triggers == 1, "regenerations are never counted as full triggers"


# ==================================================================================== #
# controller: pause / reset / user checkpoint
# ==================================================================================== #


def test_pause_ignores_screen_events_and_resume_forces_a_capture(bus, hooks, clock, cfg, make_controller) -> None:
    c = make_controller(cfg, bus, hooks, clock)
    c.start()
    assert wait_until(lambda: hooks.capture_calls == [False])

    c.set_paused(True)
    assert wait_until(lambda: c.state == "paused")

    publish_screen(bus, lines=TRIGGER_LINES, changed=True)
    time.sleep(0.3)
    assert hooks.checkpoint_calls == 0
    assert c.state == "paused"

    c.set_paused(False)
    assert wait_until(lambda: hooks.capture_calls == [False, False])


def test_reset_forgets_the_last_task_and_forces_a_capture(bus, hooks, clock, cfg, make_controller) -> None:
    c = make_baseline(bus, hooks, clock, cfg, make_controller)
    assert hooks.checkpoint_calls == 1

    c.reset()
    assert wait_until(lambda: hooks.capture_calls == [False, False])
    # the exact same screen as before must now count as a NEW task (the last one was forgotten)
    settle(bus, clock, c, BASELINE_LINES, changed=False)
    expect_checkpoint(bus, hooks, 2)
    complete_step(bus, hooks, "text", 2)
    complete_step(bus, hooks, "code", 2)


def test_own_checkpoint_is_absorbed_silently(bus, hooks, clock, cfg, make_controller) -> None:
    c = make_controller(cfg, bus, hooks, clock)
    start_and_settle(bus, clock, hooks, c, TRIGGER_LINES)
    assert wait_until(lambda: hooks.checkpoint_calls == 1)
    # the engine's own checkpoint() call publishes this as a side effect -- must be a no-op here
    bus.publish(CheckpointEvent(t=0.0, segment_id=None))
    time.sleep(0.2)
    assert hooks.capture_calls == [False], "the controller's own checkpoint must not force another capture"
    assert c.state in ("checkpoint", "text")


def test_user_checkpoint_forces_a_capture_and_redo(bus, hooks, clock, cfg, make_controller) -> None:
    c = make_baseline(bus, hooks, clock, cfg, make_controller)
    assert hooks.checkpoint_calls == 1
    assert hooks.capture_calls == [False]

    # not one of the controller's own (already consumed by make_baseline's expect_checkpoint):
    # this is the user pressing the checkpoint hotkey themselves
    bus.publish(CheckpointEvent(t=0.0, segment_id=None))
    assert wait_until(lambda: hooks.capture_calls == [False, False])

    settle(bus, clock, c, BASELINE_LINES, changed=False)
    expect_checkpoint(bus, hooks, 2)
    complete_step(bus, hooks, "text", 2)
    complete_step(bus, hooks, "code", 2)


# ==================================================================================== #
# controller: context_text()
# ==================================================================================== #


def test_context_text_is_empty_when_off_or_nothing_read(bus, hooks, clock, cfg, make_controller) -> None:
    c = make_controller(cfg, bus, hooks, clock)
    assert c.context_text() == ""  # off

    c.start()
    assert wait_until(lambda: hooks.capture_calls == [False])
    assert c.context_text() == ""  # nothing settled/read yet


def test_context_text_carries_the_current_screen_after_an_evaluation(bus, hooks, clock, cfg, make_controller) -> None:
    c = make_controller(cfg, bus, hooks, clock)
    start_and_settle(bus, clock, hooks, c, TRIGGER_LINES)
    assert wait_until(lambda: "two_sum" in c.context_text())
    assert "Two Sum" in c.context_text()

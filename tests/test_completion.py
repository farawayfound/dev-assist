"""Offline tests for completion mode: matching helpers, prompt cleaning, caret helpers, and the
Completer state machine driven tick-by-tick with a fake tracker/backend/clock (no COM, no LLM)."""
from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from exocortex.bus import AssistDoneEvent, CompletionEvent, EventBus, drain
from exocortex.completion import Completer, consume_suggestion, is_monospace
from exocortex.config import Config
from exocortex.llm import prompts
from exocortex.screen.caret import CaretInfo, detect_placeholder


# --------------------------------------------------------------------------- helpers ----
def make_info(text_before: str = "", text_after: str = "", **kw: Any) -> CaretInfo:
    base = dict(
        hwnd=100, pid=42, app="Code.exe", field_name="Message input", field_hint="", control_type=50004,
        x=500, y=400, h=22, dpi=96, font_pt=10.0, font_name="Segoe UI",
        text_before=text_before, text_after=text_after, source="uia+msaa",
    )
    base.update(kw)
    return CaretInfo(**base)


class FakeTracker:
    available = True
    detail = "fake"

    def __init__(self) -> None:
        self.info: CaretInfo | None = None
        self.last_reject = ""

    def attach(self) -> bool:
        return True

    def detach(self) -> None:
        pass

    def probe(self) -> CaretInfo | None:
        return self.info


class FakeBackend:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str]] = []
        self.cancelled: list[int] = []
        self._next = 0

    def complete(self, prefix: str, suffix: str, field: str) -> int:
        self._next += 1
        self.calls.append((prefix, suffix, field))
        return self._next

    def cancel_job(self, job_id: int) -> None:
        self.cancelled.append(job_id)


class Harness:
    """Drives the Completer synchronously: no thread, explicit clock."""

    def __init__(self, **cfg_overrides: Any) -> None:
        self.cfg = Config()
        self.cfg.completion.debounce_ms = 1000
        self.cfg.completion.min_interval_ms = 500
        self.cfg.completion.max_lines = 3            # the tests below spell out 3-line ghost texts
        for k, v in cfg_overrides.items():
            setattr(self.cfg.completion, k, v)
        self.bus = EventBus()
        self.out = self.bus.subscribe()
        self.tracker = FakeTracker()
        self.backend = FakeBackend()
        self.now = 100.0
        self.c = Completer(self.cfg, self.bus, self.backend, self.tracker, clock=lambda: self.now)
        self.c._sub = self.bus.subscribe()

    def tick(self, dt: float = 0.15, info: CaretInfo | None = "keep") -> None:  # type: ignore[assignment]
        if info != "keep":
            self.tracker.info = info
        self.now += dt
        self.c._drain_bus()
        self.c._tick(self.now)

    def wait(self, seconds: float, info: CaretInfo | None = "keep") -> None:  # type: ignore[assignment]
        steps = max(1, int(seconds / 0.15))
        for _ in range(steps):
            self.tick(0.15, info)

    def finish(self, job_id: int, text: str, cancelled: bool = False) -> None:
        self.bus.publish(AssistDoneEvent(job_id=job_id, text=text, ttft_s=0.1, tokens=5, tok_per_s=50.0, cancelled=cancelled))

    def ghost_events(self) -> list[CompletionEvent]:
        return [e for e in drain(self.out) if isinstance(e, CompletionEvent)]

    def last_ghost(self) -> CompletionEvent | None:
        evs = self.ghost_events()
        return evs[-1] if evs else None


# ------------------------------------------------------------------- consume_suggestion --
@pytest.mark.parametrize(
    "suggestion, typed, expected",
    [
        ("foo bar", "", "foo bar"),
        ("foo bar", "foo", " bar"),
        ("foo bar", "foo bar", ""),
        ("foo bar", "foo bar   ", ""),
        ("foo bar", "fox", None),
        ("return x", "return  x", ""),
        ("\n    return x", "\n\treturn", " x"),
        ("\n    return x", "return", " x"),
        ("a b", "a  b c", None),
        ("x = 1", " x", " = 1"),
        ("Hello", "hello", None),
    ],
)
def test_consume_suggestion(suggestion: str, typed: str, expected: str | None) -> None:
    assert consume_suggestion(suggestion, typed) == expected


def test_is_monospace() -> None:
    assert is_monospace("Consolas", "x")
    assert is_monospace("JetBrains Mono", "x")
    assert not is_monospace("Segoe UI", "Code.exe")
    assert is_monospace("", "Code.exe")
    assert not is_monospace("", "chrome.exe")


# ------------------------------------------------------------------- prompt cleaning -------
def test_clean_completion_strips_fences_echo_and_limits_lines() -> None:
    assert prompts.clean_completion("def fo", "def foo():\n    return 1", 3) == "o():\n    return 1"
    assert prompts.clean_completion("x", "```python\nprint(1)\n```", 3) == "print(1)"
    assert prompts.clean_completion("a", "l1\nl2\nl3\nl4", 2) == "l1\nl2"
    assert prompts.clean_completion("a", "  \n ", 3) == ""
    assert prompts.clean_completion("the cat sat ", "the cat sat on the mat", 3) == "on the mat"
    assert prompts.clean_completion("hello wor", "world today", 3) == "ld today"


def test_trim_repetition() -> None:
    assert prompts.trim_repetition(" team, 2. 3. 4. 5. 6. 7. 8. 9. 1") == " team,"
    assert prompts.trim_repetition("the the the plan") == "the"
    assert prompts.trim_repetition("1. draft, 2. review, 3. present.") == "1. draft, 2. review, 3. present."
    assert prompts.trim_repetition("and ") == "and "


def test_system_and_user_complete() -> None:
    system = prompts.system_complete("I am an engineer.", "Python 3, terse")
    assert "Output ONLY the continuation" in system
    assert "Python 3, terse" in system and "I am an engineer." in system
    user = prompts.user_complete([("transcript", "t"), ("answers", ""), ("screen", "s")], "Code.exe - editor", "after")
    assert user.index("<transcript>") < user.index("<screen>") < user.index("<field>") < user.index("<after_cursor>")
    assert "<answers>" not in user   # blank blocks are skipped
    assert prompts.user_complete([], "f", "").count("(no session context yet)") == 1
    rendered = prompts.render_chatml(system, user, prompts.NO_THINK_PREFIX + "def fo")
    assert rendered.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\ndef fo")


# ------------------------------------------------------------------- caret helpers ----------
def test_detect_placeholder() -> None:
    assert detect_placeholder("", "Search…", "Search", "")
    assert detect_placeholder("", "Queue another message…", "Message input", "")
    assert detect_placeholder("Queue another message…", "", "Message input", "", 186, 13.3)   # Chromium quirk
    assert not detect_placeholder("ab", "", "Message input", "", 0, 13.3)
    assert not detect_placeholder("", "line one\nline two\n" * 10, "Doc", "")
    assert not detect_placeholder("hello world", "more", "x", "")
    assert not detect_placeholder("", "", "x", "")


def test_caret_info_properties() -> None:
    info = make_info("line one\nline two", font_pt=12.0, dpi=144)
    assert info.current_line == "line two"
    assert info.font_px == pytest.approx(24.0)
    assert info.key == (100, "Message input", 50004)
    assert make_info(font_pt=0.0).font_px == 0.0


# ------------------------------------------------------------------- state machine ------------
def test_no_target_publishes_hidden_and_never_requests() -> None:
    h = Harness()
    h.wait(3.0, info=None)
    assert h.backend.calls == []
    evs = h.ghost_events()
    assert evs and all(not e.visible for e in evs)


def test_request_only_after_debounce_and_streams_to_ghost() -> None:
    h = Harness()
    h.tick(info=make_info("hello wor"))
    h.wait(0.6)                                  # inside the debounce window
    assert h.backend.calls == []
    h.wait(0.6)                                  # past it
    assert len(h.backend.calls) == 1
    prefix, suffix, field = h.backend.calls[0]
    assert prefix == "hello wor" and "Code.exe" in field and "Message input" in field
    h.finish(1, "ld today")
    h.tick()
    ev = h.last_ghost()
    assert ev is not None and ev.visible and ev.text == "ld today"
    assert (ev.x, ev.y, ev.line_h) == (500, 400, 22)
    assert ev.font_px == pytest.approx(10.0 * 96 / 72)
    assert ev.current_line == "hello wor"


def test_typing_the_suggestion_advances_without_new_request() -> None:
    h = Harness()
    h.tick(info=make_info("hello wor"))
    h.wait(1.2)
    h.finish(1, "ld today")
    h.tick()
    h.ghost_events()
    h.tick(info=make_info("hello worl", x=508))
    ev = h.last_ghost()
    assert ev is not None and ev.visible and ev.text == "d today" and ev.x == 508
    h.tick(info=make_info("hello world ", x=530))
    ev = h.last_ghost()
    assert ev is not None and ev.text == "today"
    assert len(h.backend.calls) == 1          # never re-asked while the user follows the suggestion
    h.tick(info=make_info("hello world today", x=600))
    h.wait(1.5)                               # suggestion consumed: a new request only after the debounce
    assert len(h.backend.calls) == 2
    assert h.backend.calls[1][0] == "hello world today"


def test_deviation_hides_then_reasks_after_quiet_period() -> None:
    h = Harness()
    h.tick(info=make_info("hello wor"))
    h.wait(1.2)
    h.finish(1, "ld today")
    h.tick()
    h.ghost_events()
    h.tick(info=make_info("hello worx"))      # deviation
    ev = h.last_ghost()
    assert ev is not None and not ev.visible
    h.wait(0.5)
    assert len(h.backend.calls) == 1          # still quiet-period
    h.wait(0.8)
    assert len(h.backend.calls) == 2 and h.backend.calls[1][0] == "hello worx"


def test_backspacing_into_anchor_grows_suggestion_back() -> None:
    h = Harness()
    h.tick(info=make_info("hello wor"))
    h.wait(1.2)
    h.finish(1, "ld")
    h.tick()
    h.ghost_events()
    h.tick(info=make_info("hello wo"))
    ev = h.last_ghost()
    assert ev is not None and ev.visible and ev.text == "rld"
    assert len(h.backend.calls) == 1


def test_switching_field_clears_state_and_cancels_inflight() -> None:
    h = Harness()
    h.tick(info=make_info("hello wor"))
    h.wait(1.2)
    assert len(h.backend.calls) == 1
    h.tick(info=make_info("completely different", hwnd=200, field_name="Search"))
    assert h.backend.cancelled == [1]
    h.wait(1.2)
    assert len(h.backend.calls) == 2 and h.backend.calls[1][0] == "completely different"


def test_same_prefix_is_never_asked_twice() -> None:
    h = Harness()
    h.tick(info=make_info("hello wor"))
    h.wait(1.2)
    h.finish(1, "")                            # model had nothing to add
    h.wait(3.0)
    assert len(h.backend.calls) == 1


def test_abandoned_inflight_discards_the_normalised_key_so_it_can_be_asked_again() -> None:
    """upstream finding 6 (ported from mac-cortex): `_asked` stores the whitespace/case-normalised
    key (see `_norm_key`), so the abandon-after-timeout path must discard THAT key, not the raw
    prefix, or a stalled request permanently blacklists this exact caret position for any prefix
    normalisation actually changes (here: a double space collapses to one)."""
    from exocortex.completion import INFLIGHT_TIMEOUT_S

    h = Harness()
    prefix = "hello  world"  # double space -> _norm_key(prefix) != prefix: the bug's precondition
    h.tick(info=make_info(prefix))
    h.wait(1.2)
    assert len(h.backend.calls) == 1
    # the job never produces anything and INFLIGHT_TIMEOUT_S elapses: it is abandoned, and (with
    # debounce/min_interval long since satisfied) the same prefix must be askable again right away.
    h.wait(INFLIGHT_TIMEOUT_S + 1.0, info=make_info(prefix))
    assert len(h.backend.calls) == 2, "the abandoned prefix must not be permanently blacklisted"


def test_asked_prefix_eviction_drops_the_oldest_not_a_random_subset() -> None:
    """upstream finding 11 (ported from mac-cortex): a plain `set()`'s iteration order is
    hash-bucket order, not insertion order, so slicing "the last 100" kept an effectively random
    subset instead of the most recently asked prefixes. The OrderedDict-based `_asked` must evict
    the true oldest."""
    h = Harness()
    info = make_info("")
    for i in range(600):
        h.c._request(info, f"prefix number {i:04d}", "", h.now)
        h.now += 0.001
    kept = {int(k.rsplit(" ", 1)[-1]) for k in h.c._asked}
    assert kept == set(range(401, 600)), "eviction must keep exactly the most recently asked prefixes"


def test_pause_hides_and_blocks_requests() -> None:
    h = Harness()
    h.tick(info=make_info("hello wor"))
    h.c.set_paused(True)
    h.wait(2.0)
    assert h.backend.calls == []
    h.c.set_paused(False)
    h.wait(1.2)
    assert len(h.backend.calls) == 1


def test_empty_field_without_hint_is_skipped_but_placeholder_counts() -> None:
    h = Harness()
    h.tick(info=make_info(""))
    h.wait(1.5)
    assert h.backend.calls == []
    h.tick(info=make_info("", field_hint="Queue another message…"))
    h.wait(1.5)
    assert len(h.backend.calls) == 1 and "placeholder: Queue another message…" in h.backend.calls[0][2]


def test_cancelled_result_is_ignored_and_repetition_trimmed() -> None:
    h = Harness()
    h.tick(info=make_info("so I will "))
    h.wait(1.2)
    h.finish(1, "1. 2. 3. 4. 5.", cancelled=True)
    h.tick()
    ev = h.last_ghost()
    assert ev is None or not ev.visible
    h.tick(info=make_info("so I will x"))
    h.wait(1.5)
    h.finish(2, "draft it, 2. 3. 4. 5. 6.")
    h.tick()
    ev = h.last_ghost()
    assert ev is not None and ev.visible and ev.text == "draft it,"


def test_reset_forgets_suggestion() -> None:
    h = Harness()
    h.tick(info=make_info("hello wor"))
    h.wait(1.2)
    h.finish(1, "ld")
    h.tick()
    h.ghost_events()
    h.c.reset()
    h.tick()
    ev = h.last_ghost()
    assert ev is not None and not ev.visible


def test_inflight_request_times_out_and_can_be_retried() -> None:
    h = Harness()
    h.tick(info=make_info("hello wor"))
    h.wait(1.2)
    assert len(h.backend.calls) == 1
    h.wait(31.0)                               # the job never answers
    assert h.backend.cancelled == [1]
    h.tick(info=make_info("hello worl"))
    h.wait(1.5)
    assert len(h.backend.calls) == 2


# ------------------------------------------------------------------- blind editors ------------
from exocortex.completion import reconcile_fuzzy  # noqa: E402
from exocortex.screen.context import ScreenSnapshot  # noqa: E402
from exocortex.screen.ocr import OcrLine  # noqa: E402


def _blind_snapshot(extra: str = "", ts: float = 1.0) -> ScreenSnapshot:
    def L(text: str, x: int, y: int) -> OcrLine:
        return OcrLine(text=text, box=(x, y, 8 * len(text), 18), confidence=0.9)
    lines = [L("1", 880, 100), L("class Solution:", 900, 100), L("2", 880, 120), L("def twoSum(self, nums, target):", 932, 120),
             L("3", 880, 140), L("seen = {}", 964, 140), L("4", 880, 160), L("for i, n in enumerate(nums):", 964, 160), L("5", 880, 180)]
    if extra:
        lines.append(L(extra, 996, 180))
    return ScreenSnapshot(lines=tuple(lines), rect=(0, 0, 1600, 900), captured_at=ts)


def blind_info(x: int = 996, y: int = 180, **kw: Any) -> CaretInfo:
    return make_info("", "", app="chrome.exe", field_name="The editor is not accessible at this time", blind=True,
                     x=x, y=y, h=20, font_pt=10.0, font_name="Consolas", **kw)


class BlindHarness(Harness):
    def __init__(self) -> None:
        super().__init__()
        self.snapshot: ScreenSnapshot | None = _blind_snapshot()
        self.captures = 0
        self.c._screen_source = lambda: self.snapshot
        self.c._capture_now = self._capture

    def _capture(self) -> None:
        self.captures += 1


def test_reconcile_fuzzy_tolerates_ocr_noise_above() -> None:
    anchor = "class Solution:\n    def twoSum(self):\n        seen = {}\n        "
    prefix = "class So1ution:\n    def twoSum(self):\n        seen = {}\n        if n"    # OCR misread + typed "if n"
    assert reconcile_fuzzy(anchor, prefix, "if n in seen:") == " in seen:"
    assert reconcile_fuzzy(anchor, "totally different\ncontext\n        if", "if n") is None
    assert reconcile_fuzzy(anchor, anchor[:-4] + "    x", "if n in seen") is None                 # deviation
    assert reconcile_fuzzy(anchor, anchor[:-2], "if n") == "  if n"                               # backspaced


def test_blind_editor_prefix_comes_from_ocr_and_requests() -> None:
    h = BlindHarness()
    h.tick(info=blind_info())
    h.wait(1.3)
    assert len(h.backend.calls) == 1
    prefix, suffix, field = h.backend.calls[0]
    assert prefix.endswith("for i, n in enumerate(nums):\n            ") and "class Solution:" in prefix
    assert "not accessible" in field and "OCR" in field
    h.finish(1, "if n in seen:")
    h.tick()
    ev = h.last_ghost()
    assert ev is not None and ev.visible and ev.text == "if n in seen:" and ev.mono
    assert ev.current_line == " " * 12          # continuation lines align with the indentation


def test_blind_typing_advances_by_caret_movement_then_resyncs_with_ocr() -> None:
    h = BlindHarness()
    h.tick(info=blind_info())
    h.wait(1.3)
    h.finish(1, "if n in seen:")
    h.tick()
    h.ghost_events()
    # the caret moved 4 characters right (8 px each) with no new OCR frame yet: optimistic advance
    h.tick(info=blind_info(x=996 + 4 * 8))
    ev = h.last_ghost()
    assert ev is not None and ev.visible and ev.text == " in seen:"
    assert h.captures >= 1                        # a fresh OCR frame was requested
    assert len(h.backend.calls) == 1
    # the OCR frame arrives showing what was really typed: the ghost keeps going (same text and
    # position, so nothing new is published - check the completer's state instead)
    h.snapshot = _blind_snapshot(extra="if n", ts=2.0)
    h.tick(info=blind_info(x=996 + 4 * 8))
    last = h.c._last_event
    assert last is not None and last.visible and last.text == " in seen:"
    assert h.c._shown is not None and (h.c._last_prefix or "").endswith("if n")
    assert len(h.backend.calls) == 1


def test_blind_ocr_shows_a_deviation() -> None:
    h = BlindHarness()
    h.tick(info=blind_info())
    h.wait(1.3)
    h.finish(1, "if n in seen:")
    h.tick()
    h.ghost_events()
    h.snapshot = _blind_snapshot(extra="x = 5", ts=2.0)       # the user typed something else
    h.tick(info=blind_info(x=996 + 5 * 8))
    ev = h.last_ghost()
    assert ev is not None and not ev.visible


def test_blind_without_screen_source_requests_capture_and_waits() -> None:
    h = BlindHarness()
    h.snapshot = None
    h.tick(info=blind_info())
    h.wait(2.0)
    assert h.backend.calls == [] and h.captures >= 1


# ------------------------------------------------------------------- explanations (the plan) --
from exocortex.plan import PlanView  # noqa: E402

PLAN_CODE = (
    "class Solution:\n"
    "    def twoSum(self, nums, target):\n"
    "        # value -> index\n"
    "        seen = {}\n"
    "        for i, n in enumerate(nums):\n"
    "            if target - n in seen:\n"
    "                return [seen[target - n], i]\n"
    "            seen[n] = i"
)


class PlanBackend(FakeBackend):
    def __init__(self) -> None:
        super().__init__()
        self.view: PlanView | None = None
        self.typed: list[int] = []

    def plan(self) -> PlanView | None:
        return self.view

    def plan_typed(self, plan_id: int) -> None:
        self.typed.append(plan_id)


class PlanHarness(Harness):
    def __init__(self, **cfg_overrides: Any) -> None:
        super().__init__(**cfg_overrides)
        self.cfg.explain.enabled = True
        self.backend = PlanBackend()
        self.c._backend = self.backend


def test_explanations_comment_skip_keeps_the_rest_of_a_model_suggestion() -> None:
    h = PlanHarness()
    h.tick(info=make_info("x = 1\n"))
    h.wait(1.2)
    h.finish(1, "# note about y\ny = 2")
    h.tick()
    h.ghost_events()
    h.tick(info=make_info("x = 1\ny", y=422))                   # typed past the comment straight into the code
    ev = h.last_ghost()
    assert ev is not None and ev.visible and ev.text == " = 2"
    assert len(h.backend.calls) == 1                             # nothing regenerated


@pytest.mark.parametrize("level, tolerated", [("normal", True), ("minimal", True), ("detailed", True), ("none", False)])
def test_comment_skip_follows_the_comment_level_with_explanations_off(level: str, tolerated: bool) -> None:
    """Review finding: `prompts.system_complete` asks model completions for `cfg.assist.comments`-
    level comments even without a plan, but `_reconcile` only tolerated typing past one while
    explanations were on -- under the new "normal" default every skipped comment would have been a
    false deviation (suggestion dropped, another model round-trip for nothing). The gate now
    follows the level; only "none" (no comments requested) keeps the strict behaviour."""
    h = Harness()
    h.cfg.explain.enabled = False
    h.cfg.assist.comments = level
    h.tick(info=make_info("x = 1\n"))
    h.wait(1.2)
    h.finish(1, "# note about y\ny = 2")
    h.tick()
    h.ghost_events()
    h.tick(info=make_info("x = 1\ny", y=422))                   # typed past the comment straight into the code
    ev = h.last_ghost()
    assert ev is not None
    if tolerated:
        assert ev.visible and ev.text == " = 2"
        assert len(h.backend.calls) == 1                         # nothing regenerated
    else:
        assert not ev.visible                                    # a deviation: the ghost is taken down


def test_plan_streaming_waits_instead_of_asking_the_model() -> None:
    h = PlanHarness()
    h.backend.view = PlanView(id=7, solution="", is_code=False, done=False)   # walkthrough still streaming
    h.tick(info=make_info("class Solution:\n    "))
    h.wait(3.0)
    assert h.backend.calls == []
    evs = h.ghost_events()
    assert evs and all(not e.visible for e in evs)


def test_plan_serves_ghost_text_without_model_calls_and_reports_typed_out() -> None:
    h = PlanHarness()
    h.backend.view = PlanView(id=7, solution=PLAN_CODE, is_code=True, done=True)
    header = "class Solution:\n    def twoSum(self, nums, target):\n        "
    h.tick(info=make_info(header))
    ev = h.last_ghost()
    assert ev is not None and ev.visible
    assert ev.text == "# value -> index\n        seen = {}\n        for i, n in enumerate(nums):"
    h.wait(3.0)
    assert h.backend.calls == [] and h.c.plan_served >= 1
    # skipping the comment: the ghost drops it and keeps the code
    h.tick(info=make_info(header + "seen"))
    ev = h.last_ghost()
    assert ev is not None and ev.text == " = {}\n        for i, n in enumerate(nums):\n            if target - n in seen:"
    # ... all the way to the last line of code
    typed = "\n".join(ln for ln in PLAN_CODE.split("\n") if not ln.strip().startswith("#"))
    h.tick(info=make_info(typed[:-1]))
    ev = h.last_ghost()
    assert ev is not None and ev.visible and ev.text == "i" and h.backend.typed == []
    h.tick(info=make_info(typed))
    ev = h.last_ghost()
    assert ev is not None and not ev.visible
    assert h.backend.typed == [7]
    h.wait(2.0)
    assert h.backend.typed == [7] and h.backend.calls == []     # reported once; nothing asked after the end


def test_leaving_the_plan_falls_back_to_the_model_and_returning_cancels_it() -> None:
    h = PlanHarness()
    h.backend.view = PlanView(id=7, solution=PLAN_CODE, is_code=True, done=True)
    header = "class Solution:\n    def twoSum(self, nums, target):\n        "
    h.tick(info=make_info(header))
    h.tick(info=make_info(header + "print(nums)"))               # not in the plan
    h.wait(1.5)
    assert len(h.backend.calls) == 1 and h.backend.calls[0][0].endswith("print(nums)")
    h.tick(info=make_info(header + "seen = {}"))                 # back on the plan
    assert h.backend.cancelled == [1]
    ev = h.last_ghost()
    assert ev is not None and ev.visible and ev.text.startswith("\n        for i, n in enumerate(nums):")


def test_plan_is_ignored_when_explanations_are_off() -> None:
    h = PlanHarness()
    h.cfg.explain.enabled = False
    h.backend.view = PlanView(id=7, solution=PLAN_CODE, is_code=True, done=True)
    h.tick(info=make_info("class Solution:\n    "))
    h.wait(1.5)
    assert len(h.backend.calls) == 1 and h.c.plan_served == 0


def test_plan_ghost_in_a_blind_editor_follows_the_caret_between_ocr_frames() -> None:
    """The plan ghost is also the 'shown' suggestion, so blind mode can advance it by caret dx
    (one char per char_w) instead of hiding until the next OCR frame."""
    h = BlindHarness()
    h.cfg.explain.enabled = True
    backend = PlanBackend()
    h.backend = backend
    h.c._backend = backend
    solution = ("class Solution:\n    def twoSum(self, nums, target):\n        seen = {}\n"
                "        for i, n in enumerate(nums):\n            if n in seen:\n                return [seen[n], i]")
    backend.view = PlanView(id=3, solution=solution, is_code=True, done=True)
    h.tick(info=blind_info())
    ev = h.last_ghost()
    assert ev is not None and ev.visible and ev.text == "if n in seen:\n                return [seen[n], i]"
    assert h.backend.calls == []
    h.tick(info=blind_info(x=996 + 4 * 8))          # typed "if n" (4 chars), no OCR frame yet
    ev = h.last_ghost()
    assert ev is not None and ev.visible and ev.text == " in seen:\n                return [seen[n], i]"



def test_defaults_predict_several_lines() -> None:
    cfg = Config()
    assert cfg.completion.max_lines >= 8 and cfg.completion.max_tokens >= 100
    h = PlanHarness(max_lines=cfg.completion.max_lines)
    h.backend.view = PlanView(id=7, solution=PLAN_CODE, is_code=True, done=True)
    h.tick(info=make_info('class Solution:\n    def twoSum(self, nums, target):\n        '))
    ev = h.last_ghost()
    assert ev is not None and ev.visible and ev.text.count('\n') == 5      # the whole rest of the plan

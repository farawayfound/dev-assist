"""Tests for exocortex.context.TranscriptWindow (the anchored window) and Transcript.lines()."""
from __future__ import annotations

import threading

import pytest

from exocortex.bus import Source, TranscriptSegment
from exocortex.context import TranscriptWindow
from exocortex.transcript import Transcript


def _seg(seg_id: int, text: str, *, t: float | None = None, label: str = "Speaker 1") -> TranscriptSegment:
    start = float(seg_id * 3) if t is None else t
    return TranscriptSegment(id=seg_id, source=Source.SYSTEM, speaker_id="spk1", speaker_label=label,
                             text=text, t_start=start, t_end=start + 1.0, confidence=0.9)


def _fill(tr: Transcript, start: int, stop: int, width: int = 30) -> None:
    for i in range(start, stop):
        tr.add(_seg(i, f"utterance {i:03d} " + "w" * width))


def _line(i: int, width: int = 30) -> str:
    return f"[{i * 3 // 60:02d}:{i * 3 % 60:02d}] Speaker 1: utterance {i:03d} " + "w" * width


# ---- Transcript.lines ----------------------------------------------------------

def test_claim_compaction_with_max_chars_takes_the_oldest_lines_that_fit_and_keeps_the_rest() -> None:
    """2026-10-06 addendum: a compaction prompt has only the notes lane's room, so a long backlog
    is folded in order, chunk by chunk, like the notes job's own backlog."""
    tr = Transcript()
    w = TranscriptWindow(tr)
    w.restore_compaction(None, "line one\nline two\nline three\nline four")
    claimed = w.claim_compaction(None, max_chars=19)
    assert claimed == ("", "line one\nline two")
    assert w.has_pending(None)
    assert w.claim_compaction(None, max_chars=5) == ("", "line three")       # at least one line, always
    assert w.claim_compaction(None, max_chars=None) == ("", "line four")
    assert w.claim_compaction(None) is None
    assert not w.has_pending(None)


def test_lines_returns_ids_and_formatted_lines() -> None:
    tr = Transcript()
    tr.add(_seg(1, "hello", t=5))
    tr.add(_seg(2, "world", t=65, label="Me"))
    assert tr.lines() == [(1, "[00:05] Speaker 1: hello"), (2, "[01:05] Me: world")]


def test_lines_filters_like_window_text() -> None:
    tr = Transcript()
    _fill(tr, 0, 10)
    assert [i for i, _ in tr.lines(since_id=3)] == [4, 5, 6, 7, 8, 9]
    assert [i for i, _ in tr.lines(until_id=3)] == [0, 1, 2, 3]
    assert [i for i, _ in tr.lines(since_id=2, until_id=5)] == [3, 4, 5]
    joined = "\n".join(t for _i, t in tr.lines(since_id=2, until_id=5))
    assert joined == tr.window_text(10_000, since_id=2, until_id=5)


def test_lines_empty_transcript() -> None:
    assert Transcript().lines() == []


def test_lines_reflects_renames() -> None:
    tr = Transcript()
    tr.add(_seg(1, "hi"))
    tr.rename_speaker("spk1", "Alice")
    assert tr.lines()[0][1].endswith("Alice: hi")


# ---- render: within budget -----------------------------------------------------

def test_render_empty_transcript() -> None:
    w = TranscriptWindow(Transcript())
    r = w.render(since_id=None, until_id=None, max_chars=1000, trim_step=0.25)
    assert (r.text, r.anchor_moved, r.dropped_text, r.chars, r.line_count) == ("", False, "", 0, 0)
    assert r.scope is None


def test_render_within_budget_returns_everything_and_keeps_anchor() -> None:
    tr = Transcript()
    _fill(tr, 0, 5)
    w = TranscriptWindow(tr)
    r = w.render(since_id=None, until_id=None, max_chars=10_000, trim_step=0.25)
    assert r.text == "\n".join(_line(i) for i in range(5))
    assert r.anchor_moved is False and r.dropped_text == "" and r.line_count == 5 and r.chars == len(r.text)
    assert r.windowed is False and w.anchor(None) is None
    assert not w.has_pending(None)


def test_text_exactly_at_the_budget_does_not_trim() -> None:
    tr = Transcript()
    _fill(tr, 0, 3)
    exact = len("\n".join(_line(i) for i in range(3)))
    w = TranscriptWindow(tr)
    assert w.render(since_id=None, until_id=None, max_chars=exact, trim_step=0.25).anchor_moved is False
    assert w.render(since_id=None, until_id=None, max_chars=exact - 1, trim_step=0.25).anchor_moved is True


def test_zero_budget_returns_empty_without_moving_the_anchor() -> None:
    tr = Transcript()
    _fill(tr, 0, 5)
    w = TranscriptWindow(tr)
    r = w.render(since_id=None, until_id=None, max_chars=0, trim_step=0.25)
    assert r.text == "" and not r.anchor_moved and w.anchor(None) is None


# ---- render: prefix stability --------------------------------------------------

def test_prefix_stable_while_the_transcript_grows_within_budget() -> None:
    tr = Transcript()
    w = TranscriptWindow(tr)
    previous = ""
    for i in range(20):
        tr.add(_seg(i, f"utterance {i:03d} " + "w" * 30))
        r = w.render(since_id=None, until_id=None, max_chars=5000, trim_step=0.25)
        assert r.text.startswith(previous)
        assert not r.anchor_moved
        previous = r.text


def test_after_a_trim_the_next_turns_extend_the_previous_text() -> None:
    tr = Transcript()
    _fill(tr, 0, 30)
    w = TranscriptWindow(tr)
    budget = 1000
    r1 = w.render(since_id=None, until_id=None, max_chars=budget, trim_step=0.25)
    assert r1.anchor_moved is True
    previous = r1.text
    moved_after = 0
    for i in range(30, 40):
        tr.add(_seg(i, f"utterance {i:03d} " + "w" * 30))
        r = w.render(since_id=None, until_id=None, max_chars=budget, trim_step=0.25)
        if r.anchor_moved:
            moved_after += 1
            previous = r.text
            continue
        assert r.text.startswith(previous), "prefix must stay byte-identical until the next trim"
        previous = r.text
    assert moved_after <= 3     # ~one trim per 250 chars of growth (about 4 lines), not one per turn


def test_sliding_window_changes_prefix_on_every_turn_but_stepping_does_not() -> None:
    """The point of the whole design: count how many turns change the prompt prefix."""
    def run(trim_step: float) -> int:
        tr = Transcript()
        w = TranscriptWindow(tr)
        first_lines: list[str] = []
        for i in range(120):
            tr.add(_seg(i, f"utterance {i:03d} " + "w" * 30))
            r = w.render(since_id=None, until_id=None, max_chars=1000, trim_step=trim_step)
            first_lines.append(r.text.split("\n", 1)[0])
        return sum(1 for a, b in zip(first_lines, first_lines[1:]) if a != b)

    sliding, stepped = run(0.0), run(0.25)
    assert sliding > 60          # once the budget is hit, nearly every turn moves the start
    assert stepped < sliding / 3


def test_anchor_advances_by_at_least_trim_step_of_the_budget() -> None:
    tr = Transcript()
    _fill(tr, 0, 30)
    w = TranscriptWindow(tr)
    budget = 1000
    r = w.render(since_id=None, until_id=None, max_chars=budget, trim_step=0.25)
    assert r.anchor_moved
    assert len(r.text) <= budget * 0.75            # the kept window fits in budget * (1 - step)
    assert r.windowed is True


def test_trim_step_zero_drops_only_what_is_needed_like_window_text() -> None:
    tr = Transcript()
    _fill(tr, 0, 30)
    for budget in (500, 1000, 1234):
        w = TranscriptWindow(tr)
        r = w.render(since_id=None, until_id=None, max_chars=budget, trim_step=0.0)
        assert r.text == tr.window_text(budget)


def test_dropped_text_is_exactly_the_dropped_lines() -> None:
    tr = Transcript()
    _fill(tr, 0, 30)
    w = TranscriptWindow(tr)
    r = w.render(since_id=None, until_id=None, max_chars=1000, trim_step=0.25)
    all_lines = [_line(i) for i in range(30)]
    assert r.dropped_text + "\n" + r.text == "\n".join(all_lines)
    assert r.line_count == len(r.text.split("\n"))


def test_at_least_one_line_dropped_and_newest_line_kept_with_huge_trim_step() -> None:
    tr = Transcript()
    _fill(tr, 0, 30)
    w = TranscriptWindow(tr)
    r = w.render(since_id=None, until_id=None, max_chars=1000, trim_step=5.0)     # clamped to 0.9
    assert r.text.endswith(_line(29)) and r.anchor_moved and r.dropped_text


def test_newest_line_always_survives_even_when_it_alone_exceeds_the_budget() -> None:
    tr = Transcript()
    _fill(tr, 0, 3)
    tr.add(_seg(3, "x" * 500))
    w = TranscriptWindow(tr)
    r = w.render(since_id=None, until_id=None, max_chars=100, trim_step=0.25)
    assert r.text == tr.window_text(100)            # the tail of the newest line, like window_text
    assert len(r.text) == 100 and r.text.endswith("x" * 100)
    assert r.anchor_moved and r.line_count == 1


def test_single_oversized_line_is_tail_cut_without_moving_the_anchor() -> None:
    tr = Transcript()
    tr.add(_seg(1, "y" * 500))
    w = TranscriptWindow(tr)
    r = w.render(since_id=None, until_id=None, max_chars=100, trim_step=0.25)
    assert len(r.text) == 100 and not r.anchor_moved and r.dropped_text == "" and r.windowed
    assert not w.has_pending(None)


def test_budget_shrinking_later_trims_again() -> None:
    tr = Transcript()
    _fill(tr, 0, 30)
    w = TranscriptWindow(tr)
    r1 = w.render(since_id=None, until_id=None, max_chars=1500, trim_step=0.25)
    r2 = w.render(since_id=None, until_id=None, max_chars=600, trim_step=0.25)
    assert r2.anchor_moved and len(r2.text) <= 600 and len(r2.text) < len(r1.text)
    assert r2.text.endswith(_line(29))


def test_until_id_limits_the_window() -> None:
    tr = Transcript()
    _fill(tr, 0, 10)
    w = TranscriptWindow(tr)
    r = w.render(since_id=None, until_id=4, max_chars=10_000, trim_step=0.25)
    assert r.text.split("\n")[-1] == _line(4) and r.line_count == 5


# ---- compaction buffer ---------------------------------------------------------

def test_pending_accumulates_across_two_trims_and_claim_clears_it() -> None:
    tr = Transcript()
    _fill(tr, 0, 30)
    w = TranscriptWindow(tr)
    r1 = w.render(since_id=None, until_id=None, max_chars=1000, trim_step=0.25)
    _fill(tr, 30, 60)
    r2 = w.render(since_id=None, until_id=None, max_chars=1000, trim_step=0.25)
    assert r1.anchor_moved and r2.anchor_moved
    assert w.has_pending(None)
    claimed = w.claim_compaction(None)
    assert claimed is not None
    previous, pending = claimed
    assert previous == ""
    assert pending == r1.dropped_text + "\n" + r2.dropped_text
    assert w.claim_compaction(None) is None and not w.has_pending(None)


def test_claim_returns_the_current_summary() -> None:
    tr = Transcript()
    _fill(tr, 0, 30)
    w = TranscriptWindow(tr)
    w.render(since_id=None, until_id=None, max_chars=1000, trim_step=0.25)
    w.set_summary(None, "so far: greetings")
    prev, pending = w.claim_compaction(None)  # type: ignore[misc]
    assert prev == "so far: greetings" and pending
    assert w.earlier_summary(None) == "so far: greetings"      # claiming does not touch the summary


def test_claim_is_none_when_nothing_pending() -> None:
    w = TranscriptWindow(Transcript())
    assert w.claim_compaction(None) is None
    assert w.claim_compaction(7) is None


def test_restore_prepends_to_text_dropped_in_the_meantime() -> None:
    tr = Transcript()
    _fill(tr, 0, 30)
    w = TranscriptWindow(tr)
    w.render(since_id=None, until_id=None, max_chars=1000, trim_step=0.25)
    _prev, claimed = w.claim_compaction(None)  # type: ignore[misc]
    _fill(tr, 30, 60)
    r2 = w.render(since_id=None, until_id=None, max_chars=1000, trim_step=0.25)
    w.restore_compaction(None, claimed)
    _prev2, again = w.claim_compaction(None)  # type: ignore[misc]
    assert again == claimed + "\n" + r2.dropped_text


def test_restore_into_an_empty_buffer_and_restore_empty_text() -> None:
    w = TranscriptWindow(Transcript())
    w.restore_compaction(None, "")
    assert w.claim_compaction(None) is None
    w.restore_compaction(None, "old text")
    assert w.claim_compaction(None) == ("", "old text")


def test_pending_buffer_is_bounded_to_four_budgets_on_a_line_boundary() -> None:
    tr = Transcript()
    w = TranscriptWindow(tr)
    budget = 300
    for start in range(0, 600, 20):
        _fill(tr, start, start + 20)
        w.render(since_id=None, until_id=None, max_chars=budget, trim_step=0.25)
    _prev, pending = w.claim_compaction(None)  # type: ignore[misc]
    assert len(pending) <= 4 * budget
    assert pending.startswith("[") and pending.endswith("w" * 30)      # starts on a whole line
    assert all(ln.startswith("[") for ln in pending.split("\n"))


# ---- scopes --------------------------------------------------------------------

def test_scopes_are_independent() -> None:
    tr = Transcript()
    _fill(tr, 0, 60)
    w = TranscriptWindow(tr)
    whole = w.render(since_id=None, until_id=None, max_chars=1000, trim_step=0.25)
    since = w.render(since_id=40, until_id=None, max_chars=10_000, trim_step=0.25)
    assert whole.anchor_moved and not since.anchor_moved
    assert since.text.startswith(_line(41)) and since.scope == 40
    assert w.anchor(None) is not None and w.anchor(40) is None
    w.set_summary(None, "whole")
    w.set_summary(40, "since")
    assert (w.earlier_summary(None), w.earlier_summary(40), w.earlier_summary(41)) == ("whole", "since", "")


def test_scope_window_never_shows_lines_at_or_before_since_id() -> None:
    tr = Transcript()
    _fill(tr, 0, 60)
    w = TranscriptWindow(tr)
    r = w.render(since_id=50, until_id=None, max_chars=200, trim_step=0.25)
    assert r.text.endswith(_line(59))
    assert _line(50) not in r.text and r.anchor_moved
    assert w.anchor(50) is not None and w.anchor(50) > 50


def test_forget_scope_drops_only_that_scope() -> None:
    tr = Transcript()
    _fill(tr, 0, 60)
    w = TranscriptWindow(tr)
    w.render(since_id=None, until_id=None, max_chars=500, trim_step=0.25)
    w.render(since_id=10, until_id=None, max_chars=500, trim_step=0.25)
    w.set_summary(10, "s")
    w.forget_scope(10)
    assert w.anchor(10) is None and w.earlier_summary(10) == "" and not w.has_pending(10)
    assert w.anchor(None) is not None
    w.forget_scope(999)         # unknown scope: no error


def test_reset_clears_everything() -> None:
    tr = Transcript()
    _fill(tr, 0, 60)
    w = TranscriptWindow(tr)
    w.render(since_id=None, until_id=None, max_chars=500, trim_step=0.25)
    w.set_summary(None, "kept?")
    w.reset()
    assert w.anchor(None) is None and w.earlier_summary(None) == "" and w.claim_compaction(None) is None
    r = w.render(since_id=None, until_id=None, max_chars=100_000, trim_step=0.25)
    assert r.line_count == 60 and not r.windowed


def test_a_cleared_transcript_starts_the_scope_afresh() -> None:
    tr = Transcript()
    _fill(tr, 0, 30)
    w = TranscriptWindow(tr)
    w.render(since_id=None, until_id=None, max_chars=500, trim_step=0.25)
    assert w.anchor(None) is not None
    tr.clear()
    r = w.render(since_id=None, until_id=None, max_chars=500, trim_step=0.25)
    assert r.text == "" and w.anchor(None) is None
    tr.add(_seg(0, "fresh start"))
    assert w.render(since_id=None, until_id=None, max_chars=500, trim_step=0.25).text.endswith("fresh start")


def test_window_is_thread_safe() -> None:
    tr = Transcript()
    w = TranscriptWindow(tr)
    errors: list[Exception] = []

    def writer() -> None:
        try:
            for i in range(200):
                tr.add(_seg(i, f"utterance {i:03d} " + "w" * 30))
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    def reader() -> None:
        try:
            for _ in range(200):
                r = w.render(since_id=None, until_id=None, max_chars=800, trim_step=0.25)
                assert len(r.text) <= 800
                w.claim_compaction(None)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    ts = [threading.Thread(target=writer), threading.Thread(target=reader), threading.Thread(target=reader)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert not errors


@pytest.mark.parametrize("step", [0.1, 0.25, 0.5])
def test_kept_window_fits_within_budget_times_one_minus_step(step: float) -> None:
    tr = Transcript()
    _fill(tr, 0, 80)
    w = TranscriptWindow(tr)
    r = w.render(since_id=None, until_id=None, max_chars=1200, trim_step=step)
    assert r.anchor_moved and len(r.text) <= 1200 * (1 - step) + 1


def test_a_staged_summary_is_promoted_at_the_next_anchor_move() -> None:
    """2026-10-06 (second live run): installing a finished summary at once re-evaluated the
    prompt on two extra turns; staged, it enters the prompt exactly when the anchor moves."""
    tr = Transcript()
    for i in range(1, 31):
        tr.add(TranscriptSegment(id=i, source=Source.SYSTEM, speaker_id="s", speaker_label="S", text=f"line {i:03d} " + "x" * 30,
                                 t_start=float(i), t_end=float(i) + 1, confidence=0.9))
    w = TranscriptWindow(tr)
    first = w.render(since_id=None, until_id=None, max_chars=400, trim_step=0.25)
    assert first.anchor_moved
    w.stage_summary(None, "so far: lines one to ten")
    assert w.earlier_summary(None) == "" and w.staged_summary(None) == "so far: lines one to ten"
    again = w.render(since_id=None, until_id=None, max_chars=400, trim_step=0.25)     # fits: no move
    assert not again.anchor_moved and w.earlier_summary(None) == ""
    for i in range(31, 61):
        tr.add(TranscriptSegment(id=i, source=Source.SYSTEM, speaker_id="s", speaker_label="S", text=f"line {i:03d} " + "x" * 30,
                                 t_start=float(i), t_end=float(i) + 1, confidence=0.9))
    moved = w.render(since_id=None, until_id=None, max_chars=400, trim_step=0.25)
    assert moved.anchor_moved
    assert w.earlier_summary(None) == "so far: lines one to ten" and w.staged_summary(None) == ""
    # a claim folds into the newest text, staged or active
    w.stage_summary(None, "newer")
    assert w.claim_compaction(None)[0] == "newer"   # type: ignore[index]
    w.set_summary(None, "direct")                   # set_summary is immediate and clears the staged one
    assert w.earlier_summary(None) == "direct" and w.staged_summary(None) == ""

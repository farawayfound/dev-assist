"""Tests for the forced, on-demand compaction added for the overlay's "Compact" control
(2026-10-07): ``TranscriptWindow.compact_now`` and ``ContextManager.compact_now``.

Unlike an overflow-driven trim, ``compact_now`` ignores the budget: it is the user asking to
free room now. It drops everything but the newest couple of lines of the current window into the
pending buffer, moves the anchor past them and returns the characters dropped, so the caller can
queue a compaction job that folds them into the running summary.
"""
from __future__ import annotations

from exocortex.bus import Source, TranscriptSegment
from exocortex.context import TranscriptWindow
from exocortex.transcript import Transcript


def _seg(seg_id: int, text: str, *, label: str = "Speaker 1") -> TranscriptSegment:
    start = float(seg_id * 3)
    return TranscriptSegment(id=seg_id, source=Source.SYSTEM, speaker_id="spk1", speaker_label=label,
                             text=text, t_start=start, t_end=start + 1.0, confidence=0.9)


def _fill(tr: Transcript, start: int, stop: int) -> None:
    for i in range(start, stop):
        tr.add(_seg(i, f"utterance {i:03d}"))


def test_compact_now_drops_all_but_the_newest_lines_into_pending_and_moves_the_anchor() -> None:
    tr = Transcript()
    _fill(tr, 0, 10)
    w = TranscriptWindow(tr)
    # Nothing windowed yet: the full window is visible.
    before = w.render(since_id=None, until_id=None, max_chars=10_000, trim_step=0.25)
    assert before.line_count == 10 and not before.windowed

    dropped = w.compact_now(None, keep_lines=2)
    assert dropped > 0
    assert w.has_pending(None)                     # the dropped text waits to be folded

    after = w.render(since_id=None, until_id=None, max_chars=10_000, trim_step=0.25)
    assert after.line_count == 2                    # only the newest two lines remain live
    assert after.windowed                           # the window no longer starts at the top
    assert "utterance 009" in after.text and "utterance 000" not in after.text

    # What was dropped is claimable for the summary job, oldest first, and nothing is lost.
    claimed = w.claim_compaction(None)
    assert claimed is not None
    _prev, pending = claimed
    assert "utterance 000" in pending and "utterance 007" in pending
    assert "utterance 008" not in pending           # kept live, not dropped


def test_compact_now_is_a_noop_when_the_window_is_already_small() -> None:
    tr = Transcript()
    _fill(tr, 0, 2)
    w = TranscriptWindow(tr)
    assert w.compact_now(None, keep_lines=2) == 0
    assert not w.has_pending(None)


def test_compact_now_respects_a_checkpoint_scope() -> None:
    tr = Transcript()
    _fill(tr, 0, 10)
    w = TranscriptWindow(tr)
    # Scope = 4 means "only segments after id 4" (the checkpoint boundary).
    dropped = w.compact_now(4, keep_lines=1)
    assert dropped > 0
    after = w.render(since_id=4, until_id=None, max_chars=10_000, trim_step=0.25)
    assert after.line_count == 1 and "utterance 009" in after.text
    # The whole-meeting scope is untouched: compacting one scope never disturbs another.
    whole = w.render(since_id=None, until_id=None, max_chars=10_000, trim_step=0.25)
    assert whole.line_count == 10 and not whole.windowed


def test_compact_now_preserves_the_whole_dropped_text_even_past_the_old_pending_bound() -> None:
    """A forced drop can exceed the budget-sized pending limit; it must not be truncated before
    its summary job runs."""
    tr = Transcript()
    for i in range(40):
        tr.add(_seg(i, f"line {i:03d} " + "x" * 300))   # ~40 * ~310 chars, well past the 8000 default bound
    w = TranscriptWindow(tr)
    dropped = w.compact_now(None, keep_lines=1)
    assert dropped > 8000
    _prev, pending = w.claim_compaction(None)
    assert "line 000" in pending and "line 038" in pending


def test_manager_compact_now_delegates_to_the_window() -> None:
    from exocortex.context import ContextManager, ContextStore
    from exocortex.config import Config

    tr = Transcript()
    _fill(tr, 0, 8)
    cfg = Config()
    store = ContextStore(None)
    mgr = ContextManager(cfg, store, tr)
    dropped = mgr.compact_now(None, keep_lines=2)
    assert dropped > 0
    assert mgr.window.has_pending(None)

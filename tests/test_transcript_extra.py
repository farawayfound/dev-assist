"""Additional verifier tests for exocortex.transcript.Transcript: boundary conditions,
empty-input behaviour, and rename propagation into every export/read path."""
from __future__ import annotations

import json
import threading
import time

from exocortex.bus import Source, TranscriptSegment
from exocortex.transcript import Transcript, _format_line, _format_timestamp, _trim_to_whole_lines


def _seg(
    seg_id: int,
    t_start: float,
    *,
    t_end: float | None = None,
    source: Source = Source.SYSTEM,
    speaker_id: str = "spk1",
    speaker_label: str = "Speaker 1",
    text: str = "hello",
    confidence: float = 0.9,
    is_question: bool = False,
) -> TranscriptSegment:
    return TranscriptSegment(
        id=seg_id,
        source=source,
        speaker_id=speaker_id,
        speaker_label=speaker_label,
        text=text,
        t_start=t_start,
        t_end=t_end if t_end is not None else t_start + 1.0,
        confidence=confidence,
        is_question=is_question,
    )


# ---- last(): negative n ------------------------------------------------------

def test_last_negative_n_returns_empty() -> None:
    t = Transcript()
    t.add(_seg(0, 0.0))
    assert t.last(n=-1) == []
    assert t.last(n=-100) == []


# ---- empty-transcript reads ---------------------------------------------------

def test_empty_transcript_reads() -> None:
    t = Transcript()
    assert t.word_count() == 0
    assert t.word_count(since_id=5) == 0
    assert t.text_since(None) == ""
    assert t.text_since(0) == ""
    assert t.segments() == []
    assert t.last() == []
    assert t.last_id() is None
    assert t.to_markdown("Empty") == "# Empty\n"
    assert t.to_jsonl() == ""


# ---- until_id combined with max_chars trimming --------------------------------

def test_window_text_until_id_and_trim_combined() -> None:
    t = Transcript()
    for i in range(10):
        t.add(_seg(i, t_start=float(i), text=f"segment number {i} with padding words here"))
    # only consider ids 0..5 (6 segments), then trim to the newest 2 of those
    cutoff_lines = [_format_line(_seg(i, float(i), text=f"segment number {i} with padding words here")) for i in range(6)]
    last2 = "\n".join(cutoff_lines[-2:])
    out = t.window_text(len(last2), until_id=5)
    assert out == last2
    # segments beyond the cutoff must never leak in even though they'd fit
    assert "segment number 6" not in out
    assert "segment number 9" not in out


def test_window_text_until_id_zero_and_negative() -> None:
    t = Transcript()
    for i in range(3):
        t.add(_seg(i, t_start=float(i), text=f"t{i}"))
    # until_id below every stored id -> no segments considered
    assert t.window_text(1000, until_id=-1) == ""


# ---- rename propagation into every read/export path ----------------------------

def test_rename_speaker_reflected_everywhere() -> None:
    t = Transcript()
    t.add(_seg(0, t_start=0.0, speaker_id="spk1", speaker_label="Speaker 1", text="hi"))
    t.add(_seg(1, t_start=1.0, speaker_id="spk2", speaker_label="Speaker 2", text="yo"))
    t.rename_speaker("spk1", "Alice")

    assert t.last(n=2)[0].speaker_label == "Alice"
    assert "Alice" in t.to_markdown("T")
    assert "Speaker 1" not in t.to_markdown("T")

    jsonl_objs = [json.loads(line) for line in t.to_jsonl().strip("\n").split("\n")]
    labels = {o["speaker_id"]: o["speaker_label"] for o in jsonl_objs}
    assert labels["spk1"] == "Alice"
    assert labels["spk2"] == "Speaker 2"

    assert "Alice" in t.text_since(None)


def test_rename_speaker_preserves_identity_fields() -> None:
    """dataclasses.replace should only touch speaker_label; everything else, including id/order,
    stays exactly as stored (regression guard against accidental field clobbering)."""
    t = Transcript()
    seg = _seg(
        0,
        t_start=1.25,
        t_end=2.5,
        speaker_id="spk1",
        speaker_label="Speaker 1",
        text="original text",
        confidence=0.77,
        is_question=True,
    )
    t.add(seg)
    t.rename_speaker("spk1", "Zed")
    got = t.segments()[0]
    assert got.id == seg.id
    assert got.text == seg.text
    assert got.t_start == seg.t_start
    assert got.t_end == seg.t_end
    assert got.confidence == seg.confidence
    assert got.is_question == seg.is_question
    assert got.source == seg.source
    assert got.speaker_label == "Zed"


# ---- to_jsonl excludes the computed `duration` property -------------------------

def test_to_jsonl_excludes_duration_property() -> None:
    t = Transcript()
    t.add(_seg(0, t_start=1.0, t_end=4.0))
    obj = json.loads(t.to_jsonl().strip("\n"))
    assert "duration" not in obj
    assert set(obj.keys()) == {
        "id", "source", "speaker_id", "speaker_label", "text", "t_start", "t_end", "confidence", "is_question",
    }


# ---- _format_timestamp boundary values -------------------------------------------

def test_format_timestamp_boundaries() -> None:
    assert _format_timestamp(0.0) == "00:00"
    assert _format_timestamp(59.9) == "00:59"
    assert _format_timestamp(60.0) == "01:00"
    assert _format_timestamp(3599.0) == "59:59"
    assert _format_timestamp(3600.0) == "1:00:00"
    assert _format_timestamp(-5.0) == "00:00"  # never goes negative


# ---- _trim_to_whole_lines direct unit tests --------------------------------------

def test_trim_to_whole_lines_exact_fit_boundary() -> None:
    lines = ["aaa", "bb", "c"]  # joined = "aaa\nbb\nc" == 8 chars
    joined = "\n".join(lines)
    assert len(joined) == 8
    assert _trim_to_whole_lines(lines, 8) == joined
    # one char short of the full join -> drops the oldest whole line ("aaa"), keeps "bb\nc"
    assert _trim_to_whole_lines(lines, 7) == "bb\nc"


def test_trim_to_whole_lines_empty_list() -> None:
    assert _trim_to_whole_lines([], 100) == ""


# ---- thread-safety: rename never yields a torn read ------------------------------

def test_concurrent_rename_never_shows_mixed_labels_for_same_speaker() -> None:
    """While one thread continuously renames spk1 between two labels, readers must always see a
    single consistent label across ALL segments belonging to spk1 (proves rename swaps the whole
    list atomically under the lock rather than mutating segments one at a time)."""
    t = Transcript()
    for i in range(50):
        t.add(_seg(i, t_start=float(i), speaker_id="spk1", speaker_label="Speaker 1"))

    stop = threading.Event()
    inconsistent: list[str] = []

    def renamer() -> None:
        toggle = False
        while not stop.is_set():
            t.rename_speaker("spk1", "A" if toggle else "B")
            toggle = not toggle

    def reader() -> None:
        for _ in range(500):
            labels = {s.speaker_label for s in t.segments() if s.speaker_id == "spk1"}
            if len(labels) > 1:
                inconsistent.append(str(labels))

    r_thread = threading.Thread(target=renamer)
    reader_threads = [threading.Thread(target=reader) for _ in range(4)]
    r_thread.start()
    for th in reader_threads:
        th.start()
    for th in reader_threads:
        th.join()
    stop.set()
    r_thread.join()

    assert inconsistent == []


def test_concurrent_add_last_id_is_monotonic_snapshot() -> None:
    """last_id() must always be an id that was actually added, never a partially-constructed value,
    while adds are happening concurrently on other threads."""
    t = Transcript()
    all_ids: set[int] = set()
    lock = threading.Lock()

    def worker(base: int) -> None:
        for i in range(200):
            seg_id = base * 200 + i
            t.add(_seg(seg_id, t_start=float(seg_id)))
            with lock:
                all_ids.add(seg_id)

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(5)]
    for th in threads:
        th.start()
    for _ in range(20):
        lid = t.last_id()
        if lid is not None:
            assert lid in range(0, 5 * 200)
        time.sleep(0.001)
    for th in threads:
        th.join()

    assert t.last_id() in all_ids

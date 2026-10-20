"""Tests for exocortex.transcript.Transcript."""
from __future__ import annotations

import json
import threading

from exocortex.bus import Source, TranscriptSegment
from exocortex.transcript import Transcript


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


# ---- ordering ---------------------------------------------------------------

def test_add_and_segments_preserve_order() -> None:
    t = Transcript()
    segs = [_seg(i, t_start=float(i * 2), text=f"seg{i}") for i in range(5)]
    for s in segs:
        t.add(s)
    assert t.segments() == segs
    assert [s.id for s in t.segments()] == [0, 1, 2, 3, 4]
    assert t.last_id() == 4


def test_segments_since_id_filters() -> None:
    t = Transcript()
    for i in range(5):
        t.add(_seg(i, t_start=float(i)))
    assert [s.id for s in t.segments(since_id=2)] == [3, 4]
    assert [s.id for s in t.segments(since_id=None)] == [0, 1, 2, 3, 4]


def test_last_returns_most_recent_and_filters_by_source() -> None:
    t = Transcript()
    system_segs = [_seg(i, t_start=float(i), source=Source.SYSTEM) for i in range(3)]
    for s in system_segs:
        t.add(s)
    assert t.last(n=2) == system_segs[-2:]

    mic_seg = _seg(10, t_start=20.0, source=Source.MIC, speaker_id="me", speaker_label="Me", text="hi")
    t.add(mic_seg)
    assert t.last(source=Source.MIC, n=1) == [mic_seg]
    assert t.last(source=Source.SYSTEM, n=1) == [system_segs[-1]]
    assert t.last(n=0) == []
    assert t.last(source=Source.MIC, n=5) == [mic_seg]


def test_last_id_none_when_empty() -> None:
    assert Transcript().last_id() is None


# ---- window_text formatting & trimming --------------------------------------

def test_window_text_basic_format_mmss() -> None:
    t = Transcript()
    t.add(_seg(0, t_start=5.0, text="Hello there", speaker_label="Speaker 1"))
    t.add(_seg(1, t_start=65.0, text="Second line", speaker_label="Speaker 2"))
    out = t.window_text(1000)
    assert out == "[00:05] Speaker 1: Hello there\n[01:05] Speaker 2: Second line"


def test_window_text_hour_format() -> None:
    t = Transcript()
    t.add(_seg(0, t_start=3665.0, text="hour test"))
    out = t.window_text(1000)
    assert out == "[1:01:05] Speaker 1: hour test"


def test_window_text_empty() -> None:
    assert Transcript().window_text(1000) == ""


def test_window_text_trims_from_front_on_whole_lines() -> None:
    t = Transcript()
    for i in range(10):
        t.add(_seg(i, t_start=float(i), text=f"segment number {i} with some extra padding words"))
    full = t.window_text(100_000)
    all_lines = full.split("\n")
    assert len(all_lines) == 10

    last3 = "\n".join(all_lines[-3:])
    trimmed = t.window_text(len(last3))
    assert trimmed == last3
    # confirm whole-line trimming: no line is partially cut
    for line in trimmed.split("\n"):
        assert line in all_lines


def test_window_text_single_line_longer_than_max_keeps_tail() -> None:
    t = Transcript()
    long_text = " ".join(f"word{i}" for i in range(200))
    t.add(_seg(0, t_start=0.0, text=long_text))
    full = t.window_text(100_000)
    trimmed = t.window_text(20)
    assert len(trimmed) == 20
    assert trimmed == full[-20:]


def test_window_text_until_id() -> None:
    t = Transcript()
    for i in range(5):
        t.add(_seg(i, t_start=float(i), text=f"t{i}"))
    out = t.window_text(100_000, until_id=2)
    lines = out.split("\n")
    assert len(lines) == 3
    assert "t3" not in out and "t4" not in out
    assert "t2" in out


def test_window_text_max_chars_zero_or_negative() -> None:
    t = Transcript()
    t.add(_seg(0, t_start=0.0, text="anything"))
    assert t.window_text(0) == ""
    assert t.window_text(-5) == ""


# ---- window_text since_id (checkpoint boundary) -------------------------------------


def test_window_text_since_id_excludes_older_segments() -> None:
    t = Transcript()
    for i in range(5):
        t.add(_seg(i, t_start=float(i), text=f"t{i}"))
    out = t.window_text(100_000, since_id=2)
    lines = out.split("\n")
    assert len(lines) == 2
    assert "t0" not in out and "t1" not in out and "t2" not in out
    assert "t3" in out and "t4" in out


def test_window_text_since_id_none_keeps_everything() -> None:
    t = Transcript()
    for i in range(3):
        t.add(_seg(i, t_start=float(i), text=f"t{i}"))
    out = t.window_text(100_000, since_id=None)
    assert "t0" in out and "t1" in out and "t2" in out


def test_window_text_since_id_and_until_id_combined() -> None:
    t = Transcript()
    for i in range(6):
        t.add(_seg(i, t_start=float(i), text=f"t{i}"))
    out = t.window_text(100_000, since_id=1, until_id=4)
    lines = out.split("\n")
    assert len(lines) == 3
    assert all(f"t{i}" not in out for i in (0, 1, 5))
    assert "t2" in out and "t3" in out and "t4" in out


def test_window_text_since_id_past_last_segment_is_empty() -> None:
    t = Transcript()
    for i in range(3):
        t.add(_seg(i, t_start=float(i), text=f"t{i}"))
    assert t.window_text(100_000, since_id=2) == ""


# ---- rename rewrites stored segments -----------------------------------------

def test_rename_speaker_rewrites_all_matching_segments() -> None:
    t = Transcript()
    t.add(_seg(0, t_start=0.0, speaker_id="spk1", speaker_label="Speaker 1", text="a"))
    t.add(_seg(1, t_start=1.0, speaker_id="spk1", speaker_label="Speaker 1", text="b"))
    t.add(_seg(2, t_start=2.0, speaker_id="spk2", speaker_label="Speaker 2", text="c"))

    t.rename_speaker("spk1", "Alice")

    segs = t.segments()
    assert [s.speaker_label for s in segs] == ["Alice", "Alice", "Speaker 2"]
    assert [s.id for s in segs] == [0, 1, 2]
    assert [s.text for s in segs] == ["a", "b", "c"]
    # other fields untouched
    assert segs[0].speaker_id == "spk1"
    assert segs[0].t_start == 0.0


def test_rename_speaker_reflected_in_window_text() -> None:
    t = Transcript()
    t.add(_seg(0, t_start=5.0, speaker_id="spk1", speaker_label="Speaker 1", text="hi"))
    t.rename_speaker("spk1", "Bob")
    assert t.window_text(1000) == "[00:05] Bob: hi"


def test_rename_speaker_no_match_is_noop() -> None:
    t = Transcript()
    seg = _seg(0, t_start=0.0, speaker_id="spk1", speaker_label="Speaker 1")
    t.add(seg)
    t.rename_speaker("spk-does-not-exist", "Nobody")
    assert t.segments() == [seg]


# ---- to_markdown ---------------------------------------------------------------

def test_to_markdown_format() -> None:
    t = Transcript()
    t.add(_seg(0, t_start=5.0, speaker_label="Speaker 1", text="Hello"))
    md = t.to_markdown("My Meeting")
    assert md == "# My Meeting\n\n**[00:05] Speaker 1:** Hello\n"


def test_to_markdown_multiple_segments_and_empty() -> None:
    t = Transcript()
    t.add(_seg(0, t_start=0.0, text="one"))
    t.add(_seg(1, t_start=1.0, text="two"))
    md = t.to_markdown("Title")
    assert md.startswith("# Title\n\n")
    assert "**[00:00] Speaker 1:** one" in md
    assert "**[00:01] Speaker 1:** two" in md

    assert Transcript().to_markdown("Empty") == "# Empty\n"


# ---- to_jsonl round trip -------------------------------------------------------

def test_to_jsonl_round_trip_fields() -> None:
    t = Transcript()
    seg = _seg(
        0,
        t_start=1.5,
        t_end=3.25,
        source=Source.MIC,
        speaker_id="me",
        speaker_label="Me",
        text="hello world",
        confidence=0.87,
        is_question=True,
    )
    t.add(seg)
    jsonl = t.to_jsonl()
    assert jsonl.endswith("\n")
    line = jsonl.strip("\n")
    obj = json.loads(line)
    assert obj == {
        "id": 0,
        "source": "mic",
        "speaker_id": "me",
        "speaker_label": "Me",
        "text": "hello world",
        "t_start": 1.5,
        "t_end": 3.25,
        "confidence": 0.87,
        "is_question": True,
    }


def test_to_jsonl_multiple_lines_and_empty() -> None:
    t = Transcript()
    t.add(_seg(0, t_start=0.0, source=Source.SYSTEM, text="first"))
    t.add(_seg(1, t_start=1.0, source=Source.MIC, speaker_id="me", speaker_label="Me", text="second"))
    lines = t.to_jsonl().strip("\n").split("\n")
    assert len(lines) == 2
    objs = [json.loads(line) for line in lines]
    assert [o["text"] for o in objs] == ["first", "second"]
    assert [o["source"] for o in objs] == ["system", "mic"]

    assert Transcript().to_jsonl() == ""


# ---- text_since / word_count ----------------------------------------------------

def test_text_since_and_word_count() -> None:
    t = Transcript()
    for i in range(4):
        t.add(_seg(i, t_start=float(i * 10), text=f"line{i}"))

    ts = t.text_since(1)
    assert ts == "[00:20] Speaker 1: line2\n[00:30] Speaker 1: line3"

    ts_all = t.text_since(None)
    assert ts_all.count("\n") == 3

    assert t.text_since(3) == ""

    assert t.word_count() == 4  # "lineN" is a single word per segment
    assert t.word_count(since_id=1) == 2


# ---- clear ---------------------------------------------------------------------

def test_clear_empties_store() -> None:
    t = Transcript()
    t.add(_seg(0, t_start=0.0))
    t.add(_seg(1, t_start=1.0))
    t.clear()
    assert t.segments() == []
    assert t.last_id() is None
    assert t.to_jsonl() == ""


# ---- concurrency smoke -----------------------------------------------------------

def test_concurrent_add_1000_segments() -> None:
    t = Transcript()
    n_threads = 10
    per_thread = 100

    def worker(base: int) -> None:
        for i in range(per_thread):
            seg_id = base * per_thread + i
            t.add(_seg(seg_id, t_start=float(seg_id), text=f"t{seg_id}"))

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(n_threads)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    segs = t.segments()
    assert len(segs) == n_threads * per_thread
    ids = sorted(s.id for s in segs)
    assert ids == list(range(n_threads * per_thread))
    assert t.word_count() == n_threads * per_thread


def test_concurrent_add_and_rename_smoke() -> None:
    t = Transcript()

    def add_worker(base: int) -> None:
        for i in range(50):
            seg_id = base * 50 + i
            t.add(_seg(seg_id, t_start=float(seg_id), speaker_id="spk1", speaker_label="Speaker 1"))

    def rename_worker() -> None:
        for _ in range(50):
            t.rename_speaker("spk1", "Renamed")

    threads = [threading.Thread(target=add_worker, args=(k,)) for k in range(4)]
    threads.append(threading.Thread(target=rename_worker))
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    assert len(t.segments()) == 200
    # rename_worker ran last-ish; regardless, no crash and store integrity holds
    assert all(s.speaker_id == "spk1" for s in t.segments())


# ---- head_text (upstream finding 3, Windows fix) ----------------------------

def test_head_text_empty_and_non_positive_budget() -> None:
    t = Transcript()
    assert t.head_text(100) == ("", None)
    t.add(_seg(1, t_start=0.0, text="hello"))
    assert t.head_text(0) == ("", None)
    assert t.head_text(100, since_id=1) == ("", None)


def test_head_text_returns_everything_when_it_fits() -> None:
    t = Transcript()
    for i in range(3):
        t.add(_seg(i, t_start=float(i), text=f"line {i}"))
    text, last = t.head_text(10_000)
    assert text == t.text_since(None)
    assert last == 2


def test_head_text_keeps_the_oldest_lines_and_chaining_covers_every_segment_once() -> None:
    """The notes job walks the transcript with this: oldest chunk first, checkpoint = the last
    segment included, next call continues from there -- nothing skipped, nothing repeated."""
    t = Transcript()
    for i in range(10):
        t.add(_seg(i, t_start=float(i), text=f"segment number {i:02d}"))
    full_lines = t.text_since(None).split("\n")
    budget = len(full_lines[0]) * 3 + 2   # room for exactly three lines (two joining newlines)
    text, last = t.head_text(budget)
    assert text.split("\n") == full_lines[:3]
    assert last == 2

    seen: list[str] = []
    since: int | None = None
    for _ in range(20):
        chunk, last = t.head_text(budget, since_id=since)
        if last is None:
            break
        assert len(chunk) <= budget
        seen.extend(chunk.split("\n"))
        since = last
    assert seen == full_lines
    assert since == 9


def test_head_text_overlong_first_line_is_returned_whole() -> None:
    t = Transcript()
    t.add(_seg(1, t_start=0.0, text="x" * 500))
    t.add(_seg(2, t_start=1.0, text="short"))
    text, last = t.head_text(50)
    assert text == t.text_since(None).split("\n")[0]
    assert last == 1

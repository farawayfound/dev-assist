"""Thread-safe ordered store of TranscriptSegment.

Holds every segment produced by the speech pipeline for the life of a session, in the
order they were added (segment ids are expected to be monotonically increasing, as
handed out by the Engine's `next_id()`). Renaming a speaker rewrites `speaker_label`
on every already-stored segment for that speaker, so history reflects the latest name.
"""
from __future__ import annotations

import dataclasses
import json
import logging
import threading

from exocortex.bus import Source, TranscriptSegment

log = logging.getLogger(__name__)


def _format_timestamp(t_start: float) -> str:
    """Render seconds-since-session-start as "mm:ss", or "h:mm:ss" once an hour has passed."""
    total = max(0, int(t_start))
    hours, rem = divmod(total, 3600)
    minutes, seconds = divmod(rem, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def _format_line(seg: TranscriptSegment) -> str:
    """"[mm:ss] Label: text" for one segment."""
    return f"[{_format_timestamp(seg.t_start)}] {seg.speaker_label}: {seg.text}"


def _trim_to_whole_lines(lines: list[str], max_chars: int) -> str:
    """Join `lines` (oldest -> newest), dropping from the front on whole-line boundaries
    until the joined text fits within `max_chars`. If even the single newest line alone
    exceeds `max_chars`, its tail (the last `max_chars` characters) is returned instead."""
    if max_chars <= 0 or not lines:
        return ""
    kept: list[str] = []
    total = 0
    for line in reversed(lines):
        extra = len(line) + (1 if kept else 0)  # +1 for the joining "\n" once we have a line
        if total + extra > max_chars:
            if not kept:
                return line[-max_chars:]
            break
        kept.append(line)
        total += extra
    kept.reverse()
    return "\n".join(kept)


class Transcript:
    """Thread-safe ordered store of TranscriptSegment. Labels of past segments are rewritten on rename."""

    def __init__(self) -> None:
        self._segments: list[TranscriptSegment] = []
        self._lock = threading.RLock()

    def add(self, seg: TranscriptSegment) -> None:
        """Append a segment to the store."""
        with self._lock:
            self._segments.append(seg)
        log.debug("transcript: added segment id=%s source=%s", seg.id, seg.source)

    def segments(self, since_id: int | None = None) -> list[TranscriptSegment]:
        """All stored segments, oldest first. With `since_id`, only those with id > since_id."""
        with self._lock:
            if since_id is None:
                return list(self._segments)
            return [s for s in self._segments if s.id > since_id]

    def last(self, source: Source | None = None, n: int = 1) -> list[TranscriptSegment]:
        """The most recent `n` segments (oldest -> newest), optionally filtered by source."""
        if n <= 0:
            return []
        with self._lock:
            pool = self._segments if source is None else [s for s in self._segments if s.source == source]
            return pool[-n:]

    def rename_speaker(self, speaker_id: str, label: str) -> None:
        """Rewrite `speaker_label` on every stored segment (past and future references) for `speaker_id`."""
        with self._lock:
            self._segments = [
                dataclasses.replace(s, speaker_label=label) if s.speaker_id == speaker_id else s
                for s in self._segments
            ]
        log.debug("transcript: renamed speaker %s -> %r", speaker_id, label)

    def window_text(self, max_chars: int, *, until_id: int | None = None, since_id: int | None = None) -> str:
        """Oldest -> newest "[mm:ss] Label: text" lines, trimmed from the front on whole lines
        to fit `max_chars`. With `until_id`, only segments with id <= until_id are considered;
        with `since_id`, only segments with id > since_id (a checkpoint boundary)."""
        with self._lock:
            segs = [
                s for s in self._segments
                if (until_id is None or s.id <= until_id) and (since_id is None or s.id > since_id)
            ]
            lines = [_format_line(s) for s in segs]
        return _trim_to_whole_lines(lines, max_chars)

    def text_since(self, since_id: int | None) -> str:
        """Same line format as `window_text`, for segments with id > since_id (all if None)."""
        segs = self.segments(since_id)
        return "\n".join(_format_line(s) for s in segs)

    def head_text(self, max_chars: int, *, since_id: int | None = None) -> tuple[str, int | None]:
        """upstream finding 3 (Windows fix): the OLDEST segments after `since_id` that fit in
        `max_chars` (whole lines, the same "[mm:ss] Label: text" format as `window_text`), plus
        the id of the last segment included -- the checkpoint a consumer should advance to.

        `window_text` keeps the NEWEST lines (a prompt prefix that should track the present);
        this is for a consumer that walks the transcript in order in bounded chunks and must not
        skip anything: the notes job. Chaining calls with the returned id covers every segment
        exactly once. Returns ``("", None)`` when nothing follows `since_id` (or `max_chars` is
        not positive). If even the first line alone exceeds `max_chars` it is returned whole,
        never truncated, so the consumer still advances past it.
        """
        if max_chars <= 0:
            return "", None
        kept: list[str] = []
        total = 0
        last_id: int | None = None
        for s in self.segments(since_id):
            line = _format_line(s)
            extra = len(line) + (1 if kept else 0)  # +1 for the joining "\n" once we have a line
            if kept and total + extra > max_chars:
                break
            kept.append(line)
            total += extra
            last_id = s.id
        return "\n".join(kept), last_id

    def word_count(self, since_id: int | None = None) -> int:
        """Total word count across the selected segments' text."""
        segs = self.segments(since_id)
        return sum(len(s.text.split()) for s in segs)

    def last_id(self) -> int | None:
        """id of the most recently added segment, or None if the transcript is empty."""
        with self._lock:
            return self._segments[-1].id if self._segments else None

    def to_markdown(self, title: str) -> str:
        """"# <title>" followed by one paragraph per segment: "**[mm:ss] Label:** text"."""
        segs = self.segments()
        paragraphs = [f"**[{_format_timestamp(s.t_start)}] {s.speaker_label}:** {s.text}" for s in segs]
        if not paragraphs:
            return f"# {title}\n"
        return f"# {title}\n\n" + "\n\n".join(paragraphs) + "\n"

    def to_jsonl(self) -> str:
        """One JSON object per line with every TranscriptSegment field (source as its `.value` string)."""
        segs = self.segments()
        if not segs:
            return ""
        out_lines: list[str] = []
        for s in segs:
            d = dataclasses.asdict(s)
            d["source"] = s.source.value
            out_lines.append(json.dumps(d, ensure_ascii=False))
        return "\n".join(out_lines) + "\n"

    def clear(self) -> None:
        """Remove all stored segments."""
        with self._lock:
            self._segments.clear()
        log.debug("transcript: cleared")

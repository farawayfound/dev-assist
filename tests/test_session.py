"""Tests for exocortex.session.SessionRecorder (fake transcript/assistant, no real deps)."""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import pytest

from exocortex.bus import (
    EventBus,
    FinalTranscriptEvent,
    SessionEvent,
    Source,
    SpeakerRenamedEvent,
    TranscriptSegment,
    drain,
)
from exocortex.config import Config, config_as_dict
from exocortex.session import SessionRecorder


class FakeTranscript:
    """Minimal stand-in for exocortex.transcript.Transcript."""

    def __init__(self) -> None:
        self.body = ""

    def to_markdown(self, title: str) -> str:
        return f"# {title}\n\n{self.body}"

    def to_jsonl(self) -> str:
        return self.body

    def segments(self, since_id: int | None = None) -> list[TranscriptSegment]:
        return []


class FakeAssistant:
    """Minimal stand-in for exocortex.llm.assistant.Assistant."""

    def __init__(self) -> None:
        self.notes_markdown = ""
        self.answers: list[tuple[float, str, str]] = []


def _make_cfg(tmp_path: Path, autosave_interval_s: int = 30) -> Config:
    cfg = Config()
    cfg.paths.sessions_dir = str(tmp_path)
    cfg.paths.autosave_interval_s = autosave_interval_s
    return cfg


def _make_segment(seg_id: int = 1, text: str = "hello") -> TranscriptSegment:
    return TranscriptSegment(
        id=seg_id,
        source=Source.SYSTEM,
        speaker_id="spk1",
        speaker_label="Speaker 1",
        text=text,
        t_start=0.0,
        t_end=1.0,
        confidence=0.9,
    )


def test_start_creates_dated_folder_and_publishes_new_event(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path)
    bus = EventBus()
    sub = bus.subscribe()
    recorder = SessionRecorder(cfg, bus, FakeTranscript(), FakeAssistant())

    folder = recorder.start()

    assert folder.is_dir()
    assert folder.parent == tmp_path
    datetime.strptime(folder.name, "%Y-%m-%d_%H-%M-%S")  # raises if the format doesn't match
    event = sub.get(timeout=1)
    assert isinstance(event, SessionEvent)
    assert event.kind == "new"
    assert event.path == str(folder)
    recorder.stop()


def test_folder_property_before_start_raises() -> None:
    cfg = Config()
    recorder = SessionRecorder(cfg, EventBus(), FakeTranscript(), FakeAssistant())
    with pytest.raises(RuntimeError):
        _ = recorder.folder


def test_stop_writes_every_file_with_current_content(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path)
    bus = EventBus()
    transcript = FakeTranscript()
    transcript.body = "hello world"
    assistant = FakeAssistant()
    assistant.notes_markdown = "## Summary\nstuff happened"
    assistant.answers = [(75.4, "What is X?", "X is Y.")]
    recorder = SessionRecorder(cfg, bus, transcript, assistant)

    folder = recorder.start()
    recorder.stop()

    assert (folder / "transcript.md").read_text(encoding="utf-8") == transcript.to_markdown(folder.name)
    assert (folder / "transcript.jsonl").read_text(encoding="utf-8") == transcript.to_jsonl()
    assert (folder / "notes.md").read_text(encoding="utf-8") == assistant.notes_markdown

    answers_text = (folder / "answers.md").read_text(encoding="utf-8")
    assert "## [01:15] What is X?" in answers_text
    assert "X is Y." in answers_text

    session_data = json.loads((folder / "session.json").read_text(encoding="utf-8"))
    assert session_data["config"] == config_as_dict(cfg)
    assert session_data["speakers"] == {}
    datetime.fromisoformat(session_data["started_at"])  # must not raise


def test_export_saves_immediately_and_returns_folder(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path)
    bus = EventBus()
    transcript = FakeTranscript()
    recorder = SessionRecorder(cfg, bus, transcript, FakeAssistant())

    folder = recorder.start()
    transcript.body = "updated content"
    result = recorder.export()

    assert result == folder
    assert "updated content" in (folder / "transcript.md").read_text(encoding="utf-8")
    recorder.stop()


def test_export_picks_up_pending_speaker_rename(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path)
    bus = EventBus()
    recorder = SessionRecorder(cfg, bus, FakeTranscript(), FakeAssistant())

    folder = recorder.start()
    bus.publish(SpeakerRenamedEvent(speaker_id="spk1", new_label="Alice"))
    recorder.export()

    data = json.loads((folder / "session.json").read_text(encoding="utf-8"))
    assert data["speakers"] == {"spk1": "Alice"}
    recorder.stop()


def test_export_publishes_exported_session_event(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path)
    bus = EventBus()
    sub = bus.subscribe()
    recorder = SessionRecorder(cfg, bus, FakeTranscript(), FakeAssistant())
    recorder.start()
    sub.get(timeout=1)  # the "new" event from start()

    recorder.export()

    event = sub.get(timeout=1)
    assert isinstance(event, SessionEvent)
    assert event.kind == "exported"
    recorder.stop()


def test_autosave_thread_saves_after_final_transcript_event(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path, autosave_interval_s=1)
    bus = EventBus()
    transcript = FakeTranscript()
    recorder = SessionRecorder(cfg, bus, transcript, FakeAssistant())

    folder = recorder.start()
    transcript.body = "autosaved content"
    bus.publish(FinalTranscriptEvent(segment=_make_segment()))

    transcript_path = folder / "transcript.md"
    deadline = time.monotonic() + 5.0
    saved = False
    while time.monotonic() < deadline:
        if transcript_path.is_file() and "autosaved content" in transcript_path.read_text(encoding="utf-8"):
            saved = True
            break
        time.sleep(0.1)

    assert saved, "autosave thread did not persist the change within the timeout"
    recorder.stop()


def test_autosave_thread_does_not_save_without_a_change(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path, autosave_interval_s=1)
    bus = EventBus()
    sub = bus.subscribe()
    recorder = SessionRecorder(cfg, bus, FakeTranscript(), FakeAssistant())

    recorder.start()
    assert sub.get(timeout=1).kind == "new"
    time.sleep(2.5)  # a couple of autosave ticks with nothing published

    # The periodic loop must not have published a "saved" event on its own; stop() always
    # performs exactly one final save regardless of "dirty" state, so that is the only one
    # that should show up on the bus.
    recorder.stop()
    kinds = [event.kind for event in drain(sub) if isinstance(event, SessionEvent)]
    assert kinds == ["saved"]

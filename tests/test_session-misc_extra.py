"""Extra regression tests for exocortex.session, exocortex.models_dl and exocortex.logging_setup.

Written by an independent verifier pass over the "session-misc" module: these cover boundary
conditions and thread-safety/resource-leak scenarios not exercised by test_session.py /
test_models_dl.py, and add missing coverage for logging_setup.py (no dedicated test file existed).
"""
from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path

import httpx
import pytest

from exocortex import models_dl
from exocortex.bus import EventBus, SessionEvent, SpeakerRenamedEvent
from exocortex.config import Config
from exocortex.logging_setup import setup_logging
from exocortex.models_dl import ModelDownloadError, ensure_models
from exocortex.session import SessionRecorder, _atomic_write_text, _format_mmss

# ---------------------------------------------------------------------------
# session.py: fakes shared with test_session.py's shape
# ---------------------------------------------------------------------------


class FakeTranscript:
    def __init__(self) -> None:
        self.body = ""

    def to_markdown(self, title: str) -> str:
        return f"# {title}\n\n{self.body}"

    def to_jsonl(self) -> str:
        return self.body

    def segments(self, since_id: int | None = None) -> list:
        return []


class FakeAssistant:
    def __init__(self) -> None:
        self.notes_markdown = ""
        self.answers: list[tuple[float, str, str]] = []


def _make_cfg(tmp_path: Path, autosave_interval_s: int = 30) -> Config:
    cfg = Config()
    cfg.paths.sessions_dir = str(tmp_path)
    cfg.paths.autosave_interval_s = autosave_interval_s
    return cfg


# ---------------------------------------------------------------------------
# _format_mmss: hour rollover
# ---------------------------------------------------------------------------


def test_format_mmss_rolls_over_after_an_hour() -> None:
    # A meeting co-pilot has to handle meetings that run past 60 minutes; answers.md should
    # use the same "h:mm:ss" style as transcript.md instead of overflowing minutes past 59.
    assert _format_mmss(3661) == "1:01:01"
    assert _format_mmss(59) == "00:59"
    assert _format_mmss(3599) == "59:59"
    assert _format_mmss(3600) == "1:00:00"


def test_format_mmss_matches_transcript_timestamp_format() -> None:
    from exocortex.transcript import _format_timestamp

    for t in (0, 1, 59, 60, 3599, 3600, 3661, 7325):
        assert _format_mmss(t) == _format_timestamp(t)


# ---------------------------------------------------------------------------
# stop()/export() before start()
# ---------------------------------------------------------------------------


def test_stop_before_start_is_a_safe_noop() -> None:
    cfg = Config()
    recorder = SessionRecorder(cfg, EventBus(), FakeTranscript(), FakeAssistant())
    recorder.stop()  # must not raise, per the class's own "always safe to call" contract
    with pytest.raises(RuntimeError):
        _ = recorder.folder


def test_export_before_start_raises_like_folder_property() -> None:
    cfg = Config()
    recorder = SessionRecorder(cfg, EventBus(), FakeTranscript(), FakeAssistant())
    with pytest.raises(RuntimeError):
        recorder.export()


def test_stop_after_start_still_saves_and_can_be_called_again(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path)
    recorder = SessionRecorder(cfg, EventBus(), FakeTranscript(), FakeAssistant())
    folder = recorder.start()
    recorder.stop()
    assert (folder / "session.json").is_file()
    recorder.stop()  # calling stop() again must not raise
    recorder.stop()


# ---------------------------------------------------------------------------
# start() called again while already running: no leaked thread/subscription
# ---------------------------------------------------------------------------


def test_restarting_a_running_recorder_does_not_leak_thread_or_subscription(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path, autosave_interval_s=30)
    bus = EventBus()
    recorder = SessionRecorder(cfg, bus, FakeTranscript(), FakeAssistant())

    first_folder = recorder.start()
    assert len(bus._subs) == 1  # noqa: SLF001 - white-box leak check
    assert sum(1 for t in threading.enumerate() if t.name == "SessionRecorder") == 1

    second_folder = recorder.start()

    # the previous session must have been stopped, not merely abandoned
    assert len(bus._subs) == 1  # noqa: SLF001
    assert sum(1 for t in threading.enumerate() if t.name == "SessionRecorder") == 1
    assert (first_folder / "session.json").is_file(), "restarting must finalize the previous session"

    recorder.stop()
    assert len(bus._subs) == 0  # noqa: SLF001
    assert sum(1 for t in threading.enumerate() if t.name == "SessionRecorder") == 0
    assert second_folder != first_folder


def test_two_sessions_started_within_the_same_second_get_distinct_folders(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path)
    bus = EventBus()
    rec1 = SessionRecorder(cfg, bus, FakeTranscript(), FakeAssistant())
    rec2 = SessionRecorder(cfg, bus, FakeTranscript(), FakeAssistant())

    f1 = rec1.start()
    f2 = rec2.start()  # may land in the same wall-clock second as f1 on a fast machine

    assert f1 != f2, "two independent sessions must never share a folder"
    rec1.stop()
    rec2.stop()


# ---------------------------------------------------------------------------
# concurrent export()/autosave + speaker renames: no crash, data lands correctly
# ---------------------------------------------------------------------------


def test_concurrent_renames_and_saves_do_not_raise(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path, autosave_interval_s=1)
    bus = EventBus()
    recorder = SessionRecorder(cfg, bus, FakeTranscript(), FakeAssistant())
    recorder.start()

    errors: list[BaseException] = []
    n_renames = 2000

    def rename_loop() -> None:
        for i in range(n_renames):
            bus.publish(SpeakerRenamedEvent(speaker_id=f"spk{i % 8}", new_label=f"Person {i}"))

    def export_loop() -> None:
        for _ in range(50):
            try:
                recorder.export()
            except BaseException as exc:  # noqa: BLE001 - want to catch and report anything
                errors.append(exc)

    renamer = threading.Thread(target=rename_loop)
    renamer.start()
    export_loop()
    renamer.join(timeout=10.0)
    assert not renamer.is_alive()
    recorder.export()  # pick up anything published after the last export() in the loop above
    recorder.stop()

    assert errors == []
    data = json.loads((recorder.folder / "session.json").read_text(encoding="utf-8"))
    assert isinstance(data["speakers"], dict)
    assert data["speakers"], "at least one rename should have been observed"


# ---------------------------------------------------------------------------
# upstream finding 5 (ported from mac-cortex): atomic writes
# ---------------------------------------------------------------------------


def test_atomic_write_text_replaces_file_content(tmp_path: Path) -> None:
    path = tmp_path / "transcript.md"
    path.write_text("old content", encoding="utf-8")
    _atomic_write_text(path, "new content")
    assert path.read_text(encoding="utf-8") == "new content"
    # no leftover temp file
    assert not (tmp_path / "transcript.md.tmp").exists()


def test_atomic_write_text_never_leaves_a_partial_file_on_first_write(tmp_path: Path) -> None:
    path = tmp_path / "brand_new.md"
    _atomic_write_text(path, "hello")
    assert path.read_text(encoding="utf-8") == "hello"


def test_atomic_write_text_crash_between_write_and_replace_keeps_old_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Simulate the process dying after the temp file is fully written but before the rename:
    `os.replace` is monkeypatched to raise, exactly the window upstream finding 5 flags. The
    previous good file must be left completely untouched -- never truncated, never half-new."""
    import exocortex.session as session_mod

    path = tmp_path / "session.json"
    path.write_text('{"good": true}', encoding="utf-8")

    def failing_replace(*args: object, **kwargs: object) -> None:
        raise OSError("simulated crash between write and replace")

    monkeypatch.setattr(session_mod.os, "replace", failing_replace)

    with pytest.raises(OSError):
        _atomic_write_text(path, '{"corrupt": "should never land"}')

    # the previous good file is untouched
    assert path.read_text(encoding="utf-8") == '{"good": true}'
    # the fully-written temp file is left on disk (exactly what a real crash would also leave)
    assert (tmp_path / "session.json.tmp").read_text(encoding="utf-8") == '{"corrupt": "should never land"}'


def test_save_writes_every_file_atomically(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SessionRecorder._save() must go through `_atomic_write_text` for every file it writes,
    not a direct `Path.write_text` (regression guard: it is easy to reintroduce the non-atomic
    write by "simplifying" `_save()` back to the old shape)."""
    import exocortex.session as session_mod

    calls: list[Path] = []
    original = session_mod._atomic_write_text

    def spy(path: Path, text: str) -> None:
        calls.append(path)
        original(path, text)

    monkeypatch.setattr(session_mod, "_atomic_write_text", spy)

    cfg = _make_cfg(tmp_path)
    recorder = SessionRecorder(cfg, EventBus(), FakeTranscript(), FakeAssistant())
    recorder.start()
    recorder.stop()

    names = {p.name for p in calls}
    assert names == {"transcript.md", "transcript.jsonl", "notes.md", "answers.md", "session.json"}


# ---------------------------------------------------------------------------
# upstream finding 2 (ported from mac-cortex): the autosave loop survives a save failure
# ---------------------------------------------------------------------------


def test_autosave_survives_a_save_exception_and_keeps_ticking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One bad autosave tick (a transient write failure) must not kill the daemon thread: the
    next tick, after the transient condition clears, must still save normally."""
    cfg = _make_cfg(tmp_path, autosave_interval_s=1)
    bus = EventBus()
    recorder = SessionRecorder(cfg, bus, FakeTranscript(), FakeAssistant())

    calls = {"n": 0}
    original_save = recorder._save

    def flaky_save(*, kind: str) -> None:
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("simulated transient write failure")
        original_save(kind=kind)

    monkeypatch.setattr(recorder, "_save", flaky_save)

    folder = recorder.start()
    try:
        bus.publish(SpeakerRenamedEvent(speaker_id="spk1", new_label="Alice"))

        # upstream finding 2 (Windows fix): a generous deadline, not a tight one. The daemon
        # thread's tick interval floors at 1s (session.py's `interval = max(1, ...)`, so it
        # cannot be shortened from here), and the worst case legitimately needs 3 ticks -- tick 1
        # can race this thread's bus.publish() above and see nothing dirty yet, tick 2 hits the
        # injected failure, tick 3 must succeed -- before this loop's condition can be true. Under
        # concurrent load (e.g. another agent's test suite running at the same time on this
        # machine), Event.wait(1) wakeup jitter can push those 3 ticks past a tight deadline with
        # no bug present, so the margin here is generous rather than tuned to the happy path.
        session_path = folder / "session.json"
        deadline = time.monotonic() + 20.0
        saved = False
        while time.monotonic() < deadline:
            if session_path.is_file() and '"Alice"' in session_path.read_text(encoding="utf-8"):
                saved = True
                break
            time.sleep(0.1)

        assert saved, "autosave must recover and save on a later tick after one failing tick"
        assert calls["n"] >= 2, "the flaky save must have been retried, not just called once"
        # the thread must still be alive and stoppable normally (not dead from the exception)
        assert sum(1 for t in threading.enumerate() if t.name == "SessionRecorder") == 1
    finally:
        # upstream finding 2 (Windows fix): always stop the recorder, even when an assertion
        # above fails. Without this, a failed assertion returns via exception before stop() ever
        # runs, and the daemon autosave thread -- writing with the real _save once monkeypatch's
        # teardown reverts the patch -- keeps ticking and writing files for the rest of the
        # pytest process: a thread/resource leak on the failure path of a test whose purpose is
        # verifying clean thread lifecycle behavior.
        recorder.stop()


# ---------------------------------------------------------------------------
# models_dl.py: mkdir failure is wrapped, not a raw OSError
# ---------------------------------------------------------------------------


def test_ensure_models_wraps_mkdir_failure_as_model_download_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover - never reached
        raise AssertionError("must not attempt a download when the destination dir can't be created")

    monkeypatch.setattr(
        models_dl, "_make_client", lambda: httpx.Client(transport=httpx.MockTransport(handler))
    )

    def failing_mkdir(self: Path, *args: object, **kwargs: object) -> None:
        raise OSError("simulated permission error")

    monkeypatch.setattr(Path, "mkdir", failing_mkdir)

    with pytest.raises(ModelDownloadError):
        ensure_models(tmp_path, names=["silero_vad.onnx"])


def test_ensure_models_partial_failure_leaves_earlier_downloads_in_place(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the second of two models fails, the first must already be on disk (not rolled back)."""

    def handler(request: httpx.Request) -> httpx.Response:
        if "silero_vad" in str(request.url):
            return httpx.Response(200, content=b"x" * 200_000)
        return httpx.Response(404, content=b"not found")

    monkeypatch.setattr(
        models_dl, "_make_client", lambda: httpx.Client(transport=httpx.MockTransport(handler))
    )

    with pytest.raises(ModelDownloadError):
        ensure_models(tmp_path, names=["silero_vad.onnx", "wespeaker_en_voxceleb_CAM++.onnx"])

    assert (tmp_path / "silero_vad.onnx").is_file()
    assert not (tmp_path / "wespeaker_en_voxceleb_CAM++.onnx").exists()


def test_model_urls_path_segments_are_not_mangled_by_httpx() -> None:
    """The literal '++' in the wespeaker URL must survive unencoded (verifies the spec_issue
    raised by the implementer is a non-issue: httpx does not treat '+' specially in a path)."""
    url = httpx.URL(models_dl.MODEL_URLS["wespeaker_en_voxceleb_CAM++.onnx"])
    assert url.path.endswith("wespeaker_en_voxceleb_CAM++.onnx")


# ---------------------------------------------------------------------------
# logging_setup.py: no dedicated test file existed before this pass
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _restore_root_logger():
    """setup_logging() mutates the global root logger; restore it after each test."""
    root = logging.getLogger()
    original_handlers = list(root.handlers)
    original_level = root.level
    yield
    for h in list(root.handlers):
        if h not in original_handlers:
            root.removeHandler(h)
            h.close()
    root.setLevel(original_level)


def test_setup_logging_creates_log_file_under_given_dir(tmp_path: Path) -> None:
    log_path = setup_logging("DEBUG", log_dir=tmp_path)
    assert log_path == tmp_path / "exocortex.log"
    logging.getLogger("exocortex.session").info("hello")
    for h in logging.getLogger().handlers:
        h.flush()
    assert log_path.is_file()
    assert "hello" in log_path.read_text(encoding="utf-8")


def test_setup_logging_is_idempotent_no_duplicate_handlers(tmp_path: Path) -> None:
    setup_logging("INFO", log_dir=tmp_path)
    first_count = len(logging.getLogger().handlers)
    setup_logging("INFO", log_dir=tmp_path)
    second_count = len(logging.getLogger().handlers)
    setup_logging("DEBUG", log_dir=tmp_path)
    third_count = len(logging.getLogger().handlers)

    assert first_count == second_count == third_count


def test_setup_logging_quiets_noisy_third_party_loggers(tmp_path: Path) -> None:
    setup_logging("DEBUG", log_dir=tmp_path)
    for name in ("httpx", "httpcore", "faster_whisper", "huggingface_hub"):
        assert logging.getLogger(name).level == logging.WARNING


def test_setup_logging_returns_none_when_log_dir_unusable(tmp_path: Path) -> None:
    blocked = tmp_path / "not_a_dir"
    blocked.write_text("x", encoding="utf-8")  # a file, so mkdir(parents=True) under it fails

    log_path = setup_logging("INFO", log_dir=blocked / "logs")

    assert log_path is None
    # console logging must still be configured
    assert any(isinstance(h, logging.StreamHandler) for h in logging.getLogger().handlers)


def test_setup_logging_sets_root_level(tmp_path: Path) -> None:
    setup_logging("WARNING", log_dir=tmp_path)
    assert logging.getLogger().level == logging.WARNING
    setup_logging("DEBUG", log_dir=tmp_path)
    assert logging.getLogger().level == logging.DEBUG

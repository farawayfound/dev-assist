"""Engine's context-library surface (2026-10-06): the seven EngineActions context methods the
Models window's "Context" tab calls, driven against a real ContextStore in a tmp directory and a
fake Assistant that records `digest_document` calls. Nothing here touches the network, the GPU or
disk outside tmp_path."""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import pytest

from exocortex import app as app_mod
from exocortex.bus import (
    ComponentStatusEvent,
    ContextLibraryEvent,
    EventBus,
    drain,
)
from exocortex.config import Config
from exocortex.context import ContextManager, ContextStore


# ---- fakes ----------------------------------------------------------------------------------
class _Assistant:
    """Just the surface the engine touches here: `digest_document` (queued, one id per call),
    `reset` and `cancel`."""

    def __init__(self) -> None:
        self.digested: list[str] = []
        self.resets = 0

    def digest_document(self, doc_id: str) -> None:
        self.digested.append(doc_id)

    def reset(self) -> None:
        self.resets += 1

    def cancel(self) -> None:
        pass


class _NoDigestAssistant:
    """An assistant without `digest_document` (the engine must call it only when present)."""

    def reset(self) -> None:
        pass


class _ExplodingAssistant(_Assistant):
    def digest_document(self, doc_id: str) -> None:
        raise RuntimeError("boom")


def _events(q: Any) -> list[Any]:
    return drain(q)


def _library_events(events: list[Any]) -> list[ContextLibraryEvent]:
    return [e for e in events if isinstance(e, ContextLibraryEvent)]


def _app_errors(events: list[Any]) -> list[ComponentStatusEvent]:
    return [e for e in events if isinstance(e, ComponentStatusEvent) and e.component == "app" and e.state == "error"]


@pytest.fixture()
def ctx_dir(tmp_path: Path) -> Path:
    return tmp_path / "ctx"


@pytest.fixture()
def eng(tmp_path: Path, ctx_dir: Path) -> app_mod.Engine:
    cfg = Config()
    cfg.path = tmp_path / "config.toml"
    cfg.paths.models_dir = str(tmp_path / "models")
    cfg.context.dir = str(ctx_dir)
    e = app_mod.Engine(cfg, EventBus())
    e.assistant = _Assistant()
    return e


@pytest.fixture()
def q(eng: app_mod.Engine) -> Any:
    return eng.bus.subscribe()


def _manifest(ctx_dir: Path) -> dict[str, Any]:
    return json.loads((ctx_dir / "manifest.json").read_text(encoding="utf-8"))


# ---- construction ---------------------------------------------------------------------------
def test_engine_builds_the_store_and_manager_without_touching_disk(eng: app_mod.Engine, ctx_dir: Path) -> None:
    assert isinstance(eng.context_store, ContextStore)
    assert isinstance(eng.context, ContextManager)
    assert eng.context.store is eng.context_store
    assert not ctx_dir.exists()          # created on the first write, never at construction


def test_default_context_dir_is_under_models_dir(tmp_path: Path) -> None:
    cfg = Config()
    cfg.paths.models_dir = str(tmp_path / "models")
    e = app_mod.Engine(cfg, EventBus())
    assert cfg.context_dir() == tmp_path / "models" / "context"
    e.add_context_text("Notes", "alpha beta gamma")
    assert (tmp_path / "models" / "context" / "manifest.json").is_file()


def test_start_hands_the_context_manager_to_the_assistant(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import exocortex.llm.assistant as assistant_mod
    import exocortex.session as session_mod
    from exocortex.llm.route import Route

    made: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    class _FakeAssistant:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            made.append((args, kwargs))

        def stop(self) -> None:
            pass

    class _FakeRecorder:
        folder = tmp_path

        def __init__(self, *a: Any, **k: Any) -> None:
            pass

        def start(self) -> Path:
            return tmp_path

        def stop(self) -> None:
            pass

    monkeypatch.setattr(assistant_mod, "Assistant", _FakeAssistant)
    monkeypatch.setattr(session_mod, "SessionRecorder", _FakeRecorder)
    monkeypatch.setattr(app_mod, "make_client", lambda *a, **k: object())
    for name in ("_init_llm", "_init_speech", "_init_screen", "_start_gpu_guard", "_preload_native"):
        monkeypatch.setattr(app_mod.Engine, name, lambda self, *a, **k: None)
    monkeypatch.setattr(app_mod.cuda_env, "setup_cuda_dlls", lambda: None)

    cfg = Config()
    cfg.path = tmp_path / "config.toml"
    cfg.paths.models_dir = str(tmp_path / "models")
    cfg.completion.enabled = False
    cfg.handsfree.enabled = False
    route = Route(backend="openai", base_url="http://127.0.0.1:9", reason="test")
    e = app_mod.Engine(cfg, EventBus(), route_resolver=lambda *a, **k: route)
    e.start()
    assert len(made) == 1
    args, kwargs = made[0]
    assert kwargs.get("context") is e.context
    assert args[:4] == (cfg, e.bus, e.client, e.transcript)


# ---- add_context_text -----------------------------------------------------------------------
def test_add_text_publishes_event_writes_manifest_and_queues_digest(eng: app_mod.Engine, q: Any, ctx_dir: Path) -> None:
    eng.add_context_text("Design notes", "# Heading\n\nThe cache is warmed by the anchored window.")
    events = _events(q)
    lib = _library_events(events)
    assert [e.reason for e in lib] == ["added"]
    docs = eng.context_store.documents()
    assert len(docs) == 1 and lib[0].doc_id == docs[0].id
    assert docs[0].name == "Design notes" and docs[0].mode == "reference"      # [context].default_mode
    assert docs[0].digest_state == "pending"
    assert eng.assistant.digested == [docs[0].id]
    on_disk = _manifest(ctx_dir)["documents"]
    assert [d["id"] for d in on_disk] == [docs[0].id]
    assert Path(on_disk[0]["path"]).is_file()
    assert not _app_errors(events)


def test_add_text_inject_mode_queues_no_digest(eng: app_mod.Engine, q: Any) -> None:
    eng.add_context_text("Glossary", "TTFT means time to first token.", mode="inject")
    doc = eng.context_store.documents()[0]
    assert doc.mode == "inject" and doc.digest_state == "n/a"
    assert eng.assistant.digested == []
    assert [e.reason for e in _library_events(_events(q))] == ["added"]


def test_add_text_respects_default_mode_from_config(eng: app_mod.Engine) -> None:
    eng.cfg.context.default_mode = "inject"
    eng.add_context_text("A", "some text to inject")
    assert eng.context_store.documents()[0].mode == "inject"
    assert eng.assistant.digested == []


def test_add_text_with_digest_off_leaves_the_state_alone(eng: app_mod.Engine) -> None:
    eng.cfg.context.digest = False
    eng.add_context_text("A", "some reference text")
    doc = eng.context_store.documents()[0]
    assert doc.mode == "reference" and doc.digest_state == "n/a"
    assert eng.assistant.digested == []


def test_add_text_blank_name_falls_back_to_untitled(eng: app_mod.Engine) -> None:
    eng.add_context_text("   ", "body text here")
    assert eng.context_store.documents()[0].name == "Untitled"


def test_add_empty_text_publishes_an_error_and_never_raises(eng: app_mod.Engine, q: Any) -> None:
    eng.add_context_text("Empty", "   \n  ")
    events = _events(q)
    lib = _library_events(events)
    assert len(lib) == 1 and lib[0].reason.startswith("error:")
    assert _app_errors(events) and "Empty" in _app_errors(events)[0].detail
    assert eng.context_store.documents() == []


def test_add_text_with_unknown_mode_is_an_error_event(eng: app_mod.Engine, q: Any) -> None:
    eng.add_context_text("A", "text", mode="telepathy")
    lib = _library_events(_events(q))
    assert len(lib) == 1 and lib[0].reason.startswith("error:")
    assert eng.context_store.documents() == []


def test_same_text_added_twice_is_one_document(eng: app_mod.Engine) -> None:
    eng.add_context_text("A", "identical body text")
    eng.add_context_text("A again", "identical body text")
    assert len(eng.context_store.documents()) == 1


# ---- add_context_document -------------------------------------------------------------------
def test_add_file_publishes_event_and_queues_digest(eng: app_mod.Engine, q: Any, tmp_path: Path) -> None:
    f = tmp_path / "README.md"
    f.write_text("# Project\n\nIt does things.\n", encoding="utf-8")
    eng.add_context_document(str(f))
    lib = _library_events(_events(q))
    assert [e.reason for e in lib] == ["added"]
    doc = eng.context_store.get(lib[0].doc_id)
    assert doc is not None and doc.name == "README.md" and doc.kind == "markdown"
    assert doc.source_path == str(f)
    assert eng.assistant.digested == [doc.id]


def test_add_file_accepts_a_quoted_path_and_a_mode(eng: app_mod.Engine, tmp_path: Path) -> None:
    f = tmp_path / "notes.txt"
    f.write_text("plain notes for the talk", encoding="utf-8")
    eng.add_context_document(f'"{f}"', mode="inject")
    doc = eng.context_store.documents()[0]
    assert doc.mode == "inject" and doc.name == "notes.txt"
    assert eng.assistant.digested == []


def test_add_missing_file_publishes_an_error_and_never_raises(eng: app_mod.Engine, q: Any, tmp_path: Path) -> None:
    eng.add_context_document(str(tmp_path / "nope.md"))
    events = _events(q)
    lib = _library_events(events)
    assert len(lib) == 1 and lib[0].reason.startswith("error:") and lib[0].doc_id == ""
    errors = _app_errors(events)
    assert errors and "nope.md" in errors[0].detail
    assert eng.context_store.documents() == []
    assert eng.assistant.digested == []


def test_add_unparseable_file_publishes_an_error(eng: app_mod.Engine, q: Any, tmp_path: Path) -> None:
    bad = tmp_path / "broken.docx"
    bad.write_bytes(b"this is not a zip archive at all")
    eng.add_context_document(str(bad))
    events = _events(q)
    assert [e.reason.startswith("error:") for e in _library_events(events)] == [True]
    assert _app_errors(events)
    assert eng.context_store.documents() == []


def test_add_a_directory_publishes_an_error(eng: app_mod.Engine, q: Any, tmp_path: Path) -> None:
    eng.add_context_document(str(tmp_path))
    assert _library_events(_events(q))[0].reason.startswith("error:")


def test_add_binary_blob_publishes_an_error(eng: app_mod.Engine, q: Any, tmp_path: Path) -> None:
    blob = tmp_path / "data.bin"
    blob.write_bytes(bytes(range(256)) * 40)
    eng.add_context_document(str(blob))
    assert _library_events(_events(q))[0].reason.startswith("error:")
    assert eng.context_store.documents() == []


# ---- the assistant is optional --------------------------------------------------------------
def test_add_before_the_assistant_exists_leaves_the_digest_pending(eng: app_mod.Engine, q: Any) -> None:
    eng.assistant = None
    eng.add_context_text("A", "reference body")
    assert eng.context_store.documents()[0].digest_state == "pending"
    assert [e.reason for e in _library_events(_events(q))] == ["added"]


def test_an_assistant_without_digest_document_is_tolerated(eng: app_mod.Engine) -> None:
    eng.assistant = _NoDigestAssistant()
    eng.add_context_text("A", "reference body")
    assert eng.context_store.documents()[0].digest_state == "pending"


def test_a_failing_digest_queue_never_raises(eng: app_mod.Engine, q: Any) -> None:
    eng.assistant = _ExplodingAssistant()
    eng.add_context_text("A", "reference body")
    assert [e.reason for e in _library_events(_events(q))] == ["added"]
    assert not _app_errors(_events(q))


# ---- context_documents ----------------------------------------------------------------------
def test_context_documents_rows_carry_the_contract_keys_and_a_token_estimate(eng: app_mod.Engine) -> None:
    eng.add_context_text("A", "x" * 400)
    rows = eng.context_documents()
    assert len(rows) == 1
    row = rows[0]
    for key in ("id", "name", "kind", "chars", "tokens", "enabled", "mode", "digest", "digest_state",
                "source_path", "added_at"):
        assert key in row
    assert row["chars"] == 400
    assert row["tokens"] == eng.context.estimator.tokens_for_len(400) > 0
    assert row["enabled"] is True and row["digest_state"] == "pending"


def test_context_documents_empty_library(eng: app_mod.Engine) -> None:
    assert eng.context_documents() == []


def test_context_documents_never_raises(eng: app_mod.Engine, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(eng.context_store, "documents", lambda: (_ for _ in ()).throw(RuntimeError("x")))
    assert eng.context_documents() == []


def test_documents_survive_an_engine_restart(tmp_path: Path, eng: app_mod.Engine, ctx_dir: Path) -> None:
    eng.add_context_text("Kept", "text that must persist")
    cfg = Config()
    cfg.context.dir = str(ctx_dir)
    cfg.paths.models_dir = str(tmp_path / "models")
    again = app_mod.Engine(cfg, EventBus())
    assert [r["name"] for r in again.context_documents()] == ["Kept"]


# ---- remove ---------------------------------------------------------------------------------
def test_remove_deletes_the_document_and_its_stored_copy(eng: app_mod.Engine, q: Any, ctx_dir: Path) -> None:
    eng.add_context_text("A", "to be removed")
    doc = eng.context_store.documents()[0]
    stored = Path(doc.path)
    assert stored.is_file()
    _events(q)
    eng.remove_context_document(doc.id)
    lib = _library_events(_events(q))
    assert [(e.reason, e.doc_id) for e in lib] == [("removed", doc.id)]
    assert eng.context_store.documents() == []
    assert not stored.exists()
    assert _manifest(ctx_dir)["documents"] == []


def test_remove_unknown_id_is_an_error_event(eng: app_mod.Engine, q: Any) -> None:
    eng.remove_context_document("deadbeef0000")
    events = _events(q)
    assert _library_events(events)[0].reason.startswith("error:")
    assert _app_errors(events)


# ---- set_context_document -------------------------------------------------------------------
def test_set_enabled_toggles_and_publishes(eng: app_mod.Engine, q: Any, ctx_dir: Path) -> None:
    eng.add_context_text("A", "some text")
    doc = eng.context_store.documents()[0]
    _events(q)
    eng.set_context_document(doc.id, enabled=False)
    lib = _library_events(_events(q))
    assert [(e.reason, e.doc_id) for e in lib] == [("updated", doc.id)]
    assert eng.context_store.get(doc.id).enabled is False
    assert _manifest(ctx_dir)["documents"][0]["enabled"] is False
    assert eng.context_store.enabled() == []
    eng.set_context_document(doc.id, enabled=True)
    assert eng.context_store.get(doc.id).enabled is True


def test_set_none_attributes_changes_nothing(eng: app_mod.Engine) -> None:
    eng.add_context_text("A", "some text", mode="inject")
    doc = eng.context_store.documents()[0]
    eng.set_context_document(doc.id)
    after = eng.context_store.get(doc.id)
    assert (after.enabled, after.mode) == (True, "inject")


def test_switch_to_reference_without_a_digest_stays_pending_and_queues_it(eng: app_mod.Engine, q: Any) -> None:
    eng.add_context_text("A", "some text", mode="inject")
    doc = eng.context_store.documents()[0]
    assert eng.assistant.digested == []
    _events(q)
    eng.set_context_document(doc.id, mode="reference")
    after = eng.context_store.get(doc.id)
    assert after.mode == "reference" and after.digest_state == "pending"
    assert eng.assistant.digested == [doc.id]
    assert [e.reason for e in _library_events(_events(q))] == ["updated"]


def test_switch_to_reference_with_digest_off_queues_nothing(eng: app_mod.Engine) -> None:
    eng.cfg.context.digest = False
    eng.add_context_text("A", "some text", mode="inject")
    doc = eng.context_store.documents()[0]
    eng.set_context_document(doc.id, mode="reference")
    assert eng.context_store.get(doc.id).digest_state == "n/a"
    assert eng.assistant.digested == []


def test_switch_back_to_reference_keeps_a_ready_digest(eng: app_mod.Engine) -> None:
    eng.add_context_text("A", "some text")
    doc = eng.context_store.documents()[0]
    eng.context_store.set_digest(doc.id, "A short digest.", "ready")
    eng.assistant.digested.clear()
    eng.set_context_document(doc.id, mode="inject")
    assert eng.context_store.get(doc.id).digest_state == "n/a"
    eng.set_context_document(doc.id, mode="reference")
    after = eng.context_store.get(doc.id)
    assert after.digest_state == "ready" and after.digest == "A short digest."
    assert eng.assistant.digested == []


def test_set_enabled_and_mode_together(eng: app_mod.Engine) -> None:
    eng.add_context_text("A", "some text")
    doc = eng.context_store.documents()[0]
    eng.set_context_document(doc.id, enabled=False, mode="inject")
    after = eng.context_store.get(doc.id)
    assert (after.enabled, after.mode) == (False, "inject")


def test_set_unknown_mode_is_an_error_and_changes_nothing(eng: app_mod.Engine, q: Any) -> None:
    eng.add_context_text("A", "some text")
    doc = eng.context_store.documents()[0]
    _events(q)
    eng.set_context_document(doc.id, enabled=False, mode="bogus")
    events = _events(q)
    assert _library_events(events)[0].reason.startswith("error:")
    assert eng.context_store.get(doc.id).enabled is True      # validated before anything was applied


def test_set_unknown_document_is_an_error_event(eng: app_mod.Engine, q: Any) -> None:
    eng.set_context_document("nosuchdoc000", enabled=False)
    events = _events(q)
    assert _library_events(events)[0].reason.startswith("error:")
    assert _app_errors(events)


# ---- refresh_context_digest -----------------------------------------------------------------
def test_refresh_marks_pending_and_queues_the_digest(eng: app_mod.Engine, q: Any) -> None:
    eng.add_context_text("A", "some text")
    doc = eng.context_store.documents()[0]
    eng.context_store.set_digest(doc.id, "old digest", "ready")
    eng.assistant.digested.clear()
    _events(q)
    eng.refresh_context_digest(doc.id)
    assert eng.context_store.get(doc.id).digest_state == "pending"
    assert eng.assistant.digested == [doc.id]
    lib = _library_events(_events(q))
    assert [(e.reason, e.doc_id) for e in lib] == [("digest-queued", doc.id)]


def test_refresh_on_an_injected_document_is_skipped_not_an_error(eng: app_mod.Engine, q: Any) -> None:
    eng.add_context_text("A", "some text", mode="inject")
    doc = eng.context_store.documents()[0]
    _events(q)
    eng.refresh_context_digest(doc.id)
    events = _events(q)
    assert [e.reason for e in _library_events(events)] == ["digest-skipped"]
    assert not _app_errors(events)
    assert eng.assistant.digested == []
    assert eng.context_store.get(doc.id).digest_state == "n/a"


def test_refresh_unknown_document_is_an_error_event(eng: app_mod.Engine, q: Any) -> None:
    eng.refresh_context_digest("nosuchdoc000")
    events = _events(q)
    assert _library_events(events)[0].reason.startswith("error:")
    assert eng.assistant.digested == []


def test_refresh_works_with_digest_config_off(eng: app_mod.Engine) -> None:
    """An explicit Re-digest click is honoured even when [context].digest is off."""
    eng.cfg.context.digest = False
    eng.add_context_text("A", "some text")
    doc = eng.context_store.documents()[0]
    eng.refresh_context_digest(doc.id)
    assert eng.assistant.digested == [doc.id]


# ---- context_usage --------------------------------------------------------------------------
def test_context_usage_shape_before_any_job(eng: app_mod.Engine) -> None:
    u = eng.context_usage()
    for key in ("slot_tokens", "chars_per_token", "earlier_summary", "documents", "has_usage",
                "budget_tokens", "used_tokens", "transcript_tokens", "reference_tokens", "lookup_tokens",
                "earlier_tokens", "screen_tokens", "prompt_tokens", "cached_tokens", "trimmed"):
        assert key in u
    assert u["has_usage"] is False and u["documents"] == 0 and u["earlier_summary"] == ""
    assert u["slot_tokens"] == eng.context.slot_tokens


def test_context_usage_counts_enabled_documents_and_the_latest_event(eng: app_mod.Engine) -> None:
    from exocortex.bus import ContextUsageEvent

    eng.add_context_text("A", "alpha text")
    eng.add_context_text("B", "beta text")
    b = [d for d in eng.context_store.documents() if d.name == "B"][0]
    eng.set_context_document(b.id, enabled=False)
    eng.context.record_usage(ContextUsageEvent(job_id=7, kind="assist", slot_tokens=4096, used_tokens=1234,
                                               prompt_tokens=1300, cached_tokens=1000))
    u = eng.context_usage()
    assert u["documents"] == 1
    assert u["has_usage"] is True and u["job_id"] == 7 and u["used_tokens"] == 1234
    assert u["prompt_tokens"] == 1300 and u["cached_tokens"] == 1000


def test_context_usage_never_raises(eng: app_mod.Engine, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(eng.context, "usage_snapshot", lambda: (_ for _ in ()).throw(RuntimeError("x")))
    assert eng.context_usage() == {}


# ---- new_session ----------------------------------------------------------------------------
def test_new_session_keeps_documents_and_resets_the_window(eng: app_mod.Engine) -> None:
    eng.add_context_text("A", "persistent reference text")
    assert eng.context.window is not None
    called: list[int] = []
    real_reset = eng.context.reset_session
    eng.context.reset_session = lambda: (called.append(1), real_reset())[1]    # type: ignore[method-assign]
    eng.context.window.set_summary(None, "an earlier summary")
    assert eng.context.window.earlier_summary(None) == "an earlier summary"
    eng.new_session()
    assert called == [1]
    assert eng.context.window.earlier_summary(None) == ""
    assert [d.name for d in eng.context_store.documents()] == ["A"]
    assert eng.assistant.resets == 1


def test_new_session_resets_the_window_even_with_no_assistant(eng: app_mod.Engine) -> None:
    eng.assistant = None
    eng.context.window.set_summary(None, "stale")
    eng.new_session()
    assert eng.context.window.earlier_summary(None) == ""


def test_new_session_survives_a_failing_window_reset(eng: app_mod.Engine, q: Any) -> None:
    def boom() -> None:
        raise RuntimeError("x")

    eng.context.reset_session = boom       # type: ignore[method-assign]
    eng.new_session()
    assert any(isinstance(e, ComponentStatusEvent) and e.component == "session" for e in _events(q))


# ---- apply_settings -------------------------------------------------------------------------
def test_apply_settings_context_keys_need_nothing_and_do_not_raise(eng: app_mod.Engine) -> None:
    eng.cfg.context.lookup_tokens = 300
    eng.apply_settings({"context.lookup_tokens": 300, "context.budget_mode": "fixed"})
    assert eng.cfg.path.is_file()          # saved like every other settings change


def test_apply_settings_context_dir_logs_that_it_takes_effect_next_launch(
    eng: app_mod.Engine, caplog: pytest.LogCaptureFixture, tmp_path: Path,
) -> None:
    store_before = eng.context_store
    with caplog.at_level("INFO", logger="exocortex.app"):
        eng.apply_settings({"context.dir": str(tmp_path / "elsewhere")})
    assert any("next launch" in r.getMessage() for r in caplog.records)
    assert eng.context_store is store_before        # the live library does not move mid-session


# ---- concurrency ----------------------------------------------------------------------------
def test_adds_from_several_threads_all_land(eng: app_mod.Engine) -> None:
    def add(i: int) -> None:
        eng.add_context_text(f"doc {i}", f"body number {i} with some words")

    threads = [threading.Thread(target=add, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)
    assert len(eng.context_documents()) == 8
    assert sorted(eng.assistant.digested) == sorted(d.id for d in eng.context_store.documents())

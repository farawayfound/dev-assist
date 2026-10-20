"""Engine wiring tests: components are faked, the composition logic is real."""
from __future__ import annotations

import queue
import threading
import time
from pathlib import Path
from typing import Any

import httpx
import pytest

from exocortex import app as app_mod
from exocortex.bus import (
    AssistStartedEvent,
    CheckpointEvent,
    ComponentStatusEvent,
    EventBus,
    FinalTranscriptEvent,
    SpeakerRenamedEvent,
    Source,
    TranscriptSegment,
    drain,
)
from exocortex.config import Config


# ---- fakes -----------------------------------------------------------------------------
class FakeTranscriber:
    instances: list["FakeTranscriber"] = []

    def __init__(self, cfg: Any, on_status: Any = None, **kw: Any) -> None:
        self.cfg, self.on_status = cfg, on_status
        self.loaded = False
        FakeTranscriber.instances.append(self)

    def load(self) -> None:
        self.loaded = True
        if self.on_status:
            self.on_status("ready", "fake")

    device_used = "fake-cpu"
    compute_type_used = "int8"


class FakeClusterer:
    def __init__(self, path: str, **kw: Any) -> None:
        self.path = path
        self.reset_calls = 0
        self.labels_: dict[str, str] = {}

    def rename(self, sid: str, label: str) -> None:
        self.labels_[sid] = label

    def reset(self) -> None:
        self.reset_calls += 1


class FakePipeline:
    instances: list["FakePipeline"] = []

    def __init__(self, cfg: Any, bus: Any, transcriber: Any, clusterer: Any, clock: Any, next_id: Any, **kw: Any) -> None:
        self.cfg, self.bus, self.transcriber, self.clusterer, self.clock, self.next_id = cfg, bus, transcriber, clusterer, clock, next_id
        self.started = self.stopped = False
        self.muted = False
        FakePipeline.instances.append(self)

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def set_mic_muted(self, muted: bool) -> None:
        self.muted = muted

    @property
    def mic_muted(self) -> bool:
        return self.muted

    def rename_speaker(self, sid: str, label: str) -> None:
        self.clusterer.rename(sid, label)
        self.bus.publish(SpeakerRenamedEvent(sid, label))

    def emit_final(self, text: str, source: Source = Source.SYSTEM) -> TranscriptSegment:
        seg = TranscriptSegment(self.next_id(), source, "spk1" if source is Source.SYSTEM else "me",
                                "Speaker 1" if source is Source.SYSTEM else "Me", text, self.clock(), self.clock() + 1, 0.9)
        self.bus.publish(FinalTranscriptEvent(seg))
        return seg


class FakeScreenReader:
    instances: list["FakeScreenReader"] = []

    def __init__(self, cfg: Any, bus: Any, clock: Any, **kw: Any) -> None:
        self.cfg = cfg
        self.calls: list[tuple[str, Any]] = []
        self.paused_ = False
        self.latest_text = "screen text"
        FakeScreenReader.instances.append(self)

    def start(self) -> None:
        self.calls.append(("start", None))

    def stop(self) -> None:
        self.calls.append(("stop", None))

    def set_mode(self, mode: str) -> None:
        self.calls.append(("mode", mode))

    def set_region(self, x: int, y: int, w: int, h: int) -> None:
        self.calls.append(("region", (x, y, w, h)))

    def capture_now(self) -> None:
        self.calls.append(("capture", None))

    def set_snapshot_dir(self, p: Any) -> None:
        self.calls.append(("snapshot_dir", p))

    @property
    def paused(self) -> bool:
        return self.paused_

    def pause(self) -> None:
        self.paused_ = True

    def resume(self) -> None:
        self.paused_ = False


def _wait(pred: Any, timeout: float = 8.0) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.02)
    return False


@pytest.fixture()
def engine_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Config, EventBus]:
    cfg = Config()
    cfg.path = tmp_path / "config.toml"
    cfg.paths.sessions_dir = str(tmp_path / "sessions")
    cfg.paths.models_dir = str(tmp_path / "models")
    cfg.llm.backend = "openai"
    cfg.llm.base_url = "http://127.0.0.1:9"        # nothing listens -> llm error status, no hang
    cfg.llm.request_timeout_s = 2.0
    cfg.notes.auto_interval_s = 0
    cfg.screen.mode = "off"
    (tmp_path / "models").mkdir()
    for name in ("silero_vad.onnx", cfg.diarization.model):
        (tmp_path / "models" / name).write_bytes(b"x" * 200_000)   # ensure_models sees them as present

    import exocortex.screen.reader as reader_mod
    import exocortex.speech.asr as asr_mod
    import exocortex.speech.diarize as diar_mod
    import exocortex.speech.pipeline as pipe_mod

    monkeypatch.setattr(asr_mod, "Transcriber", FakeTranscriber)
    monkeypatch.setattr(diar_mod, "SpeakerClusterer", FakeClusterer)
    monkeypatch.setattr(pipe_mod, "SpeechPipeline", FakePipeline)
    monkeypatch.setattr(reader_mod, "ScreenReader", FakeScreenReader)

    # LLM client: mock transport so health checks fail fast and deterministically
    import exocortex.llm.client as client_mod

    real_client = client_mod.LMStudioClient

    def _factory(llm_cfg: Any, **kw: Any) -> Any:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused", request=request)

        return real_client(llm_cfg, transport=httpx.MockTransport(handler))

    monkeypatch.setattr(client_mod, "LMStudioClient", _factory)
    FakePipeline.instances.clear()
    FakeTranscriber.instances.clear()
    FakeScreenReader.instances.clear()
    return cfg, EventBus()


def test_engine_starts_wires_and_stops(engine_env: tuple[Config, EventBus], tmp_path: Path) -> None:
    cfg, bus = engine_env
    q = bus.subscribe()
    eng = app_mod.Engine(cfg, bus)
    eng.start()
    try:
        assert _wait(lambda: FakePipeline.instances and FakePipeline.instances[0].started)
        pipe = FakePipeline.instances[0]
        assert FakeTranscriber.instances[0].loaded
        assert eng.transcriber is not None and eng.clusterer is not None
        assert FakeScreenReader.instances and ("start", None) in FakeScreenReader.instances[0].calls

        # finals published by the pipeline land in the transcript BEFORE fan-out
        seg = pipe.emit_final("What is the plan for Q4?")
        assert eng.transcript.segments()[-1].text == seg.text
        # rename flows through the recording bus into the transcript
        eng.rename_speaker("spk1", "Alice")
        assert eng.transcript.segments()[-1].speaker_label == "Alice"
        # mute routes to the pipeline and into the config
        eng.set_mic_muted(True)
        assert pipe.muted is True and cfg.audio.mic_start_muted is True
        # screen actions route to the reader
        eng.set_screen_mode("auto")
        eng.set_screen_region(10, 20, 300, 200)
        eng.capture_screen_now()
        assert eng.toggle_screen_paused() is True and eng.toggle_screen_paused() is False
        calls = FakeScreenReader.instances[0].calls
        assert ("mode", "auto") in calls and ("region", (10, 20, 300, 200)) in calls and ("capture", None) in calls
        assert cfg.screen.mode == "region" and cfg.screen.region == [10, 20, 300, 200]

        # status events were published (app loading/ready, asr ready, llm error because nothing listens)
        assert _wait(lambda: any(isinstance(e, ComponentStatusEvent) and e.component == "app" and e.state == "ready" for e in _peek(q)))
        events = _peek(q)
        assert any(isinstance(e, ComponentStatusEvent) and e.component == "asr" and e.state == "ready" for e in events)
        assert _wait(lambda: any(isinstance(e, ComponentStatusEvent) and e.component == "llm" and e.state == "error" for e in _peek(q)), 10)

        folder = eng.export_session()
        assert folder.is_dir() and (folder / "transcript.md").is_file()
        assert "What is the plan" in (folder / "transcript.md").read_text(encoding="utf-8")
    finally:
        eng.stop()
    assert FakePipeline.instances[0].stopped
    assert ("stop", None) in FakeScreenReader.instances[0].calls
    assert (tmp_path / "config.toml").is_file()
    assert "mic_start_muted = true" in (tmp_path / "config.toml").read_text(encoding="utf-8")
    # no engine threads left running
    names = {t.name for t in threading.enumerate()}
    assert not any(n.startswith(("engine-init", "llm-init", "assist-", "notes-", "bus-listener")) for n in names)


_seen: list[Any] = []


def _peek(q: "queue.Queue[Any]") -> list[Any]:
    _seen.extend(drain(q))
    return list(_seen)


def test_new_session_resets_state(engine_env: tuple[Config, EventBus]) -> None:
    cfg, bus = engine_env
    eng = app_mod.Engine(cfg, bus)
    eng.start()
    try:
        assert _wait(lambda: FakePipeline.instances and FakePipeline.instances[0].started)
        FakePipeline.instances[0].emit_final("hello there everyone")
        assert eng.transcript.segments()
        first_folder = eng.recorder.folder
        eng.new_session()
        assert eng.transcript.segments() == []
        assert eng.clusterer.reset_calls == 1
        assert eng.recorder.folder != first_folder or first_folder.is_dir()
    finally:
        eng.stop()


class _FakeTracker:
    available = True
    detail = "fake tracker"
    last_reject = ""

    def __init__(self, *a: Any, **kw: Any) -> None:
        pass

    def attach(self) -> bool:
        return True

    def detach(self) -> None:
        pass

    def probe(self) -> Any:
        return None


def test_completion_mode_waits_for_llm_then_starts(engine_env: tuple[Config, EventBus], monkeypatch: pytest.MonkeyPatch) -> None:
    import exocortex.screen.caret as caret_mod
    from exocortex.bus import CompletionModeEvent

    monkeypatch.setattr(caret_mod, "CaretTracker", _FakeTracker)
    cfg, bus = engine_env
    q = bus.subscribe()
    eng = app_mod.Engine(cfg, bus)
    eng.start()
    try:
        eng.set_completion_mode(True)             # before the LLM is up: deferred, not started
        assert cfg.completion.enabled is True
        assert eng.completer is None or eng._llm_ready.is_set()
        assert _wait(lambda: eng.completer is not None, 10)
        assert _wait(lambda: any(isinstance(e, ComponentStatusEvent) and e.component == "completion" and e.state == "ready"
                                 for e in _peek(q)))
        assert any(isinstance(e, CompletionModeEvent) and e.enabled for e in _peek(q))
        eng.set_paused(True)
        assert eng.completer.paused is True
        eng.set_completion_context({"screen": False}, "terse")
        assert cfg.completion.use_screen is False and cfg.completion.hint == "terse"
        eng.new_session()
        eng.set_completion_mode(False)
        assert eng.completer is None and cfg.completion.enabled is False
    finally:
        eng.stop()
    assert not any(t.name == "completer" for t in threading.enumerate())


def test_explanations_toggle_and_explain_now(engine_env: tuple[Config, EventBus]) -> None:
    cfg, bus = engine_env
    q = bus.subscribe()
    eng = app_mod.Engine(cfg, bus)
    eng.start()
    try:
        assert cfg.explain.enabled is False
        eng.set_explanations(True)
        assert cfg.explain.enabled is True
        assert _wait(lambda: any(isinstance(e, ComponentStatusEvent) and e.detail == "explanations on" for e in _peek(q)))
        eng.explain_now()                     # LLM not ready: deferred, never raises
        assert ("explain", "", "") in eng._pending or eng._llm_ready.is_set()
        eng.set_explanations(False)
        assert cfg.explain.enabled is False
    finally:
        eng.stop()
    text = (cfg.path).read_text(encoding="utf-8")
    assert "[explain]" in text and "enabled = false" in text.split("[explain]")[1]


def test_reset_in_completion_mode_rebuilds_the_plan(engine_env: tuple[Config, EventBus]) -> None:
    cfg, bus = engine_env
    cfg.explain.enabled = True
    eng = app_mod.Engine(cfg, bus)
    eng.start()
    try:
        assert _wait(lambda: FakePipeline.instances and FakePipeline.instances[0].started)
        calls: list[str] = []

        class _Asst:
            def reset(self) -> None:
                calls.append("reset")

            def ensure_plan(self) -> None:
                calls.append("ensure_plan")

        real_assistant = eng.assistant
        eng.assistant = _Asst()
        eng._llm_ready.set()
        eng.new_session()
        assert calls == ["reset"]                    # full overlay: no completer, no plan needed
        eng.completer = type("C", (), {"reset": lambda self: None, "stop": lambda self: None})()
        eng.new_session()
        assert calls == ["reset", "reset", "ensure_plan"]
        eng.completer = None
        eng.assistant = real_assistant               # so stop() shuts the real worker threads down
    finally:
        eng.stop()


def test_checkpoint_works_before_the_llm_is_ready(engine_env: tuple[Config, EventBus]) -> None:
    """`Engine.checkpoint()` is never deferred (SPEC.md / the design's engine contract): the
    Assistant object exists from `start()` on, and a checkpoint makes no model call."""
    cfg, bus = engine_env
    q = bus.subscribe()
    eng = app_mod.Engine(cfg, bus)
    eng.start()
    try:
        assert eng.assistant is not None
        eng.checkpoint()
        assert not any(p[0] == "checkpoint" for p in eng._pending)
        assert _wait(lambda: any(isinstance(e, CheckpointEvent) for e in _peek(q)), 3.0)
    finally:
        eng.stop()


def test_checkpoint_resets_the_completer_when_one_exists(engine_env: tuple[Config, EventBus]) -> None:
    cfg, bus = engine_env
    eng = app_mod.Engine(cfg, bus)
    eng.start()
    try:
        calls: list[str] = []
        eng.completer = type("C", (), {"reset": lambda self: calls.append("reset")})()
        eng.checkpoint()
        assert calls == ["reset"]
        eng.completer = None
    finally:
        eng.stop()


def test_code_is_deferred_and_replayed_once_the_llm_is_ready(engine_env: tuple[Config, EventBus]) -> None:
    cfg, bus = engine_env
    q = bus.subscribe()
    eng = app_mod.Engine(cfg, bus)
    eng.start()
    try:
        eng.code("fix the bug")           # LLM not ready yet: deferred, never raises
        assert ("code", "fix the bug", "") in eng._pending or eng._llm_ready.is_set()
        assert _wait(lambda: eng._llm_ready.is_set())
        assert _wait(lambda: any(isinstance(e, AssistStartedEvent) and e.kind == "code" for e in _peek(q)))
    finally:
        eng.stop()


def test_text_is_deferred_and_replayed_once_the_llm_is_ready(engine_env: tuple[Config, EventBus]) -> None:
    cfg, bus = engine_env
    q = bus.subscribe()
    eng = app_mod.Engine(cfg, bus)
    eng.start()
    try:
        eng.text("walk me through it")     # LLM not ready yet: deferred, never raises
        assert ("text", "walk me through it", "") in eng._pending or eng._llm_ready.is_set()
        assert _wait(lambda: eng._llm_ready.is_set())
        assert _wait(lambda: any(isinstance(e, AssistStartedEvent) and e.kind == "text" for e in _peek(q)))
    finally:
        eng.stop()


def test_text_before_start_is_a_noop(engine_env: tuple[Config, EventBus]) -> None:
    """`Engine.text()` called before `start()` (assistant is still None) must not raise and
    must not queue anything for later replay."""
    cfg, bus = engine_env
    eng = app_mod.Engine(cfg, bus)
    assert eng.assistant is None
    eng.text("walk me through it")
    assert eng._pending == []


# ---- backend routing (exocortex.llm.route / exocortex.llm.client.make_client) ------------------
class FakeOllamaClient:
    """Just enough of the client surface Assistant.start()/Engine.stop() touch during a plain
    startup (no assist job is submitted in these tests): health(), list_models(), close()."""

    instances: list["FakeOllamaClient"] = []

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        self.closed = False
        FakeOllamaClient.instances.append(self)

    def health(self) -> Any:
        from exocortex.llm.client import Health

        return Health(reachable=True, model_listed=True, model_loaded=True, detail="fake ollama ready")

    def list_models(self) -> list[Any]:
        from exocortex.llm.client import ModelInfo

        return [ModelInfo(id="qwen3:8b", loaded=True)]

    def ensure_loaded(self) -> bool:
        return True

    def close(self) -> None:
        self.closed = True


def test_engine_publishes_the_route_reason(engine_env: tuple[Config, EventBus], monkeypatch: pytest.MonkeyPatch) -> None:
    """`Engine.start()` resolves a route (`app_mod.resolve_route`, monkeypatchable) and publishes
    its `reason` as the first "llm" status -- the doctor/log text and the overlay's status line
    both come from that same one sentence."""
    from exocortex.llm.route import Route

    cfg, bus = engine_env
    fake_route = Route(backend="openai", base_url=cfg.llm.base_url, reason="TEST ROUTE REASON 12345",
                       checked=("openai: probed 200",))
    monkeypatch.setattr(app_mod, "resolve_route", lambda *a, **k: fake_route)
    q = bus.subscribe()
    eng = app_mod.Engine(cfg, bus)
    eng.start()
    try:
        assert _wait(lambda: any(isinstance(e, ComponentStatusEvent) and e.component == "llm"
                                 and e.detail == "TEST ROUTE REASON 12345" for e in _peek(q)))
        assert eng.route is fake_route
    finally:
        eng.stop()


def test_engine_routes_to_ollama_via_a_fake_client(engine_env: tuple[Config, EventBus], monkeypatch: pytest.MonkeyPatch) -> None:
    """A route resolving to "ollama" makes Engine build its client through `make_client`
    (monkeypatched here) instead of always using LMStudioClient, and never touches `self.server`
    (that is the llamacpp-only path)."""
    from exocortex.llm.route import Route

    cfg, bus = engine_env
    fake_route = Route(backend="ollama", base_url=cfg.llm.ollama_url, reason="fake ollama route",
                       checked=("ollama: fake 200",))
    monkeypatch.setattr(app_mod, "resolve_route", lambda *a, **k: fake_route)
    made: list[str] = []

    def fake_make_client(llm_cfg: Any, backend: str, base_url: str | None = None) -> Any:
        made.append(backend)
        return FakeOllamaClient(llm_cfg)

    monkeypatch.setattr(app_mod, "make_client", fake_make_client)
    FakeOllamaClient.instances.clear()
    q = bus.subscribe()
    eng = app_mod.Engine(cfg, bus)
    eng.start()
    try:
        assert _wait(lambda: any(isinstance(e, ComponentStatusEvent) and e.component == "llm" and e.state == "ready"
                                 for e in _peek(q)))
        assert made == ["ollama"]
        assert isinstance(eng.client, FakeOllamaClient)
        assert eng.server is None
    finally:
        eng.stop()
    assert FakeOllamaClient.instances and FakeOllamaClient.instances[0].closed is True


class FakeUnreachableOllamaClient:
    """An `ollama`-routed client whose health() never comes back reachable -- drives
    `_init_llm`'s retry-then-fallback branch (app.py's `elif backend == "ollama":`)."""

    instances: list["FakeUnreachableOllamaClient"] = []

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        self.closed = False
        FakeUnreachableOllamaClient.instances.append(self)

    def health(self) -> Any:
        from exocortex.llm.client import Health

        return Health(reachable=False, model_listed=False, model_loaded=False, detail="fake: ollama down")

    def close(self) -> None:
        self.closed = True


class FakeOpenAIClient:
    """The client `make_client` hands back once the retry falls through to "openai"."""

    instances: list["FakeOpenAIClient"] = []

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        self.closed = False
        FakeOpenAIClient.instances.append(self)

    def health(self) -> Any:
        from exocortex.llm.client import Health

        return Health(reachable=True, model_listed=True, model_loaded=True, detail="fake openai ready")

    def list_models(self) -> list[Any]:
        from exocortex.llm.client import ModelInfo

        return [ModelInfo(id="gpt-oss", loaded=True)]

    def ensure_loaded(self) -> bool:
        return True

    def close(self) -> None:
        self.closed = True


def test_engine_keeps_assistant_client_in_sync_after_ollama_falls_back_to_openai(
    engine_env: tuple[Config, EventBus], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the fast probe in start() picks "ollama" but the up-to-20s retry in _init_llm finds
    Ollama gone and falls back to "openai", Engine.client must be rebuilt *and* the already-
    constructed Assistant must be repointed at that new client -- not left holding the dead
    Ollama client forever (regression test for the route-flip-leaves-Assistant-stale bug)."""
    from exocortex.llm.route import Route

    cfg, bus = engine_env
    first_route = Route(backend="ollama", base_url=cfg.llm.ollama_url, reason="fake ollama route",
                        checked=("ollama: fake 200",))
    retried_route = Route(backend="openai", base_url=cfg.llm.base_url, reason="fell back to openai",
                          checked=("ollama: fake down", "openai: fake 200"))
    routes = iter([first_route, retried_route])
    monkeypatch.setattr(app_mod, "resolve_route", lambda *a, **k: next(routes))

    made: list[str] = []

    def fake_make_client(llm_cfg: Any, backend: str, base_url: str | None = None) -> Any:
        made.append(backend)
        if backend == "ollama":
            return FakeUnreachableOllamaClient(llm_cfg)
        return FakeOpenAIClient(llm_cfg)

    monkeypatch.setattr(app_mod, "make_client", fake_make_client)
    FakeUnreachableOllamaClient.instances.clear()
    FakeOpenAIClient.instances.clear()
    q = bus.subscribe()
    eng = app_mod.Engine(cfg, bus)
    eng.start()
    try:
        assert _wait(lambda: any(isinstance(e, ComponentStatusEvent) and e.component == "llm" and e.state == "ready"
                                 for e in _peek(q)))
        assert made == ["ollama", "openai"]
        assert eng.route.backend == "openai"
        assert isinstance(eng.client, FakeOpenAIClient)
        # The bug: Assistant kept the old (dead) client by reference forever.
        assert eng.assistant._client is eng.client
        assert isinstance(eng.assistant._client, FakeOpenAIClient)
    finally:
        eng.stop()
    # The orphaned Ollama client is closed rather than left dangling once replaced.
    assert FakeUnreachableOllamaClient.instances and FakeUnreachableOllamaClient.instances[0].closed is True


def test_engine_reports_a_none_route_as_an_llm_error(engine_env: tuple[Config, EventBus], monkeypatch: pytest.MonkeyPatch) -> None:
    """A route that resolves to "none" (no backend usable anywhere) must surface as an "llm"
    error carrying the route's actionable reason, and must never start the assistant's worker
    threads (there is nothing for them to talk to)."""
    from exocortex.llm.route import Route

    cfg, bus = engine_env
    none_route = Route(backend="none", base_url="", reason="no backend available: install LM Studio or Ollama",
                       checked=("llamacpp: no pack", "ollama: x 0", "openai: x 0"))
    monkeypatch.setattr(app_mod, "resolve_route", lambda *a, **k: none_route)
    q = bus.subscribe()
    eng = app_mod.Engine(cfg, bus)
    eng.start()
    try:
        assert _wait(lambda: any(isinstance(e, ComponentStatusEvent) and e.component == "llm" and e.state == "error"
                                 and e.detail == none_route.reason for e in _peek(q)))
        assert not eng._llm_ready.is_set()
    finally:
        eng.stop()


# ---- upstream finding 9: deferred requests must never be silently dropped ---------------------
def test_defer_rechecks_llm_ready_inside_the_lock(engine_env: tuple[Config, EventBus]) -> None:
    """upstream finding 9 (ported from mac-cortex): deterministic regression test for the TOCTOU
    race between `_defer()`'s pre-lock `is_set()` check and `_init_llm()`'s set()+drain, both
    under `self._pending_lock`. A lock wrapper simulates `_init_llm` completing that set()+drain
    exactly inside the window the first (pre-lock) check would miss: `_llm_ready` becomes set the
    instant `_defer` acquires `self._pending_lock`. The second, in-lock check must see it and bail
    out instead of appending -- the request must never enter `self._pending` (where nothing would
    ever drain it again)."""
    cfg, bus = engine_env
    eng = app_mod.Engine(cfg, bus)
    eng.assistant = object()   # only its not-None-ness is checked by _defer(); no real assistant needed
    real_lock = eng._pending_lock

    class _RacingLock:
        def __enter__(self) -> None:
            eng._llm_ready.set()   # _init_llm's set()+drain lands here, right as _defer takes the lock
            real_lock.acquire()

        def __exit__(self, *exc_info: Any) -> None:
            real_lock.release()

    eng._pending_lock = _RacingLock()
    assert eng._defer("ask", "hi") is False
    assert eng._pending == []


def test_drop_pending_reports_a_request_deferred_before_init_llm_fails(
    engine_env: tuple[Config, EventBus], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """upstream finding 9 (ported from mac-cortex): a request `_defer()`red before `_init_llm`
    fails must not sit in `self._pending` forever (the only drain lives on the success tail,
    which a failure never reaches). Gates `_drop_pending` on a `threading.Event` so the deferred
    `ask` is guaranteed to land in `self._pending` before the failure branch is allowed to drain
    it -- deterministic ordering, no reliance on thread-scheduling luck or sleeps."""
    from exocortex.llm.route import Route

    cfg, bus = engine_env
    none_route = Route(backend="none", base_url="", reason="no backend available: install LM Studio or Ollama",
                       checked=("llamacpp: no pack", "ollama: x 0", "openai: x 0"))
    monkeypatch.setattr(app_mod, "resolve_route", lambda *a, **k: none_route)

    gate = threading.Event()
    real_drop_pending = app_mod.Engine._drop_pending

    def _gated_drop_pending(self: Any, reason: str) -> None:
        assert gate.wait(5.0), "test never released the gate"
        real_drop_pending(self, reason)

    monkeypatch.setattr(app_mod.Engine, "_drop_pending", _gated_drop_pending)

    q = bus.subscribe()
    eng = app_mod.Engine(cfg, bus)
    eng.start()
    try:
        eng.ask("what is the plan")   # deferred while _init_llm's failure sits behind the gate
        assert ("ask", "what is the plan", "") in eng._pending
        gate.set()   # let _init_llm's "none" branch proceed into _drop_pending
        assert _wait(lambda: eng._pending == [])
        assert _wait(lambda: any(isinstance(e, ComponentStatusEvent) and e.component == "llm"
                                 and e.state == "error" and "ask" in e.detail for e in _peek(q)))
    finally:
        eng.stop()


def test_requests_after_an_llm_init_failure_are_refused_with_a_notice(engine_env: tuple[Config, EventBus]) -> None:
    """upstream finding 9 (Windows fix): once `_init_llm` has failed for good (`_drop_pending`
    ran; nothing ever sets `_llm_ready` afterwards), a later request must not be queued behind a
    "still starting" notice that is no longer true -- it is consumed with an "llm"/"error" status
    naming the request, and `_pending` stays empty."""
    cfg, bus = engine_env
    q = bus.subscribe()
    eng = app_mod.Engine(cfg, bus)
    eng.assistant = object()   # _defer only checks that an assistant exists
    eng._drop_pending("no LLM backend available")
    assert eng._defer("ask", "hi") is True
    assert eng._pending == []
    statuses = [e for e in _peek(q)
                if isinstance(e, ComponentStatusEvent) and e.component == "llm" and e.state == "error"]
    assert any("ask" in e.detail and "no LLM backend available" in e.detail for e in statuses)

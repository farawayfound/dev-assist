"""Wiring tests for hands-free mode, OUTSIDE exocortex.handsfree.HandsFreeController itself
(see exocortex/handsfree.py's module docstring for the controller's own contract, which this
file does not re-test):

  (a) exocortex.config -- HandsFreeConfig defaults, the new ScreenConfig.interval_s/ocr_device
      defaults, HotkeysConfig.toggle_handsfree, and that [handsfree] round-trips through
      to_toml()/load_config() (including an old config file with no [handsfree] section at all).
  (b) exocortex.bus -- HandsFreeEvent defaults and that ScreenTextEvent is still constructible
      without its new engine/accurate fields.
  (c) exocortex.llm.assistant.Assistant -- set_hands_free() gates auto-solve/suggest and
      screen-question answers off and back on, text()/code() return job ids, recent_code() fills
      from a finished code job and is cleared by checkpoint(), and _screen_text_for_job's
      hands-free context (cfg.handsfree.context_chars, not cfg.screen.max_text_chars) feeds
      "text"/"code" jobs specifically -- an "ask" job still gets the plain screen provider, and an
      empty context falls back to it too.
  (d) exocortex.app.Engine -- set_hands_free() creates/starts/stops a HandsFreeController and
      tells the assistant, set_paused()/new_session()/stop() propagate to it (stop() before the
      screen reader), apply_settings({"handsfree.enabled": ...}) wires the same path and saves
      the config, _init_screen registers RapidOCR as the accurate engine exactly when
      cfg.handsfree.accurate_ocr is on and the sampling engine isn't already rapidocr, its
      engine_factory calls exocortex.screen.ocr.create_engine with the configured device and
      cache dir, _HandsFreeHooks.text()/code()/recent_code() proxy the Engine/assistant, and
      cfg.handsfree.enabled=True turns it on from start().

Fully offline throughout: (c) reuses tests/test_assistant.py's FakeClient/drain_until/
make_segment and tests/test_assistant_screen.py's ScreenTextEvent-publishing pattern against a
real EventBus + real Transcript + real Assistant; (d) uses a fake HandsFreeController
(monkeypatched into exocortex.handsfree, which exocortex.app.Engine.set_hands_free imports
lazily -- so no hands-free thread ever runs here) and, for the two _init_screen tests, a richer
FakeScreenReader than tests/test_app.py's (it also implements capture_now(accurate=) and
set_accurate_engine, and records the engine_factory it was constructed with) monkeypatched into
exocortex.screen.reader the same way tests/test_app.py does.
"""
from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

import exocortex.handsfree as handsfree_module
import exocortex.llm.assistant as assistant_module
import exocortex.screen.ocr as ocr_module
from exocortex import app as app_mod
from exocortex.bus import (
    AssistDoneEvent,
    AssistStartedEvent,
    EventBus,
    HandsFreeEvent,
    ScreenTextEvent,
    drain,
)
from exocortex.config import Config, HandsFreeConfig, load_config, to_toml
from exocortex.llm.assistant import Assistant
from exocortex.transcript import Transcript
from tests.test_assistant import FakeClient, drain_until, make_segment

# =====================================================================================
# (a) exocortex.config
# =====================================================================================


def test_handsfree_config_defaults() -> None:
    hf = Config().handsfree
    assert hf.enabled is False
    assert hf.settle_s == 1.0
    assert hf.change_threshold == 0.6
    assert hf.min_changed_lines == 3
    assert hf.deviation_threshold == 0.5
    assert hf.context_chars == 6000
    assert hf.run_text is True
    assert hf.run_code is True
    assert hf.code_first is False
    assert hf.accurate_ocr is True


def test_screen_config_interval_and_ocr_device_defaults() -> None:
    screen = Config().screen
    assert screen.interval_s == 1.5
    assert screen.ocr_device == "auto"


def test_hotkeys_toggle_handsfree_default() -> None:
    assert Config().hotkeys.toggle_handsfree == "ctrl+alt+h"


def test_to_toml_round_trips_a_handsfree_section(tmp_path: Path) -> None:
    cfg = Config()
    cfg.handsfree.enabled = True
    cfg.handsfree.settle_s = 2.5
    cfg.handsfree.change_threshold = 0.75
    cfg.handsfree.min_changed_lines = 5
    cfg.handsfree.deviation_threshold = 0.25
    cfg.handsfree.context_chars = 9000
    cfg.handsfree.run_text = False
    cfg.handsfree.code_first = True
    cfg.handsfree.accurate_ocr = False
    text = to_toml(cfg)
    assert "[handsfree]" in text

    p = tmp_path / "config.toml"
    p.write_text(text, encoding="utf-8")
    loaded = load_config(p)
    assert loaded.handsfree == cfg.handsfree


def test_old_config_without_handsfree_section_loads_with_defaults(tmp_path: Path) -> None:
    p = tmp_path / "config.toml"
    p.write_text('[llm]\nmodel = "custom-model"\n', encoding="utf-8")
    cfg = load_config(p)
    assert cfg.llm.model == "custom-model"          # the rest of the file still loaded
    assert cfg.handsfree == HandsFreeConfig()        # missing section -> all defaults


# =====================================================================================
# (b) exocortex.bus
# =====================================================================================


def test_handsfree_event_defaults() -> None:
    ev = HandsFreeEvent()
    assert ev.enabled is False
    assert ev.state == "off"
    assert ev.detail == ""


def test_screen_text_event_constructible_without_engine_or_accurate() -> None:
    ev = ScreenTextEvent(
        mode="auto", source_name="Fake Window", rect=(0, 0, 100, 100), text="hi",
        lines=("hi",), changed=True, ocr_ms=1.0, captured_at=0.0,
    )
    assert ev.engine == ""
    assert ev.accurate is False


# =====================================================================================
# (c) exocortex.llm.assistant.Assistant
# =====================================================================================


def make_screen_event(
    *,
    lines: tuple[str, ...],
    text: str | None = None,
    changed: bool = True,
    mode: str = "auto",
    source_name: str = "Fake Window",
    rect: tuple[int, int, int, int] = (0, 0, 100, 100),
    ocr_ms: float = 1.0,
    captured_at: float = 0.0,
) -> ScreenTextEvent:
    return ScreenTextEvent(
        mode=mode, source_name=source_name, rect=rect,
        text=text if text is not None else "\n".join(lines), lines=tuple(lines),
        changed=changed, ocr_ms=ocr_ms, captured_at=captured_at,
    )


def _make_assistant(cfg: Config, bus: EventBus, transcript: Transcript, client: FakeClient) -> Assistant:
    return Assistant(cfg, bus, client, transcript, clock=time.monotonic)


@pytest.fixture()
def bus() -> EventBus:
    return EventBus()


@pytest.fixture()
def transcript() -> Transcript:
    return Transcript()


# ---- set_hands_free() gates auto-solve/suggest and screen questions -------------------------


def test_hands_free_on_suppresses_auto_solve_and_screen_question(bus: EventBus, transcript: Transcript) -> None:
    cfg = Config()
    cfg.screen.auto_suggest = True
    cfg.screen.answer_screen_questions = True
    cfg.screen.suggest_min_interval_s = 0
    client = FakeClient(token_batches=[["x"] for _ in range(4)], chunk_delay=0.0)
    assistant = _make_assistant(cfg, bus, transcript, client)
    assistant.set_hands_free(True)
    sub = bus.subscribe()
    assistant.start()
    try:
        question = "How would you implement a thread-safe LRU cache here?"
        bus.publish(make_screen_event(lines=(question,)))
        time.sleep(0.3)
        assert not any(isinstance(e, AssistStartedEvent) for e in drain(sub))
    finally:
        assistant.stop()


def test_hands_free_off_restores_auto_solve(bus: EventBus, transcript: Transcript) -> None:
    cfg = Config()
    cfg.screen.auto_suggest = True
    cfg.screen.answer_screen_questions = False   # isolate from the screen-question path
    cfg.screen.suggest_min_interval_s = 0
    client = FakeClient(token_batches=[["an answer"]], chunk_delay=0.0)
    assistant = _make_assistant(cfg, bus, transcript, client)
    assistant.set_hands_free(True)
    assistant.set_hands_free(False)
    sub = bus.subscribe()
    assistant.start()
    try:
        bus.publish(make_screen_event(lines=("Some entirely new screen content appears now.",)))
        events = drain_until(sub, lambda e: isinstance(e, AssistStartedEvent))
        started = next(e for e in events if isinstance(e, AssistStartedEvent))
        assert started.kind == "solve"          # cfg.assist.autonomous defaults to True
    finally:
        assistant.stop()


def test_hands_free_off_restores_screen_question_answers(
    bus: EventBus, transcript: Transcript, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(assistant_module, "SCREEN_QUESTION_MIN_INTERVAL_S", 0.05)
    cfg = Config()
    cfg.screen.answer_screen_questions = True
    cfg.screen.auto_suggest = False   # isolate from the auto-suggest path
    client = FakeClient(token_batches=[["- an answer"]], chunk_delay=0.0)
    assistant = _make_assistant(cfg, bus, transcript, client)
    assistant.set_hands_free(True)
    assistant.set_hands_free(False)
    sub = bus.subscribe()
    assistant.start()
    try:
        question = "How would you implement a thread-safe LRU cache here?"
        bus.publish(make_screen_event(lines=(question,)))
        events = drain_until(sub, lambda e: isinstance(e, AssistStartedEvent))
        started = next(e for e in events if isinstance(e, AssistStartedEvent))
        assert started.kind == "screen"
        assert started.question == question
    finally:
        assistant.stop()


# ---- text()/code() job ids; recent_code() -----------------------------------------------------


def test_text_and_code_return_job_ids_matching_the_started_event(bus: EventBus, transcript: Transcript) -> None:
    cfg = Config()
    client = FakeClient(token_batches=[["a walkthrough"], ["some code"]])
    assistant = _make_assistant(cfg, bus, transcript, client)
    sub = bus.subscribe()
    assistant.start()
    try:
        text_id = assistant.text()
        events = drain_until(sub, lambda e: isinstance(e, AssistDoneEvent))
        started = next(e for e in events if isinstance(e, AssistStartedEvent))
        assert started.kind == "text" and started.job_id == text_id

        code_id = assistant.code()
        events2 = drain_until(sub, lambda e: isinstance(e, AssistDoneEvent))
        started2 = next(e for e in events2 if isinstance(e, AssistStartedEvent))
        assert started2.kind == "code" and started2.job_id == code_id
        assert code_id != text_id
    finally:
        assistant.stop()


def test_recent_code_empty_then_filled_by_code_job_then_cleared_by_checkpoint(
    bus: EventBus, transcript: Transcript
) -> None:
    cfg = Config()
    client = FakeClient(token_batches=[["def solve():\n    return 42"]])
    assistant = _make_assistant(cfg, bus, transcript, client)
    sub = bus.subscribe()
    assistant.start()
    try:
        assert assistant.recent_code() == ""
        assistant.code()
        drain_until(sub, lambda e: isinstance(e, AssistDoneEvent))
        assert "def solve():" in assistant.recent_code()
        assistant.checkpoint()
        assert assistant.recent_code() == ""
    finally:
        assistant.stop()


# ---- hands-free context feeding "text"/"code" jobs --------------------------------------------


def test_hands_free_context_feeds_text_and_code_jobs_without_max_text_chars_truncation(
    bus: EventBus, transcript: Transcript
) -> None:
    cfg = Config()
    long_text = "line of screen content " * 200
    long_text = long_text[: cfg.screen.max_text_chars + 500]   # > max_text_chars, < context_chars
    assert cfg.screen.max_text_chars < len(long_text) < cfg.handsfree.context_chars
    client = FakeClient(token_batches=[["ok"], ["ok"]])
    assistant = _make_assistant(cfg, bus, transcript, client)
    assistant.set_hands_free(True, lambda: long_text)
    sub = bus.subscribe()
    assistant.start()
    try:
        assistant.text()
        drain_until(sub, lambda e: isinstance(e, AssistDoneEvent))
        assert long_text in client.calls[-1]["prompt"]

        assistant.code()
        drain_until(sub, lambda e: isinstance(e, AssistDoneEvent))
        assert long_text in client.calls[-1]["prompt"]
    finally:
        assistant.stop()


def test_ask_job_uses_the_plain_screen_provider_even_with_hands_free_context_set(
    bus: EventBus, transcript: Transcript
) -> None:
    cfg = Config()
    client = FakeClient(token_batches=[["ok"]])
    assistant = _make_assistant(cfg, bus, transcript, client)
    assistant.set_screen_provider(lambda: "plain screen provider text")
    assistant.set_hands_free(True, lambda: "cumulative hands-free context text")
    sub = bus.subscribe()
    assistant.start()
    try:
        assistant.ask("what's going on?")
        drain_until(sub, lambda e: isinstance(e, AssistDoneEvent))
        prompt = client.calls[-1]["prompt"]
        assert "plain screen provider text" in prompt
        assert "cumulative hands-free context text" not in prompt
    finally:
        assistant.stop()


def test_hands_free_context_provider_returning_empty_falls_back_to_screen_provider(
    bus: EventBus, transcript: Transcript
) -> None:
    cfg = Config()
    client = FakeClient(token_batches=[["ok"]])
    assistant = _make_assistant(cfg, bus, transcript, client)
    assistant.set_screen_provider(lambda: "fallback screen provider text")
    assistant.set_hands_free(True, lambda: "")
    sub = bus.subscribe()
    assistant.start()
    try:
        assistant.text()
        drain_until(sub, lambda e: isinstance(e, AssistDoneEvent))
        assert "fallback screen provider text" in client.calls[-1]["prompt"]
    finally:
        assistant.stop()


# =====================================================================================
# (d) exocortex.app.Engine
# =====================================================================================


class FakeAssistant:
    """Stand-in for exocortex.llm.assistant.Assistant: only the surface app.Engine touches for
    hands-free wiring (set_hands_free/set_paused/reset/stop/checkpoint/code/text/recent_code)."""

    def __init__(self) -> None:
        self.hands_free_calls: list[tuple[bool, Any]] = []
        self.paused_calls: list[bool] = []
        self.reset_calls = 0
        self.stop_calls = 0
        self.checkpoint_calls = 0
        self.code_calls: list[str] = []
        self.text_calls: list[str] = []
        self.code_labels: list[str] = []
        self.text_labels: list[str] = []
        self.recent_code_value = ""
        self._next_id = 100

    def set_hands_free(self, enabled: bool, context_provider: Any = None) -> None:
        self.hands_free_calls.append((enabled, context_provider))

    def set_paused(self, paused: bool) -> None:
        self.paused_calls.append(paused)

    def reset(self) -> None:
        self.reset_calls += 1

    def stop(self) -> None:
        self.stop_calls += 1

    def checkpoint(self) -> None:
        self.checkpoint_calls += 1

    def code(self, focus: str = "", label: str = "") -> int:
        self.code_calls.append(focus)
        self.code_labels.append(label)
        self._next_id += 1
        return self._next_id

    def text(self, focus: str = "", label: str = "") -> int:
        self.text_calls.append(focus)
        self.text_labels.append(label)
        self._next_id += 1
        return self._next_id

    def recent_code(self) -> str:
        return self.recent_code_value


class FakeHandsFreeController:
    """Stand-in for exocortex.handsfree.HandsFreeController: records calls, spawns no thread."""

    instances: list["FakeHandsFreeController"] = []

    def __init__(self, cfg: Config, bus: EventBus, hooks: Any, clock: Any) -> None:
        self.cfg = cfg
        self.bus = bus
        self.hooks = hooks
        self.clock = clock
        self.calls: list[str] = []
        self.paused_ = False
        self.reset_calls = 0
        FakeHandsFreeController.instances.append(self)

    def start(self) -> None:
        self.calls.append("start")

    def stop(self) -> None:
        self.calls.append("stop")

    def set_paused(self, paused: bool) -> None:
        self.paused_ = paused
        self.calls.append(f"paused:{paused}")

    def reset(self) -> None:
        self.reset_calls += 1
        self.calls.append("reset")

    def context_text(self) -> str:
        return "cumulative hands-free context"


@pytest.fixture()
def bare_engine(monkeypatch: pytest.MonkeyPatch) -> app_mod.Engine:
    """A minimal, never-started Engine with a fake assistant and the real handsfree module's
    HandsFreeController monkeypatched to a fake -- for the wiring that touches no other
    component (set_hands_free/set_paused/new_session/stop/apply_settings)."""
    cfg = Config()
    eng = app_mod.Engine(cfg, EventBus())
    eng.assistant = FakeAssistant()
    monkeypatch.setattr(handsfree_module, "HandsFreeController", FakeHandsFreeController)
    FakeHandsFreeController.instances.clear()
    return eng


def test_set_hands_free_true_creates_controller_wires_assistant_and_starts(bare_engine: app_mod.Engine) -> None:
    eng = bare_engine
    eng.set_hands_free(True)
    assert isinstance(eng.handsfree, FakeHandsFreeController)
    assert eng.cfg.handsfree.enabled is True
    assert eng.assistant.hands_free_calls[-1] == (True, eng.handsfree.context_text)
    assert "start" in eng.handsfree.calls
    assert eng.handsfree.paused_ is False   # eng._paused defaults to False


def test_set_hands_free_false_stops_controller_and_notifies_assistant(bare_engine: app_mod.Engine) -> None:
    eng = bare_engine
    eng.set_hands_free(True)
    eng.set_hands_free(False)
    assert eng.cfg.handsfree.enabled is False
    assert "stop" in eng.handsfree.calls
    assert eng.assistant.hands_free_calls[-1] == (False, None)


def test_set_paused_propagates_to_handsfree_controller(bare_engine: app_mod.Engine) -> None:
    eng = bare_engine
    eng.set_hands_free(True)
    eng.set_paused(True)
    assert eng.handsfree.paused_ is True
    eng.set_paused(False)
    assert eng.handsfree.paused_ is False


def test_new_session_resets_handsfree_controller(bare_engine: app_mod.Engine) -> None:
    eng = bare_engine
    eng.set_hands_free(True)
    eng.new_session()
    assert eng.handsfree.reset_calls == 1


def test_stop_stops_handsfree_controller_before_the_screen_reader(
    bare_engine: app_mod.Engine, tmp_path: Path
) -> None:
    eng = bare_engine
    eng.cfg.path = tmp_path / "config.toml"
    eng.set_hands_free(True)
    order: list[str] = []
    eng.handsfree.stop = lambda: order.append("hands-free")   # type: ignore[method-assign]
    eng.screen = SimpleNamespace(stop=lambda: order.append("screen"))
    eng.stop()
    assert order == ["hands-free", "screen"]


def test_apply_settings_handsfree_enabled_turns_it_on_and_saves(bare_engine: app_mod.Engine, tmp_path: Path) -> None:
    eng = bare_engine
    eng.cfg.path = tmp_path / "config.toml"
    eng.apply_settings({"handsfree.enabled": True})
    assert eng.cfg.handsfree.enabled is True
    assert isinstance(eng.handsfree, FakeHandsFreeController)
    assert "start" in eng.handsfree.calls
    saved = load_config(eng.cfg.path)
    assert saved.handsfree.enabled is True


def test_handsfree_hooks_text_code_and_recent_code_proxy_the_engine_and_assistant() -> None:
    cfg = Config()
    eng = app_mod.Engine(cfg, EventBus())
    fake = FakeAssistant()
    eng.assistant = fake
    hooks = app_mod._HandsFreeHooks(eng)

    # the LLM isn't ready yet: Engine.text()/code() defer instead of calling the assistant,
    # and the hooks return exactly that (None), never raising.
    assert hooks.text() is None
    assert hooks.code() is None
    assert fake.text_calls == [] and fake.code_calls == []

    # once ready, the hooks return exactly what Engine.text()/code() return.
    eng._llm_ready.set()
    text_id = hooks.text()
    assert fake.text_calls == [""]
    assert text_id == fake._next_id

    code_id = hooks.code()
    assert fake.code_calls == [""]
    assert code_id == fake._next_id
    assert code_id != text_id

    fake.recent_code_value = "def f():\n    return 1"
    assert hooks.recent_code() == fake.recent_code_value


def test_handsfree_hooks_code_forwards_focus_and_label_through_engine_to_the_assistant() -> None:
    """`_HandsFreeHooks.code(focus, label)` reaches `Engine.code(focus, label)`
    (exocortex.handsfree's HANDS_FREE_CODE_FOCUS / HANDS_FREE_CODE_LABEL for a code task), which
    forwards `label` on to the assistant as `label=` (`label` is what the Assist tab / history
    show instead of `focus`)."""
    cfg = Config()
    eng = app_mod.Engine(cfg, EventBus())
    fake = FakeAssistant()
    eng.assistant = fake
    eng._llm_ready.set()
    hooks = app_mod._HandsFreeHooks(eng)

    job_id = hooks.code(handsfree_module.HANDS_FREE_CODE_FOCUS, handsfree_module.HANDS_FREE_CODE_LABEL)

    assert fake.code_calls == [handsfree_module.HANDS_FREE_CODE_FOCUS]
    assert fake.code_labels == [handsfree_module.HANDS_FREE_CODE_LABEL]
    assert job_id == fake._next_id

    # text() forwards the same way
    text_job_id = hooks.text("a text focus", "a text label")
    assert fake.text_calls == ["a text focus"]
    assert fake.text_labels == ["a text label"]
    assert text_job_id == fake._next_id


# ---- _init_screen: the accurate engine + engine_factory ---------------------------------------


class FakeTranscriber:
    def __init__(self, cfg: Any, on_status: Any = None, **kw: Any) -> None:
        self.cfg, self.on_status = cfg, on_status
        self.loaded = False

    def load(self) -> None:
        self.loaded = True
        if self.on_status:
            self.on_status("ready", "fake")

    device_used = "fake-cpu"
    compute_type_used = "int8"


class FakeClusterer:
    def __init__(self, path: str, **kw: Any) -> None:
        self.path = path

    def rename(self, sid: str, label: str) -> None:
        pass

    def reset(self) -> None:
        pass


class FakePipeline:
    instances: list["FakePipeline"] = []

    def __init__(self, cfg: Any, bus: Any, transcriber: Any, clusterer: Any, clock: Any, next_id: Any, **kw: Any) -> None:
        self.started = self.stopped = False
        FakePipeline.instances.append(self)

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def set_mic_muted(self, muted: bool) -> None:
        pass


class FakeScreenReader:
    """Richer than tests/test_app.py's FakeScreenReader: also implements
    capture_now(accurate=)/set_accurate_engine (hands-free's needs) and records the
    engine_factory it was constructed with, so a test can call it directly to exercise
    Engine._init_screen's exocortex.screen.ocr.create_engine wiring."""

    instances: list["FakeScreenReader"] = []

    def __init__(self, cfg: Any, bus: Any, clock: Any, *, engine_factory: Any = None, **kw: Any) -> None:
        self.cfg = cfg
        self.bus = bus
        self.clock = clock
        self.engine_factory = engine_factory
        self.calls: list[tuple[str, Any]] = []
        self.paused_ = False
        self.latest_text = "screen text"
        self.accurate_engine_name: str | None = None
        FakeScreenReader.instances.append(self)

    def start(self) -> None:
        self.calls.append(("start", None))

    def stop(self) -> None:
        self.calls.append(("stop", None))

    def capture_now(self, accurate: bool = False) -> None:
        self.calls.append(("capture", accurate))

    def set_accurate_engine(self, name: str | None) -> None:
        self.accurate_engine_name = name
        self.calls.append(("accurate_engine", name))

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
    """Same shape as tests/test_app.py's engine_env fixture, but with the richer FakeScreenReader
    above (that file's own fake lacks capture_now(accurate=) and set_accurate_engine)."""
    cfg = Config()
    cfg.path = tmp_path / "config.toml"
    cfg.paths.sessions_dir = str(tmp_path / "sessions")
    cfg.paths.models_dir = str(tmp_path / "models")
    cfg.llm.backend = "openai"
    cfg.llm.base_url = "http://127.0.0.1:9"       # nothing listens -> llm error status, no hang
    cfg.llm.request_timeout_s = 2.0
    cfg.notes.auto_interval_s = 0
    cfg.screen.mode = "off"
    (tmp_path / "models").mkdir()
    for name in ("silero_vad.onnx", cfg.diarization.model):
        (tmp_path / "models" / name).write_bytes(b"x" * 200_000)

    import exocortex.screen.reader as reader_mod
    import exocortex.speech.asr as asr_mod
    import exocortex.speech.diarize as diar_mod
    import exocortex.speech.pipeline as pipe_mod

    monkeypatch.setattr(asr_mod, "Transcriber", FakeTranscriber)
    monkeypatch.setattr(diar_mod, "SpeakerClusterer", FakeClusterer)
    monkeypatch.setattr(pipe_mod, "SpeechPipeline", FakePipeline)
    monkeypatch.setattr(reader_mod, "ScreenReader", FakeScreenReader)

    import exocortex.llm.client as client_mod

    real_client = client_mod.LMStudioClient

    def _factory(llm_cfg: Any, **kw: Any) -> Any:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused", request=request)

        return real_client(llm_cfg, transport=httpx.MockTransport(handler))

    monkeypatch.setattr(client_mod, "LMStudioClient", _factory)
    FakePipeline.instances.clear()
    FakeScreenReader.instances.clear()
    return cfg, EventBus()


@pytest.mark.parametrize(
    "accurate_ocr, ocr_engine, expected",
    [
        (True, "auto", "rapidocr"),        # accurate_ocr on, sampling engine isn't rapidocr -> registered
        (True, "rapidocr", None),          # sampling engine already rapidocr -> nothing to register
        (False, "auto", None),             # accurate_ocr off -> never registered
    ],
)
def test_init_screen_registers_rapidocr_as_the_accurate_engine_per_config(
    engine_env: tuple[Config, EventBus], accurate_ocr: bool, ocr_engine: str, expected: str | None
) -> None:
    cfg, bus = engine_env
    cfg.handsfree.accurate_ocr = accurate_ocr
    cfg.screen.ocr_engine = ocr_engine
    eng = app_mod.Engine(cfg, bus)
    eng.start()
    try:
        assert FakeScreenReader.instances
        assert FakeScreenReader.instances[0].accurate_engine_name == expected
    finally:
        _wait(lambda: eng._llm_ready.is_set())
        eng.stop()


def test_init_screen_engine_factory_calls_ocr_create_engine_with_device_and_cache_dir(
    engine_env: tuple[Config, EventBus], monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg, bus = engine_env
    calls: list[tuple[str, str, Any]] = []

    def fake_create_engine(name: str, device: str = "auto", cache_dir: Any = None) -> Any:
        calls.append((name, device, cache_dir))
        return SimpleNamespace(name=name, device=device)

    monkeypatch.setattr(ocr_module, "create_engine", fake_create_engine)
    eng = app_mod.Engine(cfg, bus)
    eng.start()
    try:
        reader = FakeScreenReader.instances[0]
        assert reader.engine_factory is not None
        engine = reader.engine_factory("windows")
        assert calls == [("windows", cfg.screen.ocr_device, cfg.models_dir() / "ov_cache")]
        assert engine.name == "windows"
    finally:
        _wait(lambda: eng._llm_ready.is_set())
        eng.stop()


def test_handsfree_enabled_in_config_turns_it_on_at_engine_start(
    engine_env: tuple[Config, EventBus], monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg, bus = engine_env
    cfg.handsfree.enabled = True
    monkeypatch.setattr(handsfree_module, "HandsFreeController", FakeHandsFreeController)
    eng = app_mod.Engine(cfg, bus)
    eng.start()
    try:
        assert isinstance(eng.handsfree, FakeHandsFreeController)
        assert "start" in eng.handsfree.calls
        assert cfg.handsfree.enabled is True
    finally:
        _wait(lambda: eng._llm_ready.is_set())
        eng.stop()

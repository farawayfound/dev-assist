"""Engine's model-manager surface (2026-09-27, portable-install): the EngineActions methods the
Models window calls, driven with a fake llama-server manager and a monkeypatched
exocortex.llm.library / exocortex.hardware. Nothing here touches the network, the GPU or disk
outside tmp_path; every background thread is joined through the events it publishes."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from exocortex import app as app_mod
from exocortex import hardware
from exocortex.bus import (
    ComponentStatusEvent,
    EventBus,
    ModelInfoEvent,
    ModelLibraryEvent,
    ModelPullEvent,
    ModelSearchEvent,
    drain,
)
from exocortex.config import Config, load_config
from exocortex.llm import library


# ---- fakes ----------------------------------------------------------------------------------
@dataclass
class _Meta:
    n_ctx_train: int | None = 8192


@dataclass
class _Model:
    key: str
    name: str = ""
    size_bytes: int = 7_000_000_000
    source: str = "library"
    quant: str = "Q4_K_M"
    path: Path = Path("X:/lib/a/b/file.gguf")
    meta: Any = field(default_factory=_Meta)

    def __post_init__(self) -> None:
        self.name = self.name or self.key.split("/")[-1]


class _Server:
    def __init__(self, model_key: str | None = "unsloth/gemma-4-12b-it") -> None:
        self.model = _Model(model_key) if model_key else None
        self.running = model_key is not None
        self.stopped = 0
        self.swapped: list[str] = []

    def stop(self) -> None:
        self.stopped += 1
        self.running = False

    def swap_model(self, spec: str, *, on_status: Any = None) -> str:
        self.swapped.append(spec)
        self.model = _Model(spec)
        self.running = True
        return "http://127.0.0.1:4321"


class _Client:
    def __init__(self) -> None:
        self.base_urls: list[str] = []

    def set_base_url(self, url: str) -> None:
        self.base_urls.append(url)


class _Assistant:
    def __init__(self) -> None:
        self.cancelled = 0
        self.models: list[str] = []

    def cancel(self) -> None:
        self.cancelled += 1

    def set_model(self, model_id: str) -> None:
        self.models.append(model_id)


def _wait(q: Any, pred: Any, timeout: float = 5.0) -> list[Any]:
    """Drain `q` until an event satisfies `pred` (or `timeout`); returns everything drained."""
    seen: list[Any] = []
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for ev in drain(q):
            seen.append(ev)
            if pred(ev):
                return seen
        time.sleep(0.01)
    raise AssertionError(f"event not seen within {timeout}s; saw {seen}")


@pytest.fixture()
def eng(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> app_mod.Engine:
    cfg = Config()
    cfg.path = tmp_path / "config.toml"
    cfg.llm.library_dir = str(tmp_path / "llm")
    cfg.llm.model = "unsloth/gemma-4-12b-it"
    bus = EventBus()
    e = app_mod.Engine(cfg, bus)
    e.server = _Server()
    e.client = _Client()
    e.assistant = _Assistant()
    e._llm_ready.set()
    e._assistant_started = True          # as after a normal _init_llm
    from exocortex.llm.route import Route

    e.route = Route(backend="llamacpp", base_url="", reason="test")
    # a static hardware report: no probes
    report = hardware.HardwareReport(
        os="w", cpu_name="c", physical_cores=12, logical_cores=24, ram_total_mb=32000, ram_free_mb=16000,
        gpus=[{"name": "RTX", "vendor": "nvidia"}], primary_index=0, vendor="nvidia", budget_mb=16003,
        vram_total_mb=16003, vram_free_mb=15000, nvidia_driver="592.14", library_dir=str(tmp_path / "llm"),
        library_free_mb=100_000, runtime={"flavor": "nvidia-cuda12", "framework": "cuda"},
        recommended_policy="auto", generated_at="t", notes=[],
    )
    monkeypatch.setattr(hardware, "load", lambda path: report)
    monkeypatch.setattr(hardware, "collect", lambda cfg, **kw: report)
    monkeypatch.setattr(hardware, "save", lambda r, p: p)
    from exocortex import gpu

    monkeypatch.setattr(gpu, "primary_budget_mb", lambda gpus=None: 16003)
    return e


@pytest.fixture()
def lib(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """A fake library: two own models + one LM Studio model, a profile for the first."""
    models = [_Model("unsloth/gemma-4-12b-it"), _Model("empero-ai/Qwen3.8-9B-Distill", size_bytes=5_800_000_000),
              _Model("lmstudio-community/gemma-4-12B-it-QAT", source="lmstudio", quant="Q4_0")]
    calls: dict[str, list[Any]] = {"delete": [], "set_profile": [], "pull": [], "search": [], "info": []}
    profiles = {"unsloth/gemma-4-12b-it": library.ModelProfile(ctx_size=8192, note="tuned")}

    monkeypatch.setattr(library, "scan_library", lambda cfg, **kw: list(models))
    monkeypatch.setattr(library, "load_profiles", lambda cfg: dict(profiles))
    monkeypatch.setattr(library, "profile_for", lambda cfg, key: profiles.get(key))

    def set_profile(cfg: Any, key: str, profile: Any) -> None:
        calls["set_profile"].append((key, profile))
        if profile is None:
            profiles.pop(key, None)
        else:
            profiles[key] = profile

    monkeypatch.setattr(library, "set_profile", set_profile)

    def delete_model(model: Any) -> list[Path]:
        if model.source != "library":
            raise PermissionError(f"{model.key} belongs to LM Studio")
        calls["delete"].append(model.key)
        models[:] = [m for m in models if m.key != model.key]
        return [model.path]

    monkeypatch.setattr(library, "delete_model", delete_model)
    return {"models": models, "calls": calls, "profiles": profiles}


# ---- library / loaded / default ---------------------------------------------------------------
def test_model_library_merges_sources_and_marks_loaded_default_and_profiles(eng: app_mod.Engine, lib: dict) -> None:
    rows = eng.model_library()
    assert [r["key"] for r in rows] == ["unsloth/gemma-4-12b-it", "empero-ai/Qwen3.8-9B-Distill",
                                        "lmstudio-community/gemma-4-12B-it-QAT"]
    first = rows[0]
    assert first["loaded"] is True and first["is_default"] is True and first["has_profile"] is True
    assert first["context_length"] == 8192 and first["quant"] == "Q4_K_M" and first["source"] == "library"
    assert rows[1]["loaded"] is False and rows[1]["has_profile"] is False
    assert rows[2]["source"] == "lmstudio"
    assert eng.list_models() == [r["key"] for r in rows]


def test_loaded_model_follows_the_server(eng: app_mod.Engine) -> None:
    assert eng.loaded_model() == "unsloth/gemma-4-12b-it"
    eng.server.running = False
    assert eng.loaded_model() == ""
    eng.server = None
    assert eng.loaded_model() == ""


def test_load_model_swaps_and_announces(eng: app_mod.Engine) -> None:
    q = eng.bus.subscribe()
    eng.load_model("empero-ai/Qwen3.8-9B-Distill")
    seen = _wait(q, lambda ev: isinstance(ev, ModelLibraryEvent) and ev.reason == "loaded")
    assert eng.server.swapped == ["empero-ai/Qwen3.8-9B-Distill"]
    assert eng.client.base_urls == ["http://127.0.0.1:4321"]
    assert eng.assistant.models == ["empero-ai/Qwen3.8-9B-Distill"]
    assert seen[-1].loaded_key == "empero-ai/Qwen3.8-9B-Distill"
    assert eng.cfg.llm.model == "empero-ai/Qwen3.8-9B-Distill"


def test_unload_model_stops_the_server_and_says_so(eng: app_mod.Engine) -> None:
    q = eng.bus.subscribe()
    eng.unload_model()
    seen = _wait(q, lambda ev: isinstance(ev, ModelLibraryEvent) and ev.reason == "unloaded")
    assert eng.server.stopped == 1 and eng.assistant.cancelled == 1
    assert not eng._llm_ready.is_set()
    status = [ev for ev in seen if isinstance(ev, ComponentStatusEvent)]
    assert status and status[-1].component == "llm" and status[-1].state == "info" and "Models" in status[-1].detail
    assert eng.loaded_model() == ""
    # review round two, 2026-09-27 (E3): the unload thread must announce it is running -- stop()
    # can take ~10s, and the Models window otherwise learns nothing until it is already done.
    assert status[0].state == "loading" and "unloading" in status[0].detail
    assert "unsloth/gemma-4-12b-it" in status[0].detail


def test_set_default_model_persists(eng: app_mod.Engine, tmp_path: Path) -> None:
    q = eng.bus.subscribe()
    eng.set_default_model("empero-ai/Qwen3.8-9B-Distill")
    assert load_config(tmp_path / "config.toml").llm.model == "empero-ai/Qwen3.8-9B-Distill"
    evs = drain(q)
    assert any(isinstance(ev, ModelLibraryEvent) and ev.reason.startswith("default") for ev in evs)


# ---- delete ---------------------------------------------------------------------------------------
def test_delete_model_refuses_the_loaded_one(eng: app_mod.Engine, lib: dict) -> None:
    q = eng.bus.subscribe()
    eng.delete_model("unsloth/gemma-4-12b-it")
    evs = drain(q)
    assert lib["calls"]["delete"] == []
    assert any(isinstance(ev, ComponentStatusEvent) and ev.state == "error" and "unload" in ev.detail for ev in evs)


def test_delete_model_refuses_lm_studio_files(eng: app_mod.Engine, lib: dict) -> None:
    q = eng.bus.subscribe()
    eng.delete_model("lmstudio-community/gemma-4-12B-it-QAT")
    evs = drain(q)
    assert lib["calls"]["delete"] == []
    assert any(isinstance(ev, ComponentStatusEvent) and ev.state == "error" and "LM Studio" in ev.detail for ev in evs)


def test_delete_model_removes_files_and_profile(eng: app_mod.Engine, lib: dict) -> None:
    q = eng.bus.subscribe()
    eng.delete_model("empero-ai/Qwen3.8-9B-Distill")
    evs = drain(q)
    assert lib["calls"]["delete"] == ["empero-ai/Qwen3.8-9B-Distill"]
    assert lib["calls"]["set_profile"] == [("empero-ai/Qwen3.8-9B-Distill", None)]
    assert any(isinstance(ev, ModelLibraryEvent) and "deleted" in ev.reason for ev in evs)
    assert len(lib["models"]) == 2


# ---- pull -----------------------------------------------------------------------------------------
def _fake_pull(calls: dict[str, list[Any]], *, model: _Model | None = None, fail: Exception | None = None,
               block: threading.Event | None = None) -> Any:
    def pull_model(repo_id: str, quant: str, cfg: Any, *, include_mmproj: bool = False, policy: str = "auto",
                   budget_mb: int | None = None, progress: Any = None, cancelled: Any = None, http: Any = None) -> Any:
        calls["pull"].append((repo_id, quant, include_mmproj, policy, budget_mb))
        progress("file-00001-of-00002.gguf", 10, 100)
        if block is not None:
            block.wait(5.0)
        if cancelled is not None and cancelled():
            raise library.PullCancelled("cancelled")
        progress("file-00002-of-00002.gguf", 100, 100)
        if fail is not None:
            raise fail
        return model or _Model(repo_id.replace("-GGUF", ""))

    return pull_model


def test_pull_model_reports_progress_done_and_sets_the_first_default(eng: app_mod.Engine, lib: dict,
                                                                     monkeypatch: pytest.MonkeyPatch,
                                                                     tmp_path: Path) -> None:
    monkeypatch.setattr(library, "pull_model", _fake_pull(lib["calls"]))
    eng.cfg.llm.model = "auto"
    q = eng.bus.subscribe()
    eng.pull_model("unsloth/gemma-4-E2B-it-GGUF")
    seen = _wait(q, lambda ev: isinstance(ev, ModelLibraryEvent) and ev.reason.startswith("pulled"))
    states = [ev.state for ev in seen if isinstance(ev, ModelPullEvent)]
    assert states[0] == "queued" and "downloading" in states and states[-1] == "done"
    done = [ev for ev in seen if isinstance(ev, ModelPullEvent) and ev.state == "done"][0]
    assert done.model_key == "unsloth/gemma-4-E2B-it" and done.total_bytes == 7_000_000_000
    assert lib["calls"]["pull"] == [("unsloth/gemma-4-E2B-it-GGUF", "auto", False, "auto", 16003)]
    eng._pull_thread.join(5.0)
    assert eng.cfg.llm.model == "unsloth/gemma-4-E2B-it"
    assert load_config(tmp_path / "config.toml").llm.model == "unsloth/gemma-4-E2B-it"


def test_pull_model_keeps_an_explicit_loaded_default(eng: app_mod.Engine, lib: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(library, "pull_model", _fake_pull(lib["calls"]))
    q = eng.bus.subscribe()
    eng.pull_model("empero-ai/Qwen3.8-4B-Distill-GGUF", "Q6_K", include_mmproj=True)
    _wait(q, lambda ev: isinstance(ev, ModelPullEvent) and ev.state == "done")
    eng._pull_thread.join(5.0)
    assert lib["calls"]["pull"][0][:3] == ("empero-ai/Qwen3.8-4B-Distill-GGUF", "Q6_K", True)
    assert eng.cfg.llm.model == "unsloth/gemma-4-12b-it"      # the loaded default stays


def test_pull_model_error_and_second_pull_refused(eng: app_mod.Engine, lib: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    gate = threading.Event()
    monkeypatch.setattr(library, "pull_model", _fake_pull(lib["calls"], block=gate, fail=RuntimeError("disk full")))
    q = eng.bus.subscribe()
    eng.pull_model("a/b-GGUF")
    _wait(q, lambda ev: isinstance(ev, ModelPullEvent) and ev.state == "downloading")
    eng.pull_model("c/d-GGUF")
    seen = _wait(q, lambda ev: isinstance(ev, ModelPullEvent) and ev.repo_id == "c/d-GGUF")
    refused = [ev for ev in seen if isinstance(ev, ModelPullEvent) and ev.repo_id == "c/d-GGUF"][0]
    assert refused.state == "error" and "another pull" in refused.detail
    gate.set()
    seen = _wait(q, lambda ev: isinstance(ev, ModelPullEvent) and ev.state == "error" and ev.repo_id == "a/b-GGUF")
    assert "disk full" in seen[-1].detail
    assert len(lib["calls"]["pull"]) == 1


def test_cancel_pull_and_stop_join_the_thread(eng: app_mod.Engine, lib: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    gate = threading.Event()
    monkeypatch.setattr(library, "pull_model", _fake_pull(lib["calls"], block=gate))
    q = eng.bus.subscribe()
    eng.pull_model("a/b-GGUF")
    _wait(q, lambda ev: isinstance(ev, ModelPullEvent) and ev.state == "downloading")
    eng.cancel_pull()
    gate.set()
    seen = _wait(q, lambda ev: isinstance(ev, ModelPullEvent) and ev.state == "cancelled")
    assert seen[-1].detail == "cancelled"
    eng._pull_thread.join(5.0)
    # stop() with a pull in flight sets the flag and joins.
    gate2 = threading.Event()
    monkeypatch.setattr(library, "pull_model", _fake_pull(lib["calls"], block=gate2))
    eng.pull_model("a/b-GGUF")
    _wait(q, lambda ev: isinstance(ev, ModelPullEvent) and ev.state == "downloading")
    gate2.set()
    eng.stop()
    assert eng._pull_cancel.is_set()
    assert not eng._pull_thread.is_alive()


# ---- search / inspect -----------------------------------------------------------------------
def test_search_models_answers_with_the_request_id(eng: app_mod.Engine, lib: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(library, "hf_search", lambda query, limit=25: [library.HfRepo("a/b", 5, 1, "2026", False, False)])
    q = eng.bus.subscribe()
    eng.search_models("b", 7)
    seen = _wait(q, lambda ev: isinstance(ev, ModelSearchEvent))
    ev = seen[-1]
    assert ev.request_id == 7 and ev.query == "b" and ev.results[0]["id"] == "a/b" and ev.error == ""

    def boom(query: str, limit: int = 25) -> list:
        raise ConnectionError("offline")

    monkeypatch.setattr(library, "hf_search", boom)
    eng.search_models("b", 8)
    seen = _wait(q, lambda ev: isinstance(ev, ModelSearchEvent) and ev.request_id == 8)
    assert seen[-1].error == "offline" and seen[-1].results == ()


def test_inspect_model_lists_quants_and_the_fit(eng: app_mod.Engine, lib: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    files = (library.HfFile("a-Q4_K_M.gguf", 7_000_000_000, "ab"),)
    quants = (library.QuantOption("Q4_K_M", files, 7_000_000_000),
              library.QuantOption("Q8_0", files, 13_000_000_000))
    info = library.HfRepoInfo(id="a/b", sha="s", files=files, architecture="gemma4", params=12_000_000_000,
                              context_length=262144, quants=quants, mmproj=(library.HfFile("mmproj.gguf", 1, ""),))
    monkeypatch.setattr(library, "hf_repo_info", lambda repo_id: info)
    monkeypatch.setattr(library, "choose_quant", lambda options, *, budget_mb, policy="auto": options[0])
    q = eng.bus.subscribe()
    eng.inspect_model("a/b", 3)
    seen = _wait(q, lambda ev: isinstance(ev, ModelInfoEvent))
    ev = seen[-1]
    assert ev.request_id == 3 and ev.chosen == "Q4_K_M" and ev.mmproj is True and ev.architecture == "gemma4"
    assert [(d["quant"], d["fits"]) for d in ev.quants] == [("Q4_K_M", True), ("Q8_0", False)]


# ---- hardware / profiles ----------------------------------------------------------------------
def test_hardware_report_uses_the_saved_report_unless_refreshed(eng: app_mod.Engine, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"collect": 0}
    original = hardware.collect

    def counting(cfg: Any, **kw: Any) -> Any:
        calls["collect"] += 1
        return original(cfg, **kw)

    monkeypatch.setattr(hardware, "collect", counting)
    assert eng.hardware_report()["budget_mb"] == 16003 and calls["collect"] == 0
    assert eng.hardware_report(refresh=True)["budget_mb"] == 16003 and calls["collect"] == 1


def test_model_profile_round_trip_and_reload_of_the_loaded_model(eng: app_mod.Engine, lib: dict) -> None:
    assert eng.model_profile("unsloth/gemma-4-12b-it")["ctx_size"] == 8192
    assert eng.model_profile("nope") is None
    q = eng.bus.subscribe()
    eng.set_model_profile("unsloth/gemma-4-12b-it", {"ctx_size": 4096, "kv_cache_type": "f16", "bogus": 1})
    seen = _wait(q, lambda ev: isinstance(ev, ModelLibraryEvent) and ev.reason == "loaded")
    key, profile = lib["calls"]["set_profile"][-1]
    assert key == "unsloth/gemma-4-12b-it" and profile.ctx_size == 4096 and profile.kv_cache_type == "f16"
    assert profile.auto is False and profile.gpu_layers is None
    assert eng.server.swapped == ["unsloth/gemma-4-12b-it"]        # relaunched with the new flags
    assert any(isinstance(ev, ModelLibraryEvent) and ev.reason.startswith("profile") for ev in seen)
    eng.set_model_profile("empero-ai/Qwen3.8-9B-Distill", None)
    assert lib["calls"]["set_profile"][-1] == ("empero-ai/Qwen3.8-9B-Distill", None)
    assert eng.server.swapped == ["unsloth/gemma-4-12b-it"]        # not the loaded one: no reload


def test_set_model_profile_coerces_text_values_from_the_tuning_tab(eng: app_mod.Engine, lib: dict) -> None:
    """E4 (round two, 2026-09-27): the Tuning tab's widgets round-trip every value as a str (a
    literal "[]" for extra_args when there are none); set_model_profile must coerce those instead
    of handing them straight to `library.ModelProfile(**values)`, where a str extra_args later
    explodes character-by-character into llama-server's argv (`list(profile.extra_args)`)."""
    q = eng.bus.subscribe()
    eng.set_model_profile("empero-ai/Qwen3.8-9B-Distill", {
        "ctx_size": "8192", "temperature": "0.7", "top_k": "not-a-number",
        "extra_args": "--foo bar --baz",
    })
    _wait(q, lambda ev: isinstance(ev, ModelLibraryEvent) and ev.reason.startswith("profile"))
    _, profile = lib["calls"]["set_profile"][-1]
    assert profile.ctx_size == 8192 and isinstance(profile.ctx_size, int)
    assert profile.temperature == 0.7 and isinstance(profile.temperature, float)
    assert profile.top_k is None                          # unparsable -> dropped, not stored as text
    assert profile.extra_args == ["--foo", "bar", "--baz"]

    eng.set_model_profile("empero-ai/Qwen3.8-9B-Distill", {"extra_args": "[]"})
    assert lib["calls"]["set_profile"][-1][1].extra_args == []

    eng.set_model_profile("empero-ai/Qwen3.8-9B-Distill", {"extra_args": ["--a", 1]})
    assert lib["calls"]["set_profile"][-1][1].extra_args == ["--a", "1"]


def test_auto_tune_model_uses_the_hardware_report(eng: app_mod.Engine, lib: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    seen_kwargs: dict[str, Any] = {}

    def auto_profile(model: Any, **kw: Any) -> library.ModelProfile:
        seen_kwargs.update(kw)
        return library.ModelProfile(ctx_size=16384, note="auto")

    monkeypatch.setattr(library, "auto_profile", auto_profile)
    out = eng.auto_tune_model("empero-ai/Qwen3.8-9B-Distill")
    assert out["ctx_size"] == 16384 and out["note"] == "auto"
    assert seen_kwargs["budget_mb"] == 16003 and seen_kwargs["ram_total_mb"] == 32000
    assert seen_kwargs["physical_cores"] == 12 and seen_kwargs["gpu_framework"] == "cuda"
    assert lib["calls"]["set_profile"][-1][0] == "empero-ai/Qwen3.8-9B-Distill"
    assert eng.auto_tune_model("nope") is None


def test_start_publishes_empty_library_for_a_fresh_machine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Route resolved to llamacpp/none with no model anywhere -> the overlay is told to open Models."""
    from exocortex.llm.route import Route

    cfg = Config()
    cfg.path = tmp_path / "config.toml"
    cfg.llm.library_dir = str(tmp_path / "llm")
    bus = EventBus()
    q = bus.subscribe()
    monkeypatch.setattr(library, "scan_library", lambda cfg, **kw: [])
    route = Route(backend="none", base_url="", reason="no model", model_count=0)
    eng = app_mod.Engine(cfg, bus, route_resolver=lambda llm_cfg, **kw: route)
    # Keep start() from touching real components: stop before the threads do anything heavy.
    monkeypatch.setattr(eng, "_init_llm", lambda: None)
    monkeypatch.setattr(eng, "_init_speech", lambda: None)
    monkeypatch.setattr(eng, "_init_screen", lambda: None)
    monkeypatch.setattr(eng, "_start_gpu_guard", lambda: None)
    monkeypatch.setattr(app_mod.Engine, "_preload_native", staticmethod(lambda: None))
    try:
        eng.start()
        evs = drain(q)
    finally:
        eng.stop()
    assert any(isinstance(ev, ModelLibraryEvent) and ev.reason == "empty-library" for ev in evs)

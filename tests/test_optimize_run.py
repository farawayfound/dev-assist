"""Tests for exocortex.optimize_run.OptimizeController (2026-10-07) against a fake engine and a
stubbed exocortex.optimize_bench: no GPU, no network, no child processes, no real llama-server --
a fixture makes LlamaServerManager's constructor raise so a benchmark can never build one. The
store writes under tmp_path; the config never saves (cfg.path points into tmp_path)."""
from __future__ import annotations

import dataclasses
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from exocortex import optimize as O
from exocortex import optimize_bench as ob
from exocortex import optimize_run as R
from exocortex.bus import EventBus, OptimizeEvent, drain
from exocortex.config import Config
from exocortex.llm import library


# ---- fakes ----------------------------------------------------------------------------------



@dataclasses.dataclass
class _Meta:
    architecture: str = "qwen35"
    n_layer: int = 65
    n_ctx_train: int = 262144
    n_head_kv: int | None = 4
    head_dim: int | None = 256
    kv_heads_total: int | None = 260
    full_attention_interval: int | None = 4
    nextn_predict_layers: int | None = 1
    ssm_conv_kernel: int | None = 4
    ssm_state_size: int | None = 128
    ssm_inner_size: int | None = 6144
    ssm_group_count: int | None = 16


@dataclasses.dataclass
class _Model:
    key: str
    size_bytes: int = 16_810_714_336
    source: str = "lmstudio"
    meta: Any = dataclasses.field(default_factory=_Meta)
    path: Path = Path("X:/lib/q.gguf")


class _Server:
    def __init__(self, key: str = "lmstudio-community/Qwen3.8-27B") -> None:
        self.model = _Model(key)
        self.running = True
        self.base_url = "http://127.0.0.1:5555"
        self.relaunched: list[str] = []


class _Client:
    def __init__(self) -> None:
        self.n_ctx: int | None = 32768
        self.base_url = "http://127.0.0.1:5555"

    def context_window(self) -> int | None:
        return self.n_ctx


class _Assistant:
    _current_job = None

    def cancel(self) -> None:
        pass


class _Segment:
    def __init__(self, t_end: float) -> None:
        self.t_end = t_end


class _Transcript:
    def __init__(self) -> None:
        self.segments: list[_Segment] = []

    def last(self, source: Any = None, n: int = 1) -> list[_Segment]:
        return self.segments[-n:]


class _Transcriber:
    engine_used = "faster-whisper"
    device_used = "cpu"
    compute_type_used = "int8"


class FakeEngine:
    """The duck-typed surface OptimizeController uses, with recorders."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.bus = EventBus()
        self.server: Any = _Server()
        self.client = _Client()
        self.assistant = _Assistant()
        self.transcriber: Any = _Transcriber()
        self.pipeline = None
        self.transcript = _Transcript()
        self._llm_ready = threading.Event()
        self._llm_ready.set()
        self._swap_thread: threading.Thread | None = None
        self._stopped = threading.Event()
        self._paused = False
        self.handsfree = None
        self.route = None
        self.statuses: list[tuple[str, str, str]] = []
        self.relaunch_errors: list[str | None] = []     # queue of results for _relaunch_blocking
        self.relaunched: list[str] = []
        self.swap_refuse = False
        self._t = 100.0

    def clock(self) -> float:
        return self._t

    def hardware_report(self, refresh: bool = False) -> dict[str, object]:
        return {"cpu_name": "Intel Core Ultra 9 285H", "physical_cores": 16, "ram_total_mb": 64957, "ram_free_mb": 30000,
                "vram_total_mb": 24463, "vram_free_mb": 22400, "nvidia_driver": "610.47",
                "gpus": [{"name": "NVIDIA GeForce RTX 5090 Laptop GPU", "vendor": "nvidia"}], "primary_index": 0,
                "runtime": {"flavor": "nvidia-cuda12", "version": "2.44.0"}}

    def loaded_model(self) -> str:
        return self.server.model.key if self.server is not None and self.server.running else ""

    def _status(self, component: str, state: str, detail: str = "") -> None:
        self.statuses.append((component, state, detail))

    def _start_swap(self, work: Any, what: str) -> bool:
        if self.swap_refuse or (self._swap_thread is not None and self._swap_thread.is_alive()):
            return False
        self._swap_thread = threading.Thread(target=work, name="llm-swap", daemon=True)
        self._swap_thread.start()
        return True

    def _relaunch_blocking(self, model_id: str) -> str | None:
        self.relaunched.append(model_id)
        return self.relaunch_errors.pop(0) if self.relaunch_errors else None


class FakeBench:
    """Stand-in for the functions OptimizeController reaches through ``exocortex.optimize_bench``;
    the real Store is kept (it only writes under tmp_path)."""

    Store = ob.Store
    ASR_VARIANTS = ob.ASR_VARIANTS
    fingerprint = staticmethod(ob.fingerprint)
    iso_now = staticmethod(ob.iso_now)

    def __init__(self, gpu_state: dict[str, Any] | None = None) -> None:
        self.gpu = gpu_state if gpu_state is not None else {"free": 22400}
        self.free_mb: int | None = None          # None = follow the gpu fake, like the real sampler does
        self.probe_result: dict = {"pp": 900.0, "tg": 20.0, "cached_ttft_ms": 500.0, "incr_prompt_n": 950, "incr_pp": 800.0,
                                   "spread": 0.03, "noisy": False, "n_ctx": 32768, "slot": 1, "rounds": [], "error": None}
        self.probe_calls: list[dict] = []
        self.children: list[list[str]] = []
        self.child_results: dict[str, dict] = {}
        self.canary_ok = True
        self.swap = False
        self.probe_exc: Exception | None = None

    def sample_vram(self, n: int = 3, gap_s: float = 1.0, read: Any = None, sleep: Any = None) -> tuple[int | None, bool]:
        return (self.free_mb if self.free_mb is not None else int(self.gpu["free"])), True

    def swap_resident(self, url: str, timeout_s: float = 1.5, http_get: Any = None) -> bool | None:
        return self.swap

    def canary(self, base_url: str, timeout_s: float = 10.0, http: Any = None) -> tuple[bool, float]:
        return self.canary_ok, 12.0

    def probe_llm(self, base_url: str, **kw: Any) -> dict:
        self.probe_calls.append({"base_url": base_url, **kw})
        if self.probe_exc is not None:
            raise self.probe_exc
        if kw.get("progress"):
            kw["progress"]("round 1", 0.5)
        return dict(self.probe_result)

    def run_child(self, argv_tail: list[str], *, out_path: Path, deadline_s: float, cancel: Any = None,
                  still_ok: Any = None, procs: Any = None, python: Any = None, popen: Any = None, cwd: Any = None) -> dict:
        self.children.append(list(argv_tail))
        if argv_tail[0] == "probe-devices":
            return {"openvino": {"CPU": "x", "GPU.0": "Intel Arc (iGPU)", "NPU": "AI Boost"}, "igpu": "Intel Arc (iGPU)",
                    "npu": "AI Boost", "ct2_cuda": True, "sherpa_ready": True, "mem_mts": 7467, "power": None}
        variant = argv_tail[argv_tail.index("--variant") + 1]
        if still_ok is not None and not still_ok():
            return {"error": "cancelled"}
        return self.child_results.get(variant, {"label": variant, "engine": "sherpa" if "sherpa" in variant else "faster-whisper",
                                                "device": "cpu" if "cpu" in variant else "cuda", "compute_type": "int8",
                                                "threads": 4, "load_s": 2.0, "median_s": 0.2, "max_s": 0.3, "overlap": 0.99,
                                                "vram_mb": 0 if "cpu" in variant else 874, "vram_source": "nvml", "error": None})


@pytest.fixture(autouse=True)
def _never_a_real_server(monkeypatch: pytest.MonkeyPatch) -> None:
    from exocortex.llm import server as srv

    def _boom(*a: Any, **k: Any) -> None:
        raise AssertionError("a benchmark must never build a llama-server")

    monkeypatch.setattr(srv.LlamaServerManager, "__init__", _boom)
    monkeypatch.setattr(srv.LlamaServerManager, "start", _boom)
    monkeypatch.setattr(srv.LlamaServerManager, "swap_model", _boom)


@pytest.fixture()
def gpu_fake(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    from exocortex import gpu

    state = {"total": 24463, "free": 22400}
    monkeypatch.setattr(gpu, "vram_total_mb", lambda: state["total"])
    monkeypatch.setattr(gpu, "vram_free_mb", lambda: state["free"])
    monkeypatch.setattr(gpu, "power_state", lambda: None)
    return state


@pytest.fixture()
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, gpu_fake: dict[str, Any]):
    cfg = Config()
    cfg.path = tmp_path / "config.toml"
    cfg.llm.library_dir = str(tmp_path / "llm")
    cfg.llm.model = "lmstudio-community/Qwen3.8-27B"
    cfg.llm.swap_unload_url = "http://127.0.0.1:8081"
    (tmp_path / "llm").mkdir()
    eng = FakeEngine(cfg)
    bench = FakeBench(gpu_fake)
    models = [_Model("lmstudio-community/Qwen3.8-27B"), _Model("deepgrove/maple-preview", size_bytes=4_980_000_000, meta=_Meta(
        architecture="maple", n_layer=24, n_ctx_train=131072, n_head_kv=4, head_dim=128, kv_heads_total=96,
        full_attention_interval=None, nextn_predict_layers=None, ssm_inner_size=None))]
    profiles: dict[str, library.ModelProfile] = {"lmstudio-community/Qwen3.8-27B": library.ModelProfile(ctx_size=32768, auto=False, temperature=0.5)}
    monkeypatch.setattr(library, "scan_library", lambda cfg, **kw: list(models))
    monkeypatch.setattr(library, "profile_for", lambda cfg, key: profiles.get(key))

    def set_profile(cfg_: Any, key: str, profile: Any) -> None:
        if profile is None:
            profiles.pop(key, None)
        else:
            profiles[key] = profile

    monkeypatch.setattr(library, "set_profile", set_profile)
    monkeypatch.setattr(R.OptimizeController, "_bench", lambda self: bench)
    monkeypatch.setattr(R, "CLIP", tmp_path / "clip.wav")
    (tmp_path / "clip.wav").write_bytes(b"RIFF")
    ctl = R.OptimizeController(eng, store=ob.Store(tmp_path / "llm"), now=lambda: 1_000_000.0)
    ctl._baseline = (24463, 2063)
    return {"cfg": cfg, "eng": eng, "bench": bench, "ctl": ctl, "profiles": profiles, "tmp": tmp_path, "gpu": gpu_fake}


def _join(ctl: R.OptimizeController, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    threads = list(ctl._threads)
    swap = getattr(ctl._engine, "_swap_thread", None)
    if swap is not None:
        threads.append(swap)
    for t in threads:
        t.join(timeout=max(0.0, deadline - time.monotonic()))
    assert not any(t.is_alive() for t in threads), "a worker thread did not finish"


def _events(bus: EventBus, q: Any) -> list[OptimizeEvent]:
    return [e for e in drain(q) if isinstance(e, OptimizeEvent)]


# ---- snapshot / analyze ----------------------------------------------------------------------
def test_snapshot_before_anything_is_plain_and_never_raises(world: dict) -> None:
    snap = world["ctl"].snapshot()
    assert snap["analysis"]["state"] == "idle" and snap["plans"] == {} and snap["run"]["state"] == "idle"
    world["ctl"]._snap = None  # type: ignore[assignment]
    assert world["ctl"].snapshot() == {}


def test_analyze_fills_every_block_and_caches_the_plans(world: dict) -> None:
    ctl, eng = world["ctl"], world["eng"]
    q = eng.bus.subscribe()
    ctl.analyze("context", {"reserve_vram_mb": 3000, "ctx_cap": 0, "asr_gpu": "auto"}, "")
    _join(ctl)
    snap = ctl.snapshot()
    assert snap["analysis"]["state"] == "done" and set(snap["plans"]) == set(O.GOALS) and snap["selected"] == "context"
    facts = snap["facts"]
    assert facts["cpu"].startswith("Intel") and facts["gpu"]["total_mb"] == 24463 and facts["vram"]["others_mb"] == 2063
    assert facts["igpu"] and facts["npu"] and facts["devices_known"] and facts["runtime"] == "nvidia-cuda12 2.44.0"
    assert [m["key"] for m in snap["models"]] == ["lmstudio-community/Qwen3.8-27B", "deepgrove/maple-preview"]
    assert snap["engine"] == {"ready": True, "backend": "llamacpp", "model": "lmstudio-community/Qwen3.8-27B", "swap_resident": False}
    assert snap["gates"]["llm"] == [True, ""] and snap["gates"]["apply"] == [True, ""]
    plan = snap["plans"]["context"]
    assert plan["vram"]["reserve_mb"] == 3000                      # the UNSAVED option reached the planner
    assert plan["digest"] in ctl._plans
    kinds = [e.kind for e in _events(eng.bus, q)]
    assert "analysis" in kinds
    # the device probe child ran once and is cached afterwards
    assert world["bench"].children.count(["probe-devices", "--models-dir", str(world["cfg"].models_dir()), "--sherpa-model", world["cfg"].asr.sherpa_model]) == 1
    ctl.analyze("latency", None, "")
    _join(ctl)
    assert sum(1 for c in world["bench"].children if c[0] == "probe-devices") == 1
    assert ctl.snapshot()["selected"] == "latency"


def test_analyze_for_another_model_and_with_a_foreign_backend(world: dict) -> None:
    ctl, eng = world["ctl"], world["eng"]
    ctl.analyze("balanced", None, "deepgrove/maple-preview")
    _join(ctl)
    plan = ctl.snapshot()["plans"]["balanced"]
    assert plan["model_key"] == "deepgrove/maple-preview"
    eng.route = type("R", (), {"backend": "ollama"})()
    ctl.analyze("balanced", None, "")
    _join(ctl)
    snap = ctl.snapshot()
    assert snap["engine"]["backend"] == "ollama" and snap["gates"]["llm"][0] is False


def test_analysis_failure_is_reported_not_raised(world: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    ctl = world["ctl"]
    monkeypatch.setattr(ctl, "_gather", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    ctl.analyze("balanced", None, "")
    _join(ctl)
    snap = ctl.snapshot()
    assert snap["analysis"]["state"] == "error" and "boom" in snap["analysis"]["message"]
    assert not ctl._analyze_lock.locked()


def test_activity_reads_engine_state_defensively(world: dict) -> None:
    ctl, eng = world["ctl"], world["eng"]
    act = ctl._activity()
    assert act.llm_ready and act.server_running and act.transcriber_ready and act.speech_idle_s > 100 and act.swap_resident is False
    eng.transcript.segments.append(_Segment(t_end=99.0))
    assert ctl._activity().speech_idle_s == pytest.approx(1.0)
    eng.assistant._current_job = object()
    assert ctl._activity().assist_busy
    eng.server = None
    eng.transcriber = None
    act = ctl._activity()
    assert not act.server_running and not act.transcriber_ready
    assert O.gate("llm", act)[0] is False


def test_a_getattr_catch_all_fake_is_not_optimize_capable() -> None:
    from tests.test_overlay_completion import FakeActions

    assert not callable(getattr(type(FakeActions()), "optimize_snapshot", None))
    from exocortex.app import Engine

    assert callable(getattr(Engine, "optimize_snapshot", None))


# ---- benchmark -------------------------------------------------------------------------------
def test_benchmark_probes_the_running_server_and_speech_then_replans(world: dict) -> None:
    ctl, eng, bench = world["ctl"], world["eng"], world["bench"]
    world["gpu"]["free"] = 24463 - 2063 - 18000           # the server holds 18,000 MiB
    q = eng.bus.subscribe()
    assert ctl.benchmark() is True
    assert ctl.benchmark() is False                        # single flight
    _join(ctl)
    snap = ctl.snapshot()
    assert snap["run"]["state"] == "done" and snap["run"]["fraction"] == 1.0
    assert bench.probe_calls and bench.probe_calls[0]["base_url"] == "http://127.0.0.1:5555" and bench.probe_calls[0]["n_ctx"] == 32768
    variants = [c[c.index("--variant") + 1] for c in bench.children if c[0] == "probe-asr"]
    assert variants == ["whisper-cpu-int8", "sherpa-cpu", "whisper-cuda-int8_float16"]
    ev = ctl._get_store().evidence(None, 1_000_000.0)
    assert len(ev.llm) == 1 and ev.llm[0].source == "probe" and ev.llm[0].pp == 900.0 and ev.llm[0].gpu_used_mb == 18000
    assert len(ev.footprints) == 1 and ev.footprints[0].app_mb == 18000
    assert sorted(r.label for r in ev.asr) == ["sherpa-cpu", "whisper-cpu-int8", "whisper-cuda-int8_float16"]
    assert snap["measured"]["llm"]["pp"] == 900.0 and len(snap["measured"]["asr"]) == 3
    assert snap["plans"]["balanced"]["vram"]["tier"] in ("calibrated", "fit", "header")
    kinds = [e.kind for e in _events(eng.bus, q)]
    assert kinds[0] == "started" and "progress" in kinds and kinds[-2:] == ["done", "analysis"]
    assert not ctl._busy.locked()


def test_benchmark_is_refused_by_the_gates_and_reports_why(world: dict) -> None:
    ctl, eng = world["ctl"], world["eng"]
    eng.assistant._current_job = object()
    assert ctl.benchmark(["llm"]) is False
    assert "answer is being generated" in ctl.snapshot()["run"]["message"]
    eng.assistant._current_job = None
    eng.transcript.segments.append(_Segment(t_end=99.5))
    assert ctl.benchmark(["llm"]) is False
    assert ctl.benchmark(["llm"], force=True) is True
    _join(ctl)
    assert ctl.snapshot()["run"]["state"] == "done"


def test_probe_exception_releases_busy_and_marks_the_run_partial(world: dict) -> None:
    ctl, bench = world["ctl"], world["bench"]
    bench.probe_exc = RuntimeError("socket closed")
    assert ctl.benchmark(["llm"]) is True
    _join(ctl)
    snap = ctl.snapshot()
    assert snap["run"]["state"] == "done" and "skipped" in snap["run"]["message"]
    assert not ctl._busy.locked()
    assert ctl.benchmark(["llm"]) is True
    _join(ctl)


def test_speech_step_skips_the_cuda_variant_without_room_and_the_clip(world: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    ctl, bench = world["ctl"], world["bench"]
    bench.free_mb = 1800
    world["gpu"]["free"] = 1800
    assert ctl.benchmark(["speech"]) is True
    _join(ctl)
    variants = [c[c.index("--variant") + 1] for c in bench.children if c[0] == "probe-asr"]
    assert "whisper-cuda-int8_float16" not in variants and "sherpa-cpu" in variants
    bench.children.clear()
    monkeypatch.setattr(R, "CLIP", world["tmp"] / "missing.wav")
    assert ctl.benchmark(["speech"]) is True
    _join(ctl)
    assert not [c for c in bench.children if c[0] == "probe-asr"]
    assert "clip" in ctl.snapshot()["run"]["message"] or ctl.snapshot()["run"]["state"] == "done"


def test_cancel_and_shutdown_stop_the_run(world: dict) -> None:
    ctl, bench = world["ctl"], world["bench"]
    gate = threading.Event()

    def slow_probe(base_url: str, **kw: Any) -> dict:
        gate.wait(5.0)
        return {"error": "cancelled"}

    bench.probe_llm = slow_probe  # type: ignore[assignment]
    assert ctl.benchmark(["llm"]) is True
    ctl.cancel()
    gate.set()
    ctl.shutdown()
    assert not any(t.is_alive() for t in ctl._threads)
    assert ctl.snapshot()["run"]["state"] in ("cancelled", "done")


# ---- apply -----------------------------------------------------------------------------------
def _analysed(world: dict, goal: str = "context") -> dict:
    ctl = world["ctl"]
    ctl.analyze(goal, None, "")
    _join(ctl)
    return ctl.snapshot()["plans"][goal]


def test_apply_success_journals_merges_relaunches_and_verifies(world: dict) -> None:
    ctl, eng, profiles = world["ctl"], world["eng"], world["profiles"]
    world["gpu"]["free"] = 24463 - 2063 - 17500           # the server holds 17,500 MiB before and after
    plan = _analysed(world, "context")
    assert plan["profile"] and plan["profile"]["ctx_size"] > 32768
    eng.client.n_ctx = plan["profile"]["ctx_size"]
    q = eng.bus.subscribe()
    assert ctl.apply(plan["digest"]) is True, ctl.snapshot()["apply"]
    _join(ctl)
    assert eng.relaunched == ["lmstudio-community/Qwen3.8-27B"]
    prof = profiles["lmstudio-community/Qwen3.8-27B"]
    assert prof.ctx_size == plan["profile"]["ctx_size"] and prof.temperature == 0.5 and prof.auto is False   # sampling override kept
    journal = ctl._get_store().journal()
    assert [e["state"] for e in journal] == ["verified"] and journal[0]["previous"]["ctx_size"] == 32768
    snap = ctl.snapshot()
    assert snap["apply"]["state"] == "verified" and snap["apply"]["undo"] is True
    assert "applied" in [e.kind for e in _events(eng.bus, q)]
    ev = ctl._get_store().evidence(None, 1_000_000.0)
    assert any(f.source == "apply" for f in ev.footprints)
    assert not ctl._busy.locked()


@pytest.mark.parametrize("failure", ["swap", "ctx", "vram", "canary"])
def test_apply_failures_roll_back_and_relaunch_the_old_profile(world: dict, failure: str) -> None:
    ctl, eng, bench, profiles = world["ctl"], world["eng"], world["bench"], world["profiles"]
    plan = _analysed(world, "context")
    eng.client.n_ctx = plan["profile"]["ctx_size"]
    if failure == "swap":
        eng.relaunch_errors = ["CUDA out of memory"]
    elif failure == "ctx":
        eng.client.n_ctx = 4096
    elif failure == "vram":
        bench.free_mb = 900
    elif failure == "canary":
        bench.canary_ok = False
    assert ctl.apply(plan["digest"]) is True, ctl.snapshot()["apply"]
    _join(ctl)
    assert eng.relaunched == ["lmstudio-community/Qwen3.8-27B"] * 2          # apply, then the restore
    assert profiles["lmstudio-community/Qwen3.8-27B"].ctx_size == 32768        # the previous profile is back
    journal = ctl._get_store().journal()
    assert journal[-1]["state"] == "rolled-back" and journal[-1]["detail"]
    snap = ctl.snapshot()
    assert snap["apply"]["state"] == "rolled-back" and snap["apply"]["undo"] is False
    assert any(s[0] == "app" and s[1] == "error" for s in eng.statuses)
    assert not ctl._busy.locked()


def test_apply_refusals_write_nothing(world: dict) -> None:
    ctl, eng, bench = world["ctl"], world["eng"], world["bench"]
    plan = _analysed(world, "context")
    assert ctl.apply("not-a-digest") is False and "analyse again" in ctl.snapshot()["apply"]["message"]
    eng.assistant._current_job = object()
    assert ctl.apply(plan["digest"]) is False
    eng.assistant._current_job = None
    bench.swap = True
    assert ctl.apply(plan["digest"]) is False and "shared model server" in ctl.snapshot()["apply"]["message"]
    bench.swap = False
    eng.swap_refuse = True
    assert ctl.apply(plan["digest"]) is False and "still running" in ctl.snapshot()["apply"]["message"]
    eng.swap_refuse = False
    assert ctl._get_store().journal() == [] and eng.relaunched == []
    assert not ctl._busy.locked()
    # a stale plan (older than 15 minutes) is refused too
    ctl._now = lambda: 1_000_000.0 + 2000
    assert ctl.apply(plan["digest"]) is False and "15 minutes" in ctl.snapshot()["apply"]["message"]


def test_apply_to_a_model_that_is_not_loaded_only_saves(world: dict) -> None:
    ctl, eng, profiles = world["ctl"], world["eng"], world["profiles"]
    ctl.analyze("context", None, "deepgrove/maple-preview")
    _join(ctl)
    plan = ctl.snapshot()["plans"]["context"]
    if not plan["profile"]:
        pytest.skip("the small model needs no change in this fixture")
    assert ctl.apply(plan["digest"]) is True, ctl.snapshot()["apply"]
    _join(ctl)
    assert eng.relaunched == [] and profiles["deepgrove/maple-preview"].ctx_size == plan["profile"]["ctx_size"]
    assert ctl._get_store().journal()[-1]["state"] == "verified"
    assert "takes effect when" in ctl.snapshot()["apply"]["message"]


def test_undo_restores_the_previous_profile_through_the_same_transaction(world: dict) -> None:
    ctl, eng, profiles = world["ctl"], world["eng"], world["profiles"]
    plan = _analysed(world, "context")
    eng.client.n_ctx = plan["profile"]["ctx_size"]
    assert ctl.apply(plan["digest"]) is True, ctl.snapshot()["apply"]
    _join(ctl)
    assert ctl.undo() is True, ctl.snapshot()["apply"]
    _join(ctl)
    assert profiles["lmstudio-community/Qwen3.8-27B"].ctx_size == 32768
    journal = ctl._get_store().journal()
    assert [e["state"] for e in journal] == ["verified", "verified"] and journal[-1]["kind"] == "undo"
    assert ctl.snapshot()["apply"]["undo"] is False    # the newest verified change is the undo itself: nothing further to undo
    assert ctl.undo() is False
    assert len(eng.relaunched) == 2


def test_recover_restores_a_pending_entry_at_start(world: dict) -> None:
    ctl, profiles = world["ctl"], world["profiles"]
    store = ctl._get_store()
    store.journal_write([{"id": "1-apply", "at": "x", "goal": "context", "key": "lmstudio-community/Qwen3.8-27B", "digest": "d",
                          "previous": {"ctx_size": 8192, "auto": True}, "new": {"ctx_size": 200000}, "state": "pending", "detail": "", "kind": "apply"}])
    profiles["lmstudio-community/Qwen3.8-27B"] = library.ModelProfile(ctx_size=200000)
    ctl.recover()
    assert profiles["lmstudio-community/Qwen3.8-27B"].ctx_size == 8192
    assert store.journal()[0]["state"] == "recovered"
    assert any("undone" in s[2] for s in world["eng"].statuses)
    ctl.recover()                                        # idempotent
    assert store.journal()[0]["state"] == "recovered"


def test_note_baseline_ignores_a_resident_shared_model(world: dict) -> None:
    ctl = world["ctl"]
    ctl._baseline = None
    world["gpu"]["free"] = 22400
    ctl.note_baseline()
    assert ctl._baseline == (24463, 2063)
    ctl._baseline = None
    world["gpu"]["free"] = 5000
    ctl.note_baseline()
    assert ctl._baseline is None


def test_engine_delegates_never_raise(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from exocortex import app as app_mod

    cfg = Config()
    cfg.path = tmp_path / "config.toml"
    eng = app_mod.Engine(cfg, EventBus())

    class _Broken:
        def snapshot(self) -> dict:
            raise RuntimeError("x")

        def analyze(self, *a: Any) -> None:
            raise RuntimeError("x")

        def benchmark(self, *a: Any) -> bool:
            raise RuntimeError("x")

        def cancel(self) -> None:
            raise RuntimeError("x")

        def apply(self, *a: Any) -> bool:
            raise RuntimeError("x")

        def undo(self) -> bool:
            raise RuntimeError("x")

    eng._optimize_ctl = _Broken()
    assert eng.optimize_snapshot() == {}
    assert eng.optimize_analyze("balanced") is None
    assert eng.optimize_benchmark() is False
    assert eng.optimize_cancel() is None
    assert eng.optimize_apply("d") is False
    assert eng.optimize_undo() is False
    eng._optimize_ctl = None
    snap = eng.optimize_snapshot()                      # the real controller, lazily built
    assert isinstance(snap, dict) and snap.get("analysis", {}).get("state") == "idle"

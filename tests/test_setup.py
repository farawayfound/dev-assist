"""Offline tests for exocortex.setup (the first-run step run.bat calls after pip).

Every network / disk-heavy step is injected: a fake llama-server finder, a fake hardware
collector, fake speech-model ensurers, a fake library scan / Hub reader / puller. Nothing here
touches the network, the GPU, LM Studio or the real models folder.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from exocortex import setup as setup_mod
from exocortex.config import Config, load_config


# ---- fakes ----------------------------------------------------------------------------------
@dataclass
class _Binary:
    exe: Path
    flavor: str = "nvidia-cuda12"
    version: str = "b11200"
    source: str = "upstream"


class _Out:
    def __init__(self) -> None:
        self.text = ""

    def __call__(self, s: str) -> None:
        self.text += s


def _report(budget_mb: int | None = 16003, vendor: str = "nvidia") -> Any:
    from exocortex import hardware

    return hardware.HardwareReport(
        os="Windows-11", cpu_name="Ryzen AI 9 HX 375", physical_cores=12, logical_cores=24,
        ram_total_mb=32000, ram_free_mb=20000,
        gpus=[{"index": 0, "name": "NVIDIA GeForce RTX 5080 Laptop GPU", "vendor": vendor, "dedicated_mb": budget_mb or 0}],
        primary_index=0 if budget_mb else None, vendor=vendor if budget_mb else "none", budget_mb=budget_mb,
        vram_total_mb=budget_mb, vram_free_mb=budget_mb, nvidia_driver="592.14", library_dir="X:/lib",
        library_free_mb=500_000, runtime=None,
        recommended_policy="auto" if budget_mb else "fit", generated_at="2026-09-27T12:00:00", notes=[],
    )


def _quants(*pairs: tuple[str, float]) -> tuple[Any, ...]:
    from exocortex.llm import library

    return tuple(library.QuantOption(quant=q, files=(), size_bytes=int(gb * 1e9)) for q, gb in pairs)


@dataclass
class _Info:
    id: str
    quants: tuple[Any, ...]


@dataclass
class _Repo:
    id: str
    downloads: int = 1


@dataclass
class _Model:
    key: str
    size_bytes: int = 7_000_000_000


def _cfg(tmp_path: Path) -> Config:
    cfg = Config()
    cfg.path = tmp_path / "config.toml"
    cfg.paths.models_dir = str(tmp_path / "models")
    cfg.llm.library_dir = str(tmp_path / "models" / "llm")
    return cfg


def _probes(tmp_path: Path, *, scan_result: list[Any] | None = None, budget_mb: int | None = 16003) -> dict[str, Any]:
    calls: dict[str, list[Any]] = {"find": [], "ensure_models": [], "sherpa": [], "whisper": [], "pull": [], "search": []}
    infos = {
        "lmstudio-community/gemma-4-12B-it-QAT-GGUF": _Info("lmstudio-community/gemma-4-12B-it-QAT-GGUF", _quants(("Q4_0", 6.98))),
        "unsloth/gemma-4-12b-it-GGUF": _Info("unsloth/gemma-4-12b-it-GGUF", _quants(("Q4_K_M", 7.12), ("Q8_0", 12.67), ("Q2_K", 4.6))),
        "empero-ai/Qwen3.8-9B-Distill-GGUF": _Info("empero-ai/Qwen3.8-9B-Distill-GGUF", _quants(("Q4_K_M", 5.78), ("Q8_0", 9.79))),
        "lmstudio-community/gemma-4-E4B-it-GGUF": _Info("lmstudio-community/gemma-4-E4B-it-GGUF", _quants(("Q4_K_M", 5.34))),
        "empero-ai/Qwen3.8-4B-Distill-GGUF": _Info("empero-ai/Qwen3.8-4B-Distill-GGUF", _quants(("Q4_K_M", 2.78))),
        "unsloth/gemma-4-E2B-it-GGUF": _Info("unsloth/gemma-4-E2B-it-GGUF", _quants(("Q4_K_M", 3.11))),
        "unsloth/Qwen3.8-27B-GGUF": _Info("unsloth/Qwen3.8-27B-GGUF", _quants(("UD-Q4_K_M", 16.46))),
        "someone/found-model-GGUF": _Info("someone/found-model-GGUF", _quants(("Q4_K_M", 4.0))),
    }

    def find_binary(explicit: str, **kw: Any) -> _Binary:
        calls["find"].append(kw)
        return _Binary(tmp_path / "llama-server.exe")

    def collect(llm_cfg: Any, *, probe_runtime: bool = True) -> Any:
        return _report(budget_mb)

    def ensure_models(models_dir: Path, progress: Any = None) -> dict[str, Path]:
        calls["ensure_models"].append(models_dir)
        return {}

    def ensure_sherpa(name: str, models_dir: Path, progress: Any = None) -> Path:
        calls["sherpa"].append(name)
        return models_dir / name

    def ensure_whisper(name: str, *, dry_run: bool) -> str:
        calls["whisper"].append((name, dry_run))
        return f"whisper {name}: fake"

    def scan(llm_cfg: Any, **kw: Any) -> list[Any]:
        return list(scan_result or [])

    def repo_info(repo_id: str, **kw: Any) -> _Info:
        return infos[repo_id]

    def search(query: str, *, limit: int = 25, **kw: Any) -> list[_Repo]:
        calls["search"].append(query)
        return [_Repo("someone/found-model-GGUF", 42)]

    def pull(repo_id: str, quant: str, llm_cfg: Any, **kw: Any) -> _Model:
        calls["pull"].append((repo_id, quant, kw.get("budget_mb"), kw.get("policy")))
        return _Model(repo_id.replace("-GGUF", ""))

    return dict(find_binary=find_binary, collect=collect, ensure_models=ensure_models, ensure_sherpa=ensure_sherpa,
                ensure_whisper=ensure_whisper, scan=scan, repo_info=repo_info, search=search, pull=pull, _calls=calls)


@pytest.fixture(autouse=True)
def _sherpa_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    from exocortex.speech import asr

    monkeypatch.setattr(asr, "resolve_engine", lambda cfg, *a, **k: "sherpa")
    monkeypatch.delenv("EXO_SETUP_NONINTERACTIVE", raising=False)


def _run(cfg: Config, probes: dict[str, Any], **kw: Any) -> tuple[setup_mod.SetupResult, str]:
    out = _Out()
    p = {k: v for k, v in probes.items() if not k.startswith("_")}
    result = setup_mod.run_setup(cfg, out=out, **p, **kw)
    return result, out.text


# ---- recommendations ------------------------------------------------------------------------
def test_recommendations_follow_the_gpu_budget() -> None:
    ids = lambda budget: [r.repo_id for r in setup_mod.recommendations_for(budget)]  # noqa: E731
    assert ids(None) == ["empero-ai/Qwen3.8-4B-Distill-GGUF", "unsloth/gemma-4-E2B-it-GGUF"]
    assert ids(4096) == ["empero-ai/Qwen3.8-4B-Distill-GGUF", "unsloth/gemma-4-E2B-it-GGUF"]
    assert ids(8192)[:2] == ["unsloth/gemma-4-12b-it-GGUF", "empero-ai/Qwen3.8-9B-Distill-GGUF"]
    assert ids(16003)[0] == "lmstudio-community/gemma-4-12B-it-QAT-GGUF"
    assert "unsloth/Qwen3.8-27B-GGUF" not in ids(16003)
    assert ids(24576)[0] == "unsloth/Qwen3.8-27B-GGUF"
    assert len(ids(24576)) == 4      # capped


# ---- the four steps ---------------------------------------------------------------------------
def test_dry_run_touches_nothing_and_reports_each_step(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    probes = _probes(tmp_path)

    def find_binary_needs_fetch(explicit: str, **kw: Any) -> _Binary:
        return _Binary(kw["ensure"]("cuda13", tmp_path / "rt", "b11200"))

    probes["find_binary"] = find_binary_needs_fetch
    result, text = _run(cfg, probes, dry_run=True, stdin_is_tty=True)
    assert result.rc == 0
    assert "would fetch the official llama.cpp build b11200 (cuda13)" in result.runtime
    assert "would write" in result.hardware and "hardware.json" in result.hardware
    assert "[dryrun] would download silero_vad.onnx, wespeaker_en_voxceleb_CAM++.onnx" in result.speech
    assert "would download sherpa" in result.speech
    assert result.model == "[dryrun] would offer to pull a model (library empty)"
    assert not (tmp_path / "models").exists()
    assert probes["_calls"]["pull"] == [] and probes["_calls"]["ensure_models"] == []
    assert "RTX 5080" in text


def test_runtime_failure_sets_rc_1_but_the_other_steps_still_run(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    probes = _probes(tmp_path, scan_result=[_Model("a/b")])

    def find_binary_fails(explicit: str, **kw: Any) -> _Binary:
        raise RuntimeError("no llama.cpp runtime pack for a nvidia GPU found")

    probes["find_binary"] = find_binary_fails
    result, _text = _run(cfg, probes, stdin_is_tty=False)
    assert result.rc == 1
    assert result.runtime.startswith("no llama.cpp runtime: no llama.cpp runtime pack")
    assert "hardware report saved to" in result.hardware and (tmp_path / "models" / "llm" / "hardware.json").is_file()
    assert probes["_calls"]["ensure_models"] == [tmp_path / "models"]
    assert probes["_calls"]["sherpa"] == ["parakeet-tdt-0.6b-v2-int8"]
    assert result.model == "1 model(s) in the library; default maple-preview" or "not one of them" in result.model


def test_whisper_path_when_the_engine_is_faster_whisper(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from exocortex.speech import asr

    monkeypatch.setattr(asr, "resolve_engine", lambda cfg, *a, **k: "faster-whisper")
    cfg = _cfg(tmp_path)
    probes = _probes(tmp_path, scan_result=[_Model("a/b")])
    result, _ = _run(cfg, probes, stdin_is_tty=False)
    assert probes["_calls"]["whisper"] == [("large-v3-turbo", False)]
    assert probes["_calls"]["sherpa"] == []
    assert "whisper large-v3-turbo: fake" in result.speech


def test_configured_model_missing_from_library_is_flagged(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    cfg.llm.model = "maple-preview"
    probes = _probes(tmp_path, scan_result=[_Model("unsloth/gemma-4-12b-it")])
    result, _ = _run(cfg, probes, stdin_is_tty=False)
    assert "'maple-preview' is not one of them" in result.model
    cfg.llm.model = "gemma-4-12b-it"
    result, _ = _run(cfg, probes, stdin_is_tty=False)
    assert result.model == "1 model(s) in the library; default gemma-4-12b-it"


# ---- the prompt ------------------------------------------------------------------------------
def test_prompt_is_skipped_without_a_console_or_under_quiet(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    probes = _probes(tmp_path)
    result, _ = _run(cfg, probes, stdin_is_tty=False)
    assert "Models..." in result.model and "not an interactive console" in result.model
    result, _ = _run(cfg, probes, stdin_is_tty=True, quiet=True)
    assert "prompt skipped: --quiet" in result.model
    assert probes["_calls"]["pull"] == []
    assert cfg.llm.model == "maple-preview"


def test_env_noninteractive_skips_the_prompt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EXO_SETUP_NONINTERACTIVE", "1")
    cfg = _cfg(tmp_path)
    probes = _probes(tmp_path)
    result, _ = _run(cfg, probes, stdin_is_tty=True, ask=lambda _q: pytest.fail("must not prompt"))
    assert "prompt skipped" in result.model


def test_prompt_lists_recommendations_with_sizes_and_pulls_the_pick(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    probes = _probes(tmp_path)
    answers = iter(["2"])
    result, text = _run(cfg, probes, stdin_is_tty=True, ask=lambda _q: next(answers))
    assert "[1] lmstudio-community/gemma-4-12B-it-QAT-GGUF" in text and "Q4_0" in text and "7.0 GB" in text
    assert "[2] unsloth/gemma-4-12b-it-GGUF" in text and "Q4_K_M" in text
    assert "[3] empero-ai/Qwen3.8-9B-Distill-GGUF" in text
    assert "RTX 5080 Laptop GPU (16003 MiB)" in text
    assert probes["_calls"]["pull"] == [("unsloth/gemma-4-12b-it-GGUF", "Q4_K_M", 16003, "auto")]
    assert cfg.llm.model == "unsloth/gemma-4-12b-it"
    assert result.model.startswith("pulled unsloth/gemma-4-12b-it")
    saved = load_config(tmp_path / "config.toml")
    assert saved.llm.model == "unsloth/gemma-4-12b-it"


def test_prompt_default_answer_is_the_first_recommendation(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    probes = _probes(tmp_path)
    _run(cfg, probes, stdin_is_tty=True, ask=lambda _q: "")
    assert probes["_calls"]["pull"][0][:2] == ("lmstudio-community/gemma-4-12B-it-QAT-GGUF", "Q4_0")


def test_prompt_declined_leaves_everything_alone(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    probes = _probes(tmp_path)
    result, _ = _run(cfg, probes, stdin_is_tty=True, ask=lambda _q: "n")
    assert "(declined)" in result.model and probes["_calls"]["pull"] == []
    assert not (tmp_path / "config.toml").exists()


def test_prompt_search_path(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    probes = _probes(tmp_path)
    answers = iter(["s", "found model", "1"])
    result, text = _run(cfg, probes, stdin_is_tty=True, ask=lambda _q: next(answers))
    assert probes["_calls"]["search"] == ["found model"]
    assert "[1] someone/found-model-GGUF" in text
    assert probes["_calls"]["pull"] == [("someone/found-model-GGUF", "Q4_K_M", 16003, "auto")]
    assert result.model.startswith("pulled someone/found-model")


def test_prompt_rejects_garbage_then_accepts(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    probes = _probes(tmp_path)
    answers = iter(["zzz", "99", "3"])
    _, text = _run(cfg, probes, stdin_is_tty=True, ask=lambda _q: next(answers))
    assert text.count("please answer with a number") == 2
    assert probes["_calls"]["pull"][0][0] == "empero-ai/Qwen3.8-9B-Distill-GGUF"


def test_yes_pulls_the_first_recommendation_without_asking(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    probes = _probes(tmp_path)
    result, _ = _run(cfg, probes, stdin_is_tty=False, yes=True, ask=lambda _q: pytest.fail("must not prompt"))
    assert probes["_calls"]["pull"][0][:2] == ("lmstudio-community/gemma-4-12B-it-QAT-GGUF", "Q4_0")
    assert result.model.startswith("pulled")


def test_no_gpu_offers_the_cpu_sized_models(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    probes = _probes(tmp_path, budget_mb=None)
    _, text = _run(cfg, probes, stdin_is_tty=True, ask=lambda _q: "n")
    assert "[1] empero-ai/Qwen3.8-4B-Distill-GGUF" in text
    assert "[2] unsloth/gemma-4-E2B-it-GGUF" in text
    assert "gemma-4-12B" not in text
    assert "no GPU: CPU only" in text


def test_hub_unreachable_still_offers_search_and_not_now(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    probes = _probes(tmp_path)

    def repo_info_down(repo_id: str, **kw: Any) -> Any:
        raise ConnectionError("offline")

    probes["repo_info"] = repo_info_down
    result, text = _run(cfg, probes, stdin_is_tty=True, ask=lambda _q: "n")
    assert "could not be reached" in text and "[S] search" in text
    assert "(declined)" in result.model
    result, _ = _run(cfg, probes, stdin_is_tty=False, yes=True)
    assert "the Hub was unreachable" in result.model


def test_pull_failure_is_reported_not_raised(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    probes = _probes(tmp_path)

    def pull_fails(*a: Any, **kw: Any) -> Any:
        raise RuntimeError("disk full")

    probes["pull"] = pull_fails
    result, _ = _run(cfg, probes, stdin_is_tty=True, ask=lambda _q: "1")
    assert result.model == "model pull failed: disk full" and result.rc == 0
    assert cfg.llm.model == "maple-preview"


# ---- the CLI ------------------------------------------------------------------------------------
def test_main_maps_flags_onto_run_setup(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    seen: dict[str, Any] = {}

    def fake_run(cfg: Config, **kw: Any) -> setup_mod.SetupResult:
        seen.update(kw)
        seen["cfg_path"] = cfg.path
        return setup_mod.SetupResult(rc=3)

    monkeypatch.setattr(setup_mod, "run_setup", fake_run)
    monkeypatch.chdir(tmp_path)
    assert setup_mod.main(["--quiet", "--dry-run", "--no-speech"]) == 3
    assert seen["quiet"] is True and seen["dry_run"] is True and seen["do_speech"] is False
    assert seen["yes"] is False and seen["do_runtime"] is True and seen["do_model"] is True
    assert setup_mod.main(["--yes", "--no-runtime", "--no-model"]) == 3
    assert seen["yes"] is True and seen["do_runtime"] is False and seen["do_model"] is False


def test_progress_line_is_in_place_and_throttled() -> None:
    out = _Out()
    bar = setup_mod._Progress(out, label="pulling")
    bar("a.gguf", 0, 100)
    bar("a.gguf", 50, 100)          # within 0.2 s: throttled
    bar("a.gguf", 100, 100)         # done: always shown
    bar.finish()
    assert out.text.startswith("\r[ExoCortex] pulling a.gguf   0%")
    assert "100%" in out.text and out.text.endswith("\n")
    assert out.text.count("\r") == 2

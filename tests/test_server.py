"""Tests for exocortex.llm.server (llama-server manager)."""
from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import threading
import time
import types
from pathlib import Path

import pytest

from exocortex import gpu
from exocortex.config import LLMConfig
from exocortex.llm import runtimes
from exocortex.llm import server as srv


def _touch(p: Path, size: int = 10) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"x" * size)


@pytest.fixture(autouse=True)
def _isolated_upstream_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """find_server_binary()'s upstream fallback defaults to project_root()/.tools/llama.cpp -- this
    repo's real one already holds runtimes.py's own live downloads (vulkan, sycl; 2026-09-26), so
    every test in this file gets an empty tmp_path stand-in unless it passes its own
    `upstream_root=` explicitly. Without this, a test asserting "no GPU pack -> CPU" could instead
    find a real upstream Vulkan/SYCL build on THIS machine and silently pick that instead."""
    monkeypatch.setattr(srv, "project_root", lambda: tmp_path)


@pytest.fixture()
def library(tmp_path: Path) -> Path:
    lib = tmp_path / "models"
    _touch(lib / "deepgrove" / "maple-preview-GGUF" / "maple-preview-TQ1_0-head-Q4_K.gguf", 300)
    _touch(lib / "lmstudio-community" / "gemma-4-E4B-it-GGUF" / "gemma-4-E4B-it-Q4_K_M.gguf", 200)
    _touch(lib / "lmstudio-community" / "gemma-4-E4B-it-GGUF" / "mmproj-gemma-4-E4B-it-BF16.gguf", 50)
    _touch(lib / "flat.gguf", 20)
    return lib


def test_scan_models_keys_and_skips_projectors(library: Path) -> None:
    models = srv.scan_models(library)
    keys = [m.key for m in models]
    assert "deepgrove/maple-preview" in keys
    assert "lmstudio-community/gemma-4-E4B-it" in keys
    assert all("mmproj" not in m.path.name for m in models)
    assert any(m.key == "local/flat" for m in models)


def test_resolve_model_by_name_key_file_and_path(library: Path) -> None:
    m = srv.resolve_model("maple-preview", library)
    assert m.path.name == "maple-preview-TQ1_0-head-Q4_K.gguf"
    assert srv.resolve_model("deepgrove/maple-preview", library).path == m.path
    assert srv.resolve_model("maple-preview-TQ1_0-head-Q4_K.gguf", library).path == m.path
    assert srv.resolve_model(str(m.path), library).path == m.path
    assert srv.resolve_model("GEMMA-4-e4b", library).path.name.startswith("gemma-4-E4B")


def test_resolve_model_unknown_lists_known(library: Path) -> None:
    with pytest.raises(srv.LlamaServerError) as ei:
        srv.resolve_model("does-not-exist", library)
    assert "deepgrove/maple-preview" in str(ei.value)


def test_lmstudio_models_dir_reads_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = tmp_path / "lmshome"
    home.mkdir()
    (home / "settings.json").write_text(json.dumps({"downloadsFolder": str(tmp_path / "dl")}), encoding="utf-8")
    monkeypatch.setenv("LMSTUDIO_HOME", str(home))
    assert srv.lmstudio_models_dir() == tmp_path / "dl"
    cfg = LLMConfig(lmstudio_models_dir=str(tmp_path / "override"))
    assert srv.lmstudio_models_dir(cfg) == tmp_path / "override"


def test_find_server_binary_prefers_cuda12_and_newest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = tmp_path / "lmshome"
    backends = home / "extensions" / "backends"
    for name, ver in (("llama.cpp-win-x86_64-nvidia-cuda12-avx2-2.31.2", "2.31.2"),
                      ("llama.cpp-win-x86_64-nvidia-cuda12-avx2-2.41.0", "2.41.0"),
                      ("llama.cpp-win-x86_64-vulkan-avx2-2.50.0", "2.50.0")):
        d = backends / name
        _touch(d / "llama-server.exe")
        (d / "backend-manifest.json").write_text(json.dumps({"version": ver}), encoding="utf-8")
    (backends / "vendor" / "win-llama-cuda12-vendor-v2").mkdir(parents=True)
    monkeypatch.setenv("LMSTUDIO_HOME", str(home))
    b = srv.find_server_binary(vendor="nvidia")
    assert b.flavor == "nvidia-cuda12" and b.version == "2.41.0"
    assert b.exe.parent.name.endswith("2.41.0")
    assert any(d.name == "win-llama-cuda12-vendor-v2" for d in b.dll_dirs)
    assert b.framework == "cuda"
    assert "first choice" in b.reason


def test_find_server_binary_explicit(tmp_path: Path) -> None:
    exe = tmp_path / "rel" / ("llama-server.exe" if sys.platform == "win32" else "llama-server")
    _touch(exe)
    assert srv.find_server_binary(str(exe)).exe == exe
    assert srv.find_server_binary(str(exe.parent)).exe == exe
    with pytest.raises(srv.LlamaServerError):
        srv.find_server_binary(str(tmp_path / "nope"))


def test_build_args_reflect_config(library: Path) -> None:
    cfg = LLMConfig(ctx_size=4096, gpu_layers=12, threads=6, parallel=2, flash_attn=False, load_mode="mmap+mlock",
                    llama_extra_args=["--cache-reuse", "256"])
    mgr = srv.LlamaServerManager(cfg)
    m = srv.resolve_model("maple-preview", library)
    args = mgr.build_args(m, 5555)
    s = " ".join(args)
    assert "--model" in args and str(m.path) in args
    assert "--port 5555" in s and "--ctx-size 4096" in s and "--n-gpu-layers 12" in s
    assert "--threads 6" in s and "--parallel 2" in s and "--kv-unified" in s
    assert "--flash-attn off" in s and "--load-mode mmap+mlock" in s and "--cache-reuse 256" in s
    cfg1 = LLMConfig(parallel=1, threads=0)
    a1 = " ".join(srv.LlamaServerManager(cfg1).build_args(m, 1))
    assert "--kv-unified" not in a1 and f"--threads {srv.default_threads()}" in a1


def test_free_port_is_free() -> None:
    p = srv.free_port()
    assert 1024 < p < 65536


@pytest.mark.live
def test_live_start_query_stop() -> None:
    """Spawns the real llama-server with the configured model (needs LM Studio's runtime + library)."""
    import httpx

    cfg = LLMConfig()
    mgr = srv.LlamaServerManager(cfg)
    url = mgr.start()
    try:
        assert mgr.running
        r = httpx.get(url + "/health", timeout=5)
        assert r.status_code == 200
        body = {"prompt": "<|im_start|>user\nSay OK<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
                "max_tokens": 8, "stop": ["<|im_end|>"]}
        t0 = time.perf_counter()
        r = httpx.post(url + "/v1/completions", json=body, timeout=120)
        assert r.status_code == 200 and (time.perf_counter() - t0) < 30
    finally:
        mgr.stop()
    assert not mgr.running
    if os.name == "nt":
        import subprocess

        out = subprocess.run(["tasklist", "/FI", "IMAGENAME eq llama-server.exe"], capture_output=True, text=True).stdout
        # other llama-servers (LM Studio's own) may exist; ours must be gone -> cannot assert on count reliably
        assert isinstance(out, str)


# ---------------------------------------------------------------- KV cache type + threads ----
def test_build_args_quantises_the_kv_cache_by_default(library: Path) -> None:
    """q8_0 K and V by default: the same GGUF runs under llama-swap with q8_0, and the freed
    VRAM is paging margin on the display GPU (LLMConfig.kv_cache_type; "f16" opts out)."""
    m = srv.resolve_model("maple-preview", library)
    args = srv.LlamaServerManager(LLMConfig()).build_args(m, 5000)
    assert args[args.index("--cache-type-k") + 1] == "q8_0"
    assert args[args.index("--cache-type-v") + 1] == "q8_0"


def test_build_args_f16_or_blank_opts_out_of_kv_quantisation(library: Path) -> None:
    m = srv.resolve_model("maple-preview", library)
    for opt_out in ("f16", "F16", "", "  "):
        args = srv.LlamaServerManager(LLMConfig(kv_cache_type=opt_out)).build_args(m, 5000)
        assert "--cache-type-k" not in args and "--cache-type-v" not in args, opt_out


def test_default_threads_uses_the_physical_core_count(monkeypatch: pytest.MonkeyPatch) -> None:
    """The old heuristic halved os.cpu_count() assuming SMT; on a no-SMT chip (16 cores = 16
    logical, like the Core Ultra 9 285H) that starved the pool at 6 threads instead of 14."""
    monkeypatch.setattr(srv, "_physical_cores", lambda: 16)
    assert srv.default_threads() == 14
    monkeypatch.setattr(srv, "_physical_cores", lambda: 4)
    assert srv.default_threads() == 4                 # floor of 4
    monkeypatch.setattr(srv, "_physical_cores", lambda: None)
    monkeypatch.setattr(srv.os, "cpu_count", lambda: 16)
    assert srv.default_threads() == max(4, 16 // 2 - 2)   # the old heuristic as the fallback


def test_default_threads_on_this_machine_is_sane() -> None:
    n = srv.default_threads()
    assert isinstance(n, int) and n >= 4
    phys = srv._physical_cores()
    assert phys is None or (isinstance(phys, int) and phys >= 1)


# ---------------------------------------------------------------- backend_ladder (vendor ladders) ----
def test_backend_ladder_nvidia_is_unchanged() -> None:
    """Hard constraint: the NVIDIA ladder must never move (do not degrade NVIDIA machines)."""
    assert srv.backend_ladder("nvidia") == ("nvidia-cuda12", "nvidia-cuda", "vulkan", "cpu")
    assert srv.backend_ladder("nvidia", "auto") == srv.backend_ladder("nvidia")


def test_backend_ladder_amd_prefers_rocm_then_vulkan() -> None:
    """Measured 2026-09-26 (Radeon 8060S, Qwen3.8-27B Q4_K_M): ROCm 300 vs Vulkan 160 tok/s prompt
    processing and 0.4 s vs 3.4 s cached time-to-first-token, generation equal. Vulkan stays the
    fall-through for AMD GPUs LM Studio's ROCm pack does not cover (the 780M)."""
    assert srv.backend_ladder("amd") == ("amd-rocm", "vulkan", "cpu")


def test_backend_ladder_intel() -> None:
    assert srv.backend_ladder("intel") == ("vulkan", "cpu")


def test_backend_ladder_no_adapter_detected_keeps_the_legacy_order() -> None:
    """"none" means DXGI found nothing -- not that the machine has no GPU. A detection hiccup on an
    NVIDIA laptop must keep choosing the CUDA pack, so the ladder is the pre-vendor-aware order."""
    assert srv.backend_ladder("none") == ("nvidia-cuda12", "nvidia-cuda", "vulkan", "cpu")


@pytest.mark.parametrize("vendor", ["other", "qualcomm"])
def test_backend_ladder_unknown_vendor_is_vulkan_then_cpu(vendor: str) -> None:
    assert srv.backend_ladder(vendor) == ("vulkan", "cpu")


def test_backend_ladder_requested_overrides_the_head() -> None:
    assert srv.backend_ladder("amd", "vulkan")[0] == "vulkan"
    assert srv.backend_ladder("amd", "rocm")[:1] == ("amd-rocm",)
    ladder = srv.backend_ladder("amd", "cuda")
    assert ladder[:2] == ("nvidia-cuda12", "nvidia-cuda")
    assert set(ladder) == {"nvidia-cuda12", "nvidia-cuda", "vulkan", "amd-rocm", "cpu"}  # deduplicated fallback


def test_backend_ladder_requested_cpu_is_cpu_only_no_fallback() -> None:
    assert srv.backend_ladder("nvidia", "cpu") == ("cpu",)
    assert srv.backend_ladder("amd", "cpu") == ("cpu",)


def test_backend_ladder_unknown_requested_warns_and_behaves_like_auto(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level("WARNING"):
        assert srv.backend_ladder("amd", "bogus") == srv.backend_ladder("amd", "auto")
    assert "bogus" in caplog.text


# ---------------------------------------------------------------- _scan_packs / manifest detection ----
def _make_pack(backends: Path, dirname: str, *, manifest: dict | None = None) -> Path:
    d = backends / dirname
    _touch(d / "llama-server.exe")
    if manifest is not None:
        (d / "backend-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return d


def test_scan_packs_manifest_driven_flavor_detection(tmp_path: Path) -> None:
    backends = tmp_path / "backends"
    _make_pack(backends, "llama.cpp-win-x86_64-vulkan-avx2-2.46.0",
               manifest={"version": "2.46.0", "gpu": {"framework": "Vulkan"}, "vendor_lib_package_names": ["win-llama-vulkan-vendor-v2"]})
    _make_pack(backends, "llama.cpp-win-x86_64-amd-rocm-avx2-2.29.1",
               manifest={"version": "2.29.1", "gpu": {"framework": "ROCm", "targets": ["gfx1151"]}, "vendor_lib_package_names": ["win-llama-rocm-vendor-v6"]})
    _make_pack(backends, "llama.cpp-win-x86_64-nvidia-cuda-avx2-2.28.2",
               manifest={"version": "2.28.2", "gpu": {"framework": "CUDA"}})
    _make_pack(backends, "llama.cpp-win-x86_64-avx2-2.28.2", manifest={"version": "2.28.2"})
    (backends / "vendor" / "win-llama-vulkan-vendor-v2").mkdir(parents=True)
    (backends / "vendor" / "win-llama-rocm-vendor-v6" / "bin").mkdir(parents=True)
    packs = {p.flavor: p for p in srv._scan_packs(backends)}
    assert set(packs) == {"vulkan", "amd-rocm", "nvidia-cuda", "cpu"}
    assert any(d.name == "win-llama-vulkan-vendor-v2" for d in packs["vulkan"].dll_dirs)
    rocm_dirs = [d.name for d in packs["amd-rocm"].dll_dirs]
    assert "win-llama-rocm-vendor-v6" in rocm_dirs and "bin" in rocm_dirs  # the ROCm vendor DLLs live in bin/


def test_scan_packs_manifest_driven_flavor_wins_over_unusual_dirname(tmp_path: Path) -> None:
    """The manifest is authoritative even when the directory name gives no hint at all."""
    backends = tmp_path / "backends"
    _make_pack(backends, "llama.cpp-win-x86_64-mystery-9.9.9", manifest={"version": "9.9.9", "gpu": {"framework": "Vulkan"}})
    packs = srv._scan_packs(backends)
    assert len(packs) == 1 and packs[0].flavor == "vulkan"


def test_scan_packs_falls_back_to_dirname_when_manifest_missing_or_malformed(tmp_path: Path) -> None:
    backends = tmp_path / "backends"
    _make_pack(backends, "llama.cpp-win-x86_64-nvidia-cuda12-avx2-2.41.0", manifest=None)
    d = backends / "llama.cpp-win-x86_64-vulkan-avx2-2.46.0"
    _touch(d / "llama-server.exe")
    (d / "backend-manifest.json").write_text("{not json", encoding="utf-8")
    packs = {p.flavor: p for p in srv._scan_packs(backends)}
    assert set(packs) == {"nvidia-cuda12", "vulkan"}


def test_scan_packs_rocm_vendor_dir_glob_fallback_picks_newest(tmp_path: Path) -> None:
    """No vendor_lib_package_names in the manifest -> glob backends/vendor/win-llama-rocm-vendor-v*."""
    backends = tmp_path / "backends"
    _make_pack(backends, "llama.cpp-win-x86_64-amd-rocm-avx2-2.29.1",
               manifest={"version": "2.29.1", "gpu": {"framework": "ROCm"}})
    (backends / "vendor" / "win-llama-rocm-vendor-v5" / "bin").mkdir(parents=True)
    (backends / "vendor" / "win-llama-rocm-vendor-v6" / "bin").mkdir(parents=True)
    packs = srv._scan_packs(backends)
    assert len(packs) == 1
    names = [d.name for d in packs[0].dll_dirs]
    assert "win-llama-rocm-vendor-v6" in names
    assert "win-llama-rocm-vendor-v5" not in names


def test_scan_packs_ignores_dirs_without_llama_server_exe(tmp_path: Path) -> None:
    backends = tmp_path / "backends"
    (backends / "llama.cpp-win-x86_64-vulkan-avx2-2.0.0").mkdir(parents=True)  # no exe inside
    assert srv._scan_packs(backends) == []


# ---------------------------------------------------------------- find_server_binary per vendor ----
def _lmstudio_packs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    home = tmp_path / "lmshome"
    backends = home / "extensions" / "backends"
    monkeypatch.setenv("LMSTUDIO_HOME", str(home))
    return backends


def test_find_server_binary_amd_prefers_vulkan_over_cuda12_and_rocm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    backends = _lmstudio_packs(monkeypatch, tmp_path)
    _make_pack(backends, "llama.cpp-win-x86_64-nvidia-cuda12-avx2-2.41.0", manifest={"version": "2.41.0", "gpu": {"framework": "CUDA"}})
    _make_pack(backends, "llama.cpp-win-x86_64-amd-rocm-avx2-2.29.1", manifest={"version": "2.29.1", "gpu": {"framework": "ROCm"}})
    _make_pack(backends, "llama.cpp-win-x86_64-vulkan-avx2-2.31.2", manifest={"version": "2.31.2", "gpu": {"framework": "Vulkan"}})
    _make_pack(backends, "llama.cpp-win-x86_64-vulkan-avx2-2.46.0", manifest={"version": "2.46.0", "gpu": {"framework": "Vulkan"}})
    (backends / "vendor" / "win-llama-vulkan-vendor-v2").mkdir(parents=True)
    b = srv.find_server_binary(vendor="amd")
    assert b.flavor == "amd-rocm" and b.version == "2.29.1" and b.framework == "rocm"
    v = srv.find_server_binary(vendor="amd", exclude={"amd-rocm"})
    assert v.flavor == "vulkan" and v.version == "2.46.0" and v.framework == "vulkan"
    assert any(d.name == "win-llama-vulkan-vendor-v2" for d in v.dll_dirs)
    assert "after amd-rocm failed" in v.reason
    assert srv.find_server_binary(vendor="amd", backend="vulkan").flavor == "vulkan"   # explicit override


def test_find_server_binary_amd_rocm_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    backends = _lmstudio_packs(monkeypatch, tmp_path)
    _make_pack(backends, "llama.cpp-win-x86_64-amd-rocm-avx2-2.29.1", manifest={"version": "2.29.1", "gpu": {"framework": "ROCm"}})
    _make_pack(backends, "llama.cpp-win-x86_64-avx2-2.28.2", manifest={"version": "2.28.2"})
    (backends / "vendor" / "win-llama-rocm-vendor-v6" / "bin").mkdir(parents=True)
    b = srv.find_server_binary(vendor="amd")
    assert b.flavor == "amd-rocm" and b.framework == "rocm"
    names = [d.name for d in b.dll_dirs]
    assert "win-llama-rocm-vendor-v6" in names and "bin" in names


def test_find_server_binary_intel_uses_vulkan(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    backends = _lmstudio_packs(monkeypatch, tmp_path)
    _make_pack(backends, "llama.cpp-win-x86_64-vulkan-avx2-2.46.0", manifest={"version": "2.46.0", "gpu": {"framework": "Vulkan"}})
    b = srv.find_server_binary(vendor="intel")
    assert b.flavor == "vulkan"


def test_find_server_binary_no_adapter_detected_takes_the_best_installed_pack(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Detection failure ("none") must not demote an NVIDIA machine: with a CUDA 12 pack present
    it is chosen exactly as before; with only a CPU pack (a truly GPU-less machine) the CPU pack."""
    backends = _lmstudio_packs(monkeypatch, tmp_path)
    _make_pack(backends, "llama.cpp-win-x86_64-avx2-2.28.2", manifest={"version": "2.28.2"})
    _make_pack(backends, "llama.cpp-win-x86_64-vulkan-avx2-2.46.0", manifest={"version": "2.46.0", "gpu": {"framework": "Vulkan"}})
    _make_pack(backends, "llama.cpp-win-x86_64-nvidia-cuda12-avx2-2.41.0", manifest={"version": "2.41.0", "gpu": {"framework": "CUDA"}})
    b = srv.find_server_binary(vendor="none")
    assert b.flavor == "nvidia-cuda12" and b.framework == "cuda"


def test_find_server_binary_no_gpu_and_only_a_cpu_pack_uses_cpu(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    backends = _lmstudio_packs(monkeypatch, tmp_path)
    _make_pack(backends, "llama.cpp-win-x86_64-avx2-2.28.2", manifest={"version": "2.28.2"})
    b = srv.find_server_binary(vendor="none")
    assert b.flavor == "cpu" and b.framework == "cpu"


def test_find_server_binary_amd_backend_rocm_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    backends = _lmstudio_packs(monkeypatch, tmp_path)
    _make_pack(backends, "llama.cpp-win-x86_64-amd-rocm-avx2-2.29.1", manifest={"version": "2.29.1", "gpu": {"framework": "ROCm"}})
    _make_pack(backends, "llama.cpp-win-x86_64-vulkan-avx2-2.46.0", manifest={"version": "2.46.0", "gpu": {"framework": "Vulkan"}})
    b = srv.find_server_binary(vendor="amd", backend="rocm")
    assert b.flavor == "amd-rocm"


def test_find_server_binary_amd_backend_cuda_falls_back_to_vulkan_with_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    backends = _lmstudio_packs(monkeypatch, tmp_path)
    _make_pack(backends, "llama.cpp-win-x86_64-vulkan-avx2-2.46.0", manifest={"version": "2.46.0", "gpu": {"framework": "Vulkan"}})
    with caplog.at_level("WARNING"):
        b = srv.find_server_binary(vendor="amd", backend="cuda")
    assert b.flavor == "vulkan"
    assert caplog.text  # a warning was logged because vulkan is not backend_ladder's head here


def test_find_server_binary_amd_with_only_cuda_pack_raises_and_mentions_vulkan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On "amd" the ladder never contains a CUDA flavor, so a CUDA-only install is a hard error."""
    backends = _lmstudio_packs(monkeypatch, tmp_path)
    _make_pack(backends, "llama.cpp-win-x86_64-nvidia-cuda12-avx2-2.41.0", manifest={"version": "2.41.0", "gpu": {"framework": "CUDA"}})
    with pytest.raises(srv.LlamaServerError) as ei:
        srv.find_server_binary(vendor="amd")
    assert "vulkan" in str(ei.value).lower()


def test_find_server_binary_no_packs_at_all_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _lmstudio_packs(monkeypatch, tmp_path)
    with pytest.raises(srv.LlamaServerError):
        srv.find_server_binary(vendor="nvidia")


def test_find_server_binary_uses_gpu_vendor_when_vendor_omitted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backends = _lmstudio_packs(monkeypatch, tmp_path)
    _make_pack(backends, "llama.cpp-win-x86_64-amd-rocm-avx2-2.29.1", manifest={"version": "2.29.1", "gpu": {"framework": "ROCm"}})
    monkeypatch.setattr(srv.gpu, "gpu_vendor", lambda *a, **k: "amd")
    b = srv.find_server_binary(backend="rocm")
    assert b.flavor == "amd-rocm"


# ---------------------------------------------------------------- parse_list_devices ----
def test_parse_list_devices_the_three_sample_lines() -> None:
    text = (
        "Available devices:\n"
        "  Vulkan0: AMD Radeon(TM) 8060S Graphics (48971 MiB, 46523 MiB free)\n"
        "  ROCm0: AMD Radeon(TM) 8060S Graphics (44920 MiB, 44765 MiB free)\n"
        "  CUDA0: NVIDIA GeForce RTX 5090 Laptop GPU (24135 MiB, 23000 MiB free)\n"
    )
    devices = srv.parse_list_devices(text)
    assert [d.name for d in devices] == ["Vulkan0", "ROCm0", "CUDA0"]
    v = devices[0]
    assert v.description == "AMD Radeon(TM) 8060S Graphics"
    assert v.total_mb == 48971 and v.free_mb == 46523
    assert devices[2].description == "NVIDIA GeForce RTX 5090 Laptop GPU"


def test_parse_list_devices_tolerates_noise_and_crlf() -> None:
    text = "\r\n".join([
        "ggml_vulkan: found 1 Vulkan devices:",
        "Available devices:",
        "  Vulkan0: AMD Radeon(TM) 8060S Graphics (48971 MiB, 46523 MiB free)",
        "some trailing log line",
        "",
    ])
    devices = srv.parse_list_devices(text)
    assert len(devices) == 1 and devices[0].name == "Vulkan0"


def test_parse_list_devices_tolerates_missing_memory_parentheses() -> None:
    devices = srv.parse_list_devices("  Vulkan0: AMD Radeon(TM) 8060S Graphics\n")
    assert len(devices) == 1
    assert devices[0].total_mb is None and devices[0].free_mb is None
    assert devices[0].description == "AMD Radeon(TM) 8060S Graphics"


def test_parse_list_devices_empty_text() -> None:
    assert srv.parse_list_devices("") == []


# ---------------------------------------------------------------- choose_device ----
_VULKAN_TWO = [
    srv.LlamaDevice(name="Vulkan0", description="AMD Radeon(TM) 8060S Graphics", total_mb=48971, free_mb=46523),
    srv.LlamaDevice(name="Vulkan1", description="AMD Radeon(TM) 780M Graphics", total_mb=16000, free_mb=15000),
]


def test_choose_device_cuda_flavor_never_gets_a_device() -> None:
    devices = [srv.LlamaDevice(name="CUDA0", description="NVIDIA GeForce RTX 5090", total_mb=24135, free_mb=23000)] * 2
    assert srv.choose_device(devices, None, "auto", flavor="nvidia-cuda12") is None
    assert srv.choose_device(devices, None, "auto", flavor="nvidia-cuda") is None


def test_choose_device_single_device_returns_none() -> None:
    devices = [srv.LlamaDevice(name="Vulkan0", description="AMD Radeon(TM) 8060S Graphics", total_mb=48971, free_mb=46523)]
    assert srv.choose_device(devices, None, "auto", flavor="vulkan") is None
    assert srv.choose_device([], None, "auto", flavor="vulkan") is None


def test_choose_device_matches_primary_gpu_by_name() -> None:
    primary = gpu.GpuInfo(index=0, name="AMD Radeon(TM) 8060S Graphics", vendor="amd", vendor_id=0x1002, device_id=1,
                           dedicated_mb=32590, shared_mb=16203, uma=True, luid="x")
    assert srv.choose_device(_VULKAN_TWO, primary, "auto", flavor="vulkan") == "Vulkan0"


def test_choose_device_falls_back_to_largest_total_mb() -> None:
    assert srv.choose_device(_VULKAN_TWO, None, "auto", flavor="vulkan") == "Vulkan0"


def test_choose_device_explicit_name_is_used_even_if_unlisted(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level("WARNING"):
        name = srv.choose_device(_VULKAN_TWO, None, "Vulkan9", flavor="vulkan")
    assert name == "Vulkan9"
    assert caplog.text
    # a listed name warns nothing
    caplog.clear()
    with caplog.at_level("WARNING"):
        assert srv.choose_device(_VULKAN_TWO, None, "Vulkan0", flavor="vulkan") == "Vulkan0"
    assert not caplog.text


def test_choose_device_empty_string_means_let_llama_cpp_decide() -> None:
    assert srv.choose_device(_VULKAN_TWO, None, "", flavor="vulkan") is None


# ---------------------------------------------------------------- pick_default_model / choose_model ----
def _model(key: str, size_gb: float) -> srv.GgufModel:
    return srv.GgufModel(key=key, name=key.split("/")[-1], path=Path(f"/{key}.gguf"), size_bytes=int(size_gb * 1e9))


def test_pick_default_model_empty_library() -> None:
    assert srv.pick_default_model([], 40000) is None
    assert srv.pick_default_model([], None) is None


def test_pick_default_model_no_budget_picks_largest() -> None:
    models = [_model("a", 3.5), _model("b", 16.8), _model("c", 16.75)]
    assert srv.pick_default_model(models, None).key == "b"


def test_pick_default_model_with_budget_picks_largest_that_fits() -> None:
    models = [_model("a", 3.5), _model("b", 16.8), _model("c", 16.75)]
    # 0.75 * 48793 MiB ~= 36.6 GB -- both b and c fit; b is larger
    assert srv.pick_default_model(models, 48793).key == "b"


def test_pick_default_model_nothing_fits_picks_smallest() -> None:
    models = [_model("a", 20.0), _model("b", 30.0)]
    assert srv.pick_default_model(models, 4000).key == "a"  # 0.75*4000 MiB ~= 3.1 GB, neither fits


def test_choose_model_auto_nothing_fits_note_says_may_not_fit(
    library: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When pick_default_model hits its smallest-fallback, the note must not claim 'fits'."""
    big = srv.GgufModel(key="pub/big", name="big", path=Path("/big.gguf"), size_bytes=11_000_000_000)
    monkeypatch.setattr(srv, "scan_models", lambda models_dir: [big])
    model, note = srv.choose_model("auto", library, 1000)
    assert model.key == "pub/big"
    assert "fits 1000 MiB" not in note
    assert "may not fit" in note
    assert "smallest model available" in note


def test_choose_model_auto_075_to_09_band_does_not_falsely_claim_fits(
    library: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A model between 0.75x and 0.9x of budget hits the fallback but is missed by start()'s
    separate 0.9x warning -- the note is the only place this can be surfaced honestly."""
    mid = srv.GgufModel(key="pub/mid", name="mid", path=Path("/mid.gguf"), size_bytes=int(850 * 2**20))
    monkeypatch.setattr(srv, "scan_models", lambda models_dir: [mid])
    model, note = srv.choose_model("auto", library, 1000)
    assert model.key == "pub/mid"
    assert "fits 1000 MiB" not in note
    assert "may not fit" in note


def test_choose_model_auto_genuine_fit_still_says_fits(
    library: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    small = srv.GgufModel(key="pub/small", name="small", path=Path("/small.gguf"), size_bytes=int(500 * 2**20))
    monkeypatch.setattr(srv, "scan_models", lambda models_dir: [small])
    model, note = srv.choose_model("auto", library, 1000)
    assert model.key == "pub/small"
    assert "fits 1000 MiB" in note
    assert "may not fit" not in note


def test_choose_model_auto(library: Path) -> None:
    model, note = srv.choose_model("auto", library, None)
    assert model is not None and "auto:" in note
    model2, note2 = srv.choose_model("AUTO", library, None)
    assert model2.key == model.key
    model3, note3 = srv.choose_model("", library, None)
    assert model3.key == model.key


def test_choose_model_exact_spec_has_no_note(library: Path) -> None:
    model, note = srv.choose_model("maple-preview", library, None)
    assert model.path.name == "maple-preview-TQ1_0-head-Q4_K.gguf"
    assert note == ""


def test_choose_model_unknown_spec_falls_back_with_warning(library: Path, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level("WARNING"):
        model, note = srv.choose_model("does-not-exist", library, None)
    assert model is not None
    assert "does-not-exist" in note and "not found" in note
    assert "does-not-exist" in caplog.text


def test_choose_model_empty_library_raises(tmp_path: Path) -> None:
    empty = tmp_path / "empty_models"
    empty.mkdir()
    with pytest.raises(srv.LlamaServerError) as ei:
        srv.choose_model("auto", empty, None)
    assert str(empty) in str(ei.value)
    assert "LM Studio" in str(ei.value)


def test_choose_model_unknown_spec_and_empty_library_raises(tmp_path: Path) -> None:
    empty = tmp_path / "empty_models"
    empty.mkdir()
    with pytest.raises(srv.LlamaServerError):
        srv.choose_model("does-not-exist", empty, None)


# ---------------------------------------------------------------- resolve_from / choose_from ----
class _FakeLibraryModel:
    """A stand-in for exocortex.llm.library.LibraryModel: any object with .key/.name/.path/
    .size_bytes works with resolve_from/choose_from (2026-09-27, portable-install)."""

    def __init__(self, key: str, size_bytes: int) -> None:
        self.key = key
        self.name = key.split("/")[-1]
        self.path = Path(f"/{key}.gguf")
        self.size_bytes = size_bytes


def test_resolve_from_matches_key_name_stem_and_file_name(library: Path) -> None:
    models = srv.scan_models(library)
    m = srv.resolve_from(models, "maple-preview")
    assert m.path.name == "maple-preview-TQ1_0-head-Q4_K.gguf"
    assert srv.resolve_from(models, "deepgrove/maple-preview").path == m.path
    assert srv.resolve_from(models, "maple-preview-TQ1_0-head-Q4_K.gguf").path == m.path


def test_resolve_from_unknown_raises_naming_where_and_known_keys(library: Path) -> None:
    models = srv.scan_models(library)
    with pytest.raises(srv.LlamaServerError) as ei:
        srv.resolve_from(models, "does-not-exist", where="my custom library")
    assert "my custom library" in str(ei.value)
    assert "deepgrove/maple-preview" in str(ei.value)


def test_resolve_from_works_on_a_library_model_shaped_object() -> None:
    """resolve_from is the model-list half of resolve_model, factored out so it also works over
    exocortex.llm.library's richer LibraryModel objects -- not only GgufModel."""
    models = [_FakeLibraryModel("pub/a", 100), _FakeLibraryModel("pub/b", 200)]
    m = srv.resolve_from(models, "pub/a")
    assert m.key == "pub/a"


def test_choose_from_auto_resolve_fallback_and_empty(library: Path) -> None:
    models = srv.scan_models(library)
    model, note = srv.choose_from(models, "auto", None)
    assert model is not None and "auto:" in note
    model_exact, note_exact = srv.choose_from(models, "maple-preview", None)
    assert model_exact.path.name == "maple-preview-TQ1_0-head-Q4_K.gguf" and note_exact == ""
    model_fb, note_fb = srv.choose_from(models, "does-not-exist", None, where="the library")
    assert model_fb is not None
    assert "does-not-exist" in note_fb and "not found" in note_fb
    with pytest.raises(srv.LlamaServerError) as ei:
        srv.choose_from([], "auto", None, where="the library")
    assert "the library" in str(ei.value)


def test_choose_from_works_on_library_model_shaped_objects() -> None:
    models = [_FakeLibraryModel("pub/a", 100), _FakeLibraryModel("pub/b", 200)]
    model, note = srv.choose_from(models, "auto", None)
    assert model.key == "pub/b"                    # the largest, no budget
    model2, _note2 = srv.choose_from(models, "pub/a", None)
    assert model2.key == "pub/a"


# ---------------------------------------------------------------- build_args: device / cpu flavor ----
def test_build_args_appends_device_when_given(library: Path) -> None:
    m = srv.resolve_model("maple-preview", library)
    args = srv.LlamaServerManager(LLMConfig()).build_args(m, 5555, device="Vulkan0")
    s = " ".join(args)
    assert "--device Vulkan0" in s


def test_build_args_no_device_flag_when_none(library: Path) -> None:
    m = srv.resolve_model("maple-preview", library)
    args = srv.LlamaServerManager(LLMConfig()).build_args(m, 5555)
    assert "--device" not in args


def test_build_args_cpu_flavor_forces_zero_gpu_layers(library: Path) -> None:
    m = srv.resolve_model("maple-preview", library)
    mgr = srv.LlamaServerManager(LLMConfig(gpu_layers=999999))
    mgr.binary = srv.ServerBinary(exe=Path("llama-server.exe"), dll_dirs=(), flavor="cpu", version="1.0",
                                   framework="cpu", reason="test")
    args = mgr.build_args(m, 5555)
    assert args[args.index("--n-gpu-layers") + 1] == "0"


def test_build_args_non_cpu_flavor_keeps_configured_gpu_layers(library: Path) -> None:
    m = srv.resolve_model("maple-preview", library)
    mgr = srv.LlamaServerManager(LLMConfig(gpu_layers=12))
    mgr.binary = srv.ServerBinary(exe=Path("llama-server.exe"), dll_dirs=(), flavor="vulkan", version="1.0",
                                   framework="vulkan", reason="test")
    args = mgr.build_args(m, 5555)
    assert args[args.index("--n-gpu-layers") + 1] == "12"


# ------------------------------------------------- startup timeout, media, log warnings, sniffing ----
def test_estimate_startup_timeout_keeps_the_configured_floor_on_fast_media() -> None:
    """A 16.8 GB GGUF on a fixed disk reads in ~57 s at the 300 MiB/s floor: the 180 s default stands
    (NVIDIA laptops with an internal NVMe see no change)."""
    size = int(16.8e9)
    assert srv.estimate_startup_timeout_s(180.0, size, "fixed") == 180.0
    assert srv.estimate_startup_timeout_s(400.0, size, "removable") >= 400.0


def test_estimate_startup_timeout_grows_for_a_big_model_on_removable_media() -> None:
    """Measured 2026-09-26: 323 s cold load of the 16.8 GB Qwen3.8-27B from the exFAT card in this
    box's card reader -- the old fixed 180 s deadline gave up on a model that was loading fine."""
    size = int(16.8e9)
    t = srv.estimate_startup_timeout_s(180.0, size, "removable")
    assert 500 < t < 700
    assert srv.estimate_startup_timeout_s(180.0, 10_000_000, "removable") == 180.0   # a tiny model: floor


def test_media_kind_never_raises_and_returns_a_triple(tmp_path: Path) -> None:
    kind, fs, root = srv.media_kind(tmp_path)
    assert kind in {"removable", "fixed", "remote", "cdrom", "ramdisk", "unknown"}
    assert isinstance(fs, str) and isinstance(root, str)
    assert srv.media_kind(Path("Z:/does/not/exist/model.gguf"))[0] in {"unknown", "removable", "fixed", "remote", "cdrom", "ramdisk"}


@pytest.mark.parametrize("dll,expected", [("ggml-cuda.dll", "cuda"), ("ggml-cuda-12.dll", "cuda"), ("ggml-hip.dll", "rocm"),
                                          ("ggml-vulkan.dll", "vulkan"), ("ggml-sycl.dll", "sycl"),
                                          ("ggml-openvino.dll", "openvino"), ("ggml-cpu-zen4.dll", "cpu")])
def test_sniff_framework_reads_the_backend_dlls(tmp_path: Path, dll: str, expected: str) -> None:
    _touch(tmp_path / dll)
    _touch(tmp_path / "ggml-base.dll")
    assert srv.sniff_framework(tmp_path) == expected


def test_sniff_framework_unknown_folder_is_custom(tmp_path: Path) -> None:
    _touch(tmp_path / "llama-server.exe")
    assert srv.sniff_framework(tmp_path) == "custom"


def test_find_server_binary_explicit_cuda_folder_is_recognised_as_cuda(tmp_path: Path) -> None:
    """An [llm].llama_server_path pointing at a CUDA build must be treated like a CUDA pack -- in
    particular it must never be handed --device (LlamaServerManager.start gates on framework)."""
    exe = tmp_path / "rel" / ("llama-server.exe" if sys.platform == "win32" else "llama-server")
    _touch(exe)
    _touch(exe.parent / "ggml-cuda.dll")
    b = srv.find_server_binary(str(exe.parent))
    assert b.flavor == "custom" and b.framework == "cuda"
    assert "cuda" in b.reason


def test_build_args_flash_attn_off_keeps_k_quantised_but_forces_v_to_f16(library: Path, caplog: pytest.LogCaptureFixture) -> None:
    """llama.cpp refuses "quantized V cache requires flash_attn to be enabled" (measured 2026-09-26,
    Vulkan pack): flash_attn=False with the default q8_0 KV used to be a hard start failure."""
    m = srv.resolve_model("maple-preview", library)
    with caplog.at_level(logging.WARNING):
        args = srv.LlamaServerManager(LLMConfig(flash_attn=False)).build_args(m, 5000)
    assert args[args.index("--cache-type-k") + 1] == "q8_0"
    assert args[args.index("--cache-type-v") + 1] == "f16"
    assert "flash_attn is off" in caplog.text
    on = srv.LlamaServerManager(LLMConfig(flash_attn=True)).build_args(m, 5000)
    assert on[on.index("--cache-type-v") + 1] == "q8_0"           # unchanged with flash attention on


def test_scan_log_warnings_recognises_cache_reuse_and_cpu_fallback_lines() -> None:
    text = ("0.00.123 I srv  init: cache_reuse is not supported by this context, it will be disabled\r\n"
            "load_tensors: tensor token_embd.weight (q1_0) cannot be used with preferred buffer type Vulkan_Host, using CPU instead\n")
    warnings = srv.scan_log_warnings(text)
    assert len(warnings) == 2
    assert any("cache-reuse" in w for w in warnings) and any("run on the CPU" in w for w in warnings)
    assert srv.scan_log_warnings("all good\nllama-server ready") == []


def test_manager_collects_log_warnings_once(tmp_path: Path) -> None:
    mgr = srv.LlamaServerManager(LLMConfig(), log_dir=tmp_path)
    mgr.log_path = tmp_path / "llama-server.log"
    mgr.log_path.write_text("x\ncache_reuse is not supported by this context, it will be disabled\n", encoding="utf-8")
    mgr._collect_log_warnings()
    mgr._collect_log_warnings()
    assert len(mgr.warnings) == 1 and "cache-reuse" in mgr.warnings[0]


def test_choose_device_explicit_name_is_matched_case_insensitively_and_returns_the_real_spelling() -> None:
    devices = [srv.LlamaDevice("Vulkan0", "AMD Radeon(TM) 8060S Graphics", 48971, 46523),
               srv.LlamaDevice("Vulkan1", "Other", 1000, 900)]
    assert srv.choose_device(devices, None, "vulkan1", flavor="vulkan") == "Vulkan1"
    assert srv.choose_device(devices, None, " Vulkan0 ", flavor="vulkan") == "Vulkan0"
    assert srv.choose_device(devices, None, "ROCm0", flavor="vulkan") == "ROCm0"   # unknown: passed through, warned


def test_scan_packs_rocm_vendor_dir_glob_sorts_numerically_not_lexically(tmp_path: Path) -> None:
    backends = tmp_path / "backends"
    _make_pack(backends, "llama.cpp-win-x86_64-amd-rocm-avx2-2.29.1",
               manifest={"version": "2.29.1", "gpu": {"framework": "ROCm"}})
    (backends / "vendor" / "win-llama-rocm-vendor-v2" / "bin").mkdir(parents=True)
    (backends / "vendor" / "win-llama-rocm-vendor-v10" / "bin").mkdir(parents=True)
    names = [d.name for d in srv._scan_packs(backends)[0].dll_dirs]
    assert "win-llama-rocm-vendor-v10" in names and "win-llama-rocm-vendor-v2" not in names


# ------------------------------------------------------------------ start() fall-through ----
class _Attempts:
    """Scripted outcomes for LlamaServerManager._attempt, keyed by the flavor being tried. Install
    with ``monkeypatch.setattr(srv.LlamaServerManager, "_attempt", script.method)`` -- a plain
    function, so Python binds ``self`` (an instance with ``__call__`` would not be a descriptor)."""

    def __init__(self, outcomes: dict[str, str]) -> None:
        self.outcomes = outcomes
        self.tried: list[str] = []
        script = self

        def method(mgr: srv.LlamaServerManager, model_spec, *, on_status=None, conservative=False) -> str:
            flavor = mgr.binary.flavor
            script.tried.append(flavor + (":conservative" if conservative else ""))
            outcome = script.outcomes.get(flavor, "ok")
            if outcome != "ok":
                raise srv.LlamaServerError(outcome)
            return f"http://127.0.0.1:1/{flavor}"

        self.method = method


def _amd_packs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    backends = _lmstudio_packs(monkeypatch, tmp_path)
    _make_pack(backends, "llama.cpp-win-x86_64-amd-rocm-avx2-2.29.1", manifest={"version": "2.29.1", "gpu": {"framework": "ROCm"}})
    _make_pack(backends, "llama.cpp-win-x86_64-vulkan-avx2-2.46.0", manifest={"version": "2.46.0", "gpu": {"framework": "Vulkan"}})
    _make_pack(backends, "llama.cpp-win-x86_64-avx2-2.29.1", manifest={"version": "2.29.1"})
    return backends


def test_start_falls_through_from_rocm_to_vulkan_when_rocm_dies(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """docs/UPSTREAM_SUGGESTIONS.md finding #2: the one-time same-pack conservative retry is
    Intel-only (see test_start_conservative_retry_only_applies_to_intel below) -- AMD's own
    rocm -> vulkan -> cpu fall-through must keep its pre-branch attempt count/timing exactly as
    main had it: rocm dies ONCE, start() falls straight through to vulkan, no retry in between."""
    _amd_packs(monkeypatch, tmp_path)
    monkeypatch.setattr(srv.gpu, "gpu_vendor", lambda gpus=None: "amd")
    script = _Attempts({"amd-rocm": "llama-server exited with code 1 while loading"})
    monkeypatch.setattr(srv.LlamaServerManager, "_attempt", script.method)
    mgr = srv.LlamaServerManager(LLMConfig())
    statuses: list[str] = []
    url = mgr.start(on_status=statuses.append)
    assert url.endswith("/vulkan") and script.tried == ["amd-rocm", "vulkan"]
    assert mgr.binary.flavor == "vulkan" and "after amd-rocm failed" in mgr.binary.reason
    assert len(mgr.fallbacks) == 1
    assert "exited with code 1" in mgr.fallbacks[0]
    assert not any("retrying amd-rocm" in s for s in statuses)
    assert any("trying the next runtime pack" in s for s in statuses)


def test_start_reports_the_original_error_when_every_pack_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """AMD gets no conservative retry at all (finding #2) -- three attempts total (rocm, vulkan,
    cpu), matching main's pre-branch worst case -- so the final reported error is still the CPU
    pack's own, the last thing start() actually tried."""
    _amd_packs(monkeypatch, tmp_path)
    monkeypatch.setattr(srv.gpu, "gpu_vendor", lambda gpus=None: "amd")
    script = _Attempts({"amd-rocm": "rocm boom", "vulkan": "vulkan boom", "cpu": "cpu boom"})
    monkeypatch.setattr(srv.LlamaServerManager, "_attempt", script.method)
    with pytest.raises(srv.LlamaServerError) as ei:
        srv.LlamaServerManager(LLMConfig()).start()
    assert script.tried == ["amd-rocm", "vulkan", "cpu"]
    assert "cpu boom" in str(ei.value)


def test_start_conservative_retry_only_applies_to_intel(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """docs/UPSTREAM_SUGGESTIONS.md finding #2: the one-time same-pack conservative retry (Vulkan/
    SYCL/OpenVINO drivers nobody here can test) is gated on vendor == "intel", not merely on
    framework -- otherwise it would also fire for AMD's ROCm pack and for a Vulkan pack an NVIDIA
    machine fell back to, silently doubling attempt counts the owner's hard rule forbids moving."""
    backends = _lmstudio_packs(monkeypatch, tmp_path)
    _make_pack(backends, "llama.cpp-win-x86_64-vulkan-avx2-2.46.0", manifest={"version": "2.46.0", "gpu": {"framework": "Vulkan"}})
    _make_pack(backends, "llama.cpp-win-x86_64-avx2-2.29.1", manifest={"version": "2.29.1"})
    monkeypatch.setattr(srv.gpu, "gpu_vendor", lambda gpus=None: "intel")
    script = _Attempts({"vulkan": "llama-server exited with code 1 while loading"})
    monkeypatch.setattr(srv.LlamaServerManager, "_attempt", script.method)
    mgr = srv.LlamaServerManager(LLMConfig())
    url = mgr.start()
    assert url.endswith("/cpu") and script.tried == ["vulkan", "vulkan:conservative", "cpu"]
    assert "flash-attn auto" in mgr.fallbacks[0]


def test_start_never_falls_through_on_cuda(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """NVIDIA behaviour unchanged: a CUDA pack that fails raises at once, no Vulkan detour."""
    backends = _lmstudio_packs(monkeypatch, tmp_path)
    _make_pack(backends, "llama.cpp-win-x86_64-nvidia-cuda12-avx2-2.41.0", manifest={"version": "2.41.0", "gpu": {"framework": "CUDA"}})
    _make_pack(backends, "llama.cpp-win-x86_64-vulkan-avx2-2.46.0", manifest={"version": "2.46.0", "gpu": {"framework": "Vulkan"}})
    monkeypatch.setattr(srv.gpu, "gpu_vendor", lambda gpus=None: "nvidia")
    script = _Attempts({"nvidia-cuda12": "cuda boom"})
    monkeypatch.setattr(srv.LlamaServerManager, "_attempt", script.method)
    with pytest.raises(srv.LlamaServerError, match="cuda boom"):
        srv.LlamaServerManager(LLMConfig()).start()
    assert script.tried == ["nvidia-cuda12"]


def test_can_fall_through_allows_upstream_cuda_but_not_lmstudio_or_custom_cuda() -> None:
    """2026-09-27 (portable-install): a CUDA pack fetched from source "upstream" may now fall
    through to the next flavor (a driver too old for the pinned CUDA 13 build should land on
    CUDA 12, then Vulkan, then CPU) -- but a CUDA pack from an LM Studio pack or an explicit
    llama_server_path (source "custom") never does, exactly as before this branch."""
    mgr = srv.LlamaServerManager(LLMConfig())
    mgr.binary = srv.ServerBinary(exe=Path("x"), dll_dirs=(), flavor="nvidia-cuda12", version="1",
                                  framework="cuda", reason="", source="upstream")
    assert mgr._can_fall_through(set()) is True
    mgr.binary = srv.ServerBinary(exe=Path("x"), dll_dirs=(), flavor="nvidia-cuda12", version="1",
                                  framework="cuda", reason="", source="lmstudio")
    assert mgr._can_fall_through(set()) is False
    mgr.binary = srv.ServerBinary(exe=Path("x"), dll_dirs=(), flavor="custom", version="1",
                                  framework="cuda", reason="", source="custom")
    assert mgr._can_fall_through(set()) is False


def test_start_falls_through_from_an_upstream_cuda_pack_when_the_driver_is_too_old(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An nvidia machine with no LM Studio packs at all, source "upstream": a failing upstream CUDA
    build falls through to Vulkan, then CPU, exactly like any other GPU framework -- unlike
    source "lmstudio"/"custom" CUDA, which never falls through (see the two tests above).

    The ladder's second CUDA rung ("nvidia-cuda") is the 12.4 build ONLY when the first rung was
    the driver-gated 13.4 build; with a driver that already resolved to 12.4 there is nothing new
    to try there, so it is skipped rather than re-running the identical binary (review, 2026-09-27)."""
    _lmstudio_packs(monkeypatch, tmp_path)         # no LM Studio packs at all
    monkeypatch.setattr(srv.gpu, "gpu_vendor", lambda gpus=None: "nvidia")
    monkeypatch.setattr(srv.gpu, "nvidia_driver_version", lambda *a, **k: None, raising=False)
    upstream_root = tmp_path / "upstream"
    ensured: list[str] = []

    def fake_ensure_runtime(kind, root, *, build, progress=None, cancelled=None, **kw):
        ensured.append(kind)
        folder = Path(root) / build / kind
        _touch(folder / "llama-server.exe")
        return folder

    monkeypatch.setattr(srv.runtimes, "ensure_runtime", fake_ensure_runtime)
    cfg = LLMConfig(llama_source="upstream")
    cfg.runtime_dir = str(upstream_root)           # LLMConfig gains this field (agent D); set directly

    # A driver that only resolves to the 12.4 build: one CUDA attempt, then Vulkan, then CPU.
    monkeypatch.setattr(runtimes, "cuda_kind_for_driver", lambda driver: "cuda12", raising=False)
    script = _Attempts({"nvidia-cuda12": "cuda boom", "nvidia-cuda": "cuda boom too", "vulkan": "vulkan boom"})
    monkeypatch.setattr(srv.LlamaServerManager, "_attempt", script.method)
    url = srv.LlamaServerManager(cfg).start()
    assert url.endswith("/cpu")
    assert script.tried == ["nvidia-cuda12", "vulkan", "cpu"]
    assert "cuda13" not in ensured

    # A >= 580 driver: the 13.4 build first, the 12.4 build as the second rung, then Vulkan, CPU.
    # (A fresh runtime root: the 12.4 build the first case installed would otherwise be found on
    # disk, which is correct behaviour but hides the fetch this assertion is about.)
    monkeypatch.setattr(runtimes, "cuda_kind_for_driver", lambda driver: "cuda13", raising=False)
    shutil.rmtree(upstream_root, ignore_errors=True)
    ensured.clear()
    script = _Attempts({"nvidia-cuda12": "cuda boom", "nvidia-cuda": "cuda boom too", "vulkan": "vulkan boom"})
    monkeypatch.setattr(srv.LlamaServerManager, "_attempt", script.method)
    url = srv.LlamaServerManager(cfg).start()
    assert url.endswith("/cpu")
    assert script.tried == ["nvidia-cuda12", "nvidia-cuda", "vulkan", "cpu"]
    assert ensured[:2] == ["cuda13", "cuda12"]


def test_start_never_falls_through_on_an_explicit_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    exe = tmp_path / "rel" / ("llama-server.exe" if sys.platform == "win32" else "llama-server")
    _touch(exe)
    _touch(exe.parent / "ggml-vulkan.dll")
    monkeypatch.setattr(srv.gpu, "gpu_vendor", lambda gpus=None: "amd")
    script = _Attempts({"custom": "custom boom"})
    monkeypatch.setattr(srv.LlamaServerManager, "_attempt", script.method)
    with pytest.raises(srv.LlamaServerError, match="custom boom"):
        srv.LlamaServerManager(LLMConfig(llama_server_path=str(exe.parent))).start()
    assert script.tried == ["custom"]


def test_attempt_refuses_a_gpu_pack_that_lists_no_device(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, library: Path) -> None:
    """A ROCm pack on a GPU it does not support prints no device: it must not be spawned (the
    caller then falls through to Vulkan)."""
    _amd_packs(monkeypatch, tmp_path)
    monkeypatch.setattr(srv.gpu, "gpu_vendor", lambda gpus=None: "amd")
    monkeypatch.setattr(srv, "list_devices", lambda binary, timeout_s=20.0: [])
    monkeypatch.setattr(srv, "lmstudio_models_dir", lambda cfg=None: library)
    mgr = srv.LlamaServerManager(LLMConfig(model="maple-preview"))
    mgr.binary = srv.find_server_binary(vendor="amd")
    assert mgr.binary.framework == "rocm"
    with pytest.raises(srv.LlamaServerError, match="lists no GPU device"):
        mgr._attempt(None)


# ---------------------------------------------------------------- upstream source resolution ----
def test_find_server_binary_upstream_source_downloads_via_ensure(tmp_path: Path) -> None:
    upstream_root = tmp_path / "upstream"
    calls: list[tuple[str, Path, str]] = []

    def ensure(kind: str, root: Path, build: str) -> Path:
        calls.append((kind, root, build))
        folder = root / build / kind
        _touch(folder / "llama-server.exe")
        return folder

    b = srv.find_server_binary(vendor="intel", source="upstream", upstream_root=upstream_root, ensure=ensure)
    assert b.flavor == "vulkan" and b.source == "upstream" and b.framework == "vulkan"
    assert b.dll_dirs == (upstream_root / runtimes.PINNED_BUILD / "vulkan",)
    assert calls == [("vulkan", upstream_root, runtimes.PINNED_BUILD)]
    assert "upstream llama.cpp" in b.reason


def test_find_server_binary_upstream_source_finds_already_installed_without_downloading(tmp_path: Path) -> None:
    """installed_runtime() is checked BEFORE any ensure() call -- an already-installed upstream
    runtime is used with no network call at all (ensure=None must not crash this path)."""
    upstream_root = tmp_path / "upstream"
    _touch(upstream_root / runtimes.PINNED_BUILD / "vulkan" / "llama-server.exe")
    b = srv.find_server_binary(vendor="intel", source="upstream", upstream_root=upstream_root, ensure=None)
    assert b.flavor == "vulkan" and b.source == "upstream"


def test_find_server_binary_upstream_source_fetches_cuda_and_rocm_upstream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """2026-09-27 (portable-install): [llm].llama_source = 'upstream' now covers CUDA and ROCm too
    -- a fresh machine with no LM Studio installed at all must still be able to fetch a working
    llama.cpp runtime for an NVIDIA or AMD GPU. cuda_kind_for_driver (agent C) and
    nvidia_driver_version (agent B) are monkeypatched here (raising=False) since neither has
    necessarily landed in this tree yet; _upstream_kind_for falls back to plain attribute lookups
    on the real gpu/runtimes modules, so this is exactly what it will see once they do."""
    upstream_root = tmp_path / "upstream"
    calls: list[str] = []

    def ensure(kind: str, root: Path, build: str) -> Path:
        calls.append(kind)
        folder = root / build / kind
        _touch(folder / "llama-server.exe")
        return folder

    monkeypatch.setattr(
        runtimes, "cuda_kind_for_driver",
        lambda driver: "cuda13" if driver and int(driver.split(".")[0]) >= 580 else "cuda12",
        raising=False,
    )

    monkeypatch.setattr(srv.gpu, "nvidia_driver_version", lambda *a, **k: "592.14", raising=False)
    b = srv.find_server_binary(vendor="nvidia", source="upstream", upstream_root=upstream_root, ensure=ensure)
    assert calls == ["cuda13"]
    assert b.flavor == "nvidia-cuda12" and b.source == "upstream"

    calls.clear()
    monkeypatch.setattr(srv.gpu, "nvidia_driver_version", lambda *a, **k: None, raising=False)
    b2 = srv.find_server_binary(vendor="nvidia", source="upstream", upstream_root=upstream_root, ensure=ensure)
    assert calls == ["cuda12"]
    assert b2.flavor == "nvidia-cuda12" and b2.source == "upstream"

    calls.clear()
    b3 = srv.find_server_binary(vendor="amd", source="upstream", upstream_root=upstream_root, ensure=ensure)
    assert calls == ["rocm"]
    assert b3.flavor == "amd-rocm" and b3.source == "upstream"


def test_find_server_binary_upstream_source_still_skips_amd_ladder_head_when_backend_overrides(
    tmp_path: Path,
) -> None:
    """[llm].llama_backend = 'rocm' on an "amd"-vendor machine puts amd-rocm first in the ladder
    (backend_ladder's override) -- this still ends up fetching it upstream now (unlike before this
    branch, where CUDA/ROCm were never fetched at all): the override changes ORDER, not which
    flavors are eligible for an upstream build."""
    upstream_root = tmp_path / "upstream"
    calls: list[str] = []

    def ensure(kind: str, root: Path, build: str) -> Path:
        calls.append(kind)
        folder = root / build / kind
        _touch(folder / "llama-server.exe")
        return folder

    b = srv.find_server_binary(vendor="amd", backend="rocm", source="upstream",
                               upstream_root=upstream_root, ensure=ensure)
    assert calls == ["rocm"]
    assert b.flavor == "amd-rocm" and b.source == "upstream"


def test_find_server_binary_lmstudio_source_never_fetches_even_for_sycl_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """[llm].llama_source = 'lmstudio' (the pre-2026-09-26 behaviour) never calls `ensure` -- not
    even for backend "sycl"/"openvino", which LM Studio's own catalog never packages: it just
    falls back down the ladder to Vulkan/CPU like any other missing pack (see the llama_backend
    config comment)."""
    backends = _lmstudio_packs(monkeypatch, tmp_path)
    _make_pack(backends, "llama.cpp-win-x86_64-vulkan-avx2-2.46.0",
               manifest={"version": "2.46.0", "gpu": {"framework": "Vulkan"}})

    def ensure(kind, root, build):
        raise AssertionError("ensure must never be called under source='lmstudio'")

    b = srv.find_server_binary(vendor="intel", backend="sycl", source="lmstudio", ensure=ensure)
    assert b.flavor == "vulkan" and b.source == "lmstudio"


def test_find_server_binary_auto_fetches_upstream_only_when_no_lmstudio_gpu_pack_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """vendor="intel" (never "nvidia"/"amd" -- see the two tests right below this one, added for
    docs/UPSTREAM_SUGGESTIONS.md finding #1: those two vendors' own hard "byte-identical to main"
    rule means "auto" must never reach for an upstream build, fetched or already on disk)."""
    _lmstudio_packs(monkeypatch, tmp_path)         # no LM Studio packs installed at all
    upstream_root = tmp_path / "upstream"
    _touch(upstream_root / runtimes.PINNED_BUILD / "vulkan" / "llama-server.exe")
    b = srv.find_server_binary(vendor="intel", source="auto", upstream_root=upstream_root)
    assert b.flavor == "vulkan" and b.source == "upstream"
    assert "no LM Studio pack for any intel GPU flavor" in b.reason


@pytest.mark.parametrize(
    "vendor,flavor,dirname,manifest",
    [
        ("nvidia", "nvidia-cuda12", "llama.cpp-win-x86_64-nvidia-cuda12-avx2-2.41.0",
         {"version": "2.41.0", "gpu": {"framework": "CUDA"}}),
        ("amd", "amd-rocm", "llama.cpp-win-x86_64-amd-rocm-avx2-2.29.1",
         {"version": "2.29.1", "gpu": {"framework": "ROCm"}}),
    ],
)
def test_find_server_binary_auto_with_an_lmstudio_pack_is_byte_identical_for_nvidia_and_amd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, vendor: str, flavor: str, dirname: str, manifest: dict
) -> None:
    """2026-09-27 (portable-install): an NVIDIA or AMD machine that DOES have an LM Studio pack for
    its own vendor must stay byte-identical to before this branch under the default [llm].llama_source
    ("auto") -- no ensure() call, no network, ever. This is the half of the old, now-split
    "never touches upstream for nvidia or amd" test that is still true after this branch: only
    "no LM Studio pack anywhere for this vendor" changed (see the test right below this one)."""
    backends = _lmstudio_packs(monkeypatch, tmp_path)
    _make_pack(backends, dirname, manifest=manifest)

    def ensure(kind: str, root: Path, build: str) -> Path:
        raise AssertionError(f"ensure must never be called for vendor={vendor!r} when an LM Studio pack exists")

    b = srv.find_server_binary(vendor=vendor, source="auto", upstream_root=tmp_path / "upstream", ensure=ensure)
    assert b.flavor == flavor and b.source == "lmstudio"


def test_find_server_binary_auto_fetches_upstream_gpu_then_cpu_for_nvidia_and_amd_with_no_lmstudio_packs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """2026-09-27 (portable-install): with NO LM Studio packs installed AT ALL (a genuinely fresh
    machine, not just this vendor's own missing flavor), "auto" must now reach for the upstream
    llama.cpp build too, exactly like it already did for Intel -- this is the half of the old
    "never touches upstream for nvidia or amd" test that this branch deliberately changes: the
    whole point of "portable install" is a machine with neither LM Studio nor Ollama still getting
    a working llama-server from one run. Also covers falling all the way to the upstream CPU build
    when the GPU fetch itself fails (network down)."""
    _lmstudio_packs(monkeypatch, tmp_path)         # nothing installed anywhere
    monkeypatch.setattr(runtimes, "cuda_kind_for_driver", lambda driver: "cuda12", raising=False)
    monkeypatch.setattr(srv.gpu, "nvidia_driver_version", lambda *a, **k: None, raising=False)
    upstream_root = tmp_path / "upstream"
    calls: list[str] = []

    def ensure_ok(kind: str, root: Path, build: str) -> Path:
        calls.append(kind)
        folder = root / build / kind
        _touch(folder / "llama-server.exe")
        return folder

    b = srv.find_server_binary(vendor="nvidia", source="auto", upstream_root=upstream_root, ensure=ensure_ok)
    assert calls == ["cuda12"] and b.flavor == "nvidia-cuda12" and b.source == "upstream"
    assert "no LM Studio pack for any nvidia GPU flavor" in b.reason

    calls.clear()
    b2 = srv.find_server_binary(vendor="amd", source="auto", upstream_root=upstream_root, ensure=ensure_ok)
    assert calls == ["rocm"] and b2.flavor == "amd-rocm" and b2.source == "upstream"

    # and when the GPU fetch itself fails, fall all the way to the upstream CPU build
    calls.clear()

    def ensure_gpu_fails(kind: str, root: Path, build: str) -> Path:
        calls.append(kind)
        if kind in ("cuda12", "rocm"):
            raise runtimes.UpstreamRuntimeError("network down")
        folder = root / build / kind
        _touch(folder / "llama-server.exe")
        return folder

    upstream_root2 = tmp_path / "upstream2"
    b3 = srv.find_server_binary(vendor="nvidia", source="auto", upstream_root=upstream_root2, ensure=ensure_gpu_fails)
    assert calls == ["cuda12", "cpu"] and b3.flavor == "cpu" and b3.source == "upstream"


def test_find_server_binary_auto_prefers_a_later_lmstudio_pack_over_fetching_an_earlier_gpu_flavor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """intel-sycl (the ladder's first flavor here, via backend='sycl') has no LM Studio pack and
    would be "fetchable", but vulkan (later in the ladder) DOES have one -- the LM Studio pack
    must win, with no upstream fetch attempted at all."""
    backends = _lmstudio_packs(monkeypatch, tmp_path)
    _make_pack(backends, "llama.cpp-win-x86_64-vulkan-avx2-2.46.0",
               manifest={"version": "2.46.0", "gpu": {"framework": "Vulkan"}})

    def ensure(kind, root, build):
        raise AssertionError(f"must not fetch {kind}: vulkan (later in the ladder) has an LM Studio pack")

    b = srv.find_server_binary(vendor="intel", backend="sycl", source="auto",
                               upstream_root=tmp_path / "upstream", ensure=ensure)
    assert b.flavor == "vulkan" and b.source == "lmstudio"


def test_find_server_binary_auto_falls_all_the_way_to_upstream_cpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _lmstudio_packs(monkeypatch, tmp_path)         # nothing installed anywhere
    calls: list[str] = []

    def ensure(kind: str, root: Path, build: str) -> Path:
        calls.append(kind)
        if kind == "vulkan":
            raise runtimes.UpstreamRuntimeError("network down")
        folder = root / build / kind
        _touch(folder / "llama-server.exe")
        return folder

    b = srv.find_server_binary(vendor="intel", source="auto", upstream_root=tmp_path / "upstream", ensure=ensure)
    assert calls == ["vulkan", "cpu"]
    assert b.flavor == "cpu" and b.source == "upstream"


def test_find_server_binary_source_none_behaves_like_auto(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`source=None` (LlamaServerManager's own default when nothing overrides it) must be
    identical to `source="auto"` -- byte-identical LM-Studio-pack behaviour for NVIDIA/AMD."""
    backends = _lmstudio_packs(monkeypatch, tmp_path)
    _make_pack(backends, "llama.cpp-win-x86_64-nvidia-cuda12-avx2-2.41.0",
               manifest={"version": "2.41.0", "gpu": {"framework": "CUDA"}})
    b = srv.find_server_binary(vendor="nvidia", source=None)
    assert b.flavor == "nvidia-cuda12" and b.source == "lmstudio" and "first choice" in b.reason


def test_find_server_binary_unknown_source_warns_and_behaves_like_auto(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    backends = _lmstudio_packs(monkeypatch, tmp_path)
    _make_pack(backends, "llama.cpp-win-x86_64-vulkan-avx2-2.46.0",
               manifest={"version": "2.46.0", "gpu": {"framework": "Vulkan"}})
    with caplog.at_level("WARNING"):
        b = srv.find_server_binary(vendor="intel", source="bogus")
    assert b.flavor == "vulkan" and b.source == "lmstudio"
    assert "bogus" in caplog.text


# ---------------------------------------------------------------- Intel Arc device selection ----
def test_parse_list_devices_intel_arc_discrete_and_igpu_and_sycl_style_names() -> None:
    """B580 (Battlemage, discrete) + a Lunar-Lake-style Arc iGPU under Vulkan, and a SYCL-style
    device line -- the generic device-name pattern already covers "SYCL0" without any regex
    change (see find_server_binary's module docstring, item 6)."""
    text = (
        "Vulkan0: Intel(R) Arc(TM) B580 Graphics (12288 MiB, 11800 MiB free)\n"
        "Vulkan1: Intel(R) Arc(TM) Graphics (128 MiB, 100 MiB free)\n"
        "SYCL0: Intel(R) Arc(TM) B580 Graphics (12288 MiB, 11800 MiB free)\n"
    )
    devices = srv.parse_list_devices(text)
    assert [d.name for d in devices] == ["Vulkan0", "Vulkan1", "SYCL0"]
    assert devices[0].description == "Intel(R) Arc(TM) B580 Graphics" and devices[0].total_mb == 12288
    assert devices[1].description == "Intel(R) Arc(TM) Graphics" and devices[1].total_mb == 128
    assert devices[2].name == "SYCL0" and devices[2].total_mb == 12288


def test_choose_device_intel_arc_b580_matches_by_name_over_the_igpu() -> None:
    devices = [
        srv.LlamaDevice(name="Vulkan0", description="Intel(R) Arc(TM) B580 Graphics", total_mb=12288, free_mb=11800),
        srv.LlamaDevice(name="Vulkan1", description="Intel(R) Arc(TM) Graphics", total_mb=128, free_mb=100),
    ]
    primary = gpu.GpuInfo(index=0, name="Intel(R) Arc(TM) B580 Graphics", vendor="intel", vendor_id=0x8086,
                          device_id=1, dedicated_mb=12288, shared_mb=0, uma=False, luid="x")
    assert srv.choose_device(devices, primary, "auto", flavor="vulkan") == "Vulkan0"


def test_choose_device_intel_arc_a770_matches_over_intel_uhd_igpu() -> None:
    devices = [
        srv.LlamaDevice(name="Vulkan0", description="Intel(R) Arc(TM) A770 Graphics", total_mb=16384, free_mb=15800),
        srv.LlamaDevice(name="Vulkan1", description="Intel(R) UHD Graphics 770", total_mb=128, free_mb=100),
    ]
    primary = gpu.GpuInfo(index=0, name="Intel(R) Arc(TM) A770 Graphics", vendor="intel", vendor_id=0x8086,
                          device_id=2, dedicated_mb=16384, shared_mb=0, uma=False, luid="y")
    assert srv.choose_device(devices, primary, "auto", flavor="vulkan") == "Vulkan0"


def test_choose_device_intel_arc_discrete_wins_over_a_bigger_amd_igpu() -> None:
    """A UMA AMD iGPU reports far more "memory" (shared system RAM) than the Intel discrete card,
    but with a matching primary name the discrete GPU must still be chosen -- the AMD box this
    branch was built on cannot have an Intel Arc card, so this is unit-tested only."""
    devices = [
        srv.LlamaDevice(name="Vulkan0", description="AMD Radeon(TM) 8060S Graphics", total_mb=48971, free_mb=46523),
        srv.LlamaDevice(name="Vulkan1", description="Intel(R) Arc(TM) B570 Graphics", total_mb=10240, free_mb=9800),
    ]
    primary = gpu.GpuInfo(index=0, name="Intel(R) Arc(TM) B570 Graphics", vendor="intel", vendor_id=0x8086,
                          device_id=3, dedicated_mb=10240, shared_mb=0, uma=False, luid="z")
    assert srv.choose_device(devices, primary, "auto", flavor="vulkan") == "Vulkan1"


def test_pick_default_model_intel_b580_budget_picks_largest_under_9216_mib() -> None:
    """12288 MiB budget (Arc B580): 0.75 * 12288 = 9216 MiB -- the largest GGUF at or under that."""
    models = [_model("a", 8.5), _model("b", 9.0), _model("c", 11.0)]   # b: 9.0 GB ~ 8590 MiB <= 9216
    assert srv.pick_default_model(models, 12288).key == "b"


# ---------------------------------------------------------------- build_args conservative retry ----
def test_build_args_conservative_uses_flash_attn_auto_and_no_kv_cache_flags(library: Path) -> None:
    m = srv.resolve_model("maple-preview", library)
    args = srv.LlamaServerManager(LLMConfig()).build_args(m, 5555, conservative=True)
    assert "--flash-attn auto" in " ".join(args)
    assert "--cache-type-k" not in args and "--cache-type-v" not in args
    normal = srv.LlamaServerManager(LLMConfig()).build_args(m, 5555)
    assert "--flash-attn on" in " ".join(normal) and "--cache-type-k" in normal


# ---------------------------------------------------------------- per-model tuning profile ----
def _install_fake_library(monkeypatch: pytest.MonkeyPatch, **attrs: object) -> types.ModuleType:
    """Installs a fake exocortex.llm.library module for the duration of the test, via srv._library()
    -- the one indirection build_args()/available_models()/_model_search_where() all call instead of
    each doing its own inline ``from exocortex.llm import library``. Patching sys.modules directly
    does NOT reliably work here: once the real exocortex.llm.library has been imported successfully
    anywhere in this process, CPython caches it as an attribute of the exocortex.llm package, and an
    inline ``from exocortex.llm import library`` then reads that cached attribute without even
    consulting sys.modules again -- see srv._library()'s own docstring. Monkeypatching the accessor
    function itself sidesteps that caching question entirely."""
    mod = types.ModuleType("exocortex.llm.library")
    for k, v in attrs.items():
        setattr(mod, k, v)
    monkeypatch.setattr(srv, "_library", lambda: mod)
    return mod


class _FakeProfile:
    def __init__(self, **kw: object) -> None:
        for name in ("ctx_size", "gpu_layers", "kv_cache_type", "flash_attn", "threads", "parallel"):
            setattr(self, name, kw.get(name))
        self.extra_args = kw.get("extra_args", [])


def test_build_args_applies_a_per_model_profile_override(
    monkeypatch: pytest.MonkeyPatch, library: Path, caplog: pytest.LogCaptureFixture
) -> None:
    m = srv.resolve_model("maple-preview", library)
    profile = _FakeProfile(ctx_size=16384, gpu_layers=20, kv_cache_type="f16", flash_attn=False,
                           threads=3, parallel=4, extra_args=["--foo", "bar"])
    _install_fake_library(monkeypatch, profile_for=lambda cfg, key: profile)
    mgr = srv.LlamaServerManager(LLMConfig(ctx_size=8192, gpu_layers=999999, kv_cache_type="q8_0",
                                           flash_attn=True, threads=0, parallel=2))
    with caplog.at_level(logging.INFO):
        args = mgr.build_args(m, 5555)
    s = " ".join(args)
    assert "--ctx-size 16384" in s and "--n-gpu-layers 20" in s
    assert "--threads 3" in s and "--parallel 4" in s and "--kv-unified" in s
    assert "--flash-attn off" in s
    assert "--cache-type-k" not in args           # f16 opts out of KV quantisation
    assert "--foo bar" in s
    assert mgr.profile is profile
    assert "profile for" in caplog.text and m.key in caplog.text


def test_build_args_profile_fields_left_none_fall_back_to_cfg(monkeypatch: pytest.MonkeyPatch, library: Path) -> None:
    """A profile that only pins one field (ctx_size) leaves every other field exactly as [llm]
    configures it -- None means "use the global value", never "zero it out"."""
    m = srv.resolve_model("maple-preview", library)
    profile = _FakeProfile(ctx_size=32768)
    _install_fake_library(monkeypatch, profile_for=lambda cfg, key: profile)
    mgr = srv.LlamaServerManager(LLMConfig(ctx_size=8192, gpu_layers=12, threads=6, parallel=2, flash_attn=True))
    args = mgr.build_args(m, 5555)
    s = " ".join(args)
    assert "--ctx-size 32768" in s
    assert "--n-gpu-layers 12" in s and "--threads 6" in s and "--parallel 2" in s
    assert "--flash-attn on" in s


def test_build_args_profile_never_overrides_cpu_flavor_zero_gpu_layers(
    monkeypatch: pytest.MonkeyPatch, library: Path
) -> None:
    """The CPU-flavor "--n-gpu-layers 0" rule outranks a profile's own gpu_layers value."""
    m = srv.resolve_model("maple-preview", library)
    profile = _FakeProfile(gpu_layers=20)
    _install_fake_library(monkeypatch, profile_for=lambda cfg, key: profile)
    mgr = srv.LlamaServerManager(LLMConfig(gpu_layers=999999))
    mgr.binary = srv.ServerBinary(exe=Path("llama-server.exe"), dll_dirs=(), flavor="cpu", version="1.0",
                                   framework="cpu", reason="test")
    args = mgr.build_args(m, 5555)
    assert args[args.index("--n-gpu-layers") + 1] == "0"


def test_build_args_with_no_profile_module_is_unaffected(monkeypatch: pytest.MonkeyPatch, library: Path) -> None:
    """exocortex.llm.library not built yet (or just not importable, for any reason) must never
    break a start -- build_args() imports it lazily and guards with try/except ImportError. Setting
    a sys.modules entry to None is the standard way to force an ImportError deterministically,
    regardless of whether the real module exists on disk in this tree."""
    monkeypatch.setitem(sys.modules, "exocortex.llm.library", None)
    m = srv.resolve_model("maple-preview", library)
    mgr = srv.LlamaServerManager(LLMConfig(ctx_size=8192))
    args = mgr.build_args(m, 5555)
    assert mgr.profile is None
    assert "--ctx-size 8192" in " ".join(args)


def test_build_args_profile_for_raising_is_swallowed(monkeypatch: pytest.MonkeyPatch, library: Path) -> None:
    """A library module that exists but whose profile_for() raises (a corrupt profiles.json, say)
    must never break a start either."""
    m = srv.resolve_model("maple-preview", library)

    def boom(cfg: object, key: str) -> None:
        raise ValueError("corrupt profiles.json")

    _install_fake_library(monkeypatch, profile_for=boom)
    mgr = srv.LlamaServerManager(LLMConfig())
    args = mgr.build_args(m, 5555)
    assert mgr.profile is None
    assert "--ctx-size 8192" in " ".join(args)


# ---------------------------------------------------------------- available_models ----
def test_available_models_falls_back_to_scan_models_without_library(
    monkeypatch: pytest.MonkeyPatch, library: Path
) -> None:
    monkeypatch.setitem(sys.modules, "exocortex.llm.library", None)
    monkeypatch.setattr(srv, "lmstudio_models_dir", lambda cfg=None: library)
    mgr = srv.LlamaServerManager(LLMConfig())
    models = mgr.available_models()
    assert any(m.key == "deepgrove/maple-preview" for m in models)


def test_available_models_uses_library_scan_when_present(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_models = ["a", "b"]
    _install_fake_library(monkeypatch, scan_library=lambda cfg, **kw: fake_models)
    mgr = srv.LlamaServerManager(LLMConfig())
    assert mgr.available_models() == fake_models


# ---------------------------------------------------------------- [llm].llama_env ----
class _NoOpProc:
    def __init__(self, pid: int = 4321) -> None:
        self.pid = pid

    def poll(self) -> "int | None":
        return None


def _attempt_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, library: Path, cfg: LLMConfig,
                 flavor: str, framework: str) -> dict:
    monkeypatch.setattr(srv, "lmstudio_models_dir", lambda cfg=None: library)
    monkeypatch.setattr(srv.LlamaServerManager, "_wait_ready", lambda self: None)
    captured: dict = {}

    def fake_popen(cmd, *, env, **kwargs):
        captured["env"] = env
        return _NoOpProc()

    monkeypatch.setattr(srv.subprocess, "Popen", fake_popen)
    mgr = srv.LlamaServerManager(cfg)
    mgr.binary = srv.ServerBinary(exe=Path("llama-server.exe"), dll_dirs=(tmp_path,), flavor=flavor,
                                  version="b11200", framework=framework, reason="test", source="upstream")
    mgr._attempt(None)
    return captured["env"]


def test_attempt_merges_llama_env_after_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, library: Path) -> None:
    cfg = LLMConfig(model="maple-preview", llama_device="", llama_env={"GGML_VK_VISIBLE_DEVICES": "0"})
    env = _attempt_env(tmp_path, monkeypatch, library, cfg, "vulkan", "vulkan")
    assert env["GGML_VK_VISIBLE_DEVICES"] == "0"
    assert str(tmp_path) in env["PATH"]


def test_attempt_sycl_framework_defaults_zes_enable_sysman(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, library: Path
) -> None:
    cfg = LLMConfig(model="maple-preview", llama_device="", llama_env={"GGML_VK_VISIBLE_DEVICES": "0"})
    env = _attempt_env(tmp_path, monkeypatch, library, cfg, "intel-sycl", "sycl")
    assert env["ZES_ENABLE_SYSMAN"] == "1"
    assert env["GGML_VK_VISIBLE_DEVICES"] == "0"


def test_attempt_llama_env_can_override_the_sycl_sysman_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, library: Path
) -> None:
    cfg = LLMConfig(model="maple-preview", llama_device="", llama_env={"ZES_ENABLE_SYSMAN": "0"})
    env = _attempt_env(tmp_path, monkeypatch, library, cfg, "intel-sycl", "sycl")
    assert env["ZES_ENABLE_SYSMAN"] == "0"


def test_attempt_non_sycl_framework_gets_no_sysman_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, library: Path
) -> None:
    cfg = LLMConfig(model="maple-preview", llama_device="")
    env = _attempt_env(tmp_path, monkeypatch, library, cfg, "vulkan", "vulkan")
    assert "ZES_ENABLE_SYSMAN" not in env


# ---------------------------------------------------------------- quit-during-load (stop race) ----
class _NeverReadyProc:
    """A Popen stand-in that never exits on its own and records terminate()/kill()."""

    def __init__(self) -> None:
        self.pid = 999
        self.terminated = False
        self.killed = False

    def poll(self) -> "int | None":
        return None

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True

    def wait(self, timeout: "float | None" = None) -> int:
        return 0


class _AlwaysFailingHttpxClient:
    """httpx.Client stand-in whose GET never succeeds -- the server is "never ready" until
    something else (a stop) ends the wait."""

    def __init__(self, *a, **k) -> None:
        pass

    def __enter__(self) -> "_AlwaysFailingHttpxClient":
        return self

    def __exit__(self, *a) -> bool:
        return False

    def get(self, url: str):
        raise srv.httpx.HTTPError("not ready")


def test_start_raises_stopped_promptly_when_stopped_mid_load_and_never_tries_the_next_pack(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, library: Path
) -> None:
    """docs/UPSTREAM_SUGGESTIONS.md finding #1: a stop() during a slow/hung load must return
    promptly (not block behind the ~180s+ startup timeout) and start() must raise "stopped", not
    a timeout -- and the fall-through loop must not try vulkan/cpu after a deliberate stop."""
    _amd_packs(monkeypatch, tmp_path)
    monkeypatch.setattr(srv.gpu, "gpu_vendor", lambda gpus=None: "amd")
    monkeypatch.setattr(srv, "lmstudio_models_dir", lambda cfg=None: library)
    proc = _NeverReadyProc()
    monkeypatch.setattr(srv.subprocess, "Popen", lambda *a, **k: proc)
    monkeypatch.setattr(srv.httpx, "Client", _AlwaysFailingHttpxClient)

    cfg = LLMConfig(model="maple-preview", llama_device="", llama_startup_timeout_s=180.0)
    mgr = srv.LlamaServerManager(cfg)
    result: dict = {}

    def run() -> None:
        try:
            mgr.start()
        except srv.LlamaServerError as e:
            result["error"] = e

    t = threading.Thread(target=run, daemon=True)
    t.start()
    time.sleep(0.3)                     # let _wait_ready reach its poll loop
    t0 = time.monotonic()
    mgr.stop()
    elapsed = time.monotonic() - t0
    t.join(timeout=5)
    assert not t.is_alive()
    assert elapsed < 1.5, f"stop() took {elapsed:.2f}s -- should return almost immediately"
    assert "error" in result and str(result["error"]) == "stopped"
    assert proc.terminated
    assert mgr.fallbacks == []           # no conservative retry, no fall-through after a stop
    assert not mgr.running


def test_start_after_stop_during_load_does_not_resurrect_the_old_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, library: Path
) -> None:
    """A fresh start() called right after a stop()-during-load must proceed normally (not be born
    already "stopped")."""
    _amd_packs(monkeypatch, tmp_path)
    monkeypatch.setattr(srv.gpu, "gpu_vendor", lambda gpus=None: "amd")
    monkeypatch.setattr(srv, "lmstudio_models_dir", lambda cfg=None: library)
    monkeypatch.setattr(srv.httpx, "Client", _AlwaysFailingHttpxClient)

    first_proc = _NeverReadyProc()
    monkeypatch.setattr(srv.subprocess, "Popen", lambda *a, **k: first_proc)
    cfg = LLMConfig(model="maple-preview", llama_device="", llama_startup_timeout_s=180.0)
    mgr = srv.LlamaServerManager(cfg)

    def run_first() -> None:
        try:
            mgr.start()
        except srv.LlamaServerError:
            pass

    t = threading.Thread(target=run_first, daemon=True)
    t.start()
    time.sleep(0.3)
    mgr.stop()
    t.join(timeout=5)
    assert not t.is_alive()

    # Second start(): a genuinely ready server this time (health check succeeds immediately).
    second_proc = _NeverReadyProc()
    monkeypatch.setattr(srv.subprocess, "Popen", lambda *a, **k: second_proc)

    class _ReadyHttpxClient(_AlwaysFailingHttpxClient):
        def get(self, url: str):
            class _R:
                status_code = 200
            return _R()

    monkeypatch.setattr(srv.httpx, "Client", _ReadyHttpxClient)
    url = mgr.start()
    assert url == mgr.base_url
    assert mgr.running
    mgr.stop()                          # clean up: no process left "running"


def test_start_falls_through_after_a_genuine_timeout_without_reporting_stopped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, library: Path
) -> None:
    """docs/UPSTREAM_SUGGESTIONS.md finding #4: _wait_ready()'s own deadline-timeout cleanup used
    to call self.stop(), which set self._stop_event -- the SAME event a caller-requested stop()
    sets. start()'s except-handler then saw that set event on its very next attempt and treated a
    genuine timeout as if the user had asked to quit, aborting the whole fall-through ladder with
    a misleading "stopped" instead of trying the next pack. This drives a REAL _wait_ready() (no
    scripted _attempt) through a never-ready vulkan pack past a short configured timeout, and
    asserts start() falls through to cpu -- not "stopped" -- and the hung vulkan process was
    still torn down.

    Also covers finding #5 in the same run: vendor="intel" makes the conservative retry eligible
    by vendor, so if a genuine timeout were (wrongly) retried instead of just falling through, the
    vulkan flavor would appear twice in Popen's call count / mgr.fallbacks. It must not."""
    backends = _lmstudio_packs(monkeypatch, tmp_path)
    _make_pack(backends, "llama.cpp-win-x86_64-vulkan-avx2-2.46.0", manifest={"version": "2.46.0", "gpu": {"framework": "Vulkan"}})
    _make_pack(backends, "llama.cpp-win-x86_64-avx2-2.29.1", manifest={"version": "2.29.1"})
    monkeypatch.setattr(srv.gpu, "gpu_vendor", lambda gpus=None: "intel")
    monkeypatch.setattr(srv, "lmstudio_models_dir", lambda cfg=None: library)

    hung_proc = _NeverReadyProc()
    ready_proc = _NeverReadyProc()          # "ready" here means the health check returns 200
    popen_calls: list[str] = []

    def fake_popen(*a, **k):
        popen_calls.append("popen")
        return hung_proc if len(popen_calls) == 1 else ready_proc

    monkeypatch.setattr(srv.subprocess, "Popen", fake_popen)

    class _FirstNeverReadyThenReadyClient(_AlwaysFailingHttpxClient):
        _n = 0

        def __init__(self, *a, **k) -> None:
            _FirstNeverReadyThenReadyClient._n += 1
            self._call_index = _FirstNeverReadyThenReadyClient._n

        def get(self, url: str):
            if self._call_index == 1:
                raise srv.httpx.HTTPError("not ready")

            class _R:
                status_code = 200
            return _R()

    monkeypatch.setattr(srv.httpx, "Client", _FirstNeverReadyThenReadyClient)

    cfg = LLMConfig(model="maple-preview", llama_device="", llama_startup_timeout_s=0.3)
    mgr = srv.LlamaServerManager(cfg)
    statuses: list[str] = []
    url = mgr.start(on_status=statuses.append)

    assert url == mgr.base_url
    assert mgr.binary is not None and mgr.binary.flavor == "cpu"
    assert popen_calls == ["popen", "popen"]                 # vulkan tried once, then cpu -- no retry
    assert hung_proc.terminated                              # the hung vulkan process was killed
    assert not mgr._stop_event.is_set()                       # the internal timeout must not look like a stop()
    assert len(mgr.fallbacks) == 1
    assert "did not become ready" in mgr.fallbacks[0]
    assert not any("retrying vulkan" in s for s in statuses)  # no conservative retry on a timeout


def test_attempt_terminates_a_process_spawned_after_a_stop_raced_the_popen_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, library: Path
) -> None:
    """docs/UPSTREAM_SUGGESTIONS.md finding #6: a stop() landing strictly between _attempt()'s
    pre-Popen stop_event check and the locked self._proc assignment used to see self._proc as
    still None (Popen had not returned yet), terminate nothing, and return -- leaving the process
    it just spawned tracked only by self._proc, never cleaned up, while start() reported a clean
    "stopped". This simulates that exact race by calling stop() from inside the (mocked) Popen
    call itself, and asserts the just-spawned process is torn down anyway."""
    _amd_packs(monkeypatch, tmp_path)
    monkeypatch.setattr(srv.gpu, "gpu_vendor", lambda gpus=None: "amd")
    monkeypatch.setattr(srv, "lmstudio_models_dir", lambda cfg=None: library)

    proc = _NeverReadyProc()
    cfg = LLMConfig(model="maple-preview", llama_device="")
    mgr = srv.LlamaServerManager(cfg)

    def fake_popen(*a, **k):
        # A concurrent stop() landing exactly between the pre-Popen check (already passed) and
        # self._proc being assigned below.
        mgr.stop()
        return proc

    monkeypatch.setattr(srv.subprocess, "Popen", fake_popen)

    with pytest.raises(srv.LlamaServerError, match="stopped"):
        mgr.start()
    assert proc.terminated                  # must not be left running, orphaned, tracked nowhere
    assert mgr._proc is None
    assert not mgr.running


def test_find_server_binary_stop_check_raised_between_flavors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """docs/UPSTREAM_SUGGESTIONS.md finding #3 (partial mitigation -- the mid-download case still
    needs a companion change in exocortex.llm.runtimes.ensure_runtime, out of this file's scope):
    find_server_binary() now polls an optional `stop_check` between ladder flavors (and again
    before the deferred "auto" fetch and the final CPU rung), so a stop() landing between two fast,
    local-disk flavor checks is caught immediately instead of only being noticed back in
    LlamaServerManager.start()'s own outer loop."""
    backends = _lmstudio_packs(monkeypatch, tmp_path)
    _make_pack(backends, "llama.cpp-win-x86_64-amd-rocm-avx2-2.29.1", manifest={"version": "2.29.1", "gpu": {"framework": "ROCm"}})
    _make_pack(backends, "llama.cpp-win-x86_64-vulkan-avx2-2.46.0", manifest={"version": "2.46.0", "gpu": {"framework": "Vulkan"}})

    with pytest.raises(srv.LlamaServerError, match="stopped"):
        srv.find_server_binary(vendor="amd", stop_check=lambda: True)


def test_find_server_binary_stop_check_not_polled_when_absent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No `stop_check` given (every existing caller before this fix): behaviour is untouched."""
    backends = _lmstudio_packs(monkeypatch, tmp_path)
    _make_pack(backends, "llama.cpp-win-x86_64-amd-rocm-avx2-2.29.1", manifest={"version": "2.29.1", "gpu": {"framework": "ROCm"}})
    b = srv.find_server_binary(vendor="amd")
    assert b.flavor == "amd-rocm"


# ------------------------------------------------- upstream fetch: cancellation vs failure ----
def test_try_upstream_propagates_a_cancelled_download_as_stopped(tmp_path: Path) -> None:
    def ensure(kind: str, root: Path, build: str) -> Path:
        raise runtimes.UpstreamRuntimeCancelled("quit mid-download")

    with pytest.raises(srv.LlamaServerError) as ei:
        srv._try_upstream("vulkan", tmp_path, "b11200", ensure)
    assert ei.value.args[0] == srv._STOPPED


def test_try_upstream_treats_a_failed_download_or_oserror_as_unavailable(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    def failing(kind: str, root: Path, build: str) -> Path:
        raise runtimes.UpstreamRuntimeError("network down")

    def disk_full(kind: str, root: Path, build: str) -> Path:
        raise OSError(28, "No space left on device")

    with caplog.at_level(logging.WARNING):
        assert srv._try_upstream("vulkan", tmp_path, "b11200", failing) is None
        assert srv._try_upstream("vulkan", tmp_path, "b11200", disk_full) is None
    assert caplog.text.count("could not fetch the upstream vulkan runtime") == 2


def test_start_ensure_closure_passes_the_stop_event_as_the_cancel_hook(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The manager wires runtimes.ensure_runtime(cancelled=self._stop_event.is_set): a stop()
    during the download makes the fetch raise Cancelled, which start() reports as "stopped"."""
    _lmstudio_packs(monkeypatch, tmp_path)         # no LM Studio packs at all
    monkeypatch.setattr(srv.gpu, "gpu_vendor", lambda gpus=None: "intel")
    captured: dict[str, object] = {}

    def fake_ensure_runtime(kind, root, *, build, progress=None, cancelled=None, **kw):
        captured["cancelled"] = cancelled
        # simulate stop() arriving while the bytes stream in
        mgr.stop()
        assert cancelled is not None and cancelled() is True
        raise runtimes.UpstreamRuntimeCancelled("cancelled by stop()")

    monkeypatch.setattr(srv.runtimes, "ensure_runtime", fake_ensure_runtime)
    mgr = srv.LlamaServerManager(LLMConfig(llama_source="upstream", llama_backend="vulkan"))
    monkeypatch.setattr(srv, "project_root", lambda: tmp_path)
    with pytest.raises(srv.LlamaServerError) as ei:
        mgr.start()
    assert ei.value.args[0] == srv._STOPPED
    hook = captured["cancelled"]
    assert getattr(hook, "__self__", None) is mgr._stop_event and hook() is True   # bound to the manager's stop event

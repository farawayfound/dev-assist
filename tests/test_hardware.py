"""Tests for exocortex.hardware: every probe injected (or monkeypatched on the exocortex.gpu
module) so nothing here touches real hardware, the network, or LM Studio -- collect() itself is
pure orchestration over whatever probes it is handed."""
from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import pytest

from exocortex import hardware
from exocortex.config import LLMConfig, project_root
from exocortex.gpu import GpuInfo


def _gpu(**kw: Any) -> GpuInfo:
    base: dict[str, Any] = dict(
        index=0, name="Fake GPU", vendor="nvidia", vendor_id=0x10DE, device_id=0,
        dedicated_mb=16384, shared_mb=0, uma=False, luid="0x00000000_0x00000001", software=False,
    )
    base.update(kw)
    return GpuInfo(**base)


class _FakeServerBinary:
    def __init__(self, **kw: Any) -> None:
        self.exe = kw.get("exe", Path("C:/fake/llama-server.exe"))
        self.flavor = kw.get("flavor", "nvidia-cuda12")
        self.version = kw.get("version", "1.2.3")
        self.source = kw.get("source", "lmstudio")
        self.framework = kw.get("framework", "cuda")
        self.reason = kw.get("reason", "vendor nvidia -> cuda12 (first choice)")


def _no_vram(monkeypatch: pytest.MonkeyPatch) -> None:
    """collect() reads live VRAM through exocortex.gpu.vram_total_mb/vram_free_mb directly (not an
    injectable parameter) -- parked here in every test so nothing touches NVML/PDH."""
    monkeypatch.setattr(hardware.gpu, "vram_total_mb", lambda: 15000)
    monkeypatch.setattr(hardware.gpu, "vram_free_mb", lambda: 14000)


# --------------------------------------------------------------------------------------- collect
def test_collect_with_every_probe_injected(monkeypatch: pytest.MonkeyPatch) -> None:
    _no_vram(monkeypatch)
    fake_gpus = [_gpu(name="RTX 5090", dedicated_mb=32768)]
    binary = _FakeServerBinary()
    report = hardware.collect(
        LLMConfig(),
        detect=lambda: fake_gpus,
        memory=lambda: (65536, 40000),
        cores=lambda: 16,
        driver=lambda gpus: "592.14",
        disk=lambda path: types.SimpleNamespace(total=1_000_000_000_000, used=0, free=500_000_000_000),
        find_binary=lambda *a, **kw: binary,
    )
    assert isinstance(report.cpu_name, str)  # cpu_name() is real (not injectable); just must not raise
    assert report.physical_cores == 16
    assert report.logical_cores is not None
    assert report.ram_total_mb == 65536
    assert report.ram_free_mb == 40000
    assert report.gpus == [
        {
            "index": 0, "name": "RTX 5090", "vendor": "nvidia", "vendor_id": 0x10DE, "device_id": 0,
            "dedicated_mb": 32768, "shared_mb": 0, "uma": False, "luid": "0x00000000_0x00000001",
            "software": False,
        }
    ]
    assert report.primary_index == 0
    assert report.vendor == "nvidia"
    assert report.budget_mb == 32768
    assert report.vram_total_mb == 15000
    assert report.vram_free_mb == 14000
    assert report.nvidia_driver == "592.14"
    assert report.library_free_mb == 500_000_000_000 // (1024 * 1024)
    assert report.runtime == {
        "flavor": "nvidia-cuda12", "version": "1.2.3", "source": "lmstudio",
        "exe": str(Path("C:/fake/llama-server.exe")), "framework": "cuda",
        "reason": "vendor nvidia -> cuda12 (first choice)",
    }
    assert report.recommended_policy == "quality"          # 32768 MiB >= 20480
    assert report.generated_at
    assert isinstance(report.notes, list)


def test_collect_runtime_probe_failure_becomes_a_note_not_a_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    _no_vram(monkeypatch)

    def raising_find_binary(*_a: Any, **_kw: Any) -> Any:
        raise RuntimeError("no llama.cpp runtime pack for a nvidia GPU found")

    report = hardware.collect(
        LLMConfig(),
        detect=lambda: [],
        memory=lambda: (None, None),
        cores=lambda: None,
        driver=lambda gpus: None,
        disk=lambda path: types.SimpleNamespace(total=0, used=0, free=0),
        find_binary=raising_find_binary,
    )
    assert report.runtime is None
    assert any("no llama.cpp runtime" in n for n in report.notes)


def test_collect_probe_runtime_false_skips_the_runtime_probe_entirely(monkeypatch: pytest.MonkeyPatch) -> None:
    _no_vram(monkeypatch)

    def must_not_be_called(*_a: Any, **_kw: Any) -> Any:
        raise AssertionError("find_binary must not be called when probe_runtime=False")

    report = hardware.collect(
        LLMConfig(), probe_runtime=False,
        detect=lambda: [], memory=lambda: (None, None), cores=lambda: None, driver=lambda gpus: None,
        disk=lambda path: types.SimpleNamespace(total=0, used=0, free=0),
        find_binary=must_not_be_called,
    )
    assert report.runtime is None
    assert not any("runtime" in n.lower() for n in report.notes)


def test_collect_wraps_a_raising_detect_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    _no_vram(monkeypatch)

    def raising_detect() -> list[GpuInfo]:
        raise OSError("DXGI enumeration failed")

    report = hardware.collect(
        LLMConfig(), probe_runtime=False,
        detect=raising_detect, memory=lambda: (None, None), cores=lambda: None, driver=lambda gpus: None,
        disk=lambda path: types.SimpleNamespace(total=0, used=0, free=0),
    )
    assert report.gpus == []
    assert report.primary_index is None
    assert report.budget_mb is None
    assert any("GPU detection failed" in n for n in report.notes)


def test_collect_wraps_a_raising_memory_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    _no_vram(monkeypatch)

    def raising_memory() -> tuple[int | None, int | None]:
        raise OSError("GlobalMemoryStatusEx failed")

    report = hardware.collect(
        LLMConfig(), probe_runtime=False,
        detect=lambda: [], memory=raising_memory, cores=lambda: None, driver=lambda gpus: None,
        disk=lambda path: types.SimpleNamespace(total=0, used=0, free=0),
    )
    assert report.ram_total_mb is None
    assert report.ram_free_mb is None
    assert any("system_memory_mb" in n for n in report.notes)


@pytest.mark.parametrize("budget_mb,expected", [
    (20480, "quality"), (30000, "quality"), (6144, "auto"), (10000, "auto"),
    (6143, "fit"), (1024, "fit"), (None, "fit"),
])
def test_recommended_policy_bands(budget_mb: int | None, expected: str) -> None:
    assert hardware._recommended_policy(budget_mb) == expected


def test_collect_recommended_policy_matches_bands(monkeypatch: pytest.MonkeyPatch) -> None:
    _no_vram(monkeypatch)
    fake_gpus = [_gpu(dedicated_mb=8000, shared_mb=0, uma=False)]
    report = hardware.collect(
        LLMConfig(), probe_runtime=False,
        detect=lambda: fake_gpus, memory=lambda: (None, None), cores=lambda: None, driver=lambda gpus: None,
        disk=lambda path: types.SimpleNamespace(total=0, used=0, free=0),
    )
    assert report.budget_mb == 8000
    assert report.recommended_policy == "auto"


# ------------------------------------------------------------------------------- library_dir
def test_resolve_library_dir_falls_back_when_library_module_is_not_importable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``sys.modules[name] = None`` is the standard way to force an ImportError deterministically,
    regardless of whether exocortex.llm.library actually exists on disk yet (a parallel branch)."""
    monkeypatch.setitem(sys.modules, "exocortex.llm.library", None)
    notes: list[str] = []
    path = hardware._resolve_library_dir(LLMConfig(), notes)
    assert path == project_root() / "models" / "llm"
    assert any("not importable" in n for n in notes)


def test_resolve_library_dir_uses_the_library_module_when_present(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_mod = types.SimpleNamespace(library_dir=lambda cfg: Path("Z:/fake/library"))
    import exocortex.llm as llm_pkg

    monkeypatch.setattr(llm_pkg, "library", fake_mod, raising=False)
    monkeypatch.setitem(sys.modules, "exocortex.llm.library", fake_mod)
    notes: list[str] = []
    path = hardware._resolve_library_dir(LLMConfig(), notes)
    assert path == Path("Z:/fake/library")
    assert notes == []


def test_resolve_library_dir_tolerates_library_dir_raising(monkeypatch: pytest.MonkeyPatch) -> None:
    def bad_library_dir(cfg: LLMConfig) -> Path:
        raise ValueError("boom")

    fake_mod = types.SimpleNamespace(library_dir=bad_library_dir)
    import exocortex.llm as llm_pkg

    monkeypatch.setattr(llm_pkg, "library", fake_mod, raising=False)
    monkeypatch.setitem(sys.modules, "exocortex.llm.library", fake_mod)
    notes: list[str] = []
    path = hardware._resolve_library_dir(LLMConfig(), notes)
    assert path == project_root() / "models" / "llm"
    assert any("library.library_dir() failed" in n for n in notes)


def test_report_path_is_under_the_library_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "exocortex.llm.library", None)
    assert hardware.report_path(LLMConfig()) == project_root() / "models" / "llm" / "hardware.json"


# ------------------------------------------------------------------------- to_dict/save/load
def test_to_dict_save_load_round_trip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _no_vram(monkeypatch)
    report = hardware.collect(
        LLMConfig(), probe_runtime=False,
        detect=lambda: [_gpu()], memory=lambda: (32768, 16000), cores=lambda: 8, driver=lambda gpus: None,
        disk=lambda path: types.SimpleNamespace(total=0, used=0, free=0),
    )
    d = hardware.to_dict(report)
    assert d["ram_total_mb"] == 32768
    assert d["gpus"][0]["name"] == "Fake GPU"

    out = hardware.save(report, tmp_path / "sub" / "hardware.json")
    assert out.is_file()
    loaded = hardware.load(out)
    assert loaded is not None
    assert loaded == report


def test_load_returns_none_for_a_missing_file(tmp_path: Path) -> None:
    assert hardware.load(tmp_path / "nope.json") is None


def test_load_returns_none_for_corrupt_json(tmp_path: Path) -> None:
    p = tmp_path / "bad.json"
    p.write_text("not json at all {{{", encoding="utf-8")
    assert hardware.load(p) is None


def test_load_ignores_unknown_keys_from_a_future_schema(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _no_vram(monkeypatch)
    report = hardware.collect(
        LLMConfig(), probe_runtime=False,
        detect=lambda: [], memory=lambda: (None, None), cores=lambda: None, driver=lambda gpus: None,
        disk=lambda path: types.SimpleNamespace(total=0, used=0, free=0),
    )
    d = hardware.to_dict(report)
    d["some_future_field"] = "unexpected"
    p = tmp_path / "future.json"
    import json

    p.write_text(json.dumps(d), encoding="utf-8")
    loaded = hardware.load(p)
    assert loaded is not None
    assert loaded.vendor == report.vendor


# --------------------------------------------------------------------------------------- describe
def test_describe_contains_the_key_numbers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hardware.gpu, "vram_total_mb", lambda: 32768)
    monkeypatch.setattr(hardware.gpu, "vram_free_mb", lambda: 30000)
    report = hardware.collect(
        LLMConfig(), probe_runtime=False,
        detect=lambda: [_gpu(name="RTX 5090", dedicated_mb=32768)],
        memory=lambda: (65536, 40000), cores=lambda: 16, driver=lambda gpus: "592.14",
        disk=lambda path: types.SimpleNamespace(total=0, used=0, free=2_000_000_000),
    )
    text = hardware.describe(report)
    assert "RTX 5090" in text
    assert "nvidia" in text
    assert "32768" in text            # budget_mb
    assert "30000" in text            # vram_free_mb
    assert "592.14" in text           # nvidia_driver
    assert "65536" in text            # ram_total_mb
    assert "quality" in text          # recommended_policy for a 32 GB budget
    assert "none found yet" in text   # runtime probe was skipped


def test_describe_with_no_gpu_and_notes() -> None:
    report = hardware.HardwareReport(
        os="Windows-11", cpu_name="", physical_cores=None, logical_cores=None, ram_total_mb=None,
        ram_free_mb=None, gpus=[], primary_index=None, vendor="none", budget_mb=None,
        vram_total_mb=None, vram_free_mb=None, nvidia_driver=None, library_dir="C:/models/llm",
        library_free_mb=None, runtime=None, recommended_policy="fit", generated_at="2026-09-27T00:00:00",
        notes=["cpu_name() failed: boom"],
    )
    text = hardware.describe(report)
    assert "none detected" in text
    assert "note: cpu_name() failed: boom" in text


# -------------------------------------------------------------------------------------------- CLI
def test_main_default_prints_describe(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    fake_report = hardware.HardwareReport(
        os="Windows-11", cpu_name="Fake CPU", physical_cores=8, logical_cores=16, ram_total_mb=32768,
        ram_free_mb=16000, gpus=[], primary_index=None, vendor="none", budget_mb=None, vram_total_mb=None,
        vram_free_mb=None, nvidia_driver=None, library_dir="C:/models/llm", library_free_mb=None,
        runtime=None, recommended_policy="fit", generated_at="2026-09-27T00:00:00", notes=[],
    )
    from exocortex.config import Config

    monkeypatch.setattr("exocortex.config.load_config", lambda: Config())
    monkeypatch.setattr(hardware, "collect", lambda cfg: fake_report)
    rc = hardware.main([])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Fake CPU" in out


def test_main_json_prints_to_dict(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    fake_report = hardware.HardwareReport(
        os="Windows-11", cpu_name="Fake CPU", physical_cores=8, logical_cores=16, ram_total_mb=32768,
        ram_free_mb=16000, gpus=[], primary_index=None, vendor="none", budget_mb=None, vram_total_mb=None,
        vram_free_mb=None, nvidia_driver=None, library_dir="C:/models/llm", library_free_mb=None,
        runtime=None, recommended_policy="fit", generated_at="2026-09-27T00:00:00", notes=[],
    )
    from exocortex.config import Config

    monkeypatch.setattr("exocortex.config.load_config", lambda: Config())
    monkeypatch.setattr(hardware, "collect", lambda cfg: fake_report)
    rc = hardware.main(["--json"])
    out = capsys.readouterr().out.strip()
    assert rc == 0
    import json

    payload = json.loads(out)
    assert payload["cpu_name"] == "Fake CPU"


def test_main_save_writes_the_report(monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
                                       capsys: pytest.CaptureFixture) -> None:
    fake_report = hardware.HardwareReport(
        os="Windows-11", cpu_name="Fake CPU", physical_cores=8, logical_cores=16, ram_total_mb=32768,
        ram_free_mb=16000, gpus=[], primary_index=None, vendor="none", budget_mb=None, vram_total_mb=None,
        vram_free_mb=None, nvidia_driver=None, library_dir="C:/models/llm", library_free_mb=None,
        runtime=None, recommended_policy="fit", generated_at="2026-09-27T00:00:00", notes=[],
    )
    from exocortex.config import Config

    monkeypatch.setattr("exocortex.config.load_config", lambda: Config())
    monkeypatch.setattr(hardware, "collect", lambda cfg: fake_report)
    saved_to = tmp_path / "hardware.json"
    monkeypatch.setattr(hardware, "report_path", lambda cfg: saved_to)
    rc = hardware.main(["--save"])
    assert rc == 0
    assert saved_to.is_file()


def test_main_never_fails_on_a_bad_flag() -> None:
    assert hardware.main(["--nonsense"]) == 0


def test_runtime_probe_passes_a_custom_runtime_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    """2026-09-27 review follow-up: a build `python -m exocortex.setup` installed under a custom
    [llm].runtime_dir was invisible to the report (it said "none found yet") because the probe
    left find_server_binary on its default root."""
    captured: dict = {}

    def fake_find(explicit: str, **kw: object) -> object:
        captured.update(kw)
        raise RuntimeError("no runtime")

    cfg = LLMConfig(runtime_dir="X:/custom/rt")
    report = hardware.collect(cfg, detect=lambda: [], memory=lambda: (None, None), cores=lambda: None,
                              driver=lambda gpus=None: None, find_binary=fake_find)
    assert captured["upstream_root"] == Path("X:/custom/rt")
    assert report.runtime is None
    captured.clear()
    hardware.collect(LLMConfig(), detect=lambda: [], memory=lambda: (None, None), cores=lambda: None,
                     driver=lambda gpus=None: None, find_binary=fake_find)
    assert "upstream_root" not in captured

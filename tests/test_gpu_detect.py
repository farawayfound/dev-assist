"""Pure-logic tests for exocortex/gpu.py's vendor-neutral discovery (DXGI-free: every GpuInfo is
synthetic here) plus the NVML-then-PDH vram_free_mb fall-through and the CLI. Discovery itself
(DXGI enumeration, on this OS) gets one non-live smoke test at the bottom; everything else injects
GpuInfo lists or monkeypatches the NVML/PDH state so it passes on any machine."""
from __future__ import annotations

import json
import re
from typing import Any

import pytest

from exocortex import gpu
from exocortex.gpu import GpuInfo


def _gpu(**kw: Any) -> GpuInfo:
    base: dict[str, Any] = dict(
        index=0, name="GPU", vendor="other", vendor_id=0, device_id=0,
        dedicated_mb=0, shared_mb=0, uma=False, luid="0x00000000_0x00000000", software=False,
    )
    base.update(kw)
    return GpuInfo(**base)


# ------------------------------------------------------------------------------- primary_gpu ---
def test_nvidia_discrete_beats_an_intel_uma_igpu() -> None:
    nvidia = _gpu(index=0, name="RTX 5090", vendor="nvidia", vendor_id=0x10DE, dedicated_mb=16384,
                  uma=False, luid="0x00000000_0x00000001")
    intel = _gpu(index=1, name="Intel Arc", vendor="intel", vendor_id=0x8086, dedicated_mb=128,
                 shared_mb=8192, uma=True, luid="0x00000000_0x00000002")
    gpus = [nvidia, intel]
    assert gpu.primary_gpu(gpus) is nvidia
    assert gpu.gpu_vendor(gpus) == "nvidia"


def test_amd_uma_only_is_primary() -> None:
    amd = _gpu(index=0, name="AMD Radeon(TM) 8060S Graphics", vendor="amd", vendor_id=0x1002,
               dedicated_mb=32590, shared_mb=16203, uma=True, luid="0x00000000_0x00016a0e")
    assert gpu.primary_gpu([amd]) is amd
    assert gpu.gpu_vendor([amd]) == "amd"


def test_amd_discrete_beats_amd_uma() -> None:
    igpu = _gpu(index=0, name="AMD 780M", vendor="amd", dedicated_mb=512, shared_mb=8192, uma=True,
                luid="0x00000000_0x00000001")
    discrete = _gpu(index=1, name="Radeon RX 7900", vendor="amd", dedicated_mb=20480, uma=False,
                     luid="0x00000000_0x00000002")
    assert gpu.primary_gpu([igpu, discrete]) is discrete
    assert gpu.primary_gpu([discrete, igpu]) is discrete


def test_two_discrete_gpus_the_bigger_one_wins() -> None:
    small = _gpu(index=0, name="GTX 1660", vendor="nvidia", dedicated_mb=6144, uma=False,
                 luid="0x00000000_0x00000001")
    big = _gpu(index=1, name="RTX 5090", vendor="nvidia", dedicated_mb=32768, uma=False,
               luid="0x00000000_0x00000002")
    assert gpu.primary_gpu([small, big]) is big
    assert gpu.primary_gpu([big, small]) is big


def test_software_adapters_are_ignored() -> None:
    real = _gpu(index=0, name="AMD Radeon(TM) 8060S Graphics", vendor="amd", dedicated_mb=32590,
                uma=True, luid="0x00000000_0x00016a0e")
    warp = _gpu(index=1, name="Microsoft Basic Render Driver", vendor="microsoft", dedicated_mb=0,
                uma=True, luid="0x00000000_0x00000003", software=True)
    assert gpu.primary_gpu([real, warp]) is real
    assert gpu.primary_gpu([warp]) is None


def test_empty_gpu_list() -> None:
    assert gpu.primary_gpu([]) is None
    assert gpu.gpu_vendor([]) == "none"


# --------------------------------------------------------------------------------- budget_mb ---
def test_budget_mb_adds_shared_only_for_uma() -> None:
    igpu = _gpu(dedicated_mb=32590, shared_mb=16203, uma=True)
    discrete = _gpu(dedicated_mb=16384, shared_mb=4096, uma=False)
    assert igpu.budget_mb == 32590 + 16203
    assert discrete.budget_mb == 16384


# ----------------------------------------------------------------------------------- describe ---
def test_describe_marks_the_primary() -> None:
    a = _gpu(index=0, name="Intel Arc", vendor="intel", dedicated_mb=128, uma=True,
             luid="0x00000000_0x00000001")
    b = _gpu(index=1, name="RTX 5090", vendor="nvidia", dedicated_mb=32768, uma=False,
             luid="0x00000000_0x00000002")
    lines = gpu.describe([a, b]).splitlines()
    assert "<- primary" not in lines[0]
    assert "<- primary" in lines[1]


def test_describe_with_no_gpus() -> None:
    assert "no GPU adapter" in gpu.describe([])


def test_describe_includes_driver_when_known(monkeypatch: pytest.MonkeyPatch) -> None:
    a = _gpu(index=0, name="Intel Arc(TM) B580 Graphics", vendor="intel", vendor_id=0x8086, device_id=0xE20B,
             dedicated_mb=12288, uma=False)
    monkeypatch.setattr(gpu, "driver_version", lambda g, **_kw: "32.0.101.6795")
    assert "driver 32.0.101.6795" in gpu.describe([a])


def test_describe_omits_driver_clause_when_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    a = _gpu(index=0, name="Mystery GPU", vendor="other")
    monkeypatch.setattr(gpu, "driver_version", lambda g, **_kw: None)
    assert "driver" not in gpu.describe([a])


def test_describe_never_shows_driver_for_amd_or_nvidia_even_when_a_driver_is_found(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression for the parity bug: describe() must stay byte-identical to before driver_version
    existed for every non-Intel vendor, even when the registry lookup would succeed (as it does on
    this real AMD dev box: driver_version() is never even reached for these adapters)."""
    amd = _gpu(index=0, name="AMD Radeon(TM) 8060S Graphics", vendor="amd", dedicated_mb=32590,
               shared_mb=16203, uma=True)
    nvidia = _gpu(index=1, name="RTX 5080", vendor="nvidia", dedicated_mb=16384, uma=False,
                  luid="0x00000000_0x00000002")

    def fake_driver_version(g: GpuInfo, **_kw: Any) -> str:
        raise AssertionError(f"driver_version must not be called for vendor={g.vendor!r}")

    monkeypatch.setattr(gpu, "driver_version", fake_driver_version)
    text = gpu.describe([amd, nvidia])
    assert "driver" not in text
    assert "AMD Radeon(TM) 8060S Graphics [amd, integrated/UMA, 32590 MiB dedicated + 16203 MiB shared]" in text
    assert "RTX 5080 [nvidia, discrete, 16384 MiB dedicated + 0 MiB shared]  <- primary" in text


def test_describe_disambiguates_two_identical_intel_arc_cards_by_rank(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two Intel Arc cards of the identical model (same vendor_id/device_id) must not be shown the
    same driver string -- describe() ranks each same-model adapter among its siblings and passes
    that rank through to driver_version()."""
    card0 = _gpu(index=0, name="Intel Arc(TM) B580 Graphics", vendor="intel", vendor_id=0x8086,
                 device_id=0xE20B, dedicated_mb=12288, uma=False, luid="0x00000000_0x00000001")
    card1 = _gpu(index=1, name="Intel Arc(TM) B580 Graphics", vendor="intel", vendor_id=0x8086,
                 device_id=0xE20B, dedicated_mb=12288, uma=False, luid="0x00000000_0x00000002")

    def fake_driver_version(g: GpuInfo, *, rank: int = 0, **_kw: Any) -> str:
        return f"driver-for-rank-{rank}"

    monkeypatch.setattr(gpu, "driver_version", fake_driver_version)
    lines = gpu.describe([card0, card1]).splitlines()
    assert "driver-for-rank-0" in lines[0]
    assert "driver-for-rank-1" in lines[1]


# ---------------------------------------------------------------------------- driver_version ---
def _reader(subkeys: list[tuple[str, dict[str, str]]]) -> Any:
    return lambda: list(subkeys)


def test_driver_version_matches_by_vendor_and_device_id() -> None:
    g = _gpu(vendor="intel", vendor_id=0x8086, device_id=0x56A0)
    reader = _reader([
        ("0000", {"MatchingDeviceID": "PCI\\VEN_10DE&DEV_2882&SUBSYS_00001458", "DriverVersion": "31.0.15.5222"}),
        ("0001", {"MatchingDeviceID": "PCI\\VEN_8086&DEV_56A0&SUBSYS_00000000", "DriverVersion": "32.0.101.6795"}),
    ])
    assert gpu.driver_version(g, reg_reader=reader) == "32.0.101.6795"


def test_driver_version_is_case_insensitive_on_both_sides() -> None:
    g = _gpu(vendor="intel", vendor_id=0x8086, device_id=0x56A0)
    reader = _reader([("0000", {"matchingdeviceid": "pci\\ven_8086&dev_56a0", "driverversion": "32.0.101.6795"})])
    assert gpu.driver_version(g, reg_reader=reader) == "32.0.101.6795"


def test_driver_version_checks_deviceinstanceid_field() -> None:
    g = _gpu(vendor="amd", vendor_id=0x1002, device_id=0x1900)
    reader = _reader([("0000", {"DeviceInstanceID": "PCI\\VEN_1002&DEV_1900&SUBSYS_00000000",
                                "DriverVersion": "32.0.31032.1003"})])
    assert gpu.driver_version(g, reg_reader=reader) == "32.0.31032.1003"


def test_driver_version_checks_hardwareinformation_star_fields() -> None:
    g = _gpu(vendor="amd", vendor_id=0x1002, device_id=0x1900)
    reader = _reader([("0000", {"HardwareInformation.AdapterString": "AMD VEN_1002&DEV_1900 device",
                                "DriverVersion": "32.0.31032.1003"})])
    assert gpu.driver_version(g, reg_reader=reader) == "32.0.31032.1003"


def test_driver_version_returns_none_when_no_subkey_matches() -> None:
    g = _gpu(vendor="nvidia", vendor_id=0x10DE, device_id=0x2882)
    reader = _reader([("0000", {"MatchingDeviceID": "PCI\\VEN_8086&DEV_1234", "DriverVersion": "1.2.3.4"})])
    assert gpu.driver_version(g, reg_reader=reader) is None


def test_driver_version_returns_none_when_matching_subkey_has_no_driverversion() -> None:
    g = _gpu(vendor_id=0x8086, device_id=0x56A0)
    reader = _reader([("0000", {"MatchingDeviceID": "VEN_8086&DEV_56A0"})])
    assert gpu.driver_version(g, reg_reader=reader) is None


def test_driver_version_returns_none_on_reader_exception() -> None:
    g = _gpu(vendor_id=0x1234, device_id=0x5678)

    def bad_reader() -> Any:
        raise OSError("registry access denied")

    assert gpu.driver_version(g, reg_reader=bad_reader) is None


def test_driver_version_injected_reader_bypasses_the_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """An injected `reg_reader` must never read or write `_driver_cache` -- otherwise one test's
    fake answer could leak into another test (or the real machine) for the same ids."""
    monkeypatch.setitem(gpu._driver_cache, (0xAAAA, 0xBBBB, 0), "stale-cached-value")
    g = _gpu(vendor_id=0xAAAA, device_id=0xBBBB)
    reader = _reader([("0000", {"MatchingDeviceID": "VEN_AAAA&DEV_BBBB", "DriverVersion": "9.9.9.9"})])
    assert gpu.driver_version(g, reg_reader=reader) == "9.9.9.9"
    assert gpu._driver_cache[(0xAAAA, 0xBBBB, 0)] == "stale-cached-value"   # untouched by the injected call


def test_driver_version_default_reader_is_cached_per_vendor_device(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(gpu._driver_cache, (0x1111, 0x2222, 0), raising=False)
    calls: list[int] = []

    def counting_reader() -> Any:
        calls.append(1)
        return [("0000", {"MatchingDeviceID": "VEN_1111&DEV_2222", "DriverVersion": "5.5.5.5"})]

    monkeypatch.setattr(gpu, "_registry_subkeys", counting_reader)
    g = _gpu(vendor_id=0x1111, device_id=0x2222)
    assert gpu.driver_version(g) == "5.5.5.5"
    assert gpu.driver_version(g) == "5.5.5.5"
    assert len(calls) == 1


def test_driver_version_rank_selects_among_multiple_matching_subkeys() -> None:
    """Two subkeys matching the same vendor/device id (two physically distinct adapters of the
    identical model) -- `rank` picks which one, instead of always the first."""
    g = _gpu(vendor_id=0x8086, device_id=0xE20B)
    reader = _reader([
        ("0000", {"MatchingDeviceID": "PCI\\VEN_8086&DEV_E20B&SUBSYS_0001", "DriverVersion": "32.0.101.6789"}),
        ("0001", {"MatchingDeviceID": "PCI\\VEN_8086&DEV_E20B&SUBSYS_0002", "DriverVersion": "32.0.101.6795"}),
    ])
    assert gpu.driver_version(g, reg_reader=reader, rank=0) == "32.0.101.6789"
    assert gpu.driver_version(g, reg_reader=reader, rank=1) == "32.0.101.6795"


def test_driver_version_rank_beyond_available_matches_clamps_to_the_last_one() -> None:
    g = _gpu(vendor_id=0x8086, device_id=0xE20B)
    reader = _reader([("0000", {"MatchingDeviceID": "VEN_8086&DEV_E20B", "DriverVersion": "32.0.101.6789"})])
    assert gpu.driver_version(g, reg_reader=reader, rank=5) == "32.0.101.6789"


def test_driver_version_cache_key_includes_rank(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two adapters that share (vendor_id, device_id) but have different `rank` must not read or
    write the same cache slot -- otherwise the second identical adapter would be served the
    first's cached answer forever (the bug this disambiguation fixes)."""
    monkeypatch.delitem(gpu._driver_cache, (0x9999, 0x8888, 0), raising=False)
    monkeypatch.delitem(gpu._driver_cache, (0x9999, 0x8888, 1), raising=False)

    def reader() -> Any:
        return [
            ("0000", {"MatchingDeviceID": "VEN_9999&DEV_8888", "DriverVersion": "1.1.1.1"}),
            ("0001", {"MatchingDeviceID": "VEN_9999&DEV_8888", "DriverVersion": "2.2.2.2"}),
        ]

    monkeypatch.setattr(gpu, "_registry_subkeys", reader)
    g = _gpu(vendor_id=0x9999, device_id=0x8888)
    assert gpu.driver_version(g, rank=0) == "1.1.1.1"
    assert gpu.driver_version(g, rank=1) == "2.2.2.2"
    assert gpu._driver_cache[(0x9999, 0x8888, 0)] == "1.1.1.1"
    assert gpu._driver_cache[(0x9999, 0x8888, 1)] == "2.2.2.2"


# --------------------------------------------------------------------------- intel_arc_notes ---
def test_intel_arc_notes_empty_for_non_intel_vendor() -> None:
    amd = _gpu(vendor="amd", vendor_id=0x1002, device_id=0x1900, uma=False)
    assert gpu.intel_arc_notes(amd, "32.0.31032.1003") == []


def test_intel_arc_notes_empty_for_an_intel_integrated_gpu() -> None:
    """The Arc 140V/140T iGPUs already get ordinary vendor='intel' Vulkan routing; no special-case
    notes for them here."""
    igpu = _gpu(vendor="intel", vendor_id=0x8086, device_id=0x64A0, name="Intel Arc 140T Graphics", uma=True)
    assert gpu.intel_arc_notes(igpu, "32.0.101.6795") == []


def test_intel_arc_notes_always_recommends_vulkan() -> None:
    b580 = _gpu(vendor="intel", vendor_id=0x8086, device_id=0xE20B, name="Intel Arc(TM) B580 Graphics",
                dedicated_mb=12288, uma=False)
    notes = gpu.intel_arc_notes(b580, "32.0.101.6795")
    assert any("vulkan" in n.lower() and "recommended" in n.lower() for n in notes)


def test_intel_arc_notes_always_mentions_untested_hardware() -> None:
    a750 = _gpu(vendor="intel", vendor_id=0x8086, device_id=0x56A1, name="Intel Arc(TM) A750 Graphics",
                dedicated_mb=8192, uma=False)
    notes = gpu.intel_arc_notes(a750, None)
    assert any("no intel gpu was available to test" in n.lower() for n in notes)


def test_intel_arc_notes_battlemage_old_driver_warns_to_update() -> None:
    b580 = _gpu(vendor="intel", vendor_id=0x8086, device_id=0xE20B, name="Intel Arc(TM) B580 Graphics",
                dedicated_mb=12288, uma=False)
    notes = gpu.intel_arc_notes(b580, "31.0.101.5522")
    assert any("update the intel arc driver" in n.lower() for n in notes)


def test_intel_arc_notes_battlemage_new_driver_has_no_update_warning() -> None:
    b580 = _gpu(vendor="intel", vendor_id=0x8086, device_id=0xE20B, name="Intel Arc(TM) B580 Graphics",
                dedicated_mb=12288, uma=False)
    notes = gpu.intel_arc_notes(b580, "32.0.101.6795")
    assert not any("update the intel arc driver" in n.lower() for n in notes)
    assert any("cooperative-matrix" in n.lower() for n in notes)


def test_intel_arc_notes_battlemage_unknown_driver_gets_a_neutral_note() -> None:
    b580 = _gpu(vendor="intel", vendor_id=0x8086, device_id=0xE20B, name="Intel Arc(TM) B580 Graphics",
                dedicated_mb=12288, uma=False)
    notes = gpu.intel_arc_notes(b580, None)
    assert not any("update the intel arc driver" in n.lower() for n in notes)
    assert any("could not read the driver version" in n.lower() for n in notes)


def test_intel_arc_notes_identifies_battlemage_by_device_id_when_name_is_generic() -> None:
    generic = _gpu(vendor="intel", vendor_id=0x8086, device_id=0xE20B, name="Intel Graphics",
                    dedicated_mb=12288, uma=False)
    notes = gpu.intel_arc_notes(generic, "31.0.101.5522")
    assert any("update the intel arc driver" in n.lower() for n in notes)   # only reachable via the id fallback


def test_intel_arc_notes_alchemist_mentions_disable_f16() -> None:
    a770 = _gpu(vendor="intel", vendor_id=0x8086, device_id=0x56A0, name="Intel Arc(TM) A770 Graphics",
                dedicated_mb=16384, uma=False)
    notes = gpu.intel_arc_notes(a770, "31.0.101.5378")
    assert any("GGML_VK_DISABLE_F16" in n for n in notes)
    assert not any("cooperative-matrix" in n.lower() for n in notes)   # that's an Xe2/Battlemage note, not Alchemist


def test_intel_arc_notes_identifies_alchemist_by_device_id_when_name_is_generic() -> None:
    generic = _gpu(vendor="intel", vendor_id=0x8086, device_id=0x56A5, name="Intel Graphics",
                    dedicated_mb=16384, uma=False)
    notes = gpu.intel_arc_notes(generic, None)
    assert any("GGML_VK_DISABLE_F16" in n for n in notes)


def test_intel_arc_notes_budget_line_scales_with_dedicated_mb() -> None:
    twelve_gb = _gpu(vendor="intel", vendor_id=0x8086, device_id=0xE20B, name="Intel Arc(TM) B580 Graphics",
                     dedicated_mb=12288, uma=False)
    assert any("~9 GB" in n for n in gpu.intel_arc_notes(twelve_gb, "32.0.101.6795"))
    pro_b70 = _gpu(vendor="intel", vendor_id=0x8086, device_id=0xE211, name="Intel Arc(TM) Pro B70 Graphics",
                   dedicated_mb=32768, uma=False)
    assert any("~24 GB" in n for n in gpu.intel_arc_notes(pro_b70, "32.0.101.6795"))


# ----------------------------------------------------------------------------- extras_for_vendor
@pytest.mark.parametrize("vendor,expected", [
    ("nvidia", ["requirements-nvidia.txt"]),
    ("amd", ["requirements-directml.txt"]),
    ("intel", []),
    ("qualcomm", []),
    ("other", []),
    ("none", []),
])
def test_extras_for_vendor(vendor: str, expected: list[str]) -> None:
    assert gpu.extras_for_vendor(vendor) == expected


# ------------------------------------------------------------------------------------- CLI main
def test_main_vendor_prints_exactly_one_word(monkeypatch: pytest.MonkeyPatch,
                                              capsys: pytest.CaptureFixture) -> None:
    amd = _gpu(vendor="amd", dedicated_mb=32590, uma=True)
    monkeypatch.setattr(gpu, "detect_gpus", lambda **_k: [amd])
    rc = gpu.main(["--vendor"])
    out = capsys.readouterr().out.strip()
    assert rc == 0
    assert out == "amd"
    assert len(out.split()) == 1


def test_main_vendor_none_when_no_gpu(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    monkeypatch.setattr(gpu, "detect_gpus", lambda **_k: [])
    rc = gpu.main(["--vendor"])
    assert rc == 0
    assert capsys.readouterr().out.strip() == "none"


def test_main_json_is_valid_and_carries_primary_and_vendor(monkeypatch: pytest.MonkeyPatch,
                                                             capsys: pytest.CaptureFixture) -> None:
    a = _gpu(index=0, name="AMD Radeon(TM) 8060S Graphics", vendor="amd", dedicated_mb=32590,
             shared_mb=16203, uma=True, luid="0x00000000_0x00016a0e")
    monkeypatch.setattr(gpu, "detect_gpus", lambda **_k: [a])
    rc = gpu.main(["--json"])
    out = capsys.readouterr().out.strip()
    assert rc == 0
    payload = json.loads(out)
    assert payload["vendor"] == "amd"
    assert payload["primary"] == 0
    assert payload["adapters"][0]["name"] == a.name
    assert payload["adapters"][0]["dedicated_mb"] == 32590


def test_main_json_primary_is_none_when_no_gpu(monkeypatch: pytest.MonkeyPatch,
                                                capsys: pytest.CaptureFixture) -> None:
    monkeypatch.setattr(gpu, "detect_gpus", lambda **_k: [])
    rc = gpu.main(["--json"])
    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload == {"adapters": [], "primary": None, "vendor": "none"}


def test_main_json_includes_driver_per_adapter(monkeypatch: pytest.MonkeyPatch,
                                                capsys: pytest.CaptureFixture) -> None:
    a = _gpu(index=0, name="Intel Arc(TM) B580 Graphics", vendor="intel", vendor_id=0x8086, device_id=0xE20B,
             dedicated_mb=12288, uma=False)
    monkeypatch.setattr(gpu, "detect_gpus", lambda **_k: [a])
    monkeypatch.setattr(gpu, "driver_version", lambda g, **_kw: "32.0.101.6795")
    rc = gpu.main(["--json"])
    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload["adapters"][0]["driver"] == "32.0.101.6795"


def test_main_json_omits_driver_key_for_amd_and_nvidia(monkeypatch: pytest.MonkeyPatch,
                                                        capsys: pytest.CaptureFixture) -> None:
    """Regression for the parity bug: the --json adapter dict must not gain a "driver" key for any
    non-Intel vendor, even when the registry lookup would succeed."""
    amd = _gpu(index=0, name="AMD Radeon(TM) 8060S Graphics", vendor="amd", dedicated_mb=32590,
               shared_mb=16203, uma=True)

    def fake_driver_version(g: GpuInfo, **_kw: Any) -> str:
        raise AssertionError(f"driver_version must not be called for vendor={g.vendor!r}")

    monkeypatch.setattr(gpu, "detect_gpus", lambda **_k: [amd])
    monkeypatch.setattr(gpu, "driver_version", fake_driver_version)
    rc = gpu.main(["--json"])
    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert "driver" not in payload["adapters"][0]


def test_main_json_disambiguates_two_identical_intel_arc_cards_by_rank(monkeypatch: pytest.MonkeyPatch,
                                                                        capsys: pytest.CaptureFixture) -> None:
    card0 = _gpu(index=0, name="Intel Arc(TM) B580 Graphics", vendor="intel", vendor_id=0x8086,
                 device_id=0xE20B, dedicated_mb=12288, uma=False, luid="0x00000000_0x00000001")
    card1 = _gpu(index=1, name="Intel Arc(TM) B580 Graphics", vendor="intel", vendor_id=0x8086,
                 device_id=0xE20B, dedicated_mb=12288, uma=False, luid="0x00000000_0x00000002")

    def fake_driver_version(g: GpuInfo, *, rank: int = 0, **_kw: Any) -> str:
        return f"driver-for-rank-{rank}"

    monkeypatch.setattr(gpu, "detect_gpus", lambda **_k: [card0, card1])
    monkeypatch.setattr(gpu, "driver_version", fake_driver_version)
    rc = gpu.main(["--json"])
    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload["adapters"][0]["driver"] == "driver-for-rank-0"
    assert payload["adapters"][1]["driver"] == "driver-for-rank-1"


def test_main_extras_prints_the_mapped_requirement_files(monkeypatch: pytest.MonkeyPatch,
                                                           capsys: pytest.CaptureFixture) -> None:
    monkeypatch.setattr(gpu, "detect_gpus", lambda **_k: [_gpu(vendor="nvidia", dedicated_mb=100)])
    rc = gpu.main(["--extras"])
    assert rc == 0
    assert capsys.readouterr().out.strip() == "requirements-nvidia.txt"


def test_main_default_prints_describe(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    monkeypatch.setattr(gpu, "detect_gpus", lambda **_k: [])
    rc = gpu.main([])
    assert rc == 0
    assert "no GPU adapter" in capsys.readouterr().out


def test_main_never_fails_on_a_bad_flag(capsys: pytest.CaptureFixture) -> None:
    assert gpu.main(["--nonsense"]) == 0


# ------------------------------------------------------------------ vram_free_mb / total / source
def _park_nvml_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(gpu._state, "ok", False)
    monkeypatch.setitem(gpu._state, "lib", None)
    monkeypatch.setitem(gpu._state, "handle", None)


def test_vram_free_mb_falls_through_to_pdh_for_a_non_nvidia_primary(monkeypatch: pytest.MonkeyPatch) -> None:
    _park_nvml_off(monkeypatch)
    amd = _gpu(vendor="amd", dedicated_mb=32590, luid="0x00000000_0x00016a0e")
    monkeypatch.setattr(gpu, "primary_gpu", lambda: amd)
    monkeypatch.setattr(gpu, "_pdh_dedicated_used_mb", lambda luid: 1590)
    assert gpu.vram_free_mb() == 32590 - 1590


def test_vram_free_mb_clamps_at_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    _park_nvml_off(monkeypatch)
    amd = _gpu(vendor="amd", dedicated_mb=100)
    monkeypatch.setattr(gpu, "primary_gpu", lambda: amd)
    monkeypatch.setattr(gpu, "_pdh_dedicated_used_mb", lambda luid: 999)
    assert gpu.vram_free_mb() == 0


def test_vram_free_mb_is_none_when_pdh_also_has_no_reading(monkeypatch: pytest.MonkeyPatch) -> None:
    _park_nvml_off(monkeypatch)
    amd = _gpu(vendor="amd", dedicated_mb=32590)
    monkeypatch.setattr(gpu, "primary_gpu", lambda: amd)
    monkeypatch.setattr(gpu, "_pdh_dedicated_used_mb", lambda luid: None)
    assert gpu.vram_free_mb() is None


def test_vram_free_mb_never_falls_through_for_an_nvidia_primary(monkeypatch: pytest.MonkeyPatch) -> None:
    _park_nvml_off(monkeypatch)
    nvidia = _gpu(vendor="nvidia", dedicated_mb=32768)
    monkeypatch.setattr(gpu, "primary_gpu", lambda: nvidia)
    calls: list[str] = []
    monkeypatch.setattr(gpu, "_pdh_dedicated_used_mb", lambda luid: (calls.append(luid), 1)[1])
    assert gpu.vram_free_mb() is None
    assert calls == []


def test_vram_free_mb_is_none_with_no_primary_at_all(monkeypatch: pytest.MonkeyPatch) -> None:
    _park_nvml_off(monkeypatch)
    monkeypatch.setattr(gpu, "primary_gpu", lambda: None)
    assert gpu.vram_free_mb() is None


def test_vram_total_mb_uses_dxgi_dedicated_when_nvml_is_off(monkeypatch: pytest.MonkeyPatch) -> None:
    _park_nvml_off(monkeypatch)
    amd = _gpu(vendor="amd", dedicated_mb=32590)
    monkeypatch.setattr(gpu, "primary_gpu", lambda: amd)
    assert gpu.vram_total_mb() == 32590


def test_vram_total_mb_is_none_with_no_primary(monkeypatch: pytest.MonkeyPatch) -> None:
    _park_nvml_off(monkeypatch)
    monkeypatch.setattr(gpu, "primary_gpu", lambda: None)
    assert gpu.vram_total_mb() is None


def test_vram_source_reports_pdh(monkeypatch: pytest.MonkeyPatch) -> None:
    _park_nvml_off(monkeypatch)
    amd = _gpu(vendor="amd", dedicated_mb=32590)
    monkeypatch.setattr(gpu, "primary_gpu", lambda: amd)
    monkeypatch.setattr(gpu, "_pdh_dedicated_used_mb", lambda luid: 1590)
    assert gpu.vram_source() == "pdh"


def test_vram_source_reports_none_when_nothing_works(monkeypatch: pytest.MonkeyPatch) -> None:
    _park_nvml_off(monkeypatch)
    monkeypatch.setattr(gpu, "primary_gpu", lambda: None)
    assert gpu.vram_source() == "none"


def test_vram_source_reports_nvml(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(gpu._state, "ok", True)
    monkeypatch.setitem(gpu._state, "lib", object())
    monkeypatch.setitem(gpu._state, "handle", object())
    assert gpu.vram_source() == "nvml"


# ------------------------------------------------------------------------------------- smoke ---
def test_detect_gpus_smoke() -> None:
    """detect_gpus() must never raise on this OS, whatever hardware it finds (possibly none)."""
    gpus = gpu.detect_gpus(refresh=True)
    assert isinstance(gpus, list)
    luid_pattern = re.compile(r"^0x[0-9a-f]{8}_0x[0-9a-f]{8}$")
    for g in gpus:
        assert isinstance(g, GpuInfo)
        assert g.name
        assert luid_pattern.match(g.luid), g.luid


# ------------------------------------------------------------------------ primary_budget_mb ---
def test_primary_budget_mb_uses_the_primary_adapter() -> None:
    from exocortex.gpu import GpuInfo, primary_budget_mb

    g = GpuInfo(index=0, name="RTX", vendor="nvidia", vendor_id=0x10DE, device_id=1, dedicated_mb=16000, shared_mb=30000,
                uma=False, luid="0x00000000_0x00000001")
    assert primary_budget_mb([g]) == 16000


def test_primary_budget_mb_falls_back_to_nvml_total_when_dxgi_saw_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """The NVIDIA-hiccup case: DXGI enumerates nothing, NVML still knows the card -- the automatic
    model choice must still be sized to the real VRAM instead of 'no budget' (= largest file)."""
    from exocortex import gpu

    monkeypatch.setattr(gpu, "detect_gpus", lambda refresh=False: [])
    monkeypatch.setattr(gpu, "vram_total_mb", lambda: 24135)
    assert gpu.primary_budget_mb() == 24135
    monkeypatch.setattr(gpu, "vram_total_mb", lambda: None)
    assert gpu.primary_budget_mb() is None
    assert gpu.primary_budget_mb([]) is None            # an explicit empty list never consults NVML


# ------------------------------------------------------------- portable-install hardware helpers
# 2026-09-27 (portable-install): nvidia_driver_version / system_memory_mb / cpu_name --
# exocortex.hardware.collect()'s three injectable probes.
def test_nvidia_driver_version_none_for_non_nvidia_primary() -> None:
    amd = _gpu(vendor="amd", vendor_id=0x1002, dedicated_mb=32590, uma=True)
    assert gpu.nvidia_driver_version([amd]) is None


def test_nvidia_driver_version_none_with_no_gpus_at_all() -> None:
    assert gpu.nvidia_driver_version([]) is None


def test_nvidia_driver_version_prefers_nvml(monkeypatch: pytest.MonkeyPatch) -> None:
    nvidia = _gpu(vendor="nvidia", vendor_id=0x10DE, dedicated_mb=16384, uma=False)
    monkeypatch.setattr(gpu, "_nvml_driver_version", lambda: "592.14")

    def fail_registry(g: Any, **_kw: Any) -> str:
        raise AssertionError("driver_version must not be reached when NVML answered")

    monkeypatch.setattr(gpu, "driver_version", fail_registry)
    assert gpu.nvidia_driver_version([nvidia]) == "592.14"


def test_nvidia_driver_version_falls_back_to_registry_when_nvml_has_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    nvidia = _gpu(vendor="nvidia", vendor_id=0x10DE, dedicated_mb=16384, uma=False)
    monkeypatch.setattr(gpu, "_nvml_driver_version", lambda: None)
    monkeypatch.setattr(gpu, "driver_version", lambda g, **_kw: "32.0.15.9200")
    assert gpu.nvidia_driver_version([nvidia]) == "592.00"


def test_nvidia_driver_version_none_when_neither_source_answers(monkeypatch: pytest.MonkeyPatch) -> None:
    nvidia = _gpu(vendor="nvidia", vendor_id=0x10DE, dedicated_mb=16384, uma=False)
    monkeypatch.setattr(gpu, "_nvml_driver_version", lambda: None)
    monkeypatch.setattr(gpu, "driver_version", lambda g, **_kw: None)
    assert gpu.nvidia_driver_version([nvidia]) is None


@pytest.mark.parametrize("raw,expected", [
    ("32.0.15.9200", "592.00"),
    ("31.0.15.5222", "552.22"),
    (None, None),
    ("", None),
    ("not-a-driver-string", None),
])
def test_nvidia_version_from_registry_string_table(raw: str | None, expected: str | None) -> None:
    assert gpu._nvidia_version_from_registry_string(raw) == expected


def test_system_memory_mb_smoke() -> None:
    """Must never raise; on this OS (Windows) both halves should come back as sane positive ints
    with total >= available."""
    total_mb, avail_mb = gpu.system_memory_mb()
    assert total_mb is None or isinstance(total_mb, int)
    assert avail_mb is None or isinstance(avail_mb, int)
    if total_mb is not None and avail_mb is not None:
        assert total_mb > 0
        assert 0 <= avail_mb <= total_mb


def test_cpu_name_smoke() -> None:
    """Must never raise and must return a string (possibly empty off Windows/on failure)."""
    name = gpu.cpu_name()
    assert isinstance(name, str)

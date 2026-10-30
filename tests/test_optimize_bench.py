"""Tests for exocortex.optimize_bench (2026-10-07): the benchmark store, the sweep importers, the live
LLM probe, the child runner and the helpers. Offline: the "server" is an httpx.MockTransport, the
children are fake Popen classes, nothing touches NVML, the network or a file outside tmp_path."""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from exocortex import optimize, optimize_bench
from exocortex.config import project_root
from exocortex.optimize import AsrRow, Footprint, LlmRun, OcrRow
from exocortex.optimize_bench import (
    BENCH_FILE,
    DEVICES_FRESH_S,
    JOURNAL_FILE,
    STALE_AFTER_S,
    ImportReport,
    Store,
    canary,
    fingerprint,
    import_asr_json,
    import_bench_llm_json,
    import_files,
    import_ocr_json,
    iso_now,
    parse_at,
    probe_llm,
    python_cmd,
    run_child,
    sample_vram,
    swap_resident,
)

NOW = 1_800_000_000.0
DAY = 86400.0
MODEL = "lmstudio-community/Qwen3.8-27B"

HW = {
    "cpu_name": "Intel(R) Core(TM) Ultra 9 285H",
    "gpus": [{"index": 0, "name": "NVIDIA GeForce RTX 5090 Laptop GPU"}, {"index": 1, "name": "Intel(R) Arc(TM) 140T GPU"}],
    "primary_index": 0,
    "nvidia_driver": "610.47",
    "runtime": {"flavor": "nvidia-cuda12", "version": "2.44.0", "source": "lmstudio"},
    "ram_total_mb": 64957,
}
FP = {"cpu": "Intel(R) Core(TM) Ultra 9 285H", "gpu": "NVIDIA GeForce RTX 5090 Laptop GPU", "driver": "610.47",
      "runtime": "nvidia-cuda12 2.44.0", "ram_mb": 64957}


def _run(**kw: Any) -> LlmRun:
    base = dict(model=MODEL, ctx=32768, kv="q8_0", parallel=2, ubatch=512, flags=(), pp=950.0, tg=24.0,
                cached_ttft_ms=210.0, incr_prompt_n=300.0, at=iso_now(NOW), source="probe")
    base.update(kw)
    return LlmRun(**base)


def _asr(label: str = "sherpa-cpu", **kw: Any) -> AsrRow:
    base = dict(label=label, engine="sherpa", device="cpu", compute="int8", threads=4, median_s=0.15, overlap=0.99,
                vram_mb=0, vram_source="nvml", at=iso_now(NOW), source="probe")
    base.update(kw)
    return AsrRow(**base)


def _fp(ctx: int, mb: float, **kw: Any) -> Footprint:
    base = dict(model=MODEL, ctx=ctx, kv="q8_0", ubatch=512, parallel=2, app_mb=mb, source="probe", at=iso_now(NOW))
    base.update(kw)
    return Footprint(**base)


# ============================================================================================
# time + fingerprint
# ============================================================================================
def test_iso_now_round_trips_through_parse_at() -> None:
    s = iso_now(NOW)
    assert parse_at(s) == pytest.approx(NOW, abs=1)
    assert s[:4].isdigit() and ("+" in s[10:] or "-" in s[10:] or s.endswith("Z"))
    assert parse_at(iso_now()) == pytest.approx(time.time(), abs=3)


@pytest.mark.parametrize("bad", ["", "yesterday", None, "2026-13-45T00:00:00", [], {}])
def test_parse_at_garbage_is_none(bad: Any) -> None:
    assert parse_at(bad) is None


def test_parse_at_naive_reads_as_local_time() -> None:
    assert parse_at("2026-10-07T12:00:00") == pytest.approx(time.mktime((2026, 10, 7, 12, 0, 0, 0, 0, -1)), abs=3700)


def test_fingerprint_from_a_hardware_report() -> None:
    assert fingerprint(HW) == FP


def test_fingerprint_runtime_argument_wins_and_primary_index_selects() -> None:
    fp = fingerprint({**HW, "primary_index": 1}, runtime="custom 9")
    assert fp["gpu"] == "Intel(R) Arc(TM) 140T GPU"
    assert fp["runtime"] == "custom 9"


def test_fingerprint_tolerates_missing_everything() -> None:
    assert fingerprint({}) == {"cpu": "", "gpu": "", "driver": "", "runtime": "", "ram_mb": 0}
    assert fingerprint(None)["gpu"] == ""  # type: ignore[arg-type]
    assert fingerprint({"gpus": ["junk"], "ram_total_mb": "lots", "runtime": None})["ram_mb"] == 0


# ============================================================================================
# Store
# ============================================================================================
def test_store_reads_empty_when_nothing_exists(tmp_path: Path) -> None:
    data = Store(tmp_path / "nope").load()
    assert data == {"schema": 1, "fingerprint": {}, "devices": {}, "llm": [], "asr": [], "ocr": [], "footprints": []}
    ev = Store(tmp_path / "nope").evidence(FP, NOW)
    assert ev.llm == () and ev.asr == () and ev.ocr == () and ev.footprints == ()


def test_store_round_trip_every_kind(tmp_path: Path) -> None:
    s = Store(tmp_path)
    run = _run(flags=(("--spec-type", "draft-mtp"),), spread=0.04, gpu_used_mb=17347.0, load_s=21.4)
    s.add_llm(run)
    s.add_asr(_asr())
    s.add_ocr(OcrRow(label="rapidocr/igpu", device="igpu", median_ms=1334.0, cpu_ms=1344.0, at=iso_now(NOW), source="probe"))
    s.add_footprint(_fp(32768, 17347.0, flags=(("--spec-type", "draft-mtp"),)))
    s.set_fingerprint(FP)
    s.set_devices({"igpu": "Intel Arc", "ct2_cuda": True}, NOW)

    fresh = Store(tmp_path)                       # a second instance reads the same files
    assert (tmp_path / BENCH_FILE).is_file()
    ev = fresh.evidence(FP, NOW)
    assert ev.llm == (run,)                       # tuples (flags) survive the JSON round trip
    assert ev.llm[0].flags == (("--spec-type", "draft-mtp"),)
    assert ev.asr[0].label == "sherpa-cpu" and ev.ocr[0].device == "igpu"
    assert ev.footprints[0].app_mb == 17347.0 and ev.footprints[0].flags == (("--spec-type", "draft-mtp"),)
    assert fresh.load()["fingerprint"] == FP
    assert fresh.devices(FP, NOW) == {"igpu": "Intel Arc", "ct2_cuda": True}


def test_store_save_is_atomic_and_leaves_no_temp_files(tmp_path: Path) -> None:
    s = Store(tmp_path)
    s.add_llm(_run())
    before = (tmp_path / BENCH_FILE).read_text(encoding="utf-8")
    assert list(tmp_path.glob("*.tmp")) == []
    assert json.loads(before)["schema"] == 1


def test_store_failed_replace_keeps_the_old_file_and_cleans_up(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    s = Store(tmp_path)
    s.add_llm(_run(ctx=8192))
    old = (tmp_path / BENCH_FILE).read_text(encoding="utf-8")

    def boom(src: Any, dst: Any) -> None:
        raise OSError("disk says no")

    monkeypatch.setattr(optimize_bench.os, "replace", boom)
    with pytest.raises(OSError, match="disk says no"):
        s.add_llm(_run(ctx=16384))
    monkeypatch.undo()
    assert (tmp_path / BENCH_FILE).read_text(encoding="utf-8") == old
    assert list(tmp_path.glob("*.tmp")) == []


def test_store_retries_a_replace_blocked_by_a_scanner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    real = os.replace
    calls = {"n": 0}

    def flaky(src: Any, dst: Any) -> None:
        calls["n"] += 1
        if calls["n"] < 3:
            raise PermissionError("held by another process")
        real(src, dst)

    monkeypatch.setattr(optimize_bench.os, "replace", flaky)
    Store(tmp_path).add_llm(_run())
    assert calls["n"] == 3
    assert len(Store(tmp_path).load()["llm"]) == 1


@pytest.mark.parametrize("content", ["{not json", "[1, 2]", '"text"', '{"schema": 2, "llm": [{"model": "x"}]}', ""])
def test_store_corrupt_or_foreign_file_reads_empty_and_warns(tmp_path: Path, caplog: pytest.LogCaptureFixture,
                                                             content: str) -> None:
    (tmp_path / BENCH_FILE).write_text(content, encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="exocortex.optimize_bench"):
        data = Store(tmp_path).load()
    assert data["llm"] == [] and data["asr"] == [] and data["fingerprint"] == {}
    assert any("optimize" in r.message for r in caplog.records)
    # and the store is usable again: the next write replaces the bad file
    Store(tmp_path).add_llm(_run())
    assert len(Store(tmp_path).load()["llm"]) == 1


def test_store_drops_non_dict_rows_but_keeps_the_rest(tmp_path: Path) -> None:
    (tmp_path / BENCH_FILE).write_text(json.dumps({"schema": 1, "llm": [1, "x", optimize.to_dict(_run())], "asr": "oops"}),
                                       encoding="utf-8")
    data = Store(tmp_path).load()
    assert len(data["llm"]) == 1 and data["asr"] == []


def test_llm_dedup_newest_wins_per_configuration(tmp_path: Path) -> None:
    s = Store(tmp_path)
    s.add_llm(_run(pp=900.0))
    s.add_llm(_run(ctx=8192, pp=1100.0))
    s.add_llm(_run(pp=950.0, at=iso_now(NOW + 5)))                       # same key as the first -> replaces it
    s.add_llm(_run(flags=(("--spec-type", "draft-mtp"),), pp=800.0))     # other flags -> its own row
    s.add_llm(_run(kv="f16", pp=700.0))
    s.add_llm(_run(ubatch=1024, pp=1000.0))
    s.add_llm(_run(parallel=1, pp=1000.0))
    rows = s.evidence(None, NOW).llm
    assert len(rows) == 6
    same = [r for r in rows if r.ctx == 32768 and r.kv == "q8_0" and r.parallel == 2 and r.ubatch == 512 and not r.flags]
    assert [r.pp for r in same] == [950.0]
    assert rows[-1].parallel == 1            # insertion order = oldest first
    assert [r.ctx for r in rows].count(8192) == 1


def test_caps_keep_the_newest_rows(tmp_path: Path) -> None:
    s = Store(tmp_path)
    for i in range(70):
        s.add_llm(_run(ctx=2048 * (i + 1)))
    for i in range(45):
        s.add_asr(_asr(f"variant-{i}"))
    for i in range(25):
        s.add_ocr(OcrRow(label=f"rapidocr/d{i}", device=f"d{i}", median_ms=1000.0 + i))
    for i in range(90):
        s.add_footprint(_fp(2048 * (i + 1), 1000.0 + i))
    ev = s.evidence(None, NOW)
    assert len(ev.llm) == 60 and ev.llm[0].ctx == 2048 * 11 and ev.llm[-1].ctx == 2048 * 70
    assert len(ev.asr) == 40 and ev.asr[0].label == "variant-5" and ev.asr[-1].label == "variant-44"
    assert len(ev.ocr) == 20 and ev.ocr[-1].device == "d24"
    assert len(ev.footprints) == 80 and ev.footprints[-1].ctx == 2048 * 90


def test_asr_dedup_per_label_and_ocr_dedup_per_device(tmp_path: Path) -> None:
    s = Store(tmp_path)
    s.add_asr(_asr("sherpa-cpu", median_s=0.20))
    s.add_asr(_asr("whisper-cpu-int8", engine="faster-whisper", median_s=6.2))
    s.add_asr(_asr("sherpa-cpu", median_s=0.14))
    s.add_ocr(OcrRow(label="a", device="igpu", median_ms=1400.0))
    s.add_ocr(OcrRow(label="b", device="igpu", median_ms=1300.0))
    ev = s.evidence(None, NOW)
    assert [(r.label, r.median_s) for r in ev.asr] == [("whisper-cpu-int8", 6.2), ("sherpa-cpu", 0.14)]
    assert [(r.device, r.median_ms) for r in ev.ocr] == [("igpu", 1300.0)]


def test_rows_go_stale_when_the_fingerprint_differs(tmp_path: Path) -> None:
    s = Store(tmp_path)
    s.set_fingerprint(FP)
    s.add_llm(_run())
    s.add_asr(_asr())
    s.add_ocr(OcrRow(label="x", device="igpu", median_ms=1.0, at=iso_now(NOW)))
    s.add_footprint(_fp(32768, 17000.0))

    same = s.evidence(FP, NOW)
    assert not same.llm[0].stale and not same.asr[0].stale and not same.ocr[0].stale and len(same.footprints) == 1
    assert not s.evidence(None, NOW).llm[0].stale                       # no fingerprint given: cannot tell

    other = s.evidence({**FP, "gpu": "NVIDIA GeForce RTX 4090 Laptop GPU"}, NOW)
    assert other.llm[0].stale and other.asr[0].stale and other.ocr[0].stale
    assert other.footprints == ()                                       # a footprint from another GPU is not evidence
    assert s.evidence({**FP, "driver": "999.0"}, NOW).llm[0].stale
    assert s.evidence({**FP, "runtime": "vulkan 1.0"}, NOW).llm[0].stale
    assert s.evidence({**FP, "ram_mb": 16000}, NOW).llm[0].stale


def test_fingerprint_comparison_is_tolerant_of_unknown_parts(tmp_path: Path) -> None:
    s = Store(tmp_path)
    s.set_fingerprint(FP)
    s.add_llm(_run())
    for given in (
        {**FP, "driver": ""},                     # unknown driver reading
        {**FP, "ram_mb": 64900},                  # RAM total wobble under 2 %
        {**FP, "gpu": FP["gpu"].upper()},         # case
        {"gpu": FP["gpu"]},                       # a partial fingerprint
        {},
    ):
        assert not s.evidence(given, NOW).llm[0].stale, given


def test_rows_go_stale_by_age(tmp_path: Path) -> None:
    s = Store(tmp_path)
    s.add_llm(_run(ctx=2048, at=iso_now(NOW - (STALE_AFTER_S + DAY))))
    s.add_llm(_run(ctx=4096, at=iso_now(NOW - (STALE_AFTER_S - DAY))))
    s.add_llm(_run(ctx=6144, at="not a date"))
    s.add_llm(_run(ctx=8192, at=""))
    s.add_asr(_asr(at=iso_now(NOW - 40 * DAY)))
    s.add_footprint(_fp(32768, 17000.0, at=iso_now(NOW - 90 * DAY)))
    ev = s.evidence(None, NOW)
    assert [r.stale for r in ev.llm] == [True, False, False, False]
    assert ev.asr[0].stale
    assert len(ev.footprints) == 1            # age alone never drops a footprint
    # and "now" moving forward ages the same data
    assert all(r.stale for r in s.evidence(None, NOW + 400 * DAY).llm[:2])


def test_changing_the_recorded_fingerprint_marks_old_rows_stale_for_good(tmp_path: Path) -> None:
    s = Store(tmp_path)
    s.set_fingerprint(FP)
    s.add_llm(_run(ctx=8192))
    s.add_footprint(_fp(8192, 16000.0))
    new_fp = {**FP, "driver": "620.0"}
    s.set_fingerprint(new_fp)
    s.add_llm(_run(ctx=16384))
    ev = s.evidence(new_fp, NOW)              # the stored fingerprint now equals the machine...
    by_ctx = {r.ctx: r.stale for r in ev.llm}
    assert by_ctx == {8192: True, 16384: False}      # ...but the older row stays stale
    assert ev.footprints == ()


def test_set_fingerprint_same_value_does_not_rewrite(tmp_path: Path) -> None:
    s = Store(tmp_path)
    s.set_fingerprint(FP)
    mtime = (tmp_path / BENCH_FILE).stat().st_mtime_ns
    s.set_fingerprint(dict(FP))
    assert (tmp_path / BENCH_FILE).stat().st_mtime_ns == mtime


def test_devices_cache_freshness_and_fingerprint(tmp_path: Path) -> None:
    s = Store(tmp_path)
    assert s.devices(FP, NOW) is None
    s.set_fingerprint(FP)
    s.set_devices({"igpu": "Arc", "npu": "AI Boost", "openvino": {"NPU": "Intel(R) AI Boost"}}, NOW - 3600)
    cached = s.devices(FP, NOW)
    assert cached == {"igpu": "Arc", "npu": "AI Boost", "openvino": {"NPU": "Intel(R) AI Boost"}}   # no "at" in it
    assert s.devices(None, NOW) == cached
    assert s.devices({**FP, "gpu": "other"}, NOW) is None                  # another machine
    assert s.devices(FP, NOW - 3600 + DEVICES_FRESH_S + 1) is None         # older than 7 days
    assert s.devices(FP, NOW - 3600 + DEVICES_FRESH_S - 60) == cached


def test_devices_with_an_unparsable_stamp_is_not_trusted(tmp_path: Path) -> None:
    (tmp_path / BENCH_FILE).write_text(json.dumps({"schema": 1, "devices": {"at": "soon", "igpu": "x"}}), encoding="utf-8")
    assert Store(tmp_path).devices(None, NOW) is None


def test_journal_is_its_own_file_and_keeps_the_last_ten(tmp_path: Path) -> None:
    s = Store(tmp_path)
    assert s.journal() == []
    s.journal_write([{"id": i, "state": "verified"} for i in range(15)])
    assert [e["id"] for e in s.journal()] == list(range(5, 15))
    assert (tmp_path / JOURNAL_FILE).is_file()
    assert not (tmp_path / BENCH_FILE).exists()
    assert json.loads((tmp_path / JOURNAL_FILE).read_text(encoding="utf-8"))["schema"] == 1
    s.journal_write([{"id": "a"}, "junk", {"id": "b"}])  # type: ignore[list-item]
    assert [e["id"] for e in s.journal()] == ["a", "b"]


def test_a_corrupt_benchmark_store_cannot_touch_the_journal(tmp_path: Path) -> None:
    s = Store(tmp_path)
    s.journal_write([{"id": 1, "state": "pending", "previous": {"ctx_size": 81920}}])
    (tmp_path / BENCH_FILE).write_text("\x00garbage", encoding="utf-8")
    s.add_llm(_run())                                   # rewrites the bad store
    assert s.journal() == [{"id": 1, "state": "pending", "previous": {"ctx_size": 81920}}]
    (tmp_path / JOURNAL_FILE).write_text("{oops", encoding="utf-8")
    assert s.journal() == []
    assert len(s.load()["llm"]) == 1                    # and the other way round


def test_concurrent_adds_lose_nothing(tmp_path: Path) -> None:
    stores = [Store(tmp_path) for _ in range(8)]        # separate instances share the module lock
    errors: list[BaseException] = []

    def work(i: int) -> None:
        try:
            for j in range(3):
                stores[i].add_llm(_run(ctx=2048 * (i * 3 + j + 1)))
        except BaseException as exc:  # noqa: BLE001 - surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert sorted(r.ctx for r in Store(tmp_path).evidence(None, NOW).llm) == [2048 * k for k in range(1, 25)]


# ============================================================================================
# importer: bench_llm.py --protocol full --json
# ============================================================================================
def bench_llm_json(extra: list[str] | None = None, **over: Any) -> dict:
    """A result shaped like ``scripts/bench_llm.py --protocol full --json`` for the owner's default model."""
    rounds = []
    for pp, tg, c_pp in ((940.0, 25.5, 700.0), (960.0, 26.1, 710.0), (980.0, 26.6, 720.0)):
        rounds.append({
            "a_fresh": {"text": "- Kafka keeps a log", "text_len": 120, "ttft_s": 2.1,
                        "timings": {"prompt_n": 2012, "prompt_ms": 2100.0, "prompt_per_second": pp,
                                    "predicted_n": 96, "predicted_ms": 3700.0, "predicted_per_second": tg}},
            "b_cached_repeat": {"text": "- Kafka", "text_len": 80, "wall_ttft_s": 0.21, "timings": {"prompt_n": 3}},
            "c_incremental": {"text": "-", "text_len": 40, "ttft_s": 0.5,
                              "timings": {"prompt_n": 318, "prompt_per_second": c_pp}},
        })
    obj = {
        "requested": {"model": MODEL, "backend": "auto", "server_path": "", "flash_attn": True, "kv_cache_type": "q8_0",
                      "load_mode": "mmap", "llama_extra_args": extra if extra is not None else ["--ctx-size", "32768", "--spec-type", "draft-mtp"],
                      "ctx_size": 8192, "parallel": 2, "prompt_tokens": 2000, "extra_prompt_tokens": 300, "rounds": 3,
                      "max_tokens": 160},
        "binary": {"flavor": "nvidia-cuda12", "version": "2.44.0", "exe": "llama-server.exe", "reason": "x"},
        "model": {"key": MODEL, "size_gb": 16.81},
        "load_s": 21.4, "aborted": False, "gpu_free_before_mb": 21400, "gpu_free_after_mb": 4053, "gpu_used_mb": 17347,
        "log_signals": {"matches": {"n_ubatch": "n_ubatch = 512"}, "warnings": []},
        "rounds": rounds,
        "medians": {"pp": 960.0, "tg": 26.1, "cached_ttft_ms": 210.0, "incremental_pp": 710.0, "incremental_prompt_n": 318,
                    "spread_pp": 40.0, "spread_tg": 1.1},
    }
    obj.update(over)
    return obj


def test_import_llm_realistic_result() -> None:
    run, fp = import_bench_llm_json(bench_llm_json(), at="2026-10-07T10:00:00+00:00")
    assert run is not None and fp is not None
    assert run.model == MODEL
    assert (run.ctx, run.kv, run.parallel, run.ubatch) == (32768, "q8_0", 2, 512)   # the repeated flag overrides requested.ctx_size
    assert run.flags == (("--spec-type", "draft-mtp"),)
    assert (run.pp, run.tg, run.cached_ttft_ms, run.incr_prompt_n) == (960.0, 26.1, 210.0, 318.0)
    assert run.gpu_used_mb == 17347.0 and run.load_s == 21.4
    assert run.ok and run.clean_log and run.source == "sweep" and run.at == "2026-10-07T10:00:00+00:00" and not run.stale
    assert run.spread == pytest.approx(max((980 - 940) / 960, (26.6 - 25.5) / 26.1), abs=1e-3)
    assert fp == Footprint(model=MODEL, ctx=32768, kv="q8_0", ubatch=512, parallel=2, flags=(("--spec-type", "draft-mtp"),),
                           app_mb=17347.0, source="sweep", at="2026-10-07T10:00:00+00:00")


def test_import_llm_flag_parsing_is_last_wins() -> None:
    extra = ["--ctx-size", "8192", "--ctx-size", "65536", "--cache-type-k", "f16", "--ubatch-size", "1024",
             "--parallel", "1", "--parallel", "3", "--threads-batch", "16"]
    run, fp = import_bench_llm_json(bench_llm_json(extra))
    assert run is not None and fp is not None
    assert (run.ctx, run.kv, run.parallel, run.ubatch) == (65536, "f16", 3, 1024)
    assert run.flags == (("--threads-batch", "16"), ("--ubatch-size", "1024"))        # sorted
    assert fp.ctx == 65536 and fp.kv == "f16" and fp.flags == run.flags


def test_import_llm_without_overrides_uses_the_requested_values() -> None:
    run, _fp = import_bench_llm_json(bench_llm_json([]))
    assert run is not None
    assert (run.ctx, run.kv, run.parallel, run.ubatch, run.flags) == (8192, "q8_0", 2, 512, ())


def test_import_llm_builtin_valued_flags_do_not_count_as_differences() -> None:
    run, _fp = import_bench_llm_json(bench_llm_json(["--ubatch-size", "512", "--load-mode", "mmap", "--ctx-size", "8192"]))
    assert run is not None and run.flags == () and run.ubatch == 512


def test_import_llm_load_mode_from_requested() -> None:
    obj = bench_llm_json([])
    obj["requested"]["load_mode"] = "mmap+mlock"
    run, _ = import_bench_llm_json(obj)
    assert run is not None and run.flags == (("--load-mode", "mmap+mlock"),)
    # an explicit --load-mode in the extra args is what the server saw last
    obj2 = bench_llm_json(["--load-mode", "mmap"])
    obj2["requested"]["load_mode"] = "mmap+mlock"
    run2, _ = import_bench_llm_json(obj2)
    assert run2 is not None and run2.flags == ()


def test_import_llm_unmanaged_flags_make_the_run_inert_not_a_baseline() -> None:
    plain, _ = import_bench_llm_json(bench_llm_json(["--ctx-size", "32768"]))
    ckpt, ckpt_fp = import_bench_llm_json(bench_llm_json(["--ctx-size", "32768", "--ctx-checkpoints", "64", "--cache-ram", "16384"]))
    assert plain is not None and ckpt is not None
    assert plain.flags == ()
    assert ckpt.flags == (("--cache-ram", "16384"), ("--ctx-checkpoints", "64"))
    assert ckpt_fp is not None                       # host-side flags: the VRAM footprint is still the normal one
    assert optimize_bench._llm_key(optimize.to_dict(plain)) != optimize_bench._llm_key(optimize.to_dict(ckpt))
    assert not [r for r in (ckpt,) if not r.flags]   # the planner's baseline filter (`not r.flags`) skips it


def test_import_llm_memory_moving_flags_record_no_footprint() -> None:
    run, fp = import_bench_llm_json(bench_llm_json(["--ctx-size", "131072", "--no-kv-offload"]))
    assert run is not None and run.flags == (("--no-kv-offload", "1"),)
    assert fp is None


def test_import_llm_aborted_run_is_recorded_as_not_ok() -> None:
    obj = {"requested": bench_llm_json()["requested"], "load_s": 240.1, "aborted": True, "error": "llama-server failed to start"}
    run, fp = import_bench_llm_json(obj, at="T")
    assert run is not None and fp is None
    assert run.ok is False and run.pp is None and run.tg is None and run.gpu_used_mb is None
    assert run.model == MODEL and run.ctx == 32768 and run.load_s == 240.1


def test_import_llm_missing_medians_is_not_ok_and_has_no_footprint() -> None:
    obj = bench_llm_json()
    obj["medians"] = {}
    run, fp = import_bench_llm_json(obj)
    assert run is not None and run.ok is False and fp is None


@pytest.mark.parametrize("line,clean", [
    ("load_tensors: tensor 'blk.3.ffn' not supported on this device, falling back", False),
    ("ggml_cuda_init: CUDA error: out of memory", False),
    ("cudaMalloc failed: out of memory", False),
    ("ggml_backend_cuda_buffer_type_alloc_buffer: OOM", False),
    ("llama_model_load: error loading model", False),
    ("warning: n_ctx_seq (32768) < n_ctx_train (262144); there is room for more", True),
    ("srv  warn: cache_reuse is not supported by this context, it will be disabled", True),   # printed by every 2-slot launch
    ("common_init_from_params: warming up the model", True),
])
def test_import_llm_clean_log_rule(line: str, clean: bool) -> None:
    obj = bench_llm_json()
    obj["log_signals"]["warnings"] = [line]
    run, fp = import_bench_llm_json(obj)
    assert run is not None
    assert run.clean_log is clean
    assert (fp is not None) is clean                 # an unclean run does not feed the footprint fit


def test_import_llm_spread_falls_back_to_the_scripts_absolute_spread() -> None:
    obj = bench_llm_json()
    del obj["rounds"]
    run, _ = import_bench_llm_json(obj)
    assert run is not None
    assert run.spread == pytest.approx(max(40 / 960, 1.1 / 26.1), abs=1e-3)
    one = bench_llm_json()
    one["rounds"] = one["rounds"][:1]
    one["medians"].update(spread_pp=0.0, spread_tg=0.0)
    r1, _ = import_bench_llm_json(one)
    assert r1 is not None and r1.spread == 0.0


@pytest.mark.parametrize("junk", [None, [], "x", 3, {}, {"unrelated": 1}])
def test_import_llm_rejects_things_that_are_not_results(junk: Any) -> None:
    assert import_bench_llm_json(junk) == (None, None)


def test_import_llm_no_footprint_when_the_gpu_reading_is_missing_or_nonsense() -> None:
    for bad in (None, -5, 0, float("nan"), "n/a"):
        run, fp = import_bench_llm_json(bench_llm_json(gpu_used_mb=bad))
        assert run is not None and fp is None
        assert run.gpu_used_mb is None                         # only a positive, finite reading is kept
        assert run.ok                                          # ... and the timings still count


# ============================================================================================
# importer: speech
# ============================================================================================
def asr_rows() -> list[dict]:
    def row(label: str, engine: str, device: str, compute: str, threads: int, median: float, overlap: float, vram: float | None,
            **kw: Any) -> dict:
        return {"label": label, "engine": engine, "device": device, "compute_type": compute, "cpu_threads": threads,
                "median_s": median, "overlap_mean": overlap, "vram_delta_mb": vram, "load_s": 3.0, "error": None, **kw}
    return [
        row("whisper-turbo cuda int8_float16", "faster-whisper", "cuda", "int8_float16", 0, 0.112, 1.0, 874.3),
        row("whisper-turbo cuda float16", "faster-whisper", "cuda", "float16", 0, 0.126, 1.0, 2380.7),
        row("whisper-turbo cpu int8 4thr", "faster-whisper", "cpu", "int8", 4, 6.21, 1.0, -12.0),
        row("parakeet cpu 4thr", "sherpa", "cpu", "int8", 4, 0.15, 0.99, -3011.0),
        row("parakeet cpu 8thr", "sherpa", "cpu", "int8", 8, 0.19, 0.99, 3.0),
        row("whisper-turbo cpu int8 8thr", "faster-whisper", "cpu", "int8", 8, 6.0, 1.0, 5.0),
        row("whisper-turbo cuda float16", "faster-whisper", "cuda", "float16", 0, 0.0, 0.0, None, error="CUDA out of memory"),
    ]


def test_import_asr_normalises_labels_and_maps_fields() -> None:
    rows = import_asr_json(asr_rows(), at="T0")
    by = {r.label: r for r in rows}
    assert list(by) == ["whisper-cuda-int8_float16", "whisper-cuda-float16", "whisper-cpu-int8", "sherpa-cpu",
                        "sherpa-cpu-8thr", "whisper-cpu-int8-8thr"]            # the failed row is gone
    g = by["whisper-cuda-int8_float16"]
    assert (g.engine, g.device, g.compute, g.threads, g.median_s, g.overlap) == ("faster-whisper", "cuda", "int8_float16", 0, 0.112, 1.0)
    assert (g.vram_mb, g.vram_source, g.source, g.at, g.stale) == (874, "nvml", "sweep", "T0", False)
    assert by["whisper-cuda-float16"].vram_mb == 2381
    s = by["sherpa-cpu"]
    assert (s.engine, s.device, s.compute, s.threads) == ("sherpa", "cpu", "int8", 4)
    assert by["sherpa-cpu-8thr"].threads == 8 and by["whisper-cpu-int8-8thr"].threads == 8
    assert by["whisper-cpu-int8"].threads == 4 and by["whisper-cpu-int8"].device == "cpu"


def test_import_asr_implausible_vram_falls_back_to_the_prior() -> None:
    by = {r.label: r for r in import_asr_json(asr_rows())}
    assert (by["sherpa-cpu"].vram_mb, by["sherpa-cpu"].vram_source) == (0, "prior")                  # the -3011 MiB reading
    assert (by["whisper-cpu-int8"].vram_mb, by["whisper-cpu-int8"].vram_source) == (0, "prior")      # negative
    assert by["sherpa-cpu-8thr"].vram_source == "nvml" and by["sherpa-cpu-8thr"].vram_mb == 3
    huge = [{"label": "whisper-turbo cuda int8_float16", "median_s": 0.11, "vram_delta_mb": 9500.0}]
    r = import_asr_json(huge)[0]
    assert (r.vram_mb, r.vram_source) == (874, "prior")
    edge = [{"label": "whisper-turbo cuda float16", "median_s": 0.11, "vram_delta_mb": 8000.0}]
    assert import_asr_json(edge)[0].vram_source == "nvml"
    cpu_noise = [{"label": "parakeet cpu 4thr", "median_s": 0.15, "vram_delta_mb": 700.0}]
    assert import_asr_json(cpu_noise)[0].vram_mb == 0                                                 # a CPU engine holds none


def test_import_asr_unknown_variants_keep_their_label() -> None:
    odd = [{"label": "moonshine tiny", "median_s": 0.3, "vram_delta_mb": -1.0}, {"label": "moonshine base", "median_s": 0.4, "vram_delta_mb": 120.0}]
    a, b = import_asr_json(odd)
    assert a.label == "moonshine tiny" and a.vram_mb is None and a.vram_source == "none"
    assert b.vram_mb == 120 and b.vram_source == "nvml"


def test_import_asr_uses_row_fields_when_the_label_is_vague() -> None:
    row = {"label": "turbo fp16", "engine": "faster-whisper", "device": "cuda", "compute_type": "float16", "median_s": 0.12,
           "overlap": 1.0, "vram_mb": 2400}
    r = import_asr_json([row])[0]
    assert (r.label, r.engine, r.device, r.compute, r.vram_mb, r.overlap) == ("whisper-cuda-float16", "faster-whisper", "cuda", "float16", 2400, 1.0)


def test_import_asr_accepts_wrappers_aliases_and_skips_junk() -> None:
    rows = asr_rows()[:2]
    assert len(import_asr_json({"results": rows})) == 2
    assert len(import_asr_json(rows[0])) == 1
    assert import_asr_json({"median": 0.2, "label": "parakeet cpu 4thr"})[0].median_s == 0.2
    assert import_asr_json([{"label": "x"}, {"median_s": 0.1}, {"label": "y", "median_s": 0}, "junk", None, 4]) == []   # type: ignore[list-item]
    assert import_asr_json("nope") == [] and import_asr_json(None) == []          # type: ignore[arg-type]


def test_import_ocr_json() -> None:
    results = [
        {"ok": True, "label": "rapidocr/igpu", "init_ms": 900, "wall_ms": [1300.0, 1350.0, 1400.0], "cpu_ms": [1340.0, 1344.0, 1350.0]},
        {"ok": True, "label": "windows", "wall_ms": [140.0]},
        {"ok": False, "label": "rapidocr/npu", "error": "exit 1: boom"},
        {"ok": True, "label": "rapidocr/cpu", "wall_ms": [3200.0]},
    ]
    rows = import_ocr_json(results, at="T")
    assert [(r.device, r.median_ms, r.cpu_ms, r.source) for r in rows] == [("igpu", 1350.0, 1344.0, "sweep"), ("cpu", 3200.0, None, "sweep")]


# ============================================================================================
# import_files
# ============================================================================================
def test_import_files_sniffs_each_format(tmp_path: Path) -> None:
    d = tmp_path / "bench"
    d.mkdir()
    (d / "base.json").write_text(json.dumps(bench_llm_json([])), encoding="utf-8")
    (d / "ctx32k.json").write_text(json.dumps(bench_llm_json(["--ctx-size", "32768"])), encoding="utf-8")
    (d / "asr.json").write_text(json.dumps(asr_rows()), encoding="utf-8")
    (d / "ocr.json").write_text(json.dumps([{"ok": True, "label": "rapidocr/igpu", "wall_ms": [1300.0]}]), encoding="utf-8")
    (d / "broken.json").write_text("{nope", encoding="utf-8")
    (d / "other.json").write_text(json.dumps({"hello": 1}), encoding="utf-8")
    (d / "empty.json").write_text("[]", encoding="utf-8")
    (d / "notes.txt").write_text("ignored: not json", encoding="utf-8")
    stamp = NOW - DAY
    for f in d.glob("*.json"):
        os.utime(f, (stamp, stamp))

    store = Store(tmp_path / "lib")
    report = import_files([d], store)                       # a directory expands to its *.json
    assert isinstance(report, ImportReport)
    assert report.imported == 2 + 6 + 1                     # two runs, six usable speech rows, one OCR row
    assert report.skipped == 4                              # the failed ASR row, broken, other, empty
    text = "\n".join(report.warnings)
    assert "broken.json" in text and "other.json" in text and "empty.json" in text and "notes.txt" not in text
    ev = store.evidence(None, NOW)
    assert sorted(r.ctx for r in ev.llm) == [8192, 32768]
    assert {r.at for r in ev.llm} == {iso_now(stamp)}       # dated by the file, not by today
    assert all(r.source == "sweep" for r in ev.llm)
    assert sorted(f.ctx for f in ev.footprints) == [8192, 32768]
    assert {r.label for r in ev.asr} >= {"sherpa-cpu", "whisper-cuda-int8_float16"}
    assert [r.device for r in ev.ocr] == ["igpu"]


def test_import_files_handles_lists_of_results_and_missing_paths(tmp_path: Path) -> None:
    f = tmp_path / "all.json"
    f.write_text(json.dumps([bench_llm_json([]), bench_llm_json(["--ctx-size", "16384"]), asr_rows()[3]]), encoding="utf-8")
    bom = tmp_path / "bom.json"
    bom.write_bytes(b"\xef\xbb\xbf" + json.dumps(bench_llm_json(["--ctx-size", "4096"])).encode("utf-8"))
    store = Store(tmp_path / "lib")
    report = import_files([f, tmp_path / "missing.json", bom], store)
    assert report.imported == 4 and report.skipped == 1
    assert any("missing.json" in w for w in report.warnings)
    assert len(store.evidence(None, NOW).llm) == 3


def test_import_files_never_raises_on_a_store_that_cannot_be_written(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    f = tmp_path / "x.json"
    f.write_text(json.dumps(bench_llm_json()), encoding="utf-8")

    def boom(src: Any, dst: Any) -> None:
        raise OSError("read-only volume")

    monkeypatch.setattr(optimize_bench.os, "replace", boom)
    report = import_files([f], Store(tmp_path / "lib"))
    assert report.imported == 0 and report.skipped == 1 and "read-only volume" in report.warnings[0]


# ============================================================================================
# probe_llm against an in-process fake llama-server (httpx.MockTransport)
# ============================================================================================
def _sse(chunks: list[dict]) -> bytes:
    return ("".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n").encode("utf-8")


class FakeLlama:
    """Serves /props, /slots, SSE /v1/completions (with a final "timings" chunk) and the slot erase."""

    def __init__(self, *, n_ctx: int = 32768, slots: Any = "default", props_status: int = 200, slots_status: int = 200,
                 completions_status: int = 200, pps: list[float] | None = None, tgs: list[float] | None = None,
                 timings: bool = True, fail_on_completion: int | None = None) -> None:
        self.n_ctx = n_ctx
        self.slots = ([{"id": 0, "n_ctx": n_ctx, "is_processing": False, "n_past": 900},
                       {"id": 1, "n_ctx": n_ctx, "is_processing": False, "n_past": 10}] if slots == "default" else slots)
        self.props_status, self.slots_status, self.completions_status = props_status, slots_status, completions_status
        self.pps, self.tgs = list(pps or [900.0, 1000.0]), list(tgs or [20.0, 24.0])
        self.timings = timings
        self.fail_on_completion = fail_on_completion
        self.calls: list[str] = []                # "GET /props", "POST /v1/completions", ...
        self.bodies: list[dict] = []              # /v1/completions request bodies
        self.erased: list[str] = []
        self._seen: set[str] = set()

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        self.calls.append(f"{request.method} {path}")
        if path == "/props":
            if self.props_status != 200:
                return httpx.Response(self.props_status, json={"error": "nope"})
            return httpx.Response(200, json={"default_generation_settings": {"n_ctx": self.n_ctx}, "total_slots": 2})
        if path == "/slots":
            return httpx.Response(self.slots_status, json=self.slots if self.slots_status == 200 else {"error": "not enabled"})
        if path.startswith("/slots/") and request.url.params.get("action") == "erase":
            self.erased.append(path)
            return httpx.Response(200, json={"id_slot": 1, "n_erased": 1})
        if path == "/v1/completions":
            body = json.loads(request.content)
            self.bodies.append(body)
            if self.completions_status != 200 or self.fail_on_completion == len(self.bodies):
                return httpx.Response(self.completions_status if self.completions_status != 200 else 500, json={"error": "boom"})
            prompt = body["prompt"]
            if prompt in self._seen:
                t = {"prompt_n": 3, "prompt_per_second": 5000.0, "predicted_per_second": 25.0}
            elif "follow-up" in prompt:
                t = {"prompt_n": 230, "prompt_per_second": 720.0, "predicted_per_second": 25.0}
            else:
                t = {"prompt_n": 1500, "prompt_per_second": self.pps.pop(0) if self.pps else 900.0,
                     "predicted_per_second": self.tgs.pop(0) if self.tgs else 24.0}
            self._seen.add(prompt)
            chunks: list[dict] = [{"choices": [{"text": "- Kafka"}]}, {"choices": [{"text": " keeps a log"}]},
                                  {"choices": [{"text": "", "finish_reason": "stop"}]}]
            if self.timings:
                chunks[-1]["timings"] = t
            return httpx.Response(200, content=_sse(chunks), headers={"content-type": "text/event-stream"})
        return httpx.Response(404, json={"error": "unknown"})

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self.handler))


URL = "http://127.0.0.1:8080"


def test_probe_llm_medians_slot_pin_and_erase() -> None:
    fake = FakeLlama()
    steps: list[tuple[str, float]] = []
    res = probe_llm(URL, rounds=2, http=fake.client(), progress=lambda m, f: steps.append((m, f)))
    assert res["error"] is None and res["cancelled"] is False
    assert res["pp"] == 950.0 and res["tg"] == 22.0                       # medians of (900, 1000) and (20, 24)
    assert res["incr_prompt_n"] == 230.0 and res["incr_pp"] == 720.0
    assert 0.0 < res["cached_ttft_ms"] < 5000.0
    assert res["n_ctx"] == 32768 and res["slot"] == 1 and res["prompt_tokens"] == 1500
    assert res["spread"] == pytest.approx(max(100 / 950, 4 / 22), abs=1e-3) and res["noisy"] is False
    assert len(res["rounds"]) == 2 and res["rounds"][0]["pp"] == 900.0 and res["rounds"][1]["prompt_n"] == 1500
    assert res["warnings"] == []
    # request shape: three per round, all pinned to the idle slot with the fewest cached tokens
    assert len(fake.bodies) == 6
    assert all(b["id_slot"] == 1 and b["stream"] is True and b["temperature"] == 0.0 for b in fake.bodies)
    assert all(b["max_tokens"] == 96 and "<|im_end|>" in b["stop"] for b in fake.bodies)
    a1, b1, c1, a2 = (fake.bodies[i]["prompt"] for i in range(4))
    assert b1 == a1 and a2 != a1 and c1 != a1 and "follow-up" in c1 and "follow-up" not in a1
    assert a1.startswith("<|im_start|>system") and a1.rstrip().endswith("</think>")
    # the slot was erased at the end, and the probe never touched anything but the server it was given
    assert fake.erased == ["/slots/1"]
    assert fake.calls[:2] == ["GET /props", "GET /slots"] and fake.calls[-1] == "POST /slots/1"
    assert steps and steps[-1][1] == pytest.approx(1.0) and all(0.0 <= f <= 1.0 for _m, f in steps)


def test_probe_llm_picks_the_idle_slot_with_the_fewest_cached_tokens() -> None:
    slots = [{"id": 0, "is_processing": True, "n_past": 0}, {"id": 1, "is_processing": False, "n_past": 4000},
             {"id": 2, "is_processing": False, "n_past": 120}, {"id": 3, "is_processing": False, "n_past": 900}]
    fake = FakeLlama(slots=slots)
    res = probe_llm(URL, rounds=1, http=fake.client())
    assert res["slot"] == 2 and all(b["id_slot"] == 2 for b in fake.bodies) and fake.erased == ["/slots/2"]


def test_probe_llm_refuses_when_every_slot_is_busy() -> None:
    fake = FakeLlama(slots=[{"id": 0, "is_processing": True}, {"id": 1, "is_processing": True}])
    res = probe_llm(URL, rounds=1, http=fake.client())
    assert "busy" in res["error"] and res["pp"] is None
    assert fake.bodies == [] and fake.erased == []


def test_probe_llm_single_slot_warns() -> None:
    fake = FakeLlama(slots=[{"id": 0, "n_ctx": 8192, "is_processing": False}])
    res = probe_llm(URL, rounds=1, http=fake.client())
    assert res["error"] is None and res["slot"] == 0
    assert any("single slot" in w for w in res["warnings"])
    assert fake.bodies[0]["id_slot"] == 0


def test_probe_llm_without_slots_endpoint_warns_and_does_not_pin() -> None:
    fake = FakeLlama(slots_status=404)
    res = probe_llm(URL, rounds=1, http=fake.client())
    assert res["error"] is None and res["slot"] is None and res["pp"] == 900.0
    assert any("/slots" in w for w in res["warnings"])
    assert all("id_slot" not in b for b in fake.bodies)
    assert fake.erased == []


def test_probe_llm_without_props_warns_but_measures() -> None:
    fake = FakeLlama(props_status=500)
    res = probe_llm(URL, rounds=1, http=fake.client())
    assert res["error"] is None and any("/props" in w for w in res["warnings"])
    assert res["prompt_tokens"] == 1500          # unknown window: the default prompt, not a crash


def test_probe_llm_cancel_between_requests_returns_partial_and_still_erases() -> None:
    fake = FakeLlama()
    checks = {"n": 0}

    def cancel() -> bool:
        checks["n"] += 1
        return checks["n"] >= 5                  # props, A, B, C of round 1 pass; round 2's A is stopped

    res = probe_llm(URL, rounds=2, http=fake.client(), cancel=cancel)
    assert res["error"] == "cancelled" and res["cancelled"] is True
    assert len(fake.bodies) == 3 and res["pp"] == 900.0 and len(res["rounds"]) == 2 and res["rounds"][1] == {"round": 1}
    assert fake.erased == ["/slots/1"]


def test_probe_llm_cancel_before_anything_makes_no_request() -> None:
    fake = FakeLlama()
    res = probe_llm(URL, http=fake.client(), cancel=lambda: True)
    assert res["error"] == "cancelled" and fake.calls == []


def test_probe_llm_a_raising_cancel_predicate_does_not_abort() -> None:
    def cancel() -> bool:
        raise RuntimeError("ui went away")

    res = probe_llm(URL, rounds=1, http=FakeLlama().client(), cancel=cancel)
    assert res["error"] is None and res["pp"] == 900.0


def test_probe_llm_http_error_returns_an_error_dict_with_what_was_measured() -> None:
    fake = FakeLlama(fail_on_completion=4)       # round 1 completes, round 2's first request returns 500
    res = probe_llm(URL, rounds=2, http=fake.client())
    assert "500" in res["error"] and "/v1/completions" in res["error"]
    assert res["pp"] == 900.0 and len(res["rounds"]) == 2
    assert fake.erased == ["/slots/1"]
    all_fail = probe_llm(URL, rounds=1, http=FakeLlama(completions_status=503).client())
    assert "503" in all_fail["error"] and all_fail["pp"] is None


def test_probe_llm_connection_failure_is_an_error_not_an_exception() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("All connection attempts failed", request=request)

    res = probe_llm(URL, http=httpx.Client(transport=httpx.MockTransport(refuse)))
    assert res["error"] and "ConnectError" in res["error"]
    assert res["pp"] is None and res["rounds"] == []


def test_probe_llm_missing_timings_is_an_error() -> None:
    res = probe_llm(URL, rounds=1, http=FakeLlama(timings=False).client())
    assert "timings" in res["error"] and res["pp"] is None and res["tg"] is None


def test_probe_llm_flags_a_noisy_measurement() -> None:
    res = probe_llm(URL, rounds=2, http=FakeLlama(pps=[500.0, 1000.0]).client())
    assert res["spread"] == pytest.approx(500 / 750, abs=1e-3) and res["noisy"] is True
    quiet = probe_llm(URL, rounds=2, http=FakeLlama(pps=[950.0, 960.0]).client())
    assert quiet["noisy"] is False


def test_probe_llm_prompt_size_follows_the_window() -> None:
    sizes = {}
    for n_ctx in (4096, 32768):
        fake = FakeLlama(n_ctx=n_ctx)
        res = probe_llm(URL, rounds=1, http=fake.client())
        sizes[n_ctx] = (res["prompt_tokens"], len(fake.bodies[0]["prompt"]))
    assert sizes[4096][0] == 1024 and sizes[32768][0] == 1500
    assert sizes[4096][1] < sizes[32768][1]
    fake = FakeLlama()
    res = probe_llm(URL, rounds=1, prompt_tokens=800, n_ctx=2048, http=fake.client())
    assert res["prompt_tokens"] == 800 and res["n_ctx"] == 2048 and len(fake.bodies[0]["prompt"]) >= 4 * (800 - 80)
    tiny = probe_llm(URL, rounds=1, prompt_tokens=5000, n_ctx=2048, http=FakeLlama().client())
    assert tiny["prompt_tokens"] < 2048


def test_probe_llm_accepts_a_client_factory_and_closes_only_what_it_made() -> None:
    fake = FakeLlama()
    made: list[httpx.Client] = []

    def factory() -> httpx.Client:
        made.append(fake.client())
        return made[-1]

    res = probe_llm(URL, rounds=1, http=factory)
    assert res["error"] is None and made[0].is_closed
    shared = FakeLlama().client()
    probe_llm(URL, rounds=1, http=shared)
    assert not shared.is_closed
    shared.close()


def test_probe_llm_never_constructs_a_llama_server_manager(monkeypatch: pytest.MonkeyPatch) -> None:
    from exocortex.llm import server

    def boom(*a: Any, **k: Any) -> None:
        raise AssertionError("the probe must not build a LlamaServerManager")

    monkeypatch.setattr(server.LlamaServerManager, "__init__", boom)
    monkeypatch.setattr(server.LlamaServerManager, "start", boom)
    monkeypatch.setattr(server.LlamaServerManager, "swap_model", boom)
    res = probe_llm(URL, rounds=1, http=FakeLlama().client())
    assert res["error"] is None and res["pp"] == 900.0


def test_build_transcript_is_seed_unique_and_ends_with_the_question() -> None:
    a, b = optimize_bench.build_transcript(1200, 1), optimize_bench.build_transcript(1200, 2)
    assert a != b and a.rstrip().endswith(optimize_bench.QUESTION)
    assert not set(a.split("\n")[:-1]) & set(b.split("\n")[:-1])      # only the shared question line repeats
    assert len(a) >= 4 * 1000
    extra = optimize_bench.extra_transcript_lines(200, 1, 10_000)
    assert "follow-up" in extra and len(extra) >= 4 * 150
    joined = optimize_bench._insert_before_question(a, extra)
    assert joined.split("\n")[-1] == a.split("\n")[-1] and extra in joined


# ============================================================================================
# run_child
# ============================================================================================
def make_popen(*, exit_after: int | None = None, payload: dict | None = None, raw: str | None = None, rc: int = 0,
               record: list | None = None):
    record = record if record is not None else []

    class FakeProc:
        def __init__(self, cmd: list[str], **kw: Any) -> None:
            self.cmd, self.kw, self.waits, self.killed, self.returncode = cmd, kw, 0, False, None
            record.append(self)

        def wait(self, timeout: float | None = None) -> int | None:
            self.waits += 1
            if self.killed:
                return self.returncode
            if exit_after is not None and self.waits > exit_after:
                out = Path(self.cmd[self.cmd.index("--out") + 1])
                if payload is not None:
                    out.write_text(json.dumps(payload), encoding="utf-8")
                if raw is not None:
                    out.write_text(raw, encoding="utf-8")
                self.returncode = rc
                return rc
            raise subprocess.TimeoutExpired(self.cmd, timeout or 0)

        def kill(self) -> None:
            self.killed = True
            self.returncode = -9

    return FakeProc


def test_run_child_returns_the_result_file_and_launches_correctly(tmp_path: Path) -> None:
    record: list = []
    out = tmp_path / "res" / "devices.json"
    procs: set = set()
    seen: list[int] = []
    result = run_child(["probe-devices"], out_path=out, deadline_s=30.0, procs=procs, popen=make_popen(
        exit_after=2, payload={"igpu": "Arc", "ct2_cuda": True}, record=record),
        cancel=lambda: seen.append(len(procs)) or False)
    assert result == {"igpu": "Arc", "ct2_cuda": True}
    assert seen == [1, 1] and procs == set()                               # tracked while it ran, released after
    proc = record[0]
    assert proc.cmd[0] == sys.executable and proc.cmd[1:4] == ["-m", "exocortex.optimize_bench", "probe-devices"]
    assert proc.cmd[-2:] == ["--out", str(out)]
    kw = proc.kw
    assert kw["cwd"] == str(project_root())
    assert kw["env"]["HF_HUB_OFFLINE"] == "1"
    assert kw["stdin"] == kw["stdout"] == kw["stderr"] == subprocess.DEVNULL
    if sys.platform == "win32":
        assert kw["creationflags"] & 0x08000000 and kw["creationflags"] & 0x00004000     # no window, below normal
    else:
        assert "creationflags" not in kw
    assert not proc.killed


def test_run_child_python_and_cwd_overrides(tmp_path: Path) -> None:
    record: list = []
    run_child(["probe-asr", "--variant", "sherpa-cpu"], out_path=tmp_path / "o.json", deadline_s=5, python="py-x", cwd=tmp_path,
              popen=make_popen(exit_after=0, payload={"label": "sherpa-cpu"}, record=record))
    assert record[0].cmd[0] == "py-x" and record[0].kw["cwd"] == str(tmp_path)
    assert record[0].cmd[3:7] == ["probe-asr", "--variant", "sherpa-cpu", "--out"]


def test_run_child_kills_at_the_deadline(tmp_path: Path) -> None:
    record: list = []
    procs: set = set()
    t0 = time.monotonic()
    result = run_child(["probe-asr"], out_path=tmp_path / "o.json", deadline_s=0.05, procs=procs, popen=make_popen(record=record))
    assert result == {"error": "timeout after 0.05 s"}
    assert record[0].killed and procs == set()
    assert time.monotonic() - t0 < 5


def test_run_child_cancel_kills_and_reports_cancelled(tmp_path: Path) -> None:
    record: list = []
    result = run_child(["x"], out_path=tmp_path / "o.json", deadline_s=60, popen=make_popen(record=record),
                       cancel=lambda: record[0].waits >= 3)
    assert result == {"error": "cancelled"} and record[0].killed and record[0].waits >= 3


def test_run_child_still_ok_false_kills_and_reports_cancelled(tmp_path: Path) -> None:
    record: list = []
    procs: set = set()
    state = {"checks": 0}

    def still_ok() -> bool:
        state["checks"] += 1
        return state["checks"] < 3                # speech resumes on the third look

    result = run_child(["x"], out_path=tmp_path / "o.json", deadline_s=60, procs=procs, popen=make_popen(record=record),
                       still_ok=still_ok)
    assert result == {"error": "cancelled"} and record[0].killed and procs == set()
    assert state["checks"] == 3


def test_run_child_broken_predicates_neither_kill_nor_hang(tmp_path: Path) -> None:
    def boom() -> bool:
        raise RuntimeError("x")

    result = run_child(["x"], out_path=tmp_path / "o.json", deadline_s=30, popen=make_popen(exit_after=2, payload={"ok": 1}),
                       cancel=boom, still_ok=boom)
    assert result == {"ok": 1}


def test_run_child_missing_garbage_and_non_object_results(tmp_path: Path) -> None:
    out = tmp_path / "o.json"
    missing = run_child(["x"], out_path=out, deadline_s=30, popen=make_popen(exit_after=0))
    assert missing == {"error": "no result file"}
    crashed = run_child(["x"], out_path=out, deadline_s=30, popen=make_popen(exit_after=0, rc=3))
    assert crashed["error"].startswith("no result file") and "3" in crashed["error"]
    garbage = run_child(["x"], out_path=out, deadline_s=30, popen=make_popen(exit_after=0, raw="{half a js"))
    assert garbage["error"].startswith("unreadable result")
    listy = run_child(["x"], out_path=out, deadline_s=30, popen=make_popen(exit_after=0, raw="[1, 2]"))
    assert listy["error"].startswith("unreadable result")


def test_run_child_never_reads_a_previous_runs_file(tmp_path: Path) -> None:
    out = tmp_path / "o.json"
    out.write_text(json.dumps({"from": "an earlier child"}), encoding="utf-8")
    result = run_child(["x"], out_path=out, deadline_s=30, popen=make_popen(exit_after=0))
    assert result == {"error": "no result file"}


def test_run_child_popen_failure_is_an_error_dict(tmp_path: Path) -> None:
    def broken(cmd: list[str], **kw: Any) -> Any:
        raise OSError("python.exe is blocked")

    procs: set = set()
    result = run_child(["x"], out_path=tmp_path / "o.json", deadline_s=5, popen=broken, procs=procs)
    assert "python.exe is blocked" in result["error"] and procs == set()


def test_run_child_real_subprocess_round_trip(tmp_path: Path) -> None:
    """The real Popen path end to end (no fakes, no hardware): the child module parses its arguments,
    answers an unknown variant as data, writes the --out file and exits 0."""
    out = tmp_path / "sub" / "asr.json"
    result = run_child(["probe-asr", "--variant", "no-such-variant", "--clip", str(tmp_path / "x.wav")], out_path=out,
                       deadline_s=120)
    assert result["label"] == "no-such-variant" and "unknown variant" in result["error"]
    bad = run_child(["not-a-command"], out_path=tmp_path / "o2.json", deadline_s=120)    # argparse exits 2
    assert bad["error"].startswith("no result file") and "2" in bad["error"]


# ============================================================================================
# small helpers
# ============================================================================================
def test_sample_vram_stable_median() -> None:
    readings = iter([20000, 20100, 20050])
    sleeps: list[float] = []
    assert sample_vram(read=lambda: next(readings), sleep=sleeps.append) == (20050, True)
    assert sleeps == [1.0, 1.0]                      # between readings, not after the last
    r2 = iter([20000, 20256, 20100])
    assert sample_vram(gap_s=0.1, read=lambda: next(r2), sleep=lambda s: None) == (20100, True)      # exactly 256 is stable


def test_sample_vram_unstable_missing_and_failing() -> None:
    r = iter([20000, 21000, 22000])
    assert sample_vram(read=lambda: next(r), sleep=lambda s: None) == (21000, False)
    assert sample_vram(read=lambda: None, sleep=lambda s: None) == (None, False)
    r3 = iter([None, 100, None])
    assert sample_vram(read=lambda: next(r3), sleep=lambda s: None) == (100, False)      # one reading of three: not stable

    def boom() -> int:
        raise OSError("nvml gone")

    assert sample_vram(read=boom, sleep=lambda s: None) == (None, False)
    assert sample_vram(n=1, read=lambda: 5000, sleep=lambda s: None) == (5000, True)


def test_sample_vram_defaults_to_the_gpu_module(monkeypatch: pytest.MonkeyPatch) -> None:
    from exocortex import gpu

    monkeypatch.setattr(gpu, "vram_free_mb", lambda: 12345)
    assert sample_vram(sleep=lambda s: None) == (12345, True)


@pytest.mark.parametrize("status,body,expected", [
    (200, {"running": [{"model": "fleet-27b", "state": "ready"}]}, True),
    (200, {"running": []}, False),
    (200, [{"model": "x"}], True),
    (200, [], False),
    (200, {"unexpected": 1}, None),
    (200, "text", None),
    (500, {"running": [{"model": "x"}]}, None),
    (404, {}, None),
])
def test_swap_resident_shapes(status: int, body: Any, expected: bool | None) -> None:
    calls: list[tuple[str, float]] = []

    def get(url: str, timeout: float = 0.0) -> Any:
        calls.append((url, timeout))
        return SimpleNamespace(status_code=status, json=lambda: body)

    assert swap_resident("http://127.0.0.1:8081/", http_get=get) is expected
    assert calls == [("http://127.0.0.1:8081/running", 1.5)]


def test_swap_resident_unreachable_malformed_and_empty() -> None:
    def refuse(url: str, timeout: float = 0.0) -> Any:
        raise httpx.ConnectError("refused")

    def not_json(url: str, timeout: float = 0.0) -> Any:
        def j() -> Any:
            raise ValueError("no json")
        return SimpleNamespace(status_code=200, json=j)

    assert swap_resident("http://x", http_get=refuse) is None
    assert swap_resident("http://x", http_get=not_json) is None
    assert swap_resident("", http_get=refuse) is None
    seen: list[float] = []
    swap_resident("http://x", timeout_s=0.5, http_get=lambda u, timeout=0: seen.append(timeout) or SimpleNamespace(status_code=200, json=lambda: []))
    assert seen == [0.5]


def test_swap_resident_default_getter_is_httpx(monkeypatch: pytest.MonkeyPatch) -> None:
    got: list[str] = []

    def fake_get(url: str, timeout: float = 0.0) -> Any:
        got.append(url)
        return SimpleNamespace(status_code=200, json=lambda: {"running": []})

    monkeypatch.setattr(optimize_bench.httpx, "get", fake_get)
    assert swap_resident("http://127.0.0.1:8081") is False and got == ["http://127.0.0.1:8081/running"]


def test_canary_ok_and_failures() -> None:
    bodies: list[dict] = []

    def ok(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        assert request.url.path == "/v1/completions"
        return httpx.Response(200, json={"choices": [{"text": "!"}]})

    good, ms = canary(URL, http=httpx.Client(transport=httpx.MockTransport(ok)))
    assert good is True and 0.0 <= ms < 5000.0
    assert bodies[0]["max_tokens"] == 1 and bodies[0]["stream"] is False and bodies[0]["prompt"]

    def server_error(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "x"})

    def no_choices(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": []})

    def refused(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    def not_json(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"<html>")

    for handler in (server_error, no_choices, refused, not_json):
        okay, took = canary(URL, http=httpx.Client(transport=httpx.MockTransport(handler)))
        assert okay is False and took >= 0.0


def test_canary_accepts_a_client_factory() -> None:
    made: list[httpx.Client] = []

    def factory() -> httpx.Client:
        made.append(httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"choices": [{}]}))))
        return made[-1]

    assert canary(URL, http=factory)[0] is True and made[0].is_closed


def test_python_cmd_is_this_interpreter() -> None:
    assert python_cmd() == sys.executable


# ============================================================================================
# child entry points
# ============================================================================================
OV = {"CPU": "Intel(R) Core(TM) Ultra 9 285H", "GPU.0": "Intel(R) Arc(TM) 140T GPU (32GB) (iGPU)",
      "GPU.1": "NVIDIA GeForce RTX 5090 Laptop GPU (dGPU)", "NPU": "Intel(R) AI Boost"}


def _patch_probes(monkeypatch: pytest.MonkeyPatch, **over: Any) -> None:
    defaults = {"_probe_openvino": lambda: dict(OV), "_probe_ct2_cuda": lambda: True,
                "_probe_mem_mts": lambda: 7467, "_probe_power": lambda: {"power_capped": True, "power_w": 115.0}}
    defaults.update(over)
    for name, fn in defaults.items():
        monkeypatch.setattr(optimize_bench, name, fn)


@pytest.mark.parametrize("devices,igpu,npu", [
    (OV, "Intel(R) Arc(TM) 140T GPU (32GB) (iGPU)", "Intel(R) AI Boost"),
    ({"CPU": "x", "GPU.0": "Intel(R) Arc(TM) A370M Graphics"}, "Intel(R) Arc(TM) A370M Graphics", ""),
    ({"CPU": "x", "GPU.0": "NVIDIA GeForce RTX 5090 Laptop GPU (dGPU)"}, "", ""),
    ({"CPU": "x", "GPU.0": "Intel(R) Arc(TM) B580 Graphics (dGPU)"}, "", ""),
    ({}, "", ""),
])
def test_classify_openvino(devices: dict, igpu: str, npu: str) -> None:
    assert optimize_bench._classify_openvino(devices) == (igpu, npu)


def test_collect_devices_assembles_every_probe(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    bundle = tmp_path / "parakeet-tdt-0.6b-v2-int8"
    bundle.mkdir()
    for f in ("encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt"):
        (bundle / f).write_bytes(b"x")
    _patch_probes(monkeypatch)
    d = optimize_bench.collect_devices(tmp_path)
    assert d["openvino"] == OV and d["igpu"].endswith("(iGPU)") and d["npu"] == "Intel(R) AI Boost"
    assert d["ct2_cuda"] is True and d["mem_mts"] == 7467 and d["power"]["power_capped"] is True
    assert d["sherpa_ready"] is True and d["sherpa_models"] == {"parakeet-tdt-0.6b-v2-int8": True, "parakeet-tdt-0.6b-v3-int8": False}
    assert d["errors"] == {}
    json.dumps(d)                                                          # the file the child writes must serialise


def test_collect_devices_every_probe_fails_soft(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def boom() -> Any:
        raise RuntimeError("dll load failed")

    _patch_probes(monkeypatch, _probe_openvino=boom, _probe_ct2_cuda=boom, _probe_mem_mts=boom, _probe_power=boom)
    d = optimize_bench.collect_devices(tmp_path)
    assert d["openvino"] == {} and d["igpu"] == "" and d["npu"] == "" and d["ct2_cuda"] is False
    assert d["mem_mts"] is None and d["power"] is None and d["sherpa_ready"] is False
    assert set(d["errors"]) == {"openvino", "ct2_cuda", "mem_mts", "power"} and "dll load failed" in d["errors"]["ct2_cuda"]


def test_collect_devices_checks_the_requested_sherpa_bundle(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    v3 = tmp_path / "parakeet-tdt-0.6b-v3-int8"
    v3.mkdir()
    for f in ("encoder.onnx", "decoder.onnx", "joiner.onnx", "tokens.txt"):
        (v3 / f).write_bytes(b"x")
    _patch_probes(monkeypatch)
    assert optimize_bench.collect_devices(tmp_path)["sherpa_ready"] is False
    assert optimize_bench.collect_devices(tmp_path, "parakeet-tdt-0.6b-v3-int8")["sherpa_ready"] is True


class FakeTranscriber:
    """Stands in for exocortex.speech.asr.Transcriber; calls the injected ensure_sherpa_model like the real one."""
    instances: list["FakeTranscriber"] = []
    fallback = False
    texts: list[str] = []
    elapsed: list[float] = []

    def __init__(self, cfg: Any, models_dir: Path | None = None, ensure_sherpa_model: Any = None, **kw: Any) -> None:
        self.cfg, self.models_dir, self.ensure = cfg, models_dir, ensure_sherpa_model
        self.engine_used = self.device_used = self.compute_type_used = ""
        self.heard: list[tuple[int, int]] = []
        FakeTranscriber.instances.append(self)

    def load(self) -> None:
        self.engine_used = self.cfg.engine
        if self.cfg.engine == "sherpa":
            self.ensure(self.cfg.sherpa_model, self.models_dir, None)
            self.device_used, self.compute_type_used = "cpu", "int8"
        elif FakeTranscriber.fallback:
            self.device_used, self.compute_type_used = "cpu", "int8"
        else:
            self.device_used, self.compute_type_used = self.cfg.device, self.cfg.compute_type

    def transcribe(self, audio: Any, *, partial: bool = False) -> Any:
        self.heard.append((audio[0], len(audio)))
        i = len(self.heard) - 1
        text = FakeTranscriber.texts[i] if i < len(FakeTranscriber.texts) else ""
        return SimpleNamespace(text=text, elapsed_s=FakeTranscriber.elapsed[i % len(FakeTranscriber.elapsed)])


@pytest.fixture()
def asr_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    FakeTranscriber.instances, FakeTranscriber.fallback = [], False
    longest = sorted(optimize_bench.TWO_SPEAKERS_TRUTH, key=lambda r: r[1] - r[0], reverse=True)[:4]
    FakeTranscriber.texts = [r[3] for r in longest] * 2
    FakeTranscriber.elapsed = [0.10, 0.12, 0.14, 0.16]
    clip = tmp_path / "clip.wav"
    clip.write_bytes(b"RIFF")
    audio = list(range(16000 * 30))
    # a probe must never reach the real download code
    from exocortex import models_dl

    def no_download(*a: Any, **k: Any) -> None:
        raise AssertionError("a probe tried to download a sherpa bundle")

    monkeypatch.setattr(models_dl, "ensure_sherpa_model", no_download)
    monkeypatch.setattr(models_dl, "_download_one", no_download)
    return SimpleNamespace(clip=clip, audio=audio, models=tmp_path / "models", longest=longest, tmp=tmp_path)


def _probe(env: SimpleNamespace, variant: str, **kw: Any) -> dict:
    kw.setdefault("load_audio", lambda p: (env.audio, 16000))
    return optimize_bench.probe_asr(variant, models_dir=env.models, clip=env.clip, transcriber_cls=FakeTranscriber,
                                    sleep=lambda s: None, **kw)


def _bundle(models: Path, name: str = "parakeet-tdt-0.6b-v2-int8") -> None:
    d = models / name
    d.mkdir(parents=True)
    for f in ("encoder.onnx", "decoder.onnx", "joiner.onnx", "tokens.txt"):
        (d / f).write_bytes(b"x")


def test_probe_asr_sherpa_result_schema_and_the_four_longest_utterances(asr_env: SimpleNamespace) -> None:
    _bundle(asr_env.models)
    res = _probe(asr_env, "sherpa-cpu", threads=6)
    assert res["error"] is None
    assert set(res) == {"label", "engine", "device", "compute_type", "threads", "load_s", "median_s", "max_s", "overlap",
                        "vram_mb", "vram_source", "error"}
    assert (res["label"], res["engine"], res["device"], res["compute_type"], res["threads"]) == ("sherpa-cpu", "sherpa", "cpu", "int8", 6)
    assert res["median_s"] == pytest.approx(0.13) and res["max_s"] == 0.16 and res["overlap"] == 1.0
    assert res["vram_mb"] is None and res["vram_source"] == "none"          # a CPU engine is never sampled
    assert res["load_s"] is not None and res["load_s"] >= 0.0
    t = FakeTranscriber.instances[0]
    assert t.cfg.cpu_threads == 6 and t.cfg.engine == "sherpa"
    expected = [(int(s * 16000), int((e - s) * 16000) + 0) for s, e, _sp, _tx in asr_env.longest] * 2
    assert [h[0] for h in t.heard] == [e[0] for e in expected]              # the four longest, in duration order, twice
    assert len(t.heard) == 8
    blob = json.dumps(res)
    assert "Kafka" not in blob and "everyone" not in blob                   # the result carries no transcript text


def test_probe_asr_overlap_reflects_what_was_heard(asr_env: SimpleNamespace) -> None:
    _bundle(asr_env.models)
    FakeTranscriber.texts = ["", *FakeTranscriber.texts[1:]]
    res = _probe(asr_env, "sherpa-cpu")
    assert res["overlap"] == pytest.approx(7 / 8, abs=1e-3)


def test_probe_asr_sherpa_never_downloads_a_missing_bundle(asr_env: SimpleNamespace) -> None:
    res = _probe(asr_env, "sherpa-cpu")                        # models dir does not exist
    assert "never downloads" in res["error"] and "FileNotFoundError" in res["error"]
    assert res["median_s"] is None and res["overlap"] is None
    stand_in = optimize_bench._offline_sherpa
    with pytest.raises(FileNotFoundError):
        stand_in("parakeet-tdt-0.6b-v2-int8", asr_env.models, None)
    _bundle(asr_env.models)
    assert stand_in("parakeet-tdt-0.6b-v2-int8", asr_env.models, None) == asr_env.models / "parakeet-tdt-0.6b-v2-int8"


def test_probe_asr_whisper_cpu_uses_the_thread_count(asr_env: SimpleNamespace) -> None:
    res = _probe(asr_env, "whisper-cpu-int8", threads=8)
    cfg = FakeTranscriber.instances[0].cfg
    assert res["error"] is None and res["threads"] == 8
    assert (cfg.engine, cfg.device, cfg.compute_type, cfg.cpu_threads) == ("faster-whisper", "cpu", "int8", 8)
    assert res["vram_source"] == "none"


def test_probe_asr_cuda_variant_measures_vram(asr_env: SimpleNamespace) -> None:
    readings = iter([20000, 20010, 19990, 19100, 19110, 19090])
    res = _probe(asr_env, "whisper-cuda-int8_float16", read_vram=lambda: next(readings), threads=4)
    cfg = FakeTranscriber.instances[0].cfg
    assert res["error"] is None and (cfg.device, cfg.compute_type, cfg.cpu_threads) == ("cuda", "int8_float16", 0)
    assert res["vram_mb"] == 900 and res["vram_source"] == "nvml" and res["threads"] == 0


@pytest.mark.parametrize("readings", [
    [19000, 19010, 18990, 20000, 20010, 19990],        # free memory GREW: a negative cost
    [20000, 20300, 20100, 19100, 19110, 19090],        # the "before" triple wobbles by 300 MiB
    [20000, 20010, 19990, 19100, 19400, 19090],        # the "after" triple wobbles by 310 MiB
    [20000, 20010, 19990],                              # NVML died after the first triple
])
def test_probe_asr_rejects_untrustworthy_vram(asr_env: SimpleNamespace, readings: list[int]) -> None:
    it = iter(readings)
    res = _probe(asr_env, "whisper-cuda-float16", read_vram=lambda: next(it, None))
    assert res["error"] is None and res["vram_mb"] is None and res["vram_source"] == "none"
    assert res["median_s"] is not None                                      # the timing is still reported


def test_probe_asr_a_cuda_variant_that_fell_back_to_cpu_is_an_error(asr_env: SimpleNamespace) -> None:
    FakeTranscriber.fallback = True
    res = _probe(asr_env, "whisper-cuda-int8_float16", read_vram=lambda: 20000)
    assert "unavailable" in res["error"] and "cpu" in res["error"]
    assert res["median_s"] is None and res["vram_mb"] is None


def test_probe_asr_reports_instead_of_raising(asr_env: SimpleNamespace) -> None:
    assert "unknown variant" in _probe(asr_env, "whisper-turbo-9000")["error"]
    missing = optimize_bench.probe_asr("sherpa-cpu", models_dir=asr_env.models, clip=asr_env.tmp / "nope.wav",
                                       transcriber_cls=FakeTranscriber)
    assert "clip not found" in missing["error"]
    wrong_rate = _probe(asr_env, "whisper-cpu-int8", load_audio=lambda p: (asr_env.audio, 44100))
    assert "16 kHz" in wrong_rate["error"]

    def bad_audio(p: Path) -> Any:
        raise RuntimeError("soundfile exploded")

    assert "soundfile exploded" in _probe(asr_env, "whisper-cpu-int8", load_audio=bad_audio)["error"]


def test_truth_table_matches_the_benchmark_script() -> None:
    script = Path(__file__).resolve().parent.parent / "scripts" / "bench_asr.py"
    if not script.is_file():
        pytest.skip("scripts/ not present")
    text = script.read_text(encoding="utf-8")
    for start, end, _speaker, utterance in optimize_bench.TWO_SPEAKERS_TRUTH:
        assert f'({start:.2f}, {end:.2f},' in text and utterance in text
    assert len(optimize_bench.TWO_SPEAKERS_TRUTH) == 9
    assert optimize_bench._word_overlap("hi everyone", "Hi everyone!") == 1.0
    assert optimize_bench._word_overlap("", "Hi everyone!") == 0.0
    assert optimize_bench._word_overlap("anything", "") == 1.0


def test_main_probe_devices_writes_its_out_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch_probes(monkeypatch)
    out = tmp_path / "devices.json"
    assert optimize_bench.main(["probe-devices", "--out", str(out), "--models-dir", str(tmp_path)]) == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["igpu"].endswith("(iGPU)") and data["ct2_cuda"] is True and data["mem_mts"] == 7467


def test_main_probe_devices_prints_without_out(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    _patch_probes(monkeypatch)
    assert optimize_bench.main(["probe-devices", "--models-dir", str(tmp_path)]) == 0
    assert json.loads(capsys.readouterr().out)["npu"] == "Intel(R) AI Boost"


def test_main_probe_asr_always_answers_and_stays_offline(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")           # registers the restore of what main() sets
    out = tmp_path / "asr.json"
    rc = optimize_bench.main(["probe-asr", "--variant", "no-such-variant", "--clip", str(tmp_path / "x.wav"),
                              "--models-dir", str(tmp_path), "--out", str(out)])
    assert rc == 0 and os.environ["HF_HUB_OFFLINE"] == "1"
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["label"] == "no-such-variant" and "unknown variant" in data["error"]


def test_main_probe_asr_survives_an_unexpected_exception(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")

    def boom(*a: Any, **k: Any) -> dict:
        raise RuntimeError("kaboom")

    monkeypatch.setattr(optimize_bench, "probe_asr", boom)
    out = tmp_path / "asr.json"
    assert optimize_bench.main(["probe-asr", "--variant", "sherpa-cpu", "--clip", "c.wav", "--out", str(out)]) == 0
    assert "kaboom" in json.loads(out.read_text(encoding="utf-8"))["error"]


def test_main_out_file_that_cannot_be_written_returns_1(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch_probes(monkeypatch)
    blocker = tmp_path / "file"
    blocker.write_text("x", encoding="utf-8")
    assert optimize_bench.main(["probe-devices", "--out", str(blocker / "sub" / "d.json"), "--models-dir", str(tmp_path)]) == 1


def test_main_import_adds_files_to_the_store(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    f = tmp_path / "run.json"
    f.write_text(json.dumps(bench_llm_json()), encoding="utf-8")
    store_dir = tmp_path / "lib"
    assert optimize_bench.main(["import", str(f), str(tmp_path / "gone.json"), "--store-dir", str(store_dir)]) == 0
    text = capsys.readouterr().out
    assert "imported 1" in text and "skipped 1" in text and "gone.json" in text
    assert len(Store(store_dir).evidence(None, NOW).llm) == 1
    assert optimize_bench.main(["import", str(tmp_path / "gone.json"), "--store-dir", str(store_dir)]) == 1


def test_main_import_defaults_to_the_configured_library(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from exocortex.config import Config
    from exocortex.llm import library

    cfg = Config()
    cfg.llm.library_dir = str(tmp_path / "configured")
    monkeypatch.setattr("exocortex.config.load_config", lambda path=None: cfg)
    f = tmp_path / "run.json"
    f.write_text(json.dumps(bench_llm_json()), encoding="utf-8")
    assert optimize_bench.main(["import", str(f)]) == 0
    assert (library.library_dir(cfg.llm) / BENCH_FILE).is_file()


def test_importing_the_module_loads_no_native_stack() -> None:
    code = ("import sys, exocortex.optimize_bench;"
            "bad=[m for m in ('numpy','ctranslate2','openvino','sherpa_onnx','faster_whisper','onnxruntime',"
            "'exocortex.llm.server','exocortex.screen.ocr','PySide6') if m in sys.modules];"
            "print(','.join(bad))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60, cwd=str(project_root()))
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == ""


def test_import_treats_the_two_everyday_llama_server_warnings_as_clean() -> None:
    """Every healthy launch with two slots prints "cache_reuse is not supported by this context", and
    --fit prints "common_fit_params: failed to fit params ... already set by user" because the app pins
    -ngl; neither may flag the run unclean (that would veto every footprint and every knob)."""
    from exocortex.optimize_bench import import_bench_llm_json

    base = {"requested": {"model": "m/x", "ctx_size": 8192, "kv_cache_type": "q8_0", "parallel": 2, "load_mode": "mmap",
                          "llama_extra_args": ["--ctx-size", "32768"]},
            "model": {"key": "m/x"}, "medians": {"pp": 900.0, "tg": 20.0, "cached_ttft_ms": 500.0}, "gpu_used_mb": 16932,
            "aborted": False, "rounds": []}
    benign = dict(base, log_signals={"warnings": [
        "\x1b[34m0.1\x1b[0m \x1b[35mW srv load_model: cache_reuse is not supported by this context, it will be disabled",
        "W common_fit_params: failed to fit params to free device memory: n_gpu_layers already set by user to 999999, abort",
    ]})
    run, fp = import_bench_llm_json(benign, at="2026-10-07T21:00:00")
    assert run is not None and run.clean_log and fp is not None and fp.app_mb == 16932 and fp.ctx == 32768
    bad = dict(base, log_signals={"warnings": ["\x1b[31mE ggml_cuda: out of memory\x1b[0m"]})
    run2, fp2 = import_bench_llm_json(bad, at="2026-10-07T21:00:00")
    assert run2 is not None and not run2.clean_log and fp2 is None


def test_import_llm_uses_the_model_profile_for_the_effective_context() -> None:
    """``requested.ctx_size`` is the GLOBAL value; the server ran with the model's profile overlaid
    (the owner's 81,920-token profile over an 8,192 global). The profile wins unless the extra args
    repeat the flag, which wins over both."""
    from exocortex.optimize_bench import import_bench_llm_json

    base = {"requested": {"model": "m/x", "ctx_size": 8192, "kv_cache_type": "q8_0", "parallel": 2, "load_mode": "mmap",
                          "llama_extra_args": []},
            "model": {"key": "m/x"}, "medians": {"pp": 900.0, "tg": 20.0, "cached_ttft_ms": 500.0}, "gpu_used_mb": 18884,
            "aborted": False, "rounds": [], "log_signals": {"warnings": []}}
    run, fp = import_bench_llm_json(base, at="2026-10-07T21:00:00", profile={"ctx_size": 81920, "parallel": None, "kv_cache_type": None})
    assert run is not None and run.ctx == 81920 and run.parallel == 2 and run.kv == "q8_0"
    assert fp is not None and fp.ctx == 81920
    run2, _fp2 = import_bench_llm_json(base, at="x")
    assert run2 is not None and run2.ctx == 8192
    forced = dict(base, requested=dict(base["requested"], llama_extra_args=["--ctx-size", "32768", "--parallel", "1"]))
    run3, _fp3 = import_bench_llm_json(forced, at="x", profile={"ctx_size": 81920})
    assert run3 is not None and run3.ctx == 32768 and run3.parallel == 1

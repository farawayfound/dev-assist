"""Tests for exocortex.optimize, the pure Optimize planner (2026-10-07). Everything here is
arithmetic over hand-built facts: no hardware, no files, no network. The fixtures mirror the
owner's machine (RTX 5090 Laptop 24 GB, Qwen3.8-27B Q4_K_M -- a hybrid ``qwen35`` header) because
those are the numbers the planner was built against; a dense model and a GPU-less machine cover
the other branches."""
from __future__ import annotations

import dataclasses
import json
import socket
import subprocess
from pathlib import Path

import pytest

from exocortex import optimize as O
from exocortex.config import LLMConfig

# --------------------------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------------------------
TOTAL_MB = 24463


def qwen35_27b() -> O.ModelFacts:
    """The real header of lmstudio-community/Qwen3.8-27B-Q4_K_M.gguf (read 2026-10-07)."""
    return O.ModelFacts(key="lmstudio-community/Qwen3.8-27B", size_bytes=16_810_714_336, arch="qwen35", n_layer=65,
                        n_ctx_train=262144, n_head_kv=4, head_dim=256, kv_heads_total=260, full_attention_interval=4,
                        nextn_layers=1, ssm_conv=4, ssm_state=128, ssm_inner=6144, ssm_groups=16)


def dense_8b() -> O.ModelFacts:
    return O.ModelFacts(key="acme/dense-8b", size_bytes=4_900_000_000, arch="llama", n_layer=32, n_ctx_train=131072,
                        n_head_kv=8, head_dim=128, kv_heads_total=256)


def gpu_5090(**kw) -> O.GpuFacts:
    base = dict(name="NVIDIA GeForce RTX 5090 Laptop GPU", total_mb=TOTAL_MB, free_mb=22400, baseline_mb=2063,
                stable=True, swap_resident=False)
    base.update(kw)
    return O.GpuFacts(**base)


def machine_5090(**kw) -> O.Machine:
    base = dict(cpu="Intel(R) Core(TM) Ultra 9 285H", physical_cores=16, ram_total_mb=64957, ram_free_mb=30000,
                mem_mts=7467, gpu=gpu_5090(), igpu="Intel(R) Arc(TM) 140T GPU (iGPU)", npu="Intel(R) AI Boost",
                devices_known=True, ct2_cuda=True, sherpa_ready=True, runtime="nvidia-cuda12 2.44.0")
    base.update(kw)
    return O.Machine(**base)


def launch_owner(**kw) -> O.LaunchConfig:
    base = dict(ctx=81920, kv="q8_0", parallel=2, ubatch=512, threads=0, flash=True, gpu_layers=999999)
    base.update(kw)
    return O.LaunchConfig(**base)


def cfg_owner(**kw) -> O.CfgFacts:
    base = dict(asr_now="whisper-cpu-int8", ocr_now="igpu", asr_device="cpu", asr_compute="int8", asr_threads=4)
    base.update(kw)
    return O.CfgFacts(**base)


def asr_sweep() -> tuple[O.AsrRow, ...]:
    return (
        O.AsrRow(label="whisper-cuda-int8_float16", engine="faster-whisper", device="cuda", compute="int8_float16",
                 median_s=0.110, overlap=1.0, vram_mb=874, vram_source="nvml", source="sweep"),
        O.AsrRow(label="whisper-cuda-float16", engine="faster-whisper", device="cuda", compute="float16",
                 median_s=0.125, overlap=1.0, vram_mb=2374, vram_source="nvml", source="sweep"),
        O.AsrRow(label="whisper-cpu-int8", engine="faster-whisper", device="cpu", compute="int8", threads=4,
                 median_s=6.22, overlap=1.0, vram_mb=0, source="sweep"),
        O.AsrRow(label="sherpa-cpu", engine="sherpa", device="cpu", compute="int8", threads=4,
                 median_s=0.343, overlap=0.99, vram_mb=0, source="sweep"),
    )


def footprints_sweep(key: str = "lmstudio-community/Qwen3.8-27B") -> tuple[O.Footprint, ...]:
    return tuple(O.Footprint(model=key, ctx=c, kv="q8_0", app_mb=u, source="sweep")
                 for c, u in ((8192, 16059), (32768, 16932), (81920, 18884), (131072, 18866)))


def inputs(goal: str = "balanced", **kw) -> O.PlanInputs:
    base = dict(settings=O.Settings(goal=goal), machine=machine_5090(), model=qwen35_27b(), launch=launch_owner(),
                profile={"ctx_size": 81920, "auto": False, "extra_args": []}, cfg=cfg_owner(),
                evidence=O.Evidence(asr=asr_sweep(), footprints=footprints_sweep()), now_epoch=1_000_000.0)
    base.update(kw)
    return O.PlanInputs(**base)


# --------------------------------------------------------------------------------------------
# KV math
# --------------------------------------------------------------------------------------------
def test_hybrid_kv_bytes_per_token_counts_only_attention_layers() -> None:
    m = qwen35_27b()
    assert m.attn_layers == 16 and m.is_hybrid and m.has_mtp
    assert O.kv_bytes_per_token(m, "q8_0") == (34816, "header")
    assert O.kv_bytes_per_token(m, "f16") == (65536, "header")
    assert O.kv_bytes_per_token(m, "q4_0") == (18432, "header")
    # the dense formula (what exocortex.llm.library.kv_bytes_per_token gives) is 4x larger
    dense = O.ModelFacts(key="x", n_layer=65, kv_heads_total=260, head_dim=256)
    assert O.kv_bytes_per_token(dense, "q8_0") == (141440, "header")


def test_partial_hybrid_header_falls_back_to_the_pessimistic_bound() -> None:
    m = dataclasses.replace(qwen35_27b(), n_head_kv=None)
    bytes_, tier = O.kv_bytes_per_token(m, "q8_0")
    assert tier == "pessimistic" and bytes_ == 141440
    assert O.kv_bytes_per_token(O.ModelFacts(key="y"), "q8_0") == (None, "unknown")


def test_recurrent_state_is_about_150_mib_per_slot_and_zero_for_dense() -> None:
    m = qwen35_27b()
    per_slot = O.recurrent_mb(m, 1)
    assert 140 <= per_slot <= 160
    assert O.recurrent_mb(m, 2) == pytest.approx(2 * per_slot)
    assert O.recurrent_mb(dense_8b(), 2) == 0.0
    no_ssm = dataclasses.replace(m, ssm_inner=None)
    assert O.recurrent_mb(no_ssm, 3) == 3 * O.STATE_MB_PER_SLOT_FALLBACK


def test_compute_mb_interpolates_and_clamps() -> None:
    assert O.compute_mb(512) == 700.0
    assert O.compute_mb(2048) == 1900.0
    assert 700.0 < O.compute_mb(768) < 1100.0
    assert O.compute_mb(64) == 700.0 and O.compute_mb(8192) == 1900.0


# --------------------------------------------------------------------------------------------
# VRAM model
# --------------------------------------------------------------------------------------------
@pytest.mark.parametrize("goal", O.GOALS)
@pytest.mark.parametrize("kv", ["q8_0", "f16", "q4_0"])
@pytest.mark.parametrize("free", [6000, 12000, 18000, 22400, 24000])
def test_max_ctx_inverts_need_exactly_and_never_exceeds_the_budget(goal: str, kv: str, free: int) -> None:
    inp = inputs(goal, machine=machine_5090(gpu=gpu_5090(free_mb=free)), evidence=O.Evidence())
    vm = O.vram_model(inp, goal=goal, kv=kv, ubatch=512, parallel=2)
    assert vm is not None
    ctx = vm.max_ctx()
    assert ctx % O.CTX_STEP == 0
    if ctx > 0:
        assert vm.need(ctx) <= vm.avail_mb + 1e-6
        assert vm.need(ctx + O.CTX_STEP) > vm.avail_mb


def test_fit_tier_uses_measured_footprints_but_never_a_slope_below_the_header() -> None:
    inp = inputs()
    vm = O.vram_model(inp, goal="context", kv="q8_0", ubatch=512, parallel=2)
    assert vm is not None and vm.tier == "fit"
    # the 2026-10-07 sweep's slope (~24 KiB/token) is below the header's 34 KiB: the header wins
    assert vm.mib_per_token == pytest.approx(34816 / O.MIB)
    header = O.vram_model(dataclasses.replace(inp, evidence=O.Evidence()), goal="context", kv="q8_0", ubatch=512, parallel=2)
    assert header is not None and header.tier == "header" and header.unc == O.UNC["header"]
    assert vm.max_ctx() >= header.max_ctx()          # measured intercept beats seed overhead + 12 % margin


def test_fit_footprint_rejects_too_few_points_or_a_contradicting_slope() -> None:
    pts = footprints_sweep()
    assert O.fit_footprint(pts[:1], 34816) is None
    close = (O.Footprint(ctx=8192, app_mb=16000), O.Footprint(ctx=10240, app_mb=16100))
    assert O.fit_footprint(close, 34816) is None            # ctx range narrower than 1.5x
    wild = (O.Footprint(ctx=8192, app_mb=16000), O.Footprint(ctx=65536, app_mb=23000))   # 125 KiB/token: not this model
    assert O.fit_footprint(wild, 34816) is None
    fitted = O.fit_footprint(pts, 34816)
    assert fitted is not None and 15000 < fitted[0] < 16800 and fitted[1] > 0


def test_calibrated_tier_measures_the_overhead_from_the_running_server() -> None:
    # running server: weights 16,032 + KV(81,920) 2,720 + state 299 + ~1,000 of buffers = 20,051 MiB
    # of app footprint on top of a 2,063 MiB baseline
    gpu = gpu_5090(free_mb=TOTAL_MB - 2063 - 20051)
    inp = inputs(machine=machine_5090(gpu=gpu), running=True, evidence=O.Evidence())
    vm = O.vram_model(inp, goal="balanced", kv="q8_0", ubatch=512, parallel=2)
    assert vm is not None and vm.tier == "calibrated" and vm.unc == O.UNC["calibrated"]
    assert 300 <= vm.parts["overhead"] <= 3000
    unstable = dataclasses.replace(inp, machine=machine_5090(gpu=dataclasses.replace(gpu, stable=False)))
    vm2 = O.vram_model(unstable, goal="balanced", kv="q8_0", ubatch=512, parallel=2)
    assert vm2 is not None and vm2.tier == "header"


def test_reserve_is_floored_at_the_guard_plus_256_and_a_user_value_is_used_as_given() -> None:
    m = machine_5090()
    assert O._reserve_mb(O.Settings(), "context", m.guard_floor_mb) == 1756
    assert O._reserve_mb(O.Settings(), "latency", m.guard_floor_mb) == 1756 + 1280
    assert O._reserve_mb(O.Settings(reserve_vram_mb=4000), "context", m.guard_floor_mb) == 4000
    assert O._reserve_mb(O.Settings(reserve_vram_mb=100), "context", m.guard_floor_mb) == 1756


def test_more_free_vram_never_gives_a_smaller_window() -> None:
    prev = -1
    for free in range(4000, 24000, 2000):
        inp = inputs("context", machine=machine_5090(gpu=gpu_5090(free_mb=free, baseline_mb=None)), evidence=O.Evidence())
        ctx = O.recommend(inp, "context").vram.get("max_ctx", 0)
        assert ctx >= prev
        prev = ctx


# --------------------------------------------------------------------------------------------
# the plan: goals
# --------------------------------------------------------------------------------------------
def _item(plan: O.Plan, iid: str) -> O.Item:
    return next(i for i in plan.items if i.id == iid)


def test_goal_ordering_context_ge_balanced_ge_latency() -> None:
    inp = inputs()
    ctx = {g: _item(O.recommend(inp, g), "llm").profile.get("ctx_size", inp.launch.ctx) for g in ("latency", "balanced", "context")}
    assert ctx["context"] >= ctx["balanced"] >= ctx["latency"]
    assert ctx["context"] % O.CTX_STEP == 0


def test_owner_machine_plans_are_sensible() -> None:
    inp = inputs()
    plans = O.plan_all(inp)
    assert set(plans) == set(O.GOALS)
    ctx = plans["context"]
    llm = _item(ctx, "llm")
    assert llm.changed and llm.applies == "model-restart" and llm.tick
    assert llm.profile["ctx_size"] > 81920                   # the whole fitting window is more than today's profile
    assert ctx.vram["tier"] == "fit" and ctx.vram["need_mb"] <= ctx.vram["avail_mb"]
    bal = _item(plans["balanced"], "llm")
    assert bal.profile.get("kv_cache_type", "q8_0") == "q8_0"
    # the speech row: whisper on the CPU is replaced in every plan
    for g, p in plans.items():
        sp = _item(p, "speech")
        assert sp.changed and sp.applies == "next-launch" and sp.settings["asr.engine"] in ("sherpa", "faster-whisper"), g
        assert any("cannot keep up" in w for w in p.warnings), g


def test_latency_goal_takes_the_fastest_speech_engine_and_the_others_take_the_lowest_vram_one() -> None:
    inp = inputs()
    lat = _item(O.recommend(inp, "latency"), "speech")
    assert lat.proposed == "whisper-cuda-int8_float16"
    assert lat.settings == {"asr.engine": "faster-whisper", "asr.device": "cuda", "asr.compute_type": "int8_float16"}
    for g in ("balanced", "context", "quality"):
        sp = _item(O.recommend(inp, g), "speech")
        assert sp.proposed == "sherpa-cpu" and sp.settings == {"asr.engine": "sherpa", "asr.cpu_threads": 4}, g
        assert "Whisper on the GPU would take 0.11 s" in sp.why


def test_asr_gpu_never_keeps_whisper_off_the_gpu_even_for_latency() -> None:
    inp = inputs("latency", settings=O.Settings(goal="latency", asr_gpu="never"))
    assert _item(O.recommend(inp, "latency"), "speech").proposed == "sherpa-cpu"


def test_asr_filters_hotwords_language_bundle_and_cuda_availability() -> None:
    base = inputs()
    hot = dataclasses.replace(base, cfg=cfg_owner(hotwords=True))
    assert _item(O.recommend(hot, "balanced"), "speech").proposed.startswith("whisper-cuda")
    ja = dataclasses.replace(base, cfg=cfg_owner(asr_language="ja"))
    assert _item(O.recommend(ja, "balanced"), "speech").proposed.startswith("whisper-cuda")
    fr = dataclasses.replace(base, cfg=cfg_owner(asr_language="fr"))
    assert _item(O.recommend(fr, "balanced"), "speech").proposed == "sherpa-cpu"
    no_bundle = dataclasses.replace(base, machine=machine_5090(sherpa_ready=False))
    assert _item(O.recommend(no_bundle, "balanced"), "speech").proposed.startswith("whisper-cuda")
    nothing = dataclasses.replace(base, machine=machine_5090(sherpa_ready=False, ct2_cuda=False))
    sp = _item(O.recommend(nothing, "balanced"), "speech")
    assert not sp.changed and "no faster engine" in sp.why


def test_asr_keep_up_rule_follows_the_partial_interval() -> None:
    slow = O.Evidence(asr=(O.AsrRow(label="sherpa-cpu", engine="sherpa", device="cpu", compute="int8", threads=4,
                                   median_s=0.9, overlap=0.99, vram_mb=0, source="probe"),), footprints=footprints_sweep())
    inp = inputs(evidence=slow, settings=O.Settings(asr_gpu="never"), machine=machine_5090(ct2_cuda=False))
    assert _item(O.recommend(inp, "balanced"), "speech").changed is False          # 0.9 s > 0.6 s: not viable
    relaxed = dataclasses.replace(inp, cfg=cfg_owner(partials_enabled=False))
    assert _item(O.recommend(relaxed, "balanced"), "speech").proposed == "sherpa-cpu"  # 1.5 s threshold without partials


def test_asr_measured_probe_rows_outrank_sweep_and_priors() -> None:
    probe = O.AsrRow(label="sherpa-cpu", engine="sherpa", device="cpu", compute="int8", threads=4, median_s=0.16,
                     overlap=0.99, vram_mb=0, source="probe")
    inp = inputs(evidence=O.Evidence(asr=asr_sweep() + (probe,), footprints=footprints_sweep()))
    sp = _item(O.recommend(inp, "balanced"), "speech")
    assert sp.evidence_kind == "measured" and "0.16 s" in sp.why


def test_cuda_speech_memory_reduces_the_window() -> None:
    inp = inputs("latency")
    with_gpu_asr = O.vram_model(inp, goal="latency", kv="q8_0", ubatch=512, parallel=2, asr_mb=874)
    without = O.vram_model(inp, goal="latency", kv="q8_0", ubatch=512, parallel=2, asr_mb=0)
    assert with_gpu_asr is not None and without is not None
    lost = without.max_ctx() - with_gpu_asr.max_ctx()
    assert 20000 <= lost <= 30000                           # ~874 MiB at 34 KiB/token is about 26k tokens


def test_quality_uses_f16_only_when_32k_still_fits() -> None:
    inp = inputs("quality")
    q = O.recommend(inp, "quality")
    llm = _item(q, "llm")
    if llm.profile.get("kv_cache_type") == "f16":
        assert q.vram["max_ctx"] >= O.QUALITY_MIN_CTX
    small = inputs("quality", machine=machine_5090(gpu=gpu_5090(free_mb=19000, baseline_mb=None)), evidence=O.Evidence())
    q2 = O.recommend(small, "quality")
    assert _item(q2, "llm").profile.get("kv_cache_type", "q8_0") == "q8_0"
    assert any("16-bit KV" in w for w in q2.warnings)


def test_pinned_profile_leaves_a_smaller_window_unticked_but_a_larger_one_ticked() -> None:
    inp = inputs()
    lat = _item(O.recommend(inp, "latency"), "llm")
    assert lat.changed and lat.profile["ctx_size"] < 81920 and lat.tick is False
    ctx = _item(O.recommend(inp, "context"), "llm")
    assert ctx.profile["ctx_size"] > 81920 and ctx.tick is True
    auto = dataclasses.replace(inp, profile={"ctx_size": 81920, "auto": True})
    assert _item(O.recommend(auto, "latency"), "llm").tick is True


def test_latency_shrinks_the_window_only_when_a_longer_one_measurably_costs_time() -> None:
    key = qwen35_27b().key
    slow_long = (
        O.LlmRun(model=key, ctx=8192, kv="q8_0", parallel=2, pp=950, tg=24, cached_ttft_ms=500, spread=0.02, source="probe"),
        O.LlmRun(model=key, ctx=81920, kv="q8_0", parallel=2, pp=600, tg=16, cached_ttft_ms=1500, spread=0.02, source="probe"),
    )
    assert O.ctx_is_free(O.Evidence(llm=slow_long), key, "q8_0", 2) is False
    free = (slow_long[0], dataclasses.replace(slow_long[1], pp=940, tg=23.5, cached_ttft_ms=520))
    assert O.ctx_is_free(O.Evidence(llm=free), key, "q8_0", 2) is True
    assert O.ctx_is_free(O.Evidence(llm=slow_long[:1]), key, "q8_0", 2) is None
    inp = inputs("latency", evidence=O.Evidence(llm=slow_long, asr=asr_sweep(), footprints=footprints_sweep()))
    shrunk = _item(O.recommend(inp, "latency"), "llm").profile["ctx_size"]
    inp_free = inputs("latency", evidence=O.Evidence(llm=free, asr=asr_sweep(), footprints=footprints_sweep()))
    not_shrunk = _item(O.recommend(inp_free, "latency"), "llm").profile.get("ctx_size", 81920)
    assert shrunk < not_shrunk
    assert "measurably costs answer time" in _item(O.recommend(inp, "latency"), "llm").why


def test_model_that_does_not_fit_gives_advice_only() -> None:
    inp = inputs("context", machine=machine_5090(gpu=gpu_5090(free_mb=TOTAL_MB - 2063, total_mb=TOTAL_MB)),
                 model=dataclasses.replace(qwen35_27b(), size_bytes=23_000_000_000), evidence=O.Evidence())
    p = O.recommend(inp, "context")
    llm = _item(p, "llm")
    assert llm.proposed == "advice only" and not llm.enabled and p.profile is None
    assert any("does not fit" in w for w in p.warnings)


def test_no_gpu_or_foreign_backend_skips_the_llm_row_but_still_plans_speech() -> None:
    cpu_only = inputs(machine=machine_5090(gpu=None, ct2_cuda=False), evidence=O.Evidence())
    p = O.recommend(cpu_only, "balanced")
    assert _item(p, "llm").proposed == "advice only" and _item(p, "speech").proposed == "sherpa-cpu"
    ollama = inputs(backend="ollama")
    assert "ollama" in _item(O.recommend(ollama, "balanced"), "llm").now
    assert "GPU power" not in [i.label for i in O.recommend(ollama, "balanced").items]


def test_swap_resident_blocks_the_plan_but_still_shows_the_math() -> None:
    inp = inputs(machine=machine_5090(gpu=gpu_5090(swap_resident=True)))
    p = O.recommend(inp, "context")
    assert p.blocked and _item(p, "llm").enabled is False and p.vram["max_ctx"] > 0


def test_prompt_budgets_scale_with_the_goal_and_stay_in_the_field_ranges() -> None:
    inp = inputs()
    lat = O.recommend(inp, "latency")
    assert lat.settings["screen.max_text_chars"] == 1500 and lat.settings["handsfree.context_chars"] == 3500
    ctx = O.recommend(inp, "context")
    assert ctx.settings["screen.max_text_chars"] == 4000 and ctx.settings["handsfree.context_chars"] == 9500
    assert "screen.max_text_chars" not in O.recommend(inp, "balanced").settings
    done = dataclasses.replace(inp, cfg=cfg_owner(screen_max_text_chars=4000, handsfree_context_chars=9500))
    assert "screen.max_text_chars" not in O.recommend(done, "context").settings    # already there: no row


def test_ocr_choice_honours_openvino_devices_and_auto() -> None:
    inp = inputs()
    ocr = _item(O.recommend(inp, "balanced"), "ocr")
    assert not ocr.changed and "auto already picks the igpu" in ocr.why
    no_igpu = dataclasses.replace(inp, machine=machine_5090(igpu=""), cfg=cfg_owner(ocr_now="npu"))
    assert not _item(O.recommend(no_igpu, "balanced"), "ocr").changed
    forced_cpu = dataclasses.replace(inp, cfg=cfg_owner(ocr_device="cpu", ocr_now="cpu"))
    o = _item(O.recommend(forced_cpu, "balanced"), "ocr")
    assert o.settings == {"screen.ocr_device": "igpu"} and o.applies == "next-launch"
    unknown = dataclasses.replace(inp, machine=machine_5090(devices_known=False))
    assert "not checked" in _item(O.recommend(unknown, "balanced"), "ocr").why


def test_power_capped_gpu_gets_an_explanatory_row() -> None:
    inp = inputs(machine=machine_5090(gpu=gpu_5090(power_capped=True, power_w=115.0, power_limit_w=115.1, sm_mhz=700, sm_max_mhz=3090)))
    row = _item(O.recommend(inp, "balanced"), "gpu")
    assert "power-capped" in row.now and row.tick is False and "power limit" in row.why
    assert "gpu" not in [i.id for i in O.recommend(inputs(), "balanced").items]


# --------------------------------------------------------------------------------------------
# knobs: only measured wins become flags
# --------------------------------------------------------------------------------------------
def _runs(key: str, *, gain: float, flags: tuple[tuple[str, str], ...], parallel: int = 2, spread: float = 0.02,
          clean: bool = True, stale: bool = False) -> tuple[O.LlmRun, ...]:
    base = O.LlmRun(model=key, ctx=32768, kv="q8_0", parallel=2, pp=900.0, tg=20.0, cached_ttft_ms=600.0, spread=spread,
                    gpu_used_mb=16932, source="probe")
    t = O.typical_answer_s(base)
    assert t is not None
    # a candidate whose typical answer is `gain` faster: scale tg and pp together
    cand = dataclasses.replace(base, flags=flags, parallel=parallel, pp=base.pp / (1 - gain), tg=base.tg / (1 - gain),
                               cached_ttft_ms=base.cached_ttft_ms * (1 - gain), gpu_used_mb=17300, clean_log=clean, stale=stale)
    return base, cand


def test_knob_adopted_only_above_the_noise_floor() -> None:
    key = qwen35_27b().key
    good = _runs(key, gain=0.14, flags=(("--ubatch-size", "2048"),))
    inp = inputs("latency", evidence=O.Evidence(llm=good))
    k = O.choose_knobs(inp, "latency")
    assert k.flags == {"--ubatch-size": "2048"} and k.gains["ubatch"] > 0.1 and k.extra_mb == pytest.approx(368)
    small = _runs(key, gain=0.03, flags=(("--ubatch-size", "2048"),))
    assert O.choose_knobs(inputs(evidence=O.Evidence(llm=small)), "latency").flags == {}
    noisy = _runs(key, gain=0.14, flags=(("--ubatch-size", "2048"),), spread=0.10)
    assert O.choose_knobs(inputs(evidence=O.Evidence(llm=noisy)), "latency").flags == {}
    dirty = _runs(key, gain=0.30, flags=(("--ubatch-size", "2048"),), clean=False)
    assert O.choose_knobs(inputs(evidence=O.Evidence(llm=dirty)), "latency").flags == {}
    old = _runs(key, gain=0.30, flags=(("--ubatch-size", "2048"),), stale=True)
    assert O.choose_knobs(inputs(evidence=O.Evidence(llm=old)), "latency").flags == {}
    # the context goal never adopts a batching knob
    assert O.choose_knobs(inputs(evidence=O.Evidence(llm=good)), "context").flags == {}


def test_spec_knob_needs_the_mtp_head_and_fifteen_percent() -> None:
    key = qwen35_27b().key
    mtp = _runs(key, gain=0.20, flags=(("--spec-type", "draft-mtp"),))
    k = O.choose_knobs(inputs(evidence=O.Evidence(llm=mtp)), "balanced")
    assert k.flags == {"--spec-type": "draft-mtp"}
    weak = _runs(key, gain=0.12, flags=(("--spec-type", "draft-mtp"),))
    assert O.choose_knobs(inputs(evidence=O.Evidence(llm=weak)), "balanced").flags == {}
    no_head = inputs(model=dataclasses.replace(qwen35_27b(), nextn_layers=0), evidence=O.Evidence(llm=mtp))
    assert O.choose_knobs(no_head, "balanced").flags == {}
    # ngram speculation needs no head; a draft-model type (needs a file) is never adopted
    ngram = _runs(key, gain=0.20, flags=(("--spec-type", "ngram-mod"),))
    assert O.choose_knobs(inputs(evidence=O.Evidence(llm=ngram)), "balanced").flags == {"--spec-type": "ngram-mod"}
    draft_file = _runs(key, gain=0.40, flags=(("--spec-type", "draft-simple"),))
    assert O.choose_knobs(inputs(evidence=O.Evidence(llm=draft_file)), "balanced").flags == {}


def test_single_slot_speculation_is_latency_only_and_changes_parallel() -> None:
    key = qwen35_27b().key
    single = _runs(key, gain=0.25, flags=(("--spec-draft-n-max", "3"), ("--spec-type", "draft-mtp")), parallel=1)
    k = O.choose_knobs(inputs(evidence=O.Evidence(llm=single)), "latency")
    assert k.parallel == 1 and k.flags["--spec-type"] == "draft-mtp" and any("single slot" in n for n in k.notes)
    assert O.choose_knobs(inputs(evidence=O.Evidence(llm=single)), "balanced").flags == {}
    plan = O.recommend(inputs("latency", evidence=O.Evidence(llm=single, asr=asr_sweep(), footprints=footprints_sweep())), "latency")
    llm = _item(plan, "llm")
    assert llm.profile["parallel"] == 1 and "--spec-type" in llm.profile["extra_args"]


def test_merge_extra_args_replaces_only_managed_flags() -> None:
    existing = ["--n-cpu-moe", "4", "--ubatch-size", "1024", "--cache-reuse", "512", "--spec-type", "ngram-mod"]
    out = O.merge_extra_args(existing, {"--ubatch-size": "2048", "--spec-type": None})
    assert out == ["--n-cpu-moe", "4", "--cache-reuse", "512", "--ubatch-size", "2048"]
    assert O.merge_extra_args([], {"--load-mode": "mmap+mlock"}) == ["--load-mode", "mmap+mlock"]


def test_parse_launch_args_last_wins_and_booleans() -> None:
    parsed = O.parse_launch_args(["--ctx-size", "8192", "--kv-unified", "--ctx-size", "32768", "--flash-attn", "on"])
    assert parsed == {"--ctx-size": "32768", "--kv-unified": "1", "--flash-attn": "on"}
    assert O.managed_flags_of(["--spec-type", "draft-mtp", "--ctx-size", "4"]) == (("--spec-type", "draft-mtp"),)


# --------------------------------------------------------------------------------------------
# effective_launch mirrors build_args (the drift guard)
# --------------------------------------------------------------------------------------------
@pytest.mark.parametrize("profile", [
    None,
    {"ctx_size": 81920},
    {"ctx_size": 4096, "kv_cache_type": "f16", "parallel": 1, "flash_attn": False, "threads": 6},
    {"extra_args": ["--ctx-size", "32768", "--ubatch-size", "2048", "--parallel", "1"]},
])
def test_effective_launch_matches_build_args(profile: dict | None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from exocortex.llm import library, server as srv

    cfg = LLMConfig(ctx_size=8192, parallel=2, kv_cache_type="q8_0", flash_attn=True, llama_extra_args=["--cache-reuse", "256"])
    mgr = srv.LlamaServerManager(cfg)
    prof = library.ModelProfile(**profile) if profile else None
    monkeypatch.setattr(mgr, "_profile_for", lambda model: prof)

    class _M:
        key = "x/y"
        path = tmp_path / "y.gguf"

    args = mgr.build_args(_M(), 1234)
    parsed = O.parse_launch_args(args)
    eff = O.effective_launch(cfg, dataclasses.asdict(prof) if prof else None)
    assert eff.ctx == int(parsed["--ctx-size"])
    assert eff.parallel == int(parsed["--parallel"])
    assert eff.ubatch == int(parsed["--ubatch-size"])
    assert eff.kv == parsed.get("--cache-type-k", "f16")
    assert eff.flash == (parsed["--flash-attn"] == "on")


# --------------------------------------------------------------------------------------------
# hygiene: plan data, digest, validation, gates, describe, purity
# --------------------------------------------------------------------------------------------
def test_plan_round_trips_through_json_and_the_digest_is_stable_and_sensitive() -> None:
    inp = inputs()
    p1 = O.recommend(inp, "context")
    p2 = O.recommend(inp, "context")
    assert p1.digest == p2.digest and len(p1.digest) == 12
    rt = O.Plan.from_dict(json.loads(json.dumps(p1.to_dict())))
    assert rt == p1
    other = O.recommend(dataclasses.replace(inp, settings=O.Settings(goal="context", ctx_cap=16384)), "context")
    assert other.digest != p1.digest and _item(other, "llm").profile["ctx_size"] <= 16384
    assert set(p1.settings) == {k for i in p1.items if i.tick and i.enabled for k in i.settings}
    assert all(i.settings == {} or i.applies in ("live", "next-launch") for i in p1.items)


def test_plan_is_a_fixed_point_after_apply() -> None:
    inp = inputs()
    p = O.recommend(inp, "context")
    llm = _item(p, "llm")
    applied_profile = {**(inp.profile or {}), **llm.profile}
    cfg2 = cfg_owner(asr_now="sherpa-cpu", asr_engine="sherpa", screen_max_text_chars=p.settings["screen.max_text_chars"],
                     handsfree_context_chars=p.settings["handsfree.context_chars"])
    launch2 = dataclasses.replace(inp.launch, ctx=llm.profile["ctx_size"])
    again = O.recommend(dataclasses.replace(inp, profile=applied_profile, launch=launch2, cfg=cfg2), "context")
    assert not any(i.changed for i in again.items), [i.id for i in again.items if i.changed]
    assert again.summary[0].startswith("nothing to change")


def test_validate_plan_refuses_stale_moved_or_foreign_states() -> None:
    inp = inputs()
    p = O.recommend(inp, "context")
    ok = O.FreshFacts(free_mb=22400, loaded_key=p.model_key, swap_resident=False, now_epoch=inp.now_epoch + 10)
    assert O.validate_plan(p, ok) == ()
    assert "15 minutes" in " ".join(O.validate_plan(p, dataclasses.replace(ok, now_epoch=inp.now_epoch + 1000)))
    assert "different model" in " ".join(O.validate_plan(p, dataclasses.replace(ok, loaded_key="other/model")))
    assert "shared model server" in " ".join(O.validate_plan(p, dataclasses.replace(ok, swap_resident=True)))
    assert "GPU memory moved" in " ".join(O.validate_plan(p, dataclasses.replace(ok, free_mb=4000)))


def test_gate_matrix() -> None:
    idle = O.Activity(llm_ready=True, server_running=True, transcriber_ready=True)
    assert O.gate("llm", idle) == (True, "")
    assert O.gate("speech", idle) == (True, "")
    assert O.gate("apply", idle) == (True, "")
    assert O.gate("llm", dataclasses.replace(idle, backend="ollama"))[0] is False
    assert O.gate("llm", dataclasses.replace(idle, llm_ready=False))[0] is False
    assert O.gate("llm", dataclasses.replace(idle, assist_busy=True))[1] == "an answer is being generated"
    assert O.gate("llm", dataclasses.replace(idle, swap_busy=True))[0] is False
    assert O.gate("apply", dataclasses.replace(idle, swap_resident=True))[0] is False
    talking = dataclasses.replace(idle, speech_idle_s=1.0)
    assert O.gate("llm", talking)[0] is False and O.gate("llm", talking, force=True)[0] is True
    assert O.gate("apply", talking, force=True)[0] is False
    assert O.gate("speech", dataclasses.replace(idle, transcriber_ready=False))[0] is False
    assert O.gate("bogus", idle)[0] is False


def test_describe_lists_every_item_and_the_window_numbers() -> None:
    p = O.recommend(inputs(), "context")
    text = O.describe(p)
    assert text.startswith("Optimize for context -- lmstudio-community/Qwen3.8-27B")
    for it in p.items:
        assert it.label in text
    assert "cold re-read" in text and "GPU memory (fit)" in text
    assert p.predicted["hours_held"] > 1.0 and p.predicted["cold_reread_s"] > 0


def test_working_set_floor_and_window_numbers() -> None:
    cfg = cfg_owner()
    floor = O.working_set_floor(cfg, 2)
    assert floor >= O.CTX_FLOOR and floor % O.CTX_STEP == 0
    assert O.working_set_floor(cfg, 1) <= floor
    w = O.window_numbers(32768, cfg, 2, None)
    assert w["answer_tokens"] == int(32768 * 0.7) - 256 and w["pp_basis"] == "typical"
    assert O.window_numbers(32768, cfg, 1, 900.0)["pp_basis"] == "measured"


def test_parse_server_log_strips_ansi_and_returns_none_on_garbage() -> None:
    text = "\x1b[34m0.1\x1b[0m llama_kv_cache: size = 16384.00 MiB (262144 cells, 16 layers, 4/1 seqs)\n" \
           "llama_memory_recurrent: size = 598.50 MiB (4 seqs)\n n_ctx = 262144"
    assert O.parse_server_log(text) == {"kv_mb": 16384.0, "cells": 262144, "kv_layers": 16, "recurrent_mb": 598.5, "n_ctx": 262144}
    assert O.parse_server_log("slot print_timing: id 1") is None and O.parse_server_log("") is None


def test_planner_is_pure(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*a, **k):  # noqa: ANN001, ANN002, ANN003
        raise AssertionError("the planner must not do I/O")

    monkeypatch.setattr(socket, "socket", _boom)
    monkeypatch.setattr(subprocess, "Popen", _boom)
    plans = O.plan_all(inputs())
    assert len(plans) == 4


def test_facts_adapters_tolerate_missing_attributes() -> None:
    class _Meta:
        architecture = "qwen35"
        n_layer = 65
        n_ctx_train = 262144
        n_head_kv = 4
        head_dim = 256
        kv_heads_total = 260
        full_attention_interval = 4
        nextn_predict_layers = 1

    mf = O.model_facts_from_meta("k", 10, _Meta())
    assert mf.is_hybrid and mf.has_mtp and mf.ssm_inner is None
    assert O.model_facts_from_meta("k", 0, None) == O.ModelFacts(key="k")
    from exocortex.config import Config

    cfg = Config()
    cf = O.cfg_facts_from_config(cfg, asr_now="x", ocr_now="igpu")
    assert cf.max_answer_tokens == 700 and cf.asr_now == "x" and cf.ocr_now == "igpu"
    cfg.optimize.goal = "context"
    assert O.settings_from_config(cfg).goal == "context"
    assert O.settings_from_config(object()) == O.Settings()

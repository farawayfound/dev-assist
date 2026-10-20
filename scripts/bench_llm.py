"""Benchmark the configured LLM backend: time-to-first-token and tokens/s for the live-assist prompt.

    .venv\\Scripts\\python.exe scripts\\bench_llm.py [--config config.toml] [--runs 5] [--model KEY]

With backend = "llamacpp" this spawns llama-server exactly like the app does (and stops it after).

Plain invocation (no new flags) is byte-for-byte the original script: a short live-assist prompt,
``--runs`` repeats, ttft/tok-per-s summary from ``exocortex.llm.client.LMStudioClient``.

``--protocol full`` (added 2026-09-26, AMD/iGPU routing bench) switches to a heavier per-config
measurement used to pick per-vendor llama-server defaults:

    .venv\\Scripts\\python.exe scripts\\bench_llm.py --protocol full --backend vulkan \\
        --model lmstudio-community/Qwen3.8-27B --prompt-tokens 2000 --rounds 3 --json out\\vulkan.json

One config per process (start the server once, measure, stop it, exit) -- run the script again with
different flags for the next config; that is also what keeps "one job at a time, stop() before the
next" true when a shell loop drives several configs in a row. Per round it measures:

    (a) a FRESH ~``--prompt-tokens``-token ChatML transcript prompt (varied wording each round, so
        it is genuinely uncached) -> prompt_per_second (pp) and predicted_per_second (tg), read
        directly off the final streamed SSE chunk's ``"timings"`` (llama-server puts prompt_n,
        prompt_ms, prompt_per_second, predicted_n, predicted_ms, predicted_per_second there --
        ``LMStudioClient`` does not expose this, so this script streams httpx itself, see
        ``stream_completion_timings``);
    (b) the EXACT SAME prompt again -> wall-clock time to the first content delta, with the prefix
        now cached;
    (c) the same prompt with ``--extra-prompt-tokens`` more transcript appended before the question
        -> pp of just the incremental part (``--cache-reuse 256``, on by default in
        ``exocortex.llm.server.build_args``, should make ``timings.prompt_n`` for this call close to
        ``--extra-prompt-tokens``, not the full prompt).

``--rounds`` repeats (a)-(c) with fresh transcript wording each time; medians are printed. Load time
(``start()`` until ``/health`` 200 -- ``LlamaServerManager.start()`` already blocks on exactly that)
and the GPU-memory delta (``exocortex.gpu.vram_free_mb()`` before ``start()`` vs. right after) are
measured once per process. ``--json PATH`` dumps every raw number (per round, per request) plus the
llama-server.log lines this script found interesting (n_ubatch, flash_attn, cache types, warnings).
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from exocortex.config import load_config  # noqa: E402
from exocortex.llm.client import LMStudioClient  # noqa: E402
from exocortex.llm.prompts import (  # noqa: E402
    NO_THINK_PREFIX,
    STOP,
    approx_tokens,
    render_chatml,
    system_assist,
    user_assist,
)

TRANSCRIPT = """[00:12] Speaker 1: Thanks for joining. Today we need to decide on the message queue for the new ingestion service.
[00:31] Me: Sure, I've looked at both options over the week.
[00:40] Speaker 1: So what would you say is the main trade-off between Kafka and RabbitMQ for our use case?"""
QUESTION = "What would you say is the main trade-off between Kafka and RabbitMQ for our use case?"

# =================================================================================================
# --protocol full: a realistic, variable-length meeting transcript (fresh wording per round so a
# "fresh prompt" round is never served from llama-server's KV/cache-reuse by accident).
# =================================================================================================
_FILLER_TOPICS = [
    "the retry backoff strategy for the ingestion workers",
    "whether we need exactly-once delivery or at-least-once is good enough here",
    "how partitioning should map onto the downstream consumers",
    "the schema registry and how we handle breaking changes",
    "back-pressure once the warehouse copy job falls behind",
    "the on-call runbook if a partition rebalance stalls",
    "cost of running a 3-broker cluster versus a managed offering",
    "how we replay a bad batch without double-processing everything downstream",
    "the dead-letter queue policy and who triages it",
    "whether the ordering guarantee actually matters for this event type",
    "compaction settings for the changelog topic",
    "how client library upgrades get rolled out without downtime",
    "the monitoring dashboards we still need before this ships",
    "load testing results from last week's staging run",
    "how this affects the SLA we already promised the analytics team",
]
_SPEAKERS = ("Speaker 1", "Me", "Speaker 2")


def build_transcript(target_tokens: int, round_seed: int) -> str:
    """A synthetic meeting transcript of roughly ``target_tokens`` tokens (``approx_tokens``,
    ~4 chars/token -- the actual token count is verified from the server's own ``timings.prompt_n``
    once rendered into ChatML). Content is a deterministic function of ``round_seed`` so consecutive
    "fresh prompt" rounds never repeat the same text (and so never hit llama-server's own
    ``--cache-reuse``), while still being reproducible."""
    lines: list[str] = []
    idx = 0
    minute = 0
    budget = max(0, target_tokens - approx_tokens(QUESTION) - 40)
    while approx_tokens("\n".join(lines)) < budget:
        topic = _FILLER_TOPICS[(idx + round_seed) % len(_FILLER_TOPICS)]
        speaker = _SPEAKERS[idx % len(_SPEAKERS)]
        minute = idx * 7 // 60
        second = (idx * 7) % 60
        lines.append(
            f"[{minute:02d}:{second:02d}] {speaker}: (round {round_seed}, item {idx}) let's talk about "
            f"{topic} -- I think it changes the latency and cost trade-offs for the new ingestion "
            "service, and we should decide before the next sprint."
        )
        idx += 1
    lines.append(f"[{minute:02d}:{second + 3:02d}] Speaker 1: {QUESTION}")
    return "\n".join(lines)


def extra_transcript_lines(extra_tokens: int, round_seed: int, start_idx: int) -> str:
    """~``extra_tokens`` more transcript lines, continuing from ``start_idx`` (so they never repeat
    lines ``build_transcript`` already produced for the same round) -- appended BEFORE the question
    to exercise ``--cache-reuse`` (config (c) in the bench protocol)."""
    lines: list[str] = []
    idx = start_idx
    while approx_tokens("\n".join(lines)) < extra_tokens:
        topic = _FILLER_TOPICS[(idx + round_seed) % len(_FILLER_TOPICS)]
        speaker = _SPEAKERS[idx % len(_SPEAKERS)]
        minute = idx * 7 // 60
        second = (idx * 7) % 60
        lines.append(
            f"[{minute:02d}:{second:02d}] {speaker}: (round {round_seed}, item {idx}, follow-up) one "
            f"more thing on {topic} before we move on."
        )
        idx += 1
    return "\n".join(lines)


def _insert_before_question(transcript: str, extra: str) -> str:
    """Splice ``extra`` right before the final (question) line of ``transcript``."""
    lines = transcript.split("\n")
    return "\n".join(lines[:-1] + [extra, lines[-1]])


# ---- direct httpx streaming: llama-server's own "timings" object -------------------------------
def stream_completion_timings(
    base_url: str, prompt: str, *, max_tokens: int, temperature: float = 0.0,
    stop: list[str] | None = None, timeout_s: float = 180.0,
) -> tuple[str, float, dict[str, Any]]:
    """POST ``/v1/completions`` (stream=True) directly with httpx and return
    ``(text, wall_s_to_first_content_delta, timings)``. ``timings`` is llama-server's own object off
    the final SSE chunk (``{prompt_n, prompt_ms, prompt_per_second, predicted_n, predicted_ms,
    predicted_per_second}``), ``{}`` if the server never sent one (older build / non-llama.cpp
    server). ``LMStudioClient.stream_completion`` does not expose this -- see the module docstring.
    """
    body = {
        "model": "", "prompt": prompt, "max_tokens": max_tokens, "temperature": temperature,
        "stop": stop or STOP, "stream": True,
    }
    text_parts: list[str] = []
    timings: dict[str, Any] = {}
    ttft: float | None = None
    t0 = time.perf_counter()
    with httpx.Client(timeout=httpx.Timeout(timeout_s, connect=5.0)) as c:
        with c.stream("POST", base_url.rstrip("/") + "/v1/completions", json=body) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if isinstance(line, bytes):
                    line = line.decode("utf-8", errors="ignore")
                line = line.strip("\r")
                if not line or not line.startswith("data:"):
                    continue
                payload = line[len("data:"):].strip()
                if payload == "[DONE]" or not payload:
                    continue
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                choices = chunk.get("choices") or []
                text = choices[0].get("text", "") if choices else ""
                if text:
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    text_parts.append(text)
                if isinstance(chunk.get("timings"), dict):
                    timings = chunk["timings"]
    return "".join(text_parts), (ttft if ttft is not None else 0.0), timings


# ---- llama-server.log signals (device / n_ubatch / flash_attn / cache types / warnings) --------
_LOG_PATTERNS: dict[str, re.Pattern[str]] = {
    "device": re.compile(r"(?:load_tensors|ggml_vulkan|ggml_cuda|ROCm|Vulkan0|CUDA0).*(?:device|Device|assigned to).*", re.IGNORECASE),
    "n_ubatch": re.compile(r"\bn_ubatch\s*[=:]\s*\d+"),
    "n_batch": re.compile(r"\bn_batch\s*[=:]\s*\d+"),
    "flash_attn": re.compile(r"\bflash_attn\s*[=:]\s*\S+", re.IGNORECASE),
    "cache_type_k": re.compile(r"\bcache_type_k\s*[=:]\s*\S+", re.IGNORECASE),
    "cache_type_v": re.compile(r"\bcache_type_v\s*[=:]\s*\S+", re.IGNORECASE),
    "n_gpu_layers": re.compile(r"\bn_gpu_layers\s*[=:]\s*[-\d]+"),
}
_WARNING_KEYWORDS = (
    "not supported", "unsupported", "fallback", "falling back", "cannot", "failed",
    "warn", "no cuda", "no vulkan", "cpu backend", "mismatch",
)


def extract_log_signals(log_path: Path) -> dict[str, Any]:
    """Lines of interest from llama-server's own ``--log-file``: the last match of each pattern in
    ``_LOG_PATTERNS`` (llama.cpp prints these once per param, sometimes more than once) plus every
    line containing a warning keyword (deduplicated, capped) -- used to (a) verify a repeated
    ``llama_extra_args`` flag really overrides the earlier value and (b) flag anything alarming
    (e.g. a quant's tensors falling back to the CPU)."""
    out: dict[str, Any] = {"matches": {}, "warnings": []}
    if not log_path or not log_path.is_file():
        out["error"] = f"log file not found: {log_path}"
        return out
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        out["error"] = str(exc)
        return out
    lines = text.splitlines()
    for name, pat in _LOG_PATTERNS.items():
        found = [ln.strip() for ln in lines if pat.search(ln)]
        if found:
            out["matches"][name] = found[-1]
    seen: set[str] = set()
    warnings: list[str] = []
    for ln in lines:
        low = ln.lower()
        if any(k in low for k in _WARNING_KEYWORDS):
            key = ln.strip()
            if key and key not in seen:
                seen.add(key)
                warnings.append(key)
    out["warnings"] = warnings[:40]
    return out


def _median(vals: list[float]) -> float | None:
    return statistics.median(vals) if vals else None


# =================================================================================================
def run_simple(cfg, args) -> int:
    """The ORIGINAL script body, unchanged: a short live-assist prompt, ``--runs`` repeats,
    ttft/tok-per-s summary via ``LMStudioClient``. Kept byte-identical in behaviour so a plain
    invocation (no ``--protocol full``) is exactly what it always was."""
    mgr = None
    client = LMStudioClient(cfg.llm)
    if cfg.llm.backend.lower() == "llamacpp":
        from exocortex.llm.server import LlamaServerManager

        mgr = LlamaServerManager(cfg.llm)
        t0 = time.perf_counter()
        url = mgr.start(on_status=print)
        print(f"llama-server up at {url} in {time.perf_counter() - t0:.1f} s ({mgr.model.key if mgr.model else '?'})")
        client.set_base_url(url)
    else:
        print(f"backend openai at {cfg.llm.base_url}, model {cfg.llm.model}")

    prompt = render_chatml(system_assist(cfg.assist.persona, cfg.assist.style), user_assist(TRANSCRIPT, QUESTION), NO_THINK_PREFIX)
    print(f"prompt ~{len(prompt) // 4} tokens; {args.runs} runs, max_tokens {args.max_tokens}\n")
    ttfts, tps = [], []
    text = ""
    try:
        for i in range(args.runs):
            text = "".join(client.stream_completion(prompt, max_tokens=args.max_tokens, stop=STOP))
            st = client.last_stats
            if st:
                ttfts.append(st.ttft_s)
                tps.append(st.tok_per_s)
                print(f"run {i + 1}: ttft {st.ttft_s * 1000:6.0f} ms   {st.tok_per_s:5.1f} tok/s   {st.tokens} tokens")
        print("\n" + text.strip() + "\n")
        if ttfts:
            print(f"ttft: first {ttfts[0] * 1000:.0f} ms (uncached), median of rest {statistics.median(ttfts[1:] or ttfts) * 1000:.0f} ms")
            print(f"throughput: median {statistics.median(tps):.1f} tok/s")
    finally:
        client.close()
        if mgr is not None:
            mgr.stop()
    return 0


def run_full(cfg, args) -> int:
    """The heavier per-config protocol -- see the module docstring. Spawns exactly one llama-server
    for this process, measures load time / GPU memory / three request shapes over ``--rounds``
    rounds of fresh transcript text, prints medians, dumps everything to ``--json`` if given, and
    always stops the server before returning (even on an abort)."""
    if cfg.llm.backend.lower() != "llamacpp":
        print(f"--protocol full requires [llm].backend = llamacpp (got {cfg.llm.backend!r}); "
              "pass --backend/--server-path or set it in the config.")
        return 2

    from exocortex import gpu
    from exocortex.llm.server import LlamaServerError, LlamaServerManager

    cfg.llm.llama_startup_timeout_s = max(cfg.llm.llama_startup_timeout_s, args.abort_load_s)
    mgr = LlamaServerManager(cfg.llm, log_dir=Path(args.log_dir) if args.log_dir else (ROOT / "scripts" / "_bench_logs"))

    result: dict[str, Any] = {
        "requested": {
            "model": cfg.llm.model, "backend": cfg.llm.llama_backend, "server_path": cfg.llm.llama_server_path,
            "flash_attn": cfg.llm.flash_attn, "kv_cache_type": cfg.llm.kv_cache_type, "load_mode": cfg.llm.load_mode,
            "llama_extra_args": list(cfg.llm.llama_extra_args), "ctx_size": cfg.llm.ctx_size, "parallel": cfg.llm.parallel,
            "prompt_tokens": args.prompt_tokens, "extra_prompt_tokens": args.extra_prompt_tokens,
            "rounds": args.rounds, "max_tokens": args.max_tokens,
        },
    }

    free_before = gpu.vram_free_mb()
    t0 = time.perf_counter()
    try:
        url = mgr.start(on_status=print)
    except LlamaServerError as exc:
        load_s = time.perf_counter() - t0
        print(f"ABORT: llama-server failed to start after {load_s:.1f}s: {exc}")
        result.update(load_s=load_s, aborted=True, error=str(exc))
        if args.json:
            Path(args.json).write_text(json.dumps(result, indent=2), encoding="utf-8")
        return 1
    load_s = time.perf_counter() - t0
    free_after = gpu.vram_free_mb()
    gpu_used_mb = (free_before - free_after) if (free_before is not None and free_after is not None) else None
    binary_info = {
        "flavor": mgr.binary.flavor, "version": mgr.binary.version, "exe": str(mgr.binary.exe), "reason": mgr.binary.reason,
    }
    model_info = {"key": mgr.model.key, "size_gb": round(mgr.model.size_bytes / 1e9, 2)}
    print(f"load: {load_s:.1f}s   binary={binary_info['flavor']} {binary_info['version']}   "
          f"model={model_info['key']} ({model_info['size_gb']} GB)   "
          f"gpu_free_before={free_before}  gpu_free_after={free_after}  used={gpu_used_mb} MiB")
    log_signals = extract_log_signals(mgr.log_path) if mgr.log_path else {"error": "no log_path"}
    for name, line in log_signals.get("matches", {}).items():
        print(f"  log[{name}]: {line}")
    if log_signals.get("warnings"):
        print(f"  log: {len(log_signals['warnings'])} warning-like line(s) (see --json for full list)")
        for w in log_signals["warnings"][:5]:
            print(f"    ! {w}")

    result.update(binary=binary_info, model=model_info, load_s=load_s, aborted=False,
                   gpu_free_before_mb=free_before, gpu_free_after_mb=free_after, gpu_used_mb=gpu_used_mb,
                   log_signals=log_signals, rounds=[])

    try:
        pp_a, tg_a, ttft_b_ms, incr_pp, incr_n = [], [], [], [], []
        for r in range(args.rounds):
            transcript = build_transcript(args.prompt_tokens, round_seed=r)
            prompt_a = render_chatml(system_assist(cfg.assist.persona, cfg.assist.style),
                                     user_assist(transcript, QUESTION), NO_THINK_PREFIX)
            text_a, ttft_a_s, timings_a = stream_completion_timings(url, prompt_a, max_tokens=args.max_tokens)

            wall0 = time.perf_counter()
            text_b, ttft_b_s, timings_b = stream_completion_timings(url, prompt_a, max_tokens=args.max_tokens)
            cached_ttft_ms = ttft_b_s * 1000

            extra = extra_transcript_lines(args.extra_prompt_tokens, round_seed=r, start_idx=10_000 + r * 1000)
            transcript_c = _insert_before_question(transcript, extra)
            prompt_c = render_chatml(system_assist(cfg.assist.persona, cfg.assist.style),
                                     user_assist(transcript_c, QUESTION), NO_THINK_PREFIX)
            text_c, ttft_c_s, timings_c = stream_completion_timings(url, prompt_c, max_tokens=args.max_tokens)

            round_result = {
                "a_fresh": {"text": text_a[:600], "text_len": len(text_a), "ttft_s": ttft_a_s, "timings": timings_a},
                "b_cached_repeat": {"text": text_b[:600], "text_len": len(text_b), "wall_ttft_s": cached_ttft_ms / 1000, "timings": timings_b},
                "c_incremental": {"text": text_c[:600], "text_len": len(text_c), "ttft_s": ttft_c_s, "timings": timings_c},
            }
            result["rounds"].append(round_result)

            pp = timings_a.get("prompt_per_second")
            tg = timings_a.get("predicted_per_second")
            c_pp = timings_c.get("prompt_per_second")
            c_n = timings_c.get("prompt_n")
            if pp is not None:
                pp_a.append(pp)
            if tg is not None:
                tg_a.append(tg)
            ttft_b_ms.append(cached_ttft_ms)
            if c_pp is not None:
                incr_pp.append(c_pp)
            if c_n is not None:
                incr_n.append(c_n)
            print(f"round {r + 1}: prompt_n={timings_a.get('prompt_n')} pp={pp} tok/s   tg={tg} tok/s   "
                  f"cached_ttft={cached_ttft_ms:.0f} ms   incr_prompt_n={c_n} incr_pp={c_pp} tok/s")
    finally:
        mgr.stop()

    print("\n--- medians over {} round(s) ---".format(args.rounds))
    print(f"  pp (fresh prompt):        {_median(pp_a)}")
    print(f"  tg (generation):          {_median(tg_a)}")
    print(f"  cached ttft (ms):         {_median(ttft_b_ms)}")
    print(f"  incremental pp:           {_median(incr_pp)}")
    print(f"  incremental prompt_n:     {_median(incr_n)}")
    result["medians"] = {
        "pp": _median(pp_a), "tg": _median(tg_a), "cached_ttft_ms": _median(ttft_b_ms),
        "incremental_pp": _median(incr_pp), "incremental_prompt_n": _median(incr_n),
        "spread_pp": (max(pp_a) - min(pp_a)) if len(pp_a) > 1 else 0.0,
        "spread_tg": (max(tg_a) - min(tg_a)) if len(tg_a) > 1 else 0.0,
    }
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--runs", type=int, default=5, help="[simple protocol] number of repeats")
    ap.add_argument("--model")
    ap.add_argument("--max-tokens", type=int, default=160)
    # ---- added for the AMD/iGPU routing bench (2026-09-26) ----
    ap.add_argument("--protocol", choices=["simple", "full"], default="simple",
                     help="simple (default, original script) or full (load time / GPU mem / pp,tg,cached-ttft,incremental-pp)")
    ap.add_argument("--backend", choices=["auto", "vulkan", "rocm", "cpu"], help="overrides [llm].llama_backend")
    ap.add_argument("--server-path", help="overrides [llm].llama_server_path (an explicit llama.cpp release folder)")
    ap.add_argument("--extra-args", action="append", default=[],
                     help="extra llama-server args, e.g. --extra-args \"--ubatch-size 256\" (repeatable; shlex-split)")
    ap.add_argument("--flash-attn", choices=["on", "off"], help="overrides [llm].flash_attn")
    ap.add_argument("--kv", choices=["q8_0", "f16"], help="overrides [llm].kv_cache_type")
    ap.add_argument("--load-mode", help="overrides [llm].load_mode, e.g. mmap+mlock")
    ap.add_argument("--prompt-tokens", type=int, default=2000, help="[full protocol] target size of the fresh transcript prompt")
    ap.add_argument("--extra-prompt-tokens", type=int, default=300, help="[full protocol] tokens appended for request (c)")
    ap.add_argument("--rounds", type=int, default=3, help="[full protocol] repeats of (a)/(b)/(c), fresh wording each round")
    ap.add_argument("--abort-load-s", type=float, default=240.0, help="[full protocol] give up waiting for /health after this long")
    ap.add_argument("--log-dir", help="[full protocol] where llama-server.log is written (default scripts/_bench_logs)")
    ap.add_argument("--json", help="[full protocol] dump raw results to this file")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.model:
        cfg.llm.model = args.model
    if args.backend:
        cfg.llm.llama_backend = args.backend
    if args.server_path:
        cfg.llm.llama_server_path = args.server_path
    if args.flash_attn:
        cfg.llm.flash_attn = args.flash_attn == "on"
    if args.kv:
        cfg.llm.kv_cache_type = args.kv
    if args.load_mode:
        cfg.llm.load_mode = args.load_mode
    if args.extra_args:
        import shlex

        extra: list[str] = []
        for chunk in args.extra_args:
            extra.extend(shlex.split(chunk))
        cfg.llm.llama_extra_args = list(cfg.llm.llama_extra_args) + extra

    if args.protocol == "full":
        cfg.llm.backend = "llamacpp"  # full protocol always spawns our own llama-server
        return run_full(cfg, args)
    return run_simple(cfg, args)


if __name__ == "__main__":
    sys.exit(main())

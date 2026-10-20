"""Benchmark faster-whisper large-v3-turbo and sherpa-onnx Parakeet-TDT on the CPU.

    .venv\\Scripts\\python.exe scripts\\bench_asr.py [--engine faster-whisper sherpa] [--threads 4 8]

Transcribes each slice of `tests/data/two_speakers_16k.wav` (the truth table in
`tests/conftest.py::TWO_SPEAKERS_TRUTH`) with a *fresh* model loaded per (engine, thread count)
combination -- CTranslate2 pins its thread pool at load time, so reusing one model across thread
counts would only measure the first one -- and prints each slice's elapsed time and word overlap
with the truth text, plus the medians per (engine, thread count).

Written for the 2026-09-25 vendor-aware CPU thread work (`exocortex.speech.asr.default_cpu_threads`)
and extended 2026-09-26 (asr-sherpa) with a second engine, sherpa-onnx's NeMo Parakeet-TDT
transducer (`exocortex.speech.asr.SherpaTransducerModel`) -- the CPU speech-recognition path for
non-CUDA machines (`ASRConfig.engine`; see its comment in exocortex/config.py for the numbers this
script measured on this dev box, Ryzen AI MAX+ 395 / Radeon 8060S, and the resulting "auto"
decision). On this dev box Whisper runs on the CPU always (no CUDA), so the thread count is most of
the story for its latency; sherpa runs int8-quantised regardless of `--threads`.

The first faster-whisper run downloads large-v3-turbo (~1.6 GB) from Hugging Face; the first
sherpa run downloads and extracts the Parakeet-TDT bundle (~460-490 MB) via
`exocortex.models_dl.ensure_sherpa_model` into `<project>/models/<sherpa_model>/`.
"""
from __future__ import annotations

import argparse
import re
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from exocortex.cuda_env import setup_cuda_dlls  # noqa: E402

setup_cuda_dlls()

from exocortex.config import ASRConfig  # noqa: E402
from exocortex.speech.asr import Transcriber  # noqa: E402

TWO_SPEAKERS_WAV = ROOT / "tests" / "data" / "two_speakers_16k.wav"
MODELS_DIR = ROOT / "models"

# Mirrors tests/conftest.py::TWO_SPEAKERS_TRUTH (kept as a literal copy so this script has no
# pytest dependency and can run standalone via plain python.exe).
TWO_SPEAKERS_TRUTH = [
    (0.13, 0.98, "A", "Hi everyone!"),
    (1.35, 2.29, "A", "Thanks for joining."),
    (3.17, 6.86, "A", "Today we need to decide on the message queue for the new ingestion service."),
    (7.65, 8.18, "B", "Sure."),
    (9.03, 13.81, "B", "What would you say is the main trade-off between Kafka and RabbitMQ for our use case?"),
    (14.63, 15.38, "A", "Good question."),
    (16.23, 19.89, "A", "Let's also make sure someone owns the migration plan by Friday."),
    (20.71, 22.26, "B", "I can take that action item"),
    (23.11, 26.74, "B", "Can you walk me through how you'd design a rate limiter for the public API?"),
]


def _word_overlap(hypothesis: str, truth: str) -> float:
    """Fraction of truth's words also present in hypothesis (case-insensitive, alnum tokens)."""
    words = lambda s: set(re.findall(r"[a-z0-9']+", s.lower()))  # noqa: E731
    truth_words = words(truth)
    if not truth_words:
        return 1.0
    return len(words(hypothesis) & truth_words) / len(truth_words)


def _bench_one(engine: str, audio, sr: int, threads: int) -> list[tuple[float, float]]:
    """Returns [(elapsed_s, overlap), ...] for every truth-table slice, on a freshly loaded model."""
    if engine == "sherpa":
        cfg = ASRConfig(engine="sherpa", cpu_threads=threads)
        label = f"sherpa-onnx {cfg.sherpa_model}"
    else:
        cfg = ASRConfig(model="large-v3-turbo", engine="faster-whisper", device="cpu",
                         compute_type="int8", cpu_threads=threads)
        label = f"faster-whisper {cfg.model} int8"

    t = Transcriber(cfg, on_status=lambda state, detail: print(f"   [{state}] {detail}"), models_dir=MODELS_DIR)
    print(f"\n-- loading {label} on cpu, cpu_threads={threads} --")
    t0 = time.perf_counter()
    t.load()  # includes warm-up (and, for sherpa, the model download/extract on first run)
    print(f"   loaded in {time.perf_counter() - t0:.1f}s (engine={t.engine_used}, "
          f"model={t.model_used}, cpu_threads_used={t.cpu_threads_used})")

    results: list[tuple[float, float]] = []
    for start, end, speaker, truth_text in TWO_SPEAKERS_TRUTH:
        clip = audio[int(start * sr): int(end * sr)]
        result = t.transcribe(clip)
        overlap = _word_overlap(result.text, truth_text)
        results.append((result.elapsed_s, overlap))
        print(f"   [{speaker}] {result.elapsed_s:5.2f}s  overlap {overlap:4.2f}  got={result.text!r}")
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", nargs="+", choices=["faster-whisper", "sherpa"], default=["faster-whisper", "sherpa"])
    ap.add_argument("--threads", type=int, nargs="+", default=None,
                     help="default: [4, 8] for faster-whisper, [4, 8] for sherpa")
    ap.add_argument("--compute-type", default="int8", help="faster-whisper only (sherpa is always int8)")
    args = ap.parse_args()

    if not TWO_SPEAKERS_WAV.is_file():
        print(f"missing {TWO_SPEAKERS_WAV}", file=sys.stderr)
        return 1

    import soundfile as sf

    audio, sr = sf.read(str(TWO_SPEAKERS_WAV), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    sr = int(sr)

    threads_list = args.threads if args.threads is not None else [4, 8]

    summary: dict[tuple[str, int], tuple[float, float]] = {}
    for engine in args.engine:
        for threads in threads_list:
            rows = _bench_one(engine, audio, sr, threads)
            elapsed_median = statistics.median(e for e, _ in rows)
            overlap_mean = statistics.mean(o for _, o in rows)
            summary[(engine, threads)] = (elapsed_median, overlap_mean)

    print("\n== medians ==")
    for (engine, threads), (elapsed_median, overlap_mean) in summary.items():
        flag = "OK" if overlap_mean >= 0.7 else "LOW"
        print(f"engine={engine:14s} threads={threads}: elapsed median {elapsed_median:.3f}s, "
              f"word-overlap mean {overlap_mean:.2f} [{flag}]")

    if len(args.engine) == 2:
        # Decision rule (exocortex.config.ASRConfig.engine / exocortex.speech.asr.resolve_engine):
        # "auto" picks sherpa on non-CUDA machines if its mean word overlap is >= whisper's - 0.05
        # AND its median latency is <= 50% of whisper's, both at the same (largest) thread count.
        threads = max(threads_list)
        fw_elapsed, fw_overlap = summary[("faster-whisper", threads)]
        sh_elapsed, sh_overlap = summary[("sherpa", threads)]
        overlap_ok = sh_overlap >= fw_overlap - 0.05
        latency_ok = sh_elapsed <= 0.5 * fw_elapsed
        decision = "sherpa" if (overlap_ok and latency_ok) else "faster-whisper"
        print(f"\n== decision rule @ threads={threads} ==")
        print(f"sherpa overlap {sh_overlap:.2f} >= whisper {fw_overlap:.2f} - 0.05 = {fw_overlap - 0.05:.2f}: {overlap_ok}")
        print(f"sherpa median {sh_elapsed:.3f}s <= 50% of whisper median {fw_elapsed:.3f}s = {0.5 * fw_elapsed:.3f}s: {latency_ok}")
        print(f"=> \"auto\" without CUDA should resolve to: {decision}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

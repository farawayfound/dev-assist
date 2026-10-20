# ExoCortex — architecture & module contracts

A fully local, screen-share-invisible meeting co-pilot for Windows 11.
Listens to the meeting (system audio via WASAPI loopback + the user's mic), transcribes live with
speaker labels, takes structured notes, detects questions and streams concise answers from a local
LLM (LM Studio's llama.cpp runtime pack for this machine's GPU vendor by default, model
`maple-preview`; or Ollama, or any OpenAI-compatible server -- `[llm].backend = "auto"` picks, see
"GPU vendor routing" below) into an always-on-top overlay that screen capture cannot see.

Everything runs on this machine (since 2026-09-21): RTX 5090 Laptop (24 GB), Intel Core Ultra 9 285H with an
Intel Arc 140T iGPU and an Intel AI Boost NPU, 64 GB RAM, Python 3.12. (Developed on an RTX 5080 Laptop /
Ryzen AI 9 HX 375 / 32 GB machine, where the pre-2026-09-21 measurements below were taken.)

A second dev box was added 2026-09-25 to build and verify AMD/Intel iGPU support: an ASUS ROG Flow
Z13, AMD Ryzen AI MAX+ 395 (16C/32T Zen 5), Radeon 8060S iGPU (RDNA 3.5, gfx1151, UMA: DXGI 32590
MiB dedicated + 16203 MiB shared; llama.cpp's Vulkan device reports 48971 MiB), no NVIDIA GPU at
all. Its measurements are marked as such throughout this document; the NVIDIA laptop above remains
the machine every NVIDIA-vendor code path is verified against, unchanged.

## Threat model & non-goals

What the hardening protects: **meeting content** reaching a capture surface of the machine the app runs
on — DWM capture APIs (`WDA_EXCLUDEFROMCAPTURE`), legacy window-surface reads (`WS_EX_LAYERED`), Aero
Peek / thumbnails (`DWMWA_EXCLUDED_FROM_PEEK`), and, by *not rendering at all*, mirrors, IDD virtual
monitors and the RDP/Citrix pipeline (the exposure guard plus the fail-closed veil). No accounts, no
cloud, no telemetry: nothing it captures leaves the machine.

What is explicitly **not** provided, and is not to be added: concealment of the program from the
machine's owner, its administrators or its security tooling. So no API unhooking or direct syscalls,
no hiding from process / service / network enumeration, no faking of the session type (a console
session is reported as a console session), no kernel drivers and no traffic obfuscation. The process,
its windows, its hotkeys, its audio capture and its CPU/GPU use stay plainly observable; a detection
only ever *adds* protection (veil or hide) and is never used to hide activity. Every probe is advisory
in the safe direction — `[]` / `None` / `False` mean "nothing detected", never "safe" — and a renamed
binary or an unlisted display driver defeats it.

Since 2026-09-24 the footprint is at least *managed*, lightly, by `[stealth]` (`exocortex/stealth.py`),
which softens two of the items above instead of removing them: a live process listing reads
`ExoCortex` instead of `pythonw.exe -m exocortex` (the PEB caption — but the kernel keeps the command
line it copied at process creation, so ETW process-start events still carry the original), the priority
class drops to `below_normal` (inherited by `llama-server`), and session folders are stamped to the
meeting and pruned by age and/or count rather than accumulating forever. That is a smaller, duller
footprint, not concealment: nothing is hidden from enumeration, every call is advisory in the same safe
direction (`None` / `False` / `[]` = "could not do it"), and the caption rewrite is coarse enough to
replace the *whole* command line — which is why `--replace` is handed the caption, so a session that
has rewritten itself is still recognised as one of ours (`exocortex/instance.py`).

## Verified facts (measured on 2026-09-20 on the target machine — do not re-derive)

| Item | Result |
|---|---|
| Whisper `large-v3-turbo`, CUDA float16 via faster-whisper 1.2.1 / ctranslate2 4.8.2 | loads in 2 s (cached), **0.11–0.14 s per 1–5 s utterance**, first call ~12 s (CUDA warm-up → do a warm-up transcribe at startup) |
| CUDA DLLs | only work when `exocortex.cuda_env.setup_cuda_dlls()` runs **before** importing `faster_whisper`/`ctranslate2` (prepends `site-packages/nvidia/*/bin` to `PATH`) |
| Silence hallucination | Whisper returns " Thank you." on pure silence → never transcribe audio the VAD did not mark as speech; drop text when `no_speech_prob > 0.7` or `avg_logprob < -1.2` |
| WASAPI devices (pyaudiowpatch 0.2.12.8) | default loopback = index 14 "Odyssey G95SD (NVIDIA High Definition Audio) [Loopback]" 48000 Hz 2 ch; mic = "Microphone Array (AMD Audio Device)" 44100/48000 Hz 2 ch. Loopback devices have `isLoopbackDevice=True`; `p.get_default_wasapi_loopback()` works |
| sherpa-onnx 1.13.8 VAD | `VadModelConfig().silero_vad.{model,threshold,min_silence_duration,min_speech_duration,max_speech_duration,window_size=512}`, `sample_rate`, `num_threads`; `VoiceActivityDetector(cfg, buffer_size_in_seconds=60)`; methods `accept_waveform(np.float32[512])`, `is_speech_detected()`, `empty()`, `front` (`.start` sample index, `.samples`), `pop()`, `flush()`, `reset()`. Model: `models/silero_vad.onnx` (already downloaded). 27 s of audio → 9 clean sentence segments in 0.08 s |
| sherpa-onnx speaker embeddings | `SpeakerEmbeddingExtractorConfig(model=..., num_threads=2, provider="cpu")`, `SpeakerEmbeddingExtractor(cfg)`, `.dim` (512 for CAM++), `st=ext.create_stream(); st.accept_waveform(sample_rate=16000, waveform=x); st.input_finished(); emb=ext.compute(st)` → ~16 ms per utterance on CPU. Model `models/wespeaker_en_voxceleb_CAM++.onnx` (downloaded). Cosine same-speaker ≈ 0.5–0.9 for utterances ≥ 1.5 s; **utterances < 1 s are unreliable (cross-speaker similarity up to 0.9)** |
| LM Studio 0.4.23 | server **`http://127.0.0.1:1234`** (never `localhost`: LM Studio does not listen on IPv6 and Windows tries `::1` first, costing ~2.3 s per fresh connection; with 127.0.0.1 + one keep-alive `httpx.Client` TTFT is 0.05–0.4 s), model id **`maple-preview`** (deepgrove/maple-preview, ternary MoE 256 experts/8 active, ctx 131072, Qwen3-style ChatML template that always opens `<think>` in the generation prompt). `/v1/models` lists ids; `GET /api/v0/models` gives `state` ("loaded"/"not-loaded"), `quantization`, `max_context_length`, `loaded_context_length`, `arch`; `/v1/completions` with a raw prompt works and does NOT apply a template (its final SSE chunk always carries `usage`); `/v1/chat/completions` streams reasoning as `delta.reasoning_content` and strips `<think>` tags (its `usage` chunk only appears with `stream_options: {"include_usage": true}`). **No parameter disables thinking for this model** (`reasoning_effort`, `chat_template_kwargs.enable_thinking`, `think`, `/api/v1/chat reasoning` all no-op or 400). Streaming is genuine token-by-token SSE `data: {...}` / `data: [DONE]`. Generation ≈ 55 tok/s; prompt eval ≈ 150 tok/s (slow → keep prompts short and prefix-stable so llama.cpp's KV prefix cache hits; cached prefixes re-evaluate only the new tokens) |
| **LLM backend = llama.cpp (default, decided 2026-09-20)** | ExoCortex spawns its own hidden `llama-server.exe` from LM Studio's runtime pack (`~/.lmstudio/extensions/backends/llama.cpp-win-x86_64-nvidia-cuda12-avx2-<ver>/`, vendor DLLs in `.../backends/vendor/win-llama-cuda12-vendor-v2`) on 127.0.0.1 and a free port, inside a Job Object (dies with the app). Model files come from LM Studio's library (`downloadsFolder` in `~/.lmstudio/settings.json` = `D:\AI\models`); `[llm].model` accepts a key (`maple-preview`), a file name or a path. LM Studio itself does not need to run. Measured: load 1.3–3 s, TTFT 60 ms (cached prefix) / ~0.5 s (uncached 70-token prompt), ~62 tok/s, prompt cache works (`timings.cache_n`). Flags used: `--n-gpu-layers 999999 --n-cpu-moe 0 --flash-attn on --kv-offload --parallel 2 --kv-unified --ctx-size 8192 --threads <cores-2> --no-webui`. The TQ1_0 ternary tensors of maple-preview are NOT CUDA-accelerated ("cannot be used with preferred buffer type CUDA0, using CPU"): 4.6 GB runs on the CPU, only the 215 MB head on the GPU; so `threads` matters and a Q4_K model would use the GPU fully. llama-server's `/v1/models` reports the GGUF path as the model id and ignores the `model` field; `/health` → 200 when loaded; final SSE chunk carries `timings` (`predicted_n`, `predicted_per_second`). The `LMStudioClient` is OpenAI-compatible and works unchanged (`set_base_url()` after the server is up). `[llm].backend = "openai"` talks to any OpenAI-compatible server instead (llama-swap on :8081, LM Studio on :1234). |
| No-think trick | raw prompt ending in `<|im_start|>assistant\n<think>\n\n</think>\n\n` makes the model answer directly (good quality, 3 bullets in ~1.3 s server-side). Sampling per Qwen3 guidance (non-thinking): temperature 0.6–0.7, top_p 0.8–0.85, top_k 20, min_p 0; add presence_penalty ~0.5–1.0 for long summaries if repetition appears |
| `lms` CLI | `lms load maple-preview --gpu max --context-length 16384 --identifier maple-preview -y`, `lms ps`, `lms server start --port 1234`. Requires the LM Studio app running |
| Test audio | `tests/data/two_speakers_16k.wav` (27 s, 16 kHz mono, two Windows TTS voices, contains two questions) |
| **OCR on Intel iGPU / NPU via OpenVINO (measured 2026-09-23)** | This laptop: Core Ultra 9 285H (16 cores), Arc 140T iGPU, AI Boost NPU, RTX 5090 Laptop 24 GB, 64 GB RAM, Windows 11 26200; dense 2560x1600 page, 97 lines, OCR input downscaled to 2200x1375. Windows OCR 108 ms wall / 109 ms CPU (81 lines). RapidOCR PP-OCRv6 small: onnxruntime-CPU 3.7 s wall / 37 s CPU (all cores); OpenVINO-CPU 1.9 s / 9.9 s CPU; OpenVINO iGPU (`GPU.0`) **0.73 s / 0.74 s CPU** (first-ever compile ~13 s + ~50 s first-inference kernel builds, ~2 s + ~2 s once `CACHE_DIR` has the compiled kernels; cache 327 files / 190 MB); OpenVINO NPU (static shapes, one compile per input-shape bucket, 8-14 s each uncached / ~0.1-0.2 s cached) 1.45 s / 0.6 s CPU. The text classifier (`cls`) is off (screens are never upside down; saves ~150 ms). OCR never runs on the RTX 5090 (it holds Whisper + llama-server) so the 5090's load is unchanged -- the win is CPU time and latency, not GPU headroom. OpenVINO 2026.4.0 `core.available_devices` on this machine: `['CPU', 'GPU.0', 'GPU.1', 'NPU']` -- `GPU.0` = Intel Arc 140T (iGPU), `GPU.1` = RTX 5090 (dGPU, never picked by `resolve_ocr_device`), `NPU` = Intel AI Boost. |
| **RapidOCR / Windows OCR process-order rule (measured 2026-09-23)** | Creating a `RapidOcr` (any backend: onnxruntime or OpenVINO on any device) AFTER a `WindowsOcr` exists in the same process crashes the interpreter outright (access violation, 0xC0000005); creating `RapidOcr` first and `WindowsOcr` second, in either order after that, coexist fine for the life of the process. So anything that wants both must build the RapidOCR engine first -- `exocortex.screen.reader.ScreenReader` builds its accurate re-read engine (RapidOCR) before its sampling engine (Windows OCR by default), and `exocortex.screen.ocr.RapidOcr.__init__` refuses with `OcrUnavailable` (instead of crashing) when a `WindowsOcr` already exists in the process (module-level `_windows_ocr_created` flag). |
| **Hands-free root causes (from `%LOCALAPPDATA%\ExoCortex\logs\exocortex.log`, session of 2026-09-23 22:44-22:50, pre-hardening build)** | Three failure modes in one session drove the redesign below. (1) The old whole-screen "technical cue" gate vetoed real slide changes: `22:46:27,536` / `22:46:30,054` / `22:46:35,131` each logged `hands-free: new screen but nothing technical on it; ignoring` although the slide had changed. (2) The old `min_changed_lines` check counted only the lines differing between two *samples*, undercounting a genuine one-line question change on an otherwise static slide: `22:46:58,577` logged `hands-free: settled screen has nothing new (2 line(s)); no cycle` against `min_changed_lines = 3`. (3) The ordinary Code ask does not cover a question ABOUT the code shown (a quiz slide with a snippet), so the model correctly answered the skip marker -- but a skip was indistinguishable from a preemption, so the controller retried it twice before giving up: three `assist job N (code): model skipped ((nothing to do))` lines per coding slide, at `22:44:58,870` / `22:44:59,285` / `22:44:59,721` (jobs 4/5/6) and again at `22:50:22,625` / `22:50:23,077` / `22:50:23,519` (jobs 25/26/27), each of the first two paired with `hands-free: code step was preempted; retrying (n/2)`. |
| **AMD/Intel iGPU dev box (measured 2026-09-25/26 on a Ryzen AI MAX+ 395, Radeon 8060S RDNA 3.5 gfx1151, NO NVIDIA driver)** | `exocortex.gpu.detect_gpus()` (DXGI): vendor `0x1002`, `DedicatedVideoMemory` 32590 MiB, `SharedSystemMemory` 16204 MiB (16203 in the task's own framing), D3D12 `UMA = 1` -> `budget_mb` (dedicated+shared, UMA) = 48793 MiB; `llama-server --list-devices` (Vulkan) independently reports the same adapter at 48971 MiB. PDH `\GPU Adapter Memory(luid_<luid>_phys_*)\Dedicated Usage` free reading ~31 GB, ~0.007 ms per reading (`vram_free_mb()` falls through to PDH here since NVML is absent). LM Studio's runtime packs present: `amd-rocm-avx2-2.29.1` (b10355, the ladder's first choice for `vendor="amd"` since the 2026-09-26 idle benchmark below), `vulkan-avx2-2.46.0` (llama.cpp b11189, newest of five installed Vulkan versions, the fall-through), `nvidia-cuda-avx2-2.28.2` (unusable here), CPU packs. `choose_model("auto", ..., budget_mb=48793)` picked `lmstudio-community/Qwen3.8-27B` (16.8 GB) over the complete-but-smaller `Muse-Glimmer-30B` (16.75 GB) and the 1-bit `Bonsai-27B-Q1_0` (3.5 GB) -- the largest file clearing the 0.75x margin. |
| **ROCm vs Vulkan on this box (`bench_llm.py --protocol full`, 2026-09-26)** | Idle machine, `Qwen3.8-27B-Q4_K_M` (16.8 GB), ctx 8192, `--parallel 2`, two rounds each, max_tokens 96, fresh 2651-token prompt / cached repeat / 953-token incremental append: **ROCm 2.29.1** pp 300 tok/s, tg 11.2 tok/s, cached TTFT 0.4 s, incremental pp 273 tok/s, 16069 MiB GPU; **Vulkan 2.46.0** pp 158-162 tok/s, tg 11.4-12.4 tok/s, cached TTFT 3.4 s, incremental pp 91-93 tok/s, 16166 MiB GPU; load 361 s cold from the exFAT card, 15.9 s (Vulkan) / 16.8 s (ROCm) warm. Hence `_VENDOR_LADDER["amd"] = ("amd-rocm", "vulkan", "cpu")` with `LlamaServerManager.start()` falling through to the next pack when a GPU pack lists no device or fails while loading. An earlier same-day sweep on the 1-bit `Bonsai-27B-Q1_0` under heavy concurrent load (Whisper benchmarks and five test runs in parallel) had shown ROCm generating 3x slower (2.50 vs 7.75 tok/s) and Qwen at 2.9 tok/s on Vulkan -- contention artefacts, superseded by the idle run. Still valid from that sweep: `flash_attn=False` with the default `kv_cache_type="q8_0"` is a hard llama-server crash (`quantized V cache requires flash_attn to be enabled`) -- `build_args` now forces the V cache to `f16` in that case (K stays quantized); `--cache-reuse 256` is disabled by llama-server itself ("not supported by this context") whenever `--parallel 2` forces `--kv-unified`, on both backends (`scan_log_warnings` surfaces it; unverified on NVIDIA, same default); a repeated flag in `llama_extra_args` overrides the earlier value (`--ubatch-size` verified in the log); `Bonsai-27B-Q1_0` runs coherently on Vulkan with one tensor (`token_embd.weight`) on the CPU. Cold loads of the 16.8 GB file from the removable card: 323-361 s, past the 180 s default `llama_startup_timeout_s` -- what `estimate_startup_timeout_s()` (below) exists for. |
| **ASR engine choice on this box (`scripts/bench_asr.py --engine faster-whisper sherpa --threads 8 4`, `tests/data/two_speakers_16k.wav`, 2026-09-26)** | faster-whisper `large-v3-turbo` int8/CPU: 6.03 s median latency at 8 threads (7.16-7.34 s at 4 threads across two runs), mean word overlap 1.00. sherpa-onnx `parakeet-tdt-0.6b-v2-int8` int8/CPU: 0.19-0.20 s median at 8 threads (0.14-0.16 s at 4 -- thread count barely matters at this model size), mean word overlap 0.99. Decision rule at threads=8: overlap 0.99 >= 1.00 - 0.05 (True) AND median 0.187-0.203s <= 50% of whisper's 6.03-6.06s = 3.0s (True, sherpa was ~3% of whisper's latency) -> `resolve_engine()`'s "auto" resolves to "sherpa" off CUDA. Live download+extract of `parakeet-tdt-0.6b-v2-int8.tar.bz2` (Content-Length 482,468,385 bytes) verified via `httpx` GET/stream; `SherpaTransducerModel` construction (`num_threads=8`) 1.7-3.6 s, RSS delta ~739 MB. `provider="directml"` does not raise on this sherpa-onnx 1.13.8 wheel (no DirectML EP compiled in) -- it prints a native "... Fallback to cpu!" line and silently builds a CPU session; `self.provider` still reads `"directml"` (see `SherpaTransducerModel`'s docstring for why this could not be reliably detected from Python). This dev box's faster-whisper number moved from a 2026-09-25 baseline of 3.24 s median to 6.03 s measured 2026-09-26 (machine under other concurrent load); either number leads to the same "auto" -> sherpa decision by a wide margin. |
| **OCR: OpenVINO's GPU plugin crashes on this AMD iGPU (measured 2026-09-25)** | OpenVINO 2026.4.0 `core.available_devices` here: `['CPU', 'GPU']` (`GPU` `FULL_DEVICE_NAME` = "gfx1151 (iGPU)"). Compiling RapidOCR's det/rec models through OpenVINO's `GPU` device crashes the interpreter outright (`STATUS_STACK_BUFFER_OVERRUN`, 0xC0000409) inside the plugin's kernel compiler -- it emits Intel GEN/vISA inline assembly (literal `.decl` / `.implicit_PSEUDO_INPUT` directives) the AMD OpenCL ICD cannot parse; reproducible cold and with a warm on-disk `CACHE_DIR`. `resolve_ocr_device()` therefore never routes a non-Intel OpenVINO `GPU` device this way; a non-NVIDIA hardware GPU goes through onnxruntime's DirectML execution provider instead (`directml_available()`, `_pick_dml_adapter`, `_OrtSession`, `requirements-directml.txt`). Measured dense-screen numbers on this iGPU: `onnxruntime-directml` ~0.95 s wall / ~1 s CPU time; plain OpenVINO CPU ~1.1 s / 5-6 s CPU; plain onnxruntime (no DirectML) ~1.5 s / 22 s CPU; Windows OCR 0.13 s. DirectML is never offered on the discrete NVIDIA GPU (`_pick_dml_adapter` returns `None` when any detected adapter is NVIDIA) -- OCR still never touches that card, unchanged. |
| **Upstream llama.cpp runtime, live check on the AMD box (2026-09-26, intel-integration)** | With b11200's vulkan/sycl/cpu zips already extracted under `.tools/llama.cpp` (no network call needed), `find_server_binary(vendor="intel", source="upstream", upstream_root=Path(".tools/llama.cpp"))` -> flavor "vulkan", source "upstream", reason "vendor intel -> vulkan (upstream llama.cpp b11200)" -- exactly as designed (Intel's ladder is vulkan > cpu). A full `LlamaServerManager(LLMConfig(llama_source="upstream", llama_backend="vulkan", model="lmstudio-community/Bonsai-27B")).start()` on this box (vendor auto-detects "amd" here; backend forced to vulkan) loaded the upstream Vulkan build end to end: ready in 87.4 s (a cold 3.8 GB read off the E: exFAT card), `/health` 200, one `/v1/completions` (2 tokens, "OK") answered in 4.09 s at ~3.7 tok/s prompt-eval / ~11.8 tok/s generation (a fresh, cold-cache read, not a warm-model number), `fallbacks == []`, `stop()` left no `llama-server.exe` behind. This exercises the upstream fetch/verify/spawn path end to end without any Intel hardware -- the AMD vendor detection only decided which model-budget math ran, not which runtime binary loaded. |
| **SYCL negative control, live on the AMD box (2026-09-26, intel-integration)** | `LLMConfig(llama_source="upstream", llama_backend="sycl", model="lmstudio-community/Bonsai-27B")` on the same box: the already-installed upstream SYCL runtime lists no device (there is no Intel GPU here, exactly as expected -- this is the negative control the checklist calls for), the one conservative retry (flash-attn auto + f16 KV) sees none either (device listing does not depend on that flag), then the fall-through lands on the upstream Vulkan build (already installed too) and loads normally. `mgr.fallbacks == ["retrying intel-sycl with flash-attn auto + f16 KV", "intel-sycl b11200: sycl llama-server lists no GPU device on this machine"]` -- exactly the "untested vendor degrades to a working server instead of failing the start" behaviour this branch adds, proven end to end even though no SYCL-capable device exists to load onto. |
| **Upstream CUDA runtime, live check on the RTX 5080 Laptop dev box (2026-09-27, portable-install)** | `driver_version` 592.82 (`>= 580`, so `cuda_kind_for_driver` resolves `"cuda13"`), `LMSTUDIO_HOME` pointed at an empty folder so no LM Studio pack existed for any flavor at all. `find_server_binary()` under the default `source="auto"` fetched `llama-b11200-bin-win-cuda-13.4-x64.zip` + `cudart-llama-bin-win-cuda-13.4-x64.zip` (423.5 MB in 24.5 s for the cudart zip alone, both verified by exact size + sha256) into `.tools/llama.cpp/b11200/cuda13/`, returning flavor `"nvidia-cuda12"` (the ladder's first-rung NAME; the actual installed KIND is `cuda13`, per `_upstream_kind_for`), source `"upstream"`, framework `"cuda"`. A full `LlamaServerManager(LLMConfig()).start()` on it (Qwen2.5-0.5B-Instruct Q4_K_M): **ready 40.0 s after `start()`, including both downloads**, `fallbacks == []`, `/health` 200, a 24-token `/v1/completions` in 0.22 s -- 318 tok/s prompt-eval / 473 tok/s generation. This is the "no LM Studio, no Ollama, first launch on an NVIDIA laptop" scenario end to end: the upstream CUDA fetch (two zips, cudart included), a genuinely cold two-zip download inside `start()`'s own timeout budget, and a working GPU completion, with no LM Studio pack present at all. |
| **Hugging Face pull + auto-profile, live on the same box (2026-09-27, portable-install)** | `pull_model("Qwen/Qwen2.5-0.5B-Instruct-GGUF", "Q4_K_M", cfg)`: 491 MB pulled in 34 s, sha256 matched the Hub's own LFS hash for every shard, `auto_profile` chose `ctx_size=32768` (the model's trained context fits the budget whole) / `kv_cache_type="q8_0"` / `flash_attn=True`; a re-pull of the same file after completion skipped it (exact size match, no request made) in 1.4 s. Scanning 13 LM Studio models with `with_meta=True` (full GGUF header parse, every file) took 1.5 s. `hf_search("qwen", limit=25)` 0.4 s; `hf_repo_info` (listing the repo's shards/quants) 0.4 s. |

## Package layout

```
exocortex/
  __init__.py, __main__.py        python -m exocortex [--config path] [--doctor]
  cuda_env.py                      (done) DLL setup — import first
  bus.py                           (done) EventBus + all event/data types
  config.py                        (done) Config dataclasses + TOML
  logging_setup.py                 setup_logging(level, log_dir) -> rotating file + stderr
  gpu.py                            vendor-neutral GPU discovery (DXGI/D3D12) + NVML/PDH VRAM watch + VramGuard;
                                     `python -m exocortex.gpu --vendor/--json/--extras` (added 2026-09-25, GPU support)
  hardware.py                      collect(cfg.llm) -> HardwareReport (GPU/CPU/RAM/disk, the runtime pack that
                                     would be picked, a recommended model-fit policy); describe/to_dict/save/load,
                                     report_path(cfg.llm) = <library_dir>/hardware.json (added 2026-09-27,
                                     portable-install)
  setup.py                         python -m exocortex.setup: run_setup(cfg) -- the four first-run steps
                                     run.bat/run.vbs call on every launch (added 2026-09-27, portable-install)
  models_dl.py                     ensure_models(cfg) -> downloads silero/speaker ONNX if missing;
                                     ensure_sherpa_model() -> downloads/extracts a sherpa-onnx ASR bundle (added 2026-09-26)
  transcript.py                    Transcript store
  session.py                       SessionRecorder (autosave/export)
  stealth.py                       [stealth]: PEB caption + priority class; session-folder stamping + pruning (added 2026-09-24)
  audio/capture.py                 AudioCapture (WASAPI loopback + mic → 16 kHz mono float32 frames)
  audio/resample.py                Resampler (stateful, numpy only)
  speech/vad.py                    VadSegmenter
  speech/asr.py                    Transcriber (faster-whisper, or sherpa-onnx via `[asr].engine`, added 2026-09-26)
  speech/diarize.py                SpeakerClusterer
  speech/pipeline.py               SpeechPipeline (threads: capture → VAD → ASR → diarization → bus)
  llm/server.py                    LlamaServerManager: spawns/supervises a private llama-server from an LM Studio
                                     runtime pack, vendor-aware pack/device/model selection (added 2026-09-25);
                                     source-aware (LM Studio pack vs. upstream build) and stop-event based
                                     quit-during-load fix (added 2026-09-26, Intel Arc discrete-GPU support)
  llm/runtimes.py                  fetch/verify (size + sha256)/extract a pinned official llama.cpp Windows
                                     build (vulkan/sycl/openvino/cpu/cuda12/cuda13/rocm) -- a second runtime
                                     source alongside LM Studio's packs, tried once no LM Studio pack covers a
                                     flavor at all (added 2026-09-26 for Intel Arc SYCL/OpenVINO; note,
                                     2026-09-27 -- "Portable install + model manager" below -- extended to
                                     every vendor, CUDA/ROCm included: the 2026-09-26 "never used for
                                     CUDA/ROCm" line above is superseded, not a design that changed shape)
  llm/library.py                   this app's own GGUF library (discovery, GGUF header metadata, Hugging
                                     Face search/pull, per-model tuning profiles) -- added 2026-09-27, see
                                     "Portable install + model manager" below
  llm/route.py                     resolve_route(cfg): llamacpp -> ollama -> openai -> "none", for [llm].backend = "auto"
                                     (added 2026-09-25)
  llm/client.py                    LMStudioClient (httpx, streaming, cancel); OllamaClient (Ollama's native API,
                                     added 2026-09-25); make_client(cfg, backend)
  llm/prompts.py                   ChatML rendering + prompt builders
  llm/questions.py                 is_question / extract_question heuristics; focus_level / is_task_line / is_relevant
  llm/choices.py                   multiple-choice: option-set detection, <choices> prompt block, answer verdict line
  llm/assistant.py                 Assistant (auto/manual answers, ask, notes)
  handsfree.py                     HandsFreeController + ScreenContext; classify_change/is_new_task/task_kind (hands-free mode, added 2026-09-23; manual advance / trigger_step, 2026-09-25)
  ui/win32.py                      ctypes helpers (capture exclusion, tool window, click-through, topmost)
  ui/hotkeys.py                    GlobalHotkeys (RegisterHotKey + Qt native event filter)
  ui/llhooks.py                    MouseWheelHook / BareKeyHook / PressHoldMouseHook (WH_MOUSE_LL / WH_KEYBOARD_LL, added 2026-09-23, later; press-and-hold 2026-09-25)
  ui/theme.py                      QSS + colors
  ui/widgets.py                    TranscriptView, AssistView, NotesView, StatusBar, AskBox
  ui/overlay.py                    OverlayWindow
  ui/models.py                     ModelsDialog: Library / Discover / Tuning / Server / Hardware tabs
                                     (added 2026-09-27, portable-install)
  doctor.py                        environment diagnostics: GPU (vendor-neutral), audio, models, screen/OCR,
                                     completion, LLM backend (route-aware), overlay; `python -m exocortex --doctor`
  app.py                           Engine (composition root) + main()
scripts/doctor.py                  thin wrapper: `load_config()` + `exocortex.doctor.run_doctor()`
scripts/bench_llm.py               TTFT / tok/s benchmark for the configured model; `--protocol full` measures
                                     load time, GPU-MiB delta and llama-server's own pp/tg/cache timings (added 2026-09-26)
scripts/bench_asr.py               faster-whisper vs sherpa-onnx latency/accuracy benchmark (added 2026-09-26)
scripts/bench_ocr.py               OCR device/engine benchmark (Windows OCR / RapidOCR: onnxruntime, OpenVINO, DirectML)
tests/                             pytest; offline (no LM Studio needed) except tests marked `live`
models/                            silero_vad.onnx, wespeaker_en_voxceleb_CAM++.onnx, and (sherpa engine)
                                     parakeet-tdt-0.6b-v2-int8/ or -v3-int8/ (extracted on first use)
config.example.toml, requirements.txt, requirements-nvidia.txt, requirements-directml.txt (added 2026-09-25),
README.md, docs/INSTALL.md, run.bat, run.vbs
```

## Threading model

```
pyaudio callback threads (2)  ──frames──▶  SpeechPipeline per-source worker (2)  ──jobs──▶  ASR worker (1, owns WhisperModel)
                                             (resample done in capture, VAD here)                 │ finals → SpeakerClusterer (CPU, ~16 ms)
                                                                                                    ▼
                                                                                                  EventBus ──▶ Qt UI (QTimer poll, 30 ms)
                                                                                                           ──▶ Assistant worker (question detection, LLM jobs)
                                                                                                           ──▶ SessionRecorder (autosave)
```

- No module imports Qt except `exocortex/ui/*` and `app.py`.
- No module blocks the audio callback; it only resamples and enqueues.
- The ASR worker uses a priority queue: **final** jobs before **partial** jobs; only the newest partial per source is kept (stale partials are dropped).
- All cross-thread communication goes through `EventBus` or `queue.Queue`. UI never calls engine methods that block; engine methods return immediately and publish events later.
- Session clock: `Engine.clock()` returns seconds since session start (`time.monotonic() - t0`). Everything time-stamped uses it.

## Module contracts

Type hints are mandatory; every public class/function has a docstring. Use `logging.getLogger(__name__)`.
No prints outside `scripts/` and `__main__`. Python 3.12 features are fine.

### audio/resample.py
```python
class Resampler:
    """Stateful mono resampler. Keeps fractional phase and filter history across calls so
    consecutive chunks join seamlessly. Numpy only.
    - For integer ratios (48000→16000 = 3:1) use a windowed-sinc low-pass FIR + decimation.
    - For non-integer (44100→16000) use polyphase-free approach: FIR low-pass at min(in,out)/2 then linear
      interpolation with a carried fractional position. Good enough for ASR."""
    def __init__(self, in_rate: int, out_rate: int) -> None: ...
    def process(self, mono: np.ndarray) -> np.ndarray: ...   # float32 in, float32 out
    def reset(self) -> None: ...

def downmix(interleaved: np.ndarray, channels: int) -> np.ndarray: ...  # mean over channels → mono float32
class FrameChunker:
    """Re-buffers arbitrary-length arrays into exact fixed-size frames (e.g. 512 samples)."""
    def __init__(self, frame_size: int) -> None: ...
    def push(self, x: np.ndarray) -> list[np.ndarray]: ...
    def flush(self) -> np.ndarray | None: ...
```

### audio/capture.py
```python
@dataclass
class DeviceInfo:
    index: int; name: str; is_loopback: bool; is_input: bool; sample_rate: int; channels: int; is_default: bool

def list_devices() -> list[DeviceInfo]        # WASAPI host API only
def resolve_device(kind: Literal["mic","system"], spec: str) -> DeviceInfo   # "default" | name substring | int

class AudioCapture:
    """Opens one pyaudiowpatch input stream (loopback or mic) with the device's native rate/channels,
    converts to 16 kHz mono float32 and delivers exact `frame_samples` frames via `on_frames`.
    - Uses the callback API (stream_callback) with frames_per_buffer ≈ 20 ms at device rate.
    - `on_level(rms, peak)` is called at most every 100 ms.
    - `on_status(state, device_name, detail)` mirrors AudioStatusEvent states.
    - Device watchdog: a daemon thread ticks every 2 s to check stream health only (`is_active()`
      False / callback silent for 3 s while running); full device re-enumeration (loopback follows
      the default output; mic follows the default input) runs every 10 s per stream, or at once
      when a cheap stdlib `winmm` device-count probe changes -- either way it reopens the stream
      and reports "device_changed" (docs/UPSTREAM_SUGGESTIONS.md #4 has the reasoning: reusing the
      open PyAudio instance cannot work, since PortAudio snapshots its device list once at
      `Pa_Initialize` and never refreshes it).
    - Muting (`set_muted(True)`) keeps the stream open but delivers zeros and suppresses levels."""
    def __init__(self, source: Source, spec: str = "default", *, target_rate: int = 16000,
                 frame_samples: int = 512, on_frames: Callable[[np.ndarray], None],
                 on_level: Callable[[float, float], None] | None = None,
                 on_status: Callable[[str, str, str], None] | None = None) -> None: ...
    def start(self) -> DeviceInfo: ...
    def stop(self) -> None: ...
    def set_muted(self, muted: bool) -> None: ...
    @property
    def device(self) -> DeviceInfo | None: ...
    @property
    def running(self) -> bool: ...
```
pyaudiowpatch specifics (verified): `p = pyaudiowpatch.PyAudio()`; `p.get_host_api_info_by_type(pyaudiowpatch.paWASAPI)`;
device dicts have `isLoopbackDevice`, `maxInputChannels`, `defaultSampleRate`, `name`; `p.get_default_wasapi_loopback()`;
`p.get_default_input_device_info()`; loopback streams must be opened with `channels=dev["maxInputChannels"]`,
`rate=int(dev["defaultSampleRate"])`, `input=True`, `input_device_index=dev["index"]`, `format=paFloat32`.
When nothing is playing, loopback delivers zeros (not an error).

### speech/vad.py
```python
@dataclass(frozen=True)
class SpeechStart:   t_start: float
@dataclass(frozen=True)
class SpeechOngoing: t_start: float; audio: np.ndarray        # audio since start (incl. pre-roll)
@dataclass(frozen=True)
class SpeechEnd:     t_start: float; t_end: float; audio: np.ndarray
VadEvent = SpeechStart | SpeechOngoing | SpeechEnd

class VadSegmenter:
    """Wraps sherpa_onnx Silero VAD. `process(frame)` takes exactly window_size (512) float32 samples
    at 16 kHz and returns 0..n events. Time is derived from the number of samples fed (`t0` offset
    passed at construction, so the caller can align to the session clock).
    Emits SpeechOngoing at most every `ongoing_interval_s` while speech continues.
    Uses `vad.front.start`/`.samples` for the authoritative SpeechEnd audio, and its own ring buffer
    (pre_roll_s + audio since detection) for SpeechOngoing."""
    def __init__(self, model_path: str, *, sample_rate: int = 16000, threshold: float = 0.5,
                 min_silence_s: float = 0.45, min_speech_s: float = 0.25, max_speech_s: float = 20.0,
                 pre_roll_s: float = 0.3, ongoing_interval_s: float = 1.2, num_threads: int = 1) -> None: ...
    def set_time_origin(self, t0: float) -> None: ...   # session-clock time of the next sample
    def process(self, frame: np.ndarray) -> list[VadEvent]: ...
    def flush(self) -> list[VadEvent]: ...
    def reset(self) -> None: ...
    @property
    def in_speech(self) -> bool: ...
```

### speech/asr.py
```python
@dataclass(frozen=True)
class AsrResult:
    text: str; avg_logprob: float; no_speech_prob: float; audio_s: float; elapsed_s: float; dropped: bool; reason: str

class Transcriber:
    """faster-whisper wrapper. `load()` resolves device/compute_type ("auto": cuda+float16 if
    ctranslate2.get_cuda_device_count()>0 else cpu+int8), loads, and runs a warm-up transcribe of 1 s
    silence. Thread-safe via an internal lock (one transcribe at a time).
    transcribe(): beam_size from config, language fixed, vad_filter=False (we already segment),
    condition_on_previous_text=False, without_timestamps=True, initial_prompt built from hotwords
    ("Glossary: a, b, c." style) when provided. Applies drop rules (no_speech/logprob thresholds and a
    hallucination blocklist for clips < 1.5 s: "thank you", "thanks for watching", "you", "bye", "." ...).
    For `partial=True` use beam_size=1 and skip the blocklist except empty text."""
    def __init__(self, cfg: ASRConfig, on_status: Callable[[str, str], None] | None = None) -> None: ...
    def load(self) -> None: ...
    @property
    def device_used(self) -> str: ...
    @property
    def compute_type_used(self) -> str: ...
    def transcribe(self, audio: np.ndarray, *, partial: bool = False) -> AsrResult: ...
```

### speech/diarize.py
```python
class SpeakerClusterer:
    """Online cosine clustering over sherpa-onnx speaker embeddings (L2-normalised).
    assign(audio) → (speaker_id, label). Rules:
    - duration < enroll_min_s: never creates/updates a speaker. Assign to the best centroid if
      sim ≥ threshold + 0.15, else to the most recent speaker (continuity), else "spk1".
    - otherwise: best centroid sim ≥ threshold → assign and update centroid (running mean, re-normalised);
      else create a new speaker "spkN"/"Speaker N" (up to max_speakers; beyond that assign to best).
    - rename(id, label) is remembered; labels() returns {id: label}. reset() clears everything.
    Embedding extraction ~16 ms CPU; called from the ASR worker thread only."""
    def __init__(self, model_path: str, *, threshold: float = 0.45, enroll_min_s: float = 1.5,
                 max_speakers: int = 8, num_threads: int = 2, sample_rate: int = 16000) -> None: ...
    def assign(self, audio: np.ndarray) -> tuple[str, str]: ...
    def rename(self, speaker_id: str, label: str) -> None: ...
    def labels(self) -> dict[str, str]: ...
    def reset(self) -> None: ...
```

### speech/pipeline.py
```python
class SpeechPipeline:
    """Owns AudioCapture×2 (per AudioConfig flags), VadSegmenter×2, one ASR worker thread, the clusterer.
    - MIC finals → speaker ("me", "Me"). SYSTEM finals → clusterer.assign (if enabled) else ("spk1","Speaker 1").
    - Echo suppression (cfg.audio.echo_suppression): drop a MIC final whose normalised token set overlaps
      ≥ 0.8 with any SYSTEM final that ended within the last 8 s (and vice-versa, keep the earlier one).
    - Publishes AudioLevelEvent (throttled), AudioStatusEvent, PartialTranscriptEvent, FinalTranscriptEvent,
      ComponentStatusEvent("asr"/"audio"/"diarization").
    - Segment ids are monotonically increasing ints from `next_id()` supplied by the Engine.
    - is_question on the segment is set using llm.questions.is_question(text) (SYSTEM source only).
    Integration facts learned from the core build (must be handled here):
    - sherpa-onnx does NOT split continuous speech at max_speech_duration. When a SpeechOngoing's audio
      reaches cfg.audio.max_utterance_s, call segmenter.flush() to force a SpeechEnd for the audio so far
      (the VAD keeps running; the rest of the monologue becomes the next segment). Verify with a test.
    - VadSegmenter.process() requires exactly 512 samples; validate before calling (assert may be stripped).
    - Time alignment: frames may stop arriving (loopback idles when nothing plays and keepalive is off,
      or a device change). If no frame arrived for > 1 s and the segmenter is not in speech, call
      segmenter.set_time_origin(clock()) before feeding the next frame so timestamps track the session clock.
    - Partials: only the newest SpeechOngoing per source is kept in the ASR queue; finals have priority.
      A partial transcription result is published as PartialTranscriptEvent only if it is still the newest.
    - The ASR worker calls SpeakerClusterer.assign() (CPU, ~10 ms) after the final transcription; rename()
      may be called from another thread (GIL-safe by design)."""
    def __init__(self, cfg: Config, bus: EventBus, transcriber: Transcriber,
                 clusterer: SpeakerClusterer | None, clock: Callable[[], float],
                 next_id: Callable[[], int]) -> None: ...
    def start(self) -> None: ...        # starts captures; ASR must already be loaded
    def stop(self) -> None: ...
    def set_mic_muted(self, muted: bool) -> None: ...
    @property
    def mic_muted(self) -> bool: ...
    def rename_speaker(self, speaker_id: str, label: str) -> None: ...
```

### transcript.py
```python
class Transcript:
    """Thread-safe ordered store of TranscriptSegment. Labels of past segments are rewritten on rename."""
    def add(self, seg: TranscriptSegment) -> None: ...
    def segments(self, since_id: int | None = None) -> list[TranscriptSegment]: ...
    def last(self, source: Source | None = None, n: int = 1) -> list[TranscriptSegment]: ...
    def rename_speaker(self, speaker_id: str, label: str) -> None: ...
    def window_text(self, max_chars: int, *, until_id: int | None = None) -> str: ...  # oldest→newest "[mm:ss] Label: text" lines, trimmed from the front on whole lines
    def text_since(self, since_id: int | None) -> str: ...
    def word_count(self, since_id: int | None = None) -> int: ...
    def last_id(self) -> int | None: ...
    def to_markdown(self, title: str) -> str: ...
    def to_jsonl(self) -> str: ...
    def clear(self) -> None: ...
```

### llm/client.py
```python
@dataclass(frozen=True)
class ModelInfo: id: str; loaded: bool | None; arch: str = ""; quant: str = ""; context_length: int | None = None
@dataclass(frozen=True)
class Health: reachable: bool; model_listed: bool; model_loaded: bool | None; detail: str = ""
@dataclass(frozen=True)
class StreamStats: ttft_s: float; tokens: int; elapsed_s: float; tok_per_s: float; cancelled: bool

class LMStudioClient:
    """httpx-based client for LM Studio. Sync API, used from worker threads.
    - list_models(): try GET /api/v0/models (has "state": "loaded"/"not-loaded", "quantization",
      "max_context_length", "arch") and fall back to GET /v1/models (loaded=None).
    - health(): reachable + whether cfg.model is listed/loaded.
    - ensure_loaded(): if not loaded and auto_load: send a 1-token /v1/completions request (LM Studio JIT-loads),
      return True on success.
    - stream_completion(prompt, ..., cancel): POST /v1/completions stream=True; yields text deltas as they arrive;
      stops early when `cancel.is_set()` (closes the response). Sets `self.last_stats` (StreamStats).
    - stream_chat(messages, ..., cancel): POST /v1/chat/completions stream=True; yields ChatDelta(content, reasoning)
      where reasoning comes from delta.reasoning_content / delta.reasoning if present; if content contains
      <think>…</think> inline, split it so reasoning never leaks into content.
    - Errors raise LLMError(str) with a human-readable message (connection refused → "LM Studio server not reachable
      at <url>. Start LM Studio and enable the server (Developer tab) or run `lms server start`.")."""
    def __init__(self, cfg: LLMConfig) -> None: ...
    def list_models(self) -> list[ModelInfo]: ...
    def health(self) -> Health: ...
    def ensure_loaded(self) -> bool: ...
    def stream_completion(self, prompt: str, *, max_tokens: int, temperature: float | None = None,
                          top_p: float | None = None, stop: list[str] | None = None,
                          cancel: threading.Event | None = None) -> Iterator[str]: ...
    def stream_chat(self, messages: list[dict[str, str]], *, max_tokens: int, temperature: float | None = None,
                    top_p: float | None = None, cancel: threading.Event | None = None) -> Iterator[ChatDelta]: ...
    last_stats: StreamStats | None
```

### llm/prompts.py
```python
IM_START = "<|im_start|>"; IM_END = "<|im_end|>"
def render_chatml(system: str, user: str, assistant_prefix: str = "") -> str
    # f"{IM_START}system\n{system}{IM_END}\n{IM_START}user\n{user}{IM_END}\n{IM_START}assistant\n{assistant_prefix}"
NO_THINK_PREFIX = "<think>\n\n</think>\n\n"
STOP = ["<|im_end|>", "<|im_start|>"]
def system_assist(persona: str, style: str) -> str        # concise co-pilot; bullets/short/detailed rules; "answer as if you were the user"; never mention being an AI; no preamble
def system_ask(persona: str) -> str
def system_notes() -> str                                  # markdown with fixed headings: ## Summary, ## Key points, ## Decisions, ## Action items, ## Open questions
def user_assist(transcript_window: str, question: str) -> str
def user_ask(transcript_window: str, question: str) -> str
def user_notes(previous_notes: str, new_transcript: str) -> str   # "update the notes, keep everything still true, merge"
def approx_tokens(text: str) -> int                         # len(text)//4 + 1
```
Prompt layout for cache hits: system prompt (stable) → transcript window (grows at the end only) → the short instruction last.

### llm/questions.py
```python
def is_question(text: str, min_words: int = 3) -> bool
    # True if ends with "?" OR starts (after stripping fillers like "so", "okay", "um", "and") with an interrogative
    # (what/why/how/when/where/who/which/can/could/would/should/do/does/did/is/are/have/has/tell me/walk me/explain/describe)
    # AND has ≥ min_words words. Rhetorical/backchannel ("right?", "you know?") with ≤ 2 words → False.
def extract_question(segments_text: list[str]) -> str
    # joins the last consecutive question-bearing sentences; returns the most recent question sentence(s), max ~400 chars
```

### llm/assistant.py
```python
class Assistant:
    """Runs on its own threads. Subscribes to the bus for FinalTranscriptEvent.
    Auto mode: on a SYSTEM final with is_question, start a coalesce timer (cfg.assist.coalesce_ms); when it fires,
    build the question from the last ≤3 SYSTEM segments within 15 s and run an assist job (kind="auto").
    A new assist job cancels an in-flight assist job (not a notes job). Jobs publish AssistStarted/Delta/Done/Error.
    answer_last(): kind="manual", question = extract_question(last ≤3 SYSTEM segments) or the last SYSTEM segment text.
    ask(text): kind="ask", larger max_tokens.
    generate_notes(force): runs on the notes thread; skip if not force and new words < cfg.notes.min_new_words;
    auto timer every cfg.notes.auto_interval_s (0 disables). Keeps `self.notes_markdown`.
    Thinking: cfg.llm.thinking == "off" → render_chatml(..., NO_THINK_PREFIX) + stream_completion;
    "on" → stream_chat, hide reasoning, stream only content.
    Also publishes ComponentStatusEvent("llm", ...) after a startup health check and on errors.
    Integration facts learned from the core build:
    - questions.extract_question() only inspects the LAST element of the list it is given, so pass
      [" ".join(texts of the recent SYSTEM segments)] (one joined string) to get cross-segment context.
    - LMStudioClient.last_stats is thread-local (read it on the same worker thread that streamed).
    - Client temperature/top_p default to the config when None is passed; pass max_tokens explicitly.
    - Transcript.window_text(max_chars) trims whole lines from the front; build the window once per job
      with max_chars = cfg.llm.context_tokens * 4 and never reorder earlier lines (KV prefix cache)."""
    def __init__(self, cfg: Config, bus: EventBus, client: LMStudioClient, transcript: Transcript,
                 clock: Callable[[], float]) -> None: ...
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def answer_last(self) -> None: ...
    def ask(self, text: str) -> None: ...
    def generate_notes(self, force: bool = True) -> None: ...
    def cancel(self) -> None: ...
    def set_auto(self, enabled: bool) -> None: ...
    @property
    def auto_enabled(self) -> bool: ...
    @property
    def notes_markdown(self) -> str: ...
    @property
    def answers(self) -> list[tuple[float, str, str]]: ...   # (t, question, answer) history for export
```

### session.py
```python
class SessionRecorder:
    """Creates <sessions_dir>/<YYYY-MM-DD_HH-MM-SS>/ on start; subscribes to the bus; autosaves every
    cfg.paths.autosave_interval_s and on stop: transcript.md, transcript.jsonl, notes.md, answers.md, session.json
    (config snapshot + speaker labels). export() saves immediately and returns the folder."""
    def __init__(self, cfg: Config, bus: EventBus, transcript: Transcript, assistant: Assistant) -> None: ...
    def start(self) -> Path: ...
    def stop(self) -> None: ...
    def export(self) -> Path: ...
    @property
    def folder(self) -> Path: ...
```

### ui/win32.py (ctypes only, no pywin32)
```python
WDA_NONE=0x0; WDA_MONITOR=0x1; WDA_EXCLUDEFROMCAPTURE=0x11
def set_capture_exclusion(hwnd: int, exclude: bool) -> bool      # SetWindowDisplayAffinity; returns success
def get_display_affinity(hwnd: int) -> int | None                 # GetWindowDisplayAffinity
def set_tool_window(hwnd: int, enabled: bool) -> None             # WS_EX_TOOLWINDOW on, WS_EX_APPWINDOW off (hides from Alt-Tab/taskbar)
def set_click_through(hwnd: int, enabled: bool) -> None           # WS_EX_TRANSPARENT | WS_EX_LAYERED
def set_topmost(hwnd: int) -> None                                 # SetWindowPos HWND_TOPMOST, NOMOVE|NOSIZE|NOACTIVATE
def set_no_activate(hwnd: int, enabled: bool) -> None             # WS_EX_NOACTIVATE
def is_layered(hwnd: int) -> bool                                    # WS_EX_LAYERED (what keeps the legacy DC reads black)
def ensure_layered(hwnd: int) -> bool                                # add WS_EX_LAYERED if missing (Qt only layers translucent windows)
def capture_exclusion_verified(hwnd: int) -> bool                 # strict read: affinity IS WDA_EXCLUDEFROMCAPTURE *and* the window is still WS_EX_LAYERED
def apply_capture_exclusion(hwnd: int) -> bool                    # layer + affinity + peek, then verify; False = the caller must fail closed

# exposure probes (advisory: [] / None / False mean "nothing detected", never "safe")
def is_remote_session() -> bool                                   # SM_REMOTESESSION + WTSClientProtocolType
def capture_agents() -> list[str]                                 # running VNC-class products (Toolhelp name list)
def exposure_reason(agents=None, hwnd=0) -> str | None            # combines the three probes; hwnd adds the monitor check
def monitor_display_device(hwnd: int) -> dict | None               # monitor showing hwnd -> its display adapter
def relayed_display_reason(hwnd: int) -> str | None                # "relayed display (<adapter>)" or None (physical/unknown)
def is_relayed_display(hwnd: int) -> bool
def relayed_adapters() -> list[str]                                # attached relayed/virtual adapters (doctor run, no hwnd)

DISPLAY_DEVICE_ATTACHED_TO_DESKTOP=0x1; DISPLAY_DEVICE_PRIMARY_DEVICE=0x4
DISPLAY_DEVICE_MIRRORING_DRIVER=0x8; DISPLAY_DEVICE_RDPUDD=0x01000000; DISPLAY_DEVICE_REMOTE=0x04000000
RELAYED_ADAPTER_MARKERS = ("indirect display", "iddsample", "mmidd", "virtual display")
```
*Relayed display* = a display target whose pixels are not produced by DWM: an Indirect Display Driver
virtual monitor (`MonitorFromWindow` → `GetMonitorInfoW` → `EnumDisplayDevicesW` on the adapter, matched
by those `DISPLAY_DEVICE_*` bits or by the adapter description), a mirror driver, or the RDPUDD chained
display. Only `DISPLAY_DEVICE_ATTACHED_TO_DESKTOP` adapters count in `relayed_adapters()` — a console
session enumerates an *unattached* RDPUDD adapter that means nothing. This is content protection for the
window on that monitor; it is not concealment of the process (see "Non-goals").

`capture_exclusion_verified` is the strict read a *window* must pass before painting content: the affinity
has to read back as `WDA_EXCLUDEFROMCAPTURE` (an in-process caller can always reset it) **and** the window
has to still carry `WS_EX_LAYERED`, because the legacy `PrintWindow` / `BitBlt(GetWindowDC)` reads ignore
display affinity and return a non-layered window's pixels. `ui/overlay.py` re-checks it on the topmost tick
and covers its content with the fail-closed veil without it; `ui/ghost.py` re-checks it on every completion
and withholds the ghost text without it; `ui/region_picker.py` refuses to appear without it. Use
`apply_capture_exclusion` when a window needs both the apply and the verdict in one call — it calls
`ensure_layered` first, which is what supplies the style for the ordinary windows Qt never layers (dialogs,
menus, combo popups, tooltips: only a translucent `WA_TranslucentBackground` window is layered by Qt).

Ordinary pop-ups are layered, never hidden: the app-wide `_CaptureExclusionFilter` hides a pop-up only when
the *affinity itself* could not be verified, because hiding a dialog the user needs (the settings dialog,
a menu) to protect no session content makes the app unconfigurable — the failure was reported once as
"*I can no longer see the settings at all*" when layering was made part of that decision. A window that
refuses the style keeps its affinity, logs a warning, and stays visible.

Use `SetWindowLongPtrW`/`GetWindowLongPtrW` with `ctypes.windll.user32`, `GWL_EXSTYLE=-20`, and `ctypes.get_last_error()`
via `use_last_error=True` for diagnostics.

### ui/hotkeys.py
```python
class GlobalHotkeys(QAbstractNativeEventFilter):
    """RegisterHotKey-based global hotkeys that work while a fullscreen meeting app has focus.
    parse("ctrl+alt+space") → (modifiers, vk). Registers in `install(app)`, handles WM_HOTKEY (0x0312) in
    nativeEventFilter and invokes the callback on the Qt main thread. Reports which bindings failed
    (already taken) via `failed: list[str]`. MOD_NOREPEAT (0x4000) is set."""
    def __init__(self, bindings: dict[str, Callable[[], None]]) -> None: ...
    def install(self, app: QCoreApplication) -> None: ...
    def uninstall(self) -> None: ...
    failed: list[str]
```

### ui/overlay.py
```python
class OverlayWindow(QWidget):
    """Frameless, always-on-top, translucent 'glass' panel. Window flags:
    Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool ; WA_TranslucentBackground.
    Capture hardening ([ui].hide_from_capture): prime win32.set_tool_window + set_capture_exclusion +
    DWMWA_EXCLUDED_FROM_PEEK *before* the first show (winId() creates the HWND early; in the gap before
    the affinity lands a foreign PrintWindow caller gets the content, after it gets black), verify with
    get_display_affinity → status bar shows "🕶 hidden from capture" or "⚠ visible to capture (reason)";
    re-apply in showEvent (Qt may recreate the native window); a QApplication event filter layers
    (`win32.ensure_layered` — Qt only layers a translucent window) and applies+verifies the affinity to
    pop-ups/dialogs (each is its own HWND, affinity does not propagate), hiding one only while the affinity
    itself is unverified — the panel itself is exempt, so it can still veil its content and explain in the
    status bar, and a hidden pop-up is toasted so it never looks like a broken app; the topmost tick
    re-asserts it and reports in-process tampering (anything running inside the process can always set
    WDA_NONE). The window must stay WS_EX_LAYERED: PrintWindow without PW_RENDERFULLCONTENT and
    BitBlt(GetWindowDC) read the window's own surface — display affinity does not cover that read, and
    no in-process hook sees WM_PRINT/WM_PRINTCLIENT (measured, build 26200) — but both return black for
    layered windows. Fail-closed content veil (_set_content_veiled): while the affinity is *not verified*
    (`_capture_ok` False — the call was rejected, build < 19041, or a tampering caller cleared it) an
    opaque label covers the tab strip, the stacked views and the AskBox instead of painting transcript /
    answers / notes into a window that may be capturable; the topmost tick keeps retrying
    (_CAPTURE_RETRY_TICKS, 1.5 s each) and lifts the veil as soon as the affinity verifies again. The veil
    also takes down ghost-text completions, and focus_ask() refuses with a toast while it is up, because
    the ask box is under it. Exposure guard ([ui].hide_when_exposed, 2.5 s tick): while a remote session
    (SM_REMOTESESSION / WTS client protocol), a known VNC/remote-control process (Toolhelp name list), or
    a relayed display under the panel itself (win32.relayed_display_reason(hwnd): an IDD virtual monitor,
    a mirror driver, the RDPUDD chained display) is detected, hide the whole panel — those capture paths
    read the display pipeline and ignore display affinity; toggle/focus attempts stay refused while
    exposed and say so via toast.
    Layout (top→bottom): title bar (drag handle, auto-assist toggle, mic mute, opacity slider, hide button) /
    tab strip [Live | Assist | Notes] / content / AskBox (QLineEdit + Send) / StatusBar (levels, ASR, LLM, model, capture).
    Live tab: transcript list with colored speaker labels (Me = accent color; each Speaker N a distinct hue),
    timestamps, partial line in italics at the bottom, auto-scroll unless user scrolled up, double-click a speaker
    label to rename. Detected questions get a subtle highlight and a "Answer" button.
    Assist tab: current streaming answer (markdown-lite: bullets/bold rendered) with the question above it, history below.
    Notes tab: rendered markdown notes + "Regenerate" button.
    Consumes events from `bus.subscribe()` via a QTimer(30 ms) using bus.drain(). Never blocks the UI thread.
    Resizable from every edge and corner (invisible 6 px `_EdgeGrip` bands) plus a bottom-right size grip;
    draggable by the title bar; remembers geometry in cfg.ui (saved on quit). The title strip is kept on a
    screen (`_on_screen_pos`: drag, nudge, restore, and a 1.5 s tick), restored size is clamped to the screen,
    and the collapsible history lists live in height-capped scroll areas so content can never grow the window.
    Right-click anywhere opens the "..." menu (Quit); text widgets keep Copy/Select-all plus Hide/Quit.
    Public slots used by hotkeys/engine: toggle_visible(), focus_ask(), nudge(dx, dy), set_click_through(bool),
    set_opacity(float), show_toast(text)."""
    def __init__(self, cfg: Config, bus: EventBus, actions: "EngineActions") -> None: ...
```
`EngineActions` (in app.py) is a small Protocol the UI calls: `answer_last()`, `ask(text)`, `generate_notes()`,
`set_auto(bool)`, `set_mic_muted(bool)`, `rename_speaker(id, label)`, `export_session() -> Path`,
`new_session()`, `set_model(model_id)`, `quit()`.

### app.py
```python
class Engine(EngineActions):
    """Composition root. start(): logging → cuda_env.setup_cuda_dlls() → ensure_models → Transcriber.load (in a
    background thread, publishing ComponentStatusEvent asr loading/ready) → SpeakerClusterer → SpeechPipeline.start
    → LMStudioClient/Assistant.start (health check in background) → SessionRecorder.start.
    stop(): reverse order, saving the session. All heavy work off the Qt thread."""
def main(argv: list[str] | None = None) -> int
    # parses --config, --doctor, --no-ui (headless: prints transcript/answers to stdout, for testing), --log-level
```

## Screen reading (added 2026-09-20)

Live screen reading: capture the active window / the primary monitor / a user-drawn region, OCR it
locally, and use the text (a) as extra context for answers, (b) for proactive suggestions and
completions, (c) to auto-answer questions that appear on screen.

Measured on this machine: `mss` grabs a 2560×1600 monitor in 45 ms and a window in 15 ms. Windows
OCR (`winrt-Windows.Media.Ocr`, PyWinRT 3.2.1) reads a full screen in **120–220 ms** (44 lines,
slightly noisy). RapidOCR (`rapidocr` 3.9.2, PP-OCRv6 ONNX, CPU) takes **1.3–1.7 s** for the same
screen (53 lines, more accurate); half-resolution does not make it faster. Monitors here:
laptop 2560×1600 at (-2560, 848) *not primary*, Odyssey 5120×1440 at (0, 0) *primary* — always pick
the monitor with `is_primary`, never `monitors[1]`. Our overlay windows are capture-excluded, so
they never appear in grabs. Windows OCR needs the language pack of the user profile (English
present). `OcrEngine.max_image_dimension` is 10000. RapidOCR logs to stdout in colour → silence
its logger. Windows OCR usage that works:

```python
from winrt.windows.media.ocr import OcrEngine
from winrt.windows.graphics.imaging import SoftwareBitmap, BitmapPixelFormat
from winrt.windows.storage.streams import DataWriter
dw = DataWriter(); dw.write_bytes(bgra_bytes); ib = dw.detach_buffer()
bmp = SoftwareBitmap.create_copy_from_buffer(ib, BitmapPixelFormat.BGRA8, w, h)
eng = OcrEngine.try_create_from_user_profile_languages()      # None -> unavailable
res = asyncio.run(eng.recognize_async(bmp))                    # res.lines[i].text, .words[j].bounding_rect (x,y,width,height)
```
RapidOCR usage: `from rapidocr import RapidOCR; eng = RapidOCR(); out = eng(bgr_uint8)`; `out.txts`
(tuple of str or None), `out.boxes` (N×4×2 float points), `out.scores`.

Foreground window rect: `DwmGetWindowAttribute(hwnd, DWMWA_EXTENDED_FRAME_BOUNDS=9, byref(RECT), sizeof(RECT))`
(exact visible bounds; `GetWindowRect` includes invisible borders). Skip windows of our own process
(`GetWindowThreadProcessId`) and Progman/WorkerW (the desktop).

### Config, events, actions (done)
`ScreenConfig` in config.py (`mode`, `interval_s`, `ocr_engine`, `monitor`, `region`, `use_in_answers`,
`auto_suggest`, `suggest_min_interval_s`, `answer_screen_questions`, `max_text_chars`, `save_text`).
bus.py: `ScreenTextEvent(mode, source_name, rect, text, lines, changed, ocr_ms, captured_at)`,
`ScreenStatusEvent(state: active|paused|off|error, detail)`; `AssistKind` gained `"screen"` and `"suggest"`.
actions.py: `set_screen_mode`, `set_screen_region`, `capture_screen_now`, `suggest_now`, `set_auto_suggest`.
prompts.py: `user_assist(..., screen_text="")`, `user_ask(..., screen_text="")` (the `<screen>` block sits
between the transcript and the question), `system_suggest(persona)`, `user_suggest(transcript_window, screen_text, focus="")`.
Hotkeys: `select_region` (ctrl+alt+r), `suggest_now` (ctrl+alt+s), `toggle_screen` (ctrl+alt+v).

### screen/foreground.py
```python
@dataclass(frozen=True)
class WindowInfo: hwnd: int; title: str; process_name: str; rect: tuple[int, int, int, int]  # x, y, w, h screen px
def foreground_window() -> WindowInfo | None   # None for our own windows, the desktop/taskbar, minimized or empty rects
def monitor_rects() -> list[tuple[int, int, int, int]]   # via mss; index 0 = primary
def virtual_screen_rect() -> tuple[int, int, int, int]
```
### screen/capture.py
```python
@dataclass(frozen=True)
class Frame: image: np.ndarray  # H×W×3 uint8 BGR, C-contiguous
             rect: tuple[int, int, int, int]; source_name: str; mode: str; ts: float
class ScreenGrabber:
    """One mss.MSS per instance (mss is not thread-safe across threads; the reader owns one)."""
    def grab(self, mode: str, *, region: tuple[int, int, int, int] | None = None, monitor: int = 0) -> Frame | None
    # auto: foreground window rect (fallback: primary monitor); monitor: monitors()[monitor] (0 = primary);
    # region: clipped to the virtual screen; returns None when the rect is empty/invalid
    def monitors(self) -> list[tuple[int, int, int, int]]
def signature(image: np.ndarray) -> np.ndarray       # 32×18 float32 grayscale thumbnail (area-averaged)
def is_changed(a: np.ndarray | None, b: np.ndarray, threshold: float = 4.0) -> bool  # mean abs diff > threshold (0..255 scale)
```
### screen/ocr.py
```python
@dataclass(frozen=True)
class OcrLine: text: str; box: tuple[int, int, int, int]; confidence: float   # box relative to the frame
@dataclass(frozen=True)
class OcrResult: lines: tuple[OcrLine, ...]; text: str; elapsed_ms: float; engine: str
class OcrUnavailable(RuntimeError): ...
class OcrEngine(Protocol):
    name: str
    def recognize(self, image_bgr: np.ndarray) -> OcrResult: ...
class WindowsOcr: ...      # PyWinRT; a private asyncio event loop reused per instance; raises OcrUnavailable if no engine
class RapidOcr: ...        # rapidocr; downscale so the longer side <= 2200 px, scale boxes back; logger silenced
def create_engine(name: str = "auto") -> OcrEngine     # auto -> WindowsOcr if it can be created, else RapidOcr
def lines_to_text(lines: Sequence[OcrLine]) -> str      # reading order: bucket by row (y-centre within 0.6×median height), sort rows top→bottom and items left→right; join items with two spaces, rows with "\n"; collapse repeated whitespace; drop empty lines
def new_lines(previous: Sequence[str], current: Sequence[str]) -> list[str]   # lines in current not in previous (normalised: casefold, collapse spaces)
```
### screen/reader.py
```python
class ScreenReader:
    """Worker thread "screen-reader": every cfg.interval_s (or immediately on capture_now()) grab →
    signature → skip if unchanged (unless forced) → OCR → lines_to_text → truncate to cfg.max_text_chars
    (keep the beginning) → publish ScreenTextEvent(changed = text != previous text). Publishes
    ScreenStatusEvent on mode/source/pause changes and on errors (back off 5 s after an error).
    mode "off" idles (publishes ScreenStatusEvent("off") once). pause()/resume() keep the thread alive.
    OCR engine is created lazily on the worker thread (creation can take ~0.4 s and may raise OcrUnavailable
    → status error + fall back to the other engine). If cfg.save_text and a snapshot dir is set, write
    <dir>/screen/<HHMMSS>.txt for each changed capture (never images)."""
    def __init__(self, cfg: ScreenConfig, bus: EventBus, clock: Callable[[], float], *,
                 grabber_factory: Callable[[], ScreenGrabber] | None = None,
                 engine_factory: Callable[[str], OcrEngine] | None = None) -> None: ...
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def set_mode(self, mode: str) -> None: ...                    # off | auto | monitor | region (validated)
    def set_region(self, x: int, y: int, w: int, h: int) -> None: ...   # also switches to region mode; writes cfg.region
    def set_monitor(self, index: int) -> None: ...
    def capture_now(self) -> None: ...                            # wake the worker; forced OCR; event has changed=False if same
    def pause(self) -> None: ...
    def resume(self) -> None: ...
    def set_snapshot_dir(self, path: Path | None) -> None: ...
    @property
    def latest_text(self) -> str: ...
    @property
    def latest_lines(self) -> tuple[str, ...]: ...
    @property
    def latest_event(self) -> ScreenTextEvent | None: ...
    @property
    def mode(self) -> str: ...
    @property
    def paused(self) -> bool: ...
```
### Assistant extension (llm/assistant.py)
- `set_screen_provider(fn: Callable[[], str] | None)`: when set and `cfg.screen.use_in_answers`, every
  assist/ask job passes `screen_text=fn()` (truncated to `cfg.screen.max_text_chars`) to user_assist/user_ask.
- `suggest_now(focus: str = "")`: job kind `"suggest"` using system_suggest/user_suggest with
  `max_tokens=cfg.llm.assist_max_tokens`; publishes the usual Assist* events with kind "suggest".
  A suggest job never cancels a running auto/manual/ask/screen job (it is skipped if one is running,
  and re-queued once); a new auto/manual/ask job cancels a running suggest job.
- Auto-suggest: subscribe to ScreenTextEvent; when `changed`, `cfg.screen.auto_suggest`, at least
  `cfg.screen.suggest_min_interval_s` since the last suggest, and the line set differs from the last
  suggested one (Jaccard < 0.6) → `suggest_now()`.
- Screen questions: on a changed ScreenTextEvent, take `new_lines(prev, cur)`, keep those with
  `is_question(line, min_words=6)`, and if `cfg.screen.answer_screen_questions` and the normalised
  question was not answered before (bounded set of 50) → job kind `"screen"` with that question and
  the screen text; at most one every 15 s.
### UI extension (ui/overlay.py, ui/widgets.py, new ui/region_picker.py)
- New **Screen** tab: mode buttons (Off / Active window / Monitor / Region…), "Select region" (opens the
  picker), "Suggest now", toggles (use in answers, auto-suggest, answer screen questions),
  a read-only text view of the latest screen text (monospace-ish, dim), and a suggestions panel
  (streaming, same rendering as answers, history of the last 10). Status bar gets "Screen: auto · Zoom
  Meeting · 140 ms" / "Screen: paused".
- `RegionPicker(QWidget)`: one translucent, capture-excluded, always-on-top window covering the whole
  virtual screen (`QRect` union of all screens), crosshair cursor, rubber-band drag, Esc cancels,
  Enter/double-click accepts; result in screen pixels (multiply by devicePixelRatio for non-100 % DPI);
  returns via a signal `selected(x, y, w, h)`. **Fails closed**: `_ensure_capture_exclusion()` applies and
  verifies the affinity *before* the first paint and, while it cannot be confirmed, `begin()` does not show
  the window at all — it emits `failed(reason)` instead (the overlay toasts it and reverts the mode) —
  because a full-desktop window in a screen share would advertise the app and outline what is about to be
  read. `showEvent` re-verifies and takes the window back down if the affinity was lost.
- Answers of kind "screen" go to the Assist tab (question shown as "On screen: …"); kind "suggest"
  goes to the Screen tab's suggestions panel.
- Hotkeys: select_region, suggest_now, toggle_screen.

### Autonomous mode + UX changes (added 2026-09-21)
- **`AssistKind` gained `"solve"`.** `AssistConfig.autonomous` (default True): when the screen changes,
  `_maybe_auto_suggest` enqueues a `"solve"` side-lane job (non-cancelling, like `"suggest"`) instead
  of `"suggest"`. `Assistant.solve(focus="")` and `Engine.solve(focus)` / `actions.solve(focus)` expose
  it; deferred as pending kind `"solve"` before the LLM is ready. Prompts: `system_solve(persona)`,
  `user_solve(transcript_window, screen_text, focus="")` — instruct the model to fully solve the on-screen
  task (code in a fenced block + complexity; a direct answer; or exact text to type) and NOT ask how to
  proceed. Budget `cfg.llm.solve_max_tokens` (default 700). A `"solve"` result is stored as an *answer*
  (not a suggestion) and routed to the Assist tab (question label "Solving: …").
- **Choices (no-typing forks).** When a task genuinely forks the model emits lines
  `prompts.CHOICE_PREFIX + " <label>"` (`[[CHOICE]]`). `prompts.split_choices(text) -> (body, labels)`
  (dedup, case-insensitive). `AssistView` renders up to 4 as buttons under the answer; clicking one calls
  `actions.solve(label)`.
- **Read / Revert / Pause / Reset bottom controls** (`OverlayWindow._build_control_row`, always visible under
  the ask box, in that order). Read → `actions.capture_screen_now()` (a forced capture + OCR; it replaced the
  Screen tab's "Read now" button). Revert → `AssistView.revert()` (steps back through answer history). Pause (checkable) →
  `actions.set_paused(bool)` → `Engine.set_paused` pauses the screen reader + `Assistant.set_paused`
  (gates auto-answer, screen questions, auto-solve/suggest, and cancels the running job). Reset →
  `actions.new_session()` (+ `Assistant.reset()` clears answers/suggestions/screen-dedup/notes) and the UI
  clears the Assist/Live/Notes/Screen panels (`TranscriptView.clear()`, `AssistView.clear()`).
- **Screen OCR readout hidden by default** (`ScreenConfig.show_text=False`): `ScreenView` folds it behind a
  collapsible "Screen text (N lines)" toggle; `set_screen_text` only paints while the toggle is checked.
- **Windowless launch.** `run.vbs` runs `.venv\Scripts\pythonw.exe -m exocortex` hidden (no console, no
  taskbar entry for the launcher); `run.bat` launches via `start … pythonw.exe` and exits. `logging_setup`
  skips the console StreamHandler when `sys.stderr`/`sys.stdout` are None (pythonw), file logging only.

### Completion mode (added 2026-09-21)
Inline ghost-text completions at the caret of whatever app has focus; the overlay collapses to a slim bar.

- **Sensing: `screen/caret.py`** — `CaretTracker(prefix_chars, suffix_chars)`; `attach()` on the polling
  thread (comtypes `CoInitialize`, `CUIAutomation8`, MSAA typelib); `probe() -> CaretInfo | None`
  (`hwnd, pid, app, field_name, field_hint, control_type, x, y, h, dpi, font_pt, font_name, text_before,
  text_after, source`; props `key`, `font_px`, `current_line`); `last_reject` says why a probe found nothing.
  Text: UIA `TextPattern2.GetCaretRange` or `TextPattern.GetSelection` (collapsed to its End), ranges
  **clamped to `DocumentRange`** (Chromium ranges span the whole page), font from the range's
  `FontSize`/`FontName` attributes. Position tiers: MSAA `OBJID_CARET.accLocation` (works for Chromium,
  which has no Win32 caret) → `GetGUIThreadInfo.rcCaret` → UIA character rects; each checked against the
  element's bounding rect. Skips: password fields, our own PID, non-text controls, read-only values.
  `detect_placeholder()` recognises an empty field showing its placeholder (caret at offset 0, or Chromium's
  quirk: UIA selection at the END of the placeholder while the MSAA caret sits at the field's left edge —
  detected via the x gap > 2 chars). Verified live: VS Code message input (Edit, TextPattern, font 10pt),
  Windows 11 Notepad (Document, TextPattern2), probe 4–15 ms. Chromium's first query returns its top-level
  pane (accessibility switches on), the next returns the real element.
- **Controller: `completion.py`** — `Completer(cfg, bus, backend, tracker, clock)` thread "completer",
  ticks every `poll_ms` (150): probe → state machine. `consume_suggestion(suggestion, typed)` is
  whitespace-tolerant (auto-indent, doubled/skipped spaces): returns the remaining ghost text, "" when typed
  out, `None` on deviation. Shown suggestion + matching typing → shrink and follow the caret, no model
  call; backspacing into the anchor (≤ 40 chars) grows it back; deviation → hide, then a new request only
  after `debounce_ms` (1200) of quiet AND `min_interval_ms` (800) since the last request, never twice for
  the same prefix, never for an empty field without a placeholder. In-flight requests stream to the ghost
  and are cancelled when the user moves away (`backend.cancel_job`). Field change → clear. `set_paused`,
  `reset`, `requests`/`shown_count` counters. Publishes `CompletionEvent(visible, text, x, y, line_h,
  font_px, font_name, mono, current_line, app)` (physical px) only on change, and
  `ComponentStatusEvent("completion", ready|error, detail)`.
- **Model: `llm/assistant.py`** — `complete(prefix, suffix, field) -> job_id` (kind `"complete"`, side lane,
  latest wins), `cancel_job(id)`, `set_auto_suppressed(bool)` (completion mode suppresses auto-answer /
  screen questions / auto-solve; manual ask/answer still work). Prompt (`prompts.system_complete(persona,
  hint)`, `user_complete(context_blocks, field, suffix)`): context stable-first (transcript, answers, notes,
  screen — each gated by `cfg.completion.use_*`), and **the typed prefix is the start of the assistant turn**
  (`render_chatml(..., NO_THINK_PREFIX + prefix)`) so the model continues it verbatim and the prompt only
  grows at its end (prefix cache: first request TTFT ~1.8 s, follow-ups ~0.2 s). `max_tokens` 48,
  temperature 0.2, `repeat_penalty` 1.1 (per-request `extra_body`, new `stream_completion` kwarg).
  `prompts.clean_completion(prefix, text, max_lines)` strips fences, a repeated tail of the prefix (echo),
  degenerate repetition (`trim_repetition`: bare "2. 3. 4." runs, tripled words, repeated phrases) and cuts
  to `max_lines`. Results are ephemeral (never in `answers`/`suggestions`).
- **Rendering: `ui/ghost.py`** — `GhostText(cfg)`: frameless/tool/topmost/no-focus/click-through window,
  capture-excluded and *verified*: `_ensure_capture_exclusion()` runs before every paint (and before the
  first `show()`, so the affinity is on the HWND before anything is drawn); while the exclusion cannot be
  verified — the call was refused, or the window lost `WS_EX_LAYERED` — `show_completion` withholds the
  text and takes down a completion already on screen instead of painting it into a window a capture path
  might see (the ghost's own fail-closed rule, matching the panel's veil), and re-applies the affinity on
  the next completion. `show_completion(ev)` sizes the font to `font_px` (fallback `line_h / 1.3`, capped at
  95 % of `line_h`) divided by the monitor DPI (`win32.dpi_for_point`), lays lines out (continuations
  offset by the measured width of `current_line`), and places the window in physical pixels via
  `win32.set_window_rect_physical` (SetWindowPos, no Qt logical mapping). Verified visually on Notepad:
  prose and multi-line indented code line up with the caret and the text grid.
- **UI: `ui/overlay.py`** — `completion_btn` ("⌨", left of ⚡, `Ctrl+Alt+K`) ↔ `_set_completion_mode`:
  hides title bar/tabs/stack/ask box/control row/status and shows `slim` (`_TitleBar` drag handle):
  `Full`, `Pause/Resume` (mirrors the main Pause), `Context ▾` (checkable sources + `Hint…`/`Clear hint`
  → `actions.set_completion_context`), `Reset`, status dot/label, hide, `⋯` (main menu). Window shrinks to
  ~42 px (`_SLIM_MIN_WIDTH` 300, max 460) at the same right edge; `_normal_geometry` restored on exit and
  is what `_save_geometry` persists. `CompletionEvent` → ghost (only in completion mode and never while
  the exposure guard hides the overlay); kind `"complete"` Assist events never touch the tabs;
  `CompletionModeEvent` syncs the button; `cfg.completion.enabled` starts in completion mode.
- **Engine/actions** — `set_completion_mode(enabled)` (creates `CaretTracker` + `Completer` lazily, starts /
  stops it, `assistant.set_auto_suppressed`, publishes `CompletionModeEvent`), `set_completion_context
  (sources, hint)`; `set_paused` and `new_session` propagate to the completer; `stop()` stops it first.
- **Config `[completion]`**: `enabled, poll_ms, debounce_ms, min_interval_ms, max_tokens, max_lines,
  prefix_chars, suffix_chars, min_prefix_chars, temperature, repeat_penalty, use_transcript, use_screen,
  use_answers, use_notes, hint, ghost_color, ghost_opacity, font_scale`; hotkey `toggle_completion`
  (ctrl+alt+k). Dependency: `comtypes` (pure Python COM). Doctor: "Completion mode (caret tracking)".

### Technical focus + blind editors (added 2026-09-21, second pass)
- **Relevance filter** `llm/questions.py`: `DEFAULT_IGNORE_PATTERNS` (polls/surveys/feedback/chrome, incl.
  "have you seen this in an interview"), `has_technical_cue()` (term list + code-token regex),
  `is_relevant(text, focus, ignore_patterns)`. Applied to on-screen questions only (`_maybe_answer_screen_question`);
  spoken questions are unfiltered. `AssistConfig.focus` ("technical" | "all"), `AssistConfig.ignore_patterns`.
- **Prompts**: `_FOCUS_RULE` added to `system_assist`, `system_suggest`, `system_solve`, and a note in
  `system_complete`; `SKIP_MARKER = "(nothing to do)"` + `is_skip()` — a solve/screen answer that is only the
  marker is dropped (Done published with `cancelled=True`, not stored).
- **No-cancel screen jobs**: `_submit_assist(kind, question, cancel_running=True)`; kind "screen" passes
  `cancel_running=False` so a running solve/answer/completion is never thrown away by an on-screen question.
- **Blind editors** (`CaretInfo.blind`): the tracker flags fields whose accessible name/help contains
  `BLIND_MARKERS` ("not accessible", "screen reader", ...) — Monaco's hidden textarea — and clears their text;
  a caret shorter than `MIN_CARET_H` (6 px) is reported as unknown height. Text ranges are now built from
  `DocumentRange` → caret and caret → `DocumentRange` end (Chromium's `MoveEndpointByUnit` is unreliable).
- **`screen/context.py`**: `ScreenSnapshot(lines: OcrLine[], rect, captured_at, source_name)` (kept by
  `ScreenReader.latest_snapshot` on every OCR); `context_at(snapshot, x, y, line_h, font_px)` → `CaretContext`
  (`above`, `current_left`, `current_full`, `below`, `pane_left`, `char_w`, `has_line`; `.prefix`, `.suffix`):
  caret line found by geometry, pane = lines within 60 chars left of it (excludes a description pane), gutter
  numbers dropped/stripped (merged ones flagged, margin anchored on unmerged lines), indentation =
  round((x - pane_left) / char_w), char_w = 0.6·font_px (or 0.46·line_h).
- **Completer blind mode**: `Completer(..., screen_source, capture_now)`; `_blind_text()` derives prefix/suffix
  from the latest snapshot at the caret, follows typing between frames by caret dx (one char per `char_w`,
  optimistic: assumes the shown suggestion is being typed), requests a forced capture (≤ 1/s) whenever the
  caret moves or the line changes, and re-syncs on each new frame; `reconcile_fuzzy()` tolerates OCR noise in
  the lines above (difflib ratio ≥ 0.8) while the caret line must extend the anchor; asked-prefix de-dup uses a
  whitespace/case-normalised key. Ghost text in blind mode is monospace and aligned from the rebuilt line.
- **Ghost font**: `line_h < 6` treated as unknown; font = reported `font_px` → `line_h/1.3` →
  `default_font_pt` (config) × DPI; the line-height cap applies only to a plausible line height.

### Explanations (added 2026-09-21, third pass)
Toggle `[explain].enabled` (💡 title-bar button left of ⌨, `Ctrl+Alt+E` = `hotkeys.toggle_explanations`,
"Explanations" checkable in the slim bar's Context menu). `ExplainConfig`: `enabled, walkthrough_max_tokens
(220), explain_max_tokens (420), typed_fraction (0.9)`. `AssistKind` gained `"explain"`.

- **The plan (`exocortex/plan.py`, pure).** With explanations on a `"solve"` answer has the shape
  `Walkthrough:` + one block (no blank lines) + blank line + fenced solution (comments per `[assist].comments`)
  + complexity line (`prompts.system_solve(persona, explain, comments)` → `_solve_shape`). `split_plan(text) -> PlanParts
  (walkthrough, solution, is_code, language, after, code_open)` parses it while streaming (an unclosed fence
  keeps only complete lines; no fence = prose: first block is the walkthrough, the rest the solution).
  `consume_suggestion(suggestion, typed, skip_comments=False)` moved here (re-exported by `completion.py`);
  with `skip_comments` a mismatch that lands on a comment (`#` unless a C preprocessor directive, `//`,
  `/*`, `<!--`, `-- `) skips to the end of that line: typing past a comment drops it, nothing regenerates.
  `align_plan(solution, prefix, hint) -> PlanCursor(line, rest, at_end, score) | None`: the caret line must
  be a whitespace-insensitive prefix (`squash`) of a solution line and the last ≤ 3 typed lines must
  anchor on the solution lines before it (`_match_back`: solution comments/blank lines skipped, typed
  comments ignored, `_similar` = exact | same 12-char head with `(`/`:` (annotated signatures) | difflib
  ≥ 0.85 for OCR noise); nothing typed → line 0; comment lines are candidates only while being typed
  (blank caret line → the comment shows first, then the code); ties prefer forward progress (`hint`).
  `plan_ghost(solution, cursor, max_lines)`; `typed_fraction(solution, screen_lines)` (exact / gutter-merged
  tail / fuzzy, code lines ≥ 4 chars, comments excluded); `containment(old, new)`; `PlanView(id, solution,
  is_code, done)` is the completer's snapshot.
- **Assistant.** Four `_JobSlot` lanes, dispatch priority main → side (suggest/solve) → explain → complete
  (`_assist_slots()`, `_next_assist_job`); `_CANCELS` = what a new job cancels if running (suggest/solve:
  suggest|solve|explain; explain: explain; complete: complete) — a completion never displaces or delays a
  queued solve (the plan) any more. `_Plan(id, screen_lines, created, text, done, explaining, explained,
  explanation)` under `_plan_lock`: `_begin_plan` at solve start (explanations on), `_plan_delta` per
  delta, `_settle_plan_state` on every exit (kept only when not cancelled/failed/skipped). Public:
  `plan() -> PlanView | None` (None when explanations are off), `plan_walkthrough`, `plan_explanation`,
  `plan_typed(plan_id)` (completer reached the last code line → `_request_explain`, once per plan, not
  while paused), `explain_now()` (manual, any time), `ensure_plan()` (solve unless the plan still matches
  the screen), `_maybe_explain_from_screen(lines)` on every changed `ScreenTextEvent` (plan done, is_code,
  `typed_fraction ≥ cfg.explain.typed_fraction`). `_maybe_auto_suggest`: with explanations on the solve is
  NOT suppressed in completion mode (it is the ghost text's source), and while a plan exists a screen change
  with `containment(plan.screen_lines, lines) ≥ PLAN_KEEP_FRACTION (0.4)` is the same task (typing the
  solution adds lines, it never removes the problem statement) → no re-solve. Prompts: `system_explain`,
  `user_explain(walkthrough, solution, screen_text)`, budget `explain_max_tokens`; solve budget
  `solve_max_tokens + walkthrough_max_tokens`; `system_complete(..., explain=True)` + a `("solution", …)`
  context block so model completions follow the plan. `"explain"` results are answers (question
  "Explanation"). `reset()` drops the plan.
- **Completer.** `_serve_plan(prefix, info)` runs before the model path when explanations are on: no plan →
  model path; plan without a solution yet and not done → cancel in-flight, hide, wait; aligned → ghost text
  from the plan (`plan_served` counter, `_plan_line` hint, `_plan_id`), in-flight model job cancelled;
  `at_end` on a done code plan → `backend.plan_typed(id)` once (`_plan_typed_ids`); ghost empty (caught up
  with the stream / all typed) → hidden; misaligned → model path (comment skipping also applies to model
  suggestions via `_reconcile`/`reconcile_fuzzy(skip_comments)`). `CompletionBackend` gained `plan()` /
  `plan_typed()` (looked up with `getattr`, optional).
- **UI.** `explain_btn` (💡) ↔ `actions.set_explanations` (also sets `cfg.explain.enabled` locally);
  `slim_explain_btn` ("Explain", visible only when on) with `_explain_menu` (Walkthrough / Explanation →
  `_open_panel`, Explain now → `actions.explain_now`, Rethink → `actions.solve("")`, Close panel);
  `_ExplainPanel` (`slim_panel`, `_PANEL_HEIGHT` 230: two tab buttons, `×`, `QTextBrowser`, lazy `flush()`
  from `_poll`, placeholders) sits under the slim bar; `_apply_slim_size` activates the layouts and re-fits
  the window (`_slim_height` + layout minimum) so it grows/shrinks with the panel; hidden when leaving
  completion mode, on Reset, or when explanations go off. Events: solve (explanations on) → walkthrough text
  (`split_plan(buffer).walkthrough`), status `thinking…` → `solution ready — type it out`, panel auto-opens
  in completion mode; explain → `explaining…` → `explanation ready`, panel opens on Explanation. The Assist
  tab still receives both (label "Explanation"; `_QUESTION_PREFIXES["explain"] = ""`).
- **Engine/actions.** `set_explanations(enabled)` (config, `assistant.set_explanations` -- off drops a queued /
  running explanation --, `completer.reset()`, `assistant.ensure_plan()` when the LLM is ready), `explain_now()`
  (deferred kind `"explain"` before the LLM is ready); `set_completion_mode(True)` and `new_session()` (in
  completion mode) call `ensure_plan()` when explanations are on.
- **Review-driven invariants (adversarial review, 10 confirmed findings fixed).** `_AssistJob.plan_id` pins an
  explain job to its plan: a job whose plan was replaced before it ran is skipped (Done, cancelled) and
  `_finish_explanation` never attaches to another plan. `_begin_plan` runs only after the cancelled-before-
  streaming check (a superseded solve must not wipe a finished plan); `_settle_plan_state(job, is_plan, …)` uses
  the job's start-time `is_plan`, so toggling explanations off mid-stream cannot leave a plan stuck `done=False`.
  `_JobSlot.remove_if(pred)` makes `cancel_job` atomic. `split_plan` takes the block still open while streaming,
  else the LARGEST fenced block (an aside in the walkthrough is not the solution). `align_plan` scores exact
  anchors 2 and fuzzy ones 1 (near-duplicate lines) and `Completer` caches the alignment per
  `(plan id, solution length, done, prefix)`; the plan ghost is also the `_shown` suggestion so blind editors keep
  following typing between OCR frames and a deviation hides it through the normal reconcile. The overlay ignores
  plan/explain events once the 💡 toggle is off (no panel re-opening) and `AssistView.on_done` now strips the
  default "Q: " prefix (history no longer shows "Q: Q:"). Title-bar spacing is 4 px so the 8th icon button fits
  the 340 px floor.

### Settings dialog + popup z-order (added 2026-09-21, fourth pass)
- **`ui/settings.py`**: `Field(section, key, label, kind, lo, hi, step, choices, restart, help)`; `FIELDS` (tab →
  fields: Completion, Explanations, Assist, Screen, Model, Window, Audio) mirrors README "Configuration";
  `all_fields()`, `sections_covered()`. `SettingsDialog(cfg, list_models, parent)` (non-modal, Tool + StaysOnTop,
  tabbed `QFormLayout`s in scroll areas): `load()`, `changes() -> {"section.key": value}`, `apply()` (writes cfg,
  emits `applied(dict)`, note lists restart-only fields), `restore_defaults()` (keeps `llm.model`), OK/Cancel.
- **Wiring**: `OverlayWindow.open_settings()` (from `⋯ › Settings…` and `Context ▾ › Settings…`), `_on_settings_applied`
  mirrors `ui.opacity` / `explain.enabled` / `assist.auto_answer_questions` / `audio.mic_start_muted` in the
  overlay's controls with signals blocked, then `actions.apply_settings(changed)`; `Engine.apply_settings` reacts
  (`llm.model` → `set_model`, `assist.auto_answer_questions` → `assistant.set_auto`, `explain.enabled` →
  `set_explanations`, `audio.mic_start_muted` → `set_mic_muted`, any `completion.*` → `completer.reset()`,
  `screen.interval_s` → `capture_now()`) and `save_config()` (extracted from `stop()`). Everything else is read
  live from the shared `Config`; restart-only fields are just persisted.
- **Defaults**: `completion.max_lines` 3 → 8, `completion.max_tokens` 48 → 120 (several lines predicted at once).
- **Popups above the overlay**: `_CaptureExclusionFilter` now also calls `win32.set_topmost(hwnd)` on every newly
  shown top-level window (menus, dialogs, tooltips) — a plain popup is not topmost and opened *behind* the
  always-on-top overlay; `_reassert_topmost` → `_raise_transients()` re-raises the active popup / the settings
  dialog after the overlay's own `HWND_TOPMOST` push. Verified live with `EnumWindows` z-order (menu and dialog
  above the overlay before and after the backstop; dialog affinity 0x11). Pop-ups also get the
  `WS_EX_LAYERED` style (`win32.ensure_layered` — Qt only layers a translucent window), which is what the
  legacy `PrintWindow` / `BitBlt(GetWindowDC)` reads need to come back black for them, and the filter hides
  one only while the affinity itself cannot be verified — toasting it on the panel so a withheld pop-up
  never looks like a broken app. The overlay is exempt and keeps veiling its own content.

### Multiple-choice questions (added 2026-09-21, fifth pass)
- **`llm/choices.py` (pure, no threads/Qt/model).** `find_choices(text, typed=False, question="") -> ChoiceSet | None`
  finds the option run of a multiple-choice question in screen text or a typed question: labelled `A. / A) / (A)`,
  `1. / 1) / (1)`, `i. ii.`, circled `①`/`Ⓐ` (normalised to `1`/`A`), Cyrillic `А)`, radio/checkbox glyphs
  (`○ ◯ ● □ ☐ ☑ ( ) [ ]` and the `O`/`o`/`0` misread) and -- only next to a question word -- plain `-`/`•` bullets.
  A row holding several options is split (`A) 4  B) 8`, the two-space join of `lines_to_text`; single spaces as a
  fallback and for typed text), a label OCR'd onto its own line is paired with the next line, `1.4 / 2.8 / 3.16`
  (dropped spaces, ≥ 3 lines, not all single digits, no `=`) reads as options 1..3, and `1. 2. C. D.` is one run.
  Evidence rules -- every run needs a question next to it, because agendas, slide bullets and instruction steps
  look identical: a *question cue* (question word / "Question N" header within `_CUE_WINDOW` lines above, or a
  question-word line ≤ 2 lines below) licenses any style; lettered runs and radio glyphs also take a statement
  right above them (a sentence ending in `. ? : !`) or a closed vocabulary (True/False, Yes/No, …); numbered runs
  whose options are all ≤ 2 words take a bare `?` directly above ("2 XOR 6 = ?"). Rejected: a cue matching
  `questions.DEFAULT_IGNORE_PATTERNS` (site polls) or a ranking cue ("Rank the following"), a list introduced by a
  plain colon without a question below it, duplicate option texts, 2..10 options violated. Bare `True/False` /
  `Yes/No` lines under a question are a NONE-style set. Several runs: the one whose cue line shares the most words
  with `question`, else the last one. `ChoiceSet(choices, style, span, lead, trail, cue)` (`labeled`, `labels`,
  `texts`, `by_label`), `Choice(label, text)`, styles `letter | number | roman | none`.
- `annotate_choices(text, cs)` rewrites the run's lines as `Choice 1: 4` (keeping `lead`/`trail` text that shared a
  row, e.g. the stem of a typed one-line question); unlabelled runs are left untouched. `choices_block(cs)` is the
  volatile `<choices>` block: options as `label: text` (or `- text`), "at most two short sentences", "do not default
  to the first one", the exact required last line. `finalize_answer(text, cs)`: the model's own `Answer:` line(s)
  are removed and resolved -- label, `label — text` (the text wins when they disagree), text, `option 3`, `(c)`,
  `1.4` = label 1 + text 4 when 1.4 is no option, comma/and-separated multi-answers, a bare token that is both a
  label and a text reads as the value -- else the verdict is inferred from the body (last *committing* sentence:
  positive cue such as "so"/"correct"/"the answer is", or a sentence that is exactly an option; "both A and C" →
  all named labels; an option only ever rejected -- "A is wrong" -- is skipped; a body that never commits →
  unclear). Result: body + blank line + `**Answer: <labels> — <texts>**` / `**Answer: <text>**` /
  `**Answer: (unclear)**`.
- **Prompts.** `_CHOICE_RULE` (byte-identical every turn, in `system_assist`/`system_ask`; `_CHOICE_RULE_SOLVE` in
  `system_solve` adds "a `<choices>` block is never a `[[CHOICE]]` fork"): "1. 4" is a label + text, never 1.4; if,
  and only if, a `<choices>` block is present, commit to one option, ≤ 2 sentences, exact last line; otherwise ignore.
  `user_solve/user_assist/user_ask(..., choices="")` place the block after `<screen>` and before the ask/`<question>`
  (still volatile-last for the KV prefix cache).
- **Assistant.** `_detect_choices(job, screen_text) -> (ChoiceSet | None, on_screen)`: never for suggest/explain/
  complete; the question text first (`typed=True`); then the screen for `solve`/`screen` jobs, and for `ask`/`auto`/
  `manual` only when the question refers to options (`_REFERS_TO_OPTIONS_RE`: "which one", "option", "correct", …),
  so a quiz in another tab never forces a verdict onto an unrelated question. `_build_prompt_parts(job, window,
  screen, choices, question)` adds the block and `CHOICES_TOKEN_HEADROOM` (64) to the budget; the annotated screen /
  question copies are prompt-local (plan fingerprints, the suggest throttle and screen-question de-dup keep raw OCR
  text; the answers history keeps the raw question). After streaming -- not cancelled, not the skip marker, not a
  `[[CHOICE]]` fork -- `_finalize_choices` appends the verdict: the tail is published as one more `AssistDeltaEvent`
  (the headless console prints deltas only), `AssistDoneEvent.text` and `answers` carry the finalised text, while
  `_settle_plan_state` gets the raw text (the completer never offers the verdict as ghost text).
- **UI.** `markdown_lite`: an ordered-list line out of sequence (a lone `3. 16`, an item after a blank line) starts a
  new `<ol start="N">` (QTextBrowser honours `start`) instead of an `<ol>` renumbered from 1; the bold last line
  renders as its own paragraph.
- Tests: `tests/test_choices.py` (detection positives/negatives incl. the adversarial set, annotation, block,
  explicit and inferred verdicts), `tests/test_assistant_screen.py` (solve/screen/ask wiring, skip, fork, cancel,
  plan isolation), `tests/test_overlay.py` (rendering).

### Checkpoints, Code button, primary language, resizable history (added 2026-09-22)
- **Checkpoint = soft reset of the CODING context only** (`Assistant.checkpoint()`, `EngineActions.checkpoint()`).
  Cancels a running solve/code/explain/complete job and drops any queued suggest/explain/complete-lane job (a
  running or queued main-lane auto/manual/ask/screen job is never touched); records `_checkpoint_seg_id =
  transcript.last_id()`, `_checkpoint_answers = len(_answers)`, `_checkpoint_time = clock()`, clears
  `_code_answers`, drops the plan (`_plan = None`, `_plan_parsed = None`); publishes `bus.CheckpointEvent(t,
  segment_id)`. From then on `solve`, `code` and `complete` jobs read `transcript.window_text(..., since_id=
  self._checkpoint_seg_id)` (only segments with id > since_id) and `_complete_prompt_parts` uses
  `self._answers[self._checkpoint_answers:][-3:]`; every other kind, notably `ask`, keeps the full transcript/
  answers. NOT touched by a checkpoint: the transcript store, the Live tab, notes, the answer history/export, the
  ask box, speaker labels, the screen-question de-dup memory, and the auto-suggest throttle state
  (`_last_suggested_lines`/`_last_suggest_time` — clearing them would let the very next OCR jitter re-solve the
  old problem still on screen). `Assistant.reset()` also clears the checkpoint state (`_checkpoint_seg_id`,
  `_checkpoint_answers`, `_checkpoint_time`, `_code_answers`). `Transcript.window_text(max_chars, *, until_id=
  None, since_id=None)` grew the `since_id` parameter for this.
- **`AssistKind` gains `"code"`** (the Code button). `Assistant.code(focus: str = "")` →
  `_enqueue_side_job("code", ...)`: same side lane as suggest/solve (`_CANCELS["code"] = ("suggest", "solve",
  "code", "explain")`; "suggest"/"solve" also cancel a running "code" — it never cancels a main-lane job), but
  unlike `solve()` it is MANUAL and NOT refused while paused (`EngineActions.code`). Stored as an answer like
  solve (appended to `_answers`) and also to `_code_answers` (both "solve" and "code" answers feed it, capped at
  10 by `_append_code_answer`). With explanations on it IS the plan (`is_plan = job.kind in ("solve", "code") and
  cfg.explain.enabled`). Never subject to multiple-choice detection (`_detect_choices` returns `(None, False)`
  for "code"; `system_code` carries no `CHOICE_PREFIX` fork protocol — the language default resolves the one
  thing that could otherwise look like a fork). `_record_suggest_start` runs for "code" too (auto-solve does not
  immediately re-solve the same screen). Question-label prefix `_QUESTION_PREFIXES["code"] = "Code: "` (default
  question "the task being discussed"; a non-blank `focus` is the label instead, like solve). Budget:
  `cfg.llm.solve_max_tokens` (+ `explain.walkthrough_max_tokens` when explanations are on). Prompt:
  `prompts.system_code(persona, explain, comments, language=cfg.assist.primary_language)` (`comments`
  is now the comment-level string, `prompts.COMMENT_LEVELS`, not the old boolean) +
  `prompts.user_code(window, screen_text, previous, job.focus)`, where `previous` = `Assistant.
  _previous_code_text()` — the most recent `_code_answers` entry, its `exocortex.plan.split_plan(text).solution`
  when that is non-empty else the whole text, capped at 2400 chars, `""` when there is none yet.
- **Primary language** (`AssistConfig.primary_language: str = "Python"`). One shared rule, `prompts.
  _language_rule(language)`, used by `system_solve`, `system_code` and `system_complete`: a language named in the
  conversation/problem statement, the visible editor's language, or one the task implies wins; only an open,
  ambiguous or unindicated choice falls back to `language`, and the model is told never to ask which language to
  use. `_solve_shape(explain, comments, language="Python")` now says "({language} if none is indicated)";
  `system_solve`'s fork example no longer lists "two plausible target languages" (language is never a fork any
  more). The `language` parameters default to `"Python"` (existing callers/tests keep working); `Assistant`
  always passes `cfg.assist.primary_language`. The system prompt stays byte-identical across turns for a fixed
  config (KV-cache) — `_language_rule` depends only on `language`, nothing volatile. Settings dialog:
  `Field("assist", "primary_language", "Primary language", "text", ...)` on the **Assist** tab
  (`settings.FIELDS["Assist"]`, placed after "Autonomous"). `config.example.toml`: `primary_language = "Python"`
  under `[assist]`.
- **Control row** (`OverlayWindow._build_control_row`), one row: `Read · Revert · Reset · Pause · Checkpoint ·
  <stretch> · Text · Code` (Text added 2026-09-23, left of Code). Attributes: `read_btn`, `revert_btn`,
  `reset_btn`, `pause_btn`, `checkpoint_btn`, `text_btn`, `code_btn`; `text_btn`/`code_btn.setObjectName("primary")`
  (accent colour). Measured under the Windows platform plugin (Segoe UI Variable 9 pt): 363 px (322 px before
  Text was added), inside the 330 px the real 346 px floor leaves after the root margins -- see the 2026-09-23
  section for the resulting clipping note. The row's
  horizontal size policy is `Ignored` so it never raises the window's minimum width — the offscreen QPA's
  fallback font is ~60 % wider (512 px for the same row), so tests assert structure, never these pixel numbers.
- **Slim bar** (`_build_slim_bar`, completion mode), left → right: `Full · Context ▾ · Explain ▾ (hidden unless
  explanations are on) · Reset · Pause · Checkpoint · Code · ● status · − · ⋯`. Attributes `slim_checkpoint_btn`,
  `slim_code_btn` (`code_btn`-style accent). `_SLIM_MAX_WIDTH` raised 460 → 560: the bar's own minimum (Windows
  plugin) is 456 px with Explain hidden / 530 px shown, i.e. a 474 / 548 px window; `_set_completion_mode` applies
  `max(_SLIM_MIN_WIDTH, min(normal width, _SLIM_MAX_WIDTH))` and then widens by any shortfall against
  `slim.minimumSizeHint()` after a forced layout pass, so the bar is never squeezed below its minimum whatever the
  cap or the full-mode width. The slim `Context ▾` menu gains "Checkpoint (new coding task)" and "Code now"
  entries before the "Settings…" separator.
- **UI wiring**: `checkpoint_btn` / `slim_checkpoint_btn` / the menu entry / `h.checkpoint` (`ctrl+alt+b`) →
  `OverlayWindow._on_checkpoint()` → `actions.checkpoint()` ONLY — every panel update is driven off the bus
  `CheckpointEvent` in `_handle_event` → `_on_checkpoint_event`: `assist_view.mark_checkpoint(HH:MM:SS)`, drops
  `_plan_text`/`_explain_text`/`_plan_job`/`_explain_job`, clears and hides `slim_panel` (+ `_apply_slim_size()`),
  toasts "Checkpoint — coding context starts fresh". `code_btn` / `slim_code_btn` / the menu entry / `h.code`
  (`ctrl+alt+g`) → `_on_code()` → `_show_tab(1)` (full mode only) + `actions.code()`.
- **Engine** (`app.py`): `Engine.checkpoint()` → `assistant.checkpoint()` + `completer.reset()` when a completer
  exists; never deferred (the Assistant object exists from `start()` on, even while the LLM is still loading — a
  checkpoint just records a boundary, no model call). `Engine.code(focus="")` → `assistant.code(focus)` unless
  `_defer("code", focus)`; `_init_llm` replays a pending "code". Headless console: `/checkpoint` and `/code`
  stdin commands; prints `[checkpoint] coding context reset (segment {segment_id})` on `CheckpointEvent`.
- **History divider**: `AssistView.mark_checkpoint(when: str)` inserts `(None, f"— checkpoint {when} —")` at the
  front of `_history` (rendered dim, centred, plain text); no-op when the history is empty or its newest entry is
  already a divider. Also clears the choice buttons and sets the footer to "checkpoint — the next Code / solve
  starts fresh"; the current answer text is left alone (the history entry already has it). `revert()` steps over
  divider entries (`_history[idx][0] is None`).
- **Resizable history** (Assist tab): `AssistView.history_splitter` — a vertical `QSplitter` between the top pane
  (question label, answer browser, choice buttons, footer, Stop/Answer-last row) and the bottom pane
  (`history_toggle` + `history_scroll`); both panes non-collapsible. Collapsed: `history_scroll` hidden, the
  bottom pane's max height pinned to the toggle's own `sizeHint().height()`, the handle
  (`history_splitter.handle(1)`) disabled. Expanded: handle enabled, bottom pane restored to `_history_height`
  (remembered across collapse/expand, starts at `_HISTORY_MAX_HEIGHT` = 200); `_on_splitter_moved` updates
  `_history_height` while expanded (a drag can't happen while collapsed). `history_scroll` no longer has a fixed
  maximum height (`_history_scroll(..., max_height=None)`) — the splitter bounds it now — but keeps a
  `_HISTORY_MIN_HEIGHT` (48 px) floor so it never disappears under the handle. The Screen tab's suggestion
  history keeps the old hard cap (`_history_scroll(..., max_height=_HISTORY_MAX_HEIGHT)`, the default).
- Tests: `tests/test_assistant_checkpoint.py` (checkpoint window/answers scoping, plan drop, `CheckpointEvent`,
  code job lane/pause/plan/previous-answer behaviour, solve/suggest cancelling a running code, reset), `tests/
  test_prompts_code.py` (language rule in solve/code/complete, fork example has no language case, `user_code`
  ordering/placeholders, system-prompt stability), `tests/test_overlay_checkpoint.py` (control-row/slim-bar
  order, checkpoint/code wiring, divider + toast, history splitter collapse/expand), plus additions to
  `tests/test_app.py` (checkpoint before the LLM is ready, code deferred/replayed), `tests/test_transcript.py`
  (`since_id`) and `tests/test_settings.py` (`assist.primary_language` on the Assist tab).

### Comment levels + the Text button (added 2026-09-23)
- **Config.** `AssistConfig.comments: str = "normal"` replaces the old `ExplainConfig.comments: bool`
  (REMOVED -- an old `config.toml` that still has `comments` under `[explain]` just logs one
  ignored-key warning at startup via `config._apply`'s existing unknown-key path, nothing else
  breaks). `LLMConfig.text_max_tokens: int = 400` -- the Text button's walkthrough budget, alongside
  `solve_max_tokens`/`assist_max_tokens`/etc.
- **`prompts.py` comment levels.** `COMMENT_LEVELS = ("none", "minimal", "normal", "detailed")`.
  `normalize_comment_level(value) -> str`: case/whitespace-insensitive match against `COMMENT_LEVELS`;
  `True` -> `"minimal"`, `False` -> `"none"` (a leftover caller still passing the old boolean keeps
  its old behaviour instead of silently landing on the new default); unknown or blank -> `"normal"`.
  `_comment_rule(level)` is the instruction spliced into `_solve_shape(explain, comments="normal",
  language)`, which both `system_solve(persona, explain=False, comments="normal", language="Python")`
  and `system_code(...)` (identical signature) call. Distinctive phrase per level (tests key on
  them): `none` -> "no comments at all"; `minimal` -> "minimal but important comments"; `normal` ->
  "well commented"; `detailed` -> "pseudo-code" + "requirements" + a leading comment block inside the
  fenced code (the way an engineer works a challenge through in the editor) before the real code,
  then a comment above most of its lines. Every commented level also demands each comment be its own
  line, "never trailing" at the end of a code line -- `exocortex/plan.py`'s completion-mode comment
  skipping (`consume_suggestion(..., skip_comments=True)`) and its anchoring (`_match_back` ignoring
  comment lines) both assume a comment can never share a line with code, so this phrasing is
  load-bearing, not stylistic. Because model completions now carry comments at every level but
  `none` (below), `Completer._reconcile` tolerates typing past a comment (`skip_comments`) whenever
  `cfg.explain.enabled` OR `normalize_comment_level(cfg.assist.comments) != "none"` -- it used to be
  gated on explanations alone, which would have turned every skipped comment into a false deviation
  (suggestion dropped, a fresh model round-trip) under the new default. `_comment_rule_complete(level)`
  is appended to `system_complete(persona,
  hint="", explain=False, language="Python", comments="normal")` UNCONDITIONALLY -- every level,
  including `none`, and regardless of `explain` -- right after the optional `explain`-only paragraph
  telling the model to follow a `<solution>` block when the context carries one; with `explain=True`
  that paragraph's existing "the user may skip typing comments" sentence is untouched by this feature
  (a plan's `<solution>` block already carries its comments from `_comment_rule` at solve time;
  `_comment_rule_complete` only ever governs a continuation typed WITHOUT such a block). Phrases:
  `none` -> "never add comments"; `minimal` -> "a genuinely important step"; `normal` -> "each
  logical step"; `detailed` -> "most lines of code".
- **`prompts.py` Text prompts.** `system_text(persona)` / `user_text(transcript_window, screen_text,
  previous="", focus="")` mirror `system_code`/`user_code`'s framing and layout exactly
  (checkpoint-scoped `<transcript>`, an optional `<previous>` with code produced earlier, `<screen>`,
  no clarifying questions, no preamble, `_FOCUS_RULE`, `SKIP_MARKER`) but ask for a plain-English
  walkthrough instead of code: what is asked, the key insight, the approach step by step, the edge
  cases, the complexity -- "Prose only" is the literal instruction, no code fences, at most a few
  inline tokens of code. `system_text` contains the literal `pressed "Text"`, mentions
  `<transcript>`/`<screen>`/`<previous>` the same way `system_code` does, and carries neither
  `CHOICE_PREFIX` ("[[CHOICE]]") nor any mention of a `<choices>` block -- a walkthrough reasons its
  way to an answer in prose, it never commits to a multiple-choice verdict line. `user_text`'s default
  ask ends "Output the walkthrough now."; placeholders match `user_code`'s: "(nothing said since the
  checkpoint)" for an empty transcript, "(nothing readable on screen)" for no screen text; `<previous>`
  is omitted entirely when blank (most turns for a brand-new task).
- **Assistant.** `AssistKind` gains `"text"`. `Assistant.text(focus="")` ->
  `_enqueue_side_job("text", ...)`: the same side lane as suggest/solve/code (never cancels a
  main-lane job), default display question "the task being discussed" (like `code`). `_CANCELS["text"]
  == ("text",)` and "text" appears in no OTHER kind's cancel tuple: a walkthrough neither preempts
  nor is preempted by a running solve/code/explain (they simply queue behind each other), only a
  newer "text" replaces an older one -- a Text press must never throw away a streaming solve/plan,
  and an automatic re-solve must never silently kill a manual walkthrough request. The same holds
  for QUEUED jobs because "text" has its own single-item mailbox, `_text_slot`, dispatched after the
  side lane and before the explain lane (`_assist_slots()` = main → side → text → explain →
  complete): a `_JobSlot.put` replaces whatever is queued, so sharing `_suggest_slot` would have let
  the next auto-solve silently overwrite a queued walkthrough (and a Text press a queued Code) with
  no event ever published (found by the adversarial review). `checkpoint()` now also cancels a
  running "text" job (`_cancel_running_of`'s tuple grew to include `"text"`) and clears `_text_slot`
  along with the suggest/explain/complete slots; `reset()` (New session) now clears every queued lane
  too -- `cancel()` alone only clears the main lane's slot, so a queued walkthrough used to survive a
  reset and land in the new session's answers (same pre-existing gap for a queued solve/code/
  suggest, closed at the same time). "text" jobs are checkpoint-scoped exactly like
  "solve"/"code"/"complete": `_run_assist_job`'s `since_id` gate (`job.kind in ("solve", "code",
  "text", "complete")`) reads `transcript.window_text(..., since_id=self._checkpoint_boundary())`.
  `_record_suggest_start` does NOT run for "text" (only suggest/solve/code): a walkthrough solves
  nothing, and bumping `_last_suggest_time` would make `_maybe_auto_suggest` skip the automatic solve
  of the NEXT screen for `suggest_min_interval_s`; the same screen needs no guard since auto-solve
  only fires on a changed `ScreenTextEvent`. `_detect_choices`
  returns `(None, False)` for "text" (alongside "code"): a walkthrough is prose, never a
  multiple-choice verdict. `_screen_text_for_job` always fetches the screen for "text" (same reasoning
  as "code": "the task being discussed / shown" needs the screen, never gated by
  `cfg.screen.use_in_answers` the way auto/manual/ask are). The prompt is `system_text(persona)` +
  `user_text(window, screen_text, self._previous_code_text(), job.focus)`, budget
  `cfg.llm.text_max_tokens` (flat -- no `COMMENT_TOKEN_HEADROOM`, no walkthrough-mode extra: Text has
  its own dedicated budget, unlike solve/code which share `_code_budget`). A finished "text" answer is
  appended to `Assistant.answers` (shows in the Assist tab and session export) but NEVER to
  `_code_answers` -- prose is not something a later "code"/"text" job's `<previous>` should build on
  -- and `is_plan` stays `job.kind in ("solve", "code")` only: a walkthrough is never the plan the
  completer follows.
- **Comment-level token budget.** `COMMENT_TOKEN_HEADROOM: dict[str, int] = {"none": 0, "minimal": 0,
  "normal": 100, "detailed": 350}`. `Assistant._code_budget(explain, level) = cfg.llm.solve_max_tokens
  + (cfg.explain.walkthrough_max_tokens if explain else 0) + COMMENT_TOKEN_HEADROOM[level]`, used by
  both the "solve" and "code" branches of `_build_prompt_parts` with `level =
  prompts.normalize_comment_level(cfg.assist.comments)`.
- **Engine/actions.** `EngineActions.text(focus="")` (Protocol); `Engine.text(focus)` ->
  `assistant.text(focus)` unless `self._defer("text", focus)` (LLM not ready yet -- deferred the same
  way as `code`/`checkpoint`); `_init_llm` replays a pending `"text"` the same way it replays `"code"`.
  Headless console: `/text` stdin command (`engine.text()`), alongside `/checkpoint` and `/code`.
- **UI.** `OverlayWindow.text_btn` ("Text", `objectName("primary")`, accent colour) sits in the
  control row immediately LEFT of `code_btn`, after the stretch: `read_btn, revert_btn, reset_btn,
  pause_btn, checkpoint_btn, <stretch>, text_btn, code_btn` -- 8 layout items, up from 7.
  `OverlayWindow._on_text()`: `_show_tab(1)` (the Assist tab) unless in completion mode, then
  `actions.text()` -- the same pattern as `_on_code()`. `widgets._QUESTION_PREFIXES["text"] =
  "Text: "`. The control row measures 363 px under the real Windows platform plugin (Segoe UI
  Variable 9 pt), up from 322 px before Text existed (six buttons); at the 330 px floor the row now
  clips the right-hand Code button a little (there is ~80 px to spare at the 460 px default window
  width) -- the row's horizontal size policy stays `Ignored` so this never raises the window's
  minimum width. Text is deliberately ABSENT from the slim bar (`_build_slim_bar`) and its
  `Context ▾` menu: a walkthrough is prose meant to be read in the Assist tab, which completion
  mode's slim bar hides entirely, so there is nowhere for it to land there. No hotkey is bound for
  Text (no `HotkeysConfig` field, no `hotkeys.text`).
- **Settings.** `Field("assist", "comments", "Comments in generated code", "choice", choices=("none",
  "minimal", "normal", "detailed"))` on the **Assist** tab, placed right after `primary_language`.
  `Field("llm", "text_max_tokens", "Text walkthrough budget (tokens)", "int", 100, 2000, 50)` on the
  **Model** tab, placed right after `solve_max_tokens`. The old `Field("explain", "comments", ...,
  "bool")` is gone from `FIELDS["Explanations"]`. `config.example.toml`: `comments = "normal"` under
  `[assist]`, `text_max_tokens = 400` under `[llm]`; `[explain]` no longer has a `comments` key.
- Tests: `tests/test_prompts_comments.py` (level normalisation, the distinctive phrase per level in
  `_comment_rule`/`_comment_rule_complete`, the own-line/never-trailing wording, budget headroom
  values), `tests/test_prompts_text.py` (`system_text`/`user_text` framing, placeholders, no
  `CHOICE_PREFIX`/`<choices>`), `tests/test_assistant_text.py` (side-lane placement, `_CANCELS`
  isolation, checkpoint scoping and cancellation, screen always fetched, answers-but-not-
  `_code_answers`, never the plan, pause behaviour), `tests/test_overlay_text.py` (control-row order
  and width, `_on_text` wiring, absence from the slim bar / Context menu), plus additions to
  `tests/test_app.py` (Text deferred before the LLM is ready and replayed once it is, like `code`)
  and `tests/test_settings.py` (`assist.comments` and `llm.text_max_tokens` fields).

### Hands-free mode + OCR on the iGPU / NPU (added 2026-09-23)

- **Config (`config.py`).** `HandsFreeConfig` (`[handsfree]`): `enabled: bool = False` (the
  switch; mirrored live by `Engine.set_hands_free`/`apply_settings`), `settle_s: float = 1.0`
  (no further change for this long before the settled screen is acted on), `change_threshold:
  float = 0.6` (share of lines that differ between two *consecutive* samples, `1 - Jaccard`, to
  count as a new task / big scroll), `min_changed_lines: int = 3` (and at least this many lines
  must differ), `deviation_threshold: float = 0.5` (share of on-screen code lines that are NOT
  the app's own to trigger a Code-only regeneration; `0` disables it), `context_chars: int =
  6000` (cumulative screen context budget, bigger than `[screen].max_text_chars` on purpose),
  `run_text: bool = True`, `run_code: bool = True`, `code_first: bool = False`, `accurate_ocr:
  bool = True` (re-read the settled screen with RapidOCR before evaluating it, when the sampling
  engine is Windows OCR). There is deliberately **no minimum-interval and no sampling-interval
  field**: the screen alone decides (a massive change that settles is acted on at once, however
  soon after the previous cycle -- `HandsFreeController` has no timer gate), and hands-free
  samples at `[screen].interval_s`, the ordinary capture interval (David: "respect that"), whose
  default changed **2.0 -> 1.5**. `ScreenConfig.ocr_device: str = "auto"` picks where RapidOCR's
  models run (`"auto" | "igpu" | "npu" | "cpu"`). `HotkeysConfig.toggle_handsfree: str =
  "ctrl+alt+h"`.
- **Bus (`bus.py`).** `HandsFreeState = Literal["off", "watching", "settling", "reading",
  "checkpoint", "text", "code", "paused"]`. `HandsFreeEvent(enabled: bool, state: HandsFreeState,
  detail: str = "")` -- published on every state change (de-duplicated: `HandsFreeController`
  only re-publishes when `(enabled, state, detail)` actually changed), drives the switch, the
  slim bar's menu entry and the title-bar subtitle. `ScreenTextEvent` gained `engine: str = ""`
  (`"windows"` / `"rapidocr/igpu"` etc., from `ocr.engine_label`) and `accurate: bool = False`
  (set when the event answers a `capture_now(accurate=True)` request).
- **Actions (`actions.py`).** `EngineActions.set_hands_free(enabled: bool) -> None`: on/off;
  suppresses auto-solve and on-screen-question answers while on (the cycle replaces them);
  publishes `HandsFreeEvent`s.
- **The controller (`exocortex/handsfree.py`, new module).** Pure decision helpers at the top,
  `HandsFreeController` (a "hands-free" daemon thread draining a bus subscription) below;
  `HandsFreeHooks` is the `Protocol` it needs from the engine: `capture_now(accurate=False)`,
  `checkpoint()`, `text() -> int | None`, `code() -> int | None`, `recent_code() -> str`.
  - `sample_diff(previous: set[str] | None, current: set[str]) -> tuple[float, int]`:
    `score = 1 - |A & B| / |A | B|` (Jaccard distance) and the count of lines in one set but not
    the other, over `normalize_lines()`-normalised (casefolded, whitespace-collapsed, < 2 chars
    dropped) OCR line sets. `_on_screen` (every `ScreenTextEvent` with `changed` and not
    `accurate`) calls `sample_diff(self._prev_sample, current)` and **always reassigns
    `self._prev_sample = current` afterwards** -- the comparison base is the immediately
    preceding sample, never the screen a cycle last triggered on and never however long ago the
    last cycle ran, so a screen that has drifted a lot over many small typing-sized steps still
    never trips `change_threshold` on its own; only a jump between two adjacent samples does.
    `_big_change` latches `True` the tick the threshold is crossed and stays set through the
    settle wait.
  - **Settling.** Every screen event while `_big_change` (or any changed event at all) resets
    `_last_change = now` and moves the watch state to `"settling"`; `_tick()` (the thread's
    `_TICK_S = 0.1` s wake-up) fires `_evaluate_pending()` once `now - _last_change >= settle_s`.
    With `accurate_ocr` on, that requests `capture_now(accurate=True)` (state `"reading"`,
    `ACCURATE_READ_TIMEOUT_S = 15.0` s ceiling before falling back to the sampled lines) and
    `_evaluate()` runs on the accurate `ScreenTextEvent`; otherwise it runs directly on the
    sampled lines.
  - **`_evaluate(lines, text)`** (module docstring's decision point): with `_big_change` set,
    `sample_diff(self._last_trigger_lines or None, current)` against the screen the *last cycle
    fired on* (not the previous sample) decides whether this is really new -- below
    `change_threshold` = the same screen already handled (a popup that came and went), too few
    changed lines = nothing new, `[assist].focus == "technical"` and
    `questions.has_technical_cue(text)` is false = nothing worth solving; otherwise `_trigger()`
    (records `_last_trigger_lines`, resets `ScreenContext`, `triggers += 1`,
    `_start_cycle(_planned_steps(), checkpoint=True)`). Whether or not a new task fired, the
    settled screen is folded into `self.context` (`ScreenContext.merge`); when no cycle is
    running and `own_code_lines(hooks.recent_code())` is non-empty, `deviation()` is checked
    (below) for a Code-only regeneration.
  - **`_planned_steps()`**: `["text", "code"]` (or reversed with `code_first`), `"text"` dropped
    when `not run_text or cfg.explain.enabled or cfg.completion.enabled`, `"code"` dropped when
    `not run_code`. The cycle (`_start_cycle` -> `_next_step` -> `_request_step`) runs
    `hooks.checkpoint()` then each step in order, waiting for that job's `AssistDoneEvent` /
    `AssistErrorEvent` (matched by `_awaiting_job_id`) before requesting the next; a step that
    comes back `cancelled=True` (a spoken question's answer preempted it) is re-requested up to
    `MAX_STEP_RETRIES = 2` times instead of leaving a blank answer; `STEP_TIMEOUT_S = 240.0` s
    moves on regardless if a step never finishes. A screen event that crosses `change_threshold`
    again *while a cycle is running* still only sets `_big_change`/re-settles -- the next
    `_evaluate` sees the cycle already in `_phase` and, once it decides this is genuinely new,
    calls `_trigger()` again, which calls `_start_cycle(..., checkpoint=True)` -> `_abort_cycle()`
    first: the running job's checkpoint cancels whatever was still in flight and starts over,
    there is no minimum time between cycles.
  - **`is_code_line(line)`** / **`own_code_lines(code_text)`** / **`is_own_code_line(line, own)`**:
    turn `hooks.recent_code()` (every solve/code answer since the checkpoint) into squashed
    (whitespace-free) comparison keys, and classify an OCR line as code (bracket/keyword/assignment
    heuristics, comments and long sentences excluded) matching one of them (exact, gutter-merged
    tail, or `plan._similar` fuzzy).
  - **`deviation(lines, own, exclude=None) -> (fraction, user_lines)`**: of the code-looking lines
    on screen (excluding the lines the task statement itself contributed, `_last_trigger_lines`),
    the share that are NOT the app's own code. In `_evaluate`, when `fraction >=
    deviation_threshold` AND `len(user_lines) >= min_changed_lines` AND the user lines *grew*
    by at least `min_changed_lines` since the last check (`_last_deviation`), `regenerations += 1`
    and `_start_cycle(["code"], checkpoint=False)` -- Code only, no checkpoint, so the new
    solution is built with the checkpoint-scoped transcript/context unchanged, just re-run against
    what is actually on screen now. Verified in `tests/test_handsfree.py` (being written
    alongside this feature).
  - **`ScreenContext`** (cumulative OCR, thread-safe: written by the controller thread, read by
    the assistant's worker thread via `HandsFreeController.context_text()`): `merge(lines, text)`
    records every normalised line ever seen (first-seen original text kept) plus the verbatim
    current screen; `text(max_chars)` renders lines no longer on screen (an edited line's earlier
    reading is dropped via a shared 12-character squashed-head match, `_SAME_LINE_HEAD`) under an
    `"[earlier on screen, now scrolled out of view]"` header, oldest trimmed first to fit the
    budget, followed by `"[on screen now]"` and the current screen intact and never trimmed. This
    is what answers the user's "if I scroll down, the model should still see the code typed
    further up" requirement.
- **Reader / OCR (`screen/reader.py`, `screen/ocr.py`).** `ScreenReader.capture_now(accurate:
  bool = False)`: `accurate=True` forces the next capture through the engine set with
  `set_accurate_engine(name)`, publishing `ScreenTextEvent(accurate=True)`;
  `accurate_engine_ready` reports whether it was built. `_get_engine` builds the accurate engine
  (when configured, via `set_accurate_engine`) **before** the sampling engine, once, on the
  worker thread, and reuses it for sampling too if the sampling engine turns out to be the same
  kind -- because of the **process-order rule** (`screen/ocr.py` module docstring, measured
  2026-09-23): creating a `RapidOcr` (onnxruntime or OpenVINO, any device) AFTER a `WindowsOcr`
  exists in the same process crashes the interpreter outright (access violation); RapidOCR first,
  Windows OCR second, coexist for the process's life. `RapidOcr.__init__` refuses
  (`OcrUnavailable`) instead of crashing when the module-level `_windows_ocr_created` flag is
  already set. `Engine._init_screen` wires this: `[handsfree].accurate_ocr` on and
  `[screen].ocr_engine != "rapidocr"` -> `screen.set_accurate_engine("rapidocr")`, so a
  Windows-OCR-sampling setup still gets an accurate RapidOCR re-read for hands-free without
  crashing. New in `screen/ocr.py`: `openvino_devices()` (`{device: full name}` via
  `ov.Core().available_devices` / `FULL_DEVICE_NAME`), `resolve_ocr_device(requested, devices=
  None) -> (kind, ov_device)` (`"auto"` = igpu > npu > cpu; the iGPU is the `GPU.x` whose name
  says `"(iGPU)"`, never a `(dGPU)`, so OCR never lands on the discrete NVIDIA card; an explicit
  device that is missing falls back to cpu with a warning), `OCR_DEVICES = ("auto", "igpu",
  "npu", "cpu")`, `_OpenVinoSession(core, model_path, device, *, static, kind, bucket=None)`
  (drop-in for RapidOCR's onnxruntime session, two independent knobs: `bucket` -- iGPU and NPU --
  pads inputs to a small set of shapes, det to a 256 px grid (`pad_det_input`, prediction map
  cropped back) and rec/cls to batch 6 x a width bucket (`pad_static_batch`, `npu_width_bucket`,
  buckets 320..4096), because both accelerators pay per distinct input shape -- the GPU plugin
  JIT-compiles kernels for every new crop width, measured 37 s for one fresh screen without
  bucketing; `static` -- NPU only -- reshapes + compiles the model per shape on first use, while
  the iGPU keeps the dynamic-shape model because a reshaped model loaded back from `CACHE_DIR`
  on the GPU came back wrong (measured 0-1 lines instead of 95) and the dynamic model caches
  correctly; the CPU plugin gets `INFERENCE_NUM_THREADS = cores // 2`; calls are serialised by a
  per-session lock; `warm_rec_buckets()` compiles `_WARM_REC_WIDTH_BUCKETS` (320..1280) eagerly;
  everything cached through `core.set_property({"CACHE_DIR": ...})` at `<models_dir>/ov_cache`),
  `RapidOcr(device="cpu", cache_dir=None)` (swaps RapidOCR's det/rec onnxruntime sessions for
  `_OpenVinoSession`s when `ov_device` is set; the text-orientation classifier `Global.use_cls`
  is off; `recognize` holds an instance lock), `RapidOcr.warm(width, height)` (rec buckets, then
  a synthetic page of text recognised once for the det shape -- the reader runs it on its own
  `screen-ocr-warmup` thread so Windows OCR sampling starts at once and the first accurate read
  merely waits: ~25 s the first time on a machine, ~7 s with the cache), `engine_label(engine) ->
  str` (`"windows"` / `"rapidocr/igpu"`), `create_engine(name="auto", device="cpu",
  cache_dir=None)` (the defaults build `RapidOcr()` exactly as before; the app passes
  `[screen].ocr_device`).
  Measured (this laptop -- Core Ultra 9 285H / Arc 140T iGPU / AI Boost NPU / RTX 5090 Laptop;
  dense 2560x1600 page, 97 lines, OCR input downscaled to 2200x1375; full context and the
  process-order rule in the Verified facts table above):

  | Engine / device | Wall | CPU |
  |---|---|---|
  | Windows OCR (CPU) | 108 ms | 109 ms |
  | RapidOCR, onnxruntime CPU | 3.7 s | 37 s (all cores) |
  | RapidOCR, OpenVINO CPU | 1.9 s | 9.9 s |
  | RapidOCR, OpenVINO iGPU (`GPU.0`) | 0.73 s | 0.74 s |
  | RapidOCR, OpenVINO NPU | 1.45 s | 0.6 s |

  Compile costs (paid once per machine, cached under `models/ov_cache`, ~190 MB): iGPU ~2-3 s per
  rec width bucket and ~3 s for a det shape, i.e. ~25 s of background warm-up the first time and
  ~7 s from the cache; NPU 8-14 s per bucket uncached, ~0.2 s cached. Through the real
  `ScreenReader` on a dense 2560x1600 desktop (220 Windows-OCR lines, 186 RapidOCR lines) an
  accurate iGPU read takes ~1.5 s; Windows OCR sampling starts 2.3 s after the reader does. OCR
  never runs on the RTX 5090 (Whisper + llama-server's card) -- the win is CPU time and latency,
  never GPU headroom borrowed from anything else.
- **Assistant gates + context path (`llm/assistant.py`).** `set_hands_free(enabled: bool,
  context_provider: Callable[[], str] | None = None)`: sets `self._hands_free` and
  `self._hands_free_context`; `_maybe_answer_screen_question`/`_maybe_auto_suggest` both gate on
  `not self._hands_free` (a screen change never starts an automatic solve/suggestion or answers a
  new on-screen question while hands-free owns the cycle) -- spoken-question auto-answer,
  explanations and every manual action are untouched. `recent_code() -> str`: every solve/code
  answer since the current checkpoint, newest last, blank-line-joined (`""` when none) --
  `HandsFreeController._evaluate` feeds this through `own_code_lines()` for the deviation check.
  `_screen_text_for_job(job)`: for `"text"`/`"code"` jobs while hands-free is on,
  `self._hands_free_context()` replaces the plain screen-reader text (falling back to it when the
  provider returns `""`), and the char budget becomes `max(normal budget,
  cfg.handsfree.context_chars)` -- so the cumulative context, not just the current screen, reaches
  the model. `HandsFreeHooks.text`/`.code` are `Assistant.text`/`.code`'s existing side-lane jobs
  (`_enqueue_side_job`, checkpoint-scoped, never cancel a main-lane job) -- hands-free adds no new
  job kind, it only drives the existing ones from the controller thread.
- **Engine wiring (`app.py`).** `Engine.handsfree: HandsFreeController | None = None` (created on
  first `set_hands_free(True)`, kept afterwards). `set_hands_free(enabled)`: persists
  `cfg.handsfree.enabled`, lazily builds `HandsFreeController(cfg, bus, _HandsFreeHooks(self),
  clock)`, mirrors the engine's current `_paused` into it, `assistant.set_hands_free(enabled,
  handsfree.context_text)`, `handsfree.start()`/`.stop()`. `_HandsFreeHooks` (module-private, not
  part of `EngineActions` -- `capture_now(accurate=...)` and the reader cadence are not something
  the UI calls directly) forwards `capture_now`/`checkpoint`/`text`/`code`/`recent_code` onto the
  `Engine`/`Assistant`. `Engine.start()`: `_init_screen()`, then `cfg.completion.enabled` ->
  `set_completion_mode(True)` and `cfg.handsfree.enabled` -> `set_hands_free(True)` at launch.
  `_init_screen()`: builds the OCR engine factory (binds `ocr_device` +
  `<models_dir>/ov_cache`) and, per the process-order rule above, registers the accurate RapidOCR
  engine before starting the reader. `Engine.set_paused`/`.new_session` (Reset) propagate
  directly to `self.handsfree.set_paused`/`.reset()` when it exists; a user-initiated
  `Engine.checkpoint()` does NOT call it directly -- `HandsFreeController` already subscribes to
  the bus and handles the resulting `CheckpointEvent` itself (its own checkpoints, published
  while starting a cycle, are counted in `_own_checkpoints_pending` and skipped; any other
  `CheckpointEvent` while not paused is treated as the user saying "redo this screen"). `apply_settings`:
  `"handsfree.enabled"` in the changed set re-runs `set_hands_free`; every other `[handsfree]`
  value needs no engine call -- `HandsFreeController` reads `self._cfg.handsfree` live on every
  evaluation. Headless console: `/handsfree` / `/handsfree on` / `/handsfree off` stdin commands
  (`/handsfree` alone toggles); prints `[hands-free] on/off: <state> - <detail>` on every
  `HandsFreeEvent`.
- **UI (`ui/overlay.py`, `ui/widgets.py`).** `ToggleSwitch(QWidget)` (`ui/widgets.py`): a
  hand-painted pill on/off switch (`toggled` signal, theme colours, faded while disabled) --
  `handsfree_switch` + `handsfree_label` ("Hands-free") sit at the right of the tab strip,
  `handsfree_switch.setEnabled(...)` only while the Assist tab is current or hands-free is
  already on (`_update_handsfree_switch_enabled`, re-run on every tab change and layout apply).
  `_set_hands_free(enabled, notify_engine=True)`: flips `self._hands_free`, syncs the switch
  without re-entering its own `toggled` signal, calls `_apply_layout()`, and (unless the change
  came FROM a `HandsFreeEvent`) calls `actions.set_hands_free(enabled)`. `_apply_layout()`
  (no-op in completion mode, whose slim bar owns its own layout): hides `askbox`, `control_row`,
  `status`, `size_grip`, `clickthrough_btn` and the tab buttons while hands-free is on, forces the
  Assist tab current, calls `AssistView.set_hands_free_view(hf)`, and restores the user's normal
  click-through setting on the way out. `AssistView.set_hands_free_view(on)` (`ui/widgets.py`):
  hides the question label, footer, Answer-last button and history toggle (answer text and an
  already-expanded history list are all that remains), hides the Stop button and choice buttons,
  disables the history splitter handle; `False` restores everything including Stop for a
  still-streaming answer. `_apply_hands_free_subtitle()`: the title bar's subtitle reads
  `f"hands-free · {self._hf_state or 'on'}"` while on, the model id otherwise; `_hf_state` is set
  from every `HandsFreeEvent` as `ev.state` (+ `f" · {ev.detail}"` when `ev.detail` is set), so
  the shown text is "hands-free · <state>" or "hands-free · <state> · <detail>"; driven by
  `_on_hands_free_event`. **Cursor-tracking partial
  click-through** (`_update_hands_free_click_through`, polled every UI tick while hands-free is on
  and not in completion mode): `_hands_free_clickable_at(pos)` checks the title bar and the switch
  + its label; the window is made `WS_EX_TRANSPARENT` (clicks pass through) whenever the global
  cursor is elsewhere, and clickable again over those widgets -- `QCursor.pos()` keeps working
  while the flag is set (the OS routes clicks, not cursor queries), so this needs no extra hook.
  The flag is only touched on an actual change (`_hf_click_through` remembers the last applied
  value); `_restore_click_through()` reverts to the user's own click-through setting
  (`cfg.ui.click_through`) when the hands-free view ends. The `◈` click-through toggle
  button (`clickthrough_btn`) is one of the widgets `_apply_layout` hides while hands-free is on,
  since the view is managing click-through itself. Slim bar's `Context ▾` menu
  (`_rebuild_context_menu`) gains a checkable "Hands-free (read · checkpoint · text
  · code by itself)" entry alongside Explanations/Checkpoint/Code. Hotkey `h.toggle_handsfree`
  (`ctrl+alt+h`) -> `_set_hands_free(not self._hands_free)`. `_on_hands_free_event` also syncs
  `_hands_free` (and re-applies the layout) when the state changed from elsewhere (settings, the
  console `/handsfree` command).
- **Settings (`ui/settings.py`).** New **"Hands-free"** tab (between Explanations and Assist):
  `change_threshold`, `min_changed_lines`, `settle_s`, `deviation_threshold`, `context_chars`,
  `run_text`, `run_code`, `code_first`, `accurate_ocr` (`restart=True`), `enabled`. **"Screen"**
  tab gains `Field("screen", "ocr_device", "OCR device (RapidOCR)", "choice", choices=("auto",
  "igpu", "npu", "cpu"), restart=True)`.
- Tests (being written alongside this feature): `tests/test_handsfree.py` (the pure decision
  helpers -- `sample_diff`, `deviation`, `own_code_lines`/`is_own_code_line`, `ScreenContext` --
  and `HandsFreeController`'s state machine: trigger threshold using the previous *sample* not
  the previous *trigger*, settle timing, checkpoint -> text -> code sequencing, retry-on-preempt,
  mid-cycle re-trigger with no minimum interval, deviation-driven Code-only regeneration, pause/
  reset/user-checkpoint semantics), `tests/test_screen_ocr_device.py` (`resolve_ocr_device`,
  `openvino_devices`, `npu_width_bucket`, `pad_det_input`/`pad_static_batch`, the iGPU-never-
  discrete-GPU rule), `tests/test_handsfree_wiring.py` (`Engine.set_hands_free`,
  `_HandsFreeHooks`, `_init_screen`'s accurate-engine registration and the process-order
  build-before-sampling sequencing, `apply_settings`, the headless `/handsfree` command,
  `Assistant.set_hands_free`/`recent_code`/`_screen_text_for_job` context substitution),
  `tests/test_overlay_handsfree.py` (the switch, `_apply_layout`, `AssistView.
  set_hands_free_view`, the subtitle, the slim-bar menu entry, the hotkey, cursor-tracking
  click-through, `HandsFreeEvent` handling).

### Hands-free hardening: change classifier, focus levels, hover hooks (2026-09-23, later)

- **Root causes (Verified facts table above has the exact log lines).** One session exposed three
  bugs in the pre-hardening build. (1) The old whole-screen "technical cue" gate vetoed real slide
  changes -- `22:46:27` / `:30` / `:35` each logged `new screen but nothing technical on it;
  ignoring` for slides that had genuinely changed. (2) The old `min_changed_lines` check compared
  only the two most recent *samples*, so a one-line question change on an otherwise static slide
  measured as `2 line(s)` against the default `min_changed_lines = 3` and started nothing --
  `22:46:58`, `settled screen has nothing new (2 line(s)); no cycle`. (3) The ordinary Code ask has
  no notion of a question ABOUT the code shown, so a coding slide correctly got the skip marker
  three times running (`22:44:58`-`:59`, jobs 4/5/6; `22:50:22`-`:23`, jobs 25/26/27) -- and because
  nothing distinguished "the model chose not to answer" from "a higher-priority job preempted this
  step", the controller retried each skip as a preemption, burning both `MAX_STEP_RETRIES` retries
  on an answer that was never coming.
- **The decision pipeline (`handsfree.py`).** `classify_change(previous, current, typed=())` (raw
  OCR lines or keys; `previous=None` = no earlier screen) replaces the old sample-vs-sample
  Jaccard-only check with a `ChangeReport(score, changed, fresh, fresh_prose, fresh_code,
  questions, edits, typed, gone)`: every line of `current` absent from `previous` is classified as
  an **edit** (paired via `is_edit` with a line that disappeared), **typed** (paired with a
  squashed key already in the controller's typed memory), or **fresh** (neither).
  `ChangeReport.new_text` is the fresh prose lines with >= `MIN_WORDS_NEW_TEXT = 3` words.
  `is_new_task(report, hf, focus=..., ignore_patterns=...) -> (bool, reason)` then applies the
  module docstring's three rules in order -- a relevant line in `report.questions`; >=
  `hf.min_new_lines` relevant, task-like lines in `report.new_text`; or `report.score >=
  hf.change_threshold` with `len(report.fresh) + report.gone >= hf.min_changed_lines` and at
  least one relevant, task-like fresh-prose line -- the reason is logged and surfaces as
  `HandsFreeEvent.detail`.
  - **`is_edit(a, b)`** (squashed keys) is the same line at two moments, not two different lines,
    when either (1) one extends the other -- typing, backspacing, one trailing character of OCR
    noise allowed, >= 4 characters in common (an editor gutter's counter ticking, `"Ln 12"` ->
    `"Ln 13"`), or (2) the two are within 2 characters of the same length and differ in at most
    `EDIT_MAX_DIFF = 2` characters, one in ten for long lines (OCR noise on a re-read, a typo
    fix). Two DIFFERENT questions of similar shape -- `"what is a hash map?"` vs `"what is a
    heap?"` -- satisfy neither rule (no long common prefix; more than 2 characters differ) and so
    are correctly two fresh lines, never an edit of one another.
  - **The typed memory.** `HandsFreeController._track_sample` folds every sample's
    `classify_change` result into `self._typed` (squashed key -> `None`, capped at
    `_MAX_TYPED_MEMORY = 400`): an `edits` pairing moves a line's squashed key from its old form
    to its new one, so a line being actively typed is never "fresh" however many samples it takes
    to finish; `_trigger()` (a new task firing) clears the memory, so an old typed line never
    suppresses a genuinely new one later.
  - **The veto against the last-triggered screen** reuses the same machinery: on a settled screen,
    `classify_change(self._last_trigger_keys or None, keys, typed=self._typed)` is run back
    through `is_new_task` -- a popup that came and went, or an alt-tab and back, produces a
    `ChangeReport` with nothing fresh against the screen the LAST CYCLE fired on (even though it
    differs plenty from the immediately preceding sample, which is what armed `_big_change` in the
    first place), so it is correctly not a new task, and `detail = "same task"`.
  - **`task_kind(lines, text, window_title, window_process, mode) -> (kind, why)`**: `"code"`
    when >= `MIN_CODE_LINES = 2` lines look like code (`is_code_line`), or >= `MIN_GUTTER_LINES =
    3` lines match an editor's line-number gutter (`"12  def f():"`), or (outside `"region"`
    mode, where the foreground window and the region read are unrelated) `window_title` /
    `window_process` name a development window or process (`_DEV_WINDOW_RE` / `_DEV_PROCESS_RE`:
    VS Code, PyCharm, a terminal, LeetCode... / `Code.exe`, `pycharm64.exe`...), or the text
    matches `_CODING_TASK_RE` (`"write a function..."`, `"given an array..."`, `"Input:"`,
    `"time complexity"`, `"sql"`...); else `"text"`. `[handsfree].on_code` shapes
    `_planned_steps` for it: `"code"` (default) makes a code task **Code only**; `"code_first"`
    runs Code then Text; `"ignore"` leaves the ordinary order alone. For a code task under
    `on_code = "code"` the Code step is requested as `hooks.code(HANDS_FREE_CODE_FOCUS,
    HANDS_FREE_CODE_LABEL)`, an ask that explicitly covers a question ABOUT the code on screen
    ("If it asks ABOUT the code shown -- its output, a bug, its complexity, which option is
    correct -- give that answer in one line first, then the code that demonstrates it. ... never
    answer that there is nothing to do.") with a short label (`"the coding task on screen"`)
    standing in for that paragraph wherever the UI shows the question; every other step is
    requested with no arguments (`hooks.text()` / `hooks.code()`), the ordinary default ask.
  - **`AssistDoneEvent.skipped: bool`** (new field, `bus.py`, always paired with `cancelled=True`)
    is set when the model answered with `prompts.SKIP_MARKER` (`Assistant._run_job`); the
    controller's step-completion handler now branches on it: `skipped=True` is a finished step,
    no retry (the model looked and genuinely found nothing); `cancelled=True` without `skipped` is
    a preemption (a main-lane job, e.g. a spoken question's answer, cancelled the step) and is
    re-requested up to `MAX_STEP_RETRIES = 2` times as before. This alone fixes root cause (3): a
    coding slide the Code ask has nothing left to add to is marked done on the first skip, not
    retried twice for nothing.
- **The focus gradient (`llm/questions.py`).** `FOCUS_LEVELS = ("technical", "auto", "all")`;
  `focus_level(focus)` folds any `[assist].focus` value to one of them (`"auto"` for anything
  else, including the old default and blank). `is_task_line(text)`: a question (`is_question`,
  2+ words), an opening task verb via `_TASK_LINE_RE` (`"Implement..."`, `"Given an array..."`,
  `"Q3: Explain..."`), or a technical cue (`has_technical_cue`) -- what `"auto"` focus wants of a
  fresh line, both for an on-screen question and, via `is_new_task`'s `task_like`, for the "most
  of the view changed" rule. `is_relevant(text, focus="auto", ignore_patterns=None)`:
  `DEFAULT_IGNORE_PATTERNS` (site polls, feedback, cookie/sign-up chrome) plus
  `[assist].ignore_patterns` are dropped at every level; `"all"` then passes everything else;
  `"auto"` (default, code-biased) drops a personal question (`_PERSONAL_RE`: "have you...", "do
  you like...") unless it carries a technical cue, and otherwise passes; `"technical"`
  additionally requires a technical cue in the text itself. `HandsFreeConfig` carries no `focus`
  field of its own -- `is_new_task` reads `cfg.assist.focus` (through `questions.focus_level`), so
  on-screen questions and hands-free share the exact same one setting.
- **The libraries option (`llm/prompts.py`), soft like `primary_language`.** `LIBRARY_LEVELS =
  ("none", "basic", "all")`, `normalize_library_level(level)` (folds to one of them, `"basic"`
  default). `_library_rule(libraries="basic")` renders one line: *"Libraries: follow what the
  task says about them -- the problem statement, a remark in the conversation ('without any
  libraries', 'basic functions only', 'no imports', 'you may use any library') or the libraries
  the code on screen already imports -- and when that ask changes mid-task, redo the code
  accordingly. Only when nothing is indicated: <default>."* -- a SOFT rule, never a lock: the
  task's own statement, the conversation, or the code already on screen always wins over the
  config value; only the open case falls back to `_LIBRARY_DEFAULTS[level]` (`"none"`: built-in
  constructs and basic built-in functions only, no imports; `"basic"`: the standard library's
  commonly known basics, imports written out, no third-party package; `"all"`: anything
  importable, third-party packages included). `system_solve(...)` / `system_code(...)` both
  gained a `libraries: str = "basic"` parameter and insert `_library_rule(libraries)` right after
  `_language_rule(language)`; `Assistant` passes `libraries=cfg.assist.libraries` at both call
  sites. The `"basic"` default names no language, so a custom `primary_language` never sees the
  literal word "Python" appear anywhere in the prompt when nothing else names one.
- **The hooks (`ui/llhooks.py`, new module).** Two `SetWindowsHookExW` low-level hooks
  (`WH_MOUSE_LL = 14`, `WH_KEYBOARD_LL = 13`) for what `RegisterHotKey` cannot do:
  `MouseWheelHook(on_wheel)` hands every wheel notch (`WM_MOUSEWHEEL` / `WM_MOUSEHWHEEL`, delta
  from `MSLLHOOKSTRUCT.mouseData`) to `on_wheel(delta, horizontal) -> bool`, swallowing it
  (blocking delivery to the window underneath) when the callback returns True;
  `BareKeyHook(bindings, on_tap=None, max_tap_s=0.6)` calls a binding's callback when its key is
  **tapped** -- pressed and released with no other key pressed in between, within `max_tap_s` --
  tracked via `WM_KEYDOWN`/`WM_KEYUP`/`WM_SYSKEYDOWN`/`WM_SYSKEYUP` on `KBDLLHOOKSTRUCT.vkCode`,
  `GetAsyncKeyState` over `_HELD_CHECK_VKS` deciding whether another key is already held (a
  modifier held makes the press a combination, not a tap); a modifier key (`MODIFIER_KEYS`:
  `rshift`/`lshift`/`shift`/`rctrl`/.../`rwin`/`lwin`/`win`) is never swallowed (so Shift+letter
  for a capital keeps typing working), a plain key (e.g. `"right"`) IS swallowed while it acts as
  the toggle (no modifier held). `parse_bare_key(text)` / `is_bare_key(text)` recognise a single
  key name (no `+`) against `MODIFIER_KEYS` and `ui/hotkeys.py`'s `_NAMED_VKS`, raising /
  returning False for a `"+"` combination, which stays on `GlobalHotkeys`/`RegisterHotKey`.
  **Installed only while hovering.** The overlay installs `MouseWheelHook` only while the cursor
  is over an output pane and the Up/Down `BareKeyHook` only while it is over the expanded history
  (`_update_hands_free_hover_hooks`), so the per-move cost of a Python callback on the Qt main
  thread is paid only then; `_pause_key_hook` (the `toggle_pause` bare key) is the exception --
  installed once for the life of the window, and only on the real `"windows"` Qt platform
  (`_install_pause_key`, reached from `_install_hotkeys`, never under the `"offscreen"` platform
  tests use). **Fallback.** `_wheel_hook_usable` is False when `llhooks` is unavailable (off
  Windows) or the hook failed to install (`MouseWheelHook.failed`); `_hands_free_clickable_at`
  then also returns True over an output pane (`_hands_free_scroll_area_at`), so the window is
  left clickable there instead and the OS's own wheel delivery reaches the pane. **The ~300 ms
  hook timeout** is why callbacks defer work: Windows silently drops a low-level hook whose
  callback exceeds an internal timeout (~300 ms by default), so `MouseWheelHook`/`BareKeyHook`
  callbacks do as little as possible (read the message, decide, return) and hand the real work (a
  scroll, a history snap, a pause toggle) to the overlay through `on_wheel` / `on_tap`, which the
  overlay always defers with `QTimer.singleShot(0, ...)` rather than acting inline in the hook
  callback. `user32` is injectable (`_LowLevelHook.__init__(user32=...)`); tests drive the
  captured hook procedure directly with `(n_code, w_param, l_param)`, `l_param =
  ctypes.addressof(struct)`, against a fake object exposing `SetWindowsHookExW`/
  `UnhookWindowsHookEx`/`CallNextHookEx`/`GetAsyncKeyState`.
- **`ScreenTextEvent.window_title` / `.window_process`** (`bus.py`, new fields, default `""`):
  the foreground window at capture time, whatever the capture mode (`auto` / `monitor` /
  `region`), filled by `ScreenReader._foreground_info()` -- `exocortex.screen.foreground
  .foreground_window()` wrapped so any failure, and running off Windows or with our own overlay
  in front, yields `("", "")` rather than raising. `task_kind` reads these to tell a development
  window in front (outside `"region"` mode).
- **The job label (`llm/assistant.py`, `app.py`).** `Assistant.code(focus="", label="")` /
  `.text(focus="", label="")` gained `label`: what `AssistStartedEvent.question` (and so the
  Assist tab and the history entry) show instead of `focus`, for when `focus` itself is a long
  standing instruction rather than something worth showing verbatim (`HANDS_FREE_CODE_LABEL =
  "the coding task on screen"` vs. the paragraph-long `HANDS_FREE_CODE_FOCUS`). `Engine.code`/
  `.text` (`app.py`) take the same `label` and thread it through `_defer`, whose pending-request
  tuples are now `(kind, text, label)` 3-tuples (`Engine._pending`) so a request queued while the
  LLM is still starting is replayed with its label intact rather than losing it;
  `_HandsFreeHooks.text`/`.code` (the `HandsFreeHooks` Protocol implementation `handsfree.py`
  calls into) pass `focus` and `label` straight through to `Engine.text`/`.code`.

### One session per launch + Assist tab at start (added 2026-09-24)
- **`--replace`** (`app.main`; passed by `run.bat` and `run.vbs`): right after `load_config`, before the
  engine or Qt load anything, `exocortex.instance.replace_previous(cfg.sessions_dir())` force-closes every
  other ExoCortex and waits (up to 10 s) until each has exited -- the old llama-server must be gone from the
  GPU before the new one loads (two 27B copies on the 24 GB display GPU stalled the desktop on 2026-09-24).
  Its `ReplaceResult.notice` ("Closed the previous ExoCortex session", or "Could not close an earlier
  ExoCortex (...): access denied (is it running as administrator?)") becomes a toast 3 s after start-up
  (`_run_ui(cfg, notice)`; after the overlay's own hotkey-conflict toast) or a `[replace] ...` line in
  headless mode. Never raises: a failure is logged and the new session starts anyway. A plain
  `python -m exocortex` (no flag), `--doctor` and `--no-ui` never close anything.
- **`exocortex/instance.py`.** `select_targets(procs, self_pid, sessions_dir, caption)` (pure): roots =
  Python interpreters (`python*/pythonw*/py/pyw`) whose argv runs the package (`is_exocortex_argv`:
  `-m exocortex` or `-m exocortex.__main__` after any interpreter options / `py.exe` version selectors,
  or a script path ending `exocortex\__main__.py`; `-m pytest -m exocortex`, `-c ...` and llama-server's
  `-m <model>` are not; and, when `caption` is the `[stealth].process_caption` this process actually
  applied, a one-word command line equal to it -- the caption rewrite replaces the *whole* command line in
  the PEB, which is precisely the field `snapshot()` reads back); their descendants only of the kinds
  ExoCortex starts (llama-server, Python, conhost/OpenConsole --
  an app the old session opened survives), a child counted only when created after its parent (PID
  reuse); plus stray llama-servers whose `--log-file` lies inside the sessions folder. Never `self_pid` or
  its ancestors (the venv's `pythonw.exe` stub runs `-m exocortex` too and the py launcher's job object
  would take this process down with it), never anything created after `self_pid` ("newest wins": two quick
  launches cannot close each other), nothing when `self_pid` is missing from the table. Win32 layer, ctypes
  only: `snapshot()` = Toolhelp32 (pid, ppid, exe) + `OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION)` ->
  `GetProcessTimes` and, for interpreters / llama-servers, `NtQueryInformationProcess(ProcessCommandLine
  Information=60)` + `CommandLineToArgvW` (311 processes in ~18 ms here; processes of other users or
  elevated ones are unreadable, so never selectable -- the llmstack's are); `terminate(targets, timeout_s)`
  = `TerminateProcess` then `WaitForSingleObject` against one deadline, re-checking the creation time before
  terminating (a reused PID is left alone), an already-gone process counting as closed. No psutil.
- **Start tab.** `overlay._START_TAB = 1`: `_build_ui` calls `_show_tab(_START_TAB)` once the stack is
  filled (the Assist button checked, the hands-free switch enabled from the first frame). Nothing persists
  the tab; hands-free at start mirrors onto the same tab, and leaving completion mode's slim bar lands on it.
- **Answer-last button fix** (surfaced by the start tab): `AssistView.answer_last_btn` had a horizontal
  `Ignored` size policy since the first commit; next to the row's stretch that laid it out at zero width,
  so it was never visible (the real-app screenshot shows the gap). Now `Preferred` + `setMinimumWidth(1)`:
  its natural width, still able to shrink rather than raise the 340 px window floor.
- Tests: `tests/test_instance.py` (argv recognition, selection on synthetic tables incl. ancestors, the
  llmstack, PID reuse, stray servers, newest-wins; notices; `replace_previous` never raising; `main()`
  wiring; the launchers pass the flag; a read-only real snapshot; a real takeover restricted to a dummy
  `<tmp>\exocortex\__main__.py` tree started by the test itself), `tests/test_overlay_start.py` (Assist at
  start, hidden start, after a previous window left on Notes, hands-free, completion mode).

### GPU pressure guard, shared-server unload + KV/threads diet (added 2026-09-25)
Root cause of the 2026-09-24/25 mid-meeting freezes (measured): a separate llama-swap service on
127.0.0.1:8081 loads a SECOND copy of the same 27B on remote fleet demand; two 16.8 GB weight sets
oversubscribe the 24 GB display GPU, WDDM pages into system RAM, the desktop stalls. Prompt sizes
and speeds in the session logs stay bounded/healthy until that moment -- context trimming was ruled
out. What ships here: detection, a smaller footprint, and -- owner-approved on 2026-09-25 --
freeing the shared model server at launch and quit. (A "foreground lease" file advertising the
running session shipped briefly the same day and was removed on request, unused.)
- **`exocortex/gpu.py`.** `vram_free_mb() -> int | None`: NVML via ctypes (`nvml.dll`, lazy init,
  any failure parks it at None forever -- a driver reset must not spam). `VramGuard(free_mb_fn,
  threshold_mb, consecutive=2, cooldown_s=180, clock)`: `sample() -> str | None`; alert only after
  `consecutive` low samples (a brief spike never alerts), repeat at most every `cooldown_s`,
  one recovery message above threshold + `RECOVERY_MARGIN_MB` (512, anti-flap); None readings
  reset the streak. Engine wiring: `_start_gpu_housekeeping()` in `start()`, a "gpu-guard" daemon
  thread (`_gpu_guard_loop`, cadence `app.GPU_GUARD_INTERVAL_S` = 5 s, watches
  `self._stopped`); first probe None -> thread exits quietly. Alerts publish
  `ComponentStatusEvent("app", "error", ...)` (toast, no dot damage), recoveries `"info"`. Config:
  `LLMConfig.vram_guard = True`, `vram_guard_free_mb = 1500`.
- **Shared-server unload** (`exocortex/instance.py`): `free_llama_swap(url, timeout_s=45,
  http_get=None, clock=None, sleep=None) -> str` -- GET `<url>/running`; unreachable / non-200 /
  malformed / empty -> "" (a machine without llama-swap costs one failed local GET); else GET
  `<url>/unload` and poll `/running` once per second until empty ("Freed the shared model server
  (<n> model(s))") or `timeout_s` ("The shared model server did not release its model in time").
  Never raises; `http_get(url, timeout) -> (status, body)` injectable, default httpx. Wired
  twice, both gated on `backend == "llamacpp"` and a non-blank `LLMConfig.swap_unload_url`
  (default `http://127.0.0.1:8081`; "" disables): `main()` under `--replace` right after
  `replace_previous` (its notice joins the replace notice with "; ") so the old copy is off the
  GPU before ours loads, and `Engine.stop()` after the components (10 s budget) so quitting
  leaves the GPU empty. `tests/conftest.py` has an autouse fixture stubbing `free_llama_swap` to
  "" so no test can ever unload a model the developer is using; wiring tests install their own
  recorder over the stub.
- **Server diet** (`exocortex/llm/server.py`): `LLMConfig.kv_cache_type = "q8_0"` ->
  `--cache-type-k/-v` unless "f16"/blank (llama-swap runs the same GGUF with q8_0; hybrid qwen35
  keeps KV small, this is margin). `default_threads()` now counts PHYSICAL cores via
  `GetLogicalProcessorInformationEx(RelationProcessorCore)` (walk variable-size records by their
  Size field), falling back to the old `cores // 2` SMT guess; on the 16-core no-SMT 285H that is
  14 threads, not 6.
- David's `config.toml`: `[asr] device = "cpu"`, `compute_type = "int8"` -- Whisper off the display
  GPU (~1.5-2.5 GB freed; CTranslate2 int8 on the CPU is fast enough per utterance).
- Tests: `tests/test_gpu_guard.py` (guard logic incl. cooldown/hysteresis/None, NVML-off paths,
  Engine wiring incl. thread lifecycle), `tests/test_instance_swap.py` (free_llama_swap flows +
  launch/quit wiring) and additions to `tests/test_server.py` (cache-type flags, physical-core
  thread count).

### Manual advance mode (added 2026-09-25)

A Manual / Auto toggle next to the hands-free switch. Auto is everything as it was. Manual
(`[handsfree].manual`) preserves GPU compute and context: nothing starts a model call by itself
-- the transcript, screen scans (OCR), hands-free's watching / settle / task detection, its
auto-checkpoint on each new task (free: "a checkpoint just records a boundary, no model call")
and the cumulative screen context all keep running, but hands-free Text/Code, spoken-question
auto-answers, on-screen question answers, auto solve/suggest, auto notes and auto explanations
all wait. Generation happens only on a press-and-hold mouse gesture, anywhere on screen: the
LEFT button held `manual_hold_s` (default 0.8 s, settings-exposed) = Text, the MIDDLE button
(scroll wheel) = Code; with `manual_middle_tap` (default on) a QUICK middle-click also makes
Code, so a touchpad's three-finger tap works once Windows maps it to a middle click. The gesture
never brings the overlay to the foreground (nothing calls show/raise/activate on that path; the
hook swallows nothing, so the press still reaches the app underneath). Completion mode and the
deliberate mode activations (`ensure_plan`) are NOT gated; every manual action stays ungated.

- Config (`exocortex/config.py`): `HandsFreeConfig.manual` (False), `manual_hold_s` (0.8),
  `manual_middle_tap` (True) -- all read live, like the other `[handsfree]` values;
  `HotkeysConfig.toggle_manual` ("ctrl+alt+j"). Mirrored in `config.example.toml` and README.
- Bus (`exocortex/bus.py`): `ManualAdvanceEvent(enabled)` (the sync event every surface follows);
  `HandsFreeState` gains `"armed"` (a task read and checkpointed, generation waiting for the
  gesture).
- Controller (`exocortex/handsfree.py`): in manual mode `_trigger` still checkpoints and resets
  the cumulative context but arms (`self._armed`, steps `[]`) instead of running; the state
  publishes as `"armed"` with `ARMED_DETAIL` ("hold left = Text, middle = Code"); the
  deviation-driven Code regeneration is skipped, and manual flipped ON mid-cycle arms the
  cycle's remaining queued steps instead of letting them keep auto-firing (`_next_step` re-checks
  the live flag before popping a step). `trigger_step(kind)` (any thread, posts a
  `("manual", kind)` command; ignored while paused) aborts any cycle and runs that one step
  through the ordinary `_request_step` machinery -- a code task keeps `HANDS_FREE_CODE_FOCUS` /
  `HANDS_FREE_CODE_LABEL`, retries and timeouts work as in a cycle, and the state returns to
  `"armed"` afterwards. `resume_auto()` (posted the same way) runs an armed task's planned steps
  when manual is switched back off -- no new checkpoint, `self._task` kept (not
  `_start_cycle(checkpoint=False)`, which is the deviation path and would overwrite it).
- Assistant (`exocortex/llm/assistant.py`): `_manual_advance()` (reads `cfg.handsfree.manual`
  live) is OR-ed into the automatic gates -- the spoken-question coalesce arm and `_fire_auto`,
  `_maybe_answer_screen_question`, `_maybe_auto_suggest`, `plan_typed` and
  `_maybe_explain_from_screen` -- and the notes auto-timer skips its tick. Note: broader than
  hands-free's own gating, which leaves spoken answers and notes running.
- Engine (`exocortex/app.py`): `EngineActions.set_manual_advance` flips the flag, pokes
  `handsfree.resume_auto()` on the way back to Auto (an armed task runs, as Auto would have run
  it), publishes `ManualAdvanceEvent`, logs; `EngineActions.manual_advance(kind)` routes to
  `handsfree.trigger_step` when the controller is on and the engine not paused, else a plain
  `text()`/`code()` (which, with hands-free on, still get the cumulative context --
  `Assistant._hands_free_context` keys on the mode, not the caller). `apply_settings` handles
  `"handsfree.manual"` so the settings dialog syncs the switch through the same event.
- Hook (`exocortex/ui/llhooks.py`): `PressHoldMouseHook` (WH_MOUSE_LL), a sibling of
  `MouseWheelHook`: arms on WM_L/MBUTTONDOWN (position + `clock()`), fires ONCE at the hold time
  through an injected `schedule(delay_s, check)` (the overlay passes `QTimer.singleShot`; the
  module stays Qt-free) or on release without one; cancelled by movement past `move_px` (12 px),
  any other button, or a wheel notch; a quick middle-click fires when `middle_tap` is on;
  injected input (`LLMHF_INJECTED`) ignored; `hold_s` / `middle_tap` may be callables read at
  each press. Never swallows a message.
- UI (`exocortex/ui/overlay.py`): `manual_label` + `manual_switch` (a `ToggleSwitch`) left of the
  hands-free pair in the tab strip, enabled on every tab and kept clickable in the hands-free
  view (`_hands_free_clickable_at`); `_set_manual` mirrors the switch, installs/uninstalls the
  hook (only on the real Windows platform -- offscreen tests never hook the real mouse), notifies
  the engine and toasts; the slim bar's Context menu gains a checkable "Manual advance" entry;
  `ManualAdvanceEvent` syncs without echoing back; `_manual_fire` (deferred off the hook with
  `QTimer.singleShot(0, ...)`) drops a fire whose press was over the app's own UI
  (`_gesture_blocked_at`: the settings dialog / region picker always; the overlay itself only
  where it actually takes the click -- in the hands-free view just the title bar and the switches
  (`_hands_free_clickable_at`), with the user's click-through on nowhere, else its whole frame)
  and otherwise calls `actions.manual_advance(kind)` -- no window call of any kind on that path.
  The hook is uninstalled in `closeEvent`, and `PressHoldMouseHook.uninstall()` forgets any press
  in flight (its release is never seen once unhooked; stale state would read every later press
  as a chord).
- Settings (`exocortex/ui/settings.py`): three new Hands-free-tab fields -- `manual` (bool),
  `manual_hold_s` (float 0.2-3.0, step 0.1), `manual_middle_tap` (bool).
- Tests: `tests/test_manual_advance.py` (the hook driven through the fake user32 with an
  injectable clock and a hand-run scheduler, incl. uninstall-mid-press; controller arm /
  trigger_step / mid-cycle flip / resume_auto / no-deviation-regen / paused; assistant gating on
  the live flag; engine routing + resume-on-Auto + event + apply_settings; config round-trip
  incl. an old file without the fields; settings fields),
  `tests/test_manual_advance_early_timer.py` (a coarse timer firing early re-schedules the
  remainder) and `tests/test_overlay_manual.py` (the switch, the spy hook lifecycle, the hotkey
  binding, the menu entry, `_manual_fire` routing incl. the hands-free click-through and
  click-through-mode cases / no window-state change); `tests/test_overlay_handsfree.py`'s
  tab-strip order test now expects the Manual pair between the stretch and the hands-free pair.

### Manual advance: per-action press-and-hold + injected taps (2026-09-25, later)

Field feedback from the first build: the touchpad three-finger tap never fired. Two causes, both
fixed here; this supersedes the previous section's `manual_middle_tap` design.

- A tap cannot be held, so the hold requirement is now PER ACTION: `HandsFreeConfig.manual_hold_text`
  (default True -- left clicks happen constantly and must never generate; the settings help says
  in capitals what unchecking means) and `manual_hold_code` (default False -- a quick middle
  press, or the touchpad tap, triggers Code right away). `manual_middle_tap` is REMOVED (an old
  config carrying it just logs the unknown-key warning); `manual_hold_s` applies to whichever
  buttons still require the hold, and movement past `move_px` cancels a press whether or not its
  hold is required.
- Windows delivers the three-finger-tap middle click through `SendInput`, so it arrives at the
  WH_MOUSE_LL hook with `LLMHF_INJECTED` set -- the first build's injected-input filter ate it
  even on a quick click. `PressHoldMouseHook` now processes injected input like real input (the
  filter is gone; a comment in `_on_message` and the class docstring say why). The residual
  exposure -- synthetic input from automation firing a generation -- is bounded by the hold
  requirement on the left button and the rarity of synthetic middle clicks.
- `PressHoldMouseHook`: the `middle_tap` ctor param is replaced by `require_hold` (a bool for
  all buttons, a `{"left": bool, "middle": bool}` mapping, or a callable taking the button name,
  read at each RELEASE so a settings change applies live). The overlay passes a callable over
  `cfg.handsfree.manual_hold_text` / `manual_hold_code`. `ARMED_DETAIL` now reads
  "hold left = Text, click middle = Code" to match the defaults; tooltips / toast / README /
  SHORTCUTS / config.example.toml updated to the same wording; the settings tab exposes the two
  checkboxes ("Press and hold for Text (left button)" / "... for Code (middle button)").
- Tests updated in `tests/test_manual_advance.py` (quick middle-click keyed off
  `manual_hold_code`; quick left click fires only with its hold unchecked; an injected middle
  click fires like a real one; `require_hold` and `hold_s` read live; config round-trip incl. an
  old file still carrying `manual_middle_tap`) and `tests/test_overlay_manual.py` (the
  `require_hold` callable maps left/middle onto the two config fields).

### GPU vendor routing: AMD / Intel iGPU support, Ollama, one run script (added 2026-09-25/26)

Everything in this section was developed and measured on a second dev box added 2026-09-25: an
ASUS ROG Flow Z13, AMD Ryzen AI MAX+ 395 (16C/32T Zen 5), Radeon 8060S iGPU (RDNA 3.5, gfx1151,
UMA: DXGI 32590 MiB dedicated + 16203 MiB shared; llama.cpp's own Vulkan device report agrees at
48971 MiB), **no NVIDIA driver installed at all**. HARD RULE the whole branch was reviewed
against (owner, 2026-09-25): **every code path taken when `gpu_vendor() == "nvidia"` is
byte-identical to before this branch** -- the RTX 5080/5090 laptops' CUDA ladder, `--device`
behaviour, thread counts and `[llm].backend = "auto"` resolution are unchanged; AMD/Intel/Ollama
are additions layered next to that path, never a rewrite of it. The "Verified facts" table above
holds the dated measurements this section's decisions rest on; this section holds the module
contracts and the WHY.

- **GPU discovery (`exocortex/gpu.py`, new module).** `detect_gpus(*, refresh: bool = False) ->
  list[GpuInfo]` enumerates hardware display adapters via DXGI + a throw-away D3D12 device (no
  dependency, ctypes only): `GpuInfo(index, name, vendor, vendor_id, device_id, dedicated_mb,
  shared_mb, uma, luid, software=False)`, `.integrated` (== `uma`), `.budget_mb` (dedicated-only
  on a discrete card; dedicated+shared on a UMA iGPU, where both halves are the same DRAM).
  `primary_gpu(gpus=None) -> GpuInfo | None` picks a discrete GPU with the most dedicated memory
  over any integrated one (unchanged: an RTX next to an Intel iGPU still targets the RTX), else
  the biggest integrated adapter. `gpu_vendor(gpus=None) -> str` names its vendor
  (`"nvidia"`/`"amd"`/`"intel"`/...), falling back to an NVML handle-acquisition probe as proof of
  an NVIDIA GPU when DXGI enumerated nothing (a driver hiccup must not demote an NVIDIA laptop to
  the CPU/Vulkan paths). `primary_budget_mb(gpus=None) -> int | None` sizes `[llm].model = "auto"`.
  `vram_free_mb()`/`vram_total_mb()`/`vram_source()` extend the existing NVML VRAM-guard reading
  with a **PDH fallback** for non-NVIDIA GPUs (`\GPU Adapter Memory(luid_<luid>_phys_*)\Dedicated
  Usage`, matched by LUID, vendor-neutral -- the same "GPU Adapter Memory" counter Task Manager
  reads): an NVIDIA primary with NVML unavailable never falls through to PDH (NVML stays
  authoritative for NVIDIA; that case is "no reading", not a guess). `VramGuard` itself
  (hysteresis/cooldown alert logic) is unchanged, just fed a vendor-neutral reading now.
  `extras_for_vendor(vendor) -> list[str]` maps a vendor to the `requirements-*.txt` files
  `run.bat` should install on top of the core (`{"nvidia": ["requirements-nvidia.txt"], "amd":
  ["requirements-directml.txt"]}`; Intel/no-GPU: `[]`, since `requirements.txt` already bundles
  OpenVINO). `python -m exocortex.gpu` (`--vendor` | `--json` | `--extras`) is the CLI `run.bat`
  and the doctor shell out to; always exits 0 (a batch file has no exception handling).
- **llama.cpp runtime pack / device / model selection (`exocortex/llm/server.py`, extended).**
  `backend_ladder(vendor: str, requested: str = "auto") -> tuple[str, ...]`: NVIDIA's ladder
  (`nvidia-cuda12, nvidia-cuda, vulkan, cpu`) is the exact pre-existing order, unchanged; AMD's is
  `amd-rocm, vulkan, cpu` (ROCm first: measured 2026-09-26 on the Radeon 8060S with the 16.8 GB Qwen3.8-27B Q4_K_M, machine idle, ctx 8192, 2 slots: ROCm 2.29.1 processes a fresh 2651-token prompt at ~300 tok/s and answers a cached prompt in 0.4 s, Vulkan 2.46.0 at ~160 tok/s and 3.4 s; generation is equal (11.2 vs 12.4 tok/s); an incremental 953-token append runs at ~273 vs ~92 tok/s; LM Studio only installs
  the ROCm pack for the `gfx` targets it supports -- the 780M is not one -- and `start()` falls
  through to Vulkan, which runs on every AMD GPU, when a ROCm pack lists no device or dies while
  loading -- see the fall-through paragraph below); Intel is
  `vulkan, cpu`; `"none"` (DXGI enumerated no adapter -- not proof of a GPU-less machine, could be
  a driver hiccup) keeps the pre-vendor-aware NVIDIA-first order, since whichever pack LM Studio
  actually installed is itself evidence of the hardware. `[llm].llama_backend` ("auto" | "cuda" |
  "rocm" | "vulkan" | "cpu") moves a flavor to the front of the vendor ladder rather than
  replacing it (a missing pack still falls back down the same ladder with a warning); `"cpu"` is a
  hard stop, no GPU fallback. `find_server_binary(explicit="", *, vendor=None, backend="auto",
  exclude=frozenset()) -> ServerBinary` (`exclude`: flavors that already failed on this machine) scans `~/.lmstudio/extensions/backends/llama.cpp-*` (`_scan_packs`, manifest-
  driven flavor detection with a directory-name fallback, `_flavor_of`/`_flavor_from_dirname`),
  picks the newest version of the ladder's first present flavor, and raises `LlamaServerError`
  naming what to install for *this* vendor when none is. Picking the wrong pack is not slow, it is
  a crash (LM Studio's CUDA `llama-server.exe` exits `0xC0000135`, DLL not found, with no NVIDIA
  driver) -- this ladder is what a vendor-blind "prefer CUDA" picker used to produce. Vendor DLL
  directories (`_vendor_names_for`) resolve from the manifest's `vendor_lib_package_names` when
  present, else a fixed map for CUDA/CUDA12/Vulkan, else -- ROCm's vendor package is versioned --
  the newest `win-llama-rocm-vendor-v<N>` found by globbing, plus its `bin` subfolder when present.
  `list_devices(binary, timeout_s=20.0) -> list[LlamaDevice]` runs `--list-devices` and
  `parse_list_devices()` parses `"Vulkan0: AMD Radeon(TM) 8060S Graphics (48971 MiB, 46523 MiB
  free)"`-style lines (tolerant of CRLF, extra log lines, a missing memory parenthetical).
  `choose_device(devices, primary, requested="auto", *, flavor) -> str | None`: `None` (no
  `--device` flag) for a CUDA flavor or <=1 device (unchanged NVIDIA behaviour, and this box's own
  single-adapter case), else the device matching `primary`'s name, else the one with the most
  `total_mb`. `pick_default_model(models, budget_mb) -> GgufModel | None`: the largest GGUF within
  75% of the GPU budget (headroom for KV cache/activations), or the smallest model if none fit, or
  the largest overall with no GPU budget known; `choose_model(spec, models_dir, budget_mb) ->
  tuple[GgufModel, str]` resolves `"auto"`/`""` this way and falls an unresolvable named spec
  (including the bare default `"maple-preview"` on a machine that lacks it) back to `"auto"` with
  a warning instead of failing the start. `media_kind(path) -> (kind, filesystem, root)`
  (`GetDriveTypeW`/`GetVolumeInformationW`) and `estimate_startup_timeout_s(configured_s,
  size_bytes, kind) -> float` (never less than `[llm].llama_startup_timeout_s`, nor less than a
  cold read of `size_bytes` at the volume kind's read floor + 60 s slack) exist because a 16.8 GB
  GGUF on this box's removable exFAT card measured a 323 s cold load -- past the 180 s default --
  while the same file on a fixed disk estimates 117 s (the default still stands). `default_threads()`
  now counts *physical* cores (`_physical_cores()`, `GetLogicalProcessorInformationEx`, walking
  variable-size records by their `Size` field) instead of assuming SMT via `cpu_count() // 2`.
  `scan_log_warnings(text) -> list[str]` turns two llama-server.log patterns into operational
  warnings surfaced by `LlamaServerManager.warnings` / the doctor: `cache_reuse is not supported`
  (measured on every run with `--parallel 2` -> `--kv-unified`, both Vulkan and ROCm) and `cannot
  be used with preferred buffer type` (a tensor with no GPU kernel in this build falling back to
  CPU). `LlamaServerManager.build_args()` forces the V-cache type to `"f16"` (K stays quantized)
  when `flash_attn` is off and `kv_cache_type` is quantized -- llama-server itself hard-refuses to
  start otherwise (`quantized V cache requires flash_attn to be enabled`; measured on the Vulkan
  pack), so what used to be an obscure exit code is now a working start plus a log warning.
- **Fall-through between runtime packs (`LlamaServerManager.start()`, 2026-09-26).** `start()` is now a loop over
  `find_server_binary(..., exclude=<flavors that failed>)` + `_attempt()` (the old body): when the pack is a GPU
  framework other than CUDA and came from the ladder (not an explicit `llama_server_path`), a pack whose
  `--list-devices` is empty is refused before spawning (`"<framework> llama-server lists no GPU device on this
  machine"`), and one that exits or times out while loading is put aside; the next flavor of the ladder is tried,
  at most three packs per `start()`, each skip recorded in `mgr.fallbacks` ("<flavor> <version>: <error>") and
  reported through `on_status`. When nothing is left the ORIGINAL error is raised. CUDA packs and explicit paths
  raise at once, exactly as before (NVIDIA behaviour and an explicit choice are never second-guessed).
  `sniff_framework(exe_dir) -> "cuda" | "rocm" | "vulkan" | "cpu" | "custom"` reads the `ggml-*.dll` next
  to an explicit `llama_server_path` so a hand-picked CUDA folder is treated like a CUDA pack (never given
  `--device`). Tests: `test_start_falls_through_from_rocm_to_vulkan_when_rocm_dies`,
  `test_start_reports_the_original_error_when_every_pack_fails`, `test_start_never_falls_through_on_cuda`,
  `test_start_never_falls_through_on_an_explicit_path`, `test_attempt_refuses_a_gpu_pack_that_lists_no_device`,
  `test_sniff_framework_*`, `test_find_server_binary_explicit_cuda_folder_is_recognised_as_cuda`.
- **Backend route (`exocortex/llm/route.py`, new module).** `resolve_route(cfg, *, vendor=None,
  find_binary=None, scan=None, http_get=None, start_ollama=False, ollama_launcher=None,
  clock=None, sleep=None) -> Route` is the single place `[llm].backend = "auto"` gets decided:
  llamacpp (a runtime pack for this vendor + >=1 GGUF in the library) -> ollama (already running
  at `[llm].ollama_url`, or started here when `start_ollama` and `ollama_installed()`) -> openai
  (any server answering at `[llm].base_url`) -> `Route("none", ..., reason=...)` naming what to
  install for *this* GPU vendor. HARD RULE (owner, 2026-09-25): an explicit `[llm].backend`
  (`"llamacpp"`/`"ollama"`/`"openai"`) is returned as-is with **no probing** -- this is what keeps
  a tuned NVIDIA machine's behaviour exactly as before once the vendor-aware pieces land; "auto"
  simply now also succeeds on machines that used to have no viable backend at all. Every probe
  (`find_binary`, `scan`, `http_get`, the Ollama launcher, `clock`/`sleep`) is injectable so the
  whole rule set is unit-testable without touching a real machine; `resolve_route` never raises --
  a probe that blows up is "not available", same as one that answers "no" cleanly. `Route.checked`
  keeps one line per probe for `describe_route()` / `--doctor`.
- **`OllamaClient` (`exocortex/llm/client.py`, extended).** Subclasses `LMStudioClient`, reusing
  its streaming/cancellation/`<think>`-splitting/stats plumbing; only discovery, health, warm-up
  and the two request bodies differ because Ollama's native endpoints are not OpenAI-shaped:
  `list_models()` (`GET /api/tags` + `GET /api/ps` for what's actually loaded -- Ollama can be
  reachable with nothing resident), `health()` (auto-picks the largest model fitting 75% of the
  GPU budget when `cfg.model` is `"auto"`/`""`), `ensure_loaded()` (`POST /api/generate` with an
  empty prompt and `keep_alive: "30m"`), `stream_completion()`/`stream_chat()` (`POST
  /api/generate` `raw: true` / `POST /api/chat`, NDJSON one-object-per-line, no SSE framing).
  `make_client(cfg, backend) -> LMStudioClient` picks `OllamaClient` for `"ollama"`,
  `LMStudioClient` otherwise. A mid-stream `{"error": ...}` NDJSON line now raises `LLMError`
  (a review fix, 2026-09-26) instead of silently ending the stream as if it had completed cleanly.
- **ASR engine split (`exocortex/speech/asr.py`, `exocortex/models_dl.py`, extended).**
  CTranslate2 (faster-whisper's backend) has no ROCm/Vulkan/DirectML device, so every non-NVIDIA
  machine ran Whisper on the CPU already -- what's new is a second, faster CPU engine.
  `resolve_engine(cfg, cuda_available=_cuda_available_default) -> str`: an explicit
  `"faster-whisper"`/`"sherpa"` is returned unchanged; `"auto"` resolves to `"faster-whisper"` when
  `cfg.device == "cuda"` or `cuda_available()` (the exact CUDA probe `Transcriber` itself uses --
  so this never changes anything on the NVIDIA laptops), else `"sherpa"`. `resolve_sherpa_model(cfg)
  -> str` swaps the English-only default bundle for the multilingual `-v3-int8` one when
  `[asr].language` is set to non-English and `sherpa_model` is still at its default (never
  overrides an explicit choice -- the same "soft default" pattern as `AssistConfig.primary_language`).
  `SherpaTransducerModel(model_dir, num_threads, provider="cpu")` wraps a sherpa-onnx
  `OfflineRecognizer.from_transducer` over a NeMo Parakeet-TDT export, satisfying the same
  `_ModelLike.transcribe(audio, **kwargs) -> (segments, info)` duck type as `WhisperModel` (ignores
  faster-whisper-only kwargs, logged once at debug), so `Transcriber.transcribe()`'s drop rules
  needed no change. `Transcriber.__init__` gained `models_dir`, `progress`, `sherpa_model_factory`
  and `ensure_sherpa_model` kwargs (all injectable for tests); `load()` now dispatches on
  `resolve_engine()` to `_load_faster_whisper` (unchanged body -- same factory, same kwargs, same
  CPU fallback, as required for the NVIDIA path) or `_load_sherpa` (downloads/extracts the bundle
  via `exocortex.models_dl.ensure_sherpa_model` first); `engine_used`/`model_used` properties
  record which. `default_cpu_threads(vendor=None, physical=None) -> int`: `"nvidia"` always
  returns 4 (unchanged -- the CPU also feeds llama-server's CPU-bound TQ1_0 tensors there); every
  other vendor gets `max(4, min(8, physical // 2))` -- 8 on this 16-core box, chosen because 4
  threads did not keep large-v3-turbo comfortably under the utterance length on it.
  `exocortex.models_dl.SHERPA_MODELS` (a name -> k2-fsa/sherpa-onnx release URL map) and
  `ensure_sherpa_model(name, models_dir, progress=None) -> Path` mirror `ensure_models()`'s
  streaming-download/validation pattern but for a `.tar.bz2` archive:
  `_safe_extract_tar_bz2()` rejects absolute paths, `..` components, more than one top-level
  directory, and (belt-and-suspenders) any member whose resolved path would land outside the
  extraction root, before calling `extractall(..., filter="data")`; idempotent
  (`_sherpa_model_ready()` checks for `encoder*.onnx`/`decoder*.onnx`/`joiner*.onnx`/`tokens.txt`
  before ever touching the network) and self-cleaning (a failed extraction deletes the archive
  rather than leave a corrupt one for the next idempotence check to trip over).
- **OCR device routing (`exocortex/screen/ocr.py`, extended).** OpenVINO's `GPU` plugin crashes
  the interpreter outright on this AMD iGPU (`STATUS_STACK_BUFFER_OVERRUN` compiling Intel
  GEN/vISA inline assembly the AMD OpenCL ICD cannot parse -- see the Verified facts table), so
  `resolve_ocr_device()` now deliberately excludes a non-Intel OpenVINO `GPU` device from the
  `"igpu"` step and offers `directml_available() -> bool` (`"DmlExecutionProvider" in
  onnxruntime.get_available_providers()`, cached, never raises) instead:
  `_pick_dml_adapter(gpus) -> (device_id, is_uma) | None` refuses when any detected adapter is
  NVIDIA (that GPU is never OCR's, unchanged) and otherwise picks adapter 0 for a single GPU, or
  -- for more than one -- the last index if one of them is UMA (a heuristic about DirectML's
  `DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE` ordering, unverified on a real multi-GPU-plus-AMD-iGPU
  machine; see README Limitations). The resolution order for `"igpu"`/the igpu step of `"auto"` is
  now: (a) an Intel-named OpenVINO `GPU` device (unchanged for Intel hardware) -> (b) DirectML on
  a non-NVIDIA adapter -> (c) CPU. `describe_ocr_device()` reports which (`"igpu via DirectML
  device 0"` vs `"igpu via OpenVINO GPU"`). `requirements-directml.txt` pins
  `onnxruntime-directml==1.24.4`, which *replaces* plain `onnxruntime` (same import path); `run.bat`
  uninstalls the CPU package first on an AMD machine so a later `pip install -r requirements.txt`
  never silently reintroduces it.
- **Doctor (`exocortex/doctor.py`, extended).** `_check_gpu` now prints vendor-neutral discovery
  (`gpu.describe()`, primary adapter + vendor, the VRAM reading and which source it came from) and
  only runs the CUDA-specific checks on an NVIDIA primary; every other vendor gets an
  engine-/model-aware ASR line (`resolve_engine()`/`resolve_sherpa_model()` -- names whichever
  engine "auto" actually picked, not just "Whisper"). `_check_llm` resolves the route first when
  `[llm].backend == "auto"` (`route.resolve_route`) and dispatches on the *resolved* backend;
  `_check_llm_ollama` (new) builds an `OllamaClient`, checks reachability, lists models and times a
  completion, same shape as the existing llamacpp/openai checks. Every probe that depends on
  another module's contract is wrapped so a missing/failing piece degrades to a WARN line rather
  than crashing the whole report.
- **`run.bat` / `run.vbs` (extended).** Python discovery tries `py -3.12` / `-3.13` / `-3.11`,
  then a bare `python` if its version is in range, and only then bootstraps a private interpreter
  via [uv](https://astral.sh/uv) (`uv python install 3.12` + `uv venv --seed`) -- unchanged
  behaviour, now exercised on a machine with no Python at all in `tests/test_run_script.py`.
  `:ensure_deps` installs the vendor-neutral core (`requirements.txt`) plus this machine's extras
  from `exocortex.gpu --extras` (`requirements-nvidia.txt` on NVIDIA, `requirements-directml.txt`
  on AMD, nothing on Intel/no-GPU -- `:install_extra` uninstalls plain `onnxruntime` first for the
  DirectML case), then writes a stamp (`.venv\exocortex-setup.stamp`) recording the vendor and
  date. A later run's freshness check now ALSO re-probes `exocortex.gpu --vendor` (a stdlib-only,
  few-ms call) and treats a mismatch against the stamp's recorded vendor as staleness -- a laptop
  whose primary GPU changes between runs (driver update, hybrid-graphics/Optimus switch, a dock
  change) would otherwise read "Setup is up to date." forever and never get the new vendor's
  extras. `:bootstrap_uv_python` reuses a cached `.tools\uv\uv.exe` instead of re-downloading it on
  every fresh-`.venv` setup. `run.vbs` (a silent double-click launcher) used to only check whether
  the stamp file *existed*; it now replicates `run.bat`'s own staleness check (the same watched
  files, plus the vendor recheck) in VBScript, so it only shows a visible console on a genuinely
  fresh install (no `.venv\Scripts\pythonw.exe` at all) and otherwise runs `run.bat --setup-only`
  for real, just with its window hidden, when setup is stale -- an already-working install no
  longer sees an unexpected console just to catch a `requirements*.txt`/vendor change. (Bug found
  and fixed along the way: VBScript's `Trim()` only strips spaces, not CR/LF, so a subprocess's
  captured stdout needed explicit `Replace(..., vbCr, "")`/`Replace(..., vbLf, "")` before the
  vendor-word comparison -- caught against the real `cscript.exe`, not just by reading the code.)
- **Config (`exocortex/config.py`).** New `LLMConfig` fields: `ollama_url` (default
  `"http://127.0.0.1:11434"`), `llama_backend` (default `"auto"`), `llama_device` (default
  `"auto"`). New `ASRConfig` fields: `engine` (default `"auto"`), `sherpa_model` (default
  `"parakeet-tdt-0.6b-v2-int8"`); `cpu_threads`'s default changed from a hardcoded `4` to `0`
  ("auto" -> `default_cpu_threads()`, vendor-aware). A regression caught in review and fixed
  2026-09-26: `LLMConfig.model`'s default was accidentally swapped from `"maple-preview"` to
  `"auto"` alongside `backend`'s (correct) swap to `"auto"` -- reverted to `"maple-preview"` so a
  fresh/no-config install on an NVIDIA machine still loads exactly the model this app was verified
  against; `backend = "auto"` stayed, since `resolve_route()` already resolves it to `"llamacpp"`
  on any working NVIDIA install (existing tests lock this in).
- **Tests.** New: `tests/test_gpu_detect.py` (DXGI/D3D12 discovery, vendor mapping, budget_mb,
  `extras_for_vendor`), `tests/test_route.py` (`resolve_route`'s full rule order incl. the
  explicit-backend no-probing rule, injected probes), `tests/test_client_ollama.py`
  (`OllamaClient` against `httpx.MockTransport`, incl. the mid-stream-error regression),
  `tests/test_doctor.py` (vendor-aware GPU/LLM sections, incl. a regression pinning
  `LLMConfig.backend`/`model` defaults), `tests/test_asr_sherpa.py` (30 tests: `resolve_engine`,
  `resolve_sherpa_model`, `SherpaTransducerModel` via a fake `sherpa_onnx` module, `Transcriber`'s
  sherpa path via fakes), `tests/test_run_script.py` (`run.bat`/`run.vbs` via `EXO_SETUP_DRYRUN`
  and real `cscript.exe` invocations against safe stand-in executables, never a real Python
  interpreter). Extended: `tests/test_server.py` (backend ladder, pack scanning, device selection,
  `choose_model`, startup-timeout scaling, the flash_attn/kv_cache_type crash-avoidance fix),
  `tests/test_screen_ocr_device.py` (DirectML resolution, the AMD-GPU-plugin exclusion,
  `_pick_dml_adapter`), `tests/test_models_dl.py` (10 new: sherpa download/extract/idempotence/
  traversal-rejection/multi-top-dir-rejection/undersized/unknown-name), `tests/test_gpu_guard.py`
  and `tests/test_instance_swap.py` (vendor-neutral VRAM reading plumbed through unchanged guard
  logic), `tests/test_settings.py` / `tests/test_app.py` / `tests/test_pipeline_extra.py` (config
  round-trips and wiring for the new fields). Full repository suite after every fix in this branch
  landed: 1955 passed, 0 failed, 0 errors, 0 skipped (`.venv\Scripts\python.exe -m pytest -q`, ~
  3.5 minutes) -- run offline; `-m live` tests (GPU/audio/llama-server) are additional and were
  exercised by hand on this AMD box (see the Verified facts table) rather than in that count.

### Intel Arc discrete GPUs, upstream llama.cpp runtimes, quit-during-load fix (added 2026-09-26)

No Intel Arc GPU (A770/A750 "Alchemist", B580/B570 "Battlemage", Pro B60/B70) exists on either dev
box this branch was built on (the RTX 5080/5090 laptops; the Ryzen AI MAX+ 395 / Radeon 8060S box
above) -- everything Intel-Arc-specific below is a unit-tested design against synthetic `GpuInfo`
/ `--list-devices` data, not a live measurement, and is labelled as such throughout. What WAS run
live, on the AMD box, with no Intel hardware involved at all: the upstream-runtime fetch/spawn
mechanism end to end (Vulkan), the SYCL fall-through as a negative control (no device -> falls
through cleanly), and the quit-during-load fix -- see the three new "Verified facts" rows above.
HARD RULE this branch was reviewed against (owner, 2026-09-26): **NVIDIA and AMD behaviour is
unchanged, including at the "auto"-vs-upstream boundary** -- see the vendor gates below, which are
narrower than the first cut of this branch (a review caught and fixed two places where they had
not been applied strictly enough; see "Fixes applied after review"). *Dated note (2026-09-27): the
owner's later "Portable install + model manager" section below lifts this exact boundary for
NVIDIA and AMD -- a fresh machine with no LM Studio pack for either vendor now gets an upstream
fetch instead of an immediate offline error. What this HARD RULE still guarantees, unchanged: a
machine that DOES have an LM Studio pack for the flavor it lands on is byte-identical to before
either branch, no network call, ever. Read the vendor-gate bullets just below as the 2026-09-26
boundary; the 2026-09-27 section states the current one.*

- **A second runtime source (`exocortex/llm/runtimes.py`, new module).** LM Studio's own runtime
  catalog packages CUDA/ROCm/Vulkan/CPU only -- no SYCL or OpenVINO pack exists there for any
  vendor, so an Intel Arc card only ever gets LM Studio's Vulkan pack today, which matches the
  llama.cpp community's own recommendation for Arc anyway (Vulkan over SYCL, as of 2026-07;
  GitHub discussion #12570 measures a B580 at ~30 tok/s on Qwen3-8B Q4_K_M; discussion #13530
  covers the Xe2 cooperative-matrix driver threshold). This module fetches upstream llama.cpp's
  own self-contained Windows release zips (`github.com/ggml-org/llama.cpp`) directly, so a SYCL or
  OpenVINO build (or a newer Vulkan than LM Studio ships) is one call away instead of a manual
  install. It is a SEPARATE source from LM Studio's packs and never touches `server.py`/`config.py`
  itself -- callers ask it for a folder holding `llama-server.exe` and hand that folder to
  `server.py` the same way an explicit `[llm].llama_server_path` is used today.
  - `RUNTIME_KINDS: tuple[str, ...] = ("vulkan", "sycl", "openvino", "cpu")` -- CUDA and ROCm are
    deliberately absent: NVIDIA and AMD keep their tuned LM Studio packs; this module never offers
    an alternative for either. *Dated note (2026-09-27): superseded -- `RUNTIME_KINDS` grew
    `"cuda12"`, `"cuda13"` and `"rocm"` in the "Portable install + model manager" section below,
    each fetchable the same way once no LM Studio pack covers that flavor.*
  - `PINNED_BUILD = "b11200"`; `PINNED_ASSETS: dict[str, RuntimeAsset]` records that build's four
    Windows assets by hand (fetched from the GitHub release API on 2026-09-26): `win-vulkan-x64`
    33,062,078 bytes, `win-sycl-x64` 120,815,015 bytes (SYCL.md: self-contained, no separate oneAPI
    install), `win-openvino-2026.4-x64` 88,333,295 bytes, `win-cpu-x64` 19,154,722 bytes, each with
    its published sha256. Installing the pinned build never makes a network call before the
    download itself; any other `build` resolves its asset from the release API at call time.
  - `RuntimeAsset(kind, build, name, url, size, sha256)` (frozen dataclass) -- everything needed to
    download an asset and prove it arrived intact. `UpstreamRuntimeError(RuntimeError)` is the only
    exception this module raises on top of `OSError` (see "Fixes applied after review" below for
    what changed about that boundary).
  - `ensure_runtime(kind, root, *, build=PINNED_BUILD, ...) -> Path`: downloads (if not already
    installed), verifies BOTH exact byte size and sha256 before extraction (integrity is mandatory,
    not advisory), and refuses outright -- never installs on a "probably fine" basis -- any asset
    that a non-pinned release's API response does not itself attest a sha256 digest for. Extraction
    (`_safe_extract_zip`) guards against zip-slip, absolute member names, and unix-symlink-mode
    entries, the same threat model as `exocortex.models_dl`'s tar.bz2 extraction, adapted for zip's
    `ZipInfo` (no "exactly one top-level directory" rule, since upstream's zips are not consistent
    about wrapping their contents in one).
  - `installed_runtime(root, kind, build=PINNED_BUILD) -> Path | None`: an already-installed
    build's folder (searching one level deep for whichever layout the zip happened to use), or
    `None` -- the "already on disk" check `find_server_binary` makes before ever fetching.
  - `describe_runtime(folder) -> str` and `vulkan_env_switches(folder) -> list[str]`: the latter
    scans the folder's `ggml-vulkan*.dll` for the `GGML_VK_*` environment-variable strings that
    exact build actually recognises (e.g. `GGML_VK_DISABLE_COOPMAT`, `GGML_VK_DISABLE_F16`), which
    the doctor prints -- so `[llm].llama_env` workarounds are never guesses at flags a given binary
    ignores.
- **`find_server_binary` gained a source dimension (`exocortex/llm/server.py`).** New signature:
  `find_server_binary(explicit="", *, vendor=None, backend="auto", exclude=frozenset(),
  source=None, upstream_root=None, upstream_build=None, ensure=None, stop_check=None) ->
  ServerBinary`. `source` ("auto" | "lmstudio" | "upstream"):
  - `"lmstudio"`: LM Studio packs only, for every flavor of `backend_ladder` -- the pre-2026-09-26
    behaviour, byte-identical, never fetches anything.
  - `"upstream"`: an already-installed upstream build (`installed_runtime`, under `upstream_root`,
    default `project_root()/.tools/llama.cpp`), else -- only for a flavor with an upstream kind
    (never CUDA/ROCm) and only when `ensure` was given -- a freshly fetched one; else the next
    flavor. The CPU pack falls back to an upstream CPU build too when LM Studio has none.
  - `"auto"` (default): LM Studio's pack first, for every flavor; when a GPU flavor has none, it is
    remembered as the first "fetchable" flavor (has an upstream kind) and the next flavor is
    tried. Only once EVERY GPU flavor of the ladder has been tried with no LM Studio pack found
    (including one already in `exclude`) does the remembered fetchable flavor get the same
    installed-or-fetch treatment as explicit `"upstream"` -- before the CPU pack, not after.
  - `stop_check: Callable[[], bool] | None` (added in the post-review fixer pass, finding #3
    partial close -- see below): polled between flavors, and once more before the deferred
    "auto"-fetch and the final CPU rung, so a concurrent `LlamaServerManager.stop()` lands within
    one flavor's worth of (fast, local-disk) work instead of only being noticed back in `start()`'s
    own loop. Raises `LlamaServerError("stopped")` the moment it returns true. Does NOT interrupt a
    single upstream download already in flight (see "Open issues" below).
  - **Vendor gate (finding #1, tightened by the post-review fixer -- see "Fixes applied after
    review"):** under the default `source="auto"`, a vendor of `"nvidia"` or `"amd"` NEVER reaches
    the upstream mechanism at all -- not the fetch, and not even the free "already installed on
    disk" check -- so those two vendors are byte-identical to before this field existed, not merely
    "no network call". `source="upstream"` (an explicit opt-in) is a different code path and is
    NOT gated this way; it is the user's own choice. *Dated note (2026-09-27): this blanket gate is
    superseded by the "Portable install + model manager" section below -- the gate now applies only
    when an LM Studio pack for that vendor's flavor exists (still byte-identical in that case);
    when none does, NVIDIA/AMD reach the same installed-or-fetch treatment as every other vendor.*
- **`ServerBinary.source: str = "lmstudio"`** (new field) -- `"lmstudio"` (LM Studio's own pack, the
  only source before this branch), `"upstream"` (an official llama.cpp release build fetched by
  `exocortex.llm.runtimes`), or `"custom"` (an explicit `[llm].llama_server_path`). Surfaced by the
  doctor's llama.cpp section alongside `describe_runtime()`/`vulkan_env_switches()` for an upstream
  binary.
- **The quit-during-load race (docs/UPSTREAM_SUGGESTIONS.md finding #1).** Before this fix,
  `start()`/`_attempt()` held `self._lock` across the blocking `_wait_ready()` poll, so a
  concurrent `stop()` (quitting mid-load) could not acquire the lock it needed and blocked for the
  REST of the startup timeout -- worse than before this branch, now that a size-scaled timeout can
  run up to ~10 minutes on slow/removable model storage. The fix:
  - A `threading.Event` (`self._stop_event`) is checked at the top of `start()`'s retry loop,
    before spawning (`_attempt`, both before `Popen` and again right after, see the TOCTOU fix
    below), and inside `_wait_ready()`'s poll loop (`_stop_event.wait(0.25)` in place of
    `time.sleep(0.25)`, so a stop lands within one 0.25 s tick instead of only at the next
    iteration's start).
  - **Exact stop-event contract: `stop()` leaves the event set; `start()` clears it.** `stop()`
    sets `self._stop_event` FIRST (before anything that could block) and deliberately does NOT
    clear it itself. `start()` clears it at its own top, before doing anything else. Clearing it
    inside `stop()` would race whichever `_wait_ready()` loop it just interrupted: if the clear ran
    before that loop's next `is_set()` check, the loop would never notice the stop and would poll
    all the way to the real timeout. Leaving the event set until the next `start()` closes that
    window completely.
  - `start()` then raises `LlamaServerError("stopped")` (the `_STOPPED` sentinel string, matched by
    value rather than by re-reading the event, since a concurrent `stop()` may have already cleared
    it for the NEXT `start()` by the time the loading thread's except-handler runs) promptly
    instead of a bogus "did not become ready" timeout, and without trying the next runtime pack.
  - The same event also cancels an in-flight upstream runtime DOWNLOAD (added after the review
    round): `start()`'s `_ensure` closure passes `cancelled=self._stop_event.is_set` to
    `exocortex.llm.runtimes.ensure_runtime(...)`, whose chunk loop polls it every 64 KiB and raises
    `UpstreamRuntimeCancelled` (a subclass of `UpstreamRuntimeError`) after deleting the `.part`
    file -- the file handle is closed first, since Windows refuses to delete an open file;
    `_try_upstream` turns that one subclass into `LlamaServerError("stopped")` while every other
    `UpstreamRuntimeError` / `OSError` still means "this runtime is unavailable, try the next
    flavor". Tests: `test_ensure_runtime_cancelled_mid_download_removes_the_part_and_installs_nothing`,
    `test_ensure_runtime_cancelled_before_the_first_byte`,
    `test_try_upstream_propagates_a_cancelled_download_as_stopped`,
    `test_try_upstream_treats_a_failed_download_or_oserror_as_unavailable`,
    `test_start_ensure_closure_passes_the_stop_event_as_the_cancel_hook`.
  - Live on the AMD box (2026-09-26): starting the slow-loading 16.8 GB Qwen3.8-27B (LM Studio's
    ROCm pack, a 600 s configured timeout) on a thread and calling `stop()` 3.0 s in, `start()`
    raised `LlamaServerError("stopped")` ~0.6 ms after the stop event was set. `stop()` itself took
    ~5.0 s to return to ITS caller -- dominated by the pre-existing, UNCHANGED
    `terminate() -> wait(5) -> kill() -> wait(5)` teardown of a process with a large in-flight mmap
    read from slow removable media, which this box's OS took just over the first 5 s budget to tear
    down. No `llama-server.exe` remained a few seconds later. The number that matters is the
    ABSENCE of the ~600 s block: before this fix, the same scenario would have hung `stop()` behind
    the whole remaining startup timeout, because `start()`/`_attempt()` held the lock `stop()`
    needs for its entire duration.
  - **`_TIMED_OUT` sentinel, added by the post-review fixer (finding #5).** A GENUINE
    `_wait_ready()` deadline timeout now raises a message prefixed `"timed_out: ..."`
    (`_TIMED_OUT = "timed_out"`), matched the same way `_STOPPED` is, so `start()`'s retry
    condition can tell a pack that merely outlasted its startup deadline (still loading, not
    crashed) apart from one that actually died while loading. Before this, a legitimately slow
    load that just hit `estimate_startup_timeout_s()` got a second full-length wait of the SAME
    pack before any fall-through -- up to doubling (chained across a ladder, ~4x) the worst-case
    time on exactly the slow-media boxes this branch targets.
  - **`_kill_hung_process()` / `_terminate_process()`, added by the post-review fixer (finding
    #4).** `_wait_ready()`'s deadline-timeout branch now calls a new `_kill_hung_process()` --
    the same grab-and-clear-`self._proc`-under-the-lock, `terminate() -> wait(5) -> kill()`
    teardown `stop()` performs, factored out as a shared `_terminate_process()` staticmethod --
    WITHOUT touching `self._stop_event`. Calling `self.stop()` from inside a genuine timeout (the
    pre-fix behaviour) would have set the stop event, which `start()`'s own retry/fall-through
    logic and `_attempt()`'s pre-Popen guard would then have wrongly treated as "the user asked to
    stop", short-circuiting the conservative retry AND the whole ROCm -> Vulkan -> CPU fall-through
    ladder on the very first genuine timeout. `stop()` itself is unchanged in behavior -- it still
    sets the event, then delegates teardown to the same shared helper.
  - **TOCTOU fix in `_attempt()` (finding #6).** A concurrent `stop()` landing strictly between the
    pre-`Popen` `_stop_event` check and the locked `self._proc = proc` assignment used to read
    `self._proc` as still `None` (Popen had not returned yet), terminate nothing, and return --
    leaving the just-spawned process tracked by `self._proc` with nothing left to clean it up.
    `_attempt()` now re-checks `self._stop_event` immediately after that assignment (same short
    lock discipline); on a lost race it tears the process down itself before raising `"stopped"`,
    exactly as `stop()` would have.
- **A conservative retry within a pack, Intel-only (vendor-gated by the post-review fixer, finding
  #2).** Nobody can test an Intel Arc driver's Vulkan flash-attention / quantized-KV support here,
  and older "Alchemist" drivers are documented to have fp16/garbage-output issues under llama.cpp's
  Vulkan backend. Before falling through to the NEXT flavor, a GPU pack (framework `rocm`,
  `vulkan`, `sycl` or `openvino` -- `_GPU_FRAMEWORKS_FOR_RETRY`) that exits while loading gets ONE
  retry of the SAME pack with `--flash-attn auto` and an f16 KV cache
  (`build_args(..., conservative=True)`), skipping the K/V-cache-type flags entirely under that
  mode. The retry condition is `vendor == "intel" and not timed_out and
  self.binary.framework in _GPU_FRAMEWORKS_FOR_RETRY and self.binary.flavor != "custom" and
  self.binary.flavor not in retried` -- gating on vendor, not just framework, because ROCm and
  Vulkan are frameworks AMD also uses: the first cut of this branch gated on framework alone,
  which retried AMD's ROCm pack too (and, narrowly, a Vulkan pack an NVIDIA machine fell back to),
  moving the AMD rocm > vulkan > cpu fall-through's attempt count and timing -- exactly what the
  owner's hard rule says must not happen. Restricting this to `vendor == "intel"` keeps NVIDIA/AMD's
  fall-through byte-identical to before this branch (their tests were updated to assert this: see
  "Tests" below).
- **`[llm].llama_env: dict[str, str]`** (new `LLMConfig` field) merges extra environment variables
  into llama-server's process, after `PATH` -- an Arc "Alchemist" workaround
  (`{"GGML_VK_DISABLE_COOPMAT": "1"}`, etc. -- see `vulkan_env_switches` above for the exact
  switches a given `ggml-vulkan.dll` build recognises) or SYCL's device selector
  (`{"ONEAPI_DEVICE_SELECTOR": "level_zero:0"}`). Framework `"sycl"` also gets
  `ZES_ENABLE_SYSMAN=1` by default (SYCL.md: enables free-VRAM reporting) unless the user's own
  `llama_env` already sets that key.
- **`gpu.driver_version(g, *, reg_reader=None, rank=0) -> str | None`** (new, `exocortex/gpu.py`)
  reads a display adapter's driver version out of the registry (`winreg`, stdlib; every `NNNN`
  subkey of the display class GUID scanned for a `MatchingDeviceID`/`DeviceInstanceID`/
  `HardwareInformation.*` match on `VEN_<vendor_id:04x>&DEV_<device_id:04x>`), verified live on the
  AMD box (`"32.0.31032.1003"`, matching Win32_VideoController). `rank` disambiguates two adapters
  sharing a (vendor_id, device_id) pair; cached per `(vendor_id, device_id, rank)` except when
  `reg_reader` is injected (tests never see a stale answer). It is Intel-only in `describe()`'s
  lines and the `--json` output (gated on `vendor == "intel"`) so NVIDIA and AMD output stays
  byte-identical to before this field existed.
  `gpu.intel_arc_notes(g, driver) -> list[str]` turns that driver string plus a discrete Intel
  `GpuInfo` into advisory lines: Vulkan-over-SYCL (llama.cpp's current community recommendation),
  the Xe2 (Battlemage) cooperative-matrix driver threshold (`32.0.101.6795` -- discussion #13530),
  and Alchemist's `GGML_VK_DISABLE_F16` folklore. `[]` for any non-Intel vendor or an integrated
  Intel adapter (the 140V/140T iGPUs already get ordinary Vulkan routing with no special-casing
  needed). Unit-tested only (`tests/test_gpu_detect.py`) -- no Intel Arc GPU exists on either dev
  box.
- **OCR rule (a2) (`exocortex/screen/ocr.py`, `resolve_ocr_device`).** An OpenVINO `GPU` device
  whose name says Intel AND `(dGPU)` (e.g. `"Intel(R) Arc(TM) B580 Graphics (dGPU)"`) is now
  accepted for the `"igpu"` step's `kind="gpu"` result, but ONLY when (a) found no Intel iGPU AND
  `gpus` (default `exocortex.gpu.detect_gpus()`) confirms no NVIDIA adapter shares the machine. The
  existing rule (a) already excludes an Intel `(dGPU)` name for the NVIDIA-laptop case that rule
  was written for (Arc 140T next to an RTX 5090: OCR must stay off the machine's real GPU); (a2)
  only reopens that exclusion once hardware enumeration proves rule (a)'s premise false -- there is
  no NVIDIA adapter to shadow, so the Arc card IS this machine's GPU (an Intel-only desktop, or an
  Arc dGPU next to a non-Intel iGPU). An EMPTY or failed `detect_gpus()` is inconclusive, not proof
  there is no NVIDIA adapter, so it blocks (a2) too (same posture as `_pick_dml_adapter`'s own
  `not gpus` guard). Rule (a)'s own no-suffix fallback is cross-checked against `gpus` for the same
  reason: when every Intel adapter reported is confirmed discrete, the no-suffix name is left for
  (a2)/(b) to route instead of bypassing the NVIDIA gate. `resolve_ocr_device()`'s full order for
  the `"igpu"` step is now: (a) Intel-named OpenVINO `(iGPU)` -> (a2) Intel-named OpenVINO `(dGPU)`,
  NVIDIA-gated -> (b) DirectML on a non-NVIDIA adapter -> (c) CPU. Unit-tested only
  (`tests/test_screen_ocr_device.py`) -- no Intel Arc hardware exists to verify live.
- **`exocortex/config.py` (`LLMConfig`), new fields.** `llama_source: str = "auto"`,
  `llama_upstream_build: str = "b11200"`, `llama_env: dict[str, str] = field(default_factory=dict)`
  -- see their code comments (quoted in the "Configuration" README section and `config.example.toml`)
  for the full contract; `llama_backend`'s comment was extended to document the `"sycl"`/`"openvino"`
  explicit opt-ins (Intel-only, upstream-only frameworks LM Studio's catalog never packages).
- **Fixes applied after review.** A code review against this branch's first cut found 6 confirmed
  issues in `server.py`/`config.py` and 2 in `runtimes.py`, all fixed without moving NVIDIA/AMD
  behaviour:
  - Finding #1 (nvidia/amd parity): the "auto"-path upstream-fetch gate is now checked in TWO
    places -- the deferred `fetchable_flavor` capture and the final CPU-pack fallback -- both
    excluding `vendor in ("nvidia", "amd")`, so those vendors never touch the upstream mechanism
    under "auto", not even the free "already on disk" check.
  - Finding #2 (amd parity): the conservative retry is gated on `vendor == "intel"`, not just
    framework (see above) -- AMD's rocm/vulkan/cpu fall-through is back to exactly 3 attempts.
  - Finding #4 (amd parity): `_kill_hung_process()` lets a genuine `_wait_ready()` timeout tear
    down the process without setting `_stop_event` (see above).
  - Finding #5 (retry-on-crash, not retry-on-timeout): the `_TIMED_OUT` sentinel (see above) keeps
    the conservative retry from firing on a pack that was simply still loading.
  - Finding #6 (TOCTOU leak): `_attempt()` re-checks `_stop_event` after the locked `self._proc =
    proc` assignment and tears down a process orphaned by a race with `stop()` (see above).
  - Finding #3 (in-flight download not interruptible) -- **partially closed, open issue below.**
  - Runtimes.py, finding #1 (unwrapped `OSError`): the install-dir `mkdir` and `_safe_extract_zip`
    are now wrapped and re-raised as `UpstreamRuntimeError` (with `shutil.rmtree(tmp_dir,
    ignore_errors=True)` cleanup on the extraction path), so a disk-full or `MAX_PATH` failure mid-
    install never leaks a raw `OSError` past this module's documented `UpstreamRuntimeError`/
    `OSError`-only contract, and never leaves a half-written directory behind.
- **Open issue: `stop()` during an in-flight upstream download still blocks.** `find_server_binary`'s
  `stop_check` closes the "stop() lands between two flavor attempts" gap (each flavor's local-disk
  check is fast), but `exocortex.llm.runtimes.ensure_runtime`'s `iter_bytes` download loop has no
  cancellation hook of its own -- a `stop()` during an ACTUAL in-flight multi-MB fetch (up to
  ~120 MB for SYCL) still blocks until that fetch finishes or fails. Closing this fully means
  adding a cancellation hook inside `runtimes.py` itself; nobody has done that yet.
- **Tests added.** `tests/test_runtimes.py` (23 tests: happy path, idempotence, size/sha256
  mismatch, traversal-member rejection, missing-server-exe rejection and cleanup, the two OSError-
  wrapping fixes and their cleanup, an install-race that yields the winner's folder, no-digest
  refusal, release-API resolution for a non-pinned build, `describe_runtime`/`vulkan_env_switches`
  incl. no-DLL/empty-folder cases). `tests/test_server.py` (extended): `backend_ladder`/
  `find_server_binary` Intel and upstream-source cases (`test_backend_ladder_intel`,
  `test_find_server_binary_intel_uses_vulkan`, `test_find_server_binary_upstream_source_*`,
  `test_find_server_binary_auto_never_touches_upstream_for_nvidia_or_amd`,
  `test_find_server_binary_lmstudio_source_never_fetches_even_for_sycl_backend`), device/model
  parsing for Arc names (`test_parse_list_devices_intel_arc_discrete_and_igpu_and_sycl_style_names`,
  `test_choose_device_intel_arc_*`, `test_pick_default_model_intel_b580_budget_*`), the conservative
  retry (`test_build_args_conservative_uses_flash_attn_auto_and_no_kv_cache_flags`,
  `test_start_conservative_retry_only_applies_to_intel`), the SYCL sysman default
  (`test_attempt_sycl_framework_defaults_zes_enable_sysman`,
  `test_attempt_llama_env_can_override_the_sycl_sysman_default`,
  `test_attempt_non_sycl_framework_gets_no_sysman_default`), and the quit-during-load/regression
  set (`test_start_raises_stopped_promptly_when_stopped_mid_load_and_never_tries_the_next_pack`,
  `test_start_falls_through_after_a_genuine_timeout_without_reporting_stopped`, plus the
  TOCTOU/retry-only-intel/upstream-never-for-nvidia-amd regressions listed under "Fixes applied
  after review"). `tests/test_gpu_detect.py` (extended): `driver_version`/`intel_arc_notes` --
  registry matching, case-insensitivity, rank disambiguation and caching, per-generation
  (Battlemage/Alchemist) advisory text, and the NVIDIA/AMD byte-identical-output guards
  (`test_describe_never_shows_driver_for_amd_or_nvidia_even_when_a_driver_is_found`,
  `test_main_json_omits_driver_key_for_amd_and_nvidia`). `tests/test_doctor.py` (extended): the
  Arc driver/notes lines, `ServerBinary.source` reporting, `[llm].llama_env` display, and the
  upstream-runtimes-on-disk listing. `tests/test_screen_ocr_device.py` (extended): rule (a2) --
  Arc-alone, Arc-beats-DirectML-on-AMD, NVIDIA-blocks-a2, iGPU-still-wins-over-a2, and the empty-
  `gpus`-fails-closed-to-CPU case.

### Portable install + model manager (added 2026-09-27)

HARD RULE the whole branch was reviewed against (owner, 2026-09-27): **a machine with NEITHER LM
Studio NOR Ollama installed gets a working local LLM from one `run.bat` run** -- the llama.cpp
runtime for its GPU, the speech models, and (an empty library) a prompt to pull a model, ending in
a loaded answer. LM Studio and Ollama stay fully optional, never required, and a machine that
already has an LM Studio pack for the flavor it lands on stays byte-identical to before this
section existed (no network call, no `ensure()` call, ever) -- this is what supersedes the older
"never for NVIDIA/AMD" carve-outs the 2026-09-25/26 sections above wrote when that guarantee still
depended on excluding two vendors from the upstream mechanism entirely; it now depends on an
LM-Studio-pack check instead, which is a strictly wider net (every vendor and flavor, not just
NVIDIA and AMD) that happens to produce the same byte-identical answer for those two vendors on
any machine that already has LM Studio. Five agents landed this section in the same working tree
at once, each owning one module and reading (never editing) the other four across the boundaries
listed under "Module contracts" below; "Verified facts" above holds this section's measurements,
this section holds the contracts and the why.

**Module contracts (owner per module, read-only across the others):**

- **`exocortex/llm/runtimes.py` (extended).** `RUNTIME_KINDS: tuple[str, ...] = ("vulkan", "sycl",
  "openvino", "cpu", "cuda12", "cuda13", "rocm")` -- three new kinds alongside the 2026-09-26 four.
  `PINNED_ASSETS: dict[str, tuple[RuntimeAsset, ...]]` changed shape from one asset per kind to a
  TUPLE: every existing kind still gets one, but `"cuda12"`/`"cuda13"` each get TWO --
  `llama-b11200-bin-win-cuda-{12.4,13.4}-x64.zip` (the binaries) plus a same-release
  `cudart-llama-bin-win-cuda-{12.4,13.4}-x64.zip` (the CUDA runtime DLLs upstream does not
  statically link: `cudart64_*`, `cublas64_*`, ...) -- both required, both verified by exact byte
  size and sha256, both extracted into the same install directory. `cuda_kind_for_driver(driver: str
  | None) -> str`: `"cuda13"` once the driver string's leading major version is `>= 580` (the
  minimum for the 13.4 build, and the only one with native Blackwell/RTX-50 kernels -- 12.4 still
  runs on Blackwell via PTX JIT, just slower to first load), else `"cuda12"`; a missing/unparseable
  driver conservatively resolves `"cuda12"`. `ensure_runtime(kind, root, *, build=PINNED_BUILD, ...)
  -> Path` now downloads/verifies/extracts EVERY asset of a multi-asset kind in one call before
  checking once for `llama-server.exe`; a `.exocortex-runtime.json` manifest
  (`{"kind", "build", "assets": [names]}`) is written under the install directory's top level so
  `describe_runtime()` can list every asset a multi-zip kind actually pulled (absent for a runtime
  installed before the manifest existed, or dropped in by hand -- callers must not depend on it).
- **`exocortex/llm/server.py` (extended).** The rule from here on: **an LM Studio pack wins first,
  for every GPU flavor of every vendor's ladder; upstream covers whatever LM Studio doesn't, for
  every vendor, CUDA and ROCm included.** `find_server_binary`'s `source="auto"` (default) tries
  the LM Studio pack for each flavor of the ladder in turn; the first flavor with none is
  remembered as the first "fetchable" one (has an upstream kind -- every flavor does now, per
  `_upstream_kind_for`); only once EVERY GPU flavor has been tried with no LM Studio pack found at
  all does the remembered flavor get the same installed-or-fetch treatment as explicit
  `source="upstream"`, before the CPU pack. The 2026-09-25/26 vendor gate (`vendor in ("nvidia",
  "amd")` never reaching the upstream mechanism under `"auto"`) is REMOVED -- that was how the
  byte-identical guarantee used to be enforced; it is now enforced by the LM-Studio-pack check
  itself, which happens to answer the same way for those two vendors whenever LM Studio is
  present, and answers "fetch upstream" instead of "raise immediately, offline" when it is not.
  `LlamaServerManager._can_fall_through` additionally allows an upstream-sourced CUDA pack (never
  one from an LM Studio pack or an explicit `llama_server_path`) to fall through to the next
  flavor on a load failure -- CUDA packs from LM Studio still never fall through (NVIDIA's own
  tuning is never second-guessed); this is the one place a CUDA flavor's historical "raise at once,
  never retried" behavior changes, and only for a binary this branch itself fetched.
  `available_models() -> list` (new) returns `exocortex.llm.library.scan_library(self.cfg)` (this
  app's own library + LM Studio's, merged) when `exocortex.llm.library` is importable, else falls
  back to this module's own `scan_models(lmstudio_models_dir(self.cfg))` -- the sole point where
  `server.py` reaches into `library.py`, late enough (inside a method body) that a circular import
  with `library.py`'s own late reach back into `server.py` (for `default_threads`/
  `lmstudio_models_dir`) never forms. `choose_from`/`resolve_from` were factored out of
  `choose_model`/`resolve_model` to work over either a `GgufModel` or a `LibraryModel` list, so
  `available_models()`'s richer scan and the pre-2026-09-27 `scan_models()` path share one
  resolution rule instead of two.
- **`exocortex/llm/library.py` (new module, owned entirely by this section).** This app's own GGUF
  library at `<library_dir>/<publisher>/<repo>/*.gguf` (`library_dir(cfg) -> Path`, `[llm].library_dir`,
  empty = `<repo>/models/llm` -- the same two-level layout LM Studio uses, so a file moved between
  the two just works), with LM Studio's own folder read as an optional SECOND source
  (`library_roots(cfg) -> list[tuple[str, Path]]`). `scan_library(cfg, *, with_meta=False) ->
  list[LibraryModel]` merges both roots, groups a quant's shards under one entry, and attaches a
  `mmproj` vision projector to its base model instead of listing it on its own
  (`_best_mmproj_match`). `gguf_metadata(path) -> GgufMeta` reads ONLY the `GGUF` magic, version,
  tensor/KV counts and the KV section itself -- stopping at the first byte of tensor-info and never
  touching tensor data, cheap enough to run over an entire library on every scan; a value this
  module does not care about is skipped by seeking past fixed-size scalars or reading only an
  array/string's length prefix(es), never its bytes. `hf_search(query, *, limit=25, http=None) ->
  list[HfRepo]`, `hf_repo_info(repo_id, *, http=None) -> HfRepoInfo`,
  `group_quants(files) -> tuple[QuantOption, ...]` (groups a repo's GGUF files by quant name,
  keeping only the quant's own shards). `choose_quant(options, *, budget_mb, policy="auto") ->
  QuantOption | None` -- see "Quant policy" below. `pull_model(repo_id, quant, cfg, *,
  include_mmproj=False, policy="auto", budget_mb=None, progress=None, cancelled=None, http=None) ->
  LibraryModel`: every shard of one quant, streamed to a `.part` file with an HTTP Range resume (a
  server that ignores the Range request restarts from scratch rather than silently duplicate
  bytes), each file's exact byte size and sha256 checked against the Hub's own LFS metadata before
  an atomic `part.replace(dest)`, a free-space check (`shutil.disk_usage`) before any request is
  made, `HF_TOKEN` (environment) sent as a bearer token for a gated repo, and -- when `budget_mb`
  is given -- an `auto_profile` generated and saved for the pulled model. `PullCancelled` (a
  `PullError` subclass) keeps the `.part` file on a `cancelled()` hook firing mid-download, so the
  next `pull_model` call resumes rather than restarts. `ModelProfile` (per-model overrides of the
  global `[llm]` tuning; `None` field = "use the global value"; `auto: bool` marks one
  `auto_profile` generated, so the Tuning tab's own edits can clear it) with
  `load_profiles`/`save_profiles` (`profiles.json` under `library_dir`, atomic write) and
  `profile_for`/`set_profile`. `delete_model(model) -> list[Path]` -- see "Coding rules" below for
  the ownership rule it enforces. `auto_profile(model, *, budget_mb, ram_total_mb, physical_cores,
  gpu_framework, base) -> ModelProfile` -- see "Auto-profile formula" below.
- **`exocortex/hardware.py` (new module).** `collect(cfg.llm, *, probe_runtime=True, ...) ->
  HardwareReport` -- a single, side-effect-free snapshot (besides an optional `--save`): GPU/CPU/
  RAM, `library_dir`/`library_free_mb`, and (when `probe_runtime`) what runtime pack
  `find_server_binary(..., ensure=None)` would pick RIGHT NOW without fetching anything --
  `runtime = None` plus a note when neither an LM Studio pack nor an already-installed upstream
  build exists. `recommended_policy(budget_mb) -> str` feeds `choose_quant`'s `policy` argument.
  Every probe inside `collect` is injectable and wrapped so a single failure adds one line to
  `HardwareReport.notes` and leaves that field `None`, rather than failing the whole report.
  `to_dict`/`save`/`load` round-trip through plain JSON; `report_path(cfg.llm) ->
  <library_dir>/hardware.json`; `describe(report) -> str` is what `python -m exocortex.hardware`
  and the setup script's console output print. `python -m exocortex.hardware [--json] [--save]`.
- **`exocortex/setup.py` (new module).** `run_setup(cfg, *, quiet=False, yes=False, dry_run=False,
  do_runtime=True, do_speech=True, do_model=True, out=_stdout_write, stdin_is_tty=None, ask=input,
  **probes) -> SetupResult` -- the four steps below, each with its own injectable probes so the
  whole thing is unit-tested offline. `python -m exocortex.setup [--config PATH] [--quiet] [--yes]
  [--dry-run] [--no-runtime] [--no-speech] [--no-model]`; exit code 1 only when no llama.cpp
  runtime could be found or fetched at all (every other step failure is reported in `SetupResult`
  and the process still exits 0, so a `run.bat` launch is never blocked by a bad network day).
  1. **The llama.cpp runtime** -- `find_server_binary(cfg.llm.llama_server_path, vendor=gpu.gpu_vendor(),
     backend=cfg.llm.llama_backend, source=cfg.llm.llama_source, upstream_build=cfg.llm.llama_upstream_build,
     ensure=<a closure that calls runtimes.ensure_runtime with a console progress bar>)`.
  2. **The hardware report** -- `hardware.collect(cfg.llm, probe_runtime=True)`, printed via
     `hardware.describe()` and saved to `hardware.report_path(cfg.llm)`.
  3. **The speech models** -- `models_dl.ensure_models` (VAD + speaker embeddings), then
     `asr.resolve_engine(cfg.asr)`'s answer decides between `models_dl.ensure_sherpa_model` (the
     Parakeet bundle) and a faster-whisper `download_model` cache check/fetch.
  4. **At least one LLM** -- `library.scan_library(cfg.llm)`; an empty result prints
     `recommendations_for(budget_mb)` (the curated `RECOMMENDED` table below, filtered to what this
     machine's budget clears, with LIVE sizes/quants from `hf_repo_info` + `choose_quant`), then
     either an interactive prompt (`[1..N]` / `[S]earch` / `[N]ot now`), `--yes` (takes the first
     recommendation), or a one-line hint to use `... > Models...` when the prompt is skipped (see
     "Skip conditions" below). A successful pull sets `cfg.llm.model` to the pulled key and saves
     `config.toml` at once, so the very next launch loads exactly what was just pulled.
  - **Skip conditions** (`_interactive`): the prompt is skipped when `EXO_SETUP_NONINTERACTIVE=1`
    is set, when `stdin_is_tty`/`sys.stdin.isatty()` says this is not a real console, or when the
    caller passed `quiet=True` (`--quiet`, what `run.vbs` sets `EXO_SETUP_QUIET=1` for on its
    hidden catch-up runs -- nobody could answer a prompt in a hidden window).
  - **`RECOMMENDED: tuple[Recommendation, ...]`** -- curated 2026-09-27 from the Hub's
    most-downloaded GGUF repos per size class (`repo_id`, `why`, `min_budget_mb`): `unsloth/Qwen3.8-27B-GGUF`
    (22528 MiB min), `lmstudio-community/gemma-4-12B-it-QAT-GGUF` (9728), `unsloth/gemma-4-12b-it-GGUF`
    (6144), `empero-ai/Qwen3.8-9B-Distill-GGUF` (8192), `lmstudio-community/gemma-4-E4B-it-GGUF`
    (7168), `empero-ai/Qwen3.8-4B-Distill-GGUF` (0 -- CPU-capable), `unsloth/gemma-4-E2B-it-GGUF` (0).
    `recommendations_for(budget_mb, *, limit=4) -> list[Recommendation]` keeps the ones this
    machine's budget clears, best first; a `None` budget (no GPU) keeps only the `min_budget_mb=0`
    entries.
- **`exocortex/config.py` (extended).** `LLMConfig` gained `server_url: str = ""` (replacing the
  always-both-populated `base_url`/`ollama_url`), `library_dir: str = ""`, `runtime_dir: str = ""`.
  `effective_server_url(backend) -> str`: `server_url` verbatim if non-empty, else
  `STANDARD_SERVER_URLS[backend]` (`{"ollama": "http://127.0.0.1:11434", "openai":
  "http://127.0.0.1:8081"}`). `base_url`/`ollama_url` are now properties (getter = a backend-pinned
  view over `effective_server_url`; setter writes `server_url`) -- NOT dataclass fields, so reading
  or writing either never touches `config.toml` directly except through `server_url`.
  `_migrate_llm_url_keys(llm_data)` runs inside `load_config`, before `_apply`: an old
  `base_url`/`ollama_url` key folds into `server_url` (following whichever the config's own
  `backend` names, or whichever legacy value differs from its own standard default, else `""`),
  UNLESS `server_url` is already present (it wins; the legacy keys are dropped silently) -- one
  INFO log line names the migration when it actually changes something. **Two protocols, one
  field, in two sentences (for the README's "Configuration" section):** the two old fields existed
  because `backend = "ollama"` and `backend = "openai"` speak two different wire protocols
  (Ollama's native `/api/generate`/`/api/chat` vs. an OpenAI-compatible `/v1/chat/completions`),
  each needing its own address; one field is enough now because only one of those two backends is
  ever the *external* server a given `[llm].backend` value talks to at a time, so `server_url`'s
  meaning simply follows `backend` instead of needing a name of its own per protocol.
- **`exocortex/app.py`/`actions.py`/`bus.py` (extended).** `EngineActions` gained
  `model_library() -> list[dict]`, `loaded_model() -> str`, `load_model(key)`, `unload_model()`,
  `set_default_model(key)`, `delete_model(key)`, `pull_model(repo_id, quant="auto", *,
  include_mmproj=False)`, `cancel_pull()`, `search_models(query, request_id)`,
  `inspect_model(repo_id, request_id)`, `hardware_report(refresh=False) -> dict`,
  `model_profile(key) -> dict | None`, `set_model_profile(key, values)`, `auto_tune_model(key) ->
  dict | None`. New events `ModelPullEvent` (`repo_id`, `state` in `{"queued", "downloading",
  "cancelled", "error", "done"}`, `file`, `detail`, `model_key`), `ModelLibraryEvent` (`reason`,
  `loaded_key`), `ModelSearchEvent` (`request_id`, `query`, `results`, `error`), `ModelInfoEvent`
  (`request_id`, `repo_id`, `quants`, `error`). An empty library at engine start publishes
  `ModelLibraryEvent(reason="empty-library")` -- what makes the overlay open the Models window on
  Discover by itself. `unload_model()` stops `llama-server` (or, on Ollama, sets `keep_alive: 0`);
  every answer path is refused with a notice until the next `load_model`/auto-load.
- **`exocortex/ui/models.py` (new module) / `ui/settings.py` (extended).** `ModelsDialog` (`⋯ ›
  Models...`, and the slim bar's `Context ▾` menu): tabs `("Library", "Discover", "Tuning",
  "Server", "Hardware")`. Library: load / unload / set default / delete (own-library models only
  -- a click on an LM Studio model's row disables Delete) / open folder. Discover: Hub search, a
  quant drop-down with `"auto (<quant>, <size> - fits this machine)"` first, an mmproj checkbox,
  pull progress + cancel, "Load now" once a pull finishes. Tuning: one override checkbox per
  `ModelProfile` field, presets "Speed"/"Quality"/"Low memory", "Auto-tune for this machine", "Use
  global defaults". Server: the `[llm]` fields that used to be `SettingsDialog`'s "Model" tab.
  Hardware: the same report `exocortex.hardware.collect` writes, with a "Refresh" button.
  `SettingsDialog` no longer has a Model tab and no longer touches `[llm]` at all; the old "Choose
  model" submenu is gone -- `ModelsDialog` is now the one place both live.

**Quant policy (`exocortex.llm.library.choose_quant(options, *, budget_mb, policy="auto")`):**
`"auto"` (default) -- the first quant of a preference ladder (`Q4_K_M` first, then the rest of the
K/IQ ladder) that both exists in the repo and fits 75% of `budget_mb` (headroom for the KV cache
and activations), falling back to the largest non-float quant that fits, or -- with no `budget_mb`
at all (no GPU detected) -- the first preference-list quant present regardless of size. `"quality"`
-- the LARGEST of a smaller, higher-precision preference ladder (`Q8_0`, `Q6_K`, ...) that fits,
not merely the first in list order. `"fit"` -- the largest non-float quant that fits at all, full
stop. A named quant (e.g. `"Q5_K_M"`) pins that one exactly, or `None` if the repo has no such
quant.

**Auto-profile formula (`exocortex.llm.library.auto_profile`):** no GPU (`budget_mb is None`) ->
`ctx_size=4096, gpu_layers=0`, CPU-only. No usable GGUF metadata -> no opinion on ctx/layers,
`[llm]`'s own global values stand for those two. Otherwise: let `margin = 0.75 * budget_mb`; the
largest `ctx` that is a multiple of 2048 in `[4096, min(the model's trained context, 32768)]` such
that `weights + kv_bytes_per_token(kv_cache_type) * ctx + 768 MiB <= margin`; when even `ctx=4096`
does not fit, partial GPU offload instead --
`gpu_layers = floor(n_layer * (margin - kv_bytes_per_token * 4096 - 768 MiB) / weights)`, clamped
to `[0, n_layer]`, at a fixed 4096-token context. A GPU runtime gets `kv_cache_type="q8_0"` +
`flash_attn=True`; a CPU-only one gets `f16` + no flash attention (llama-server hard-refuses a
quantized V-cache without flash attention, same rule as the 2026-09-25/26 sections above).

**The setup steps and the `run.bat`/`run.vbs` contract:** see "Module contracts" above (`setup.py`)
for the four steps' exact call sequence. `run.bat`'s `:first_run` label calls `python -m
exocortex.setup` right after `:ensure_deps` (the pip step), on EVERY launch, not only when the pip
stamp is stale -- a model pulled or deleted by hand, or a runtime that went missing, must be
noticed without touching `requirements.txt`. `EXO_SETUP_QUIET=1` (`run.vbs` sets it for its hidden
catch-up runs) maps to `--quiet`; `EXO_SETUP_DRYRUN=1` makes every download/setup step -- including
this one -- print what it would do instead of touching the network or disk (what
`tests/test_run_script.py` runs under, so the suite needs no network and takes seconds). A
non-zero exit from this step is reported (`WARNING: the first-run setup reported a problem`) but
never blocks the launch that follows -- the overlay still transcribes, and `find_server_binary`
retries the runtime fetch itself at the engine's own start.

**Live-verified 2026-09-27** (RTX 5080 Laptop, driver 592.82, `LMSTUDIO_HOME` pointed at an empty
folder): see the three new "Verified facts" rows above (upstream CUDA fetch/start end to end, the
Hugging Face pull + auto-profile, and the resulting numbers) -- this is the "no LM Studio, no
Ollama" scenario proven live, once, on the NVIDIA laptop this app is tuned on.

**Still unverified, labelled as such (see the README's honesty notes for the reader-facing
version):**
- AMD ROCm and Intel SYCL/OpenVINO upstream fetches under the new every-vendor rule -- a
  unit-tested design (`tests/test_runtimes.py`'s cuda12/cuda13/rocm cases,
  `tests/test_server.py`'s extended vendor-gate tests) but not fetched live for those two flavors
  under this specific rule; the AMD box has only run the 2026-09-26 Vulkan/SYCL-negative-control
  scenario, which predates the cuda12/cuda13/rocm kinds.
- The mac-cortex fork's port of this feature -- explicitly a separate follow-up, not part of this
  branch (see `docs/UPSTREAM_SUGGESTIONS.md`).
- `ModelsDialog` itself -- driven only through offscreen tests (`tests/test_models_dialog.py`,
  `tests/test_overlay_models.py`); nobody has clicked through Library/Discover/Tuning/Server/
  Hardware on a real screen yet.

**Tests.** `tests/test_setup.py` (the four `run_setup` steps, skip conditions, `--yes`/`--dry-run`,
the recommendation list, exit-code-1-only-on-no-runtime), `tests/test_app_models.py` (the new
`EngineActions` methods and events end to end against a fake bus), `tests/test_hardware.py`
(`collect`'s injected-probe-failure tolerance, `describe`/`to_dict`/`save`/`load` round-tripping,
`recommended_policy`), `tests/test_runtimes.py` (extended: cuda12/cuda13 two-asset installs,
`cuda_kind_for_driver`, rocm, the manifest), `tests/test_library.py` (GGUF header parsing,
`choose_quant`'s three policies, `pull_model`'s resume/verify/cancel/gated-repo/disk-space paths,
`auto_profile`'s ctx/offload math, profile load/save), `tests/test_models_dialog.py` (offscreen:
every tab's controls, the own-library-only delete guard, the quant-combo ordering),
`tests/test_overlay_models.py` (offscreen: the `... > Models...` menu entry, the Context ▾ slim-bar
entry, the empty-library auto-open), `tests/test_config_urls.py` (`server_url`
migration/`effective_server_url`/`base_url`/`ollama_url` property round-tripping),
`tests/test_run_script.py` (extended: the `:first_run` label under `EXO_SETUP_DRYRUN`,
`EXO_SETUP_QUIET` threading through to `--quiet`), `tests/test_review_fixes.py` (the seven review
pins below). Full suite: `2395 passed, 0 failed, 24 live deselected
(2026-09-27, offline, on the RTX 5080 laptop)`; the one remaining red test before this pass,
`tests/test_asr_sherpa.py::test_transcriber_load_faster_whisper_path_unaffected_by_sherpa_wiring`,
assumed a CPU-only box (it asserted `device_used == "cpu"` with the default `device = "auto"`, which
resolves to CUDA here) and now pins `device = "cpu"` explicitly, so the suite is green on any GPU.

**Fixes applied after review (2026-09-27, five finder lenses -> two independent refuters per
finding; six of seven findings confirmed, each fixed and pinned in `tests/test_review_fixes.py`):**

1. *Critical -- the first load on a fresh machine was a no-op.* With an empty library at start the
   route resolves to `"none"`, `_init_llm` never builds a `LlamaServerManager`, and `set_model`'s
   old `if self.server is not None` branch fell through to `assistant.set_model` (a name change,
   no server). `Engine.set_model` now builds the manager on first use (`_ensure_server`, only ever
   from the swap thread), starts the assistant if it never ran (`_assistant_started`), lifts the
   `_llm_failed` refusal `_drop_pending` set, sets `_llm_ready` and replays queued requests
   (`_mark_llm_ready` / `_replay_pending`, the latter factored out of `_init_llm`'s tail). Before
   `start()` (no route) or for `ollama`/`openai` routes nothing is spawned (`_can_spawn_server`).
2. *Critical -- `resolve_route`'s default probe only saw LM Studio's folder.* `_default_scan` now
   delegates to `library.scan_library(cfg)` (the merged own-library + LM Studio view
   `available_models()` serves), and `_default_find_binary_for(cfg)` passes
   `llama_source`/`llama_upstream_build`/`runtime_dir` through -- a build the setup step installed
   under a custom `runtime_dir` is found by the probe (never downloaded: no `ensure`). Verified
   live on the dev laptop after the fix: empty `LMSTUDIO_HOME`, one GGUF in the app library,
   `python -m exocortex.setup` (runtime step) then `resolve_route` -> `llamacpp`, upstream cuda13,
   1 model, 0.5 s. The same probe kwargs now reach `hardware.collect`'s runtime line.
3. *High -- unguarded concurrent model changes.* `set_model` and `unload_model` used to spawn a
   fresh `"llm-swap"` thread each time; two overlapping `swap_model()`/`stop()` calls on the one
   `LlamaServerManager` race on `_proc` and can orphan a llama-server. `Engine._start_swap` keeps
   one tracked swap thread and refuses a second change with an `"llm"` info notice while it runs;
   the Models window disables Load / Unload / Delete while the `"llm"` component reports
   `"loading"`.
4. *Medium -- free-space check ignored bytes already on disk.* `pull_model` now subtracts a
   resumed `.part`'s bytes and skips shards that already finished, so a nearly complete resume is
   never refused for the space it no longer needs.
5. *Medium -- the second CUDA ladder rung re-ran the same binary.* `_upstream_kind_for` resolves
   `"nvidia-cuda"` (the rung after `"nvidia-cuda12"`) to the 12.4 build only when the first rung
   was the driver-gated 13.4 build, and to nothing when the first rung already was 12.4 -- the
   fall-through goes straight to Vulkan instead of retrying an identical exe under another name.
6. *Medium -- an orphan widget tree on the Server tab.* `_build_server_tab` built a `FieldForm`
   (which builds its own scroll area + form) and then re-laid its widgets out in a second form,
   leaving the first one floating over the Backend row; the tab now uses `FieldForm.widget`
   directly, like the Settings tabs.
7. *Defense in depth (the refuted path-traversal finding).* The Hub's repo id and file names each
   become exactly one path component under the library: `pull_model` refuses `""`, `"."`, `".."`,
   separators, drive prefixes or characters outside the Hub's naming rules (`_check_path_component`);
   a path with directory parts is reduced to its basename, as before (never a traversal).
   `route._install_hint` no longer tells a fresh machine to install LM Studio or Ollama.

**Second review round (2026-09-27, five new lenses, one reproducing verifier per finding; eight
findings, all confirmed, fixed in the ten changes below -- two findings had an engine half and a
dialog half -- and each fix re-verified by an independent agent re-running the original failure
scenario; `exocortex --doctor` warns when `[llm].dropped_legacy_url` is set):**  <!-- review round two, 2026-09-27 -->

1. A refused model change no longer rewrites `[llm].model`.
2. `stop()` joins the swap thread and stops a server created by a racing first load.
3. `unload` publishes a `"loading"` status and the refusal notice carries the `"busy: "` prefix.
4. `set_model_profile` coerces `extra_args` and numeric strings.
5. The Tuning form is reloaded only when the model changes.
6. A second Pull click can no longer corrupt the running pull's progress.
7. `extra_args` round-trips as a list and an empty list is no override.
8. The busy gate follows operation start / completion and double-click honours it.
9. The legacy-URL migration warns loudly and records the dropped address.
10. The README "fit" wording.

## Coding rules for implementers
- Windows-only OK, but guard imports so `pytest` runs (e.g., import pyaudiowpatch inside functions or try/except at module import with a clear error at use time).
- Never call the network except LM Studio (localhost), the model download in `models_dl.py`, and
  (2026-09-27) `huggingface.co` and `github.com/ggml-org/llama.cpp` release downloads -- and only
  through `exocortex.llm.library` (Hub search/repo-info/pull) or `exocortex.llm.runtimes` (upstream
  llama.cpp runtime fetch) respectively; no other module reaches either host directly.
- Every module must be importable without side effects (no model loading at import).
- Unit tests must not require the GPU, audio devices, or LM Studio: mock them. Tests that need them are marked `@pytest.mark.live` and skipped by default (`-m "not live"` is the default via `pytest.ini`).
- Use `tests/data/two_speakers_16k.wav` for VAD/diarization tests (these run on CPU with the ONNX models in `models/`; skip if the model files are absent).

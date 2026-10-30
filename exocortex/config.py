"""Configuration: typed dataclasses + TOML load/save (no third-party deps).

Search order for the config file (first found wins):
  1. explicit path passed to load_config()
  2. ./config.toml (current working directory)
  3. <project root>/config.toml
  4. %APPDATA%/ExoCortex/config.toml
If none exists, defaults are used and `Config.path` is None.

All sections are flat (scalars, lists of scalars, or a flat dict for extra_body)
so the tiny TOML writer below is sufficient. Unknown keys are ignored with a warning.
"""
from __future__ import annotations

import dataclasses
import logging
import os
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


#: Standard address for each backend that talks to an EXTERNAL server (used when [llm].server_url
#: is left at "" -- see LLMConfig.effective_server_url). "llamacpp" is not here: that backend's
#: port is ephemeral (LlamaServerManager picks a free one at startup) and never read from this.
STANDARD_SERVER_URLS: dict[str, str] = {
    "ollama": "http://127.0.0.1:11434",
    "openai": "http://127.0.0.1:8081",
}


@dataclass
class LLMConfig:
    # "auto":     (default) route at startup -- exocortex.llm.route.resolve_route: "llamacpp" when
    #             LM Studio's llama.cpp runtime pack for this machine's GPU and at least one GGUF
    #             are present; else "ollama" when an Ollama server answers at
    #             effective_server_url("ollama") (or the standard Ollama address); else "openai"
    #             when a server answers at effective_server_url("openai"); else a clear error
    #             naming what to install (LM Studio, Ollama, or just pull a model into the app's
    #             own library -- see library_dir below). The choice is logged and shown by
    #             --doctor; it is never written back into this field.
    # "llamacpp": ExoCortex spawns its own hidden llama-server (llama.cpp) on a private port.
    #             Models come from the app's own library (library_dir below; LM Studio's folder,
    #             if present, is read as an extra source -- no LM Studio install is required) and
    #             a llama.cpp runtime pack for this GPU: CUDA on NVIDIA, Vulkan or ROCm on AMD,
    #             Vulkan on Intel, CPU otherwise -- see llama_backend. No LM Studio process
    #             needs to run. See llama_source below for where the llama-server BINARY comes
    #             from (LM Studio's own packs vs. an official upstream llama.cpp build, fetched
    #             into runtime_dir).
    # "ollama":   talk to a running Ollama at effective_server_url("ollama"): its own model store,
    #             its own GPU routing.
    # "openai":   talk to any OpenAI-compatible server at effective_server_url("openai")
    #             (llama-swap, LM Studio, ...).
    backend: str = "auto"
    # 2026-09-27 (portable-install): the external server for backend = "ollama" / "openai" -- ONE
    # field where there used to be two (base_url / ollama_url): a machine only ever talks to one
    # such server at a time, and the split invited the two drifting out of sync. "" = that
    # backend's standard address (Ollama 127.0.0.1:11434, OpenAI-compatible 127.0.0.1:8081 -- see
    # STANDARD_SERVER_URLS / effective_server_url below). Use 127.0.0.1, NOT "localhost": Windows
    # tries IPv6 first and servers that only listen on IPv4 cost ~2.3 s per fresh connection. The
    # old base_url / ollama_url keys in config.toml still load -- load_config migrates them into
    # this field -- and `cfg.llm.base_url` / `cfg.llm.ollama_url` still work as read/write
    # properties over this same field (see the bottom of this class), for any code that has not
    # moved to server_url yet.
    server_url: str = ""
    # review round two, 2026-09-27: when a config.toml customises BOTH legacy base_url/ollama_url
    # to different non-standard addresses, only one survives into server_url (see
    # _migrate_llm_url_keys); the other is recorded here so the doctor / UI can surface it and the
    # owner can recover it (config.toml lines are given in the WARNING log line). NOT a persisted
    # field -- to_toml skips it (metadata "toml": False), like `path` is skipped on Config -- it is
    # set only in memory, fresh on every load_config() call.
    dropped_legacy_url: str = field(default="", compare=False, metadata={"toml": False})
    # Default is the specific, tuned model ("maple-preview") so a fresh/no-config install (and any
    # NVIDIA machine that doesn't pin this key) loads exactly what the app was verified against,
    # matching main's behaviour byte-for-byte. "auto": the largest model in the library that fits
    # this machine's GPU memory budget (exocortex.llm.server.pick_default_model; ollama: the
    # largest model Ollama lists) -- opt in explicitly via config.toml for machines that lack
    # "maple-preview" (e.g. AMD/Intel iGPU boxes without that GGUF). Otherwise llamacpp: an LM
    # Studio model key ("maple-preview", "deepgrove/maple-preview"), a GGUF file name, or an
    # absolute path to a .gguf; ollama / openai: the model id the server expects. A name that is
    # not found (including "maple-preview" itself, on a machine that lacks it) falls back to
    # "auto" with a warning instead of failing the start -- this is what covers non-NVIDIA
    # machines using the bare default.
    model: str = "maple-preview"
    # ---- llamacpp backend ----
    llama_server_path: str = ""        # "" -> the best llama-server.exe among LM Studio's runtime packs for this GPU
    # Which llama.cpp GPU framework to run among LM Studio's runtime packs: "auto" picks by the
    # primary GPU's vendor (exocortex.gpu): NVIDIA -> cuda12 > cuda > vulkan > cpu; AMD -> rocm >
    # vulkan > cpu (measured 2026-09-26 on the Radeon 8060S with a 16.8 GB Qwen3.8-27B: ROCm
    # processes prompts at ~300 tok/s vs Vulkan's ~160 and answers a cached prompt in 0.4 s vs
    # 3.4 s, generation equal; LM Studio only installs ROCm on GPUs it supports -- the 780M is
    # not one -- and a ROCm pack that sees no device or dies while loading falls through to
    # Vulkan, which runs on every AMD GPU); Intel -> vulkan > cpu; no GPU detected -> the legacy
    # cuda12 > cuda > vulkan > cpu order. Explicit: "cuda" | "rocm" | "vulkan" | "cpu" | "sycl" |
    # "openvino" (a missing pack falls back down the same ladder with a warning). 2026-09-26:
    # "sycl" and "openvino" are Intel-only frameworks that exist only as upstream llama.cpp builds
    # -- LM Studio's own runtime catalog ships neither -- so in practice requesting one only gets
    # a binary at all when llama_source (below) is "auto" or "upstream"; under "lmstudio" (packs
    # only) it simply falls back down the ladder to Vulkan/CPU exactly like any other missing pack.
    llama_backend: str = "auto"
    # Which GPU llama-server uses when the chosen framework sees several (`--device`): "auto" =
    # the primary GPU (exocortex.gpu.primary_gpu, matched by name in `llama-server --list-devices`;
    # a single device gets no flag, and CUDA builds are never given one); "" = let llama.cpp
    # decide; or a device name as --list-devices prints it ("Vulkan0", "ROCm0").
    llama_device: str = "auto"
    # 2026-09-26 (intel-integration): where the llama-server BINARY for the chosen llama_backend
    # flavor comes from -- a second source alongside LM Studio's own runtime packs, added because
    # LM Studio's catalog ships no SYCL/OpenVINO pack for Intel Arc discrete GPUs (A770/A750,
    # B580/B570, Pro B70) at all. "auto" (default): NEVER fetches anything for NVIDIA/AMD -- those
    # two vendors are byte-identical to before this field existed, full stop, including "no LM
    # Studio pack anywhere for this vendor" (that raises immediately, offline, exactly like main;
    # it does not reach for an upstream build). For every OTHER vendor (Intel included): LM
    # Studio's packs first, and only when NO LM Studio pack exists for ANY GPU flavor of this
    # vendor's ladder, or every one already failed this run (the fall-through loop exhausted
    # them), fetch the official llama.cpp build (exocortex.llm.runtimes) for the ladder's first GPU
    # flavor, before finally dropping to the CPU pack. "lmstudio": LM Studio's packs only -- the
    # pre-2026-09-26 behaviour, never fetches.
    # "upstream": official llama.cpp release builds only, ignoring LM Studio's packs entirely (the
    # only way "sycl"/"openvino" above get a binary under a strict "lmstudio" setting is to change
    # this to "auto" or "upstream"). CUDA/ROCm are never fetched upstream under any setting --
    # NVIDIA and AMD keep their tuned LM Studio packs (see exocortex.llm.server's module docstring).
    llama_source: str = "auto"
    # The llama.cpp release tag exocortex.llm.runtimes fetches builds from when llama_source is
    # "upstream" (or "auto" falls through to it) -- pinned so a fresh install never makes a network
    # call just to discover asset digests (runtimes.PINNED_ASSETS/PINNED_BUILD hold that build's
    # sizes/sha256 by hand); any other tag is resolved live from GitHub's release API.
    llama_upstream_build: str = "b11200"
    # 2026-09-27 (portable-install): where exocortex.llm.runtimes caches a fetched upstream
    # llama.cpp runtime pack (see llama_source above). "" -> <project root>/.tools/llama.cpp.
    runtime_dir: str = ""
    # Extra environment variables merged into llama-server's process, after PATH -- e.g. an Intel
    # Arc "Alchemist" Vulkan workaround ({"GGML_VK_DISABLE_COOPMAT": "1"}; see
    # exocortex.llm.runtimes.vulkan_env_switches, which reads the exact GGML_VK_* switches a given
    # ggml-vulkan.dll build actually recognises -- the doctor prints them) or SYCL's device
    # selector ({"ONEAPI_DEVICE_SELECTOR": "level_zero:0"}). Framework "sycl" additionally gets
    # ZES_ENABLE_SYSMAN=1 by default (SYCL.md: enables free-VRAM reporting) unless a key of this
    # same name is already set here.
    llama_env: dict[str, str] = field(default_factory=dict)
    lmstudio_models_dir: str = ""      # "" -> read from %USERPROFILE%/.lmstudio/settings.json (downloadsFolder)
    # 2026-09-27 (portable-install): the app's OWN GGUF library -- <library_dir>/<publisher>/
    # <repo>/*.gguf, populated by exocortex.llm.library.pull_model (Hugging Face Hub search +
    # whole-model pulls, auto-chosen quant for this machine's GPU budget). "" -> <project
    # root>/models/llm. LM Studio's own folder (lmstudio_models_dir above), when present, is
    # scanned too as a second source (exocortex.llm.library.library_roots) -- no LM Studio
    # install is required for this app to have a model library.
    library_dir: str = ""
    llama_port: int = 0                # 0 -> pick a free port
    ctx_size: int = 8192
    gpu_layers: int = 999999
    threads: int = 0                   # 0 -> physical cores - 2 (matters for CPU-bound quants such as TQ1_0)
    parallel: int = 2                  # 2 slots so a notes job never blocks a live answer
    flash_attn: bool = True
    load_mode: str = "mmap"            # mmap | mmap+mlock (mlock pins the weights in RAM)
    llama_extra_args: list[str] = field(default_factory=list)
    # The floor of the startup deadline; the manager waits longer for a big model on slow media
    # (exocortex.llm.server.estimate_startup_timeout_s: size / read floor + 60 s -- a 16.8 GB
    # GGUF on the exFAT card in this box's card reader measured a 323 s cold load, so the wait
    # becomes ~600 s there; on an internal drive the 180 s stands).
    llama_startup_timeout_s: float = 180.0
    # "off": raw /v1/completions with an empty <think></think> block (fastest, recommended).
    #        maple-preview's template always thinks and LM Studio has no working toggle for it.
    # "on":  let the model think; reasoning is streamed but hidden, only the answer is shown
    thinking: str = "off"
    assist_max_tokens: int = 220
    ask_max_tokens: int = 400
    solve_max_tokens: int = 700         # autonomous "solve what's on screen" answers (code needs room)
    text_max_tokens: int = 400          # the Text button: a plain-English walkthrough of the task
    notes_max_tokens: int = 900
    insights_max_tokens: int = 320      # co-pilot insights: questions to ask + blind spots / follow-ups
    recap_max_tokens: int = 360         # the Recap action: where the session stands, ready to read out
    # One dashboard refresh (exocortex.dashboard): the model reports only what CHANGED (new items,
    # answered / done / dropped ones) against the open items it was shown, so this stays small.
    dashboard_max_tokens: int = 420
    # Qwen3-family non-thinking guidance: temperature 0.7 / top_p 0.8 / top_k 20 / min_p 0.
    # Greedy decoding (0.0) is discouraged upstream (repetition); we go slightly lower for factual answers.
    temperature: float = 0.6
    top_p: float = 0.85
    # transcript context handed to the model for assist/ask (approximate tokens; 4 chars ~ 1 token)
    context_tokens: int = 2500
    auto_load_model: bool = True       # ask LM Studio to load the model if it is not loaded
    request_timeout_s: float = 120.0
    # extra JSON merged into every request body (LM Studio accepts llama.cpp sampling keys)
    extra_body: dict[str, Any] = field(default_factory=lambda: {"top_k": 20, "min_p": 0.0})
    # KV cache quantisation for the spawned llama-server ("f16" opts out). q8_0 by default: the
    # very same GGUF runs under this machine's llama-swap with q8_0 K and V, and the hybrid
    # qwen35 architecture keeps the KV small anyway -- the freed VRAM is margin against the
    # display GPU tipping into WDDM paging mid-meeting.
    kv_cache_type: str = "q8_0"
    # GPU pressure guard: a "gpu-guard" thread samples free VRAM (NVML) every ~5 s and raises a
    # toast + status event when it stays under vram_guard_free_mb -- the signature of a SECOND
    # large model landing on the display GPU (this machine's fleet gateway can load one on
    # remote demand), which pages the card into system RAM and stalls the whole desktop.
    vram_guard: bool = True
    vram_guard_free_mb: int = 1500
    # The local llama-swap service (this laptop's fleet node) can hold its own copy of the model;
    # a --replace launch and a normal quit ask it to unload (GET /unload) so the display GPU is
    # never shared between two copies (the 2026-09-24/25 freezes). "" disables; only used with
    # backend = "llamacpp" (with backend "openai" that server IS the backend).
    swap_unload_url: str = "http://127.0.0.1:8081"

    # ---- server_url helpers / legacy aliases (2026-09-27, portable-install) -------------------
    def effective_server_url(self, backend: str) -> str:
        """The address to talk to for `backend` ("ollama" | "openai"): `server_url` verbatim
        (trailing slash stripped) when it is set, else that backend's standard address
        (STANDARD_SERVER_URLS). Used directly by exocortex.llm.route/client/doctor; `base_url`
        and `ollama_url` below are thin, backend-pinned views over the same thing."""
        return self.server_url.rstrip("/") or STANDARD_SERVER_URLS.get(backend, "")

    @property
    def base_url(self) -> str:
        """Legacy alias for ``effective_server_url("openai")``. NOT a dataclass field -- reading
        or writing it never touches config.toml directly; it is a view over `server_url`."""
        return self.effective_server_url("openai")

    @base_url.setter
    def base_url(self, value: str) -> None:
        self.server_url = value

    @property
    def ollama_url(self) -> str:
        """Legacy alias for ``effective_server_url("ollama")``. NOT a dataclass field, same as
        `base_url` above."""
        return self.effective_server_url("ollama")

    @ollama_url.setter
    def ollama_url(self, value: str) -> None:
        self.server_url = value


@dataclass
class ASRConfig:
    model: str = "large-v3-turbo"      # any faster-whisper model name or CTranslate2 dir
    device: str = "auto"               # auto | cuda | cpu
    compute_type: str = "auto"         # auto | float16 | int8_float16 | int8
    language: str = "en"
    beam_size: int = 1
    partial_interval_s: float = 1.2    # how often to transcribe the in-progress utterance
    partials_enabled: bool = True
    hotwords: list[str] = field(default_factory=list)  # names/jargon to bias recognition
    no_speech_threshold: float = 0.7   # drop results with no_speech_prob above this
    logprob_threshold: float = -1.2    # drop results with avg_logprob below this
    # CTranslate2 threads when Whisper runs on the CPU (device "cpu", or "auto" without CUDA).
    # 0 = auto (exocortex.speech.asr.default_cpu_threads): 4 on a machine with an NVIDIA GPU
    # (unchanged: the CPU also feeds llama-server there), else min(8, max(4, physical cores // 2))
    # -- 8 on a 16-core Ryzen AI MAX+ 395, where int8 large-v3-turbo then keeps up with speech.
    cpu_threads: int = 0
    # 2026-09-26 (asr-sherpa): which speech-recognition engine `Transcriber.load()` builds.
    # "auto" (default, exocortex.speech.asr.resolve_engine): "faster-whisper" whenever CUDA is
    # available (cfg.device == "cuda", or -- same as `_resolve_device_and_compute_type`'s own
    # probe -- ctranslate2.get_cuda_device_count() > 0 after cuda_env.setup_cuda_dlls()): the
    # NVIDIA laptops (RTX 5080/5090) transcribe a 1-5 s utterance in 0.12 s there, and nothing
    # else needs to change on those machines (the owner's hard rule). Off CUDA, "auto" resolves
    # to "sherpa": measured on this dev box (Ryzen AI MAX+ 395, no NVIDIA GPU) with
    # `scripts/bench_asr.py --engine faster-whisper sherpa --threads 8 4` on
    # tests/data/two_speakers_16k.wav (TWO_SPEAKERS_TRUTH) -- faster-whisper large-v3-turbo
    # int8/CPU: 6.03 s median / utterance at 8 threads (7.16 s at 4 -- this machine's other load
    # at measurement time was heavier than the 2026-09-25 note above assumed; Whisper's encoder
    # always runs its padded 30 s window regardless, and no Whisper GPU path exists here: no
    # CUDA, whisper.cpp ships no Windows binary, CTranslate2 has no Vulkan/DirectML), mean word
    # overlap 1.00. sherpa-onnx Parakeet-TDT (`sherpa_model` below) int8/CPU: 0.19 s median at 8
    # threads (0.14 s at 4 -- thread count barely matters at this model size, unlike Whisper),
    # mean word overlap 0.99 -- comfortably clears the decision rule (>= whisper's overlap - 0.05
    # AND <= 50% of whisper's median latency: sherpa was ~3% of Whisper's latency here, not just
    # under half), so "auto" prefers it on every non-CUDA machine (AMD, Intel-only, no GPU).
    # "faster-whisper" / "sherpa" pin one explicitly.
    engine: str = "auto"
    # sherpa-onnx model bundle name (a key of exocortex.models_dl.SHERPA_MODELS), used when
    # `engine` resolves to "sherpa". "parakeet-tdt-0.6b-v2-int8": NVIDIA Parakeet-TDT 0.6B v2, a
    # NeMo transducer, English only, top of the Open ASR leaderboard for its size, ~10-20x
    # realtime on a CPU at int8 (measured above: a ~2-4 s utterance transcribed in ~0.15-0.3 s on
    # this dev box, i.e. comfortably in that range -- unlike Whisper, which pays a fixed 30 s
    # encoder window no matter how short the clip). `resolve_sherpa_model()` swaps this default
    # to the multilingual "parakeet-tdt-0.6b-v3-int8"
    # (25 European languages) automatically when `language` is set to anything other than "en"
    # and this field is still at its default -- an explicit non-default value here is never
    # overridden.
    sherpa_model: str = "parakeet-tdt-0.6b-v2-int8"


@dataclass
class AudioConfig:
    capture_system: bool = True        # WASAPI loopback of the default output device
    capture_mic: bool = True
    system_device: str = "default"     # "default" or a device name substring or an integer index
    mic_device: str = "default"
    frame_ms: int = 32                 # 512 samples @ 16 kHz, matches Silero VAD window
    vad_threshold: float = 0.5
    min_silence_s: float = 0.4         # silence that ends an utterance (0.45 merges closely spoken sentences)
    min_speech_s: float = 0.25
    max_utterance_s: float = 20.0      # force-split long monologues
    pre_roll_s: float = 0.3
    echo_suppression: bool = True      # drop mic text that duplicates recent system text
    mic_start_muted: bool = False
    # WASAPI loopback delivers NO callbacks while nothing is playing (Windows idles the endpoint).
    # A ~-80 dBFS keep-alive tone on the same output keeps capture continuous (verified on this machine).
    loopback_keepalive: bool = True


@dataclass
class DiarizationConfig:
    enabled: bool = True
    model: str = "wespeaker_en_voxceleb_CAM++.onnx"   # file name inside models_dir
    threshold: float = 0.45            # cosine similarity to join an existing speaker
    enroll_min_s: float = 1.5          # utterances shorter than this never create/update speakers
    max_speakers: int = 8
    num_threads: int = 2


@dataclass
class AssistConfig:
    auto_answer_questions: bool = True # answer questions detected in system audio automatically
    coalesce_ms: int = 700             # wait for a follow-on sentence before answering
    style: str = "bullets"             # bullets | short | detailed
    persona: str = ""                  # optional: "I am a backend engineer interviewing for ..."
    min_question_words: int = 3
    # Autonomous mode: when the screen changes, SOLVE whatever task/problem is shown (a coding
    # problem, a question, a form) and show the finished answer, instead of proposing talking
    # points. Only asks the user to choose when the task genuinely forks (rendered as buttons).
    autonomous: bool = True
    # What is worth a model call -- on-screen questions here, and every new task hands-free mode
    # sees -- least to most permissive. "technical": only text with technical substance (theory,
    # concepts, design, math, code). "auto" (default, code-biased): questions and lines that read
    # as a task ("Implement...", "Given an array...", a technical term); personal-opinion
    # questions without technical substance are dropped, and in hands-free mode a big change of
    # view needs one such line among the new ones -- a chat or an email drifting into view does
    # not distract it. "all": read everything, answer as much as possible. Site polls / surveys /
    # feedback prompts (`ignore_patterns` and the built-in list, e.g. LeetCode's "Have you seen
    # this in an interview before?") are dropped at every level.
    focus: str = "auto"
    ignore_patterns: list[str] = field(default_factory=list)   # extra regexes (case-insensitive) to ignore
    # Language for solutions / the Code button when the task does not indicate one. A soft
    # default, never a lock: a SQL question still gets SQL, code in the visible editor keeps its
    # language, a language named in the conversation wins; only an open or ambiguous choice
    # falls back to this.
    primary_language: str = "Python"
    # Which libraries the generated code (solve, the Code button) may use when the task does not
    # say. A soft default like `primary_language`, never a lock: "without any libraries" / "basic
    # functions only" / "no imports" / "use whatever you like" in the conversation or the statement,
    # or the imports already in the editor, always win, and a change of ask mid-task redoes the
    # code. "none": the language's built-in constructs and basic built-in functions only -- no
    # imports, no library calls, the helpers written out. "basic" (default): the standard library's
    # commonly known basics (Python's collections / itertools / heapq / math / re, Java's java.util,
    # C++'s STL ...), no third-party package. "all": whatever library or importable function makes
    # the solution shortest and clearest, third-party packages included.
    libraries: str = "basic"
    # How much the generated code (solve, the Code button, and code continued by completions) is
    # commented. "none": no comments at all. "minimal": 2-3 comment lines before the important
    # steps only. "normal" (default): a short comment above every logical step plus a one-line
    # summary at the top. "detailed": the code opens with a comment block stating the requirements
    # and the pseudo-code -- the way an engineer works a challenge through in the editor -- and
    # then carries a comment above most lines. Comments are always whole lines of their own, so
    # completion mode can still let the user type past them (exocortex/plan.py).
    comments: str = "normal"
    # ---- pair programming / team coding (2026-10-20) ----
    # What the session is about: the project, the stack, today's goal ("FastAPI + Postgres
    # billing service; adding per-tenant rate limiting"). Handed to every answer, insight and
    # recap as standing context, so suggestions use the team's real names and constraints.
    # Empty = none. Also settable from the Assist menu ("Session context...").
    session_context: str = ""
    # Your role right now; steers the suggested questions and blind spots. "driver" (default):
    # you are typing -- questions to put to the navigator / team, what you may be missing while
    # heads-down. "navigator": a teammate is typing and you review -- questions to ask the driver,
    # review points on the code on screen. "participant": a design / team discussion. Pairs swap
    # roles often, so the Assist menu switches it in one click.
    role: str = "driver"
    # Proactive co-pilot insights, refreshed in the background into the Notes tab's Insights pane
    # (never interrupting the Assist tab or switching tabs): questions worth asking right now, and
    # blind spots / follow-ups (unhandled edge cases, missing tests, risky assumptions, things
    # to follow up on). Both are always available on demand from the Assist menu / Notes tab.
    suggest_questions: bool = True
    flag_blind_spots: bool = True
    # A background pass runs at most this often (seconds; 0 = on demand only), and only once
    # insight_min_new_words of new speech arrived since the last one -- or the screen (your
    # code) changed substantially. Every pass is one local model call: this is the GPU cost knob.
    insight_interval_s: int = 120
    insight_min_new_words: int = 60
    insight_max_items: int = 4         # bullets per section (questions / blind spots)


@dataclass
class NotesConfig:
    auto_interval_s: int = 180         # 0 disables automatic notes
    min_new_words: int = 120           # only regenerate when this much new speech arrived
    # "dev" (default): the same five headings, steered toward a working session -- what changed in
    # the code, bugs found and their causes, design choices and trade-offs, commands / config,
    # TODOs. "meeting": the original general-meeting wording.
    template: str = "dev"


@dataclass
class DashboardConfig:
    """Dashboard mode (2026-10-20): the overlay as a multi-column board -- session notes, action
    items (primary tasks with their sub-tasks and suggested prompts), goals (acceptance criteria),
    constraints, questions to ask, blind spots & follow-ups -- fed by a small database
    (exocortex.dashboard.store, SQLite in the session folder) that the model never holds: every
    refresh hands the model only the OPEN items plus what is new since the last refresh, and the
    model answers with the CHANGES (new / answered / done / dropped), which the store applies.
    The view is toggled with the ▦ title-bar button, Ctrl+Alt+D, or the "⋯" menu; the tracking
    runs whether or not the view is open. Screen capture / OCR keep their own cadence
    ([screen].interval_s) in every view mode: a refresh reads whatever the screen reader last saw.
    """
    enabled: bool = False              # start in the dashboard view (also mirrors the toggle; persisted)
    # Background tracking passes. While on, these REPLACE the Notes tab's background insights
    # passes ([assist].suggest_questions / flag_blind_spots) -- the Insights pane and insights.md
    # then show the dashboard's open questions and blind spots, so nothing runs twice.
    track: bool = True
    refresh_interval_s: int = 90       # a background refresh at most this often (0 = on demand only)
    min_new_words: int = 40            # ... once this much new speech arrived -- or the screen changed a lot
    max_open_items: int = 8            # open items per column handed to the model (oldest first; x2 for tasks and goals)
    max_new_items: int = 4             # new items the model may add per column (and per task) per refresh
    suggest_prompts: bool = True       # action items also get suggested prompts to type into Claude Code
    show_resolved: bool = True         # answered / done items stay visible, collapsed under the open ones
    resolved_keep: int = 12            # ... how many of them each column keeps on screen
    # Which columns the view shows (also the View menu's check marks). show_nextsteps is the Action
    # items column; show_prompts shows the suggested prompts under the action items they serve.
    show_notes: bool = True
    show_nextsteps: bool = True
    show_goals: bool = True
    show_constraints: bool = True
    show_questions: bool = True
    show_blindspots: bool = True
    show_prompts: bool = True
    width: int = 1180                  # window width in dashboard mode (the normal size is restored on exit)
    view_refresh_s: int = 5            # the view re-renders its timestamps this often (no model call)
    persist: bool = True               # keep the tracked items in <session folder>/dashboard.sqlite


@dataclass
class ProjectConfig:
    """What the model may look up beyond the transcript and the screen (2026-10-20): the
    repository being worked on, and the Claude Code toolkit installed in it. Nothing here is
    injected wholesale -- the repo is SEARCHED for the file names and identifiers a prompt
    mentions, and only the matching lines go in; the toolkit is a compact, auto-discovered
    catalog of what is installed (skills, agents, rules, workflows, hooks) plus, on mention,
    one part's full text. Edited on the settings dialog's "Project" tab.
    """
    # The project directory ("" = none): searched on demand, never read wholesale. Also where the
    # Claude Code toolkit is looked for (<repo_root>/.claude, <repo_root>/CLAUDE.local.md).
    repo_root: str = ""
    # With no repo_root: use the directory Claude Code was most recently run in (read from its
    # session logs under ~/.claude/projects), when that was within repo_auto_max_age_h hours. The
    # toolkit installed in that repository is then discovered with no setup at all.
    repo_auto: bool = True
    repo_auto_max_age_h: int = 12
    repo_search: bool = True           # search the repo for files / identifiers mentioned; inject matching lines only
    repo_max_files: int = 6            # files per lookup
    repo_max_chars: int = 1800         # characters of matched lines handed to the model
    repo_exclude: list[str] = field(default_factory=lambda: [
        ".git", "node_modules", ".venv", "venv", "dist", "build", "__pycache__", ".tools", "models",
        ".pytest_cache", ".mypy_cache", ".ruff_cache", ".idea", ".vscode", ".claude",
    ])
    # The Claude Code toolkit (claude-toolkit's skills, agents, rules, workflows, hooks) as
    # installed in the repo: auto-discovered from <repo_root>/.claude and CLAUDE.local.md, re-checked
    # every rescan_s, so an install or update mid-session is picked up without a restart. Nothing
    # about the toolkit is hard-coded: the catalog is built from the files' own frontmatter and
    # the house rules' "Loop" section.
    toolkit: bool = True
    toolkit_lookup: bool = True        # also hand over the full text of a skill / agent that is mentioned
    toolkit_in_answers: bool = True    # the catalog also feeds answers, ask and recap (not only the dashboard)
    toolkit_user_layer: bool = False   # also scan ~/.claude/skills and ~/.claude/agents (the opt-in user layer)
    # The toolkit CLONE (optional): richer one-line descriptions from its modules/index.json for
    # the parts installed in the repo. "" = look in ~/claude-toolkit and ~/.claude/dev-toolkit.
    toolkit_dir: str = ""
    toolkit_max_chars: int = 3000      # catalog budget (chars) in a system prompt; ~750 tokens, prefix-cached
    rescan_s: int = 30                 # how often the repo's .claude/ is re-checked


@dataclass
class ScreenConfig:
    """Live screen reading: capture -> local OCR -> context for answers / suggestions."""
    mode: str = "auto"                 # off | auto (active window) | monitor (whole primary monitor) | region
    # Capture cadence (OCR only runs when the image changed); hands-free mode samples at this
    # same interval. 1.5 s keeps the cheap sampling OCR under 10 % of one core.
    interval_s: float = 1.5
    ocr_engine: str = "auto"           # auto (Windows OCR if available, else rapidocr) | windows | rapidocr
    # Where RapidOCR's PP-OCR models run (Windows OCR is always CPU): "igpu" (an Intel iGPU via
    # OpenVINO -- measured 0.73 s / 0.74 s CPU per dense 2560x1600 screen -- or, when OpenVINO sees
    # no Intel GPU, a non-NVIDIA GPU via DirectML / onnxruntime-directml: the AMD Radeon 8060S /
    # 780M iGPUs, an AMD dGPU, an Intel GPU without OpenVINO), "npu" (Intel AI Boost via OpenVINO,
    # static shapes; 1.45 s / 0.6 s CPU), "cpu" (OpenVINO CPU when installed, else onnxruntime:
    # 1.9 s / 9.9 s CPU vs 3.7 s / 37 s CPU), "auto" = igpu > npu > cpu. Compiled kernels are
    # cached under <models_dir>/ov_cache. Never the discrete NVIDIA GPU (it holds Whisper and
    # llama-server).
    ocr_device: str = "auto"
    monitor: int = 0                   # 0 = primary monitor (for mode = monitor)
    region: list[int] = field(default_factory=lambda: [0, 0, 0, 0])  # x, y, w, h (screen px) for mode = region
    use_in_answers: bool = True        # include the current screen text in assist/ask prompts
    # Solve (autonomous) / propose talking points whenever the screen changes. Off by default
    # since 2026-10-20: while you drive, your editor changes constantly and every change would
    # start an unrequested solve that takes over the Assist tab. The co-pilot insights read the
    # screen instead, and Text / Code / Recap are one press away. Turn on for an interview or
    # coding-challenge screen.
    auto_suggest: bool = False
    suggest_min_interval_s: int = 20   # never suggest more often than this
    # Auto-answer a NEW question that appears on screen. Off by default since 2026-10-20 for the
    # same reason: in an editor or terminal, a comment, docstring or log line that reads as a
    # question would take over the Assist tab. Spoken questions are still answered.
    answer_screen_questions: bool = False
    # Screen text handed to the model (kept from the top). It changes every capture, so it is never
    # prefix-cached: with a CPU-bound model every 1000 chars cost ~1 s of prompt processing.
    max_text_chars: int = 2500
    save_text: bool = False            # write screen text snapshots into the session folder
    show_text: bool = False            # show the raw OCR readout in the Screen tab (off: less clutter)
    # Blank out this app's own visible windows (the overlay, the ghost text, menus) in every
    # captured frame before OCR. The overlay is an ordinary, capturable window, so without this
    # the screen text would include the co-pilot's own answers and suggested questions and feed
    # them back into the next prompt.
    exclude_own_windows: bool = True


@dataclass
class UIConfig:
    opacity: float = 0.93
    width: int = 460
    height: int = 680
    x: int = -1                        # -1 = top-right corner of the primary screen
    y: int = -1
    font_size: int = 11
    start_hidden: bool = False
    click_through: bool = False
    always_on_top: bool = True


@dataclass
class CompletionConfig:
    """Completion mode: inline ghost-text completions at the caret of whatever text field or code
    editor has focus, from the whole session's context (transcript, answers, notes, screen).

    The text under the caret is read through Windows accessibility (UI Automation / MSAA), never a
    keyboard hook; password fields and our own windows are always skipped.
    """
    enabled: bool = False              # start in completion mode
    poll_ms: int = 150                 # caret / typed-text sampling cadence (cheap; ~1-10 ms per probe)
    debounce_ms: int = 1200            # quiet typing time after a deviation before asking for a new completion
    min_interval_ms: int = 800         # never ask the model more often than this
    max_tokens: int = 120              # model completions: enough for several lines (streams as it goes)
    max_lines: int = 8                 # ghost-text lines shown at once (plan mode: lines of the solution)
    prefix_chars: int = 1500           # text before the caret handed to the model (clamped to the field)
    suffix_chars: int = 200            # text after the caret
    min_prefix_chars: int = 2          # do not complete an empty field unless it has a placeholder/label
    temperature: float = 0.2
    repeat_penalty: float = 1.1        # llama.cpp repeat_penalty for completions (1.0 = off); curbs "2. 3. 4." loops
    use_transcript: bool = True        # context sources ("Context" menu in the slim bar)
    use_screen: bool = True
    use_answers: bool = True
    use_notes: bool = True
    hint: str = ""                     # standing instruction, e.g. "Python 3, terse", "reply in French"
    ghost_color: str = "#9aa0a6"       # ghost text colour (semi-transparent grey reads on light and dark)
    ghost_opacity: float = 0.75
    font_scale: float = 1.0            # multiply the matched font size (fine-tune per DPI/app)
    default_font_pt: float = 11.0      # ghost font when the target's font/line height is unknown


@dataclass
class ExplainConfig:
    """Explanations (toggle: the 💡 button / Ctrl+Alt+E / the slim bar's Context menu).

    When on, a solve is a *plan*: a short walkthrough of the problem and the thought process is
    written BEFORE the solution, the code carries comments per ``[assist].comments``, and once the
    code has been typed out on screen an explanation of the finished solution follows. In
    completion mode the ghost text is served from the plan (no model call per line; comments are
    optional to type -- typing past one just drops it) and the walkthrough / explanation live in
    the slim bar's "Explain" drop-down.
    """
    enabled: bool = False
    walkthrough_max_tokens: int = 220  # extra budget a solve gets for the walkthrough part
    explain_max_tokens: int = 420      # budget for the explanation of the finished solution
    typed_fraction: float = 0.9        # share of the solution's code lines seen on screen that counts as "typed out"


@dataclass
class HandsFreeConfig:
    """Hands-free mode (the switch at the right of the tab strip, Ctrl+Alt+H).

    While on, the screen is sampled every ``[screen].interval_s`` (the ordinary capture
    interval, 1.5 s by default). A MASSIVE change between two consecutive samples -- a new
    question, a big scroll -- followed by the picture settling makes the app do by itself what
    the user does by hand today: Read (an accurate OCR pass) -> Checkpoint -> Text (the
    walkthrough) -> Code (the solution). Typing changes a line or two per sample and never
    counts. What scrolls out of view stays in the context handed to the model (cumulative OCR),
    and when the code on screen has drifted away from the app's own code, Code is regenerated
    (no checkpoint) so it builds on what was actually typed. The overlay collapses to a
    click-through output pane meanwhile (only the title bar and the switch stay clickable).
    """
    enabled: bool = False              # start in hands-free mode
    settle_s: float = 1.0              # the screen must stay unchanged this long before it is acted on
    # What hands-free answers is [assist].focus ("auto": any new task on screen; "technical": only
    # screens with a technical cue in the new text or code on screen); site polls / feedback
    # prompts ([assist].ignore_patterns and the built-in list) never count.
    # What counts as a NEW TASK between one screen sample and the next (exocortex.handsfree.is_new_task),
    # any one of:
    #  - new text: at least `min_new_lines` lines of prose (3+ words) that were not on screen before,
    #    are not an edit of a line that was (OCR noise, a counter, a typo fix) and were not typed
    #    progressively -- a new question on a slide, however short, even under a static header;
    #    a new question line (ends in "?" / starts with what-how-why...) always counts;
    #  - or most of the view changed at once: the share of lines that differ (1 - Jaccard of the
    #    two line sets) reaches `change_threshold` AND at least `min_changed_lines` differ -- a
    #    big scroll through code, a completely different screen.
    # Typing never gets there: the line being typed is an edit of itself, sample after sample.
    # Lines matching [assist].ignore_patterns (site polls, feedback prompts) never count.
    min_new_lines: int = 1
    change_threshold: float = 0.6
    min_changed_lines: int = 3         # for the "most of the view changed" rule (and the deviation rule)
    # When the settled screen shows code (a code block, an editor's gutter, a development window in
    # front such as VS Code / PyCharm / LeetCode, or a task that asks for code): "code" = run Code
    # only, with an ask that also covers questions ABOUT the code shown (its output, a bug, which
    # option is right) so the model never skips it; "code_first" = Code, then Text; "ignore" = the
    # ordinary order.
    on_code: str = "code"
    # Regenerate the code when the code on screen has drifted from the app's own solution: the
    # share of on-screen code lines that are NOT the app's (0 = off). Code only, no checkpoint.
    deviation_threshold: float = 0.5
    # Cumulative screen context handed to hands-free Text / Code jobs (what scrolled out of view
    # plus the current screen); larger than [screen].max_text_chars on purpose, the task's
    # statement and the code typed so far both have to fit.
    context_chars: int = 6000
    # Text is skipped automatically while explanations are on (the code answer already opens with a
    # walkthrough) and in completion mode (the slim bar has nowhere to show prose).
    run_text: bool = True              # write the plain-English walkthrough (Text) for each new task
    run_code: bool = True              # write the code (Code) for each new task
    code_first: bool = False           # Code before Text (default: Text streams first, Code follows)
    # Re-read the settled screen with the accurate RapidOCR engine ([screen].ocr_device) before
    # evaluating it and handing it to the model, when the sampling engine is Windows OCR.
    accurate_ocr: bool = True
    # Manual advance (the Manual switch next to Hands-free, Ctrl+Alt+J). While on, nothing starts
    # a model call by itself: watching, screen scans (OCR), the cumulative context, auto-
    # checkpoints, the transcript and question detection all keep running, but hands-free
    # Text/Code, spoken-question auto-answers, on-screen question answers, automatic solves /
    # suggestions, automatic notes and automatic explanations all wait. Generation happens only on
    # a press-and-hold of a mouse button, anywhere on screen, without bringing the overlay to the
    # front: the LEFT button held `manual_hold_s` makes Text, the MIDDLE button (the scroll wheel)
    # makes Code. Moving the pointer while holding (a drag, a selection) cancels the hold, so
    # ordinary clicking never triggers. Works with hands-free on (the gesture runs that step for
    # the task on screen, with the cumulative context) or off (a plain Text / Code).
    manual: bool = False
    manual_hold_s: float = 0.8         # press-and-hold time before a held button generates
    # Whether each action needs the press-and-hold, or a plain click is enough. Text (the left
    # button) needs the hold by default -- ordinary left clicks happen constantly and must never
    # generate. Code (the middle button) does NOT need it by default: a quick press of the scroll
    # wheel, or a touchpad three-finger tap mapped to "Middle mouse button" in Windows Settings >
    # Bluetooth & devices > Touchpad (a tap cannot be held), triggers Code right away. Moving the
    # pointer while the button is down (a drag, an autoscroll) always cancels.
    manual_hold_text: bool = True
    manual_hold_code: bool = False


@dataclass
class HotkeysConfig:
    toggle_visible: str = "ctrl+alt+space"
    answer_last: str = "ctrl+alt+a"
    ask_focus: str = "ctrl+alt+q"
    notes: str = "ctrl+alt+n"
    toggle_auto: str = "ctrl+alt+t"
    toggle_mic: str = "ctrl+alt+m"
    click_through: str = "ctrl+alt+c"
    # Nudge the panel 40 px. Off (empty) by default since 2026-10-20: Ctrl+Alt+arrows are VS Code's
    # add-cursor-above/below and JetBrains' navigate back/forward, which a driver uses constantly,
    # and a global hotkey swallows them in every app. Drag the title bar instead, or bind e.g.
    # "ctrl+alt+shift+left". Any hotkey left empty is simply not registered.
    move_left: str = ""
    move_right: str = ""
    move_up: str = ""
    move_down: str = ""
    select_region: str = "ctrl+alt+r"   # drag a screen region to read
    suggest_now: str = "ctrl+alt+s"     # suggestions from screen + transcript right now
    toggle_screen: str = "ctrl+alt+v"   # pause / resume screen reading
    toggle_completion: str = "ctrl+alt+k"   # completion mode (slim bar + ghost-text completions)
    toggle_explanations: str = "ctrl+alt+e" # explanations (walkthrough before, comments, explanation after)
    checkpoint: str = "ctrl+alt+b"      # checkpoint: the coding context (solve / code / completions) starts fresh
    # Pause / resume everything automatic (the Pause button). A bare key TAP (pressed and released
    # with no other key in between, under 0.6 s) rather than a combination: "rshift" (the right
    # Shift key -- typing capitals never trips it), "rctrl", "ralt", or a plain key such as
    # "right" (the arrow key, which is then swallowed while no modifier is held). A "+" combination
    # ("ctrl+alt+p") works too. "" = off.
    toggle_pause: str = "rshift"
    code: str = "ctrl+alt+g"            # write the code for the task discussed / shown since the last checkpoint
    toggle_handsfree: str = "ctrl+alt+h"   # hands-free mode (auto read -> checkpoint -> text -> code on screen change)
    toggle_manual: str = "ctrl+alt+j"   # manual advance (generate only on a press-and-hold: left = Text, middle = Code)
    insights: str = "ctrl+alt+w"        # refresh suggested questions + blind spots / follow-ups now
    recap: str = "ctrl+alt+y"           # recap the session so far (goal, done, current state, next)
    toggle_dashboard: str = "ctrl+alt+d"   # dashboard mode on / off (the ▦ title-bar button)
    quit: str = "ctrl+alt+x"


@dataclass
class PathsConfig:
    models_dir: str = ""               # "" -> <project>/models
    sessions_dir: str = ""             # "" -> ~/Documents/ExoCortex/sessions
    autosave_interval_s: int = 30


@dataclass
class Config:
    llm: LLMConfig = field(default_factory=LLMConfig)
    asr: ASRConfig = field(default_factory=ASRConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    diarization: DiarizationConfig = field(default_factory=DiarizationConfig)
    assist: AssistConfig = field(default_factory=AssistConfig)
    notes: NotesConfig = field(default_factory=NotesConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)
    project: ProjectConfig = field(default_factory=ProjectConfig)
    screen: ScreenConfig = field(default_factory=ScreenConfig)
    completion: CompletionConfig = field(default_factory=CompletionConfig)
    explain: ExplainConfig = field(default_factory=ExplainConfig)
    handsfree: HandsFreeConfig = field(default_factory=HandsFreeConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    hotkeys: HotkeysConfig = field(default_factory=HotkeysConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    path: Path | None = field(default=None, compare=False)

    # ---- derived paths ----
    def models_dir(self) -> Path:
        if self.paths.models_dir:
            return Path(self.paths.models_dir).expanduser()
        return project_root() / "models"

    def sessions_dir(self) -> Path:
        if self.paths.sessions_dir:
            return Path(self.paths.sessions_dir).expanduser()
        return Path.home() / "Documents" / "ExoCortex" / "sessions"


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def appdata_config_path() -> Path:
    base = os.environ.get("APPDATA") or str(Path.home())
    return Path(base) / "ExoCortex" / "config.toml"


def find_config_path() -> Path | None:
    for p in (Path.cwd() / "config.toml", project_root() / "config.toml", appdata_config_path()):
        if p.is_file():
            return p
    return None


def _apply(section_obj: Any, data: dict[str, Any], section_name: str) -> None:
    known = {f.name: f for f in fields(section_obj)}
    for k, v in data.items():
        if k not in known:
            log.warning("config: unknown key [%s].%s ignored", section_name, k)
            continue
        cur = getattr(section_obj, k)
        if isinstance(cur, bool):
            v = bool(v)
        elif isinstance(cur, int) and isinstance(v, (int, float)):
            v = int(v)
        elif isinstance(cur, float) and isinstance(v, (int, float)):
            v = float(v)
        setattr(section_obj, k, v)


def _migrate_llm_url_keys(llm_data: dict[str, Any]) -> None:
    """2026-09-27 (portable-install): fold the old [llm] ``base_url`` / ``ollama_url`` keys into
    the single ``server_url`` field, in place, BEFORE ``_apply`` runs -- so neither legacy name
    ever reaches ``_apply`` (which would otherwise warn "unknown key") and any config.toml
    written since 2026-09-20 keeps loading exactly as it did before this field existed.

    - Neither legacy key present: no-op.
    - ``server_url`` already present too: it wins: the legacy keys are dropped silently (no log).
    - Otherwise: the migrated value follows [llm].backend -- ``ollama_url`` for backend
      "ollama", ``base_url`` for backend "openai", else whichever of the two differs from its
      own standard default (so an explicit, still-default value on the OTHER key never wins by
      accident), else "". A legacy value that is identical to its standard default migrates to
      "" either way, so a bare default 8081/11434 never gets pinned into server_url. One INFO
      log line records what happened.

    review round two, 2026-09-27 (C1): there is deliberately ONE server_url field -- the fix for
    "both legacy keys customised to different addresses, the other one silently vanishes" is NOT a
    second field, it is making the loss loud and recoverable. When BOTH legacy keys were customised
    (each differs from ITS OWN standard default) to DIFFERENT addresses, the precedence above still
    decides which one is kept, but now: a WARNING (not INFO) names both old values, which one was
    kept, which one was dropped, and the exact two config.toml lines that would recover the dropped
    one (backend + server_url); and the dropped value is recorded on ``llm_data["dropped_legacy_url"]``
    -- a real ``LLMConfig`` field that ``_apply`` then sets like any other key, but one ``to_toml``
    never writes (metadata "toml": False, same trick as ``Config.path``) -- so `cfg.llm.doctor` / the
    UI can surface it, and it never gets persisted into config.toml as a fake "setting".
    """
    if "base_url" not in llm_data and "ollama_url" not in llm_data:
        return
    base_val = llm_data.pop("base_url", STANDARD_SERVER_URLS["openai"])
    ollama_val = llm_data.pop("ollama_url", STANDARD_SERVER_URLS["ollama"])
    if "server_url" in llm_data:
        return  # server_url wins; legacy keys dropped silently
    backend = str(llm_data.get("backend", "auto")).strip().lower()

    def _clean(value: Any, standard: str) -> Any:
        return "" if value == standard else value

    base_customised = base_val != STANDARD_SERVER_URLS["openai"]
    ollama_customised = ollama_val != STANDARD_SERVER_URLS["ollama"]

    kept_backend: str | None
    dropped_backend: str | None
    dropped_val: Any
    if backend == "ollama":
        chosen = _clean(ollama_val, STANDARD_SERVER_URLS["ollama"])
        kept_backend, dropped_backend, dropped_val = "ollama", "openai", base_val
    elif backend == "openai":
        chosen = _clean(base_val, STANDARD_SERVER_URLS["openai"])
        kept_backend, dropped_backend, dropped_val = "openai", "ollama", ollama_val
    else:
        chosen = ""
        kept_backend = dropped_backend = None
        dropped_val = None
        for value, standard, name in (
            (base_val, STANDARD_SERVER_URLS["openai"], "openai"),
            (ollama_val, STANDARD_SERVER_URLS["ollama"], "ollama"),
        ):
            if value != standard:
                chosen = value
                kept_backend = name
                break
        if kept_backend == "openai":
            dropped_backend, dropped_val = "ollama", ollama_val
        elif kept_backend == "ollama":
            dropped_backend, dropped_val = "openai", base_val
    llm_data["server_url"] = chosen
    log.info("config: [llm].base_url/ollama_url migrated to server_url=%r", chosen)

    # Loss only happens when BOTH keys were customised to genuinely different addresses -- one of
    # them equal to its own default, or both customised to the SAME address, loses nothing.
    lost = (
        base_customised and ollama_customised and base_val != ollama_val and kept_backend is not None
    )
    if lost:
        llm_data["dropped_legacy_url"] = dropped_val
        log.warning(
            "config: [llm].base_url=%r and [llm].ollama_url=%r were both customised to different "
            "addresses; there is one server_url field by design, so only one survives. Kept the %s "
            "address (backend=%r) as server_url=%r and dropped the %s address %r -- it is NOT lost, "
            "it is recorded on cfg.llm.dropped_legacy_url this run, but config.toml will not "
            "remember it. To use the dropped address instead, put these two lines in config.toml:\n"
            '  backend = "%s"\n'
            '  server_url = "%s"',
            base_val, ollama_val, kept_backend, backend, chosen, dropped_backend, dropped_val,
            dropped_backend, dropped_val,
        )


#: The keys ``[project]`` took over from ``[context]`` (2026-10-29): until then the repository /
#: toolkit settings lived in a ``[context]`` section, which exo-cortex's reference documents and
#: context window now own. No upstream ``[context]`` key shares one of these names.
_PROJECT_KEYS = frozenset(f.name for f in fields(ProjectConfig))


def _migrate_project_keys(data: dict[str, Any]) -> None:
    """Move the repository / toolkit keys of an older config.toml from ``[context]`` to
    ``[project]``, in place, BEFORE ``_apply`` runs. A key already present in ``[project]`` wins;
    the next save writes everything under ``[project]``. A ``[context]`` section left empty by
    the move is dropped, so it is not reported as unknown."""
    old = data.get("context")
    if not isinstance(old, dict):
        return
    moved = {k: old.pop(k) for k in [k for k in old if k in _PROJECT_KEYS]}
    if not moved:
        return
    project = data.setdefault("project", {})
    if isinstance(project, dict):
        for k, v in moved.items():
            project.setdefault(k, v)
    if not old:
        del data["context"]
    log.info("config: moved [context] %s to [project]", ", ".join(sorted(moved)))


def load_config(path: str | os.PathLike[str] | None = None) -> Config:
    cfg = Config()
    p = Path(path) if path else find_config_path()
    if p is None:
        return cfg
    with open(p, "rb") as f:
        data = tomllib.load(f)
    _migrate_project_keys(data)
    for sec_name, sec_data in data.items():
        if not hasattr(cfg, sec_name) or not isinstance(sec_data, dict):
            log.warning("config: unknown section [%s] ignored", sec_name)
            continue
        if sec_name == "llm":
            _migrate_llm_url_keys(sec_data)
        _apply(getattr(cfg, sec_name), sec_data, sec_name)
    cfg.path = p
    return cfg


def _toml_value(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, str):
        return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_toml_value(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{ " + ", ".join(f"{k} = {_toml_value(x)}" for k, x in v.items()) + " }"
    raise TypeError(f"unsupported TOML value: {v!r}")


def to_toml(cfg: Config) -> str:
    lines: list[str] = ["# ExoCortex configuration", ""]
    for f in fields(cfg):
        if f.name == "path":
            continue
        sec = getattr(cfg, f.name)
        lines.append(f"[{f.name}]")
        for sf in fields(sec):
            # review round two, 2026-09-27: non-persisted fields (e.g. LLMConfig.dropped_legacy_url)
            # opt out via metadata, the same way Config.path is skipped above by name.
            if sf.metadata.get("toml", True) is False:
                continue
            lines.append(f"{sf.name} = {_toml_value(getattr(sec, sf.name))}")
        lines.append("")
    return "\n".join(lines)


def save_config(cfg: Config, path: str | os.PathLike[str] | None = None) -> Path:
    p = Path(path) if path else (cfg.path or project_root() / "config.toml")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(to_toml(cfg), encoding="utf-8")
    cfg.path = p
    return p


def config_as_dict(cfg: Config) -> dict[str, Any]:
    d = dataclasses.asdict(cfg)
    d.pop("path", None)
    return d

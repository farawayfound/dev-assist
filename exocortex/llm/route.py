"""Backend routing: picks which LLM backend to talk to, or spawn, at startup.

``[llm].backend = "auto"`` (the default since the 2026-09-25 GPU-support work) no longer means
"assume LM Studio's CUDA pack": this machine (an AMD Ryzen AI MAX+ 395 / Radeon 8060S laptop, no
NVIDIA driver at all) has to land on LM Studio's Vulkan or ROCm llama.cpp pack instead, and a
machine with neither a usable GPU pack nor a GGUF library has to fall through to Ollama, then any
OpenAI-compatible server, before giving up with an actionable message. :func:`resolve_route` is
the single place that decision is made, with every probe (find the llama-server binary, scan the
model library, GET a URL, launch Ollama, sleep/clock) injectable so it is unit-testable on any
machine; only the *defaults* touch this one's disk/network.

HARD RULE (owner, 2026-09-25): an explicit ``[llm].backend`` is returned as-is, with NO probing --
this is what keeps every NVIDIA-machine code path (where the field is usually left "auto" but was
tuned assuming CUDA) working exactly as before once the other agents' vendor-aware
``find_server_binary``/``primary_gpu`` land: "auto" simply now also succeeds on machines that used
to have no viable backend at all.

Rule order for "auto" (first hit wins): llamacpp (a runtime pack + >=1 GGUF) -> ollama (already
running, or started here) -> openai (any server at ``cfg.effective_server_url("openai")``) ->
"none" (nothing usable; the reason names what to install for *this* GPU vendor).
:func:`resolve_route` never raises: a probe that blows up is treated as "not available", exactly
like a probe that answers "no" cleanly.

2026-09-27 (portable-install): the Ollama probe (both the plain check and the autostart poll) no
longer tries a single URL -- it walks :func:`ollama_candidates` (the configured
``[llm].server_url`` first, then Ollama's own standard address), so a ``server_url`` that was set
for a different backend (or left at its default) never hides a real Ollama server running on the
standard port, and vice versa. ``checked`` grows one ``"ollama: <url> <status>"`` line per
candidate tried, in order, until one answers 200 or the list is exhausted.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from exocortex.config import STANDARD_SERVER_URLS, LLMConfig

log = logging.getLogger(__name__)

#: Fast, local-connect probes only (Engine.start() must return quickly).
_PROBE_TIMEOUT_S = 1.5
#: How long the Ollama autostart path waits for "ollama serve" to answer /api/tags.
_OLLAMA_START_TIMEOUT_S = 20.0
_OLLAMA_POLL_INTERVAL_S = 0.5

HttpGet = Callable[[str, float], "tuple[int, str]"]


@dataclass(frozen=True)
class Route:
    """The backend Engine.start() should use, and why -- shown by the doctor and logged."""

    backend: str                       # "llamacpp" | "ollama" | "openai" | "none"
    base_url: str
    reason: str                        # one sentence for logs / status / doctor
    checked: tuple[str, ...] = ()       # what was probed, one entry each ("llamacpp: ok (...)", ...)
    binary: Any | None = None          # exocortex.llm.server.ServerBinary when backend == "llamacpp"
    model_count: int = 0


def describe_route(route: Route) -> str:
    """Multi-line description for ``--doctor``: backend, reason, then every probe that ran."""
    lines = [f"backend: {route.backend}", f"reason: {route.reason}"]
    lines.extend(f"  - {c}" for c in route.checked)
    return "\n".join(lines)


# ---- default probes (only these touch the real machine) --------------------------------------
def _default_http_get(url: str, timeout: float) -> "tuple[int, str]":
    """(status, body); every transport failure maps to (0, "") -- like instance._default_http_get."""
    import httpx

    try:
        r = httpx.get(url, timeout=timeout)
        return r.status_code, r.text
    except Exception:  # noqa: BLE001 - no server there is the normal case on most machines
        log.debug("route: GET %s failed", url, exc_info=True)
        return 0, ""


def _safe_get(get: HttpGet, url: str, timeout: float) -> "tuple[int, str]":
    """Never let a misbehaving injected ``http_get`` escape (the contract maps failures to (0, ""),
    but 'never raises' must not depend on an injectable keeping its promises)."""
    try:
        return get(url, timeout)
    except Exception:  # noqa: BLE001
        log.debug("route: GET %s raised", url, exc_info=True)
        return 0, ""


def _default_find_binary() -> Callable[..., Any]:
    from exocortex.llm import server

    def find(explicit: str, *, vendor: str | None = None, backend: str = "auto") -> Any:
        return server.find_server_binary(explicit, vendor=vendor, backend=backend)

    return find


def _default_find_binary_for(cfg: LLMConfig) -> Callable[..., Any]:
    """2026-09-27 (portable-install): the real probe honours the runtime settings the launcher's
    setup step used -- `[llm].llama_source` / `llama_upstream_build` / `runtime_dir` -- so a
    build `python -m exocortex.setup` installed under a custom `runtime_dir` is found here too.
    Never downloads (no `ensure`): a probe must stay fast and offline."""
    from exocortex.llm import server

    def find(explicit: str, *, vendor: str | None = None, backend: str = "auto") -> Any:
        kwargs: dict[str, Any] = {"vendor": vendor, "backend": backend}
        if cfg.llama_source:
            kwargs["source"] = cfg.llama_source
        if cfg.llama_upstream_build:
            kwargs["upstream_build"] = cfg.llama_upstream_build
        if cfg.runtime_dir:
            kwargs["upstream_root"] = Path(cfg.runtime_dir).expanduser()
        return server.find_server_binary(explicit, **kwargs)

    return find


def _default_scan(cfg: LLMConfig) -> Callable[[], list]:
    """2026-09-27 (portable-install): the app's own library PLUS LM Studio's folder
    (:func:`exocortex.llm.library.scan_library`) -- the same merged view
    `LlamaServerManager.available_models()` serves from. Scanning LM Studio's folder alone (the
    pre-2026-09-27 probe) reported "no models" on exactly the machine this pass is for: a fresh
    install whose only GGUF was just pulled into ``models/llm``."""
    from exocortex.llm import server

    def scan() -> list:
        try:
            from exocortex.llm import library
        except ImportError:
            return server.scan_models(server.lmstudio_models_dir(cfg))
        return library.scan_library(cfg)

    return scan


def _safe_gpu_vendor() -> str | None:
    """The primary GPU's vendor (exocortex.gpu), or ``None`` when it cannot be determined -- never
    raises, so a DXGI hiccup degrades to "no vendor hint" rather than taking startup down."""
    try:
        from exocortex import gpu

        return gpu.gpu_vendor()
    except Exception:  # noqa: BLE001
        log.debug("route: gpu_vendor() failed", exc_info=True)
        return None


def ollama_installed() -> bool:
    """Ollama on PATH (``ollama.exe``), or at its default per-user install location."""
    if shutil.which("ollama"):
        return True
    local = os.environ.get("LOCALAPPDATA")
    if local:
        exe = Path(local) / "Programs" / "Ollama" / "ollama.exe"
        if exe.is_file():
            return True
    return False


def _default_ollama_launcher() -> None:
    """``ollama serve``, hidden, detached from us -- it is the owner's process from here on, and
    must survive our exit (never waited on, never added to our own Job Object)."""
    creation = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    subprocess.Popen(
        ["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL, creationflags=creation,
    )


def _count_ollama_models(body: str) -> int:
    try:
        data = json.loads(body)
    except (ValueError, TypeError):
        return 0
    items = data.get("models") if isinstance(data, dict) else None
    return len(items) if isinstance(items, list) else 0


def ollama_candidates(cfg: LLMConfig) -> list[str]:
    """Ollama URLs to try, in order: the configured server (``[llm].server_url``, or its
    standard address when unset) first, then Ollama's own standard address again -- deduplicated.

    Covers both directions a single ``server_url`` field can go wrong for "auto" routing: a
    ``server_url`` that happens to point Ollama at a non-default port (tried first, so it is
    found without ever touching the standard one), and a ``server_url`` that is actually some
    OTHER (openai) server, or was simply never set for Ollama at all -- either way the standard
    Ollama address still gets a look before giving up on Ollama entirely. Used by
    :func:`resolve_route`'s "auto" probe and by the doctor.
    """
    urls: list[str] = []
    for url in (cfg.effective_server_url("ollama"), STANDARD_SERVER_URLS["ollama"]):
        if url and url not in urls:
            urls.append(url)
    return urls


def _install_hint(vendor: str | None) -> str:
    """2026-09-27 (portable-install): nothing has to be installed by hand any more -- the launcher's
    setup step fetches the llama.cpp runtime for this GPU and the Models window pulls a model; LM
    Studio and Ollama are alternatives, not requirements."""
    gpu_name = vendor or "this GPU"
    return (
        f"run run.bat (or 'python -m exocortex.setup') to fetch the llama.cpp runtime for {gpu_name} and pull "
        "a model, or pull one from ... > Models... in the overlay; LM Studio (a downloaded GGUF) or Ollama "
        "('ollama pull <model>') work too"
    )


# ---- llamacpp probe -----------------------------------------------------------------------
def _try_llamacpp(
    cfg: LLMConfig, vendor: str | None, find_binary: Callable[..., Any] | None, scan: Callable[[], list] | None,
) -> "tuple[Any | None, int, str | None]":
    """(binary, model_count, why_not); ``why_not`` is ``None`` only on outright success."""
    find = find_binary or _default_find_binary_for(cfg)
    try:
        binary = find(cfg.llama_server_path, vendor=vendor, backend=cfg.llama_backend)
    except Exception as exc:  # noqa: BLE001 - a missing pack is an ordinary "no" here
        return None, 0, str(exc)
    scan_fn = scan or _default_scan(cfg)
    try:
        models = scan_fn()
    except Exception as exc:  # noqa: BLE001
        return binary, 0, f"found {binary.flavor} {binary.version} but the model scan failed: {exc}"
    if not models:
        return binary, 0, f"found {binary.flavor} {binary.version} but no GGUF models are in the library"
    return binary, len(models), None


# ---- the router ---------------------------------------------------------------------------
def resolve_route(
    cfg: LLMConfig,
    *,
    vendor: str | None = None,
    find_binary: Callable[..., Any] | None = None,
    scan: Callable[[], list] | None = None,
    http_get: HttpGet | None = None,
    start_ollama: bool = False,
    ollama_launcher: Callable[[], None] | None = None,
    clock: Callable[[], float] | None = None,
    sleep: Callable[[float], None] | None = None,
) -> Route:
    """Pick the backend for ``cfg`` (see the module docstring for the rules). Never raises."""
    backend = (cfg.backend or "auto").strip().lower()
    if backend in ("llamacpp", "ollama", "openai"):
        base_url = "" if backend == "llamacpp" else cfg.effective_server_url(backend)
        return Route(backend=backend, base_url=base_url, reason=f"configured: [llm].backend = {backend!r}")

    get = http_get or _default_http_get
    clock = clock or time.monotonic
    sleep = sleep or time.sleep
    actual_vendor = vendor if vendor is not None else _safe_gpu_vendor()
    checked: list[str] = []

    binary, model_count, why_not = _try_llamacpp(cfg, actual_vendor, find_binary, scan)
    if binary is not None and why_not is None and model_count >= 1:
        reason = f"llama.cpp {binary.flavor} {binary.version} with {model_count} GGUF model(s) in the library"
        checked.append(f"llamacpp: ok ({reason})")
        return Route("llamacpp", "", reason, tuple(checked), binary, model_count)
    llamacpp_why = why_not or "no llama-server runtime pack found for this GPU"
    checked.append(f"llamacpp: {llamacpp_why}")

    ollama_urls = ollama_candidates(cfg)
    for url in ollama_urls:
        status, body = _safe_get(get, url + "/api/tags", _PROBE_TIMEOUT_S)
        checked.append(f"ollama: {url} {status}")
        if status == 200:
            n = _count_ollama_models(body)
            return Route("ollama", url, f"Ollama at {url} ({n} models)", tuple(checked))

    if start_ollama and ollama_installed():
        launcher = ollama_launcher or _default_ollama_launcher
        try:
            launcher()
        except Exception as exc:  # noqa: BLE001 - autostart failing must not take resolve_route down
            checked.append(f"ollama: launch failed ({exc})")
        else:
            deadline = clock() + _OLLAMA_START_TIMEOUT_S
            while clock() < deadline:
                for url in ollama_urls:
                    status, body = _safe_get(get, url + "/api/tags", _PROBE_TIMEOUT_S)
                    if status == 200:
                        n = _count_ollama_models(body)
                        checked.append(f"ollama: {url} {status} (started just now)")
                        return Route("ollama", url, f"started Ollama at {url} ({n} models)", tuple(checked))
                sleep(_OLLAMA_POLL_INTERVAL_S)
            checked.append(
                f"ollama: did not come up within {_OLLAMA_START_TIMEOUT_S:.0f}s of starting it "
                f"(tried {', '.join(ollama_urls)})"
            )
    elif start_ollama:
        checked.append("ollama: not installed (no ollama.exe on PATH or in %LOCALAPPDATA%\\Programs\\Ollama)")

    base_url = cfg.effective_server_url("openai")
    status, _body = _safe_get(get, base_url + "/v1/models", _PROBE_TIMEOUT_S)
    checked.append(f"openai: {base_url} {status}")
    if status == 200:
        return Route("openai", base_url, f"OpenAI-compatible server at {base_url}", tuple(checked))

    reason = f"no local LLM found ({llamacpp_why}); {_install_hint(actual_vendor)}"
    return Route("none", "", reason, tuple(checked))

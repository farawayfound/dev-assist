"""HTTP clients for the LLM backends: model discovery, health checks and streaming generation.

`LMStudioClient` talks to LM Studio's server, any OpenAI-compatible server (`base_url`), and the
llama-server ExoCortex spawns itself (backend "llamacpp" -- `set_base_url` points it at the
ephemeral port). `OllamaClient` (below) talks to Ollama's own native API instead, added
2026-09-25 for machines without an NVIDIA GPU that would rather run models through Ollama than
LM Studio's llama.cpp packs; it shares everything else (streaming, cancellation, `<think>`
splitting, stats) with its parent.

Synchronous, built on `httpx`, intended to be called from worker threads (never
the Qt UI thread). Every public method maps connection failures and HTTP error
responses to `LLMError` with a human-readable message; nothing else raises.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import httpx

from exocortex.config import LLMConfig

log = logging.getLogger(__name__)

_CONNECT_TIMEOUT_S = 3.0
_ENSURE_LOADED_TIMEOUT_S = 180.0


@dataclass(frozen=True)
class ModelInfo:
    """One model as reported by LM Studio."""

    id: str
    loaded: bool | None
    arch: str = ""
    quant: str = ""
    context_length: int | None = None


@dataclass(frozen=True)
class Health:
    """Result of a reachability + model-availability check."""

    reachable: bool
    model_listed: bool
    model_loaded: bool | None
    detail: str = ""


@dataclass(frozen=True)
class StreamStats:
    """Timing/throughput summary for one completed (or cancelled) stream."""

    ttft_s: float
    tokens: int
    elapsed_s: float
    tok_per_s: float
    cancelled: bool
    #: 2026-10-06 (context library): what the server says it processed for the PROMPT of this
    #: request (the whole prompt, cached part included) and how much of it came from the KV prefix
    #: cache. None when the backend did not report it. They sit at the END so positional
    #: construction (tests, older callers) keeps working. `ContextManager` calibrates its
    #: chars-per-token estimate from `prompt_tokens`; the Models window's usage panel shows both.
    prompt_tokens: int | None = None
    cached_tokens: int | None = None
    #: 2026-10-07 (capabilities): why the stream ended -- "stop" (end of text, or one of the
    #: `stop` strings: llama-server / OpenAI `finish_reason`, Ollama `done_reason`), "length"
    #: (`max_tokens` reached) or "" (the backend did not say, or the stream was cancelled). The
    #: tool loop uses it to tell a call that ended on its closing tag from one cut short.
    finish_reason: str = ""


@dataclass(frozen=True)
class ChatDelta:
    """One incremental piece of a chat completion, with reasoning split out."""

    content: str = ""
    reasoning: str = ""


class LLMError(Exception):
    """Raised for any LM Studio connectivity or protocol error."""


class _ThinkSplitter:
    """Splits inline ``<think>...</think>`` spans out of streamed text.

    Feed arbitrary chunks of streamed text and get back the content-only and
    reasoning-only portions of that chunk. A tag that arrives split across two
    chunks (e.g. ``"<th"`` then ``"ink>"``) is handled by holding back the
    ambiguous suffix until the next `feed` call resolves it.
    """

    OPEN = "<think>"
    CLOSE = "</think>"
    _MAX_TAG_LEN = max(len(OPEN), len(CLOSE))

    def __init__(self) -> None:
        self._buf = ""
        self._in_think = False

    def feed(self, text: str) -> tuple[str, str]:
        """Return `(content_piece, reasoning_piece)` for this chunk of text."""
        self._buf += text
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        while True:
            tag = self.CLOSE if self._in_think else self.OPEN
            idx = self._buf.find(tag)
            if idx == -1:
                break
            before, self._buf = self._buf[:idx], self._buf[idx + len(tag):]
            (reasoning_parts if self._in_think else content_parts).append(before)
            self._in_think = not self._in_think
        hold = self._holdback_len()
        emit_end = len(self._buf) - hold
        emit, self._buf = self._buf[:emit_end], self._buf[emit_end:]
        if emit:
            (reasoning_parts if self._in_think else content_parts).append(emit)
        return "".join(content_parts), "".join(reasoning_parts)

    def flush(self) -> tuple[str, str]:
        """Return any buffered leftover text (e.g. an unterminated tag prefix)."""
        text, self._buf = self._buf, ""
        if not text:
            return "", ""
        return ("", text) if self._in_think else (text, "")

    def _holdback_len(self) -> int:
        limit = min(self._MAX_TAG_LEN, len(self._buf))
        for n in range(limit, 0, -1):
            suffix = self._buf[-n:]
            if self.OPEN.startswith(suffix) or self.CLOSE.startswith(suffix):
                return n
        return 0


def _completion_tokens_from_chunk(chunk: dict[str, Any]) -> int | None:
    """Extract a completion-token count from an SSE chunk's usage/stats field, if present."""
    usage = chunk.get("usage")
    if isinstance(usage, dict):
        val = usage.get("completion_tokens")
        if isinstance(val, int):
            return val
    stats = chunk.get("stats")
    if isinstance(stats, dict):
        for key in ("completion_tokens", "predicted_tokens_count", "tokens_predicted", "generated_tokens", "tokens"):
            val = stats.get(key)
            if isinstance(val, int):
                return val
    timings = chunk.get("timings")  # llama-server adds this to its final streamed chunk
    if isinstance(timings, dict):
        val = timings.get("predicted_n")
        if isinstance(val, int):
            return val
    return None


def _int_or_none(val: Any) -> int | None:
    """`val` when it is a real (non-bool) int, else None."""
    return val if isinstance(val, int) and not isinstance(val, bool) else None


def _prompt_tokens_from_chunk(chunk: dict[str, Any]) -> tuple[int | None, int | None]:
    """`(prompt_tokens, cached_tokens)` from an SSE chunk, either part None when absent.

    2026-10-06 (context library). Shapes seen: OpenAI `usage.prompt_tokens` (+
    `usage.prompt_tokens_details.cached_tokens` when the server reports a cache hit); LM Studio's
    `stats` block (any of a few key spellings); llama-server's `timings` on its final streamed
    chunk, where `prompt_n` is only the part it had to EVALUATE and `cache_n` the part it reused,
    so the whole prompt is their sum -- reporting `prompt_n` alone would make a warm-cache turn
    look like a 40-token prompt and wreck the chars-per-token calibration built on it. Sources
    are consulted in that order and the first to give a number for a part wins.
    """
    prompt: int | None = None
    cached: int | None = None
    usage = chunk.get("usage")
    if isinstance(usage, dict):
        prompt = _int_or_none(usage.get("prompt_tokens"))
        details = usage.get("prompt_tokens_details")
        if isinstance(details, dict):
            cached = _int_or_none(details.get("cached_tokens"))
    stats = chunk.get("stats")
    if isinstance(stats, dict):
        if prompt is None:
            for key in ("prompt_tokens", "prompt_tokens_count", "input_tokens", "prompt_eval_count"):
                prompt = _int_or_none(stats.get(key))
                if prompt is not None:
                    break
        if cached is None:
            for key in ("cached_tokens", "cache_tokens", "prompt_cached_tokens"):
                cached = _int_or_none(stats.get(key))
                if cached is not None:
                    break
    timings = chunk.get("timings")  # llama-server adds this to its final streamed chunk
    if isinstance(timings, dict):
        evaluated = _int_or_none(timings.get("prompt_n"))
        reused = _int_or_none(timings.get("cache_n"))
        if cached is None:
            cached = reused
        if prompt is None and evaluated is not None:
            prompt = evaluated + (reused or 0)
    return prompt, cached


def _finish_reason_from_chunk(chunk: dict[str, Any]) -> str | None:
    """The stream's end reason, when this chunk carries one: OpenAI-style
    ``choices[0].finish_reason`` ("stop" / "length"; LM Studio and llama-server send it on the
    last content chunk, None before) or Ollama's ``done_reason``. None when absent."""
    choices = chunk.get("choices") or []
    if choices and isinstance(choices[0], dict):
        reason = choices[0].get("finish_reason")
        if isinstance(reason, str) and reason:
            return reason
    reason = chunk.get("done_reason")
    if isinstance(reason, str) and reason:
        return reason
    return None


def _extract_items(data: Any) -> list[dict[str, Any]]:
    """Pull the list of model entries out of a `/v1/models` or `/api/v0/models` payload."""
    items = data.get("data", []) if isinstance(data, dict) else data
    return list(items) if isinstance(items, list) else []


class LMStudioClient:
    """httpx-based client for LM Studio. Sync API, used from worker threads."""

    def __init__(self, cfg: LLMConfig, *, transport: httpx.BaseTransport | None = None) -> None:
        self.cfg = cfg
        self._local = threading.local()
        self._transport = transport
        base = cfg.effective_server_url("openai")
        self._base_url = base
        self._client = self._build_client(base)
        # Set by health()/ensure_loaded() when cfg.model is "auto"/"" and the server lists more
        # than one model: the id actually picked, so requests carry a real model id while
        # cfg.model itself stays "auto" (and so never gets written back into config.toml -- see
        # `model_id`). None until the first successful resolution.
        self.resolved_model: str | None = None

    @property
    def model_id(self) -> str:
        """The model id to put in request bodies: `cfg.model` verbatim, unless it is empty or
        "auto", in which case whatever `health()`/`ensure_loaded()` last resolved it to (or an
        empty string before either has run once)."""
        model = self.cfg.model
        if model and model != "auto":
            return model
        return self.resolved_model or ""

    def _build_client(self, base_url: str) -> httpx.Client:
        timeout = httpx.Timeout(self.cfg.request_timeout_s, connect=_CONNECT_TIMEOUT_S)
        client_kwargs: dict[str, Any] = {"base_url": base_url, "timeout": timeout}
        if self._transport is not None:
            client_kwargs["transport"] = self._transport
        return httpx.Client(**client_kwargs)

    @property
    def base_url(self) -> str:
        return str(self._client.base_url).rstrip("/")

    def set_base_url(self, base_url: str) -> None:
        """Point the client at another server (e.g. the llama-server the app just spawned).

        `cfg.base_url` is left untouched on purpose: for the llamacpp backend the port is ephemeral
        and must not be persisted into config.toml.
        """
        old = self._client
        self._base_url = base_url
        self._client = self._build_client(base_url)
        try:
            old.close()
        except Exception:  # noqa: BLE001
            pass

    def close(self) -> None:
        """Release the connection pool."""
        try:
            self._client.close()
        except Exception:  # noqa: BLE001
            pass

    @property
    def last_stats(self) -> StreamStats | None:
        """Stats for the most recently completed stream call on the *current* thread.

        Stored per-thread: `Assistant` streams answers and notes from separate
        worker threads that share one client, and a single shared attribute would
        let one thread's stats clobber another's between the moment a stream
        finishes and the moment its caller reads this property.
        """
        return getattr(self._local, "value", None)

    @last_stats.setter
    def last_stats(self, value: StreamStats | None) -> None:
        self._local.value = value

    # ---- discovery -----------------------------------------------------

    def list_models(self) -> list[ModelInfo]:
        """List models known to LM Studio, preferring the richer `/api/v0/models`."""
        resp = self._get("/api/v0/models")
        if resp.status_code < 400:
            try:
                return [self._parse_v0_model(item) for item in _extract_items(resp.json())]
            except (ValueError, KeyError, TypeError) as exc:
                log.debug("client: unparsable /api/v0/models response, falling back: %s", exc)
        else:
            log.debug("client: /api/v0/models -> HTTP %s, falling back to /v1/models", resp.status_code)
        resp = self._get("/v1/models")
        if resp.status_code >= 400:
            raise LLMError(self._status_error_message(resp))
        return [ModelInfo(id=str(item.get("id", "")), loaded=None) for item in _extract_items(resp.json())]

    def health(self) -> Health:
        """Check reachability and whether `cfg.model` is listed/loaded.

        `cfg.model` "auto" (or ""): resolve it against a multi-model server -- the first LOADED
        model, else the first listed -- and remember the pick in `resolved_model` (`model_id`,
        and so every request body, then carries that real id); reported in `detail` too.
        """
        try:
            models = self.list_models()
        except LLMError as exc:
            return Health(reachable=False, model_listed=False, model_loaded=None, detail=str(exc))
        if not self.cfg.model or self.cfg.model == "auto":
            if not models:
                return Health(reachable=True, model_listed=False, model_loaded=None, detail="no models listed")
            chosen = next((m for m in models if m.loaded), models[0])
            self.resolved_model = chosen.id
            return Health(reachable=True, model_listed=True, model_loaded=chosen.loaded,
                          detail=f"auto-resolved to {chosen.id}")
        match = next((m for m in models if m.id == self.cfg.model), None)
        if match is None and len(models) == 1 and models[0].loaded is None:
            # Single-model servers (llama-server) report the GGUF path as the id, ignore the
            # "model" field in requests and answer /health with 200 once the model is loaded.
            single = models[0]
            try:
                healthy = self._client.get("/health").status_code == 200
            except httpx.HTTPError:
                healthy = False
            if healthy:
                return Health(reachable=True, model_listed=True, model_loaded=True, detail=f"single model: {single.id}")
        if match is None:
            detail = f"model '{self.cfg.model}' not found among {len(models)} listed model(s)"
            return Health(reachable=True, model_listed=False, model_loaded=None, detail=detail)
        return Health(reachable=True, model_listed=True, model_loaded=match.loaded, detail="")

    def ensure_loaded(self) -> bool:
        """If the model is not (known to be) loaded, ask LM Studio to JIT-load it."""
        h = self.health()
        if h.model_loaded is True:
            return True
        if not self.cfg.auto_load_model:
            return False
        body = {"model": self.model_id, "prompt": "hi", "max_tokens": 1}
        try:
            resp = self._client.post(
                "/v1/completions", json=body, timeout=httpx.Timeout(_ENSURE_LOADED_TIMEOUT_S, connect=_CONNECT_TIMEOUT_S)
            )
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            raise LLMError(self._connect_message()) from exc
        except httpx.TimeoutException as exc:
            raise LLMError(
                f"Timed out waiting for LM Studio to load model '{self.model_id}' "
                f"(>{_ENSURE_LOADED_TIMEOUT_S:.0f}s)."
            ) from exc
        except httpx.TransportError as exc:
            raise LLMError(f"LM Studio request failed while loading model '{self.model_id}': {exc}") from exc
        if resp.status_code >= 400:
            raise LLMError(self._status_error_message(resp))
        return True

    def context_window(self) -> int | None:
        """The per-request context (tokens) the server will actually give one request, or None.

        2026-10-06 (context library): `Assistant` sizes every prompt to this instead of trusting
        the configured `ctx_size` (a per-model tuning profile may override it, and an LM Studio
        model may be loaded with any length; measured the same day: with `--parallel 2` llama-server
        runs `--kv-unified` and reports the FULL ctx_size per slot -- the two slots share that one
        pool, which `ContextManager.job_limit` accounts for). llama-server
        answers `GET /props` with `default_generation_settings.n_ctx`; LM Studio has no `/props`,
        so its `/api/v0/models` entry for `model_id` is the fallback (`loaded_context_length`,
        else `max_context_length`). Cheap (the shared client, default timeouts) and NEVER raises:
        a 404, a non-JSON body, a connect error or an unexpected shape all mean "unknown" -> None,
        and the caller falls back to its own estimate.
        """
        try:
            resp = self._client.get("/props")
            if resp.status_code < 400:
                settings = resp.json().get("default_generation_settings")
                n_ctx = _int_or_none(settings.get("n_ctx")) if isinstance(settings, dict) else None
                if n_ctx and n_ctx > 0:
                    return n_ctx
        except Exception:  # noqa: BLE001 - any failure just means "unknown"
            log.debug("client: GET /props failed", exc_info=True)
        try:
            resp = self._client.get("/api/v0/models")
            if resp.status_code >= 400:
                return None
            wanted = self.model_id
            for item in _extract_items(resp.json()):
                if not isinstance(item, dict) or not wanted or str(item.get("id", "")) != wanted:
                    continue
                for key in ("loaded_context_length", "max_context_length"):
                    n = _int_or_none(item.get(key))
                    if n and n > 0:
                        return n
        except Exception:  # noqa: BLE001 - any failure just means "unknown"
            log.debug("client: GET /api/v0/models failed", exc_info=True)
        return None

    # ---- streaming generation -------------------------------------------

    def stream_completion(
        self,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float | None = None,
        top_p: float | None = None,
        stop: list[str] | None = None,
        cancel: threading.Event | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> Iterator[str]:
        """POST `/v1/completions` (stream=True) and yield text deltas as they arrive.

        `extra_body` (per request, e.g. a repeat penalty for inline completions) is merged after
        `cfg.extra_body`, so it can override the configured sampling keys.
        """
        body: dict[str, Any] = {
            "model": self.model_id,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": self.cfg.temperature if temperature is None else temperature,
            "top_p": self.cfg.top_p if top_p is None else top_p,
            "stop": stop,
            "stream": True,
        }
        body.update(self.cfg.extra_body)
        if extra_body:
            body.update(extra_body)

        start = time.monotonic()
        ttft = 0.0
        chunk_count = 0
        usage_tokens: int | None = None
        prompt_tokens: int | None = None
        cached_tokens: int | None = None
        finish_reason = ""
        cancelled = False
        first = True
        try:
            with self._client.stream("POST", "/v1/completions", json=body) as r:
                if r.status_code >= 400:
                    raise LLMError(self._status_error_message(r))
                for chunk in self._iter_sse_chunks(r):
                    choices = chunk.get("choices") or []
                    text = choices[0].get("text", "") if choices else ""
                    reason = _finish_reason_from_chunk(chunk)
                    if reason is not None:
                        finish_reason = reason
                    tok = _completion_tokens_from_chunk(chunk)
                    if tok is not None:
                        usage_tokens = tok
                    p_tok, c_tok = _prompt_tokens_from_chunk(chunk)
                    if p_tok is not None:
                        prompt_tokens = p_tok
                    if c_tok is not None:
                        cached_tokens = c_tok
                    if text:
                        chunk_count += 1
                        if first:
                            ttft = time.monotonic() - start
                            first = False
                        yield text
                    if cancel is not None and cancel.is_set():
                        cancelled = True
                        break
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            raise LLMError(self._connect_message()) from exc
        except httpx.TimeoutException as exc:
            raise LLMError(self._timeout_message()) from exc
        except httpx.TransportError as exc:
            raise LLMError(f"LM Studio request failed: {exc}") from exc
        finally:
            self.last_stats = self._make_stats(
                start, ttft, chunk_count, usage_tokens, cancelled,
                prompt_tokens=prompt_tokens, cached_tokens=cached_tokens, finish_reason=finish_reason,
            )

    def stream_chat(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        temperature: float | None = None,
        top_p: float | None = None,
        cancel: threading.Event | None = None,
        stop: list[str] | None = None,
    ) -> Iterator[ChatDelta]:
        """POST `/v1/chat/completions` (stream=True) and yield `ChatDelta`s.

        Reasoning is taken from `delta.reasoning_content` / `delta.reasoning` when the
        server sends it separately, and additionally split out of inline
        ``<think>...</think>`` spans within `delta.content` so it never leaks into
        the visible answer text. `stop` (2026-10-07, capabilities) adds stop strings exactly
        like `stream_completion`'s: the tool loop stops the chat path on the tool call's
        closing tag; None sends none, as before.
        """
        body: dict[str, Any] = {
            "model": self.model_id,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": self.cfg.temperature if temperature is None else temperature,
            "top_p": self.cfg.top_p if top_p is None else top_p,
            "stream": True,
        }
        if stop:
            body["stop"] = list(stop)
        body.update(self.cfg.extra_body)

        start = time.monotonic()
        ttft = 0.0
        chunk_count = 0
        usage_tokens: int | None = None
        prompt_tokens: int | None = None
        cached_tokens: int | None = None
        finish_reason = ""
        cancelled = False
        first = True
        splitter = _ThinkSplitter()
        try:
            with self._client.stream("POST", "/v1/chat/completions", json=body) as r:
                if r.status_code >= 400:
                    raise LLMError(self._status_error_message(r))
                for chunk in self._iter_sse_chunks(r):
                    choices = chunk.get("choices") or []
                    reason = _finish_reason_from_chunk(chunk)
                    if reason is not None:
                        finish_reason = reason
                    delta_obj: dict[str, Any] = {}
                    if choices:
                        delta_obj = choices[0].get("delta") or {}
                    raw_content = delta_obj.get("content") or ""
                    reasoning_field = delta_obj.get("reasoning_content")
                    if reasoning_field is None:
                        reasoning_field = delta_obj.get("reasoning")
                    reasoning_field = reasoning_field or ""
                    content, split_reasoning = splitter.feed(raw_content)
                    reasoning = reasoning_field + split_reasoning
                    tok = _completion_tokens_from_chunk(chunk)
                    if tok is not None:
                        usage_tokens = tok
                    p_tok, c_tok = _prompt_tokens_from_chunk(chunk)
                    if p_tok is not None:
                        prompt_tokens = p_tok
                    if c_tok is not None:
                        cached_tokens = c_tok
                    if content or reasoning:
                        chunk_count += 1
                        if first:
                            ttft = time.monotonic() - start
                            first = False
                        yield ChatDelta(content=content, reasoning=reasoning)
                    if cancel is not None and cancel.is_set():
                        cancelled = True
                        break
                else:
                    tail_content, tail_reasoning = splitter.flush()
                    if tail_content or tail_reasoning:
                        chunk_count += 1
                        yield ChatDelta(content=tail_content, reasoning=tail_reasoning)
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            raise LLMError(self._connect_message()) from exc
        except httpx.TimeoutException as exc:
            raise LLMError(self._timeout_message()) from exc
        except httpx.TransportError as exc:
            raise LLMError(f"LM Studio request failed: {exc}") from exc
        finally:
            self.last_stats = self._make_stats(
                start, ttft, chunk_count, usage_tokens, cancelled,
                prompt_tokens=prompt_tokens, cached_tokens=cached_tokens, finish_reason=finish_reason,
            )

    # ---- internals -------------------------------------------------------

    @staticmethod
    def _parse_v0_model(item: dict[str, Any]) -> ModelInfo:
        loaded = {"loaded": True, "not-loaded": False}.get(item.get("state"))
        return ModelInfo(
            id=str(item.get("id", "")),
            loaded=loaded,
            arch=str(item.get("arch") or ""),
            quant=str(item.get("quantization") or ""),
            context_length=item.get("max_context_length"),
        )

    @staticmethod
    def _make_stats(
        start: float, ttft: float, chunk_count: int, usage_tokens: int | None, cancelled: bool,
        *, prompt_tokens: int | None = None, cached_tokens: int | None = None, finish_reason: str = "",
    ) -> StreamStats:
        elapsed = time.monotonic() - start
        tokens = usage_tokens if usage_tokens is not None else chunk_count
        # Throughput is a *generation* rate: exclude the time to first token (prompt evaluation),
        # otherwise a long prompt makes a fast model look slow.
        gen_time = elapsed - ttft if tokens > 1 else elapsed
        tok_per_s = tokens / gen_time if gen_time > 0 else 0.0
        return StreamStats(
            ttft_s=ttft, tokens=tokens, elapsed_s=elapsed, tok_per_s=tok_per_s, cancelled=cancelled,
            prompt_tokens=prompt_tokens, cached_tokens=cached_tokens, finish_reason=finish_reason,
        )

    @staticmethod
    def _iter_sse_chunks(r: httpx.Response) -> Iterator[dict[str, Any]]:
        for line in r.iter_lines():
            if isinstance(line, bytes):
                line = line.decode("utf-8", errors="ignore")
            line = line.strip("\r")
            if not line or not line.startswith("data:"):
                continue
            payload = line[len("data:"):].strip()
            if payload == "[DONE]":
                return
            if not payload:
                continue
            try:
                yield json.loads(payload)
            except json.JSONDecodeError:
                log.warning("client: unparsable SSE payload: %.200s", payload)
                continue

    def _get(self, path: str) -> httpx.Response:
        try:
            return self._client.get(path)
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            raise LLMError(self._connect_message()) from exc
        except httpx.TimeoutException as exc:
            raise LLMError(self._timeout_message()) from exc
        except httpx.TransportError as exc:
            raise LLMError(f"LM Studio request failed: {exc}") from exc

    def _connect_message(self) -> str:
        hint = (
            "Start LM Studio and enable the server (Developer tab) or run `lms server start`."
            if ":1234" in self._base_url
            else "If ExoCortex spawns llama-server itself, check the llama-server log in the session folder."
        )
        return f"LLM server not reachable at {self._base_url}. {hint}"

    def _timeout_message(self) -> str:
        return f"LLM request timed out after {self.cfg.request_timeout_s:.0f}s at {self._base_url}."

    def _status_error_message(self, r: httpx.Response) -> str:
        if not r.is_stream_consumed:
            try:
                r.read()
            except httpx.HTTPError:
                pass
        text = (r.text or "")[:300]
        return f"LLM server returned HTTP {r.status_code} for {r.request.url}: {text}"


#: Ollama sampling keys this client will forward when present in `cfg.extra_body`/`extra_body`
#: (llama.cpp-style keys `list_models`'s callers may already carry, e.g. via `LLMConfig.extra_body`
#: -- everything else Ollama does not know about is dropped rather than sent and ignored/rejected).
_OLLAMA_OPTION_PASSTHROUGH = ("top_k", "min_p", "repeat_penalty")


class OllamaClient(LMStudioClient):
    """Client for Ollama's own API (http://127.0.0.1:11434 by default) -- an alternative to
    LM Studio for machines that would rather manage models with ``ollama pull`` than through LM
    Studio's runtime packs (added 2026-09-25 alongside vendor-neutral GPU routing). Reuses its
    parent's streaming/cancellation/`<think>`-splitting/stats plumbing; only discovery, health,
    warm-up and the two request bodies differ, because Ollama's native endpoints (``/api/tags``,
    ``/api/ps``, ``/api/generate``, ``/api/chat``) are not OpenAI-shaped.
    """

    def __init__(self, cfg: LLMConfig, *, transport: httpx.BaseTransport | None = None) -> None:
        super().__init__(cfg, transport=transport)
        self.set_base_url(cfg.effective_server_url("ollama"))
        #: model name -> size in bytes, from the most recent `list_models()` (Ollama's `/api/tags`
        #: reports this; nothing else here does), used to auto-pick a model that fits the GPU.
        self.model_sizes: dict[str, int] = {}

    # ---- discovery -----------------------------------------------------

    def list_models(self) -> list[ModelInfo]:
        """`GET /api/tags` (the library), augmented with `GET /api/ps` (what is actually loaded
        right now) when that call answers -- Ollama can be reachable with nothing resident."""
        resp = self._get("/api/tags")
        if resp.status_code >= 400:
            raise LLMError(self._status_error_message(resp))
        try:
            data = resp.json()
        except ValueError as exc:
            raise LLMError(f"Ollama returned unparsable JSON from /api/tags: {exc}") from exc
        items = data.get("models", []) if isinstance(data, dict) else []
        loaded_names = self._loaded_model_names()
        self.model_sizes = {}
        models: list[ModelInfo] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or item.get("model") or "")
            if not name:
                continue
            details = item.get("details") or {}
            size = item.get("size")
            if isinstance(size, int):
                self.model_sizes[name] = size
            loaded = None if loaded_names is None else (name in loaded_names)
            models.append(ModelInfo(
                id=name, loaded=loaded, arch=str(details.get("family") or ""),
                quant=str(details.get("quantization_level") or ""), context_length=None,
            ))
        return models

    def _loaded_model_names(self) -> set[str] | None:
        """Names Ollama reports as currently loaded (`/api/ps`), or `None` when that call itself
        did not answer -- callers then leave `ModelInfo.loaded` at `None` rather than guessing."""
        try:
            resp = self._client.get("/api/ps")
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.TimeoutException, httpx.TransportError):
            return None
        if resp.status_code >= 400:
            return None
        try:
            data = resp.json()
        except ValueError:
            return None
        items = data.get("models", []) if isinstance(data, dict) else []
        names: set[str] = set()
        for item in items:
            if isinstance(item, dict):
                name = item.get("name") or item.get("model")
                if name:
                    names.add(str(name))
        return names

    def health(self) -> Health:
        """Reachability + whether `cfg.model` is in the library; "auto"/"" picks the largest
        model that fits 0.75x the primary GPU's memory budget (the same margin llama.cpp's own
        default GPU-layer heuristics leave), or the largest of all when no GPU is known."""
        try:
            models = self.list_models()
        except LLMError as exc:
            return Health(reachable=False, model_listed=False, model_loaded=None, detail=str(exc))
        if not models:
            return Health(reachable=True, model_listed=False, model_loaded=None,
                          detail="Ollama has no models pulled -- run 'ollama pull <model>'")
        if not self.cfg.model or self.cfg.model == "auto":
            chosen = self._pick_auto_model(models)
            self.resolved_model = chosen.id
            return Health(reachable=True, model_listed=True, model_loaded=chosen.loaded,
                          detail=f"auto-resolved to {chosen.id}")
        match = next((m for m in models if m.id == self.cfg.model), None)
        if match is None:
            detail = f"model '{self.cfg.model}' not found among {len(models)} listed model(s)"
            return Health(reachable=True, model_listed=False, model_loaded=None, detail=detail)
        return Health(reachable=True, model_listed=True, model_loaded=match.loaded, detail="")

    def _pick_auto_model(self, models: list[ModelInfo]) -> ModelInfo:
        budget_mb = self._gpu_budget_mb()
        if budget_mb is not None:
            limit_bytes = 0.75 * budget_mb * 1024 * 1024
            fitting = [m for m in models if self.model_sizes.get(m.id, 0) <= limit_bytes]
            pool = fitting or models
        else:
            pool = models
        return max(pool, key=lambda m: self.model_sizes.get(m.id, 0))

    @staticmethod
    def _gpu_budget_mb() -> float | None:
        try:
            from exocortex import gpu

            primary = gpu.primary_gpu()
            return float(primary.budget_mb) if primary is not None else None
        except Exception:  # noqa: BLE001 - no GPU reading here just means "pick the largest"
            log.debug("client: primary_gpu() failed", exc_info=True)
            return None

    def ensure_loaded(self) -> bool:
        """Ollama's documented way to force a model into memory ahead of the first real request:
        an empty-prompt `/api/generate` with a `keep_alive` (30 min -- long enough to outlast the
        gaps between meeting answers without holding it forever on a machine that also runs other
        things on the GPU)."""
        h = self.health()
        if h.model_loaded is True:
            return True
        if not self.cfg.auto_load_model:
            return False
        body = {"model": self.model_id, "prompt": "", "keep_alive": "30m"}
        try:
            resp = self._client.post(
                "/api/generate", json=body, timeout=httpx.Timeout(_ENSURE_LOADED_TIMEOUT_S, connect=_CONNECT_TIMEOUT_S)
            )
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            raise LLMError(self._connect_message()) from exc
        except httpx.TimeoutException as exc:
            raise LLMError(
                f"Timed out waiting for Ollama to load model '{self.model_id}' (>{_ENSURE_LOADED_TIMEOUT_S:.0f}s)."
            ) from exc
        except httpx.TransportError as exc:
            raise LLMError(f"Ollama request failed while loading model '{self.model_id}': {exc}") from exc
        if resp.status_code >= 400:
            raise LLMError(self._status_error_message(resp))
        return True

    def unload(self) -> bool:
        """2026-09-27 (portable-install, the Models window's Unload): Ollama's documented way to
        evict a model right away -- an empty-prompt `/api/generate` with ``keep_alive: 0``. Ollama
        keeps its own copy on disk, so this only frees memory; `ensure_loaded` brings it back."""
        body = {"model": self.model_id, "prompt": "", "keep_alive": 0}
        try:
            resp = self._client.post("/api/generate", json=body,
                                     timeout=httpx.Timeout(_ENSURE_LOADED_TIMEOUT_S, connect=_CONNECT_TIMEOUT_S))
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            raise LLMError(self._connect_message()) from exc
        except httpx.TransportError as exc:
            raise LLMError(f"Ollama request failed while unloading model '{self.model_id}': {exc}") from exc
        if resp.status_code >= 400:
            raise LLMError(self._status_error_message(resp))
        return True

    def context_window(self) -> int | None:
        """2026-10-06 (context library): every Ollama request is sent with `num_ctx` =
        `cfg.ctx_size` (`_ollama_options`), so that is the window -- no server query needed."""
        n = _int_or_none(self.cfg.ctx_size)
        return n if n and n > 0 else None

    # ---- streaming generation -------------------------------------------

    def _ollama_options(
        self, *, max_tokens: int, temperature: float | None, top_p: float | None,
        stop: list[str] | None, extra_body: dict[str, Any] | None,
    ) -> dict[str, Any]:
        options: dict[str, Any] = {
            "num_predict": max_tokens,
            "temperature": self.cfg.temperature if temperature is None else temperature,
            "top_p": self.cfg.top_p if top_p is None else top_p,
            "stop": stop or [],
            "num_ctx": self.cfg.ctx_size,
        }
        merged: dict[str, Any] = dict(self.cfg.extra_body)
        if extra_body:
            merged.update(extra_body)
        for key in _OLLAMA_OPTION_PASSTHROUGH:
            if key in merged:
                options[key] = merged[key]
        return options

    def stream_completion(
        self,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float | None = None,
        top_p: float | None = None,
        stop: list[str] | None = None,
        cancel: threading.Event | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> Iterator[str]:
        """POST `/api/generate` (`raw: true` -- the prompt is already fully formatted, same as the
        parent's `/v1/completions` use) and yield text deltas from Ollama's NDJSON stream."""
        body: dict[str, Any] = {
            "model": self.model_id,
            "prompt": prompt,
            "raw": True,
            "stream": True,
            "keep_alive": "30m",
            "options": self._ollama_options(max_tokens=max_tokens, temperature=temperature, top_p=top_p,
                                            stop=stop, extra_body=extra_body),
        }
        start = time.monotonic()
        ttft = 0.0
        chunk_count = 0
        eval_count: int | None = None
        prompt_eval_count: int | None = None
        finish_reason = ""
        cancelled = False
        first = True
        try:
            with self._client.stream("POST", "/api/generate", json=body) as r:
                if r.status_code >= 400:
                    raise LLMError(self._status_error_message(r))
                for chunk in self._iter_ndjson_lines(r):
                    if chunk.get("error"):
                        raise LLMError(f"Ollama error: {chunk['error']}")
                    if isinstance(chunk.get("eval_count"), int):
                        eval_count = chunk["eval_count"]
                    if isinstance(chunk.get("prompt_eval_count"), int):
                        prompt_eval_count = chunk["prompt_eval_count"]
                    reason = _finish_reason_from_chunk(chunk)
                    if reason is not None:
                        finish_reason = reason
                    text = chunk.get("response", "")
                    if text:
                        chunk_count += 1
                        if first:
                            ttft = time.monotonic() - start
                            first = False
                        yield text
                    if cancel is not None and cancel.is_set():
                        cancelled = True
                        break
                    if chunk.get("done"):
                        break
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            raise LLMError(self._connect_message()) from exc
        except httpx.TimeoutException as exc:
            raise LLMError(self._timeout_message()) from exc
        except httpx.TransportError as exc:
            raise LLMError(f"Ollama request failed: {exc}") from exc
        finally:
            self.last_stats = self._make_stats(
                start, ttft, chunk_count, eval_count, cancelled, prompt_tokens=prompt_eval_count,
                finish_reason=finish_reason,
            )

    def stream_chat(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        temperature: float | None = None,
        top_p: float | None = None,
        cancel: threading.Event | None = None,
        stop: list[str] | None = None,
    ) -> Iterator[ChatDelta]:
        """POST `/api/chat` and yield `ChatDelta`s; Ollama's own `message.thinking` field is used
        when present, alongside the same inline ``<think>`` splitting the parent does (a model
        served through Ollama may still emit inline tags instead of the separate field). `stop`
        (2026-10-07, capabilities) goes into `options.stop` like `stream_completion`'s."""
        body: dict[str, Any] = {
            "model": self.model_id,
            "messages": messages,
            "stream": True,
            "keep_alive": "30m",
            "options": self._ollama_options(max_tokens=max_tokens, temperature=temperature, top_p=top_p,
                                            stop=stop, extra_body=None),
        }
        start = time.monotonic()
        ttft = 0.0
        chunk_count = 0
        eval_count: int | None = None
        prompt_eval_count: int | None = None
        finish_reason = ""
        cancelled = False
        first = True
        splitter = _ThinkSplitter()
        try:
            with self._client.stream("POST", "/api/chat", json=body) as r:
                if r.status_code >= 400:
                    raise LLMError(self._status_error_message(r))
                for chunk in self._iter_ndjson_lines(r):
                    if chunk.get("error"):
                        raise LLMError(f"Ollama error: {chunk['error']}")
                    if isinstance(chunk.get("eval_count"), int):
                        eval_count = chunk["eval_count"]
                    if isinstance(chunk.get("prompt_eval_count"), int):
                        prompt_eval_count = chunk["prompt_eval_count"]
                    reason = _finish_reason_from_chunk(chunk)
                    if reason is not None:
                        finish_reason = reason
                    message = chunk.get("message") or {}
                    raw_content = message.get("content") or ""
                    reasoning_field = message.get("thinking") or ""
                    content, split_reasoning = splitter.feed(raw_content)
                    reasoning = reasoning_field + split_reasoning
                    if content or reasoning:
                        chunk_count += 1
                        if first:
                            ttft = time.monotonic() - start
                            first = False
                        yield ChatDelta(content=content, reasoning=reasoning)
                    if cancel is not None and cancel.is_set():
                        cancelled = True
                        break
                    if chunk.get("done"):
                        tail_content, tail_reasoning = splitter.flush()
                        if tail_content or tail_reasoning:
                            chunk_count += 1
                            yield ChatDelta(content=tail_content, reasoning=tail_reasoning)
                        break
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            raise LLMError(self._connect_message()) from exc
        except httpx.TimeoutException as exc:
            raise LLMError(self._timeout_message()) from exc
        except httpx.TransportError as exc:
            raise LLMError(f"Ollama request failed: {exc}") from exc
        finally:
            self.last_stats = self._make_stats(
                start, ttft, chunk_count, eval_count, cancelled, prompt_tokens=prompt_eval_count,
                finish_reason=finish_reason,
            )

    # ---- internals -------------------------------------------------------

    @staticmethod
    def _iter_ndjson_lines(r: httpx.Response) -> Iterator[dict[str, Any]]:
        """Ollama streams one JSON object per line (no `data:`/`[DONE]` SSE framing)."""
        for line in r.iter_lines():
            if isinstance(line, bytes):
                line = line.decode("utf-8", errors="ignore")
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                log.warning("client: unparsable Ollama NDJSON line: %.200s", line)
                continue

    def _connect_message(self) -> str:
        return (
            f"Ollama not reachable at {self._base_url}. Start Ollama (the tray app, or "
            "'ollama serve') and pull a model ('ollama pull qwen3:8b')."
        )


def make_client(cfg: LLMConfig, backend: str, base_url: str | None = None) -> LMStudioClient:
    """The client for `backend`: `OllamaClient` for "ollama", `LMStudioClient` for everything else
    (openai, llamacpp -- `set_base_url` points it at the spawned server -- and "none", so the rest
    of the app always has a real client object to construct around).

    2026-09-27 (portable-install): `base_url`, when given and non-empty, is applied via
    `set_base_url` right after construction -- this is how a `Route` that found Ollama (or an
    OpenAI-compatible server) at a non-standard address is honoured without writing that address
    into `cfg` (`cfg.base_url`/`cfg.ollama_url` must never be mutated by `set_base_url`; only the
    client's own, separate `_base_url` changes).
    """
    client = OllamaClient(cfg) if (backend or "").strip().lower() == "ollama" else LMStudioClient(cfg)
    if base_url:
        client.set_base_url(base_url)
    return client

"""Unit tests for the 2026-10-06 context-library additions to exocortex.llm.client: prompt-token
extraction into StreamStats and `context_window()`. Offline via httpx.MockTransport, same pattern
as tests/test_client.py and tests/test_client_ollama.py."""
from __future__ import annotations

import json
from typing import Any, Callable

import httpx
import pytest

from exocortex.config import LLMConfig
from exocortex.llm.client import (
    LMStudioClient,
    OllamaClient,
    StreamStats,
    _completion_tokens_from_chunk,
    _prompt_tokens_from_chunk,
)

Handler = Callable[[httpx.Request], httpx.Response]


def make_client(handler: Handler, **cfg_kwargs: Any) -> LMStudioClient:
    cfg = LLMConfig(server_url="http://unit.test", model=cfg_kwargs.pop("model", "maple-preview"), **cfg_kwargs)
    return LMStudioClient(cfg, transport=httpx.MockTransport(handler))


def make_ollama(handler: Handler, **cfg_kwargs: Any) -> OllamaClient:
    cfg = LLMConfig(server_url="http://unit.test", model=cfg_kwargs.pop("model", "qwen3:8b"), **cfg_kwargs)
    return OllamaClient(cfg, transport=httpx.MockTransport(handler))


def sse_body(events: list[dict[str, Any] | None]) -> bytes:
    lines = []
    for ev in events:
        payload = "[DONE]" if ev is None else json.dumps(ev)
        lines.append(f"data: {payload}\n\n")
    return "".join(lines).encode("utf-8")


def ndjson_body(lines: list[dict[str, Any]]) -> bytes:
    return b"".join((json.dumps(line) + "\n").encode("utf-8") for line in lines)


def route(routes: dict[str, Handler]) -> Handler:
    def handler(request: httpx.Request) -> httpx.Response:
        found = routes.get(request.url.path)
        return found(request) if found is not None else httpx.Response(404, content=b"not found")

    return handler


def json_ok(data: Any) -> Handler:
    return lambda request: httpx.Response(200, content=json.dumps(data).encode("utf-8"))


# ---------------------------------------------------------------------------
# StreamStats
# ---------------------------------------------------------------------------


def test_stream_stats_new_fields_default_to_none_and_positional_construction_still_works() -> None:
    stats = StreamStats(0.1, 5, 1.0, 5.0, False)
    assert stats.prompt_tokens is None
    assert stats.cached_tokens is None


# ---------------------------------------------------------------------------
# _prompt_tokens_from_chunk
# ---------------------------------------------------------------------------


def test_chunk_without_any_prompt_info_gives_none_none() -> None:
    assert _prompt_tokens_from_chunk({"choices": [{"text": "x"}]}) == (None, None)


def test_chunk_openai_usage_prompt_tokens() -> None:
    assert _prompt_tokens_from_chunk({"usage": {"prompt_tokens": 321, "completion_tokens": 9}}) == (321, None)


def test_chunk_openai_usage_cached_tokens_from_details() -> None:
    chunk = {"usage": {"prompt_tokens": 500, "prompt_tokens_details": {"cached_tokens": 450}}}
    assert _prompt_tokens_from_chunk(chunk) == (500, 450)


def test_chunk_llama_server_timings_sum_evaluated_and_cached() -> None:
    # prompt_n is only what was evaluated; the whole prompt is prompt_n + cache_n
    chunk = {"timings": {"prompt_n": 120, "cache_n": 880, "predicted_n": 40}}
    assert _prompt_tokens_from_chunk(chunk) == (1000, 880)


def test_chunk_llama_server_timings_without_cache_n() -> None:
    assert _prompt_tokens_from_chunk({"timings": {"prompt_n": 333}}) == (333, None)


def test_chunk_usage_wins_over_timings_for_the_prompt_total_but_timings_fill_the_cache() -> None:
    chunk = {"usage": {"prompt_tokens": 900}, "timings": {"prompt_n": 100, "cache_n": 800}}
    assert _prompt_tokens_from_chunk(chunk) == (900, 800)


def test_chunk_lm_studio_stats_keys() -> None:
    assert _prompt_tokens_from_chunk({"stats": {"prompt_tokens_count": 77}}) == (77, None)
    assert _prompt_tokens_from_chunk({"stats": {"input_tokens": 12, "cached_tokens": 4}}) == (12, 4)


def test_chunk_ignores_non_int_and_bool_values() -> None:
    chunk = {"usage": {"prompt_tokens": "12"}, "timings": {"prompt_n": True, "cache_n": 1.5}}
    assert _prompt_tokens_from_chunk(chunk) == (None, None)


def test_completion_tokens_extraction_is_unchanged() -> None:
    assert _completion_tokens_from_chunk({"timings": {"predicted_n": 7}}) == 7
    assert _completion_tokens_from_chunk({"usage": {"completion_tokens": 3}}) == 3


# ---------------------------------------------------------------------------
# LMStudioClient streams
# ---------------------------------------------------------------------------


def test_stream_completion_parses_llama_server_timings_into_stats() -> None:
    events = [
        {"choices": [{"text": "Hel"}]},
        {"choices": [{"text": "lo"}]},
        {"choices": [{"text": ""}], "timings": {"prompt_n": 60, "cache_n": 1940, "predicted_n": 2}},
        None,
    ]
    client = make_client(route({"/v1/completions": lambda r: httpx.Response(200, content=sse_body(events))}))

    assert "".join(client.stream_completion("p", max_tokens=8)) == "Hello"

    stats = client.last_stats
    assert stats is not None
    assert stats.tokens == 2
    assert stats.prompt_tokens == 2000
    assert stats.cached_tokens == 1940


def test_stream_completion_parses_openai_usage_into_stats() -> None:
    events = [
        {"choices": [{"text": "ok"}]},
        {"choices": [], "usage": {"prompt_tokens": 410, "completion_tokens": 1,
                                  "prompt_tokens_details": {"cached_tokens": 256}}},
        None,
    ]
    client = make_client(route({"/v1/completions": lambda r: httpx.Response(200, content=sse_body(events))}))

    list(client.stream_completion("p", max_tokens=8))

    assert client.last_stats is not None
    assert client.last_stats.prompt_tokens == 410
    assert client.last_stats.cached_tokens == 256


def test_stream_completion_without_prompt_info_leaves_none() -> None:
    events = [{"choices": [{"text": "ok"}]}, None]
    client = make_client(route({"/v1/completions": lambda r: httpx.Response(200, content=sse_body(events))}))

    list(client.stream_completion("p", max_tokens=8))

    assert client.last_stats is not None
    assert client.last_stats.prompt_tokens is None
    assert client.last_stats.cached_tokens is None


def test_a_later_chunk_without_prompt_info_does_not_erase_an_earlier_one() -> None:
    events = [
        {"choices": [{"text": "a"}], "usage": {"prompt_tokens": 99}},
        {"choices": [{"text": "b"}]},
        None,
    ]
    client = make_client(route({"/v1/completions": lambda r: httpx.Response(200, content=sse_body(events))}))

    list(client.stream_completion("p", max_tokens=8))

    assert client.last_stats is not None
    assert client.last_stats.prompt_tokens == 99


def test_stream_chat_parses_prompt_tokens_into_stats() -> None:
    events = [
        {"choices": [{"delta": {"content": "hi"}}]},
        {"choices": [], "timings": {"prompt_n": 10, "cache_n": 90, "predicted_n": 1}},
        None,
    ]
    client = make_client(route({"/v1/chat/completions": lambda r: httpx.Response(200, content=sse_body(events))}))

    deltas = list(client.stream_chat([{"role": "user", "content": "x"}], max_tokens=8))

    assert "".join(d.content for d in deltas) == "hi"
    assert client.last_stats is not None
    assert client.last_stats.prompt_tokens == 100
    assert client.last_stats.cached_tokens == 90


def test_stats_are_recorded_even_when_the_stream_errors_out() -> None:
    client = make_client(route({"/v1/completions": lambda r: httpx.Response(500, content=b"boom")}))
    with pytest.raises(Exception):
        list(client.stream_completion("p", max_tokens=8))
    assert client.last_stats is not None
    assert client.last_stats.prompt_tokens is None


# ---------------------------------------------------------------------------
# OllamaClient streams
# ---------------------------------------------------------------------------


def test_ollama_stream_completion_reports_prompt_eval_count() -> None:
    lines = [
        {"response": "Hi", "done": False},
        {"response": "", "done": True, "eval_count": 2, "prompt_eval_count": 345},
    ]
    client = make_ollama(route({"/api/generate": lambda r: httpx.Response(200, content=ndjson_body(lines))}))

    assert "".join(client.stream_completion("p", max_tokens=8)) == "Hi"

    stats = client.last_stats
    assert stats is not None
    assert stats.tokens == 2
    assert stats.prompt_tokens == 345
    assert stats.cached_tokens is None


def test_ollama_stream_chat_reports_prompt_eval_count() -> None:
    lines = [
        {"message": {"content": "Yo"}, "done": False},
        {"message": {"content": ""}, "done": True, "eval_count": 1, "prompt_eval_count": 58},
    ]
    client = make_ollama(route({"/api/chat": lambda r: httpx.Response(200, content=ndjson_body(lines))}))

    list(client.stream_chat([{"role": "user", "content": "x"}], max_tokens=8))

    assert client.last_stats is not None
    assert client.last_stats.prompt_tokens == 58
    assert client.last_stats.cached_tokens is None


def test_ollama_without_prompt_eval_count_leaves_none() -> None:
    lines = [{"response": "Hi", "done": True, "eval_count": 1}]
    client = make_ollama(route({"/api/generate": lambda r: httpx.Response(200, content=ndjson_body(lines))}))

    list(client.stream_completion("p", max_tokens=8))

    assert client.last_stats is not None
    assert client.last_stats.prompt_tokens is None


# ---------------------------------------------------------------------------
# context_window
# ---------------------------------------------------------------------------


def test_context_window_reads_llama_server_props() -> None:
    props = {"default_generation_settings": {"n_ctx": 2048}, "total_slots": 2}
    client = make_client(route({"/props": json_ok(props)}))
    assert client.context_window() == 2048


def test_context_window_props_404_without_lm_studio_listing_is_none() -> None:
    client = make_client(route({}))
    assert client.context_window() is None


def test_context_window_props_non_json_is_none() -> None:
    client = make_client(route({"/props": lambda r: httpx.Response(200, content=b"<html>hello</html>")}))
    assert client.context_window() is None


def test_context_window_props_without_n_ctx_is_none() -> None:
    client = make_client(route({"/props": json_ok({"default_generation_settings": {}})}))
    assert client.context_window() is None
    client = make_client(route({"/props": json_ok(["not", "a", "dict"])}))
    assert client.context_window() is None


def test_context_window_connect_error_is_none() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    assert make_client(handler).context_window() is None


def test_context_window_unexpected_exception_is_none() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise RuntimeError("transport exploded")

    assert make_client(handler).context_window() is None


def test_context_window_falls_back_to_lm_studio_loaded_length() -> None:
    models = {"data": [
        {"id": "other", "state": "loaded", "loaded_context_length": 999, "max_context_length": 9999},
        {"id": "maple-preview", "state": "loaded", "loaded_context_length": 8192, "max_context_length": 32768},
    ]}
    client = make_client(route({"/api/v0/models": json_ok(models)}))
    assert client.context_window() == 8192


def test_context_window_lm_studio_uses_max_length_when_not_loaded() -> None:
    models = {"data": [{"id": "maple-preview", "state": "not-loaded", "max_context_length": 32768}]}
    client = make_client(route({"/api/v0/models": json_ok(models)}))
    assert client.context_window() == 32768


def test_context_window_lm_studio_model_not_listed_is_none() -> None:
    models = {"data": [{"id": "other", "loaded_context_length": 4096}]}
    client = make_client(route({"/api/v0/models": json_ok(models)}))
    assert client.context_window() is None


def test_context_window_props_wins_over_lm_studio_listing() -> None:
    calls: list[str] = []

    def spy(path: str, data: Any) -> Handler:
        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(path)
            return httpx.Response(200, content=json.dumps(data).encode("utf-8"))

        return handler

    client = make_client(route({
        "/props": spy("/props", {"default_generation_settings": {"n_ctx": 4096}}),
        "/api/v0/models": spy("/api/v0/models", {"data": []}),
    }))
    assert client.context_window() == 4096
    assert calls == ["/props"]


def test_ollama_context_window_is_the_configured_ctx_size() -> None:
    client = make_ollama(route({}), ctx_size=6144)
    assert client.context_window() == 6144

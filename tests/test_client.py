"""Unit tests for exocortex.llm.client, all offline via httpx.MockTransport.

The one @pytest.mark.live test talks to a real LM Studio server at
http://localhost:1234 with model "maple-preview" and is skipped by default.
"""
from __future__ import annotations

import json
import threading
import time
from typing import Any, Callable

import httpx
import pytest

from exocortex.config import LLMConfig
from exocortex.llm.client import ChatDelta, Health, LLMError, LMStudioClient, ModelInfo, StreamStats

Handler = Callable[[httpx.Request], httpx.Response]


def make_client(handler: Handler, **cfg_kwargs: Any) -> LMStudioClient:
    cfg = LLMConfig(server_url="http://unit.test", model="maple-preview", **cfg_kwargs)
    return LMStudioClient(cfg, transport=httpx.MockTransport(handler))


def sse_body(events: list[dict[str, Any] | None]) -> bytes:
    """Build an SSE byte body from a list of chunk dicts; `None` renders as [DONE]."""
    lines = []
    for ev in events:
        payload = "[DONE]" if ev is None else json.dumps(ev)
        lines.append(f"data: {payload}\n\n")
    return "".join(lines).encode("utf-8")


def completion_chunk(text: str) -> dict[str, Any]:
    return {"choices": [{"text": text}]}


def chat_chunk(content: str = "", reasoning_content: str | None = None) -> dict[str, Any]:
    delta: dict[str, Any] = {"content": content}
    if reasoning_content is not None:
        delta["reasoning_content"] = reasoning_content
    return {"choices": [{"delta": delta}]}


def router(routes: dict[str, Handler]) -> Handler:
    def handler(request: httpx.Request) -> httpx.Response:
        route = routes.get(request.url.path)
        if route is None:
            return httpx.Response(404, content=b"not found")
        return route(request)

    return handler


def json_response(status: int, data: Any) -> Handler:
    return lambda request: httpx.Response(status, content=json.dumps(data).encode("utf-8"))


def sse_response(events: list[dict[str, Any] | None]) -> Handler:
    return lambda request: httpx.Response(200, content=sse_body(events))


# ---------------------------------------------------------------------------
# list_models / health / ensure_loaded
# ---------------------------------------------------------------------------


def test_list_models_v0_parses_state_and_metadata() -> None:
    v0_payload = {
        "data": [
            {
                "id": "maple-preview",
                "state": "loaded",
                "arch": "qwen3moe",
                "quantization": "Q4_K_M",
                "max_context_length": 131072,
                "type": "llm",
            },
            {"id": "other-model", "state": "not-loaded", "arch": "llama", "quantization": "Q8_0"},
        ]
    }
    client = make_client(router({"/api/v0/models": json_response(200, v0_payload)}))

    models = client.list_models()

    assert models == [
        ModelInfo(id="maple-preview", loaded=True, arch="qwen3moe", quant="Q4_K_M", context_length=131072),
        ModelInfo(id="other-model", loaded=False, arch="llama", quant="Q8_0", context_length=None),
    ]


def test_list_models_falls_back_to_v1_when_v0_unavailable() -> None:
    handler = router(
        {
            "/api/v0/models": lambda r: httpx.Response(404, content=b"not found"),
            "/v1/models": json_response(200, {"data": [{"id": "maple-preview", "object": "model"}]}),
        }
    )
    client = make_client(handler)

    models = client.list_models()

    assert models == [ModelInfo(id="maple-preview", loaded=None)]


def test_health_reports_loaded_model() -> None:
    payload = {"data": [{"id": "maple-preview", "state": "loaded"}]}
    client = make_client(router({"/api/v0/models": json_response(200, payload)}))

    h = client.health()

    assert h == Health(reachable=True, model_listed=True, model_loaded=True, detail="")


def test_health_reports_model_not_listed() -> None:
    payload = {"data": [{"id": "some-other-model", "state": "loaded"}]}
    client = make_client(router({"/api/v0/models": json_response(200, payload)}))

    h = client.health()

    assert h.reachable is True
    assert h.model_listed is False
    assert h.model_loaded is None
    assert "maple-preview" in h.detail


def test_ensure_loaded_noop_when_already_loaded() -> None:
    calls: list[str] = []

    def completions(request: httpx.Request) -> httpx.Response:
        calls.append("completions")
        return httpx.Response(200, content=sse_body([None]))

    payload = {"data": [{"id": "maple-preview", "state": "loaded"}]}
    handler = router({"/api/v0/models": json_response(200, payload), "/v1/completions": completions})
    client = make_client(handler)

    assert client.ensure_loaded() is True
    assert calls == []


def test_ensure_loaded_sends_warmup_request_when_not_loaded() -> None:
    calls: list[dict[str, Any]] = []

    def completions(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content))
        return httpx.Response(200, content=b'{"choices": [{"text": "hi"}]}')

    payload = {"data": [{"id": "maple-preview", "state": "not-loaded"}]}
    handler = router({"/api/v0/models": json_response(200, payload), "/v1/completions": completions})
    client = make_client(handler)

    assert client.ensure_loaded() is True
    assert len(calls) == 1
    assert calls[0]["model"] == "maple-preview"
    assert calls[0]["prompt"] == "hi"
    assert calls[0]["max_tokens"] == 1


def test_ensure_loaded_respects_auto_load_disabled() -> None:
    calls: list[str] = []

    def completions(request: httpx.Request) -> httpx.Response:
        calls.append("hit")
        return httpx.Response(200, content=b"{}")

    payload = {"data": [{"id": "maple-preview", "state": "not-loaded"}]}
    handler = router({"/api/v0/models": json_response(200, payload), "/v1/completions": completions})
    client = make_client(handler, auto_load_model=False)

    assert client.ensure_loaded() is False
    assert calls == []


# ---------------------------------------------------------------------------
# error mapping
# ---------------------------------------------------------------------------


def test_connect_error_maps_to_llm_error_with_base_url() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    client = make_client(handler)

    with pytest.raises(LLMError) as exc_info:
        client.list_models()
    assert client.cfg.base_url in str(exc_info.value)

    h = client.health()
    assert h == Health(reachable=False, model_listed=False, model_loaded=None, detail=str(exc_info.value))


def test_connect_timeout_maps_to_llm_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("timed out", request=request)

    client = make_client(handler)

    with pytest.raises(LLMError) as exc_info:
        client.list_models()
    assert client.cfg.base_url in str(exc_info.value)


def test_http_error_status_includes_response_text() -> None:
    long_body = ("boom " * 100).encode("utf-8")
    handler = router(
        {
            "/api/v0/models": lambda r: httpx.Response(500, content=long_body),
            "/v1/models": lambda r: httpx.Response(500, content=long_body),
        }
    )
    client = make_client(handler)

    with pytest.raises(LLMError) as exc_info:
        client.list_models()
    msg = str(exc_info.value)
    assert "500" in msg
    assert "boom" in msg
    # only the first 300 chars of the body are included
    assert len(msg) < len(long_body.decode()) + 100


def test_stream_completion_http_error_raises_llm_error() -> None:
    handler = router({"/v1/completions": lambda r: httpx.Response(503, content=b"overloaded")})
    client = make_client(handler)

    with pytest.raises(LLMError) as exc_info:
        next(client.stream_completion("hi", max_tokens=10))
    assert "503" in str(exc_info.value)
    assert "overloaded" in str(exc_info.value)


# ---------------------------------------------------------------------------
# stream_completion
# ---------------------------------------------------------------------------


def test_stream_completion_yields_text_and_sets_stats() -> None:
    events = [completion_chunk("Hel"), completion_chunk("lo"), completion_chunk(" world"), None]
    handler = router({"/v1/completions": sse_response(events)})
    client = make_client(handler)

    out = list(client.stream_completion("hi", max_tokens=10))

    assert out == ["Hel", "lo", " world"]
    stats = client.last_stats
    assert isinstance(stats, StreamStats)
    assert stats.cancelled is False
    assert stats.tokens == 3
    assert stats.ttft_s >= 0.0
    assert stats.elapsed_s >= 0.0


def test_stream_completion_uses_usage_completion_tokens_when_present() -> None:
    events = [
        completion_chunk("a"),
        completion_chunk("b"),
        {"choices": [], "usage": {"completion_tokens": 42}},
        None,
    ]
    handler = router({"/v1/completions": sse_response(events)})
    client = make_client(handler)

    list(client.stream_completion("hi", max_tokens=10))

    assert client.last_stats is not None
    assert client.last_stats.tokens == 42


def test_stream_completion_uses_stats_object_when_no_usage() -> None:
    events = [completion_chunk("a"), {"choices": [], "stats": {"tokens_predicted": 7}}, None]
    handler = router({"/v1/completions": sse_response(events)})
    client = make_client(handler)

    list(client.stream_completion("hi", max_tokens=10))

    assert client.last_stats is not None
    assert client.last_stats.tokens == 7


def test_stream_completion_cancel_mid_stream_stops_generator() -> None:
    events = [completion_chunk("a"), completion_chunk("b"), completion_chunk("c"), None]
    handler = router({"/v1/completions": sse_response(events)})
    client = make_client(handler)
    cancel = threading.Event()

    gen = client.stream_completion("hi", max_tokens=10, cancel=cancel)
    first = next(gen)
    assert first == "a"
    cancel.set()
    with pytest.raises(StopIteration):
        next(gen)

    assert client.last_stats is not None
    assert client.last_stats.cancelled is True
    assert client.last_stats.tokens == 1


def test_stream_completion_body_merges_extra_body_and_fields() -> None:
    captured: dict[str, Any] = {}

    def completions(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(200, content=sse_body([completion_chunk("ok"), None]))

    handler = router({"/v1/completions": completions})
    client = make_client(handler, extra_body={"repeat_penalty": 1.05})

    list(client.stream_completion("hello", max_tokens=5, stop=["\n"]))

    assert captured["model"] == "maple-preview"
    assert captured["prompt"] == "hello"
    assert captured["max_tokens"] == 5
    assert captured["stop"] == ["\n"]
    assert captured["stream"] is True
    assert captured["repeat_penalty"] == 1.05
    # defaults from cfg used when temperature/top_p not passed
    assert captured["temperature"] == client.cfg.temperature
    assert captured["top_p"] == client.cfg.top_p


# ---------------------------------------------------------------------------
# stream_chat
# ---------------------------------------------------------------------------


def test_stream_chat_separates_explicit_reasoning_field() -> None:
    events = [
        chat_chunk(reasoning_content="thinking..."),
        chat_chunk(content="Hello"),
        chat_chunk(content=" there"),
        None,
    ]
    handler = router({"/v1/chat/completions": sse_response(events)})
    client = make_client(handler)

    deltas = list(client.stream_chat([{"role": "user", "content": "hi"}], max_tokens=10))

    content = "".join(d.content for d in deltas)
    reasoning = "".join(d.reasoning for d in deltas)
    assert content == "Hello there"
    assert reasoning == "thinking..."


def test_stream_chat_splits_inline_think_tags() -> None:
    events = [
        chat_chunk(content="<think>reasoning here</think>The answer is 42."),
        None,
    ]
    handler = router({"/v1/chat/completions": sse_response(events)})
    client = make_client(handler)

    deltas = list(client.stream_chat([{"role": "user", "content": "hi"}], max_tokens=10))

    content = "".join(d.content for d in deltas)
    reasoning = "".join(d.reasoning for d in deltas)
    assert "<think>" not in content
    assert "</think>" not in content
    assert content == "The answer is 42."
    assert reasoning == "reasoning here"


def test_stream_chat_splits_inline_think_tag_across_chunk_boundary() -> None:
    events = [
        chat_chunk(content="<thi"),
        chat_chunk(content="nk>secret</thi"),
        chat_chunk(content="nk>visible answer"),
        None,
    ]
    handler = router({"/v1/chat/completions": sse_response(events)})
    client = make_client(handler)

    deltas = list(client.stream_chat([{"role": "user", "content": "hi"}], max_tokens=10))

    content = "".join(d.content for d in deltas)
    reasoning = "".join(d.reasoning for d in deltas)
    assert content == "visible answer"
    assert reasoning == "secret"


def test_stream_chat_cancel_mid_stream_stops_generator() -> None:
    events = [chat_chunk(content="a"), chat_chunk(content="b"), chat_chunk(content="c"), None]
    handler = router({"/v1/chat/completions": sse_response(events)})
    client = make_client(handler)
    cancel = threading.Event()

    gen = client.stream_chat([{"role": "user", "content": "hi"}], max_tokens=10, cancel=cancel)
    first = next(gen)
    assert first == ChatDelta(content="a", reasoning="")
    cancel.set()
    with pytest.raises(StopIteration):
        next(gen)

    assert client.last_stats is not None
    assert client.last_stats.cancelled is True


def test_stream_chat_body_has_no_stop_field_and_merges_extra_body() -> None:
    captured: dict[str, Any] = {}

    def chat(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(200, content=sse_body([chat_chunk(content="ok"), None]))

    handler = router({"/v1/chat/completions": chat})
    client = make_client(handler, extra_body={"seed": 7})

    messages = [{"role": "user", "content": "hi"}]
    list(client.stream_chat(messages, max_tokens=5))

    assert captured["model"] == "maple-preview"
    assert captured["messages"] == messages
    assert captured["max_tokens"] == 5
    assert captured["stream"] is True
    assert captured["seed"] == 7
    assert "stop" not in captured


# ---------------------------------------------------------------------------
# additional coverage: transport-error mapping, thread isolation, edge cases
# ---------------------------------------------------------------------------


def test_list_models_maps_generic_transport_error_to_llm_error() -> None:
    # e.g. the server closes the connection mid-response: not a ConnectError/Timeout,
    # but still must surface as LLMError per the module's "nothing else raises" contract.
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.RemoteProtocolError("peer closed connection", request=request)

    client = make_client(handler)

    with pytest.raises(LLMError) as exc_info:
        client.list_models()
    assert "peer closed connection" in str(exc_info.value)


def test_stream_completion_maps_read_error_to_llm_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadError("connection reset", request=request)

    client = make_client(handler)

    with pytest.raises(LLMError):
        next(client.stream_completion("hi", max_tokens=10))


def test_stream_chat_maps_read_error_to_llm_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadError("connection reset", request=request)

    client = make_client(handler)

    with pytest.raises(LLMError):
        next(client.stream_chat([{"role": "user", "content": "hi"}], max_tokens=10))


def test_ensure_loaded_maps_generic_transport_error_to_llm_error() -> None:
    def completions(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadError("connection reset", request=request)

    payload = {"data": [{"id": "maple-preview", "state": "not-loaded"}]}
    handler = router({"/api/v0/models": json_response(200, payload), "/v1/completions": completions})
    client = make_client(handler)

    with pytest.raises(LLMError):
        client.ensure_loaded()


def test_stream_chat_combines_explicit_reasoning_field_and_inline_think_tags() -> None:
    """Both reasoning channels can be present in the same chunk; neither should be lost."""
    events = [
        chat_chunk(content="<think>inline</think>visible", reasoning_content="explicit "),
        None,
    ]
    handler = router({"/v1/chat/completions": sse_response(events)})
    client = make_client(handler)

    deltas = list(client.stream_chat([{"role": "user", "content": "hi"}], max_tokens=10))

    content = "".join(d.content for d in deltas)
    reasoning = "".join(d.reasoning for d in deltas)
    assert content == "visible"
    assert reasoning == "explicit inline"


def test_last_stats_none_before_any_call() -> None:
    client = make_client(router({}))
    assert client.last_stats is None


def test_last_stats_is_isolated_per_thread() -> None:
    """Two threads sharing one client (e.g. Assistant's assist + notes workers) must each
    see their own stream's stats, even if the other thread's call finishes and overwrites
    the client in between one thread's call returning and it reading `last_stats`."""
    lock = threading.Lock()

    def completions(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        n = 1 if body["prompt"] == "a-prompt" else 5
        with lock:
            return httpx.Response(200, content=sse_body([completion_chunk("x")] * n + [None]))

    handler = router({"/v1/completions": completions})
    client = make_client(handler)

    results: dict[str, int] = {}

    def worker_a() -> None:
        list(client.stream_completion("a-prompt", max_tokens=10))
        # Simulate real downstream work (building an event, etc.) between the stream
        # finishing and this thread reading back its own stats.
        time.sleep(0.15)
        assert client.last_stats is not None
        results["a"] = client.last_stats.tokens

    def worker_b() -> None:
        time.sleep(0.05)  # start after A's call has completed, during A's gap
        list(client.stream_completion("b-prompt", max_tokens=10))
        assert client.last_stats is not None
        results["b"] = client.last_stats.tokens

    ta = threading.Thread(target=worker_a)
    tb = threading.Thread(target=worker_b)
    ta.start()
    tb.start()
    ta.join(timeout=5)
    tb.join(timeout=5)

    assert results == {"a": 1, "b": 5}


# ---------------------------------------------------------------------------
# make_client (backend dispatch + an explicit Route base_url override)
# ---------------------------------------------------------------------------


def test_make_client_openai_backend_uses_effective_server_url() -> None:
    from exocortex.llm.client import make_client as build_client

    cfg = LLMConfig(server_url="http://h:8081", model="m")
    client = build_client(cfg, "openai")
    assert isinstance(client, LMStudioClient)
    assert client.base_url == "http://h:8081"


def test_make_client_ollama_backend_returns_an_ollama_client() -> None:
    from exocortex.llm.client import OllamaClient
    from exocortex.llm.client import make_client as build_client

    cfg = LLMConfig(model="m")
    client = build_client(cfg, "ollama")
    assert isinstance(client, OllamaClient)
    assert client.base_url == cfg.effective_server_url("ollama")


def test_make_client_applies_an_explicit_base_url_override_without_touching_cfg() -> None:
    """This is how a Route that found the server at a non-standard address (Ollama on a custom
    port, or an OpenAI-compatible server) is honoured -- without writing that address into cfg."""
    from exocortex.llm.client import make_client as build_client

    cfg = LLMConfig(model="m")  # cfg.base_url is still the standard openai address
    client = build_client(cfg, "openai", base_url="http://found-here:9999")
    assert client.base_url == "http://found-here:9999"
    assert cfg.base_url != "http://found-here:9999"  # set_base_url never mutates cfg


def test_make_client_with_no_base_url_override_keeps_the_cfg_derived_default() -> None:
    from exocortex.llm.client import make_client as build_client

    cfg = LLMConfig(server_url="http://h:1", model="m")
    client = build_client(cfg, "openai", base_url=None)
    assert client.base_url == "http://h:1"


def test_make_client_ignores_an_empty_base_url_override() -> None:
    from exocortex.llm.client import make_client as build_client

    cfg = LLMConfig(server_url="http://h:1", model="m")
    client = build_client(cfg, "openai", base_url="")
    assert client.base_url == "http://h:1"


# ---------------------------------------------------------------------------
# live (real LM Studio, skipped by default)
# ---------------------------------------------------------------------------


@pytest.mark.live
def test_live_stream_completion_against_lm_studio() -> None:
    # maple-preview is a ChatML/Qwen3-style chat model: /v1/completions does not apply a
    # template, so a plain raw prompt yields an immediate stop token. Render the ChatML
    # "no-think" prompt (see SPEC.md) to get a direct answer, same as llm/prompts.py will.
    prompt = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\nSay hello in one short sentence.<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
    cfg = LLMConfig(server_url="http://localhost:1234", model="maple-preview")
    client = LMStudioClient(cfg)
    try:
        client.ensure_loaded()
    except LLMError as exc:
        pytest.skip(f"LM Studio not reachable: {exc}")

    text = "".join(client.stream_completion(prompt, max_tokens=40, stop=["<|im_end|>", "<|im_start|>"]))

    assert text.strip() != ""
    assert client.last_stats is not None
    assert client.last_stats.tok_per_s > 0

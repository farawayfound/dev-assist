"""Unit tests for exocortex.llm.client.OllamaClient, all offline via httpx.MockTransport (see
tests/test_client.py for the LMStudioClient/parent tests these mirror)."""
from __future__ import annotations

import json
import threading
from typing import Any, Callable

import httpx
import pytest

from exocortex.config import LLMConfig
from exocortex.llm.client import ChatDelta, Health, LLMError, ModelInfo, OllamaClient

Handler = Callable[[httpx.Request], httpx.Response]


def make_ollama_client(handler: Handler, **cfg_kwargs: Any) -> OllamaClient:
    cfg = LLMConfig(server_url="http://unit.test", model=cfg_kwargs.pop("model", "qwen3:8b"), **cfg_kwargs)
    return OllamaClient(cfg, transport=httpx.MockTransport(handler))


def router(routes: dict[str, Handler]) -> Handler:
    def handler(request: httpx.Request) -> httpx.Response:
        route = routes.get(request.url.path)
        if route is None:
            return httpx.Response(404, content=b"not found")
        return route(request)

    return handler


def json_response(status: int, data: Any) -> Handler:
    return lambda request: httpx.Response(status, content=json.dumps(data).encode("utf-8"))


def ndjson_body(lines: list[dict[str, Any]]) -> bytes:
    return b"".join((json.dumps(line) + "\n").encode("utf-8") for line in lines)


def ndjson_response(lines: list[dict[str, Any]]) -> Handler:
    return lambda request: httpx.Response(200, content=ndjson_body(lines))


TAGS_TWO_MODELS = {
    "models": [
        {"name": "qwen3:8b", "size": 5_000_000_000, "details": {"family": "qwen3", "quantization_level": "Q4_K_M"}},
        {"name": "llama3.1:70b", "size": 40_000_000_000, "details": {"family": "llama", "quantization_level": "Q4_K_M"}},
    ]
}


# ---------------------------------------------------------------------------
# list_models / /api/ps
# ---------------------------------------------------------------------------


def test_list_models_parses_tags_and_remembers_sizes() -> None:
    handler = router({
        "/api/tags": json_response(200, TAGS_TWO_MODELS),
        "/api/ps": json_response(200, {"models": []}),
    })
    client = make_ollama_client(handler)

    models = client.list_models()

    assert models == [
        ModelInfo(id="qwen3:8b", loaded=False, arch="qwen3", quant="Q4_K_M", context_length=None),
        ModelInfo(id="llama3.1:70b", loaded=False, arch="llama", quant="Q4_K_M", context_length=None),
    ]
    assert client.model_sizes == {"qwen3:8b": 5_000_000_000, "llama3.1:70b": 40_000_000_000}


def test_list_models_marks_loaded_from_api_ps() -> None:
    handler = router({
        "/api/tags": json_response(200, TAGS_TWO_MODELS),
        "/api/ps": json_response(200, {"models": [{"name": "qwen3:8b"}]}),
    })
    client = make_ollama_client(handler)

    models = client.list_models()

    assert [m.loaded for m in models] == [True, False]


def test_list_models_leaves_loaded_none_when_api_ps_does_not_answer() -> None:
    handler = router({
        "/api/tags": json_response(200, TAGS_TWO_MODELS),
        "/api/ps": lambda r: httpx.Response(500, content=b"boom"),
    })
    client = make_ollama_client(handler)

    models = client.list_models()

    assert [m.loaded for m in models] == [None, None]


def test_list_models_raises_llm_error_on_http_failure() -> None:
    handler = router({"/api/tags": json_response(500, {"error": "boom"})})
    client = make_ollama_client(handler)
    with pytest.raises(LLMError):
        client.list_models()


# ---------------------------------------------------------------------------
# health() -- explicit model and "auto"
# ---------------------------------------------------------------------------


def test_health_reports_model_listed() -> None:
    handler = router({
        "/api/tags": json_response(200, TAGS_TWO_MODELS),
        "/api/ps": json_response(200, {"models": [{"name": "qwen3:8b"}]}),
    })
    client = make_ollama_client(handler, model="qwen3:8b")

    h = client.health()

    assert h == Health(reachable=True, model_listed=True, model_loaded=True, detail="")


def test_health_reports_model_not_listed() -> None:
    handler = router({
        "/api/tags": json_response(200, TAGS_TWO_MODELS),
        "/api/ps": json_response(200, {"models": []}),
    })
    client = make_ollama_client(handler, model="mistral:7b")

    h = client.health()

    assert h.model_listed is False
    assert "mistral:7b" in h.detail


def test_health_auto_picks_the_largest_model_when_no_gpu_is_known(monkeypatch: pytest.MonkeyPatch) -> None:
    import exocortex.gpu as gpu_mod

    monkeypatch.setattr(gpu_mod, "primary_gpu", lambda: None)
    handler = router({
        "/api/tags": json_response(200, TAGS_TWO_MODELS),
        "/api/ps": json_response(200, {"models": []}),
    })
    client = make_ollama_client(handler, model="auto")

    h = client.health()

    assert h.reachable is True
    assert client.resolved_model == "llama3.1:70b"          # the larger of the two
    assert "llama3.1:70b" in h.detail
    assert client.model_id == "llama3.1:70b"


def test_health_auto_picks_the_largest_model_that_fits_the_gpu_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    import exocortex.gpu as gpu_mod

    class FakeGpu:
        budget_mb = 40_000   # 0.75 * 40000 MiB ~= 31.5 GB: the 70B (40 GB) does not fit, the 8B does

    monkeypatch.setattr(gpu_mod, "primary_gpu", lambda: FakeGpu())
    handler = router({
        "/api/tags": json_response(200, TAGS_TWO_MODELS),
        "/api/ps": json_response(200, {"models": []}),
    })
    client = make_ollama_client(handler, model="")

    h = client.health()

    assert client.resolved_model == "qwen3:8b"
    assert h.model_listed is True


def test_health_unreachable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    client = make_ollama_client(handler)
    h = client.health()
    assert h.reachable is False
    assert "Ollama" in h.detail


def test_health_no_models_pulled() -> None:
    handler = router({"/api/tags": json_response(200, {"models": []})})
    client = make_ollama_client(handler, model="auto")
    h = client.health()
    assert h.reachable is True
    assert h.model_listed is False


# ---------------------------------------------------------------------------
# ensure_loaded
# ---------------------------------------------------------------------------


def test_ensure_loaded_sends_the_documented_warmup_request() -> None:
    calls: list[dict[str, Any]] = []

    def generate(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content))
        return httpx.Response(200, content=b'{"done": true}')

    handler = router({
        "/api/tags": json_response(200, {"models": [{"name": "qwen3:8b"}]}),
        "/api/ps": json_response(200, {"models": []}),
        "/api/generate": generate,
    })
    client = make_ollama_client(handler, model="qwen3:8b")

    assert client.ensure_loaded() is True
    assert calls == [{"model": "qwen3:8b", "prompt": "", "keep_alive": "30m"}]


def test_ensure_loaded_noop_when_already_loaded() -> None:
    calls: list[str] = []

    def generate(request: httpx.Request) -> httpx.Response:
        calls.append("hit")
        return httpx.Response(200, content=b"{}")

    handler = router({
        "/api/tags": json_response(200, {"models": [{"name": "qwen3:8b"}]}),
        "/api/ps": json_response(200, {"models": [{"name": "qwen3:8b"}]}),
        "/api/generate": generate,
    })
    client = make_ollama_client(handler, model="qwen3:8b")

    assert client.ensure_loaded() is True
    assert calls == []


# ---------------------------------------------------------------------------
# stream_completion
# ---------------------------------------------------------------------------


def test_stream_completion_parses_ndjson_and_reports_stats_from_eval_count() -> None:
    lines = [
        {"response": "Hel", "done": False},
        {"response": "lo", "done": False},
        {"response": "", "done": True, "eval_count": 7, "prompt_eval_count": 12, "done_reason": "stop"},
    ]
    handler = router({"/api/generate": ndjson_response(lines)})
    client = make_ollama_client(handler, model="qwen3:8b")

    out = "".join(client.stream_completion("hi", max_tokens=50))

    assert out == "Hello"
    assert client.last_stats.tokens == 7
    assert client.last_stats.cancelled is False


def test_stream_completion_body_maps_options_and_drops_unknown_keys() -> None:
    captured: dict[str, Any] = {}

    def generate(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(200, content=ndjson_body([{"response": "x", "done": True}]))

    handler = router({"/api/generate": generate})
    client = make_ollama_client(
        handler, model="qwen3:8b", ctx_size=4096,
        extra_body={"top_k": 20, "min_p": 0.0, "repeat_penalty": 1.1, "frequency_penalty": 9},
    )

    list(client.stream_completion("hi", max_tokens=50, stop=["</s>"]))

    assert captured["model"] == "qwen3:8b"
    assert captured["prompt"] == "hi"
    assert captured["raw"] is True
    assert captured["stream"] is True
    assert captured["keep_alive"] == "30m"
    options = captured["options"]
    assert options["num_predict"] == 50
    assert options["stop"] == ["</s>"]
    assert options["num_ctx"] == 4096
    assert options["top_k"] == 20
    assert options["min_p"] == 0.0
    assert options["repeat_penalty"] == 1.1
    assert "frequency_penalty" not in options            # unknown to Ollama: dropped


def test_stream_completion_stop_defaults_to_empty_list() -> None:
    captured: dict[str, Any] = {}

    def generate(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(200, content=ndjson_body([{"response": "x", "done": True}]))

    handler = router({"/api/generate": generate})
    client = make_ollama_client(handler, model="qwen3:8b")
    list(client.stream_completion("hi", max_tokens=10))
    assert captured["options"]["stop"] == []


def test_stream_completion_cancel_mid_stream_stops_early() -> None:
    lines = [{"response": f"tok{i} ", "done": False} for i in range(20)]
    handler = router({"/api/generate": ndjson_response(lines)})
    client = make_ollama_client(handler, model="qwen3:8b")
    cancel = threading.Event()

    pieces = []
    for i, piece in enumerate(client.stream_completion("hi", max_tokens=100, cancel=cancel)):
        pieces.append(piece)
        if i == 2:
            cancel.set()

    assert len(pieces) <= 4
    assert client.last_stats.cancelled is True


def test_stream_completion_raises_on_mid_stream_error_line() -> None:
    lines = [
        {"response": "partial answer", "done": False},
        {"error": "model requires more system memory (16.0 GiB) than is available (8.0 GiB)"},
    ]
    handler = router({"/api/generate": ndjson_response(lines)})
    client = make_ollama_client(handler, model="qwen3:8b")

    gen = client.stream_completion("hi", max_tokens=50)
    chunks: list[str] = []
    with pytest.raises(LLMError) as exc_info:
        for piece in gen:
            chunks.append(piece)

    assert chunks == ["partial answer"]
    assert "model requires more system memory" in str(exc_info.value)
    assert client.last_stats.cancelled is False


def test_stream_completion_connect_error_message_mentions_ollama() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    client = make_ollama_client(handler, model="qwen3:8b")
    with pytest.raises(LLMError) as exc_info:
        list(client.stream_completion("hi", max_tokens=10))
    msg = str(exc_info.value)
    assert "Ollama" in msg and "ollama serve" in msg and "ollama pull" in msg


# ---------------------------------------------------------------------------
# stream_chat
# ---------------------------------------------------------------------------


def test_stream_chat_yields_content_and_native_thinking_field() -> None:
    lines = [
        {"message": {"role": "assistant", "content": "", "thinking": "let me think"}, "done": False},
        {"message": {"role": "assistant", "content": "the answer"}, "done": False},
        {"message": {"role": "assistant", "content": ""}, "done": True, "eval_count": 3},
    ]
    handler = router({"/api/chat": ndjson_response(lines)})
    client = make_ollama_client(handler, model="qwen3:8b")

    deltas = list(client.stream_chat([{"role": "user", "content": "hi"}], max_tokens=50))

    reasoning = "".join(d.reasoning for d in deltas)
    content = "".join(d.content for d in deltas)
    assert reasoning == "let me think"
    assert content == "the answer"
    assert client.last_stats.tokens == 3


def test_stream_chat_splits_inline_think_tags_too() -> None:
    lines = [
        {"message": {"role": "assistant", "content": "<think>hmm</think>answer"}, "done": True, "eval_count": 2},
    ]
    handler = router({"/api/chat": ndjson_response(lines)})
    client = make_ollama_client(handler, model="qwen3:8b")

    deltas = [d for d in client.stream_chat([{"role": "user", "content": "hi"}], max_tokens=50) if d.content or d.reasoning]

    assert "".join(d.reasoning for d in deltas) == "hmm"
    assert "".join(d.content for d in deltas) == "answer"


def test_stream_chat_raises_on_mid_stream_error_line() -> None:
    lines = [
        {"message": {"role": "assistant", "content": "partial "}, "done": False},
        {"error": "model requires more system memory (16.0 GiB) than is available (8.0 GiB)"},
    ]
    handler = router({"/api/chat": ndjson_response(lines)})
    client = make_ollama_client(handler, model="qwen3:8b")

    gen = client.stream_chat([{"role": "user", "content": "hi"}], max_tokens=50)
    contents: list[str] = []
    with pytest.raises(LLMError) as exc_info:
        for delta in gen:
            contents.append(delta.content)

    assert "".join(contents) == "partial "
    assert "model requires more system memory" in str(exc_info.value)
    assert client.last_stats.cancelled is False


def test_stream_chat_body_shape() -> None:
    captured: dict[str, Any] = {}

    def chat(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(200, content=ndjson_body([{"message": {"content": "x"}, "done": True}]))

    handler = router({"/api/chat": chat})
    client = make_ollama_client(handler, model="qwen3:8b")
    messages = [{"role": "user", "content": "hi"}]

    list(client.stream_chat(messages, max_tokens=30))

    assert captured["model"] == "qwen3:8b"
    assert captured["messages"] == messages
    assert captured["stream"] is True
    assert captured["keep_alive"] == "30m"
    assert captured["options"]["num_predict"] == 30

"""Tests for exocortex.llm.route (backend routing at startup): every probe is injected, so these
run on any machine, incl. one with no GPU, no LM Studio and no Ollama. Only tests marked
@pytest.mark.live touch this box's real llama.cpp packs / Ollama installation (see test_live.py)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

import pytest

from exocortex.config import STANDARD_SERVER_URLS, LLMConfig
from exocortex.llm import route as route_mod
from exocortex.llm.route import Route, describe_route, ollama_candidates, resolve_route


@dataclass(frozen=True)
class _FakeBinary:
    flavor: str = "vulkan"
    version: str = "2.46.0"


def _find_ok(*_a: Any, **_k: Any) -> _FakeBinary:
    return _FakeBinary()


def _find_fails(*_a: Any, **_k: Any) -> Any:
    raise RuntimeError("no LM Studio runtime packs found")


def _scan(n: int) -> Callable[[], list]:
    return lambda: [object()] * n


def _http_get(routes: dict[str, "tuple[int, str] | list[tuple[int, str]]"]) -> Callable[[str, float], "tuple[int, str]"]:
    """Scripted GET: each URL maps to a fixed (status, body), or a list consumed one-at-a-time
    (last entry repeats). Unlisted URLs answer (0, "") -- "nothing there"."""
    calls: list[str] = []

    def get(url: str, timeout: float) -> "tuple[int, str]":
        calls.append(url)
        for prefix, value in routes.items():
            if url.startswith(prefix):
                if isinstance(value, list):
                    return value.pop(0) if len(value) > 1 else value[0]
                return value
        return 0, ""

    get.calls = calls  # type: ignore[attr-defined]
    return get


def ticks() -> Callable[[], float]:
    it = iter(range(100_000))
    return lambda: float(next(it))


def ollama_body(n: int) -> str:
    return json.dumps({"models": [{"name": f"m{i}"} for i in range(n)]})


# ---------------------------------------------------------------------------------- explicit ---
@pytest.mark.parametrize(
    "backend,expected_url",
    [("llamacpp", ""), ("ollama", "http://127.0.0.1:11434"), ("openai", "http://127.0.0.1:8081")],
)
def test_explicit_backend_skips_all_probing(backend: str, expected_url: str) -> None:
    cfg = LLMConfig(backend=backend)
    calls: list[str] = []

    def boom(*_a: Any, **_k: Any) -> Any:
        calls.append("called")
        raise AssertionError("must not probe when the backend is explicit")

    route = resolve_route(cfg, find_binary=boom, scan=boom, http_get=boom)
    assert route.backend == backend
    assert route.base_url == expected_url
    assert route.reason == f"configured: [llm].backend = {backend!r}"
    assert route.checked == ()
    assert calls == []


# --------------------------------------------------------------------------------------- auto ---
def test_auto_picks_llamacpp_when_a_pack_and_a_gguf_are_both_present() -> None:
    cfg = LLMConfig(backend="auto")
    route = resolve_route(cfg, find_binary=_find_ok, scan=_scan(3), http_get=_http_get({}))
    assert route.backend == "llamacpp"
    assert route.base_url == ""
    assert "vulkan" in route.reason and "2.46.0" in route.reason and "3" in route.reason
    assert route.model_count == 3
    assert isinstance(route.binary, _FakeBinary)
    assert route.checked == (f"llamacpp: ok ({route.reason})",)


def test_auto_skips_llamacpp_when_no_pack_is_found() -> None:
    cfg = LLMConfig(backend="auto")
    get = _http_get({"http://127.0.0.1:11434/api/tags": (200, ollama_body(2))})
    route = resolve_route(cfg, find_binary=_find_fails, scan=_scan(5), http_get=get)
    assert route.backend == "ollama"
    assert "no LM Studio runtime packs" in route.checked[0]


def test_auto_skips_llamacpp_when_the_pack_exists_but_the_library_is_empty() -> None:
    cfg = LLMConfig(backend="auto")
    get = _http_get({"http://127.0.0.1:11434/api/tags": (200, ollama_body(1))})
    route = resolve_route(cfg, find_binary=_find_ok, scan=_scan(0), http_get=get)
    assert route.backend == "ollama"
    assert "no GGUF models" in route.checked[0]


def test_auto_falls_to_ollama_when_it_answers_at_a_non_default_server_url() -> None:
    cfg = LLMConfig(backend="auto", server_url="http://h:11434")
    get = _http_get({"http://h:11434/api/tags": (200, ollama_body(4))})
    route = resolve_route(cfg, find_binary=_find_fails, scan=_scan(0), http_get=get)
    assert route.backend == "ollama"
    assert route.base_url == "http://h:11434"
    assert route.reason == "Ollama at http://h:11434 (4 models)"
    assert route.checked[-1] == "ollama: http://h:11434 200"


def test_auto_finds_ollama_at_the_standard_port_when_server_url_points_at_an_openai_server() -> None:
    """[llm].server_url is a single field shared by the ollama/openai probes: when it points at an
    OpenAI-compatible server instead, Ollama's own standard address still gets tried (see
    ollama_candidates) before the router falls through past it."""
    cfg = LLMConfig(backend="auto", server_url="http://h:9999")
    get = _http_get({
        "http://h:9999/v1/models": (200, "{}"),
        "http://127.0.0.1:11434/api/tags": (200, ollama_body(3)),
    })
    route = resolve_route(cfg, find_binary=_find_fails, scan=_scan(0), http_get=get)
    assert route.backend == "ollama"
    assert route.base_url == "http://127.0.0.1:11434"
    assert route.checked[-2] == "ollama: http://h:9999 0"
    assert route.checked[-1] == "ollama: http://127.0.0.1:11434 200"


def test_auto_falls_to_openai_when_ollama_is_unreachable() -> None:
    cfg = LLMConfig(backend="auto", server_url="http://h:8081")
    get = _http_get({"http://h:8081/v1/models": (200, "{}")})
    route = resolve_route(cfg, find_binary=_find_fails, scan=_scan(0), http_get=get)
    assert route.backend == "openai"
    assert route.base_url == "http://h:8081"
    assert "http://h:8081" in route.reason
    assert route.checked[-1] == "openai: http://h:8081 200"


# --------------------------------------------------------------------------- ollama_candidates ---
def test_ollama_candidates_is_just_the_standard_address_when_server_url_is_unset() -> None:
    assert ollama_candidates(LLMConfig()) == [STANDARD_SERVER_URLS["ollama"]]


def test_ollama_candidates_tries_server_url_first_then_the_standard_address() -> None:
    cfg = LLMConfig(server_url="http://h:9")
    assert ollama_candidates(cfg) == ["http://h:9", STANDARD_SERVER_URLS["ollama"]]


def test_ollama_candidates_dedups_when_server_url_already_is_the_standard_address() -> None:
    cfg = LLMConfig(server_url=STANDARD_SERVER_URLS["ollama"])
    assert ollama_candidates(cfg) == [STANDARD_SERVER_URLS["ollama"]]


def test_auto_gives_none_with_an_actionable_reason_when_nothing_answers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(route_mod, "_safe_gpu_vendor", lambda: "amd")
    cfg = LLMConfig(backend="auto")
    route = resolve_route(cfg, find_binary=_find_fails, scan=_scan(0), http_get=_http_get({}))
    assert route.backend == "none"
    assert route.base_url == ""
    assert "LM Studio" in route.reason and "Ollama" in route.reason
    assert "no LM Studio runtime packs" in route.reason        # names why llamacpp was skipped
    assert len(route.checked) == 3                              # llamacpp, ollama, openai


def test_vendor_is_passed_through_to_find_binary() -> None:
    seen: dict[str, Any] = {}

    def find(explicit: str, *, vendor: str | None = None, backend: str = "auto") -> _FakeBinary:
        seen["vendor"], seen["backend"] = vendor, backend
        return _FakeBinary()

    cfg = LLMConfig(backend="auto", llama_backend="rocm")
    resolve_route(cfg, vendor="amd", find_binary=find, scan=_scan(1), http_get=_http_get({}))
    assert seen == {"vendor": "amd", "backend": "rocm"}


def test_vendor_defaults_to_gpu_vendor_when_not_given(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(route_mod, "_safe_gpu_vendor", lambda: "intel")
    seen: dict[str, Any] = {}

    def find(explicit: str, *, vendor: str | None = None, backend: str = "auto") -> _FakeBinary:
        seen["vendor"] = vendor
        return _FakeBinary()

    resolve_route(LLMConfig(backend="auto"), find_binary=find, scan=_scan(1), http_get=_http_get({}))
    assert seen == {"vendor": "intel"}


# -------------------------------------------------------------------------- ollama autostart ---
def test_ollama_autostart_launches_and_waits_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(route_mod, "ollama_installed", lambda: True)
    cfg = LLMConfig(backend="auto", server_url="http://h:11434")
    get = _http_get({"http://h:11434/api/tags": [(0, ""), (0, ""), (200, ollama_body(2))]})
    launched: list[str] = []
    slept: list[float] = []

    route = resolve_route(
        cfg, find_binary=_find_fails, scan=_scan(0), http_get=get, start_ollama=True,
        ollama_launcher=lambda: launched.append("started"), clock=ticks(), sleep=slept.append,
    )
    assert launched == ["started"]
    assert route.backend == "ollama"
    assert route.reason == "started Ollama at http://h:11434 (2 models)"
    assert slept == [0.5]                          # one poll answered "not yet", the next succeeded
    assert route.checked[-1].startswith("ollama: http://h:11434 200")


def test_ollama_autostart_polls_every_candidate_each_round(monkeypatch: pytest.MonkeyPatch) -> None:
    """The autostart poll walks the SAME candidate list as the plain probe: a server_url that
    points somewhere else must not stop it from finding Ollama once it comes up at the standard
    address."""
    monkeypatch.setattr(route_mod, "ollama_installed", lambda: True)
    cfg = LLMConfig(backend="auto", server_url="http://h:9999")
    get = _http_get({"http://127.0.0.1:11434/api/tags": [(0, ""), (200, ollama_body(1))]})

    route = resolve_route(
        cfg, find_binary=_find_fails, scan=_scan(0), http_get=get, start_ollama=True,
        ollama_launcher=lambda: None, clock=ticks(), sleep=lambda _s: None,
    )
    assert route.backend == "ollama"
    assert route.base_url == "http://127.0.0.1:11434"
    assert route.reason == "started Ollama at http://127.0.0.1:11434 (1 models)"


def test_ollama_autostart_is_skipped_when_ollama_is_not_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(route_mod, "ollama_installed", lambda: False)
    cfg = LLMConfig(backend="auto")
    launched: list[str] = []
    route = resolve_route(
        cfg, find_binary=_find_fails, scan=_scan(0), http_get=_http_get({}), start_ollama=True,
        ollama_launcher=lambda: launched.append("started"),
    )
    assert launched == []
    assert route.backend == "none"
    assert any("not installed" in c for c in route.checked)


def test_ollama_autostart_gives_up_after_the_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(route_mod, "ollama_installed", lambda: True)
    cfg = LLMConfig(backend="auto")

    class Clock:
        def __init__(self) -> None:
            self.t = 0.0

        def __call__(self) -> float:
            self.t += 3.0            # advances past the 20s budget after ~7 polls
            return self.t

    slept: list[float] = []
    route = resolve_route(
        cfg, find_binary=_find_fails, scan=_scan(0), http_get=_http_get({}), start_ollama=True,
        ollama_launcher=lambda: None, clock=Clock(), sleep=slept.append,
    )
    assert route.backend == "none"
    assert any("did not come up" in c for c in route.checked)
    assert slept                                    # it did poll before giving up


def test_ollama_autostart_launch_failure_is_reported_and_never_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(route_mod, "ollama_installed", lambda: True)
    cfg = LLMConfig(backend="auto")

    def boom() -> None:
        raise OSError("ollama.exe missing despite the check")

    route = resolve_route(
        cfg, find_binary=_find_fails, scan=_scan(0), http_get=_http_get({}), start_ollama=True,
        ollama_launcher=boom,
    )
    assert route.backend == "none"
    assert any("launch failed" in c for c in route.checked)


# ------------------------------------------------------------------------------- never raises ---
def test_a_raising_http_get_is_treated_as_unreachable() -> None:
    def boom(url: str, timeout: float) -> "tuple[int, str]":
        raise RuntimeError("transport bug")

    route = resolve_route(LLMConfig(backend="auto"), find_binary=_find_fails, scan=_scan(0), http_get=boom)
    assert route.backend == "none"


def test_a_raising_find_binary_is_treated_as_no_pack() -> None:
    def boom(*_a: Any, **_k: Any) -> Any:
        raise RuntimeError("weird ctypes failure")

    route = resolve_route(LLMConfig(backend="auto"), find_binary=boom, scan=_scan(1), http_get=_http_get({}))
    assert route.backend == "none"
    assert "weird ctypes failure" in route.checked[0]


def test_a_raising_scan_is_treated_as_no_models() -> None:
    def boom() -> list:
        raise OSError("permission denied")

    route = resolve_route(LLMConfig(backend="auto"), find_binary=_find_ok, scan=boom, http_get=_http_get({}))
    assert route.backend != "llamacpp"


# ------------------------------------------------------------------------------ describe_route ---
def test_describe_route_lists_backend_reason_and_every_probe() -> None:
    r = Route(backend="none", base_url="", reason="nothing available", checked=("llamacpp: no pack", "ollama: x 0"))
    text = describe_route(r)
    lines = text.splitlines()
    assert lines[0] == "backend: none"
    assert lines[1] == "reason: nothing available"
    assert lines[2] == "  - llamacpp: no pack"
    assert lines[3] == "  - ollama: x 0"


def test_ollama_installed_true_when_on_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(route_mod.shutil, "which", lambda name: "C:/ollama/ollama.exe")
    assert route_mod.ollama_installed() is True


def test_ollama_installed_false_when_nowhere_to_be_found(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    monkeypatch.setattr(route_mod.shutil, "which", lambda name: None)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    assert route_mod.ollama_installed() is False

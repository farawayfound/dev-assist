"""Tests for freeing the shared llama-swap model server (exocortex/instance.py free_llama_swap)
and its wiring: a --replace launch unloads it before our model loads, and Engine.stop() unloads
it on the way out. All HTTP is faked (tests/conftest.py additionally stubs free_llama_swap
itself for every test, so nothing here -- or anywhere -- can reach the machine's real llama-swap;
the unit tests below hold a direct reference to the original function, which survives the stub)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from exocortex import app as app_mod
from exocortex import instance as inst
from exocortex.instance import ReplaceResult, free_llama_swap


class FakeSwap:
    """Scripted llama-swap: a queue of /running bodies, plus a log of every request."""

    def __init__(self, running_bodies: list[Any], unload_status: int = 200) -> None:
        self.bodies = list(running_bodies)
        self.unload_status = unload_status
        self.calls: list[str] = []

    def __call__(self, url: str, timeout: float) -> tuple[int, str]:
        self.calls.append(url)
        if url.endswith("/unload"):
            return self.unload_status, "OK"
        body = self.bodies.pop(0) if len(self.bodies) > 1 else self.bodies[0]
        if isinstance(body, tuple):
            return body
        return 200, json.dumps(body)


def ticks() -> Any:
    it = iter(range(10_000))
    return lambda: float(next(it))


ONE = {"running": [{"model": "qwen3.8-27b", "state": "ready"}]}
NONE = {"running": []}


# ------------------------------------------------------------------------ free_llama_swap ---
def test_unloads_and_reports_when_a_model_is_resident() -> None:
    swap = FakeSwap([ONE, NONE])
    notice = free_llama_swap("http://s:1/", http_get=swap, clock=ticks(), sleep=lambda s: None)
    assert notice == "Freed the shared model server (1 model(s))"
    assert swap.calls == ["http://s:1/running", "http://s:1/unload", "http://s:1/running"]


def test_counts_models_in_the_notice() -> None:
    two = {"running": [{"model": "a"}, {"model": "b"}]}
    notice = free_llama_swap("http://s:1", http_get=FakeSwap([two, NONE]), clock=ticks(), sleep=lambda s: None)
    assert notice == "Freed the shared model server (2 model(s))"


def test_nothing_resident_is_silent_and_never_unloads() -> None:
    swap = FakeSwap([NONE])
    assert free_llama_swap("http://s:1", http_get=swap) == ""
    assert swap.calls == ["http://s:1/running"]


@pytest.mark.parametrize("first", [(0, ""), (500, "err"), (200, "not json"), (200, '{"odd": 1}')])
def test_unreachable_or_malformed_server_is_silent(first: tuple[int, str]) -> None:
    swap = FakeSwap([first])
    assert free_llama_swap("http://s:1", http_get=swap) == ""
    assert swap.calls == ["http://s:1/running"]          # never escalates to /unload


def test_blank_url_does_nothing_at_all() -> None:
    swap = FakeSwap([ONE])
    assert free_llama_swap("", http_get=swap) == ""
    assert free_llama_swap("   ", http_get=swap) == ""
    assert swap.calls == []


def test_timeout_when_the_model_never_releases() -> None:
    slept: list[float] = []
    swap = FakeSwap([ONE])                                # /running answers ONE forever
    notice = free_llama_swap("http://s:1", timeout_s=3.0, http_get=swap,
                             clock=ticks(), sleep=slept.append)
    assert notice == "The shared model server did not release its model in time"
    assert slept and all(s == 1.0 for s in slept)


def test_a_raising_http_get_never_escapes() -> None:
    def boom(url: str, timeout: float) -> tuple[int, str]:
        raise RuntimeError("transport bug")

    # free_llama_swap promises never to raise; the injectable contract maps failures to (0, ""),
    # but even a broken injectable must not take a launch or a quit down with it.
    try:
        result = free_llama_swap("http://s:1", http_get=boom)
    except RuntimeError:
        pytest.fail("free_llama_swap raised")
    assert result == ""


# --------------------------------------------------------------------------- main() wiring ---
@pytest.fixture()
def cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """main() with the takeover, the swap unload and both runners replaced by recorders."""
    calls: list[tuple] = []
    config = tmp_path / "config.toml"
    config.write_text(
        f'[paths]\nsessions_dir = "{(tmp_path / "sessions").as_posix()}"\n', encoding="utf-8"
    )
    monkeypatch.setattr("exocortex.logging_setup.setup_logging", lambda *a, **k: None)
    monkeypatch.setattr(
        inst, "replace_previous",
        lambda *a, **k: calls.append(("replace",)) or ReplaceResult(
            closed=[inst.ProcInfo(pid=1, ppid=0, name="pythonw.exe", created=1,
                                  argv=("pythonw.exe", "-m", "exocortex"))]),
    )
    monkeypatch.setattr(inst, "free_llama_swap",
                        lambda url, **k: calls.append(("swap", url)) or "Freed the shared model server (1 model(s))")
    monkeypatch.setattr(app_mod, "_run_ui", lambda cfg, notice="": calls.append(("ui", notice)) or 0)
    monkeypatch.setattr(app_mod, "_run_headless", lambda cfg, duration_s=0.0: calls.append(("headless",)) or 0)
    return config, calls


def test_replace_unloads_the_swap_after_the_takeover_and_merges_notices(cli) -> None:
    """The `cli` fixture's config has no `[llm]` section, so `backend` is the "auto" default
    (2026-09-25: routing at startup, exocortex.llm.route) rather than the old "llamacpp" default --
    the swap-unload gate in `main()` treats "auto" as llamacpp-like for exactly this reason (the
    route is not resolved yet this early, and unloading the shared server is harmless either way)."""
    config, calls = cli
    assert app_mod.main(["--config", str(config), "--replace"]) == 0
    assert calls == [
        ("replace",),
        ("swap", "http://127.0.0.1:8081"),
        ("ui", "Closed the previous ExoCortex session; Freed the shared model server (1 model(s))"),
    ]


def test_openai_backend_never_unloads_its_own_server(cli, tmp_path: Path) -> None:
    config, calls = cli
    config.write_text(
        '[llm]\nbackend = "openai"\n'
        f'[paths]\nsessions_dir = "{(tmp_path / "sessions").as_posix()}"\n', encoding="utf-8"
    )
    assert app_mod.main(["--config", str(config), "--replace"]) == 0
    assert ("swap", "http://127.0.0.1:8081") not in calls          # that server IS the backend
    assert calls[0] == ("replace",)


def test_blank_swap_url_disables_the_launch_unload(cli, tmp_path: Path) -> None:
    config, calls = cli
    config.write_text(
        '[llm]\nswap_unload_url = ""\n'
        f'[paths]\nsessions_dir = "{(tmp_path / "sessions").as_posix()}"\n', encoding="utf-8"
    )
    assert app_mod.main(["--config", str(config), "--replace"]) == 0
    assert not any(c[0] == "swap" for c in calls)


def test_without_replace_nothing_is_touched(cli) -> None:
    config, calls = cli
    assert app_mod.main(["--config", str(config)]) == 0
    assert calls == [("ui", "")]


# ------------------------------------------------------------------------ Engine.stop wiring ---
from exocortex.bus import EventBus  # noqa: E402
from exocortex.config import Config  # noqa: E402
from tests.test_app import engine_env  # noqa: E402,F401  (fixture reuse)


def test_engine_stop_unloads_the_swap(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The quit path with the llamacpp backend. The engine is constructed but never start()ed, so
    `self.route` is still None and `Engine.stop()` falls back to the config test -- `Config()`'s
    default `backend` is "auto" (2026-09-25), which counts as llamacpp-like there for the same
    reason `main()`'s gate does (see the "auto" comment in `Engine.stop()`). Starting for real with
    backend "llamacpp" would try to spawn a real llama-server (the engine_env fixture avoids that
    by using "openai", but that backend legitimately skips the unload); stop() runs the same
    shutdown sequence either way."""
    cfg = Config()
    cfg.path = tmp_path / "config.toml"                            # conftest guard: never his file
    calls: list[tuple[str, float]] = []
    monkeypatch.setattr(inst, "free_llama_swap",
                        lambda url, timeout_s=45.0, **k: calls.append((url, timeout_s)) or "")
    eng = app_mod.Engine(cfg, EventBus())
    eng.stop()
    assert calls == [("http://127.0.0.1:8081", 10.0)]              # quit budget: stays snappy
    eng.stop()                                                     # idempotent stop: no second call
    assert len(calls) == 1


def test_engine_stop_skips_the_swap_for_openai_backend(
    engine_env: "tuple[Config, EventBus]", monkeypatch: pytest.MonkeyPatch
) -> None:
    """Full start/stop cycle on the engine_env fixture (backend "openai"): that server IS the
    backend, so quitting must not unload it."""
    cfg, bus = engine_env
    calls: list[str] = []
    monkeypatch.setattr(inst, "free_llama_swap", lambda url, **k: calls.append(url) or "")
    eng = app_mod.Engine(cfg, bus)
    eng.start()
    eng.stop()
    assert calls == []


def test_engine_stop_skips_the_swap_with_a_blank_url(monkeypatch: pytest.MonkeyPatch,
                                                     tmp_path: Path) -> None:
    cfg = Config()
    cfg.path = tmp_path / "config.toml"
    cfg.llm.swap_unload_url = ""
    calls: list[str] = []
    monkeypatch.setattr(inst, "free_llama_swap", lambda url, **k: calls.append(url) or "")
    eng = app_mod.Engine(cfg, EventBus())
    eng.stop()
    assert calls == []

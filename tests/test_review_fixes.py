"""Pins for the 2026-09-27 adversarial-review findings on the portable-install pass (each was
confirmed by two independent refuters before being fixed):

1. a fresh machine (route "none": no model at start) can load its first pulled model without a
   restart -- Engine.set_model creates the LlamaServerManager, starts the assistant, lifts the
   "LLM unavailable" refusal and replays queued requests;
2. one model change at a time -- a Load/Unload while another swap is running is refused with a
   notice instead of racing on the one LlamaServerManager;
3. resolve_route's default probes see the app's own library (not just LM Studio's folder) and
   honour [llm].llama_source / llama_upstream_build / runtime_dir;
4. pull_model's free-space check credits bytes already on disk (a resumed .part, a finished shard);
5. Hub-supplied path components can never leave the library directory;
6. the ladder's second CUDA rung is the 12.4 build after a 13.4 first rung, and nothing when the
   first rung already was 12.4 (server._upstream_kind_for);
7. the Models window's Server tab builds exactly one form, and Load/Unload/Delete are disabled
   while llama-server is switching models.

Round two (same date, a second adversarial review of the same pass):

8.  Engine.set_model only writes `[llm].model` once a swap was actually accepted -- a request
    _start_swap refuses as "still busy" leaves cfg untouched (E1);
9.  Engine.stop() stops whatever self.server ends up being (not just the one it snapshotted before
    a racing first Load could create one) and joins the "llm-swap" thread too (E2);
10. that "still busy" refusal carries the stable "busy: " prefix, and unload_model's thread
    publishes a "loading" status as soon as it starts running (E3);
11. set_model_profile coerces the settings dialog's plain-text values -- extra_args (str -> argv
    tokens, "[]"/"" -> []) and the numeric fields (str -> int/float, or dropped) -- instead of
    storing whatever `ModelProfile(**values)` happened to receive (E4).
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from exocortex import app as app_mod
from exocortex.bus import ComponentStatusEvent, EventBus, ModelLibraryEvent, drain
from exocortex.config import Config, LLMConfig
from exocortex.llm import library, route, server as srv
from exocortex.llm.route import Route


def _wait(q: Any, pred: Any, timeout: float = 5.0) -> list[Any]:
    seen: list[Any] = []
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for ev in drain(q):
            seen.append(ev)
            if pred(ev):
                return seen
        time.sleep(0.01)
    raise AssertionError(f"event not seen within {timeout}s; saw {seen}")


# ---- 1 + 2: engine ---------------------------------------------------------------------------
@dataclass
class _Model:
    key: str


class _FakeManager:
    """Stands in for LlamaServerManager: swap_model blocks on `gate` when one is given."""

    instances: list["_FakeManager"] = []
    gate: threading.Event | None = None

    def __init__(self, cfg: Any, *, log_dir: Any = None) -> None:
        self.cfg = cfg
        self.model: _Model | None = None
        self.running = False
        self.stopped = 0
        _FakeManager.instances.append(self)

    def swap_model(self, spec: str, *, on_status: Any = None) -> str:
        if _FakeManager.gate is not None:
            _FakeManager.gate.wait(5.0)
        self.model = _Model(spec)
        self.running = True
        return "http://127.0.0.1:5555"

    def stop(self) -> None:
        self.stopped += 1
        self.running = False


class _Assistant:
    def __init__(self) -> None:
        self.started = 0
        self.asked: list[str] = []
        self.models: list[str] = []

    def start(self) -> None:
        self.started += 1

    def cancel(self) -> None:
        pass

    def set_model(self, model_id: str) -> None:
        self.models.append(model_id)

    def ask(self, text: str) -> None:
        self.asked.append(text)


class _Client:
    def __init__(self) -> None:
        self.urls: list[str] = []

    def set_base_url(self, url: str) -> None:
        self.urls.append(url)


@pytest.fixture()
def fresh_engine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> app_mod.Engine:
    """An engine whose start() resolved to backend "none" (empty library): no server, assistant
    never started, _drop_pending already refused requests."""
    _FakeManager.instances.clear()
    _FakeManager.gate = None
    import exocortex.llm.server as server_mod

    monkeypatch.setattr(server_mod, "LlamaServerManager", _FakeManager)
    cfg = Config()
    cfg.path = tmp_path / "config.toml"
    cfg.llm.library_dir = str(tmp_path / "llm")
    eng = app_mod.Engine(cfg, EventBus())
    eng.route = Route(backend="none", base_url="", reason="no model in the library yet")
    eng.client = _Client()
    eng.assistant = _Assistant()
    eng.recorder = None
    eng._llm_failed = "no model in the library yet"
    return eng


def test_first_load_on_a_fresh_machine_creates_the_server_and_starts_the_assistant(fresh_engine: app_mod.Engine) -> None:
    eng = fresh_engine
    q = eng.bus.subscribe()
    assert eng.server is None and not eng._llm_ready.is_set()
    # A request made while nothing could answer: refused, not queued (upstream finding 9).
    assert eng._defer("ask", "too early") is True
    # Now the model is pulled and loaded from the Models window:
    eng.load_model("unsloth/gemma-4-E2B-it")
    seen = _wait(q, lambda ev: isinstance(ev, ModelLibraryEvent) and ev.reason == "loaded")
    eng._swap_thread.join(5.0)
    assert len(_FakeManager.instances) == 1 and eng.server is _FakeManager.instances[0]
    assert eng.server.model.key == "unsloth/gemma-4-E2B-it" and eng.server.running
    assert eng.client.urls == ["http://127.0.0.1:5555"]
    assert eng.assistant.started == 1 and eng.assistant.models == ["unsloth/gemma-4-E2B-it"]
    assert eng._llm_ready.is_set() and eng._llm_failed is None
    assert seen[-1].loaded_key == "unsloth/gemma-4-E2B-it"
    assert eng.loaded_model() == "unsloth/gemma-4-E2B-it"
    # From here on requests run immediately, and a request queued during the load is replayed.
    assert eng._defer("ask", "now") is False
    eng._llm_ready.clear()
    assert eng._defer("ask", "queued during reload") is True
    eng.load_model("empero-ai/Qwen3.8-4B-Distill")
    _wait(q, lambda ev: isinstance(ev, ModelLibraryEvent) and ev.reason == "loaded" and "Qwen" in ev.loaded_key)
    eng._swap_thread.join(5.0)
    assert eng.assistant.asked == ["queued during reload"]
    assert eng.assistant.started == 1                    # started once, never twice


def test_no_server_is_spawned_before_start_or_for_a_remote_backend(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import exocortex.llm.server as server_mod

    _FakeManager.instances.clear()
    monkeypatch.setattr(server_mod, "LlamaServerManager", _FakeManager)
    cfg = Config()
    cfg.path = tmp_path / "config.toml"
    eng = app_mod.Engine(cfg, EventBus())
    eng.assistant = _Assistant()
    eng.set_model("x")                                   # no route yet: the old synchronous path
    assert eng.assistant.models == ["x"] and eng.server is None and _FakeManager.instances == []
    assert eng.cfg.llm.model == "x"                      # E1: this branch still sets cfg directly
    eng.route = Route(backend="ollama", base_url="http://127.0.0.1:11434", reason="t")
    eng.set_model("y")
    assert eng.assistant.models == ["x", "y"] and eng.server is None and _FakeManager.instances == []
    assert eng.cfg.llm.model == "y"


def test_a_second_model_change_is_refused_while_one_is_running(fresh_engine: app_mod.Engine) -> None:
    eng = fresh_engine
    eng._llm_failed = None
    _FakeManager.gate = threading.Event()
    q = eng.bus.subscribe()
    eng.load_model("a/b")
    _wait(q, lambda ev: isinstance(ev, ComponentStatusEvent) and ev.state == "loading")
    eng.load_model("c/d")
    eng.unload_model()
    evs = drain(q)
    notices = [ev for ev in evs if isinstance(ev, ComponentStatusEvent) and "still busy" in ev.detail]
    assert len(notices) == 2 and "loading c/d skipped" in notices[0].detail and "unloading skipped" in notices[1].detail
    # review round two, 2026-09-27 (E3): the stable "busy: " prefix lets a consumer ignore this
    # refusal for state tracking, while "still busy" (asserted above) stays in the text.
    assert all(n.detail.startswith("busy: ") for n in notices)
    assert len(_FakeManager.instances) == 1
    _FakeManager.gate.set()
    _wait(q, lambda ev: isinstance(ev, ModelLibraryEvent) and ev.reason == "loaded")
    eng._swap_thread.join(5.0)
    assert eng.server.model.key == "a/b"                 # the first request won; nothing raced it
    _FakeManager.gate = None
    eng.unload_model()
    _wait(q, lambda ev: isinstance(ev, ModelLibraryEvent) and ev.reason == "unloaded")
    eng._swap_thread.join(5.0)
    assert eng.server.stopped == 1 and eng.loaded_model() == ""


# ---- 8: E1 -- cfg.llm.model changes only when the swap was accepted ---------------------------
def test_set_model_leaves_cfg_untouched_when_the_swap_is_refused(fresh_engine: app_mod.Engine) -> None:
    """A load refused as "still busy" (E1) must not become the persisted default -- only the
    accepted swap that follows may change `cfg.llm.model`."""
    eng = fresh_engine
    eng._llm_failed = None
    _FakeManager.gate = threading.Event()
    q = eng.bus.subscribe()
    eng.load_model("a/b")
    _wait(q, lambda ev: isinstance(ev, ComponentStatusEvent) and ev.state == "loading")
    eng.load_model("c/d")                                # refused: a swap is already running
    assert eng.cfg.llm.model != "c/d"                    # the refused request never committed
    _FakeManager.gate.set()
    _wait(q, lambda ev: isinstance(ev, ModelLibraryEvent) and ev.reason == "loaded")
    eng._swap_thread.join(5.0)
    assert eng.cfg.llm.model == "a/b"                    # the accepted swap did commit it
    _FakeManager.gate = None


# ---- 9: E2 -- stop() catches a server the swap thread creates ---------------------------------
def test_stop_stops_a_server_created_by_a_racing_load(fresh_engine: app_mod.Engine) -> None:
    """E2: self.server is still None when stop()'s component-loop tuple is built, mirroring a
    fresh machine; a first Load races the quit and creates/starts a LlamaServerManager (here from
    inside another component's own .stop(), to land deterministically within the loop) that was
    never in that tuple. stop() must still stop it, not just whatever it snapshotted up front."""
    eng = fresh_engine
    _FakeManager.gate = None

    class _RacingAssistant(_Assistant):
        def stop(self) -> None:
            manager = _FakeManager(eng.cfg.llm)
            manager.swap_model("raced/in")
            eng.server = manager                          # assigned AFTER stop()'s tuple was built

    eng.assistant = _RacingAssistant()
    assert eng.server is None
    eng.stop()
    assert eng.server is not None and eng.server.running is False and eng.server.stopped >= 1


def test_stop_stops_a_server_the_swap_thread_creates_during_the_join(
    fresh_engine: app_mod.Engine, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """E2: the in-flight swap thread can still be between its two `_stopped` checks -- past the
    first (before `_ensure_server`) but not yet at the second (before `swap_model`) -- when stop()
    runs. stop() must catch the server that thread creates while the join is still waiting, not
    only the one it saw (or didn't see) right before the join started."""
    eng = fresh_engine
    eng._llm_failed = None
    original_ensure = eng._ensure_server

    def slow_ensure_server() -> Any:
        time.sleep(0.3)          # gives stop() time to run its first "self.server is not None" check
        return original_ensure()

    monkeypatch.setattr(eng, "_ensure_server", slow_ensure_server)
    eng.load_model("a/b")
    assert eng.server is None    # the swap thread is still inside _ensure_server's sleep
    eng.stop()                   # sets _stopped, sees server is still None, then joins the thread
    assert eng.server is not None and eng.server.stopped >= 1
    assert eng._swap_thread is not None and not eng._swap_thread.is_alive()


# ---- 3: route probes --------------------------------------------------------------------------
def test_default_route_probes_use_the_merged_library_and_the_runtime_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_find(explicit: str, **kw: Any) -> Any:
        captured.update(kw)
        return srv.ServerBinary(exe=Path("X:/rt/llama-server.exe"), dll_dirs=(), flavor="nvidia-cuda12",
                                version="b11200", framework="cuda", reason="t", source="upstream")

    monkeypatch.setattr(srv, "find_server_binary", fake_find)
    monkeypatch.setattr(srv, "scan_models", lambda d: [])                    # LM Studio's folder: empty
    monkeypatch.setattr(library, "scan_library", lambda cfg, **kw: [_Model("own/model")])
    cfg = LLMConfig(backend="auto", llama_source="upstream", runtime_dir="X:/rt", llama_upstream_build="b11200")
    r = route.resolve_route(cfg, vendor="nvidia", http_get=lambda url, t: (0, ""))
    assert r.backend == "llamacpp" and r.model_count == 1
    assert captured["source"] == "upstream" and captured["upstream_build"] == "b11200"
    assert captured["upstream_root"] == Path("X:/rt") and captured["vendor"] == "nvidia"

    monkeypatch.setattr(library, "scan_library", lambda cfg, **kw: [])
    r = route.resolve_route(cfg, vendor="nvidia", http_get=lambda url, t: (0, ""))
    assert r.backend == "none" and "no GGUF models" in r.reason


# ---- 4 + 5: pull_model --------------------------------------------------------------------------
def _repo_info(repo_id: str, size: int, name: str = "m-Q4_K_M.gguf") -> library.HfRepoInfo:
    f = library.HfFile(name, size, "")
    return library.HfRepoInfo(id=repo_id, sha="s", files=(f,), architecture="x", params=None, context_length=None,
                              quants=(library.QuantOption("Q4_K_M", (f,), size),), mmproj=())


@pytest.fixture()
def pull_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    cfg = LLMConfig(library_dir=str(tmp_path / "llm"))
    state: dict[str, Any] = {"free": 10_000, "pulled": []}
    monkeypatch.setattr(library, "hf_repo_info", lambda repo_id, http=None: _repo_info(repo_id, 100))

    class _Usage:
        def __init__(self, free: int) -> None:
            self.free = free

    monkeypatch.setattr(shutil, "disk_usage", lambda p: _Usage(state["free"]))

    def fake_pull_one(client: Any, repo_id: str, hf_file: Any, dest: Path, progress: Any, cancelled: Any) -> None:
        state["pulled"].append(dest)
        dest.write_bytes(b"x" * hf_file.size)

    monkeypatch.setattr(library, "_pull_one_file", fake_pull_one)
    monkeypatch.setattr(library, "_new_hf_client", lambda: type("C", (), {"close": lambda self: None})())
    state["cfg"] = cfg
    return state


def test_free_space_check_credits_a_resumed_part_and_finished_shards(pull_env: dict[str, Any], tmp_path: Path) -> None:
    cfg = pull_env["cfg"]
    dest_root = tmp_path / "llm" / "acme" / "demo"
    dest_root.mkdir(parents=True)
    pull_env["free"] = 20
    with pytest.raises(library.PullError, match="need 0.00 GB more"):
        library.pull_model("acme/demo", "Q4_K_M", cfg)      # 100 bytes needed, 20 free
    (dest_root / "m-Q4_K_M.gguf.part").write_bytes(b"y" * 90)   # 90 already on disk: only 10 more needed
    model = library.pull_model("acme/demo", "Q4_K_M", cfg)
    assert model.key == "acme/demo" and pull_env["pulled"] == [dest_root / "m-Q4_K_M.gguf"]
    pull_env["free"] = 0                                          # the finished shard needs nothing more
    library.pull_model("acme/demo", "Q4_K_M", cfg)
    assert len(pull_env["pulled"]) == 2


@pytest.mark.parametrize("repo_id", ["../escape", "acme/../../x", "C:/abs/path", "acme/re/po", "", "acme/.."])
def test_unsafe_repo_ids_are_refused(pull_env: dict[str, Any], repo_id: str) -> None:
    with pytest.raises(library.PullError, match="unsafe"):
        library.pull_model(repo_id, "Q4_K_M", pull_env["cfg"])
    assert pull_env["pulled"] == []


@pytest.mark.parametrize("name", ["..", "."])
def test_unsafe_file_names_are_refused(pull_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    monkeypatch.setattr(library, "hf_repo_info", lambda repo_id, http=None: _repo_info(repo_id, 100, name))
    with pytest.raises(library.PullError, match="unsafe"):
        library.pull_model("acme/demo", "Q4_K_M", pull_env["cfg"])
    assert pull_env["pulled"] == []


@pytest.mark.parametrize("name", ["../../evil.gguf", "C:\\evil.gguf", "sub/dir/evil.gguf"])
def test_file_paths_with_directories_land_inside_the_repo_folder(pull_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
                                                                 name: str, tmp_path: Path) -> None:
    """A Hub path with directory parts is reduced to its basename: the file lands under the repo
    folder, never where the directory parts pointed."""
    monkeypatch.setattr(library, "hf_repo_info", lambda repo_id, http=None: _repo_info(repo_id, 100, name))
    library.pull_model("acme/demo", "Q4_K_M", pull_env["cfg"])
    assert pull_env["pulled"] == [tmp_path / "llm" / "acme" / "demo" / "evil.gguf"]
    assert not (tmp_path / "evil.gguf").exists() and not Path("C:/evil.gguf").exists()


# ---- 6: the second CUDA rung -----------------------------------------------------------------
def test_second_cuda_rung_is_the_12_4_build_only_after_a_13_4_first_rung(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(srv.gpu, "nvidia_driver_version", lambda *a, **k: "592.14", raising=False)
    monkeypatch.setattr(srv.runtimes, "cuda_kind_for_driver", lambda d: "cuda13", raising=False)
    assert srv._upstream_kind_for("nvidia-cuda12", "nvidia") == "cuda13"
    assert srv._upstream_kind_for("nvidia-cuda", "nvidia") == "cuda12"
    monkeypatch.setattr(srv.runtimes, "cuda_kind_for_driver", lambda d: "cuda12", raising=False)
    assert srv._upstream_kind_for("nvidia-cuda12", "nvidia") == "cuda12"
    assert srv._upstream_kind_for("nvidia-cuda", "nvidia") is None
    assert srv._upstream_kind_for("amd-rocm", "amd") == "rocm"
    assert srv._upstream_kind_for("vulkan", "intel") == "vulkan"


# ---- 7: the Models window ----------------------------------------------------------------------
class _DialogActions:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []

    def model_library(self) -> list[dict[str, Any]]:
        return [{"key": "own/a", "name": "a", "size_bytes": 1e9, "source": "library", "quant": "Q4_K_M",
                 "path": "X:/lib/own/a/a.gguf", "context_length": 8192, "loaded": True, "is_default": True,
                 "has_profile": False},
                {"key": "own/b", "name": "b", "size_bytes": 2e9, "source": "library", "quant": "Q6_K",
                 "path": "X:/lib/own/b/b.gguf", "context_length": 8192, "loaded": False, "is_default": False,
                 "has_profile": False}]

    def loaded_model(self) -> str:
        return "own/a"

    def hardware_report(self, refresh: bool = False) -> dict[str, Any]:
        return {"budget_mb": 16003, "vendor": "nvidia", "gpus": [], "notes": [], "recommended_policy": "auto"}

    def model_profile(self, key: str) -> None:
        return None

    def __getattr__(self, name: str) -> Any:
        def rec(*a: Any, **k: Any) -> None:
            self.calls.append((name, a))
        return rec


@pytest.fixture(scope="module")
def qapp() -> Any:
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication(["exocortex-tests"])


def test_server_tab_builds_exactly_one_form(qapp: Any) -> None:
    from PySide6.QtWidgets import QScrollArea

    from exocortex.ui.models import ModelsDialog

    dlg = ModelsDialog(Config(), _DialogActions(), None)
    try:
        dlg.show_tab("Server")
        server_tab = dlg.tabs.widget(list(ModelsDialog.TABS).index("Server"))
        assert len(server_tab.findChildren(QScrollArea)) == 1
        assert "llm.backend" in dlg._server_form.widgets and "llm.server_url" in dlg._server_form.widgets
    finally:
        dlg.close()
        qapp.processEvents()


def test_library_buttons_are_disabled_while_the_model_is_switching(qapp: Any) -> None:
    from exocortex.ui.models import ModelsDialog

    actions = _DialogActions()
    dlg = ModelsDialog(Config(), actions, None)
    try:
        dlg.show_tab("Library")
        dlg.library_table.selectRow(1)                     # "own/b", not loaded
        qapp.processEvents()
        assert dlg.load_btn.isEnabled() and dlg.unload_btn.isEnabled() and dlg.delete_btn.isEnabled()
        dlg.on_event(ComponentStatusEvent("llm", "loading", "switching to own/b"))
        assert not dlg.load_btn.isEnabled() and not dlg.unload_btn.isEnabled() and not dlg.delete_btn.isEnabled()
        assert dlg.set_default_btn.isEnabled()             # harmless while loading: only writes config
        dlg.on_event(ComponentStatusEvent("llm", "ready", "own/b"))
        assert dlg.load_btn.isEnabled() and dlg.unload_btn.isEnabled() and dlg.delete_btn.isEnabled()
    finally:
        dlg.close()
        qapp.processEvents()

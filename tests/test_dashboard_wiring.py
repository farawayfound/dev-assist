"""Engine wiring for dashboard mode and the context lookups (2026-10-20): config round trip of
the new sections, the deferred dashboard actions, insights_now routing to the dashboard while it
tracks, the Context-menu switches, apply_settings reconfiguring the provider and the mode event,
and that no view mode touches the screen reader's cadence."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from exocortex.bus import ComponentStatusEvent, DashboardModeEvent, EventBus, drain
from exocortex.config import Config, load_config, save_config


def test_the_new_sections_round_trip_through_toml(tmp_path: Path) -> None:
    cfg = Config()
    cfg.dashboard.enabled = True
    cfg.dashboard.refresh_interval_s = 45
    cfg.dashboard.show_notes = False
    cfg.context.repo_root = "D:/repos/thing"
    cfg.context.repo_exclude = [".git", "vendor"]
    cfg.context.toolkit_dir = "~/claude-toolkit"
    cfg.llm.dashboard_max_tokens = 500
    cfg.hotkeys.toggle_dashboard = "ctrl+alt+shift+d"
    path = save_config(cfg, tmp_path / "config.toml")
    text = path.read_text(encoding="utf-8")
    assert "[dashboard]" in text and "[context]" in text and 'toggle_dashboard = "ctrl+alt+shift+d"' in text
    loaded = load_config(path)
    assert loaded.dashboard.enabled and loaded.dashboard.refresh_interval_s == 45 and not loaded.dashboard.show_notes
    assert loaded.context.repo_root == "D:/repos/thing" and loaded.context.repo_exclude == [".git", "vendor"]
    assert loaded.context.toolkit_dir == "~/claude-toolkit" and loaded.llm.dashboard_max_tokens == 500
    defaults = Config()
    assert defaults.dashboard.track and defaults.dashboard.refresh_interval_s == 90 and defaults.dashboard.width == 1180
    assert defaults.dashboard.show_goals and defaults.dashboard.show_constraints and defaults.dashboard.show_prompts
    assert defaults.context.toolkit and defaults.context.repo_search and defaults.context.repo_root == ""
    assert defaults.context.repo_auto and defaults.context.repo_auto_max_age_h == 12
    assert defaults.hotkeys.toggle_dashboard == "ctrl+alt+d"


class _Service:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []

    def refresh_now(self, sections: Any = ()) -> int:
        self.calls.append(("refresh_now", tuple(sections)))
        return 1

    def item_action(self, item_id: int, action: str, text: str = "") -> bool:
        self.calls.append(("item_action", item_id, action, text))
        return True

    def add_item(self, kind: str, text: str, parent_id: int | None = None) -> bool:
        self.calls.append(("add_item", kind, text, parent_id))
        return True

    def set_paused(self, paused: bool) -> None:
        self.calls.append(("set_paused", paused))


class _Assistant:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []

    def insights_now(self, sections: Any = None) -> int:
        self.calls.append(("insights_now", sections))
        return 1


def test_dashboard_refresh_is_deferred_while_the_llm_starts_then_replayed() -> None:
    from exocortex.app import Engine

    eng = Engine(Config(), EventBus())
    eng.assistant = _Assistant()
    service = _Service()
    eng.dashboard = service
    eng.dashboard_refresh(("questions",))
    eng.dashboard_refresh(("questions", "nextsteps"))    # latest request wins
    assert service.calls == [] and [p[0] for p in eng._pending] == ["dashboard"]
    eng._llm_ready.set()
    eng._replay_pending()
    assert service.calls == [("refresh_now", ("questions", "nextsteps"))]
    eng.dashboard_refresh()
    assert service.calls[-1] == ("refresh_now", ())
    eng.dashboard_item(4, "answer", "yes")
    eng.dashboard_add("question", "Who?")
    eng.dashboard_add("nextstep", "a step", 4)
    assert service.calls[-3:] == [("item_action", 4, "answer", "yes"), ("add_item", "question", "Who?", None),
                                  ("add_item", "nextstep", "a step", 4)]


def test_insights_now_refreshes_the_dashboard_while_it_tracks() -> None:
    from exocortex.app import Engine

    cfg = Config()
    eng = Engine(cfg, EventBus())
    assistant = _Assistant()
    service = _Service()
    eng.assistant = assistant
    eng.dashboard = service
    eng._llm_ready.set()
    eng.insights_now(("questions",))
    assert service.calls == [("refresh_now", ("questions",))] and assistant.calls == []
    cfg.dashboard.track = False
    eng.insights_now(("blindspots",))
    assert assistant.calls == [("insights_now", ("blindspots",))] and len(service.calls) == 1
    eng.dashboard = None
    eng.dashboard_refresh()                                 # no service: a status line, no crash
    eng.set_paused(True)
    eng.set_paused(False)


def test_set_dashboard_mode_publishes_the_event_and_persists_the_flag() -> None:
    from exocortex.app import Engine

    cfg = Config()
    bus = EventBus()
    sub = bus.subscribe()
    eng = Engine(cfg, bus)
    eng.set_dashboard_mode(True)
    assert cfg.dashboard.enabled
    events = [e for e in drain(sub) if isinstance(e, DashboardModeEvent)]
    assert events and events[-1].enabled
    eng.set_dashboard_mode(False)
    assert not cfg.dashboard.enabled


def test_completion_context_switches_the_lookups_too() -> None:
    from exocortex.app import Engine

    cfg = Config()
    eng = Engine(cfg, EventBus())
    eng.set_completion_context({"repo": False, "toolkit": False, "screen": False})
    assert not cfg.context.repo_search and not cfg.context.toolkit_in_answers and not cfg.completion.use_screen
    eng.set_completion_context({"repo": True})
    assert cfg.context.repo_search and not cfg.context.toolkit_in_answers


def test_apply_settings_reconfigures_the_context_and_toggles_the_mode(tmp_path: Path) -> None:
    from exocortex.app import Engine

    class _Context:
        def __init__(self) -> None:
            self.reconfigured = 0

        def reconfigure(self) -> None:
            self.reconfigured += 1

        def context_summary(self) -> str:
            return "repo: no toolkit installed"

    cfg = Config()
    cfg.path = tmp_path / "config.toml"
    bus = EventBus()
    sub = bus.subscribe()
    eng = Engine(cfg, bus)
    ctx = _Context()
    eng.context = ctx
    repo = tmp_path / "repo"
    repo.mkdir()
    cfg.context.repo_root = str(repo)
    cfg.dashboard.enabled = True
    cfg.screen.interval_s = 1.5
    eng.apply_settings({"context.repo_root": str(repo), "dashboard.enabled": True, "dashboard.track": True})
    assert ctx.reconfigured == 1
    events = drain(sub)
    assert any(isinstance(e, DashboardModeEvent) and e.enabled for e in events)
    assert any(isinstance(e, ComponentStatusEvent) and e.component == "context" and e.detail == "Context: repo: no toolkit installed"
               for e in events)
    cfg.screen.monitor = 1
    eng.apply_settings({"screen.monitor": 1})               # no reader yet: nothing to re-point, no crash
    assert load_config(cfg.path).screen.monitor == 1
    saved = load_config(cfg.path)
    assert saved.context.repo_root == str(repo) and saved.dashboard.enabled
    assert saved.screen.interval_s == 1.5                   # the view mode never touches the capture cadence
    assert eng.screen is None                               # and nothing here started or re-tuned the reader

"""Which repository is being worked on (exocortex.context.detect, 2026-10-21): the newest Claude
Code session log names it; the context provider uses it when no [context].repo_root is set."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from exocortex.config import Config
from exocortex.context import detect
from exocortex.context.provider import ContextProvider


class ManualClock:
    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


@pytest.fixture(autouse=True)
def _scratch_is_elsewhere(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """pytest's tmp_path lives under the system temp dir, which detection treats as scratch space:
    point "temp" somewhere else for these tests (and use it for the scratch case)."""
    monkeypatch.setattr(detect.tempfile, "gettempdir", lambda: str(tmp_path / "scratch"))
    monkeypatch.setattr("exocortex.context.provider.detect_repo_root", detect.detect_repo_root)


def _log(home: Path, folder: str, cwd: Path | None, mtime: float, name: str = "session.jsonl", *, noise: int = 2) -> None:
    d = home / ".claude" / "projects" / folder
    d.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps({"type": "queue-operation", "n": n}) for n in range(noise)]
    if cwd is not None:
        lines.append(json.dumps({"type": "system", "cwd": str(cwd)}))
    f = d / name
    f.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.utime(f, (mtime, mtime))


def _repo(base: Path, name: str, *, skill: str = "") -> Path:
    root = base / name
    root.mkdir(parents=True)
    if skill:
        (root / ".claude" / "skills" / skill).mkdir(parents=True)
        (root / ".claude" / "skills" / skill / "SKILL.md").write_text(
            f"---\nname: {skill}\ndescription: look around\nargument-hint: '[topic]'\n---\nbody", encoding="utf-8")
    return root


def test_the_most_recently_written_log_names_the_repository(tmp_path: Path) -> None:
    home, now = tmp_path / "home", time.time()
    a, b = _repo(tmp_path, "scenario-a"), _repo(tmp_path, "scenario-b")
    _log(home, "a", a, now - 3600)
    _log(home, "b", b, now - 60)
    _log(home, "b", a, now - 7200, name="older-session.jsonl")      # only the newest log of a folder counts
    found = detect.recent_projects(home, now=now)
    assert [p for p, _ in found] == [b, a] and found[0][1] == pytest.approx(now - 60, abs=1)
    assert detect.detect_repo_root(home, now=now) == b


def test_unusable_candidates_are_skipped(tmp_path: Path) -> None:
    home, now = tmp_path / "home", time.time()
    real = _repo(tmp_path, "real")
    scratch = tmp_path / "scratch" / "psmux-test"
    scratch.mkdir(parents=True)
    _log(home, "gone", tmp_path / "was-deleted", now - 10)           # the directory no longer exists
    _log(home, "scratch", scratch, now - 20)                          # a scratch session under the temp dir
    _log(home, "nocwd", None, now - 30)                               # a log that never recorded a cwd
    _log(home, "real", real, now - 40)
    (home / ".claude" / "projects" / "empty").mkdir()
    (home / ".claude" / "projects" / "stray.txt").write_text("x", encoding="utf-8")
    assert [p for p, _ in detect.recent_projects(home, now=now)] == [real]


def test_age_limit_and_a_missing_projects_folder(tmp_path: Path) -> None:
    home, now = tmp_path / "home", time.time()
    old = _repo(tmp_path, "old")
    _log(home, "old", old, now - 5 * 3600)
    assert detect.detect_repo_root(home, max_age_s=3600, now=now) is None
    assert detect.detect_repo_root(home, max_age_s=0, now=now) == old          # 0 = no age limit
    assert detect.detect_repo_root(tmp_path / "nobody", now=now) is None


def test_the_cwd_is_found_past_a_few_noise_lines_and_bad_json(tmp_path: Path) -> None:
    home, now = tmp_path / "home", time.time()
    repo = _repo(tmp_path, "r")
    d = home / ".claude" / "projects" / "r"
    d.mkdir(parents=True)
    (d / "s.jsonl").write_text('not json\n{"cwd": 5}\n' + '{"x": 1}\n' * 10 + json.dumps({"cwd": str(repo)}) + "\n", encoding="utf-8")
    assert detect.detect_repo_root(home, now=now + 1) == repo


def test_the_provider_uses_the_detected_repository_when_none_is_configured(tmp_path: Path) -> None:
    home, now = tmp_path / "home", time.time()
    a = _repo(tmp_path, "scenario-a", skill="recon")
    _log(home, "a", a, now - 60)
    cfg = Config()
    clock = ManualClock()
    provider = ContextProvider(cfg, clock, home=home)
    assert provider.effective_root() == str(a) and provider.repo_source == "auto"
    assert provider.repo_root == a and provider.has_toolkit and provider.toolkit_summary().startswith("1 skill")
    assert provider.context_summary() == "scenario-a (detected from Claude Code): 1 skill"
    assert provider.toolkit_example() == "/recon [topic]" and provider.check_prompt("/recon x") and not provider.check_prompt("/ghost x")
    # another project becomes the active one: picked up by the next refresh, after the short cache
    b = _repo(tmp_path, "scenario-b")
    _log(home, "b", b, time.time())
    assert provider.refresh() is False and provider.repo_root == a              # cached for a few seconds
    clock.t += 20
    assert provider.refresh() is True and provider.repo_root == b and not provider.has_toolkit
    assert provider.context_summary() == "scenario-b (detected from Claude Code): no toolkit installed"


def test_an_explicit_root_wins_and_the_switch_turns_detection_off(tmp_path: Path) -> None:
    home, now = tmp_path / "home", time.time()
    detected, chosen = _repo(tmp_path, "detected"), _repo(tmp_path, "chosen")
    _log(home, "d", detected, now - 5)
    cfg = Config()
    provider = ContextProvider(cfg, ManualClock(), home=home)
    assert provider.repo_root == detected
    cfg.context.repo_root = str(chosen)
    provider.reconfigure()
    assert provider.repo_root == chosen and provider.repo_source == "config"
    assert provider.context_summary() == "chosen ([context].repo_root): no toolkit installed"
    cfg.context.repo_root = ""
    cfg.context.repo_auto = False
    provider.reconfigure()
    assert provider.repo_root is None and provider.repo_source == "" and "no repository" in provider.context_summary()
    cfg.context.repo_auto = True
    cfg.context.repo_auto_max_age_h = 0
    provider.reconfigure()
    assert provider.repo_root == detected                                        # 0 hours = no age limit

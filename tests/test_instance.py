"""Tests for ``exocortex/instance.py`` (``--replace``: force-close every earlier ExoCortex so a
``run.bat`` launch is always the only session) and its wiring in ``exocortex/app.py``'s ``main()``.

The selection is pure and tested on synthetic process tables. The Win32 layer runs for real only
in ways that cannot touch anything but processes these tests start themselves: a read-only
snapshot, and a takeover restricted to a dummy "exocortex" process -- a sleeping script at
``<tmp>\\exocortex\\__main__.py``, run by path and never as ``-m exocortex``, so the real app (and
its 27B model) can never start by accident.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from exocortex import app as app_mod
from exocortex import instance as inst
from exocortex.instance import ProcInfo, ReplaceResult, is_exocortex_argv, select_targets

STUB = r"D:\Repos\exo-cortex\.venv\Scripts\pythonw.exe"      # the venv's launcher stub
BASE = r"C:\Python312\pythonw.exe"                            # the interpreter it runs as a child
LLAMA = r"C:\Users\u\.lmstudio\extensions\backends\llama.cpp\llama-server.exe"
SESSIONS = r"C:\Users\u\Documents\ExoCortex\sessions"


def P(pid: int, ppid: int, name: str, created: int | None, *argv: str) -> ProcInfo:
    return ProcInfo(pid=pid, ppid=ppid, name=name, created=created, argv=tuple(argv) if argv else None)


def machine() -> list[ProcInfo]:
    """A realistic table: an old session, the llmstack, other programs, and this (new) launch."""
    return [
        P(4, 0, "System", None),
        # the old session: venv stub -> real interpreter -> its llama-server, console host, an app it opened
        P(100, 50, "pythonw.exe", 10, STUB, "-m", "exocortex"),
        P(101, 100, "pythonw.exe", 11, BASE, "-m", "exocortex"),
        P(102, 101, "llama-server.exe", 12, LLAMA, "--model", r"D:\m.gguf",
          "--log-file", SESSIONS + r"\2026-09-24_01-07-34\llama-server.log"),
        P(103, 101, "conhost.exe", 12),
        P(104, 101, "notepad.exe", 13),
        # the llmstack: llama-swap and its llama-server (unreadable: no argv) -- never touched
        P(200, 190, "llama-swap.exe", 5),
        P(201, 200, "llama-server.exe", 20),
        # another Python program that merely mentions exocortex
        P(300, 1, "python.exe", 15, BASE, "-m", "pytest", "-m", "exocortex"),
        # this launch (run.bat): cmd -> venv stub -> this interpreter
        P(400, 1, "cmd.exe", 29),
        P(401, 400, "pythonw.exe", 30, STUB, "-m", "exocortex", "--replace"),
        P(402, 401, "pythonw.exe", 31, BASE, "-m", "exocortex", "--replace"),
        # a stray ExoCortex llama-server whose parent is long gone
        P(500, 999, "llama-server.exe", 8, LLAMA, "--log-file", SESSIONS + r"\2026-09-23_22-00-00\llama-server.log"),
        # a process that inherited a stale parent id (created before its "parent" 101): a stranger
        P(600, 101, "python.exe", 3, BASE, "tool.py"),
        # an interpreter whose command line could not be read (another user / elevated)
        P(700, 1, "pythonw.exe", 9),
    ]


def pids(procs: list[ProcInfo]) -> set[int]:
    return {p.pid for p in procs}


# ------------------------------------------------------------------------ is_exocortex_argv ---
@pytest.mark.parametrize("argv", [
    (BASE, "-m", "exocortex"),
    (STUB, "-m", "exocortex", "--replace"),
    (r"C:\Python312\python.exe", "-X", "utf8", "-u", "-m", "exocortex", "--no-ui"),
    ("python.exe", "-Xutf8", "-m", "exocortex"),
    ("python.exe", "-mexocortex"),
    ("python.exe", "-uB", "-m", "exocortex"),
    ("py.exe", "-3.12", "-m", "exocortex"),
    ("py", "-V:3.12", "-m", "exocortex"),
    ("python3.12.exe", "-m", "exocortex.__main__"),
    ("PYTHONW.EXE", "-m", "exocortex"),
    ("python.exe", r"D:\Repos\exo-cortex\exocortex\__main__.py"),
    ("python.exe", "-u", "D:/repo/exocortex/__main__.py", "--doctor"),
    ("python.exe", r"D:\Repos\exo-cortex\exocortex"),
    ("python.exe", "--", r"exocortex\__main__.py"),
])
def test_exocortex_command_lines_are_recognised(argv: tuple[str, ...]) -> None:
    assert is_exocortex_argv(argv)


@pytest.mark.parametrize("argv", [
    None,
    (),
    (BASE,),
    (BASE, "-m"),
    (BASE, "-m", "pytest", "-m", "exocortex"),          # pytest's -m, not the interpreter's
    (BASE, "-m", "exocortexx"),
    (BASE, "-m", "exocortex_tools"),
    (BASE, "-c", "import exocortex"),
    (BASE, r"scripts\ui_snapshot.py"),
    (BASE, "-W", "ignore", "-m", "pytest"),
    (BASE, r"D:\x\not_exocortex\__main__.py"),
    (LLAMA, "-m", "exocortex"),                        # llama-server's -m is a model, not a module
    ("cmd.exe", "/c", "python -m exocortex"),
])
def test_other_command_lines_are_not(argv: tuple[str, ...] | None) -> None:
    assert not is_exocortex_argv(argv)


# --------------------------------------------------------------------------- select_targets ---
def test_closes_the_old_session_tree_and_the_stray_model_server_only() -> None:
    assert pids(select_targets(machine(), 402, SESSIONS)) == {100, 101, 102, 103, 500}


def test_never_its_own_ancestors_even_though_they_run_exocortex_too() -> None:
    """The venv's stub (401) runs `-m exocortex --replace` too; closing it could take this process
    down with it (the py launcher's job object), so ancestors are protected."""
    chosen = pids(select_targets(machine(), 402, SESSIONS))
    assert not chosen & {400, 401, 402}


def test_leaves_the_llmstack_other_programs_and_unreadable_processes_alone() -> None:
    chosen = pids(select_targets(machine(), 402, SESSIONS))
    assert not chosen & {4, 200, 201, 300, 700}


def test_descendants_are_limited_to_what_exocortex_itself_starts() -> None:
    """notepad (104), opened by the old session, survives; so does 600, whose parent id only
    points at 101 because Windows reused the number (it is older than 101)."""
    chosen = pids(select_targets(machine(), 402, SESSIONS))
    assert 104 not in chosen and 600 not in chosen
    assert {102, 103} <= chosen


def test_without_a_sessions_dir_the_stray_server_is_not_recognised() -> None:
    chosen = pids(select_targets(machine(), 402, None))
    assert 500 not in chosen
    assert {100, 101, 102, 103} <= chosen          # the old session's own server still goes with it


def test_stray_server_needs_a_log_file_inside_the_sessions_folder() -> None:
    table = machine() + [
        P(501, 999, "llama-server.exe", 8, LLAMA, "--log-file", r"C:\elsewhere\llama-server.log"),
        P(502, 999, "llama-server.exe", 8, LLAMA, f"--log-file={SESSIONS.lower()}\\x\\llama-server.log"),
        P(503, 999, "llama-server.exe", 8, LLAMA, "--log-file", SESSIONS + "-old\\x\\llama-server.log"),
        P(504, 999, "llama-server.exe", 8, LLAMA),
    ]
    chosen = pids(select_targets(table, 402, SESSIONS))
    assert 502 in chosen                       # --log-file=..., case-insensitive
    assert not chosen & {501, 503, 504}        # elsewhere, a sibling folder, no log file


def test_newest_wins_two_launches_cannot_close_each_other() -> None:
    """Two double-clicks: each only closes what started before it, so exactly one survives."""
    table = [
        P(10, 1, "pythonw.exe", 30, BASE, "-m", "exocortex", "--replace"),
        P(11, 1, "pythonw.exe", 31, BASE, "-m", "exocortex", "--replace"),
    ]
    assert pids(select_targets(table, 10, SESSIONS)) == set()      # the older one spares the newer
    assert pids(select_targets(table, 11, SESSIONS)) == {10}       # the newer one closes the older


def test_a_model_server_started_after_this_process_is_never_stray() -> None:
    table = machine() + [P(510, 999, "llama-server.exe", 50, LLAMA, "--log-file", SESSIONS + r"\n\llama-server.log")]
    assert 510 not in pids(select_targets(table, 402, SESSIONS))


def test_nothing_is_closed_when_this_process_is_not_in_the_table() -> None:
    assert select_targets(machine(), 12345, SESSIONS) == []


# ---------------------------------------------------------------------------- ReplaceResult ---
def test_notice_texts() -> None:
    old = [p for p in machine() if p.pid in (100, 101, 102, 103)]
    assert ReplaceResult(closed=old).sessions_closed == 1
    assert ReplaceResult(closed=old).notice == "Closed the previous ExoCortex session"
    two = old + [P(900, 1, "pythonw.exe", 5, BASE, "-m", "exocortex")]
    assert ReplaceResult(closed=two).notice == "Closed 2 previous ExoCortex sessions"
    stray = [p for p in machine() if p.pid == 500]
    assert ReplaceResult(closed=stray).notice == "Closed a stray ExoCortex model server"
    denied = ReplaceResult(failed=[(old[0], "access denied (is it running as administrator?)")])
    assert denied.notice.startswith("Could not close an earlier ExoCortex (pythonw.exe, pid 100)")
    assert "administrator" in denied.notice
    assert ReplaceResult().notice == ""


# ------------------------------------------------------------------------- replace_previous ---
def test_replace_previous_kills_exactly_the_selection_and_reports_it() -> None:
    seen: list[tuple[set[int], float]] = []

    def kill(targets, timeout_s):
        seen.append((pids(list(targets)), timeout_s))
        return list(targets), []

    result = inst.replace_previous(SESSIONS, list_processes=machine, kill=kill, self_pid=402)
    assert seen == [({100, 101, 102, 103, 500}, 10.0)]
    assert result.notice == "Closed the previous ExoCortex session"


def test_replace_previous_with_nothing_to_close_never_calls_kill() -> None:
    def kill(targets, timeout_s):  # pragma: no cover - must not run
        raise AssertionError("kill called")

    table = [p for p in machine() if p.pid not in (100, 101, 102, 103, 500)]
    result = inst.replace_previous(SESSIONS, list_processes=lambda: table, kill=kill, self_pid=402)
    assert result.closed == [] and result.failed == [] and result.notice == ""


def test_replace_previous_never_raises() -> None:
    def broken_listing() -> list[ProcInfo]:
        raise OSError("snapshot failed")

    assert inst.replace_previous(SESSIONS, list_processes=broken_listing, self_pid=402).notice == ""

    def broken_kill(targets, timeout_s):
        raise OSError("boom")

    result = inst.replace_previous(SESSIONS, list_processes=machine, kill=broken_kill, self_pid=402)
    assert pids([p for p, _why in result.failed]) == {100, 101, 102, 103, 500}
    assert result.notice.startswith("Could not close an earlier ExoCortex")


# ------------------------------------------------------------------------ main() wiring ---
@pytest.fixture()
def cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """main() with logging, the takeover and both runners replaced by recorders; a temp config
    whose sessions folder is known (never David's config.toml or his log file)."""
    sessions = tmp_path / "sessions"
    config = tmp_path / "config.toml"
    config.write_text(f'[paths]\nsessions_dir = "{sessions.as_posix()}"\n', encoding="utf-8")
    calls: list[tuple] = []

    def fake_replace(sessions_dir=None, **kw):
        calls.append(("replace", Path(sessions_dir)))
        return ReplaceResult(closed=[P(1, 0, "pythonw.exe", 1, BASE, "-m", "exocortex")])

    monkeypatch.setattr("exocortex.logging_setup.setup_logging", lambda *a, **k: None)
    monkeypatch.setattr(inst, "replace_previous", fake_replace)
    monkeypatch.setattr(app_mod, "_run_ui", lambda cfg, notice="": calls.append(("ui", notice)) or 0)
    monkeypatch.setattr(app_mod, "_run_headless", lambda cfg, duration_s=0.0: calls.append(("headless",)) or 0)
    return config, sessions, calls


def test_main_with_replace_closes_the_old_session_before_starting_the_ui(cli) -> None:
    config, sessions, calls = cli
    assert app_mod.main(["--config", str(config), "--replace"]) == 0
    assert calls == [("replace", sessions), ("ui", "Closed the previous ExoCortex session")]


def test_main_without_replace_leaves_other_sessions_alone(cli) -> None:
    config, _sessions, calls = cli
    assert app_mod.main(["--config", str(config)]) == 0
    assert calls == [("ui", "")]


def test_main_headless_with_replace_prints_the_notice(cli, capsys: pytest.CaptureFixture[str]) -> None:
    config, sessions, calls = cli
    assert app_mod.main(["--config", str(config), "--no-ui", "--replace"]) == 0
    assert calls == [("replace", sessions), ("headless",)]
    assert "[replace] Closed the previous ExoCortex session" in capsys.readouterr().out


def test_launchers_pass_replace() -> None:
    root = Path(__file__).resolve().parent.parent
    bat = (root / "run.bat").read_text(encoding="utf-8")
    assert bat.count("-m exocortex --replace %*") == 2        # pythonw and the python fallback
    vbs = (root / "run.vbs").read_text(encoding="utf-8")
    assert '-m exocortex --replace"' in vbs


# --------------------------------------------------------------- the real Win32 layer ---
win32_only = pytest.mark.skipif(sys.platform != "win32", reason="Win32 process APIs")


def _subtree(rows: list[ProcInfo], root_pid: int) -> set[int]:
    """``root_pid`` and every process below it (parent id + creation order, like the module)."""
    by_parent: dict[int, list[ProcInfo]] = {}
    for p in rows:
        by_parent.setdefault(p.ppid, []).append(p)
    by_pid = {p.pid: p for p in rows}
    out, stack = set(), [by_pid[root_pid]] if root_pid in by_pid else []
    while stack:
        cur = stack.pop()
        out.add(cur.pid)
        for child in by_parent.get(cur.pid, ()):
            if child.pid not in out and (child.created or 0) >= (cur.created or 0):
                stack.append(child)
    return out


def _filetime_now() -> int:
    return int((time.time() + 11_644_473_600) * 10_000_000)


@win32_only
def test_snapshot_reads_this_process_command_line_and_creation_time() -> None:
    rows = inst.snapshot()
    me = {p.pid: p for p in rows}[os.getpid()]
    assert me.created and me.argv
    assert me.argv[0].lower().endswith(".exe") and any("pytest" in a for a in me.argv)


@win32_only
def test_real_takeover_closes_a_dummy_exocortex_tree_and_nothing_else(tmp_path: Path) -> None:
    pkg = tmp_path / "exocortex"
    pkg.mkdir()
    script = pkg / "__main__.py"
    script.write_text("import time\ntime.sleep(60)\n", encoding="utf-8")
    dummy = subprocess.Popen(
        [sys.executable, str(script)], cwd=str(tmp_path),
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    try:
        # The venv's python.exe is a stub that runs the real interpreter as a child: wait for it.
        deadline = time.monotonic() + 10.0
        while True:
            rows = inst.snapshot()
            tree = _subtree(rows, dummy.pid)
            interpreters = [p for p in rows if p.pid in tree and is_exocortex_argv(p.argv)]
            if len(interpreters) >= 2 or time.monotonic() > deadline:
                break
            time.sleep(0.1)
        by_pid = {p.pid: p for p in rows}
        assert is_exocortex_argv(by_pid[dummy.pid].argv)       # the real command line, read and parsed
        # Select as a process started after the dummy would (it is "older"), then act ONLY on the
        # dummy's own tree: a real ExoCortex running on this machine right now is never touched.
        me = ProcInfo(pid=-1, ppid=-1, name="python.exe", created=_filetime_now(), argv=("python.exe", "-m", "pytest"))
        targets = [p for p in select_targets(rows + [me], -1) if p.pid in tree]
        assert pids(targets) == tree
        closed, failed = inst.terminate(targets, 10.0)
        assert failed == [] and pids(closed) == tree
        assert dummy.wait(timeout=5) is not None
        alive = {(p.pid, p.created) for p in inst.snapshot()}
        assert not any((p.pid, p.created) in alive for p in targets)
    finally:
        if dummy.poll() is None:
            dummy.kill()


@win32_only
def test_terminate_counts_an_already_exited_process_as_closed() -> None:
    proc = subprocess.Popen([sys.executable, "-c", "pass"], creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    api = inst._api()
    handle = api.open(proc.pid, inst.PROCESS_QUERY_LIMITED_INFORMATION)
    assert handle
    try:
        created = api.creation_time(handle)
    finally:
        api.close(handle)
    proc.wait(timeout=30)
    # The Popen object still holds its handle, so this pid cannot have been reused yet.
    gone = ProcInfo(pid=proc.pid, ppid=os.getpid(), name="python.exe", created=created)
    closed, failed = inst.terminate([gone], 2.0)
    assert failed == [] and pids(closed) == {proc.pid}

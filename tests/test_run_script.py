"""Tests for run.bat / run.vbs: the seamless-setup contract (as few prerequisites as possible).

Windows-only (batch files, `py`/`where`/PowerShell). Nothing here touches the developer's real
``.venv`` or network: every scenario runs in a ``tmp_path`` copy with ``EXO_SETUP_DRYRUN=1``, which
makes run.bat print every pip/uv/download command instead of running it (see run.bat's header).

Two Python-discovery paths are exercised, matching how :create_venv in run.bat is meant to behave:

* No usable Python anywhere on PATH -> falls through to the uv bootstrap, which (under dry-run)
  only echoes the download/install commands -- this proves the "no prerequisites but LM Studio"
  path is reachable without needing network access in a test.
* A `py` launcher IS on PATH (a tiny fake ``py.bat``) and answers ``-3.12 -m venv <dir>`` by
  creating a working venv-like interpreter: a copy of this test's own ``sys.executable`` next to a
  ``pyvenv.cfg`` pointing ``home`` at the real installation (the real venv module's own trick --
  no third-party install, no site-packages needed since `exocortex.gpu` is stdlib-only). With
  ``PYTHONPATH`` set to the repo root that copied interpreter can then actually run
  ``-m exocortex.gpu --vendor`` / ``--extras``, so the GPU-extras line run.bat prints can be
  checked against a direct invocation of the same command instead of a hard-coded vendor name
  (the vendor/extras CLI contract belongs to a different task; this stays correct however it
  ends up answering on whatever machine runs the test).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="run.bat/run.vbs are Windows-only")

ROOT = Path(__file__).resolve().parent.parent
SYSTEM32 = str(Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32")
MINIMAL_PATH = SYSTEM32 + os.pathsep + str(Path(SYSTEM32) / "WindowsPowerShell" / "v1.0")


def _copy_run_files(dest: Path) -> None:
    for name in ("run.bat", "run.vbs", "requirements.txt", "requirements-nvidia.txt"):
        shutil.copy(ROOT / name, dest / name)
    directml = ROOT / "requirements-directml.txt"
    if directml.is_file():
        shutil.copy(directml, dest / "requirements-directml.txt")


def _run_setup_only(dest: Path, env: dict[str, str], timeout: float = 25.0) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["cmd", "/c", str(dest / "run.bat"), "--setup-only"],
        cwd=str(dest),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def test_requirements_txt_has_no_nvidia_lines() -> None:
    text = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    nvidia_lines = [ln for ln in text.splitlines() if ln.strip().lower().startswith("nvidia-")]
    assert nvidia_lines == [], f"nvidia-* pins must live in requirements-nvidia.txt, found: {nvidia_lines}"


def test_requirements_nvidia_txt_has_exactly_the_two_pins() -> None:
    text = (ROOT / "requirements-nvidia.txt").read_text(encoding="utf-8")
    pins = [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    assert pins == ["nvidia-cublas-cu12==12.9.2.10", "nvidia-cudnn-cu12==9.26.0.51"]


def test_dryrun_with_no_python_falls_back_to_uv_bootstrap(tmp_path: Path) -> None:
    """No `py`, no `python`, nothing installed: run.bat must still exit 0 under --setup-only,
    having only *echoed* the uv download/install/venv commands (EXO_SETUP_DRYRUN=1)."""
    _copy_run_files(tmp_path)

    env = dict(os.environ)
    env["PATH"] = MINIMAL_PATH  # strips every python/py from PATH
    env["EXO_SETUP_DRYRUN"] = "1"
    env.pop("PYTHONPATH", None)

    result = _run_setup_only(tmp_path, env)

    assert result.returncode == 0, result.stdout + result.stderr
    out = result.stdout
    assert "No usable system Python" in out
    assert "uv-x86_64-pc-windows-msvc.zip" in out
    assert "uv.exe python install 3.12" in out
    assert "uv.exe venv .venv --python 3.12 --seed" in out
    # Nothing was actually downloaded or installed, and no real venv was created.
    assert not (tmp_path / ".venv" / "Scripts" / "python.exe").exists()
    assert not (tmp_path / ".tools").exists()


def _write_fake_py_launcher(bin_dir: Path) -> None:
    """A `py.bat` that answers `-3.12 -m venv <dir>` the way the real `py -3.12 -m venv` would as
    far as run.bat can tell: it leaves a `<dir>\\Scripts\\python.exe` behind. It is a copy of THIS
    test's own interpreter with a `pyvenv.cfg` pointing `home` at the real installation, so it
    still runs -- just with no site-packages of its own (fine: exocortex.gpu is stdlib-only)."""
    bin_dir.mkdir(parents=True, exist_ok=True)
    real_python = Path(sys.executable)
    real_home = real_python.parent
    script = textwrap.dedent(f"""\
        @echo off
        if /I "%~1"=="-3.12" if /I "%~2"=="-m" if /I "%~3"=="venv" (
          mkdir "%~4\\Scripts" >nul 2>nul
          copy /y "{real_python}" "%~4\\Scripts\\python.exe" >nul
          > "%~4\\pyvenv.cfg" echo home = {real_home}
          exit /b 0
        )
        exit /b 1
        """)
    (bin_dir / "py.bat").write_text(script, encoding="utf-8")


def _seed_broken_venv(dest: Path, *, home: Path, uv: bool) -> None:
    """A ``.venv`` whose interpreter no longer starts -- what deleting the Python it was built from
    leaves behind (a cleanup tool emptying %APPDATA%\\uv, a python.org uninstall): python.exe is
    still on disk, but every call fails. The stand-in is a `timeout.exe` copy, which exits non-zero
    at once on ``-E -c "import sys"`` just as the uv trampoline / venv launcher do then, and can
    never run anything. ``uv`` adds the ``uv = ...`` line uv writes into the pyvenv.cfg of a venv it
    created, which is what run.bat reads to decide whether its cached uv.exe can repair it."""
    scripts = dest / ".venv" / "Scripts"
    scripts.mkdir(parents=True, exist_ok=True)
    timeout_exe = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "timeout.exe"
    shutil.copy(timeout_exe, scripts / "python.exe")
    shutil.copy(timeout_exe, scripts / "pythonw.exe")
    cfg = f"home = {home}\n" + ("uv = 0.12.23\n" if uv else "")
    (dest / ".venv" / "pyvenv.cfg").write_text(cfg, encoding="utf-8")


def test_dryrun_with_fake_py_creates_venv_installs_extras_and_writes_stamp(tmp_path: Path) -> None:
    _copy_run_files(tmp_path)
    bin_dir = tmp_path / "fakebin"
    _write_fake_py_launcher(bin_dir)

    env = dict(os.environ)
    env["PATH"] = str(bin_dir) + os.pathsep + MINIMAL_PATH
    env["EXO_SETUP_DRYRUN"] = "1"
    env["PYTHONPATH"] = str(ROOT)  # so the copied interpreter can `import exocortex`

    result = _run_setup_only(tmp_path, env)
    assert result.returncode == 0, result.stdout + result.stderr

    venv_python = tmp_path / ".venv" / "Scripts" / "python.exe"
    assert venv_python.is_file(), "the fake py.bat should have produced a venv-like interpreter"

    stamp = tmp_path / ".venv" / "exocortex-setup.stamp"
    assert stamp.is_file(), "a successful setup must write the stamp file"

    # pip itself was never actually invoked (dry run).
    assert "[dryrun] .venv\\Scripts\\python.exe -m pip install -r requirements.txt" in result.stdout
    # 2026-09-27 (portable-install): the first-run step (llama.cpp runtime, speech models, the
    # hardware report, the "pull a model" prompt) runs after the pip step, echoed under dry-run,
    # with no --quiet flag unless EXO_SETUP_QUIET=1 (run.vbs's hidden catch-up runs set it).
    assert "[dryrun] .venv\\Scripts\\python.exe -m exocortex.setup\n" in result.stdout.replace("\r\n", "\n")
    assert "--quiet" not in result.stdout

    # The GPU-extras line run.bat printed must match a direct invocation of the same command
    # (whatever `exocortex.gpu --vendor`/`--extras` currently answer on this machine).
    probe_env = dict(env)
    vendor = subprocess.run(
        [str(venv_python), "-m", "exocortex.gpu", "--vendor"],
        cwd=str(tmp_path), env=probe_env, capture_output=True, text=True, timeout=15,
    ).stdout.strip()
    extras = subprocess.run(
        [str(venv_python), "-m", "exocortex.gpu", "--extras"],
        cwd=str(tmp_path), env=probe_env, capture_output=True, text=True, timeout=15,
    ).stdout.strip()
    expected_extras = extras if extras else "none"
    assert f"GPU: {vendor} -> extras: {expected_extras}" in result.stdout, result.stdout

    # A second run with nothing changed must be fast and skip the install -- but the first-run
    # step still runs (it is idempotent and is what notices a model pulled or deleted by hand).
    result2 = _run_setup_only(tmp_path, env)
    assert result2.returncode == 0, result2.stdout + result2.stderr
    assert "Setup is up to date." in result2.stdout
    assert "Installing dependencies" not in result2.stdout
    assert "[dryrun] .venv\\Scripts\\python.exe -m exocortex.setup" in result2.stdout

    # EXO_SETUP_QUIET=1 (what run.vbs sets for a hidden run) turns into --quiet for the step.
    quiet_env = dict(env)
    quiet_env["EXO_SETUP_QUIET"] = "1"
    result3 = _run_setup_only(tmp_path, quiet_env)
    assert result3.returncode == 0, result3.stdout + result3.stderr
    assert "[dryrun] .venv\\Scripts\\python.exe -m exocortex.setup --quiet" in result3.stdout


def test_dryrun_no_python_reuses_cached_uv_exe_skips_download(tmp_path: Path) -> None:
    """A machine with no system Python that already has a working ``.tools\\uv\\uv.exe`` from a
    prior bootstrap (e.g. after ``.venv`` was deleted to force a refresh) must reuse it instead of
    re-downloading uv.zip from GitHub every time -- otherwise a later offline/proxy-blocked run
    fails outright even though a perfectly good uv.exe is sitting on disk."""
    _copy_run_files(tmp_path)
    uv_dir = tmp_path / ".tools" / "uv"
    uv_dir.mkdir(parents=True)
    cached_uv = uv_dir / "uv.exe"
    cached_uv.write_bytes(b"not a real uv.exe, just needs to already exist")

    env = dict(os.environ)
    env["PATH"] = MINIMAL_PATH  # strips every python/py from PATH
    env["EXO_SETUP_DRYRUN"] = "1"
    env.pop("PYTHONPATH", None)

    result = _run_setup_only(tmp_path, env)

    assert result.returncode == 0, result.stdout + result.stderr
    out = result.stdout
    assert "No usable system Python" in out
    # The download/extract steps must be skipped because the cached uv.exe was reused...
    assert "uv-x86_64-pc-windows-msvc.zip" not in out
    assert "Expand-Archive" not in out
    # ...but the rest of the bootstrap (installing Python, creating the venv) still runs.
    assert "uv.exe python install 3.12" in out
    assert "uv.exe venv .venv --python 3.12 --seed" in out
    # The cached uv.exe itself must be left untouched, not re-downloaded over.
    assert cached_uv.read_bytes() == b"not a real uv.exe, just needs to already exist"
    assert not (tmp_path / ".venv" / "Scripts" / "python.exe").exists()


def test_dryrun_vendor_mismatch_forces_reinstall_even_when_stamp_is_fresh(tmp_path: Path) -> None:
    """:ensure_deps must not rely on file timestamps alone: if the stamp's recorded GPU vendor no
    longer matches what ``exocortex.gpu --vendor`` currently reports, it must reinstall the GPU
    extras even though the stamp is newer than every watched file (e.g. a driver update, a
    hybrid-graphics/Optimus switch, or a dock/monitor change flipped the detected primary GPU)."""
    _copy_run_files(tmp_path)
    bin_dir = tmp_path / "fakebin"
    _write_fake_py_launcher(bin_dir)

    env = dict(os.environ)
    env["PATH"] = str(bin_dir) + os.pathsep + MINIMAL_PATH
    env["EXO_SETUP_DRYRUN"] = "1"
    env["PYTHONPATH"] = str(ROOT)

    result = _run_setup_only(tmp_path, env)
    assert result.returncode == 0, result.stdout + result.stderr

    venv_python = tmp_path / ".venv" / "Scripts" / "python.exe"
    stamp = tmp_path / ".venv" / "exocortex-setup.stamp"
    assert stamp.is_file()

    actual_vendor = subprocess.run(
        [str(venv_python), "-m", "exocortex.gpu", "--vendor"],
        cwd=str(tmp_path), env=dict(env), capture_output=True, text=True, timeout=15,
    ).stdout.strip()

    # Rewrite the stamp to claim a different vendor than what's actually detected, but keep its
    # timestamp the newest of anything watched (so only the vendor mismatch, not a stale mtime,
    # can be what triggers the reinstall below).
    bogus_vendor = "nvidia" if actual_vendor != "nvidia" else "amd"
    stamp.write_text(f"{bogus_vendor} bogus-date\n", encoding="utf-8")
    now = time.time() + 5
    os.utime(stamp, (now, now))
    for watched in ("requirements.txt", "requirements-nvidia.txt", "requirements-directml.txt", "run.bat"):
        path = tmp_path / watched
        if path.is_file():
            os.utime(path, (now - 100, now - 100))

    result2 = _run_setup_only(tmp_path, env)
    assert result2.returncode == 0, result2.stdout + result2.stderr
    assert "Setup is up to date." not in result2.stdout
    assert "Installing dependencies" in result2.stdout
    assert f"GPU: {actual_vendor} ->" in result2.stdout, result2.stdout

    # The stamp must now record the real vendor again.
    assert stamp.read_text(encoding="utf-8").split()[0] == actual_vendor


def test_dryrun_broken_uv_venv_reinstalls_python_then_rebuilds(tmp_path: Path) -> None:
    """The .venv's python.exe is on disk but no longer starts: the uv-managed Python it was built on
    is gone. run.bat used to check only that python.exe exists, so every launch failed at its first
    pip call, behind a console that closed at once. It must try to put that Python back with the
    cached uv.exe first and -- when the interpreter still does not start, as always under a dry
    run -- rebuild .venv, then carry on with the normal setup."""
    _copy_run_files(tmp_path)
    bin_dir = tmp_path / "fakebin"
    _write_fake_py_launcher(bin_dir)
    uv_dir = tmp_path / ".tools" / "uv"
    uv_dir.mkdir(parents=True)
    (uv_dir / "uv.exe").write_bytes(b"not a real uv.exe, just needs to already exist")
    _seed_broken_venv(tmp_path, home=tmp_path / "deleted-python", uv=True)

    env = dict(os.environ)
    env["PATH"] = str(bin_dir) + os.pathsep + MINIMAL_PATH
    env["EXO_SETUP_DRYRUN"] = "1"
    env["PYTHONPATH"] = str(ROOT)

    result = _run_setup_only(tmp_path, env)
    assert result.returncode == 0, result.stdout + result.stderr
    out = result.stdout
    assert "Python no longer starts" in out
    assert "[dryrun] .tools\\uv\\uv.exe python install 3.12" in out
    assert "rebuilding the virtual environment" in out
    assert "Creating virtual environment" in out

    # The broken venv was replaced by a working one (the fake py launcher's), and setup went on.
    probe = subprocess.run(
        [str(tmp_path / ".venv" / "Scripts" / "python.exe"), "-E", "-c", "import sys"],
        capture_output=True, timeout=15,
    )
    assert probe.returncode == 0
    assert (tmp_path / ".venv" / "exocortex-setup.stamp").is_file()


def test_dryrun_broken_venv_whose_base_python_still_exists_is_left_alone(tmp_path: Path) -> None:
    """python.exe does not start although the Python its pyvenv.cfg names is still installed: the
    cause is something else (an antivirus block), and a rebuild would only throw away a multi-GB
    venv to hit the same wall. run.bat must stop with an error and leave .venv as it is."""
    _copy_run_files(tmp_path)
    base = tmp_path / "installed-python"
    base.mkdir()
    (base / "python.exe").write_bytes(b"")
    _seed_broken_venv(tmp_path, home=base, uv=False)

    env = dict(os.environ)
    env["PATH"] = MINIMAL_PATH
    env["EXO_SETUP_DRYRUN"] = "1"
    env.pop("PYTHONPATH", None)

    result = _run_setup_only(tmp_path, env)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "still installed" in result.stdout
    assert (tmp_path / ".venv" / "Scripts" / "python.exe").is_file(), "the venv must be left alone"
    assert (tmp_path / ".venv" / "pyvenv.cfg").is_file()


def test_dryrun_quiet_run_keeps_a_broken_venv_instead_of_rebuilding(tmp_path: Path) -> None:
    """A hidden run (EXO_SETUP_QUIET=1, what run.vbs sets) tries the non-destructive uv repair and
    nothing more: a rebuild that then failed (offline) would leave no .venv at all, and run.vbs
    takes a missing .venv for a fresh install -- the one case it shows a console for, to a meeting
    on a shared screen. The broken venv must still be there afterwards (no app, no window)."""
    _copy_run_files(tmp_path)
    bin_dir = tmp_path / "fakebin"
    _write_fake_py_launcher(bin_dir)
    uv_dir = tmp_path / ".tools" / "uv"
    uv_dir.mkdir(parents=True)
    (uv_dir / "uv.exe").write_bytes(b"not a real uv.exe, just needs to already exist")
    _seed_broken_venv(tmp_path, home=tmp_path / "deleted-python", uv=True)

    env = dict(os.environ)
    env["PATH"] = str(bin_dir) + os.pathsep + MINIMAL_PATH
    env["EXO_SETUP_DRYRUN"] = "1"
    env["EXO_SETUP_QUIET"] = "1"
    env["PYTHONPATH"] = str(ROOT)

    result = _run_setup_only(tmp_path, env)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "[dryrun] .tools\\uv\\uv.exe python install 3.12" in result.stdout
    assert "rebuilding" not in result.stdout
    assert "Creating virtual environment" not in result.stdout
    assert (tmp_path / ".venv" / "pyvenv.cfg").read_text(encoding="utf-8").startswith(
        f"home = {tmp_path / 'deleted-python'}"), "the broken venv must be kept, not replaced"


def test_failed_setup_never_waits_for_a_keypress(tmp_path: Path) -> None:
    """A failed setup exits at once, like every other run.bat outcome -- it must never `pause` to
    keep its console up: consoles are not excluded from screen capture, and one left waiting
    mid-meeting is seen by everyone watching the share. stdin stays OPEN here, so a `pause` would
    block until the timeout rather than return on EOF. Not a dry run (the failure path must be
    the real one), but nothing here can reach the network or start the app: setup stops at the
    broken-venv check first, and the dummy uv.exe stands in the way of a real download should that
    check ever regress."""
    _copy_run_files(tmp_path)
    base = tmp_path / "installed-python"
    base.mkdir()
    (base / "python.exe").write_bytes(b"")
    _seed_broken_venv(tmp_path, home=base, uv=False)
    uv_dir = tmp_path / ".tools" / "uv"
    uv_dir.mkdir(parents=True)
    (uv_dir / "uv.exe").write_bytes(b"not a real uv.exe, just needs to already exist")

    env = dict(os.environ)
    env["PATH"] = MINIMAL_PATH
    env.pop("EXO_SETUP_DRYRUN", None)
    env.pop("EXO_SETUP_QUIET", None)
    env.pop("PYTHONPATH", None)

    proc = subprocess.Popen(
        ["cmd", "/c", str(tmp_path / "run.bat")],
        cwd=str(tmp_path), env=env, stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    timed_out = False
    try:
        rc = proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        timed_out = True
        proc.kill()
        rc = proc.wait()
    proc.stdin.close()
    out = proc.stdout.read()
    proc.stdout.close()
    assert not timed_out, "run.bat sat waiting for a keypress after a failed setup:\n" + out
    assert rc == 1, out
    assert "still installed" in out


# --- run.vbs: does it actually decide correctly whether to call run.bat, and how visibly? ------
#
# run.vbs itself never runs real setup or launches the real overlay in these tests: the run.bat it
# calls is a stand-in ("recorder") that never touches pip/network, and the "pythonw.exe"/
# "python.exe" it (and the pre-seeded "existing venv" scenarios) leave behind is a copy of the
# always-present system `timeout.exe`, never a real Python interpreter -- so even the final
# `pyw -m exocortex --replace` launch step run.vbs always performs cannot start anything resembling
# ExoCortex or its overlay. The recorder also asks PowerShell whether ITS OWN console window is
# currently visible (GetConsoleWindow + IsWindowVisible), which is exactly the signal
# `shell.Run ..., <style>, True` controls -- so these tests observe the real window-style decision
# run.vbs makes, not just whether run.bat was called.

_CONSOLE_VISIBLE_PS1 = textwrap.dedent(r"""
    Add-Type -Name Win32 -Namespace ExoTest -MemberDefinition @'
    [DllImport("user32.dll")] public static extern bool IsWindowVisible(System.IntPtr hWnd);
    [DllImport("kernel32.dll")] public static extern System.IntPtr GetConsoleWindow();
    '@
    $h = [ExoTest.Win32]::GetConsoleWindow()
    [ExoTest.Win32]::IsWindowVisible($h)
    """)


def _write_fake_bat_recorder(dest: Path, *, starting_python: bool = False) -> None:
    """Overwrites ``run.bat`` in ``dest`` (already containing the real ``run.vbs``, copied
    separately) with a stand-in that: records it was called (``calls.log``), records whether its
    own console window was visible (``visible_state.txt``), and then leaves behind a working
    ``.venv\\Scripts\\pythonw.exe``/``python.exe`` (timeout.exe copies) and a fresh stamp, exactly
    as a real ``run.bat --setup-only`` would on success -- so run.vbs's post-setup launch step has
    something harmless to invoke. ``starting_python``: python.exe is instead the interpreter stand-in
    `_seed_existing_venv` uses, one that passes run.vbs's "does it start" check -- what a repaired
    venv looks like; without it, a broken venv's python.exe stays as broken as before."""
    (dest / "check_console_visible.ps1").write_text(_CONSOLE_VISIBLE_PS1, encoding="utf-8")
    if starting_python:
        real_python = Path(sys.executable)
        python_step = (f'copy /y "{real_python}" ".venv\\Scripts\\python.exe" >nul\n'
                       f'> ".venv\\pyvenv.cfg" echo home = {real_python.parent}\n')
    else:
        python_step = 'copy /y "%WINDIR%\\System32\\timeout.exe" ".venv\\Scripts\\python.exe" >nul\n'
    (dest / "run.bat").write_text(
        textwrap.dedent(r"""
            @echo off
            setlocal
            cd /d "%~dp0"
            >> "%~dp0calls.log" echo CALLED
            >> "%~dp0calls.log" echo QUIET=%EXO_SETUP_QUIET%
            powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0check_console_visible.ps1" > "%~dp0visible_state.txt"
            if not exist ".venv\Scripts" mkdir ".venv\Scripts" >nul 2>nul
            copy /y "%WINDIR%\System32\timeout.exe" ".venv\Scripts\pythonw.exe" >nul
            """) + python_step + textwrap.dedent(r"""
            > ".venv\exocortex-setup.stamp" echo fake-vendor stamped-by-test
            exit /b 0
            """),
        encoding="utf-8",
    )


def _seed_existing_venv(dest: Path) -> None:
    """Simulates an existing, already-working ``.venv`` without running the recorder -- for
    scenarios that must NOT need (or must need, but silently) setup. ``pythonw.exe`` is a
    `timeout.exe` copy (harmless: run.vbs's final launch step invokes it, and it must never be
    able to import -- let alone run -- the real `exocortex` package). ``python.exe`` is instead a
    copy of THIS test's own interpreter with a `pyvenv.cfg` pointing `home` at the real
    installation (the real venv module's own trick), because run.vbs's vendor recheck
    (``exocortex.gpu --vendor``, stdlib-only, side-effect-free) needs a real answer -- it is never
    invoked with `-m exocortex --replace`, so it can never start the overlay."""
    scripts = dest / ".venv" / "Scripts"
    scripts.mkdir(parents=True, exist_ok=True)
    timeout_exe = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "timeout.exe"
    shutil.copy(timeout_exe, scripts / "pythonw.exe")
    real_python = Path(sys.executable)
    shutil.copy(real_python, scripts / "python.exe")
    (dest / ".venv" / "pyvenv.cfg").write_text(f"home = {real_python.parent}\n", encoding="utf-8")


def _run_vbs(dest: Path, timeout: float = 20.0) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT)  # so a real python.exe stand-in can `import exocortex`
    return subprocess.run(
        ["cscript", "//nologo", str(dest / "run.vbs")],
        cwd=str(dest),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def test_vbs_fresh_install_runs_setup_visibly(tmp_path: Path) -> None:
    """No ``.venv`` at all yet: run.vbs must call run.bat --setup-only in a VISIBLE window (the
    one-time install the user is meant to watch)."""
    _copy_run_files(tmp_path)
    _write_fake_bat_recorder(tmp_path)

    result = _run_vbs(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr

    assert (tmp_path / "calls.log").is_file(), "run.bat should have been called"
    visible = (tmp_path / "visible_state.txt").read_text(encoding="utf-8").strip()
    assert visible == "True", "a genuinely fresh install must show a visible setup window"
    # Visible = someone can answer the "pull a model" prompt: run.vbs must NOT ask for --quiet.
    assert "QUIET=\n" in (tmp_path / "calls.log").read_text(encoding="utf-8").replace("\r\n", "\n")


def test_vbs_existing_venv_missing_stamp_runs_setup_hidden(tmp_path: Path) -> None:
    """An existing, already-working .venv (pythonw.exe present) from before the stamp file
    existed -- every pre-upgrade install, NVIDIA included -- must still run the real setup (so the
    stamp gets migrated and any real dependency change lands), but WITHOUT flashing a console."""
    _copy_run_files(tmp_path)
    _write_fake_bat_recorder(tmp_path)
    _seed_existing_venv(tmp_path)
    assert not (tmp_path / ".venv" / "exocortex-setup.stamp").exists()

    result = _run_vbs(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr

    assert (tmp_path / "calls.log").is_file(), "run.bat should still have been called (migration)"
    visible = (tmp_path / "visible_state.txt").read_text(encoding="utf-8").strip()
    assert visible == "False", "an existing install must never show a console just to migrate"
    # Hidden = nobody can answer a prompt: run.vbs must hand run.bat EXO_SETUP_QUIET=1.
    assert "QUIET=1" in (tmp_path / "calls.log").read_text(encoding="utf-8")


def test_vbs_stale_requirements_runs_setup_hidden(tmp_path: Path) -> None:
    """An existing, already-working .venv whose stamp predates a requirements.txt change: run.vbs
    itself (not just run.bat) must detect the staleness via file timestamps and re-run setup, but
    -- like the migration case -- without showing a console, since the install already works."""
    _copy_run_files(tmp_path)
    _write_fake_bat_recorder(tmp_path)
    _seed_existing_venv(tmp_path)

    now = time.time()
    stamp = tmp_path / ".venv" / "exocortex-setup.stamp"
    stamp.write_text("amd old-date\n", encoding="utf-8")
    os.utime(stamp, (now - 100, now - 100))
    os.utime(tmp_path / "requirements.txt", (now, now))  # newer than the stamp -> stale

    result = _run_vbs(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr

    assert (tmp_path / "calls.log").is_file(), "a stale requirements.txt must trigger setup"
    visible = (tmp_path / "visible_state.txt").read_text(encoding="utf-8").strip()
    assert visible == "False", "a stale-but-existing install must not show a console either"


def test_vbs_up_to_date_never_calls_run_bat(tmp_path: Path) -> None:
    """Nothing changed since the stamp was written -- including the detected GPU vendor, which
    run.vbs itself now rechecks (mirroring run.bat's own :ensure_deps) -- so run.vbs must not
    invoke run.bat at all (the fully silent, common-case path)."""
    _copy_run_files(tmp_path)
    _write_fake_bat_recorder(tmp_path)
    _seed_existing_venv(tmp_path)

    venv_python = tmp_path / ".venv" / "Scripts" / "python.exe"
    probe_env = dict(os.environ)
    probe_env["PYTHONPATH"] = str(ROOT)
    actual_vendor = subprocess.run(
        [str(venv_python), "-m", "exocortex.gpu", "--vendor"],
        cwd=str(tmp_path), env=probe_env, capture_output=True, text=True, timeout=15,
    ).stdout.strip()
    assert actual_vendor, "the real venv_python stand-in must be able to answer --vendor"

    now = time.time()
    for watched in ("requirements.txt", "requirements-nvidia.txt", "requirements-directml.txt", "run.bat"):
        path = tmp_path / watched
        if path.is_file():
            os.utime(path, (now - 100, now - 100))
    stamp = tmp_path / ".venv" / "exocortex-setup.stamp"
    stamp.write_text(f"{actual_vendor} up-to-date\n", encoding="utf-8")
    os.utime(stamp, (now, now))  # newer than every watched file, and vendor matches

    result = _run_vbs(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert not (tmp_path / "calls.log").exists(), "run.bat must not be called when nothing is stale"


def test_vbs_vendor_mismatch_runs_setup_hidden(tmp_path: Path) -> None:
    """No watched file changed, but the stamp's recorded vendor no longer matches what
    ``exocortex.gpu --vendor`` currently reports: run.vbs must detect this itself (mirroring
    run.bat's own :ensure_deps fix) and re-run setup -- hidden, like every other case where a
    working install is just catching up."""
    _copy_run_files(tmp_path)
    _write_fake_bat_recorder(tmp_path)
    _seed_existing_venv(tmp_path)

    venv_python = tmp_path / ".venv" / "Scripts" / "python.exe"
    probe_env = dict(os.environ)
    probe_env["PYTHONPATH"] = str(ROOT)
    actual_vendor = subprocess.run(
        [str(venv_python), "-m", "exocortex.gpu", "--vendor"],
        cwd=str(tmp_path), env=probe_env, capture_output=True, text=True, timeout=15,
    ).stdout.strip()

    now = time.time()
    for watched in ("requirements.txt", "requirements-nvidia.txt", "requirements-directml.txt", "run.bat"):
        path = tmp_path / watched
        if path.is_file():
            os.utime(path, (now - 100, now - 100))
    stamp = tmp_path / ".venv" / "exocortex-setup.stamp"
    bogus_vendor = "nvidia" if actual_vendor != "nvidia" else "amd"
    stamp.write_text(f"{bogus_vendor} bogus-date\n", encoding="utf-8")
    os.utime(stamp, (now, now))  # newer than every watched file -- only the vendor differs

    result = _run_vbs(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr

    assert (tmp_path / "calls.log").is_file(), "a vendor mismatch alone must still trigger setup"
    visible = (tmp_path / "visible_state.txt").read_text(encoding="utf-8").strip()
    assert visible == "False", "an existing install's vendor-drift setup must stay hidden too"


def test_vbs_broken_interpreter_runs_setup_hidden(tmp_path: Path) -> None:
    """pythonw.exe is on disk but the venv's Python no longer starts (the interpreter it was built
    from was deleted). run.vbs used to take that for a working install and launch it -- to no
    effect, on every double-click. It must run the repair, HIDDEN like every other catch-up of an
    existing install (consoles are not excluded from screen capture), with EXO_SETUP_QUIET=1 --
    then launch, with no second run.bat call, once the interpreter starts again."""
    _copy_run_files(tmp_path)
    _write_fake_bat_recorder(tmp_path, starting_python=True)
    _seed_broken_venv(tmp_path, home=tmp_path / "deleted-python", uv=True)
    now = time.time()
    stamp = tmp_path / ".venv" / "exocortex-setup.stamp"
    stamp.write_text("nvidia up-to-date\n", encoding="utf-8")
    os.utime(stamp, (now, now))  # nothing stale: only the broken interpreter can trigger setup

    result = _run_vbs(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr

    calls = (tmp_path / "calls.log").read_text(encoding="utf-8").replace("\r\n", "\n")
    assert calls.count("CALLED") == 1, calls
    visible = (tmp_path / "visible_state.txt").read_text(encoding="utf-8").strip()
    assert visible == "False", "repairing an existing install must never show a console"
    assert "QUIET=1" in calls, "a hidden setup must not prompt -- or rebuild (see run.bat)"


def test_vbs_broken_interpreter_still_broken_after_setup_stays_silent(tmp_path: Path) -> None:
    """The hidden repair did not take: python.exe still does not start after run.bat --setup-only.
    Launching pythonw.exe would do nothing, and an existing install never gets a console -- not
    even a run.bat window to explain the failure. So: one hidden run.bat call, then nothing. A
    console fallback would be fire-and-forget, so give one time to show up before checking."""
    _copy_run_files(tmp_path)
    _write_fake_bat_recorder(tmp_path)  # leaves python.exe as broken as it found it
    _seed_broken_venv(tmp_path, home=tmp_path / "deleted-python", uv=True)

    result = _run_vbs(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr

    log = tmp_path / "calls.log"
    deadline = time.time() + 3
    while time.time() < deadline and log.read_text(encoding="utf-8").count("CALLED") < 2:
        time.sleep(0.2)
    calls = log.read_text(encoding="utf-8").replace("\r\n", "\n")
    assert calls.count("CALLED") == 1, calls
    assert "QUIET=1" in calls
    visible = (tmp_path / "visible_state.txt").read_text(encoding="utf-8").strip()
    assert visible == "False", "a failed repair must not show a console either"

@echo off
setlocal
cd /d "%~dp0"

:: ExoCortex launcher -----------------------------------------------------------------------
:: Goal: NO prerequisites. One run on a fresh machine installs everything: Python itself is
:: optional (see :create_venv), the pip requirements and this GPU's extras (:ensure_deps), and
:: then -- 2026-09-27 -- everything the app used to expect from LM Studio or Ollama (:first_run,
:: `python -m exocortex.setup`): the llama.cpp runtime for this GPU (LM Studio's pack when one is
:: installed, else the official llama.cpp build), the speech models, a hardware report, and a
:: prompt to pull at least one LLM into the app's own library when it holds none yet.
::
:: Usage:
::   run.bat               creates/refreshes .venv, runs the first-run setup, starts the overlay
::   run.bat --setup-only  creates/refreshes .venv + first-run setup and exits (run.vbs uses it)
::   run.bat --doctor      runs `python -m exocortex --doctor` in a visible console, then pauses
::
:: EXO_SETUP_DRYRUN=1 makes every pip/uv/download/setup step print the command instead of running
:: it (used by tests\test_run_script.py so they do not need network access or take minutes).
:: EXO_SETUP_QUIET=1 (run.vbs sets it for its hidden catch-up runs) skips the interactive
:: "pull a model" prompt; the app then offers its Models window instead.
:: -----------------------------------------------------------------------------------------

if /I "%~1"=="--doctor" goto :run_doctor

call :ensure_setup
if errorlevel 1 exit /b 1

if /I "%~1"=="--setup-only" goto :eof

:: Launch windowless (pythonw = no console) and detached, then exit so this console
:: window closes immediately and leaves nothing in the taskbar. For a visible console
:: (logs/--no-ui/--doctor) call python.exe directly, e.g.:
::     .venv\Scripts\python.exe -m exocortex --doctor
:: --replace: force-close any ExoCortex that is already running (and its model server)
:: first, so this launch is always the only session.
if exist ".venv\Scripts\pythonw.exe" (
  start "" "%~dp0.venv\Scripts\pythonw.exe" -m exocortex --replace %*
) else (
  start "" "%~dp0.venv\Scripts\python.exe" -m exocortex --replace %*
)
goto :eof

:run_doctor
call :ensure_setup
if errorlevel 1 exit /b 1
".venv\Scripts\python.exe" -m exocortex --doctor
pause
exit /b 0

:: =============================================================================================
:: :ensure_setup -- make sure .venv exists and its dependencies are current. Safe to call every
:: run: a "stamp" file records when it last succeeded, so a machine that is already set up pays
:: only the cost of a python -m exocortex.gpu call (a few ms), not a pip resolve.
:: =============================================================================================
:ensure_setup
call :create_venv
if errorlevel 1 exit /b 1

if not exist ".venv\Scripts\python.exe" (
  if "%EXO_SETUP_DRYRUN%"=="1" (
    echo [ExoCortex] [dryrun] no virtual environment available yet ^(the uv bootstrap above was
    echo [ExoCortex] [dryrun] simulated^); skipping the dependency install.
    exit /b 0
  )
  echo [ExoCortex] ERROR: could not set up a Python virtual environment.
  exit /b 1
)

call :ensure_deps
if errorlevel 1 exit /b 1

call :first_run
exit /b 0

:: ---------------------------------------------------------------------------------------------
:: :first_run -- `python -m exocortex.setup`: the llama.cpp runtime for this GPU (fetched only
:: when no LM Studio pack covers it), the speech models, the hardware report the Models window
:: and the auto-tuned model profiles are built from, and -- when the library holds no LLM yet --
:: the prompt to pull one. Idempotent and quick (~1 s) once everything is present, so it runs on
:: EVERY launch after :ensure_deps, not just when the pip stamp is stale: a model pulled or
:: deleted by hand must be noticed without touching requirements.txt. A problem here (offline,
:: a refused download) is reported and never blocks the launch -- the overlay still transcribes,
:: and the app retries the runtime fetch itself at start. Placed right after :ensure_setup for
:: the same forward-CALL reason as :bootstrap_uv_python below.
:: ---------------------------------------------------------------------------------------------
:first_run
set "SETUP_FLAGS="
if "%EXO_SETUP_QUIET%"=="1" set "SETUP_FLAGS=--quiet"
if "%EXO_SETUP_DRYRUN%"=="1" goto :first_run_dryrun
echo [ExoCortex] Checking the llama.cpp runtime, speech models and model library...
".venv\Scripts\python.exe" -m exocortex.setup %SETUP_FLAGS%
if errorlevel 1 echo [ExoCortex] WARNING: the first-run setup reported a problem ^(see above^); starting anyway.
exit /b 0
:first_run_dryrun
if "%SETUP_FLAGS%"=="" echo [dryrun] .venv\Scripts\python.exe -m exocortex.setup
if not "%SETUP_FLAGS%"=="" echo [dryrun] .venv\Scripts\python.exe -m exocortex.setup %SETUP_FLAGS%
exit /b 0

:: ---------------------------------------------------------------------------------------------
:: :bootstrap_uv_python -- download uv (a single self-contained .exe, astral.sh) into .tools\uv\,
:: then use it to install Python 3.12 and create .venv with pip seeded into it ("--seed"), so
:: every step after this one is unchanged whether the interpreter came from "py" or from uv.
:: Defined here, right after :ensure_setup and BEFORE :create_venv (which calls it): cmd.exe's
:: parser has been observed to fail a forward CALL ("cannot find the batch label specified")
:: when the target label sits far enough into a long file with several nested if-blocks ahead
:: of the call site -- reproducible on this Windows 11 build regardless of the label's own
:: content. Keeping frequently-forward-called subroutines early avoids it; see
:: tests\test_run_script.py for the regression coverage.
:: ---------------------------------------------------------------------------------------------
:bootstrap_uv_python
if not "%EXO_SETUP_DRYRUN%"=="1" goto :bootstrap_real

if exist ".tools\uv\uv.exe" goto :bootstrap_dryrun_uv_ready
echo [dryrun] Invoke-WebRequest -Uri https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip -OutFile .tools\uv\uv.zip
echo [dryrun] Expand-Archive -Path .tools\uv\uv.zip -DestinationPath .tools\uv -Force
:bootstrap_dryrun_uv_ready
echo [dryrun] .tools\uv\uv.exe python install 3.12
echo [dryrun] .tools\uv\uv.exe venv .venv --python 3.12 --seed
exit /b 0

:bootstrap_real
:: Reuse a cached .tools\uv\uv.exe from a prior bootstrap instead of always re-downloading it --
:: this is the only thing standing between a working offline re-setup (e.g. after deleting .venv)
:: and a hard "no internet?" failure on a machine that already has a perfectly good uv.exe.
if exist ".tools\uv\uv.exe" goto :uv_ready

if not exist ".tools\uv" mkdir ".tools\uv" >nul 2>nul
echo [ExoCortex] Downloading uv...
powershell -NoProfile -ExecutionPolicy Bypass -Command "Invoke-WebRequest -Uri 'https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip' -OutFile '.tools\uv\uv.zip'"
if errorlevel 1 (
  echo [ExoCortex] ERROR: could not download uv ^(no internet?^).
  echo [ExoCortex] Install Python 3.12 from python.org, or let this script download it ^(needs internet^).
  exit /b 1
)

echo [ExoCortex] Extracting uv...
powershell -NoProfile -ExecutionPolicy Bypass -Command "Expand-Archive -Path '.tools\uv\uv.zip' -DestinationPath '.tools\uv' -Force"

:uv_ready
if not exist ".tools\uv\uv.exe" (
  for /r ".tools\uv" %%f in (uv.exe) do copy /y "%%f" ".tools\uv\uv.exe" >nul 2>nul
)
if not exist ".tools\uv\uv.exe" (
  echo [ExoCortex] ERROR: uv.exe not found after extracting the download.
  echo [ExoCortex] Install Python 3.12 from python.org, or let this script download it ^(needs internet^).
  exit /b 1
)

echo [ExoCortex] Installing Python 3.12 via uv...
".tools\uv\uv.exe" python install 3.12
if errorlevel 1 (
  echo [ExoCortex] ERROR: uv could not install Python 3.12.
  echo [ExoCortex] Install Python 3.12 from python.org, or let this script download it ^(needs internet^).
  exit /b 1
)

echo [ExoCortex] Creating virtual environment via uv...
".tools\uv\uv.exe" venv .venv --python 3.12 --seed
if errorlevel 1 (
  echo [ExoCortex] ERROR: uv could not create the virtual environment.
  exit /b 1
)
exit /b 0

:: ---------------------------------------------------------------------------------------------
:: :create_venv -- a no-op when .venv\Scripts\python.exe already exists (created once, never
:: recreated). Otherwise: prefer the newest system Python (py -3.12, then 3.13, then 3.11 --
:: 3.12 first because that is what this app is verified on), then a bare "python" if its version
:: is in range, and only if none of those exist, bootstrap a private Python with uv (:above).
:: Flat sequential checks with GOTO, no nested "if ( ... )" blocks, on purpose -- see the note
:: above :bootstrap_uv_python. Every py/python invocation goes through CALL: "py"/"python" is
:: usually python.exe, but a version manager (pyenv-win and friends) can shim it as a .bat/.cmd,
:: and running a batch file without CALL from inside another batch file hands control to it and
:: never returns -- CALL is correct either way and costs nothing extra for a real .exe.
:: ---------------------------------------------------------------------------------------------
:create_venv
if exist ".venv\Scripts\python.exe" exit /b 0

echo [ExoCortex] Creating virtual environment...

where py >nul 2>nul
if errorlevel 1 goto :no_py
echo [ExoCortex] Trying "py -3.12"...
call py -3.12 -m venv .venv >nul 2>nul
if exist ".venv\Scripts\python.exe" exit /b 0
echo [ExoCortex] Trying "py -3.13"...
call py -3.13 -m venv .venv >nul 2>nul
if exist ".venv\Scripts\python.exe" exit /b 0
echo [ExoCortex] Trying "py -3.11"...
call py -3.11 -m venv .venv >nul 2>nul
if exist ".venv\Scripts\python.exe" exit /b 0

:no_py
where python >nul 2>nul
if errorlevel 1 goto :no_python
call python -c "import sys; sys.exit(0 if (3,11) <= sys.version_info[:2] <= (3,13) else 1)" >nul 2>nul
if errorlevel 1 goto :no_python
echo [ExoCortex] Using system "python" ^(3.11-3.13^)...
call python -m venv .venv >nul 2>nul
if exist ".venv\Scripts\python.exe" exit /b 0

:no_python
echo [ExoCortex] No usable system Python ^(3.11-3.13^) found; downloading a private one with uv...
call :bootstrap_uv_python
exit /b %errorlevel%

:: ---------------------------------------------------------------------------------------------
:: :ensure_deps -- core install (requirements.txt) plus this machine's GPU extras
:: (requirements-nvidia.txt on NVIDIA, requirements-directml.txt on AMD, nothing otherwise, per
:: `exocortex.gpu --extras`), re-run whenever the stamp is missing, older than any input, or
:: records a different GPU vendor than what's currently detected.
::
:: Timestamp check: a plain PowerShell (Get-Item).LastWriteTime comparison rather than
:: "xcopy /D" or "forfiles" -- both of those compare against the SYSTEM CLOCK or a fixed date
:: (not against another file) without extra parsing, and forfiles' date matching is locale
:: dependent; comparing two LastWriteTime objects with PowerShell needs no date parsing at all
:: and is what the rest of this repo already reaches for on Windows.
::
:: Vendor check: exocortex.gpu is stdlib-only (ctypes), so paying for this "few ms" probe on every
:: launch -- even the fast "up to date" path -- is cheap and was always the documented intent (see
:: :ensure_setup above). Without it, a laptop whose detected primary GPU changes between runs
:: (driver update, hybrid-graphics/Optimus switch, dock change) with no watched file touched would
:: keep "Setup is up to date." forever and never re-run :install_extra for the new vendor.
:: ---------------------------------------------------------------------------------------------
:ensure_deps
set "CUR_VENDOR="
for /f "usebackq delims=" %%v in (`".venv\Scripts\python.exe" -m exocortex.gpu --vendor`) do set "CUR_VENDOR=%%v"

powershell -NoProfile -Command "$stamp = '.venv\exocortex-setup.stamp'; if (-not (Test-Path $stamp)) { exit 1 }; $t = (Get-Item $stamp).LastWriteTime; $newer = $false; foreach ($s in 'requirements.txt','requirements-nvidia.txt','requirements-directml.txt','run.bat') { if ((Test-Path $s) -and ((Get-Item $s).LastWriteTime -gt $t)) { $newer = $true } }; if ($newer) { exit 1 }; $stampVendor = ((Get-Content $stamp -TotalCount 1) -split '\s+')[0]; if ($stampVendor -ne $env:CUR_VENDOR) { exit 1 }; exit 0"
if not errorlevel 1 (
  echo [ExoCortex] Setup is up to date.
  exit /b 0
)

echo [ExoCortex] Installing dependencies...
if "%EXO_SETUP_DRYRUN%"=="1" (
  echo [dryrun] .venv\Scripts\python.exe -m pip install --upgrade pip
  echo [dryrun] .venv\Scripts\python.exe -m pip install -r requirements.txt
) else (
  ".venv\Scripts\python.exe" -m pip install --upgrade pip
  if errorlevel 1 exit /b 1
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt
  if errorlevel 1 exit /b 1
)

:: GPU-specific extras: exocortex.gpu is stdlib-only (ctypes), so this is cheap and safe to run
:: for real even under EXO_SETUP_DRYRUN -- only the pip installs it may trigger are simulated.
:: VENDOR was already probed above (as CUR_VENDOR) to decide freshness; reuse it here so it's
:: only probed once per run.
set "VENDOR=%CUR_VENDOR%"
set "EXTRAS="
for /f "usebackq delims=" %%e in (`".venv\Scripts\python.exe" -m exocortex.gpu --extras`) do set "EXTRAS=%%e"

if "%EXTRAS%"=="" (
  echo [ExoCortex] GPU: %VENDOR% -^> extras: none
) else (
  echo [ExoCortex] GPU: %VENDOR% -^> extras: %EXTRAS%
  for %%F in (%EXTRAS%) do call :install_extra "%%F"
)

:: Stamp records the vendor and the date this setup ran, mainly to make a stale stamp file
:: self-explanatory when someone stumbles on it; only its timestamp is checked above.
> ".venv\exocortex-setup.stamp" echo %VENDOR% %DATE% %TIME%
exit /b 0

:: ---------------------------------------------------------------------------------------------
:: :install_extra <requirements file> -- onnxruntime-directml REPLACES the CPU "onnxruntime"
:: package that faster-whisper/rapidocr depend on, so it must be uninstalled first, every time
:: (a later `pip install -r requirements.txt` would otherwise pull plain onnxruntime back in and
:: silently undo the DirectML install).
:: ---------------------------------------------------------------------------------------------
:install_extra
set "EXTRA_FILE=%~1"
echo %EXTRA_FILE% | findstr /I "directml" >nul
if not errorlevel 1 (
  if "%EXO_SETUP_DRYRUN%"=="1" (
    echo [dryrun] .venv\Scripts\python.exe -m pip uninstall -y onnxruntime
  ) else (
    ".venv\Scripts\python.exe" -m pip uninstall -y onnxruntime
  )
)

if "%EXO_SETUP_DRYRUN%"=="1" (
  echo [dryrun] .venv\Scripts\python.exe -m pip install -r %EXTRA_FILE%
  exit /b 0
)

if not exist "%EXTRA_FILE%" (
  echo [ExoCortex] WARNING: %EXTRA_FILE% not found ^(skipping this GPU extra^).
  exit /b 0
)
".venv\Scripts\python.exe" -m pip install -r "%EXTRA_FILE%"
if errorlevel 1 (
  echo [ExoCortex] WARNING: installing %EXTRA_FILE% failed; continuing without it.
)
exit /b 0

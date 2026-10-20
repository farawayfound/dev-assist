# Suggestions for exo-cortex (found while porting to macOS)

These findings were produced by five independent code readers, each assigned a different lens (concurrency, resources, robustness, security-privacy, design-optimization), reading the pristine exo-cortex source at the mac-cortex fork's base commit `3ae4898`. Every finding was then judged by two independent reviewers: one trying to refute it directly against the code and tests, one judging whether it is material enough to matter. Only findings both reviewers accepted are listed below as confirmed; the rest are listed separately as rejected, with both judges' reasoning, for your own judgement. All line numbers refer to commit `3ae4898` and may have drifted on the Windows machine since; please verify each item against your current source before changing anything.

## Summary of confirmed findings

| # | Severity | Category | File:line | Title | Status in mac-cortex | Status in exo-cortex (Windows) |
|---|----------|----------|-----------|-------|----------------------|-------------------------------|
| 1 | critical | race | `exocortex/llm/server.py:361` | Quitting while llama-server is still loading can hang shutdown for up to 180s | fixed (shared code) | fixed (Intel-integration, `server.py`; verified live) |
| 2 | critical | robustness | `exocortex/session.py:184` | Autosave thread has no exception guard: one write failure permanently kills all future autosaves | fixed (shared code) | fixed (ported) |
| 3 | high | resource | `exocortex/llm/assistant.py:1472` | Notes generation sends an unbounded, ever-growing transcript slice | fixed (shared code) | fixed (ported, then adjusted -- see below) |
| 4 | high | optimization | `exocortex/audio/capture.py:592` | Audio-capture watchdog re-inits a full PyAudio/PortAudio host every 2s, forever, per stream | fixed on macOS only; Windows path unchanged | fixed (Windows-specific design -- see below) |
| 5 | high | robustness | `exocortex/session.py:208` | Session files (transcript.md, session.json, etc.) are written non-atomically | fixed (shared code) | fixed (ported) |
| 6 | high | bug | `exocortex/completion.py:344` | Abandoned-completion retry discards the wrong cache key, permanently blacklisting the prefix | fixed (shared code) | fixed (ported) |
| 7 | high | privacy | `exocortex/speech/pipeline.py:628` | Echo-suppressed transcript text logged verbatim to the persistent app log | fixed (shared code) | fixed (ported) |
| 8 | high | optimization | `exocortex/llm/prompts.py:465` | Hands-free Text then Code run as two full, cache-breaking LLM calls over the same transcript+screen content | open | open, deliberately (analysis below) |
| 9 | medium | race | `exocortex/app.py:227` | A deferred request can be silently and permanently dropped when the LLM becomes ready | fixed (shared code) | fixed (ported and extended) |
| 10 | medium | bug | `exocortex/handsfree.py:985` | Deviation heuristic can misclassify mid-typed (incomplete) code lines as user drift, wrongly regenerating Code | fixed (shared code) | fixed (ported, plus a casefold bug the fork shares) |
| 11 | medium | optimization | `exocortex/completion.py:451` | Completion's asked-prefix cache pruning drops entries at random instead of the oldest ones | fixed (shared code) | fixed (ported) |
| 12 | medium | optimization | `exocortex/speech/pipeline.py:525` | Live ASR partials re-transcribe the whole growing utterance every 1.2s instead of the new audio only | mitigated | mitigated (ported, adjusted to keep a Windows-pinned contract) |
| 13 | medium | optimization | `exocortex/screen/reader.py:481` | Accurate OCR engine (RapidOCR/OpenVINO) is compiled synchronously on the screen-reader thread, delaying the very first screen read | mitigated (moot with Apple Vision) | not changed, deliberately (analysis below) |

**Status in mac-cortex** (verified 2026-09-26 against `git diff 3ae4898 HEAD`, one checker per finding):

- *fixed (shared code)*: the module has no Windows/macOS split, so the same change would fix the
  Windows build. Each is marked `mac-cortex:` in the code so it can be cherry-picked. It was tested
  on macOS only; verify on the Windows machine before merging.
- *fixed on macOS only*: #4. The macOS watchdog uses sounddevice's cheap device list (or skips the
  check for ScreenCaptureKit). The Windows pyaudiowpatch path in `_win_resolve_device` still builds
  and tears down a full PyAudio host every 2 s.
- *mitigated*: #12 skips a partial while the ASR worker is busy and slows the cadence to 2 s on
  macOS, but each partial still re-transcribes the whole utterance. #13 is moot for the default
  macOS engine (Apple Vision starts in ~130 ms), but the RapidOCR path still builds its engine
  synchronously on the reader thread.
- *open*: #8. `system_text` and `system_code` still differ from their first line, so a hands-free
  Text then Code pair re-evaluates the whole prompt. The transcript-window hysteresis added in the
  fork keeps the `<transcript>` block stable but does not touch the system prompt.

**Status in exo-cortex (Windows)** (2026-09-26, branch `Intel-integration`, committed alongside the Intel Arc work; every
change is marked `upstream finding N (ported from mac-cortex)` or `upstream finding N (Windows
fix)` in the code, one adversarial review pair per finding, tests green in the touched files):

- **#1** -- fixed by the parallel Intel-integration work on `exocortex/llm/server.py` (confirmed
  2026-09-26): `start()`/`_attempt()` no longer hold `self._lock` across `_wait_ready()`; a
  `threading.Event` (`self._stop_event`) lets `stop()` interrupt the `/health` poll within one
  0.25 s tick, after which `start()` raises `LlamaServerError("stopped")` instead of trying the
  next runtime pack. One deliberate deviation from the fork: `stop()` leaves the event set and only
  the next `start()` clears it, because clearing at the end of `stop()` can race a poll that has
  not yet observed the event and lose the interrupt. Verified live against a 16.8 GB Qwen3.8-27B
  load: `start()` raised "stopped" in under a second; `stop()` returned in ~5 s, dominated by the
  pre-existing terminate/wait/kill teardown of a large in-flight mmap, not by lock contention.
  Tests: the stop-during-start cases in `tests/test_server.py` (fake Popen + fake httpx).
- **#2, #5** (`session.py`) -- ported as is: the autosave tick and `stop()`'s final save are
  guarded (log, retry next tick; `stop()` still unsubscribes), every session file goes through
  `_atomic_write_text` (sibling `.tmp` + `os.replace`). Per-file atomicity only: the five files are
  still written one after another. Windows note: `os.replace` raises `PermissionError` while AV or
  OneDrive holds the destination -- exactly the transient class the tick guard now survives.
- **#3** (`assistant.py`, `transcript.py`) -- the bound is ported, the *shape* is not. The fork keeps
  the NEWEST `context_tokens * 4` chars of an oversized slice and advances `_notes_last_id` on
  every failure; both silently drop transcript from the notes (a transient backend error loses
  that window; a backlog bigger than the budget loses its older part the moment notes recover).
  Windows instead uses `Transcript.head_text`: the OLDEST chunk that fits, the checkpoint advances
  exactly to the last segment sent, the next chunk is queued immediately while a backlog remains,
  and a failure leaves the checkpoint alone. Every prompt stays bounded, nothing is skipped, and
  the fork's wedge (a frozen checkpoint resending an ever-growing slice) cannot recur. Recommended
  for mac-cortex too.
- **#4** (`audio/capture.py`) -- Windows-specific fix. The audit's suggestion (reuse the open
  `self._pa`) cannot work: PortAudio snapshots its device list at `Pa_Initialize` and never
  refreshes it, so a fresh `PyAudio()` is the only way to see a new default device. The 2 s
  watchdog tick stays for stream health; the full re-enumeration now runs every
  `_DEVICE_POLL_INTERVAL_S` (10 s) per stream, or at once when a cheap stdlib `winmm`
  `waveInGetNumDevs + waveOutGetNumDevs` probe changes (a plug/unplug is still caught within one
  tick). Only a pure default-device switch in Settings waits up to 10 s. SPEC.md's "checks every
  2 s whether the default device index changed" sentence is folded into the Intel-integration
  docs pass.
- **#6, #11** (`completion.py`) -- ported as is (`_asked` is an `OrderedDict`; the abandon path
  discards the normalised key). The sibling `_plan_typed_ids` prune sorts by plan id, a real
  recency key, and was left alone.
- **#7** (`speech/pipeline.py`) -- ported as is (length only). No other INFO-level line in the
  module logs recognised text.
- **#8** (`prompts.py`) -- left open on purpose. Verified against llama.cpp's current server: a
  request goes to the idle slot whose cached prompt shares the longest common *prefix*
  (`--slot-prompt-similarity`, default 0.10), else the least-recently-used slot. With this repo's
  default `parallel = 2` the Text and Code prompts, which diverge in their first system line,
  therefore land in separate slots and each reuses its own previous prompt; the pair still shares
  nothing, and any third job kind (notes, ask, auto-answer) evicts one of them. `--cache-reuse 256`
  does not help: llama-server disables it under `--parallel 2` / `--kv-unified` (SPEC.md). So the
  cost is real but smaller than the audit states, and a unified preamble would change prompts
  David tuned live; it needs a live A/B (`scripts/bench_llm.py --protocol full`) before landing.
- **#9** (`app.py`) -- ported (`_defer` re-checks readiness under `_pending_lock`; `_drop_pending`
  reports what was queued from every `_init_llm` failure branch, which Windows has four of) and
  extended: after such a failure, later requests are refused with an `llm`/`error` status naming
  the request instead of being queued behind a "still starting" notice that is no longer true.
- **#10** (`handsfree.py`) -- ported, plus a bug the fork's fix has too: `squash(line)` compared a
  raw OCR line against `self._typed` keys that are casefolded, so any uppercase character in the
  half-typed line (class names, `True`/`None`, camelCase) made the exclusion miss. Windows squashes
  `normalize_line(line)`, the same form the keys are built from. mac-cortex should take this.
- **#12** (`speech/pipeline.py`) -- mitigated like the fork, adjusted: an unconditional "skip the
  partial while the worker is busy" broke a Windows-only pinned contract
  (`tests/test_pipeline_extra.py::test_slow_transcribe_never_publishes_or_runs_a_superseded_partial`:
  the *last* partial must still run once the worker catches up), so a partial that arrives while
  busy is stashed (newest per source) and queued the moment the worker frees up, atomically with
  the busy flag. The headline cost (each partial re-decodes the whole utterance) is unchanged and
  matters less here: Windows ASR runs at ~0.12 s (CUDA) / ~0.19 s (sherpa, CPU) per utterance.
- **#13** (`screen/reader.py`) -- not changed. The measured process-order rule (a RapidOCR engine
  created after a Windows OCR engine crashes the interpreter) forbids the suggested fix of serving
  early ticks with the cheap engine while RapidOCR compiles in the background; the accurate engine
  is already built on the very first tick after `start()`, so the delay is startup-only and bounded
  (~2 s with a warm OpenVINO cache, ~13 s first ever). Moving both constructions to a background
  thread would not make the first read earlier.
- **R1-R6** -- not acted on (the audit's materiality call stands). R3 and R6 are the same
  log-hygiene class as #7 and would be one-line changes if wanted.

Found while porting, for mac-cortex: the #10 casefold miss above; #3's tail-cap plus
advance-on-failure drops content (see above); the fork's
`test_autosave_survives_a_save_exception_and_keeps_ticking` needs a wider deadline than 5 s on a
loaded machine (the autosave interval floors at 1 s, and the worst case takes three ticks).

## 1. Quitting while llama-server is still loading can hang shutdown for up to 180s

**Severity:** critical  
**Category:** race  
**File:** `exocortex/llm/server.py:361`  
**Lens:** concurrency

**Claim:**

LlamaServerManager.start() holds self._lock for its entire body, including the blocking _wait_ready() poll loop (up to cfg.llm.llama_startup_timeout_s, default 180s), and stop() also needs self._lock before it can terminate the child process — so a concurrent stop() called during startup (e.g. the user quitting the app) blocks behind the whole startup timeout instead of promptly killing the process.

**Evidence:**

start() (lines 361-391) wraps its whole body, including self._wait_ready(), in `with self._lock:`. _wait_ready() (lines 393-412) loops `while time.monotonic() < deadline: ... time.sleep(0.25)` with deadline = now + cfg.llm.llama_startup_timeout_s (default 180.0 in config.py / config.example.toml) and never checks any external stop/cancel signal. stop() (lines 423-438) begins `with self._lock:` before it can call p.terminate()/p.kill(). Engine.stop() (app.py ~343-356) calls comp.stop() for ("llama-server", self.server) synchronously as part of normal shutdown (wired to app.aboutToQuit in _run_ui), on whatever thread stop() is invoked from (the Qt main thread via aboutToQuit). tests/test_server.py has no test for stop()-during-start() concurrency.

**Failure scenario / payoff:**

User launches the app with a large/slow-to-load model (e.g. this M4's Ternary-Bonsai-2-27B via a hypothetical local llama-server-equivalent path, or any big GGUF on the original Windows/CUDA target) and quits before the model finishes loading (or the server never becomes healthy, e.g. wrong path/OOM). Engine.stop() -> LlamaServerManager.stop() blocks on self._lock, which start()/_wait_ready() still holds, for up to the remaining startup timeout (tens of seconds to 3 minutes) before the child process is actually killed and the app can exit. The app appears completely frozen on quit for that whole window.

**Suggested fix:**

Do not hold the lock across the blocking readiness wait: acquire the lock only to mutate self._proc/self.binary/self.model/self.port, release it before polling, and have _wait_ready() (or the caller) watch a cancellation/stop event so a concurrent stop() can terminate the process immediately and unblock the waiter (e.g. poll a threading.Event with a short timeout instead of holding the RLock through the whole sleep loop).

**Finder confidence:** 0.82

**Reviewer notes:** Both reviewers verified the mechanism directly against server.py and app.py: start() holds the RLock across its full body including the blocking _wait_ready() poll (server.py ~361-391, ~393-412), stop() (~423-438) needs the same lock before it can terminate the process, and the default `llama_startup_timeout_s` is 180s (config.py:50). Engine.start() runs server.start() on a background "llm-init" thread (app.py ~134-135) while Engine.stop() is wired synchronously to the Qt main thread's `aboutToQuit` (app.py:882, 343-360), so quitting during startup blocks the main thread for up to the remaining timeout. `tests/test_server.py` only exercises the sequential start-then-stop path; no test or doc pins a slow/hung quit during model load as accepted. Both judges kept the severity at critical, agreeing this is a genuine, easily-triggered UI hang (quit shortly after launch, or with a slow/misconfigured model) with a straightforward fix.

## 2. Autosave thread has no exception guard: one write failure permanently kills all future autosaves

**Severity:** critical  
**Category:** robustness  
**File:** `exocortex/session.py:184`  
**Lens:** robustness

**Claim:**

SessionRecorder._run() (the daemon autosave loop) wraps nothing in try/except, unlike the otherwise-identical patterns in Completer._run() (completion.py:267-271) and HandsFreeController._run() (handsfree.py:749-754). Any exception raised inside _drain_pending()/_save() -- e.g. a transient PermissionError/OSError from write_text() while an AV/OneDrive process has the session folder open, a full disk, or any other I/O hiccup -- propagates uncaught and terminates the daemon thread.

**Evidence:**

```
def _run(self) -> None:
    interval = max(1, self._cfg.paths.autosave_interval_s)
    while not self._stop_event.is_set():
        self._drain_pending()
        if self._dirty.is_set():
            self._save(kind="saved")
        self._stop_event.wait(interval)
(session.py:184-191) -- compare to completion.py:267-271 and handsfree.py:749-754, both of which wrap their per-tick work in `try: ... except Exception: log.exception(...)`.
```

**Failure scenario / payoff:**

Mid-meeting, Windows Defender (or OneDrive sync) transiently locks a file inside the freshly created session folder while _save() is writing transcript.md/session.json. write_text() raises OSError, the exception is uncaught, and the 'SessionRecorder' daemon thread silently dies (only a stderr traceback nobody sees in a packaged/windowless app). From that point on, no more autosaves happen for the rest of the meeting -- transcript, notes and answers are only whatever was captured before the fault. If stop()/export() later hits the same transient condition (or the app crashes before stop() runs), the whole meeting's record is lost, i.e. exactly the failure mode a 'never lose the meeting' co-pilot must not have.

**Suggested fix:**

Wrap the body of the while loop (drain + save) in try/except Exception, log.exception it, and keep looping -- exactly the pattern already used in completion.py and handsfree.py. Optionally also publish a ComponentStatusEvent/error so the UI can surface 'autosave failed' instead of failing completely silently.

**Finder confidence:** 0.9

**Reviewer notes:** Both reviewers confirmed `session.py`'s `SessionRecorder._run()` (184-191) has no try/except around `_drain_pending()`/`_save()`, unlike the equivalent daemon loops in `completion.py`'s `Completer._run()` and `handsfree.py`'s `HandsFreeController._run()`, both of which wrap per-tick work in `try/except Exception: log.exception(...)`. Neither `tests/test_session.py` nor `tests/test_session-misc_extra.py` exercises a save-time exception, so nothing pins the missing guard as intentional. The first reviewer initially rated it high, reasoning that `stop()`/`export()` would likely still run and surface the same class of exception directly to the user at end-of-meeting; the second reviewer raised it to critical, citing README.md:50-51's own statement that a force-closed session "keeps what it last autosaved," meaning a dead autosave thread silently voids that guarantee for the rest of the meeting with no user-visible signal. The recorded corrected severity is critical, and both agree the fix (mirror the try/except pattern already used in `completion.py` and `handsfree.py`) is trivial.

## 3. Notes generation sends an unbounded, ever-growing transcript slice

**Severity:** high  
**Category:** resource  
**File:** `exocortex/llm/assistant.py:1472`  
**Lens:** resources

**Claim:**

Unlike every other prompt path in the app, the notes job's user turn is built from `Transcript.text_since(since_id)`, which has no character/token cap. Every other consumer of the transcript (assist/ask/solve/code/text/suggest, via `Transcript.window_text(max_chars, ...)`) is explicitly trimmed to `cfg.llm.context_tokens * 4` chars, but `_run_notes_job` (assistant.py:1472) calls `text_since()` (transcript.py `text_since`, no `max_chars` parameter at all) and feeds the full, uncapped text into `prompts.user_notes()`. Because `_notes_last_id` is only advanced on a *successful* run (assistant.py:1490), a single failure (e.g. the very prompt this bug produces exceeding `ctx_size=8192` and being rejected/erroring out) permanently freezes `since_id` at that point: every subsequent auto-notes cycle (every `notes.auto_interval_s`, default 180s, for the rest of the session) resends the same (still growing) transcript slice, guaranteeing repeated failure, wasted prompt-eval time (llama.cpp prompt eval on this rig is the documented bottleneck, ~150 tok/s), and a notes feature that is silently dead for the rest of the meeting.

**Evidence:**

exocortex/transcript.py `text_since()` (no max_chars arg, unlike `window_text(max_chars, ...)`); exocortex/llm/assistant.py:1458-1491 `_run_notes_job`: `new_text = self._transcript.text_since(since_id)` then `user = prompts.user_notes(previous, new_text)`, and `self._notes_last_id = until_id` only happens after a successful `_stream_text` (line 1490), inside the same `with self._notes_lock` block that follows the try/except that returns early on `LLMError`/`Exception`.

**Failure scenario / payoff:**

A 2+ hour meeting where notes are only forced once at the end, or where auto-notes is briefly disabled/paused for a stretch then re-enabled, or where one notes call fails for any transient reason: the very next (and every subsequent) notes prompt carries the full multi-hour transcript with no cap, likely exceeding `ctx_size=8192`, causing llama-server to error or truncate unpredictably, and burning multiple seconds of prompt evaluation on a CPU-bound quant every 180s for nothing, with notes never recovering until the app restarts.

**Suggested fix:**

Give `Transcript` a capped variant (or reuse `window_text` with `since_id=since_id, until_id=until_id` and a `max_chars` budget derived from `cfg.llm.context_tokens`) for the notes job's new-transcript slice, and advance `_notes_last_id` to `until_id` even on failure (or at least once the slice is capped) so a stuck `since_id` cannot compound across cycles.

**Finder confidence:** 0.75

**Reviewer notes:** Both reviewers verified that `Transcript.text_since()` (transcript.py:104-107) has no `max_chars` cap, unlike `window_text()` used by every other job (assistant.py:1116), and that `_run_notes_job` (assistant.py:1458-1491) feeds its output uncapped into `prompts.user_notes()`. Both confirmed `_notes_last_id` only advances to `until_id` after a successful `_stream_text` call (assistant.py:1487-1490), so any failed run (e.g. from exceeding `ctx_size=8192`, config.py:43, with no `--context-shift` flag configured, server.py:338) freezes the checkpoint and compounds on every subsequent `notes.auto_interval_s` cycle (default 180s). `tests/test_assistant.py:436` pins that notes deliberately use the full transcript-since-checkpoint rather than a sliding window, but neither that test nor any other exercises the missing cap or the stuck-checkpoint-on-failure path. Both judges rated this high rather than critical: no crash, hang, or data loss occurs, but the feature can go silently and permanently dead for the rest of a session while repeatedly burning prompt-eval time on the documented CPU-bound bottleneck.

## 4. Audio-capture watchdog re-inits a full PyAudio/PortAudio host every 2s, forever, per stream

**Severity:** high  
**Category:** optimization  
**File:** `exocortex/audio/capture.py:592`  
**Lens:** resources

**Claim:**

`AudioCapture._watchdog_tick()` calls `_check_device_changed()` every `_WATCHDOG_INTERVAL_S` (2.0s) for the whole life of the app; that method calls `resolve_device(self._kind, self._spec, pa_module=self._get_pa_module())`, and `resolve_device()` unconditionally constructs a brand-new `mod.PyAudio()` (a full PortAudio/WASAPI host-API + device enumeration, i.e. COM activation of every audio endpoint) and tears it down again (`_safe_terminate`) on every single call, instead of reusing the capture's own already-open PyAudio handle (`self._pa`, opened once in `_open()`). With MIC + SYSTEM both enabled this is two full host-API init/enumerate/terminate cycles every 2 seconds, indefinitely, purely to notice a device-list change that almost never happens.

**Evidence:**

exocortex/audio/capture.py:33-34 (`_WATCHDOG_INTERVAL_S = 2.0`), 566-571 (`_watchdog_loop` ticks every 2s), 573-579 (`_watchdog_tick` calls `_check_device_changed`), 592-601 (`_check_device_changed` calls `resolve_device(...)`), and `resolve_device()` at 159-214, specifically `pa = mod.PyAudio()` at line 172 and `finally: _safe_terminate(pa)` at 175-176 — a fresh instance every call, never reusing `self._pa` (opened in `_open()`, line 422).

**Failure scenario / payoff:**

For the whole duration of any meeting, this runs continuously on two background daemon threads, spinning up and tearing down the native audio subsystem roughly once per second combined — real, continuous CPU and OS-handle churn for a check that only needs to compare the current default device to what is already open, and a known class of intermittent glitch source (PortAudio/WASAPI re-enumeration briefly interfering with an open exclusive-ish stream).

**Suggested fix:**

Reuse the already-open `self._pa` (or a single long-lived PyAudio instance kept just for polling) inside the watchdog's device-change check instead of constructing and terminating a new one every tick; only fully re-enumerate when actually reopening the stream.

**Finder confidence:** 0.65

**Reviewer notes:** Both reviewers traced the chain directly: `_watchdog_loop` ticks every `_WATCHDOG_INTERVAL_S=2.0s` (capture.py:34, 566-571), `_check_device_changed` (592-601) calls `resolve_device()` every tick, and `resolve_device()` (159-176) unconditionally constructs and tears down a fresh `PyAudio()`/WASAPI host and full device enumeration rather than reusing the capture's already-open `self._pa` (opened in `_open()`, line 422/440). With MIC and SYSTEM capture both running by default, this is roughly one full host re-init per second, sustained for the life of every meeting. `tests/test_capture.py`'s `FakePaModule` records instantiation counts but asserts no bound and does not pin the current per-tick behavior as required. Both judges agree this is a large, continuous, avoidable resource cost matching the high bar; one judge notes the finding's side-claim that this is "a known class of intermittent glitch source" for exclusive-mode WASAPI streams is plausible from the file's own comments but not independently verified, so severity rests on the verified resource-waste claim alone.

## 5. Session files (transcript.md, session.json, etc.) are written non-atomically

**Severity:** high  
**Category:** robustness  
**File:** `exocortex/session.py:208`  
**Lens:** robustness

**Claim:**

_save() writes every session file directly with Path.write_text(), with no write-to-temp-then-rename. A crash, force-quit, or power loss during a save (which happens on every autosave tick, every export(), and at stop()) can leave any of transcript.md / transcript.jsonl / notes.md / answers.md / session.json truncated or holding invalid JSON, for a class of files whose whole purpose is being the durable record of the meeting.

**Evidence:**

```
def _save(self, *, kind) -> None:
    ...
    with self._save_lock:
        (folder / "transcript.md").write_text(self._transcript.to_markdown(title), encoding="utf-8")
        (folder / "transcript.jsonl").write_text(self._transcript.to_jsonl(), encoding="utf-8")
        (folder / "notes.md").write_text(self._assistant.notes_markdown or "", encoding="utf-8")
        (folder / "answers.md").write_text(self._render_answers(), encoding="utf-8")
        (folder / "session.json").write_text(self._render_session_json(speakers_snapshot), encoding="utf-8")
        self._dirty.clear()
(session.py:208-219). No .tmp file + os.replace() anywhere in the module.
```

**Failure scenario / payoff:**

The app (or the machine) is killed mid-write during one of the periodic autosave ticks -- e.g. Windows Update reboots the laptop, or the process is force-terminated -- while transcript.jsonl or session.json is partway through being rewritten. The next run (or the user opening the file afterward) finds a truncated JSONL line or invalid JSON, corrupting exactly the durable artifact the whole SessionRecorder exists to produce, for a meeting that otherwise had no other backup.

**Suggested fix:**

Write to a sibling temp file (e.g. `<name>.tmp`) and os.replace() it onto the final path for each of the five files, so a save is all-or-nothing per file and a crash mid-write only ever leaves the previous good version on disk.

**Finder confidence:** 0.85

**Reviewer notes:** Both reviewers confirmed `_save()` (session.py:208-219) writes all five session files (transcript.md, transcript.jsonl, notes.md, answers.md, session.json) via plain `Path.write_text()` with no temp-file+`os.replace()` pattern anywhere in the module, and that this path runs on every autosave tick, at `export()`, and at `stop()` -- not a rare edge case. Neither SPEC.md nor README.md documents non-atomic writes as an accepted limitation (both reviewers grepped for atomic/crash/durable/tmp/os.replace and found only unrelated OCR-crash content), and existing tests (`test_session.py`, `test_session-misc_extra.py`) cover thread-safety and lifecycle but not crash-safety against partial writes. Both judges rated this high rather than critical because the trigger requires an external kill or crash at a precise moment (process termination, OS reboot, power loss) rather than occurring during ordinary operation. The suggested fix -- write to a sibling `.tmp` file and `os.replace()` per file -- is standard and low-risk.

## 6. Abandoned-completion retry discards the wrong cache key, permanently blacklisting the prefix

**Severity:** high  
**Category:** bug  
**File:** `exocortex/completion.py:344`  
**Lens:** robustness

**Claim:**

When an in-flight completion request is abandoned after INFLIGHT_TIMEOUT_S with no output, the code tries to let that prefix be asked again by removing it from the `_asked` de-dup set -- but `_asked` stores `_norm_key(prefix)` (whitespace-collapsed, casefolded) while the discard call passes the raw, unnormalized `inflight.anchor`. For any prefix containing multi-space runs, tabs, newlines or mixed case (i.e. almost any real code/text prefix), `_norm_key(prefix) != prefix`, so the discard is a silent no-op and the normalized key stays in `_asked` forever.

**Evidence:**

```
self._asked.add(_norm_key(prefix))   # completion.py:449, when the request is first sent
...
elif now - inflight.started > INFLIGHT_TIMEOUT_S and not inflight.raw:
    log.warning("completer: job %d produced nothing in %.0f s; abandoning it", ...)
    self._cancel_inflight()
    self._asked.discard(inflight.anchor)   # completion.py:344 -- raw prefix, not _norm_key(prefix)
...
if _norm_key(prefix) in self._asked:      # completion.py:353
    return
```

**Failure scenario / payoff:**

The moment one completion request stalls for 30s (a dropped job, a slow/overloaded llama-server, a cancelled-without-event race) for a given caret position, that exact spot in the file becomes permanently unable to produce a ghost-text suggestion for the rest of the time the user stays in that field -- `_norm_key(prefix) in self._asked` keeps returning True at line 353, so `_request()` is never called again for it. This is a real completion-mode feature silently going dark at exactly the spot where it just failed, with only a debug log line as evidence, until the user moves to a different field (which clears `_asked` via `_clear_all()`).

**Suggested fix:**

Discard the normalized key, not the raw one: `self._asked.discard(_norm_key(inflight.anchor))`.

**Finder confidence:** 0.9

**Reviewer notes:** Both reviewers traced the exact mismatch: `_InFlight.anchor` is set to the raw, unnormalized prefix at request time (completion.py ~434/445), while `_asked` stores `_norm_key(prefix)` (line 449); on abandonment after `INFLIGHT_TIMEOUT_S`, line 344 calls `self._asked.discard(inflight.anchor)` -- the raw string -- which silently fails to remove the normalized key whenever normalization changes the string (any prefix with a tab, multi-space run, newline, or non-lowercase character). SPEC.md:803 documents the de-dup as using "a whitespace/case-normalised key" with no carve-out for the abandon path, so this is a genuine violation of the code's own documented contract, not an accepted quirk. No test in `tests/test_completion.py` exercises the abandonment/timeout path (only the general never-asked-twice de-dup). Both judges agree the effect -- ghost-text completion silently and permanently going dark at that exact caret position until the field changes -- is a real, plausible (any stalled completion job), user-facing feature failure matching the high bar, with a one-line fix (`self._asked.discard(_norm_key(inflight.anchor))`).

## 7. Echo-suppressed transcript text logged verbatim to the persistent app log

**Severity:** high  
**Category:** privacy  
**File:** `exocortex/speech/pipeline.py:628`  
**Lens:** security-privacy

**Claim:**

Every time the echo-suppression feature drops a duplicate utterance (default `[audio].echo_suppression = true`), the full recognized text of that utterance is written at INFO level via `log.info("pipeline: echo-suppressed %s final %r", job.source.value, segment.text)`. `logging_setup.setup_logging()` attaches a `RotatingFileHandler` at root level INFO (default) writing to `%LOCALAPPDATA%/ExoCortex/logs/exocortex.log`, so this line lands in that file verbatim. That file is outside `sessions_dir`, so `exocortex.stealth.prune_sessions()` (the anti-forensics retention that bounds session folders by `sessions_keep_days`/`sessions_keep_max`) never touches it; the log file is only bounded by size (2 MB x 3 backups), not age or content sensitivity, and README/SPEC never disclose that spoken meeting content is written there (SPEC.md's own historical debugging note confirms this file has in fact carried session content before).

**Evidence:**

exocortex/speech/pipeline.py:625-628 (`if self._should_suppress_echo(segment): log.info("pipeline: echo-suppressed %s final %r", job.source.value, segment.text); return`); exocortex/logging_setup.py:37-86 (root level INFO by default, file handler always attached when the log dir is writable, no path/level ever excludes transcript-bearing modules); exocortex/stealth.py's own docstring frames the module's anti-forensics half purely as "the sessions folder" — it never reaches the LOCALAPPDATA log directory.

**Failure scenario / payoff:**

A meeting where the mic picks up the system audio (a common laptop-speaker setup, which is exactly why the echo-suppression feature exists) writes the suppressed duplicate's real spoken sentence into a plaintext, unencrypted file under `%LOCALAPPDATA%\ExoCortex\logs\exocortex.log`. That file survives after the session folder has been deleted by `[stealth]` retention (`sessions_keep_days`), and it is readable by anything with filesystem access to the user profile (an EDR/forensic reviewer, another local user, a backup agent) — exactly the exposure the user believes `[stealth]` and the capture-exclusion veil eliminate, and README never documents it.

**Suggested fix:**

Never log the raw recognized text; log only length/metrics (e.g. `len(segment.text)`, hash, or a fixed placeholder) for this and any similar diagnostic line, or gate `%r` at DEBUG and additionally scrub/redact it under the default INFO root level so it never reaches the rotating file handler in normal operation.

**Finder confidence:** 0.85

**Reviewer notes:** Both reviewers confirmed pipeline.py:628 logs the full recognized text of every echo-suppressed utterance at INFO via `%r`, that the app's default log level is INFO (app.py:892/902) with an unconditional `RotatingFileHandler` (logging_setup.py) under `%LOCALAPPDATA%\ExoCortex\logs`, and that `stealth.py`'s anti-forensics pruning (`prune_sessions()`) is explicitly scoped only to the sessions folder, never that log directory -- so this text is never aged out except by size (2MB x 3 backups). Neither reviewer found a test or documented exception covering this log line, and both note `echo_suppression` defaults to true and fires in the common scenario of a laptop mic picking up its own speaker audio, so the trigger is routine rather than rare. Both judges rated this high: it defeats a documented retention/anti-forensics guarantee via an undisclosed side channel, though it is a persistence/audit-trail leak rather than a live crash, hang, or wrong-answer failure. The suggested fix -- never log the raw text, or gate it at DEBUG with a length/hash placeholder at INFO -- is direct and cheap.

## 8. Hands-free Text then Code run as two full, cache-breaking LLM calls over the same transcript+screen content

**Severity:** high  
**Category:** optimization  
**File:** `exocortex/llm/prompts.py:465`  
**Lens:** design-optimization

**Claim:**

`system_text` (prompts.py:526) and `system_code` (prompts.py:465) are entirely different strings from their first line ('...pressed "Text"...' vs '...pressed "Code"...'), and ChatML rendering (`render_chatml`) puts the system message first in the prompt. Since llama.cpp/llama-server's KV prefix cache only reuses tokens that match from the start of the prompt, and by default (`[explain].enabled = false`, the shipped default) hands-free's `_planned_steps` (handsfree.py:1009-1023) runs BOTH 'text' and 'code' as two separate assist jobs back-to-back on the identical transcript window and identical (often 6000-char, `context_chars`, config.py:320) screen/cumulative-context text, the second call gets zero prefix-cache benefit from the first even though ~90% of its content (transcript + screen) is byte-identical.

**Evidence:**

prompts.py:465-524 (system_code/user_code) vs prompts.py:526-589 (system_text/user_text) -- both build `<transcript>...</transcript>` then `<screen>...</screen>` in the same shape but with a different system preamble; handsfree.py:1014 `run_text = bool(hf.run_text) and not self._cfg.explain.enabled and not self._cfg.completion.enabled` shows Text is skipped ONLY when explanations are on, so with the default `explain.enabled=false` both steps run every cycle (handsfree.py:1020 `steps = [kind for kind, on in (("text", run_text), ("code", run_code)) if on]`); SPEC.md's own measured fact table gives the CPU-bound prefill cost this doubles: 'Generation ~58 tok/s; prompt eval ~150 tok/s ... keep prompts short and prefix-stable so llama.cpp's KV prefix cache hits' and README's 'What to expect' table: '2-5 s when the screen text changed (it is re-evaluated by the CPU-bound model: ~1 s per 1000 characters...)'.

**Failure scenario / payoff:**

Every hands-free (or manual Text-then-Code) cycle on a real screen change pays full CPU prefill twice for content that is identical the second time. With `context_tokens=2500` (config.py:65, ~10000 chars of transcript) plus up to 6000 chars of hands-free screen context (config.py:320), a single prefill at the documented ~1 s/1000 chars can run several seconds; because the system prompt differs, the SECOND call (Code, right after Text) cannot reuse ANY of that -- it pays the same multi-second prefill again instead of the ~60 ms cached-prefix hit SPEC.md documents for a matching prefix. This is pure waste on every hands-free cycle in the shipped default configuration (explanations off).

**Suggested fix:**

Make the system message identical for the 'text' and 'code' job kinds (a single shared preamble describing the co-pilot and the shared <transcript>/<previous>/<screen> contract), and push the only differing instruction ("write a walkthrough" vs "write the code") into the trailing 'ask' sentence that user_text/user_code already append after the identical transcript+screen blocks (prompts.py:512-523, 571-589). That lets llama-server's prefix cache match through the entire transcript+screen portion of the Code call against the just-completed Text call, leaving only the last sentence (and the generation itself) to actually re-evaluate.

**Finder confidence:** 0.7

**Reviewer notes:** Both reviewers confirmed `system_code` (prompts.py:465-524) and `system_text` (prompts.py:526-589) diverge from their first line, that `render_chatml` places the system message first (prompts.py:45-56), and that with the shipped defaults (`explain.enabled=false`, `completion.enabled=false`; config.py:236,269) hands-free runs both "text" and "code" steps back-to-back on the identical transcript+screen content (handsfree.py:1009-1023). Because llama.cpp's prefix cache only reuses tokens matching from position 0, the differing system preambles mean the second call gets zero cache benefit from the first, despite the prompt module's own docstring stating the system message should stay "identical for every turn of a given (persona, style) pair" specifically for cache reuse -- both reviewers call this an overlooked exception to the codebase's own stated caching invariant. One reviewer independently verified the two jobs run strictly sequentially (not concurrently), so this is real added wall-clock latency, not just wasted FLOPs. Both agree the fix -- unify the system preamble and move the differing instruction into the trailing per-call "ask" sentence -- is concrete, low-risk, and directly closes the gap, matching the high bar for a default-path, easily quantified latency cost.

## 9. A deferred request can be silently and permanently dropped when the LLM becomes ready

**Severity:** medium (originally reported as high)  
**Category:** race  
**File:** `exocortex/app.py:227`  
**Lens:** concurrency

**Claim:**

_init_llm() sets self._llm_ready and then swaps/drains self._pending exactly once; _defer() can append a new item to self._pending under the same lock but *after* that one-time drain has already run, so the appended request is never processed again.

**Evidence:**

_init_llm() (lines 227-229): `self._llm_ready.set()` followed by `with self._pending_lock: pending, self._pending = self._pending, []` and then iterating `pending` once. _defer() (lines 270-277): checks `if self._llm_ready.is_set() ... return False` (not deferred) BEFORE taking self._pending_lock, and only appends to self._pending inside the lock afterward. There is no second/later drain of self._pending anywhere else in the module (grep confirms _pending is only read/written at lines 227-229 and 275).

**Failure scenario / payoff:**

If a caller (ask/code/text/generate_notes/solve/explain, invoked from the Qt UI thread or a hotkey) runs _defer() and reads is_set()==False just as the llm-init thread finishes _init_llm()'s set()+drain (a window that can be widened by ordinary thread scheduling delays, not just a couple of CPU instructions), the caller's append to self._pending lands in a list nobody will ever drain again. The user's request (e.g. a typed question, a Code/Text press) silently vanishes: no error, no answer, no further status update — the request simply never runs.

**Suggested fix:**

Have _defer() re-check self._llm_ready.is_set() again after acquiring self._pending_lock (and, if now set, run the request immediately instead of queuing it), or fold both the readiness flag and the pending list under one lock so the check-then-append is atomic with the set()+drain in _init_llm().

**Finder confidence:** 0.55

**Reviewer notes:** Both reviewers confirmed the TOCTOU race: `_defer()` (app.py:270-277) checks `self._llm_ready.is_set()` before acquiring `self._pending_lock`, while `_init_llm()` (227-229) does `self._llm_ready.set()` then a one-time swap-and-drain of `self._pending` under the same lock, with no later drain anywhere in the module. If a caller's `is_set()` read happens just before `_init_llm()` completes its drain, the caller's appended request sits in a list nobody will ever process again, with no error or retry. Both reviewers downgraded the finder's original "high" rating to medium: the race window is only a few Python bytecodes wide, it can only occur once during the single, brief LLM-startup phase, and its blast radius is a single dropped request, recoverable by the user simply repeating the action, rather than crashing the app, corrupting broader state, or leaking content. `tests/test_app.py` only pins the non-racing deferred-then-replayed path and does not contradict the race. The suggested fix -- re-check `is_set()` inside the lock, or unify the flag and list under one lock -- is correct and cheap.

## 10. Deviation heuristic can misclassify mid-typed (incomplete) code lines as user drift, wrongly regenerating Code

**Severity:** medium  
**Category:** bug  
**File:** `exocortex/handsfree.py:985`  
**Lens:** robustness

**Claim:**

deviation()/is_own_code_line() decide whether the code on screen has 'drifted' from the app's own solution by fuzzy-matching each on-screen code line against the app's own lines (exact match, gutter-tail match, or plan._similar's head-12-chars-with-paren shortcut / difflib >=0.85 ratio with length delta capped at max(3, len//4)). A line the user is still in the middle of typing (not yet complete) generally fails all three: it isn't equal, it has no gutter tail, its length differs from the target by more than max(3, len//4) once several characters are still missing, and (unless it's a def/class signature with an early paren) it doesn't get the 12-char-head shortcut either -- so it is counted as 'user' (deviating) code even though the user is typing exactly the plan's own solution.

**Evidence:**

```
handsfree.py:483-507 (deviation()) and handsfree.py:455-480 (own_code_lines/is_own_code_line) feed handsfree.py:985-995:
    if (fraction >= float(hf.deviation_threshold) and len(user_lines) >= max(1, int(hf.min_changed_lines))
            and grew >= max(1, int(hf.min_changed_lines))):
        ...
        self._start_cycle(["code"], checkpoint=False)   # regenerates Code, no checkpoint
This only runs once the screen has SETTLED (settle_s, default 1.0s) with self._phase is None, so it fires specifically when the user pauses mid-typing.
```

**Failure scenario / payoff:**

During a live coding interview, the user is transcribing the app's own generated solution by hand and pauses for a second (to think, or interrupted) while 3+ short body-statement lines on screen (e.g. `return dp[n]`, `total += x`, non-signature lines under the 12-char/paren shortcut) are only partially typed and none of the earlier, already-completed lines make up more than half the visible code. deviation_threshold (default 0.5) and min_changed_lines (default 3) are both met purely from in-progress lines, so hands-free silently fires an unrequested Code regeneration mid-typing -- an extra, unwanted model call and a duplicate/possibly different answer appended to the Assist tab at exactly the moment the user is mid-flow in a live meeting.

**Suggested fix:**

Exclude lines the sample-to-sample tracker already knows are 'being typed' (self._typed, already computed by classify_change/_track_sample) from deviation()'s candidate set, the same way `exclude=self._last_trigger_keys` already excludes the original statement's code lines -- a line still growing sample-to-sample should never count as drifted-away-from-own code.

**Finder confidence:** 0.6

**Reviewer notes:** Both reviewers confirmed `deviation()`/`is_own_code_line()` (handsfree.py:463-507) have no awareness of `self._typed`, the tracker that `classify_change` already maintains for lines known to be mid-growth, so a partially-typed, non-signature body line (no early paren/colon, missing more than roughly `len//4` characters) can fail the exact/gutter-tail/fuzzy checks and be counted as "user" drift even when the user is transcribing the app's own generated solution. Reaching the actual regeneration trigger (handsfree.py:985-995) additionally requires 3+ such lines forming at least half of on-screen code and net growth of 3+ since the last baseline while the screen has settled -- both reviewers note this is a narrower bar than "any pause while typing," more realistically hit via a multi-line paste/snippet edited progressively, or several lines simultaneously clipped by pane edges, than by ordinary single-cursor typing. `tests/test_handsfree.py`'s `test_deviation_typing_the_apps_code_is_low` only exercises complete-prefix lines, so the partial-line gap is untested. Both judges agree the effect -- a spurious extra Code-regeneration LLM call and a duplicate/possibly-different answer silently appended mid-interview -- is real but situational, matching medium rather than high or critical.

## 11. Completion's asked-prefix cache pruning drops entries at random instead of the oldest ones

**Severity:** medium  
**Category:** optimization  
**File:** `exocortex/completion.py:451`  
**Lens:** robustness

**Claim:**

`_asked` is a plain `set[str]`; when it exceeds 500 entries the code trims it to 100 with `set(list(self._asked)[-100:])`. Python set iteration order for strings is governed by hash order (and is subject to per-process hash randomization for str by default), not insertion order, so this slice keeps an arbitrary ~100 entries rather than the most recently asked ones the surrounding logic clearly intends (contrast with the correctly-implemented bounded caches in handsfree.py:764-766 and handsfree.py:522-523, which sort by an actual recency key -- job id / plan id -- before trimming).

**Evidence:**

```
self._asked.add(_norm_key(prefix))          # completion.py:449
if len(self._asked) > 500:
    self._asked = set(list(self._asked)[-100:])  # completion.py:450-451 -- no recency ordering
```

**Failure scenario / payoff:**

In one long-lived, continuously-focused field (a long file in one editor tab, or a long chat box, during a multi-hour session -- `_asked` is only cleared on a field/key change), once more than 500 distinct debounce-settled prefixes have been asked, the prune keeps a hash-order-dependent, effectively random subset. Prefixes that should have been evicted (long stale) can survive while recently-asked ones are dropped, causing unnecessary duplicate completion requests for text just asked moments ago -- wasted model calls/latency with no functional benefit, defeating the point of the cache.

**Suggested fix:**

Use an ordered structure that tracks true recency (e.g. `collections.OrderedDict` used as a set via `move_to_end`, or a `deque` of the same max length) so the trim reliably evicts the oldest entries.

**Finder confidence:** 0.55

**Reviewer notes:** Both reviewers confirmed `_asked` is a plain `set[str]` and that the prune at completion.py:450-451 (`set(list(self._asked)[-100:])` once the set exceeds 500 entries) relies on the mistaken assumption that Python set iteration reflects insertion order; CPython set order is hash-bucket order, affected by per-process str hash randomization, so the slice keeps an effectively arbitrary ~100 entries rather than the most recent ones. One reviewer empirically reproduced this, finding that inserting 600 sequential keys and slicing the last 100 retained only about 15 of the truly most-recent 100. Both note `_asked` persists per-field (cleared only on field/key change, completion.py:517) and is a genuine dedup cache pinned by `tests/test_completion.py:219,266`, but no test exercises the >500 eviction path. Both judges rated this medium: the trigger (one continuously-focused field accumulating 500+ distinct prefixes in a multi-hour session) is plausible but narrow, and the downside is bounded to occasional redundant completion requests rather than a crash, hang, or broken feature. One reviewer flags that the sibling cache at completion.py:522-523 (`_plan_typed_ids`) may share the same class of bug and is worth checking while fixing this one.

## 12. Live ASR partials re-transcribe the whole growing utterance every 1.2s instead of the new audio only

**Severity:** medium (originally reported as high)  
**Category:** optimization  
**File:** `exocortex/speech/pipeline.py:525`  
**Lens:** design-optimization

**Claim:**

SpeechOngoing carries the full audio since speech start (speech/vad.py's VadSegmenter._maybe_ongoing, line 187-200, slices from `self._speech_start_sample` to `self._total_fed`), and `_handle_speech_ongoing` (pipeline.py:525-535) hands that whole, ever-growing buffer straight to `_enqueue_partial` -> `Transcriber.transcribe(partial=True)` every `ongoing_interval_s`/`partial_interval_s` (1.2 s default, config.py:95). Because there is exactly one ASR worker thread (pipeline.py:280, `_asr_worker_loop`) serialized behind one lock inside `Transcriber` (SPEC.md's own contract: 'Thread-safe via an internal lock (one transcribe at a time)'), every partial re-decodes audio the model already decoded seconds ago, and a final for the SAME or another source has to wait behind whichever partial call is currently in flight.

**Evidence:**

pipeline.py:525-535 passes `event.audio` (VadSegmenter's full since-start buffer) unchanged into `_enqueue_partial`; vad.py:198 `audio = self._slice_ring(self._speech_start_sample, self._total_fed)` grows every call; config.py:95 `partial_interval_s: float = 1.2`, config.py:113 `max_utterance_s: float = 20.0`; SPEC.md line 219 documents the single 'one transcribe at a time' lock and the single 'asr-worker' thread (pipeline.py:571).

**Failure scenario / payoff:**

For one 20 s monologue (the max before force-split, config.py:113) with the default 1.2 s partial cadence, the ASR model is asked to decode audio growing 1.2, 2.4, 3.6, ... 19.2 s -- 16 calls whose audio-seconds sum to ~163 s, roughly 8x the 20 s actually spoken. On CPU int8 (the README's own recommended default for `[asr]`, keeping Whisper off the GPU) transcription time scales with audio length, so this is real, avoidable CPU burn that grows quadratically with utterance length, and because only one ASR call runs at a time, a partial re-decoding 15+ seconds of audio directly delays the FINAL transcription (and everything gated on it: question detection, auto-answer, notes) behind it -- a plausible, easily hit trigger any time someone talks for more than a few seconds without pausing.

**Suggested fix:**

Either (a) transcribe only the newest chunk since the last partial and stitch/replace the displayed partial text incrementally (accepting that faster-whisper itself isn't a streaming decoder, so this needs an app-level windowing scheme -- e.g. keep the last N seconds as the partial audio instead of the whole utterance), or (b) back off the partial cadence as the utterance grows (e.g. interval scaling with elapsed speech time) so a 15 s monologue isn't re-decoded every 1.2 s, or (c) cap the partial's audio to a trailing window (e.g. last 5-8 s) and prefix it with the already-committed partial text so cost stays roughly constant instead of growing with utterance length.

**Finder confidence:** 0.75

**Reviewer notes:** Both reviewers confirmed `VadSegmenter._maybe_ongoing` (vad.py:187-200) slices from a fixed speech-start sample to the ever-growing total-fed sample, so every `SpeechOngoing` event carries the entire utterance so far, and `pipeline.py`'s `_handle_speech_ongoing` (525-535) forwards that unmodified buffer into `Transcriber.transcribe(partial=True)` every `partial_interval_s` (1.2s default, config.py:95), serialized behind a single asr-worker thread and a single lock in `Transcriber` (asr.py:143). Both agree a long partial call can genuinely delay a subsequently queued final for the same or another source, since the priority queue only reorders jobs not yet started. `tests/test_vad.py` pins the VAD layer's growing-buffer contract but not the pipeline's choice to feed the whole buffer to ASR uncapped, so the suggested trailing-window fix does not conflict with any pinned test. The two judges split on severity -- one kept the finder's "high" given the roughly 8x audio-seconds overhead on a 20s utterance and the direct delay to downstream question-detection/notes; the other corrected it to medium, since utterances force-split at 20s bound the cost and no data is lost -- and the recorded corrected severity is medium.

## 13. Accurate OCR engine (RapidOCR/OpenVINO) is compiled synchronously on the screen-reader thread, delaying the very first screen read

**Severity:** medium  
**Category:** optimization  
**File:** `exocortex/screen/reader.py:481`  
**Lens:** design-optimization

**Claim:**

`_get_engine` (reader.py:460-508) builds the accurate OCR engine (`engine_factory(accurate_name)`, line 481) synchronously, in-line, on the screen-reader worker thread, the first time `_tick` needs any engine at all -- i.e. on the FIRST screen change it observes after startup. Only the model's kernel *warm-up* (`_warm_quietly`, line 428-444) is offloaded to a background thread; the engine construction/compile step itself is not.

**Evidence:**

reader.py:478-488: `if accurate_name and not self._accurate_attempted: ... self._accurate_engine = engine_factory(accurate_name)` runs directly inside `_get_engine`, itself called synchronously from `_tick` (reader.py:336) before any OCR result -- and hence before the first `ScreenTextEvent` -- can be published. SPEC.md's own measured OCR table gives the cost this blocks on: 'first-ever compile ~13 s + ~50 s first-inference kernel builds, ~2 s + ~2 s once CACHE_DIR has the compiled kernels'; the ~50 s kernel-build part is what `_warm_quietly` already defers to a background thread, but the ~13 s (or ~2 s cached) compile step is not.

**Failure scenario / payoff:**

Any session with `[handsfree].accurate_ocr` on and a sampling engine other than RapidOCR (the default combination: cheap sampling OCR + RapidOCR as the accurate engine, per app.py:190) has its very first `ScreenTextEvent` -- and therefore the first on-screen-question answer, the first hands-free task detection, and the first autonomous-solve context -- delayed by the accurate engine's compile time: ~2 s on a machine that already has the OpenVINO kernel cache from a previous run, ~13 s on the very first run ever, instead of the ~15-45 ms capture + ~110-220 ms OCR the README documents as normal. This directly contradicts the documented 'Startup ~5 s warm' expectation for anything that touches the screen in that window.

**Suggested fix:**

Build the accurate engine on its own background thread (the same pattern already used for `_warm_quietly`), and let the sampling engine (or a placeholder) serve the very first tick(s) while the accurate engine finishes compiling in the background, only switching `self._accurate_engine` in once ready -- mirroring how `_warm_quietly` already avoids blocking the *sampling* engine's use on kernel warm-up.

**Finder confidence:** 0.55

**Reviewer notes:** Both reviewers confirmed `_get_engine` (reader.py:460-508) builds the accurate OCR engine via `engine_factory(accurate_name)` (line 481) synchronously inline, called from `_tick` (line 336) before any `ScreenTextEvent` is published, and that only the kernel warm-up (`_warm_quietly`, 428-444) -- not engine construction/compile -- is backgrounded, contradicting README's claim that "the cheap Windows OCR sampling starts at once." One reviewer flags an important nuance: ocr.py:15-19 documents a hard, measured invariant that creating a RapidOcr engine after a WindowsOcr engine crashes the interpreter, which is exactly why the accurate (RapidOCR) engine must be constructed before the sampling engine in one place -- so the finder's proposed fix of letting the sampling engine serve early ticks while the accurate engine compiles in the background would, if implemented naively, violate that ordering and risk the very crash the current code avoids; a correct fix must move the whole accurate-then-sampling sequence to a background thread, preserving order. Both judges agree the delay is real (about 2s typical once the OpenVINO kernel cache is warm, about 13s on a first-ever run, per SPEC.md) but one-time and bounded, matching medium rather than high or critical.

## Reported but judged not material (for your judgement)

The six findings below were read as factually accurate by the code-reading (refuting) judge in every case, but were rejected by the materiality judge as not clearing the bar used for this audit (crash, hang, data loss, meeting-content leaking to capture, corrupted session, a feature silently not working, a state-corrupting race, or measurable resource waste); several of them read as real, undisclosed behavior even where rejected, and you may reasonably disagree with the materiality call and decide to act on one or more of them anyway.

### R1. EventBus's only overflow protection never engages because every real subscriber uses an unbounded queue

**File:** `exocortex/bus.py:285`  
**Lens:** concurrency

**Judge 1 verdict:** Confirmed. bus.py:274 `subscribe(self, maxsize: int = 0)` creates an unbounded `queue.Queue`, and `publish()` (bus.py:285-299) only special-cases `queue.Full` for the drop-oldest logic. Every production call site subscribes with no argument -- session.py:122, handsfree.py:680 (handsfree.py:695 even carries a `# pragma: no cover - unbounded queue` comment on its `queue.Full` handler, confirming the maintainers already know this path is dead), completion.py:219, llm/assistant.py:470, ui/overlay.py:657, app.py:778 -- so `maxsize` is always 0 and `put_nowait()` can never raise `Full` in the real app; only `test_pipeline_extra.py` exercises a bounded queue, and that's an unrelated internal ASR queue (`_asr_queue`, maxsize=3), not an EventBus subscription. I went further and found a concrete, non-hypothetical trigger that doesn't even need a 'stalled' consumer: SessionRecorder's autosave thread (session.py) wakes only every `autosave_interval_s` (default 30s, config.py:386) and each wake calls `_drain_pending()` -> `drain(self._queue)` with the default `max_items=200` (bus.py's `drain()`). AudioLevelEvent alone is throttled to 10 Hz per source with 2 sources (speech/pipeline.py:62,458) = ~600 events/30s just from audio levels, before counting PartialTranscriptEvent/ScreenTextEvent/AssistDeltaEvent fan-out into the same queue. That's already >200/cycle, so under a fully normal long meeting with default config, the recorder's own bus queue grows by roughly 400+ unconsumed events every 30 seconds indefinitely -- exactly the unbounded-growth-with-no-safety-valve outcome the finding describes, and it doesn't even require a stall. (The overlay's 30ms poll and the assistant's continuous `sub.get(timeout=0.2)` bus-listener loop both keep up fine, so this specific structural growth is isolated to the SessionRecorder subscriber, but it's real and it's the one long-running background thread every session always runs.): All of the finding's cited facts check out exactly: bus.py:274's default maxsize=0, publish()'s Full-only drop-oldest branch at bus.py:285-299, and every listed call site (session.py:122, handsfree.py:680, completion.py:219, llm/assistant.py:470, ui/overlay.py:657, app.py:778) indeed passing no maxs

**Judge 2 verdict:** The code fact is accurate: EventBus.subscribe() defaults to maxsize=0, every real subscriber (session.py:122, completion.py:219, handsfree.py:680, llm/assistant.py:470, ui/overlay.py:657, app.py:778, and the _RecordingBus.subscribe() passthrough in app.py:63) calls it with no maxsize argument, so put_nowait() in publish() (bus.py:285-296) can never raise queue.Full and the drop-oldest branch is dead in production. However, tracing the actual consumers turns up no concrete site where a subscriber thread genuinely stalls for an extended period: SessionRecorder's autosave loop (session.py:183-190) wakes every autosave_interval_s and its _save() is a few synchronous file writes that return quickly, not an indefinite block; the Assistant's bus listener runs on its own dedicated thread (_bus_listener_loop, assistant.py:1546) separate from the slow LLM-call worker thread, and its per-event handling is cheap, non-blocking logic; the overlay's QTimer poll runs on the Qt thread but Qt's nested event loop (as used by a modal settings dialog) continues to service other QTimers, so it isn't obviously starved for the long durations needed to matter. Event volumes are also modest (AudioLevelEvent throttled to 10 Hz/source per pipeline.py:62, small dataclasses), so growth absent an actual stall is negligible. No concrete, demonstrated failure trigger exists in this codebase for the 'persistently stalled consumer over a multi-hour meeting' scenario; SPEC/README don't call this out as a known issue either.: This is a genuine, verifiable code observation (dead code: the queue.Full protection can never trigger given every current call site's default maxsize=0) but it is not material by the judge's bar. Materiality requires a demonstrated crash/hang/leak/wrong-answer/content-leak/state-corruption or a qua

### R2. Assist dispatcher adds up to 50ms of avoidable latency to every side/text/explain/completion job

**File:** `exocortex/llm/assistant.py:1077`  
**Lens:** resources

**Judge 1 verdict:** Confirmed as described. Each of the five assist lanes (`_assist_slot`, `_suggest_slot`, `_text_slot`, `_explain_slot`, `_complete_slot`) is its own `_JobSlot()` instance (assistant.py:375-379), and `_JobSlot.__init__` gives every instance its own private `threading.Condition()` (line 210). `put()` only calls `self._cond.notify_all()` on its own slot's condition (line 225). `_next_assist_job()` (1060-1077) round-robins `try_get()` across all five slots but, when none had work, blocks only on `self._assist_slot.wait(timeout=_DISPATCH_POLL_S)` (line 1077, `_DISPATCH_POLL_S = 0.05` at line 117) -- tied to the main slot's condition, not the other four. So `complete()` (line 663 `self._complete_slot.put(job)`), and `suggest_now()`/`solve()`/`code()`/`text()` (line 1027-1028, putting into `_suggest_slot`/`_text_slot`/`_explain_slot`), do not wake a dispatcher currently parked in that wait -- it only notices on its next up-to-50ms poll. The module's own docstring at 1067-1068 even acknowledges this asymmetry ('`_assist_slot.put()`/`close()` wake it immediately ... since those `notify_all()` the very condition this waits on'), confirming the other four lanes do not. No test in tests/ (test_assistant.py, test_assistant_text.py:557-558) pins or exercises this cross-lane wake timing, so nothing makes this a proven non-issue beyond the docstring acknowledging it as an accepted trade-off. The scenario is plausible and frequent: any `complete()` call (fired on every debounced keystroke in completion mode) while the main lane is idle pays up to 50ms of pure dispatch delay on top of network/prompt-eval time, and the same applies to every suggest/solve/code/text/explain job start.: The claim is factually accurate: five independent `_JobSlot` instances each carry their own `threading.Condition`, and the dispatcher's idle-wait only watches the main lane's condition, so a job placed in any other lane while the main lane is empty is not woken until the next ≤50ms poll. This is rea

**Judge 2 verdict:** Mechanism confirmed but impact is negligible in context; not something the maintainer should prioritize.: The mechanism is real: `_JobSlot.__init__` gives each slot its own `threading.Condition` (assistant.py:210), `put()` only calls `self._cond.notify_all()` on that slot's own condition (assistant.py:225), `_assist_slots()` returns five distinct `_JobSlot` instances (assistant.py:1040), and `_next_assi

### R3. Text typed in any application, anywhere on the desktop, is logged while completion mode is on

**File:** `exocortex/completion.py:447`  
**Lens:** security-privacy

**Judge 1 verdict:** Confirmed: exocortex/completion.py:447-448 logs prefix[-40:] (the literal last 40 characters of whatever text field currently has focus, in any foreground app) at INFO level on every completion request. _request() is reached from _tick() (lines 283-357) with no app allowlist or content filter -- the only exclusions are password fields and the app's own windows, enforced upstream in exocortex/screen/caret.py (el.CurrentIsPassword check and the own-pid check). Nothing scopes probing/completion to meeting-related content; it runs whenever completion mode is on (ctrl+alt+k), gated only by debounce_ms/min_interval_ms/min_prefix_chars, none of which affect field selection or content. logging_setup.py confirms INFO is the default level and the file handler has no content redaction -- it does have size-based RotatingFileHandler rotation (2MB x 3 backups), so the finding's word "unpruned" is slightly imprecise (there is pruning by size, just not by age or sensitivity), but the core claim -- ordinary private typed text from arbitrary apps is persisted verbatim to a plaintext log file with no special protection -- is accurate. No test in tests/ pins this log line's content.: The cited log statement exists verbatim as quoted, the password/own-window exclusions are the only upstream filters, and the request path is reachable from any focused text field in any foreground application whenever completion mode is enabled -- there is no meeting-only or app-allowlist gate. The 

**Judge 2 verdict:** Confirmed as a real, reproducible fact of the code (exocortex/completion.py:447-448 logs `field[:40]` and `prefix[-40:]` at INFO on every completion request, i.e. routinely while typing in completion mode, into the rotating `%LOCALAPPDATA%\\ExoCortex\\logs\\exocortex.log`), but it does not clear the materiality bar as defined for this judge pass. README (line ~331) confirms completion mode is explicitly general-purpose ("wherever you are typing (a chat box, an editor, a form, a code file)") — not confined to meeting content — so this is a background diagnostic-logging privacy nit for a co-pilot feature, not "meeting content leaking to capture" (that term is used throughout SPEC.md/README.md specifically for the DWM screen-capture-exclusion mechanism, a completely different code path). It also does not cause a crash, hang, wrong/missing answer, corrupted session data, a silently-broken feature, a state-corrupting race, or any measurable perf/resource waste. The exposure itself is real but modest: only the last 40 characters (not full documents), password fields are already excluded upstream, and the file is size-rotated (2MB x 3 backups, not literally "unpruned" forever as the finder states) rather than kept indefinitely — and the app already stores far more sensitive material (full meeting transcripts, notes, answers) in plaintext files by design elsewhere in the session folder, so this is incremental rather than a novel category of exposure. No test in tests/test_completion.py pins this line, so it isn't a deliberately-accepted, tested contract either way. It's a legitimate low/medium hardening item (log counts/timings, not the raw prefix) worth a line-item fix, but it is not something that should jump the maintainer's queue ahead of crash/hang/wrong-answer/meeting-capture-leak/race-condition classes of bugs.: The claim is accurate to the letter (verified at completion.py:447-448 and config.py:230-235/caret.py:255-259), but the judge lens restricts "material" to a specific set of impacts (crash/hang/wrong-or-missing-answer/meeting-content-leaking-to-capture/corrupted-session/feature-silently-not-working/s

### R4. relayed_display_reason() silently treats "could not determine" as "not relayed", defeating the exposure guard's own documented contract

**File:** `exocortex/ui/win32.py:743`  
**Lens:** security-privacy

**Judge 1 verdict:** The mechanics are accurately described — relayed_display_reason() at exocortex/ui/win32.py:743-752 does fold the "definitively local" and "enumeration/lookup failed" cases of monitor_display_device() into the same None, and exposure_reason() (win32.py:546-564) then reads that None as "nothing detected" — but this is not a silent defect defeating a documented contract; it is the explicitly documented, intended behavior of the whole module. relayed_display_reason()'s own docstring (win32.py:745-747, cited by the finder) states outright: "None also covers 'could not tell'... - the same advisory contract as capture_agents." The module docstring (win32.py:1-17) and SPEC.md both spell out that every exposure probe is "advisory in the safe direction... never 'safe'" and that SPEC.md explicitly names this exact failure class as an accepted, known limit: "a renamed binary or an unlisted display driver defeats it" (SPEC.md:17-25). So monitor_display_device's "callers must read as unknown, not local" note is guidance for interpreting that function's raw dict in general, not a mandate that the one caller must fail closed — and that caller's own docstring, right next to the finder's cited lines, says the opposite is intended. The finder built the claim by reading one docstring while ignoring the very next function's docstring and the module/spec-level threat model that both state this is deliberate, non-hard-guarantee behavior. Additionally, per the module docstring the overlay's real hard backstop is the "fail-closed veil" (content veiled whenever affinity can't be verified), with the relayed-display/exposure check being a supplementary advisory signal for paths affinity cannot cover at all — so there is no false sense of a broken safety mechanism here.: Technically real (the None-collapsing happens exactly as described) but not material: it is the explicitly documented, intended design of the entire exposure-probe subsystem (advisory-only, "never safe," with this exact edge case named as an accepted limitation in SPEC.md), not a hidden contract vio

**Judge 2 verdict:** The code behaves as described: relayed_display_reason() and exposure_reason() collapse monitor_display_device()==None (both 'confirmed local' and 'lookup failed') into 'nothing detected'. But this is not a bug relative to the module's stated contract — it is the explicitly documented and tested design of this whole probe layer.: exposure_reason()'s own docstring at exocortex/ui/win32.py:546-549 says '"None" means "nothing detected" (never "safe")', and _check_exposure's docstring in overlay.py:2166-2176 states 'Detection is advisory - a renamed binary or an unlisted virtual monitor defeats it - so this lowers exposure; it d

### R5. The shared llama-swap unload endpoint is enabled by default and hit with no verification that the service at that URL is actually llama-swap

**File:** `exocortex/config.py:85`  
**Lens:** security-privacy

**Judge 1 verdict:** The code shape is accurate (config.py:85 default swap_unload_url="http://127.0.0.1:8081"; instance.py _swap_running_count only checks HTTP 200 + a JSON body shaped like {"running":[...]}/a list, with no service-identity check before the mutating GET /unload; app.py:364,922 call free_llama_swap unconditionally whenever backend=="llamacpp" and swap_unload_url is truthy, which is the default). But this is not an unreviewed default landing on arbitrary installs: the module docstring in instance.py explicitly documents that "this laptop also runs a llama-swap service (a fleet node)" on 8081, that unloading it is done "With the owner's sign-off", and that it "Interferes with whatever that server was doing on purpose -- the foreground session owns the GPU (owner's call, 2026-09-25)." tests/conftest.py:102 likewise states free_llama_swap "talks to the machine's real llama-swap on 127.0.0.1:8081" and stubs it out specifically because a real service is there. ExoCortex is a single-user, single-known-machine personal tool (confirmed throughout by hardware-specific comments like "the 5090's 24 GB"), not general-purpose software shipped to unknown environments, so the default targets a port the owner has confirmed is their own llama-swap. The finding's actual failure scenario -- "a future build/fork" or "some unrelated local server happens to be listening on 8081" -- is speculative/hypothetical and concerns portability/forking, which the task explicitly puts out of scope ("Ignore Windows-vs-macOS portability entirely" and "ignore ... hypothetical concerns"). On the maintainer's actual machine as documented, this is intended, sign-off behavior, not a defect with a plausible trigger.: The claim is factually accurate about the code (no identity/version check guards the mutating GET /unload, and the default is non-empty), but it is not a material finding for this task. The docstrings in exocortex/instance.py and the test comment in tests/conftest.py:102 make explicit that port 8081

**Judge 2 verdict:** Reject as non-material. The finding is technically accurate (config.py:85 defaults swap_unload_url to a non-empty URL, and instance.py's free_llama_swap does an unauthenticated GET /running + GET /unload with only shape-matching, no identity check), but this is a deliberate, dated design decision the maintainer made to fix a real, measured bug: SPEC.md:1616-1621 documents the 2026-09-24/25 mid-meeting freezes caused by a second llama-swap-loaded copy of the same 27B model contending for the GPU that drives the display, and README.md:54-64 documents the default URL and behavior with an explicit opt-out (set swap_unload_url = ""). instance.py:160's docstring says outright: "Interferes with whatever that server was doing on purpose -- the foreground session owns the GPU (owner's call, 2026-09-25)." On the maintainer's actual single-user Windows dev laptop (the only environment in scope per the task instructions, which say to ignore portability/future-macOS-port concerns entirely), port 8081 is their own known llama-swap fleet node, not an unknown third party. The finding's own scenario is speculative and hedged ("some unrelated local server happens to be listening... happens to include a running array") and explicitly relies on a future macOS port, which is out of scope. No crash, hang, wrong answer, meeting-content leak, or state corruption results; this is an accepted, documented tradeoff with a working opt-out, not a live defect worth the maintainer's attention right now.: config.py:81-85 and instance.py:145-181 do exactly what the finding describes mechanically, but SPEC.md:1616-1640 and README.md:54-64 show this is an intentional, documented fix for a measured real bug (GPU contention causing mid-meeting freezes) on the maintainer's own machine, with a clearly docum

### R6. On-screen question text logged at INFO when filtered out by the focus setting

**File:** `exocortex/llm/assistant.py:1618`  
**Lens:** security-privacy

**Judge 1 verdict:** Confirmed as described.: exocortex/llm/assistant.py:1618 does exactly what the finding claims: inside `_maybe_answer_screen_question`, when `questions.is_relevant(question, self._cfg.assist.focus, self._cfg.assist.ignore_patterns)` returns False, the code runs `log.info("assistant: ignoring on-screen question (not %s): %r",

**Judge 2 verdict:** The finding is factually accurate: exocortex/llm/assistant.py:1618 logs `log.info(\"assistant: ignoring on-screen question (not %s): %r\", self._cfg.assist.focus, question[:80])` whenever questions.is_relevant() rejects an on-screen question under [assist].focus. But this is not a materially distinct privacy hole in this codebase. The same file already logs the full (untruncated) question text at DEBUG for every submitted job regardless of relevance filtering (assistant.py:1007: `log.debug(\"assistant: submitted %s job %d: %r\", kind, job.id, question)`), and elsewhere the app logs full verbatim transcript/meeting speech at INFO as a matter of course (exocortex/speech/pipeline.py:628: `log.info(\"pipeline: echo-suppressed %s final %r\", job.source.value, segment.text)`). So writing on-screen meeting content into the local %LOCALAPPDATA% log at INFO is pervasive, by-design behavior across the app, not something unique to the focus-filter path. The log file also does have retention (logging_setup.py: RotatingFileHandler, 2MB x 3 backups), contradicting the finding's 'no retention' framing. Critically, this is a local, on-device diagnostic log under the user's own %LOCALAPPDATA%, not a transmission or a screen-capture leak -- it does not match SPEC's actual 'excluded from screen capture' privacy contract (a distinct feature about live screen-sharing/recording tools, not log files). No crash, hang, wrong answer, corrupted state, silently-broken feature, race, or measurable resource waste results; it is a logging-hygiene nit consistent with how the rest of the codebase already logs meeting/session content.: Materiality requires crash/hang/wrong-or-missing-answer/meeting-content-leaking-to-capture/corrupted-session/silently-broken-feature/state-corrupting-race/measurable-waste. Local plaintext logging of already-on-screen text into a rotating %LOCALAPPDATA% log file meets none of these categories, and i


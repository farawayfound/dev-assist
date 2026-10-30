# Syncing with exo-cortex

dev-assist started as a snapshot of exo-cortex at `18542ae` (2026-09-27), not a git fork: the two
repositories share no history, so upstream work comes over commit by commit and is merged against
the differences below. This page is the starting point for the next sync.

## Where it stands

| | |
|---|---|
| Forked from | exo-cortex `18542ae` (2026-09-27) |
| Synced through | exo-cortex `main` at `c97590e` (2026-10-29) |
| Taken | `6b90e21` context library + window, `968ddc0` View strips / Monitor / Compact, `f998fe3` Optimize, `4d40fce..5594dd4` Capabilities, `02216bc` audio devices, `1fc5aef` given code / Code from scratch / Shortcuts (and the merges `4f304c0`, `6c994ea`, `1f43166`, `c97590e`) |
| Not taken yet | `93231bf` (`feat/background-focus`: shortcuts never move the focus, multi-file workspace, read-on-shortcut, code-only answers), which reached exo-cortex `main` on 2026-10-29 at 23:42, while this sync was being validated; `feat/compiled-exe-tester-build`, still in progress |

## Differences to keep

- **No capture exclusion.** dev-assist is an ordinary window that screen sharing shows. exo-cortex's
  `stealth.py`, display affinity, the fail-closed veil, the exposure guard and their `[ui]` keys are
  not here, so upstream changes to them are dropped (968ddc0's veil fix and
  `MonitorView.set_content_allowed`, the SHORTCUTS row about the exposure guard).
- **`[project]` is dev-assist's.** The repository and Claude Code toolkit lookups live in
  `exocortex/project/` and `[project]`; upstream's `exocortex/context.py` and `[context]` are its
  document library and context window. An older `config.toml` with repo / toolkit keys under
  `[context]` is migrated on load.
- **Prompt builders** take upstream's parameters plus dev-assist's positional `context` (system
  prompts: the session context) and `extra` (user turns: the project lookups). The `insights`,
  `recap` and `dashboard` kinds are dev-assist's; they never get tools (`_NO_TOOL_KINDS`), and the
  toolkit catalog is appended to the system prompt by `_prepare`, not inside `_system_for`.
- **View menu.** One submenu, dev-assist's, carrying upstream's strip toggles under "Overlay
  strips". In dashboard mode the board takes the tab strip's place; the other strips still apply.
- **Settings > Audio.** Upstream's `AudioPanel`, built as `CheckedAudioPanel`
  (`exocortex/ui/audio_check.py`) with dev-assist's Check box (the Windows mute of a mic a Test
  found, the speaker-routing check). `capture.pa_guard` (the lock plus COM on the opening thread)
  wraps upstream's `_PA_LOCK` sites.
- **`[hotkeys]`.** The `move_*` nudges ship unbound (IDE shortcuts); `insights`, `recap` and
  `toggle_dashboard` are in the Shortcuts tab and in `OverlayWindow._trigger_table`.

## Next time

1. `git fetch --no-tags ../exo-cortex +main:refs/exo-cortex/main` (only reads exo-cortex).
2. `git log --first-parent --oneline c97590e..refs/exo-cortex/main` lists what is new.
3. Cherry-pick commit by commit, oldest first. A feature that arrived as a branch with an
   upstream merge in the middle goes over as one change: `git commit-tree <last>^{tree} -p <commit
   before it>` and cherry-pick that commit.
4. Resolve with the differences above, then run the full offline suite (`python -m pytest`).
   `tests/test_server.py::test_attempt_non_sycl_framework_gets_no_sysman_default` fails on a
   machine with `ZES_ENABLE_SYSMAN` set, before and after.
5. Line endings differ in two files: `tests/test_assistant_text.py` (CRLF here, LF upstream) and
   `tests/test_overlay.py` (LF here, CRLF upstream) conflict as a whole, so upstream's hunks are
   applied by hand.

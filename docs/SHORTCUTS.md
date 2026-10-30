# Shortcuts

Every keyboard shortcut and mouse gesture in ExoCortex. The hotkeys are **global**: they work
while another app, a fullscreen meeting included, has focus. They are Windows-only and exist only
while the overlay is running (`--no-ui` console mode has none). The keys below are the defaults;
each can be rebound on **Settings › Shortcuts** or in `[hotkeys]` of `config.toml` (see
[Changing a shortcut](#changing-a-shortcut)) -- to a key combination, a single-key tap or a mouse
button. What each feature does is described in the [README](../README.md).

## Overlay gone?

The **−** button at the right of the title bar (and of the slim bar) hides the overlay. It is not
a minimize button: a hidden overlay has no taskbar button and no Alt+Tab entry, so once it is
hidden only a hotkey brings it back:

- **`Ctrl+Alt+Space`** shows it again.
- **`Ctrl+Alt+Q`** also shows it and puts the cursor in the ask box.

If it still does not appear:

| What you see | Why | Fix |
|---|---|---|
| Nothing, and the overlay said **Hotkey conflict** at start | Another app owns `Ctrl+Alt+Space` | Rebind **Show / hide the overlay** on Settings › Shortcuts (or `toggle_visible` in `config.toml`, then relaunch) |
| The overlay is there but clicks go straight through it | Click-through is on | `Ctrl+Alt+C` |
| Only the answer text, no tabs or buttons | Hands-free view | `Ctrl+Alt+H` |
| One row of buttons | Completion mode's slim bar | `Ctrl+Alt+K`, or **Full** |
| Half off the edge of the screen | Moved past the edge | Drag it back by the title bar. The move hotkeys are off by default; bind `move_left` … `move_down` to use them |
| No ExoCortex at all | It was quit (`Ctrl+Alt+X` or **Quit**) | Start it with `run.vbs` or `run.bat` |

As a last resort, relaunch with `run.vbs` or `run.bat`. It closes the running copy first without
saving, so that session keeps only what it last autosaved (every 30 s).

## Global hotkeys

### Window

| Keys | Action | `[hotkeys]` key |
|---|---|---|
| `Ctrl+Alt+Space` | Show / hide the overlay (the **−** button) | `toggle_visible` |
| *(off by default)* | Move the overlay 40 px. Unbound since 2026-10-20: `Ctrl+Alt+←` `→` `↑` `↓` are VS Code's add-cursor and JetBrains' navigate-back/forward keys | `move_left`, `move_right`, `move_up`, `move_down` |
| `Ctrl+Alt+C` | Click-through on / off: the mouse passes through to the app underneath (**◈**) | `click_through` |
| `Ctrl+Alt+X` | Quit. Saves the session; there is no confirmation | `quit` |

### Answers and notes

| Keys | Action | `[hotkeys]` key |
|---|---|---|
| `Ctrl+Alt+A` | Answer the last thing the other side said | `answer_last` |
| `Ctrl+Alt+Q` | Show the overlay and focus the ask box | `ask_focus` |
| `Ctrl+Alt+T` | Auto-answer on / off (**⚡**) | `toggle_auto` |
| `Ctrl+Alt+S` | Suggestions from the screen and the recent conversation, now | `suggest_now` |
| `Ctrl+Alt+N` | Regenerate the notes now (they also refresh every 3 minutes) | `notes` |

### Co-pilot (pair programming)

See [Pair programming and team coding](PAIR_PROGRAMMING.md).

| Keys | Action | `[hotkeys]` key |
|---|---|---|
| `Ctrl+Alt+W` | Refresh the co-pilot insights now: questions to ask, plus blind spots and follow-ups, in the Notes tab's Insights pane (while the dashboard tracks the session, this refreshes the dashboard instead). Works while paused | `insights` |
| `Ctrl+Alt+Y` | Recap the session so far (goal, done, now, next) in the Assist tab. Works while paused | `recap` |
| `Ctrl+Alt+D` | Dashboard mode on / off (the **▦** button): the overlay as a board — session notes, action items (tasks, sub-tasks, prompts), goals, constraints, questions to ask, blind spots & follow-ups. See [DASHBOARD.md](DASHBOARD.md) | `toggle_dashboard` |

### Coding

| Keys | Action | `[hotkeys]` key |
|---|---|---|
| `Ctrl+Alt+G` | **Code**: write the code for the task discussed or shown since the last checkpoint. Works while paused | `code` |
| `Ctrl+Alt+B` | **Checkpoint**: solve, Code and completions start fresh from here. The transcript, notes and answer history keep the whole meeting | `checkpoint` |
| `Ctrl+Alt+E` | Explanations on / off (**💡**) | `toggle_explanations` |
| `Ctrl+Alt+K` | Completion mode on / off (**⌨**): the slim bar plus ghost-text completions where you type | `toggle_completion` |
| `Ctrl+Alt+H` | Hands-free mode on / off (the switch at the right of the tabs) | `toggle_handsfree` |
| `Ctrl+Alt+J` | Manual advance on / off (the **Manual** switch next to Hands-free): nothing is generated until the mouse gesture, anywhere on screen — hold the left button for **Text**, click the middle button (or three-finger-tap the touchpad) for **Code**, click mouse button 4 (the rear thumb button) for **Code from scratch**. Transcript, screen reading and checkpoints keep running | `toggle_manual` |

### Screen, mic and pause

| Keys | Action | `[hotkeys]` key |
|---|---|---|
| `Ctrl+Alt+R` | Drag a screen region to read (**Region…**) | `select_region` |
| `Ctrl+Alt+V` | Pause / resume screen reading only | `toggle_screen` |
| `Ctrl+Alt+M` | Mute / unmute your mic transcription (**🎤**) | `toggle_mic` |
| Tap **Right Shift** | Pause / resume everything automatic: screen reading, auto-answer, auto-solve and hands-free (**Pause**). A toast confirms | `toggle_pause` |

A *tap* means the key is pressed and released within 0.6 s with no other key in between, soxxx
holding Shift for a capital letter never triggers it and typing keeps working.

## Inside the overlay

These work only in the place listed, not globally.

| Where | Key or mouse | Action |
|---|---|---|
| Ask box | `Enter` | Send the question |
| Ask box | `Esc` | Clear the box |
| Region picker (`Ctrl+Alt+R`) | Drag | Select the region |
| Region picker | `Enter` or double-click | Accept the region |
| Region picker | `Esc` or right-click | Cancel |
| Live tab | Double-click a speaker name | Rename the speaker |
| Title bar or slim bar | Drag | Move the overlay |
| Window edges or bottom-right corner | Drag | Resize the overlay |
| Anywhere on the panel except text | Right-click | The **⋯** menu: **Assist ▸** (questions, blind spots, recap and the dashboard now; the automatic switches; role; session context; Assist settings…), **View ▸** (switch the view; the current view's display options — the board's columns, the OCR readout, the history pane, partial speech; the **Overlay strips** — Tabs, Controls, Status, Monitor; **Screen source ▸** off / active window / a monitor / a region; **Screen: pause/resume**; **Repository ▸** auto-detected from Claude Code or a chosen folder), **Dashboard mode**, Export session, New session, Settings…, Models…, Context documents…, **Capabilities…**, Compact context now, Rename speakers, Quit |
| Notes tab | **Questions** / **Blind spots** / **Recap** buttons | Refresh one insight section now, or recap the session |
| Dashboard mode | **✓** / **✕** / **↺** / **⧉** / **＋** on a card, or right-click a card | Mark an item answered, done or met (with an optional note), dismiss it (a constraint: lift it), reopen it, copy a prompt, add a sub-task under a task; the menu also has **Edit…** and, on a task, **Add an acceptance criterion…** / **Add a prompt…** |
| Dashboard mode | **+** / **↻** in a column header, **Refresh all** | Add an item by hand (a goal asks which task it is for); refresh that column, or everything (the items and the notes) |
| Text panes and the ask box | Right-click | The usual text menu (Copy, Select All, …) plus **Hide overlay** and **Quit ExoCortex** |
| Hands-free view, over the answer or the expanded history | Mouse wheel | Scroll it, even though the window is click-through there |
| Hands-free view, over the expanded history | Tap `↑` / `↓` | Jump to the previous / next answer, marked with an accent bar. The arrow keys work normally everywhere else |
| Manual advance on, anywhere on screen (any app) | Hold the **left** mouse button `manual_hold_s` (0.8 s) — `manual_text` | Generate **Text** for the current task. Moving the pointer while the button is down cancels; the click still reaches the app underneath; the overlay stays in the background |
| Manual advance on, anywhere on screen | **Click** the **middle** button (press the scroll wheel) — or **three-finger-tap** a touchpad whose tap is mapped to "Middle mouse button" in Windows touchpad settings — `manual_code` | Generate **Code** for the current task, working with the code already on screen when **Consider given code** (Settings › Assist) is on. No hold needed by default |
| Manual advance on, anywhere on screen | **Click mouse button 4** (the rear thumb button) — `manual_code_fresh` | Generate **Code from scratch**: the code already on screen is ignored, straight to the best solution. No hold needed by default; the press is kept from the app underneath so a browser does not go *Back* (`swallow_thumb_buttons`). Whether the three gestures require the press-and-hold is set on the Hands-free settings tab (`manual_hold_text` / `manual_hold_code` / `manual_hold_code_fresh`); the buttons themselves on the Shortcuts tab |

These have a button or menu entry but no hotkey: **Read**, **Revert**, **Reset**, **Text**,
**Compact** (fold the context window into a running summary now, like `/compact`; also in the
**⋯** menu as "Compact context now"), switching tabs, the Notes tab's **Questions** / **Blind
spots** buttons (`Ctrl+Alt+W` refreshes both), and the **⋯** menu's Assist switches, role and
session context, Export session, New session, Settings…, Models…, Context documents…,
**Capabilities…** (the Models window's Capabilities tab: what the model may read, edit, run,
fetch and call on its own, and the project maps) and Rename speakers.

## View menu — show or hide parts of the panel

Besides switching the view, the **⋯** menu's **View** submenu toggles which strips of the full
overlay are shown (under **Overlay strips**); each choice is saved to `[ui]` in `config.toml`. The
title bar (drag, toggles, menu, Hide) is always present, so the panel can never be made
unreachable.

| Entry | Hides | `[ui]` key |
| --- | --- | --- |
| **Tabs** | The top strip: the Live / Assist / Notes / Screen tabs and the Manual / Hands-free switches. The content stack keeps whatever tab was last selected | `show_tabs` |
| **Controls** | The bottom button row (Read, Revert, Reset, Pause, Checkpoint, Compact, Text, Code) | `show_controls` |
| **Status** | The bottom status strip (audio levels, ASR / LLM state, capture and screen status) | `show_status` |
| **Monitor** | *Adds* a section at the bottom that visually monitors the local model's performance (time-to-first-token, tokens/second), the context window and its contents (how the per-request budget is filled by source, plus the running summary and document count), and resource use. Off by default | `show_monitor` |

The View choices apply to the full overlay: completion mode's slim bar owns its own layout, the
hands-free view shows only its output, the title bar and the Manual / Hands-free switches, and in
dashboard mode the board takes the tab strip's place while Controls, Status and Monitor still
apply.

### Configuring the Monitor

**Settings → Monitor** chooses which readouts the Monitor section shows (`[monitor]` in
`config.toml`): LLM performance, the context window and its contents, and the utilisation of the
**CPU**, **system RAM**, the dedicated **GPU** (with VRAM in use), the integrated **iGPU**, and
the **NPU**. The GPU / iGPU / NPU rows hide automatically when the machine has no such device, or
when Windows does not expose its performance counter (the NPU counter is only present on recent
Windows builds). GPU, iGPU and NPU utilisation and system RAM come from Windows performance
counters; dedicated-GPU VRAM uses the GPU driver (NVML on NVIDIA).

## Focus

The shortcuts never move the keyboard focus. Every hotkey, tap and mouse gesture is read by Windows
or by a low-level hook, never by the overlay, so the app you are working in keeps the keyboard:
nothing is brought to the front, and the overlay is shown or updated without being activated. Two
shortcuts exist to take the keyboard and are the exceptions: `Ctrl+Alt+Q` puts the cursor in the ask
box, and the region picker (`Ctrl+Alt+R`) covers the desktop -- on Windows it reads its own Esc and
Enter through a hook, so your window stays focused even then; elsewhere it takes the focus and gives
it back when it closes.

Clicking the panel follows the same rule. A left click on a button, a switch or a tab works without
moving the focus; the ask box, the text panes (typing, selecting, Ctrl+C), the title bar and the
slim bar (dragging), the edges and the sliders take it as any window would, and so does a
right-click (the menus need it). A middle or thumb-button click on the panel never takes it. In the
hands-free view a click that lands on the click-through content never activates the panel either.

## Changing a shortcut

**Settings › Shortcuts** lists every shortcut -- the global hotkeys above and the three manual-advance
gestures -- with its current binding in words. **Set** listens for the next key combination or mouse
button, anywhere on screen: a combination is taken half a second after the last key (keep the
modifiers down until you have pressed the key), a single key or a modifier on its own becomes a
tap, a mouse button a mouse binding; Esc cancels. **Default** puts one shortcut back. **Apply** /
**OK** re-register the shortcuts at once -- no restart -- and save `config.toml`; a binding used
twice, or one the grammar rejects, is flagged on the tab and not applied. The same values live in
`[hotkeys]` of `config.toml` for editing by hand:

- A binding is `+`-separated and case-insensitive: any of `ctrl`, `alt`, `shift`, `win`, plus
  exactly one key or mouse button. The key can be a letter, a digit, `f1` to `f12`, `space`,
  `enter`, `esc`, `tab`, `backspace`, `insert`, `delete`, `home`, `end`, `pageup`, `pagedown`,
  `left`, `up`, `right`, `down`, `grave` (the `` ` `` key), `backslash`, or one of `-` `=` `[` `]`
  `;` `'` `,` `.` `/`.
- A single key with no modifier -- `rshift` (the default pause key), `lctrl`, `ralt`, `win`, or a
  plain key such as `f9` or `right` -- is a **tap**: pressed and released within 0.6 s with no other
  key in between, so holding Shift for a capital letter never triggers it. A plain key is swallowed
  while no modifier is held (`ctrl+right` still reaches the app).
- A mouse button -- `mouse_left`, `mouse_middle`, `mouse_right`, `mouse4` and `mouse5` (the thumb
  buttons: 4 is *Back*, 5 is *Forward*), alone or with modifiers (`ctrl+mouse4`) -- fires on a
  click anywhere on screen with exactly those modifiers held. The manual-advance gestures are the
  only ones with a press-and-hold (the Hands-free tab decides which). A bound thumb button is kept
  from the app underneath (`swallow_thumb_buttons`, so a browser does not go *Back*); a modified
  click is kept too; a bare left / middle / right press always reaches the app -- which is why a
  bare `mouse_left` on a global shortcut gets a warning on the tab.
- `""` unbinds a shortcut.
- If another app already owns a combination, it cannot be registered. The overlay shows **Hotkey
  conflict: …** and that shortcut does nothing until you choose another. A binding used twice is
  reported the same way; the first owner keeps it.
- Every hotkey is system-wide, so it shadows the same combination in your IDE. Several defaults
  collide with JetBrains shortcuts (`Ctrl+Alt+V` / `M` / `C` / `N` refactorings, `Ctrl+Alt+T`
  Surround With, `Ctrl+Alt+B` Go to Implementation). On layouts with **AltGr**, AltGr is
  `Ctrl+Alt`, so `Ctrl+Alt+Q` swallows the `@` of a German keyboard. Turn off what you don't use
  and move the rest to combinations your IDE leaves free; see
  [Pair programming](PAIR_PROGRAMMING.md#hotkeys-and-your-ide).

# Shortcuts

Every keyboard shortcut and mouse gesture in ExoCortex. The hotkeys are **global**: they work
while another app, a fullscreen meeting included, has focus. They are Windows-only and exist only
while the overlay is running (`--no-ui` console mode has none). The keys below are the defaults;
each can be rebound in `[hotkeys]` of `config.toml` (see [Changing a hotkey](#changing-a-hotkey)).
What each feature does is described in the [README](../README.md).

## Overlay gone?

The **−** button at the right of the title bar (and of the slim bar) hides the overlay. It is not
a minimize button: a hidden overlay has no taskbar button and no Alt+Tab entry, so once it is
hidden only a hotkey brings it back:

- **`Ctrl+Alt+Space`** shows it again.
- **`Ctrl+Alt+Q`** also shows it and puts the cursor in the ask box.

If it still does not appear:

| What you see | Why | Fix |
|---|---|---|
| Nothing, and the overlay said **Hotkey conflict** at start | Another app owns `Ctrl+Alt+Space` | Rebind `toggle_visible`, then relaunch |
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
| `Ctrl+Alt+D` | Dashboard mode on / off (the **▦** button): the overlay as a board — session notes, questions to ask, blind spots & follow-ups, next steps & prompts. See [DASHBOARD.md](DASHBOARD.md) | `toggle_dashboard` |

### Coding

| Keys | Action | `[hotkeys]` key |
|---|---|---|
| `Ctrl+Alt+G` | **Code**: write the code for the task discussed or shown since the last checkpoint. Works while paused | `code` |
| `Ctrl+Alt+B` | **Checkpoint**: solve, Code and completions start fresh from here. The transcript, notes and answer history keep the whole meeting | `checkpoint` |
| `Ctrl+Alt+E` | Explanations on / off (**💡**) | `toggle_explanations` |
| `Ctrl+Alt+K` | Completion mode on / off (**⌨**): the slim bar plus ghost-text completions where you type | `toggle_completion` |
| `Ctrl+Alt+H` | Hands-free mode on / off (the switch at the right of the tabs) | `toggle_handsfree` |
| `Ctrl+Alt+J` | Manual advance on / off (the **Manual** switch next to Hands-free): nothing is generated until the mouse gesture, anywhere on screen — hold the left button for **Text**, click the middle button (or three-finger-tap the touchpad) for **Code**. Transcript, screen reading and checkpoints keep running | `toggle_manual` |

### Screen, mic and pause

| Keys | Action | `[hotkeys]` key |
|---|---|---|
| `Ctrl+Alt+R` | Drag a screen region to read (**Region…**) | `select_region` |
| `Ctrl+Alt+V` | Pause / resume screen reading only | `toggle_screen` |
| `Ctrl+Alt+M` | Mute / unmute your mic transcription (**🎤**) | `toggle_mic` |
| Tap **Right Shift** | Pause / resume everything automatic: screen reading, auto-answer, auto-solve and hands-free (**Pause**). A toast confirms | `toggle_pause` |

A *tap* means the key is pressed and released within 0.6 s with no other key in between, so
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
| Anywhere on the panel except text | Right-click | The **⋯** menu: **Assist ▸** (questions, blind spots, recap and the dashboard now; the automatic switches; role; session context; Assist settings…), **Dashboard mode**, Export session, New session, Settings…, Models…, Rename speakers, Screen region…, Screen: pause/resume, Quit |
| Notes tab | **Questions** / **Blind spots** / **Recap** buttons | Refresh one insight section now, or recap the session |
| Dashboard mode | **✓** / **✕** / **↺** / **⧉** on a card, or right-click a card | Mark an item answered or done (with an optional answer), dismiss it, reopen it, copy a prompt; the menu also has **Edit…** |
| Dashboard mode | **+** / **↻** in a column header, **Refresh all** | Add an item by hand; refresh that column, or everything (the items and the notes) |
| Text panes and the ask box | Right-click | The usual text menu (Copy, Select All, …) plus **Hide overlay** and **Quit ExoCortex** |
| Hands-free view, over the answer or the expanded history | Mouse wheel | Scroll it, even though the window is click-through there |
| Hands-free view, over the expanded history | Tap `↑` / `↓` | Jump to the previous / next answer, marked with an accent bar. The arrow keys work normally everywhere else |
| Manual advance on, anywhere on screen (any app) | Hold the **left** mouse button `manual_hold_s` (0.8 s) | Generate **Text** for the current task. Moving the pointer while the button is down cancels; the click still reaches the app underneath; the overlay stays in the background |
| Manual advance on, anywhere on screen | **Click** the **middle** button (press the scroll wheel) — or **three-finger-tap** a touchpad whose tap is mapped to "Middle mouse button" in Windows touchpad settings | Generate **Code** for the current task. No hold needed by default; whether Text / Code require the press-and-hold is set on the Hands-free settings tab (`manual_hold_text` / `manual_hold_code`) |

These have a button or menu entry but no hotkey: **Read**, **Revert**, **Reset**, **Text**,
switching tabs, the Notes tab's **Questions** / **Blind spots** buttons (`Ctrl+Alt+W` refreshes
both), and the **⋯** menu's Assist switches, role and session context, Export session, New
session, Settings…, Models… and Rename speakers.

## Changing a hotkey

Edit `[hotkeys]` in `config.toml` and restart ExoCortex. Hotkeys are registered once at start,
and the Settings page does not list them.

- A binding is `+`-separated and case-insensitive: any of `ctrl`, `alt`, `shift`, `win`, plus
  exactly one key. The key can be a letter, a digit, `f1` to `f12`, `space`, `enter`, `esc`,
  `tab`, `backspace`, `insert`, `delete`, `home`, `end`, `pageup`, `pagedown`, `left`, `up`,
  `right`, `down`, `grave` (the `` ` `` key), `backslash`, or one of `-` `=` `[` `]` `;` `'` `,`
  `.` `/`.
- `toggle_pause` can instead be a single key, used as a tap: `rshift` (the default), `lshift`,
  `shift`, `rctrl`, `lctrl`, `ctrl`, `ralt`, `lalt`, `alt`, `rwin`, `lwin`, `win`, or a plain key
  such as `right`, which is then swallowed while no modifier is held. `""` turns it off.
- If another app already owns a combination, it cannot be registered. The overlay shows
  **Hotkey conflict: …** at start, and that hotkey does nothing until you choose another.
- Give each action its own combination. If two share one, only one of them works.
- A hotkey set to `""` is off: it is not registered at all.
- Every hotkey is system-wide, so it shadows the same combination in your IDE. Several defaults
  collide with JetBrains shortcuts (`Ctrl+Alt+V` / `M` / `C` / `N` refactorings, `Ctrl+Alt+T`
  Surround With, `Ctrl+Alt+B` Go to Implementation). On layouts with **AltGr**, AltGr is
  `Ctrl+Alt`, so `Ctrl+Alt+Q` swallows the `@` of a German keyboard. Turn off what you don't use
  and move the rest to combinations your IDE leaves free; see
  [Pair programming](PAIR_PROGRAMMING.md#hotkeys-and-your-ide).

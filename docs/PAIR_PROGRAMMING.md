# Pair programming and team coding

dev-assist is tuned for a coding session where **you drive**: you are at the keyboard, teammates
talk it through with you (over a call, or in the room), and the co-pilot keeps track of what you
are too heads-down to catch. Everything runs on your machine, in an ordinary window you can share.

What it does **by itself**, without ever taking over the panel while you type:

- **Questions to ask.** A short, living list of questions worth putting to the navigator or the
  team right now: a requirement to clarify, an assumption to pin down, a design choice or edge
  case to probe, something said but never settled. Phrased the way you would say it, with the
  real function, file and service names.
- **Blind spots & follow-ups.** Risks nobody has raised: an unhandled edge case or error path, a
  missing test, a concurrency, performance, security or data-loss concern, a breaking change for
  callers, an unverified assumption. Plus the follow-ups that must not be forgotten: a TODO,
  someone to loop in, a ticket to file.
- **Dev-session notes.** The same five headings as before (Summary, Key points, Decisions, Action
  items, Open questions), steered toward a working session: code changed, bugs and their root
  causes, design trade-offs, commands and config, TODOs.
- **Answers to spoken questions.** When a teammate asks you something, a ready-to-say answer
  streams into the Assist tab, as before.

What it does **on demand**: a **Recap** of where the session stands (goal, done, now, next),
an immediate refresh of the questions or blind spots, and the existing **Code**, **Text** and
ask box.

The questions and blind spots live in the **Notes tab's Insights pane**, above the running notes.
A background refresh never switches tabs and never touches the Assist tab. When one lands while
you are elsewhere, the Notes tab shows a dot (**Notes •**) until you look.

## Set up a session

1. **Session context.** In **⋯ > Assist > Session context…**, say what you are working on, for
   example `FastAPI + Postgres billing service; today: per-tenant rate limiting`. Every answer,
   insight and recap uses it, so suggestions use your real names and constraints.
2. **Role.** **⋯ > Assist > Role** is **Driver** by default. Switch to **Navigator** when a
   teammate takes the keyboard: questions are then for the driver, and blind spots read as review
   points on the code being written. **Participant** fits a design review or planning discussion.
   Pairs swap often, so this is one click.
3. **Speakers.** Double-click a speaker on the Live tab to give them a name. Questions and notes
   then use it.
4. **Screen.** On the Screen tab, **Active window** reads whatever you are working in (the editor,
   a terminal, a ticket). The insights use it: blind spots come from the code on screen as well as
   from the conversation.

## How the background insights work

A background refresh is one local model call that rewrites the living list. It keeps every item
still open, drops whatever the conversation has since answered or settled, and adds what is new.

A refresh starts when both of these hold:

- at least `insight_interval_s` (120 s) have passed since the last one, and
- something is new: `insight_min_new_words` (60) words of speech, or the code on screen has
  changed substantially. The second trigger means you also get blind spots while coding in silence.

It is the lowest-priority work the co-pilot does. Any other request, such as a spoken-question
answer, Code, Text or Recap, cancels a running background refresh, which is retried about 30 s
later. A background refresh never replaces one you asked for. Nothing automatic runs while you are
paused (**Pause** or a tap of **Right Shift**), in manual advance, or in completion mode.

Everything on demand works regardless of those switches, including while paused:

| Want | Do |
|---|---|
| Fresh questions and blind spots now | `Ctrl+Alt+W`, or **⋯ > Assist > Suggest questions now** |
| Just one of the two | The **Questions** or **Blind spots** button on the Notes tab |
| Where the session stands | `Ctrl+Alt+Y`, the **Recap** button on the Notes tab, or **⋯ > Assist > Recap the session** |
| The notes, refreshed now | `Ctrl+Alt+N`, or **Regenerate** on the Notes tab |

## The Assist menu

**⋯ > Assist** holds everything that changes during a session:

| Entry | What it does |
|---|---|
| Suggest questions now / Blind spots & follow-ups now / Recap the session / Regenerate notes | Run that now, and show the tab it lands in |
| Suggest questions automatically | Background refreshes of the questions (`[assist].suggest_questions`) |
| Flag blind spots & follow-ups automatically | Background refreshes of the blind spots (`[assist].flag_blind_spots`) |
| Auto-answer spoken questions | Same as the **⚡** button (`Ctrl+Alt+T`) |
| Solve / suggest when the screen changes | `[screen].auto_suggest`, off by default (see below) |
| Answer questions that appear on screen | `[screen].answer_screen_questions`, off by default |
| Role | Driver, Navigator or Participant |
| Session context… | Set or clear the session context |
| Assist settings… | The settings dialog, opened on its Assist tab |

## Settings

The Settings dialog's **Assist** tab carries the new fields after the persona. A new **Notes** tab
holds the notes settings. In `config.toml`:

| Key | Default | Meaning |
|---|---|---|
| `[assist] session_context` | `""` | What the session is about; handed to every prompt |
| `[assist] role` | `"driver"` | `driver`, `navigator` or `participant` |
| `[assist] suggest_questions` | `true` | Background refreshes of the questions |
| `[assist] flag_blind_spots` | `true` | Background refreshes of the blind spots and follow-ups |
| `[assist] insight_interval_s` | `120` | Minimum seconds between background refreshes; `0` means on demand only |
| `[assist] insight_min_new_words` | `60` | New speech needed for a refresh; a big change of the code on screen also counts |
| `[assist] insight_max_items` | `4` | Bullets per section, up to 8 |
| `[notes] template` | `"dev"` | `dev` steers the notes toward a working session; `meeting` restores the original wording |
| `[llm] insights_max_tokens` | `320` | Token budget of one insights refresh |
| `[llm] recap_max_tokens` | `360` | Token budget of a recap |
| `[hotkeys] insights` | `"ctrl+alt+w"` | Refresh questions and blind spots now |
| `[hotkeys] recap` | `"ctrl+alt+y"` | Recap now |

`insight_interval_s` is the GPU-cost knob. Each refresh re-reads the recent transcript and the
screen, which takes several seconds of prompt processing on a laptop GPU. Raise the interval if
answers to spoken questions feel slower than you would like, or set it to `0` and use `Ctrl+Alt+W`.

## Defaults changed for driving

These defaults changed on 2026-10-20. An existing `config.toml` keeps whatever values it already
has, so change them there if you want the new behaviour.

| Setting | Now | Why |
|---|---|---|
| `[screen] auto_suggest` | `false` | While you type, the editor changes constantly, and every change would start an unrequested solve that takes over the Assist tab. Turn it back on for an interview or coding-challenge screen |
| `[screen] answer_screen_questions` | `false` | A comment, docstring or log line that reads as a question would take over the Assist tab. Spoken questions are still answered |
| `[hotkeys] move_left` … `move_down` | `""` (off) | `Ctrl+Alt+arrows` are VS Code's add-cursor-above/below and JetBrains' navigate back/forward. A global hotkey swallows them in every app. Drag the title bar instead |
| `[notes] template` | `"dev"` | New key; see above |

`[screen] exclude_own_windows` (on) now does real work. The overlay is an ordinary window that
screen capture sees, so the screen reader blanks this app's own windows (the panel, menus, ghost
text) out of every frame before OCR. Without that, the co-pilot would read its own suggested
questions back as screen content.

## Hotkeys and your IDE

The global hotkeys use `Ctrl+Alt+<key>` and are registered system-wide, so each one shadows the
same combination in your editor. Several defaults collide with JetBrains IDE shortcuts, for
example `Ctrl+Alt+V`, `M`, `C` and `N` (the Extract and Inline refactorings), `Ctrl+Alt+T`
(Surround With) and `Ctrl+Alt+B` (Go to Implementation). On keyboard layouts with **AltGr**
(German, French, Polish and others), AltGr is `Ctrl+Alt`, so `Ctrl+Alt+Q` also swallows the `@`
of a German keyboard.

Set any hotkey you don't use to `""` (it is then not registered), and move the rest to
combinations your IDE leaves free. Check its keymap first: **File > Settings > Keymap** in
JetBrains, `Ctrl+K Ctrl+S` in VS Code. See [SHORTCUTS.md](SHORTCUTS.md#changing-a-hotkey).

## Session folder

Next to `transcript.md`, `notes.md` and `answers.md`, each session folder now gets an
`insights.md` with the last questions and blind spots, once there are any. A recap is saved in
`answers.md` like any other answer.

## Limits

- The insights are only as good as the local model. The prompt forbids generic questions ("any
  concerns?") and anything the transcript already answers, but a small model will still produce
  some filler. Larger models do noticeably better.
- If you pair in the same room on one microphone, everyone is transcribed as **Me**. The insights
  and notes still work. Spoken questions are only auto-answered when they come from the call
  audio, so in that setup ask through the ask box (`Ctrl+Alt+Q`) instead.
- This feature set is covered by offline unit tests against a scripted model client. It has not
  yet been tuned against a real model on every supported GPU, so treat the default intervals as a
  starting point.

# Dashboard mode

Dashboard mode turns the overlay into a board you can leave open on a second screen while you
pair: six columns, side by side, kept current without you touching anything. Any column can be
hidden (**⋯ › View**).

| Column | What it holds | Where it comes from |
|---|---|---|
| **Session notes** | The running notes: summary, key points, decisions, open questions. While the board is on, the notes keep the narrative only; the tasks, goals, constraints and questions live in the columns next to it | The Notes tab's notes job, on its own cadence (`[notes].auto_interval_s`) |
| **Action items** | The work, as a tree: each **primary task** the brief or the discussion asked for (bold), its **sub-tasks** indented under it, and under the item they serve the **suggested prompts** to type into Claude Code, exactly as you would type them. A task shows how many of its steps are done | A dashboard refresh |
| **Goals** | The acceptance criteria: how we will know an action item is done, grouped under the task each is for (**Whole session** for session-wide ones). Marked **met** with the evidence | A dashboard refresh |
| **Constraints** | The rules and guardrails in force: stack and versions, the time box, "no new dependencies", diff limits, data and safety rules, what to ask about first. A constraint is never "done": it is in force until lifted | A dashboard refresh |
| **Questions to ask** | Questions worth putting to the navigator or the team right now. Once the conversation answers one, it moves under **Answered (n)** with the answer | A dashboard refresh |
| **Blind spots & follow-ups** | Risks nobody raised, missing tests, assumptions to verify, follow-ups not to forget. Tracked the same way | A dashboard refresh |

Toggle it with the **▦** button at the left of the title-bar icons, `Ctrl+Alt+D`, **⋯ ›
Dashboard mode** or **⋯ › View › Dashboard**. The window widens to `[dashboard].width` (1180 px by
default, growing leftwards from its right edge and never past the screen); the ask box, the
bottom controls and the status bar stay, so asking, pausing, **Text** and **Code** keep working.
Leaving the mode restores the window exactly as it was. The slim bar (completion mode) and the
board cannot share the window: entering one leaves the other.

## Where the data lives

The tracked items are not something the model "remembers". They live in a small database,
`dashboard.sqlite` in the session folder (or in memory with `[dashboard].persist = false`), and
the board is also written next to the notes as `dashboard.md` after every change. Each refresh
hands the model only what it needs: the **open** items of each list, each under a short id
(`N1` action items -- sub-tasks indented under their task -- `G1` goals, `C1` constraints, `Q1`
questions, `B1` blind spots, `P1` prompts), what was said **since the last refresh**, a little
earlier context, and the screen. The model answers with the **changes** only:

```
## Action items
- NEW: Task 1: the segment evaluation endpoint
- NEW under N1: parse the filter expression
- DONE N2: the parser is in
## Goals
- NEW for N1: attribute filters return the matching customer ids
- MET G1: the unit test passes
## Constraints
- NEW: TypeScript on Node 22, no new dependencies
## Questions to ask
- ANSWERED Q2: yes, the cache is keyed by tenant id
## Blind spots & follow-ups
- DROP B1
## Suggested prompts
- NEW for N1: /recon segment evaluation: filters, customers, the in-memory store
```

A new action item takes the next free number, so a sub-task, goal or prompt written right after
it can name it (`NEW under N4: …` when `N1`–`N3` were shown). The store applies the changes: new
items are added (duplicates of an open item are folded, an item already answered is not raised
again), answered, done or met items keep their note, dropped ones disappear. Dismissing a task
dismisses its open sub-tasks, goals and prompts; finishing a task finishes its open sub-tasks
and retires their pending prompts (its goals wait for their own **met**). Anything the model did
not mention stays open, so a cut-off answer never loses an item. That is what keeps each call
short on a laptop GPU: the prompt carries a few dozen lines of items instead of the whole
history, and the answer is a handful of lines.

## When it refreshes

With **Track questions, blind spots & next steps in the background** on (the default), a
refresh runs when both hold:

- at least `[dashboard].refresh_interval_s` (90 s) have passed since the last one, and
- something is new: `[dashboard].min_new_words` (40) words of speech, or the code on screen
  changed substantially.

It is the lowest-priority work the co-pilot does, like the Notes tab's insights passes it
replaces: a spoken-question answer, **Code**, **Text** or a recap cancels a running background
refresh, which is retried about 30 s later. Nothing automatic runs while paused, in manual
advance or in completion mode. While tracking is on, the Notes tab's **Insights** pane and the
session's `insights.md` show the board's open questions and blind spots, so nothing runs twice.

On demand, any time, including while paused: **Refresh all** on the board (the items plus the
notes), the **↻** in a column header (that column only -- the Goals column also shows the model
the action items, since a goal names one), **⋯ › Assist › Refresh the dashboard now**, or the
Notes tab's **Questions** / **Blind spots** buttons and `Ctrl+Alt+W`, which refresh the board
while it tracks.

Screen capture and OCR are not part of this. They keep the cadence set on the Screen tab
(`[screen].interval_s`) in every view mode; a refresh reads whatever the screen reader last saw.

## Working the board

| Want | Do |
|---|---|
| Mark a question answered (and note the answer) | **✓** on the card, or right-click › **Mark answered…** |
| Mark a task or step done, a goal met | **✓** on the card, or right-click › **Mark done…** / **Mark met…** |
| Lift a constraint that no longer applies | **✕** on the card, or right-click › **Lift** |
| Drop an item | **✕**, or right-click › **Dismiss** (a task takes its open steps, goals and prompts with it) |
| Bring an answered item back | **↺** under **Answered (n)** / **Done (n)** / **Met (n)** |
| Reword an item | Right-click › **Edit…** |
| Copy a prompt to paste into Claude Code | **⧉** on the card, or right-click › **Copy** |
| Add a sub-task under a task | **＋** on the task's card, or right-click › **Add a sub-task…** |
| Add an acceptance criterion or a prompt for a task | Right-click the task › **Add an acceptance criterion…** / **Add a prompt…** |
| Add an item by hand | **+** in the column header. In Action items, text starting with `/` or "Use the …" becomes a prompt; in Goals you are asked which task it is for |
| Refresh one column | **↻** in its header |
| Hide or show a column, the prompts, the done items | **⋯ › View** while the board is on |

Answered, done and met items stay visible, collapsed, up to `[dashboard].resolved_keep` per
column (`show_resolved` hides them altogether).

## The View menu

**⋯ › View** (2026-10-21) always opens on the view you are in and shows that view's own
options first: the board's six columns (each one hideable), the prompts under the action items
and the collapsed done items; the Screen tab's OCR readout; the Assist tab's history pane and
the explanations switch; the Live tab's in-progress speech. Below them, the switches every view
shares:

- **Overlay strips** -- Tabs, Controls, Status, Monitor: which strips of the full overlay show
  (`[ui].show_*`, from exo-cortex; see [SHORTCUTS.md](SHORTCUTS.md#view-menu--show-or-hide-parts-of-the-panel)).
  On the board the tab strip stays hidden either way; Controls, Status and Monitor apply.
- **Screen source** -- Off, the active window, one of the **monitors** (listed by number, size
  and position; `[screen].monitor`) or a region. The same thing as the Screen tab's mode row.
- **Screen: pause/resume**.
- **Repository** -- which project the lookups and the toolkit discovery use (next section): the
  one auto-detected from Claude Code, a folder you choose, and what was found.

The first entries switch the view itself: Live, Assist, Notes, Screen, Dashboard.

## Claude Code toolkit awareness

If a Claude Code toolkit is installed in the repository you are working on, the suggested
prompts are written with it: every prompt opens with one of its slash commands or "Use the
<agent> agent on …", named exactly as installed, with the task's real details as arguments and
the constraints and criteria that matter carried in; a prompt that names a command or agent that
is not installed is thrown away before it reaches the board. The model is also asked for one
prompt per open task that has none yet, so the Action items column reads as a plan you can type.

dev-assist discovers the install by itself: it reads the skills, agents, rules, workflows and
hooks under `<repo>/.claude/` and the house rules and detected commands in `CLAUDE.local.md`,
builds a compact catalog from the files' own frontmatter, and re-checks every
`[project].rescan_s` seconds, so a `node "$HOME/claude-toolkit/install.mjs" --set live` run
mid-session is picked up within half a minute (a toast says what was found). Nothing about any
particular skill is hard-coded: an updated toolkit describes itself, and the example command the
model is shown is the first skill the toolkit's own working loop names.

The catalog names each part exactly as it is typed or used (`/recon [topic]`, "Use the reviewer
agent …"), with its one-line description, the repo's test and lint commands, and the working
loop from the house rules. It sits at the end of the system prompt, so it is prefix-cached, and
the model is told to name only what is listed. When the conversation or the screen mentions a
part (`/scope`, "the verifier agent"), that part's full text goes along too, so the suggested
prompt follows its actual steps.

### Which repository

You need not set anything: with `[project].repo_auto` on (the default) dev-assist reads Claude
Code's own session logs (`~/.claude/projects`) and uses the directory of its most recent session
-- the project you are working in -- re-checked every few seconds, so starting `claude` in the
scenario's folder is all it takes for its toolkit to be found. Only sessions active within
`repo_auto_max_age_h` (12 h) count, and scratch sessions under the temp directory never do.
**⋯ › View › Repository** shows what was found and lets you pin a folder instead; the
Settings dialog's **Project** tab holds the same: **Repository root** (explicit, wins over the
detection), the auto-detect switch and its age limit, **Claude Code toolkit context**, the
on-mention lookup, whether answers / ask / recap get the catalog too (not only the dashboard),
the opt-in `~/.claude` user layer, the catalog's size, and the optional toolkit clone
(`~/claude-toolkit` is found by itself) whose `modules/index.json` supplies fuller descriptions
for the parts installed. Parts that are not installed are never listed.

## Repository lookups

The same **Project** tab holds **Look up files / identifiers mentioned**. With a repository
known, a refresh (and an answer, an ask, a recap) searches it for the identifier-shaped names the
conversation, the question or the screen mention (`insights_now`, `ScreenReader`, `billing.py`,
a path), and hands over only the matching lines, within `repo_max_files` / `repo_max_chars`.
Nothing is read wholesale and nothing is sent when nothing identifier-like was mentioned or
nothing matched. The scan is bounded (about a second per refresh, less for an answer) and skips
`.git`, `node_modules`, `.venv`, build output and binaries (`[project].repo_exclude`). Inline
completions never trigger a lookup (latency).

The slim bar's **Context ▾** menu switches both lookups too (**Repository lookup**, **Toolkit
catalog**), next to the completion sources.

## Settings

The **Dashboard** tab of the Settings dialog, and `config.toml`:

| Key | Default | Meaning |
|---|---|---|
| `[dashboard] enabled` | `false` | Start in dashboard mode (mirrors the toggle; persisted) |
| `[dashboard] track` | `true` | Background refreshes; replaces the Notes tab's insights passes while on |
| `[dashboard] refresh_interval_s` | `90` | Minimum seconds between background refreshes; `0` = on demand only |
| `[dashboard] min_new_words` | `40` | New speech needed for a refresh; a big change of the code on screen also counts |
| `[dashboard] max_open_items` | `8` | Open items per column handed to the model (the oldest first; twice that for action items and goals) |
| `[dashboard] max_new_items` | `4` | New items the model may add per column (and per task) per refresh |
| `[dashboard] suggest_prompts` | `true` | Suggested prompts for Claude Code under the action items |
| `[dashboard] show_resolved` / `resolved_keep` | `true` / `12` | Answered / done / met items stay visible, collapsed, this many per column |
| `[dashboard] show_notes`, `show_nextsteps`, `show_goals`, `show_constraints`, `show_questions`, `show_blindspots` | `true` | Which columns the board shows (also **⋯ › View**) |
| `[dashboard] show_prompts` | `true` | The suggested prompts under their action items |
| `[dashboard] width` | `1180` | Window width in dashboard mode; resizing the board updates it |
| `[dashboard] view_refresh_s` | `5` | How often the board re-renders its timestamps (no model call) |
| `[dashboard] persist` | `true` | Keep the items in `dashboard.sqlite` in the session folder (next session) |
| `[llm] dashboard_max_tokens` | `420` | Token budget of one refresh's answer |
| `[hotkeys] toggle_dashboard` | `"ctrl+alt+d"` | The toggle |

The **Project** tab:

| Key | Default | Meaning |
|---|---|---|
| `[project] repo_root` | `""` | The project directory; `""` = the auto-detected one |
| `[project] repo_auto` / `repo_auto_max_age_h` | `true` / `12` | Use the project of Claude Code's most recent session (within that many hours; `0` = any) |
| `[project] repo_search` | `true` | Look up files / identifiers mentioned (answers, dashboard) |
| `[project] repo_max_files` / `repo_max_chars` | `6` / `1800` | Budget of one lookup |
| `[project] repo_exclude` | `.git`, `node_modules`, … | Directories never scanned (config file only) |
| `[project] toolkit` | `true` | The toolkit catalog, auto-discovered in the repository |
| `[project] toolkit_lookup` | `true` | A mentioned part's full text goes along |
| `[project] toolkit_in_answers` | `true` | The catalog also for answers, ask and recap |
| `[project] toolkit_user_layer` | `false` | Also scan `~/.claude/skills` and `~/.claude/agents` |
| `[project] toolkit_dir` | `""` | The toolkit clone, for fuller descriptions; `""` = `~/claude-toolkit`, `~/.claude/dev-toolkit` |
| `[project] toolkit_max_chars` | `3000` | Catalog budget in the system prompt |
| `[project] rescan_s` | `30` | How often the repository and its `.claude/` are re-checked |

`--doctor` prints what the context sees: the repository (and how it was found, plus the other
recent Claude Code projects), the toolkit found (with every skill, agent and workflow by name and
the example command the prompts open with), the clone, and the dashboard's tracking settings.

## Headless

`python -m exocortex --no-ui` accepts `/refresh` (one refresh) and `/dashboard` (print the board
as markdown), and prints a `[dashboard]` line after every change.

## Limits

- The refresh prompt was tuned against a scripted client in the offline tests, not yet against
  every supported local model. A small model may restate an open item as `NEW`; the store folds
  exact duplicates, not paraphrases, so expect to dismiss the odd repeat. A sub-task the model
  files under the wrong task can be dismissed and re-added under the right one.
- Answers are detected from the conversation, so a question answered in a chat window the
  screen reader did not see stays open until you mark it.
- Toolkit awareness depends on the files Claude Code itself reads (`.claude/`, `CLAUDE.local.md`);
  a toolkit installed only in `~/.claude` is seen with `toolkit_user_layer` on. The auto-detected
  repository is the one Claude Code last ran in: with two projects open at once, pin the right one
  under **⋯ › View › Repository**.

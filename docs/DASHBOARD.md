# Dashboard mode

Dashboard mode turns the overlay into a board you can leave open on a second screen while you
pair: four columns, side by side, kept current without you touching anything.

| Column | What it holds | Where it comes from |
|---|---|---|
| **Session notes** | The running notes (summary, key points, decisions, action items, open questions) | The Notes tab's notes job, on its own cadence (`[notes].auto_interval_s`) |
| **Questions to ask** | Questions worth putting to the navigator or the team right now. Each one is tracked: once the conversation answers it, it moves under **Answered (n)** with the answer | A dashboard refresh |
| **Blind spots & follow-ups** | Risks nobody raised, missing tests, assumptions to verify, follow-ups not to forget. Tracked the same way | A dashboard refresh |
| **Next steps & prompts** | The concrete next actions, and, when a step calls for the coding agent, the prompt to type into Claude Code, exactly as you would type it | A dashboard refresh |

Toggle it with the **▦** button at the left of the title-bar icons, `Ctrl+Alt+D`, or **⋯ ›
Dashboard mode**. The window widens to `[dashboard].width` (1180 px by default, growing leftwards
from its right edge and never past the screen); the ask box, the bottom controls and the status
bar stay, so asking, pausing, **Text** and **Code** keep working. Leaving the mode restores the
window exactly as it was. The slim bar (completion mode) and the board cannot share the window:
entering one leaves the other.

## Where the data lives

The tracked items are not something the model "remembers". They live in a small database,
`dashboard.sqlite` in the session folder (or in memory with `[dashboard].persist = false`), and
the board is also written next to the notes as `dashboard.md` after every change. Each refresh
hands the model only what it needs: the **open** items of each list, each under a short id
(`Q1`, `B2`, `N1`, `P1`), what was said **since the last refresh**, a little earlier context, and
the screen. The model answers with the **changes** only:

```
## Questions to ask
- NEW: Should export() paginate past 1000 rows?
- ANSWERED Q2: yes, the cache is keyed by tenant id
## Blind spots & follow-ups
- DROP B1
## Next steps
- DONE N1: the failing test is in
## Suggested prompts
- NEW: /repro-fix export() returns 500 on an empty tenant
```

The store applies them: new items are added (duplicates of an open item are folded, an item
already answered is not raised again), answered or done items keep their answer, dropped ones
disappear. Anything the model did not mention stays open, so a cut-off answer never loses an
item. That is what keeps each call short on a laptop GPU: the prompt carries a dozen lines of
items instead of the whole history, and the answer is a handful of lines.

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
notes), the **↻** in a column header (that column only), **⋯ › Assist › Refresh the dashboard
now**, or the Notes tab's **Questions** / **Blind spots** buttons and `Ctrl+Alt+W`, which refresh
the board while it tracks.

Screen capture and OCR are not part of this. They keep the cadence set on the Screen tab
(`[screen].interval_s`) in every view mode; a refresh reads whatever the screen reader last saw.

## Working the board

| Want | Do |
|---|---|
| Mark a question answered (and note the answer) | **✓** on the card, or right-click › **Mark answered…** |
| Mark a step done | **✓** on the card, or right-click › **Mark done…** |
| Drop an item | **✕**, or right-click › **Dismiss** |
| Bring an answered item back | **↺** under **Answered (n)** |
| Reword an item | Right-click › **Edit…** |
| Copy a prompt to paste into Claude Code | **⧉** on the card, or right-click › **Copy** |
| Add an item by hand | **+** in the column header. In the Next steps column, text starting with `/` or "Use the …" becomes a prompt |
| Refresh one column | **↻** in its header |

Answered and done items stay visible, collapsed, up to `[dashboard].resolved_keep` per column
(`show_resolved` hides them altogether).

## Claude Code toolkit awareness

If a Claude Code toolkit is installed in the repository you are working on, the Suggested
prompts are written with it. dev-assist discovers the install by itself: it reads the skills,
agents, rules, workflows and hooks under `<repo>/.claude/` and the house rules and detected
commands in `CLAUDE.local.md`, builds a compact catalog from the files' own frontmatter, and
re-checks every `[context].rescan_s` seconds, so a `bash ~/claude-toolkit/install.sh --set live`
run mid-session is picked up within half a minute (a toast says what was found). Nothing about
any particular skill is hard-coded: an updated toolkit describes itself.

The catalog names each part exactly as it is typed or used (`/recon [topic]`, "Use the reviewer
agent …"), with its one-line description, the repo's test and lint commands, and the working
loop from the house rules. It sits at the end of the system prompt, so it is prefix-cached, and
the model is told to name only what is listed. When the conversation or the screen mentions a
part (`/scope`, "the verifier agent"), that part's full text goes along too, so the suggested
prompt follows its actual steps.

Set it up on the Settings dialog's **Context** tab: **Repository root** is the project
directory (also where `.claude/` is looked for); **Claude Code toolkit context** switches the
catalog; the sub-options control the on-mention lookup, whether answers / ask / recap get the
catalog too (not only the dashboard), the opt-in `~/.claude` user layer, the catalog's size, and
the optional toolkit clone (`~/claude-toolkit` is found by itself) whose `modules/index.json`
supplies fuller descriptions for the parts installed. Parts that are not installed are never
listed.

## Repository lookups

The same **Context** tab holds **Look up files / identifiers mentioned**. With a repository root
set, a refresh (and an answer, an ask, a recap) searches the repository for the identifier-shaped
names the conversation, the question or the screen mention (`insights_now`, `ScreenReader`,
`billing.py`, a path), and hands over only the matching lines, within `repo_max_files` /
`repo_max_chars`. Nothing is read wholesale and nothing is sent when nothing identifier-like
was mentioned or nothing matched. The scan is bounded (about a second per refresh, less for an
answer) and skips `.git`, `node_modules`, `.venv`, build output and binaries
(`[context].repo_exclude`). Inline completions never trigger a lookup (latency).

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
| `[dashboard] max_open_items` | `8` | Open items per column handed to the model (the oldest first) |
| `[dashboard] max_new_items` | `4` | New items the model may add per column per refresh |
| `[dashboard] suggest_prompts` | `true` | The Next steps column also suggests prompts for Claude Code |
| `[dashboard] show_resolved` / `resolved_keep` | `true` / `12` | Answered items stay visible, collapsed, this many per column |
| `[dashboard] show_notes` … `show_nextsteps` | `true` | Which columns the board shows |
| `[dashboard] width` | `1180` | Window width in dashboard mode; resizing the board updates it |
| `[dashboard] view_refresh_s` | `5` | How often the board re-renders its timestamps (no model call) |
| `[dashboard] persist` | `true` | Keep the items in `dashboard.sqlite` in the session folder (next session) |
| `[llm] dashboard_max_tokens` | `420` | Token budget of one refresh's answer |
| `[hotkeys] toggle_dashboard` | `"ctrl+alt+d"` | The toggle |

The **Context** tab:

| Key | Default | Meaning |
|---|---|---|
| `[context] repo_root` | `""` | The project directory; `""` = no lookups, no toolkit discovery |
| `[context] repo_search` | `true` | Look up files / identifiers mentioned (answers, dashboard) |
| `[context] repo_max_files` / `repo_max_chars` | `6` / `1800` | Budget of one lookup |
| `[context] repo_exclude` | `.git`, `node_modules`, … | Directories never scanned (config file only) |
| `[context] toolkit` | `true` | The toolkit catalog, auto-discovered in the repository |
| `[context] toolkit_lookup` | `true` | A mentioned part's full text goes along |
| `[context] toolkit_in_answers` | `true` | The catalog also for answers, ask and recap |
| `[context] toolkit_user_layer` | `false` | Also scan `~/.claude/skills` and `~/.claude/agents` |
| `[context] toolkit_dir` | `""` | The toolkit clone, for fuller descriptions; `""` = `~/claude-toolkit`, `~/.claude/dev-toolkit` |
| `[context] toolkit_max_chars` | `3000` | Catalog budget in the system prompt |
| `[context] rescan_s` | `30` | How often the repository's `.claude/` is re-checked |

`--doctor` prints what the context sees: the repository root, the toolkit found (with every
skill, agent and workflow by name), the clone, and the dashboard's tracking settings.

## Headless

`python -m exocortex --no-ui` accepts `/refresh` (one refresh) and `/dashboard` (print the board
as markdown), and prints a `[dashboard]` line after every change.

## Limits

- The refresh prompt was tuned against a scripted client in the offline tests, not yet against
  every supported local model. A small model may restate an open item as `NEW`; the store folds
  exact duplicates, not paraphrases, so expect to dismiss the odd repeat.
- Answers are detected from the conversation, so a question answered in a chat window the
  screen reader did not see stays open until you mark it.
- Toolkit awareness depends on the files Claude Code itself reads (`.claude/`, `CLAUDE.local.md`);
  a toolkit installed only in `~/.claude` is seen with `toolkit_user_layer` on.

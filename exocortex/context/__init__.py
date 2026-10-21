"""On-demand context for the model (2026-10-20): the repository being worked on and the Claude
Code toolkit installed in it.

- :mod:`exocortex.context.toolkit` -- auto-discovers a claude-toolkit install (``<repo>/.claude``,
  ``CLAUDE.local.md``) and renders a compact catalog of the installed skills, agents, rules,
  workflows and hooks from the files' own frontmatter -- nothing hard-coded -- plus one part's
  full text when it is mentioned.
- :mod:`exocortex.context.repo` -- a lightweight file index over the repo root that is SEARCHED for
  the identifiers and file names a prompt mentions; only matching lines are handed over.
- :mod:`exocortex.context.provider` -- the one object the assistant and the dashboard ask
  (``toolkit_block`` / ``toolkit_lookup`` / ``repo_block``), reading ``[context]`` live.
"""

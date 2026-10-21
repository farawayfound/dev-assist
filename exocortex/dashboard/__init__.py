"""Dashboard mode (2026-10-20): the tracked session board behind the overlay's multi-column view.

- :mod:`exocortex.dashboard.store` -- the database the data lives in (SQLite, one file per
  session folder, or in memory): questions to ask, blind spots & follow-ups, next steps and
  suggested prompts, each with a status (open / answered / dropped) and its answer, plus the
  running notes. The model never holds this; it is the source of truth for the view.
- :mod:`exocortex.dashboard.prompts` -- one refresh's prompt: the OPEN items with short ids, what
  is new since the last refresh, the screen, the on-demand repo / toolkit context -- and the
  parser for the model's answer, which lists only CHANGES (NEW / ANSWERED / DONE / DROP).
- :mod:`exocortex.dashboard.service` -- the timer that decides when a refresh is due, builds the
  prompt from the store, applies the parsed changes and publishes ``DashboardUpdatedEvent``.
"""

"""Logging configuration for ExoCortex.

Sets up a compact console handler (stderr) plus a rotating file handler under
``%LOCALAPPDATA%/ExoCortex/logs`` (or a caller-supplied directory), and quiets a
handful of noisy third-party libraries. Safe to call more than once: repeated
calls reconfigure the level/handlers instead of stacking duplicate handlers.
"""
from __future__ import annotations

import logging
import logging.handlers
import os
import sys
from pathlib import Path

log = logging.getLogger(__name__)

_CONSOLE_FORMAT = "%(asctime)s %(levelname).1s %(name)s: %(message)s"
_CONSOLE_DATEFMT = "%H:%M:%S"
_FILE_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"

_MAX_BYTES = 2 * 1024 * 1024  # 2 MB
_BACKUP_COUNT = 3

# Third-party loggers that are useful at DEBUG for their own project but just
# add noise to ours; pin them to WARNING regardless of our own root level.
_QUIET_LIBS = ("httpx", "httpcore", "faster_whisper", "huggingface_hub")

_HANDLER_TAG = "_exocortex_handler"


def _default_log_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA") or str(Path.home())
    return Path(base) / "ExoCortex" / "logs"


def setup_logging(level: str = "INFO", log_dir: Path | None = None) -> Path | None:
    """Configure the root logger with a console handler and a rotating file handler.

    Idempotent: calling this more than once removes any handlers this function
    previously installed (tagged via an attribute on the handler) before adding
    fresh ones, so logs are never duplicated and the level/log_dir can change
    between calls.

    Returns the path to the log file, or ``None`` if the log directory could
    not be created (console logging still works in that case).
    """
    root = logging.getLogger()
    root.setLevel(level.upper())

    for h in list(root.handlers):
        if getattr(h, _HANDLER_TAG, False):
            root.removeHandler(h)
            h.close()

    # Under pythonw.exe (the windowless launcher used by run.vbs) there is no console: both
    # sys.stderr and sys.stdout are None, and a StreamHandler bound to None would raise on every
    # record. Only attach a console handler when there is a real stream to write to; file logging
    # (below) still captures everything.
    stream = sys.stderr or sys.stdout
    if stream is not None:
        console = logging.StreamHandler(stream)
        console.setFormatter(logging.Formatter(_CONSOLE_FORMAT, datefmt=_CONSOLE_DATEFMT))
        setattr(console, _HANDLER_TAG, True)
        root.addHandler(console)

    log_path: Path | None = None
    target_dir = Path(log_dir) if log_dir is not None else _default_log_dir()
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        log_path = target_dir / "exocortex.log"
        file_handler = logging.handlers.RotatingFileHandler(
            log_path, maxBytes=_MAX_BYTES, backupCount=_BACKUP_COUNT, encoding="utf-8"
        )
        file_handler.setFormatter(logging.Formatter(_FILE_FORMAT))
        setattr(file_handler, _HANDLER_TAG, True)
        root.addHandler(file_handler)
    except OSError:
        log.warning("could not set up log file under %s; console logging only", target_dir, exc_info=True)
        log_path = None

    for name in _QUIET_LIBS:
        logging.getLogger(name).setLevel(logging.WARNING)

    log.debug("logging configured: level=%s log_path=%s", level, log_path)
    return log_path

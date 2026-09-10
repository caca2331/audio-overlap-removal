"""Console and file logging, configured by the command-line entry point only."""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

LOG_LEVELS = ("debug", "info", "warning", "error")

_PACKAGE_LOGGER = "audio_overlap_removal"
_FILE_FORMAT = "%(asctime)s %(levelname)-7s %(name)s %(message)s"


class _UtcFormatter(logging.Formatter):
    """Timestamp in UTC with milliseconds; logging's default is local time.

    A run that spans a daylight-saving change, or a log compared against one
    from another machine, is unreadable without a fixed zone.
    """

    converter = time.gmtime
    default_time_format = "%Y-%m-%dT%H:%M:%S"
    default_msec_format = "%s.%03dZ"


def _configure_logging(log_path: Path | None, level: str, quiet: bool) -> None:
    """Send progress to stderr, and optionally a more detailed copy to a file.

    The console and the file share one level so that ``--log-level debug``
    alone is enough to watch the detail live; ``--quiet`` then clamps the
    console back to warnings without touching what the file records.
    """
    if level not in LOG_LEVELS:
        raise ValueError(f"Unknown log level: {level!r}.")
    numeric = getattr(logging, level.upper())
    logger = logging.getLogger(_PACKAGE_LOGGER)
    logger.setLevel(numeric)
    # Progress belongs to this tool, not to whatever the embedding process
    # configured on the root logger.
    logger.propagate = False
    for handler in list(logger.handlers):
        if not isinstance(handler, logging.NullHandler):
            logger.removeHandler(handler)
            handler.close()

    console = logging.StreamHandler(sys.stderr)
    console.setLevel(max(numeric, logging.WARNING) if quiet else numeric)
    console.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(console)

    if log_path is None:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setLevel(numeric)
    file_handler.setFormatter(_UtcFormatter(_FILE_FORMAT))
    logger.addHandler(file_handler)

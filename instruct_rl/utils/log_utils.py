"""
instruct_rl/utils/log_utils.py
===============================
Project-wide logging utilities.

Usage
------
>>> from instruct_rl.utils.log_utils import get_logger
>>> logger = get_logger(__name__)         # or get_logger(__file__)
>>> logger.info("hello %s", "world")
"""
from __future__ import annotations

import logging
import os
import re
from os.path import basename, splitext
from typing import Union

# ── Default format ──────────────────────────────────────────────────────
_DEFAULT_FMT = "[%(asctime)s][%(name)s][%(levelname)s] %(message)s"
_DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"

# The global log level can be controlled with an environment variable
_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# Flag preventing duplicate root handlers
_ROOT_CONFIGURED = False


class _MultiLineFormatter(logging.Formatter):
    """Prefix every line of a multiline message consistently."""

    def format(self, record: logging.LogRecord) -> str:
        original = super().format(record)
        # The first line is already formatted; prefix all remaining lines
        if "\n" not in record.getMessage():
            return original
        # prefix = "[2026-03-26 ...][name][LEVEL] "
        header = original[: original.index(record.getMessage())]
        lines = original.split("\n")
        return "\n".join(
            line if i == 0 else f"{header}{line}"
            for i, line in enumerate(lines)
        )


def _ensure_root_handler():
    """Add a root StreamHandler if absent and apply one format to existing handlers."""
    global _ROOT_CONFIGURED
    if _ROOT_CONFIGURED:
        return

    formatter = _MultiLineFormatter(_DEFAULT_FMT, datefmt=_DEFAULT_DATEFMT)
    root = logging.getLogger()

    if root.handlers:
        # Apply our format to existing handlers such as Hydra's
        for h in root.handlers:
            h.setFormatter(formatter)
    else:
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        root.addHandler(handler)

    root.setLevel(getattr(logging, _LOG_LEVEL, logging.INFO))
    _ROOT_CONFIGURED = True


def _clean_name(name: str | None) -> str | None:
    """Remove paths and the .py extension from a logger name.

    ``__file__`` → ``"train_cpcgrl"``
    ``__name__`` → as-is return
    """
    if not name:
        return name
    # Extract the basename when the name contains a path
    if "/" in name or "\\" in name:
        name = basename(name)
    # Remove .py/.pyc extensions
    name = re.sub(r"\.pyc?$", "", name)
    return name


def get_logger(name: Union[str, None] = None, level: Union[str, int, None] = None) -> logging.Logger:
    """Return a logger for a name.

    Parameters
    ----------
    name : str | None
        Pass ``__name__`` or ``__file__``; paths and extensions are normalized.
    level : str | int | None
        Per-logger level. ``None`` follows the global LOG_LEVEL environment setting.
    """
    _ensure_root_handler()
    name = _clean_name(name)

    logger = logging.getLogger(name)

    if level is not None:
        if isinstance(level, str):
            level = getattr(logging, level.upper(), logging.INFO)
        logger.setLevel(level)
    else:
        logger.setLevel(getattr(logging, _LOG_LEVEL, logging.INFO))

    return logger


def suppress_jax_debug_logs():
    """Suppress internal JAX DEBUG logs such as cache_key messages."""
    for jax_logger_name in (
        "jax._src.cache_key",
        "jax._src.compiler",
        "jax._src.dispatch",
        "jax._src.interpreters",
        "jax",
    ):
        logging.getLogger(jax_logger_name).setLevel(logging.WARNING)

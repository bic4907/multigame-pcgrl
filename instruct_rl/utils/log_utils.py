"""
instruct_rl/utils/log_utils.py
===============================
text to text  before text in  text for text   to text utility.

Usage
------
>>> from instruct_rl.utils.log_utils import get_logger
>>> logger = get_logger(__name__)         # text  get_logger(__file__)
>>> logger.info("hello %s", "world")
"""
from __future__ import annotations

import logging
import os
import re
from os.path import basename, splitext
from typing import Union

# ── default text ────────────────────────────────────────────────────────────
_DEFAULT_FMT = "[%(asctime)s][%(name)s][%(levelname)s] %(message)s"
_DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"

# text to   before text  to text level  text text text.
_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# text handler  duplicate text text text also text text
_ROOT_CONFIGURED = False


class _MultiLineFormatter(logging.Formatter):
    """text text of  each text in  sametext prefix  text sort  keeptext."""

    def format(self, record: logging.LogRecord) -> str:
        original = super().format(record)
        # text text   text text. remaining text in  also  sametext prefix  text.
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
    """text  to text in  StreamHandler  text if missing text text, existing handler also  text  text."""
    global _ROOT_CONFIGURED
    if _ROOT_CONFIGURED:
        return

    formatter = _MultiLineFormatter(_DEFAULT_FMT, datefmt=_DEFAULT_DATEFMT)
    root = logging.getLogger()

    if root.handlers:
        # existing handler(hydra text) in  also  text text  apply
        for h in root.handlers:
            h.setFormatter(formatter)
    else:
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        root.addHandler(handler)

    root.setLevel(getattr(logging, _LOG_LEVEL, logging.INFO))
    _ROOT_CONFIGURED = True


def _clean_name(name: str | None) -> str | None:
    """ to text name in  path and  .py expandtext  removetext.

    ``__file__`` → ``"train_cpcgrl"``
    ``__name__`` → as-is return
    """
    if not name:
        return name
    # path  text text basename extract
    if "/" in name or "\\" in name:
        name = basename(name)
    # .py / .pyc expandtext remove
    name = re.sub(r"\.pyc?$", "", name)
    return name


def get_logger(name: Union[str, None] = None, level: Union[str, int, None] = None) -> logging.Logger:
    """name based  to text  returntext.

    Parameters
    ----------
    name : str | None
        ``__name__`` text  ``__file__``   text text.
        path·expandtext  automatic as  text.
    level : str | int | None
        text  to text level. ``None``  text  before text(LOG_LEVEL text) config  text.
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
    """jax internal DEBUG  to text(cache_key text)  text."""
    for jax_logger_name in (
        "jax._src.cache_key",
        "jax._src.compiler",
        "jax._src.dispatch",
        "jax._src.interpreters",
        "jax",
    ):
        logging.getLogger(jax_logger_name).setLevel(logging.WARNING)


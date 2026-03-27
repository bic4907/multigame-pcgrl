"""
instruct_rl/utils/log_utils.py
===============================
프로젝트 전역에서 사용하는 로깅 유틸리티.

사용법
------
>>> from instruct_rl.utils.log_utils import get_logger
>>> logger = get_logger(__name__)         # 또는 get_logger(__file__)
>>> logger.info("hello %s", "world")
"""
from __future__ import annotations

import logging
import os
import re
from os.path import basename, splitext
from typing import Union

# ── 기본 포맷 ────────────────────────────────────────────────────────────
_DEFAULT_FMT = "[%(asctime)s][%(name)s][%(levelname)s] %(message)s"
_DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"

# 환경변수로 전역 로그 레벨을 제어할 수 있다.
_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# 루트 핸들러가 중복 추가되지 않도록 플래그
_ROOT_CONFIGURED = False


class _MultiLineFormatter(logging.Formatter):
    """멀티라인 메시지의 각 줄에 동일한 prefix를 붙여 정렬을 유지한다."""

    def format(self, record: logging.LogRecord) -> str:
        original = super().format(record)
        # 첫 줄은 이미 포맷됨. 나머지 줄에도 동일한 prefix를 붙인다.
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
    """루트 로거에 StreamHandler가 아직 없으면 추가하고, 기존 핸들러도 포맷을 통일한다."""
    global _ROOT_CONFIGURED
    if _ROOT_CONFIGURED:
        return

    formatter = _MultiLineFormatter(_DEFAULT_FMT, datefmt=_DEFAULT_DATEFMT)
    root = logging.getLogger()

    if root.handlers:
        # 기존 핸들러(hydra 등)에도 우리 포맷을 적용
        for h in root.handlers:
            h.setFormatter(formatter)
    else:
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        root.addHandler(handler)

    root.setLevel(getattr(logging, _LOG_LEVEL, logging.INFO))
    _ROOT_CONFIGURED = True


def _clean_name(name: str | None) -> str | None:
    """로거 이름에서 경로와 .py 확장자를 제거한다.

    ``__file__`` → ``"train_cpcgrl"``
    ``__name__`` → 그대로 반환
    """
    if not name:
        return name
    # 경로가 포함되어 있으면 basename 추출
    if "/" in name or "\\" in name:
        name = basename(name)
    # .py / .pyc 확장자 제거
    name = re.sub(r"\.pyc?$", "", name)
    return name


def get_logger(name: Union[str, None] = None, level: Union[str, int, None] = None) -> logging.Logger:
    """이름 기반 로거를 반환한다.

    Parameters
    ----------
    name : str | None
        ``__name__`` 또는 ``__file__`` 을 넘기면 된다.
        경로·확장자는 자동으로 정리된다.
    level : str | int | None
        개별 로거 레벨. ``None`` 이면 전역(LOG_LEVEL 환경변수) 설정을 따른다.
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
    """jax 내부 DEBUG 로그(cache_key 등)를 숨긴다."""
    for jax_logger_name in (
        "jax._src.cache_key",
        "jax._src.compiler",
        "jax._src.dispatch",
        "jax._src.interpreters",
        "jax",
    ):
        logging.getLogger(jax_logger_name).setLevel(logging.WARNING)


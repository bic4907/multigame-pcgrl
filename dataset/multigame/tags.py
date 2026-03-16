"""
dataset/multigame/tags.py
=========================
GameSample 태깅 & 필터링 유틸리티.

외부 의존 없음.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional

from .base import GameSample


# ── 태그 빌더 ────────────────────────────────────────────────────────────────────

def build_tags(sample: GameSample) -> Dict[str, Any]:
    """
    샘플에서 태그 dict를 추출.

    Returns
    -------
    {
        "game":        str,
        "instruction": str | None,
        "order":       int | None,
        "source_id":   str,
        "has_instruction": bool,
        "shape":       (H, W),
        **sample.meta  (instruction_slug, level_id, sample_id 등)
    }
    """
    tags: Dict[str, Any] = {
        "game":            sample.game,
        "instruction":     sample.instruction,
        "order":           sample.order,
        "source_id":       sample.source_id,
        "has_instruction": sample.instruction is not None,
        "shape":           sample.shape,
    }
    tags.update(sample.meta)
    return tags


# ── 필터 유틸 ────────────────────────────────────────────────────────────────────

def extract_by_game(
    samples: List[GameSample],
    game: str,
) -> List[GameSample]:
    """특정 game 태그 샘플만 추출."""
    return [s for s in samples if s.game == game]


def extract_by_games(
    samples: List[GameSample],
    games: List[str],
) -> List[GameSample]:
    """복수 game 태그 샘플 추출."""
    game_set = set(games)
    return [s for s in samples if s.game in game_set]


def extract_by_instruction(
    samples: List[GameSample],
    keyword: str,
    *,
    case_sensitive: bool = False,
) -> List[GameSample]:
    """instruction에 keyword가 포함된 샘플 추출."""
    kw = keyword if case_sensitive else keyword.lower()
    result = []
    for s in samples:
        if s.instruction is None:
            continue
        text = s.instruction if case_sensitive else s.instruction.lower()
        if kw in text:
            result.append(s)
    return result


def extract_with_instruction(samples: List[GameSample]) -> List[GameSample]:
    """instruction이 있는 샘플만 추출."""
    return [s for s in samples if s.instruction is not None]


def extract_without_instruction(samples: List[GameSample]) -> List[GameSample]:
    """instruction이 없는 샘플만 추출."""
    return [s for s in samples if s.instruction is None]


def extract_by_order(
    samples: List[GameSample],
    start: int,
    end: int,
) -> List[GameSample]:
    """order 범위 [start, end) 샘플 추출."""
    return [
        s for s in samples
        if s.order is not None and start <= s.order < end
    ]


def extract_by_meta(
    samples: List[GameSample],
    key: str,
    value: Any,
) -> List[GameSample]:
    """sample.meta[key] == value 인 샘플 추출."""
    return [s for s in samples if s.meta.get(key) == value]


def extract_by_predicate(
    samples: List[GameSample],
    fn: Callable[[GameSample], bool],
) -> List[GameSample]:
    """임의 조건 함수로 필터링."""
    return [s for s in samples if fn(s)]


# ── 집계 유틸 ────────────────────────────────────────────────────────────────────

def group_by_game(
    samples: List[GameSample],
) -> Dict[str, List[GameSample]]:
    """game 태그별로 샘플 그룹핑."""
    groups: Dict[str, List[GameSample]] = defaultdict(list)
    for s in samples:
        groups[s.game].append(s)
    return dict(groups)


def group_by_instruction(
    samples: List[GameSample],
) -> Dict[str, List[GameSample]]:
    """instruction 문자열별로 샘플 그룹핑 (None은 '__no_instruction__' 키)."""
    groups: Dict[str, List[GameSample]] = defaultdict(list)
    for s in samples:
        key = s.instruction if s.instruction is not None else "__no_instruction__"
        groups[key].append(s)
    return dict(groups)


def count_by_game(samples: List[GameSample]) -> Dict[str, int]:
    """게임별 샘플 수 카운트."""
    counts: Dict[str, int] = defaultdict(int)
    for s in samples:
        counts[s.game] += 1
    return dict(counts)


def count_by_instruction(samples: List[GameSample]) -> Dict[str, int]:
    """instruction별 샘플 수 카운트."""
    counts: Dict[str, int] = defaultdict(int)
    for s in samples:
        key = s.instruction or "__no_instruction__"
        counts[key] += 1
    return dict(counts)


def summary(samples: List[GameSample]) -> Dict[str, Any]:
    """전체 샘플 요약 정보."""
    return {
        "total":                 len(samples),
        "by_game":               count_by_game(samples),
        "with_instruction":      len(extract_with_instruction(samples)),
        "without_instruction":   len(extract_without_instruction(samples)),
        "unique_instructions":   len(set(
            s.instruction for s in samples if s.instruction is not None
        )),
    }


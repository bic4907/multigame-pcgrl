"""
dataset/multigame/tags.py
=========================
GameSample text & filtering utility.

text  of text none.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional

from .base import GameSample


# ── text text ────────────────────────────────────────────────────────────────────

def build_tags(sample: GameSample) -> Dict[str, Any]:
    """
    sample in  text dict  extract.

    Returns
    -------
    {
        "game":        str,
        "instruction": str | None,
        "order":       int | None,
        "source_id":   str,
        "has_instruction": bool,
        "shape":       (H, W),
        **sample.meta  (instruction_slug, level_id, sample_id text)
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


# ── filter utility ────────────────────────────────────────────────────────────────────

def extract_by_game(
    samples: List[GameSample],
    game: str,
) -> List[GameSample]:
    """text game text sampletext extract."""
    return [s for s in samples if s.game == game]


def extract_by_games(
    samples: List[GameSample],
    games: List[str],
) -> List[GameSample]:
    """text game text sample extract."""
    game_set = set(games)
    return [s for s in samples if s.game in game_set]


def extract_by_instruction(
    samples: List[GameSample],
    keyword: str,
    *,
    case_sensitive: bool = False,
) -> List[GameSample]:
    """instruction in  keyword  text sample extract."""
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
    """instruction  with sampletext extract."""
    return [s for s in samples if s.instruction is not None]


def extract_without_instruction(samples: List[GameSample]) -> List[GameSample]:
    """instruction  without sampletext extract."""
    return [s for s in samples if s.instruction is None]


def extract_by_order(
    samples: List[GameSample],
    start: int,
    end: int,
) -> List[GameSample]:
    """order range [start, end) sample extract."""
    return [
        s for s in samples
        if s.order is not None and start <= s.order < end
    ]


def extract_by_meta(
    samples: List[GameSample],
    key: str,
    value: Any,
) -> List[GameSample]:
    """sample.meta[key] == value text sample extract."""
    return [s for s in samples if s.meta.get(key) == value]


def extract_by_predicate(
    samples: List[GameSample],
    fn: Callable[[GameSample], bool],
) -> List[GameSample]:
    """text of  condition function to  filtering."""
    return [s for s in samples if fn(s)]


# ── text utility ────────────────────────────────────────────────────────────────────

def group_by_game(
    samples: List[GameSample],
) -> Dict[str, List[GameSample]]:
    """game textby sample text."""
    groups: Dict[str, List[GameSample]] = defaultdict(list)
    for s in samples:
        groups[s.game].append(s)
    return dict(groups)


def group_by_instruction(
    samples: List[GameSample],
) -> Dict[str, List[GameSample]]:
    """instruction stringby sample text (None  '__no_instruction__' text)."""
    groups: Dict[str, List[GameSample]] = defaultdict(list)
    for s in samples:
        key = s.instruction if s.instruction is not None else "__no_instruction__"
        groups[key].append(s)
    return dict(groups)


def count_by_game(samples: List[GameSample]) -> Dict[str, int]:
    """gametext sample text text."""
    counts: Dict[str, int] = defaultdict(int)
    for s in samples:
        counts[s.game] += 1
    return dict(counts)


def count_by_instruction(samples: List[GameSample]) -> Dict[str, int]:
    """instructiontext sample text text."""
    counts: Dict[str, int] = defaultdict(int)
    for s in samples:
        key = s.instruction or "__no_instruction__"
        counts[key] += 1
    return dict(counts)


def summary(samples: List[GameSample]) -> Dict[str, Any]:
    """all sample summary info."""
    return {
        "total":                 len(samples),
        "by_game":               count_by_game(samples),
        "with_instruction":      len(extract_with_instruction(samples)),
        "without_instruction":   len(extract_without_instruction(samples)),
        "unique_instructions":   len(set(
            s.instruction for s in samples if s.instruction is not None
        )),
    }


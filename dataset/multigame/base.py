"""
dataset/multigame/base.py
=========================
공통 추상 인터페이스 정의.
모든 게임 핸들러는 BaseGameHandler를 상속하고,
모든 전처리기는 BasePreprocessor를 상속한다.

외부 패키지 의존 없음 (numpy만 사용).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional
import warnings

import numpy as np


# ── 공통 태그 상수 ──────────────────────────────────────────────────────────────
class GameTag:
    """지원 게임 식별자 상수 모음."""
    ZELDA       = "zelda"
    MARIO       = "mario"
    LODE_RUNNER = "lode_runner"
    KID_ICARUS  = "kid_icarus"
    DOOM        = "doom"
    MEGA_MAN    = "mega_man"
    DUNGEON     = "dungeon"
    BOXOBAN     = "boxoban"


# ── 공통 데이터 구조 ────────────────────────────────────────────────────────────

@dataclass
class TileLegend:
    """
    타일 문자 → 의미 속성 매핑.
    char_to_attrs: {'W': ['solid', 'wall'], '-': ['passable', 'empty'], ...}
    """
    char_to_attrs: Dict[str, List[str]] = field(default_factory=dict)

    def tags_for(self, char: str) -> List[str]:
        return self.char_to_attrs.get(char, [])

    def is_passable(self, char: str) -> bool:
        return "passable" in self.tags_for(char)

    def is_solid(self, char: str) -> bool:
        return "solid" in self.tags_for(char)

    def is_enemy(self, char: str) -> bool:
        return "enemy" in self.tags_for(char)


@dataclass
class GameSample:
    """
    단일 레벨 샘플.

    Parameters
    ----------
    game        : GameTag 상수 (e.g. GameTag.ZELDA)
    source_id   : 원본 파일명 또는 npz 키 등 고유 식별자
    array       : (H, W) int32 ndarray - 정수 인코딩된 타일 그리드
    char_grid   : (H, W) 문자 그리드 (원본 txt 기반일 때 유지)
    legend      : TileLegend (None 가능)
    instruction : 자연어 명령 (dungeon 등에서 사용)
    order       : 원본 데이터셋 내 순서(index)
    meta        : 기타 부가 정보 dict
    """
    game:        str
    source_id:   str
    array:       np.ndarray                    # (H, W) int32
    char_grid:   Optional[List[List[str]]] = None
    legend:      Optional[TileLegend]      = None
    instruction: Optional[str]             = None
    order:       Optional[int]             = None
    meta:        Dict[str, Any]            = field(default_factory=dict)

    @property
    def height(self) -> int:
        return self.array.shape[0]

    @property
    def width(self) -> int:
        return self.array.shape[1]

    @property
    def shape(self):
        return self.array.shape

    def __repr__(self) -> str:
        return (
            f"GameSample(game={self.game!r}, source_id={self.source_id!r}, "
            f"shape={self.shape}, instruction={self.instruction!r})"
        )


def enforce_top_left_16x16(
    array: np.ndarray,
    *,
    game: str,
    source_id: str,
) -> np.ndarray:
    """
    Normalize any 2D level array to (16, 16).

    - If shape is already (16, 16), array is returned as-is.
    - Otherwise, top-left [:16, :16] is used.
    - If the sliced region is smaller than 16x16, remaining area is zero-padded.
    """
    # Some sources can contain an extra leading axis, e.g. (1, 16, 16).
    if array.ndim > 2:
        warnings.warn(
            (
                f"[{game}] {source_id} has ndim={array.ndim}; "
                "using the first slice on leading axes before 16x16 normalization"
            ),
            RuntimeWarning,
            stacklevel=2,
        )
        # Keep only the first sample on leading axes and retain the last 2 dims.
        array = array.reshape((-1,) + array.shape[-2:])[0]

    if array.shape == (16, 16):
        return array
    warnings.warn(
        (
            f"[{game}] {source_id} has shape {array.shape}; "
            "normalizing to (16, 16) with top-left slice and zero-padding if needed"
        ),
        RuntimeWarning,
        stacklevel=2,
    )
    out = np.zeros((16, 16), dtype=array.dtype)
    h = min(array.shape[0], 16)
    w = min(array.shape[1], 16)
    out[:h, :w] = array[:h, :w]
    return out


def enforce_char_grid_top_left_16x16(
    char_grid: List[List[str]],
) -> List[List[str]]:
    """Slice char grid to top-left 16x16 for consistency with array slicing."""
    return [row[:16] for row in char_grid[:16]]


# ── 추상 핸들러 ─────────────────────────────────────────────────────────────────

class BaseGameHandler(ABC):
    """
    단일 게임/데이터셋 소스에 대한 핸들러.
    list_entries() 로 전체 ID를 열거하고,
    load_sample()  로 GameSample을 반환한다.
    """

    @property
    @abstractmethod
    def game_tag(self) -> str:
        """GameTag 상수를 반환."""
        ...

    @abstractmethod
    def list_entries(self) -> List[str]:
        """로드 가능한 source_id 목록 반환."""
        ...

    @abstractmethod
    def load_sample(self, source_id: str, order: Optional[int] = None) -> GameSample:
        """source_id에 해당하는 GameSample 반환."""
        ...

    def __iter__(self) -> Iterator[GameSample]:
        for i, sid in enumerate(self.list_entries()):
            yield self.load_sample(sid, order=i)

    def __len__(self) -> int:
        return len(self.list_entries())

    def all_samples(self) -> List[GameSample]:
        return list(self)


# ── 추상 전처리기 ───────────────────────────────────────────────────────────────

class BasePreprocessor(ABC):
    """
    문자 그리드 → 정수 ndarray 변환 및 기타 전처리.
    각 게임마다 서브클래스를 정의한다.
    """

    @abstractmethod
    def char_to_int(self, char: str) -> int:
        """단일 문자를 정수 타일 ID로 변환."""
        ...

    def transform(self, char_grid: List[List[str]]) -> np.ndarray:
        """2D 문자 리스트 → (H, W) int32 ndarray."""
        h = len(char_grid)
        w = max(len(row) for row in char_grid) if h > 0 else 0
        arr = np.zeros((h, w), dtype=np.int32)
        for r, row in enumerate(char_grid):
            for c, ch in enumerate(row):
                arr[r, c] = self.char_to_int(ch)
        return arr

    def parse_txt(self, text: str) -> List[List[str]]:
        """텍스트 파일 내용 → 2D 문자 리스트."""
        lines = text.splitlines()
        # 빈 줄 제거
        lines = [l for l in lines if l.strip()]
        return [list(line) for line in lines]

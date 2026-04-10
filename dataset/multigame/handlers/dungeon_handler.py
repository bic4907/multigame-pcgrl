"""
dataset/multigame/handlers/dungeon_handler.py
=============================================
dungeon_level_dataset 핸들러.

- dungeon_levels.npz + dungeon_levels_metadata.csv 로드
- instruction / instruction_slug / level_id / sample_id 태깅 지원
- DungeonLevelDataset 코드를 직접 복사하지 않고 독립적으로 재구현
  (외부 패키지 참조 없음, numpy만 사용)

타일 매핑 (dungeon_level_dataset README 기준)
---------------------------------------------
0  : padding / unknown
1  : floor  (원본 값 1)
2  : wall   (원본 값 2)
3  : enemy  (원본 값 3)

전처리 필터 (캐시 저장 전 적용)
---------------------------------------------
1. ndim!=2 맵 제거 (형식 이상 맵)
2. feature 기반 필터: region / path_length / bat_amount 만 유지
   (block, bat_direction 카테고리는 퀄리티 문제로 제외)
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ..base import (
    BaseGameHandler,
    GameSample,
    GameTag,
    TileLegend,
    enforce_top_left_16x16,
)

_DEFAULT_DUNGEON_ROOT = (
    Path(__file__).parent.parent.parent / "dungeon_level_dataset"
)

# ── feature 분류 ─────────────────────────────────────────────────────────────────
# instruction_slug 키워드로 feature 카테고리를 추론한다.
# 5개 카테고리: region / path_length / bat_amount / block / bat_direction
#   → 이 중 퀄리티가 좋은 region / path_length / bat_amount 만 로드한다.
_DIRECTION_KEYWORDS: tuple[str, ...] = (
    "linear", "radial", "north", "south", "east", "west",
    "top", "bottom", "left", "right",
)

_ALLOWED_FEATURES: frozenset[str] = frozenset({
    "region",
    "path_length",
    "bat_amount",
})


def _classify_feature(slug: str) -> str:
    """instruction_slug 로부터 feature 카테고리를 추론한다.

    Returns
    -------
    'region' | 'path_length' | 'bat_amount' | 'block' | 'bat_direction' | 'unknown'
    """
    has_bat = "bat" in slug
    has_direction = any(kw in slug for kw in _DIRECTION_KEYWORDS)

    if has_bat and has_direction:
        return "bat_direction"
    if "block" in slug or "centralized" in slug or "decentralized" in slug:
        return "block"
    if has_bat:
        return "bat_amount"
    if "path" in slug:
        return "path_length"
    if "region" in slug:
        return "region"
    return "unknown"


# ── 타일 상수 ────────────────────────────────────────────────────────────────────
class DungeonTile:
    UNKNOWN  = 0
    FLOOR    = 1
    WALL     = 2
    ENEMY    = 3
    TREASURE = 4


DUNGEON_PALETTE: dict[int, tuple[int, int, int]] = {
    DungeonTile.UNKNOWN: (0,   0,   0),
    DungeonTile.FLOOR:   (200, 180, 120),
    DungeonTile.WALL:    (80,  80,  80),
    DungeonTile.ENEMY:   (220, 50,  50),
    DungeonTile.TREASURE: (200, 200, 0),
}


def _place_treasure(array: np.ndarray, key: str) -> np.ndarray:
    """
    FLOOR 타일 중 랜덤하게 0~9개를 TREASURE(4)로 교체한다.
    key(='000000' 형식)를 정수로 변환해 seed로 사용 → 재현 가능.
    FLOOR 가 없으면 array 를 그대로 반환한다.
    """
    rng = np.random.RandomState(int(key))
    n = rng.randint(0, 10)                          # 0~9 개
    if n == 0:
        return array
    floor_pos = np.argwhere(array == DungeonTile.FLOOR)
    if len(floor_pos) == 0:
        return array
    n = min(n, len(floor_pos))
    chosen = rng.choice(len(floor_pos), size=n, replace=False)
    result = array.copy()
    for idx in chosen:
        r, c = floor_pos[idx]
        result[r, c] = DungeonTile.TREASURE
    return result


def _make_legend() -> TileLegend:
    return TileLegend(char_to_attrs={
        "1": ["passable", "floor"],
        "2": ["solid", "wall"],
        "3": ["enemy", "damaging"],
    })


# ── 메타 dataclass (경량) ────────────────────────────────────────────────────────
class _DungeonMeta:
    __slots__ = ("index", "key", "instruction", "instruction_slug",
                 "level_id", "sample_id")

    def __init__(self, index, key, instruction, instruction_slug,
                 level_id, sample_id):
        self.index = int(index)
        self.key = key
        self.instruction = instruction
        self.instruction_slug = instruction_slug
        self.level_id = int(level_id)
        self.sample_id = int(sample_id)


class DungeonHandler(BaseGameHandler):
    """
    dungeon_level_dataset 핸들러.

    Parameters
    ----------
    root      : dungeon_level_dataset 폴더 경로
    npz_name  : npz 파일명 (기본 'dungeon_levels.npz')
    meta_name : csv 파일명 (기본 'dungeon_levels_metadata.csv')

    Example
    -------
        handler = DungeonHandler()
        for sample in handler:
            print(sample.instruction, sample.shape)

        # instruction으로 필터
        subset = handler.filter_by_instruction("bat swarm")
    """

    def __init__(
        self,
        root: Path | str = _DEFAULT_DUNGEON_ROOT,
        npz_name: str = "dungeon_levels.npz",
        meta_name: str = "dungeon_levels_metadata.csv",
    ) -> None:
        self._root = Path(root)
        npz_path  = self._root / npz_name
        meta_path = self._root / meta_name

        if not npz_path.exists():
            raise FileNotFoundError(f"NPZ not found: {npz_path}")
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata CSV not found: {meta_path}")

        self._archive = np.load(npz_path)
        self._legend  = _make_legend()
        self._metas: List[_DungeonMeta] = []
        self._key_to_meta: Dict[str, _DungeonMeta] = {}

        raw_metas: dict[str, _DungeonMeta] = {}
        with open(meta_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                m = _DungeonMeta(
                    index=row["index"],
                    key=row["key"],
                    instruction=row["instruction"],
                    instruction_slug=row["instruction_slug"],
                    level_id=row["level_id"],
                    sample_id=row["sample_id"],
                )
                raw_metas[m.key] = m

        # 전처리 필터 적용
        all_keys = [m.key for m in sorted(raw_metas.values(), key=lambda m: m.index)]

        # 1차: ndim!=2 맵 제거 (형식 이상 맵)
        all_keys = [k for k in all_keys if self._archive[k].ndim == 2]

        # 2차: feature 기반 필터 (region / path_length / bat_amount 만 유지)
        kept_keys = [
            k for k in all_keys
            if _classify_feature(raw_metas[k].instruction_slug) in _ALLOWED_FEATURES
        ]

        for key in kept_keys:
            m = raw_metas[key]
            self._metas.append(m)
            self._key_to_meta[key] = m

    @property
    def game_tag(self) -> str:
        return GameTag.DUNGEON

    # ── BaseGameHandler ─────────────────────────────────────────────────────────
    def list_entries(self) -> List[str]:
        """npz key 목록 반환."""
        return [m.key for m in self._metas]

    def load_sample(self, source_id: str, order: Optional[int] = None) -> GameSample:
        """npz key → GameSample 반환."""
        m = self._key_to_meta.get(source_id)
        if m is None:
            raise KeyError(f"Key not found in dungeon dataset: {source_id!r}")
        raw = self._archive[source_id]           # (16,16) int64
        array = raw.astype(np.int32)
        array = enforce_top_left_16x16(
            array,
            game=GameTag.DUNGEON,
            source_id=source_id,
        )
        # array = _place_treasure(array, source_id)
        return GameSample(
            game=GameTag.DUNGEON,
            source_id=source_id,
            array=array,
            char_grid=None,
            legend=self._legend,
            instruction=m.instruction,
            order=order if order is not None else m.index,
            meta={
                "instruction_slug": m.instruction_slug,
                "level_id":         m.level_id,
                "sample_id":        m.sample_id,
            },
        )

    # ── 확장 쿼리 메서드 ─────────────────────────────────────────────────────────
    def filter_by_instruction(
        self, keyword: str, *, case_sensitive: bool = False
    ) -> List[GameSample]:
        """instruction에 keyword가 포함된 샘플 목록 반환."""
        kw = keyword if case_sensitive else keyword.lower()
        result = []
        for i, m in enumerate(self._metas):
            text = m.instruction if case_sensitive else m.instruction.lower()
            if kw in text:
                result.append(self.load_sample(m.key, order=i))
        return result

    def group_by_instruction(self) -> Dict[str, List[GameSample]]:
        """instruction_slug → 샘플 리스트 딕셔너리."""
        groups: Dict[str, List[GameSample]] = {}
        for i, m in enumerate(self._metas):
            sample = self.load_sample(m.key, order=i)
            groups.setdefault(m.instruction_slug, []).append(sample)
        return groups


    def category_names(self) -> List[str]:
        """고유 instruction 문자열 목록."""
        seen = {}
        for m in self._metas:
            seen[m.instruction_slug] = m.instruction
        return list(seen.values())

    def __repr__(self) -> str:
        return (
            f"DungeonHandler(root={str(self._root)!r}, "
            f"samples={len(self._metas)})"
        )

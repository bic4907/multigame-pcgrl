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

전처리 필터 (캐시 저장 전 적용, legacy annotation 기준)
---------------------------------------------
1. RG(region, reward_enum==1) 값이 25 또는 35인 샘플 제거
2. BD(bat_direction, reward_enum==5) 샘플을 instruction별로 절반 제거
   (key 오름차순 정렬 후 앞 절반 유지 → 재현 가능)
3. 전체 4000개로 절단
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterator, List, Optional

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

# dataset/annotation/legacy/dungeon_reward_annotations.csv
_LEGACY_ANNOT_PATH = (
    Path(__file__).parent.parent.parent
    / "annotation" / "legacy" / "dungeon_reward_annotations.csv"
)

# ── 전처리 상수 ─────────────────────────────────────────────────────────────────
_EXCLUDE_RG: frozenset[int] = frozenset({25, 35})
_TARGET_COUNT: int = 4_000


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


def _apply_preprocess_filter(all_keys: List[str]) -> List[str]:
    """
    legacy annotation CSV 를 읽어 전처리 필터 후 남길 key 목록을 반환한다.

    필터 적용 순서
    --------------
    1. reward_enum==1 (RG) 이면서 condition_1 in _EXCLUDE_RG 인 샘플 제거
    2. reward_enum==5 (BD) 샘플을 instruction 별로 절반 유지
       (key 오름차순 정렬 후 앞 절반 → 재현 가능)
    3. 전체 _TARGET_COUNT 개로 절단 (원래 순서 유지)

    annotation CSV 가 없으면 all_keys 를 그대로 반환한다.
    """
    if not _LEGACY_ANNOT_PATH.exists():
        return all_keys

    annot: dict[str, dict] = {}
    with open(_LEGACY_ANNOT_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            annot[row["key"]] = row

    keep_set: set[str] = set()
    bd_by_instruction: dict[str, List[str]] = {}

    for key in all_keys:
        row = annot.get(key)
        if row is None:
            # annotation 에 없는 샘플은 그대로 유지
            keep_set.add(key)
            continue

        reward_enum = row.get("reward_enum", "")
        cond1_raw   = row.get("condition_1", "")

        if reward_enum == "1":
            # RG 값이 제외 대상이면 스킵
            try:
                rg = int(float(cond1_raw))
            except (ValueError, TypeError):
                rg = -1
            if rg in _EXCLUDE_RG:
                continue

        if reward_enum == "5":
            # BD 샘플은 instruction 별로 그룹화해서 나중에 처리
            instr = row.get("instruction", "")
            bd_by_instruction.setdefault(instr, []).append(key)
        else:
            keep_set.add(key)

    # BD: instruction 별로 정렬 후 앞 절반만 유지
    for instr in sorted(bd_by_instruction):
        group = sorted(bd_by_instruction[instr])   # key 오름차순 → 재현 가능
        half = max(1, len(group) // 2)
        keep_set.update(group[:half])

    # 원래 순서 유지 후 절단
    filtered = [k for k in all_keys if k in keep_set]
    return filtered[:_TARGET_COUNT]


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

        # 전처리 필터 적용 (캐시 저장 전)
        all_keys = [m.key for m in sorted(raw_metas.values(), key=lambda m: m.index)]

        # 1차: ndim!=2 맵 제거 (형식 이상 맵, RuntimeWarning 발생원)
        all_keys = [k for k in all_keys if self._archive[k].ndim == 2]

        # 2차: legacy annotation 기반 필터 (RG, BD, 4000개 절단)
        kept_keys = _apply_preprocess_filter(all_keys)

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
        array = _place_treasure(array, source_id)
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

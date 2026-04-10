"""
dataset/multigame/handlers/d5_handler.py
========================================
Dungeon D5 핸들러.

d2 핸들러를 기반으로 하되, **reward_enum=4 (collectable/treasure)** 카테고리를
추가한다.

기존 3개 enum (0=region, 1=path_length, 3=hazard/bat) 은 d2 와 동일하고,
추가로 ~1000 개 샘플을 랜덤 선택하여 `_place_treasure` 로 보물을 배치한 뒤
배치된 개수를 condition 값으로 어노테이션한다.

글로벌 5-class 체계:
    0: region      — d2 legacy
    1: path_length — d2 legacy
    3: hazard(bat) — d2 legacy
    4: collectable — ★ 신규: treasure 배치 개수

타일 매핑:
  0: padding / unknown
  1: floor
  2: wall
  3: enemy
  4: treasure
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..base import (
    BaseGameHandler,
    GameSample,
    GameTag,
    enforce_top_left_16x16,
)

# d2 핸들러에서 재사용
from .d2_handler import (
    _classify_slug,
    _quantize_instruction,
    _make_legend,
    _D2Meta,
)
from .dungeon_handler import DungeonTile, _place_treasure

logger = logging.getLogger(__name__)

_DEFAULT_D5_ROOT = (
    Path(__file__).parent.parent.parent / "dungeon_level_dataset"
)


# ── collectable 설정 ─────────────────────────────────────────────────────────────
_COLLECTABLE_ENUM = 4
_COLLECTABLE_SEED = 20260409          # 재현용 고정 시드
_COLLECTABLE_COUNT = 1000             # 랜덤 선택 샘플 수

# treasure 개수 → 4-bin 매핑 (condition_value & instruction 생성용)
#   bin 0: 1~2 → "few"       → condition = 2
#   bin 1: 3~4 → "some"      → condition = 4
#   bin 2: 5~6 → "several"   → condition = 6
#   bin 3: 7~9 → "many"      → condition = 8
_TREASURE_BIN_EDGES = [
    (2,  0),   # count <= 2  → bin 0
    (4,  1),   # count <= 4  → bin 1
    (6,  2),   # count <= 6  → bin 2
    (9,  3),   # count <= 9  → bin 3
]

# bin → 대표 condition 값 (다른 enum의 CSV 기반 값과 동일한 패턴)
_COLLECTABLE_BIN_TO_CONDITION: List[float] = [2.0, 4.0, 6.0, 8.0]

_COLLECTABLE_INSTRUCTIONS: Dict[int, List[str]] = {
    0: [
        "A few treasures are scattered on the dungeon floor.",
        "Sparse treasure placement with minimal pickups.",
        "The dungeon contains a few collectables.",
        "Minimal treasure can be found in this level.",
    ],
    1: [
        "Some treasures are placed around the dungeon.",
        "A moderate amount of treasure is hidden on the floor.",
        "Several collectables are spread across the map.",
        "The dungeon holds a decent number of treasures.",
    ],
    2: [
        "Numerous treasures are scattered throughout the dungeon.",
        "The floor is dotted with several collectables.",
        "A generous amount of treasure awaits the player.",
        "Multiple treasures are placed across the level.",
    ],
    3: [
        "Many treasures fill the dungeon floor.",
        "Treasures are abundantly spread across the map.",
        "The dungeon is rich with collectables everywhere.",
        "A large number of treasures dominate the floor tiles.",
    ],
}


def _treasure_bin(count: int) -> int:
    """treasure 개수 → 4-bin (0~3)."""
    for edge, bv in _TREASURE_BIN_EDGES:
        if count <= edge:
            return bv
    return 3


def _collectable_instruction(count: int, rng: np.random.RandomState) -> str:
    """treasure 개수에 맞는 instruction 텍스트 생성."""
    bv = _treasure_bin(count)
    templates = _COLLECTABLE_INSTRUCTIONS[bv]
    return templates[rng.randint(0, len(templates))]


def _count_treasure(array: np.ndarray) -> int:
    """array 에서 TREASURE(4) 타일 개수를 반환."""
    return int(np.sum(array == DungeonTile.TREASURE))


# ── Handler ──────────────────────────────────────────────────────────────────

class D5Handler(BaseGameHandler):
    """
    Dungeon D5 핸들러.

    d2 와 동일한 npz + metadata 를 로드하되:
      - 기존 3개 reward_enum (0, 1, 3) 은 d2 legacy 그대로
      - reward_enum=4 (collectable): ~1000 개 랜덤 샘플에
        treasure 를 배치하고, 배치 개수를 condition 으로 어노테이션
      - game_tag = "d5"

    Parameters
    ----------
    root              : dungeon_level_dataset 폴더 경로
    npz_name          : npz 파일명 (기본 'dungeon_levels.npz')
    meta_name         : csv 파일명 (기본 'dungeon_levels_metadata.csv')
    collectable_count : collectable 샘플 수 (기본 1000)
    collectable_seed  : collectable 선택/instruction 생성용 시드
    """

    def __init__(
        self,
        root: Path | str = _DEFAULT_D5_ROOT,
        npz_name: str = "dungeon_levels.npz",
        meta_name: str = "dungeon_levels_metadata.csv",
        collectable_count: int = _COLLECTABLE_COUNT,
        collectable_seed: int = _COLLECTABLE_SEED,
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
        self._metas: List[_D2Meta] = []
        self._key_to_meta: Dict[str, _D2Meta] = {}

        # ── 원본 metadata 로드 (d2 와 동일) ──────────────────────────────────
        raw_metas: list[_D2Meta] = []
        with open(meta_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                m = _D2Meta(
                    index=row["index"],
                    key=row["key"],
                    instruction=row["instruction"],
                    instruction_slug=row["instruction_slug"],
                    level_id=row["level_id"],
                    sample_id=row["sample_id"],
                )
                raw_metas.append(m)

        raw_metas.sort(key=lambda m: m.index)
        kept = [m for m in raw_metas if self._archive[m.key].ndim == 2]

        for m in kept:
            self._metas.append(m)
            self._key_to_meta[m.key] = m

        # ── collectable 샘플 선택 ────────────────────────────────────────────
        # 전체 키 중 랜덤으로 collectable_count 개 선택
        # _place_treasure 로 보물 배치 후 개수>0 인 것만 유지
        all_keys: List[str] = [m.key for m in self._metas]
        sel_rng = np.random.RandomState(collectable_seed)
        n_select = min(collectable_count, len(all_keys))
        candidate_indices: List[int] = sel_rng.choice(
            len(all_keys), size=n_select, replace=False,
        ).tolist()

        # key → (treasure_count, treasure_bin, instruction_text) 매핑
        self._collectable_keys: Dict[str, Tuple[int, int, str]] = {}
        instr_rng = np.random.RandomState(collectable_seed + 1)

        for idx in candidate_indices:
            key = str(all_keys[idx])
            raw = self._archive[key].astype(np.int32)
            arr_with_treasure = _place_treasure(raw, key)
            t_count = _count_treasure(arr_with_treasure)
            if t_count == 0:
                continue  # treasure 0 개 → collectable 에 부적합
            t_bin = _treasure_bin(t_count)
            instr = _collectable_instruction(t_count, instr_rng)
            self._collectable_keys[key] = (t_count, t_bin, instr)

        # 통계 로그
        from collections import Counter
        slug_cats = Counter()
        for m in self._metas:
            re, _ = _classify_slug(m.instruction_slug)
            slug_cats[re] += 1

        treasure_bins = Counter(tb for _, tb, _ in self._collectable_keys.values())
        logger.info(
            "D5 loaded: %d base samples, slug classification: %s, "
            "collectable: %d samples (bin dist: %s)",
            len(self._metas),
            dict(sorted(slug_cats.items())),
            len(self._collectable_keys),
            dict(sorted(treasure_bins.items())),
        )

    @property
    def game_tag(self) -> str:
        return GameTag.D5

    # ── BaseGameHandler ─────────────────────────────────────────────────────────
    def list_entries(self) -> List[str]:
        """npz key 목록 반환 (기존 + collectable 전용 키 접미사)."""
        entries = [m.key for m in self._metas]
        # collectable 샘플은 "key::collect" 형태로 구분
        for key in self._collectable_keys:
            entries.append(f"{key}::collect")
        return entries

    def __iter__(self):
        """block(reward_enum=-1) 제외 이터레이션 + collectable 샘플 추가."""
        order = 0
        # 1) 기존 d2 legacy 샘플 (region, path_length, hazard)
        for m in self._metas:
            re, _ = _classify_slug(m.instruction_slug)
            if re < 0:
                continue  # block 등 미지원 카테고리 → 스킵
            sample = self.load_sample(m.key, order=order)
            order += 1
            yield sample

        # 2) collectable 샘플
        for key in self._collectable_keys:
            sample = self.load_sample(f"{key}::collect", order=order)
            order += 1
            yield sample

    def load_sample(self, source_id: str, order: Optional[int] = None) -> GameSample:
        """npz key (또는 key::collect) → GameSample 반환."""

        # ── collectable 샘플 ────────────────────────────────────────────────
        if source_id.endswith("::collect"):
            return self._load_collectable_sample(source_id, order)

        # ── 기존 d2 legacy 샘플 ─────────────────────────────────────────────
        m = self._key_to_meta.get(source_id)
        if m is None:
            raise KeyError(f"Key not found in d5 dataset: {source_id!r}")
        raw = self._archive[source_id]
        array = raw.astype(np.int32)
        array = enforce_top_left_16x16(
            array,
            game=GameTag.D5,
            source_id=source_id,
        )
        array = _place_treasure(array, source_id)

        # reward_enum & condition: d2 legacy CSV 기반 대표 값 (enum별 4단계)
        reward_enum, feature_name = _classify_slug(m.instruction_slug)
        condition_value = _quantize_instruction(m.instruction, reward_enum)

        return GameSample(
            game=GameTag.D5,
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
                "reward_enum":      reward_enum,
                "feature_name":     feature_name,
                "sub_condition":    "",
                "conditions":       {reward_enum: condition_value},
            },
        )

    def _load_collectable_sample(
        self, source_id: str, order: Optional[int] = None,
    ) -> GameSample:
        """collectable (treasure) 샘플 로드."""
        base_key = source_id.removesuffix("::collect")
        m = self._key_to_meta.get(base_key)
        if m is None:
            raise KeyError(f"Base key not found in d5 dataset: {base_key!r}")
        info = self._collectable_keys.get(base_key)
        if info is None:
            raise KeyError(
                f"Key {base_key!r} is not a collectable sample in d5"
            )
        treasure_count, treasure_bin, instruction_text = info
        condition_value = _COLLECTABLE_BIN_TO_CONDITION[treasure_bin]

        raw = self._archive[base_key]
        array = raw.astype(np.int32)
        array = enforce_top_left_16x16(
            array,
            game=GameTag.D5,
            source_id=base_key,
        )
        array = _place_treasure(array, base_key)

        return GameSample(
            game=GameTag.D5,
            source_id=source_id,
            array=array,
            char_grid=None,
            legend=self._legend,
            instruction=instruction_text,
            order=order if order is not None else m.index,
            meta={
                "instruction_slug": f"collectable_treasure_bin{treasure_bin}",
                "level_id":         m.level_id,
                "sample_id":        m.sample_id,
                "reward_enum":      _COLLECTABLE_ENUM,
                "feature_name":     "collectable",
                "sub_condition":    "",
                "conditions":       {_COLLECTABLE_ENUM: condition_value},
                "treasure_count":   treasure_count,
                "treasure_bin":     treasure_bin,
            },
        )

    # ── 확장 쿼리 메서드 ─────────────────────────────────────────────────────────
    def filter_by_instruction(
        self, keyword: str, *, case_sensitive: bool = False
    ) -> List[GameSample]:
        """instruction 에 keyword 가 포함된 샘플 목록 반환."""
        kw = keyword if case_sensitive else keyword.lower()
        result = []
        for i, m in enumerate(self._metas):
            text = m.instruction if case_sensitive else m.instruction.lower()
            if kw in text:
                result.append(self.load_sample(m.key, order=i))
        # collectable 샘플도 검색
        for key, (_, _, instr) in self._collectable_keys.items():
            text = instr if case_sensitive else instr.lower()
            if kw in text:
                result.append(self.load_sample(f"{key}::collect"))
        return result

    def collectable_samples(self) -> List[GameSample]:
        """collectable 샘플만 반환."""
        return [
            self.load_sample(f"{key}::collect", order=i)
            for i, key in enumerate(self._collectable_keys)
        ]

    def __repr__(self) -> str:
        eligible = sum(
            1 for m in self._metas
            if _classify_slug(m.instruction_slug)[0] >= 0
        )
        return (
            f"D5Handler(root={str(self._root)!r}, "
            f"base_samples={len(self._metas)}, "
            f"eligible_legacy={eligible}, "
            f"collectable={len(self._collectable_keys)})"
        )


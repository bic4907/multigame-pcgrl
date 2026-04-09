"""
dataset/multigame/handlers/d2_handler.py
========================================
Dungeon Legacy (d2) 핸들러.

dungeon_level_dataset의 동일한 npz + metadata.csv를 로드하되,
game_tag = "d2" 로 태깅한다.

- 원본 metadata.csv의 instruction을 그대로 사용 (game-specific 텍스트)
- reward_enum, conditions는 instruction 키워드에서 유도:
    0: region      → 수량 4단계 (few/moderate/multi/many)
    1: path_length → 길이 4단계 (nano·micro/short/balanced/long)
    2: block       → 밀도 4단계 (few/some/dense/high-density)
    3: bat         → 수량 4단계 (few/moderate/many/directional)
- condition_value는 이미 quantized된 0, 1, 2, 3 (추가 양자화 불필요)

타일 매핑은 dungeon과 동일:
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
    TileLegend,
    enforce_top_left_16x16,
)

_DEFAULT_D2_ROOT = (
    Path(__file__).parent.parent.parent / "dungeon_level_dataset"
)

# DungeonHandler와 동일한 treasure 배치 로직 재사용
from .dungeon_handler import _place_treasure

logger = logging.getLogger(__name__)


# ── instruction → (reward_enum, feature_name) 분류 ───────────────────────────
# "region" 키워드를 포함하더라도 bat 관련이면 bat으로 분류해야 하므로
# bat 키워드를 먼저 검사한다.
_SLUG_CATEGORY_RULES: List[Tuple[str, int, str]] = [
    # (keyword, reward_enum, feature_name)  — 우선순위 순서
    ("path",   1, "path_length"),
    ("block",  2, "block"),
    ("bat",    3, "bat_amount"),
    ("region", 0, "region"),
]


def _classify_slug(slug: str) -> Tuple[int, str]:
    """instruction_slug에서 reward_enum과 feature_name을 결정한다."""
    slug_lower = slug.lower()
    for keyword, reward_enum, feature_name in _SLUG_CATEGORY_RULES:
        if keyword in slug_lower:
            return reward_enum, feature_name
    return -1, "unknown"


# ═══════════════════════════════════════════════════════════════════════════════
#  Instruction 키워드 → 4단계 quantized bin (0, 1, 2, 3)
# ═══════════════════════════════════════════════════════════════════════════════
# 각 enum별로 instruction의 수량/레벨 키워드를 파싱하여 4개 bin으로 매핑.
# 규칙: (keyword_list, bin) 튜플을 우선순위 순서로 정의. 먼저 매칭되면 해당 bin.
#
# enum=0 (region): few/sparse → moderate/some → multi/balanced → many/numerous
# enum=1 (path):   nano/micro/minimal → short/brief → balanced/moderate → long/extended
# enum=2 (block):  few/sparse/minimal → some/moderate → dense/numerous → high-density/packed
# enum=3 (bat):    few/small → moderate(some/several/five/multi) → many/dense/swarm → directional
# ═══════════════════════════════════════════════════════════════════════════════

_REGION_BIN_RULES: List[Tuple[List[str], int]] = [
    # bin 3 — many  (many, numerous, large number, everywhere)  — 먼저 검사 (bin 2와 구분)
    (["many", "numerous", "large number", "everywhere"], 3),
    # bin 2 — multi/balanced  (multi-region, appear in, balanced distribution, multiple placed)
    #         "appear in small clusters"가 bin 0의 "small cluster"에 선점되지 않도록 먼저 검사
    (["multi-region", "appear in", "balanced distribution", "multiple placed"], 2),
    # bin 1 — moderate  (moderate, some, several)
    (["moderate", "some ", "several"], 1),
    # bin 0 — few  (few, sparse, small cluster)
    (["a few", "sparse", "small cluster", "contains a few"], 0),
]

_PATH_BIN_RULES: List[Tuple[List[str], int]] = [
    # bin 0 — very short  (nano, micro, minimal, extremely short)
    (["nano", "micro", "minimal", "extremely short"], 0),
    # bin 1 — short  (short, brief, concise)
    (["short", "brief", "concise"], 1),
    # bin 2 — medium  (balanced, moderate, medium, reasonable)
    #         "moderate"가 " long"(along 포함)보다 먼저 매칭되도록 선행 검사
    (["balanced", "moderate", "medium", "reasonable"], 2),
    # bin 3 — long  (long, extended, prolonged, challenging)
    #         " long"/"long "으로 검사하여 "along" 오매칭 방지
    ([" long", "long ", "extended", "prolonged", "challenging"], 3),
]

_BLOCK_BIN_RULES: List[Tuple[List[str], int]] = [
    # bin 3 — very dense  (high-density, packed, overcrowding, dense barriers)
    (["high-density", "packed", "overcrowding", "dense barriers"], 3),
    # bin 2 — dense  (dense, substantial, numerous, movement is restricted)
    (["dense", "substantial", "numerous", "movement is restricted"], 2),
    # bin 0 — few  (few, sparse, minimal, "blocks are scattered")
    (["few", "sparse", "minimal", "blocks are scattered"], 0),
    # bin 1 — moderate  (some, spacing, moderate, a number of)
    (["some", "spacing", "moderate", "a number of"], 1),
]

_BAT_BIN_RULES: List[Tuple[List[str], int]] = [
    # bin 3 — directional  (linear/radial + 방향 키워드)
    (["linear", "radial", "north", "south", "east", "west",
      "top-focused", "bottom-heavy", "right-side", "left side",
      "upper section", "bottom edge", "western zone"], 3),
    # bin 2 — many  (swarm, dense, large, many, flood, dominates)
    (["swarm", "dense group", "large ", "many "], 2),
    # bin 0 — few  (a few, few , small )
    (["a few", "few ", "small "], 0),
    # bin 1 — moderate  (some, several, number, five, multi-bat, spread, clusters, roam, contains)
    (["some", "several", "number", "five", "multi-bat",
      "spread", "clusters", "roam", "contains"], 1),
]

_ENUM_BIN_RULES: Dict[int, List[Tuple[List[str], int]]] = {
    0: _REGION_BIN_RULES,
    1: _PATH_BIN_RULES,
    2: _BLOCK_BIN_RULES,
    3: _BAT_BIN_RULES,
}


def _load_bin_to_condition_mapping() -> Dict[int, List[float]]:
    """
    dungeon_instruction_reward_mapping.csv에서 bin → 실제 condition 값 매핑을 로드한다.

    CSV의 reward_enum은 1-based, D2Handler의 reward_enum은 0-based이므로 변환한다.
    Returns: {d2_enum_0based: [cond_bin0, cond_bin1, cond_bin2, cond_bin3]}
    """
    csv_path = Path(__file__).parent.parent / "annotations" / "dungeon_lagacy" / "dungeon_instruction_reward_mapping.csv"
    if not csv_path.exists():
        logger.warning("dungeon_instruction_reward_mapping.csv not found: %s", csv_path)
        return {}

    import csv as csv_mod
    enum_conds: Dict[int, set] = {}
    with open(csv_path, encoding="utf-8") as f:
        for row in csv_mod.DictReader(f):
            csv_enum = int(row["reward_enum"])   # 1-based in CSV
            cond = float(row["condition"])
            enum_conds.setdefault(csv_enum, set()).add(cond)

    mapping: Dict[int, List[float]] = {}
    for csv_enum, conds in enum_conds.items():
        d2_enum = csv_enum - 1  # CSV 1-based → D2 0-based
        mapping[d2_enum] = sorted(conds)  # bin 0→smallest, 3→largest

    logger.info("D2 bin→condition mapping loaded: %s",
                {k: v for k, v in sorted(mapping.items())})
    return mapping


# 모듈 로드 시 한 번만 로드
_BIN_TO_CONDITION: Dict[int, List[float]] = _load_bin_to_condition_mapping()


def _quantize_instruction(instruction: str, reward_enum: int) -> float:
    """
    instruction 텍스트에서 quantized condition bin (0~3)을 추출하고,
    dungeon_instruction_reward_mapping.csv 기반으로 실제 condition 값을 반환한다.

    Returns: 실제 condition 값 (e.g., 5, 15, 25, 35 for region)
    """
    rules = _ENUM_BIN_RULES.get(reward_enum, [])
    instr_lower = instruction.lower()
    bin_val = 0  # fallback
    for keywords, bv in rules:
        for kw in keywords:
            if kw in instr_lower:
                bin_val = bv
                break
        else:
            continue
        break
    else:
        logger.warning("D2 unmatched instruction (enum=%d): %r → fallback bin=0",
                       reward_enum, instruction)

    # bin → 실제 condition 값 변환
    cond_list = _BIN_TO_CONDITION.get(reward_enum, [])
    if 0 <= bin_val < len(cond_list):
        return cond_list[bin_val]
    return float(bin_val)


# ── Handler ──────────────────────────────────────────────────────────────────

def _make_legend() -> TileLegend:
    return TileLegend(char_to_attrs={
        "1": ["passable", "floor"],
        "2": ["solid", "wall"],
        "3": ["enemy", "damaging"],
    })


class _D2Meta:
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


class D2Handler(BaseGameHandler):
    """
    Dungeon Legacy (d2) 핸들러.

    dungeon_level_dataset의 npz + metadata를 로드하되,
    game_tag="d2"로 태깅하여 기존 dungeon과 독립적으로 관리한다.

    DungeonHandler와 달리 legacy annotation 기반 전처리 필터를 적용하지 않고,
    원본 metadata의 모든 유효 샘플(ndim==2)을 그대로 로드한다.

    reward_enum과 conditions는 instruction 키워드에서 유도되며,
    condition_value는 이미 4단계로 quantized된 값 (0, 1, 2, 3)이다.

    Parameters
    ----------
    root      : dungeon_level_dataset 폴더 경로
    npz_name  : npz 파일명 (기본 'dungeon_levels.npz')
    meta_name : csv 파일명 (기본 'dungeon_levels_metadata.csv')
    """

    def __init__(
        self,
        root: Path | str = _DEFAULT_D2_ROOT,
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
        self._metas: List[_D2Meta] = []
        self._key_to_meta: Dict[str, _D2Meta] = {}

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

        # index 순으로 정렬 후 ndim==2 필터 (원본 전체 로드, 절단 없음)
        raw_metas.sort(key=lambda m: m.index)
        kept = [m for m in raw_metas if self._archive[m.key].ndim == 2]

        for m in kept:
            self._metas.append(m)
            self._key_to_meta[m.key] = m

        # slug 분류 통계 로그
        from collections import Counter
        slug_cats = Counter()
        for m in self._metas:
            re, _ = _classify_slug(m.instruction_slug)
            slug_cats[re] += 1
        logger.info("D2 slug classification: %s (total=%d)", dict(sorted(slug_cats.items())), len(self._metas))

    @property
    def game_tag(self) -> str:
        return GameTag.D2

    # ── BaseGameHandler ─────────────────────────────────────────────────────────
    def list_entries(self) -> List[str]:
        """npz key 목록 반환."""
        return [m.key for m in self._metas]

    def load_sample(self, source_id: str, order: Optional[int] = None) -> GameSample:
        """npz key → GameSample 반환."""
        m = self._key_to_meta.get(source_id)
        if m is None:
            raise KeyError(f"Key not found in d2 dataset: {source_id!r}")
        raw = self._archive[source_id]
        array = raw.astype(np.int32)
        array = enforce_top_left_16x16(
            array,
            game=GameTag.D2,
            source_id=source_id,
        )
        array = _place_treasure(array, source_id)

        # reward_enum & condition: instruction 키워드 기반 quantized bin (0~3)
        reward_enum, feature_name = _classify_slug(m.instruction_slug)
        condition_value = _quantize_instruction(m.instruction, reward_enum)

        return GameSample(
            game=GameTag.D2,
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

    def __repr__(self) -> str:
        return (
            f"D2Handler(root={str(self._root)!r}, "
            f"samples={len(self._metas)})"
        )

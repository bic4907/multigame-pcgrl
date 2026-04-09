"""
dataset/multigame/handlers/d3_handler.py
========================================
Dungeon D3 핸들러.

d2 핸들러를 기반으로 하되, 새로 어노테이션한
``dungeon_reward_annotations.csv`` 의 ``instruction_uni`` 컬럼을
instruction 텍스트로 사용한다.

reward_enum 과 condition 은 **d2 (legacy)** 로직을 그대로 따른다:
  - instruction_slug 키워드 기반 분류 (0=region, 1=path_length, 3=hazard/bat)
  - dungeon_instruction_reward_mapping.csv 기반 실제 condition 값

타일 매핑은 dungeon / d2 와 동일:
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
from typing import Dict, List, Optional

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
from .dungeon_handler import _place_treasure

logger = logging.getLogger(__name__)

_DEFAULT_D3_ROOT = (
    Path(__file__).parent.parent.parent / "dungeon_level_dataset"
)

# 새 어노테이션 CSV 경로
_DEFAULT_ANNOT_CSV = (
    Path(__file__).parent.parent.parent
    / "reward_annotations"
    / "dungeon_reward_annotations.csv"
)


def _load_uni_instructions(csv_path: Path) -> Dict[tuple, str]:
    """
    dungeon_reward_annotations.csv 에서 (sample_id, reward_enum) → instruction_uni 매핑을 로드한다.

    CSV 구조:
      - key (dg000000): 전체 고유 식별자
      - sample_id (000000): 기본 레벨 식별자 (같은 level 에 5개 reward_enum)
      - reward_enum (0-4): 글로벌 reward 카테고리
      - instruction_uni: 통합 어노테이션 instruction

    각 sample_id 는 5개 reward_enum 에 대해 각각 다른 instruction_uni 를 가진다.
    매핑 키: (sample_id_stripped, reward_enum_int)

    Returns: {(sample_id_str, reward_enum_int): instruction_uni}
    """
    mapping: Dict[tuple, str] = {}
    if not csv_path.exists():
        logger.warning("D3 annotation CSV not found: %s", csv_path)
        return mapping

    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sid = row.get("sample_id", "").strip()
            uni = row.get("instruction_uni", "").strip()
            re_str = row.get("reward_enum", "").strip()
            if not sid or not uni or uni == "None" or not re_str:
                continue
            # sample_id 를 metadata key 형식으로 정규화 (zero-padded 6자리)
            key = (sid, int(re_str))
            if key not in mapping:
                mapping[key] = uni

    logger.info("D3: loaded %d (sample_id, reward_enum) → instruction_uni entries from %s",
                len(mapping), csv_path.name)
    return mapping


class D3Handler(BaseGameHandler):
    """
    Dungeon D3 핸들러.

    d2 와 동일한 npz + metadata 를 로드하되:
      - instruction 텍스트 = dungeon_reward_annotations.csv 의 instruction_uni
      - reward_enum / condition = d2 (legacy) 키워드 분류 로직
      - game_tag = "d3"

    Parameters
    ----------
    root       : dungeon_level_dataset 폴더 경로
    npz_name   : npz 파일명 (기본 'dungeon_levels.npz')
    meta_name  : csv 파일명 (기본 'dungeon_levels_metadata.csv')
    annot_csv  : instruction_uni 를 포함한 어노테이션 CSV 경로
    """

    def __init__(
        self,
        root: Path | str = _DEFAULT_D3_ROOT,
        npz_name: str = "dungeon_levels.npz",
        meta_name: str = "dungeon_levels_metadata.csv",
        annot_csv: Path | str = _DEFAULT_ANNOT_CSV,
    ) -> None:
        self._root = Path(root)
        npz_path = self._root / npz_name
        meta_path = self._root / meta_name

        if not npz_path.exists():
            raise FileNotFoundError(f"NPZ not found: {npz_path}")
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata CSV not found: {meta_path}")

        self._archive = np.load(npz_path)
        self._legend = _make_legend()
        self._metas: List[_D2Meta] = []
        self._key_to_meta: Dict[str, _D2Meta] = {}

        # instruction_uni 매핑 로드: (sample_id, reward_enum) → instruction_uni
        self._uni_map: Dict[tuple, str] = _load_uni_instructions(Path(annot_csv))

        # 원본 metadata 로드 (d2 와 동일)
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

        # slug 분류 통계 로그
        from collections import Counter
        slug_cats = Counter()
        for m in self._metas:
            re, _ = _classify_slug(m.instruction_slug)
            slug_cats[re] += 1

        # instruction_uni 매칭 통계 (d2 legacy reward_enum 기준)
        uni_matched = 0
        for m in self._metas:
            re, _ = _classify_slug(m.instruction_slug)
            if re < 0:
                continue
            if (m.key, re) in self._uni_map:
                uni_matched += 1
        uni_eligible = sum(1 for m in self._metas if _classify_slug(m.instruction_slug)[0] >= 0)
        logger.info(
            "D3 loaded: %d samples, slug classification: %s, "
            "instruction_uni matched: %d/%d (eligible only)",
            len(self._metas),
            dict(sorted(slug_cats.items())),
            uni_matched,
            uni_eligible,
        )

    @property
    def game_tag(self) -> str:
        return GameTag.D3

    # ── BaseGameHandler ─────────────────────────────────────────────────────────
    def list_entries(self) -> List[str]:
        """npz key 목록 반환."""
        return [m.key for m in self._metas]

    def __iter__(self):
        """block 등 legacy 에 없는 카테고리(reward_enum=-1) 샘플을 제외하고 이터레이션."""
        order = 0
        for m in self._metas:
            re, _ = _classify_slug(m.instruction_slug)
            if re < 0:
                continue  # block 등 미지원 카테고리 → 스킵
            sample = self.load_sample(m.key, order=order)
            order += 1
            yield sample

    def load_sample(self, source_id: str, order: Optional[int] = None) -> GameSample:
        """npz key → GameSample 반환."""
        m = self._key_to_meta.get(source_id)
        if m is None:
            raise KeyError(f"Key not found in d3 dataset: {source_id!r}")
        raw = self._archive[source_id]
        array = raw.astype(np.int32)
        array = enforce_top_left_16x16(
            array,
            game=GameTag.D3,
            source_id=source_id,
        )
        array = _place_treasure(array, source_id)

        # reward_enum & condition: d2 (legacy) 방식
        reward_enum, feature_name = _classify_slug(m.instruction_slug)
        condition_value = _quantize_instruction(m.instruction, reward_enum)

        # instruction: annotation CSV 의 instruction_uni 사용
        # 매칭 키: (npz_key, d2_legacy_reward_enum)
        instruction_text = self._uni_map.get((m.key, reward_enum), m.instruction)

        return GameSample(
            game=GameTag.D3,
            source_id=source_id,
            array=array,
            char_grid=None,
            legend=self._legend,
            instruction=instruction_text,
            order=order if order is not None else m.index,
            meta={
                "instruction_slug": m.instruction_slug,
                "instruction_raw":  m.instruction,       # 원본 instruction 보존
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
        """instruction (uni) 에 keyword 가 포함된 샘플 목록 반환."""
        kw = keyword if case_sensitive else keyword.lower()
        result = []
        for i, m in enumerate(self._metas):
            re, _ = _classify_slug(m.instruction_slug)
            if re < 0:
                continue
            text = self._uni_map.get((m.key, re), m.instruction)
            if not case_sensitive:
                text = text.lower()
            if kw in text:
                result.append(self.load_sample(m.key, order=i))
        return result

    def __repr__(self) -> str:
        eligible = sum(1 for m in self._metas if _classify_slug(m.instruction_slug)[0] >= 0)
        uni_matched = sum(
            1 for m in self._metas
            if _classify_slug(m.instruction_slug)[0] >= 0
            and (m.key, _classify_slug(m.instruction_slug)[0]) in self._uni_map
        )
        return (
            f"D3Handler(root={str(self._root)!r}, "
            f"samples={len(self._metas)}, "
            f"eligible={eligible}, "
            f"uni_mapped={uni_matched})"
        )


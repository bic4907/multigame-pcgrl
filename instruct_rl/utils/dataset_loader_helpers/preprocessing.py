"""
instruct_rl/utils/dataset_loader_helpers/preprocessing.py
==========================================================
GameSample 리스트에 대한 공통 전처리 유틸리티.

모든 데이터 로딩 파이프라인(CPCGRL, IPCGRL, VIPCGRL, MGPCGRL,
CLIP 인코더, MLP 인코더)에서 동일하게 적용한다.
"""

from __future__ import annotations

import dataclasses
import logging
import os
from os.path import basename

log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logger = logging.getLogger(basename(__file__))
logger.setLevel(getattr(logging, log_level, logging.INFO))

# (game, reward_enum, cutoff): condition >= cutoff 인 샘플 제거
LONGTAIL_CUTOFF = [
    ("dungeon", 1, 80),   # path_length >= 80
    ("pokemon", 2, 150),  # interactive_count >= 150
    ("pokemon", 4, 29),   # collectable_count >= 29
]


def _invalid_instruction(inst) -> bool:
    if inst is None:
        return True
    s = str(inst).strip()
    return s == "" or s.lower() == "none" or s.lower() == "nan"


def apply_longtail_cut(samples: list) -> list:
    """LONGTAIL_CUTOFF 기준으로 극단적 condition 값의 샘플을 제거한다."""
    def _is_longtail(s) -> bool:
        reward_enum = s.meta.get("reward_enum")
        condition_value = s.meta.get("conditions", {}).get(reward_enum)
        if condition_value is None:
            return False
        return any(
            s.game == game and reward_enum == enum and condition_value >= cutoff
            for game, enum, cutoff in LONGTAIL_CUTOFF
        )
    return [s for s in samples if not _is_longtail(s)]


def apply_tile_offset(samples: list, offset: int) -> list:
    """각 샘플의 array 타일 값에 offset을 더한 새 샘플 리스트를 반환한다."""
    if offset == 0:
        return samples
    return [dataclasses.replace(s, array=s.array + offset) for s in samples]


def preprocess_samples(samples: list, *, longtail_cut: bool = True) -> list:
    """공통 샘플 전처리: invalid instruction 필터 + longtail cut.

    인코더 학습과 RL 학습 모두에서 동일하게 적용한다.
    """
    n_before = len(samples)
    dropped_combos = sorted(set(
        (s.game, s.meta.get("reward_enum"))
        for s in samples if _invalid_instruction(s.instruction)
    ))
    samples = [s for s in samples if not _invalid_instruction(s.instruction)]
    n_dropped = n_before - len(samples)
    if n_dropped > 0:
        logger.info(
            "Instruction filter: %d → %d (dropped %d). Dropped (game, re) combos: %s",
            n_before, len(samples), n_dropped, dropped_combos,
        )
    else:
        logger.info("Instruction filter: all %d samples valid.", n_before)

    if longtail_cut:
        n_before_lt = len(samples)
        samples = apply_longtail_cut(samples)
        logger.info(
            "Longtail cut: %d → %d (removed %d)",
            n_before_lt, len(samples), n_before_lt - len(samples),
        )

    return samples

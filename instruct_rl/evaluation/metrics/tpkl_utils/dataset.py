"""
dataset.py
==========
MultiGameDataset에서 GT 레벨을 로드한다.
"""
from __future__ import annotations

import logging
import os
from os.path import basename
from typing import Iterable

import numpy as np

logger = logging.getLogger(__name__)


def load_gt_levels(
    train_ratio: float = 1.0,
    seed: int = 42,
    games: Iterable[str] | None = None,
    reward_enums: Iterable[int] | None = None,
) -> np.ndarray:
    """
    MultiGameDataset에서 reward annotation이 있는 레벨을 로드하여 (N, H, W) 배열로 반환.

    Parameters
    ----------
    train_ratio   : 0 < x ≤ 1.0. 1.0 이면 전체, 미만이면 train split만 반환
    seed          : train split 랜덤 시드
    games         : 포함할 게임 이름 목록. None 이면 전체.
    reward_enums  : 포함할 reward_enum 값 목록. None 이면 전체.

    Returns
    -------
    np.ndarray (N, H, W) int32
    """
    from dataset.multigame import MultiGameDataset

    games_set = set(games) if games is not None else None
    re_set    = set(reward_enums) if reward_enums is not None else None

    logger.info(
        "Loading MultiGameDataset (games=%s, reward_enums=%s) ...",
        games_set, re_set,
    )
    ds = MultiGameDataset(use_tile_mapping=True)
    annotated = ds.with_reward_annotation()
    logger.info("Annotated samples total: %d", len(annotated))

    raw: list = []
    for s in annotated:
        re = s.meta.get("reward_enum")
        if re is None:
            continue
        if s.meta.get("conditions", {}).get(re) is None:
            continue
        if games_set is not None and s.game not in games_set:
            continue
        if re_set is not None and re not in re_set:
            continue
        raw.append(s.array.astype(np.int32))

    if not raw:
        raise ValueError(
            f"[TPKL] No GT levels found for games={games_set}, reward_enums={re_set}"
        )

    arr = np.stack(raw)

    if train_ratio < 1.0:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(arr))
        arr = arr[: max(1, int(len(arr) * train_ratio))]

    logger.info("GT levels loaded: %d", len(arr))
    return arr


"""
dataset.py
==========
MultiGameDataset에서 GT 레벨을 로드한다.
"""
from __future__ import annotations

import logging
import os
from os.path import basename

import numpy as np

log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logger = logging.getLogger(basename(__file__))
logger.setLevel(getattr(logging, log_level, logging.INFO))


def load_gt_levels(train_ratio: float = 1.0, seed: int = 42) -> np.ndarray:
    """
    MultiGameDataset에서 reward annotation이 있는 레벨을 로드하여 (N, H, W) 배열로 반환.

    Parameters
    ----------
    train_ratio : 0 < x ≤ 1.0. 1.0 이면 전체, 미만이면 train split만 반환
    seed        : train split 랜덤 시드

    Returns
    -------
    np.ndarray (N, H, W) int32
    """
    from dataset.multigame import MultiGameDataset

    logger.info("[TPKL] Loading MultiGameDataset ...")
    ds = MultiGameDataset(use_tile_mapping=True)
    annotated = ds.with_reward_annotation()
    logger.info("[TPKL] Annotated samples: %d", len(annotated))

    raw: list = []
    for s in annotated:
        if s.meta.get("reward_enum") is None:
            continue
        if s.meta.get("conditions", {}).get(s.meta["reward_enum"]) is None:
            continue
        raw.append(s.array.astype(np.int32))

    arr = np.stack(raw)

    if train_ratio < 1.0:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(arr))
        arr = arr[: max(1, int(len(arr) * train_ratio))]

    logger.info("[TPKL] GT levels loaded: %d", len(arr))
    return arr


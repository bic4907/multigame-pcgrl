"""
distribution.py
===============
GT 레벨 집합으로부터 타일 패턴 분포를 계산한다.
"""
from __future__ import annotations

import numpy as np

from instruct_rl.evaluation.metrics.tpkl_utils.patch import (
    MAX_TILE,
    extract_windows,
    patches_to_dist,
)


def build_gt_distribution(gt_levels: np.ndarray,
                           window_sizes: tuple,
                           epsilon: float) -> list[dict]:
    """
    gt_levels: (M, H, W) → window_sizes별 패턴 분포 리스트.

    Parameters
    ----------
    gt_levels    : (M, H, W) int
    window_sizes : 슬라이딩 윈도우 크기 목록 e.g. (2, 3)
    epsilon      : Laplace smoothing 값

    Returns
    -------
    list of {hash_int: prob} — window_sizes 순서와 동일
    """
    n_tiles = int(gt_levels.max()) + 1 if gt_levels.size > 0 else MAX_TILE
    dists = []
    for k in window_sizes:
        wins = extract_windows(gt_levels, k)
        M, P, k2 = wins.shape
        patches = wins.reshape(M * P, k2)
        # n_tiles를 함께 저장해 scoring 쪽에서 해시 공간을 일치시킴
        dists.append({"n_tiles": n_tiles, "dist": patches_to_dist(patches, epsilon, n_tiles)})
    return dists


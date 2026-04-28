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
                           epsilon: float,
                           _pbar=None) -> list[dict]:
    """
    gt_levels: (M, H, W) → window_sizes별 패턴 분포 리스트.
    """
    n_tiles = int(gt_levels.max()) + 1 if gt_levels.size > 0 else MAX_TILE
    dists = []
    for k in window_sizes:
        wins = extract_windows(gt_levels, k)
        M, P, k2 = wins.shape
        patches = wins.reshape(M * P, k2)
        dists.append({"n_tiles": n_tiles, "dist": patches_to_dist(patches, epsilon, n_tiles)})
        if _pbar is not None:
            _pbar.set_postfix_str(f"GT dist w={k}")
            _pbar.update(0)   # postfix만 갱신 (update는 scoring에서)
    return dists


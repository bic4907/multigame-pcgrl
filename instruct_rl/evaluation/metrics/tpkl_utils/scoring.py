"""
scoring.py
==========
예측 레벨과 GT 분포 사이의 JSD 스코어를 계산한다.
"""
from __future__ import annotations

import numpy as np

from instruct_rl.evaluation.metrics.tpkl_utils.patch import MAX_TILE, extract_windows


def compute_jsd_scores(pred_levels: np.ndarray,
                       gt_dists: list[dict],
                       window_sizes: tuple,
                       epsilon: float) -> np.ndarray:
    """
    pred_levels 각각에 대해 GT 분포와의 JSD를 계산.

    Parameters
    ----------
    pred_levels : (N, H, W) int
    gt_dists    : build_gt_distribution() 반환값
    window_sizes: 슬라이딩 윈도우 크기 목록
    epsilon     : Laplace smoothing 값 (미등록 패턴 fallback)

    Returns
    -------
    scores : (N,) float  — 낮을수록 GT와 분포가 유사
    """
    n_tiles = int(pred_levels.max()) + 1 if pred_levels.size > 0 else MAX_TILE
    N = pred_levels.shape[0]
    scores = np.zeros(N, dtype=float)

    for k_idx, k in enumerate(window_sizes):
        wins = extract_windows(pred_levels, k)   # (N, P, k²)
        _, P, k2 = wins.shape
        gt_dist = gt_dists[k_idx]

        bases  = (n_tiles ** np.arange(k2, dtype=np.int64)).reshape(1, k2)
        hashes = (wins.astype(np.int64).reshape(N, P, k2) * bases).sum(axis=2)  # (N, P)

        gt_keys  = np.array(list(gt_dist.keys()),   dtype=np.int64)
        gt_probs = np.array(list(gt_dist.values()), dtype=float)

        for i in range(N):
            h_i   = hashes[i]
            n_bins = int(h_i.max()) + 1 if h_i.size > 0 else 1
            cnts  = np.bincount(h_i, minlength=n_bins).astype(float)
            nz    = cnts.nonzero()[0]
            sm    = cnts[nz] + epsilon
            sm   /= sm.sum()
            p_dist = dict(zip(nz.tolist(), sm.tolist()))

            # JSD = 0.5 * KL(p‖q) + 0.5 * KL(q‖p)
            qv    = np.array([gt_dist.get(int(k_), epsilon) for k_ in nz],    dtype=float)
            pv2   = np.array([p_dist.get(int(k_), epsilon) for k_ in gt_keys], dtype=float)
            kl_pq = float(np.sum(sm       * np.log(sm       / qv)))
            kl_qp = float(np.sum(gt_probs * np.log(gt_probs / pv2)))
            scores[i] += 0.5 * kl_pq + 0.5 * kl_qp

    return scores


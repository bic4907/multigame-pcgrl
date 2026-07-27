"""
scoring.py
==========
Compute JSD scores between predicted levels and the GT distribution.
"""
from __future__ import annotations

import numpy as np

from instruct_rl.evaluation.metrics.tpkl_utils.patch import MAX_TILE, extract_windows


def compute_jsd_scores(pred_levels: np.ndarray,
                       gt_dists: list[dict],
                       window_sizes: tuple,
                       epsilon: float,
                       _pbar=None) -> np.ndarray:
    """
    Compute JSD against the GT distribution for each predicted level.

    Parameters
    ----------
    pred_levels : (N, H, W) int
    gt_dists    : return value from build_gt_distribution()
                  Each item may be {hash_int: prob} or
                  {"n_tiles": int, "dist": {hash_int: prob}}.
    window_sizes: sliding-window sizes
    epsilon     : Laplace smoothing value used for unseen patterns

    Returns
    -------
    scores : (N,) float; lower means a distribution closer to GT
    """
    N = pred_levels.shape[0]
    scores = np.zeros(N, dtype=float)

    for k_idx, k in enumerate(window_sizes):
        # ── Parse gt_dist and n_tiles (backward compatibility) ──────────────
        raw = gt_dists[k_idx]
        if isinstance(raw, dict) and "n_tiles" in raw:
            n_tiles = int(raw["n_tiles"])
            gt_dist = raw["dist"]
        else:
            gt_dist = raw
            n_tiles = max(
                int(pred_levels.max()) + 1 if pred_levels.size > 0 else MAX_TILE,
                (max(gt_dist.keys()) + 1) if gt_dist else MAX_TILE,
            )

        wins = extract_windows(pred_levels, k)   # (N, P, k²)
        _, P, k2 = wins.shape

        # ── Compute hashes (N, P) ────────────────────────────────────────────
        bases  = (n_tiles ** np.arange(k2, dtype=np.int64)).reshape(1, 1, k2)
        hashes = (wins.astype(np.int64) * bases).sum(axis=2)   # (N, P)

        gt_keys  = np.array(list(gt_dist.keys()),   dtype=np.int64)
        gt_probs = np.array(list(gt_dist.values()), dtype=float)

        # ── Remap only observed hashes, reducing the representation to K dimensions ──
        # Memory: O(N*K), where K <= |GT keys| + |unique predicted keys|
        all_keys = np.unique(np.concatenate([gt_keys, hashes.ravel()]))
        K = len(all_keys)

        # Map hashes (N, P) to remap_idx (N, P) via searchsorted in O(log K)
        remap_idx = np.searchsorted(all_keys, hashes)   # (N, P)

        # ── Vectorized bincount using the offset trick ──────────────────────
        offsets   = (np.arange(N, dtype=np.int64) * K).reshape(N, 1)
        flat      = (remap_idx.astype(np.int64) + offsets).ravel()
        counts_2d = np.bincount(flat, minlength=N * K).reshape(N, K).astype(np.float32)

        # Laplace smoothing + normalize  →  (N, K)
        counts_2d += epsilon
        counts_2d /= counts_2d.sum(axis=1, keepdims=True)

        # ── Convert GT to a dense vector in the remapped space ──────────────
        gt_idx = np.searchsorted(all_keys, gt_keys)   # valid because gt_keys ⊆ all_keys
        gt_vec  = np.full(K, epsilon, dtype=np.float32)
        gt_vec[gt_idx] = gt_probs.astype(np.float32)
        gt_vec /= gt_vec.sum()

        # ── Fully vectorized JSD in float32 ─────────────────────────────────
        p = counts_2d            # (N, K)
        q = gt_vec[np.newaxis:]  # (1, K)
        m = np.float32(0.5) * (p + q)
        with np.errstate(divide="ignore", invalid="ignore"):
            kl_pm = np.where(p > 0, p * np.log(p / m), np.float32(0.0)).sum(axis=1)
            kl_qm = np.where(q > 0, q * np.log(q / m), np.float32(0.0)).sum(axis=1)
        scores += 0.5 * (kl_pm + kl_qm)
        if _pbar is not None:
            _pbar.set_postfix_str(f"JSD w={k} N={N}")
            _pbar.update(1)

    return scores

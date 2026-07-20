"""
scoring.py
==========
예측 레벨과 GT 분포 사이의 JSD 스코어를 계산한다.
"""
from __future__ import annotations

import os

import numpy as np

from instruct_rl.evaluation.metrics.tpkl_utils.patch import MAX_TILE, extract_windows

# Chunk pred_levels along N so the dense (chunk, K) count matrix stays bounded.
# Without this, np.bincount(minlength=N*K) allocates an O(N*K) array which, for
# window size 3 with large N (e.g. N=10000), balloons to tens of GB and thrashes
# RAM/CPU indefinitely.
_JSD_MAX_CELLS = int(os.environ.get("TPKL_JSD_MAX_CELLS", 50_000_000))


def compute_jsd_scores(pred_levels: np.ndarray,
                       gt_dists: list[dict],
                       window_sizes: tuple,
                       epsilon: float,
                       _pbar=None) -> np.ndarray:
    """
    pred_levels 각각에 대해 GT 분포와의 JSD를 계산.

    Parameters
    ----------
    pred_levels : (N, H, W) int
    gt_dists    : build_gt_distribution() 반환값
                  각 원소는 {hash_int: prob} 또는
                  {"n_tiles": int, "dist": {hash_int: prob}} 형태를 모두 지원
    window_sizes: 슬라이딩 윈도우 크기 목록
    epsilon     : Laplace smoothing 값 (미등록 패턴 fallback)

    Returns
    -------
    scores : (N,) float  — 낮을수록 GT와 분포가 유사
    """
    N = pred_levels.shape[0]
    scores = np.zeros(N, dtype=float)

    for k_idx, k in enumerate(window_sizes):
        # ── gt_dist & n_tiles 파싱 (하위 호환) ──────────────────────────────
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

        # ── 해시 계산 (N, P) ─────────────────────────────────────────────────
        bases  = (n_tiles ** np.arange(k2, dtype=np.int64)).reshape(1, 1, k2)
        hashes = (wins.astype(np.int64) * bases).sum(axis=2)   # (N, P)

        gt_keys  = np.array(list(gt_dist.keys()),   dtype=np.int64)
        gt_probs = np.array(list(gt_dist.values()), dtype=float)

        # ── 실제 등장한 해시만 remapping → K 차원으로 축소 ──────────────────
        # 메모리: O(N*K)  여기서 K ≤ |GT keys| + |pred unique keys|
        all_keys = np.unique(np.concatenate([gt_keys, hashes.ravel()]))
        K = len(all_keys)

        # hashes (N, P) → remap_idx (N, P)  ← searchsorted는 정렬된 배열에서 O(log K)
        remap_idx = np.searchsorted(all_keys, hashes)   # (N, P)

        # ── GT를 remapped 공간의 밀집 벡터로 변환 ────────────────────────────
        gt_idx = np.searchsorted(all_keys, gt_keys)   # valid because gt_keys ⊆ all_keys
        gt_vec  = np.full(K, epsilon, dtype=np.float32)
        gt_vec[gt_idx] = gt_probs.astype(np.float32)
        gt_vec /= gt_vec.sum()
        q = gt_vec[np.newaxis, :]  # (1, K)

        # ── 청크 단위 bincount + JSD ────────────────────────────────────────
        # 전체 (N, K) 밀집 행렬을 한 번에 만들면 N*K 가 수십억 원소로 폭증하여
        # 메모리를 고갈시키므로, 행(N)을 청크로 나눠 chunk*K 크기만 유지한다.
        chunk = max(1, min(N, _JSD_MAX_CELLS // max(K, 1)))
        for start in range(0, N, chunk):
            end = min(start + chunk, N)
            n = end - start
            remap_chunk = remap_idx[start:end]                    # (n, P)

            # ── 벡터화 bincount (offset trick, 청크 한정) ──────────────────
            offsets   = (np.arange(n, dtype=np.int64) * K).reshape(n, 1)
            flat      = (remap_chunk.astype(np.int64) + offsets).ravel()
            counts_2d = np.bincount(flat, minlength=n * K).reshape(n, K).astype(np.float32)

            # Laplace smoothing + 정규화  →  (n, K)
            counts_2d += epsilon
            counts_2d /= counts_2d.sum(axis=1, keepdims=True)

            # ── JSD (완전 벡터화, float32) ────────────────────────────────
            p = counts_2d            # (n, K)
            m = np.float32(0.5) * (p + q)
            with np.errstate(divide="ignore", invalid="ignore"):
                kl_pm = np.where(p > 0, p * np.log(p / m), np.float32(0.0)).sum(axis=1)
                kl_qm = np.where(q > 0, q * np.log(q / m), np.float32(0.0)).sum(axis=1)
            scores[start:end] += 0.5 * (kl_pm + kl_qm)

        if _pbar is not None:
            _pbar.set_postfix_str(f"JSD w={k} N={N}")
            _pbar.update(1)

    return scores


"""
patch.py
========
슬라이딩 윈도우 추출 및 패치 해싱 내부 헬퍼.
"""
from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

MAX_TILE = 64  # 타일 종류 상한; 초과 시 자동 조정됨


def hash_patches(patches: np.ndarray, n_tiles: int) -> np.ndarray:
    """(M, k²) int 패치 → (M,) int64 해시. base-n_tiles 인코딩."""
    k2 = patches.shape[1]
    bases = (n_tiles ** np.arange(k2, dtype=np.int64)).reshape(1, k2)
    return (patches.astype(np.int64) * bases).sum(axis=1)


def extract_windows(levels: np.ndarray, k: int) -> np.ndarray:
    """levels: (N, H, W) → (N, n_patches, k²), n_patches = (H-k+1)*(W-k+1)"""
    wins = sliding_window_view(levels, (k, k), axis=(1, 2))
    N, h, w, _, _ = wins.shape
    return wins.reshape(N, h * w, k * k)


def patches_to_dist(patches: np.ndarray, epsilon: float,
                    n_tiles: int | None = None) -> dict:
    """(M, k²) → {hash_int: 확률} dict. Laplace smoothing 적용."""
    if n_tiles is None:
        n_tiles = int(patches.max()) + 1 if patches.size > 0 else MAX_TILE
    hashes = hash_patches(patches, n_tiles)
    n_bins = int(hashes.max()) + 1 if hashes.size > 0 else 1
    counts = np.bincount(hashes, minlength=n_bins).astype(float)
    nonzero = counts.nonzero()[0]
    smoothed = counts[nonzero] + epsilon
    smoothed /= smoothed.sum()
    return dict(zip(nonzero.tolist(), smoothed.tolist()))


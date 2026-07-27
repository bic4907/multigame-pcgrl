"""
evaluator/metrics/tpkl.py
==========================
TPKL (Tile-Pattern KL-Divergence) evaluator.

Input: LevelBundle.array -- (H, W) int32 array with five unified categories
Score: symmetric KL divergence between sliding-window k-by-k pattern distributions;
       lower is more similar (closer to the GT distribution)

Pairwise implementation based on the tpkl_old.py sliding-window/symmetric-KL method.

Unified categories when use_tile_mapping=True:
  0=empty, 1=wall, 2=interactive, 3=hazard, 4=collectable
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, List, Tuple

import numpy as np

from .base import BaseMetricEvaluator, LevelBundle


# ── Module-level utilities matching tpkl_old.py ─────────────────────────────

def _sliding_windows(level: np.ndarray, k: int):
    """Yield k-by-k sliding-window patterns as tuples."""
    h, w = level.shape[:2]
    for i in range(h - k + 1):
        for j in range(w - k + 1):
            yield tuple(int(v) for v in level[i : i + k, j : j + k].flatten())


def _build_distribution(
    level: np.ndarray,
    window_sizes: Tuple[int, ...],
    epsilon: float,
) -> List[Dict[Tuple, float]]:
    """Return a Laplace-smoothed normalized distribution for each window size."""
    dists = []
    for k in window_sizes:
        counts: Counter = Counter()
        for key in _sliding_windows(level, k):
            counts[key] += 1
        smoothed = {key: v + epsilon for key, v in counts.items()}
        norm = sum(smoothed.values())
        dists.append({key: v / norm for key, v in smoothed.items()})
    return dists


def _kl(p: Dict, q: Dict, eps: float) -> float:
    """Compute KL(p || q), substituting eps for keys absent from q."""
    return float(sum(pv * np.log(pv / q.get(k, eps)) for k, pv in p.items()))


def _sym_kl(
    dists_p: List[Dict],
    dists_q: List[Dict],
    eps: float,
) -> float:
    """Sum symmetric KL divergence over window sizes.
    Zero means identical distributions; larger values mean greater difference.
    """
    return sum(
        0.5 * _kl(p, q, eps) + 0.5 * _kl(q, p, eps)
        for p, q in zip(dists_p, dists_q)
    )


# ── Evaluator class ──────────────────────────────────────────────────────────

class TPKLMetric(BaseMetricEvaluator):
    """
    Tile-Pattern KL-Divergence evaluator.

    Based on symmetric KL divergence between sliding-window pattern distributions.
    Lower KL divergence means greater similarity.

    To preserve the BaseMetricEvaluator interface, similarity_matrix() returns
    exp(-sym_KL) in (0, 1]. Use divergence_matrix() for raw KL divergence.

    Parameters
    ----------
    window_sizes : tuple of int
        Sliding-window sizes. Default: (2, 3).
    epsilon : float
        KL smoothing term. Default: 1e-6.
    """

    def __init__(
        self,
        window_sizes: Tuple[int, ...] = (2, 3),
        epsilon: float = 1e-6,
    ) -> None:
        self.window_sizes = window_sizes
        self.epsilon = epsilon

    # ── BaseMetricEvaluator implementation ───────────────────────────────────

    @property
    def name(self) -> str:
        return "TPKL"

    def similarity_matrix(self, bundles: List[LevelBundle]) -> np.ndarray:
        """
        (N, N) pairwise similarity matrix.

        similarity = exp(-sym_KL)  ∈ (0, 1]
        1.0 means identical distributions; values closer to zero are more different.
        """
        dists = [
            _build_distribution(b.array, self.window_sizes, self.epsilon)
            for b in bundles
        ]
        N = len(dists)
        mat = np.zeros((N, N), dtype=np.float64)
        for i in range(N):
            for j in range(N):
                kl = _sym_kl(dists[i], dists[j], self.epsilon)
                mat[i, j] = np.exp(-kl)
        return mat

    # ── Additional public API ─────────────────────────────────────────────────

    def divergence_matrix(self, bundles: List[LevelBundle]) -> np.ndarray:
        """
        (N, N) pairwise symmetric KL-divergence matrix.
        Lower is more similar; zero means identical distributions.
        """
        dists = [
            _build_distribution(b.array, self.window_sizes, self.epsilon)
            for b in bundles
        ]
        N = len(dists)
        mat = np.zeros((N, N), dtype=np.float64)
        for i in range(N):
            for j in range(N):
                mat[i, j] = _sym_kl(dists[i], dists[j], self.epsilon)
        return mat

    def score_divergence(self, a: LevelBundle, b: LevelBundle) -> float:
        """Return KL divergence for one pair; lower is more similar."""
        return float(self.divergence_matrix([a, b])[0, 1])

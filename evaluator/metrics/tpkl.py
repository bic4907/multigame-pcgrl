"""
evaluator/metrics/tpkl.py
==========================
TPKL (Tile-Pattern KL-Divergence) texttable.

text: LevelBundle.array — (H, W) int32 unified 5-category tile array
text: sliding window k×k text distribution text of  symmetric KL-divergence
      text text text text (GT distribution in   text text)

tpkl_old.py text (sliding window + symmetric KL) based pairwise text.

unified text (use_tile_mapping=True basis):
  0=empty, 1=wall, 2=interactive, 3=hazard, 4=collectable
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, List, Tuple

import numpy as np

from .base import BaseMetricEvaluator, LevelBundle


# ── text level utility (tpkl_old.py same  to text) ───────────────────────────────────

def _sliding_windows(level: np.ndarray, k: int):
    """k×k text text text also text text  tuple to  yield."""
    h, w = level.shape[:2]
    for i in range(h - k + 1):
        for j in range(w - k + 1):
            yield tuple(int(v) for v in level[i : i + k, j : j + k].flatten())


def _build_distribution(
    level: np.ndarray,
    window_sizes: Tuple[int, ...],
    epsilon: float,
) -> List[Dict[Tuple, float]]:
    """text also text sizetext Laplace-smoothed normalize distribution return."""
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
    """KL(p ‖ q)  (q  in  without key  eps text)."""
    return float(sum(pv * np.log(pv / q.get(k, eps)) for k, pv in p.items()))


def _sym_kl(
    dists_p: List[Dict],
    dists_q: List[Dict],
    eps: float,
) -> float:
    """text also text sizetext symmetric KL divergence sum.
    0 = same distribution, text  text text.
    """
    return sum(
        0.5 * _kl(p, q, eps) + 0.5 * _kl(q, p, eps)
        for p, q in zip(dists_p, dists_q)
    )


# ── texttable class ───────────────────────────────────────────────────────────────

class TPKLMetric(BaseMetricEvaluator):
    """
    Tile-Pattern KL-Divergence texttable.

    sliding window text distribution text symmetric KL-divergence based.
    KL divergence text  text text text text.

    BaseMetricEvaluator interface  keeptext abovetext
    similarity_matrix()  exp(-sym_KL) ∈ (0, 1]  to  converttext return.
    text KL divergence rowtext  text divergence_matrix() text for .

    Parameters
    ----------
    window_sizes : tuple of int
        text text text also text size list. default (2, 3).
    epsilon : float
        KL smoothing text. default 1e-6.
    """

    def __init__(
        self,
        window_sizes: Tuple[int, ...] = (2, 3),
        epsilon: float = 1e-6,
    ) -> None:
        self.window_sizes = window_sizes
        self.epsilon = epsilon

    # ── BaseMetricEvaluator text ──────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "TPKL"

    def similarity_matrix(self, bundles: List[LevelBundle]) -> np.ndarray:
        """
        (N, N) pairwise text also  rowtext.

        similarity = exp(-sym_KL)  ∈ (0, 1]
        1.0 = same distribution, 0 in   text distribution  text.
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

    # ── text  public API ─────────────────────────────────────────────────────────

    def divergence_matrix(self, bundles: List[LevelBundle]) -> np.ndarray:
        """
        (N, N) pairwise symmetric KL-divergence rowtext.
        text text text text (0 = same distribution).
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
        """text text KL divergence text (text text text)."""
        return float(self.divergence_matrix([a, b])[0, 1])

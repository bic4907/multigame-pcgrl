"""
evaluator/metrics/shannon_entropy.py
=====================================
Level-similarity metric based on Shannon entropy.

Compute the entropy of each level's tile distribution p and convert the
entropy difference between two levels to similarity.

  H(p) = -Σ p_i · log(p_i)   [nats]  ∈ [0, log(n_cats)]
  sim(i, j) = 1 - |H(p_i) - H(p_j)| / H_max

Input: LevelBundle.array -- (H, W) int32 array with five unified categories
Similarity: in [0, 1], where 1 means equal entropy (equal diversity)

Additional API:
  .entropy_scores(bundles) -> (N,) entropy values for analysis
"""
from __future__ import annotations

from typing import List

import numpy as np

from .base import BaseMetricEvaluator, LevelBundle


class ShannonEntropyMetric(BaseMetricEvaluator):
    """
    Tile-diversity similarity metric based on Shannon entropy.

    Levels in the same (game, reward_enum) group are expected to have similar
    tile diversity and therefore small entropy differences.

    Parameters
    ----------
    n_categories : int
        Number of unified tile categories (default: 5).
        H_max = log(n_categories) [nats] ≈ 1.609 (n_cats=5)
    eps : float
        Small value used to avoid log(0).
    """

    CAT_NAMES: List[str] = ["empty", "wall", "interactive", "hazard", "collectable"]

    def __init__(self, n_categories: int = 5, eps: float = 1e-10) -> None:
        self.n_categories = n_categories
        self.eps = eps
        self._h_max = float(np.log(n_categories))   # Maximum entropy for a uniform distribution

    # ── BaseMetricEvaluator implementation ───────────────────────────────────

    @property
    def name(self) -> str:
        return "ShannonEntropy"

    def similarity_matrix(self, bundles: List[LevelBundle]) -> np.ndarray:
        """
        (N, N) pairwise entropy-similarity matrix.

        sim[i, j] = 1 - |H(p_i) - H(p_j)| / H_max  ∈ [0, 1]
        The diagonal is 1.0.
        """
        entropies = self.entropy_scores(bundles)   # (N,)
        N = len(entropies)
        mat = np.zeros((N, N), dtype=np.float64)
        for i in range(N):
            for j in range(N):
                diff = abs(entropies[i] - entropies[j])
                mat[i, j] = 1.0 - diff / (self._h_max + self.eps)
        np.clip(mat, 0.0, 1.0, out=mat)
        return mat

    # ── Additional public API ─────────────────────────────────────────────────

    def entropy_scores(self, bundles: List[LevelBundle]) -> np.ndarray:
        """
        each level of  Shannon Entropy [nats] return.

        Returns
        -------
        np.ndarray : shape (N,), values in [0, log(n_categories)]
        """
        return np.array([self._entropy(b.array) for b in bundles], dtype=np.float64)

    def normalized_entropy_scores(self, bundles: List[LevelBundle]) -> np.ndarray:
        """
        Return normalized Shannon entropy in [0, 1].
        0 means only one tile type is present; 1 means all tile types have equal proportions.
        """
        return self.entropy_scores(bundles) / self._h_max

    # ── internal utility ─────────────────────────────────────────────────────────────

    def _tile_histogram(self, array: np.ndarray) -> np.ndarray:
        """(H, W) int32 → normalized histogram (n_categories,)"""
        flat   = np.clip(array.flatten(), 0, self.n_categories - 1).astype(np.int32)
        counts = np.bincount(flat, minlength=self.n_categories).astype(np.float64)
        return counts / (counts.sum() + self.eps)

    def _entropy(self, array: np.ndarray) -> float:
        """Return Shannon entropy in nats for a single level array."""
        p = self._tile_histogram(array)
        # Treat zero entries as 0*log(0) = 0 by continuity
        mask = p > self.eps
        return float(-np.sum(p[mask] * np.log(p[mask])))

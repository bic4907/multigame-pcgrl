"""
evaluator/metrics/shannon_entropy.py
=====================================
Shannon Entropy based level text also  texttable.

each level of  tile distribution p  in  text text to text  computetext,
text level of  text to text text   text also  to  converttext.

  H(p) = -Σ p_i · log(p_i)   [nats]  ∈ [0, log(n_cats)]
  sim(i, j) = 1 - |H(p_i) - H(p_j)| / H_max

text : LevelBundle.array — (H, W) int32 unified 5-category tile array
text also : ∈ [0, 1],  1 = same text to text (same text)

text  API:
  .entropy_scores(bundles)  → (N,) each level of  text to text text (text for )
"""
from __future__ import annotations

from typing import List

import numpy as np

from .base import BaseMetricEvaluator, LevelBundle


class ShannonEntropyMetric(BaseMetricEvaluator):
    """
    Shannon Entropy based tile text text also  texttable.

    same (game, reward_enum) text  text tile text   text to
    text to text text   text  text  text.

    Parameters
    ----------
    n_categories : int
        unified tile category text (default 5).
        H_max = log(n_categories) [nats] ≈ 1.609 (n_cats=5)
    eps : float
        log(0) text for  text.
    """

    CAT_NAMES: List[str] = ["empty", "wall", "interactive", "hazard", "collectable"]

    def __init__(self, n_categories: int = 5, eps: float = 1e-10) -> None:
        self.n_categories = n_categories
        self.eps = eps
        self._h_max = float(np.log(n_categories))   # text distributiontext text maximum text to text

    # ── BaseMetricEvaluator text ──────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "ShannonEntropy"

    def similarity_matrix(self, bundles: List[LevelBundle]) -> np.ndarray:
        """
        (N, N) pairwise text to text text also  rowtext.

        sim[i, j] = 1 - |H(p_i) - H(p_j)| / H_max  ∈ [0, 1]
        texteachtext = 1.0.
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

    # ── text  public API ─────────────────────────────────────────────────────────

    def entropy_scores(self, bundles: List[LevelBundle]) -> np.ndarray:
        """
        each level of  Shannon Entropy [nats] return.

        Returns
        -------
        np.ndarray : shape (N,), text range [0, log(n_categories)]
        """
        return np.array([self._entropy(b.array) for b in bundles], dtype=np.float64)

    def normalized_entropy_scores(self, bundles: List[LevelBundle]) -> np.ndarray:
        """
        normalizetext Shannon Entropy ∈ [0, 1] return.
        0 = text before  text distribution text (text tiletext text)
        1 = text distribution (text tile text  same ratio)
        """
        return self.entropy_scores(bundles) / self._h_max

    # ── internal utility ─────────────────────────────────────────────────────────────

    def _tile_histogram(self, array: np.ndarray) -> np.ndarray:
        """(H, W) int32 → normalized histogram (n_categories,)"""
        flat   = np.clip(array.flatten(), 0, self.n_categories - 1).astype(np.int32)
        counts = np.bincount(flat, minlength=self.n_categories).astype(np.float64)
        return counts / (counts.sum() + self.eps)

    def _entropy(self, array: np.ndarray) -> float:
        """text level array of  Shannon Entropy [nats]."""
        p = self._tile_histogram(array)
        # 0text text  0·log(0) = 0  as  process (text)
        mask = p > self.eps
        return float(-np.sum(p[mask] * np.log(p[mask])))


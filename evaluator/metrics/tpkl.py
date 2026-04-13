"""
evaluator/metrics/tpkl.py
==========================
TPKL (Tile Proportion KL-Divergence) 지표.

입력: LevelBundle.array — (H, W) int32 unified 5-category 타일 배열
유사도: 1 - JSD(tile_dist_i, tile_dist_j) / ln(2)  ∈ [0, 1]

unified 카테고리 (use_tile_mapping=True 기준):
  0=empty, 1=wall, 2=interactive, 3=hazard, 4=collectable
"""
from __future__ import annotations

from typing import List

import numpy as np

from .base import BaseMetricEvaluator, LevelBundle


class TPKLMetric(BaseMetricEvaluator):
    """
    Tile Proportion KL-Divergence 지표.

    게임 레벨의 타일 타입 분포(비율)를 Jensen-Shannon Divergence(JSD)로 비교.
    JSD 를 정규화하여 유사도로 변환:
        similarity = 1 - JSD(p, q) / ln(2)  ∈ [0, 1]

    Parameters
    ----------
    n_categories : int
        unified tile category 수. 기본 5.
    """

    CAT_NAMES: List[str] = ["empty", "wall", "interactive", "hazard", "collectable"]

    def __init__(self, n_categories: int = 5) -> None:
        self.n_categories = n_categories

    # ── BaseMetricEvaluator 구현 ──────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "TPKL"

    def similarity_matrix(self, bundles: List[LevelBundle]) -> np.ndarray:
        """(N, N) pairwise TPKL-similarity matrix."""
        hists = [self._tile_histogram(b.array) for b in bundles]
        N = len(hists)
        mat = np.zeros((N, N), dtype=np.float64)
        for i in range(N):
            for j in range(N):
                mat[i, j] = self._jsd_similarity(hists[i], hists[j])
        return mat

    # ── 추가 공개 API ─────────────────────────────────────────────────────────

    def histograms(self, bundles: List[LevelBundle]) -> np.ndarray:
        """(N, n_categories) 정규화 타일 히스토그램 배열."""
        return np.array([self._tile_histogram(b.array) for b in bundles])

    # ── 내부 유틸 ─────────────────────────────────────────────────────────────

    def _tile_histogram(self, array: np.ndarray) -> np.ndarray:
        """(H, W) int32 → normalized histogram (n_categories,)"""
        flat   = np.clip(array.flatten(), 0, self.n_categories - 1).astype(np.int32)
        counts = np.bincount(flat, minlength=self.n_categories).astype(np.float64)
        return counts / (counts.sum() + 1e-10)

    def _jsd(self, p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> float:
        """Jensen-Shannon Divergence ∈ [0, ln(2)]."""
        m      = 0.5 * (p + q)
        kl_pm  = np.sum(p * np.log((p + eps) / (m + eps)))
        kl_qm  = np.sum(q * np.log((q + eps) / (m + eps)))
        return float(np.clip(0.5 * (kl_pm + kl_qm), 0.0, np.log(2)))

    def _jsd_similarity(self, p: np.ndarray, q: np.ndarray) -> float:
        """JSD 기반 유사도 ∈ [0, 1].  1 = 동일 분포."""
        return 1.0 - self._jsd(p, q) / np.log(2)


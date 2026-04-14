"""
evaluator/metrics/shannon_entropy.py
=====================================
Shannon Entropy 기반 레벨 유사도 지표.

각 레벨의 타일 분포 p 에 대해 엔트로피를 계산하고,
두 레벨의 엔트로피 차이를 유사도로 변환한다.

  H(p) = -Σ p_i · log(p_i)   [nats]  ∈ [0, log(n_cats)]
  sim(i, j) = 1 - |H(p_i) - H(p_j)| / H_max

입력 : LevelBundle.array — (H, W) int32 unified 5-category 타일 배열
유사도: ∈ [0, 1],  1 = 동일 엔트로피 (동일 다양성)

추가 API:
  .entropy_scores(bundles)  → (N,) 각 레벨의 엔트로피 값 (분석용)
"""
from __future__ import annotations

from typing import List

import numpy as np

from .base import BaseMetricEvaluator, LevelBundle


class ShannonEntropyMetric(BaseMetricEvaluator):
    """
    Shannon Entropy 기반 타일 다양성 유사도 지표.

    동일 (game, reward_enum) 그룹은 비슷한 타일 다양성을 가지므로
    엔트로피 차이가 작을 것을 기대한다.

    Parameters
    ----------
    n_categories : int
        unified tile category 수 (기본 5).
        H_max = log(n_categories) [nats] ≈ 1.609 (n_cats=5)
    eps : float
        log(0) 방지용 소량.
    """

    CAT_NAMES: List[str] = ["empty", "wall", "interactive", "hazard", "collectable"]

    def __init__(self, n_categories: int = 5, eps: float = 1e-10) -> None:
        self.n_categories = n_categories
        self.eps = eps
        self._h_max = float(np.log(n_categories))   # 균등 분포일 때 최대 엔트로피

    # ── BaseMetricEvaluator 구현 ──────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "ShannonEntropy"

    def similarity_matrix(self, bundles: List[LevelBundle]) -> np.ndarray:
        """
        (N, N) pairwise 엔트로피 유사도 행렬.

        sim[i, j] = 1 - |H(p_i) - H(p_j)| / H_max  ∈ [0, 1]
        대각선 = 1.0.
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

    # ── 추가 공개 API ─────────────────────────────────────────────────────────

    def entropy_scores(self, bundles: List[LevelBundle]) -> np.ndarray:
        """
        각 레벨의 Shannon Entropy [nats] 반환.

        Returns
        -------
        np.ndarray : shape (N,), 값 범위 [0, log(n_categories)]
        """
        return np.array([self._entropy(b.array) for b in bundles], dtype=np.float64)

    def normalized_entropy_scores(self, bundles: List[LevelBundle]) -> np.ndarray:
        """
        정규화된 Shannon Entropy ∈ [0, 1] 반환.
        0 = 완전 균일 분포 아님 (한 타일만 존재)
        1 = 균등 분포 (모든 타일 타입이 동일 비율)
        """
        return self.entropy_scores(bundles) / self._h_max

    # ── 내부 유틸 ─────────────────────────────────────────────────────────────

    def _tile_histogram(self, array: np.ndarray) -> np.ndarray:
        """(H, W) int32 → normalized histogram (n_categories,)"""
        flat   = np.clip(array.flatten(), 0, self.n_categories - 1).astype(np.int32)
        counts = np.bincount(flat, minlength=self.n_categories).astype(np.float64)
        return counts / (counts.sum() + self.eps)

    def _entropy(self, array: np.ndarray) -> float:
        """단일 레벨 배열의 Shannon Entropy [nats]."""
        p = self._tile_histogram(array)
        # 0인 항목은 0·log(0) = 0 으로 처리 (극한값)
        mask = p > self.eps
        return float(-np.sum(p[mask] * np.log(p[mask])))


"""
evaluator/metrics/tpkl.py
==========================
TPKL (Tile-Pattern KL-Divergence) 지표.

입력: LevelBundle.array — (H, W) int32 unified 5-category 타일 배열
점수: sliding window k×k 패턴 분포 간의 symmetric KL-divergence
      낮을수록 더 유사 (GT 분포에 가까울수록 좋음)

tpkl_old.py 방식 (sliding window + symmetric KL) 기반 pairwise 구현.

unified 카테고리 (use_tile_mapping=True 기준):
  0=empty, 1=wall, 2=interactive, 3=hazard, 4=collectable
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, List, Tuple

import numpy as np

from .base import BaseMetricEvaluator, LevelBundle


# ── 모듈 레벨 유틸 (tpkl_old.py 동일 로직) ───────────────────────────────────

def _sliding_windows(level: np.ndarray, k: int):
    """k×k 슬라이딩 윈도우 패턴을 tuple로 yield."""
    h, w = level.shape[:2]
    for i in range(h - k + 1):
        for j in range(w - k + 1):
            yield tuple(int(v) for v in level[i : i + k, j : j + k].flatten())


def _build_distribution(
    level: np.ndarray,
    window_sizes: Tuple[int, ...],
    epsilon: float,
) -> List[Dict[Tuple, float]]:
    """윈도우 크기별 Laplace-smoothed 정규화 분포 반환."""
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
    """KL(p ‖ q)  (q 에 없는 key는 eps 대체)."""
    return float(sum(pv * np.log(pv / q.get(k, eps)) for k, pv in p.items()))


def _sym_kl(
    dists_p: List[Dict],
    dists_q: List[Dict],
    eps: float,
) -> float:
    """윈도우 크기별 symmetric KL divergence 합산.
    0 = 동일 분포, 값이 클수록 다름.
    """
    return sum(
        0.5 * _kl(p, q, eps) + 0.5 * _kl(q, p, eps)
        for p, q in zip(dists_p, dists_q)
    )


# ── 지표 클래스 ───────────────────────────────────────────────────────────────

class TPKLMetric(BaseMetricEvaluator):
    """
    Tile-Pattern KL-Divergence 지표.

    sliding window 패턴 분포 간 symmetric KL-divergence 기반.
    KL divergence 자체는 낮을수록 더 유사.

    BaseMetricEvaluator 인터페이스를 유지하기 위해
    similarity_matrix()는 exp(-sym_KL) ∈ (0, 1] 로 변환하여 반환.
    원시 KL divergence 행렬이 필요하면 divergence_matrix() 사용.

    Parameters
    ----------
    window_sizes : tuple of int
        슬라이딩 윈도우 크기 목록. 기본 (2, 3).
    epsilon : float
        KL smoothing 항. 기본 1e-6.
    """

    def __init__(
        self,
        window_sizes: Tuple[int, ...] = (2, 3),
        epsilon: float = 1e-6,
    ) -> None:
        self.window_sizes = window_sizes
        self.epsilon = epsilon

    # ── BaseMetricEvaluator 구현 ──────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "TPKL"

    def similarity_matrix(self, bundles: List[LevelBundle]) -> np.ndarray:
        """
        (N, N) pairwise 유사도 행렬.

        similarity = exp(-sym_KL)  ∈ (0, 1]
        1.0 = 동일 분포, 0에 가까울수록 분포가 다름.
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

    # ── 추가 공개 API ─────────────────────────────────────────────────────────

    def divergence_matrix(self, bundles: List[LevelBundle]) -> np.ndarray:
        """
        (N, N) pairwise symmetric KL-divergence 행렬.
        낮을수록 더 유사 (0 = 동일 분포).
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
        """단일 쌍 KL divergence 점수 (낮을수록 유사)."""
        return float(self.divergence_matrix([a, b])[0, 1])

"""
evaluator/metrics/ssim.py
==========================
SSIM (Structural Similarity Index Measure) 지표.

입력: LevelBundle.image — (H, W, 3) uint8 RGB 이미지
유사도: SSIM ∈ [-1, 1]  (1 = 완전 동일 구조)
의존: scikit-image  (pip install scikit-image)
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from .base import BaseMetricEvaluator, LevelBundle


class SSIMMetric(BaseMetricEvaluator):
    """
    Structural Similarity Index Measure (SSIM) 지표.

    skimage.metrics.structural_similarity 를 사용해
    렌더링된 RGB 이미지 쌍의 구조적 유사도를 측정한다.

    Parameters
    ----------
    win_size : int | None
        SSIM 윈도우 크기. None = skimage 기본값(7).
        이미지가 작을 경우 명시적으로 지정 필요.
    """

    def __init__(self, win_size: Optional[int] = None) -> None:
        self.win_size = win_size
        # 의존 패키지 사전 검증
        from skimage.metrics import structural_similarity  # noqa: F401

    # ── BaseMetricEvaluator 구현 ──────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "SSIM"

    def similarity_matrix(self, bundles: List[LevelBundle]) -> np.ndarray:
        """
        (N, N) pairwise SSIM matrix.  대각선 = 1.0.
        """
        N   = len(bundles)
        mat = np.eye(N, dtype=np.float64)
        for i in range(N):
            for j in range(i + 1, N):
                v         = self._ssim_pair(bundles[i].image, bundles[j].image)
                mat[i, j] = v
                mat[j, i] = v
        return mat

    # ── 내부 유틸 ─────────────────────────────────────────────────────────────

    def _ssim_pair(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Both (H, W, 3) uint8.  Returns SSIM ∈ [-1, 1]."""
        from skimage.metrics import structural_similarity as _ssim_fn
        kwargs: dict = dict(data_range=255)
        if self.win_size is not None:
            kwargs["win_size"] = self.win_size
        try:
            # scikit-image >= 0.19
            return float(_ssim_fn(img1, img2, channel_axis=2, **kwargs))
        except TypeError:
            # scikit-image < 0.19 fallback
            return float(_ssim_fn(img1, img2, multichannel=True, **kwargs))


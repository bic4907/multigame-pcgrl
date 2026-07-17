"""
evaluator/metrics/ssim.py
==========================
SSIM (Structural Similarity Index Measure) texttable.

text: LevelBundle.image — (H, W, 3) uint8 RGB image
text also : SSIM ∈ [-1, 1]  (1 = text before  same structure)
 of text: scikit-image  (pip install scikit-image)
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from .base import BaseMetricEvaluator, LevelBundle


class SSIMMetric(BaseMetricEvaluator):
    """
    Structural Similarity Index Measure (SSIM) texttable.

    skimage.metrics.structural_similarity   text for text
    renderingtext RGB image text of  structuretext text also   measuretext.

    Parameters
    ----------
    win_size : int | None
        SSIM text also text size. None = skimage default value(7).
        image  text  text text as  text text.
    """

    def __init__(self, win_size: Optional[int] = None) -> None:
        self.win_size = win_size
        #  of text text text before  validate
        from skimage.metrics import structural_similarity  # noqa: F401

    # ── BaseMetricEvaluator text ──────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "SSIM"

    def similarity_matrix(self, bundles: List[LevelBundle]) -> np.ndarray:
        """
        (N, N) pairwise SSIM matrix.  texteachtext = 1.0.
        """
        N   = len(bundles)
        mat = np.eye(N, dtype=np.float64)
        for i in range(N):
            for j in range(i + 1, N):
                v         = self._ssim_pair(bundles[i].image, bundles[j].image)
                mat[i, j] = v
                mat[j, i] = v
        return mat

    # ── internal utility ─────────────────────────────────────────────────────────────

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


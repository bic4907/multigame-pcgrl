"""
evaluator/metrics/lpips.py
===========================
LPIPS (Learned Perceptual Image Patch Similarity) 지표.

입력: LevelBundle.image — (H, W, 3) uint8 RGB 이미지
유사도: 1 - LPIPS_dist / max_dist  ∈ [0, 1]  (1 = 완전 동일)
의존: lpips, torch  (pip install lpips)
"""
from __future__ import annotations

from typing import List

import numpy as np

from .base import BaseMetricEvaluator, LevelBundle


class LPIPSMetric(BaseMetricEvaluator):
    """
    Learned Perceptual Image Patch Similarity (LPIPS) 지표.

    AlexNet/VGG 퍼셉추얼 거리를 정규화하여 유사도로 반환한다:
        similarity = 1 - dist / max_dist  ∈ [0, 1]

    Parameters
    ----------
    net : {"alex", "vgg", "squeeze"}
        LPIPS 백본 네트워크.  "alex" 가 가장 빠르고 권장됨.
    """

    def __init__(self, net: str = "alex") -> None:
        import lpips as _lpips_lib
        self.net = net
        self._loss_fn = _lpips_lib.LPIPS(net=net, verbose=False)
        self._loss_fn.eval()

    # ── BaseMetricEvaluator 구현 ──────────────────────────────────────────────

    @property
    def name(self) -> str:
        return f"LPIPS[{self.net}]"

    def similarity_matrix(self, bundles: List[LevelBundle]) -> np.ndarray:
        """
        (N, N) 유사도 행렬: sim = 1 − dist / max_dist.
        대각선 = 1.0.
        """
        dist_mat = self._distance_matrix(bundles)
        max_d    = dist_mat.max()
        sim      = 1.0 - dist_mat / max_d if max_d > 1e-8 else np.ones_like(dist_mat)
        np.fill_diagonal(sim, 1.0)
        return sim

    # ── 추가 공개 API ─────────────────────────────────────────────────────────

    def distance_matrix(self, bundles: List[LevelBundle]) -> np.ndarray:
        """(N, N) pairwise LPIPS 거리 행렬 (낮을수록 유사)."""
        return self._distance_matrix(bundles)

    # ── 내부 유틸 ─────────────────────────────────────────────────────────────

    def _to_tensor(self, img: np.ndarray):
        """(H, W, 3) uint8 → (1, 3, H, W) float32 tensor in [-1, 1]."""
        import torch
        t = torch.from_numpy(img.astype(np.float32) / 127.5 - 1.0)
        return t.permute(2, 0, 1).unsqueeze(0)

    def _distance_matrix(self, bundles: List[LevelBundle]) -> np.ndarray:
        import torch
        N       = len(bundles)
        tensors = [self._to_tensor(b.image) for b in bundles]
        mat     = np.zeros((N, N), dtype=np.float64)
        with torch.no_grad():
            for i in range(N):
                for j in range(i + 1, N):
                    d         = float(self._loss_fn(tensors[i], tensors[j]).item())
                    mat[i, j] = d
                    mat[j, i] = d
        return mat


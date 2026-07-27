"""
evaluator/metrics/lpips.py
===========================
LPIPS (Learned Perceptual Image Patch Similarity) evaluator.

Input: LevelBundle.image -- (H, W, 3) uint8 RGB image
Similarity: 1 - LPIPS_dist / max_dist in [0, 1] (1 means identical)
Dependencies: lpips and torch (pip install lpips)
"""
from __future__ import annotations

from typing import List

import numpy as np

from .base import BaseMetricEvaluator, LevelBundle


class LPIPSMetric(BaseMetricEvaluator):
    """
    Learned Perceptual Image Patch Similarity (LPIPS) evaluator.

    Normalize AlexNet/VGG perceptual distance and return it as similarity:
        similarity = 1 - dist / max_dist  ∈ [0, 1]

    Parameters
    ----------
    net : {"alex", "vgg", "squeeze"}
        LPIPS backbone. "alex" is fastest and recommended.
    """

    def __init__(self, net: str = "alex") -> None:
        import importlib
        # Load the external lpips package explicitly to avoid filename collisions
        _lpips_lib = importlib.import_module("lpips")
        self.net = net
        self._loss_fn = _lpips_lib.LPIPS(net=net, verbose=False)
        self._loss_fn.eval()

    # ── BaseMetricEvaluator implementation ───────────────────────────────────

    @property
    def name(self) -> str:
        return f"LPIPS[{self.net}]"

    def similarity_matrix(self, bundles: List[LevelBundle]) -> np.ndarray:
        """
        (N, N) similarity matrix: sim = 1 - dist / max_dist.
        The diagonal is 1.0.
        """
        dist_mat = self._distance_matrix(bundles)
        max_d    = dist_mat.max()
        sim      = 1.0 - dist_mat / max_d if max_d > 1e-8 else np.ones_like(dist_mat)
        np.fill_diagonal(sim, 1.0)
        return sim

    # ── Additional public API ─────────────────────────────────────────────────

    def distance_matrix(self, bundles: List[LevelBundle]) -> np.ndarray:
        """Return an (N, N) pairwise LPIPS distance matrix; lower is more similar."""
        return self._distance_matrix(bundles)

    # ── internal utility ─────────────────────────────────────────────────────────────

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

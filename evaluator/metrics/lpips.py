"""
evaluator/metrics/lpips.py
===========================
LPIPS (Learned Perceptual Image Patch Similarity) texttable.

text: LevelBundle.image — (H, W, 3) uint8 RGB image
text also : 1 - LPIPS_dist / max_dist  ∈ [0, 1]  (1 = text before  same)
 of text: lpips, torch  (pip install lpips)
"""
from __future__ import annotations

from typing import List

import numpy as np

from .base import BaseMetricEvaluator, LevelBundle


class LPIPSMetric(BaseMetricEvaluator):
    """
    Learned Perceptual Image Patch Similarity (LPIPS) texttable.

    AlexNet/VGG text distance  normalizetext text also  to  returntext:
        similarity = 1 - dist / max_dist  ∈ [0, 1]

    Parameters
    ----------
    net : {"alex", "vgg", "squeeze"}
        LPIPS text network.  "alex"    text text recommendedtext.
    """

    def __init__(self, net: str = "alex") -> None:
        import importlib
        # filetext text text: text lpips text  direct text load
        _lpips_lib = importlib.import_module("lpips")
        self.net = net
        self._loss_fn = _lpips_lib.LPIPS(net=net, verbose=False)
        self._loss_fn.eval()

    # ── BaseMetricEvaluator text ──────────────────────────────────────────────

    @property
    def name(self) -> str:
        return f"LPIPS[{self.net}]"

    def similarity_matrix(self, bundles: List[LevelBundle]) -> np.ndarray:
        """
        (N, N) text also  rowtext: sim = 1 − dist / max_dist.
        texteachtext = 1.0.
        """
        dist_mat = self._distance_matrix(bundles)
        max_d    = dist_mat.max()
        sim      = 1.0 - dist_mat / max_d if max_d > 1e-8 else np.ones_like(dist_mat)
        np.fill_diagonal(sim, 1.0)
        return sim

    # ── text  public API ─────────────────────────────────────────────────────────

    def distance_matrix(self, bundles: List[LevelBundle]) -> np.ndarray:
        """(N, N) pairwise LPIPS distance rowtext (text text text)."""
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


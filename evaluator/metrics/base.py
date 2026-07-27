"""
evaluator/metrics/base.py
==========================
Common interface for level-similarity metrics.

LevelBundle -- container for all representations of one level
MetricResult — evaluation results dataclass
BaseMetricEvaluator -- ABC inherited by every metric class
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Common data types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LevelBundle:
    """
    Container for all representations of a single game level.

    Each MetricEvaluator uses only the fields it needs.
      - TPKL  → array
      - SSIM / LPIPS → image
      - CLIPScore    → text + image

    Parameters
    ----------
    array : (H, W) int32 ndarray
        unified 5-category tile array (use_tile_mapping=True basis).
    image : (H, W, 3) uint8 ndarray
        Rendered RGB image.
    text : str
        Natural-language instruction.
    game : str
        Game tag (e.g. "dungeon", "doom").
    meta : dict
        Additional information such as reward_enum and conditions.
    """
    array: np.ndarray
    image: np.ndarray
    text:  str
    game:  str                         = ""
    meta:  Dict[str, Any]             = field(default_factory=dict)

    @classmethod
    def from_game_sample(
        cls,
        sample,                        # dataset.multigame.base.GameSample
        image_np: np.ndarray,
    ) -> "LevelBundle":
        """Create a LevelBundle from a GameSample and rendered image."""
        return cls(
            array = sample.array,
            image = image_np,
            text  = sample.instruction or "",
            game  = sample.game,
            meta  = dict(sample.meta),
        )


@dataclass
class MetricResult:
    """
    Evaluation result for a single metric.

    Attributes
    ----------
    name : str
        Metric name.
    same_mean : float
        Mean similarity of same-group pairs.
    diff_mean : float
        Mean similarity of different-group pairs.
    delta : float
        same_mean - diff_mean. Positive means same-group pairs are more similar.
    auc : float
        AUC-ROC ∈ [0, 1].  0.5 = random, 1.0 = perfect.
    same_scores : list[float]
        Individual same-group scores.
    diff_scores : list[float]
        Individual different-group scores.
    matrix : np.ndarray | None
        (N, N) similarity matrix, retained only when evaluate(keep_matrix=True).
    """
    name:        str
    same_mean:   float
    diff_mean:   float
    delta:       float
    auc:         float
    same_scores: List[float]
    diff_scores: List[float]
    matrix:      Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def is_supported(self) -> bool:
        """Whether the hypothesis is supported (delta > 0 and AUC > 0.5)."""
        return self.delta > 0 and self.auc > 0.5

    def __repr__(self) -> str:
        tag = "SUPPORTED" if self.is_supported else "NOT supported"
        return (
            f"MetricResult({self.name}: "
            f"delta={self.delta:+.4f}, auc={self.auc:.4f}) [{tag}]"
        )

    def summary_line(self) -> str:
        tag = "[OK]" if self.is_supported else "[NG]"
        return (
            f"{self.name:<16}  "
            f"same={self.same_mean:+.4f}  diff={self.diff_mean:+.4f}  "
            f"delta={self.delta:+.4f}  AUC={self.auc:.4f}  {tag}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Common statistical utilities
# ─────────────────────────────────────────────────────────────────────────────

def extract_pair_scores(
    matrix: np.ndarray,
    same_pairs: List[Tuple[int, int]],
    diff_pairs: List[Tuple[int, int]],
    symmetric: bool = True,
) -> Tuple[List[float], List[float]]:
    """
    Extract same-group and different-group score lists from a similarity matrix.

    symmetric=True includes both (i,j) and (j,i), for image-image comparisons.
    symmetric=False includes only (i,j), for text-image comparisons.
    """
    def _collect(pairs: List[Tuple[int, int]]) -> List[float]:
        out: List[float] = []
        for i, j in pairs:
            out.append(float(matrix[i, j]))
            if symmetric and i != j:
                out.append(float(matrix[j, i]))
        return out

    return _collect(same_pairs), _collect(diff_pairs)


def auc_roc_score(
    same_scores: List[float],
    diff_scores: List[float],
) -> float:
    """
    AUC-ROC based on the U statistic. 0.5 is random and 1.0 is perfect.
    """
    s = np.array(same_scores, dtype=np.float64)
    d = np.array(diff_scores, dtype=np.float64)
    if s.size == 0 or d.size == 0:
        return 0.5
    correct = float((s[:, None] > d[None, :]).sum())
    tie     = float((s[:, None] == d[None, :]).sum())
    return (correct + 0.5 * tie) / (s.size * d.size)


def roc_curve_points(
    same_scores: List[float],
    diff_scores: List[float],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return ROC curve points (fpr, tpr).
    """
    labels = np.concatenate([np.ones(len(same_scores)), np.zeros(len(diff_scores))])
    scores = np.concatenate([same_scores, diff_scores])
    thresholds = np.sort(np.unique(scores))[::-1]

    fprs, tprs = [0.0], [0.0]
    n_pos = labels.sum()
    n_neg = (1 - labels).sum()
    for thr in thresholds:
        pred = scores >= thr
        tp = ((pred == 1) & (labels == 1)).sum()
        fp = ((pred == 1) & (labels == 0)).sum()
        fprs.append(fp / (n_neg + 1e-10))
        tprs.append(tp / (n_pos + 1e-10))
    fprs.append(1.0)
    tprs.append(1.0)
    return np.array(fprs), np.array(tprs)


# ─────────────────────────────────────────────────────────────────────────────
# Abstract base class
# ─────────────────────────────────────────────────────────────────────────────

class BaseMetricEvaluator(ABC):
    """
    Common abstract base class for all level-similarity metrics.

    Subclasses must implement:
        name              : str property
        similarity_matrix : List[LevelBundle] → (N, N) ndarray

    Provided automatically:
        score_pair : similarity for one pair
        evaluate   : calculate all statistics and return MetricResult
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the metric's unique name."""
        ...

    @abstractmethod
    def similarity_matrix(self, bundles: List[LevelBundle]) -> np.ndarray:
        """
        (N, N) pairwise similarity matrix.
        - Higher values mean greater similarity.
        - The diagonal is 1.0 (self-similarity).
        """
        ...

    # ── Provided automatically to subclasses ─────────────────────────────────

    def score_pair(self, a: LevelBundle, b: LevelBundle) -> float:
        """Return the similarity score for one pair (a, b)."""
        return float(self.similarity_matrix([a, b])[0, 1])

    def evaluate(
        self,
        bundles: List[LevelBundle],
        same_pairs: List[Tuple[int, int]],
        diff_pairs: List[Tuple[int, int]],
        keep_matrix: bool = False,
        symmetric: bool = True,
    ) -> MetricResult:
        """
        all evaluation Usage.

        Parameters
        ----------
        bundles : list of N LevelBundles
        same_pairs : (i, j) indices from the same (game, reward_enum) group
        diff_pairs : (i, j) indices from different groups
        keep_matrix : store the matrix in MetricResult.matrix when True
        symmetric : passed to extract_pair_scores (image-image=True, text-image=False)

        Returns
        -------
        MetricResult
        """
        mat = self.similarity_matrix(bundles)
        same_sc, diff_sc = extract_pair_scores(mat, same_pairs, diff_pairs, symmetric=symmetric)

        same_mean = float(np.mean(same_sc)) if same_sc else 0.0
        diff_mean = float(np.mean(diff_sc)) if diff_sc else 0.0

        return MetricResult(
            name        = self.name,
            same_mean   = same_mean,
            diff_mean   = diff_mean,
            delta       = same_mean - diff_mean,
            auc         = auc_roc_score(same_sc, diff_sc),
            same_scores = same_sc,
            diff_scores = diff_sc,
            matrix      = mat if keep_matrix else None,
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"

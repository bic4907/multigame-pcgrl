"""
evaluator/metrics/base.py
==========================
level text also  texttable common interface.

LevelBundle  — text level of  text tabletext  text  text text
MetricResult — evaluation results dataclass
BaseMetricEvaluator — text texttable class  text ABC
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# common data text
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LevelBundle:
    """
    text game level of  text tabletext  text  text text.

    each MetricEvaluator  text in text text text text for text.
      - TPKL  → array
      - SSIM / LPIPS → image
      - CLIPScore    → text + image

    Parameters
    ----------
    array : (H, W) int32 ndarray
        unified 5-category tile array (use_tile_mapping=True basis).
    image : (H, W, 3) uint8 ndarray
        renderingtext RGB image.
    text : str
        text instruction.
    game : str
        game text (e.g. "dungeon", "doom").
    meta : dict
        reward_enum, conditions text text  info.
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
        """GameSample + renderingtext image → LevelBundle."""
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
    text texttable of  evaluation results.

    Attributes
    ----------
    name : str
        texttable name.
    same_mean : float
        same-group text mean text also .
    diff_mean : float
        diff-group text mean text also .
    delta : float
        same_mean − diff_mean.  text = same-group text text ( text text).
    auc : float
        AUC-ROC ∈ [0, 1].  0.5 = random, 1.0 = perfect.
    same_scores : list[float]
        same-group text text list.
    diff_scores : list[float]
        diff-group text text list.
    matrix : np.ndarray | None
        (N, N) text also  rowtext (keep_matrix=True  to  evaluate() call text in text preserve).
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
        """ text text text (Δ > 0 AND AUC > 0.5)."""
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
# common text utility
# ─────────────────────────────────────────────────────────────────────────────

def extract_pair_scores(
    matrix: np.ndarray,
    same_pairs: List[Tuple[int, int]],
    diff_pairs: List[Tuple[int, int]],
    symmetric: bool = True,
) -> Tuple[List[float], List[float]]:
    """
    text also  rowtext in  same / diff text text extract.

    symmetric=True  : (i,j)  and  (j,i)   text text (image-image text).
    symmetric=False : (i,j) text text (text-image text).
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
    AUC-ROC (U-text based).  0.5 = random, 1.0 = perfect.
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
    ROC text text (fpr, tpr) return.
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
# text based class
# ─────────────────────────────────────────────────────────────────────────────

class BaseMetricEvaluator(ABC):
    """
    text level text also  texttable of  common text based class.

    textclass text text:
        name              : str property
        similarity_matrix : List[LevelBundle] → (N, N) ndarray

    text  after  automatic text:
        score_pair : text text text also
        evaluate   : all text compute → MetricResult
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """texttable text name."""
        ...

    @abstractmethod
    def similarity_matrix(self, bundles: List[LevelBundle]) -> np.ndarray:
        """
        (N, N) pairwise text also  rowtext.
        - text  text text text text.
        - texteachtext = 1.0 (text text and  of  text also ).
        """
        ...

    # ── text  after  automatic text ─────────────────────────────────────────────────────

    def score_pair(self, a: LevelBundle, b: LevelBundle) -> float:
        """text text (a, b)  of  text also  text."""
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
        bundles : list of LevelBundle (Ntext)
        same_pairs : (i, j) — same (game, reward_enum) text text index
        diff_pairs : (i, j) — different text text index
        keep_matrix : True  text MetricResult.matrix  in  rowtext save
        symmetric : extract_pair_scores  in   before text (image-image=True, text-image=False)

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


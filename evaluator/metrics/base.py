"""
evaluator/metrics/base.py
==========================
레벨 유사도 지표 공통 인터페이스.

LevelBundle  — 단일 레벨의 모든 표현을 묶는 컨테이너
MetricResult — 평가 결과 dataclass
BaseMetricEvaluator — 모든 지표 클래스가 상속할 ABC
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# 공통 데이터 타입
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LevelBundle:
    """
    단일 게임 레벨의 모든 표현을 묶는 컨테이너.

    각 MetricEvaluator는 자신에게 필요한 필드만 사용한다.
      - TPKL  → array
      - SSIM / LPIPS → image
      - CLIPScore    → text + image

    Parameters
    ----------
    array : (H, W) int32 ndarray
        unified 5-category 타일 배열 (use_tile_mapping=True 기준).
    image : (H, W, 3) uint8 ndarray
        렌더링된 RGB 이미지.
    text : str
        자연어 instruction.
    game : str
        게임 태그 (e.g. "dungeon", "doom").
    meta : dict
        reward_enum, conditions 등 부가 정보.
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
        """GameSample + 렌더링된 이미지 → LevelBundle."""
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
    단일 지표의 평가 결과.

    Attributes
    ----------
    name : str
        지표 이름.
    same_mean : float
        same-group 쌍 평균 유사도.
    diff_mean : float
        diff-group 쌍 평균 유사도.
    delta : float
        same_mean − diff_mean.  양수 = same-group 더 유사 (가설 지지).
    auc : float
        AUC-ROC ∈ [0, 1].  0.5 = random, 1.0 = perfect.
    same_scores : list[float]
        same-group 개별 점수 목록.
    diff_scores : list[float]
        diff-group 개별 점수 목록.
    matrix : np.ndarray | None
        (N, N) 유사도 행렬 (keep_matrix=True 로 evaluate() 호출 시에만 보존).
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
        """가설 지지 여부 (Δ > 0 AND AUC > 0.5)."""
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
# 공통 통계 유틸
# ─────────────────────────────────────────────────────────────────────────────

def extract_pair_scores(
    matrix: np.ndarray,
    same_pairs: List[Tuple[int, int]],
    diff_pairs: List[Tuple[int, int]],
    symmetric: bool = True,
) -> Tuple[List[float], List[float]]:
    """
    유사도 행렬에서 same / diff 점수 리스트 추출.

    symmetric=True  : (i,j) 와 (j,i) 를 모두 포함 (image-image 비교).
    symmetric=False : (i,j) 만 포함 (text-image 비교).
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
    AUC-ROC (U-통계량 기반).  0.5 = random, 1.0 = perfect.
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
    ROC 곡선 점 (fpr, tpr) 반환.
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
# 추상 기반 클래스
# ─────────────────────────────────────────────────────────────────────────────

class BaseMetricEvaluator(ABC):
    """
    모든 레벨 유사도 지표의 공통 추상 기반 클래스.

    서브클래스 구현 필수:
        name              : str property
        similarity_matrix : List[LevelBundle] → (N, N) ndarray

    상속 후 자동 제공:
        score_pair : 단일 쌍 유사도
        evaluate   : 전체 통계 계산 → MetricResult
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """지표 고유 이름."""
        ...

    @abstractmethod
    def similarity_matrix(self, bundles: List[LevelBundle]) -> np.ndarray:
        """
        (N, N) pairwise 유사도 행렬.
        - 값이 높을수록 더 유사.
        - 대각선 = 1.0 (자기 자신과의 유사도).
        """
        ...

    # ── 상속 후 자동 제공 ─────────────────────────────────────────────────────

    def score_pair(self, a: LevelBundle, b: LevelBundle) -> float:
        """단일 쌍 (a, b) 의 유사도 점수."""
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
        전체 평가 실행.

        Parameters
        ----------
        bundles : list of LevelBundle (N개)
        same_pairs : (i, j) — 같은 (game, reward_enum) 그룹 쌍 인덱스
        diff_pairs : (i, j) — 다른 그룹 쌍 인덱스
        keep_matrix : True 이면 MetricResult.matrix 에 행렬 저장
        symmetric : extract_pair_scores 에 전달 (image-image=True, text-image=False)

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


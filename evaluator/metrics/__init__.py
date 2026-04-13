"""
evaluator/metrics/__init__.py
==============================
레벨 유사도 지표 패키지.

사용 예:
    from evaluator.metrics import (
        LevelBundle, MetricResult,
        CLIPScoreMetric, TPKLMetric, SSIMMetric, LPIPSMetric,
    )

    bundle = LevelBundle.from_game_sample(sample, image_np)
    metric = TPKLMetric()
    result = metric.evaluate(bundles, same_pairs, diff_pairs)
    print(result.summary_line())
"""
from .base import (
    LevelBundle,
    MetricResult,
    BaseMetricEvaluator,
    extract_pair_scores,
    auc_roc_score,
    roc_curve_points,
)
from .clip_score      import CLIPScoreMetric
from .tpkl            import TPKLMetric
from .ssim            import SSIMMetric
from .lpips           import LPIPSMetric
from .shannon_entropy import ShannonEntropyMetric

__all__ = [
    # 데이터 타입
    "LevelBundle",
    "MetricResult",
    # 추상 기반 클래스
    "BaseMetricEvaluator",
    # 지표 클래스
    "CLIPScoreMetric",
    "TPKLMetric",
    "SSIMMetric",
    "LPIPSMetric",
    "ShannonEntropyMetric",
    # 통계 유틸
    "extract_pair_scores",
    "auc_roc_score",
    "roc_curve_points",
]


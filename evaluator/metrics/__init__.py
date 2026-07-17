"""
evaluator/metrics/__init__.py
==============================
level text also  texttable text.

Example:
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
    # data text
    "LevelBundle",
    "MetricResult",
    # text based class
    "BaseMetricEvaluator",
    # texttable class
    "CLIPScoreMetric",
    "TPKLMetric",
    "SSIMMetric",
    "LPIPSMetric",
    "ShannonEntropyMetric",
    # text utility
    "extract_pair_scores",
    "auc_roc_score",
    "roc_curve_points",
]


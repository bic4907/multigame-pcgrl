"""
metrics.py
==========
Post-eval metric orchestrator.
Delegates to individual wrappers in instruct_rl.eval.wrappers:
  - DiversityWrapper   — Hamming diversity across seeds
  - HumanLikenessWrapper — ViT-based human-likeness score
  - TPKLWrapper        — Tile-Pattern KL divergence
"""
import logging

import pandas as pd

from instruct_rl.eval.wrappers import DiversityWrapper, ViTScoreWrapper, TPKLWrapper

logger = logging.getLogger(__name__)


def run_post_eval(
    config,
    instruct_df: pd.DataFrame,
    df_output: pd.DataFrame,
    eval_rendered: list,
    n_rows: int,
    n_eps: int,
    gt_levels=None,
) -> pd.DataFrame:
    """Run all enabled post-eval metrics and attach results to df_output.

    Args:
        config        : EvalConfig (or subclass).
        instruct_df   : Original instruct DataFrame (Diversity / ViT wrapper 용).
        df_output     : Result DataFrame (already contains loss etc.).
        eval_rendered : List of raw_rendered arrays per batch (for vit_score).
        n_rows        : Total number of evaluation rows.
        n_eps         : Number of episodes (seeds).
        gt_levels     : (M, H, W) int — eval_utils에서 samples 기반으로 직접 전달.
                        TPKL 계산에 사용. None 이면 AssertionError.

    Returns:
        df_output with metric columns appended.
    """
    kwargs = dict(
        instruct_df=instruct_df,
        eval_rendered=eval_rendered,
        n_rows=n_rows,
        n_eps=n_eps,
    )

    if config.diversity:
        df_output = DiversityWrapper(config).run(df_output, **kwargs)

    if config.vit_score:
        df_output = ViTScoreWrapper(config).run(df_output, **kwargs)

    if config.tpkldiv:
        assert gt_levels is not None, (
            "[run_post_eval] gt_levels must be provided for TPKL. "
            "Pass gt_levels from eval_utils via make_eval."
        )
        df_output = TPKLWrapper(config).run(df_output, gt_levels=gt_levels, **kwargs)

    return df_output


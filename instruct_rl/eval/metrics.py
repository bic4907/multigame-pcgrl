"""
metrics.py
==========
eval 후처리 메트릭 계산.
  - diversity    : Hamming distance
  - human_likeness: ViT 기반
  - tpkldiv      : TP-KL divergence
"""
import logging

import numpy as np
import pandas as pd
import wandb
from tqdm import tqdm

from instruct_rl.evaluation.hamming import compute_hamming_distance
from instruct_rl.evaluation.metrics.tpkldiv import TPKLEvaluator
from instruct_rl.evaluation.metrics.vit import ViTEvaluator

logger = logging.getLogger(__name__)


def run_post_eval(
    config,
    instruct_df: pd.DataFrame,
    df_output: pd.DataFrame,
    eval_rendered: list,
    n_rows: int,
    n_eps: int,
) -> pd.DataFrame:
    """diversity / human_likeness / tpkldiv 를 계산하여 df_output 에 컬럼으로 추가.

    Args:
        config        : EvalConfig (또는 하위 클래스).
        instruct_df   : 원본 instruct CSV DataFrame.
        df_output     : 결과 DataFrame (loss 등 이미 포함).
        eval_rendered : 배치별 raw_rendered 리스트 (human_likeness 용).
        n_rows        : 총 평가 row 수.
        n_eps         : 에피소드 수 (seed 수).

    Returns:
        메트릭 컬럼이 추가된 df_output.
    """

    # ── Diversity ─────────────────────────────────────────────────────────────
    if config.diversity:
        scores = []
        for row_i, _ in tqdm(instruct_df.iterrows(), desc="Computing Diversity"):
            states = [
                np.load(f"{config.eval_dir}/reward_{row_i}/seed_{seed_i}/state_0.npy")
                for seed_i in range(1, n_eps + 1)
            ]
            scores.append(compute_hamming_distance(np.array(states)))

        diversity_df = instruct_df.copy()
        diversity_df = diversity_df.loc[:, ~diversity_df.columns.str.startswith('embed')]
        diversity_df['diversity'] = scores

        if wandb.run:
            wandb.log({'diversity': wandb.Table(dataframe=diversity_df)})

    # ── Human-likeness ────────────────────────────────────────────────────────
    if config.human_likeness:
        eval_rendered_arr = np.concatenate(eval_rendered, axis=0)[:n_rows]
        evaluator = ViTEvaluator(normalized_vector=config.vit_normalize)

        index_ids = instruct_df.index.to_numpy()
        task_ids = np.repeat((index_ids // 4).astype(int), n_eps)
        df_output['human_likeness'] = evaluator.run(eval_rendered_arr, task_ids)

    # ── TP-KL Divergence ──────────────────────────────────────────────────────
    if config.tpkldiv:
        states = [
            np.load(f"{config.eval_dir}/reward_{row_i}/seed_{seed_i}/state_0.npy")
            for row_i, _ in tqdm(instruct_df.iterrows(), desc="Computing TPKLDiv")
            for seed_i in range(1, n_eps + 1)
        ]
        states = np.array(states)

        index_ids = instruct_df.index.to_numpy()
        task_ids = np.repeat((index_ids // 4).astype(int), n_eps)

        evaluator = TPKLEvaluator()
        df_output['tpkldiv'] = np.array(
            evaluator.run(states, task_ids, show_progress=True)
        ).reshape(-1)

    return df_output


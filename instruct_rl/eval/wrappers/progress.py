"""
progress.py
===========
Progress 측정 유틸리티.

progress = 1 - |condition - feat_final| / (|condition - feat_s0| + ε)
  - condition : 목표값 (cont_value)
  - feat_final: 최종 상태에서 측정된 feature 값
  - feat_s0   : 초기 상태(s0)에서 측정된 feature 값
  - 값은 [0, 100] 으로 클리핑됨.
  - condition == -1(null 센티널)인 행은 NaN 반환.
"""

import numpy as np
import pandas as pd

EPS = 1e-7


def calculate_progress(condition: float, feat: float, feat_s0: float) -> float:
    """스칼라 단위 progress 계산 (0~100, NaN 반환 가능)."""
    if np.isnan(condition) or condition == -1:
        return float("nan")
    raw = 1.0 - abs(condition - feat) / (abs(condition - feat_s0) + EPS)
    return float(np.clip(raw * 100.0, 0.0, 100.0))


class ProgressWrapper:
    """df_ctrl_sim DataFrame에 progress_* 컬럼을 추가한다.

    Parameters
    ----------
    n_cond : int
        condition_* / feat_* 컬럼의 개수.
    """

    def __init__(self, n_cond: int):
        self.n_cond = n_cond

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """reward_enum에 해당하는 인덱스의 condition/feat/feat_s0를 읽어
        단일 'progress' 컬럼을 추가한 복사본을 반환."""
        df = df.copy()

        def _row_progress(row):
            i = row.get("reward_enum", float("nan"))
            if pd.isna(i):
                return float("nan")
            i = int(i)
            cond = row.get(f"condition_{i}", float("nan"))
            feat = row.get(f"feat_{i}", float("nan"))
            feat_s0 = row.get(f"feat_{i}_s0", float("nan"))
            if pd.isna(feat) or pd.isna(feat_s0):
                return float("nan")
            return calculate_progress(cond, feat, feat_s0)

        df["progress"] = df.apply(_row_progress, axis=1)
        return df


"""
instruct_rl/eval/utils.py
=========================
평가 공통 유틸리티.
"""

import numpy as np
import pandas as pd


def iqr_mean(x: pd.Series) -> float:
    """IQR 기반 robust mean.

    - NaN 제외 후 계산.
    - 샘플이 4개 미만이면 IQR 필터링 없이 단순 평균 반환
      (샘플이 너무 적으면 IQR=0이 되어 전부 이상치로 판정될 수 있음).
    - 유효값이 없으면 NaN 반환.
    """
    x = x.dropna()
    if x.empty:
        return float("nan")
    if len(x) < 4:
        return float(x.mean())
    q1, q3 = x.quantile(0.25), x.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        # IQR=0 → 중앙값과 다른 값만 이상치로 처리
        median = x.median()
        filtered = x[x == median]
    else:
        filtered = x[(x >= q1 - 1.5 * iqr) & (x <= q3 + 1.5 * iqr)]
    return float(filtered.mean()) if not filtered.empty else float(x.mean())


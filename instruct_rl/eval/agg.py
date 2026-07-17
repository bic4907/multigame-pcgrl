"""
instruct_rl/eval/utils.py
=========================
evaluation common utility.
"""

import numpy as np
import pandas as pd


def iqr_mean(x: pd.Series) -> float:
    """IQR based robust mean.

    - NaN text  after  compute.
    - sample  4text less than text IQR filtering text  text mean return
      (sample  text text IQR=0  text  before text or moretext to  text text text).
    - validtext  if missing NaN return.
    """
    x = x.dropna()
    if x.empty:
        return float("nan")
    if len(x) < 4:
        return float(x.mean())
    q1, q3 = x.quantile(0.25), x.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        # IQR=0 → centertext and  different text or moretext to  process
        median = x.median()
        filtered = x[x == median]
    else:
        filtered = x[(x >= q1 - 1.5 * iqr) & (x <= q3 + 1.5 * iqr)]
    return float(filtered.mean()) if not filtered.empty else float(x.mean())


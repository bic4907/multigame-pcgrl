"""
instruct_rl/eval/utils.py
=========================
evaluation common utility.
"""

import numpy as np
import pandas as pd


def iqr_mean(x: pd.Series) -> float:
    """IQR based robust mean.

    - Exclude NaN values before calculation.
    - With fewer than four samples, return the plain mean without IQR filtering
      (too few samples can produce IQR=0 and classify every value as an outlier).
    - Return NaN when there are no valid values.
    """
    x = x.dropna()
    if x.empty:
        return float("nan")
    if len(x) < 4:
        return float(x.mean())
    q1, q3 = x.quantile(0.25), x.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        # IQR=0: treat only values different from the median as outliers
        median = x.median()
        filtered = x[x == median]
    else:
        filtered = x[(x >= q1 - 1.5 * iqr) & (x <= q3 + 1.5 * iqr)]
    return float(filtered.mean()) if not filtered.empty else float(x.mean())

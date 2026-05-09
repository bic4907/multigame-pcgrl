"""Statistical helper functions shared across results/ scripts."""
from __future__ import annotations

import math


def safe_std(values: list[float]) -> float:
    """Population standard deviation. Returns 0.0 for single-element lists."""
    if len(values) <= 1:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return math.sqrt(variance)


def to_float(text: str | None) -> float | None:
    """Parse a string to float. Returns None on failure or empty string."""
    if text is None:
        return None
    s = text.strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def iqr_mean(x: "pd.Series") -> float:  # type: ignore[name-defined]
    """IQR-filtered mean for a pandas Series. Falls back to plain mean for small samples."""
    x = x.dropna()
    if x.empty:
        return float("nan")
    if len(x) < 4:
        return float(x.mean())
    q1, q3 = x.quantile(0.25), x.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        median = x.median()
        filtered = x[x == median]
    else:
        filtered = x[(x >= q1 - 1.5 * iqr) & (x <= q3 + 1.5 * iqr)]
    return float(filtered.mean()) if not filtered.empty else float(x.mean())


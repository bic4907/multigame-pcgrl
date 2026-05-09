"""Min-max normalization helpers for benchmark reporting."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

# NormScale[game][reward_enum][metric] = {"min": float, "max": float}
NormScale = dict


def compute_normalization_scale(
    plot_rows: list[dict],
    metric_order: list[str],
) -> NormScale:
    """(game, reward_enum, metric) 별로 전체 프로젝트에 걸친 min/max를 계산한다."""
    buckets: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in plot_rows:
        game = row["game"]
        re = str(row["reward_enum"])
        for metric in metric_order:
            v = row["metrics"].get(metric)
            if v is not None:
                buckets[(game, re, metric)].append(v)

    scale: NormScale = {}
    for (game, re, metric), vals in buckets.items():
        mn, mx = min(vals), max(vals)
        scale.setdefault(game, {}).setdefault(re, {})[metric] = {"min": mn, "max": mx}
    return scale


def apply_normalization(
    plot_rows: list[dict],
    scale: NormScale,
    metric_order: list[str],
) -> list[dict]:
    """각 row의 metric 값을 (game, reward_enum) 기준 min-max 정규화한다."""
    norm_rows: list[dict] = []
    for row in plot_rows:
        game = row["game"]
        re = str(row["reward_enum"])
        new_metrics: dict[str, float] = {}
        for metric in metric_order:
            v = row["metrics"].get(metric)
            if v is None:
                continue
            bounds = scale.get(game, {}).get(re, {}).get(metric)
            if bounds is None:
                new_metrics[metric] = v
                continue
            mn, mx = bounds["min"], bounds["max"]
            new_metrics[metric] = (v - mn) / (mx - mn) if mx > mn else 0.0
        norm_rows.append({**row, "metrics": new_metrics})
    return norm_rows


def save_normalization_scale(scale: NormScale, path: Path) -> None:
    """정규화 스케일을 JSON 파일로 저장한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(scale, f, indent=2, ensure_ascii=False)


def load_normalization_scale(path: Path) -> NormScale:
    """JSON 파일에서 정규화 스케일을 로드한다."""
    with path.open(encoding="utf-8") as f:
        return json.load(f)

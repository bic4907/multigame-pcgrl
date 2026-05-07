"""CSV / file I/O helpers shared across results/ scripts."""
from __future__ import annotations

import csv
from pathlib import Path


# ---------------------------------------------------------------------------
# reward_enum normalization
# ---------------------------------------------------------------------------

def normalize_reward_enum(value: str) -> str:
    """'0.0' → '0' 등 float 형태의 reward_enum 문자열을 int 문자열로 정규화."""
    try:
        return str(int(float(value)))
    except (ValueError, TypeError):
        return value


# ---------------------------------------------------------------------------
# Run / eval name token parsing
# ---------------------------------------------------------------------------

def parse_run_tokens(run_name: str) -> dict[str, str]:
    """'cpcgrl_game-all_re-0_exp-def_s-0' → {'game': 'all', 're': '0', ...}"""
    tokens: dict[str, str] = {}
    for part in run_name.split("_"):
        if "-" not in part:
            continue
        key, value = part.split("-", 1)
        tokens[key] = value
    return tokens


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def iter_summary_paths(input_root: Path) -> list[Path]:
    return sorted(p for p in input_root.rglob("summary.csv") if p.is_file())


def iter_results_paths(input_root: Path) -> list[Path]:
    return sorted(p for p in input_root.rglob("results.csv") if p.is_file())


# ---------------------------------------------------------------------------
# CSV reading
# ---------------------------------------------------------------------------

def read_summary(summary_path: Path) -> dict[str, float]:
    """summary.csv (metric, mean columns) → {metric: mean} dict."""
    metrics: dict[str, float] = {}
    with summary_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            metric = (row.get("metric") or "").strip()
            mean_text = (row.get("mean") or "").strip()
            if not metric or not mean_text:
                continue
            try:
                metrics[metric] = float(mean_text)
            except ValueError:
                continue
    return metrics


# ---------------------------------------------------------------------------
# Sorting helpers
# ---------------------------------------------------------------------------

def sort_key_reward_enum(value: str) -> tuple[int, float | str]:
    """Sort key: numeric reward_enums first (ascending), then string values."""
    if value == "all":
        return (0, -1.0)
    try:
        return (0, float(value))
    except ValueError:
        return (1, value)


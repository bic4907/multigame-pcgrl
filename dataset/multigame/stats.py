"""
dataset/multigame/stats.py
===========================
MultiGameDataset statistics utilities.

Open the dataset and compute per-game level statistics without training.

Usage
-----
    from dataset.multigame.stats import compute_dataset_stats, print_dataset_stats

    # Compute statistics from a MultiGameDataset instance
    from dataset.multigame import MultiGameDataset
    ds = MultiGameDataset()
    stats = compute_dataset_stats(ds)
    print_dataset_stats(stats)

    # CLI Usage
    python -m dataset.multigame.stats
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List

import numpy as np

from .base import GameSample
from .tile_utils import (
    UNIFIED_CATEGORIES,
    to_unified,
    category_distribution,
)


def compute_sample_stats(sample: GameSample) -> Dict[str, Any]:
    """Return basic statistics for a single GameSample.

    Returns
    -------
    {
        "height": int,
        "width": int,
        "n_tiles": int,
        "unique_raw_tiles": int,             # Number of distinct raw tile IDs
        "category_counts": Dict[str, int],   # Tile count per unified category
        "has_instruction": bool,
        "instruction_len": int | None,        # Word count (None if no instruction)
        "has_reward_annotation": bool,
    }
    """
    arr = sample.array
    h, w = arr.shape[:2]
    n_tiles = h * w

    # Number of unique raw tile values
    unique_raw = len(np.unique(arr))

    # Unified-category distribution (counts)
    unified = to_unified(arr, sample.game, warn_unmapped=False)
    cat_counts = category_distribution(unified, normalize=False)

    # instruction
    has_inst = sample.instruction is not None
    inst_len = len(sample.instruction.split()) if has_inst else None

    # reward annotation
    has_reward = "reward_enum" in sample.meta

    return {
        "height": h,
        "width": w,
        "n_tiles": n_tiles,
        "unique_raw_tiles": unique_raw,
        "category_counts": cat_counts,
        "has_instruction": has_inst,
        "instruction_len": inst_len,
        "has_reward_annotation": has_reward,
    }


def compute_game_stats(
    samples: List[GameSample],
    game: str,
) -> Dict[str, Any]:
    """Compute aggregate statistics from samples belonging to one game.

    Returns
    -------
    {
        "game": str,
        "num_samples": int,
        "shape": {
            "heights": {"min", "max", "mean"},
            "widths":  {"min", "max", "mean"},
            "unique_shapes": List[Tuple[int,int]],
        },
        "raw_tile_vocab": {
            "min_unique": int,
            "max_unique": int,
            "mean_unique": float,
        },
        "category_distribution": {
            <category_name>: {"mean", "std", "min", "max", "mean_ratio"}
        },
        "instruction": {
            "with_instruction": int,
            "without_instruction": int,
            "ratio": float,
            "word_count": {"min", "max", "mean"} | None,
        },
        "reward_annotation": {
            "annotated": int,
            "ratio": float,
        },
    }
    """
    if not samples:
        return {"game": game, "num_samples": 0}

    per = [compute_sample_stats(s) for s in samples]
    n = len(per)

    # ── shape ──
    heights = np.array([p["height"] for p in per])
    widths = np.array([p["width"] for p in per])
    unique_shapes = sorted(set(zip(heights.tolist(), widths.tolist())))

    # ── raw tile vocab ──
    uniq_tiles = np.array([p["unique_raw_tiles"] for p in per])

    # ── category distribution ──
    cat_names = list(UNIFIED_CATEGORIES.values())
    cat_arrays: Dict[str, np.ndarray] = {}
    n_tiles_arr = np.array([p["n_tiles"] for p in per], dtype=float)
    for cn in cat_names:
        vals = np.array([p["category_counts"].get(cn, 0) for p in per], dtype=float)
        cat_arrays[cn] = vals

    cat_dist: Dict[str, Dict[str, float]] = {}
    for cn in cat_names:
        vals = cat_arrays[cn]
        ratios = vals / np.maximum(n_tiles_arr, 1)
        cat_dist[cn] = {
            "mean_count": float(np.mean(vals)),
            "std_count": float(np.std(vals)),
            "min_count": int(np.min(vals)),
            "max_count": int(np.max(vals)),
            "mean_ratio": float(np.mean(ratios)),
            "std_ratio": float(np.std(ratios)),
        }

    # ── instruction ──
    with_inst = sum(1 for p in per if p["has_instruction"])
    without_inst = n - with_inst
    inst_word_counts = [p["instruction_len"] for p in per if p["instruction_len"] is not None]
    if inst_word_counts:
        wc = np.array(inst_word_counts)
        word_stats = {
            "min": int(np.min(wc)),
            "max": int(np.max(wc)),
            "mean": float(np.mean(wc)),
        }
    else:
        word_stats = None

    # ── reward annotation ──
    annotated = sum(1 for p in per if p["has_reward_annotation"])

    return {
        "game": game,
        "num_samples": n,
        "shape": {
            "heights": {
                "min": int(np.min(heights)),
                "max": int(np.max(heights)),
                "mean": float(np.mean(heights)),
            },
            "widths": {
                "min": int(np.min(widths)),
                "max": int(np.max(widths)),
                "mean": float(np.mean(widths)),
            },
            "unique_shapes": unique_shapes,
        },
        "raw_tile_vocab": {
            "min_unique": int(np.min(uniq_tiles)),
            "max_unique": int(np.max(uniq_tiles)),
            "mean_unique": float(np.mean(uniq_tiles)),
        },
        "category_distribution": cat_dist,
        "instruction": {
            "with_instruction": with_inst,
            "without_instruction": without_inst,
            "ratio": with_inst / n if n > 0 else 0.0,
            "word_count": word_stats,
        },
        "reward_annotation": {
            "annotated": annotated,
            "ratio": annotated / n if n > 0 else 0.0,
        },
    }


def compute_dataset_stats(dataset) -> Dict[str, Any]:
    """Compute overall and per-game statistics from a MultiGameDataset instance.

    Parameters
    ----------
    dataset : MultiGameDataset (or List[GameSample])

    Returns
    -------
    {
        "total_samples": int,
        "games": List[str],
        "per_game": { game_name: <compute_game_stats result> },
        "overall_category_distribution": { <category>: mean_ratio },
    }
    """
    # Use dataset directly when it is already a list
    if isinstance(dataset, list):
        samples = dataset
    else:
        # MultiGameDataset: access _samples directly
        samples = list(dataset._samples)

    # Group by game
    game_groups: Dict[str, List[GameSample]] = defaultdict(list)
    for s in samples:
        game_groups[s.game].append(s)

    per_game: Dict[str, Dict[str, Any]] = {}
    for game in sorted(game_groups.keys()):
        per_game[game] = compute_game_stats(game_groups[game], game)

    # Overall mean category ratios
    cat_names = list(UNIFIED_CATEGORIES.values())
    overall_cat: Dict[str, float] = {}
    total = len(samples)
    if total > 0:
        for cn in cat_names:
            all_ratios = []
            for s in samples:
                unified = to_unified(s.array, s.game, warn_unmapped=False)
                dist = category_distribution(unified, normalize=True)
                all_ratios.append(dist.get(cn, 0.0))
            overall_cat[cn] = float(np.mean(all_ratios))
    else:
        overall_cat = {cn: 0.0 for cn in cat_names}

    return {
        "total_samples": total,
        "games": sorted(game_groups.keys()),
        "per_game": per_game,
        "overall_category_distribution": overall_cat,
    }


def print_dataset_stats(stats: Dict[str, Any]) -> None:
    """Print compute_dataset_stats results in a human-readable form."""
    print("=" * 72)
    print(f"  MultiGameDataset Statistics  (total: {stats['total_samples']} samples)")
    print(f"  Games: {', '.join(stats['games'])}")
    print("=" * 72)

    for game in stats["games"]:
        gs = stats["per_game"][game]
        n = gs["num_samples"]
        print(f"\n{'─' * 60}")
        print(f"  [{game.upper()}]  {n} samples")
        print(f"{'─' * 60}")

        # shape
        sh = gs["shape"]
        shapes_str = ", ".join(f"{h}×{w}" for h, w in sh["unique_shapes"])
        print(f"  Shape       : {shapes_str}")
        print(f"                H=[{sh['heights']['min']}..{sh['heights']['max']}] "
              f"(mean {sh['heights']['mean']:.1f})  "
              f"W=[{sh['widths']['min']}..{sh['widths']['max']}] "
              f"(mean {sh['widths']['mean']:.1f})")

        # raw tile vocab
        rv = gs["raw_tile_vocab"]
        print(f"  Raw tiles   : unique=[{rv['min_unique']}..{rv['max_unique']}] "
              f"(mean {rv['mean_unique']:.1f})")

        # category distribution
        print(f"  Category distribution (mean ratio ± std):")
        for cn, cd in gs["category_distribution"].items():
            bar_len = int(cd["mean_ratio"] * 30)
            bar = "█" * bar_len + "░" * (30 - bar_len)
            print(f"    {cn:12s} {bar} {cd['mean_ratio']:5.1%} ± {cd['std_ratio']:5.1%}"
                  f"  (count {cd['mean_count']:6.1f} ± {cd['std_count']:5.1f})")

        # instruction
        inst = gs["instruction"]
        print(f"  Instruction : {inst['with_instruction']}/{n} ({inst['ratio']:.0%})")
        if inst["word_count"] is not None:
            wc = inst["word_count"]
            print(f"                words=[{wc['min']}..{wc['max']}] (mean {wc['mean']:.1f})")

        # reward annotation
        ra = gs["reward_annotation"]
        print(f"  Reward ann. : {ra['annotated']}/{n} ({ra['ratio']:.0%})")

    # overall
    print(f"\n{'=' * 72}")
    print("  Overall category distribution (mean ratio across all samples):")
    for cn, ratio in stats["overall_category_distribution"].items():
        bar_len = int(ratio * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        print(f"    {cn:12s} {bar} {ratio:5.1%}")
    print("=" * 72)


# ── CLI entrypoint ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    import json as _json
    import logging

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Compute MultiGameDataset statistics")
    parser.add_argument("--json", action="store_true", help="Output in JSON format")
    parser.add_argument("--include-dungeon", action="store_true", default=True)
    parser.add_argument("--include-pokemon", action="store_true", default=True)
    parser.add_argument("--include-sokoban", action="store_true", default=True)
    parser.add_argument("--include-doom", action="store_true", default=True)
    parser.add_argument("--include-zelda", action="store_true", default=True)
    parser.add_argument("--no-dungeon", action="store_true")
    parser.add_argument("--no-pokemon", action="store_true")
    parser.add_argument("--no-sokoban", action="store_true")
    parser.add_argument("--no-doom", action="store_true")
    parser.add_argument("--no-zelda", action="store_true")
    args = parser.parse_args()

    from .dataset import MultiGameDataset

    ds = MultiGameDataset(
        include_dungeon=not args.no_dungeon,
        include_pokemon=not args.no_pokemon,
        include_sokoban=not args.no_sokoban,
        include_doom=not args.no_doom,
        include_zelda=not args.no_zelda,
        use_tile_mapping=False,  # Compute statistics from raw data
    )
    print(f"Loaded: {ds}")

    stats = compute_dataset_stats(ds)

    if args.json:
        # Convert NumPy types for serialization
        def _convert(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, tuple):
                return list(obj)
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

        print(_json.dumps(stats, indent=2, default=_convert, ensure_ascii=False))
    else:
        print_dataset_stats(stats)

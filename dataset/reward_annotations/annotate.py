#!/usr/bin/env python3
"""
dataset/reward_annotations/annotate.py
=======================================
Read Doom/Zelda/Sokoban/Pokemon/Dungeon maps from cache and
compute per-sample reward annotations and save them to {key}.ann.json.

Use tile_mapping.json unified categories in the same way as the data loader:
  raw map → to_unified (0-4) → +1 shift → MultigameTiles (1-5)

Reward enum definitions:
  0 (RG)  region              - number of connected traversable regions -> condition_0
  1 (PL)  path_length         - longest path length                       -> condition_1
  2 (IC)  interactable_count  - Interactive tile count                    -> condition_2
  3 (HC)  hazard_count        - Hazard tile count                         -> condition_3
  4 (CC)  collectable_count   - Collectable tile count                    -> condition_4

passible (region/path_length basis):
  unified EMPTY(1) + HAZARD(4) + COLLECTABLE(5), shared by all games

Usage:
  python dataset/reward_annotations/annotate.py
  python dataset/reward_annotations/annotate.py --games doom zelda dungeon
  python dataset/reward_annotations/annotate.py --cache-dir dataset/multigame/cache/artifacts
  python dataset/reward_annotations/annotate.py --force  # existing ann.json overwrite
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add the project root, two levels above dataset/reward_annotations, to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import jax.numpy as jnp

from evaluator.measures import (
    get_region as eval_get_region,
    get_path_length as eval_get_path_length,
    get_interactive_count,
    get_hazard_count,
    get_collectable_count,
)
from dataset.multigame.tile_utils import to_unified
from dataset.multigame.cache_utils import (
    save_game_annotations_to_cache,
    load_game_annotations_from_cache,
)
from envs.probs.multigame import MultigameTiles

# Unified passable tiles in MultigameTiles space: EMPTY(1) + HAZARD(4) + COLLECTABLE(5)
_UNIFIED_PASSIBLE = jnp.array(
    [int(MultigameTiles.EMPTY), int(MultigameTiles.HAZARD), int(MultigameTiles.COLLECTABLE)],
    dtype=jnp.int32,
)

# ── path ─────────────────────────────────────────────────────────────────────────
_HERE = Path(__file__).parent                              # dataset/reward_annotations/
_CACHE_DIR = _HERE.parent / "multigame" / "cache" / "artifacts"

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Per-game sub_condition labels based on tile_mapping.json categories ───────
_GAME_CONFIG = {
    "doom": {
        "sub_cond_interactable": "spawn+door+danger",
        "sub_cond_hazard":       "enemy",
        "sub_cond_collectable":  "item",
    },
    "zelda": {
        "sub_cond_interactable": "door+block+start",
        "sub_cond_hazard":       "mob",
        "sub_cond_collectable":  "object",
    },
    "sokoban": {
        "sub_cond_interactable": "box",
        "sub_cond_hazard":       "",
        "sub_cond_collectable":  "",
    },
    "pokemon": {
        "sub_cond_interactable": "spawn+water",
        "sub_cond_hazard":       "enemy",
        "sub_cond_collectable":  "object",
    },
    "dungeon": {
        "sub_cond_interactable": "",
        "sub_cond_hazard":       "enemy",
        "sub_cond_collectable":  "treasure",
    },
}

# Per-game key prefixes (two-letter abbreviations)
GAME_PREFIX: Dict[str, str] = {
    "doom":    "dm",
    "zelda":   "zl",
    "sokoban": "sk",
    "pokemon": "pk",
    "dungeon": "dg",
}


# ── cache load (standalone mode for ) ────────────────────────────────────────────────

def _load_cache_dir(cache_dir: Path) -> Optional[List[dict]]:
    """Load NPZ and JSON caches from a cache directory."""
    json_files = sorted(
        f for f in cache_dir.glob("*.json")
        if not f.name.endswith(".info.json") and not f.name.endswith(".ann.json")
    )
    if not json_files:
        return None

    meta_path = json_files[-1]
    npz_path = meta_path.with_suffix(".npz")
    if not npz_path.exists():
        logger.error(f"cache npz file none: {npz_path}")
        return None

    arrays = np.load(npz_path)["arrays"]
    meta: List[dict] = json.loads(meta_path.read_text(encoding="utf-8"))
    if len(meta) != len(arrays):
        logger.error(f"Cache metadata/array size mismatch: {cache_dir}")
        return None

    return [
        {**m, "array": arrays[i].astype(np.int32)}
        for i, m in enumerate(meta)
    ]


def _get_cache_key(cache_dir: Path) -> Optional[str]:
    """Extract a cache key from the NPZ filename in a cache directory."""
    npz_files = sorted(f for f in cache_dir.glob("*.npz"))
    if not npz_files:
        return None
    return npz_files[-1].stem


# ── sample_id shortening ─────────────────────────────────────────────────────

def _shorten_source_id(source_id: str, game: str) -> str:
    """Convert source_id to a concise per-game identifier."""
    if game == "doom":
        if "|" in source_id:
            path_part, slice_idx = source_id.rsplit("|", 1)
        else:
            path_part, slice_idx = source_id, "0"
        p = Path(path_part)
        parts = p.parts
        version = "Doom2" if any("Doom2" in part for part in parts) else "Doom1"
        return f"{version}_{p.stem}_{int(slice_idx):03d}"

    if game == "sokoban":
        if "#" in source_id:
            path_part, lvl_idx = source_id.rsplit("#", 1)
        else:
            path_part, lvl_idx = source_id, "0"
        p = Path(path_part)
        parts = p.parts
        difficulty = "hard" if any("hard" in part for part in parts) else "medium"
        return f"{difficulty}_{p.stem}_{int(lvl_idx):03d}"

    return source_id


# ── measure compute ─────────────────────────────────────────────────────────────────

def _compute_measures(
    env_map: np.ndarray,
    game: str,
) -> Tuple[float, float, float, float, float, float]:
    """
    Compute six measures from tile_mapping.json unified categories using the
    same mapping as data-loader to_unified.

      raw map → to_unified (0-4) → +1 shift → MultigameTiles (1-5)

    passable = EMPTY(1) + HAZARD(4) + COLLECTABLE(5), shared by all games

    Returns
    -------
    (rg, pl, wc, ic_inter, ic_hazard, ic_coll)
    """
    unified       = to_unified(env_map, game, warn_unmapped=False)
    multigame_map = jnp.array(unified + 1, dtype=jnp.int32)

    rg       = float(eval_get_region(multigame_map, _UNIFIED_PASSIBLE))
    pl       = float(eval_get_path_length(multigame_map, _UNIFIED_PASSIBLE))
    wc       = float(jnp.sum(multigame_map == int(MultigameTiles.WALL)))
    ic_inter = float(get_interactive_count(multigame_map))
    ic_haz   = float(get_hazard_count(multigame_map))
    ic_coll  = float(get_collectable_count(multigame_map))

    return rg, pl, wc, ic_inter, ic_haz, ic_coll


# ── row create ───────────────────────────────────────────────────────────────────────

def _make_rows(
    samples: List[dict],
    game: str,
    config: dict,
) -> List[dict]:
    """
    Create 5*N rows for all samples from one game.
    Output all reward_enum 0 rows, then 1, through 4, grouped by enum.

    samples format: [{"array": np.ndarray, "source_id": str, "instruction": str|None}, ...]
    """
    prefix = GAME_PREFIX.get(game, game[:2])

    # Step 1: compute measures for all samples
    computed = []
    for order_idx, sample in enumerate(samples):
        env_map   = sample["array"]
        source_id = _shorten_source_id(
            sample.get("source_id", str(order_idx)), game
        )

        if (order_idx + 1) % 100 == 0 or order_idx == 0:
            logger.info(f"  [{game}] {order_idx + 1}/{len(samples)} …")

        try:
            rg, pl, wc, oc, mc, ic = _compute_measures(env_map, game)
        except Exception as exc:
            logger.warning(f"  Measure failed ({source_id}): {exc}; replacing with 0")
            rg, pl, wc, oc, mc, ic = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        computed.append((order_idx, source_id, rg, pl, wc, oc, mc, ic))

    # Step 2: create rows grouped by reward_enum
    sc_object = config["sub_cond_interactable"]
    sc_mob    = config["sub_cond_hazard"]
    sc_item   = config["sub_cond_collectable"]

    # (reward_enum, feature_name, cond_col, val_idx, sub_condition)
    # val_idx: index in the computed tuple (2=rg, 3=pl, 5=oc, 6=mc, 7=ic)
    enum_specs = [
        (0, "region",             "condition_0", 2, ""),
        (1, "path_length",        "condition_1", 3, ""),
        (2, "interactable_count", "condition_2", 5, sc_object),
        (3, "hazard_count",       "condition_3", 6, sc_mob),
        (4, "collectable_count",  "condition_4", 7, sc_item),
    ]

    rows: List[dict] = []
    row_n = 0  # Sequential row number within the game

    for reward_enum, feature_name, cond_col, val_idx, sub_cond in enum_specs:
        for order_idx, source_id, *vals in computed:
            value = vals[val_idx - 2]  # offset: idx 2 → vals[0]
            row: dict = {
                "key":             f"{prefix}{row_n:06d}",
                "source_id":       source_id,
                "reward_enum":     reward_enum,
                "feature_name":    feature_name,
                "sub_condition":   sub_cond,
                "condition_0":     None,
                "condition_1":     None,
                "condition_2":     None,
                "condition_3":     None,
                "condition_4":     None,
                "instruction_raw": None,
                "instruction_uni": None,
            }
            row[cond_col] = value
            rows.append(row)
            row_n += 1

    return rows


# ── Public functions ─────────────────────────────────────────────────────────

def compute_game_annotations(
    samples,
    game: str,
) -> List[Dict[str, Any]]:
    """
    Compute and return annotation rows for five reward_enum values from a list
    of GameSamples or dictionaries containing array/source_id.

    Used for automatic annotation inside MultiGameDataset.

    Parameters
    ----------
    samples : List[GameSample] or List[dict]
        Each item must have .array/.source_id attributes or keys.
    game : str

    Returns
    -------
    List[dict] in the same format returned by _make_rows().
    """
    config = _GAME_CONFIG.get(game, {
        "sub_cond_interactable": "",
        "sub_cond_hazard":       "",
        "sub_cond_collectable":  "",
    })

    # Support both GameSample and dictionary inputs
    sample_dicts: List[dict] = []
    for s in samples:
        if isinstance(s, dict):
            sample_dicts.append(s)
        else:
            sample_dicts.append({
                "array":      s.array,
                "source_id":  s.source_id,
                "instruction": getattr(s, "instruction", None),
            })

    return _make_rows(sample_dicts, game, config)


# ── Per-game processing ───────────────────────────────────────────────────────

def annotate_game(
    game: str,
    samples: List[dict],
    cache_dir: Path,
    key: str,
    force: bool = False,
) -> int:
    """
    Compute one game's annotations, save them to {key}.ann.json, and return the row count.

    With force=False, skip when ann.json already exists.
    """
    if not force:
        existing = load_game_annotations_from_cache(cache_dir, game, key)
        if existing is not None:
            logger.info(f"  [{game}] ann.json already exists; skipped (use --force to regenerate)")
            return len(existing.get("annotations", []))

    config = _GAME_CONFIG[game]
    logger.info(f"\n=== {game.upper()} ({len(samples)} samples) ===")

    t0 = time.perf_counter()
    rows = _make_rows(samples, game, config)
    elapsed = time.perf_counter() - t0

    save_game_annotations_to_cache(
        cache_dir, game, key, rows,
        has_instructions=False,
        n_samples=len(samples),
    )

    logger.info(
        f"  → {game}/{key[:12]}….ann.json  "
        f"({len(rows)} rows, {len(samples)} samples × 5)  [{elapsed:.1f}s]"
    )
    return len(rows)


# ── CLI ───────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reward annotation create → {key}.ann.json save"
    )
    parser.add_argument(
        "--games", nargs="+",
        default=["doom", "zelda", "sokoban", "pokemon", "dungeon"],
        choices=["doom", "zelda", "sokoban", "pokemon", "dungeon"],
        help="Games to process (default: all)",
    )
    parser.add_argument(
        "--cache-dir", type=Path,
        default=_CACHE_DIR,
        help=f"cache directory (default: {_CACHE_DIR})",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite an existing ann.json",
    )
    args = parser.parse_args()

    logger.info(f"cache load: {args.cache_dir}")

    # Maximum samples per game
    _MAX_SAMPLES: Dict[str, int] = {
        "doom":    1000,
        "zelda":   1000,
        "sokoban": 1000,
        "pokemon": 1000,
        "dungeon": 4000,
    }

    # Scan all subdirectories and aggregate by game tag
    # Collect as {game: {key: samples}}
    by_game_key: Dict[str, Dict[str, List[dict]]] = {}

    if args.cache_dir.is_dir():
        for sub in sorted(args.cache_dir.iterdir()):
            if not sub.is_dir():
                continue
            key = _get_cache_key(sub)
            if key is None:
                continue
            samples = _load_cache_dir(sub)
            if not samples:
                continue
            for s in samples:
                g = s.get("game", "")
                if g:
                    by_game_key.setdefault(g, {}).setdefault(key, []).append(s)

    if not by_game_key:
        logger.error("No caches were loaded")
        return

    # max_samples apply
    for g in list(by_game_key.keys()):
        limit = _MAX_SAMPLES.get(g)
        for k in list(by_game_key[g].keys()):
            if limit is not None and len(by_game_key[g][k]) > limit:
                by_game_key[g][k] = by_game_key[g][k][:limit]

    logger.info(
        "Samples by game: "
        + ", ".join(
            f"{g}={sum(len(v) for v in ks.values())}"
            for g, ks in sorted(by_game_key.items())
        )
    )

    total_rows = 0
    for game in args.games:
        game_keys = by_game_key.get(game, {})
        if not game_keys:
            logger.warning(f"{game}: no cached samples; skipped")
            continue
        for key, samples in game_keys.items():
            n = annotate_game(game, samples, args.cache_dir, key, force=args.force)
            total_rows += n

    logger.info(f"\nDone: created {total_rows} rows")


if __name__ == "__main__":
    main()

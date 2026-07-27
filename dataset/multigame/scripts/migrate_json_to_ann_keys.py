#!/usr/bin/env python3
"""

dataset/multigame/scripts/migrate_json_to_ann_keys.py
======================================================
Temporary script for migrating legacy {key}.json caches to the new format.

Legacy format: game, source_id, instruction, order, meta
New format: game, source_id, order, ann_keys (ann.json keys only)

convert condition:
  - Calculate and add ann_keys only when ann.json exists.
  - Without ann.json, remove instruction/meta and leave ann_keys empty.

Usage:
  python dataset/multigame/scripts/migrate_json_to_ann_keys.py
  python dataset/multigame/scripts/migrate_json_to_ann_keys.py --cache-dir dataset/multigame/cache/artifacts
  python dataset/multigame/scripts/migrate_json_to_ann_keys.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from dataset.multigame.cache_utils import (
    load_game_annotations_from_cache,
    update_json_with_ann_keys,
    _stable_json,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_HERE = Path(__file__).parent
_DEFAULT_CACHE_DIR = _HERE.parent / "cache" / "artifacts"


def _is_old_format(entry: dict) -> bool:
    """Return whether data is legacy format, indicated by instruction or meta."""
    return "instruction" in entry or "meta" in entry


def migrate_game(cache_dir: Path, game: str, dry_run: bool = False) -> int:
    """Convert one game's JSON files to the new format and return the count."""
    game_dir = cache_dir / game
    if not game_dir.exists():
        return 0

    json_files = [
        f for f in game_dir.glob("*.json")
        if not f.name.endswith(".info.json") and not f.name.endswith(".ann.json")
    ]
    if not json_files:
        return 0

    converted = 0
    for meta_path in json_files:
        key = meta_path.stem
        entries = json.loads(meta_path.read_text(encoding="utf-8"))

        if not entries:
            continue

        # For new-format data, fill ann_keys only when missing
        already_new = not _is_old_format(entries[0])

        # ann.json in  ann_keys compute
        ann_data = load_game_annotations_from_cache(cache_dir, game, key)

        if already_new and ann_data is None:
            logger.info(f"  [{game}] {key[:12]}….json: new format, no ann.json -- skipped")
            continue
        if already_new and "ann_keys" in entries[0]:
            logger.info(f"  [{game}] {key[:12]}….json: new format with ann_keys -- skipped")
            continue

        # Convert to the new format
        new_entries = []
        for e in entries:
            new_entry: dict = {
                "game":      e["game"],
                "source_id": e["source_id"],
                "order":     e.get("order"),
            }
            new_entries.append(new_entry)

        n_samples = len(new_entries)
        logger.info(f"  [{game}] {key[:12]}….json: converted {n_samples} samples"
                    + (" (dry-run)" if dry_run else ""))

        if not dry_run:
            meta_path.write_text(_stable_json(new_entries), encoding="utf-8")
            converted += 1
            # Add ann_keys
            if ann_data is not None:
                update_json_with_ann_keys(cache_dir, game, key, ann_data)
                logger.info(f"  [{game}] {key[:12]}….json: added ann_keys")
            else:
                logger.warning(f"  [{game}] no ann.json -- saved without ann_keys (added after annotate.py and reload)")
        else:
            converted += 1

    return converted


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Migrate {key}.json to the new game/source_id/order/ann_keys format"
    )
    parser.add_argument("--cache-dir", type=Path, default=_DEFAULT_CACHE_DIR)
    parser.add_argument("--games", nargs="+",
                        default=["doom", "dungeon", "zelda", "pokemon", "sokoban"])
    parser.add_argument("--dry-run", action="store_true",
                        help="Print migration targets without writing")
    args = parser.parse_args()

    logger.info(f"cache directory: {args.cache_dir}")
    if args.dry_run:
        logger.info("(dry-run mode: no changes)")

    total = 0
    for game in args.games:
        n = migrate_game(args.cache_dir, game, dry_run=args.dry_run)
        if n:
            logger.info(f"[{game}] converted {n} files")
        total += n

    logger.info(f"\nDone: converted {total} files")


if __name__ == "__main__":
    main()

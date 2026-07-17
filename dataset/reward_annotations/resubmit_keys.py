#!/usr/bin/env python3
"""
dataset/reward_annotations/resubmit_keys.py
============================================
text map text in  text batch  text (failuretext sample textprocess for ).

TARGET_KEYS array in  textprocesstext map text(= ann.json row["key"])  text Usagetext text.

Usage:
    python dataset/reward_annotations/resubmit_keys.py              # text + finish text + ann.json update
    python dataset/reward_annotations/resubmit_keys.py --dry-run    # JSONL createtext, text X
    python dataset/reward_annotations/resubmit_keys.py --retrieve BATCH_ID
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE.parent.parent))

import numpy as np
from dotenv import load_dotenv

load_dotenv(_HERE.parent.parent / ".env")

from dataset.multigame.cache_utils import (
    load_game_annotations_from_cache,
    find_game_cache_key,
)
from generate_instructions import (
    build_batch_request,
    load_system_prompt,
    load_cache_by_game,
    submit_batch,
    retrieve_batch_results,
    update_caches,
    check_batch_status,
    _CACHE_DIR,
    _BATCH_DIR,
    _ENUM_TO_COND_COL,
)
from instruction_config import CUSTOM_THRESHOLDS

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# text map text list — text in  text  map text  text
# map text = ann.json each row  of  "key" text
#   doom    text: "dm000000", "dm000123"
#   zelda   text: "zl000000"
#   sokoban text: "sk000000"
#   pokemon text: "pk000000"
#   dungeon text: "dg000000"
# ─────────────────────────────────────────────────────────────────────────────
TARGET_KEYS: list[str] = [
    "pk001058",
    "dm001182"
]
# ─────────────────────────────────────────────────────────────────────────────

_ALL_GAMES = ["doom", "zelda", "sokoban", "pokemon", "dungeon"]
_POLL_INTERVAL = 10  # batch finish check text (seconds)


def build_jsonl_for_keys(
    target_keys: list[str],
    cache_dir: Path,
    cache_by_game: dict,
    system_prompt: str,
) -> tuple[Path, list[str]] | tuple[None, None]:
    """TARGET_KEYS in  text  rowtext text JSONL  createtext."""
    key_set = set(target_keys)
    lines: list[str] = []
    matched_games: set[str] = set()

    for game in _ALL_GAMES:
        cache_key = find_game_cache_key(cache_dir, game)
        if cache_key is None:
            continue
        ann_data = load_game_annotations_from_cache(cache_dir, game, cache_key)
        if ann_data is None:
            continue

        sid_map = cache_by_game.get(game, {})
        for row in ann_data.get("annotations", []):
            if row["key"] not in key_set:
                continue
            array = sid_map.get(row["source_id"])
            if array is None:
                logger.warning(f"array none: game={game} source_id={row['source_id']}")
                continue

            feature_name = row["feature_name"]
            if CUSTOM_THRESHOLDS.get(f"{game}_{feature_name}") is None:
                logger.info(f"threshold=None text: {row['key']}")
                continue

            reward_enum = int(row["reward_enum"])
            cond_col = _ENUM_TO_COND_COL.get(reward_enum)
            raw_val = row.get(cond_col)
            if raw_val is None:
                continue
            try:
                cond_val = float(raw_val)
            except (TypeError, ValueError):
                continue

            req = build_batch_request(
                row["key"], game, feature_name,
                cond_val, row.get("sub_condition", ""),
                array, system_prompt,
            )
            lines.append(json.dumps(req, ensure_ascii=False))
            matched_games.add(game)
            logger.info(f"  text : {row['key']}  game={game}  enum={reward_enum}  feature={feature_name}")

    if not lines:
        logger.warning("text text  text. TARGET_KEYS  checktext.")
        return None, None

    _BATCH_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = _BATCH_DIR / f"resubmit_{ts}.jsonl"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info(f"JSONL create: {out_path.name}  ({len(lines)} requests)")
    return out_path, list(matched_games)


def main() -> None:
    parser = argparse.ArgumentParser(description="text map text batch text")
    parser.add_argument("--dry-run", action="store_true",
                        help="JSONL createtext text batch text  text text")
    parser.add_argument("--retrieve", metavar="BATCH_ID",
                        help="finishtext batch result  text ann.json in  apply")
    args = parser.parse_args()

    # ── retrieve ──
    if args.retrieve:
        logger.info(f"result text: {args.retrieve}")
        results = retrieve_batch_results(args.retrieve)
        n = update_caches(results, _CACHE_DIR, _ALL_GAMES)
        logger.info(f"total {n}text update finish")
        return

    if not TARGET_KEYS:
        logger.error("TARGET_KEYS  text text. resubmit_keys.py top of  TARGET_KEYS in  text  text text.")
        sys.exit(1)

    logger.info(f"target map text: {len(TARGET_KEYS)}text")
    system_prompt = load_system_prompt()
    cache_by_game = load_cache_by_game(_CACHE_DIR)

    jsonl_path, matched_games = build_jsonl_for_keys(
        TARGET_KEYS, _CACHE_DIR, cache_by_game, system_prompt
    )
    if jsonl_path is None:
        sys.exit(1)

    if args.dry_run:
        logger.info("--dry-run mode: batch text text")
        return

    # ── text + finish text + ann.json update ──
    n_requests = sum(1 for _ in jsonl_path.open(encoding="utf-8"))
    batch_id = submit_batch(jsonl_path, matched_games or _ALL_GAMES, [], n_requests)
    logger.info(f"batch text finish: {batch_id}")
    logger.info(f"finish text  during  (interval={_POLL_INTERVAL}s) …")

    start_time = time.time()
    while True:
        time.sleep(_POLL_INTERVAL)
        info   = check_batch_status(batch_id)
        status = info["status"]
        counts = info["request_counts"]
        elapsed     = int(time.time() - start_time)
        elapsed_str = f"{elapsed // 60}m {elapsed % 60:02d}s"
        c, t = counts["completed"], counts["total"]
        print(f"\r\033[K  [{elapsed_str}] {batch_id}: {status}  {c}/{t} completed",
              end="", flush=True)

        if status == "completed":
            print()
            results = retrieve_batch_results(batch_id)
            n = update_caches(results, _CACHE_DIR, _ALL_GAMES)
            logger.info(f"total {n}text update finish")
            break
        elif status in ("failed", "expired", "cancelled"):
            print()
            logger.error(f"batch failure/text/text: {status}")
            logger.info(f"text text: python resubmit_keys.py --retrieve {batch_id}")
            sys.exit(1)


if __name__ == "__main__":
    main()

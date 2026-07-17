#!/usr/bin/env python3
"""
dataset/cpcgrl_buffer/build_pair_dataset.py
============================================
saves/ folder of  CPCGRL training text(.npz)  (game, reward_enum) by text
text text (env_map[t], env_map[t+1])   text,
duplicate remove  after  **text .npz file** in  dict form to  savetext.

text:
    dataset/cpcgrl_buffer/cpcgrl_pair_dataset.npz
        text structure:
            {game}_re{rn}       : (N, 2, 16, 16) int32  — env_map pairs
            {game}_re{rn}_ts    : (N,) int64             — timesteps
            _metadata           : JSON string (0-d array)

Usage:
    python dataset/cpcgrl_buffer/build_pair_dataset.py \\
        [--saves_dir saves] \\
        [--pairs_per_group 50000] \\
        [--seed 42]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import platform
import re
import socket
from datetime import datetime

import numpy as np


# ── helpers ──────────────────────────────────────────────────────────────────

def parse_game_and_re(dirname: str) -> tuple[str | None, int | None]:
    """directory name in  game name and  reward_enum text extract.
    text: 'buffer-exp-cb_game-doom_re-3_vec_ro_s-0' → ('doom', 3)
    """
    gm = re.search(r"_game-(\w+)_", dirname)
    rm = re.search(r"_re-(\d+)-?_", dirname)
    game = gm.group(1) if gm else None
    rn = int(rm.group(1)) if rm else None
    return game, rn


def load_buffer_dir(buffer_dir: str):
    """text directory of  text .npz   text (env_maps, dones, timesteps)   return."""
    npz_files = sorted(glob.glob(os.path.join(buffer_dir, "*.npz")))
    maps, dones, ts = [], [], []
    for f in npz_files:
        d = np.load(f)
        n = d["env_map"].shape[0]
        maps.append(d["env_map"])
        dones.append(d["done"])
        ts.append(np.arange(n, dtype=np.int64) + int(d["timestep"]))
        d.close()
    return (
        np.concatenate(maps, axis=0),
        np.concatenate(dones, axis=0),
        np.concatenate(ts, axis=0),
    )


def make_pairs(env_maps, dones, timesteps):
    """text 2-step text create. done text text timestep text remove."""
    n = env_maps.shape[0]
    if n < 2:
        empty = np.empty((0, 2, *env_maps.shape[1:]), dtype=env_maps.dtype)
        return empty, np.empty((0,), dtype=np.int64)

    valid = ~dones[:-1]
    td = np.diff(timesteps)
    valid &= (td > 0) & (td < 10000)

    idx = np.where(valid)[0]
    pairs = np.stack([env_maps[idx], env_maps[idx + 1]], axis=1)
    return pairs, timesteps[idx]


def deduplicate_pairs(pairs: np.ndarray) -> np.ndarray:
    """env_map text textabove to  text before  sametext row remove. return: text row index."""
    flat = pairs.reshape(pairs.shape[0], -1)
    _, unique_idx = np.unique(flat, axis=0, return_index=True)
    unique_idx.sort()
    return unique_idx


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="CPCGRL buffer → single .npz pair dataset (keyed by game/re)"
    )
    parser.add_argument("--saves_dir", default="saves")
    parser.add_argument("--pairs_per_group", type=int, default=50000,
                        help="(game, re) text text maximum extract text text")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    out_dir = os.path.join("dataset", "cpcgrl_buffer")
    os.makedirs(out_dir, exist_ok=True)

    # 1. (game, reward_enum) text text text
    exp_dirs = sorted(glob.glob(os.path.join(args.saves_dir, "*_vec_ro_s-*")))
    group_bufs: dict[tuple[str, int], str] = {}
    for ed in exp_dirs:
        game, rn = parse_game_and_re(os.path.basename(ed))
        if game is None or rn is None:
            continue
        bd = os.path.join(ed, "buffer")
        if os.path.isdir(bd) and glob.glob(os.path.join(bd, "*.npz")):
            group_bufs[(game, rn)] = bd

    found = sorted(group_bufs.keys())
    games_found = sorted(set(g for g, _ in found))
    res_found = sorted(set(r for _, r in found))
    print(f"Found {len(found)} (game, re) groups")
    print(f"  games: {games_found}")
    print(f"  reward_enums: {res_found}")
    assert found, "No buffer dirs found!"

    # 2. text text extract → dict  in  text
    arrays: dict[str, np.ndarray] = {}
    group_info = []
    total_pairs = 0
    total_before_dedup = 0

    for game, rn in found:
        buf_dir = group_bufs[(game, rn)]
        key = f"{game}_re{rn}"
        print(f"\n[{key}] {buf_dir}")

        env_maps, dones, timesteps = load_buffer_dir(buf_dir)
        print(f"  transitions: {env_maps.shape[0]}")

        pairs, pts = make_pairs(env_maps, dones, timesteps)
        print(f"  candidate pairs: {pairs.shape[0]}")

        if pairs.shape[0] == 0:
            print(f"  WARNING: skip {key}")
            continue

        # sampletext
        n_sample = min(args.pairs_per_group, pairs.shape[0])
        idx = rng.choice(pairs.shape[0], size=n_sample, replace=False)
        idx.sort()
        pairs = pairs[idx]
        pts = pts[idx]
        print(f"  sampled: {n_sample}")

        # duplicate remove
        n_before = pairs.shape[0]
        uniq_idx = deduplicate_pairs(pairs)
        pairs = pairs[uniq_idx]
        pts = pts[uniq_idx]
        print(f"  after dedup: {pairs.shape[0]} (removed {n_before - pairs.shape[0]})")

        # text
        perm = rng.permutation(pairs.shape[0])
        pairs = pairs[perm]
        pts = pts[perm]

        # dict  in  text
        arrays[key] = pairs            # (N, 2, H, W)
        arrays[f"{key}_ts"] = pts      # (N,)

        group_info.append({
            "key": key,
            "game": game,
            "reward_enum": rn,
            "n_pairs": int(pairs.shape[0]),
            "n_before_dedup": n_before,
            "tile_min": int(pairs.min()),
            "tile_max": int(pairs.max()),
        })
        total_pairs += pairs.shape[0]
        total_before_dedup += n_before

    # 3. metadata  JSON string to  save
    metadata = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "seed": args.seed,
        "pairs_per_group_requested": args.pairs_per_group,
        "saves_dir": os.path.abspath(args.saves_dir),
        "total_pairs": total_pairs,
        "total_before_dedup": total_before_dedup,
        "games": games_found,
        "reward_enums": res_found,
        "groups": group_info,
    }
    arrays["_metadata"] = np.array(json.dumps(metadata, ensure_ascii=False))

    # 4. text file save
    out_path = os.path.join(out_dir, "cpcgrl_pair_dataset.npz")
    np.savez_compressed(out_path, **arrays)

    fsize = os.path.getsize(out_path)

    # 5. summary text
    print(f"\n{'=' * 64}")
    print(f"  CPCGRL Pair Dataset")
    print(f"{'=' * 64}")
    print(f"  path          : {out_path}")
    print(f"  file size     : {fsize / 1024:.0f} KB")
    print(f"  total groups  : {len(group_info)}")
    print(f"  total pairs   : {total_pairs:,}")
    print(f"  before dedup  : {total_before_dedup:,}")
    print(f"  games         : {games_found}")
    print(f"  reward_enums  : {res_found}")

    print(f"\n  keys in .npz:")
    for gi in group_info:
        k = gi["key"]
        print(f"    {k:20s}  shape={arrays[k].shape}  "
              f"n={gi['n_pairs']:>5,}  tiles=[{gi['tile_min']},{gi['tile_max']}]")

    print(f"\n  per-game totals:")
    for g in games_found:
        n = sum(gi["n_pairs"] for gi in group_info if gi["game"] == g)
        ng = sum(1 for gi in group_info if gi["game"] == g)
        print(f"    {g:12s}: {n:>6,} pairs  ({ng} groups)")

    print(f"\n  per-re totals:")
    for r in res_found:
        n = sum(gi["n_pairs"] for gi in group_info if gi["reward_enum"] == r)
        ng = sum(1 for gi in group_info if gi["reward_enum"] == r)
        print(f"    re-{r}: {n:>6,} pairs  ({ng} groups)")

    print(f"\n  build info:")
    print(f"    created_at : {metadata['created_at']}")
    print(f"    hostname   : {metadata['hostname']}")
    print(f"    platform   : {metadata['platform']}")
    print(f"{'=' * 64}")


if __name__ == "__main__":
    main()


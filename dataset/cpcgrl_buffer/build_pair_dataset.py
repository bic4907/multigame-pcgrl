#!/usr/bin/env python3
"""
dataset/cpcgrl_buffer/build_pair_dataset.py
============================================
saves/ 폴더의 CPCGRL 학습 버퍼(.npz)를 reward_enum 별로 읽어서
연속 쌍 (env_map[t], env_map[t+1]) 을 구성하고,
re-1~5 에서 골고루 추출 · 중복 제거 후 **단일 .npz 파일**로 저장한다.

출력:
    dataset/cpcgrl_buffer/cpcgrl_pair_dataset.npz
        - env_map_pairs  : (N, 2, 16, 16) int32   # (before, after) 쌍
        - reward_enums   : (N,) int32              # 각 쌍의 reward_enum 라벨
        - timesteps      : (N,) int64              # 각 쌍의 시작 timestep

Usage:
    python dataset/cpcgrl_buffer/build_pair_dataset.py \\
        [--saves_dir saves] \\
        [--pairs_per_re 4000] \\
        [--seed 42]
"""
from __future__ import annotations

import argparse
import glob
import os
import re

import numpy as np


# ── helpers ──────────────────────────────────────────────────────────────────

def parse_reward_enum(dirname: str) -> int | None:
    """디렉토리 이름에서 reward_enum 번호 추출.
    예: '..._re-3_vec_ro_s-0' → 3,  '..._re-4-_vec_ro_s-0' → 4
    """
    m = re.search(r"_re-(\d+)-?_", dirname)
    return int(m.group(1)) if m else None


def load_buffer_dir(buffer_dir: str):
    """버퍼 디렉토리의 모든 .npz 를 읽어 (env_maps, dones, timesteps) 를 반환."""
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
    """연속 2-step 쌍 생성. done 경계 및 timestep 점프 제거."""
    n = env_maps.shape[0]
    if n < 2:
        empty = np.empty((0, 2, *env_maps.shape[1:]), dtype=env_maps.dtype)
        return empty, np.empty((0,), dtype=np.int64)

    valid = ~dones[:-1]                        # done[t]=True → t→t+1 건너뜀
    td = np.diff(timesteps)
    valid &= (td > 0) & (td < 10000)          # 비정상 점프 제거

    idx = np.where(valid)[0]
    pairs = np.stack([env_maps[idx], env_maps[idx + 1]], axis=1)  # (M, 2, H, W)
    return pairs, timesteps[idx]


def deduplicate_pairs(pairs: np.ndarray) -> np.ndarray:
    """env_map 쌍 단위로 완전 동일한 행 제거. 반환: 고유 행 인덱스."""
    # (N, 2, H, W) → (N, 2*H*W)  바이트 뷰로 비교
    flat = pairs.reshape(pairs.shape[0], -1)
    _, unique_idx = np.unique(flat, axis=0, return_index=True)
    unique_idx.sort()
    return unique_idx


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="CPCGRL buffer → deduplicated pair dataset (single file)"
    )
    parser.add_argument("--saves_dir", default="saves")
    parser.add_argument("--pairs_per_re", type=int, default=4000,
                        help="reward_enum 당 추출할 쌍 수")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    out_dir = os.path.join("dataset", "cpcgrl_buffer")
    os.makedirs(out_dir, exist_ok=True)

    # 1. reward_enum 별 버퍼 탐색
    exp_dirs = sorted(glob.glob(os.path.join(args.saves_dir, "*_vec_ro_s-*")))
    re_bufs: dict[int, str] = {}
    for ed in exp_dirs:
        rn = parse_reward_enum(os.path.basename(ed))
        if rn is None:
            continue
        bd = os.path.join(ed, "buffer")
        if os.path.isdir(bd) and glob.glob(os.path.join(bd, "*.npz")):
            re_bufs[rn] = bd

    found = sorted(re_bufs.keys())
    print(f"Found reward_enums: {found}")
    assert found, "No buffer dirs found!"

    # 2. re 별 쌍 추출
    all_pairs, all_re, all_ts = [], [], []
    for rn in found:
        print(f"\n[re-{rn}] {re_bufs[rn]}")
        env_maps, dones, timesteps = load_buffer_dir(re_bufs[rn])
        print(f"  transitions: {env_maps.shape[0]}")

        pairs, pts = make_pairs(env_maps, dones, timesteps)
        print(f"  candidate pairs: {pairs.shape[0]}")

        if pairs.shape[0] == 0:
            print(f"  WARNING: skip re-{rn}")
            continue

        n_sample = min(args.pairs_per_re, pairs.shape[0])
        idx = rng.choice(pairs.shape[0], size=n_sample, replace=False)
        idx.sort()

        all_pairs.append(pairs[idx])
        all_re.append(np.full(n_sample, rn, dtype=np.int32))
        all_ts.append(pts[idx])
        print(f"  sampled: {n_sample}")

    # 3. 병합
    merged_pairs = np.concatenate(all_pairs, axis=0)   # (N_raw, 2, H, W)
    merged_re = np.concatenate(all_re, axis=0)         # (N_raw,)
    merged_ts = np.concatenate(all_ts, axis=0)         # (N_raw,)

    print(f"\nMerged (before dedup): {merged_pairs.shape[0]}")

    # 4. 중복 제거
    uniq_idx = deduplicate_pairs(merged_pairs)
    merged_pairs = merged_pairs[uniq_idx]
    merged_re = merged_re[uniq_idx]
    merged_ts = merged_ts[uniq_idx]

    print(f"After dedup: {merged_pairs.shape[0]}")

    # 5. 셔플
    perm = rng.permutation(merged_pairs.shape[0])
    merged_pairs = merged_pairs[perm]
    merged_re = merged_re[perm]
    merged_ts = merged_ts[perm]

    # 6. 저장
    out_path = os.path.join(out_dir, "cpcgrl_pair_dataset.npz")
    np.savez_compressed(
        out_path,
        env_map_pairs=merged_pairs,
        reward_enums=merged_re,
        timesteps=merged_ts,
    )

    print(f"\n{'=' * 60}")
    print(f"  CPCGRL Pair Dataset")
    print(f"{'=' * 60}")
    print(f"  env_map_pairs : {merged_pairs.shape}  {merged_pairs.dtype}")
    print(f"  reward_enums  : {merged_re.shape}  {merged_re.dtype}")
    print(f"  timesteps     : {merged_ts.shape}  {merged_ts.dtype}")
    print(f"  file size     : {os.path.getsize(out_path) / 1024:.0f} KB")
    print(f"  path          : {out_path}")

    # re 별 분포
    print(f"\n  reward_enum distribution:")
    for rn in found:
        cnt = (merged_re == rn).sum()
        print(f"    re-{rn}: {cnt:,}")
    print(f"    total: {merged_pairs.shape[0]:,}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()


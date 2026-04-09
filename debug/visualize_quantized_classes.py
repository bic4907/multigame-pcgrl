#!/usr/bin/env python
"""
debug/visualize_quantized_classes.py
====================================
Condition quantization이 어떻게 쪼개졌는지 확인하는 스크립트.

각 (game, reward_enum)별로:
  - 원래 condition 분포 히스토그램
  - CUSTOM_THRESHOLDS 경계선
  - quantized bin별 샘플 수 바차트

Usage:
    python debug/visualize_quantized_classes.py                     # 전체 게임
    python debug/visualize_quantized_classes.py --game dungeon      # dungeon만
    python debug/visualize_quantized_classes.py --game dungeon --save  # 저장
"""
import argparse
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

# 프로젝트 루트를 path에 추가
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset.multigame import MultiGameDataset
from dataset.reward_annotations.instruction_config import CUSTOM_THRESHOLDS

REWARD_ENUM_NAMES = {
    0: "region",
    1: "path_length",
    2: "interactable",
    3: "hazard",
    4: "collectable",
}

FEATURE_NAME_MAP = {
    0: "region",
    1: "path_length",
    2: "interactable_count",
    3: "hazard_count",
    4: "collectable_count",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", type=str, default=None,
                        help="특정 게임만 (dungeon, doom, zelda, sokoban, pokemon)")
    parser.add_argument("--save", action="store_true",
                        help="그래프를 파일로 저장 (debug/quantized_classes/ 에)")
    args = parser.parse_args()

    # 데이터셋 로드
    include_all = args.game is None
    ds = MultiGameDataset(
        include_dungeon=include_all or args.game == "dungeon",
        include_doom=include_all or args.game == "doom",
        include_doom2=False,
        include_zelda=include_all or args.game == "zelda",
        include_sokoban=include_all or args.game == "sokoban",
        include_pokemon=include_all or args.game == "pokemon",
        include_d2=False,
    )

    # (game, reward_enum) → [condition_value]
    data = defaultdict(list)
    for s in ds._samples:
        re = s.meta.get("reward_enum")
        if re is None:
            continue
        conds = s.meta.get("conditions", {})
        cval = conds.get(re, next(iter(conds.values()), None))
        if cval is not None:
            data[(s.game, int(re))].append(float(cval))

    if not data:
        print("No samples with reward annotations found.")
        return

    # 그룹별 정렬
    groups = sorted(data.keys())
    n_plots = len(groups)
    ncols = min(4, n_plots)
    nrows = (n_plots + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.5 * nrows))
    if n_plots == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, (game, re_id) in enumerate(groups):
        ax = axes[idx]
        vals = np.array(data[(game, re_id)])
        feature_name = FEATURE_NAME_MAP.get(re_id, f"unknown_{re_id}")
        enum_name = REWARD_ENUM_NAMES.get(re_id, f"e{re_id}")
        threshold_key = f"{game}_{feature_name}"
        thresholds = CUSTOM_THRESHOLDS.get(threshold_key)

        # 히스토그램
        n_bins = min(60, max(10, len(set(vals)) // 2))
        ax.hist(vals, bins=n_bins, color="#6baed6", edgecolor="white", alpha=0.85,
                label=f"n={len(vals)}")

        # threshold 경계선 + bin 레이블
        if thresholds is not None:
            bin_counts = []
            boundaries = [-np.inf] + list(thresholds) + [np.inf]
            for i in range(len(boundaries) - 1):
                lo, hi = boundaries[i], boundaries[i + 1]
                cnt = np.sum((vals > lo) & (vals <= hi)) if i > 0 else np.sum(vals <= hi)
                if i == 0:
                    cnt = np.sum(vals < thresholds[0])
                bin_counts.append(cnt)
            # digitize 기준 재계산 (np.digitize 와 동일)
            digitized = np.digitize(vals, thresholds)  # 0,1,2,3
            bin_counts_exact = [int(np.sum(digitized == b)) for b in range(len(thresholds) + 1)]

            for i, t in enumerate(thresholds):
                ax.axvline(t, color="red", linestyle="--", linewidth=1.5,
                           label=f"t={t}" if i == 0 else f"t={t}")

            # bin별 개수 텍스트
            txt_parts = [f"bin{b}: {c}" for b, c in enumerate(bin_counts_exact)]
            ax.text(0.97, 0.97, "\n".join(txt_parts),
                    transform=ax.transAxes, fontsize=8, va="top", ha="right",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
        else:
            # threshold 없음 → 단일 bin
            ax.text(0.97, 0.97, f"bin0: {len(vals)}\n(no threshold)",
                    transform=ax.transAxes, fontsize=8, va="top", ha="right",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

        ax.set_title(f"{game} / e{re_id} ({enum_name})", fontsize=11, fontweight="bold")
        ax.set_xlabel("condition value")
        ax.set_ylabel("count")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(alpha=0.15)

    # 빈 subplot 숨기기
    for idx in range(n_plots, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Condition Quantization Distribution (CUSTOM_THRESHOLDS)",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()

    if args.save:
        out_dir = Path(__file__).parent / "quantized_classes"
        out_dir.mkdir(exist_ok=True)
        suffix = f"_{args.game}" if args.game else "_all"
        out_path = out_dir / f"quantized_dist{suffix}.png"
        fig.savefig(out_path, dpi=180, bbox_inches="tight")
        print(f"Saved to {out_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()


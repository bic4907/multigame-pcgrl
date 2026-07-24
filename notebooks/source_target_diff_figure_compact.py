"""
notebooks/source_target_diff_figure.py
======================================
source_target 실험의 논문용 그림 하나만 생성한다:
    source_target_diff_baseline.png / .pdf

  - 왼쪽 열: 각 target 의 target-only baseline progress (흰 배경, 숫자만)
  - 가운데: 큰 화살표
  - 오른쪽: source→target Δ = perf(target|source) − perf(target|baseline) 히트맵
            (seed-paired, reward_enum 평균). 빨강/파랑 = 도움/악화.

데이터: results/wandb_projects/aaai27_eval_mgpcgrl_source_target/<run>/<eval>/results.csv
        (results/utils/experiment/make_eval_summary.py 로 미리 만들어둬야 함)
출력  : notebooks/source_target/source_target_diff_baseline.{png,pdf}

실행:
    python notebooks/source_target_diff_figure.py
    python notebooks/source_target_diff_figure.py --vlim 8 --metric progress
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrow


def _muted_cmap(name: str, lo: float = 0.25, hi: float = 0.75, n: int = 256):
    """transferbility 와 동일하게 coolwarm 계열의 [lo, hi] 구간만 잘라 뮤트."""
    base = plt.get_cmap(name)
    return LinearSegmentedColormap.from_list(
        f"{name}_muted", base(np.linspace(lo, hi, n)))

_HERE = Path(__file__).resolve().parent          # notebooks/
_ROOT = _HERE.parent                             # project root

# ── 상수 ──────────────────────────────────────────────────────────────────────
ABBR_TO_GAME = {"dg": "dungeon", "pk": "pokemon", "sk": "sokoban", "dm": "doom", "zd": "zelda"}
GAME_ORDER = ["dg", "pk", "sk", "dm", "zd"]
BASELINE_KEY = "none"
METRIC_DISPLAY = {"progress": "Progress", "vit_score": "ViTScore",
                  "tpkldiv": "TPKL-Div", "diversity": "Diversity"}

_DEFAULT_INPUT = _ROOT / "results/wandb_projects/aaai27_eval_mgpcgrl_source_target"
_DEFAULT_OUT = _HERE / "source_target"

_NOTE = ""
#     (
#     r"$\Delta \mathrm{Progress}"
#     r" = \mathrm{perf}(\mathrm{target}\mid\mathrm{source})"
#     r" - \mathrm{perf}(\mathrm{target}\mid\mathrm{baseline})$, "
#     "\naveraged over 5 reward types"
# )
#

# ── 소형 유틸 (외부 의존 없이 인라인) ─────────────────────────────────────────
def _parse_run_tokens(run_name: str) -> dict[str, str]:
    """'mgpcgrl_game-all_re-0_..._s-0' → {'game':'all','re':'0',...}"""
    tokens = {}
    for part in run_name.split("_"):
        if "-" in part:
            k, v = part.split("-", 1)
            tokens[k] = v
    return tokens


def _safe_std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    m = sum(values) / len(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / len(values))


def _parse_source_target(tokens: dict[str, str]) -> tuple[str, str] | None:
    """(source|'none', target). encgame==target 이면 baseline, source+target 이면 source."""
    target = tokens.get("un")
    encgame = tokens.get("encgame")
    if not target or not encgame or target not in ABBR_TO_GAME:
        return None
    if encgame == target:
        return BASELINE_KEY, target
    if len(encgame) == 4 and encgame.endswith(target):
        source = encgame[:2]
        if source in ABBR_TO_GAME:
            return source, target
    return None


# ── 데이터 수집/집계 ──────────────────────────────────────────────────────────
def collect_raw(input_root: Path, metric: str) -> dict[tuple[str, str, str], dict[str, float]]:
    """(source, target, reward_enum) -> {seed: value}."""
    raw: dict[tuple[str, str, str], dict[str, float]] = defaultdict(dict)
    for results_path in sorted(input_root.rglob("results.csv")):
        rel = results_path.relative_to(input_root)
        if len(rel.parts) < 2:
            continue
        tokens = _parse_run_tokens(rel.parts[0])
        parsed = _parse_source_target(tokens)
        if parsed is None:
            continue
        source, target = parsed
        seed = tokens.get("s", rel.parts[0])
        enum = tokens.get("re")
        if enum is None:
            continue
        target_game = ABBR_TO_GAME[target]
        df = pd.read_csv(results_path)
        if metric not in df.columns or "game" not in df.columns:
            continue
        sub = df.loc[df["game"] == target_game, metric].dropna()
        if sub.empty:
            continue
        raw[(source, target, enum)][seed] = float(sub.mean())
    return raw


def build_overall_cells(raw: dict[tuple[str, str, str], dict[str, float]]) -> dict[tuple[str, str], dict]:
    """모든 reward_enum 을 합쳐 (source, target) 단위로 seed-first 집계."""
    grouped: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for (source, target, _enum), seed_vals in raw.items():
        for seed, v in seed_vals.items():
            grouped[(source, target)][seed].append(v)
    cells: dict[tuple[str, str], dict] = {}
    for key, seed_vals in grouped.items():
        seed_means = {s: sum(v) / len(v) for s, v in seed_vals.items()}
        means = list(seed_means.values())
        cells[key] = {"mean": sum(means) / len(means), "std": _safe_std(means),
                      "seed_means": seed_means}
    return cells


# ── 히트맵 그리기 ─────────────────────────────────────────────────────────────
def _draw_heatmap(ax, mat, row_labels, col_labels, *, cmap, vmin=None, vmax=None,
                  ref=None, signed=False, fmt="{:.1f}", std_mat=None, std_fmt="{:.1f}",
                  show_ylabels=True, ylabels_right=False, cell_fontsize=15,
                  std_fontsize=None, tick_fontsize=13, no_fill=False, aspect="equal"):
    if std_fontsize is None:
        std_fontsize = max(cell_fontsize - 2, 6)

    if no_fill:
        ax.set_xlim(-0.5, mat.shape[1] - 0.5)
        ax.set_ylim(mat.shape[0] - 0.5, -0.5)
        ax.set_facecolor("white")
        ax.set_aspect(aspect)
        for ri in range(mat.shape[0] + 1):
            ax.axhline(ri - 0.5, color="#cccccc", linewidth=0.8)
        for ci in range(mat.shape[1] + 1):
            ax.axvline(ci - 0.5, color="#cccccc", linewidth=0.8)
    else:
        ax.imshow(mat, cmap=cmap, aspect=aspect, vmin=vmin, vmax=vmax, interpolation="nearest")

    if ref is None:
        valid = mat[~np.isnan(mat)]
        ref = (max(abs(valid.max()), abs(valid.min())) if signed and len(valid)
               else (valid.max() if len(valid) else 1.0))
    has_std = std_mat is not None
    for ri in range(mat.shape[0]):
        for ci in range(mat.shape[1]):
            v = mat[ri, ci]
            if np.isnan(v):
                ax.text(ci, ri, "–", ha="center", va="center", fontsize=cell_fontsize, color="#999999")
                continue
            text_color = "black" if no_fill else ("white" if (abs(v) / ref if ref else 0) > 0.55 else "black")
            text = ("+" if signed and v > 0 else "") + fmt.format(v)
            if has_std and not np.isnan(std_mat[ri, ci]):
                ax.text(ci, ri - 0.12, text, ha="center", va="center",
                        fontsize=cell_fontsize, color=text_color, fontweight="bold")
                ax.text(ci, ri + 0.24, "±" + std_fmt.format(std_mat[ri, ci]), ha="center",
                        va="center", fontsize=std_fontsize, color=text_color)
            else:
                ax.text(ci, ri, text, ha="center", va="center",
                        fontsize=cell_fontsize, color=text_color, fontweight="bold")

    ax.set_xticks(range(len(col_labels)))
    ax.set_yticks(range(len(row_labels)))
    ax.set_xticklabels(col_labels, rotation=35, ha="right", fontsize=tick_fontsize)
    if show_ylabels:
        ax.set_yticklabels(row_labels, fontsize=tick_fontsize)
        if ylabels_right:
            ax.yaxis.tick_right()
            ax.yaxis.set_label_position("right")
    else:
        ax.set_yticklabels([])
    ax.tick_params(which="both", bottom=False, left=False, right=False)
    ax.grid(False)


def plot_diff_with_baseline(cells, metric, out_path: Path, note=None, vlim=None):
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["axes.unicode_minus"] = False

    rows = cols = GAME_ORDER
    row_labels = [ABBR_TO_GAME[r].capitalize() for r in rows]
    col_labels = [ABBR_TO_GAME[c].capitalize() for c in cols]

    base_mat = np.full((len(rows), 1), np.nan)
    base_std = np.full((len(rows), 1), np.nan)
    for ri, target in enumerate(rows):
        b = cells.get((BASELINE_KEY, target))
        if b:
            base_mat[ri, 0], base_std[ri, 0] = b["mean"], b["std"]

    mat = np.full((len(rows), len(cols)), np.nan)
    std_mat = np.full((len(rows), len(cols)), np.nan)
    for ri, target in enumerate(rows):
        base = cells.get((BASELINE_KEY, target))
        if not base:
            continue
        for ci, source in enumerate(cols):
            if source == target:
                continue
            stat = cells.get((source, target))
            if not stat:
                continue
            common = sorted(set(stat["seed_means"]) & set(base["seed_means"]))
            if not common:
                continue
            paired = [stat["seed_means"][s] - base["seed_means"][s] for s in common]
            mat[ri, ci] = sum(paired) / len(paired)
            std_mat[ri, ci] = _safe_std(paired)

    valid = mat[~np.isnan(mat)]
    vmax = float(max(abs(valid.max()), abs(valid.min()))) if valid.size else 1.0
    color_vmax = float(vlim) if vlim is not None else vmax
    metric_label = METRIC_DISPLAY.get(metric, metric)

    fig = plt.figure(figsize=(0.70 * len(cols) + 1.9 - 0.7, 0.6* len(rows) + 1.1))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, len(cols)], wspace=0.02)
    ax_base = fig.add_subplot(gs[0, 0])
    ax_diff = fig.add_subplot(gs[0, 1])

    limfontsize=18

    _draw_heatmap(ax_base, base_mat, row_labels, ["Target-only"], cmap="Greys",
                  fmt="{:.2f}", std_mat=base_std, std_fmt="{:.2f}",
                  show_ylabels=True, no_fill=True, tick_fontsize=limfontsize)
    # ref 를 매우 크게 → 모든 셀 글씨 검정.
    _draw_heatmap(ax_diff, mat, row_labels, col_labels, cmap=_muted_cmap("coolwarm_r"), signed=True,
                  fmt="{:.2f}", vmin=-color_vmax, vmax=color_vmax, ref=1e9,
                  std_mat=std_mat, std_fmt="{:.2f}", show_ylabels=True, ylabels_right=True,
                  tick_fontsize=limfontsize)

    ax_diff.set_xlabel("(+) Source Domain", fontsize=limfontsize+2)
    ax_diff.set_ylabel("Target Domain", fontsize=limfontsize+2, rotation=270, labelpad=18)
    ax_diff.set_yticklabels([])

    # (화살표 열 제거됨 — base 와 diff 를 바로 인접 배치)
    # if note:
    #     fig.text(0.5, -0.06, "Note: " + note, ha="center", va="top", fontsize=11,
    #              transform=fig.transFigure)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"saved: {out_path}")


def plot_diff_only(cells, metric, out_path: Path, note=None, vlim=None):
    """target-only 열 없이 Δ 히트맵만 그리는 more-compact 버전."""
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["axes.unicode_minus"] = False

    rows = cols = GAME_ORDER
    row_labels = [ABBR_TO_GAME[r].capitalize() for r in rows]
    col_labels = [ABBR_TO_GAME[c].capitalize() for c in cols]

    mat = np.full((len(rows), len(cols)), np.nan)
    std_mat = np.full((len(rows), len(cols)), np.nan)
    for ri, target in enumerate(rows):
        base = cells.get((BASELINE_KEY, target))
        if not base:
            continue
        for ci, source in enumerate(cols):
            if source == target:
                continue
            stat = cells.get((source, target))
            if not stat:
                continue
            common = sorted(set(stat["seed_means"]) & set(base["seed_means"]))
            if not common:
                continue
            paired = [stat["seed_means"][s] - base["seed_means"][s] for s in common]
            mat[ri, ci] = sum(paired) / len(paired)
            std_mat[ri, ci] = _safe_std(paired)

    valid = mat[~np.isnan(mat)]
    vmax = float(max(abs(valid.max()), abs(valid.min()))) if valid.size else 1.0
    color_vmax = float(vlim) if vlim is not None else vmax

    # compact 버전과 동일한 figsize/gridspec 사용 → diff 패널의 셀 크기·간격 동일.
    # 좌측 base 열은 그리지 않고 비워두며(tight bbox 로 잘림), row 라벨은 diff 좌측에 표시.
    fig = plt.figure(figsize=(0.70 * len(cols) + 1.9 - 0.7, 0.6 * len(rows) + 1.1))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, len(cols)], wspace=0.02)
    ax_diff = fig.add_subplot(gs[0, 1])

    limfontsize = 18

    # 게임명(행 tick)은 좌측 유지, 축 라벨만 우측으로.
    _draw_heatmap(ax_diff, mat, row_labels, col_labels, cmap=_muted_cmap("coolwarm_r"), signed=True,
                  fmt="{:.2f}", vmin=-color_vmax, vmax=color_vmax, ref=1e9,
                  std_mat=std_mat, std_fmt="{:.2f}", show_ylabels=True, ylabels_right=False,
                  tick_fontsize=limfontsize)

    ax_diff.set_xlabel("(+) Source Domain", fontsize=limfontsize)
    ax_diff.set_ylabel("Target Domain", fontsize=limfontsize, rotation=270, labelpad=18)
    ax_diff.yaxis.set_label_position("right")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"saved: {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=str(_DEFAULT_INPUT), help="eval 프로젝트 디렉토리")
    ap.add_argument("--out", default=str(_DEFAULT_OUT), help="출력 디렉토리")
    ap.add_argument("--metric", default="progress", help="metric (기본: progress)")
    ap.add_argument("--vlim", type=float, default=8.0, help="Δ 컬러 대칭 상한 ±vlim (기본: 8)")
    args = ap.parse_args()

    input_root = Path(args.input)
    if not input_root.is_dir():
        raise FileNotFoundError(f"input not found: {input_root}")
    out_root = Path(args.out)

    cells = build_overall_cells(collect_raw(input_root, args.metric))
    for ext in ("png", "pdf"):
        plot_diff_with_baseline(cells, args.metric,
                                out_root / f"source_target_diff_compact.{ext}",
                                note=_NOTE, vlim=args.vlim)
        plot_diff_only(cells, args.metric,
                       out_root / f"source_target_diff_morecompact.{ext}",
                       note=_NOTE, vlim=args.vlim)


if __name__ == "__main__":
    main()

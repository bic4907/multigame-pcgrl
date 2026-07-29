"""
notebooks/threshold_boxplot.py

dataset/reward_annotations/annotation_figure.ipynb의 마지막 플롯(게임 × reward별
CUSTOM_THRESHOLD 구간을 가로 막대로 시각화하는 threshold boxplot)만 떼어낸
독립 실행 스크립트다. 노트북처럼 ann.json을 로드하고 롱테일을 제거한 뒤,
5개 reward feature 각각에 대해 게임별 값 범위를 8-bin 구간(SEQ_COLORS)으로
나눠 그린다.

데이터 소스: dataset/multigame/cache/artifacts/<game>/*.ann.json
  (annotate.py를 먼저 실행해 캐시가 존재해야 한다)

Usage:
    python notebooks/threshold_boxplot.py                    # notebooks/threshold_boxplot/ 에 저장
    python notebooks/threshold_boxplot.py --out-dir <dir>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
import matplotlib.patheffects as pe
from matplotlib.transforms import blended_transform_factory
from matplotlib.offsetbox import TextArea, HPacker, AnnotationBbox

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "dataset" / "multigame" / "cache" / "artifacts"

# ── 폰트 (Pretendard) ─────────────────────────────────────────────────────────
_PRETENDARD = Path.home() / ".local" / "share" / "fonts" / "Pretendard-Regular.otf"
if _PRETENDARD.exists():
    fm.fontManager.addfont(str(_PRETENDARD))
    plt.rcParams["font.family"] = "Pretendard"
else:
    print(f"[WARNING] Pretendard 폰트 없음: {_PRETENDARD} (기본 폰트로 대체)")
plt.rcParams["axes.unicode_minus"] = False

# ── 스타일 ────────────────────────────────────────────────────────────────────
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"]   = "white"
plt.rcParams["savefig.facecolor"] = "white"
plt.rcParams["text.color"]       = "black"
plt.rcParams["axes.labelcolor"]  = "black"
plt.rcParams["xtick.color"]      = "black"
plt.rcParams["ytick.color"]      = "black"
plt.rcParams["axes.edgecolor"]   = "black"

GAME_COLORS = {
    "dungeon": "#e6550d",
    "doom":    "#756bb1",
    "zelda":   "#31a354",
    "pokemon": "#3182bd",
    "sokoban": "#d6616b",
}

GAMES = ["doom", "pokemon",  "sokoban", "dungeon", "zelda"]

# ── 게임 × 타일 카테고리별 specific 타일 이름 (게임 이름 아래에 표기) ─────────
# interactable / hazard / collectable feature일 때만 게임별 실제 타일 이름을 붙인다.
# None = 해당 게임에 그 카테고리 타일이 없음 (표기 안 함).
TILE_NAMES = {
    "dungeon": {"interactable_count": None,     "hazard_count": "bat",     "collectable_count": "treasure"},
    "doom":    {"interactable_count": "door",   "hazard_count": "demon",   "collectable_count": "item"},
    "zelda":   {"interactable_count": "barrel", "hazard_count": "monster", "collectable_count": "ruby"},
    "pokemon": {"interactable_count": "water",  "hazard_count": "grass",   "collectable_count": "pokeball"},
    "sokoban": {"interactable_count": "box",    "hazard_count": None,      "collectable_count": None},
}

# ── task(subplot) 제목 표기 ───────────────────────────────────────────────────
TASK_TITLES = {
    "region":             "Region count",
    "path_length":        "Path length",
    "interactable_count": "Interactable tile count",
    "hazard_count":       "Hazard tile count",
    "collectable_count":  "Collectable tile count",
}

# ── task별 x축 방향 의미 (왼쪽 ↔ 오른쪽) ──────────────────────────────────────
AXIS_SEMANTICS = {
    "region":             ("Few", "Many"),
    "path_length":        ("Short", "Long"),
    "interactable_count": ("Few", "Many"),
    "hazard_count":       ("Few", "Many"),
    "collectable_count":  ("Few", "Many"),
}

# reward 체계 (5개)
REWARDS = [
    (1, "region",             "condition_0"),
    (2, "path_length",        "condition_1"),
    (3, "interactable_count", "condition_2"),
    (4, "hazard_count",       "condition_3"),
    (5, "collectable_count",  "condition_4"),
]

# ── CUSTOM_THRESHOLDS: (game_feature) → threshold 리스트 ─────────────────────
# None = 해당 (game, feature) 조합이 없음 (N/A)
# threshold 7개 → 8구간 (8분위수 기반)
CUSTOM_THRESHOLDS = {
    "dungeon_region":             [0.5,  1.5,  3.0,  4.5,  9.5,  14.5, 19.5],
    "dungeon_path_length":        [17.5, 23.5, 28.0, 32.5, 38.5, 44.5, 50.5],
    "dungeon_interactable_count": None,
    "dungeon_hazard_count":       [3.5,  6.5,  8.5,  10.5, 14.5, 18.5, 22.5],
    "dungeon_collectable_count":  [5.5,  8.5,  10.0, 11.5, 13.5, 15.5, 19.5],

    "doom_region":                [0.5,  1.5,  2.0,  2.5,  3.0,  3.5,  4.5],
    "doom_path_length":           [21.5, 23.5, 25.5, 27.5, 29.0, 30.5, 32.5],
    "doom_interactable_count":    [-0.5, 0.5,  2.0,  3.5,  5.0,  6.5,  8.5],
    "doom_hazard_count":          [0.5,  1.5,  2.5,  3.5,  4.5,  5.5,  7.5],
    "doom_collectable_count":     [0.5,  1.5,  2.0,  2.5,  4.0,  5.5,  7.5],

    "zelda_region":               [0.5,  1.5,  2.0,  2.5,  3.5,  4.5,  6.5],
    "zelda_path_length":          [14.5, 16.5, 19.0, 21.5, 22.0, 22.5, 23.5],
    "zelda_interactable_count":   [2.5,  4.5,  6.5,  8.5,  17.5, 26.5, 35.5],
    "zelda_hazard_count":         [3.5,  5.5,  8.0,  10.5, 14.5, 18.5, 22.5],
    "zelda_collectable_count":    [0.5,  1.5,  2.5,  3.5,  9.0,  14.5, 19.5],

    "pokemon_region":             [0.5,  1.5,  2.0,  2.5,  3.5,  4.5,  6.5],
    "pokemon_path_length":        [16.5, 18.5, 21.0, 23.5, 26.5, 29.5, 32.5],
    "pokemon_interactable_count": [-0.5, 0.5,  10.5, 20.5, 40.5, 60.5, 80.5],
    "pokemon_hazard_count":       [5.5,  11.5, 28.0, 44.5, 60.5, 76.5, 92.5],
    "pokemon_collectable_count":  [-0.5, 0.5,  1.5,  2.5,  4.5,  6.5,  9.5],

    "sokoban_region":             [0.5,  1.5,  2.0,  2.5,  3.0,  3.5,  4.5],
    "sokoban_path_length":        [13.5, 17.5, 19.5, 21.5, 23.5, 25.5, 29.5],
    "sokoban_interactable_count": [1.5,  3.5,  5.0,  6.5,  8.0,  9.5,  12.5],
    "sokoban_hazard_count":       None,
    "sokoban_collectable_count":  None,
}

# preprocessing.py의 LONGTAIL_CUTOFF와 동일 조건
_LONGTAIL_CUTOFF = [
    ("dungeon", "path_length",        "condition_1", 80),
    ("pokemon", "interactable_count", "condition_2", 150),
    ("pokemon", "collectable_count",  "condition_4", 29),
]


def _is_empty(vals):
    """데이터 없음 또는 모두 0인 경우 True."""
    return len(vals) == 0 or (vals == 0).all()


def _load_ann_json(cache_dir: Path, game: str) -> list:
    """게임 캐시 디렉토리에서 ann.json을 찾아 annotations 리스트를 반환한다."""
    game_dir = cache_dir / game
    if not game_dir.exists():
        return []
    ann_files = sorted(game_dir.glob("*.ann.json"))
    if not ann_files:
        return []
    data = json.loads(ann_files[-1].read_text(encoding="utf-8"))
    rows = data.get("annotations", [])
    for r in rows:
        r["game"] = game
    return rows


def load_all_df(cache_dir: Path) -> pd.DataFrame:
    """모든 게임의 ann.json을 로드하고 롱테일을 제거한 DataFrame을 반환한다."""
    dfs = {}
    for game in GAMES:
        rows = _load_ann_json(cache_dir, game)
        if rows:
            dfs[game] = pd.DataFrame(rows)
        else:
            print(f"[WARNING] {game}: ann.json 없음 (annotate.py를 먼저 실행하세요)")

    all_df = pd.concat(dfs.values(), ignore_index=True) if dfs else pd.DataFrame()

    _before = len(all_df)
    for _game, _feat, _cond_col, _cutoff in _LONGTAIL_CUTOFF:
        _mask = (
            (all_df["game"] == _game) &
            (all_df["feature_name"] == _feat) &
            (pd.to_numeric(all_df[_cond_col], errors="coerce") >= _cutoff)
        )
        all_df = all_df[~_mask]
    all_df = all_df.reset_index(drop=True)
    print(f"전체 rows: {_before} → 롱테일 제거 후: {len(all_df)}")
    return all_df


def plot_threshold_boxplot(all_df: pd.DataFrame, out_dir: Path):
    N_MAX_BINS = 8
    # cold→warm 계열 — 낮은 level(파랑)에서 높은 level(빨강)로 자연스럽게 이어진다.
    _cmap = plt.cm.coolwarm
    SEQ_COLORS = [_cmap(0.06 + 0.88 * i / (N_MAX_BINS - 1)) for i in range(N_MAX_BINS)]

    BAR_H = 0.55

    def eff_th(th):
        return (th + 0.5) if (th % 1 == 0.0) else th

    # A.1.5가 한 페이지에 들어오도록 세로 높이는 축소하되,
    # 게임 행 간 라벨 겹침을 막기 위해 subplot 내부 높이는 확보하고
    # subplot 사이 여백(hspace)을 줄여 전체 페이지 높이를 보정한다.
    fig, axes = plt.subplots(
        len(REWARDS), 1,
        figsize=(16, 2.61 * len(REWARDS)),
        gridspec_kw={"hspace": 0.6},
    )
    fig.patch.set_facecolor("white")

    for row_i, (r_enum, feat, cond_col) in enumerate(REWARDS):
        ax = axes[row_i]
        ax.set_facecolor("white")
        sub = all_df[all_df["feature_name"] == feat]

        game_ranges = {}
        for game in GAMES:
            v = pd.to_numeric(sub[sub["game"] == game][cond_col], errors="coerce").dropna().values
            ths = CUSTOM_THRESHOLDS.get(f"{game}_{feat}")
            if not _is_empty(v) and ths is not None:
                game_ranges[game] = (float(v.min()) - 0.5, float(v.max()) + 0.5)

        if not game_ranges:
            ax.axis("off")
            continue

        x_lo_global = min(lo for lo, _ in game_ranges.values())
        x_hi_global = max(hi for _, hi in game_ranges.values())
        x_pad = (x_hi_global - x_lo_global) * 0.02

        y_ticks, y_labels = [], []

        for gi, game in enumerate(GAMES):
            y_pos = len(GAMES) - 1 - gi
            vals = pd.to_numeric(sub[sub["game"] == game][cond_col], errors="coerce").dropna().values
            ths  = CUSTOM_THRESHOLDS.get(f"{game}_{feat}")

            y_ticks.append(y_pos)
            y_labels.append(game)

            if _is_empty(vals) or ths is None:
                ax.plot([x_lo_global, x_hi_global], [y_pos, y_pos],
                        color="#cccccc", linewidth=1.5, zorder=1)
                continue

            v_lo = float(vals.min()) - 0.5
            v_hi = float(vals.max()) + 0.5

            converted = [eff_th(t) for t in sorted(ths)]
            all_edges = [v_lo] + converted + [v_hi]

            color_idx = 0
            drawn_th_positions = set()

            for i in range(len(all_edges) - 1):
                lo_e = all_edges[i]
                hi_e = all_edges[i + 1]
                raw_width = hi_e - lo_e

                if raw_width <= 0:
                    color_idx += 1
                    continue

                lo_draw = max(lo_e, v_lo)
                hi_draw = min(hi_e, v_hi)

                if hi_draw > lo_draw:
                    c = SEQ_COLORS[min(color_idx, N_MAX_BINS - 1)]
                    ax.barh(y_pos, hi_draw - lo_draw, left=lo_draw,
                            height=BAR_H, color=c, linewidth=0, zorder=2)
                    if i > 0 and v_lo < lo_e < v_hi:
                        drawn_th_positions.add(lo_e)

                color_idx += 1

            for th in sorted(drawn_th_positions):
                ax.plot([th, th], [y_pos - BAR_H / 2+0.02, y_pos + BAR_H / 2-0.02],
                        color="black", linewidth=1.5, zorder=5)

            ax.add_patch(mpatches.Rectangle(
                (v_lo, y_pos - BAR_H / 2), v_hi - v_lo, BAR_H,
                linewidth=1.0, edgecolor="black", facecolor="none", zorder=6,
            ))

        ax.set_xlim(x_lo_global - x_pad, x_hi_global + x_pad)
        ax.set_ylim(-0.6, len(GAMES) - 0.4)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([""] * len(y_ticks))
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", labelsize=12, colors="black")

        # ── 게임 이름(첫 글자 대문자) + 게임별 타일 이름/N/A (축 왼쪽 바깥) ──
        ylab_trans = blended_transform_factory(ax.transAxes, ax.transData)
        for y_pos, game in zip(y_ticks, y_labels):
            g_vals = pd.to_numeric(sub[sub["game"] == game][cond_col], errors="coerce").dropna().values
            g_ths = CUSTOM_THRESHOLDS.get(f"{game}_{feat}")
            if _is_empty(g_vals) or g_ths is None:
                sublabel, sub_color = "N/A", "#999999"   # 타일 없는 데이터도 밑에 N/A로 표기
            else:
                sublabel, sub_color = TILE_NAMES.get(game, {}).get(feat), "#333333"

            # feature 라벨(있으면)을 게임명 바로 왼쪽에 고정 간격으로 붙여
            # 하나의 묶음으로 만든 뒤, 게임명 오른쪽 끝을 축 가장자리에 정렬한다.
            game_area = TextArea(
                game.capitalize(),
                textprops=dict(fontsize=15, color="black", va="baseline"),
            )
            if sublabel:
                feat_area = TextArea(
                    sublabel,
                    textprops=dict(fontsize=12, color=sub_color,
                                   style="italic", va="baseline"),
                )
                packed = HPacker(children=[feat_area, game_area],
                                 align="baseline", pad=0, sep=8)
            else:
                packed = game_area
            ax.add_artist(AnnotationBbox(
                packed, (-0.012, y_pos), xycoords=ylab_trans,
                box_alignment=(1.0, 0.5), frameon=False, pad=0,
            ))

        # 태스크(task) 이름만 bold — Pretendard-Regular뿐이라 stroke로 faux-bold 처리
        title = ax.set_title(TASK_TITLES.get(feat, feat.replace("_", " ")).title(),
                             fontsize=16, loc="left", pad=6)
        title.set_color("black")
        title.set_path_effects([pe.withStroke(linewidth=0.7, foreground="black")])
        ax.grid(axis="x", linestyle=":", alpha=0.35, zorder=0, color="gray")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_color("black")
        for lbl in ax.get_xticklabels():
            lbl.set_color("black")

        # ── x축 방향 의미 (few/many, short/long) ──
        left_lbl, right_lbl = AXIS_SEMANTICS.get(feat, ("", ""))
        if left_lbl or right_lbl:
            ax.text(0.0, -0.22, f"← {left_lbl}", transform=ax.transAxes,
                    ha="left", va="top", fontsize=15, style="italic", color="#444444")
            ax.text(1.0, -0.22, f"{right_lbl} →", transform=ax.transAxes,
                    ha="right", va="top", fontsize=15, style="italic", color="#444444")

    # ── 하단 범례 ─────────────────────────────────────────────────────────────
    legend_patches = [
        mpatches.Patch(facecolor=SEQ_COLORS[i], edgecolor="black", linewidth=0.6,
                       label=f"Level {i + 1}")
        for i in range(N_MAX_BINS)
    ]
    fig.legend(
        handles=legend_patches,
        loc="upper center",
        ncol=N_MAX_BINS,
        fontsize=13,
        frameon=True,
        facecolor="white",
        edgecolor="#aaaaaa",
        labelcolor="black",
        bbox_to_anchor=(0.5, 0.01),
        handlelength=1.8,
        handleheight=1.0,
    )
    plt.subplots_adjust(bottom=0.055)

    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "threshold_boxplot.png"
    pdf_path = out_dir / "threshold_boxplot.pdf"
    plt.savefig(png_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    print(f"저장: {png_path}")
    print(f"저장: {pdf_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR,
                        help="ann.json 캐시 디렉토리 (기본: dataset/multigame/cache/artifacts)")
    parser.add_argument("--out-dir", type=Path,
                        default=Path(__file__).resolve().parent / "threshold_boxplot",
                        help="threshold_boxplot.png/.pdf 저장 폴더 (기본: notebooks/threshold_boxplot/)")
    args = parser.parse_args()

    all_df = load_all_df(args.cache_dir)
    if len(all_df) == 0:
        print("[ERROR] 로드된 데이터가 없습니다. annotate.py를 먼저 실행하세요.")
        return
    plot_threshold_boxplot(all_df, args.out_dir)


if __name__ == "__main__":
    main()

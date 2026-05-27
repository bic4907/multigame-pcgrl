"""
visualizer/plots.py
===================
CSV(all_checkpoints.csv)로부터 시각화 플롯을 생성한다.

플롯 목록
---------
1. Scatter (Regression)  : actual vs predicted condition value
   - game × seen/unseen × reward_enum
2. Summary stats         : 콘솔 출력

사용 예시
---------
    from analysis.reward_prediction.visualizer import VisualizerConfig, run_visualize
    cfg = VisualizerConfig(csv_path=Path("..."), output_dir=Path("..."))
    run_visualize(cfg)
"""
from __future__ import annotations

import os  # noqa: F401
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple

_ROOT = Path(__file__).resolve().parents[4]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from instruct_rl.utils.log_utils import get_logger

logger = get_logger(__file__)


# ─────────────────────────────────────────────────────────
# 설정 컨테이너
# ─────────────────────────────────────────────────────────

@dataclass
class VisualizerConfig:
    """pipeline.py 에서 visualizer 단계를 구성하는 파라미터."""

    csv_path: Path = Path("results/mgpcgrl_reward_pred_csv/all_checkpoints.csv")
    output_dir: Path = Path("results/reward_decoding_plots")
    sample_n: int = 3000        # scatter plot 샘플 수 (per game per enum)
    games: List[str] = field(
        default_factory=lambda: ["dungeon", "sokoban", "zelda", "pokemon", "doom"]
    )
    game_colors: dict = field(
        default_factory=lambda: {
            "dungeon": "#4C72B0",
            "sokoban": "#DD8452",
            "zelda":   "#55A868",
            "pokemon": "#C44E52",
            "doom":    "#8172B3",
        }
    )

    # 실행 후 채워지는 결과 경로 목록
    saved_plots: List[Path] = field(default_factory=list, init=False)


# ─────────────────────────────────────────────────────────
# 공통 헬퍼
# ─────────────────────────────────────────────────────────

def _load_dataframe(csv_path: Path) -> pd.DataFrame:
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info("Loaded CSV: %s  shape=%s", csv_path, df.shape)
    return df


def _get_regression_df(df: pd.DataFrame) -> pd.DataFrame:
    """actual_condition_active != -1 인 유효 행 반환."""
    return df[df["actual_condition_active"] != -1.0].copy()


# ─────────────────────────────────────────────────────────
# 1. Scatter — unseen, reward_enum 별
# ─────────────────────────────────────────────────────────

def plot_scatter_by_game_and_split(
    df: pd.DataFrame,
    cfg: VisualizerConfig,
    split: str = "unseen",
) -> Path:
    """
    1행 × N열 (reward_enum 수) 그리드.
    컬러 = game, 점 = (actual_condition_active, pred_condition_at_actual_enum).
    """
    reward_enums = sorted(df["actual_reward_enum"].unique().tolist())
    n_enums = len(reward_enums)

    fig, axes = plt.subplots(
        1, n_enums,
        figsize=(3.0 * n_enums, 3.4),
        squeeze=False,
    )
    plt.rcParams.update({"font.size": 11, "figure.dpi": 120})
    fig.suptitle(
        f"Regression: Actual vs Predicted Condition — by Reward Enum ({split})",
        fontsize=12, y=1.02,
    )

    reg_df = _get_regression_df(df[df["game_seen_unseen"] == split])

    for col, enum_val in enumerate(reward_enums):
        ax = axes[0][col]
        sub = reg_df[reg_df["actual_reward_enum"] == enum_val]
        if len(sub) == 0:
            ax.set_visible(False)
            continue

        for game in cfg.games:
            sg = sub[sub["game_canonical"] == game]
            if len(sg) == 0:
                sg = sub[sub["game"] == game]
            if len(sg) == 0:
                continue
            if len(sg) > cfg.sample_n:
                sg = sg.sample(cfg.sample_n, random_state=42)
            ax.scatter(
                sg["actual_condition_active"],
                sg["pred_condition_at_actual_enum"],
                alpha=0.3, s=6,
                color=cfg.game_colors.get(game, "#999999"),
                label=game, rasterized=True,
            )

        actual: np.ndarray = sub["actual_condition_active"].to_numpy(dtype=float)
        pred: np.ndarray   = sub["pred_condition_at_actual_enum"].to_numpy(dtype=float)
        lo: float = float(min(float(actual.min()), float(pred.min())))
        hi: float = float(max(float(actual.max()), float(pred.max())))
        margin: float = float((hi - lo) * 0.05) or 0.05
        ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin], "k--", lw=1.0)

        corr = float(np.corrcoef(actual, pred)[0, 1]) if len(sub) > 1 else float("nan")
        mae  = float(np.mean(np.abs(actual - pred)))
        ax.set_title(f"reward_enum={enum_val}\nr={corr:.2f}, MAE={mae:.2f}", fontsize=9)
        ax.set_xlabel("Actual", fontsize=8)
        ax.set_ylabel("Predicted", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_xlim(float(lo - margin), float(hi + margin))
        ax.set_ylim(float(lo - margin), float(hi + margin))
        ax.set_aspect("equal", adjustable="box")

    handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=cfg.game_colors.get(g, "#999999"),
                   markersize=8, label=g)
        for g in cfg.games
    ]
    fig.legend(handles=handles, title="Game", fontsize=9, title_fontsize=9,
               loc="lower center", ncol=len(cfg.games),
               bbox_to_anchor=(0.5, -0.12), framealpha=0.8)
    plt.tight_layout(rect=(0.0, 0.7, 1.0, 0.95))
    plt.subplots_adjust(wspace=0.25)

    out = cfg.output_dir / f"scatter_by_game_{split}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.close()
    logger.info("Saved plot: %s", out)
    return out


# ─────────────────────────────────────────────────────────
# 2. Accuracy bar chart — seen vs unseen per game
# ─────────────────────────────────────────────────────────

def plot_accuracy_by_game_split(df: pd.DataFrame, cfg: VisualizerConfig) -> Path:
    """게임별 · seen/unseen 별 reward_enum 분류 정확도 bar chart."""
    records = []
    for game in cfg.games:
        for split in ("seen", "unseen"):
            sub = df[(df["game_canonical"] == game) & (df["game_seen_unseen"] == split)]
            if len(sub) == 0:
                sub = df[(df["game"] == game) & (df["game_seen_unseen"] == split)]
            if len(sub) == 0:
                continue
            acc = float(sub["reward_enum_match"].mean())
            records.append({"game": game, "split": split, "accuracy": acc, "n": len(sub)})

    if not records:
        logger.warning("No data for accuracy bar chart.")
        out = cfg.output_dir / "accuracy_by_game_split.png"
        return out

    plot_df = pd.DataFrame(records)
    games_present = [g for g in cfg.games if g in plot_df["game"].values]
    x = np.arange(len(games_present))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(6, 2 * len(games_present)), 4))
    for i, split in enumerate(["seen", "unseen"]):
        vals = [
            plot_df.loc[(plot_df["game"] == g) & (plot_df["split"] == split), "accuracy"].values[0]
            if len(plot_df.loc[(plot_df["game"] == g) & (plot_df["split"] == split)]) > 0
            else 0.0
            for g in games_present
        ]
        bars = ax.bar(x + i * width, vals, width, label=split,
                      color=["#4C72B0", "#DD8452"][i], alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(games_present)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Reward Enum Accuracy")
    ax.set_title("Reward Enum Classification Accuracy — Seen vs Unseen")
    ax.legend(title="Split")
    plt.tight_layout()

    out = cfg.output_dir / "accuracy_by_game_split.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.close()
    logger.info("Saved plot: %s", out)
    return out


# ─────────────────────────────────────────────────────────
# 3. Summary stats (콘솔 출력)
# ─────────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame, games: Optional[List[str]] = None) -> None:
    if games is None:
        games = sorted(df["game_canonical"].dropna().unique().tolist()
                       if "game_canonical" in df.columns
                       else df["game"].dropna().unique().tolist())

    print("\n" + "=" * 70)
    print("REGRESSION SUMMARY  (actual_condition_active vs pred_condition_at_actual_enum)")
    print("=" * 70)
    reg = _get_regression_df(df)

    for game in games:
        for split in ("seen", "unseen"):
            col = "game_canonical" if "game_canonical" in df.columns else "game"
            sub = reg[(reg[col] == game) & (reg["game_seen_unseen"] == split)]
            if len(sub) == 0:
                continue
            a = sub["actual_condition_active"].values
            p = sub["pred_condition_at_actual_enum"].values
            corr = np.corrcoef(a, p)[0, 1] if len(sub) > 1 else float("nan")
            mae  = float(np.mean(np.abs(a - p)))
            rmse = float(np.sqrt(np.mean((a - p) ** 2)))
            print(f"  {game:10s} [{split:6s}]  n={len(sub):7,d}  r={corr:+.3f}  MAE={mae:.3f}  RMSE={rmse:.3f}")

    print("\nCLASSIFICATION SUMMARY  (reward_enum accuracy)")
    print("-" * 70)
    for game in games:
        for split in ("seen", "unseen"):
            col = "game_canonical" if "game_canonical" in df.columns else "game"
            sub = df[(df[col] == game) & (df["game_seen_unseen"] == split)]
            if len(sub) == 0:
                continue
            acc = float(sub["reward_enum_match"].mean())
            print(f"  {game:10s} [{split:6s}]  n={len(sub):7,d}  accuracy={acc:.3f}")


# ─────────────────────────────────────────────────────────
# 메인 visualize 실행 함수
# ─────────────────────────────────────────────────────────

def run_visualize(cfg: VisualizerConfig) -> VisualizerConfig:
    """
    VisualizerConfig 를 받아 전체 시각화 파이프라인을 실행하고
    saved_plots 가 채워진 cfg를 반환한다.
    """
    df = _load_dataframe(cfg.csv_path)
    print_summary(df, games=cfg.games)

    saved: List[Path] = []

    logger.info("[1/2] Plotting scatter (unseen) ...")
    saved.append(plot_scatter_by_game_and_split(df, cfg, split="unseen"))

    logger.info("[2/2] Plotting accuracy bar chart ...")
    saved.append(plot_accuracy_by_game_split(df, cfg))

    cfg.saved_plots = saved
    logger.info("Visualize done. output_dir=%s  plots=%d", cfg.output_dir.resolve(), len(saved))
    return cfg


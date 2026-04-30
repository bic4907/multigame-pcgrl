"""
Rebuild game-filtered results.csv and summary.csv from ctrl_sim.csv.

Scope:
    results/results/eval/aaai27_eval_cpcgrl

Rule:
    If run folder name is cpcgrl_game-dg_..., keep only game == "dungeon".
    (Same rule is applied to other game codes: pk/sk/dm/zd)
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TARGET_DIR = PROJECT_ROOT / "results/results/eval/aaai27_eval_cpcgrl"

GAME_CODE_TO_NAME = {
    "dg": "dungeon",
    "pk": "pokemon",
    "sk": "sokoban",
    "dm": "doom",
    "zd": "zelda",
}


def iqr_mean(x: pd.Series) -> float:
    x = x.dropna()
    if x.empty:
        return float("nan")
    if len(x) < 4:
        return float(x.mean())

    q1, q3 = x.quantile(0.25), x.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        median = x.median()
        filtered = x[x == median]
    else:
        filtered = x[(x >= q1 - 1.5 * iqr) & (x <= q3 + 1.5 * iqr)]
    return float(filtered.mean()) if not filtered.empty else float(x.mean())


def parse_game_code(folder_name: str) -> str | None:
    match = re.search(r"(?:^|_)game-([a-z0-9]+)(?:_|$)", folder_name)
    if match is None:
        return None
    return match.group(1)


def rebuild_eval_dir(eval_dir: Path, target_game: str) -> bool:
    ctrl_sim_path = eval_dir / "ctrl_sim.csv"
    results_path = eval_dir / "results.csv"
    summary_path = eval_dir / "summary.csv"
    diversity_path = eval_dir / "diversity.csv"

    if not ctrl_sim_path.exists():
        print(f"[SKIP] Missing ctrl_sim.csv: {ctrl_sim_path}")
        return False

    df_ctrl_sim = pd.read_csv(ctrl_sim_path)
    if "game" not in df_ctrl_sim.columns:
        print(f"[SKIP] Missing game column: {ctrl_sim_path}")
        return False

    df_ctrl_sim = df_ctrl_sim[df_ctrl_sim["game"].astype(str) == target_game].copy()
    if df_ctrl_sim.empty:
        print(f"[SKIP] No rows for game={target_game}: {ctrl_sim_path}")
        return False

    mean_cols = [c for c in ["progress", "vit_score", "tpkldiv"] if c in df_ctrl_sim.columns]
    if not mean_cols:
        print(f"[SKIP] No metric columns in ctrl_sim.csv: {ctrl_sim_path}")
        return False

    meta_cols = [c for c in ["row_i", "game", "instruction", "reward_enum"] if c in df_ctrl_sim.columns]

    df_results = (
        df_ctrl_sim.groupby("row_i", sort=True)[mean_cols]
        .agg(iqr_mean)
        .reset_index()
    )
    meta_df = df_ctrl_sim[meta_cols].drop_duplicates(subset="row_i").reset_index(drop=True)
    df_results = meta_df.merge(df_results, on="row_i")

    if diversity_path.exists():
        diversity_df = pd.read_csv(diversity_path)
        if {"row_i", "diversity"}.issubset(diversity_df.columns):
            diversity_df = diversity_df[["row_i", "diversity"]]
            df_results = df_results.merge(diversity_df, on="row_i", how="left")

    df_results.to_csv(results_path, index=False)

    summary_metric_cols = [c for c in ["progress", "vit_score", "tpkldiv", "diversity"] if c in df_results.columns]
    if summary_metric_cols:
        df_summary = df_results[summary_metric_cols].mean().reset_index()
        df_summary.columns = ["metric", "mean"]
        df_summary.to_csv(summary_path, index=False)
    else:
        # Keep an empty file with schema for consistency.
        pd.DataFrame(columns=["metric", "mean"]).to_csv(summary_path, index=False)

    print(f"[OK] Rebuilt: {eval_dir} (game={target_game}, rows={len(df_results)})")
    return True


def main() -> None:
    if not TARGET_DIR.exists():
        raise FileNotFoundError(f"TARGET_DIR not found: {TARGET_DIR}")

    if TARGET_DIR.name != "aaai27_eval_cpcgrl":
        raise ValueError(f"This script only targets aaai27_eval_cpcgrl: {TARGET_DIR}")

    n_done = 0
    run_dirs = sorted(p for p in TARGET_DIR.iterdir() if p.is_dir() and p.name.startswith("cpcgrl_game-"))
    for run_dir in run_dirs:
        game_code = parse_game_code(run_dir.name)
        if game_code is None:
            print(f"[SKIP] Cannot parse game code: {run_dir.name}")
            continue

        target_game = GAME_CODE_TO_NAME.get(game_code)
        if target_game is None:
            print(f"[SKIP] Unknown game code: {run_dir.name}")
            continue

        eval_dirs = sorted(p.parent for p in run_dir.rglob("ctrl_sim.csv"))
        if not eval_dirs:
            print(f"[SKIP] No ctrl_sim.csv under {run_dir}")
            continue

        for eval_dir in eval_dirs:
            if rebuild_eval_dir(eval_dir, target_game):
                n_done += 1

    print(f"[DONE] Rebuilt {n_done} eval directories in {TARGET_DIR}")


if __name__ == "__main__":
    main()

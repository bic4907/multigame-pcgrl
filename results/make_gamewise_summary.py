"""
Rebuild game-filtered results.csv and summary.csv from ctrl_sim.csv.

실험(experiment)에 속한 모든 프로젝트의 run 폴더를 순회한다.
- `*_game-{code}_*` 패턴 → game code로 단일 게임 필터링 후 집계
- `*_game-all_*`  패턴 → 필터 없이 전체 게임 집계
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_HERE = Path(__file__).resolve().parent          # results/

# sys.path에 project root 추가 (instruct_rl 임포트용)
if str(_HERE.parent) not in sys.path:
    sys.path.append(str(_HERE.parent))

from instruct_rl.utils.log_utils import get_logger  # noqa: E402

logger = get_logger(__name__)

def _load_cfg() -> dict:
    cfg_path = _HERE / "config.json"
    if cfg_path.is_file():
        with cfg_path.open(encoding="utf-8") as f:
            return json.load(f)
    return {}

_CFG = _load_cfg()

GAME_CODE_TO_NAME: dict[str, str] = _CFG.get("games", {}).get("code_to_name", {
    "dg": "dungeon",
    "pk": "pokemon",
    "sk": "sokoban",
    "dm": "doom",
    "zd": "zelda",
})


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
        return False

    df_ctrl_sim = pd.read_csv(ctrl_sim_path)
    if "game" not in df_ctrl_sim.columns:
        return False

    df_ctrl_sim = df_ctrl_sim[df_ctrl_sim["game"].astype(str) == target_game].copy()
    if df_ctrl_sim.empty:
        return False

    mean_cols = [c for c in ["progress", "vit_score", "tpkldiv"] if c in df_ctrl_sim.columns]
    if not mean_cols:
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
            df_results = df_results.merge(diversity_df[["row_i", "diversity"]], on="row_i", how="left")

    df_results.to_csv(results_path, index=False)

    summary_metric_cols = [c for c in ["progress", "vit_score", "tpkldiv", "diversity"] if c in df_results.columns]
    if summary_metric_cols:
        df_summary = df_results[summary_metric_cols].mean().reset_index()
        df_summary.columns = ["metric", "mean"]
        df_summary.to_csv(summary_path, index=False)
    else:
        pd.DataFrame(columns=["metric", "mean"]).to_csv(summary_path, index=False)

    return True


def rebuild_eval_dir_all_games(eval_dir: Path) -> bool:
    """game-all 런: 게임 필터 없이 전체 행으로 results.csv / summary.csv 재생성."""
    ctrl_sim_path = eval_dir / "ctrl_sim.csv"
    results_path  = eval_dir / "results.csv"
    summary_path  = eval_dir / "summary.csv"
    diversity_path = eval_dir / "diversity.csv"

    if not ctrl_sim_path.exists():
        return False

    df = pd.read_csv(ctrl_sim_path)
    mean_cols = [c for c in ["progress", "vit_score", "tpkldiv"] if c in df.columns]
    if not mean_cols:
        return False

    meta_cols = [c for c in ["row_i", "game", "instruction", "reward_enum"] if c in df.columns]
    df_results = (
        df.groupby("row_i", sort=True)[mean_cols]
        .agg(iqr_mean)
        .reset_index()
    )
    meta_df = df[meta_cols].drop_duplicates(subset="row_i").reset_index(drop=True)
    df_results = meta_df.merge(df_results, on="row_i")

    if diversity_path.exists():
        div_df = pd.read_csv(diversity_path)
        if {"row_i", "diversity"}.issubset(div_df.columns):
            df_results = df_results.merge(div_df[["row_i", "diversity"]], on="row_i", how="left")

    df_results.to_csv(results_path, index=False)

    summary_cols = [c for c in ["progress", "vit_score", "tpkldiv", "diversity"] if c in df_results.columns]
    if summary_cols:
        df_summary = df_results[summary_cols].mean().reset_index()
        df_summary.columns = ["metric", "mean"]
        df_summary.to_csv(summary_path, index=False)
    else:
        pd.DataFrame(columns=["metric", "mean"]).to_csv(summary_path, index=False)

    return True


def _get_project_dirs(input_root: Path, experiment: str | None) -> list[Path]:
    """experiment에 속한 프로젝트 디렉토리 목록을 반환한다."""
    if experiment:
        projects = _CFG.get("experiments", {}).get(experiment, {}).get("target_projects", [])
    else:
        default = _CFG.get("paths", {}).get("gamewise_target_dir", "wandb_projects/aaai27_eval_cpcgrl")
        projects = [Path(default).name]

    dirs = []
    for proj in projects:
        d = input_root / proj
        if d.is_dir():
            dirs.append(d)
        else:
            logger.warning("project dir not found: %s", d)
    return dirs


def _rebuild_project(project_dir: Path) -> int:
    """프로젝트 디렉토리 안의 모든 run을 처리하고 rebuild 횟수를 반환한다."""
    tasks: list[tuple[Path, str, list[Path]]] = []
    for run_dir in sorted(p for p in project_dir.iterdir() if p.is_dir() and re.search(r"_game-", p.name)):
        game_code = parse_game_code(run_dir.name)
        if game_code is None:
            continue
        eval_dirs = sorted(p.parent for p in run_dir.rglob("ctrl_sim.csv"))
        if eval_dirs:
            tasks.append((run_dir, game_code, eval_dirs))

    n_done = 0
    desc = f"rebuild  {project_dir.name}"
    with tqdm(total=len(tasks), desc=desc, unit="run", leave=True) as pbar:
        for run_dir, game_code, eval_dirs in tasks:
            pbar.set_postfix_str(run_dir.name, refresh=False)
            if game_code == "all":
                for eval_dir in eval_dirs:
                    if rebuild_eval_dir_all_games(eval_dir):
                        n_done += 1
            else:
                target_game = GAME_CODE_TO_NAME.get(game_code)
                if target_game is None:
                    pbar.update(1)
                    continue
                for eval_dir in eval_dirs:
                    if rebuild_eval_dir(eval_dir, target_game):
                        n_done += 1
            pbar.update(1)

    return n_done


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ctrl_sim.csv → results.csv / summary.csv 재생성"
    )
    parser.add_argument(
        "--input", "--input-root",
        dest="input",
        default=_CFG.get("paths", {}).get("eval_output", "wandb_projects"),
        help="다운로드 결과 루트 경로 (기본값: config.json paths.eval_output)",
    )
    parser.add_argument(
        "--experiment",
        default=None,
        help="처리할 experiment 이름 (미지정 시 config.json 기본 프로젝트 사용)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    raw = Path(args.input)
    input_root = raw if raw.is_absolute() else (_HERE / raw).resolve()
    if not input_root.exists():
        raise FileNotFoundError(f"input root not found: {input_root}")

    project_dirs = _get_project_dirs(input_root, args.experiment)
    if not project_dirs:
        logger.warning("No project dirs found for experiment=%r", args.experiment)
        return

    total = 0
    for project_dir in project_dirs:
        n = _rebuild_project(project_dir)
        total += n
    logger.info("rebuilt %d eval directories total", total)


if __name__ == "__main__":
    main()

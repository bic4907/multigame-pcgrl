"""
ctrl_sim.csv → results.csv / summary.csv 생성

실험(experiment)에 속한 모든 프로젝트의 run 폴더를 순회하며
ctrl_sim.csv를 IQR mean 집계해 results.csv / summary.csv를 생성한다.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_HERE = Path(__file__).resolve().parent          # results/

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


def process_eval_dir(eval_dir: Path) -> bool:
    """ctrl_sim.csv → results.csv / summary.csv 생성 (게임 필터 없이 전체 집계)."""
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
        projects = []

    dirs = []
    for proj in projects:
        d = input_root / proj
        if d.is_dir():
            dirs.append(d)
        else:
            logger.warning("project dir not found: %s", d)
    return dirs


def _process_project(project_dir: Path, workers: int = 4) -> int:
    """프로젝트 디렉토리 안의 모든 eval dir을 병렬 처리하고 성공 횟수를 반환한다."""
    eval_dirs = sorted({p.parent for p in project_dir.rglob("ctrl_sim.csv")})
    if not eval_dirs:
        logger.warning("ctrl_sim.csv not found under: %s", project_dir)
        return 0

    n_done = 0
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_eval_dir, d): d for d in eval_dirs}
        with tqdm(total=len(eval_dirs), desc=project_dir.name, unit="eval", leave=True) as pbar:
            for fut in as_completed(futures):
                d = futures[fut]
                pbar.set_postfix_str(d.name, refresh=False)
                try:
                    if fut.result():
                        n_done += 1
                except Exception as e:
                    logger.error("Error processing %s: %s", d, e)
                pbar.update(1)

    return n_done


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ctrl_sim.csv → results.csv / summary.csv 생성"
    )
    parser.add_argument(
        "--input", "--input-root",
        dest="input",
        default="wandb_projects",
        help="다운로드 결과 루트 경로 (기본값: wandb_projects)",
    )
    parser.add_argument(
        "--experiment",
        default=None,
        help="처리할 experiment 이름 (미지정 시 config.json 기본 프로젝트 사용)",
    )
    parser.add_argument(
        "--workers", "-j",
        type=int,
        default=min(8, os.cpu_count() or 4),
        help="병렬 worker 수 (기본값: min(8, cpu_count))",
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
        n = _process_project(project_dir, workers=args.workers)
        total += n
    logger.info("processed %d eval directories total", total)


if __name__ == "__main__":
    main()

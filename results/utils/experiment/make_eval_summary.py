"""
ctrl_sim.csv → results.csv / summary.csv 생성

실험(experiment)에 속한 모든 프로젝트의 eval 폴더를 순회하며
ctrl_sim.csv를 reward_enum 단위로 IQR mean 집계해 results.csv / summary.csv를 생성한다.
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm import tqdm

_HERE        = Path(__file__).resolve().parent   # results/utils/experiment/
_RESULTS_DIR = _HERE.parent.parent               # results/
_ROOT        = _HERE.parent.parent.parent        # project root
if str(_RESULTS_DIR) not in sys.path:
    sys.path.insert(0, str(_RESULTS_DIR))
if str(_ROOT) not in sys.path:
    sys.path.append(str(_ROOT))

from instruct_rl.utils.log_utils import get_logger
from utils.core.run_output import load_cfg
from utils.core.stats import iqr_mean

logger = get_logger(__name__)
_CFG = load_cfg()


def _iqr_mean_groupby(df: pd.DataFrame, group_col: str, value_cols: list[str]) -> pd.DataFrame:
    """그룹별 IQR-filtered mean — 벡터화 버전.

    기존 ``df.groupby(group_col)[value_cols].agg(iqr_mean)`` 와 동일한 결과를
    Python 함수 호출 없이 pandas/NumPy 연산만으로 계산한다.

    동작:
      1. 그룹별 q1, q3 를 transform 으로 broadcast
      2. [q1 - 1.5·IQR, q3 + 1.5·IQR] 밖의 값을 NaN 마스킹
      3. 그룹별 mean (NaN 자동 무시)
      4. 마스킹 후 그룹 평균이 NaN(전부 outlier) 이면 원본 평균으로 fallback
    """
    grp_vals = df.groupby(group_col, sort=True)[value_cols]
    q1 = grp_vals.transform("quantile", 0.25)
    q3 = grp_vals.transform("quantile", 0.75)
    iqr = q3 - q1
    lo  = q1 - 1.5 * iqr
    hi  = q3 + 1.5 * iqr
    vals = df[value_cols]
    mask = (vals.ge(lo)) & (vals.le(hi))
    filtered = vals.where(mask)
    result = filtered.groupby(df[group_col], sort=True).mean()
    # fallback: 필터 결과가 모두 NaN 인 그룹은 원본 평균으로 채움
    plain  = vals.groupby(df[group_col], sort=True).mean()
    return result.fillna(plain).reset_index()


def process_eval_dir(eval_dir: Path) -> bool:
    """ctrl_sim.csv → results.csv / summary.csv 생성 (게임 필터 없이 전체 집계)."""
    ctrl_sim_path  = eval_dir / "ctrl_sim.csv"
    results_path   = eval_dir / "results.csv"
    summary_path   = eval_dir / "summary.csv"
    diversity_path = eval_dir / "diversity.csv"

    if not ctrl_sim_path.exists():
        return False

    df = pd.read_csv(ctrl_sim_path)
    mean_cols = [c for c in ["progress", "vit_score", "tpkldiv"] if c in df.columns]
    if not mean_cols:
        return False

    meta_cols = [c for c in ["row_i", "game", "instruction", "reward_enum"] if c in df.columns]
    # 벡터화된 IQR mean — 기존 .agg(iqr_mean) 보다 10~50배 빠름
    df_results = _iqr_mean_groupby(df, "row_i", mean_cols)
    meta_df    = df[meta_cols].drop_duplicates(subset="row_i").reset_index(drop=True)
    df_results = meta_df.merge(df_results, on="row_i")

    if diversity_path.exists():
        div_df = pd.read_csv(diversity_path)
        if {"row_i", "diversity"}.issubset(div_df.columns):
            df_results = df_results.merge(div_df[["row_i", "diversity"]], on="row_i", how="left")

    # reward_enum float(0.0) → int 방지
    if "reward_enum" in df_results.columns:
        df_results["reward_enum"] = (
            pd.to_numeric(df_results["reward_enum"], errors="coerce").astype("Int64")
        )

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
    if experiment:
        projects = _CFG.get("experiments", {}).get(experiment, {}).get("target_projects", [])
    else:
        projects = []

    dirs: list[Path] = []
    for proj in projects:
        d = input_root / proj
        if d.is_dir():
            dirs.append(d)
        else:
            logger.warning("project dir not found: %s", d)
    return dirs


def _process_project(project_dir: Path, workers: int = 4) -> int:
    eval_dirs = sorted({p.parent for p in project_dir.rglob("ctrl_sim.csv")})
    if not eval_dirs:
        logger.warning("ctrl_sim.csv not found under: %s", project_dir)
        return 0

    n_done = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
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
    parser = argparse.ArgumentParser(description="ctrl_sim.csv → results.csv / summary.csv 생성")
    parser.add_argument("--input", "--input-root", dest="input",
                        default="wandb_projects",
                        help="다운로드 결과 루트 경로 (기본값: wandb_projects)")
    parser.add_argument("--experiment", default=None,
                        help="처리할 experiment 이름")
    parser.add_argument("--workers", "-j", type=int,
                        default=os.cpu_count() or 4,
                        help="병렬 worker 수 (기본값: cpu_count)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    raw        = Path(args.input)
    if raw.is_absolute():
        input_root = raw.resolve()
    else:
        # 상대 경로는 results/ 기준으로 먼저, 없으면 프로젝트 루트 기준으로 시도
        candidate_results = (_RESULTS_DIR / raw).resolve()
        candidate_root    = (_ROOT / raw).resolve()
        if candidate_results.exists():
            input_root = candidate_results
        elif candidate_root.exists():
            input_root = candidate_root
        else:
            input_root = candidate_results   # 에러 메시지에 results/ 기준 경로 표시
    if not input_root.exists():
        raise FileNotFoundError(f"input root not found: {input_root}")

    project_dirs = _get_project_dirs(input_root, args.experiment)
    if not project_dirs:
        logger.warning("No project dirs found for experiment=%r", args.experiment)
        return

    total = 0
    for project_dir in project_dirs:
        n = _process_project(project_dir, workers=args.workers)
        logger.debug("project %s: %d eval dirs processed", project_dir.name, n)
        total += n
    logger.info("done — %d eval directories processed total", total)


if __name__ == "__main__":
    main()

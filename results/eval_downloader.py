"""
eval_downloader.py
==================
runner.py 에서 업로드한 eval 결과물을 W&B에서 로컬로 다운받는 스크립트.

업로드된 artifact 종류
  - eval_csv  (type=dataset): ctrl_sim.csv / results.csv / diversity.csv / summary.csv
  - eval_h5_* (type=dataset): eval.h5

사용 예
-------
    python sweep/wandb_sweep/eval_downloader.py
    python sweep/wandb_sweep/eval_downloader.py --no-h5    # h5 생략
    python sweep/wandb_sweep/eval_downloader.py --output results/eval_download
    python sweep/wandb_sweep/eval_downloader.py --finished-only --workers 4
"""

import argparse
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from tqdm import tqdm

# ── 프로젝트 루트를 sys.path에 추가 ──────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from sweep.wandb_utils.config import DEFAULT_ENTITY, DEFAULT_NUM_WORKERS
from sweep.wandb_utils.downloader import get_api

# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.WARNING,   # tqdm 진행 중에는 WARNING 이상만 출력
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class _RunResult:
    run_name: str
    status: str          # "ok" | "skipped" | "error"
    error: str = ""


@dataclass
class _ProjectSummary:
    project: str
    ok: int = 0
    skipped: int = 0
    errors: list[_RunResult] = field(default_factory=list)

# 다운받을 프로젝트 목록
TARGET_PROJECTS = [
    "aaai27_eval_cpcgrl",
    "aaai27_eval_cpcgrl_vanilla",
    "aaai27_eval_cpcgrl_spread",
]

# ---------------------------------------------------------------------------
# 단일 run 다운로드
# ---------------------------------------------------------------------------


def _download_run(
    run,
    output_dir: str,
    download_csv: bool = True,
    download_h5: bool = True,
    skip_if_exists: bool = True,
) -> _RunResult:
    """단일 W&B run 에서 eval artifact 를 다운로드한다.

    Returns
    -------
    _RunResult  with status "ok" | "skipped" | "error"
    """
    # run 별 저장 폴더: exp_dir basename 또는 run.id
    folder_name = os.path.basename(run.config.get("exp_dir", None) or run.id)
    run_dir = os.path.join(output_dir, folder_name)

    # ── 스킵 체크 ─────────────────────────────────────────────────────────
    if skip_if_exists:
        ctrl_sim = os.path.join(run_dir, "ctrl_sim.csv")
        results  = os.path.join(run_dir, "results.csv")
        if os.path.isfile(ctrl_sim) and os.path.isfile(results):
            return _RunResult(run_name=run.name, status="skipped")

    os.makedirs(run_dir, exist_ok=True)

    # ── artifact 목록 조회 ────────────────────────────────────────────────
    try:
        artifacts = list(run.logged_artifacts())
    except Exception as e:
        return _RunResult(run_name=run.name, status="error", error=f"artifact 조회 실패: {e}")

    csv_artifact = None
    h5_artifacts = []

    for art in artifacts:
        if art.name.startswith("eval_csv") and art.type == "dataset":
            csv_artifact = art
        elif art.name.startswith("eval_h5") and art.type == "dataset":
            h5_artifacts.append(art)

    errors = []

    # ── eval_csv 다운로드 ─────────────────────────────────────────────────
    if download_csv:
        if csv_artifact is None:
            errors.append("eval_csv artifact 없음")
        else:
            try:
                for f in csv_artifact.files():
                    local_path = os.path.join(run_dir, f.name)
                    if skip_if_exists and os.path.isfile(local_path):
                        continue
                    f.download(root=run_dir, replace=True)
            except Exception as e:
                errors.append(f"eval_csv 오류: {e}")

    # ── eval_h5 다운로드 ──────────────────────────────────────────────────
    if download_h5 and h5_artifacts:
        latest_h5 = h5_artifacts[-1]
        h5_local = os.path.join(run_dir, "eval.h5")
        if not (skip_if_exists and os.path.isfile(h5_local)):
            try:
                for f in latest_h5.files():
                    f.download(root=run_dir, replace=True)
            except Exception as e:
                errors.append(f"eval_h5 오류: {e}")

    if errors:
        return _RunResult(run_name=run.name, status="error", error=" | ".join(errors))
    return _RunResult(run_name=run.name, status="ok")


# ---------------------------------------------------------------------------
# 프로젝트 전체 다운로드
# ---------------------------------------------------------------------------


def download_eval_project(
    project: str,
    entity: str = DEFAULT_ENTITY,
    output_dir: str = "results/eval",
    download_csv: bool = True,
    download_h5: bool = True,
    skip_if_exists: bool = True,
    n_workers: int = DEFAULT_NUM_WORKERS,
    filters: dict | None = None,
    per_page: int = 200,
) -> _ProjectSummary:
    """프로젝트의 모든 run 에서 eval artifact 를 다운로드한다."""
    api = get_api()
    proj_output_dir = os.path.join(output_dir, project)
    os.makedirs(proj_output_dir, exist_ok=True)

    runs = list(api.runs(f"{entity}/{project}", filters=filters or {}, per_page=per_page))
    summary = _ProjectSummary(project=project)

    def _worker(run):
        return _download_run(
            run,
            output_dir=proj_output_dir,
            download_csv=download_csv,
            download_h5=download_h5,
            skip_if_exists=skip_if_exists,
        )

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_worker, r): r.name for r in runs}
        with tqdm(total=len(futures), desc=project, unit="run") as pbar:
            for fut in as_completed(futures):
                try:
                    res: _RunResult = fut.result()
                except Exception as e:
                    res = _RunResult(run_name=futures[fut], status="error", error=str(e))

                if res.status == "ok":
                    summary.ok += 1
                elif res.status == "skipped":
                    summary.skipped += 1
                else:
                    summary.errors.append(res)

                pbar.update(1)

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="W&B eval artifact 다운로더 (eval_csv / eval_h5)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python sweep/wandb_sweep/eval_downloader.py
  python sweep/wandb_sweep/eval_downloader.py --no-h5
  python sweep/wandb_sweep/eval_downloader.py --output results/my_eval --finished-only
  python sweep/wandb_sweep/eval_downloader.py --projects aaai27_eval_cpcgrl --workers 4
        """,
    )
    parser.add_argument(
        "--projects",
        nargs="+",
        default=TARGET_PROJECTS,
        metavar="PROJECT",
        help=f"다운로드할 W&B 프로젝트 목록 (기본: {TARGET_PROJECTS})",
    )
    parser.add_argument(
        "--entity",
        default=DEFAULT_ENTITY,
        help=f"W&B 엔티티 (기본: {DEFAULT_ENTITY})",
    )
    parser.add_argument(
        "--output",
        default="results/eval",
        help="로컬 저장 루트 경로 (기본: results/eval)",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="eval_csv artifact(ctrl_sim/results/diversity/summary CSV) 다운로드 생략",
    )
    parser.add_argument(
        "--no-h5",
        action="store_true",
        help="eval_h5 artifact(eval.h5) 다운로드 생략",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="이미 존재하는 파일도 덮어쓰기",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
        help=f"병렬 다운로드 스레드 수 (기본: {DEFAULT_NUM_WORKERS})",
    )
    parser.add_argument(
        "--finished-only",
        action="store_true",
        help="state=finished 인 run 만 다운로드",
    )
    return parser.parse_args()


def _print_summary(summaries: list[_ProjectSummary]) -> None:
    """프로젝트별 결과를 표 형식으로 출력한다."""
    COL_W = [40, 8, 10, 8]
    headers = ["Project", "OK", "Skipped", "Error"]
    sep = "+" + "+".join("-" * (w + 2) for w in COL_W) + "+"
    fmt = "| " + " | ".join(f"{{:<{w}}}" for w in COL_W) + " |"

    print("\n" + sep)
    print(fmt.format(*headers))
    print(sep)
    for s in summaries:
        print(fmt.format(s.project, s.ok, s.skipped, len(s.errors)))
    print(sep)

    # 에러 상세
    any_error = any(s.errors for s in summaries)
    if any_error:
        print("\n[!] 에러 상세:")
        for s in summaries:
            for r in s.errors:
                print(f"  [{s.project}] {r.run_name} — {r.error}")
    print()


def main():
    args = parse_args()

    filters = {"state": "finished"} if args.finished_only else None

    summaries: list[_ProjectSummary] = []
    for project in args.projects:
        summary = download_eval_project(
            project=project,
            entity=args.entity,
            output_dir=args.output,
            download_csv=not args.no_csv,
            download_h5=not args.no_h5,
            skip_if_exists=not args.force,
            n_workers=args.workers,
            filters=filters,
        )
        summaries.append(summary)

    _print_summary(summaries)


if __name__ == "__main__":
    main()


"""
process_shared.py
=================
experiment별 결과 처리 스크립트(process_allseen.py,
process_unseen_generalizability.py)가 공유하는 유틸리티.

직접 실행하지 말 것.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import logging
import sys
import time
import traceback
from pathlib import Path

_HERE = Path(__file__).resolve().parent          # results/
_ROOT = _HERE.parent                             # project root

if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
if str(str(_ROOT)) not in sys.path:
    sys.path.append(str(_ROOT))

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_cfg() -> dict:
    cfg_path = _HERE / "config.json"
    if cfg_path.is_file():
        with cfg_path.open(encoding="utf-8") as f:
            return json.load(f)
    return {}


def get_experiment_projects(experiment: str, cfg: dict | None = None) -> list[str]:
    """config.json의 experiments.<experiment>.target_projects 반환."""
    cfg = cfg or load_cfg()
    return cfg.get("experiments", {}).get(experiment, {}).get("target_projects", [])


def get_experiment_names(cfg: dict | None = None) -> list[str]:
    cfg = cfg or load_cfg()
    return list(cfg.get("experiments", {}).keys())


# ---------------------------------------------------------------------------
# Step runner (sys.argv 패치 후 script의 main() 직접 호출)
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _patch_argv(script_path: Path, extra_args: list[str]):
    old = sys.argv[:]
    sys.argv = [str(script_path)] + extra_args
    try:
        yield
    finally:
        sys.argv = old


def run_processing_step(
    script_path: Path,
    extra_args: list[str],
    log: logging.Logger,
) -> bool:
    """
    script_path의 main()을 extra_args와 함께 직접 호출한다.
    성공 시 True, 실패 시 False 반환.
    """
    module_name = script_path.stem
    log.info("▶  %s  args=%s", module_name, extra_args)

    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        log.error("모듈 로드 실패: %s", script_path)
        return False

    module = importlib.util.module_from_spec(spec)
    start = time.time()
    ok = False
    with _patch_argv(script_path, extra_args):
        try:
            spec.loader.exec_module(module)
            module.main()
            ok = True
        except SystemExit as e:
            ok = (e.code in (None, 0))
            if not ok:
                log.error("%s: SystemExit(%s)", module_name, e.code)
        except Exception:
            log.error("%s 예외:\n%s", module_name, traceback.format_exc())

    elapsed = time.time() - start
    status = "OK" if ok else "FAILED"
    log.info("   %s  %.1fs", status, elapsed)
    return ok


# ---------------------------------------------------------------------------
# Processing steps 정의 (experiment 공통)
# ---------------------------------------------------------------------------

#: 결과 파일 처리 단계 목록 (순서 중요)
PROCESSING_STEPS: list[dict] = [
    {
        "id": 1,
        "name": "make_gamewise_summary",
        "script": _HERE / "make_gamewise_summary.py",
        "description": "ctrl_sim.csv → per-game results.csv / summary.csv",
    },
    {
        "id": 2,
        "name": "build_benchmark_table",
        "script": _HERE / "build_benchmark_table.py",
        "description": "summary.csv → Markdown/CSV 벤치마크 테이블 + 플롯",
    },
    {
        "id": 3,
        "name": "condition_progress_report",
        "script": _HERE / "condition_progress_report.py",
        "description": "ctrl_sim.csv → condition vs metric 플롯 + Markdown 리포트",
    },
]


# ---------------------------------------------------------------------------
# 공통 argparse 빌더
# ---------------------------------------------------------------------------

def build_parser(experiment: str) -> argparse.ArgumentParser:

    cfg = load_cfg()
    _exp_names = get_experiment_names(cfg)

    parser = argparse.ArgumentParser(
        description=f"결과 파일 처리 파이프라인 — experiment: {experiment}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
단계:
  1  make_gamewise_summary    ctrl_sim.csv → results.csv / summary.csv
  2  build_benchmark_table    summary.csv → 테이블 + 플롯
  3  condition_progress_report ctrl_sim.csv → 조건 플롯 + 리포트

사용 예:
  python results/process_{experiment}.py
  python results/process_{experiment}.py --steps 2 3
  python results/process_{experiment}.py --input wandb_projects --dry-run
        """,
    )
    parser.add_argument(
        "--input",
        default=cfg.get("paths", {}).get("eval_output", "wandb_projects"),
        help="다운로드된 결과 파일의 루트 경로 (기본값: config.json paths.eval_output)",
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        type=int,
        choices=[s["id"] for s in PROCESSING_STEPS],
        default=None,
        metavar="N",
        help="실행할 단계 번호 (기본값: 전체). 예: --steps 2 3",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="실제 실행 없이 실행될 단계만 출력",
    )
    parser.add_argument(
        "--continue-on-failure",
        action="store_true",
        help="단계 실패 시 중단하지 않고 계속 진행",
    )
    return parser


# ---------------------------------------------------------------------------
# 공통 실행 진입점
# ---------------------------------------------------------------------------

def run_experiment_processing(experiment: str) -> None:
    """
    experiment에 맞는 처리 단계를 실행한다.
    각 실험 스크립트의 main()에서 이 함수를 호출한다.
    """
    from instruct_rl.utils.log_utils import get_logger

    log = get_logger(f"process_{experiment}")

    parser = build_parser(experiment)
    args = parser.parse_args()

    selected_ids = set(args.steps) if args.steps else {s["id"] for s in PROCESSING_STEPS}
    selected_steps = [s for s in PROCESSING_STEPS if s["id"] in selected_ids]

    # 각 스텝에 전달할 공통 인자
    base_extra = [
        "--experiment", experiment,
        "--input", args.input,
    ]

    log.info("=== process_%s 시작 ===", experiment)
    log.info("단계: %s", [s["id"] for s in selected_steps])
    log.info("입력 경로: %s", args.input)
    if args.dry_run:
        log.info("[DRY-RUN] 실행될 단계:")
        for s in selected_steps:
            log.info("  %d  %s  —  %s", s["id"], s["name"], s["description"])
        return

    results: list[tuple[dict, bool]] = []
    for step in selected_steps:
        ok = run_processing_step(step["script"], base_extra, log)
        results.append((step, ok))
        if not ok and not args.continue_on_failure:
            log.error("[중단] 단계 %d (%s) 실패 — 이후 단계를 건너뜁니다.", step["id"], step["name"])
            break

    log.info("=== 요약 ===")
    executed_ids = {s["id"] for s, _ in results}
    for step, ok in results:
        icon = "✓" if ok else "✗"
        tag  = "OK" if ok else "FAILED"
        log.info("  %s %d [%s] %s", icon, step["id"], tag, step["name"])
    for step in selected_steps:
        if step["id"] not in executed_ids:
            log.info("  - %d [SKIP] %s", step["id"], step["name"])

    if any(not ok for _, ok in results):
        sys.exit(1)


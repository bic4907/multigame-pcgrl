"""
run_pipeline.py — unified entry point for results/ scripts

Calls each script's main() function directly in the same process.
Aborts the pipeline on the first failure by default.

Usage:
    python results/run_pipeline.py                         # full pipeline (steps 1-5)
    python results/run_pipeline.py --experiment allseen    # allseen experiment
    python results/run_pipeline.py --experiment unseen_generalizability
    python results/run_pipeline.py --steps 3               # single step
    python results/run_pipeline.py --steps 3 4 5           # multiple steps
    python results/run_pipeline.py --continue-on-failure   # keep going after failure
    python results/run_pipeline.py --dry-run               # show steps without running
    python results/run_pipeline.py --list                  # list step descriptions

Step numbers:
    1  eval_downloader          Download eval artifacts from W&B
    2  make_eval_summary        ctrl_sim.csv → per-eval results/summary.csv
    3  benchmark               summary/results.csv → Markdown/CSV tables + plots
    4  condition_progress_report  condition vs metric plots + Markdown report
    5  reward_enum_visualizer   reward_enum representative tile-map visualization (requires eval.h5)
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import logging
import os
import sys
import time
import traceback
from pathlib import Path

_HERE = Path(__file__).resolve().parent        # results/
_ROOT = _HERE.parent                           # project root

# sys.path setup and early logger (before pipeline log file is ready)
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
if str(_ROOT) not in sys.path:
    sys.path.append(str(_ROOT))
from instruct_rl.utils.log_utils import get_logger
_log = get_logger(__file__)


def _get_experiment_names() -> list[str]:
    """config.json의 experiments 키 목록을 반환한다."""
    import json
    cfg_path = _HERE / "config.json"
    if cfg_path.is_file():
        with cfg_path.open(encoding="utf-8") as f:
            cfg = json.load(f)
        return list(cfg.get("experiments", {}).keys())
    return []

STEPS: list[dict] = [
    {
        "id": 1,
        "name": "eval_downloader",
        "script": _HERE / "eval_downloader.py",
        "description": "Download eval artifacts (CSV/H5) from W&B",
    },
    {
        "id": 2,
        "name": "make_eval_summary",
        "script": _HERE / "make_eval_summary.py",
        "description": "ctrl_sim.csv → per-eval results.csv / summary.csv",
    },
    {
        "id": 3,
        "name": "benchmark",
        "script": _HERE / "benchmark.py",
        "description": "summary/results.csv → Markdown/CSV tables + comparison plots",
    },
    {
        "id": 4,
        "name": "condition_progress_report",
        "script": _HERE / "condition_progress_report.py",
        "description": "ctrl_sim.csv → condition vs metric plots + Markdown report",
    },
    {
        "id": 5,
        "name": "reward_enum_visualizer",
        "script": _HERE / "reward_enum_visualizer.py",
        "description": "ctrl_sim.csv + eval.h5 → reward_enum tile-map visualization",
    },
]


def parse_args(default_experiment: str | None = None) -> argparse.Namespace:
    _exp_names = _get_experiment_names()
    _exp_hint = ", ".join(_exp_names) if _exp_names else "none defined"
    parser = argparse.ArgumentParser(
        description="Unified pipeline entry point for results/ scripts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
examples:
  python results/run_pipeline.py                       # run all steps (1-5)
  python results/run_pipeline.py --experiment allseen
  python results/run_pipeline.py --experiment unseen_generalizability
  python results/run_pipeline.py --steps 3             # table generation only
  python results/run_pipeline.py --steps 3 4           # table + condition report
  python results/run_pipeline.py --continue-on-failure # keep going after failures
  python results/run_pipeline.py --dry-run             # show steps without running
  python results/run_pipeline.py --list                # list step descriptions

available experiments: {_exp_hint}
        """,
    )
    parser.add_argument(
        "--experiment",
        choices=_exp_names if _exp_names else None,
        default=default_experiment,
        metavar="EXPERIMENT",
        help=(
            f"Experiment group to run (choices: {_exp_hint}). "
            "Passed as --experiment to every step script."
        ),
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        type=int,
        choices=[s["id"] for s in STEPS],
        default=None,
        metavar="N",
        help="Step numbers to run (default: all). e.g. --steps 3 4",
    )
    parser.add_argument(
        "--continue-on-failure",
        action="store_true",
        help="Continue to the next step even if the current step fails (default: abort on first failure)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the list of functions to be called without actually running them",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print step numbers and descriptions then exit",
    )
    parser.add_argument(
        "--extra-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Extra CLI arguments forwarded to each script (e.g. -- --no-plot)",
    )
    return parser.parse_args()


def _list_steps(log: logging.Logger) -> None:
    log.info("Pipeline Steps:")
    for s in STEPS:
        log.info("  %d  %-35s  %s", s["id"], s["name"], s["description"])


def _setup_pipeline_logger(log_path: Path) -> logging.Logger:
    """Pipeline-level logger: get_logger + FileHandler."""
    logger = get_logger("pipeline")
    already = any(
        isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None) == str(log_path)
        for h in logger.handlers
    )
    if not already:
        fmt = logging.Formatter(
            "[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


@contextlib.contextmanager
def _patch_argv(script_path: Path, extra_args: list[str]):
    """Temporarily replace sys.argv while the script's argparse runs."""
    old = sys.argv[:]
    sys.argv = [str(script_path)] + extra_args
    try:
        yield
    finally:
        sys.argv = old


def run_step(
    step: dict,
    extra_args: list[str],
    dry_run: bool,
    log: logging.Logger | None = None,
) -> bool:
    """
    Call the script's main() function directly.
    The module is reloaded via importlib on every call so that
    module-level initialization (e.g. config loading) reruns fresh.
    """
    lg = log or _log
    script_path: Path = step["script"]
    module_name: str = script_path.stem

    lg.info("--- Step %d: %s ---", step["id"], step["name"])
    lg.info("-> %s.main()  args=%s", module_name, extra_args or [])

    if dry_run:
        lg.info("[DRY-RUN] skipped")
        return True

    # Load the module fresh from file (bypasses import cache → reruns module-level code)
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        lg.error("Failed to load module: %s", script_path)
        return False

    module = importlib.util.module_from_spec(spec)

    # sys.modules에 등록해야 @dataclass 등 타입 어노테이션 처리가 정상 동작함
    _prev = sys.modules.get(module_name)
    sys.modules[module_name] = module

    start = time.time()
    ok = False
    with _patch_argv(script_path, extra_args):
        try:
            spec.loader.exec_module(module)   # run module-level code
            module.main()                     # call main() directly
            ok = True
        except SystemExit as e:
            code = e.code if e.code is not None else 0
            if code == 0:
                ok = True
            else:
                lg.error("Step %d: SystemExit(%s)", step["id"], code)
        except Exception:
            lg.error("Step %d exception:\n%s", step["id"], traceback.format_exc())
        finally:
            # sys.modules 원상 복구 (이전 값이 있으면 되돌리고, 없으면 제거)
            if _prev is None:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = _prev

    elapsed = time.time() - start
    if ok:
        lg.info("Step %d OK  (%s) elapsed=%.1fs", step["id"], step["name"], elapsed)
    else:
        lg.error("Step %d FAILED (%s) elapsed=%.1fs", step["id"], step["name"], elapsed)
    return ok


def main(default_experiment: str | None = None) -> None:
    args = parse_args(default_experiment=default_experiment)

    if args.list:
        _list_steps(_log)
        return

    selected_ids: set[int] = set(args.steps) if args.steps else {s["id"] for s in STEPS}
    selected_steps = [s for s in STEPS if s["id"] in selected_ids]
    base_extra = [a for a in (args.extra_args or []) if a != "--"]

    # 실행할 experiment 목록 결정
    # --experiment 지정 시 해당 하나만, 미지정 시 config의 모든 experiment 순서대로
    all_exp_names = _get_experiment_names()
    if args.experiment:
        experiments_to_run = [args.experiment]
    elif all_exp_names:
        experiments_to_run = all_exp_names
    else:
        experiments_to_run = [None]   # experiment 없이 전체 실행

    # Create pipeline root directory (모든 experiment 공유)
    from utils.run_output import make_run_dir, load_cfg
    _cfg = load_cfg()
    pipeline_run_dir = make_run_dir("pipeline", cfg=_cfg)
    log_path = pipeline_run_dir / "pipeline.log"
    log = _setup_pipeline_logger(log_path)

    mode = "[DRY-RUN] " if args.dry_run else ""
    on_failure = "continue (--continue-on-failure)" if args.continue_on_failure else "abort (default)"
    log.info("=== pipeline start ===")
    log.info("%ssteps       : %s", mode, [s["id"] for s in selected_steps])
    log.info("experiments : %s", experiments_to_run)
    log.info("on failure  : %s", on_failure)
    log.info("extra args  : %s", base_extra or "none")
    log.info("pipeline dir: %s", pipeline_run_dir)

    all_ok = True
    for experiment in experiments_to_run:
        # 환경 변수로 각 step의 make_run_dir()에 경로 전달
        os.environ["PIPELINE_RUN_DIR"] = str(pipeline_run_dir)
        if experiment:
            os.environ["PIPELINE_EXPERIMENT"] = experiment
            log.info("--- [ experiment: %s ] ---", experiment)
        else:
            os.environ.pop("PIPELINE_EXPERIMENT", None)

        # --experiment를 각 step 스크립트 인자로 주입
        extra = list(base_extra)
        if experiment and "--experiment" not in extra:
            extra = ["--experiment", experiment] + extra

        exp_results: list[tuple[dict, bool]] = []
        for step in selected_steps:
            ok = run_step(step, extra, dry_run=args.dry_run, log=log)
            exp_results.append((step, ok))
            if not ok and not args.continue_on_failure:
                log.error("[ABORT] experiment=%s step %d (%s) 실패 — 다음 실험으로 건너뜁니다.",
                          experiment or "(none)", step["id"], step["name"])
                break

        # 실험별 요약
        log.info("[ %s ] 요약:", experiment or "(none)")
        executed_ids = {s["id"] for s, _ in exp_results}
        for step, ok in exp_results:
            icon, tag = ("✓", "OK    ") if ok else ("✗", "FAILED")
            log.info("  %s Step %d [%s] %s", icon, step["id"], tag.strip(), step["name"])
        for step in selected_steps:
            if step["id"] not in executed_ids:
                log.info("  - Step %d [SKIP  ] %s", step["id"], step["name"])

        if any(not ok for _, ok in exp_results):
            all_ok = False

    # 환경 변수 정리
    os.environ.pop("PIPELINE_RUN_DIR", None)
    os.environ.pop("PIPELINE_EXPERIMENT", None)

    log.info("=== pipeline end ===")

    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()


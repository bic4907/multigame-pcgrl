"""
run_pipeline.py — unified entry point for results/ scripts

Calls each script's main() function directly in the same process.
Aborts the pipeline on the first failure by default.

Usage:
    python results/run_pipeline.py                         # full pipeline (steps 1-6)
    python results/run_pipeline.py --experiment fullshot    # fullshot experiment
    python results/run_pipeline.py --experiment zeroshot
    python results/run_pipeline.py --experiment fewshot
    python results/run_pipeline.py --exclude-experiments fewshot seen_ratio_progress  # 특정 experiment 제외
    python results/run_pipeline.py --steps 3               # single step
    python results/run_pipeline.py --steps 3 4 5 6         # multiple steps
    python results/run_pipeline.py --steps 6               # 분석 리포트만 생성
    python results/run_pipeline.py --continue-on-failure   # keep going after failure
    python results/run_pipeline.py --dry-run               # show steps without running
    python results/run_pipeline.py --list                  # list step descriptions

Step numbers:
    1  eval_downloader          Download eval artifacts from W&B
    2  make_eval_summary        ctrl_sim.csv → per-eval results/summary.csv
    3  benchmark               summary/results.csv → Markdown/CSV tables + plots
                                (fullshot 등 일반 실험 전용; zeroshot 에서는 생략)    4  condition_progress_report  condition vs metric plots + Markdown report
    6  analysis_report          모델 간 % 비교 + Baseline 대비 분석을 한글 Markdown 으로 저장
                                (allseen / unseen 모두 적용)
    9  seen_ratio_progress      train_seen_ratio(데이터 양) 증가에 따른 unseen 게임
                                 progress 꺾은선 그래프 — 선 구분: seen 게임 수
                                 (seen_ratio_progress 전용; 다른 실험에서는 생략)
    10 condition_shift_perf_drop  RE별 조건 분포 변화(Wasserstein/JSD) vs 성능 하락 scatter + 상관계수
                                 (condition_shift_analysis 전용; 다른 실험에서는 생략)
    11 seen_count_progress      unseen 게임 개수별 bar chart — method 간 unseen/seen 성능 비교
                                 (unseen_count_progress → seen_count_progress.py 실행;
                                  unseen_count_progress.png + seen_progress.png 출력)
                                 (zeroshot/fewshot 계열 전용; 다른 실험에서는 생략)
    12 encoder_delta_weight_progress
                                encoder_delta_weight 변화에 따른 progress ablation plot
                                (encoder_delta_weight_progress 전용; 다른 실험에서는 생략)
    13 decoder_performance      decoder_prediction_csv artifact + delta loss history 기반
                                decoder 성능 plot
    14 dataset_unseen_ratio_progress
                                dataset_unseen_ratio 변화에 따른 unseen 게임 progress plot
                                (predictive_reward 전용; 다른 실험에서는 생략)

NOTE: step 5 (seen_unseen_report), 7 (unseen_count_progress), 8 (game_impact_analysis)
      스크립트 파일은 results/utils/experiment/ 아래에 보존되어 있으나
      파이프라인의 STEPS 목록에서는 제외되었습니다.
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
        "script": _HERE / "utils/wandb/eval_downloader.py",
        "description": "Download eval artifacts (CSV/H5) from W&B",
    },
    {
        "id": 2,
        "name": "make_eval_summary",
        "script": _HERE / "utils/experiment/make_eval_summary.py",
        "description": "ctrl_sim.csv → per-eval results.csv / summary.csv",
    },
    {
        "id": 3,
        "name": "benchmark",
        "script": _HERE / "utils/experiment/benchmark.py",
        "description": "summary/results.csv → Markdown/CSV tables + comparison plots (fullshot 전용)",
    },
    # step 4 (condition_progress_report) — 비활성화
    # step 6 (analysis_report) — 비활성화
    {
        "id": 9,
        "name": "seen_ratio_progress",
        "script": _HERE / "utils/experiment/seen_ratio_progress.py",
        "description": "train_seen_ratio(데이터 양) 증가에 따른 unseen 게임 progress 꺾은선 그래프 (seen_ratio_progress 전용)",
    },
    {
        "id": 10,
        "name": "condition_shift_perf_drop",
        "script": _HERE / "utils/experiment/condition_shift_perf_drop.py",
        "description": "RE별 조건 분포 변화(Wasserstein/JSD) vs 성능 하락 scatter + 상관계수 (condition_shift_analysis 전용)",
    },
    {
        "id": 11,
        "name": "progress",
        "script": _HERE / "utils/experiment/seen_count_progress.py",
        "description": "unseen 게임 개수별 progress bar chart — all.png / unseen.png / seen.png 출력 (zeroshot 전용)",
    },
    {
        "id": 12,
        "name": "encoder_delta_weight_progress",
        "script": _HERE / "utils/experiment/encoder_delta_weight_progress.py",
        "description": "encoder_delta_weight 변화에 따른 progress ablation plot (전용)",
    },
    {
        "id": 13,
        "name": "decoder_performance",
        "script": _HERE / "utils/experiment/decoder_performance.py",
        "description": "decoder_prediction_csv artifact + delta loss history 기반 decoder performance plot (전용)",
    },
    {
        "id": 14,
        "name": "dataset_unseen_ratio_progress",
        "script": _HERE / "utils/experiment/dataset_unseen_ratio_progress.py",
        "description": "dataset_unseen_ratio 변화에 따른 unseen 게임 progress 꺾은선 그래프 (predictive_reward 전용)",
    },
]

# 특정 experiment 에서 실행하지 않을 step id
_EXPERIMENT_SKIP: dict[str | None, set[int]] = {
    "zeroshot":        {3, 9, 10},        # benchmark / seen_ratio_progress / condition_shift 생략
    "fewshot":                  {3, 9, 10},        # benchmark / seen_ratio_progress / condition_shift 생략; step 11(progress) 사용
    "fewshot_seenrate":         {3, 9, 10},        # legacy alias
    "fewshot_delta_alignment": {3, 9, 10, 12}, # fewshot 비교 — step 11(progress)만 사용
    "directional_fewshot": {3, 9, 10, 12}, # fewshot 비교 — step 11(progress)만 사용
    "domain_condition":{3, 9, 10},        # fewshot과 동일 — step 11(progress) 사용
    "instruction_type":{3, 9, 10},        # fewshot과 동일 — step 11(progress) 사용
    "seen_ratio_progress":       {3, 4, 10, 11},    # seen_ratio_progress 전용 — step 9만 실행
    "condition_shift_analysis":  {3, 4, 6, 9, 11},  # condition_shift_analysis 전용 — step 10만 실행
    "unseen_ratio_ngames":       {3, 9, 10},         # benchmark / seen_ratio_progress / condition_shift 생략
    "encoder_delta_weight_progress": {2, 3, 9, 10, 11},  # downloader + step 12만 실행
    "decoder_performance":       {1, 2, 3, 9, 10, 11, 12},  # step 13만 실행
    "dwctrl":                    {1, 2, 3, 9, 10, 11, 12},  # step 13만 실행
    "predictive_reward":         {3, 9, 10, 11, 12, 13},  # downloader + summary + step 14만 실행
    None:                        {9, 10, 11, 12, 13, 14},    # experiment 미지정 시 생략
}
# fullshot 등 전용 실험이 아닌 일반 실험: 9, 10, 11, 12 생략
_DEFAULT_SKIP: set[int] = {9, 10, 11, 12, 13, 14}


def parse_args(default_experiment: str | None = None) -> argparse.Namespace:
    _exp_names = _get_experiment_names()
    _exp_hint = ", ".join(_exp_names) if _exp_names else "none defined"
    parser = argparse.ArgumentParser(
        description="Unified pipeline entry point for results/ scripts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
examples:
  python results/run_pipeline.py                       # run all steps (1-6)
  python results/run_pipeline.py --experiment fullshot
  python results/run_pipeline.py --experiment zeroshot
  python results/run_pipeline.py --experiment fewshot
  python results/run_pipeline.py --exclude-experiments fewshot seen_ratio_progress
  python results/run_pipeline.py --steps 3             # table generation only
  python results/run_pipeline.py --steps 4 5           # condition report + seen/unseen report
  python results/run_pipeline.py --steps 6             # 한글 분석 리포트만 생성
  python results/run_pipeline.py --continue-on-failure # keep going after failures
  python results/run_pipeline.py --dry-run             # show steps without running
  python results/run_pipeline.py --list                # list step descriptions

note:
  step 3 (benchmark)             — fullshot 등 일반 실험에서만 실행; zeroshot / fewshot 에서는 자동 생략
  step 6 (analysis_report)       — fullshot / zeroshot / fewshot 모두 실행; 한글 Markdown 리포트 생성
  step 9 (seen_ratio_progress)   — seen_ratio_progress 실험에서만 실행; train_seen_ratio vs progress 꺾은선 그래프
  step 11 (seen_count_progress)  — zeroshot/fewshot 계열에서 실행; unseen 개수별 subplot 및 seen/unseen 표
  step 14 (dataset_unseen_ratio_progress) — predictive_reward 실험에서만 실행; dataset_unseen_ratio vs unseen progress

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
        "--exclude-experiments",
        nargs="+",
        default=[],
        metavar="EXPERIMENT",
        help=(
            "Experiment names to skip when running the full pipeline "
            "(ignored if --experiment is specified). "
            f"Choices: {_exp_hint}"
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


def _compute_global_norm_scale(
    pipeline_run_dir: Path,
    base_extra: list[str],
    log: logging.Logger,
) -> bool:
    """
    모든 실험 데이터를 합산해 normalization scale 을 계산하고
    pipeline_run_dir/normalization_scale.json 에 저장한다.

    --input 인자가 base_extra 에 있으면 그 경로를 사용하고,
    없으면 'wandb_projects' 기본값을 사용한다.

    Returns True on success, False on failure.
    """
    try:
        from utils.experiment.benchmark import (
            collect_plot_rows_from_results,
            resolve_input_root,
            DEFAULT_METRIC_ORDER,
        )
        from utils.core.normalization import compute_normalization_scale, save_normalization_scale

        # base_extra 에서 --input 값 추출 (없으면 default)
        try:
            idx = base_extra.index("--input")
            input_arg = base_extra[idx + 1]
        except (ValueError, IndexError):
            input_arg = "wandb_projects"

        input_root = resolve_input_root(input_arg, _HERE)
        log.info("[norm_scale] 전체 데이터 수집 중: %s", input_root)

        all_rows = collect_plot_rows_from_results(input_root, DEFAULT_METRIC_ORDER)
        if not all_rows:
            log.warning("[norm_scale] 데이터 없음 — %s", input_root)
            return False

        scale = compute_normalization_scale(all_rows, DEFAULT_METRIC_ORDER)
        scale_path = pipeline_run_dir / "normalization_scale.json"
        save_normalization_scale(scale, scale_path)
        log.info("[norm_scale] 저장 완료: %s  (rows=%d)", scale_path, len(all_rows))
        return True
    except Exception:
        log.error("[norm_scale] 계산 중 예외:\n%s", traceback.format_exc())
        return False


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
        exclude_set = set(args.exclude_experiments or [])
        experiments_to_run = [e for e in all_exp_names if e not in exclude_set]
        if exclude_set:
            excluded = [e for e in all_exp_names if e in exclude_set]
            _log.info("제외된 experiments: %s", excluded)
    else:
        experiments_to_run = [None]   # experiment 없이 전체 실행

    # Create pipeline root directory (모든 experiment 공유)
    from utils.core.run_output import make_run_dir, load_cfg
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

    # ── 전역 Normalization Scale 사전 계산 ──────────────────────────────────
    # 모든 실험 데이터를 합산하여 normalization scale 을 한 번만 계산하고
    # pipeline_run_dir/normalization_scale.json 에 저장한다.
    # 이후 각 step 은 PIPELINE_NORM_SCALE 환경변수를 통해 이 파일을 읽는다.
    if not args.dry_run:
        _norm_ok = _compute_global_norm_scale(pipeline_run_dir, base_extra, log)
        if _norm_ok:
            os.environ["PIPELINE_NORM_SCALE"] = str(pipeline_run_dir / "normalization_scale.json")
            log.info("PIPELINE_NORM_SCALE: %s", os.environ["PIPELINE_NORM_SCALE"])
        else:
            log.warning("전역 norm scale 계산 실패 — 각 실험이 개별 scale 을 사용합니다.")
    # ────────────────────────────────────────────────────────────────────────

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
            # 실험별 자동 스킵 —————————————————————————————————————————
            skip_ids = _EXPERIMENT_SKIP.get(experiment, _DEFAULT_SKIP)
            if step["id"] in skip_ids:
                log.info(
                    "Step %d (%s) — experiment=%s 에서 생략됩니다.",
                    step["id"], step["name"], experiment or "(none)",
                )
                continue
            # ——————————————————————————————————————————————————————————
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
    os.environ.pop("PIPELINE_NORM_SCALE", None)

    log.info("=== pipeline end ===")

    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()

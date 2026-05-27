"""
pipeline.py
===========
MGPCGRL 보상 예측 분석 End-to-End 파이프라인.

단계
----
1. [Export]     체크포인트 inference → CSV 파일 생성
                (exporter.checkpoint_csv.run_export)
2. [Visualize]  CSV → 시각화 플롯 생성
                (visualizer.plots.run_visualize)

실행 예시
---------
    # 전체 파이프라인 (export + visualize)
    python analysis/reward_prediction/pipeline.py

    # export 만
    python analysis/reward_prediction/pipeline.py --skip-visualize

    # 이미 생성된 CSV로 visualize 만
    python analysis/reward_prediction/pipeline.py --skip-export \\
        --all-csv results/mgpcgrl_reward_pred_csv/all_checkpoints.csv

    # 일부 체크포인트만 (테스트)
    python analysis/reward_prediction/pipeline.py --max-checkpoints 3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

# 프로젝트 루트 import 경로 보장
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from instruct_rl.utils.log_utils import get_logger, suppress_jax_debug_logs

suppress_jax_debug_logs()
logger = get_logger(__file__)


# ─────────────────────────────────────────────────────────
# CLI 정의
# ─────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="MGPCGRL reward prediction analysis pipeline: export → visualize.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- 공통 ---
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/mgpcgrl_reward_pred_csv"),
        help="CSV 저장 디렉토리 (export 결과)",
    )
    p.add_argument(
        "--plot-dir",
        type=Path,
        default=Path("results/reward_decoding_plots"),
        help="시각화 플롯 저장 디렉토리",
    )

    # --- 단계 선택 ---
    step_grp = p.add_argument_group("Pipeline steps")
    step_grp.add_argument(
        "--skip-export",
        action="store_true",
        help="Export 단계를 건너뛴다 (기존 CSV 사용).",
    )
    step_grp.add_argument(
        "--skip-visualize",
        action="store_true",
        help="Visualize 단계를 건너뛴다.",
    )

    # --- Export 옵션 ---
    exp_grp = p.add_argument_group("Export options")
    exp_grp.add_argument(
        "--sweep-yaml",
        type=Path,
        default=Path("sweep/wandb_sweep/train_mgpcgrl_unseen.yaml"),
        help="WandB sweep yaml (encoder.ckpt_dir / ckpt_name 목록).",
    )
    exp_grp.add_argument(
        "--ckpt-dir-override",
        type=str,
        default=None,
        help="sweep yaml의 encoder.ckpt_dir 를 덮어씀.",
    )
    exp_grp.add_argument("--dataset-game", type=str, default="all")
    exp_grp.add_argument(
        "--dataset-reward-enum",
        type=str,
        default="all",
        help='"all", "0", "1", "0,1" 등',
    )
    exp_grp.add_argument(
        "--reward-decoder-mode",
        type=str,
        default="all",
        choices=["all", "unseen", "noop"],
    )
    exp_grp.add_argument(
        "--max-samples-per-game",
        type=int,
        default=0,
        help="0 = 제한 없음",
    )
    exp_grp.add_argument(
        "--max-checkpoints",
        type=int,
        default=0,
        help="0 = sweep yaml 전체 체크포인트 사용",
    )
    exp_grp.add_argument(
        "--fail-on-missing",
        action="store_true",
        help="체크포인트 미존재 시 즉시 실패 (기본: skip)",
    )

    # --- Visualize 옵션 ---
    vis_grp = p.add_argument_group("Visualize options")
    vis_grp.add_argument(
        "--all-csv",
        type=Path,
        default=None,
        help="시각화에 사용할 CSV 경로. 미지정 시 --output-dir/all_checkpoints.csv 사용.",
    )
    vis_grp.add_argument(
        "--sample-n",
        type=int,
        default=3000,
        help="scatter plot 당 최대 샘플 수",
    )

    return p


# ─────────────────────────────────────────────────────────
# 단계별 실행 함수
# ─────────────────────────────────────────────────────────

def _run_export_stage(args: argparse.Namespace) -> Path:
    """Export 단계 실행 → all_checkpoints.csv 경로 반환."""
    from exporter import ExportConfig, run_export
    from exporter.checkpoint_csv import _to_number_or_str

    cfg = ExportConfig(
        sweep_yaml=args.sweep_yaml,
        ckpt_dir_override=args.ckpt_dir_override,
        output_dir=args.output_dir,
        dataset_game=args.dataset_game,
        dataset_reward_enum=_to_number_or_str(args.dataset_reward_enum),
        reward_decoder_mode=args.reward_decoder_mode,
        max_samples_per_game=args.max_samples_per_game,
        max_checkpoints=args.max_checkpoints,
        fail_on_missing=args.fail_on_missing,
    )
    result = run_export(cfg)
    return result.all_csv_path


def _run_visualize_stage(csv_path: Path, args: argparse.Namespace) -> None:
    """Visualize 단계 실행."""
    from visualizer import VisualizerConfig, run_visualize

    cfg = VisualizerConfig(
        csv_path=csv_path,
        output_dir=args.plot_dir,
        sample_n=args.sample_n,
    )
    result = run_visualize(cfg)
    logger.info("Saved plots:")
    for p in result.saved_plots:
        logger.info("  %s", p)


# ─────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────

def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.skip_export and args.skip_visualize:
        parser.error("--skip-export 와 --skip-visualize 를 동시에 지정할 수 없습니다.")

    all_csv: Optional[Path] = args.all_csv

    # ── Step 1: Export ──
    if not args.skip_export:
        logger.info("=" * 60)
        logger.info("STEP 1/2 : Export — checkpoint inference → CSV")
        logger.info("=" * 60)
        all_csv = _run_export_stage(args)
    else:
        logger.info("STEP 1/2 : Export  [SKIPPED]")
        if all_csv is None:
            all_csv = args.output_dir / "all_checkpoints.csv"
        logger.info("Using CSV: %s", all_csv)

    # ── Step 2: Visualize ──
    if not args.skip_visualize:
        logger.info("=" * 60)
        logger.info("STEP 2/2 : Visualize — CSV → plots")
        logger.info("=" * 60)
        if all_csv is None or not Path(all_csv).is_file():
            logger.error(
                "all_checkpoints.csv 를 찾을 수 없습니다: %s\n"
                "--all-csv 옵션으로 CSV 경로를 직접 지정하거나 export 단계를 먼저 실행하세요.",
                all_csv,
            )
            sys.exit(1)
        _run_visualize_stage(Path(all_csv), args)
    else:
        logger.info("STEP 2/2 : Visualize  [SKIPPED]")

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()


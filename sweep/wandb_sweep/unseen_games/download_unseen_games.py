"""
download_unseen_games.py
========================
sweep_unseen_games 프로젝트에서 실험 결과(table/results)를 다운로드하고
로컬 CSV로 저장하는 스크립트.

사용법
------
    # 프로젝트 루트에서 실행
    python sweep/wandb_sweep/unseen_games/download_unseen_games.py

    # 모든 state의 run 다운로드 (기본: finished만)
    python sweep/wandb_sweep/unseen_games/download_unseen_games.py --all

    # 이미 다운로드된 파일도 덮어쓰기
    python sweep/wandb_sweep/unseen_games/download_unseen_games.py --no-skip

출력 구조
---------
    results/
      sweep_unseen_games/
        {exp_dir_basename}/
          results.csv     ← table/results 테이블
          config.json     ← run config
        combined_results.csv   ← 전체 병합본
"""

import argparse
import logging
import os
import sys

# ── 경로 설정: 프로젝트 루트를 sys.path에 추가 ────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
_WANDB_UTILS = os.path.join(_ROOT, "sweep", "wandb_utils")

for _p in [_ROOT, _WANDB_UTILS]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# 프로젝트 루트의 .env 로드 (dotenv 패키지 불필요)
from instruct_rl.utils.env_loader import load_dotenv
load_dotenv(os.path.join(_ROOT, ".env"))

from sweep.wandb_utils.downloader import WandbTableDownloader

# ──────────────────────────────── 설정 ────────────────────────────────────────

ENTITY  = os.getenv("WANDB_ENTITY", "st4889ha-gwangju-institute-of-science-and-technology")
PROJECT = "sweep_unseen_games"

# table/results 테이블은 wandb.log({"table/results": ...}) 로 기록됨
# W&B 내부 파일명은 "results" 를 포함하므로 아래 패턴으로 검색
TABLE_PATTERNS = {
    "results": "results",   # → results.csv
}

OUTPUT_DIR = os.path.join(_HERE, "wandb_download", PROJECT)

# ─────────────────────── run config → 추가 컬럼 ───────────────────────────────

def _extra_cols(config: dict, run) -> dict:
    """각 행에 붙일 run 레벨 메타데이터를 반환합니다.

    table/results 테이블 자체에 ratio, seen_ratio, unseen_games 컬럼이
    이미 포함되어 있으므로 run 식별 정보만 추가합니다.
    """
    return {
        "run_id":    run.id,
        "run_name":  run.name,
        "run_state": run.state,
        # 혹시 테이블에 없을 경우를 대비해 config에서도 백업
        "cfg_unseen_games": config.get("unseen_games", ""),
        "cfg_seen_ratio":   config.get("seen_ratio", ""),
        "cfg_seed":         config.get("seed", ""),
    }


# ──────────────────────────────── main ────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="sweep_unseen_games W&B 결과 다운로드")
    p.add_argument(
        "--entity",  default=ENTITY,
        help=f"W&B 엔티티 (기본값: {ENTITY})",
    )
    p.add_argument(
        "--project", default=PROJECT,
        help=f"W&B 프로젝트 이름 (기본값: {PROJECT})",
    )
    p.add_argument(
        "--output-dir", default=OUTPUT_DIR,
        help=f"결과 저장 디렉토리 (기본값: {OUTPUT_DIR})",
    )
    p.add_argument(
        "--workers", type=int, default=8,
        help="병렬 다운로드 워커 수 (기본값: 8)",
    )
    p.add_argument(
        "--all", action="store_true",
        help="finished 외 모든 state 의 run도 다운로드 (기본: finished만)",
    )
    p.add_argument(
        "--no-skip", action="store_true",
        help="이미 다운로드된 파일도 덮어쓰기",
    )
    p.add_argument(
        "--combine-only", action="store_true",
        help="다운로드 없이 기존 CSV만 병합",
    )
    p.add_argument(
        "--per-page", type=int, default=200,
        help="W&B API 페이지당 run 수 (기본값: 200)",
    )
    return p.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    args = parse_args()

    downloader = WandbTableDownloader(entity=args.entity)

    filters = None if args.all else {"state": "finished"}

    if args.combine_only:
        # 다운로드 없이 기존 파일만 합치기
        logging.info("--combine-only 모드: 다운로드 없이 CSV 병합만 수행합니다.")
        combined = downloader.combine_csvs(
            output_dir=args.output_dir,
            table_patterns=TABLE_PATTERNS,
            project=args.project,
        )
    else:
        combined = downloader.download_project(
            project=args.project,
            table_patterns=TABLE_PATTERNS,
            output_dir=args.output_dir,
            extra_cols_fn=_extra_cols,
            n_workers=args.workers,
            filters=filters,
            per_page=args.per_page,
            skip_if_exists=not args.no_skip,
        )

    if combined:
        logging.info(f"\n✅ 병합 CSV 생성 완료: {combined}")
    else:
        logging.warning("병합할 CSV가 없습니다. 다운로드된 run을 확인해 주세요.")


if __name__ == "__main__":
    main()


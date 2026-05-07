"""
process_unseen_generalizability.py
===================================
unseen_generalizability 실험의 다운로드된 결과 파일을 처리한다.

대상 프로젝트 (config.json experiments.unseen_generalizability):
    aaai27_eval_mgpcgrl, aaai27_eval_mgpcgrl_unseen

처리 단계:
    1  make_gamewise_summary    ctrl_sim.csv → per-game results.csv / summary.csv
    2  build_benchmark_table    summary.csv → 벤치마크 테이블 + 플롯
    3  condition_progress_report ctrl_sim.csv → 조건 플롯 + Markdown 리포트

사용법:
    python results/process_unseen_generalizability.py
    python results/process_unseen_generalizability.py --steps 2 3
    python results/process_unseen_generalizability.py --input wandb_projects --dry-run
    python results/process_unseen_generalizability.py --continue-on-failure
"""

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
if str(_HERE.parent) not in sys.path:
    sys.path.append(str(_HERE.parent))

from process_shared import run_experiment_processing  # noqa: E402

EXPERIMENT = "unseen_generalizability"


def main() -> None:
    run_experiment_processing(EXPERIMENT)


if __name__ == "__main__":
    main()


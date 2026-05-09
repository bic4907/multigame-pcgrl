"""
process_allseen.py
==================
allseen 실험의 다운로드된 결과 파일을 처리한다.

대상 프로젝트 (config.json experiments.allseen):
    aaai27_eval_cpcgrl, aaai27_eval_vipcgrl_nosim, aaai27_eval_mgpcgrl,
    aaai27_eval_ipcgrl, aaai27_eval_vipcgrl, aaai27_eval_random

처리 단계:
    1  make_gamewise_summary    ctrl_sim.csv → per-game results.csv / summary.csv
    2  build_benchmark_table    summary.csv → 벤치마크 테이블 + 플롯
    3  condition_progress_report ctrl_sim.csv → 조건 플롯 + Markdown 리포트

사용법:
    python results/process_allseen.py
    python results/process_allseen.py --steps 2 3
    python results/process_allseen.py --input wandb_projects --dry-run
    python results/process_allseen.py --continue-on-failure
"""

import sys
from pathlib import Path

_HERE        = Path(__file__).resolve().parent   # results/utils/experiment/
_RESULTS_DIR = _HERE.parent.parent               # results/
if str(_RESULTS_DIR) not in sys.path:
    sys.path.insert(0, str(_RESULTS_DIR))
if str(_RESULTS_DIR.parent) not in sys.path:
    sys.path.append(str(_RESULTS_DIR.parent))

from utils.pipeline import run_experiment_processing  # noqa: E402

EXPERIMENT = "allseen"


def main() -> None:
    run_experiment_processing(EXPERIMENT)


if __name__ == "__main__":
    main()


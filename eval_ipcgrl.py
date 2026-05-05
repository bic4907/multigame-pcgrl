"""
eval_ipcgrl.py
==============
IPCGRL (Instructed PCGRL) 평가 엔트리포인트.
BERT 임베딩을 nlp_obs 에 주입하여 평가한다.

실행:
    python -m eval_ipcgrl [overrides]
"""
import hydra

from conf.config import IPCGRLEvalConfig
from instruct_rl.utils.log_utils import suppress_jax_debug_logs
from instruct_rl.utils.eval_utils import main_eval_entry
from train_ipcgrl import inject_ipcgrl_obs

suppress_jax_debug_logs()


# ── Hydra entrypoint ──────────────────────────────────────────────────────────

@hydra.main(version_base=None, config_path="./conf", config_name="eval_ipcgrl")
def main(config: IPCGRLEvalConfig):
    main_eval_entry(config, inject_obs_fn=inject_ipcgrl_obs)


if __name__ == "__main__":
    main()


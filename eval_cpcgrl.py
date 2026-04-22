"""
eval_cpcgrl.py
==============
CPCGRL (Conditional PCGRL) 평가 엔트리포인트.
raw condition 벡터를 nlp_obs 에 주입하여 평가한다.

실행:
    python -m eval_cpcgrl [overrides]
"""
import hydra

from conf.config import CPCGRLEvalConfig
from instruct_rl.utils.log_utils import suppress_jax_debug_logs
from instruct_rl.utils.eval_utils import main_eval_entry
from train_cpcgrl import inject_cpcgrl_obs

suppress_jax_debug_logs()



# ── Hydra entrypoint ──────────────────────────────────────────────────────────

@hydra.main(version_base=None, config_path="./conf", config_name="eval_cpcgrl")
def main(config: CPCGRLEvalConfig):
    main_eval_entry(config, inject_obs_fn=inject_cpcgrl_obs)


if __name__ == "__main__":
    main()


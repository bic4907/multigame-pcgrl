"""
eval_cpcgrl.py
==============
CPCGRL (Conditional PCGRL) 평가 엔트리포인트.
raw condition 벡터를 nlp_obs 에 주입하여 평가한다.

실행:
    python -m eval_cpcgrl [overrides]
"""
import hydra

from conf.config import MGPCGRLEvalConfig
from instruct_rl.utils.log_utils import suppress_jax_debug_logs
from instruct_rl.utils.eval_utils import main_eval_entry
from train_mg_pcgrl import inject_vipcgrl_obs

suppress_jax_debug_logs()



# ── Hydra entrypoint ──────────────────────────────────────────────────────────

@hydra.main(version_base=None, config_path="./conf", config_name="eval_mgpcgrl")
def main(config: MGPCGRLEvalConfig):
    main_eval_entry(config, inject_obs_fn=inject_vipcgrl_obs)


if __name__ == "__main__":
    main()


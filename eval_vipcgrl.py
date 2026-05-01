"""
eval_vipcgrl.py
===============
VIPCGRL (Vision-Instructed PCGRL) 평가 엔트리포인트.
pretrained CLIP 임베딩을 nlp_obs 에 주입하여 평가한다.

실행:
    python -m eval_vipcgrl [overrides]
"""
import hydra

from conf.config import VIPCGRLEvalConfig
from instruct_rl.utils.log_utils import suppress_jax_debug_logs
from instruct_rl.utils.eval_utils import main_eval_entry
from train_vipcgrl import inject_vipcgrl_obs

suppress_jax_debug_logs()



# ── Hydra entrypoint ──────────────────────────────────────────────────────────

@hydra.main(version_base=None, config_path="./conf", config_name="eval_vipcgrl")
def main(config: VIPCGRLEvalConfig):

    if not config.encoder.ckpt_dir or not config.encoder.ckpt_name:
        raise ValueError("Both encoder.ckpt_dir and encoder.ckpt_name must be set in the configuration.")

    main_eval_entry(config, inject_obs_fn=inject_vipcgrl_obs)


if __name__ == "__main__":
    main()


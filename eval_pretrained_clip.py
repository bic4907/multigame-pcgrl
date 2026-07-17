"""
eval_pretrained_clip.py
=======================
PretrainedCLIP PCGRL evaluation entry point.
precomputed CLIP text embedding  nlp_obs  in  injecttext evaluationtext.

Usage:
    python -m eval_pretrained_clip [overrides]
    python -m eval_pretrained_clip dataset_reward_enum=4 game=all
"""
import hydra

from conf.config import PretrainedCLIPEvalConfig
from instruct_rl.utils.log_utils import suppress_jax_debug_logs
from instruct_rl.utils.eval_utils import main_eval_entry
from train_pretrained_clip import inject_pretrained_clip_pcgrl_obs

suppress_jax_debug_logs()


# ── Hydra entrypoint ──────────────────────────────────────────────────────────

@hydra.main(version_base=None, config_path="./conf", config_name="eval_pretrained_clip")
def main(config: PretrainedCLIPEvalConfig):
    main_eval_entry(config, inject_obs_fn=inject_pretrained_clip_pcgrl_obs)


if __name__ == "__main__":
    main()

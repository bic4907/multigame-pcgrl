"""
train_vipcgrl.py
=================
VIPCGRL (Vision-Instructed PCGRL) — pretrained CLIP 임베딩을 입력 피처로 사용.

기존 train.py 의 `encoder.model='cnnclip'` 모드에 해당하며,
dataset 기반 파이프라인(MultiGameDataset)으로 동작한다.

실행:
    python -m train_vipcgrl [overrides]
    python -m train_vipcgrl dataset_game=dungeon dataset_reward_enum=1 SIM_COEF=3.5
"""
import hydra

from conf.config import CPCGRLConfig
from instruct_rl.utils.log_utils import suppress_jax_debug_logs
from instruct_rl.utils.train_utils import main_entry

suppress_jax_debug_logs()


# ── VIPCGRL obs 주입: CLIP embedding → nlp_obs ───────────────────────────────

def inject_vipcgrl_obs(last_obs, env_state, instruct_sample, config, env):
    """pretrained CLIP 인코더로 계산된 임베딩을 nlp_obs 에 주입."""
    return last_obs.replace(nlp_obs=instruct_sample.embedding)


# ── Hydra entrypoint ──────────────────────────────────────────────────────────

@hydra.main(version_base=None, config_path="./conf", config_name="train_vipcgrl")
def main(config: CPCGRLConfig):
    main_entry(config, inject_obs_fn=inject_vipcgrl_obs)


if __name__ == "__main__":
    main()


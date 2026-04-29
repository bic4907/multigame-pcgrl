"""
train_mg_pcgrl.py
=================
MGPCGRL (MultiGame PCGRL) — pretrained CLIP 임베딩을 입력 피처로 사용.

기존 VIPCGRL 파이프라인을 기반으로 하되,
실험 엔트리/설정을 mgpcgrl 이름으로 분리한 실행 스크립트.

실행:
    python -m train_mg_pcgrl [overrides]
    python -m train_mg_pcgrl dataset_game=dungeon dataset_reward_enum=1 SIM_COEF=3.5
"""
import hydra

from conf.config import MGPCGRLConfig
from instruct_rl.utils.log_utils import suppress_jax_debug_logs
from instruct_rl.utils.train_utils import main_entry

suppress_jax_debug_logs()


# ── VIPCGRL obs 주입: CLIP embedding → nlp_obs ───────────────────────────────

def inject_vipcgrl_obs(last_obs, env_state, instruct_sample, config, env):
    """pretrained CLIP 인코더로 계산된 임베딩을 nlp_obs 에 주입."""
    return last_obs.replace(nlp_obs=instruct_sample.embedding)


# ── Hydra entrypoint ──────────────────────────────────────────────────────────

@hydra.main(version_base=None, config_path="./conf", config_name="train_mgpcgrl")
def main(config: MGPCGRLConfig):

    main_entry(
        config,
        inject_obs_fn=inject_vipcgrl_obs,
        # inject_reward_fn=inject_reward_fn,
    )


if __name__ == "__main__":
    main()

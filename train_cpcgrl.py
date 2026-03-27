"""
train_cpcgrl.py
================
CPCGRL (Conditional PCGRL) — raw condition 벡터를 입력 피처로 사용.

실행:
    python -m train_cpcgrl [overrides]
"""
import jax
import hydra

from conf.config import CPCGRLConfig
from instruct_rl.utils.log_utils import suppress_jax_debug_logs
from instruct_rl.utils.train_utils import main_entry

suppress_jax_debug_logs()


# ── CPCGRL obs 주입: get_cont_obs → nlp_obs ──────────────────────────────────

def inject_cpcgrl_obs(last_obs, env_state, instruct_sample, config, env):
    """env_map + condition 으로 continuous observation 을 계산하여 nlp_obs 에 주입."""
    vmap_state_fn = jax.vmap(env.prob.get_cont_obs, in_axes=(0, 0, None))
    cont_obs = vmap_state_fn(
        env_state.env_state.env_map,
        instruct_sample.condition,
        config.raw_obs,
    )
    return last_obs.replace(nlp_obs=cont_obs)


# ── Hydra entrypoint ──────────────────────────────────────────────────────────

@hydra.main(version_base=None, config_path="./conf", config_name="train_cpcgrl")
def main(config: CPCGRLConfig):
    main_entry(config, inject_obs_fn=inject_cpcgrl_obs)


if __name__ == "__main__":
    main()

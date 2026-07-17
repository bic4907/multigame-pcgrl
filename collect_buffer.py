"""
collect_buffer.py
==================
RL  in previoustext training and  text in  trajectory text  text  entry point.

training interval of  text(default 50%~100%) in  text text text(env_idx=0) basis as
text text to  (obs, action, reward, done, env_map) data  text
experiment folder of  buffer/ directory in  .npz file to  savetext.

text text  buffer_max_samples / num_steps  to  automatic computetext.

Usage:
    python -m collect_buffer [overrides]

Key parameters:
    buffer_max_samples    : text maximum transition text (default 10,000)
    collect_start_ratio   : text start text (default 0.5 = training 50%)
    collect_end_ratio     : text text text (default 1.0 = training 100%)
    buffer_save_dir       : save path (default None → exp_dir/buffer)
"""
import jax
import hydra

from conf.config import CollectBufferConfig
from instruct_rl.utils.log_utils import suppress_jax_debug_logs
from instruct_rl.utils.train_utils import main_entry

suppress_jax_debug_logs()


# ── CPCGRL obs inject: get_cont_obs → nlp_obs ──────────────────────────────────

def inject_cpcgrl_obs(last_obs, env_state, instruct_sample, config, env):
    """env_map + condition  as  continuous observation   computetext nlp_obs  in  inject."""
    vmap_state_fn = jax.vmap(env.prob.get_cont_obs, in_axes=(0, 0, None))
    cont_obs = vmap_state_fn(
        env_state.env_state.env_map,
        instruct_sample.condition,
        config.raw_obs,
    )
    return last_obs.replace(nlp_obs=cont_obs)


# ── Hydra entrypoint ──────────────────────────────────────────────────────────

@hydra.main(version_base=None, config_path="./conf", config_name="collect_buffer")
def main(config: CollectBufferConfig):
    main_entry(config, inject_obs_fn=inject_cpcgrl_obs)


if __name__ == "__main__":
    main()

"""
collect_buffer.py
==================
Entry point that replays a trained RL policy and collects trajectories.

Over a window of training (50%-100% by default) a single environment (env_idx=0)
is stepped, and its (obs, action, reward, done, env_map) data is written as
.npz files into the buffer/ directory of the experiment folder.

The number of steps is derived automatically from buffer_max_samples / num_steps.

Usage:
    python -m collect_buffer [overrides]

Key parameters:
    buffer_max_samples    : maximum number of transitions to collect (default: 10,000)
    collect_start_ratio   : start of the window (default 0.5 = 50% into training)
    collect_end_ratio     : end of the window (default 1.0 = end of training)
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
    """Build the continuous observation from env_map + condition and inject it into nlp_obs."""
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

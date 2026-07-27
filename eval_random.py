"""
eval_random.py
==============
Entry point for evaluating a PCGRL environment with a pure random policy.

Samples uniformly from the action space at every step without using a neural network.
This is a genuinely random policy, not an initialized neural-network policy.

Usage:
    python -m eval_random [overrides]
"""
import hydra

from conf.config import RandomEvalConfig
from instruct_rl.utils.log_utils import suppress_jax_debug_logs
from instruct_rl.utils.eval_utils import main_eval_entry

suppress_jax_debug_logs()


# ── Hydra entrypoint ──────────────────────────────────────────────────────────

@hydra.main(version_base=None, config_path="./conf", config_name="eval_random")
def main(config: RandomEvalConfig):
    # ── Ensure a pure random policy ──────────────────────────────────────────
    # random_agent=True makes runner.py sample uniformly from the action space
    # without a neural-network forward pass.
    config.random_agent = True

    # inject_obs_fn=None leaves observations unchanged; they are unused by the policy anyway
    main_eval_entry(config, inject_obs_fn=None)


if __name__ == "__main__":
    main()

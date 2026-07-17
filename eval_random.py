"""
eval_random.py
==============
text before  random text(Pure Random Policy) as  PCGRL text  evaluationtext  entry point.

NN  text for text text text text action space in  uniform random sampling  textrowtext.
(initializetext NN policy  text text randomtext  warning)

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
    # ── text before  random policy text ──────────────────────────────────────────────
    # random_agent=True → runner.py in  NN forward pass text
    # action space in  uniform random sampling textrow (text random, initialized policy text)
    config.random_agent = True

    # inject_obs_fn=None: obs convert text  as-is text for  (text NN text for  text to  text)
    main_eval_entry(config, inject_obs_fn=None)


if __name__ == "__main__":
    main()

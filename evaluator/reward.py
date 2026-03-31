from functools import partial

import jax
from jax import vmap
import chex
import jax.numpy as jnp
from envs.probs.dungeon3 import Dungeon3Tiles


from .rewards import *
from .rewards.multigame_amount import get_multigame_amount_reward
from .weights import RewardWeight, RewardBias


@partial(jax.jit, static_argnums=(4,))
def get_reward_batch(
    reward_i: chex.Array,
    condition: chex.Array,
    prev_env_map: chex.Array,
    curr_env_map: chex.Array,
    map_size: chex.Array = 16,
) -> chex.Array:
    """
    Compute batch rewards by mapping indices to reward functions and executing them in parallel.

    Args:
        reward_i: Array of indices mapping to functions in call_reward.
        condition: Array of conditions corresponding to each reward calculation.
        prev_env_map: Previous environment map.
        curr_env_map: Current environment map.

    Returns:
        rewards: Array of computed rewards.
    """
    # List of reward functions
    reward_funcs = [
        lambda cond, prev_map, curr_map: 0.0,  # 0
        lambda cond, prev_map, curr_map: get_region_reward(
            prev_map, curr_map, cond[0]
        ) * RewardWeight.REGION + RewardBias.REGION,  # 1 (region)
        lambda cond, prev_map, curr_map: get_path_length_reward(
            prev_map, curr_map, cond[1]
        ) * RewardWeight.PATH_LENGTH + RewardBias.PATH_LENGTH,  # 2 (diameter)
        lambda cond, prev_map, curr_map: get_multigame_amount_reward(
            prev_map, curr_map, cond[4], tile_name="interactive"
        ) * RewardWeight.MONSTER,  # 5 (interactive_amount)
        lambda cond, prev_map, curr_map: get_multigame_amount_reward(
            prev_map, curr_map, cond[5], tile_name="hazard"
        ) * RewardWeight.MONSTER,  # 6 (hazard_amount)
        lambda cond, prev_map, curr_map: get_multigame_amount_reward(
            prev_map, curr_map, cond[6], tile_name="collectable"
        ) * RewardWeight.MONSTER,  # 7 (collectable_amount)
        lambda cond, prev_map, curr_map: 0.0,  # 8+ (no function)
    ]

    # Map indices to functions using `switch`
    def compute_value(func_idx, cond_value, _prev_env_map, _curr_env_map):
        func_idx = func_idx.astype(jnp.int32)
        reward_values = vmap(lambda idx: jax.lax.switch(idx, reward_funcs, cond_value, _prev_env_map, _curr_env_map))(func_idx)
        return jnp.sum(reward_values)

    compute_reward_vmap = vmap(compute_value, in_axes=(0, 0, 0, 0))
    rewards = compute_reward_vmap(reward_i, condition, prev_env_map, curr_env_map)

    # clip
    rewards = jnp.clip(rewards, -2, 2)

    return jax.lax.stop_gradient(rewards)
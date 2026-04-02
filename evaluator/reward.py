from functools import partial

import jax
from jax import vmap
import chex
import jax.numpy as jnp


from .rewards import *
from .rewards.multigame_placement import get_multigame_tile_placement_reward
from .weights import RewardWeight, RewardBias


@partial(jax.jit, static_argnums=(4,))
def get_reward_batch(
    reward_i: chex.Array,
    condition: chex.Array,
    prev_env_map: chex.Array,
    curr_env_map: chex.Array,
    map_size: chex.Array = 16,
) -> chex.Array:
    """Compute batch rewards by mapping indices to reward functions and executing them in parallel.

    reward_i 는 다음 인덱스를 따른다.

    0: none
    1: region
    2: path_length
    3: interactive placement (multigame — 개수 + 배치품질)
    4: hazard placement (multigame)
    5: collectable placement (multigame)
    6+: none
    """
    # List of reward functions
    reward_funcs = [
        # 0: region
        lambda cond, prev_map, curr_map: get_region_reward(
            prev_map, curr_map, cond[0]
        ) * RewardWeight.REGION + RewardBias.REGION,

        # 1: path length
        lambda cond, prev_map, curr_map: get_path_length_reward(
            prev_map, curr_map, cond[1]
        ) * RewardWeight.PATH_LENGTH + RewardBias.PATH_LENGTH,

        # 2: interactive placement (개수 + cluster/access/spread)
        lambda cond, prev_map, curr_map: get_multigame_tile_placement_reward(
            prev_map, curr_map, cond[2], tile_name="interactive"
        ) * RewardWeight.MONSTER,

        # 3: hazard placement
        lambda cond, prev_map, curr_map: get_multigame_tile_placement_reward(
            prev_map, curr_map, cond[3], tile_name="hazard"
        ) * RewardWeight.MONSTER,

        # 4: collectable placement
        lambda cond, prev_map, curr_map: get_multigame_tile_placement_reward(
            prev_map, curr_map, cond[4], tile_name="collectable"
        ) * RewardWeight.MONSTER,
    ]

    # Map indices to functions using `switch`
    def compute_value(func_idx, cond_value, _prev_env_map, _curr_env_map):
        func_idx = func_idx.astype(jnp.int32)
        reward_values = vmap(
            lambda idx: jax.lax.switch(idx, reward_funcs, cond_value, _prev_env_map, _curr_env_map)
        )(func_idx)
        return jnp.sum(reward_values)

    compute_reward_vmap = vmap(compute_value, in_axes=(0, 0, 0, 0))
    rewards = compute_reward_vmap(reward_i, condition, prev_env_map, curr_env_map)

    # clip
    rewards = jnp.clip(rewards, -2, 2)

    return jax.lax.stop_gradient(rewards)
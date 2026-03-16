from functools import partial

import jax
from jax import vmap
import chex
import jax.numpy as jnp

from envs.probs.dungeon3 import Dungeon3Tiles

from .fitnesses import *
from .measures import get_amount
from .weights import FitnessWeight

@partial(jax.jit, static_argnums=())
def get_fitness_batch(
    reward_i: chex.Array,
    condition: chex.Array,
    curr_env_map: chex.Array,
    normal_weights: chex.Array,
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
    fitness_funcs = [
        lambda cond, curr_map: 0.0,  # 0
        lambda cond, curr_map: get_region_fitness(curr_map, cond[0]) * FitnessWeight.REGION,  # 1 (region)
        lambda cond, curr_map: get_path_length_fitness(curr_map, cond[1]) * FitnessWeight.PATH_LENGTH,  # 2 (diameter)
        lambda cond, curr_map: get_amount_fitness(curr_map, cond[2], Dungeon3Tiles.WALL) * FitnessWeight.WALL,  # 3 (block)
        lambda cond, curr_map: get_amount_fitness(curr_map, cond[3], Dungeon3Tiles.BAT) * FitnessWeight.MONSTER,  # 4 (bat_amount)
        lambda cond, curr_map: get_direction_fitness(curr_map, cond[4], Dungeon3Tiles.BAT) * FitnessWeight.DIRECTION,  # 5 (bat_direction)
        lambda cond, curr_map: 0.0,  # 6+
    ]

    # Map indices to functions using `switch`
    def compute_value(func_idx, cond_value, _curr_env_map):
        reward_values = jax.vmap(lambda idx: jax.lax.switch(idx, fitness_funcs, cond_value, _curr_env_map))(
            func_idx)
        return jnp.sum(reward_values)

    compute_reward_vmap = vmap(compute_value, in_axes=(0, 0, 0))
    rewards = compute_reward_vmap(reward_i, condition, curr_env_map)

    # multiply by weights
    rewards = jnp.divide(rewards, normal_weights)

    return jax.lax.stop_gradient(rewards)


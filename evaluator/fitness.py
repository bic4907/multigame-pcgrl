from functools import partial

import jax
from jax import vmap
import chex
import jax.numpy as jnp

from .fitnesses import *
from .measures.doom import DoomTile, DoomPassible
from .measures.dungeon import DungeonTile, DungeonPassible
from .measures.pokemon import PokemonTile, PokemonPassible
from .measures.sokoban import SokobanTile, SokobanPassible
from .measures.zelda import ZeldaTile, ZeldaPassible
from .weights import FitnessWeight


@partial(jax.jit, static_argnums=())
def get_doom_fitness_batch(
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
        lambda cond, curr_map: get_region_fitness(curr_map, cond[0], DoomPassible) * FitnessWeight.REGION,  # 0 (region)
        lambda cond, curr_map: get_path_length_fitness(curr_map, cond[1], DoomPassible) * FitnessWeight.PATH_LENGTH,  # 1 (diameter)
        lambda cond, curr_map: 0.0,  # 2
        lambda cond, curr_map: 0.0,  # 3
        lambda cond, curr_map: 0.0,  # 4
        lambda cond, curr_map: 0.0,  # 5+
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


@partial(jax.jit, static_argnums=())
def get_dungeon_fitness_batch(
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
        lambda cond, curr_map: get_region_fitness(curr_map, cond[0], DungeonPassible) * FitnessWeight.REGION,  # 0 (region)
        lambda cond, curr_map: get_path_length_fitness(curr_map, cond[1], DungeonPassible) * FitnessWeight.PATH_LENGTH,  # 1 (diameter)
        lambda cond, curr_map: 0.0,  # 2
        lambda cond, curr_map: 0.0,  # 3
        lambda cond, curr_map: 0.0,  # 4
        lambda cond, curr_map: 0.0,  # 5+
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


@partial(jax.jit, static_argnums=())
def get_pokemon_fitness_batch(
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
        lambda cond, curr_map: get_region_fitness(curr_map, cond[0], PokemonPassible) * FitnessWeight.REGION,  # 0 (region)
        lambda cond, curr_map: get_path_length_fitness(curr_map, cond[1], PokemonPassible) * FitnessWeight.PATH_LENGTH,  # 1 (diameter)
        lambda cond, curr_map: 0.0,  # 2
        lambda cond, curr_map: 0.0,  # 3
        lambda cond, curr_map: 0.0,  # 4
        lambda cond, curr_map: 0.0,  # 5+
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


@partial(jax.jit, static_argnums=())
def get_sokoban_fitness_batch(
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
        lambda cond, curr_map: get_region_fitness(curr_map, cond[0], SokobanPassible) * FitnessWeight.REGION,  # 0 (region)
        lambda cond, curr_map: get_path_length_fitness(curr_map, cond[1], SokobanPassible) * FitnessWeight.PATH_LENGTH,  # 1 (diameter)
        lambda cond, curr_map: 0.0,  # 2
        lambda cond, curr_map: 0.0,  # 3
        lambda cond, curr_map: 0.0,  # 4
        lambda cond, curr_map: 0.0,  # 5+
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


@partial(jax.jit, static_argnums=())
def get_zelda_fitness_batch(
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
        lambda cond, curr_map: get_region_fitness(curr_map, cond[0], ZeldaPassible) * FitnessWeight.REGION,  # 0 (region)
        lambda cond, curr_map: get_path_length_fitness(curr_map, cond[1], ZeldaPassible) * FitnessWeight.PATH_LENGTH,  # 1 (diameter)
        lambda cond, curr_map: 0.0,  # 2
        lambda cond, curr_map: 0.0,  # 3
        lambda cond, curr_map: 0.0,  # 4
        lambda cond, curr_map: 0.0,  # 5+
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
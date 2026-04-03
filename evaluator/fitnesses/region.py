import chex
import jax.numpy as jnp

from envs.probs.dungeon3 import Dungeon3Passible

from evaluator.losses import region_loss

def get_region_fitness(
    curr_env_map: chex.Array,
    cond: chex.Array,
    passable_tiles: chex.Array = Dungeon3Passible,
) -> chex.Array:
    """
    Simple fit reward function for the current map.

    Args:
        curr_env_map (chex.Array): Map state
        cond (chex.Array): User-intended number of regions
        passable_tiles (chex.Array): Types of tiles to ignore

    Returns:
        chex.Array: Reward value
    """

    curr_loss = region_loss(curr_env_map, cond, passable_tiles)
    reward = curr_loss.astype(float)

    # normalize the region reward
    # reward = jnp.divide(reward, weights)

    return reward

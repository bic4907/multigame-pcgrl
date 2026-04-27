import chex
import jax.numpy as jnp

from envs.probs.multigame import MultigamePassable
from evaluator.losses import path_length_loss, region_loss


def get_path_length_fitness(
    curr_env_map: chex.Array,
    cond: chex.Array,
    passable_tiles: chex.Array = MultigamePassable,
):
    """
    Simple fit reward function for the current map.

    Args:
        curr_env_map (chex.Array): Map state
        cond (chex.Array): User-intended path length
        passable_tiles (chex.Array): Types of tiles to ignore

    Returns:
        chex.Array: Reward value
    """


    curr_loss = path_length_loss(curr_env_map, cond, passable_tiles)
    curr_r_loss = region_loss(curr_env_map, 1, passable_tiles)

    reward = curr_loss
    reward += curr_r_loss * 0.5
    reward = reward.astype(float)

    # normalize the path length reward
    # reward = jnp.divide(reward, weights)

    return reward

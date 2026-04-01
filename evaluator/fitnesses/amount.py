import chex
import jax.numpy as jnp

from envs.probs.dungeon3 import Dungeon3Tiles
from evaluator.losses import amount_loss


def get_amount_fitness(
    curr_env_map: chex.Array,
    cond: chex.Array,
    tile_type: chex.Array,
) -> chex.Array:
    """
    Simple fit reward function for the current map.

    Args:
        curr_env_map (chex.Array): Map state
        cond (chex.Array): Desired direction by the user
        tile_type (Dungeon3Tiles): Tile type to aggregate
        rows (int, optional): Map size (rows, default: 16)
        cols (int, optional): Map size (columns, default: 16)

    Returns:
        chex.Array: Reward value
    """


    curr_loss = amount_loss(curr_env_map, tile_type, cond, absolute=False)

    reward = curr_loss
    reward = reward.astype(float)

    # reward = jnp.divide(reward, weights)

    return reward


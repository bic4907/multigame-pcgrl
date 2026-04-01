import chex
import jax.numpy as jnp

from evaluator.losses import direction_loss
from envs.probs.dungeon3 import Dungeon3Tiles


def get_direction_fitness(
    curr_env_map: chex.Array,
    cond: chex.Array,
    tile_type: chex.Array,
    rows: int = 16,
    cols: int = 16,
):
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


    curr_loss = direction_loss(curr_env_map, tile_type, cond, rows, cols)

    reward = curr_loss.astype(float)

    # reward = jnp.divide(reward, weights)

    return reward

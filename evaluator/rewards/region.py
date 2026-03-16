import chex
import jax.numpy as jnp

from envs.probs.dungeon3 import Dungeon3Passible

from evaluator.losses import region_loss


def get_region_reward(
    prev_env_map: chex.Array,
    curr_env_map: chex.Array,
    cond: chex.Array,
    passable_tiles: chex.Array = Dungeon3Passible,
) -> chex.Array:
    """
    Function to evaluate the number of independent regions.

    Args:
        prev_env_map (chex.Array): Previous map state
        curr_env_map (chex.Array): Current map state
        cond (chex.Array): User-intended number of regions
        passable_tiles (chex.Array): Types of tiles to ignore

    Returns:
        chex.Array: Evaluation result
    """

    prev_loss = jnp.abs(region_loss(prev_env_map, cond, passable_tiles))
    curr_loss = jnp.abs(region_loss(curr_env_map, cond, passable_tiles))

    reward = prev_loss - curr_loss
    reward = reward.astype(float)

    return reward

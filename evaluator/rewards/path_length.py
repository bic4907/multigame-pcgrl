import chex
import jax.numpy as jnp

from envs.probs.dungeon3 import Dungeon3Passible

from evaluator.losses import path_length_loss, region_loss


def get_path_length_reward(
    prev_env_map: chex.Array,
    curr_env_map: chex.Array,
    cond: chex.Array,
    passable_tiles: chex.Array = Dungeon3Passible,
):
    """
    Function to evaluate path length.

    Args:
        prev_env_map (chex.Array): Previous map state
        curr_env_map (chex.Array): Current map state
        cond (chex.Array): User-intended value (path length)
        passable_tiles (chex.Array): Types of tiles to ignore

    Returns:
        chex.Array: Path value with respect to path length
    """


    prev_loss = jnp.abs(path_length_loss(prev_env_map, cond, passable_tiles))
    curr_loss = jnp.abs(path_length_loss(curr_env_map, cond, passable_tiles))

    prev_r_loss = jnp.abs(region_loss(prev_env_map, 1, passable_tiles))
    curr_r_loss = jnp.abs(region_loss(curr_env_map, 1, passable_tiles))

    reward = prev_loss - curr_loss
    reward += (prev_r_loss - curr_r_loss) * 0.5
    reward = reward.astype(float)

    reward = jnp.clip(reward, -10, 10)

    return reward

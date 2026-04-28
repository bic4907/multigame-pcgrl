import chex
import jax.numpy as jnp

from envs.probs.multigame import MultigamePassable

from ..measures import get_path_length


def path_length_loss(
    env_map: chex.Array,
    cond: chex.Array,
    passable_tiles: chex.Array = MultigamePassable,
):
    """
    Loss function for path length.

    Args:
        env_map (chex.Array): Map to aggregate
        cond (chex.Array): User-intended path length
        passable_tiles (chex.Array): Types of tiles to ignore

    Returns:
        chex.Array: Loss value for the path length metric
    """

    path_length = get_path_length(env_map, passable_tiles).astype(float)

    loss = jnp.subtract(path_length, cond)

    return loss

import chex
import jax.numpy as jnp

from envs.probs.dungeon3 import Dungeon3Passible

from ..measures import get_region

def region_loss(
    env_map: chex.Array,
    cond: chex.Array,
    passable_tiles: chex.Array = Dungeon3Passible,
) -> chex.Array:
    """
    Region loss function.

    Args:
        env_map (chex.Array): Map to aggregate
        cond (chex.Array): User-intended number of regions
        passable_tiles (chex.Array): Types of tiles to ignore

    Returns:
        chex.Array: Loss value for the region metric
    """

    n_regions = get_region(env_map, passable_tiles).astype(float)

    loss = jnp.subtract(n_regions, cond)

    return loss

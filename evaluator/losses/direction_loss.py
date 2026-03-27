import chex
import jax.numpy as jnp

from ..measures import get_direction

def direction_loss(
    env_map: chex.Array,
    tile_type: chex.Array,
    direction: chex.Array,
    rows: int = 16,
    cols: int = 16,
) -> chex.Array:
    """
    Loss function for direction-based metrics.

    Args:
        env_map (chex.Array): Map to aggregate
        tile_type (chex.Array): Tile type to aggregate
        direction (chex.Array): User-intended direction
        rows (int, optional): Map size (number of rows, default: 16)
        cols (int, optional): Map size (number of columns, default: 16)

    Returns:
        chex.Array: Loss value for the direction metric
    """
    # vectorize direction and ensure int32 type
    direction = jnp.array(direction).flatten().astype(jnp.int32)

    tile_counts = get_direction(env_map, tile_type, direction, rows, cols)
    opposite_tile_counts = get_direction(
        env_map, tile_type, ((direction + 2) % 4).astype(jnp.int32), rows, cols
    )

    loss = -tile_counts + opposite_tile_counts * 0.5 #  + penalty

    return loss

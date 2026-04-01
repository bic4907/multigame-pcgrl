import chex
import jax

import jax.numpy as jnp

from evaluator.types.direction import Direction


def generate_direction_map(
    direction: chex.Array,
    rows: int = 16,
    cols: int = 16,
):
    """
    Generates a matrix where the area to be calculated according to the given direction is 1,
    and the area that will not be reflected in the calculation is 0.

    Args:
        direction:
        rows (int, optional): map size (rows, 16 by default)
        cols (int, optional): map size (cols, 16 by default)
    """

    is_direction_sets = [
        lambda _, col: jnp.less(col, jnp.divide(cols, 2)),  # 0 (west)
        lambda row, _: jnp.less(row, jnp.divide(rows, 2)),  # 1 (north)
        lambda _, col: jnp.greater_equal(col, jnp.divide(cols, 2)),  # 2 (east)
        lambda row, _: jnp.greater_equal(row, jnp.divide(rows, 2)),  # 3 (south)
    ]

    direction = direction[0].astype(jnp.int32)
    direction = jnp.clip(direction, 0, 3)
    indices_col, indices_row = jnp.meshgrid(jnp.arange(rows), jnp.arange(cols))

    return jax.vmap(
        lambda row, col: jax.lax.switch(direction, is_direction_sets, row, col),
        in_axes=(0, 0),
    )(
        indices_row.flatten(),
        indices_col.flatten(),
    ).reshape((rows, cols))


def get_direction(
    env_map: chex.Array,
    tile_type: chex.Array,
    direction: chex.Array,
    rows: int = 16,
    cols: int = 16,
):
    """
    Aggregates the number of specified tiles across all areas of the map.

    Args:
        env_map:
        tile_type:
        direction:
        rows (int, optional): map size (rows, 16 by default)
        cols (int, optional): map size (cols, 16 by default)
    """

    env_map = env_map.copy()

    # Solve the problem of recognizing the type as a scalar by using .flatten.
    direction = direction.flatten()
    direction_map = generate_direction_map(direction, rows, cols).astype(float)

    aggregated_map = jnp.where(env_map == tile_type, 1, 0)

    aggregated_count = jnp.sum(direction_map * aggregated_map, dtype=float)

    return aggregated_count

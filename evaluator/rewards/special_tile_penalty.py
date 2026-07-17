"""special tile (INTERACTIVE, HAZARD, COLLECTABLE)   map in  text  text text in
text of  penalty  text, defaulttext as    tile text  0 as  keeptext also text text also text.

penalty = (current map of  special tile total count) * weight   (per env)
→ reward  in  text text "text loss" signal  text.
"""

import chex
import jax
import jax.numpy as jnp

from envs.probs.multigame import MultigameTiles


def _tile_value_or_default(*names: str, default: int = -1) -> int:
    for name in names:
        if hasattr(MultigameTiles, name):
            return int(getattr(MultigameTiles, name))
    return default


_special_tiles_list = [
    _tile_value_or_default("INTERACTIVE", "INTERACTABLE"),
    _tile_value_or_default("HAZARD"),
    _tile_value_or_default("COLLECTABLE", "COLLECTIBLE"),
]
_special_tiles_list = [t for t in _special_tiles_list if t >= 0]
if not _special_tiles_list:
    _special_tiles_list = [-1]
_SPECIAL_TILES = jnp.array(_special_tiles_list, dtype=jnp.int32)


@jax.jit
def get_special_tile_penalty(
    prev_env_map: chex.Array,
    curr_env_map: chex.Array,
    weight: float = 0.01,
    exclude_tiles: chex.Array = jnp.array([-1], dtype=jnp.int32),
) -> chex.Array:
    """special tile count text text in  text  penalty(text = text )  return.

    Parameters
    ----------
    prev_env_map : (H, W) int map — previous text.
    curr_env_map : (H, W) int map — current text.
    weight : tile 1text text text penalty size. default 0.01 (text).

    exclude_tiles : penalty compute in  text tile text list.
        text) [INTERACTIVE, -1, -1]  text INTERACTIVE tile text  penalty in  text.
        default value [-1]   text none and  same.

    Returns
    -------
    scalar  (text = special tile text  → penalty, text = text → reward).
    """
    exclude_tiles = jnp.asarray(exclude_tiles, dtype=jnp.int32)
    excluded_mask = jnp.isin(_SPECIAL_TILES, exclude_tiles)  # (3,)

    prev_counts = jnp.sum(
        prev_env_map[..., None] == _SPECIAL_TILES, axis=(0, 1)
    ).astype(jnp.float32)
    curr_counts = jnp.sum(
        curr_env_map[..., None] == _SPECIAL_TILES, axis=(0, 1)
    ).astype(jnp.float32)

    delta_counts = curr_counts - prev_counts
    delta_counts = jnp.where(excluded_mask, 0.0, delta_counts)
    return jnp.sum(delta_counts) * weight

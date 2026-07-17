"""evaluator/rewards/multigame_amount.py

Multigame tile count based reward (previous map text text also ).
"""
import chex
import jax
from functools import partial

from evaluator.losses.multigame_amount_loss import multigame_amount_loss


@partial(jax.jit, static_argnames=("tile_name",))
def get_multigame_amount_reward(
    prev_env_map: chex.Array,
    curr_env_map: chex.Array,
    cond: chex.Array,
    tile_name: str = "interactive",
) -> chex.Array:
    """previous map text multigame tile count condition text improvement.

    Parameters
    ----------
    prev_env_map : chex.Array
        previous (H, W) integer map.
    curr_env_map : chex.Array
        current (H, W) integer map.
    cond : chex.Array
        texttable tile count.
    tile_name : str
        "interactive", "hazard", "collectable"  during  text.

    Returns
    -------
    chex.Array : reward (text = text, text = text).
    """
    prev_loss = multigame_amount_loss(prev_env_map, tile_name, cond, absolute=True)
    curr_loss = multigame_amount_loss(curr_env_map, tile_name, cond, absolute=True)

    reward = prev_loss - curr_loss
    return reward.astype(float)


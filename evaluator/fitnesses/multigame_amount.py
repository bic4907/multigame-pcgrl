"""evaluator/fitnesses/multigame_amount.py

Multigame tile count based fitness (current map of  condition achievement).
"""
import chex

from evaluator.losses.multigame_amount_loss import multigame_amount_loss


def get_multigame_amount_fitness(
    curr_env_map: chex.Array,
    cond: chex.Array,
    tile_name: str = "interactive",
) -> chex.Array:
    """Signed difference between a multigame tile count and its target in the current map.

    Parameters
    ----------
    curr_env_map : chex.Array
        (H, W) integer map.
    cond : chex.Array
        Measured tile count.
    tile_name : str
        One of "interactive", "hazard", or "collectable".

    Returns
    -------
    chex.Array : fitness value (positive above cond and negative below it).
    """
    curr_loss = multigame_amount_loss(curr_env_map, tile_name, cond, absolute=False)
    return curr_loss.astype(float)

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
    """current map in  multigame tile count and  texttable of  text  (text keep).

    Parameters
    ----------
    curr_env_map : chex.Array
        (H, W) integer map.
    cond : chex.Array
        texttable tile count.
    tile_name : str
        "interactive", "hazard", "collectable"  during  text.

    Returns
    -------
    chex.Array : fitness text (cond text text text, text text).
    """
    curr_loss = multigame_amount_loss(curr_env_map, tile_name, cond, absolute=False)
    return curr_loss.astype(float)


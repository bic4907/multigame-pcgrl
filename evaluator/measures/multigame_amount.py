"""evaluator/measures/multigame_amount.py

MultigameTiles(INTERACTIVE, HAZARD, COLLECTABLE) count  text  utility.

Usage
-----
    from evaluator.measures.multigame_amount import (
        get_multigame_tile_counts,
        get_interactive_count,
        get_hazard_count,
        get_collectable_count,
    )

    counts = get_multigame_tile_counts(env_map)
    # counts: dict{"interactive": int, "hazard": int, "collectable": int}

    n_interactive = get_interactive_count(env_map)
"""
import chex
import jax.numpy as jnp

from envs.probs.multigame import MultigameTiles


def get_interactive_count(env_map: chex.Array) -> jnp.ndarray:
    """INTERACTIVE(3) tile count  returntext."""
    return jnp.sum(env_map == MultigameTiles.INTERACTIVE).astype(float)


def get_hazard_count(env_map: chex.Array) -> jnp.ndarray:
    """HAZARD(4) tile count  returntext."""
    return jnp.sum(env_map == MultigameTiles.HAZARD).astype(float)


def get_collectable_count(env_map: chex.Array) -> jnp.ndarray:
    """COLLECTABLE(5) tile count  returntext."""
    return jnp.sum(env_map == MultigameTiles.COLLECTABLE).astype(float)


def get_multigame_tile_counts(env_map: chex.Array) -> dict:
    """INTERACTIVE, HAZARD, COLLECTABLE count  text text in  returntext.

    Parameters
    ----------
    env_map : chex.Array
        (H, W) integer array. text  MultigameTiles enum.

    Returns
    -------
    dict with keys "interactive", "hazard", "collectable"
        each text  jnp float scalar.
    """
    return {
        "interactive": get_interactive_count(env_map),
        "hazard": get_hazard_count(env_map),
        "collectable": get_collectable_count(env_map),
    }


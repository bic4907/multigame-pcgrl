"""evaluator/measures/multigame_amount.py

Utilities for counting MultigameTiles (INTERACTIVE, HAZARD, and COLLECTABLE).

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
    """Return the INTERACTIVE (3) tile count."""
    return jnp.sum(env_map == MultigameTiles.INTERACTIVE).astype(float)


def get_hazard_count(env_map: chex.Array) -> jnp.ndarray:
    """Return the HAZARD (4) tile count."""
    return jnp.sum(env_map == MultigameTiles.HAZARD).astype(float)


def get_collectable_count(env_map: chex.Array) -> jnp.ndarray:
    """Return the COLLECTABLE (5) tile count."""
    return jnp.sum(env_map == MultigameTiles.COLLECTABLE).astype(float)


def get_multigame_tile_counts(env_map: chex.Array) -> dict:
    """Return the INTERACTIVE, HAZARD, and COLLECTABLE counts together.

    Parameters
    ----------
    env_map : chex.Array
        Integer array of shape (H, W) containing MultigameTiles enum values.

    Returns
    -------
    dict with keys "interactive", "hazard", "collectable"
        Each value is a scalar jnp float.
    """
    return {
        "interactive": get_interactive_count(env_map),
        "hazard": get_hazard_count(env_map),
        "collectable": get_collectable_count(env_map),
    }

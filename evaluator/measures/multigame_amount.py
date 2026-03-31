"""evaluator/measures/multigame_amount.py

MultigameTiles(INTERACTIVE, HAZARD, COLLECTABLE) 개수를 세는 유틸.

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
    """INTERACTIVE(3) 타일 개수를 반환한다."""
    return jnp.sum(env_map == MultigameTiles.INTERACTIVE).astype(float)


def get_hazard_count(env_map: chex.Array) -> jnp.ndarray:
    """HAZARD(4) 타일 개수를 반환한다."""
    return jnp.sum(env_map == MultigameTiles.HAZARD).astype(float)


def get_collectable_count(env_map: chex.Array) -> jnp.ndarray:
    """COLLECTABLE(5) 타일 개수를 반환한다."""
    return jnp.sum(env_map == MultigameTiles.COLLECTABLE).astype(float)


def get_multigame_tile_counts(env_map: chex.Array) -> dict:
    """INTERACTIVE, HAZARD, COLLECTABLE 개수를 한 번에 반환한다.

    Parameters
    ----------
    env_map : chex.Array
        (H, W) 정수 배열. 값은 MultigameTiles enum.

    Returns
    -------
    dict with keys "interactive", "hazard", "collectable"
        각 값은 jnp float scalar.
    """
    return {
        "interactive": get_interactive_count(env_map),
        "hazard": get_hazard_count(env_map),
        "collectable": get_collectable_count(env_map),
    }


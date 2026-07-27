"""evaluator/rewards/multigame_placement.py

Multigame reward based on the placement quality of special tiles.

Rewards how the special tiles (INTERACTIVE, HAZARD, COLLECTABLE) are laid out on the map,
rather than how many of them there are.

Reward signals
-----------
1. **repetition batch penalty (cluster penalty)**
   Penalises identical tiles among the 4 neighbours.
   → discourages clumping and repetition.

2. **accessibility reward (accessibility bonus)**
   Rewards a special tile that has at least one passable neighbour
     (EMPTY, HAZARD, COLLECTABLE) among its 4 neighbours. A tile fully enclosed by
   WALL / BORDER / INTERACTIVE scores 0.
→ discourages placements that cannot be reached.
3. **spread reward (spread bonus)**
   Mean pairwise L1 distance between the special tile coordinates.
   A larger mean distance means the tiles are more spread out.
   → discourages piling every item into one spot.

Total reward = w_spread * spread_reward

Everything is JAX jit/vmap friendly.
"""
import chex
import jax
import jax.numpy as jnp
from functools import partial

from envs.probs.multigame import MultigameTiles

# Special tile values
_ITEM_TILES = jnp.array([
    MultigameTiles.INTERACTIVE,
    MultigameTiles.HAZARD,
    MultigameTiles.COLLECTABLE,
], dtype=jnp.int32)

# Passable tiles: EMPTY + HAZARD + COLLECTABLE (INTERACTIVE excluded)
_PASSABLE_TILES = jnp.array([
    MultigameTiles.EMPTY,
    MultigameTiles.HAZARD,
    MultigameTiles.COLLECTABLE,
], dtype=jnp.int32)


# ── 1. repetition batch penalty ────────────────────────────────────────────────────────

def _cluster_penalty(env_map: chex.Array) -> jnp.ndarray:
    """Count, over all special tiles, how many of their 4 neighbours share the same tile.

    A larger value means more clumping; 0 means no special tile touches another.
    """
    H, W = env_map.shape
    is_item = jnp.isin(env_map, _ITEM_TILES)  # (H, W) bool

    # Neighbour shifts — out-of-bounds is filled with -1 (never matches)
    up    = jnp.pad(env_map[:-1, :], ((1, 0), (0, 0)), constant_values=-1)
    down  = jnp.pad(env_map[1:, :],  ((0, 1), (0, 0)), constant_values=-1)
    left  = jnp.pad(env_map[:, :-1], ((0, 0), (1, 0)), constant_values=-1)
    right = jnp.pad(env_map[:, 1:],  ((0, 0), (0, 1)), constant_values=-1)

    same_neighbor = (
        (env_map == up).astype(jnp.int32) +
        (env_map == down).astype(jnp.int32) +
        (env_map == left).astype(jnp.int32) +
        (env_map == right).astype(jnp.int32)
    )  # (H, W) — number of matching neighbours per cell (0-4)

    # Sum over the special tile positions
    penalty = jnp.sum(same_neighbor * is_item).astype(float)
    return penalty


# ── 2. accessibility reward ─────────────────────────────────────────────────────────────

def _accessibility_bonus(env_map: chex.Array) -> jnp.ndarray:
    """Fraction of special tiles with at least one passable neighbour.

    1.0 = every special tile is reachable, 0.0 = none of them are.
    Returns 1.0 when there is no special tile.
    """
    H, W = env_map.shape
    is_item = jnp.isin(env_map, _ITEM_TILES)
    n_items = jnp.sum(is_item).astype(float)

    # Passable mask
    passable = jnp.isin(env_map, _PASSABLE_TILES)

    up    = jnp.pad(passable[:-1, :], ((1, 0), (0, 0)), constant_values=False)
    down  = jnp.pad(passable[1:, :],  ((0, 1), (0, 0)), constant_values=False)
    left  = jnp.pad(passable[:, :-1], ((0, 0), (1, 0)), constant_values=False)
    right = jnp.pad(passable[:, 1:],  ((0, 0), (0, 1)), constant_values=False)

    # Number of passable neighbours (padding is False, so the map border
    # behaves like a wall)
    n_passable_neighbors = (
        up.astype(jnp.int32) + down.astype(jnp.int32) +
        left.astype(jnp.int32) + right.astype(jnp.int32)
    )

    # A special tile with at least one passable neighbour counts as accessible
    accessible = ((n_passable_neighbors >= 1) & is_item).astype(float)
    n_accessible = jnp.sum(accessible)

    # No special tiles → 1.0 (no penalty)
    bonus = jnp.where(n_items > 0, n_accessible / n_items, 1.0)
    return bonus


# ── 3. spread reward ───────────────────────────────────────────────────────────────

def _spread_bonus(env_map: chex.Array, max_items: int = 32) -> jnp.ndarray:
    """Mean pairwise L1 distance between the special tiles (normalised by map size).

    Close to 1 means widely spread, close to 0 means tightly clustered.
    Returns 0.0 when there is at most one special tile (spread is undefined).

    max_items : fixed array size used to bound the number of tracked tiles.
    """
    H, W = env_map.shape
    max_dist = (H - 1.0) + (W - 1.0)  # Maximum L1 distance across the map

    is_item = jnp.isin(env_map, _ITEM_TILES)
    n_items = jnp.sum(is_item).astype(jnp.int32)

    # Extract the special tile coordinates into a fixed (max_items, 2) array
    rows, cols = jnp.where(is_item, size=max_items, fill_value=-1)
    coords = jnp.stack([rows, cols], axis=-1)  # (max_items, 2)

    # Valid mask: coordinates other than -1
    valid = (coords[:, 0] >= 0)  # (max_items,)

    # Pairwise L1 distances — (max_items, max_items)
    diff = jnp.abs(coords[:, None, :] - coords[None, :, :])  # (M, M, 2)
    pairwise_l1 = jnp.sum(diff, axis=-1)  # (M, M)

    # Valid pairs only (i != j)
    valid_pair = valid[:, None] & valid[None, :]  # (M, M)
    diag_mask = ~jnp.eye(max_items, dtype=bool)
    valid_pair = valid_pair & diag_mask

    n_pairs = jnp.sum(valid_pair).astype(float)
    total_dist = jnp.sum(pairwise_l1 * valid_pair).astype(float)

    mean_dist = jnp.where(n_pairs > 0, total_dist / n_pairs, 0.0)
    # Normalize to [0, 1] by dividing by max_dist
    bonus = jnp.where(max_dist > 0, mean_dist / max_dist, 0.0)

    # Spread is undefined for a single tile
    bonus = jnp.where(n_items > 1, bonus, 0.0)
    return bonus


# ── Public reward function ───────────────────────────────────────────────────────

@partial(jax.jit, static_argnames=("max_items",))
def get_multigame_placement_reward(
    prev_env_map: chex.Array,
    curr_env_map: chex.Array,
    w_spread: float = 1.0,
    max_items: int = 32,
) -> chex.Array:
    """Improvement in placement quality relative to the previous map.

    Parameters
    ----------
    prev_env_map, curr_env_map : chex.Array
        (H, W) integer map.
    w_spread : float
        Weight of the spread reward; the other terms are unused.
    max_items : int
        Fixed array size for the spread computation (upper bound on tracked tiles).

    Returns
    -------
    chex.Array : scalar reward (positive means improvement).
    """
    # ── Previous score ──
    prev_spread  = _spread_bonus(prev_env_map, max_items)

    # ── Current score ──
    curr_spread  = _spread_bonus(curr_env_map, max_items)

    # spread: improvement over the previous map → curr - prev
    spread_reward = (curr_spread - prev_spread)

    reward = w_spread * spread_reward
    return reward.astype(float)


# ── Export individual measures as well ───────────────────────────────────────

cluster_penalty = jax.jit(_cluster_penalty)
accessibility_bonus = jax.jit(_accessibility_bonus)
spread_bonus = jax.jit(partial(_spread_bonus, max_items=32), static_argnames=("max_items",))


# ══════════════════════════════════════════════════════════════════════════════
#  Tile-specific placement reward
#  — evaluated separately for interactive / hazard / collectable
#    combining count (amount) with placement quality (cluster / access / spread)
# ══════════════════════════════════════════════════════════════════════════════

_TILE_VALUE = {
    "interactive": int(MultigameTiles.INTERACTIVE),
    "hazard":      int(MultigameTiles.HAZARD),
    "collectable": int(MultigameTiles.COLLECTABLE),
}


def _cluster_penalty_tile(env_map: chex.Array, tile_val: int) -> jnp.ndarray:
    """Count how many of the 4 neighbours share the same tile, summed over that tile type."""
    is_target = (env_map == tile_val)

    up    = jnp.pad(env_map[:-1, :], ((1, 0), (0, 0)), constant_values=-1)
    down  = jnp.pad(env_map[1:, :],  ((0, 1), (0, 0)), constant_values=-1)
    left  = jnp.pad(env_map[:, :-1], ((0, 0), (1, 0)), constant_values=-1)
    right = jnp.pad(env_map[:, 1:],  ((0, 0), (0, 1)), constant_values=-1)

    same_neighbor = (
        (env_map == up).astype(jnp.int32) +
        (env_map == down).astype(jnp.int32) +
        (env_map == left).astype(jnp.int32) +
        (env_map == right).astype(jnp.int32)
    )
    return jnp.sum(same_neighbor * is_target).astype(float)


def _accessibility_bonus_tile(env_map: chex.Array, tile_val: int) -> jnp.ndarray:
    """Fraction of the given tile type with at least one passable neighbour."""
    is_target = (env_map == tile_val)
    n_targets = jnp.sum(is_target).astype(float)

    passable = jnp.isin(env_map, _PASSABLE_TILES)

    up    = jnp.pad(passable[:-1, :], ((1, 0), (0, 0)), constant_values=False)
    down  = jnp.pad(passable[1:, :],  ((0, 1), (0, 0)), constant_values=False)
    left  = jnp.pad(passable[:, :-1], ((0, 0), (1, 0)), constant_values=False)
    right = jnp.pad(passable[:, 1:],  ((0, 0), (0, 1)), constant_values=False)

    n_passable_neighbors = (
        up.astype(jnp.int32) + down.astype(jnp.int32) +
        left.astype(jnp.int32) + right.astype(jnp.int32)
    )

    accessible = ((n_passable_neighbors >= 1) & is_target).astype(float)
    n_accessible = jnp.sum(accessible)
    return jnp.where(n_targets > 0, n_accessible / n_targets, 1.0)


def _spread_bonus_tile(env_map: chex.Array, tile_val: int, max_items: int = 32) -> jnp.ndarray:
    """Mean pairwise L1 distance between tiles of the given type (normalised by map size)."""
    H, W = env_map.shape
    max_dist = (H - 1.0) + (W - 1.0)

    is_target = (env_map == tile_val)
    n_targets = jnp.sum(is_target).astype(jnp.int32)

    rows, cols = jnp.where(is_target, size=max_items, fill_value=-1)
    coords = jnp.stack([rows, cols], axis=-1)

    valid = (coords[:, 0] >= 0)
    diff = jnp.abs(coords[:, None, :] - coords[None, :, :])
    pairwise_l1 = jnp.sum(diff, axis=-1)

    valid_pair = valid[:, None] & valid[None, :] & ~jnp.eye(max_items, dtype=bool)
    n_pairs = jnp.sum(valid_pair).astype(float)
    total_dist = jnp.sum(pairwise_l1 * valid_pair).astype(float)

    mean_dist = jnp.where(n_pairs > 0, total_dist / n_pairs, 0.0)
    bonus = jnp.where(max_dist > 0, mean_dist / max_dist, 0.0)
    return jnp.where(n_targets > 1, bonus, 0.0)


def _tile_amount_diff(prev_env_map: chex.Array, curr_env_map: chex.Array,
                      tile_val: int, cond: chex.Array) -> jnp.ndarray:
    """Improvement in satisfying the tile-count condition (prev_loss - curr_loss)."""
    prev_count = jnp.sum(prev_env_map == tile_val).astype(float)
    curr_count = jnp.sum(curr_env_map == tile_val).astype(float)
    prev_loss = jnp.abs(prev_count - cond)
    curr_loss = jnp.abs(curr_count - cond)
    return prev_loss - curr_loss


@partial(jax.jit, static_argnames=("tile_name", "max_items"))
def get_multigame_tile_placement_reward(
    prev_env_map: chex.Array,
    curr_env_map: chex.Array,
    cond: chex.Array,
    tile_name: str = "interactive",
    w_amount: float = 0.4,
    w_spread: float = 0.2,
    max_items: int = 32,
) -> chex.Array:
    """Reward combining the count and the placement quality of one tile type.

    Parameters
    ----------
    prev_env_map, curr_env_map : (H, W) int map.
    cond : scalar target tile count.
    cond : scalar — target tile count.
    w_amount  : weight for satisfying the count condition.
    w_amount  : weight of the count condition.
    max_items : spread compute for  fixed array size.

    Returns
    -------
    scalar reward (higher is better).
    """
    tile_val = _TILE_VALUE[tile_name]

    # ── amount ──
    amount_reward = _tile_amount_diff(prev_env_map, curr_env_map, tile_val, cond)

    # ── spread (improvement over the previous map → curr − prev) ──
    spread_reward = (
        _spread_bonus_tile(curr_env_map, tile_val, max_items)
        - _spread_bonus_tile(prev_env_map, tile_val, max_items)
    )

    reward = (
        w_amount  * amount_reward  +
        w_spread  * spread_reward
    )
    return reward.astype(float)

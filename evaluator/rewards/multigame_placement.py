"""evaluator/rewards/multigame_placement.py

Multigame text text batch quality based reward.

text text(INTERACTIVE, HAZARD, COLLECTABLE)  map in  batchtext text,
text count  text **batch text**text text reward  text.

text  text signal
-----------
1. **repetition batch penalty (cluster penalty)**
   4text  text  during  same tile  text penalty.
   → text/text text.

2. **accessibility reward (accessibility bonus)**
   text text adjacent 4text  during  passage available tile(EMPTY, HAZARD, COLLECTABLE)
     1text or more text reward. text  WALL/BORDER/INTERACTIVE  to  text text  reward 0.
   → reach unavailabletext text in  text text batch text.

3. **spread reward (spread bonus)**
   text text tile coordinate of  text L1 distance mean.
   distance  text text text text  text.
   → text text in  text text also text text also .

text reward = w_spread * spread_reward

text JAX jit/vmap text.
"""
import chex
import jax
import jax.numpy as jnp
from functools import partial

from envs.probs.multigame import MultigameTiles

# text text tile text
_ITEM_TILES = jnp.array([
    MultigameTiles.INTERACTIVE,
    MultigameTiles.HAZARD,
    MultigameTiles.COLLECTABLE,
], dtype=jnp.int32)

# passage available tile: EMPTY + HAZARD + COLLECTABLE (INTERACTIVE text)
_PASSABLE_TILES = jnp.array([
    MultigameTiles.EMPTY,
    MultigameTiles.HAZARD,
    MultigameTiles.COLLECTABLE,
], dtype=jnp.int32)


# ── 1. repetition batch penalty ────────────────────────────────────────────────────────

def _cluster_penalty(env_map: chex.Array) -> jnp.ndarray:
    """text text tile of  4text  text  during  same tile text of  text.

    text  text text  text. 0 text text text text  text batch.
    """
    H, W = env_map.shape
    is_item = jnp.isin(env_map, _ITEM_TILES)  # (H, W) bool

    # text text — text outside  -1(text text )
    up    = jnp.pad(env_map[:-1, :], ((1, 0), (0, 0)), constant_values=-1)
    down  = jnp.pad(env_map[1:, :],  ((0, 1), (0, 0)), constant_values=-1)
    left  = jnp.pad(env_map[:, :-1], ((0, 0), (1, 0)), constant_values=-1)
    right = jnp.pad(env_map[:, 1:],  ((0, 0), (0, 1)), constant_values=-1)

    same_neighbor = (
        (env_map == up).astype(jnp.int32) +
        (env_map == down).astype(jnp.int32) +
        (env_map == left).astype(jnp.int32) +
        (env_map == right).astype(jnp.int32)
    )  # (H, W) — each cell of  same  text text (0~4)

    # text text abovetext in text sum
    penalty = jnp.sum(same_neighbor * is_item).astype(float)
    return penalty


# ── 2. accessibility reward ─────────────────────────────────────────────────────────────

def _accessibility_bonus(env_map: chex.Array) -> jnp.ndarray:
    """text text tile  during  4text in  passage available tile  1text or moretext ratio.

    1.0 = text text text text available, 0.0 = text text text text text.
    text text  0text 1.0 return.
    """
    H, W = env_map.shape
    is_item = jnp.isin(env_map, _ITEM_TILES)
    n_items = jnp.sum(is_item).astype(float)

    # passage available text
    passable = jnp.isin(env_map, _PASSABLE_TILES)

    up    = jnp.pad(passable[:-1, :], ((1, 0), (0, 0)), constant_values=False)
    down  = jnp.pad(passable[1:, :],  ((0, 1), (0, 0)), constant_values=False)
    left  = jnp.pad(passable[:, :-1], ((0, 0), (1, 0)), constant_values=False)
    right = jnp.pad(passable[:, 1:],  ((0, 0), (0, 1)), constant_values=False)

    #  text  during  passage availabletext tile text (text text  text text to
    # " text"text text —  text shift text to  text text text text text)
    n_passable_neighbors = (
        up.astype(jnp.int32) + down.astype(jnp.int32) +
        left.astype(jnp.int32) + right.astype(jnp.int32)
    )

    # text text abovetext in   text passage available text ≥ 1  text text available
    accessible = ((n_passable_neighbors >= 1) & is_item).astype(float)
    n_accessible = jnp.sum(accessible)

    # text text 0text → 1.0 (penalty none)
    bonus = jnp.where(n_items > 0, n_accessible / n_items, 1.0)
    return bonus


# ── 3. spread reward ───────────────────────────────────────────────────────────────

def _spread_bonus(env_map: chex.Array, max_items: int = 32) -> jnp.ndarray:
    """text text coordinate text mean L1 distance (map size to  normalize).

    1 in   text text text text, 0 in   text text text.
    text text ≤ 1text 0.0 (spread measure text ).

    max_items : fixed size array  abovetext maximum text text text.
    """
    H, W = env_map.shape
    max_dist = (H - 1.0) + (W - 1.0)  # map texteachtext L1 distance

    is_item = jnp.isin(env_map, _ITEM_TILES)
    n_items = jnp.sum(is_item).astype(jnp.int32)

    # text text coordinate extract — fixed size array (max_items, 2)
    rows, cols = jnp.where(is_item, size=max_items, fill_value=-1)
    coords = jnp.stack([rows, cols], axis=-1)  # (max_items, 2)

    # valid mask: -1  text coordinate
    valid = (coords[:, 0] >= 0)  # (max_items,)

    # text L1 distance — (max_items, max_items)
    diff = jnp.abs(coords[:, None, :] - coords[None, :, :])  # (M, M, 2)
    pairwise_l1 = jnp.sum(diff, axis=-1)  # (M, M)

    # valid text (i != j)
    valid_pair = valid[:, None] & valid[None, :]  # (M, M)
    diag_mask = ~jnp.eye(max_items, dtype=bool)
    valid_pair = valid_pair & diag_mask

    n_pairs = jnp.sum(valid_pair).astype(float)
    total_dist = jnp.sum(pairwise_l1 * valid_pair).astype(float)

    mean_dist = jnp.where(n_pairs > 0, total_dist / n_pairs, 0.0)
    # normalize: max_dist to  text 0~1 range
    bonus = jnp.where(max_dist > 0, mean_dist / max_dist, 0.0)

    # text text ≤ 1text spread measure text
    bonus = jnp.where(n_items > 1, bonus, 0.0)
    return bonus


# ── text reward function ─────────────────────────────────────────────────────────────

@partial(jax.jit, static_argnames=("max_items",))
def get_multigame_placement_reward(
    prev_env_map: chex.Array,
    curr_env_map: chex.Array,
    w_spread: float = 1.0,
    max_items: int = 32,
) -> chex.Array:
    """previous map text text text batch quality improvement.

    Parameters
    ----------
    prev_env_map, curr_env_map : chex.Array
        (H, W) integer map.
    w_spread : float
        spread reward weight. text text in  text.
    max_items : int
        spread compute text fixed array size (map  inside  maximum text text text).

    Returns
    -------
    chex.Array : scalar reward (text = text).
    """
    # ── prev text ──
    prev_spread  = _spread_bonus(prev_env_map, max_items)

    # ── curr text ──
    curr_spread  = _spread_bonus(curr_env_map, max_items)

    # spread: text text text → curr - prev
    spread_reward = (curr_spread - prev_spread)

    reward = w_spread * spread_reward
    return reward.astype(float)


# ── text measure  also  export ────────────────────────────────────────────────────

cluster_penalty = jax.jit(_cluster_penalty)
accessibility_bonus = jax.jit(_accessibility_bonus)
spread_bonus = jax.jit(partial(_spread_bonus, max_items=32), static_argnames=("max_items",))


# ══════════════════════════════════════════════════════════════════════════════
#  tiletext(tile-specific) placement reward
#  — interactive / hazard / collectable eacheach in  text
#    count(amount) + batchquality(cluster/access/spread)   text in  evaluation
# ══════════════════════════════════════════════════════════════════════════════

_TILE_VALUE = {
    "interactive": int(MultigameTiles.INTERACTIVE),
    "hazard":      int(MultigameTiles.HAZARD),
    "collectable": int(MultigameTiles.COLLECTABLE),
}


def _cluster_penalty_tile(env_map: chex.Array, tile_val: int) -> jnp.ndarray:
    """text tile of  4text  text  during  same tile text sum."""
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
    """text tile  during  4text in  passage available tile  1text or moretext ratio."""
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
    """text tile coordinate text mean L1 distance (map size to  normalize)."""
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
    """tile count condition text improvement (prev_loss − curr_loss)."""
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
    """text tile of  count + batch quality  text in  evaluationtext  text reward.

    Parameters
    ----------
    prev_env_map, curr_env_map : (H, W) int map.
    cond : scalar — texttable tile count.
    tile_name : "interactive", "hazard", "collectable".
    w_amount  : count condition text weight.
    w_spread  : spread reward weight.
    max_items : spread compute for  fixed array size.

    Returns
    -------
    scalar reward (text = text).
    """
    tile_val = _TILE_VALUE[tile_name]

    # ── amount ──
    amount_reward = _tile_amount_diff(prev_env_map, curr_env_map, tile_val, cond)

    # ── spread (text text text → curr − prev) ──
    spread_reward = (
        _spread_bonus_tile(curr_env_map, tile_val, max_items)
        - _spread_bonus_tile(prev_env_map, tile_val, max_items)
    )

    reward = (
        w_amount  * amount_reward  +
        w_spread  * spread_reward
    )
    return reward.astype(float)

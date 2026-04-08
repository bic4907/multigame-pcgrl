"""evaluator/rewards/multigame_placement.py

Multigame 아이템 배치 품질 기반 reward.

아이템(INTERACTIVE, HAZARD, COLLECTABLE)을 맵에 배치할 때,
단순 개수가 아니라 **배치 패턴**까지 고려하여 보상을 준다.

세 가지 시그널
-----------
1. **반복 배치 패널티 (cluster penalty)**
   4방향 이웃 중 같은 타일이 있으면 패널티.
   → 일직선/뭉침 방지.

2. **접근성 보상 (accessibility bonus)**
   아이템 인접 4방향 중 통행 가능 타일(EMPTY, INTERACTIVE, HAZARD, COLLECTABLE)
   이 1개 이상이면 보상. 사방이 WALL/BORDER 로 막힌 곳은 보상 0.
   → 도달 불가능한 곳에 아이템 배치 방지.

3. **분산 보상 (spread bonus)**
   아이템 타일 좌표의 쌍별 L1 거리 평균.
   거리가 클수록 골고루 퍼져 있다는 뜻.
   → 한 곳에 몰리지 않도록 유도.

최종 reward = w_spread * spread_reward

모두 JAX jit/vmap 호환.
"""
import chex
import jax
import jax.numpy as jnp
from functools import partial

from envs.probs.multigame import MultigameTiles

# 아이템 타일 값
_ITEM_TILES = jnp.array([
    MultigameTiles.INTERACTIVE,
    MultigameTiles.HAZARD,
    MultigameTiles.COLLECTABLE,
], dtype=jnp.int32)

# 통행 가능 타일: EMPTY + 아이템 타일들
_PASSABLE_TILES = jnp.array([
    MultigameTiles.EMPTY,
    MultigameTiles.INTERACTIVE,
    MultigameTiles.HAZARD,
    MultigameTiles.COLLECTABLE,
], dtype=jnp.int32)


# ── 1. 반복 배치 패널티 ────────────────────────────────────────────────────────

def _cluster_penalty(env_map: chex.Array) -> jnp.ndarray:
    """아이템 타일의 4방향 이웃 중 같은 타일 수의 합.

    값이 클수록 뭉침이 심하다. 0이면 모든 아이템이 고립 배치.
    """
    H, W = env_map.shape
    is_item = jnp.isin(env_map, _ITEM_TILES)  # (H, W) bool

    # 상하좌우 시프트 — 경계 밖은 -1(매칭 불가)
    up    = jnp.pad(env_map[:-1, :], ((1, 0), (0, 0)), constant_values=-1)
    down  = jnp.pad(env_map[1:, :],  ((0, 1), (0, 0)), constant_values=-1)
    left  = jnp.pad(env_map[:, :-1], ((0, 0), (1, 0)), constant_values=-1)
    right = jnp.pad(env_map[:, 1:],  ((0, 0), (0, 1)), constant_values=-1)

    same_neighbor = (
        (env_map == up).astype(jnp.int32) +
        (env_map == down).astype(jnp.int32) +
        (env_map == left).astype(jnp.int32) +
        (env_map == right).astype(jnp.int32)
    )  # (H, W) — 각 셀의 동일 이웃 수 (0~4)

    # 아이템 위치에서만 합산
    penalty = jnp.sum(same_neighbor * is_item).astype(float)
    return penalty


# ── 2. 접근성 보상 ─────────────────────────────────────────────────────────────

def _accessibility_bonus(env_map: chex.Array) -> jnp.ndarray:
    """아이템 타일 중 4방향에 통행 가능 타일이 1개 이상인 비율.

    1.0 = 모든 아이템 접근 가능, 0.0 = 모든 아이템 사방 막힘.
    아이템이 0개면 1.0 반환.
    """
    H, W = env_map.shape
    is_item = jnp.isin(env_map, _ITEM_TILES)
    n_items = jnp.sum(is_item).astype(float)

    # 통행 가능 마스크
    passable = jnp.isin(env_map, _PASSABLE_TILES)

    up    = jnp.pad(passable[:-1, :], ((1, 0), (0, 0)), constant_values=False)
    down  = jnp.pad(passable[1:, :],  ((0, 1), (0, 0)), constant_values=False)
    left  = jnp.pad(passable[:, :-1], ((0, 0), (1, 0)), constant_values=False)
    right = jnp.pad(passable[:, 1:],  ((0, 0), (0, 1)), constant_values=False)

    # 이웃 중 통행 가능한 타일 수 (자기 자신은 제외해야 하므로
    # "이웃"만 카운트 — 이미 shift 했으므로 자기 자신 포함 안 됨)
    n_passable_neighbors = (
        up.astype(jnp.int32) + down.astype(jnp.int32) +
        left.astype(jnp.int32) + right.astype(jnp.int32)
    )

    # 아이템 위치에서 이웃 통행 가능 수 ≥ 1 이면 접근 가능
    accessible = ((n_passable_neighbors >= 1) & is_item).astype(float)
    n_accessible = jnp.sum(accessible)

    # 아이템 0개 → 1.0 (패널티 없음)
    bonus = jnp.where(n_items > 0, n_accessible / n_items, 1.0)
    return bonus


# ── 3. 분산 보상 ───────────────────────────────────────────────────────────────

def _spread_bonus(env_map: chex.Array, max_items: int = 32) -> jnp.ndarray:
    """아이템 좌표 간 평균 L1 거리 (맵 크기로 정규화).

    1에 가까울수록 골고루 퍼져 있고, 0에 가까울수록 뭉쳐 있다.
    아이템 ≤ 1개면 0.0 (분산 측정 불가).

    max_items : 고정 크기 배열을 위한 최대 아이템 수.
    """
    H, W = env_map.shape
    max_dist = (H - 1.0) + (W - 1.0)  # 맵 대각선 L1 거리

    is_item = jnp.isin(env_map, _ITEM_TILES)
    n_items = jnp.sum(is_item).astype(jnp.int32)

    # 아이템 좌표 추출 — 고정 크기 배열 (max_items, 2)
    rows, cols = jnp.where(is_item, size=max_items, fill_value=-1)
    coords = jnp.stack([rows, cols], axis=-1)  # (max_items, 2)

    # valid mask: -1이 아닌 좌표
    valid = (coords[:, 0] >= 0)  # (max_items,)

    # 쌍별 L1 거리 — (max_items, max_items)
    diff = jnp.abs(coords[:, None, :] - coords[None, :, :])  # (M, M, 2)
    pairwise_l1 = jnp.sum(diff, axis=-1)  # (M, M)

    # valid 쌍만 (i != j)
    valid_pair = valid[:, None] & valid[None, :]  # (M, M)
    diag_mask = ~jnp.eye(max_items, dtype=bool)
    valid_pair = valid_pair & diag_mask

    n_pairs = jnp.sum(valid_pair).astype(float)
    total_dist = jnp.sum(pairwise_l1 * valid_pair).astype(float)

    mean_dist = jnp.where(n_pairs > 0, total_dist / n_pairs, 0.0)
    # 정규화: max_dist로 나누어 0~1 범위
    bonus = jnp.where(max_dist > 0, mean_dist / max_dist, 0.0)

    # 아이템 ≤ 1개면 분산 측정 불가
    bonus = jnp.where(n_items > 1, bonus, 0.0)
    return bonus


# ── 통합 보상 함수 ─────────────────────────────────────────────────────────────

@partial(jax.jit, static_argnames=("max_items",))
def get_multigame_placement_reward(
    prev_env_map: chex.Array,
    curr_env_map: chex.Array,
    w_spread: float = 1.0,
    max_items: int = 32,
) -> chex.Array:
    """이전 맵 대비 아이템 배치 품질 개선량.

    Parameters
    ----------
    prev_env_map, curr_env_map : chex.Array
        (H, W) 정수 맵.
    w_spread : float
        분산 보상 가중치. 클수록 쏠림에 민감.
    max_items : int
        spread 계산 시 고정 배열 크기 (맵 내 최대 아이템 수).

    Returns
    -------
    chex.Array : scalar reward (양수 = 개선).
    """
    # ── prev 점수 ──
    prev_spread  = _spread_bonus(prev_env_map, max_items)

    # ── curr 점수 ──
    curr_spread  = _spread_bonus(curr_env_map, max_items)

    # spread: 높을수록 좋음 → curr - prev
    spread_reward = (curr_spread - prev_spread)

    reward = w_spread * spread_reward
    return reward.astype(float)


# ── 개별 measure 도 export ────────────────────────────────────────────────────

cluster_penalty = jax.jit(_cluster_penalty)
accessibility_bonus = jax.jit(_accessibility_bonus)
spread_bonus = jax.jit(partial(_spread_bonus, max_items=32), static_argnames=("max_items",))


# ══════════════════════════════════════════════════════════════════════════════
#  타일별(tile-specific) placement reward
#  — interactive / hazard / collectable 각각에 대해
#    개수(amount) + 배치품질(cluster/access/spread) 를 한번에 평가
# ══════════════════════════════════════════════════════════════════════════════

_TILE_VALUE = {
    "interactive": int(MultigameTiles.INTERACTIVE),
    "hazard":      int(MultigameTiles.HAZARD),
    "collectable": int(MultigameTiles.COLLECTABLE),
}


def _cluster_penalty_tile(env_map: chex.Array, tile_val: int) -> jnp.ndarray:
    """특정 타일의 4방향 이웃 중 같은 타일 수 합산."""
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
    """특정 타일 중 4방향에 통행 가능 타일이 1개 이상인 비율."""
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
    """특정 타일 좌표 간 평균 L1 거리 (맵 크기로 정규화)."""
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
    """타일 개수 조건 달성 개선량 (prev_loss − curr_loss)."""
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
    """특정 타일의 개수 + 배치 품질을 동시에 평가하는 통합 reward.

    Parameters
    ----------
    prev_env_map, curr_env_map : (H, W) int 맵.
    cond : scalar — 목표 타일 개수.
    tile_name : "interactive", "hazard", "collectable".
    w_amount  : 개수 조건 달성 가중치.
    w_spread  : 분산 보상 가중치.
    max_items : spread 계산용 고정 배열 크기.

    Returns
    -------
    scalar reward (양수 = 개선).
    """
    tile_val = _TILE_VALUE[tile_name]

    # ── amount ──
    amount_reward = _tile_amount_diff(prev_env_map, curr_env_map, tile_val, cond)

    # ── spread (높을수록 좋음 → curr − prev) ──
    spread_reward = (
        _spread_bonus_tile(curr_env_map, tile_val, max_items)
        - _spread_bonus_tile(prev_env_map, tile_val, max_items)
    )

    reward = (
        w_amount  * amount_reward  +
        w_spread  * spread_reward
    )
    return reward.astype(float)

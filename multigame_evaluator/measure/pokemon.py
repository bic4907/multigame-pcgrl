"""
multigame_evaluator/measure/pokemon.py
=======================================
Pokemon 게임에 맞는 measure 함수 모음.

타일 정의 (tile_mapping.json 기준)
--------------------------------------------------------------
0  EMPTY   - 빈 공간/경계    → unified 1 (wall)
1  WALL    - 벽              → unified 1 (wall)
2  FLOOR   - 바닥            → unified 0 (empty)
3  ENEMY   - 와일드 포켓몬   → unified 3 (hazard)
4  OBJECT  - 아이템 오브젝트 → unified 4 (collectable)
5  SPAWN   - 플레이어 시작점 → unified 2 (interactive)
6  WATER   - 물              → unified 2 (interactive)
7  FENCE   - 울타리          → unified 1 (wall)
8  TREE    - 나무            → unified 1 (wall)
9  HOUSE   - 건물            → unified 1 (wall)
10 GRASS   - 잔디            → unified 0 (empty)
99 UNKNOWN - 미정의 fallback → unified 0 (empty)

분류 (unified category 기준)
--------------------------------------------------------------
Empty       : FLOOR, GRASS, UNKNOWN      (unified 0)
Wall        : EMPTY, WALL, FENCE, TREE, HOUSE  (unified 1)
Interactable: SPAWN, WATER               (unified 2)
Hazard      : ENEMY                      (unified 3)
Collectable : OBJECT                     (unified 4)

Passable (RG/PL): Empty + Hazard + Collectable
"""
from enum import IntEnum

import chex
import jax
import jax.numpy as jnp

from envs.pathfinding import calc_diameter
from evaluator.types.direction import Direction
from evaluator.utils import init_flood_net


class PokemonTile(IntEnum):
    EMPTY   = 0
    WALL    = 1
    FLOOR   = 2
    ENEMY   = 3
    OBJECT  = 4
    SPAWN   = 5
    WATER   = 6
    FENCE   = 7
    TREE    = 8
    HOUSE   = 9
    GRASS   = 10
    UNKNOWN = 99


PokemonEmpty = jnp.array([
    PokemonTile.FLOOR,
    PokemonTile.GRASS,
    PokemonTile.UNKNOWN,
], dtype=jnp.int32)

PokemonWall = jnp.array([
    PokemonTile.EMPTY,
    PokemonTile.WALL,
    PokemonTile.FENCE,
    PokemonTile.TREE,
    PokemonTile.HOUSE,
], dtype=jnp.int32)

PokemonInteractable = jnp.array([
    PokemonTile.SPAWN,
    PokemonTile.WATER,
], dtype=jnp.int32)

PokemonHazard = jnp.array([
    PokemonTile.ENEMY,
], dtype=jnp.int32)

PokemonCollectable = jnp.array([
    PokemonTile.OBJECT,
], dtype=jnp.int32)

PokemonPassible = jnp.concatenate([PokemonEmpty, PokemonHazard, PokemonCollectable])


def get_amount(
    env_map: chex.Array,
    tile_type: int = PokemonTile.ENEMY,
) -> float:
    """지정한 타일 종류의 개수를 반환한다. 기본값: ENEMY(와일드 포켓몬)."""
    return jnp.sum(env_map == tile_type, dtype=float)


def get_direction(
    env_map: chex.Array,
    tile_type: int = PokemonTile.ENEMY,
    direction: chex.Array = jnp.array([Direction.east]),
    rows: int = 16,
    cols: int = 16,
) -> float:
    """
    지정한 방향의 절반 영역에 존재하는 tile_type 개수를 반환한다.
    기본값: ENEMY 타일, east 방향.
    """
    is_direction_sets = [
        lambda _, col: jnp.less(col, jnp.divide(cols, 2)),           # west
        lambda row, _: jnp.less(row, jnp.divide(rows, 2)),           # north
        lambda _, col: jnp.greater_equal(col, jnp.divide(cols, 2)),  # east
        lambda row, _: jnp.greater_equal(row, jnp.divide(rows, 2)),  # south
    ]

    d = direction.flatten()
    indices_col, indices_row = jnp.meshgrid(jnp.arange(rows), jnp.arange(cols))
    direction_map = jax.vmap(
        lambda row, col: jax.lax.switch(d[0], is_direction_sets, row, col),
        in_axes=(0, 0),
    )(indices_row.flatten(), indices_col.flatten()).reshape((rows, cols)).astype(float)

    aggregated = jnp.where(env_map == tile_type, 1, 0)
    return jnp.sum(direction_map * aggregated, dtype=float)


def get_path_length(
    env_map: chex.Array,
    passable_tiles: chex.Array = None,
) -> float:
    """맵에서 가장 긴 경로(diameter)를 반환한다."""
    if passable_tiles is None:
        passable_tiles = PokemonPassible
    region_net, path_net = init_flood_net(env_map.shape)
    path_length, _, _, _ = calc_diameter(region_net, path_net, env_map, passable_tiles)
    return path_length.astype(float)


def get_region(
    env_map: chex.Array,
    passable_tiles: chex.Array = None,
) -> float:
    """맵에서 연결된 통로 영역의 수를 반환한다."""
    if passable_tiles is None:
        passable_tiles = PokemonPassible
    region_net, path_net = init_flood_net(env_map.shape)
    _, _, n_regions, _ = calc_diameter(region_net, path_net, env_map, passable_tiles)
    return n_regions.astype(float)

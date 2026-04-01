"""
multigame_evaluator/measure/pokemon.py
=======================================
Pokemon 게임에 맞는 measure 함수 모음.

타일 정의 (dataset/multigame/handlers/fdm_game/pokemon.py 기준)
--------------------------------------------------------------
0  EMPTY   - 빈 공간/경계 (Wall)
1  WALL    - 벽 (Wall)
2  FLOOR   - 바닥 (Empty)
3  ENEMY   - 와일드 포켓몬 (Mob — annotation 미사용)
4  OBJECT  - 아이템 오브젝트 (Item)
5  SPAWN   - 플레이어 시작점 (Object)
6  WATER   - 물 (Object)
7  FENCE   - 울타리 (Wall)
8  TREE    - 나무 (Wall)
9  HOUSE   - 건물 (Wall)

분류
--------------------------------------------------------------
Empty   : FLOOR                  (통과 가능 지형)
Wall    : TREE, HOUSE, FENCE     (막힘)
Object  : SPAWN, WATER           (구조물)
Mob     : (없음)
Item    : OBJECT                 (획득 가능)

Passable (RG/PL): Empty + Item
"""
from enum import IntEnum

import chex
import jax
import jax.numpy as jnp

from envs.pathfinding import calc_diameter
from evaluator.types.direction import Direction
from evaluator.utils import init_flood_net


class PokemonTile(IntEnum):
    EMPTY  = 0
    WALL   = 1
    FLOOR  = 2
    ENEMY  = 3
    OBJECT = 4
    SPAWN  = 5
    WATER  = 6
    FENCE  = 7
    TREE   = 8
    HOUSE  = 9


PokemonEmpty = jnp.array([
    PokemonTile.FLOOR,
], dtype=jnp.int32)

PokemonWall = jnp.array([
    PokemonTile.TREE,
    PokemonTile.HOUSE,
    PokemonTile.FENCE,
], dtype=jnp.int32)

PokemonObject = jnp.array([
    PokemonTile.SPAWN,
    PokemonTile.WATER,
], dtype=jnp.int32)

PokemonMob = jnp.array([], dtype=jnp.int32)

PokemonItem = jnp.array([
    PokemonTile.OBJECT,
], dtype=jnp.int32)

PokemonPassible = jnp.concatenate([PokemonEmpty, PokemonItem])

"""
multigame_evaluator/measure/doom.py
====================================
Doom 게임에 맞는 measure 함수 모음.

타일 정의 (dataset/multigame/handlers/vglc_games/doom.py 기준)
--------------------------------------------------------------
0  EMPTY   - 빈 공간/경계 (막힘)
1  WALL    - 벽 (막힘)
2  FLOOR   - 바닥 (통과 가능)
3  ENEMY   - 적 (Mob)
4  SPAWN   - 스폰 포인트 / 출구 (통과 가능)
5  ITEM    - 아이템 (통과 가능)
6  DANGER  - 위험 오브젝트 (Object)
7  DOOR    - 문 (Object)

분류
--------------------------------------------------------------
Empty   : FLOOR, SPAWN           (통과 가능 지형)
Wall    : WALL, EMPTY            (막힘)
Object  : DOOR, DANGER           (구조물)
Mob     : ENEMY                  (적)
Item    : ITEM                   (획득 가능)

Passable (RG/PL): Empty + Item
"""
from enum import IntEnum

import chex
import jax
import jax.numpy as jnp

from envs.pathfinding import calc_diameter
from evaluator.types.direction import Direction
from evaluator.utils import init_flood_net


class DoomTile(IntEnum):
    EMPTY  = 0
    WALL   = 1
    FLOOR  = 2
    ENEMY  = 3
    SPAWN  = 4
    ITEM   = 5
    DANGER = 6
    DOOR   = 7


DoomEmpty = jnp.array([
    DoomTile.FLOOR,
    DoomTile.SPAWN,
], dtype=jnp.int32)

DoomWall = jnp.array([
    DoomTile.WALL,
    DoomTile.EMPTY,
], dtype=jnp.int32)

DoomObject = jnp.array([
    DoomTile.DOOR,
    DoomTile.DANGER,
], dtype=jnp.int32)

DoomMob = jnp.array([
    DoomTile.ENEMY,
], dtype=jnp.int32)

DoomItem = jnp.array([
    DoomTile.ITEM,
], dtype=jnp.int32)

DoomPassible = jnp.concatenate([DoomEmpty, DoomItem])
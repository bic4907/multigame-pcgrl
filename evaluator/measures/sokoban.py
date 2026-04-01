"""
multigame_evaluator/measure/sokoban.py
=======================================
Sokoban (Boxoban) 게임에 맞는 measure 함수 모음.

타일 정의 (dataset/multigame/handlers/boxoban_handler.py 기준)
--------------------------------------------------------------
0  EMPTY   - 빈 바닥 / 목표 칸 (Empty)
1  WALL    - 벽 (Wall)
4  OBJECT  - 상자 (Object)
5  SPAWN   - 플레이어 위치 (Empty)

Note: 타일 ID 2, 3은 Sokoban에서 사용되지 않는다.

분류
--------------------------------------------------------------
Empty   : EMPTY, SPAWN           (통과 가능 지형)
Wall    : WALL                   (막힘)
Object  : OBJECT                 (상자)
Mob     : (없음)
Item    : (없음)

Passable (RG/PL): Empty + Item = Empty
"""
from enum import IntEnum

import chex
import jax
import jax.numpy as jnp

from envs.pathfinding import calc_diameter
from evaluator.types.direction import Direction
from evaluator.utils import init_flood_net


class SokobanTile(IntEnum):
    EMPTY  = 0
    WALL   = 1
    OBJECT = 4   # box
    SPAWN  = 5   # player


SokobanEmpty = jnp.array([
    SokobanTile.EMPTY,
    SokobanTile.SPAWN,
], dtype=jnp.int32)

SokobanWall = jnp.array([
    SokobanTile.WALL,
], dtype=jnp.int32)

SokobanObject = jnp.array([
    SokobanTile.OBJECT,
], dtype=jnp.int32)

SokobanMob = jnp.array([], dtype=jnp.int32)

SokobanItem = jnp.array([], dtype=jnp.int32)

SokobanPassible = jnp.concatenate([SokobanEmpty, SokobanItem])

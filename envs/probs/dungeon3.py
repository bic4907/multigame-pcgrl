from enum import IntEnum
import math
import os
from functools import partial
from typing import Optional

import chex
from flax import struct
import jax
import jax.numpy as jnp
from PIL import Image
import numpy as np


from envs.pathfinding import (
    FloodPath,
    FloodRegions,
    get_max_path_length_static, calc_diameter, get_path_coords_diam,
)
from envs.probs.problem import (
    Problem,
    ProblemState,
    draw_path
)

from envs.utils import idx_dict_to_arr, Tiles

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))


class Dungeon3Tiles(IntEnum):
    BORDER = 0
    EMPTY = 1
    WALL = 2
    BAT = 3


class Dungeon3Metrics(IntEnum):
    PLACEHOLDER = 0


@struct.dataclass
class Dungeon3State(ProblemState):
    flood_count: Optional[chex.Array] = None


Dungeon3Passible = jnp.array(
    [Dungeon3Tiles.EMPTY]
)


class Dungeon3Problem(Problem):
    tile_enum = Dungeon3Tiles
    metrics_enum = Dungeon3Metrics
    ctrl_threshes = np.zeros(len(Dungeon3Metrics))

    tile_probs = {
        Dungeon3Tiles.BORDER: 0.0,
        Dungeon3Tiles.EMPTY: 0.5,
        Dungeon3Tiles.WALL: 0.5,
        Dungeon3Tiles.BAT: 0.02,
    }
    tile_probs = tuple(idx_dict_to_arr(tile_probs))

    tile_nums = [0 for _ in range(len(tile_enum))]
    tile_nums = tuple(tile_nums)

    stat_weights = {
        Dungeon3Metrics.PLACEHOLDER: 0,
    }
    stat_weights = idx_dict_to_arr(stat_weights)

    stat_trgs = {
        Dungeon3Metrics.PLACEHOLDER: 1,
    }

    passable_tiles = Dungeon3Passible

    def __init__(self, map_shape, ctrl_metrics, pinpoints):
        self.flood_path_net = FloodPath()
        self.flood_path_net.init_params(map_shape)
        self.flood_regions_net = FloodRegions()
        self.flood_regions_net.init_params(map_shape)
        self.max_path_len = get_max_path_length_static(map_shape)
        self.n_tiles = math.prod(map_shape)

        stat_trgs = {
            Dungeon3Metrics.PLACEHOLDER: 1,
        }
        self.stat_trgs = idx_dict_to_arr(stat_trgs)
        self.ctrl_threshes[Dungeon3Metrics.PLACEHOLDER] = 1

        super().__init__(
            map_shape=map_shape, ctrl_metrics=ctrl_metrics, pinpoints=pinpoints
        )

    def get_metric_bounds(self, map_shape):
        bounds = [None] * len(Dungeon3Metrics)
        bounds[Dungeon3Metrics.PLACEHOLDER] = [0, 1]
        return jnp.array(bounds)

    def get_path_coords(self, env_map: chex.Array, prob_state: ProblemState):
        _, flood_path_state, _, _ = calc_diameter(
            self.flood_regions_net, self.flood_path_net, env_map, self.passable_tiles
        )

        return (get_path_coords_diam(flood_count=flood_path_state.flood_count,
                                     max_path_len=self.max_path_len),)

    def get_curr_stats(self, env_map: chex.Array):
        stats = jnp.zeros(len(Dungeon3Metrics))  # [0]
        stats = stats.at[Dungeon3Metrics.PLACEHOLDER].set(1)  # [1]
        state = Dungeon3State(stats=stats)  # {stats: [1]}
        return state

    def init_graphics(self):
        self.graphics = {
            Dungeon3Tiles.EMPTY: Image.open(
                f"{__location__}/tile_ims/empty.png"
            ).convert("RGBA"),
            Dungeon3Tiles.WALL: Image.open(
                f"{__location__}/tile_ims/solid.png"
            ).convert("RGBA"),
            Dungeon3Tiles.BORDER: Image.open(
                f"{__location__}/tile_ims/solid.png"
            ).convert("RGBA"),
            Dungeon3Tiles.BAT: Image.open(f"{__location__}/tile_ims/bat.png").convert("RGBA"),
            len(Dungeon3Tiles): Image.open(
                f"{__location__}/tile_ims/path_g.png"
            ).convert("RGBA")
        }
        self.graphics = jnp.array(idx_dict_to_arr(self.graphics))
        super().init_graphics()

    def draw_path(self, lvl_img, env_map, border_size, path_coords_tpl, tile_size):
        assert len(path_coords_tpl) == 1

        lvl_img = draw_path(
            prob=self,
            lvl_img=lvl_img,
            env_map=env_map,
            border_size=border_size,
            path_coords=path_coords_tpl[0],
            tile_size=tile_size,
            im_idx=-1,
        )

        return lvl_img

    @partial(jax.jit, static_argnums=(0, 3))
    def get_cont_obs(self, env_map, condition, raw_obs: bool = False) -> jnp.array:
        # Change the index number when the condition is changed
        _condition = condition[:4]

        mask = jnp.not_equal(_condition, -1).astype(jnp.float32)[:4]
        stats = jnp.full_like(_condition, -1).astype(jnp.float32)[:4]

        # Start of diameter and longest path
        flood_regions_net = FloodRegions()
        flood_regions_net.init_params(env_map.shape)

        flood_path_net = FloodPath()
        flood_path_net.init_params(env_map.shape)

        passable_tiles = Dungeon3Passible

        diameter, _, n_regions, _ = calc_diameter(
            flood_regions_net, flood_path_net, env_map, passable_tiles
        )

        stats = stats.at[0].set(n_regions)
        stats = stats.at[1].set(diameter)

        n_block = jnp.sum(env_map == Dungeon3Tiles.WALL)
        stats = stats.at[2].set(n_block)

        n_bat = jnp.sum(env_map == Dungeon3Tiles.BAT)
        stats = stats.at[3].set(n_bat)


        # Type 1 (the others)
        if raw_obs is False:
            direction = jnp.sign(_condition - stats)
        else:
            direction = _condition

        obs = jnp.where(mask == 1, direction, 0)[:4]

        onehot_cond = condition[4:5]

        def to_onehot(index, num_classes=4):
            """convert index to num_classes size one-hot vector"""
            return jnp.eye(num_classes)[index]

        expanded_onehot = jax.lax.cond(
            jnp.equal(onehot_cond[0], -1),
            lambda _: jnp.zeros((4,)),
            lambda _: to_onehot(onehot_cond[0]),
            operand=None
        )

        obs = jnp.concatenate((obs, expanded_onehot), axis=-1)

        return obs

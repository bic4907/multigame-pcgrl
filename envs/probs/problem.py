from enum import IntEnum
from functools import partial
import math
from typing import Optional
import os
import chex
from flax import struct
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image, ImageFont

from envs.utils import Tiles, create_rgba_circle


@struct.dataclass
class ProblemState:
    stats: Optional[chex.Array] = None
    region_features: Optional[chex.Array] = None

    ctrl_trgs: Optional[chex.Array] = None


def get_reward(
    stats,  # [1]
    old_stats,
    stat_weights,
    stat_trgs,
    ctrl_threshes,
):
    """
    ctrl_threshes: A vector of thresholds for each metric. If the metric is within
        an interval of this size centered at its target value, it has 0 loss.
    """
    prev_loss = jnp.abs(stat_trgs - old_stats)
    prev_loss = jnp.clip(prev_loss - ctrl_threshes, 0)

    loss = jnp.abs(stat_trgs - stats)
    loss = jnp.clip(loss - ctrl_threshes, 0)

    reward = prev_loss - loss
    reward = jnp.where(stat_trgs == jnp.inf, stats - old_stats, reward)
    reward = jnp.where(stat_trgs == -jnp.inf, old_stats - stats, reward)
    reward *= stat_weights
    reward = jnp.sum(reward)

    return reward


def get_max_loss(stat_weights, stat_trgs, ctrl_threshes, metric_bounds):
    stat_trgs = jnp.clip(stat_trgs, metric_bounds[:, 0], metric_bounds[:, 1])
    loss_0 = jnp.abs(stat_trgs - metric_bounds[:, 0])
    loss_1 = jnp.abs(stat_trgs - metric_bounds[:, 1])
    loss = jnp.where(loss_0 < loss_1, loss_1, loss_0)
    loss = jnp.clip(loss - ctrl_threshes, 0)
    loss *= stat_weights
    loss = jnp.sum(loss)
    return loss


def get_loss(stats, stat_weights, stat_trgs, ctrl_threshes, metric_bounds):
    stat_trgs = jnp.clip(stat_trgs, metric_bounds[:, 0], metric_bounds[:, 1])
    loss = jnp.abs(stat_trgs - stats)
    loss = jnp.clip(loss - ctrl_threshes, 0)
    loss *= stat_weights
    loss = jnp.sum(loss)
    return loss


def gen_ctrl_trgs(metric_bounds, rng):
    rng, _ = jax.random.split(rng)
    return jax.random.randint(
        rng, (len(metric_bounds),), metric_bounds[:, 0], metric_bounds[:, 1]
    )


@struct.dataclass
class MapData:
    env_map: chex.Array
    actual_map_shape: chex.Array
    static_map: chex.Array


@partial(
    jax.jit,
    static_argnames=(
        "tile_enum",
        "map_shape",
        "tile_probs",
        "randomize_map_shape",
        "empty_start",
        "tile_nums",
        "pinpoints",
    ),
)
def gen_init_map(
    rng,
    tile_enum,
    map_shape,
    tile_probs,
    randomize_map_shape=False,
    empty_start=False,
    tile_nums=None,
    pinpoints=False,
):
    tile_probs = np.array(tile_probs, dtype=np.float32)

    if empty_start:
        init_map = jnp.full(map_shape, dtype=jnp.int32, fill_value=tile_enum.EMPTY)
    else:
        # Randomly place tiles according to their probabilities tile_probs
        init_map = jax.random.choice(rng, len(tile_enum), shape=map_shape, p=tile_probs)

    if randomize_map_shape:
        # Randomize the actual map size
        actual_map_shape = jax.random.randint(
            rng, (2,), 3, jnp.max(jnp.array(map_shape)) + 1
        )

        # Use jnp.ogrid to create a grid of indices
        oy, ox = jnp.ogrid[: map_shape[0], : map_shape[1]]
        # Use these indices to create a mask where each dimension is less than the corresponding actual_map_shape
        mask = (oy < actual_map_shape[0]) & (ox < actual_map_shape[1])

        # Replace the rest with tile_enum.BORDER
        init_map = jnp.where(mask, init_map, tile_enum.BORDER)

    else:
        actual_map_shape = jnp.array(map_shape)

    if tile_nums is not None and pinpoints:
        non_num_tiles = jnp.array(
            [tile_idx for tile_idx, tile_num in enumerate(tile_nums) if tile_num == 0]
        )
        n_map_cells = math.prod(map_shape)

        def add_num_tiles(carry, tile_idx):
            rng, init_map = carry
            tiles_to_add = tile_nums[tile_idx]

            modifiable_map = (
                jnp.isin(init_map, non_num_tiles) & (init_map != tile_enum.BORDER)
            ).ravel()
            probs = modifiable_map / jnp.sum(modifiable_map)
            add_idxs = jax.random.choice(
                rng, n_map_cells, shape=(tiles_to_add,), p=probs, replace=False
            )

            # Adjust the map
            init_map = init_map.ravel().at[add_idxs].set(tile_idx).reshape(map_shape)

            return (rng, init_map), None

        tile_idxs = np.arange(len(tile_enum))
        # _, init_map = jax.lax.scan(adjust_tile_nums, (rng, init_map), tile_idxs)[0]
        for tile_idx in tile_idxs:
            (rng, init_map), _ = add_num_tiles((rng, init_map), tile_idx)

    return MapData(init_map, actual_map_shape, init_map == tile_enum.BORDER)


class Placeholder(IntEnum):
    pass


class Problem:
    tile_size: int = 16
    stat_weights: chex.Array
    metrics_enum: IntEnum
    region_metrics_enum: IntEnum = Placeholder
    ctrl_metrics: chex.Array
    stat_trgs: chex.Array
    ctrl_threshes: chex.Array = None
    queued_ctrl_trgs: chex.Array = None
    unavailable_tiles: list = list()  # no

    def __init__(self, map_shape, ctrl_metrics, pinpoints):
        self.map_shape = map_shape
        self.metric_bounds = self.get_metric_bounds(map_shape)
        self.ctrl_metrics = np.array(ctrl_metrics, dtype=int)
        self.ctrl_metrics_mask = np.array(
            [i in ctrl_metrics for i in range(len(self.stat_trgs))]
        )

        if self.ctrl_threshes is None:
            self.ctrl_threshes = np.zeros(len(self.stat_trgs))

        self.max_loss = get_max_loss(
            self.stat_weights, self.stat_trgs, self.ctrl_threshes, self.metric_bounds
        )

        # Dummy control observation to placate jax tree map during minibatch creation
        self.ctrl_metric_obs_idxs = (
            np.array([0]) if len(self.ctrl_metrics) == 0 else self.ctrl_metrics
        )

        self.metric_names = [metric.name for metric in self.metrics_enum]
        self.region_metric_names = [metric.name for metric in self.region_metrics_enum]

        self.queued_ctrl_trgs = jnp.zeros(
            len(self.metric_names)
        )  # dummy value to placate jax
        self.has_queued_ctrl_trgs = False

        # Make sure we don't generate pinpoint tiles if they are being treated as such
        tile_probs = np.array(self.tile_probs, dtype=np.float32)
        if self.tile_nums is not None and pinpoints:
            for tile in self.tile_enum:
                if self.tile_nums[tile] > 0:
                    tile_probs[tile] = 0
                # Normalize to make tile_probs sum to 1
            tile_probs = tile_probs / np.sum(tile_probs)
        self.tile_probs = tuple(tile_probs)

    @partial(
        jax.jit,
        static_argnames=("self", "randomize_map_shape", "empty_start", "pinpoints"),
    )
    def gen_init_map(
        self, rng, randomize_map_shape=False, empty_start=False, pinpoints=False
    ):
        return gen_init_map(
            rng,
            self.tile_enum,
            self.map_shape,
            self.tile_probs,
            randomize_map_shape=randomize_map_shape,
            empty_start=empty_start,
            tile_nums=self.tile_nums,
            pinpoints=pinpoints,
        )

    def get_metric_bounds(self, map_shape):
        raise NotImplementedError

    def get_stats(self, env_map: chex.Array, prob_state: ProblemState):
        raise NotImplementedError

    def init_graphics(self):
        self.graphics = jnp.array([np.array(g) for g in self.graphics])
        # Load TTF font (Replace 'path/to/font.ttf' with the actual path)

        current_dir = os.path.dirname(__file__)  # Get the directory of the current file
        font_path = os.path.abspath(
            os.path.join(current_dir, "fonts", "AcPlus_IBM_VGA_9x16-2x.ttf")
        )
        self.render_font = ImageFont.truetype(font_path, 20)

        ascii_chars_to_ints = {}
        self.ascii_chars_to_ims = {}
        self.render_font_shape = (16, 9)


    def observe_ctrls(self, prob_state: ProblemState):
        obs = jnp.zeros(len(self.metrics_enum))
        obs = jnp.where(
            self.ctrl_metrics_mask,
            jnp.sign(prob_state.ctrl_trgs - prob_state.stats),
            obs,
        )
        # Return a vector of only the metrics we're controlling
        obs = obs[self.ctrl_metric_obs_idxs]
        return obs

    def gen_rand_ctrl_trgs(self, rng, actual_map_shape):
        metric_bounds = self.get_metric_bounds(actual_map_shape)
        # Randomly sample some control targets
        ctrl_trgs = gen_ctrl_trgs(metric_bounds, rng)
        # change to float32
        ctrl_trgs = jnp.array(ctrl_trgs, dtype=jnp.float32)

        ctrl_trgs = jnp.where(
            self.ctrl_metrics_mask, ctrl_trgs, self.stat_trgs.astype(jnp.float32)
        )
        return ctrl_trgs

    def reset(self, env_map: chex.Array, rng, queued_state, actual_map_shape):
        ctrl_trgs = jax.lax.select(
            queued_state.has_queued_ctrl_trgs,
            queued_state.ctrl_trgs,
            self.gen_rand_ctrl_trgs(rng, actual_map_shape),
        )

        state = self.get_curr_stats(env_map)
        state = state.replace(
            ctrl_trgs=ctrl_trgs,
        )
        reward = None
        return reward, state

    def step(self, env_map: chex.Array, state: ProblemState):
        # new_stats.stats = [1]
        new_state = self.get_curr_stats(env_map=env_map)
        # jax.debug.print("stats: {stats}", stats=new_state.stats)

        reward = get_reward(
            new_state.stats,
            state.stats,
            self.stat_weights,
            state.ctrl_trgs,
            self.ctrl_threshes,
        )
        new_state = new_state.replace(
            ctrl_trgs=state.ctrl_trgs,
        )
        return reward, new_state

    def get_curr_stats(self, env_map: chex.Array) -> ProblemState:
        raise NotImplementedError

    def draw_path(self, lvl_img, env_map, border_size, path_coords_tpl, tile_size):
        # path_coords_tpl is a tuple of (1) array of of path coordinates
        assert len(path_coords_tpl) == 1
        lvl_img = draw_path(
            prob=self,
            lvl_img=lvl_img,
            env_map=env_map,
            border_size=border_size,
            path_coords=path_coords_tpl[0],
            tile_size=tile_size,
        )
        return lvl_img


def draw_path(prob, lvl_img, env_map, border_size, path_coords, tile_size, im_idx=-1):
    # Path, if applicable
    tile_img = prob.graphics[im_idx]

    def draw_path_tile(carry):
        path_coords, lvl_img, i = carry
        y, x = path_coords[i]
        tile_type = env_map[y + border_size[0]][x + border_size[1]]
        empty_tile = int(Tiles.EMPTY)

        # og_tile = lvl_img[(y + border_size[0]) * tile_size:(y + border_size[0] + 1) * tile_size,
        #                     (x + border_size[1]) * tile_size:(x + border_size[1] + 1) * tile_size, :]
        og_tile = jax.lax.dynamic_slice(
            lvl_img,
            ((y + border_size[0]) * tile_size, (x + border_size[1]) * tile_size, 0),
            (tile_size, tile_size, 4),
        )
        new_tile_img = jnp.where(tile_img[..., -1:] == 0, og_tile, tile_img)

        # Only draw path tiles on top of empty tiles
        lvl_img = jax.lax.cond(
            tile_type == empty_tile,
            lambda: jax.lax.dynamic_update_slice(
                lvl_img,
                new_tile_img,
                ((y + border_size[0]) * tile_size, (x + border_size[1]) * tile_size, 0),
            ),
            lambda: lvl_img,
        )

        return (path_coords, lvl_img, i + 1)

    def cond(carry):
        path_coords, _, i = carry
        return jnp.all(path_coords[i] != jnp.array((-1, -1)))
        # return jnp.all(path_coords[i:i+env.prob.max_path_len+1] != jnp.array(-1, -1))
        # result = jnp.any(
        #     jax.lax.dynamic_slice(path_coords, (i, 0), (prob.max_path_len+1, 2)) != jnp.array((-1, -1))
        # )
        # return result

    i = 0
    _, lvl_img, _ = jax.lax.while_loop(cond, draw_path_tile, (path_coords, lvl_img, i))

    return lvl_img


def alpha_blend(background, overlay):
    """
    Blends an overlay onto the background using the overlay's alpha channel.

    Parameters:
    - background: (h, w, 4) RGBA region from img (alpha assumed to be 255).
    - overlay: (h, w, 4) RGBA region from circle (alpha 0-255).

    Returns:
    - blended_region: (h, w, 4) RGBA blended region.
    """
    # Extract alpha channel from the overlay and normalize to 0-1
    overlay_alpha = overlay[:, :, 3:4] * 0.7 / 255.0  # Shape (h, w, 1)

    # Blend RGB channels
    blended_rgb = (
        overlay_alpha * overlay[:, :, :3] + (1 - overlay_alpha) * background[:, :, :3]
    )

    # Keep the alpha channel of the background as 255
    blended_alpha = jnp.full(
        (background.shape[0], background.shape[1], 1), 255, dtype=jnp.uint8
    )

    # Combine the blended RGB and alpha
    blended_region = jnp.concatenate([blended_rgb, blended_alpha], axis=-1)

    return blended_region.astype(jnp.uint8)


def draw_solutions(lvl_img, solutions, tile_size, border_size=(0, 0)):
    """
    Draw solutions on the level image using JAX operations with `jax.lax.cond`.

    Args:
        lvl_img (PIL.Image): The level image to draw on.
        solutions (object): Contains solution data including paths, offsets, and colors.
        tile_size (int): Size of each tile.
        border_size (tuple): Border size adjustment as (y_border, x_border).

    Returns:
        PIL.Image: The modified level image with solutions drawn.
    """
    NO_PATH = jnp.array([-1, -1])
    border_offset_y, border_offset_x = (
        border_size[0] * tile_size,
        border_size[1] * tile_size,
    )

    def draw_point(index, inputs):
        """
        Draws a single point of the path onto the image.

        Args:
            index (tuple): (solution_idx, path_idx).
            inputs (tuple): (current_img, solutions).

        Returns:
            PIL.Image: Updated image with the point drawn.
        """
        idx, i = index
        img, solutions = inputs
        point = solutions.path[idx][i]
        offset_y, offset_x = solutions.offset[idx]
        color = solutions.color[idx]

        is_image_obj = isinstance(img, Image.Image)
        circle = create_rgba_circle(
            tile_size=tile_size, color=color, alpha=0.5, return_image=is_image_obj
        )

        def no_op(img):
            return img  # If NO_PATH, do nothing

        def draw_circle(img):
            y, x = point
            adjusted_x = x * tile_size + offset_x + border_offset_x
            adjusted_y = y * tile_size + offset_y + border_offset_y

            if isinstance(img, Image.Image) and isinstance(circle, Image.Image):
                img.paste(circle, (adjusted_x, adjusted_y), circle)
            else:
                h, w = circle.shape[:2]
                x, y = adjusted_x, adjusted_y

                # Extract the region to be updated from the img
                region = jax.lax.dynamic_slice(img, (y, x, 0), (h, w, 4))
                region = alpha_blend(region, circle)
                img = jax.lax.dynamic_update_slice(img, region, (y, x, 0))
            return img

        # Use `jax.lax.cond` to decide whether to draw the circle
        return jax.lax.cond(jnp.all(point == NO_PATH), no_op, draw_circle, img)

    def draw_single_solution(idx, img):
        """
        Draws a single solution's path onto the image.

        Args:
            idx (int): The index of the solution.
            img (PIL.Image): The current image to update.

        Returns:
            PIL.Image: Updated image with the solution path drawn.
        """
        path_length = solutions.path[idx].shape[0]
        indices = jnp.arange(path_length)  # Indices for the path points

        def loop_body(i, current_img):
            return draw_point((idx, i), (current_img, solutions))

        return jax.lax.fori_loop(0, path_length, loop_body, img)

    # Loop through all solutions using `jax.lax.fori_loop`
    return jax.lax.fori_loop(0, solutions.n, draw_single_solution, lvl_img)

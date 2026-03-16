from enum import IntEnum
import os
from typing import Optional, Tuple
from flax import struct
import numpy as np
from PIL import Image

from envs.pathfinding import FloodPath, FloodRegions, calc_diameter, get_max_n_regions, \
    get_max_path_length, get_max_path_length_static, get_path_coords_diam
from envs.probs.problem import Problem, ProblemState
from envs.feature import *

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

from envs.utils import idx_dict_to_arr


class BinaryTiles(IntEnum):
    BORDER = 0
    EMPTY = 1
    WALL = 2


@struct.dataclass
class BinaryState(ProblemState):
    flood_count: Optional[chex.Array] = None


class BinaryMetrics(IntEnum):
    DIAMETER = 0

    N_REGIONS = 1

    H_SYMMETRY = 2  # Horizontal Symmetry
    V_SYMMETRY = 3  # Vertical Symmetry
    LR_DIAGONAL_SYMMETRY = 4  # Left-to-Right Diagonal Symmetry
    RL_DIAGONAL_SYMMETRY = 5  # Right-to-Left Diagonal Symmetry

    LARGEST_COMPONENT = 6  # Largest Component Normalized


class BinaryLocalMetrics(IntEnum):
    # empty spaces
    EMPTY_RATE = 0
    # Shannon entropy
    SHANNON_ENTROPY = 1


class BinaryProblem(Problem):
    tile_enum = BinaryTiles

    tile_probs = np.zeros(len(tile_enum))
    tile_probs[BinaryTiles.EMPTY] = 0.5
    tile_probs[BinaryTiles.WALL] = 0.5
    tile_probs = tuple(tile_probs)

    tile_nums = tuple([0 for _ in range(len(tile_enum))])

    stat_weights = np.zeros(len(BinaryMetrics))
    stat_weights[BinaryMetrics.DIAMETER] = 1.0
    stat_weights[BinaryMetrics.N_REGIONS] = 1.0
    stat_weights = jnp.array(stat_weights)

    stat_trgs = np.zeros(len(BinaryMetrics))
    stat_trgs[BinaryMetrics.DIAMETER] = np.inf
    stat_trgs[BinaryMetrics.N_REGIONS] = 1
    stat_trgs = jnp.array(stat_trgs)

    metrics_enum = BinaryMetrics
    region_metrics_enum = BinaryLocalMetrics

    passable_tiles = jnp.array([BinaryTiles.EMPTY])

    def __init__(self, map_shape, ctrl_metrics, pinpoints):
        self.flood_path_net = FloodPath()
        self.flood_path_net.init_params(map_shape)
        self.flood_regions_net = FloodRegions()
        self.flood_regions_net.init_params(map_shape)
        self.max_path_len = get_max_path_length_static(map_shape)
        super().__init__(map_shape=map_shape, ctrl_metrics=ctrl_metrics, pinpoints=pinpoints)

    def get_metric_bounds(self, map_shape):
        bounds = [None] * len(BinaryMetrics)
        bounds[BinaryMetrics.DIAMETER] = [0, get_max_path_length(map_shape)]
        bounds[BinaryMetrics.N_REGIONS] = [0, get_max_n_regions(map_shape)]
        bounds[BinaryMetrics.H_SYMMETRY] = [0, 1]
        bounds[BinaryMetrics.V_SYMMETRY] = [0, 1]
        bounds[BinaryMetrics.LR_DIAGONAL_SYMMETRY] = [0, 1]
        bounds[BinaryMetrics.RL_DIAGONAL_SYMMETRY] = [0, 1]
        # width * height
        bounds[BinaryMetrics.LARGEST_COMPONENT] = [0, map_shape[0] * map_shape[1]]

        # for idx in range(16):
        #     bounds[BinaryMetrics.LC_R1_EMPTY_RATE + idx] = [0, 1]
        #     bounds[BinaryMetrics.LC_R1_ENTROPY + idx] = [0, 1]

        return jnp.array(bounds)

    def get_path_coords(self, env_map: chex.Array, prob_state: BinaryState) -> Tuple[chex.Array]:
        return (get_path_coords_diam(flood_count=prob_state.flood_count, max_path_len=self.max_path_len),)

    def init_graphics(self):
        self.graphics = {
            BinaryTiles.EMPTY: Image.open(
                f"{__location__}/tile_ims/empty.png"
            ).convert('RGBA'),
            BinaryTiles.WALL: Image.open(
                f"{__location__}/tile_ims/solid.png"
            ).convert('RGBA'),
            BinaryTiles.BORDER: Image.open(
                f"{__location__}/tile_ims/solid.png"
            ).convert('RGBA')
        }
        self.graphics = jnp.array(idx_dict_to_arr(self.graphics))
        super().init_graphics()

    def get_curr_stats(self, env_map: chex.Array):
        """Get relevant metrics from the current state of the environment."""
        diameter, flood_path_state, n_regions, flood_regions_state = calc_diameter(
            self.flood_regions_net, self.flood_path_net, env_map, self.passable_tiles
        )

        # Initialize stats array with zeros
        stats = jnp.zeros(len(BinaryMetrics))

        # Set diameter and number of regions
        stats = stats.at[BinaryMetrics.DIAMETER].set(diameter)
        stats = stats.at[BinaryMetrics.N_REGIONS].set(n_regions)

        # Calculate the largest component of WALL tiles
        stats = stats.at[BinaryMetrics.LARGEST_COMPONENT].set(
            get_largest_component_size(flood_regions_state.flood_count))

        # Calculate symmetry metrics (consider only 1 and 2 blocks)
        h_symmetry = calculate_horizontal_symmetry(env_map)
        v_symmetry = calculate_vertical_symmetry(env_map)
        lr_diagonal_symmetry = calculate_lr_diagonal_symmetry(env_map)
        rl_diagonal_symmetry = calculate_rl_diagonal_symmetry(env_map)

        stats = stats.at[BinaryMetrics.H_SYMMETRY].set(h_symmetry)
        stats = stats.at[BinaryMetrics.V_SYMMETRY].set(v_symmetry)
        stats = stats.at[BinaryMetrics.LR_DIAGONAL_SYMMETRY].set(lr_diagonal_symmetry)
        stats = stats.at[BinaryMetrics.RL_DIAGONAL_SYMMETRY].set(rl_diagonal_symmetry)

        region_stats = jnp.zeros((len(BinaryLocalMetrics), 4, 4))
        region_stats = region_stats.at[BinaryLocalMetrics.EMPTY_RATE].set(self.calculate_empty_block_rates(env_map))
        region_stats = region_stats.at[BinaryLocalMetrics.SHANNON_ENTROPY].set(
            calculate_entropy_for_regions(env_map, num_unique_values=self.get_num_unique_tiles()))

        # Return state with the new stats
        state = BinaryState(
            stats=stats, flood_count=flood_path_state.flood_count, ctrl_trgs=None,
            region_features=region_stats,
        )
        return state

    def calculate_empty_block_rates(self, env_map: chex.Array) -> chex.Array:
        """
        Calculate the rate of empty blocks in each 4x4 region, returning a 4x4 array.
        """
        map_height, map_width = env_map.shape

        # Reshape the env_map into blocks of 4x4 regions
        reshaped_map = env_map.reshape(4, map_height // 4, 4, map_width // 4)

        # Calculate the number of empty blocks in each 4x4 region
        empty_blocks = (reshaped_map == BinaryTiles.EMPTY).sum(axis=(1, 3))

        # Calculate the total number of blocks in each region (region_height * region_width)
        total_blocks = (map_height // 4) * (map_width // 4)

        # Compute the empty rate for each region by dividing the number of empty blocks by the total blocks
        empty_rates = empty_blocks / total_blocks

        return empty_rates

    def get_num_unique_tiles(self):
        return len(BinaryTiles) - 1  # Subtract 1 to exclude BORDER

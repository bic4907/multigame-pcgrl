import chex

from envs.pathfinding import calc_diameter
from envs.probs.dungeon3 import Dungeon3Passible

from ..utils import init_flood_net


def get_path_length(
    env_map: chex.Array, passable_tiles: chex.Array = Dungeon3Passible
):
    region_network, path_network = init_flood_net(env_map.shape)
    path_length, _, _, _ = calc_diameter(
        region_network, path_network, env_map, passable_tiles
    )
    path_length = path_length.astype(float)
    return path_length

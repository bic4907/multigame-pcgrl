import chex

from envs.pathfinding import calc_diameter

from ..utils import init_flood_net


def get_region(
    env_map: chex.Array,
    passable_tiles: chex.Array,
):
    region_network, path_network = init_flood_net(env_map.shape)
    _, _, n_regions, _ = calc_diameter(
        region_network, path_network, env_map, passable_tiles
    )

    n_regions = n_regions.astype(float)

    return n_regions

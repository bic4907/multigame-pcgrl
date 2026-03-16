from typing import Tuple
from envs.pathfinding import FloodPath, FloodRegions


def init_flood_net(map_shape: Tuple[int, ...]) -> Tuple[FloodRegions, FloodPath]:
    """
    Function to generate both FloodRegions and FloodPath at once.

    Args:
        map_shape (Tuple[int, ...]): Map size

    Returns:
        Tuple[FloodRegions, FloodPath]: The generated FloodRegions and FloodPath
    """


    flood_regions_net = FloodRegions()
    flood_regions_net.init_params(map_shape)

    flood_path_net = FloodPath()
    flood_path_net.init_params(map_shape)

    return (flood_regions_net, flood_path_net)

import numpy as np


def pairing_maps(prev_env_map: np.ndarray, curr_env_map: np.ndarray):
    """
    Combines the previous and current environment maps into a single array.

    Parameters:
        prev_env_map (np.ndarray): 2D array representing the previous environment map.
        curr_env_map (np.ndarray): 2D array representing the current environment map.

    Returns:
        np.ndarray: 2D array representing the combined environment map.

    Example:
        >>> prev = np.array([[1, 2], [3, 4]])
        >>> curr = np.array([[5, 6], [7, 8]])
        >>> result = pairing_maps(prev, curr)
        >>> print(result)
        [[1 2 5 6]
         [3 4 7 8]]
    """
    return np.concatenate([prev_env_map, curr_env_map], axis=1)
import numpy as np


def get_unique_pair_indices(pairs: np.ndarray) -> np.ndarray:
    """
    return index in NumPy array

    Parameters:
        pairs (np.ndarray): A 2D NumPy array of shape (n, 2), where each row represents a pair.

    Returns:
        np.ndarray: A 1D array containing the indices of unique pairs in the original array, sorted in ascending order.

    Examples:
        >>> pairs = np.array([[1, 2], [3, 4], [1, 2], [5, 6]])
        >>> get_unique_pair_indices(pairs)
        array([0, 1, 3])
    """
    _, unique_indices = np.unique(pairs, axis=0, return_index=True)

    return unique_indices

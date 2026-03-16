import numpy as np
from flax.struct import dataclass


@dataclass
class Duplicates:
    """
    Data class for storing the results of duplicate pair checks.

    Attributes:
        unique_pairs (np.ndarray): 2D array containing unique pairs, with each row representing a non-duplicate pair.
        unique_counts (np.ndarray): 1D array containing the occurrence count for each unique pair.
        duplicates_ratio (float): Ratio of unique pairs to total pairs (number of unique pairs divided by total pairs).
        num_unique_value (int): Number of unique pairs.
        num_total_value (int): Total number of pairs.
        min_count (int): Minimum occurrence count among unique pairs.
        max_count (int): Maximum occurrence count among unique pairs.
    """


    unique_pairs: np.ndarray
    unique_counts: np.ndarray
    duplicates_ratio: float
    num_unique_value: int
    num_total_value: int
    min_count: int
    max_count: int


def calculate_duplicates(pairs: np.ndarray) -> Duplicates:
    """
    Calculates duplicate pairs in the given array and returns statistical information about the duplicates.

    Parameters:
        pairs (np.ndarray): 2D array containing pairs to check for duplicates. Each row represents a pair.

    Returns:
        Duplicates: A Duplicates object containing statistics related to the duplicate pair check.

    Example:
        >>> pairs = np.array([[1, 2], [3, 4], [1, 2], [5, 6]])
        >>> result = calculate_duplicates(pairs)
        >>> print(result.unique_counts)
        [2 1 1]
        >>> print(result.unique_pairs)
        [[1 2]
         [3 4]
         [5 6]]
        >>> print(result.duplicates_ratio)
        0.75
    """

    unique_pairs, unique_counts = np.unique(pairs, axis=0, return_counts=True)
    num_total_value = pairs.shape[0]
    num_unique_value = unique_pairs.shape[0]

    duplicates_ratio = (num_total_value - num_unique_value) / num_total_value
    min_count, max_count = np.min(unique_counts).item(), np.max(unique_counts).item()

    return Duplicates(
        unique_pairs=unique_pairs,
        unique_counts=unique_counts,
        duplicates_ratio=duplicates_ratio,
        num_unique_value=num_unique_value,
        num_total_value=num_total_value,
        min_count=min_count,
        max_count=max_count,
    )

import numpy as np


def generate_pair_indices(done_indices: np.ndarray):
    """
    Extracts the previous and current indices of unfinished elements from the given done_indices array.

    Args:
        done_indices (np.ndarray): A 1D boolean array indicating completion status.
                                   True indicates a finished index, False indicates unfinished.

    Returns:
        tuple:
            - prev_indices (np.ndarray): Array of previous indices for unfinished elements.
            - curr_indices (np.ndarray): Array of current indices for unfinished elements.

    Notes:
        - A patch is required to exclude cases where the step difference is 2 or more.
        - Previous indices are extracted within the valid range of undone_indices.

    Raises:
        AssertionError: If done_indices is not 1-dimensional.
    """


    if done_indices.ndim != 1:
        raise TypeError(
            f"done_indices must be 1D array. current dimentinon: {done_indices.ndim}"
        )

    # undone step index
    undone_indices = np.argwhere(done_indices != True).ravel()
    # done step index
    done_indices = np.argwhere(done_indices == True).ravel()

    prev_indices = np.argwhere(undone_indices - 1 >= 0).squeeze()
    curr_indices = (prev_indices + 1).squeeze()

    return prev_indices, curr_indices


if __name__ == "__main__":
    sample_indices = np.array([False, False, False, False, True, False, False, False])
    print(generate_pair_indices(sample_indices))

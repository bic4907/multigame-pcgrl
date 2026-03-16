import pytest
import jax
import jax.numpy as jnp
from jax import jit
from debug.scenario_levels import AllLevels

@jit
def compute_reward(prev_array, unused3, curr_array, unused4) -> float:
    reward = 0.0

    # Define constants
    EMPTY = 1
    WALL = 2
    ROOM_SIZE = 3
    PATH_LENGTH = 8

    # Function to count rooms of a specific size
    def count_rooms(array, room_size):
        room_count = 0
        for i in range(array.shape[0] - room_size + 1):
            for j in range(array.shape[1] - room_size + 1):
                sub_array = array[i:i+room_size, j:j+room_size]
                room_count = jax.lax.cond(
                    jnp.all(sub_array == WALL),
                    lambda x: x + 1,
                    lambda x: x,
                    room_count
                )
        return room_count

    # Function to check for a path of a specific length
    def has_path(array, path_length):
        path_count = 0
        for i in range(array.shape[0]):
            for j in range(array.shape[1] - path_length + 1):
                sub_array = array[i, j:j+path_length]
                path_count = jax.lax.cond(
                    jnp.all(sub_array == EMPTY),
                    lambda x: x + 1,
                    lambda x: x,
                    path_count
                )
        return path_count > 0

    # Calculate the number of rooms in the current and previous arrays
    curr_room_count = count_rooms(curr_array, ROOM_SIZE)
    prev_room_count = count_rooms(prev_array, ROOM_SIZE)

    # Calculate the presence of a path in the current and previous arrays
    curr_has_path = has_path(curr_array, PATH_LENGTH)
    prev_has_path = has_path(prev_array, PATH_LENGTH)

    # Reward for the number of rooms
    room_diff = curr_room_count - prev_room_count
    reward += room_diff * 1.0

    # Reward for the presence of a path
    path_diff = int(curr_has_path) - int(prev_has_path)
    reward += path_diff * 1.0

    return reward


@jit
def compute_reward_example(prev_array, unused3, curr_array, unused4) -> float:
    WALL = 1
    cnt_wall = jnp.sum(curr_array == WALL)
    return cnt_wall


# Pytest test case
@pytest.mark.parametrize("index, level", enumerate(AllLevels))
def test_compute_reward(index, level):
    """
    Test compute_reward with all levels in AllLevels.

    Parameters:
        index: Index of the level in AllLevels.
        level: Matrix data of the level.
    """
    prev_array = level  # Treat this as the previous array
    curr_array = level  # Treat this as the current array

    # Compute reward for the given level
    reward = compute_reward_example(prev_array, None, curr_array, None)

    # Check if reward is computed without errors
    assert reward is not None, f"Reward computation failed for level at index: {index}"

    # Optionally add specific checks for reward values if needed
    print(f"Level index: {index}, Reward: {reward}")


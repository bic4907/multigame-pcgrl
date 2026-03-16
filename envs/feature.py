import jax
import jax.numpy as jnp
import chex


def preprocess_map_for_symmetry(env_map: chex.Array) -> chex.Array:
    """Preprocess the map to consider only 1 and 2 blocks. All other blocks are treated as 1."""
    processed_map = jnp.where((env_map == 1) | (env_map == 2), env_map, 1)
    return processed_map


def calculate_horizontal_symmetry(env_map: chex.Array) -> float:
    """Calculate horizontal symmetry by comparing the map with its vertically flipped version."""
    env_map = preprocess_map_for_symmetry(env_map)
    flipped_map = jnp.flipud(env_map)  # Flip the map vertically (up-down)
    return jnp.mean(env_map == flipped_map)


def calculate_vertical_symmetry(env_map: chex.Array) -> float:
    """Calculate vertical symmetry by comparing the map with its horizontally flipped version."""
    env_map = preprocess_map_for_symmetry(env_map)
    flipped_map = jnp.fliplr(env_map)  # Flip the map horizontally (left-right)
    return jnp.mean(env_map == flipped_map)


def calculate_lr_diagonal_symmetry(env_map: chex.Array) -> float:
    """Calculate diagonal symmetry (left-top to right-bottom) by comparing the map with its transpose."""
    env_map = preprocess_map_for_symmetry(env_map)
    transposed_map = jnp.transpose(env_map)  # Transpose the map (diagonal symmetry)
    return jnp.mean(env_map == transposed_map)


def calculate_rl_diagonal_symmetry(env_map: chex.Array) -> float:
    """Calculate diagonal symmetry (right-top to left-bottom) by flipping horizontally and transposing."""
    env_map = preprocess_map_for_symmetry(env_map)
    flipped_transposed_map = jnp.transpose(
        jnp.fliplr(env_map)
    )  # Flip horizontally and transpose
    return jnp.mean(env_map == flipped_transposed_map)


def flood_fill_iterative(env_map, x, y, target_tile, visited):
    """Iterative flood fill algorithm to find connected components in JAX."""
    queue = [(x, y)]
    size = 0

    while len(queue) > 0:
        x, y = queue.pop(0)

        # Check if the current cell is within bounds and should be filled
        in_bounds = (
            (x >= 0) & (x < env_map.shape[0]) & (y >= 0) & (y < env_map.shape[1])
        )
        should_fill = in_bounds & (env_map[x, y] == target_tile) & (~visited[x, y])

        # If the cell is valid, mark it as visited and increase the component size
        if should_fill:
            visited = visited.at[x, y].set(True)
            size += 1

            # Add neighboring cells (up, down, left, right) to the queue
            queue.append((x - 1, y))
            queue.append((x + 1, y))
            queue.append((x, y - 1))
            queue.append((x, y + 1))

    return size, visited


def get_largest_component_jax(env_map: chex.Array, target_tile: int) -> int:
    """Find the largest connected component of a specific tile type using JAX."""
    visited = jnp.zeros_like(env_map, dtype=bool)
    largest_size = 0

    # Iterate over the entire map
    for x in range(env_map.shape[0]):
        for y in range(env_map.shape[1]):
            # Check if the cell hasn't been visited and matches the target tile
            component_size, visited = flood_fill_iterative(
                env_map, x, y, target_tile, visited
            )
            largest_size = jax.lax.cond(
                component_size > largest_size,
                lambda _: component_size,
                lambda _: largest_size,
                operand=None,
            )

    return largest_size


def get_largest_component_size(flood_count: chex.Array) -> int:
    """Calculate the size of the largest connected component."""
    # Set a reasonable size for the maximum number of unique regions
    max_possible_regions = (
        flood_count.size
    )  # Maximum possible regions is the total number of tiles

    # Get unique region labels and their sizes, specifying the size
    unique_regions, region_sizes = jnp.unique(
        flood_count, size=max_possible_regions, return_counts=True
    )

    # Find the largest region, excluding region 0 (which represents non-regions)
    largest_region_size = jnp.max(
        jnp.where(unique_regions > 0, jnp.max(region_sizes), 0)
    )

    return largest_region_size


def calculate_shannon_entropy_manual(
    region: chex.Array, num_unique_values: int = 3
) -> float:
    """
    Calculate the Shannon entropy of a given region manually.
    """
    # Get unique values and their counts
    unique, counts = jnp.unique(region, return_counts=True, size=num_unique_values)

    # Calculate probabilities of each unique value
    total_count = jnp.sum(counts)
    probabilities = counts / total_count

    # Use jnp.where to filter out zero probabilities (instead of boolean indexing)
    non_zero_probs = jnp.where(
        probabilities > 0, probabilities, 1e-10
    )  # Replace 0 with a small number to avoid log(0)

    # Calculate the entropy manually using the formula
    entropy_value = -jnp.sum(non_zero_probs * jnp.log2(non_zero_probs))

    return entropy_value


def calculate_entropy_for_regions(
    env_map: chex.Array, num_unique_values: int = 3
) -> chex.Array:
    """
    Calculate the Shannon entropy for each of the 16 regions (4x4).
    """
    map_height, map_width = env_map.shape

    # Reshape the map into a structure where each block represents a 4x4 region
    reshaped_map = env_map.reshape(4, map_height // 4, 4, map_width // 4)

    # Flatten each region into a 16-element array
    flattened_regions = reshaped_map.reshape(4, 4, -1)

    # Apply the calculate_shannon_entropy_manual function across each region
    entropies = jax.vmap(
        lambda region: calculate_shannon_entropy_manual(region, num_unique_values)
    )(flattened_regions.reshape(-1, 16))

    # Reshape back to 4x4 entropy array
    entropies = entropies.reshape(4, 4)

    return entropies

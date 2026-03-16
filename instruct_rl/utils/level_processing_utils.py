import jax
import jax.numpy as jnp
from jax import random, vmap
from functools import partial

def mutate_level_fn(level_array: jnp.ndarray, rng_key: jax.random.PRNGKey, mutation_rate: float) -> jnp.ndarray: 
    flat = level_array.reshape(-1)
    n = flat.shape[0]
    num_change = int(n * mutation_rate)
    
    perm_key, r_key = jax.random.split(rng_key)
    
    perm = jax.random.permutation(perm_key, n)
    chosen_idx = perm[:num_change]
    
    cur_vals = flat[chosen_idx]
    
    alt = jnp.array([[2, 3], [1, 3], [1, 2]], dtype=jnp.int32) 
    r = jax.random.randint(r_key, (num_change,), minval=0, maxval=2)
    new_vals = alt[cur_vals-1, r]  
    
    updated_flat = flat.at[chosen_idx].set(new_vals)
    return updated_flat.reshape(level_array.shape)


def mutate_batched_level(level_batch: jnp.ndarray,
                         rng_key: jax.random.PRNGKey,
                         mutation_rate: float) -> jnp.ndarray:
    """
    level_batch: shape = (B, H, W), values in {1,2,3}
    rng_key: a single PRNGKey
    mutation_rate: fraction of elements to mutate per map
    returns: mutated batch, same shape
    """
    B = level_batch.shape[0]
    # split one key into B subkeys
    keys = random.split(rng_key, B)

    # vmap over batch dimension:
    #   in_axes: (0=keys, 0=level_batch, None=mutation_rate)
    vmapped = vmap(mutate_level_fn, in_axes=(0, 0, None), out_axes=0)
    return vmapped(level_batch, keys, mutation_rate)

def reshape_level(level: jnp.ndarray) -> jnp.ndarray:
    level = jnp.reshape(level, (-1, 16, 16))  # Reshape to (B, H, W)
    return jnp.squeeze(level, axis=0)

def map2onehot(level_map: jnp.ndarray, num_classes: int=3) -> jnp.ndarray:
    """
    Convert a level map to one-hot encoding.
    
    Args:
        level_map: shape = (H, W), values in {1,2,3}
        num_classes: number of classes (default is 3 for Dungeon3)
        
    Returns:
        one_hot_map: shape = (H, W, num_classes)
    """
    level_map = reshape_level(level_map)  # Ensure level_map is (H, W)
    arr_zero_start = level_map - 1
    one_hot_map = jax.nn.one_hot(arr_zero_start, num_classes)
    return one_hot_map

def map2onehot_batch(level_map_batch: jnp.ndarray, num_classes: int = 3) -> jnp.ndarray:
    """
    One-hot encode a batch of level maps of shape (..., H, W) → (..., H, W, num_classes)
    """
    # Reshape to (-1, H, W)
    orig_shape = level_map_batch.shape
    *leading_dims, H, W = orig_shape
    flat_maps = level_map_batch.reshape(-1, H, W)

    # Apply map2onehot to each map
    onehot_flat = vmap(map2onehot, in_axes=(0, None))(flat_maps, num_classes)

    # Reshape back to (..., H, W, num_classes)
    return onehot_flat.reshape(*leading_dims, H, W, num_classes)

def add_coord_channel(level_map: jnp.ndarray) -> jnp.ndarray:
    """
    Add coordinate channels to a level map.
    
    Args:
        level_map: shape = (H, W, C)
        
    Returns:
        level_map_with_coords: shape = (H, W, C+2)
    """
    H, W, C = level_map.shape
    xs = jnp.linspace(-1, 1, W)[None, :, None]
    xs = jnp.broadcast_to(xs, (H, W, 1))
    
    ys = jnp.linspace(-1, 1, H)[:, None, None]
    ys = jnp.broadcast_to(ys, (H, W, 1))
    
    level_map_with_coords = jnp.concatenate([level_map, xs, ys], axis=-1)
    return level_map_with_coords

def add_coord_channel_batch(level_map_batch: jnp.ndarray) -> jnp.ndarray:
    """
    Add coordinate channels to a batch of level maps of shape (..., H, W, C).

    Returns:
        level_map_batch_with_coords: shape = (..., H, W, C+2)
    """
    orig_shape = level_map_batch.shape
    *leading_dims, H, W, C = orig_shape
    flat_batch = level_map_batch.reshape(-1, H, W, C)

    def add_coords_single(x):  # wrapper to match vmap input
        return add_coord_channel(x)

    with_coords = vmap(add_coords_single)(flat_batch)
    return with_coords.reshape(*leading_dims, H, W, C + 2)
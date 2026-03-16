from os.path import join, dirname, abspath

import jax.lax
import numpy as np
from PIL import Image

import jax.numpy as jnp

from instruct_rl.utils.path_utils import make_directory_recursive

image_dir = abspath(join(dirname(__file__), '..', '..', 'envs', 'probs', 'tile_ims'))


def render_level(array: np.array, tile_size: int = 16, return_numpy: bool = True):
    """
    Render a level array into an image using tile images.

    Args:
        array (np.array): The level array to render.
        tile_size (int): The size of each tile in pixels.
        return_numpy (bool): Whether to return the image as a numpy array.

    Returns:
        PIL.Image.Image or np.array: The rendered image as a PIL image or numpy array.
    """
    tiles = {
        1: Image.open(join(image_dir, 'empty.png')),
        2: Image.open(join(image_dir, 'solid.png')),
        3: Image.open(join(image_dir, 'bat.png')),
    }

    # Generate a color palette for the circles

    # Create a blank image with the appropriate size
    #  img = Image.new('RGB', (array.shape[1] * tile_size, array.shape[0] * tile_size), color='white')
    img = jnp.zeros((array.shape[0] * tile_size, array.shape[1] * tile_size, 4), dtype=jnp.uint8)

    # Paste tiles onto the image
    for y, row in enumerate(array):
        for x, tile in enumerate(row):
            val = int(tile)
            resized_tile = tiles[val].resize((tile_size, tile_size))
            resized_tile_np = np.array(resized_tile, dtype=np.uint8)

            # Use dynamic update slice to place the resized tile
            img = jax.lax.dynamic_update_slice(
                img,
                jnp.array(resized_tile_np),
                (y * tile_size, x * tile_size, 0)
            )

    if return_numpy:
        return np.array(img)
    else:
        img.show()


def convert_npy_png(task_pair):
    src_npy_path, dest_image_path = task_pair

    make_directory_recursive(dirname(dest_image_path))

    npy_obj = np.load(src_npy_path)

    # rendering the image
    image_array = render_level(npy_obj)

    # Ensure uint8 dtype
    if image_array.dtype != np.uint8:
        image_array = (255 * np.clip(image_array, 0, 1)).astype(np.uint8)

    image = Image.fromarray(image_array, mode='RGBA')
    image.save(dest_image_path)

    return dest_image_path

# convert numpy array → PIL Image
def npy_to_image(npy_array):
    normed = ((npy_array - np.min(npy_array)) / (np.max(npy_array) - np.min(npy_array)) * 255).astype(np.uint8)
    return Image.fromarray(normed)
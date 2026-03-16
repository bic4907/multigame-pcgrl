import json
from os.path import join, dirname, abspath
import argparse
import numpy as np
import jax.numpy as jnp
from PIL import Image
from glob import glob
from multiprocessing import Pool, cpu_count
from tqdm import tqdm


image_dir = join(dirname(__file__), '..', '..', '..', 'envs', 'probs', 'tile_ims')


def render_level(file_path: str, output_path: str, tile_size: int):
    """
    Render a level array into an image using tile images.

    Args:
        json_input (np.array): The level array to render.
        tile_size (int): The size of each tile in pixels.

    Returns:
        None: Saves the image as 'level.png'.
    """
    tiles = {
        1: Image.open(join(image_dir, 'empty.png')),
        2: Image.open(join(image_dir, 'solid.png')),
        3: Image.open(join(image_dir, 'bat.png')),
        4: Image.open(join(image_dir, 'bat.png')),
        5: Image.open(join(image_dir, 'scorpion.png')),
        6: Image.open(join(image_dir, 'spider.png')),
        7: Image.open(join(image_dir, 'key.png')),
        8: Image.open(join(image_dir, 'door.png')),
    }

    with open(file_path, "r") as f:
        level_data = json.load(f)

    array = np.array(level_data)
    height, width = array.shape
    img = Image.new('RGBA', (width * tile_size, height * tile_size))

    for y in range(height):
        for x in range(width):
            tile_val = int(array[y, x])
            if tile_val in tiles:
                tile_img = tiles[tile_val].resize((tile_size, tile_size))
                img.paste(tile_img, (x * tile_size, y * tile_size))

    # change the extension with png
    # file_path = file_path.replace(".json", ".png")
    img.save(output_path)


def render_numpy(file_path: str, output_path: str, tile_size: int):
    """
    Render a level array into an image using tile images.

    Args:
        json_input (np.array): The level array to render.
        tile_size (int): The size of each tile in pixels.

    Returns:
        None: Saves the image as 'level.png'.
    """
    tiles = {
        1: Image.open(join(image_dir, 'empty.png')),
        2: Image.open(join(image_dir, 'solid.png')),
        3: Image.open(join(image_dir, 'bat.png')),
        4: Image.open(join(image_dir, 'bat.png')),
        5: Image.open(join(image_dir, 'scorpion.png')),
        6: Image.open(join(image_dir, 'spider.png')),
        7: Image.open(join(image_dir, 'key.png')),
        8: Image.open(join(image_dir, 'door.png')),
    }

    level_data = np.load(file_path)
    array = np.array(level_data)
    height, width = array.shape
    img = Image.new('RGBA', (width * tile_size, height * tile_size))

    for y in range(height):
        for x in range(width):
            tile_val = int(array[y, x])
            if tile_val in tiles:
                tile_img = tiles[tile_val].resize((tile_size, tile_size))
                img.paste(tile_img, (x * tile_size, y * tile_size))

    # change the extension with png
    # file_path = file_path.replace(".json", ".png")
    img.save(output_path)



def render_image_batch(config):
    """
    Render a batch of level images from JSON files.

    Args:
        config: Argparse config with `input_dir` and `tile_size`.
    """
    files = glob(join(config.input_dir, "*.json"))
    args = [(file_path, config.tile_size) for file_path in files]

    with Pool(processes=cpu_count()) as pool:
        list(tqdm(pool.starmap(render_level, args), total=len(files), desc="Rendering levels"))


def render_array(level_array: np.array, tile_size: int = 16) -> np.array:
    """
    Render a level array into an image using tile images.

    Args:
        level_array (np.array): The level array to render.
        tile_size (int): The size of each tile in pixels.

    Returns:
        np.array: The rendered image as a NumPy array.
    """
    tiles = {
        1: Image.open(join(image_dir, 'empty.png')),
        2: Image.open(join(image_dir, 'solid.png')),
        3: Image.open(join(image_dir, 'bat.png')),
        4: Image.open(join(image_dir, 'bat.png')),
        5: Image.open(join(image_dir, 'scorpion.png')),
        6: Image.open(join(image_dir, 'spider.png')),
        7: Image.open(join(image_dir, 'key.png')),
        8: Image.open(join(image_dir, 'door.png')),
    }

    height, width = level_array.shape
    img = np.array(Image.new('RGBA', (width * tile_size, height * tile_size)))

    for y in range(height):
        for x in range(width):
            tile_val = int(level_array[y, x])
            if tile_val in tiles:
                tile_img = tiles[tile_val].resize((tile_size, tile_size))
                img[y * tile_size:(y + 1) * tile_size, x * tile_size:(x + 1) * tile_size] = np.array(tile_img)
    # rgba to rgb
    img = img[:, :, :3]

    return img

def render_array_batch(level_array: np.array, tile_size: int = 16) -> np.array:
    """
    Render a batch of level arrays into RGB images using preloaded and resized tile images.

    Args:
        level_array (np.array): The level array to render. Shape: (B, H, W)
        tile_size (int): The size of each tile in pixels.
        image_dir (str): Directory containing tile images.

    Returns:
        np.array: Rendered image batch. Shape: (B, H * tile_size, W * tile_size, 3)
    """
    tile_keys = [1, 2, 3]
    tile_imgs = {
        key: np.array(
            Image.open(join(image_dir, f"{name}.png"))
            .convert("RGB")
            .resize((tile_size, tile_size))
        )
        for key, name in zip(tile_keys, ['empty', 'solid', 'bat'])
    }

    B, H, W = level_array.shape
    rendered = np.zeros((B, H * tile_size, W * tile_size, 3), dtype=np.uint8)

    for tile_val, tile_img in tile_imgs.items():
        mask = (level_array == tile_val)
        ys, xs = np.where(mask[0])  # Only need relative positions

        for b in range(B):
            b_mask = mask[b]
            yx = np.argwhere(b_mask)
            for y, x in yx:
                y0, x0 = y * tile_size, x * tile_size
                rendered[b, y0:y0 + tile_size, x0:x0 + tile_size] = tile_img

    return rendered

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="levels", help="Directory containing level JSON files.")
    parser.add_argument("--tile_size", type=int, default=16, help="Tile size in pixels.")
    config = parser.parse_args()

    render_image_batch(config)



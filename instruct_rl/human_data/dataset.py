import os
import warnings
import logging
from os.path import basename, abspath, join
import zipfile
import io
import numpy as np
import jax.numpy as jnp
import glob

from PIL import Image

from instruct_rl.utils.level_processing_utils import add_coord_channel_batch, map2onehot_batch

log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
logger = logging.getLogger(basename(__file__))
logger.setLevel(getattr(logging, log_level, logging.INFO))

LEVEL_WIDTH = 16
LEVEL_HEIGHT = 16

IMAGE_WIDTH, IMAGE_HEIGHT = 224, 224

def warn(message: str):
    logger.warning(message)
    warnings.warn(message, stacklevel=2)

class DatasetManager:
    def __init__(self, root_dir: str):
        self.root_dir = abspath(root_dir)
        self.level_dir = join(self.root_dir, "numpy")
        self.sketch_dir = join(self.root_dir, "sketch")
        logger.info(f"DatasetManager initialized from: {self.root_dir}")

    def get_instruction_dirname(self, instruction: str) -> str:
        return instruction.replace(' ', '_').replace('.', '').lower()

    def get_levels(self, instructions: list, n=1, to_jax=False, squeeze_n=False, coord_channel=False):
        output = []
        for inst in instructions:
            dir_name = self.get_instruction_dirname(inst)
            pattern = join(self.level_dir, dir_name, "*.npy")
            files = sorted(glob.glob(pattern))

            level_list = []
            for file in files[:n]:
                try:
                    arr = np.load(file, allow_pickle=True)
                    if arr.ndim == 3:
                        arr = arr[0]
                    level_list.append(arr)
                except Exception as e:
                    warn(f"Failed to load {file}: {e}")

            if len(level_list) < n:
                warn(f"Padding missing levels for '{inst}'")
                pad = [np.zeros((LEVEL_HEIGHT, LEVEL_WIDTH), dtype=np.float32)
                       for _ in range(n - len(level_list))]
                level_list += pad

            levels = np.asarray(level_list[:n], dtype=np.float32)
            output.append(levels)

        output = np.asarray(output, dtype=np.int32)
        if to_jax:
            output = jnp.asarray(output)
        if squeeze_n:
            output = output.squeeze(axis=1)

        if coord_channel:
            output = map2onehot_batch(output)
            output = add_coord_channel_batch(output)
        return output

    def get_sketches(self, instructions: list, n=1, to_jax=False, squeeze_n=False, coord_channel=False):
        output = []
        for inst in instructions:
            dir_name = self.get_instruction_dirname(inst)
            pattern = join(self.sketch_dir, dir_name, "*.png")
            files = sorted(glob.glob(pattern))

            sketch_list = []
            for file in files[:n]:
                try:
                    with Image.open(file) as img:
                        img = img.convert("L")  # grayscale
                        arr = np.array(img, dtype=np.float32)
                        arr = arr / 255.0  # normalize to [0, 1]
                        if arr.shape != (IMAGE_HEIGHT, IMAGE_WIDTH):
                            arr = np.pad(arr[:IMAGE_HEIGHT, :IMAGE_WIDTH],
                                         ((0, max(0, IMAGE_HEIGHT - arr.shape[0])),
                                          (0, max(0, IMAGE_WIDTH - arr.shape[1]))),
                                         mode='constant')
                        arr = np.expand_dims(arr, axis=-1)  # (H, W, 1)
                        sketch_list.append(arr)
                except Exception as e:
                    warn(f"Failed to load sketch {file}: {e}")

            # pad if not enough
            if len(sketch_list) < n:
                warn(f"Padding missing sketches for '{inst}'")
                pad = [np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 1), dtype=np.float32)
                       for _ in range(n - len(sketch_list))]
                sketch_list += pad

            sketches = np.asarray(sketch_list[:n], dtype=np.float32)
            output.append(sketches)

        output = np.asarray(output, dtype=np.float32)  # (B, n, H, W, 1)
        if to_jax:
            output = jnp.asarray(output)
        if squeeze_n:
            output = output.squeeze(axis=1)  # (B, H, W, 1)

        if coord_channel:
            output = add_coord_channel_batch(output)

        return output


if __name__ == "__main__":
    dataset_manager = DatasetManager("YOUR_DATASET_PATH")
    #example instruction
    levels = dataset_manager.get_levels(['a_balanced_path_length_with_narrow_characteristics_is_created'], n=5)
    sketches = dataset_manager.get_sketches(['a_balanced_path_length_with_narrow_characteristics_is_created'], n=5)
    print("Loaded shape:", levels.shape)
    print("Number of sketches:", sketches.shape)

import argparse
import logging
from copy import deepcopy
from datetime import datetime
from glob import glob
from os.path import join, dirname, abspath

import numpy as np
from tensorflow_probability.substrates.jax.math import logerfc

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


__dirname__ = dirname(__file__)

import pandas as pd



def create_dataset(df, level_dir):

    level_dataset = dict() # (task_id, level_array)

    for i, row in df.iterrows():
        instruction = row['instruction'].lower().replace(" ", "_").replace(".", "")
        inst_level_dir = join(level_dir, instruction)

        level_files = glob(join(inst_level_dir, "*.npy"))
        logger.debug(f"Loading {len(level_files)} level files for instruction: {instruction}")

        condition_type = i // 4

        if condition_type not in level_dataset:
            level_dataset[condition_type] = list()


        for level_file in level_files:
            level_npy = join(inst_level_dir, level_file)
            level_arr = np.load(level_npy)

            if level_arr.ndim == 3:
                # If the level array is 3D, take the first slice
                level_arr = level_arr[0]
                logger.debug(f"Level array for {instruction} is 3D, taking first slice.")

            level_dataset[condition_type].append(level_arr)


    level_dataset_np = list()
    level_dataset_legacy = dict()

    for task_id, levels in deepcopy(level_dataset).items():
        level_dataset[condition_type] = np.stack(levels, axis=0)

        # if the levels are lower than 128, repeat them to make it 128 / if the levels are higher than 128, take the first 128
        if level_dataset[condition_type].shape[0] < 128:
            level_dataset[condition_type] = np.tile(level_dataset[condition_type], (128 // level_dataset[condition_type].shape[0] + 1, 1, 1))[:128]
        elif level_dataset[condition_type].shape[0] > 128:
            level_dataset[condition_type] = level_dataset[condition_type][:128]

        level_dataset_np.append(level_dataset[condition_type])
        level_dataset_legacy[str(task_id)] = level_dataset[condition_type]

        logger.info(f"Task ID: {task_id}, Levels shape: {level_dataset[condition_type].shape}")

    level_dataset_np = np.stack(level_dataset_np, axis=0)
    logger.info(f"Final dataset shape: {level_dataset_np.shape}")
    logger.info(f"Legacy dataset shape: {len(level_dataset_legacy)} tasks, each with shape {level_dataset_legacy['0'].shape if level_dataset_legacy else 'N/A'}")

    return level_dataset_np, level_dataset_legacy


if __name__ == "__main__":


    instruction = abspath(join(__file__, "..", "..", "..", "..", "..", "instruct", "sub_condition",
                               "bert", "scn-1_se-whole.csv"))

    default_output = abspath(join(__dirname__, f"human_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npz"))

    parser = argparse.ArgumentParser(description="Copy evaluation data")
    parser.add_argument('--src', type=str, default="./human_dataset", help='Source directory to copy from')
    parser.add_argument('--output_path', type=str, default=default_output, help='Output file name')

    parser.add_argument('--instruction', type=str, default=instruction)
    args = parser.parse_args()

    src = abspath(args.src)

    logger = logging.getLogger(__name__)

    npy_dir = join(src, "numpy")



    logger.info(f"Loading numpy files from {npy_dir}")
    logger.info(f"Loading instruction file from {args.instruction}")
    with open(args.instruction, "r") as f:
        inst_df = pd.read_csv(f)

    dataset, dataset_legacy = create_dataset(inst_df, npy_dir)

    print(f"Dataset created with {dataset.shape} tasks.")

    logger.info(f"Outputting dataset to {args.output_path}")

    np.savez(args.output_path, {'levels': dataset})

    # make a legacy path, insert ".lagacy" before the extension
    legacy_path = args.output_path.replace(".npz", ".legacy.npz")
    np.savez(legacy_path, **dataset_legacy)
    logger.info(f"Outputting legacy dataset to {legacy_path}")


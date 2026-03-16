import argparse
import uuid
from os.path import abspath, join
from glob import glob
import re
import logging

import numpy as np
import pandas as pd
import os


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



def copy_eval_data(src: str, dest: str, inst_df: pd.DataFrame):
    cont_pcgrl_dir_regex = f"*_vec_ro_*"
    src_dir_path = join(src, cont_pcgrl_dir_regex)

    dirs = glob(src_dir_path)

    logger.info("Destination directory: %s", dest)
    numpy_dir = join(dest, "numpy")

    pattern = re.compile(r"se-[0-9]_")

    filtered = [d for d in dirs if pattern.search(d)]

    for i, root_dir in enumerate(filtered):

        eval_dir = glob(join(root_dir, "eval*"))[0]
        reward_dirs = glob(join(eval_dir, "reward_*"))

        logger.info(f"[{i+1}/{len(filtered)}] Found {len(reward_dirs)} reward directories in {eval_dir}")

        se_num = re.search(r"se-([0-9]+)_", root_dir).group(1)

        for reward_dir in reward_dirs:
            reward_i = int(reward_dir.split("/")[-1].split("_")[-1])

            if str(inst_df.iloc[reward_i]['reward_enum']) != str(se_num):
                continue

            inst_str = inst_df.iloc[reward_i]['instruction'].lower().replace(' ', '_').replace(".", "")

            # make dir
            dest_dir = join(numpy_dir, inst_str)
            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)

            seed_dirs = glob(join(reward_dir, "seed_*"))

            for seed_dir in seed_dirs:
                state_file = glob(join(seed_dir, "state_*.npy"))[0]
                # copy the state_file with uuid

                src_npy_file = np.load(state_file)
                uid = str(uuid.uuid4()).split("-")[-1]
                dest_npy = join(dest_dir, f"{uid}.npy")

                # copy file
                np.save(dest_npy, src_npy_file)




if __name__ == "__main__":

    instruction = abspath(join(__file__, "..", "..", "..", "instruct", "test", "bert", "scn-1_se-whole.csv"))

    parser = argparse.ArgumentParser(description="Copy evaluation data")
    parser.add_argument('--instruction', type=str, default=instruction)

    parser.add_argument('--src', type=str, required=True, help='Source directory to copy from')
    parser.add_argument('--dest', type=str, required=True, help='Destination directory to copy to')

    args = parser.parse_args()

    src = abspath(args.src)
    dest = abspath(args.dest)

    with open(args.instruction, "r") as f:
        inst_df = pd.read_csv(f)

    logger.info(f"Loading instruction from {args.instruction}, {inst_df.shape[0]} rows")

    copy_eval_data(src, dest, inst_df)


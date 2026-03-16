import glob
import os

import numpy as np
import argparse
from datetime import datetime
import pandas as pd
from os.path import basename, dirname, join, abspath
from tabulate import tabulate

def export(data_path: str, instruct: pd.DataFrame) -> np.array:

    np_path = abspath(join(data_path, "numpy"))

    levels = list()
    rows = list()

    for i, row in instruct.iterrows():
        instruct = row['instruction']
        condition_id = i // 4

        sub_dir = instruct.replace(' ', '_').replace('.', '').lower()

        instruct_dir = abspath(join(np_path, sub_dir))

        npy_files = glob.glob(join(instruct_dir, "*.npy"))

        if len(levels) <= condition_id:
            levels.append(list())

        for npy_file in npy_files:
            npy = np.load(npy_file)
            levels[condition_id].append(npy)
            rows.append({'instruction': instruct, 'condition_id': condition_id, 'file': basename(npy_file)})

    result_df = pd.DataFrame(rows)
    result_df = result_df.groupby(['condition_id']).agg({'file': 'count'}).reset_index()

    print('Original Dataset:')
    print(tabulate(result_df, headers='keys', tablefmt='psql'))

    max_count = max(len(group) for group in levels)

    # repeat or truncate
    filled_levels = []
    for group in levels:
        n = len(group)
        if n < max_count:
            reps = (max_count + n - 1) // n
            group = (group * reps)[:max_count]
        else:
            group = group[:max_count]
        filled_levels.append(group)

    levels_npy = np.array(filled_levels)
    print('Fill-out Dataset:')
    print(levels_npy.shape, end='\n\n')


    return levels_npy

if __name__ == "__main__":

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")


    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default='vision_level_gpt', type=str)
    parser.add_argument("--instruct", default='scn-1_se-whole', type=str)
    parser.add_argument("--output_dir", default='output', type=str)

    args = parser.parse_args()

    instruct_root = abspath(join(dirname(abspath(__file__)), "..", "..", "instruct"))
    csv_file = abspath(join(instruct_root, "test", "bert", f"{args.instruct}.csv"))

    df = pd.read_csv(csv_file)

    output = export(args.input_dir, df)

    # Save the level as a numpy file
    os.makedirs(args.output_dir, exist_ok=True)

    output_name = f"{args.input_dir}_{timestamp}.npz"
    output_path = os.path.join(args.output_dir, output_name)
    np.savez_compressed(output_path, levels=output)

    print(f"Saved level to {output_path} ({output.shape})")




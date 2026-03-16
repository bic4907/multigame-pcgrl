import argparse
import uuid
from os.path import abspath, join, basename, dirname
from glob import glob
import re
import logging
from tabulate import tabulate
import numpy as np
import pandas as pd
import os


def copy_eval_data(src: str, inst_df: pd.DataFrame):


    types = glob(join(src, "*"))

    rows = list()

    for type_dir in types:
        type_name = type_dir.split("/")[-1].split("_")[0]
        numpy_dir = join(type_dir, "numpy")

        instruct_dirs = glob(join(numpy_dir, "*"))

        for instruct_dir in instruct_dirs:
            instruct = basename(instruct_dir)

            level_files = glob(join(instruct_dir, "*.npy"))

            for level_file in level_files:
                rows.append(
                    {
                        "type": type_name,
                        "instruct": instruct,
                        "level_file": level_file,
                    }
                )

    df = pd.DataFrame(rows)


    g_df = df.groupby(['type', 'instruct']).agg({'level_file': 'count'}).reset_index()
    # print(tabulate(g_df, headers='keys', tablefmt='psql'))

    # remove embed_* columns from the dataframe
    embed_cols = [col for col in inst_df.columns if col.startswith("embed_")]
    inst_df = inst_df.drop(columns=embed_cols + ['train'])
    inst_df = inst_df.apply(lambda x: x.astype(str).str.replace(" ", "_").str.replace(".", "").str.lower())


    # merge on dataframe
    df = pd.merge(df, inst_df, how="left", left_on="instruct", right_on="instruction")
    # print(tabulate(inst_df, headers='keys', tablefmt='psql'))

    g_df = df.groupby(['type', 'reward_enum']).agg({'level_file': 'count'}).reset_index()
    print(tabulate(g_df, headers='keys', tablefmt='psql'))


if __name__ == "__main__":

    instruction = abspath(join(__file__, "..", "..", "..", "instruct", "test",
                               "bert", "scn-1_se-whole.csv"))

    parser = argparse.ArgumentParser(description="Copy evaluation data")
    parser.add_argument('--src', type=str, required=True, help='Source directory to copy from')
    parser.add_argument('--instruction', type=str, default=instruction)

    args = parser.parse_args()

    src = abspath(args.src)

    with open(args.instruction, "r") as f:
        inst_df = pd.read_csv(f)

    copy_eval_data(src, inst_df)
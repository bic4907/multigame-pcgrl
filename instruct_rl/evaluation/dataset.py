import argparse
import os
from multiprocessing import Pool
from os.path import abspath, join, basename
from glob import glob
import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
from instruct_rl.utils.render import convert_npy_png


class LevelDataset(Dataset):
    def __init__(self):
        self.dataframe = pd.DataFrame()
        self.images = None

        self._task_filter_idx = None
        self._type_filter_idx = None
        self._filter_idx = None

    def add(self, dataframe, images, train_ratio: int = 0):

        if train_ratio > 0:
            train_indices = np.random.choice(len(dataframe), int(len(dataframe) * train_ratio), replace=False)
            dataframe = dataframe.iloc[train_indices].reset_index(drop=True)
            images = images[train_indices]

        self.dataframe = pd.concat([self.dataframe, dataframe], ignore_index=True)

        if self.images is None:
            self.images = images
        else:
            self.images = np.concatenate((self.images, images), axis=0)

        # set reward_enum type to integer
        self.dataframe['reward_enum'] = self.dataframe['reward_enum'].astype(int)

        return self

    def get_images(self):
        if self._filter_idx is not None:
            indexes = self._filter_idx
        else:
            indexes = self.dataframe.index

        return self.images[indexes]

    def set_filter(self, task_filter: int = None, type_filter: str = None):
        self.dataframe['reward_enum'] = self.dataframe['reward_enum'].astype(int)

        if task_filter is None:
            self._task_filter_idx = self.dataframe.index
        else:
            self._task_filter_idx = self.dataframe[self.dataframe['reward_enum'] == task_filter].index


        if type_filter is None:
            self._type_filter_idx = self.dataframe.index
        else:
            self._type_filter_idx = self.dataframe[self.dataframe['type'] == type_filter].index

        self._filter_idx = self._task_filter_idx.intersection(self._type_filter_idx)

        return self

    @property
    def labels(self):
        if self._filter_idx is not None:
            indexes = self._filter_idx
        else:
            indexes = self.dataframe.index

        return self.dataframe.iloc[indexes]['reward_enum'].values

    @property
    def label_to_idx(self):
        return {label: idx for idx, label in enumerate(self.labels)}

    def __len__(self):
        if self._filter_idx is not None:
            return len(self._filter_idx)
        else:
            return len(self.dataframe)

    def __getitem__(self, idx):
        if self._filter_idx is not None:
            indexes = self.dataframe[self._filter_idx].index
        else:
            indexes = self.dataframe.index

        return self.dataframe.iloc[indexes[idx]], self.images[indexes[idx]]





def get_image_path(level_file: str):
    """
    Convert a level file path to a corresponding image file path.
    Args:
        level_file (str): The path to the level file.
    Returns:
        str: The path to the corresponding image file.
    """
    arr_level_path = level_file.split("/")
    arr_level_path[-3] = "png"
    arr_level_path[-1] = arr_level_path[-1].replace(".npy", ".png")
    dest_path = join(*arr_level_path)

    if level_file.startswith("/"):
        dest_path = '/' + dest_path

    return dest_path

def get_dataset(src: str, inst_df: pd.DataFrame):

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
                        "png_file": get_image_path(level_file),
                    }
                )
    df = pd.DataFrame(rows)

    embed_cols = [col for col in inst_df.columns if col.startswith("embed_")]
    inst_df = inst_df.drop(columns=embed_cols + ['train'])
    inst_df = inst_df.apply(lambda x: x.astype(str).str.replace(" ", "_").str.replace(".", "").str.lower())


    df = pd.merge(df, inst_df, how="left", left_on="instruct", right_on="instruction")

    return df



def render_dataset(dataset: pd.DataFrame, src:str, dest: str):

    os.makedirs(dest, exist_ok=True)

    task_pairs = list()
    for i, row in dataset.iterrows():
        src_path = row['level_file']

        arr_level_path = src_path.split("/")
        arr_level_path[-3] = "png"
        arr_level_path[-1] = arr_level_path[-1].replace(".npy", ".png")
        dest_path = join(*arr_level_path)

        # is '/' is the first character of src_path, set dest_path to src_path
        if src_path.startswith("/"):
            dest_path = '/' + dest_path

        task_pairs.append((src_path, dest_path))

    with Pool(os.cpu_count()) as pool:
        results = list(tqdm(pool.imap_unordered(convert_npy_png, task_pairs), total=len(task_pairs)))
    return results


if __name__ == "__main__":

    instruction = abspath(join(__file__, "..", "..", "..", "instruct", "test", "bert", "scn-1_se-whole.csv"))

    parser = argparse.ArgumentParser(description="Copy evaluation data")
    parser.add_argument('--src', type=str, required=True, help='Source directory to copy from')
    parser.add_argument('--instruction', type=str, default=instruction)

    args = parser.parse_args()

    src = abspath(args.src)

    with open(args.instruction, "r") as f:
        inst_df = pd.read_csv(f)

    get_dataset(src, inst_df)
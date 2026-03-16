import numpy as np
import pandas as pd

from instruct_rl.evaluation.utils import get_pngs


def get_valid_dataset(dataset: pd.DataFrame, type: str, n: int = None) -> pd.DataFrame:
    level_df = dataset[dataset['type'] == type]
    human_image_list = level_df['png_file'].tolist()
    human_images = get_pngs(human_image_list)

    valid_indices = [i for i, img in enumerate(human_images) if img is not None]
    valid_indices = np.array(valid_indices)
    human_level_df = level_df.iloc[valid_indices].reset_index(drop=True)
    human_images = [human_images[i] for i in valid_indices]
    human_images = np.array(human_images)

    if n:
        random_indics = np.random.choice(len(human_images), n, replace=False)
        human_level_df = human_level_df.iloc[random_indics].reset_index(drop=True)
        human_images = [human_images[i] for i in random_indics]
        human_images = np.array(human_images)

    return human_level_df, human_images
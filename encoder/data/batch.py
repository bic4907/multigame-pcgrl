import math
import pandas as pd
from glob import glob
import numpy as np
import wandb
from chex import dataclass
from os.path import basename

from sklearn.manifold import TSNE
from tqdm import tqdm
import logging
import os
from os.path import abspath, join, dirname
from conf.config import RewardTrainConfig

from encoder.data import (get_unique_pair_indices, pairing_maps)
from evaluator import get_fitness_batch, get_reward_batch

log_level = os.getenv('LOG_LEVEL', 'INFO').upper()  # Add the environment variable ;LOG_LEVEL=DEBUG
logger = logging.getLogger(basename(__file__))
logger.setLevel(getattr(logging, log_level, logging.INFO))



@dataclass
class Dataset:
    reward_id: np.ndarray
    prev_map_obs: np.ndarray
    curr_map_obs: np.ndarray
    reward: np.ndarray
    embedding: np.ndarray
    augmentable: np.ndarray


@dataclass
class EmbedData:
    reward_id: int
    embedding: np.ndarray


def create_batches(dataset: Dataset, batch_size: int, augment: bool = False):
    num_samples = len(dataset.curr_map_obs)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_indices = indices[start_idx:end_idx]

        prev = dataset.prev_map_obs[batch_indices]
        curr = dataset.curr_map_obs[batch_indices]
        embed = dataset.embedding[batch_indices]
        augmentable = dataset.augmentable[batch_indices]

        reward_id = dataset.reward_id[batch_indices]

        if augment:
            # Only augment data where augmentable == True (1)
            augment_mask = augmentable.astype(bool)

            k = np.random.choice([0, 1, 2, 3])  # Number of 90-degree rotations

            prev[augment_mask] = np.rot90(prev[augment_mask], k=k, axes=(1, 2))
            curr[augment_mask] = np.rot90(curr[augment_mask], k=k, axes=(1, 2))

            if np.random.rand() > 0.5:
                prev[augment_mask] = np.flip(prev[augment_mask], axis=1)
                curr[augment_mask] = np.flip(curr[augment_mask], axis=1)

            if np.random.rand() > 0.5:
                prev[augment_mask] = np.flip(prev[augment_mask], axis=2)
                curr[augment_mask] = np.flip(curr[augment_mask], axis=2)

        X = (prev, curr, embed)
        y = dataset.reward[batch_indices]

        yield X, y, reward_id


def create_dataset(buffer_dir: str, instruct_csv: str, config: RewardTrainConfig):
    file_list = glob(os.path.join(buffer_dir, '*.npz'), recursive=True)
    file_list += glob(os.path.join(buffer_dir, '**', '*.npz'), recursive=True)
    total_files = len(file_list)

    assert len(file_list) > 0, f"No buffer files found in {buffer_dir}"

    config.n_buffer = int(total_files * config.buffer_ratio)

    n_buffer = config.n_buffer
    if n_buffer > 0:
        file_list = file_list[:n_buffer]
        logging.info(f"Using {n_buffer} of {total_files} buffer files")

    arr_prev_map_obs, arr_curr_map_obs, arr_prev_env_map, arr_curr_env_map = [], [], [], []
    rewards = []

    logging.info(f"Loading {len(file_list)} buffer files")

    for file in tqdm(file_list, desc="Loading buffer files"):
        data = np.load(file, allow_pickle=True).get('buffer').item()

        obs = data.get('obs')
        map_obs = np.array(obs.get('map_obs'))
        reward = data.get('reward')
        done = np.array(data.get('done'))
        env_map = np.array(data.get('env_map'))

        prev_map_obs = map_obs[:, 0:-1]
        curr_map_obs = map_obs[:, 1:]
        prev_env_map = env_map[:, 0:-1]
        curr_env_map = env_map[:, 1:]
        reward = reward[:, 1:]
        done = done[:, 1:]

        done_indices = np.where(done != True)

        prev_map_obs = prev_map_obs[done_indices[0], done_indices[1], ...]
        curr_map_obs = curr_map_obs[done_indices[0], done_indices[1], ...]
        prev_env_map = prev_env_map[done_indices[0], done_indices[1], ...]
        curr_env_map = curr_env_map[done_indices[0], done_indices[1], ...]
        reward = reward[done_indices[0], done_indices[1], ...]

        arr_curr_env_map.append(curr_env_map)
        arr_prev_env_map.append(prev_env_map)
        arr_curr_map_obs.append(curr_map_obs)
        arr_prev_map_obs.append(prev_map_obs)
        rewards.append(reward)

    # Concat
    curr_map_obs = np.concatenate(arr_curr_map_obs, axis=0)
    prev_map_obs = np.concatenate(arr_prev_map_obs, axis=0)
    curr_env_map = np.concatenate(arr_curr_env_map, axis=0)
    prev_env_map = np.concatenate(arr_prev_env_map, axis=0)
    # reward = np.concatenate(rewards, axis=0)

    if config.use_prev:
        unique_pair_indices = get_unique_pair_indices(
            pairing_maps(
                prev_env_map, curr_env_map
            )
        )
    else:
        unique_pair_indices = get_unique_pair_indices(
            curr_env_map
        )

    curr_map_obs = curr_map_obs[unique_pair_indices]
    prev_map_obs = prev_map_obs[unique_pair_indices]
    curr_env_map = curr_env_map[unique_pair_indices]
    prev_env_map = prev_env_map[unique_pair_indices]

    ############################## Embedding ##############################
    # Load instruction csv
    csv_path = abspath(join(dirname(__file__), '..', '..', 'instruct', f'{instruct_csv}.csv'))
    df = pd.read_csv(csv_path)
    logging.info(f"Loading instruction csv from {csv_path}")

    # reward_enum = np.array(df['reward_enum'].to_list())
    reward_enum_list = [[int(digit) for digit in str(num)] for num in df["reward_enum"].to_list()]
    max_len = max(len(x) for x in reward_enum_list)
    reward_enum = np.array([
        x + [0] * (max_len - len(x)) for x in reward_enum_list
    ])

    df_embed = df.filter(regex='embed_*')
    df_embed = df_embed.reindex(sorted(df_embed.columns, key=lambda x: int(x.split('_')[-1])), axis=1)
    embedding = df_embed.to_numpy()

    df_cond = df.filter(regex=r'(?<!sub_)condition_')
    condition_df = df_cond.reindex(sorted(df_cond.columns, key=lambda x: int(x.split('_')[-1])), axis=1)
    condition = condition_df.to_numpy()

    sample_size = curr_env_map.shape[0]

    repeat_count = math.ceil(sample_size / len(reward_enum))

    # make number numpy with the dataframe index

    reward_id = df.index.to_numpy()

    reward_id = np.tile(reward_id, repeat_count)[:sample_size]
    embedding = np.tile(embedding, (repeat_count, 1))[:sample_size]
    reward_enum = np.tile(reward_enum, (repeat_count, 1))[:sample_size]
    condition = np.tile(condition, (repeat_count, 1))[:sample_size]

    # if reward_enum is 5, it is not augmentable
    augmentable = np.where(np.any(reward_enum == 5, axis=1), 0, 1)

    logging.info(f"Loaded {sample_size:,} samples")

    ############################## Reward ##############################
    # Initialize an empty list to store rewards

    batch_size = min(config.n_envs, config.batch_size)
    num_batches = (sample_size + batch_size - 1) // batch_size

    recalculated_reward = list()

    for i in tqdm(range(num_batches), desc="Processing reward calculation"):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, sample_size)

        #slicing current batch data
        batch_reward_enum = reward_enum[start_idx:end_idx]
        batch_condition = condition[start_idx:end_idx]
        batch_prev_env_map = prev_env_map[start_idx:end_idx]
        batch_curr_env_map = curr_env_map[start_idx:end_idx]

        if config.use_prev:
            batch_reward = get_reward_batch(
                batch_reward_enum,
                batch_condition,
                batch_prev_env_map,
                batch_curr_env_map,
            )
        else:
            batch_reward = get_fitness_batch(
                batch_reward_enum,
                batch_condition,
                batch_curr_env_map,
                config.normal_weigth,
            )

        # save
        recalculated_reward.append(batch_reward)

    reward = np.concatenate(recalculated_reward, axis=0)

    del reward_enum, condition, curr_env_map, prev_env_map

    # Reward Replacement

    if config.zero_reward_ratio is not None:
        zero_indices = np.where((reward < 0.001) & (reward > -0.001))[0]
        zero_ratio = len(zero_indices) / len(reward) * 100

        n_keep = int(len(zero_indices) * config.zero_reward_ratio) # (111, )
        zero_indices_keep = np.random.choice(zero_indices, n_keep, replace=False)

        non_zero_indices = np.where(reward != 0)[0] # (111, )
        sample_indices = np.concatenate([non_zero_indices, zero_indices_keep], axis=0)

        non_zero_count = len(non_zero_indices)
        final_zero_count = len(zero_indices_keep)
        final_total_count = len(sample_indices)
        filtered_zero_ratio = final_zero_count / final_total_count * 100
        filtered_non_zero_ratio = 100 - filtered_zero_ratio

        # logging
        logging.info(
            f"Initial dataset: {len(reward):,} samples. Zero reward samples: {len(zero_indices):,} ({zero_ratio:.2f}%).")
        logging.info(
            f"After filtering: Non-zero samples: {non_zero_count:,}, Kept zero samples: {final_zero_count:,}.")
        logging.info(
            f"Final dataset: {final_total_count:,} samples. Zero reward: {filtered_zero_ratio:.2f}%, Non-zero: {filtered_non_zero_ratio:.2f}%.")

        curr_map_obs = curr_map_obs[sample_indices]
        prev_map_obs = prev_map_obs[sample_indices]

        reward = reward[sample_indices]
        augmentable = augmentable[sample_indices]

    dataset = Dataset(reward_id=reward_id,
                      prev_map_obs=prev_map_obs,
                      curr_map_obs=curr_map_obs,
                      reward=reward,
                      embedding=embedding,
                      augmentable=augmentable)
    del reward_id
    del prev_map_obs
    del curr_map_obs
    del reward
    del embedding
    del augmentable

    return dataset, df


def split_dataset(database: Dataset, train_ratio: float = 0.8):
    """
    Splits the dataset into train and test sets.

    Args:
        database (Dataset): The full dataset containing observations, rewards, and done flags.
        train_ratio (float): Proportion of the data to use for training. Default is 0.8.

    Returns:
        Tuple[Dataset, Dataset]: Train and Test datasets.
    """
    total_size = database.curr_map_obs.shape[0]
    train_size = int(total_size * train_ratio)

    indices = np.random.permutation(total_size)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    # Train Dataset
    train_dataset = Dataset(
        reward_id=database.reward_id[train_indices],
        prev_map_obs=database.prev_map_obs[train_indices],
        curr_map_obs=database.curr_map_obs[train_indices],
        reward=database.reward[train_indices],
        embedding=database.embedding[train_indices],
        augmentable=database.augmentable[train_indices]
    )

    # Test Dataset
    test_dataset = Dataset(
        reward_id=database.reward_id[test_indices],
        prev_map_obs=database.prev_map_obs[test_indices],
        curr_map_obs=database.curr_map_obs[test_indices],
        reward=database.reward[test_indices],
        embedding=database.embedding[test_indices],
        augmentable=database.augmentable[test_indices]
    )

    return train_dataset, test_dataset


def create_embedding_table(embed_queue, reward_df: pd.DataFrame) -> wandb.Table:
    reward_ids = [e.reward_id for e in embed_queue]
    embeds = np.array([e.embedding for e in embed_queue])

    # TSNE (2dim)
    tsne = TSNE(n_components=2, random_state=42)
    tsne_embeds = tsne.fit_transform(embeds)

    inst_cols = reward_df.iloc[reward_ids][['instruction', 'reward_enum']].reset_index()
    tsne_df = pd.DataFrame(tsne_embeds, columns=['tsne_x', 'tsne_y']).reset_index()
    df = pd.concat([inst_cols, tsne_df], axis=1).drop(columns=['index'])

    # Wandb table genarate, logging
    train_table = wandb.Table(dataframe=df)

    return train_table

import math
import pandas as pd
from glob import glob
import numpy as np
import wandb
from chex import dataclass
from os.path import basename

import jax
from sklearn.manifold import TSNE
from tqdm import tqdm
import logging
import os
from os.path import abspath, join, dirname
from conf.config import RewardTrainConfig

from encoder.data import (get_unique_pair_indices, pairing_maps)
from encoder.data.instruct_utils import apply_pretrained_model
from evaluator import get_fitness_batch, get_reward_batch
from dataset.multigame import MultiGameDataset

log_level = os.getenv('LOG_LEVEL', 'INFO').upper()  # Add the environment variable ;LOG_LEVEL=DEBUG
logger = logging.getLogger(basename(__file__))
logger.setLevel(getattr(logging, log_level, logging.INFO))



@dataclass
class Dataset:
    reward_id: np.ndarray
    curr_map_obs: np.ndarray
    reward_enum: np.ndarray
    reward: np.ndarray
    instruct: np.ndarray
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

        curr = dataset.curr_map_obs[batch_indices]
        embed = dataset.embedding[batch_indices]
        augmentable = dataset.augmentable[batch_indices]

        reward_id = dataset.reward_id[batch_indices]

        if augment:
            # Only augment data where augmentable == True (1)
            augment_mask = augmentable.astype(bool)

            k = np.random.choice([0, 1, 2, 3])  # Number of 90-degree rotations

            curr[augment_mask] = np.rot90(curr[augment_mask], k=k, axes=(1, 2))

            if np.random.rand() > 0.5:
                curr[augment_mask] = np.flip(curr[augment_mask], axis=1)

            if np.random.rand() > 0.5:
                curr[augment_mask] = np.flip(curr[augment_mask], axis=2)

        X = (curr, embed)
        y = dataset.reward[batch_indices]

        yield X, y, reward_id


def create_dataset(buffer_dir: str, dataset: MultiGameDataset, config: RewardTrainConfig):
    dataset_samples = dataset._samples

    games_type = [s.game for s in dataset_samples]
    unique_games = sorted(set(games_type))
    game2idx = {game: idx for idx, game in enumerate(unique_games)}
    game_ids = np.array([game2idx[game] for game in games_type])
    logger.info(f"Detected {len(unique_games)} unique games: {game2idx}")

    # dataset_samples 순서 기반 reward_id (0, 1, 2, ...)
    sample_reward_id = np.arange(len(dataset_samples))

    whole_language_inst_list = [s.instruction for s in dataset_samples]
    whole_reward_enum = []
    whole_condition = []
    for s in dataset_samples:
        reward_e = s.meta.get("reward_enum", None)
        conditions = s.meta.get("conditions", {})  # e.g. {2: 40.0}
        # Extract value from conditions dict (e.g., 40.0 from {2: 40.0})
        condition_value = list(conditions.values())[0] if conditions else None
        # Create tuple: (game_idx, reward_enum, condition_value)
        whole_reward_enum.append(int(reward_e))
        whole_condition.append(condition_value)

    pretrained_model, tokenizer = apply_pretrained_model(config)

    with jax.disable_jit():
        encoded_inputs = tokenizer(
            whole_language_inst_list,
            return_tensors="jax",
            padding="max_length",
            max_length=config.max_len,
            truncation=True,
        )

        encoded_outputs = pretrained_model(**encoded_inputs).last_hidden_state

        embedding_outputs = encoded_outputs[:, 0, :]     # Take the [CLS] token output (shape: [num_samples, hidden_size])

    # Convert JAX output to NumPy
    embedding_outputs = np.array(embedding_outputs)

    dataset = None
    for game in unique_games:
        file_list = glob(os.path.join(buffer_dir, game, '*.npz'), recursive=True)
        file_list += glob(os.path.join(buffer_dir, game, '**', '*.npz'), recursive=True)
        logger.info(f"Found {len(file_list)} files for game '{game}'")

        assert len(file_list) > 0, f"No buffer files found in {buffer_dir}"

        arr_curr_map_obs, arr_curr_env_map = [], []

        for file in tqdm(file_list, desc="Loading buffer files"):
            data = np.load(file, allow_pickle=True).get('buffer').item()

            obs = data.get('obs')
            map_obs = np.array(obs.get('map_obs'))
            done = np.array(data.get('done'))
            env_map = np.array(data.get('env_map'))

            curr_map_obs = map_obs[:, 1:]
            curr_env_map = env_map[:, 1:]
            done = done[:, 1:]

            done_indices = np.where(done != True)

            curr_map_obs = curr_map_obs[done_indices[0], done_indices[1], ...]
            curr_env_map = curr_env_map[done_indices[0], done_indices[1], ...]

            arr_curr_env_map.append(curr_env_map)
            arr_curr_map_obs.append(curr_map_obs)

        # Concat
        curr_map_obs = np.concatenate(arr_curr_map_obs, axis=0)
        curr_env_map = np.concatenate(arr_curr_env_map, axis=0)
        # reward = np.concatenate(rewards, axis=0)

        unique_pair_indices = get_unique_pair_indices(
                curr_env_map
            )

        curr_map_obs = curr_map_obs[unique_pair_indices]
        curr_env_map = curr_env_map[unique_pair_indices]

        game_idxes = np.where(game_ids == game2idx[game])[0]
        instruct = np.array(whole_language_inst_list)[game_idxes]
        reward_enum = np.array(whole_reward_enum)[game_idxes]
        embedding = embedding_outputs[game_idxes]
        reward_id = sample_reward_id[game_idxes]
        condition = np.array(whole_condition)[game_idxes]

        sample_size = curr_env_map.shape[0]

        repeat_count = math.ceil(sample_size / len(reward_enum))

        # make number numpy with the dataframe index

        instruct = np.tile(instruct, repeat_count)[:sample_size]
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
            batch_curr_env_map = curr_env_map[start_idx:end_idx]

            batch_reward = get_fitness_batch(
                batch_reward_enum,
                batch_condition,
                batch_curr_env_map,
                config.normal_weigth,
            )

            # save
            recalculated_reward.append(batch_reward)

        reward = np.concatenate(recalculated_reward, axis=0)

        del condition, curr_env_map

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

            reward = reward[sample_indices]
            augmentable = augmentable[sample_indices]

        if dataset is not None:
            dataset.reward_id.expend(reward_id)
            dataset.curr_map_obs.expend(curr_map_obs)
            dataset.reward_enum.expend(reward_enum)
            dataset.reward.expend(reward)
            dataset.instruct.expend(instruct)
            dataset.embedding.expend(embedding)
            dataset.augmentable.expend(augmentable)
        else:
            dataset = Dataset(reward_id=reward_id,
                              curr_map_obs=curr_map_obs,
                              reward_enum=reward_enum,
                              reward=reward,
                              instruct=instruct,
                              embedding=embedding,
                              augmentable=augmentable
                              )
        del reward_enum, reward_id, curr_map_obs, reward, embedding

    return dataset


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
        curr_map_obs=database.curr_map_obs[train_indices],
        reward_enum=database.reward_enum[train_indices],
        reward=database.reward[train_indices],
        instruct=database.instruct[train_indices],
        embedding=database.embedding[train_indices],
        augmentable=database.augmentable[train_indices]
    )

    # Test Dataset
    test_dataset = Dataset(
        reward_id=database.reward_id[test_indices],
        curr_map_obs=database.curr_map_obs[test_indices],
        reward_enum=database.reward_enum[test_indices],
        reward=database.reward[test_indices],
        instruct=database.instruct[train_indices],
        embedding=database.embedding[test_indices],
        augmentable=database.augmentable[test_indices]
    )

    return train_dataset, test_dataset


def create_embedding_table(embed_queue) -> wandb.Table:
    instruction = [e.instruct for e in embed_queue]
    reward_enum = [e.reward_enum for e in embed_queue]
    embeds = np.array([e.embedding for e in embed_queue])

    # TSNE (2dim)
    tsne = TSNE(n_components=2, random_state=42)
    tsne_embeds = tsne.fit_transform(embeds)

    instruction = pd.DataFrame(instruction, columns=['instruction'])
    reward_enum = pd.DataFrame(reward_enum, columns=['reward_enum'])
    tsne_df = pd.DataFrame(tsne_embeds, columns=['tsne_x', 'tsne_y']).reset_index()
    df = pd.concat([instruction, reward_enum, tsne_df], axis=1).drop(columns=['index'])

    # Wandb table genarate, logging
    train_table = wandb.Table(dataframe=df)

    return train_table

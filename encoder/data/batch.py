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
from evaluator import get_fitness_batch
from dataset.multigame import MultiGameDataset
from instruct_rl.utils.dataset_loader import _build_instruct

log_level = os.getenv('LOG_LEVEL', 'INFO').upper()  # Add the environment variable ;LOG_LEVEL=DEBUG
logger = logging.getLogger(basename(__file__))
logger.setLevel(getattr(logging, log_level, logging.INFO))



@dataclass
class Dataset:
    reward_id: np.ndarray
    reward_enum: np.ndarray
    reward: np.ndarray
    curr_env_map: np.ndarray
    instruct: np.ndarray
    embedding: np.ndarray


@dataclass
class EmbedData:
    reward_id: int
    reward_enum: int
    instruct: str
    embedding: np.ndarray


def create_batches(dataset: Dataset, batch_size: int):
    num_samples = len(dataset.curr_env_map)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_indices = indices[start_idx:end_idx]

        curr = dataset.curr_env_map[batch_indices]
        embed = dataset.embedding[batch_indices]

        reward_id = dataset.reward_id[batch_indices]
        reward_enum = dataset.reward_enum[batch_indices]
        instruct = dataset.instruct[batch_indices]

        # Convert env_map to one-hot encoding: (batch_size, 16, 16) → (batch_size, 16, 16, 4)
        # Handle tile values: clamp to valid range [0, 3] for Dungeon3Tiles
        curr_clamped = np.clip(curr.astype(np.int32), 0, 3)
        curr_onehot = np.zeros((*curr_clamped.shape, 4), dtype=np.float32)
        for i in range(4):
            curr_onehot[..., i] = (curr_clamped == i).astype(np.float32)

        X = (curr_onehot, embed)
        y = dataset.reward[batch_indices]

        yield X, y, reward_id, reward_enum, instruct


def create_dataset(buffer_dir: str, dataset: MultiGameDataset, config: RewardTrainConfig):
    dataset_samples = dataset._samples

    # ── max_samples: dry-run용 — BERT 임베딩 계산 전에 샘플 수를 제한 ──
    max_samples = getattr(config, 'max_samples', None)
    if max_samples is not None and len(dataset_samples) > max_samples:
        logger.info(f"[dry-run] max_samples={max_samples}: "
                    f"dataset_samples {len(dataset_samples)} → {max_samples}")
        dataset_samples = dataset_samples[:max_samples]

    games_type = [s.game for s in dataset_samples]
    unique_games = sorted(set(games_type))
    game2idx = {game: idx for idx, game in enumerate(unique_games)}
    game_ids = np.array([game2idx[game] for game in games_type])
    logger.info(f"Detected {len(unique_games)} unique games: {game2idx}")

    whole_inst = _build_instruct(sample_list=dataset_samples, config=config)

    # dataset_samples 순서 기반 reward_id (0, 1, 2, ...)
    sample_reward_id = np.arange(len(dataset_samples))

    whole_language_inst_list = [s.instruction for s in dataset_samples]

    whole_reward_enum_digits = []
    for s in dataset_samples:
        reward_e = int(s.meta.get("reward_enum", 0))

        # reward_enum → digit 배열 (e.g. 2 → [2], 12 → [1, 2])
        whole_reward_enum_digits.append([int(d) for d in str(reward_e)])

    max_re_len = max(len(x) for x in whole_reward_enum_digits)
    whole_reward_enum = np.array(
        [x + [0] * (max_re_len - len(x)) for x in whole_reward_enum_digits]
    )  # (n_samples, max_re_len)

    dataset = None
    for game in unique_games:
        # file_list = os.path.join(buffer_dir, game, '*.npz') # TODO: 데이터셋 구조에 맞게 추후 수정
        file = os.path.join(buffer_dir, 'cpcgrl_buffer', 'cpcgrl_pair_dataset.npz')

        data = np.load(file, allow_pickle=True)

        arr_env_map = data['env_map_pairs']

        # prev_env_map = arr_env_map[:, 0, :, :]
        curr_env_map = arr_env_map[:, 1, :, :]

        unique_pair_indices = get_unique_pair_indices(
                curr_env_map
            )

        curr_env_map = curr_env_map[unique_pair_indices]

        game_idxes = np.where(game_ids == game2idx[game])[0]
        instruct = np.array(whole_language_inst_list)[game_idxes]
        reward_enum = np.array(whole_reward_enum)[game_idxes]
        embedding = whole_inst.embedding[game_idxes]
        reward_id = whole_inst.reward_i[game_idxes].reshape(-1)
        condition = whole_inst.condition[game_idxes]

        sample_size = curr_env_map.shape[0]

        repeat_count = math.ceil(sample_size / len(reward_enum))

        # make number numpy with the dataframe index

        instruct = np.tile(instruct, repeat_count)[:sample_size]
        reward_id = np.tile(reward_id, repeat_count)[:sample_size]
        embedding = np.tile(embedding, (repeat_count, 1))[:sample_size]
        # reward_enum: (n_game_samples, max_re_len) → tile → (sample_size, max_re_len)
        reward_enum = np.tile(reward_enum, (repeat_count, 1))[:sample_size]
        # condition:  (n_game_samples, N_CONDITION_COLS) → tile → (sample_size, N_CONDITION_COLS)
        condition = np.tile(condition, (repeat_count, 1))[:sample_size]

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

            # Select fitness function based on game type
            batch_reward = get_fitness_batch(
                batch_reward_enum,
                batch_condition,
                batch_curr_env_map,
                config.normal_weigth,
            )

            # save
            recalculated_reward.append(batch_reward)

        reward = np.concatenate(recalculated_reward, axis=0)

        del condition

        if dataset is not None:
            dataset.reward_id.expend(reward_id)
            dataset.reward_enum.expend(reward_enum)
            dataset.reward.expend(reward)
            dataset.instruct.expend(instruct)
            dataset.embedding.expend(embedding)
        else:
            dataset = Dataset(reward_id=reward_id,
                              reward_enum=reward_enum,
                              reward=reward,
                              curr_env_map=curr_env_map,
                              instruct=instruct,
                              embedding=embedding,
                              )
        del reward_enum, reward_id, reward, embedding, curr_env_map


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
    total_size = database.curr_env_map.shape[0]
    train_size = int(total_size * train_ratio)

    indices = np.random.permutation(total_size)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    # Train Dataset
    train_dataset = Dataset(
        reward_id=database.reward_id[train_indices],
        reward_enum=database.reward_enum[train_indices],
        reward=database.reward[train_indices],
        curr_env_map=database.curr_env_map[train_indices],
        instruct=database.instruct[train_indices],
        embedding=database.embedding[train_indices],
    )

    # Test Dataset
    test_dataset = Dataset(
        reward_id=database.reward_id[test_indices],
        reward_enum=database.reward_enum[test_indices],
        reward=database.reward[test_indices],
        curr_env_map=database.curr_env_map[test_indices],
        instruct=database.instruct[test_indices],
        embedding=database.embedding[test_indices],
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

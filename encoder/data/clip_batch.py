import os
import pandas as pd
from glob import glob
import numpy as np
from chex import dataclass
from itertools import cycle
import json
import logging
from os.path import abspath, join, dirname, basename, splitext
from tqdm import tqdm
from functools import partial
from typing import List, Tuple
from transformers import CLIPProcessor
from PIL import Image
import random
from dataset.multigame import MultiGameDataset

import jax
import jax.numpy as jnp

from instruct_rl.utils.level_processing_utils import mutate_level_fn, map2onehot_batch, add_coord_channel_batch

log_level = os.getenv('LOG_LEVEL', 'INFO').upper()  # Add the environment variable ;LOG_LEVEL=DEBUG
logger = logging.getLogger(basename(__file__))
logger.setLevel(getattr(logging, log_level, logging.INFO))
logging.getLogger('absl').setLevel(logging.ERROR)


@dataclass
class CLIPDataset:
    class_ids: np.ndarray
    input_ids: np.ndarray
    attention_masks: np.ndarray
    pixel_values: np.ndarray
    is_train: np.ndarray

@dataclass
class CLIPEmbedData:
    class_ids: np.ndarray
    state_embeddings: np.ndarray
    text_embeddings: np.ndarray

@dataclass
class CLIPContrastiveBatch:
    class_ids: np.ndarray
    input_ids: np.ndarray
    attention_mask: np.ndarray
    pixel_values: np.ndarray
    duplicate_matrix: np.ndarray  # (B, B) matrix indicating positive pairs


class CLIPDatasetBuilder:
    def __init__(self,
                 processor: CLIPProcessor,
                 paired_data: MultiGameDataset,
                 rng_key:jax.random.PRNGKey,
                 train_ratio:float=0.8,
                 max_len:int=77,
                 ):
        self.processor = processor
        self.paired_data = paired_data
        self.rng_key = rng_key

        self.max_len = max_len
        self.train_ratio = train_ratio

        self.dataset = self._build_dataset()

    def get_dataset(self) -> CLIPDataset:
        return self.dataset

    def _build_dataset(self) -> CLIPDataset:
        self.preprocessed_dataset_dict = self.preprocess_paired_data()

        logging.info(f"Finished loading dataset")
        return CLIPDataset(
                class_ids=np.array(self.preprocessed_dataset_dict["class_ids"]),
                input_ids=np.array(self.preprocessed_dataset_dict["input_ids"]),
                attention_masks=np.array(self.preprocessed_dataset_dict["attention_masks"]),
                pixel_values=np.array(self.preprocessed_dataset_dict["pixel_values"]),
                is_train=np.array(self.preprocessed_dataset_dict["is_train"])
            )

    def preprocess_paired_data(self):
        samples = self.paired_data._samples

        # Extract game types and create mapping to integer IDs
        games_type = [s.game for s in samples]  # N is the number of samples
        unique_games = sorted(set(games_type))
        game2idx = {game: idx for idx, game in enumerate(unique_games)}
        game_ids = np.array([game2idx[g] for g in games_type])  # (N,)
        logger.info(f"Detected {len(unique_games)} unique games: {game2idx}")

        # Extract level arrays and language instructions
        level_arrays = jnp.stack([s.array for s in samples], 0)  # (N, 16, 16)
        level_arrays = map2onehot_batch(level_arrays)
        level_arrays = add_coord_channel_batch(level_arrays)  # (N, 16, 16, C)

        language_inst_list = [s.instruction for s in samples]

        # Language instruction tokenization
        tokenized_instrs = self.processor(
            text=language_inst_list,
            return_tensors="np",
            padding="max_length",
            truncation=True,
            max_length=self.max_len
        )
        inst_input_ids, inst_attention_masks = tokenized_instrs['input_ids'], tokenized_instrs['attention_mask']

        # Stratified train/val bool mask per game type
        is_train = np.zeros(len(samples), dtype=bool)
        rng_key = self.rng_key
        for gid in range(len(unique_games)):
            rng_key, subkey = jax.random.split(rng_key)
            idx_of_game = np.where(game_ids == gid)[0]
            n_train = max(1, int(len(idx_of_game) * self.train_ratio))
            perm = np.array(jax.random.permutation(subkey, idx_of_game))
            is_train[perm[:n_train]] = True
        logger.info(f"Train: {is_train.sum()} samples, Val: {(~is_train).sum()} samples")

        return {
            "game_type":            games_type,
            "class_ids":            game_ids,
            "language_inst":        language_inst_list,
            "input_ids":            inst_input_ids,
            "attention_masks":      inst_attention_masks,
            "pixel_values":         level_arrays,
            "is_train":             is_train,
        }

    
    def get_split_dataset(self):
        """
        Get train, test datasets based on the 'is_train' flag.
        Returns:
            Tuple[CLIPDataset, CLIPDataset]: Train and Test datasets.
        """
        train_mask = self.dataset.is_train
        test_mask = ~self.dataset.is_train

        train_dataset = CLIPDataset(
            class_ids=self.dataset.class_ids[train_mask],
            input_ids=self.dataset.input_ids[train_mask],
            attention_masks=self.dataset.attention_masks[train_mask],
            pixel_values=self.dataset.pixel_values[train_mask],
            is_train=self.dataset.is_train[train_mask]
        )
        test_dataset = CLIPDataset(
            class_ids=self.dataset.class_ids[test_mask],
            input_ids=self.dataset.input_ids[test_mask],
            attention_masks=self.dataset.attention_masks[test_mask],
            pixel_values=self.dataset.pixel_values[test_mask],
            is_train=self.dataset.is_train[test_mask]
        )

        return train_dataset, test_dataset


def create_clip_batch(dataset: CLIPDataset, batch_size: int, rng_key: jax.random.PRNGKey) -> CLIPContrastiveBatch:
    """
    Create a batch of CLIP data for contrastive learning.
    Args:
        dataset (CLIPDataset): The dataset to sample from.
        batch_size (int): The size of the batch to create.
        rng_key (jax.random.PRNGKey): Random key for sampling.
    Returns:
        CLIPContrastiveBatch: A batch of CLIP data.
    """

    n_samples = len(dataset.input_ids)
    shuffled_indices = jax.random.permutation(rng_key, n_samples)
    shuffled_indices = np.array(shuffled_indices)

    for start_idx in range(0,n_samples,batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_indices = shuffled_indices[start_idx:end_idx]
        if len(batch_indices) < batch_size:
            needed = batch_size - len(batch_indices)
            extra_indices = np.random.choice(n_samples, needed, replace=True)
            batch_indices = np.concatenate([batch_indices, extra_indices])


        class_ids = dataset.class_ids[batch_indices].squeeze()             #(B,)
        input_ids = dataset.input_ids[batch_indices]           #(B,T) -> (B,77)
        attention_mask = dataset.attention_masks[batch_indices] #(B,T) -> (B,77)
        pixel_values = dataset.pixel_values[batch_indices]     #(B,H,W,C) -> (B,16,16,5)
        duplicate_matrix = np.equal.outer(class_ids, class_ids).astype(np.float32) # (B, B)

        yield CLIPContrastiveBatch(
            class_ids=class_ids,
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            duplicate_matrix=duplicate_matrix
        )

def create_clip_embedding_table(embed_queue, reward_df: pd.DataFrame):
    pass

if __name__ == "__main__":
    rng_key = jax.random.PRNGKey(0)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    dataset = MultiGameDataset()
    dataset_builder = CLIPDatasetBuilder(processor, dataset, rng_key=rng_key, train_ratio=0.8)

    train_dataset, test_dataset = dataset_builder.get_split_dataset()

    print(f"Total samples: {len(dataset_builder.get_dataset().class_ids)}")
    print(f"Train samples: {len(train_dataset.class_ids)}")
    print(f"Test samples: {len(test_dataset.class_ids)}")

    for batch in create_clip_batch(train_dataset, batch_size=32, rng_key=rng_key):
        print("\n--- Train Batch ---")
        print("Input IDs shape:", batch.input_ids.shape)
        print("Attention Mask shape:", batch.attention_mask.shape)
        print("Pixel Values shape:", batch.pixel_values.shape)
        print("Duplicate Matrix shape:", batch.duplicate_matrix.shape)
        rng_key, subkey = jax.random.split(rng_key)
        break

    for batch in create_clip_batch(test_dataset, batch_size=32, rng_key=rng_key):
        print("\n--- Test Batch ---")
        print("Input IDs shape:", batch.input_ids.shape)
        print("Attention Mask shape:", batch.attention_mask.shape)
        print("Pixel Values shape:", batch.pixel_values.shape)
        print("Duplicate Matrix shape:", batch.duplicate_matrix.shape)
        rng_key, subkey = jax.random.split(rng_key)
        break
    
    
    
    
    
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
    reward_cond: np.ndarray
    input_ids: np.ndarray
    attention_masks: np.ndarray
    pixel_values: np.ndarray
    is_train: np.ndarray
    reward_enum_targets: np.ndarray = None   # (N,) 0-indexed reward_enum
    condition_targets: np.ndarray = None     # (N,) condition float value

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


@dataclass
class CLIPDecoderBatch:
    """Contrastive + Decoder 학습용 배치."""
    class_ids: np.ndarray
    input_ids: np.ndarray
    attention_mask: np.ndarray
    pixel_values: np.ndarray
    duplicate_matrix: np.ndarray     # (B, B)
    reward_enum_target: np.ndarray   # (B,)  — 0-indexed reward_enum 클래스
    condition_target: np.ndarray     # (B,)  — condition 값 (regression target)


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

        # Create game_name to index mapping
        unique_games = sorted(set([s.game for s in self.paired_data._samples]))
        self.game2idx = {game: idx for idx, game in enumerate(unique_games)}
        self.idx2game = {idx: game for game, idx in self.game2idx.items()}

        self.dataset = self._build_dataset()

    def get_dataset(self) -> CLIPDataset:
        return self.dataset

    def _build_dataset(self) -> CLIPDataset:
        self.preprocessed_dataset_dict = self.preprocess_paired_data()

        logging.info(f"Finished loading dataset")
        return CLIPDataset(
                class_ids=np.array(self.preprocessed_dataset_dict["class_ids"]),
                reward_cond=np.array(self.preprocessed_dataset_dict["reward_cond"]),
                input_ids=np.array(self.preprocessed_dataset_dict["input_ids"]),
                attention_masks=np.array(self.preprocessed_dataset_dict["attention_masks"]),
                pixel_values=np.array(self.preprocessed_dataset_dict["pixel_values"]),
                is_train=np.array(self.preprocessed_dataset_dict["is_train"]),
                reward_enum_targets=np.array(self.preprocessed_dataset_dict["reward_enum_targets"]),
                condition_targets=np.array(self.preprocessed_dataset_dict["condition_targets"]),
            )

    def preprocess_paired_data(self):
        samples = self.paired_data._samples

        # Filter out samples with None instruction (count logged after train/val split)
        n_before = len(samples)
        dropped_combos = sorted(set(
            (s.game, s.meta.get("reward_enum"))
            for s in samples if s.instruction is None
        ))
        samples = [s for s in samples if s.instruction is not None]

        # Extract game types and create mapping to integer IDs (필터 이후 기준)
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

        # Extract reward annotation (game_name, reward_enum, conditions) combination
        reward_cond_list = []
        for s in samples:
            game_idx = self.game2idx.get(s.game, -1)  # Get game index
            reward_enum = s.meta.get("reward_enum", 0)
            conditions = s.meta.get("conditions", {})
            # reward_enum에 해당하는 condition 값을 사용 (없으면 첫 번째 값 fallback)
            condition_value = conditions.get(reward_enum, next(iter(conditions.values()), None))
            # Create tuple: (game_idx, reward_enum, condition_value)
            reward_cond_tuple = (game_idx, int(reward_enum), condition_value)
            reward_cond_list.append(reward_cond_tuple)

        # Generate class_id based on unique (game_name, reward_enum, conditions) combinations
        unique_reward_cond = sorted(set(reward_cond_list))
        reward_cond2class_id = {rc: idx for idx, rc in enumerate(unique_reward_cond)}
        class_ids = np.array([reward_cond2class_id[rc] for rc in reward_cond_list])

        # Store the mapping for reference
        self.reward_cond2class_id = reward_cond2class_id
        self.class_id2reward_cond = {v: k for k, v in reward_cond2class_id.items()}

        # Keep reward_cond as structured data
        reward_cond = np.array([
            {
                "game_idx": rc[0],
                "game_name": self.idx2game.get(rc[0], "unknown"),
                "reward_enum": rc[1],
                "condition_value": rc[2]
            }
            for rc in reward_cond_list
        ], dtype=object)

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
        n_filtered = n_before - len(samples)
        if n_filtered > 0:
            logger.info(
                f"Filtered out {n_filtered}/{n_before} samples with invalid reward. "
                f"(game, reward_enum): {dropped_combos}"
            )
        logger.info(f"Train: {is_train.sum()} samples, Val: {(~is_train).sum()} samples")

        # ── 디코더 학습용 타겟 ──
        # reward_enum: 이미 0-indexed (0=region … 4=collectable)
        reward_enum_targets = np.array([
            int(rc[1]) for rc in reward_cond_list   # 0-indexed 그대로 사용
        ], dtype=np.int32)
        condition_targets_raw = np.array([
            float(rc[2]) if rc[2] is not None else 0.0
            for rc in reward_cond_list
        ], dtype=np.float32)

        # ── reward_enum별 min-max normalization → [0, 1] ──
        unique_enums = sorted(set(reward_enum_targets))
        cond_norm_min = {}   # {enum_idx: min_val}
        cond_norm_max = {}   # {enum_idx: max_val}
        condition_targets = condition_targets_raw.copy()

        for eidx in unique_enums:
            mask = (reward_enum_targets == eidx)
            vals = condition_targets_raw[mask]
            v_min, v_max = float(vals.min()), float(vals.max())
            cond_norm_min[int(eidx)] = v_min
            cond_norm_max[int(eidx)] = v_max
            denom = v_max - v_min if v_max != v_min else 1.0
            condition_targets[mask] = (vals - v_min) / denom

        self.cond_norm_min = cond_norm_min
        self.cond_norm_max = cond_norm_max
        logger.info(f"Condition normalization (per reward_enum, 0-indexed):")
        for eidx in unique_enums:
            logger.info(f"  enum {eidx}: min={cond_norm_min[eidx]:.2f}, max={cond_norm_max[eidx]:.2f}")

        return {
            "game_type":            games_type,
            "class_ids":            class_ids,
            "reward_cond":          reward_cond,
            "language_inst":        language_inst_list,
            "input_ids":            inst_input_ids,
            "attention_masks":      inst_attention_masks,
            "pixel_values":         level_arrays,
            "is_train":             is_train,
            "reward_enum_targets":  reward_enum_targets,
            "condition_targets":    condition_targets,
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
            reward_cond=self.dataset.reward_cond[train_mask],
            input_ids=self.dataset.input_ids[train_mask],
            attention_masks=self.dataset.attention_masks[train_mask],
            pixel_values=self.dataset.pixel_values[train_mask],
            is_train=self.dataset.is_train[train_mask],
            reward_enum_targets=self.dataset.reward_enum_targets[train_mask],
            condition_targets=self.dataset.condition_targets[train_mask],
        )
        test_dataset = CLIPDataset(
            class_ids=self.dataset.class_ids[test_mask],
            reward_cond=self.dataset.reward_cond[test_mask],
            input_ids=self.dataset.input_ids[test_mask],
            attention_masks=self.dataset.attention_masks[test_mask],
            pixel_values=self.dataset.pixel_values[test_mask],
            is_train=self.dataset.is_train[test_mask],
            reward_enum_targets=self.dataset.reward_enum_targets[test_mask],
            condition_targets=self.dataset.condition_targets[test_mask],
        )

        return train_dataset, test_dataset

    def get_class_id2reward_cond(self):
        """
        Get the mapping of class IDs to reward conditions.
        Returns:
            dict: Mapping of class IDs to reward conditions.
        """
        return self.class_id2reward_cond

    def get_condition_norm_stats(self):
        """reward_enum별 condition 정규화 파라미터 반환.

        Returns:
            (cond_norm_min, cond_norm_max): 각각 {enum_idx(0-indexed): float} dict.
            역변환: original = normalized * (max - min) + min
        """
        return self.cond_norm_min, self.cond_norm_max


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


def create_clip_decoder_batch(dataset: CLIPDataset, batch_size: int, rng_key: jax.random.PRNGKey) -> CLIPDecoderBatch:
    """Create batches for contrastive + decoder training."""
    n_samples = len(dataset.input_ids)
    shuffled_indices = jax.random.permutation(rng_key, n_samples)
    shuffled_indices = np.array(shuffled_indices)

    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_indices = shuffled_indices[start_idx:end_idx]
        if len(batch_indices) < batch_size:
            needed = batch_size - len(batch_indices)
            extra_indices = np.random.choice(n_samples, needed, replace=True)
            batch_indices = np.concatenate([batch_indices, extra_indices])

        class_ids = dataset.class_ids[batch_indices].squeeze()
        input_ids = dataset.input_ids[batch_indices]
        attention_mask = dataset.attention_masks[batch_indices]
        pixel_values = dataset.pixel_values[batch_indices]
        duplicate_matrix = np.equal.outer(class_ids, class_ids).astype(np.float32)
        reward_enum_target = dataset.reward_enum_targets[batch_indices]
        condition_target = dataset.condition_targets[batch_indices]

        yield CLIPDecoderBatch(
            class_ids=class_ids,
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            duplicate_matrix=duplicate_matrix,
            reward_enum_target=reward_enum_target,
            condition_target=condition_target,
        )

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
    
    
    
    
    
import os
import logging
import warnings

import numpy as np
from PIL import Image
import jax.numpy as jnp
from os.path import abspath, dirname, join, basename

from tqdm import tqdm
from transformers import FlaxViTModel, ViTImageProcessor

from instruct_rl.evaluation.metrics.base import BaseEvaluator
from instruct_rl.vision.data.render import render_array_batch


log_level = os.getenv('LOG_LEVEL', 'INFO').upper()  # Add the environment variable ;LOG_LEVEL=DEBUG
logger = logging.getLogger(basename(__file__))
logger.setLevel(getattr(logging, log_level, logging.INFO))

class ViTEvaluator(BaseEvaluator):

    def preload(self,
                batch_size: int = 32,
                model_name: str = "google/vit-base-patch16-224",
                gt_data_name: str = 'human_20250630_213109.legacy.npz',
                force_reload: bool = False,
                normalized_vector: bool = False):
        """
        Preload ground truth embeddings from cache or generate them if not available.

        Args:
            batch_size (int): Batch size for embedding computation.
            model_name (str): Name of the ViT model to use.
            gt_data_name (str): Name of the ground truth dataset file (.npz).
            force_reload (bool): If True, ignore cache and recompute embeddings.
        """
        self.batch_size = batch_size
        self.model_name = model_name
        self.normalized_vector = normalized_vector

        self.feature_extractor = ViTImageProcessor.from_pretrained(model_name)
        self.model = FlaxViTModel.from_pretrained(model_name)

        self.gt_data_path = abspath(join(dirname(__file__), "human_data", gt_data_name))
        cache_dir = join(dirname(self.gt_data_path), "cache")
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_gt_data_path = join(cache_dir, f"{gt_data_name}")

        use_cache = os.path.exists(self.cache_gt_data_path) and not force_reload

        if use_cache:
            logger.info(f"Loading precomputed embeddings from {self.cache_gt_data_path}")
            self.embeddings_gt = dict(np.load(self.cache_gt_data_path, allow_pickle=True))
        else:
            logger.info(f"{'Forcing reload' if force_reload else 'No cache found'}, processing dataset...")
            try:
                dataset = np.load(self.gt_data_path, allow_pickle=True)
            except FileNotFoundError:
                raise FileNotFoundError(f"Dataset not found at {self.gt_data_path}. Please check the path.")

            embeddings_gt = dict()
            for key, val in tqdm(dataset.items()):
                images = render_array_batch(val, 16)
                embeddings = self.get_embeddings(images, norm=True)
                embeddings_gt[key] = embeddings

            # Save to cache
            np.savez(self.cache_gt_data_path, **embeddings_gt)
            self.embeddings_gt = embeddings_gt
            logger.info(f"Precomputed embeddings saved to {self.cache_gt_data_path}")

        if self.normalized_vector:
            with open(join(dirname(__file__), 'normalized.npy'), 'rb') as f:
                normalized = np.load(f, allow_pickle=True)
                self.norm_mean = normalized['mean']
                self.norm_std = normalized['std']

    def get_embeddings(self, level: np.ndarray, norm: bool = False) -> np.ndarray:

        features = list()

        for start_idx in range(0, level.shape[0], self.batch_size):
            end_idx = min(start_idx + self.batch_size, level.shape[0])
            batch = [Image.fromarray(p).convert("RGB") for p in level[start_idx:end_idx]]
            inputs = self.feature_extractor(batch, return_tensors="np")
            outputs = self.model(**inputs)
            pred_feats = outputs.last_hidden_state[:, 0, :]  # (B, feat_dim)
            features.append(pred_feats)

        features = jnp.concatenate(features, axis=0)  # (N, feat_dim)

        if norm:
            features = features / jnp.linalg.norm(features, axis=1, keepdims=True)

        return features


    def run(self, level: np.ndarray, task_ids: np.array) -> np.array:
        """
        Compute cosine similarity between predicted features and GT features grouped by task_id.

        Returns:
            np.array of shape (B,), similarity score per input.
        """
        assert level.shape[0] == task_ids.shape[0], "Level and task_ids must have the same number of elements."

        # Step 1: normalize total embedding
        pred_feats = self.get_embeddings(level, norm=True)  # (B, D)

        # Step 2: Group by task_id
        unique_tasks = np.unique(task_ids)
        scores = jnp.zeros(level.shape[0])

        for task_id in unique_tasks:
            task_str = str(task_id)
            if task_str not in self.embeddings_gt:
                warnings.warn(f"Task ID {task_str} not found in ground truth embeddings.")
                continue

            # Get indices for current task_id
            indices = np.where(task_ids == task_id)[0]
            pred_task = pred_feats[indices]  # (n_task, D)
            gt_task = self.embeddings_gt[task_str]  # (N_gt, D)

            if self.normalized_vector:
                pred_task = (pred_task - self.norm_mean) / self.norm_std
                gt_task = (gt_task - self.norm_mean) / self.norm_std

            # Normalize embeddings *within* task group
            norm_pred = pred_task / jnp.linalg.norm(pred_task, axis=1, keepdims=True)  # (n_task, D)
            norm_gt = gt_task / jnp.linalg.norm(gt_task, axis=1, keepdims=True)  # (N_gt, D)

            # Cosine similarity: (n_task, N_gt)
            sim_matrix = jnp.dot(norm_pred, norm_gt.T)
            mean_sim = jnp.mean(sim_matrix, axis=1)  # (n_task,)

            # Assign scores back to full array
            scores = scores.at[indices].set(mean_sim)

        return scores # (B,)


if __name__ == "__main__":
    data_path = abspath(join(dirname(__file__), "human_data", "human_20250630_213109.legacy.npz"))
    gt_data = np.load(data_path, allow_pickle=True)

    levels = gt_data['1']

    images = render_array_batch(levels, 16)
    task_ids = np.array([1] * len(levels))

    evaluator = ViTEvaluator(gt_data_name="human_20250630_213109.legacy.npz")
    scores = evaluator.run(images, task_ids=task_ids)

    print(f"Shape of scores: {scores.shape}, Mean score: {np.mean(scores)}, "
          f"Standard deviation: {np.std(scores)}")



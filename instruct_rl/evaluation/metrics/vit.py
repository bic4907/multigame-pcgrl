import os
import logging

import numpy as np
from PIL import Image
import jax.numpy as jnp
from os.path import basename

from tqdm import tqdm
from transformers import FlaxViTModel, ViTImageProcessor

from instruct_rl.evaluation.metrics.base import BaseEvaluator


log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
logger = logging.getLogger(basename(__file__))
logger.setLevel(getattr(logging, log_level, logging.INFO))


class ViTEvaluator(BaseEvaluator):
    """ViT-based image similarity evaluator.

    Operates in stateless pairwise mode: pred and gt images are passed directly
    to run_pairwise() each time — no human data files or preloading required.
    """

    _MODEL_NAME = "google/vit-base-patch16-224"
    _DEFAULT_BATCH_SIZE = 32

    def __init__(self, batch_size: int = _DEFAULT_BATCH_SIZE):
        self.batch_size = batch_size
        self.feature_extractor = None
        self.model = None

    def _ensure_model(self) -> None:
        """Lazily initialise the ViT model on first use."""
        if self.feature_extractor is None:
            logger.info("[ViTEvaluator] Loading model: %s", self._MODEL_NAME)
            self.feature_extractor = ViTImageProcessor.from_pretrained(self._MODEL_NAME)
            self.model = FlaxViTModel.from_pretrained(self._MODEL_NAME)

    def get_embeddings(
        self,
        level: np.ndarray,
        norm: bool = False,
        desc: str = "ViT embedding",
    ) -> np.ndarray:
        """Return CLS-token embeddings for a batch of images.

        Parameters
        ----------
        level : (N, H, W, 3) uint8
        norm  : L2-normalise each embedding if True.
        desc  : tqdm description string.
        """
        self._ensure_model()
        features = []
        n = level.shape[0]
        n_batches = (n + self.batch_size - 1) // self.batch_size

        with tqdm(total=n_batches, desc=desc, leave=False) as pbar:
            for start_idx in range(0, n, self.batch_size):
                end_idx = min(start_idx + self.batch_size, n)
                batch = [Image.fromarray(p).convert("RGB") for p in level[start_idx:end_idx]]
                inputs = self.feature_extractor(batch, return_tensors="np")
                outputs = self.model(**inputs)
                pred_feats = outputs.last_hidden_state[:, 0, :]  # (B, D)
                features.append(pred_feats)
                pbar.update(1)

        features = jnp.concatenate(features, axis=0)  # (N, D)
        if norm:
            features = features / jnp.linalg.norm(features, axis=1, keepdims=True)
        return features

    def run_pairwise(
        self,
        pred_images: np.ndarray,   # (N, H, W, 3) uint8
        gt_images: np.ndarray,     # (N, H, W, 3) uint8 — 1:1 correspondence with pred
    ) -> np.ndarray:
        """Compute pairwise cosine similarity between pred and gt images.

        Parameters
        ----------
        pred_images : (N, H, W, 3) uint8 — generated level renderings
        gt_images   : (N, H, W, 3) uint8 — ground-truth renderings (pred[i] ↔ gt[i])

        Returns
        -------
        scores : (N,) float32 — higher is more similar to GT
        """
        assert pred_images.shape[0] == gt_images.shape[0], (
            f"pred/gt count mismatch: {pred_images.shape[0]} vs {gt_images.shape[0]}"
        )
        pred_feats = self.get_embeddings(pred_images, norm=True, desc="[ViT] pred embedding")
        gt_feats   = self.get_embeddings(gt_images,   norm=True, desc="[ViT] gt  embedding")

        # Element-wise dot product of L2-normalised vectors = cosine similarity
        scores = jnp.sum(pred_feats * gt_feats, axis=1)  # (N,)
        return np.array(scores)

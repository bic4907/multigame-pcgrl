"""
encoder/data/mlp_batch.py
=========================
Convert annotation-format MultiGameDataset data to an MLP encoder training dataset.

Use the same preprocessing pipeline as CLIPDatasetBuilder (instruction filtering, prefixes,
Delegate long-tail cutoff, log1p normalization, stratified splitting, and
pixel_values unchanged, then additionally calculate BERT CLS embeddings.

Keep the constructor signature as close as possible to CLIPDatasetBuilder so
the encoder type (CLIP vs MLP) changes while other variables remain controlled.
BERT embedding  instruct_rl.utils.dataset_loader._compute_bert_embeddings reuse.
"""

from __future__ import annotations

import logging
import os
from os.path import basename
from typing import Iterator, Optional, Set, Tuple

import jax
import numpy as np
from chex import dataclass
from transformers import CLIPProcessor

log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logger = logging.getLogger(basename(__file__))
logger.setLevel(getattr(logging, log_level, logging.INFO))
logging.getLogger("absl").setLevel(logging.ERROR)


# ── Dataclass ─────────────────────────────────────────────────────────────────

@dataclass
class MLPDataset:
    """Dataset for MLP encoder training.

    Correspondence with CLIPDataset:
        bert_embeddings  ← (CLIPDataset  in   none)  BERT CLS embedding
        pixel_values     ← CLIPDataset.pixel_values  level map (H, W, C)
        condition_targets           ← same
        reward_enum_targets         ← same
        game_names                  ← reward_cond  in  extract
        instructions                <- original instruction text
        is_train                    ← same (unseen game  False)
    """
    bert_embeddings: np.ndarray      # (N, nlp_input_dim)
    pixel_values: np.ndarray         # (N, H, W, C)  level map (CLIPDataset  and  same)
    condition_targets: np.ndarray    # (N,)  log1p + per-enum min-max → [0, 1]
    reward_enum_targets: np.ndarray  # (N,)  0-indexed
    game_names: np.ndarray           # (N,)  str
    instructions: np.ndarray         # (N,)  str
    is_train: np.ndarray             # (N,)  bool


# ── Dataset Builder ───────────────────────────────────────────────────────────

class MLPDatasetBuilder:
    """Create an MLPDataset from CLIPDatasetBuilder output and BERT embeddings.

    The constructor matches CLIPDatasetBuilder and adds only MLP-specific
    parameters (exclude_games and nlp_input_dim).

    Parameters follow the same order and names as CLIPDatasetBuilder.
    ----------
    processor : CLIPProcessor
        Internal CLIPDatasetBuilder tokenizer, unrelated to BERT embedding.
    paired_data : MultiGameDataset
    rng_key : jax.random.PRNGKey
    train_ratio : float
    max_len : int
    max_samples : int | None
    prepend option (single instruction_prefix)
    -------------------------------------
    instruction_prefix : str | None
        "name" / "desc" / "none" (or None), matching CLIPDatasetBuilder.
    longtail_cut : bool

    Parameters  (MLP  before  for )
    ----------------------
    exclude_games : set[str] | None
        Unseen game names. Include only ``unseen_ratio`` in training and mark
        the rest is_train=False for zero/few-shot evaluation.
    nlp_input_dim : int
        BERT embedding dimension (default 768).
    unseen_ratio : float
        Fraction of the unseen-game training pool to use (few-shot ratio).
        0.0 is zero-shot; 1.0 uses the complete unseen training pool.
    seen_ratio : float
        Fraction of each seen-game training pool to use; 1.0 uses all samples.
    """

    def __init__(
        self,
        processor: CLIPProcessor,
        paired_data,
        rng_key: jax.random.PRNGKey,
        train_ratio: float = 0.8,
        max_len: int = 77,
        max_samples: Optional[int] = None,
        instruction_prefix: Optional[str] = "name",
        longtail_cut: bool = True,
        tile_offset: int = 0,
        # MLP  before  for
        exclude_games: Optional[Set[str]] = None,
        nlp_input_dim: int = 768,
        unseen_ratio: float = 0.0,
        seen_ratio: float = 1.0,
    ) -> None:
        from encoder.data.clip_batch import CLIPDatasetBuilder

        self.exclude_games: Set[str] = exclude_games or set()
        self.nlp_input_dim = nlp_input_dim
        self.unseen_ratio = float(unseen_ratio)
        self.seen_ratio = float(seen_ratio)

        # 1. CLIPDatasetBuilder  to  preprocessing (filter·prefix·normalize·split·pixel_values)
        self._clip_builder = CLIPDatasetBuilder(
            processor=processor,
            paired_data=paired_data,
            rng_key=rng_key,
            train_ratio=train_ratio,
            max_len=max_len,
            max_samples=max_samples,
            instruction_prefix=instruction_prefix,
            longtail_cut=longtail_cut,
            tile_offset=tile_offset,
        )
        clip_ds = self._clip_builder.get_dataset()
        d = self._clip_builder.preprocessed_dataset_dict

        # 2. Compute BERT embeddings from CLIPDatasetBuilder-preprocessed instructions
        instructions: list[str] = d["language_inst"]
        bert_embeds = self._compute_bert(instructions)

        # 3. Extract game_names (stored inside reward_cond in CLIPDataset)
        game_names = np.array(d["game_type"])

        # 4. is_train: CLIPDatasetBuilder split + few-shot ratio apply
        #    - Seen games: train only on the seen_ratio prefix of the training pool
        #    - Unseen games: train only on the unseen_ratio prefix; mark the rest
        #      is_train=False for zero/few-shot evaluation
        is_train = self._apply_fewshot_split(game_names, clip_ds.is_train.copy())

        # 5. Summary log
        self._log_split(game_names, is_train)

        self._dataset = MLPDataset(
            bert_embeddings=bert_embeds,
            pixel_values=np.array(d["pixel_values"]),
            condition_targets=clip_ds.condition_targets,
            reward_enum_targets=clip_ds.reward_enum_targets,
            game_names=game_names,
            instructions=np.array(instructions),
            is_train=is_train,
        )

    # ── public API ──────────────────────────────────────────────────────────────

    def get_dataset(self) -> MLPDataset:
        return self._dataset

    def get_condition_norm_stats(self) -> tuple[dict, dict]:
        """Return CLIPDatasetBuilder's per-reward_enum condition normalization parameters."""
        return self._clip_builder.get_condition_norm_stats()

    # ── Internal methods ──────────────────────────────────────────────────────

    def _compute_bert(self, instructions: list[str]) -> np.ndarray:
        """Convert CLIPDatasetBuilder-preprocessed instruction strings to BERT CLS embeddings."""
        from instruct_rl.utils.dataset_loader import _compute_bert_embeddings

        class _FakeSample:
            __slots__ = ("instruction",)
            def __init__(self, inst: str):
                self.instruction = inst

        fake_samples = [_FakeSample(inst) for inst in instructions]
        return np.array(_compute_bert_embeddings(fake_samples, self.nlp_input_dim))

    def _apply_fewshot_split(
        self, game_names: np.ndarray, is_train: np.ndarray
    ) -> np.ndarray:
        """Apply few-shot ratios within each game's training pool.

        Given CLIPDatasetBuilder's natural train/validation split (``is_train``),
        retain only a ratio prefix of each game's training pool.

        - Seen games (game not in exclude_games): use the seen_ratio prefix
        - Unseen games (game in exclude_games): use the unseen_ratio prefix
          (0.0 excludes all, preserving legacy exclude_games zero-shot behavior)

        Prefix selection is deterministic in natural index order and
        CLIP few-shot(``build_train_indices_for_ratio``  of  pool[:n_use]) and
        has identical semantics.
        """
        new_is_train = is_train.copy()
        for game in sorted(set(game_names.tolist())):
            is_unseen = game in self.exclude_games
            ratio = self.unseen_ratio if is_unseen else self.seen_ratio
            if ratio >= 1.0:
                continue  # Use everything; no change
            game_train_idx = np.where((game_names == game) & is_train)[0]
            n_use = int(len(game_train_idx) * ratio)
            drop_idx = game_train_idx[n_use:]  # Exclude samples beyond the ratio
            new_is_train[drop_idx] = False
        return new_is_train

    def _log_split(self, game_names: np.ndarray, is_train: np.ndarray) -> None:
        unique_games = sorted(set(game_names))
        logger.info("=" * 60)
        logger.info(
            "  MLPDataset split  (unseen=%s, unseen_ratio=%.4f, seen_ratio=%.4f)",
            self.exclude_games or "none", self.unseen_ratio, self.seen_ratio,
        )
        for g in unique_games:
            mask = game_names == g
            tag = "(unseen)" if g in self.exclude_games else "(seen)"
            logger.info(
                "  %-12s %s  total=%d, train=%d, val=%d",
                g, tag, mask.sum(), (mask & is_train).sum(), (mask & ~is_train).sum(),
            )
        logger.info(
            "  Total: %d (train=%d, val=%d)",
            len(game_names), is_train.sum(), (~is_train).sum(),
        )
        logger.info("=" * 60)


# ── Batch Generator ───────────────────────────────────────────────────────────

def create_mlp_batches(
    dataset: MLPDataset,
    batch_size: int,
    train: bool,
    rng: jax.random.PRNGKey,
) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Create batches from an MLPDataset.

    Parameters
    ----------
    dataset : MLPDataset
    batch_size : int
    train : bool
        True → is_train=True sample, False → val sample.
    rng : jax.random.PRNGKey

    Yields
    ------
    (bert_embeds, pixel_values, cond_targets, game_names, reward_enums)
    """
    mask = dataset.is_train if train else ~dataset.is_train
    indices = np.where(mask)[0]

    if len(indices) == 0:
        return

    perm = np.array(jax.random.permutation(rng, len(indices)))
    indices = indices[perm]

    for start in range(0, len(indices), batch_size):
        end = min(start + batch_size, len(indices))
        batch_idx = indices[start:end]
        if len(batch_idx) < batch_size:
            extra = np.random.choice(len(indices), batch_size - len(batch_idx), replace=True)
            batch_idx = np.concatenate([batch_idx, indices[extra]])

        yield (
            dataset.bert_embeddings[batch_idx],       # (B, nlp_input_dim)
            dataset.pixel_values[batch_idx],           # (B, H, W, C)
            dataset.condition_targets[batch_idx],      # (B,)
            dataset.game_names[batch_idx],             # (B,) str
            dataset.reward_enum_targets[batch_idx],    # (B,)
        )

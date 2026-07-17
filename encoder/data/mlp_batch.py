"""
encoder/data/mlp_batch.py
=========================
Annotation text MultiGameDataset → MLP text training for  Dataset.

CLIPDatasetBuilder  and  sametext preprocessing pipeline(instruction filter, prefix,
longtail cut, log1p normalize, stratified split, pixel_values text)  as-is
abovetext, BERT CLS embeddingtext text  to  computetext.

text text(CLIP vs MLP)text text remaining text  text text to
createtext text  CLIPDatasetBuilder  and  maximumtext sametext text.
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
    """MLP text training for  Dataset.

    CLIPDataset  and  text text:
        bert_embeddings  ← (CLIPDataset  in   none)  BERT CLS embedding
        pixel_values     ← CLIPDataset.pixel_values  level map (H, W, C)
        condition_targets           ← same
        reward_enum_targets         ← same
        game_names                  ← reward_cond  in  extract
        instructions                ← text instruction text
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
    """CLIPDatasetBuilder + BERT embedding as  MLPDataset   createtext.

    createtext text  CLIPDatasetBuilder  and  sametext,
    MLP  before  for  parameter(exclude_games, nlp_input_dim)text text text.

    Parameters  (CLIPDatasetBuilder  and  sametext order·name)
    ----------
    processor : CLIPProcessor
        CLIPDatasetBuilder internal tokenizer. BERT embedding compute and   text.
    paired_data : MultiGameDataset
    rng_key : jax.random.PRNGKey
    train_ratio : float
    max_len : int
    max_samples : int | None
    prepend text (text instruction_prefix)
    -------------------------------------
    instruction_prefix : str | None
        "name" / "desc" / "none" (text  None) — CLIPDatasetBuilder  and  same.
    longtail_cut : bool

    Parameters  (MLP  before  for )
    ----------------------
    exclude_games : set[str] | None
        unseen game name text. ``unseen_ratio`` text training in  text
        remaining  is_train=False (val)  to  text zero/few-shot evaluation in  text for text.
    nlp_input_dim : int
        BERT embedding dimension (default 768).
    unseen_ratio : float
        unseen game(train pool)  during  training in  text for text ratio (few-shot ratio).
        0.0 = zero-shot (unseen training data 0%), 1.0 = unseen train pool  before text.
    seen_ratio : float
        seen game(train pool)  during  training in  text for text ratio. 1.0 =  before text text for .
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

        # 2. CLIPDatasetBuilder   preprocessingtext instruction(prefix text) as  BERT embedding compute
        instructions: list[str] = d["language_inst"]
        bert_embeds = self._compute_bert(instructions)

        # 3. game_names extract (CLIPDataset  in   reward_cond text in  text)
        game_names = np.array(d["game_type"])

        # 4. is_train: CLIPDatasetBuilder split + few-shot ratio apply
        #    - seen  game: train pool  during  seen_ratio prefix text training in  text for
        #    - unseen game: train pool  during  unseen_ratio prefix text training in  text for
        #      (remaining  is_train=False  to  text zero/few-shot evaluation in  text for )
        is_train = self._apply_fewshot_split(game_names, clip_ds.is_train.copy())

        # 5. summary  to text
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
        """CLIPDatasetBuilder  of  reward_enumtext condition normalize parameter return."""
        return self._clip_builder.get_condition_norm_stats()

    # ── internal text ───────────────────────────────────────────────────────────

    def _compute_bert(self, instructions: list[str]) -> np.ndarray:
        """CLIPDatasetBuilder   preprocessingtext instruction string → BERT CLS embedding."""
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
        """gametext train pool  in  few-shot ratio   applytext.

        CLIPDatasetBuilder  of  text train/val split result(``is_train``)  text
        each game of  train pool(is_train=True)  during  ratio prefix text training in  text.

        - seen  game (game ∉ exclude_games): seen_ratio prefix text for
        - unseen game (game ∈ exclude_games): unseen_ratio prefix text for
          (unseen_ratio=0.0 →  before text text = existing exclude_games text = zero-shot)

        prefix select  text index order basis as  deterministic(deterministic) text,
        CLIP few-shot(``build_train_indices_for_ratio``  of  pool[:n_use]) and
        sametext text  text text.
        """
        new_is_train = is_train.copy()
        for game in sorted(set(game_names.tolist())):
            is_unseen = game in self.exclude_games
            ratio = self.unseen_ratio if is_unseen else self.seen_ratio
            if ratio >= 1.0:
                continue  #  before text text for  → text none
            game_train_idx = np.where((game_names == game) & is_train)[0]
            n_use = int(len(game_train_idx) * ratio)
            drop_idx = game_train_idx[n_use:]  # ratio exceedtext  training text
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
    """MLPDataset  in  textbatch  createtext.

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

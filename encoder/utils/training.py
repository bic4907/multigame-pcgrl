"""
encoder/utils/training.py
=========================
Shared by train_ipcgrl_encoder, train_clip, and train_clip_decoder.
dataset create · checkpoint save · wandb initialize utility.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
from os.path import basename

import numpy as np
import wandb
from flax.training import checkpoints

from conf.config import Config
from dataset.multigame import MultiGameDataset
from encoder.utils.path import get_ckpt_dir
from instruct_rl.utils.logger import get_wandb_name

log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logger = logging.getLogger(basename(__file__))
logger.setLevel(getattr(logging, log_level, logging.INFO))


# ═══════════════════════════════════════════════════════════════════════════════
#  MultiGameDataset create
# ═══════════════════════════════════════════════════════════════════════════════

def build_multigame_dataset(config: Config) -> MultiGameDataset:
    """Use config.include_* flags and max_samples_per_game to
    Create a MultiGameDataset.

    Parameters
    ----------
    config : Config or a subclass such as RewardConfig or CLIPTrainConfig
        ``include_dungeon``, ``include_pokemon``, ``include_sokoban``,
        ``include_doom``, ``include_doom2``, ``include_zelda``,
        Reads ``max_samples_per_game`` and ``max_samples_seed``.

    Returns
    -------
    MultiGameDataset
    """
    dataset = MultiGameDataset(
        include_dungeon=getattr(config, "include_dungeon", False),
        include_pokemon=getattr(config, "include_pokemon", False),
        include_sokoban=getattr(config, "include_sokoban", False),
        include_doom=getattr(config, "include_doom", False),
        include_doom2=getattr(config, "include_doom2", False),
        include_zelda=getattr(config, "include_zelda", False),
        max_samples_per_game=getattr(config, "max_samples_per_game", 0),
        max_samples_seed=getattr(config, "max_samples_seed", 42),
        instruction_field=getattr(config, "instruction_field", "uni"),
    )
    dataset._game_str = getattr(config, "game", "all")
    logger.info(f"MultiGameDataset: {dataset}")
    return dataset


# ═══════════════════════════════════════════════════════════════════════════════
#  checkpoint save
# ═══════════════════════════════════════════════════════════════════════════════

def save_encoder_checkpoint(config: Config, state, step: int) -> None:
    """Save an encoder checkpoint with Flax checkpoints."""
    ckpt_dir = get_ckpt_dir(config)
    ckpt_dir = os.path.abspath(ckpt_dir)
    ckpt_keep = int(getattr(config, "ckpt_keep", 2))
    checkpoints.save_checkpoint(
        ckpt_dir, target=state, prefix="", step=step, overwrite=True, keep=ckpt_keep,
    )
    logger.info(f"Checkpoint saved at step {step}")


def save_norm_stats(config: Config, cond_norm_min: dict, cond_norm_max: dict) -> None:
    """Save condition normalization statistics (log + min-max) to a JSON file.

    Saved path: <exp_dir>/norm_stats.json  (sibling of the ckpts/ directory,
    so that Flax checkpoint cleanup does not remove it)

    Parameters
    ----------
    config : Config
        Configuration object used to derive the checkpoint directory path.
    cond_norm_min : dict[int, float]
        Per-reward_enum log-space min values.
    cond_norm_max : dict[int, float]
        Per-reward_enum log-space max values.
    """
    # Save next to ckpts/ (not inside it) so Flax checkpoint management won't remove it
    exp_dir = os.path.abspath(config.exp_dir)
    os.makedirs(exp_dir, exist_ok=True)

    stats = {
        "cond_norm_min": {str(k): float(v) for k, v in cond_norm_min.items()},
        "cond_norm_max": {str(k): float(v) for k, v in cond_norm_max.items()},
    }
    path = os.path.join(exp_dir, "norm_stats.json")
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Norm stats saved to {path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Wandb initialize
# ═══════════════════════════════════════════════════════════════════════════════

def setup_wandb(config: Config) -> None:
    """Initialize wandb.

    Prefer the .env-based ``instruct_rl.utils.env_loader.get_wandb_key()``;
    fall back to ``config.wandb_key``. Keep wandb disabled when no key exists.
    """
    from instruct_rl.utils.env_loader import get_wandb_key

    wandb_key = get_wandb_key() or getattr(config, "wandb_key", None)
    if not wandb_key:
        logger.info("No wandb key found — wandb disabled")
        return

    dt = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    wandb_name = get_wandb_name(config)
    wandb_id = f"{wandb_name}-{dt}"

    wandb.login(key=wandb_key)
    wandb.init(
        project=getattr(config, "wandb_project", "instruct_pcgrl"),
        group=getattr(config, "instruct", None),
        entity=getattr(config, "wandb_entity", None),
        name=wandb_name,
        id=wandb_id,
        save_code=True,
    )
    wandb.config.update(dict(config), allow_val_change=True)

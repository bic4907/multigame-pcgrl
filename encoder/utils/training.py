"""
encoder/utils/training.py
=========================
train_ipcgrl_encoder / train_clip / train_clip_decoder 에서 공유하는
데이터셋 생성 · 체크포인트 저장 · wandb 초기화 유틸리티.
"""

from __future__ import annotations

import datetime
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
#  MultiGameDataset 생성
# ═══════════════════════════════════════════════════════════════════════════════

def build_multigame_dataset(config: Config) -> MultiGameDataset:
    """config.include_* 플래그와 max_samples_per_game 을 사용하여
    MultiGameDataset 을 생성한다.

    Parameters
    ----------
    config : Config (또는 하위 클래스 — RewardConfig, CLIPTrainConfig 등)
        ``include_dungeon``, ``include_pokemon``, ``include_sokoban``,
        ``include_doom``, ``include_doom2``, ``include_zelda``,
        ``max_samples_per_game``, ``max_samples_seed`` 필드를 참조.

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
    )
    dataset._game_str = getattr(config, "game", "all")
    logger.info(f"MultiGameDataset: {dataset}")
    return dataset


# ═══════════════════════════════════════════════════════════════════════════════
#  체크포인트 저장
# ═══════════════════════════════════════════════════════════════════════════════

def save_encoder_checkpoint(config: Config, state, step: int) -> None:
    """flax checkpoints 를 사용해 인코더 체크포인트를 저장한다."""
    ckpt_dir = get_ckpt_dir(config)
    ckpt_dir = os.path.abspath(ckpt_dir)
    checkpoints.save_checkpoint(
        ckpt_dir, target=state, prefix="", step=step, overwrite=True, keep=3,
    )
    logger.info(f"Checkpoint saved at step {step}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Wandb 초기화
# ═══════════════════════════════════════════════════════════════════════════════

def setup_wandb(config: Config) -> None:
    """wandb 를 초기화한다.

    API 키는 ``instruct_rl.utils.env_loader.get_wandb_key()`` (.env 기반)
    을 우선 사용하고, 없으면 ``config.wandb_key`` 를 fallback 으로 확인한다.
    키가 없으면 wandb 를 비활성 상태로 둔다.
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


from __future__ import annotations

import json
import logging
import os
from os.path import basename
from pathlib import Path

import jax
import jax.numpy as jnp
import optax
from flax.training import checkpoints
from flax.training.train_state import TrainState

from conf.config import CLIPDecoderTrainConfig
from encoder.clip_model import get_cnnclip_decoder_encoder
from encoder.utils.path import get_ckpt_dir
from encoder.visualization.common import ROOT, repo_relative_path


log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logger = logging.getLogger(basename(__file__))
logger.setLevel(getattr(logging, log_level, logging.INFO))


def resolve_checkpoint_root(config: CLIPDecoderTrainConfig) -> Path:
    ckpt_path = getattr(config.encoder, "ckpt_path", None)
    if ckpt_path:
        return repo_relative_path(ckpt_path)

    ckpt_dir = getattr(config.encoder, "ckpt_dir", None)
    ckpt_name = getattr(config.encoder, "ckpt_name", None)
    if ckpt_dir and ckpt_name:
        return (repo_relative_path(ckpt_dir) / ckpt_name / "ckpts").resolve()

    ckpt_root = repo_relative_path(get_ckpt_dir(config))
    if ckpt_root.exists():
        return ckpt_root

    exp_name = ckpt_root.parent.name
    candidates = [
        ROOT / "pretrained_encoders" / exp_name / "ckpts",
        ROOT / "encoder_ckpts" / exp_name / "ckpts",
        *sorted((ROOT / "pretrained_encoders").glob(f"*/{exp_name}/ckpts")),
    ]
    for candidate in candidates:
        if candidate.exists():
            resolved = candidate.resolve()
            logger.warning(
                "Default checkpoint path not found: %s; using packaged checkpoint: %s",
                ckpt_root,
                resolved,
            )
            return resolved

    return ckpt_root


def find_checkpoint_roots(max_results: int = 12) -> list[Path]:
    roots: list[Path] = []
    for base in (ROOT / "saves", ROOT / "pretrained_encoders", ROOT / "encoder_ckpts"):
        if not base.exists():
            continue
        for path in base.rglob("ckpts"):
            if path.is_dir():
                roots.append(path.resolve())
                if len(roots) >= max_results:
                    return roots
    return roots


def checkpoint_not_found_error(ckpt_root: Path) -> FileNotFoundError:
    lines = [
        f"Checkpoint path not found: {ckpt_root}",
        "",
        "The t-SNE exporter needs a trained CLIP decoder checkpoint. "
        "Pass the directory that contains numeric step folders, or rerun training first.",
        "",
        "Examples:",
        f"  python encoder/export_tsne.py game=all encoder.ckpt_path={ckpt_root}",
        "  python encoder/export_tsne.py game=all "
        "encoder.ckpt_dir=pretrained_encoders/reward "
        "encoder.ckpt_name=clipdec-game-all_exp-def_0",
    ]

    candidates = find_checkpoint_roots()
    if candidates:
        lines.extend(["", "Checkpoint roots found in this repo:"])
        lines.extend(f"  - {path}" for path in candidates)

    return FileNotFoundError("\n".join(lines))


def resolve_checkpoint_step(ckpt_root: Path, step: int | None) -> Path:
    if not ckpt_root.exists():
        raise checkpoint_not_found_error(ckpt_root)

    if step is not None:
        step_path = ckpt_root / str(step)
        if not step_path.exists():
            step_dirs = sorted(
                [p.name for p in ckpt_root.iterdir() if p.is_dir() and p.name.isdigit()],
                key=int,
            )
            suffix = f"\nAvailable checkpoint steps: {', '.join(step_dirs)}" if step_dirs else ""
            raise FileNotFoundError(f"Checkpoint step not found: {step_path}{suffix}")
        return step_path

    if ckpt_root.name.isdigit():
        return ckpt_root

    step_dirs = [p for p in ckpt_root.iterdir() if p.is_dir() and p.name.isdigit()]
    if not step_dirs:
        return ckpt_root
    return max(step_dirs, key=lambda p: int(p.name))


def load_norm_stats(ckpt_step_path: Path, num_reward_classes: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    candidates = [
        ckpt_step_path.parent / "norm_stats.json",
        ckpt_step_path.parent.parent / "norm_stats.json",
    ]
    for path in candidates:
        if path.exists():
            with path.open("r") as f:
                stats = json.load(f)
            cond_min = stats.get("cond_norm_min", {})
            cond_max = stats.get("cond_norm_max", {})
            logger.info("Loaded norm stats: %s", path)
            return (
                jnp.array([float(cond_min.get(str(i), 0.0)) for i in range(num_reward_classes)]),
                jnp.array([float(cond_max.get(str(i), 1.0)) for i in range(num_reward_classes)]),
            )

    logger.warning("norm_stats.json not found; using default [0, 1] stats")
    return (
        jnp.zeros((num_reward_classes,), dtype=jnp.float32),
        jnp.ones((num_reward_classes,), dtype=jnp.float32),
    )


def init_module_and_variables(
    config: CLIPDecoderTrainConfig,
    cond_min: jnp.ndarray,
    cond_max: jnp.ndarray,
):
    module, _ = get_cnnclip_decoder_encoder(
        config.encoder,
        decoder_config=config.decoder,
        cond_norm_min=cond_min,
        cond_norm_max=cond_max,
        RL_training=True,
    )

    variables = module.init(
        jax.random.PRNGKey(0),
        jnp.ones((1, config.encoder.token_max_len), dtype=jnp.int32),
        jnp.ones((1, config.encoder.token_max_len), dtype=jnp.int32),
        jnp.ones((1, 16, 16, config.clip_input_channel), dtype=jnp.float32),
        reward_enum=jnp.zeros((1,), dtype=jnp.int32),
        mode="text_state",
        training=False,
    )
    return module, variables


def restore_variables(module, variables_template, ckpt_step_path: Path):
    dummy_state = TrainState.create(
        apply_fn=module.apply,
        params=variables_template,
        tx=optax.identity(),
    )
    restored = checkpoints.restore_checkpoint(str(ckpt_step_path), target=dummy_state, prefix="")
    return restored.params

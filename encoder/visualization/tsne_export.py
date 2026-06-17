from __future__ import annotations

import logging
import os
from datetime import datetime
from os.path import basename
from pathlib import Path

import numpy as np
from omegaconf import DictConfig, OmegaConf

from conf.config import CLIPDecoderTrainConfig
from encoder.visualization.checkpoints import (
    init_module_and_variables,
    load_norm_stats,
    resolve_checkpoint_root,
    resolve_checkpoint_step,
    restore_variables,
)
from encoder.visualization.common import ROOT, canonical_game_name, cfg_get, normalize_games, repo_relative_path
from encoder.visualization.dataset import build_dataset, sample_indices_by_game
from encoder.visualization.embeddings import batched_encode_state, batched_encode_text
from encoder.visualization.io import save_csv, save_embedding_npz, save_json, save_plot
from encoder.visualization.reducers import run_tsne


log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logger = logging.getLogger(basename(__file__))
logger.setLevel(getattr(logging, log_level, logging.INFO))
logging.getLogger("absl").setLevel(logging.ERROR)


def _selected_reward_metadata(full_dataset, selected: np.ndarray):
    reward_cond = full_dataset.reward_cond[selected]
    reward_enums = np.array([int(rc["reward_enum"]) for rc in reward_cond], dtype=np.int32)
    conditions = np.array(
        [np.nan if rc["condition_value"] is None else float(rc["condition_value"]) for rc in reward_cond],
        dtype=np.float32,
    )
    quantized_bins = np.array([int(rc["quantized_bin"]) for rc in reward_cond], dtype=np.int32)
    games = np.array([canonical_game_name(rc["game_name"]) for rc in reward_cond])
    return games, reward_enums, conditions, quantized_bins


def _duplicate_for_modalities(values: list[np.ndarray]) -> np.ndarray:
    return np.concatenate(values, axis=0)


def _base_meta(
    *,
    config: CLIPDecoderTrainConfig | DictConfig,
    ckpt_root: Path,
    ckpt_step_path: Path,
    games: list[str],
    samples_per_game: int,
    include_text: bool,
    n_jobs: int,
) -> dict:
    return {
        "ckpt_root": str(ckpt_root),
        "ckpt_step_path": str(ckpt_step_path),
        "games": games,
        "samples_per_game": samples_per_game,
        "include_text": include_text,
        "n_jobs": n_jobs,
        "columns": [
            "x",
            "y",
            "z",
            "game",
            "type",
            "text",
            "source_index",
            "reward_enum",
            "condition_value",
            "quantized_bin",
        ],
        "config": OmegaConf.to_container(config, resolve=True)
        if isinstance(config, DictConfig)
        else str(config),
    }


def export_tsne_from_config(config: CLIPDecoderTrainConfig | DictConfig) -> dict[str, Path]:
    games = normalize_games(cfg_get(config, "tsne.games", None))
    seed = int(cfg_get(config, "tsne.seed", config.seed))
    samples_per_game = int(cfg_get(config, "tsne.samples_per_game", 1000))
    batch_size = int(cfg_get(config, "tsne.batch_size", 256))
    include_text = bool(cfg_get(config, "tsne.include_text", True))
    allow_replacement = bool(cfg_get(config, "tsne.allow_replacement", False))
    perplexity = float(cfg_get(config, "tsne.perplexity", 30.0))
    max_iter = int(cfg_get(config, "tsne.max_iter", 1000))
    n_jobs = int(cfg_get(config, "tsne.n_jobs", -1))
    ckpt_step = cfg_get(config, "tsne.ckpt_step", None)
    ckpt_step = None if ckpt_step is None else int(ckpt_step)
    max_samples_per_game = int(cfg_get(config, "tsne.max_samples_per_game", 0))

    out_root = repo_relative_path(cfg_get(config, "tsne.out_dir", ROOT / "saves"))
    out_dir = out_root / f"tsne_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: %s", out_dir)
    suffix = "state_text" if include_text else "state"
    embeddings_path = out_dir / f"embeddings_{suffix}.npz"
    png_2d_path = out_dir / f"tsne_{suffix}_2d.png"
    csv_2d_path = out_dir / f"tsne_{suffix}_2d.csv"
    coords_2d_path = out_dir / f"tsne_{suffix}_2d.npz"
    png_3d_path = out_dir / f"tsne_{suffix}_3d.png"
    csv_3d_path = out_dir / f"tsne_{suffix}_3d.csv"
    coords_3d_path = out_dir / f"tsne_{suffix}_3d.npz"
    meta_path = out_dir / f"tsne_{suffix}_meta.json"

    ckpt_root = resolve_checkpoint_root(config)
    ckpt_step_path = resolve_checkpoint_step(ckpt_root, ckpt_step)
    sample_label = "all" if samples_per_game <= 0 else str(samples_per_game)
    logger.info("Checkpoint: %s", ckpt_step_path)
    logger.info("Games: %s, samples_per_game=%s", games, sample_label)

    cond_min, cond_max = load_norm_stats(ckpt_step_path, config.decoder.num_reward_classes)
    logger.info("Initializing model template")
    module, variables_template = init_module_and_variables(config, cond_min, cond_max)
    logger.info("Restoring checkpoint variables")
    variables = restore_variables(module, variables_template, ckpt_step_path)

    logger.info("Building multigame CLIP dataset")
    full_dataset, language_inst = build_dataset(config, games, seed, max_samples_per_game=max_samples_per_game)
    selected = sample_indices_by_game(
        full_dataset,
        games=games,
        samples_per_game=samples_per_game,
        seed=seed,
        allow_replacement=allow_replacement,
    )

    pixel_values = full_dataset.pixel_values[selected].astype(np.float32)
    reward_targets = full_dataset.reward_enum_targets[selected].astype(np.int32)
    labels_game, labels_reward_enum, labels_condition, labels_quantized_bin = _selected_reward_metadata(
        full_dataset,
        selected,
    )
    labels_text = language_inst[selected]
    source_indices = selected.copy()

    logger.info("Encoding %d state embeddings", len(pixel_values))
    embeddings = [batched_encode_state(module, variables, pixel_values, reward_targets, batch_size)]
    all_games = [labels_game]
    all_types = [np.array(["state"] * len(selected))]
    all_texts = [labels_text]
    all_indices = [source_indices]
    all_reward_enums = [labels_reward_enum]
    all_conditions = [labels_condition]
    all_quantized_bins = [labels_quantized_bin]

    if include_text:
        logger.info("Encoding %d text embeddings", len(selected))
        embeddings.append(
            batched_encode_text(
                module,
                variables,
                full_dataset.input_ids[selected].astype(np.int32),
                full_dataset.attention_masks[selected].astype(np.int32),
                batch_size,
            )
        )
        all_games.append(labels_game.copy())
        all_types.append(np.array(["text"] * len(selected)))
        all_texts.append(labels_text.copy())
        all_indices.append(source_indices.copy())
        all_reward_enums.append(labels_reward_enum.copy())
        all_conditions.append(labels_condition.copy())
        all_quantized_bins.append(labels_quantized_bin.copy())

    embeddings = _duplicate_for_modalities(embeddings)
    all_games = _duplicate_for_modalities(all_games)
    all_types = _duplicate_for_modalities(all_types)
    all_texts = _duplicate_for_modalities(all_texts)
    all_indices = _duplicate_for_modalities(all_indices)
    all_reward_enums = _duplicate_for_modalities(all_reward_enums)
    all_conditions = _duplicate_for_modalities(all_conditions)
    all_quantized_bins = _duplicate_for_modalities(all_quantized_bins)

    meta = _base_meta(
        config=config,
        ckpt_root=ckpt_root,
        ckpt_step_path=ckpt_step_path,
        games=games,
        samples_per_game=samples_per_game,
        include_text=include_text,
        n_jobs=n_jobs,
    )
    meta.update(
        {
            "num_points": int(len(embeddings)),
            "embedding_dim": int(embeddings.shape[1]),
            "outputs": {
                "embeddings": str(embeddings_path),
                "png_2d": str(png_2d_path),
                "csv_2d": str(csv_2d_path),
                "coords_2d": str(coords_2d_path),
                "png_3d": str(png_3d_path),
                "csv_3d": str(csv_3d_path),
                "coords_3d": str(coords_3d_path),
                "meta": str(meta_path),
            },
            "completed": {
                "embeddings": True,
                "tsne_2d": False,
                "tsne_3d": False,
            },
        }
    )
    save_embedding_npz(
        embeddings_path,
        embeddings,
        all_games,
        all_types,
        all_texts,
        all_indices,
        all_reward_enums,
        all_conditions,
        all_quantized_bins,
    )
    save_json(meta_path, meta)
    logger.info("Saved raw embeddings: %s", embeddings_path)
    logger.info("Saved metadata: %s", meta_path)

    logger.info("Running 2D t-SNE on %d points, dim=%d", embeddings.shape[0], embeddings.shape[1])
    coords_2d = run_tsne(
        embeddings,
        perplexity=perplexity,
        max_iter=max_iter,
        seed=seed,
        n_components=2,
        n_jobs=n_jobs,
    )
    save_plot(png_2d_path, coords_2d, all_games, all_types, f"CLIP decoder 2D t-SNE ({suffix}, {sample_label}/game)")
    save_csv(
        csv_2d_path,
        coords_2d,
        all_games,
        all_types,
        all_texts,
        all_indices,
        all_reward_enums,
        all_conditions,
        all_quantized_bins,
    )
    np.savez_compressed(
        coords_2d_path,
        coords=coords_2d,
        games=all_games,
        types=all_types,
        texts=all_texts,
        source_indices=all_indices,
        reward_enums=all_reward_enums,
        condition_values=all_conditions,
        quantized_bins=all_quantized_bins,
    )
    meta["completed"]["tsne_2d"] = True
    save_json(meta_path, meta)
    logger.info("Saved 2D plot: %s", png_2d_path)
    logger.info("Saved 2D coords: %s", csv_2d_path)
    logger.info("Saved 2D arrays: %s", coords_2d_path)

    logger.info("Running 3D t-SNE on %d points, dim=%d", embeddings.shape[0], embeddings.shape[1])
    coords_3d = run_tsne(
        embeddings,
        perplexity=perplexity,
        max_iter=max_iter,
        seed=seed,
        n_components=3,
        n_jobs=n_jobs,
    )
    save_plot(png_3d_path, coords_3d, all_games, all_types, f"CLIP decoder 3D t-SNE ({suffix}, {sample_label}/game)")
    save_csv(
        csv_3d_path,
        coords_3d,
        all_games,
        all_types,
        all_texts,
        all_indices,
        all_reward_enums,
        all_conditions,
        all_quantized_bins,
    )
    np.savez_compressed(
        coords_3d_path,
        coords=coords_3d,
        games=all_games,
        types=all_types,
        texts=all_texts,
        source_indices=all_indices,
        reward_enums=all_reward_enums,
        condition_values=all_conditions,
        quantized_bins=all_quantized_bins,
    )
    meta["completed"]["tsne_3d"] = True
    save_json(meta_path, meta)
    logger.info("Saved 3D plot: %s", png_3d_path)
    logger.info("Saved 3D coords: %s", csv_3d_path)
    logger.info("Saved 3D arrays: %s", coords_3d_path)
    return {
        "embeddings": embeddings_path,
        "png_2d": png_2d_path,
        "csv_2d": csv_2d_path,
        "coords_2d": coords_2d_path,
        "png_3d": png_3d_path,
        "csv_3d": csv_3d_path,
        "coords_3d": coords_3d_path,
        "meta": meta_path,
    }

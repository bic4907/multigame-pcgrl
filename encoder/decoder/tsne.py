from __future__ import annotations

import logging
import os
from functools import partial
from os.path import basename
from typing import Dict, List, Tuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import wandb
from flax.training.train_state import TrainState
from jax import jit

from encoder.data.clip_batch import CLIPDataset
from .visualization import _GAME_SCATTER_COLORS, _MODALITY_MARKERS, _get_game_color

log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logger = logging.getLogger(basename(__file__))
logger.setLevel(getattr(logging, log_level, logging.INFO))


@partial(jit, static_argnums=(1,))
def extract_embeddings_batch(
    train_state: TrainState,
    mode: str,
    input_ids: jnp.ndarray,
    attention_mask: jnp.ndarray,
    pixel_values: jnp.ndarray,
    reward_enum: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Extract batch embeddings with JIT compilation.
    
    Returns
    -------
    text_embed, state_embed
    """
    outputs = train_state.apply_fn(
        train_state.params,
        input_ids,
        attention_mask,
        pixel_values,
        reward_enum=reward_enum,
        mode=mode,
        training=False,
        rngs={"dropout": jax.random.PRNGKey(0)},
    )
    text_embed  = outputs["text_embed"]
    state_embed = outputs.get("state_embed", jnp.zeros_like(text_embed))
    return text_embed, state_embed


def collect_tsne_embeddings(
    train_state: TrainState,
    dataset: CLIPDataset,
    game_names: np.ndarray,
    mode: str,
    n_samples: int = 1000,
    batch_size: int = 128,
    seed: int = 0,
) -> Dict[str, np.ndarray]:
    """Randomly sample n_samples items from the dataset and collect text and level embeddings.

    Run JAX operations on the main thread and return NumPy arrays.

    Parameters
    ----------
    game_names : game-name array aligned with the full dataset indices.
    """
    rng      = np.random.RandomState(seed)
    n_total  = len(dataset.class_ids)
    n_sample = min(n_samples, n_total)
    indices  = np.sort(rng.choice(n_total, size=n_sample, replace=False))

    all_text:   List[np.ndarray] = []
    all_state:  List[np.ndarray] = []
    all_gnames: List[str]        = []

    for start in range(0, n_sample, batch_size):
        end    = min(start + batch_size, n_sample)
        bidx   = indices[start:end]          # Actual dataset indices.
        actual = len(bidx)

        # Pad to a uniform batch size.
        if actual < batch_size:
            pad      = indices[: batch_size - actual]
            bidx_pad = np.concatenate([bidx, pad])
        else:
            bidx_pad = bidx

        input_ids     = jnp.array(dataset.input_ids[bidx_pad])
        attention_mask = jnp.array(dataset.attention_masks[bidx_pad])
        pixel_values  = jnp.array(dataset.pixel_values[bidx_pad])
        reward_enum   = jnp.array(dataset.reward_enum_targets[bidx_pad])

        text_emb, state_emb = extract_embeddings_batch(
            train_state,
            mode,
            input_ids,
            attention_mask,
            pixel_values,
            reward_enum,
        )

        all_text.append(np.array(jax.device_get(text_emb))[:actual])
        all_state.append(np.array(jax.device_get(state_emb))[:actual])
        all_gnames.extend(game_names[bidx].tolist())   # bidx contains dataset indices.

    return {
        "text_embed":  np.concatenate(all_text,  axis=0),
        "state_embed": np.concatenate(all_state, axis=0),
        "game_names":  np.array(all_gnames),
    }


def create_and_upload_tsne(
    text_embeds:  np.ndarray,
    state_embeds: np.ndarray,
    game_names:   np.ndarray,
    epoch:        int,
    out_dir:      str,
    tag:          str = "train",
    seed:         int = 0,
    has_state:    bool = True,
) -> None:
    """Compute t-SNE and upload it to W&B.

    Parameters
    ----------
    has_state : whether the mode includes 'state'.
                If false, or if state_embeds is all zeros, visualize text embeddings only.
    """
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        logger.warning("scikit-learn is not installed; skipping t-SNE.")
        return

    try:
        # Select embeddings to visualize.
        state_is_valid = has_state and not np.all(state_embeds == 0)

        if state_is_valid:
            combined   = np.concatenate([text_embeds, state_embeds], axis=0)
            modalities = np.array(["text"]  * len(text_embeds) +
                                  ["level"] * len(state_embeds))
            game_labels = np.concatenate([game_names, game_names], axis=0)
        else:
            combined    = text_embeds
            modalities  = np.array(["text"] * len(text_embeds))
            game_labels = game_names

        # L2 normalization.
        norms    = np.linalg.norm(combined, axis=1, keepdims=True)
        combined = combined / (norms + 1e-8)

        # t-SNE.
        perplexity = min(30, max(5, len(combined) // 10))
        tsne = TSNE(
            n_components=2,
            random_state=seed,
            perplexity=perplexity,
            n_iter=1000,
            init="pca",
            learning_rate="auto",
        )
        embeds_2d = tsne.fit_transform(combined)

        # Plot.
        fig, ax = plt.subplots(figsize=(8, 6))

        unique_games = sorted(set(game_labels.tolist()))
        for gi, gname in enumerate(unique_games):
            color = _get_game_color(gname, _GAME_SCATTER_COLORS, gi)
            for mod, marker in _MODALITY_MARKERS.items():
                mask = (game_labels == gname) & (modalities == mod)
                if not mask.any():
                    continue
                ax.scatter(
                    embeds_2d[mask, 0],
                    embeds_2d[mask, 1],
                    c=color,
                    marker=marker,
                    s=18,
                    alpha=0.55,
                    edgecolors="none",
                    label=f"{gname} ({mod})",
                    rasterized=True,
                )

        # Deduplicate legend labels.
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(
            by_label.values(),
            by_label.keys(),
            fontsize=7,
            loc="best",
            framealpha=0.85,
            ncol=max(1, len(unique_games) // 4),
        )

        # Add a modality legend explaining marker shapes.
        from matplotlib.lines import Line2D
        modality_legend_handles = [
            Line2D([0], [0], marker=mk, color="gray", linestyle="None",
                   markersize=6, label=f"modality: {mod}")
            for mod, mk in _MODALITY_MARKERS.items()
        ]
        ax.add_artist(ax.legend(handles=modality_legend_handles,
                                fontsize=7, loc="lower right", framealpha=0.85))

        ax.set_title(f"t-SNE Embeddings [{tag}]  epoch {epoch}", fontsize=10)
        ax.set_xlabel("dim 1", fontsize=8)
        ax.set_ylabel("dim 2", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.2)

        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"tsne_{tag}_epoch{epoch:04d}.png")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        logger.info("t-SNE saved: %s", path)

        if wandb.run is not None:
            wandb.log({
                f"{tag}/tsne": wandb.Image(path),
                "total/epoch": epoch,
            })
            logger.info("t-SNE uploaded to wandb (epoch %d, tag=%s)", epoch, tag)

    except Exception as exc:
        logger.warning("t-SNE failed (epoch %d, tag=%s): %s", epoch, tag, exc)


def run_tsne_epoch(
    train_state: TrainState,
    dataset: CLIPDataset,
    game_names: np.ndarray,
    mode: str,
    epoch: int,
    config,
) -> None:
    """Collect embeddings, render a t-SNE plot, and upload it for one epoch."""
    tsne_samples = int(getattr(config, "tsne_samples", 1000))
    logger.info("  Starting t-SNE visualization (epoch %d)...", epoch + 1)

    embed_data = collect_tsne_embeddings(
        train_state=train_state,
        dataset=dataset,
        game_names=game_names,
        mode=mode,
        n_samples=tsne_samples,
        batch_size=config.batch_size,
        seed=config.seed + epoch,
    )
    create_and_upload_tsne(
        text_embeds=embed_data["text_embed"],
        state_embeds=embed_data["state_embed"],
        game_names=embed_data["game_names"],
        epoch=epoch + 1,
        out_dir=config.exp_dir,
        tag="train",
        seed=config.seed + epoch,
        has_state=config.encoder.state,
    )

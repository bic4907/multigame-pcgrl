from __future__ import annotations

import logging
import math
import os
from os.path import basename
from functools import partial
from typing import Dict, Optional, Tuple, List

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import wandb
from flax.training.train_state import TrainState
from jax import jit

from encoder.data.clip_batch import CLIPDataset
from .common import _REWARD_ENUM_NAMES

log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logger = logging.getLogger(basename(__file__))
logger.setLevel(getattr(logging, log_level, logging.INFO))


# 고정 게임별 색상 팔레트
_GAME_SCATTER_COLORS = {
    "dungeon": "#4C72B0",
    "sokoban": "#DD8452",
    "zelda": "#55A868",
    "pokemon": "#C44E52",
    "doom": "#8172B3",
}

_MODALITY_MARKERS: Dict[str, str] = {
    "text": "o",
    "level": "s",
}


def _get_game_color(game: str, color_map: Optional[Dict[str, str]] = None, fallback_seed: int = 0) -> str:
    """게임별 고정 색상을 반환한다."""
    if color_map is None:
        color_map = _GAME_SCATTER_COLORS
    if game in color_map:
        return color_map[game]

    # 새 게임 이름은 순환 팔레트로 fallback
    fallback_colors = [
        "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
        "#393b79", "#637939", "#8c6d31", "#843c39", "#8c564b",
    ]
    return fallback_colors[fallback_seed % len(fallback_colors)]


def _compute_scatter_trendline(
    x: np.ndarray,
    y: np.ndarray,
) -> Tuple[float, float, float]:
    """Return (r, slope, intercept) for finite (x, y) pairs.

    If not enough finite pairs exist, returns NaN values.
    """
    x_arr = np.asarray(x, dtype=float).reshape(-1)
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    finite_mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[finite_mask]
    y_arr = y_arr[finite_mask]

    if x_arr.size < 2 or y_arr.size < 2:
        return float("nan"), float("nan"), float("nan")
    if np.std(x_arr) < 1e-12 or np.std(y_arr) < 1e-12:
        return float("nan"), float("nan"), float("nan")

    try:
        r = float(np.corrcoef(x_arr, y_arr)[0, 1])
        slope, intercept = np.polyfit(x_arr, y_arr, 1)
    except Exception:
        return float("nan"), float("nan"), float("nan")
    return r, float(slope), float(intercept)


def create_fewshot_plot(
    results: Dict[float, Dict[str, float]],
    reg_results: Dict[float, Dict[str, float]],
    unseen_game_names: Set[str],
    out_dir: str,
) -> str:
    """Few-shot ratio sweep 결과를 reg_loss 단일 패널로 시각화한다.

    Reward Accuracy는 wandb 스칼라로만 기록하고 이미지에는 포함하지 않는다.
    """
    os.makedirs(out_dir, exist_ok=True)

    ratios = sorted([r for r in results.keys() if r < 1.0])
    all_games = sorted(
        {g for r in reg_results.values() for g in r
         if g not in ("overall", "seen_overall", "unseen_overall")}
    )

    unseen_tag = ", ".join(sorted(unseen_game_names))

    fig, ax = plt.subplots(figsize=(3.8, 2.6))

    # ── Seen / Unseen (굵은 선, legend) ──
    seen_ov = [reg_results[r].get("seen_overall", float("nan")) for r in ratios]
    unseen_ov = [reg_results[r].get("unseen_overall", float("nan")) for r in ratios]
    ax.plot(ratios, seen_ov, marker="s", markersize=4, linewidth=2.4,
            linestyle="--", color="#b2182b", label="Seen")
    ax.plot(ratios, unseen_ov, marker="o", markersize=4, linewidth=2.4,
            linestyle="-", color="#2166ac", label="Unseen")

    ax.set_xlabel("Few-shot Ratio", fontsize=8)
    ax.set_ylabel("Regression Loss (Huber)", fontsize=8)
    ax.set_title(f"Unseen: {unseen_tag}", fontsize=8.5)
    ax.set_xlim(-0.02, 1.02)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", fontsize=6, framealpha=0.85)

    path = os.path.join(out_dir, "fewshot_ratio_vs_reward_accuracy.png")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    logger.info("Few-shot plot saved: %s", path)
    return path


def create_scatter_plots(
    scatter_data: Dict[int, Dict[str, np.ndarray]],
    out_dir: str,
    max_points: int = 1000,
    seed: int = 0,
    space: str = "norm",
    game_colors: Optional[Dict[str, str]] = None,
) -> Optional[str]:
    """Per reward_enum scatter plot (pred vs target).

    Parameters
    ----------
    scatter_data : evaluate_per_game()의 반환값
    max_points   : 서브플롯별 최대 점 개수 (초과 시 random sampling)
    space        : "norm" (정규화된 [0,1] 공간) or "raw" (linear 스케일)
    """
    if not scatter_data:
        logger.warning("create_scatter_plots: empty scatter_data — skipping")
        return None

    os.makedirs(out_dir, exist_ok=True)
    pred_key = f"pred_{space}"
    target_key = f"target_{space}"

    enums = sorted(scatter_data.keys())
    n = len(enums)
    ncols = min(3, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, squeeze=False)
    fig.set_size_inches(2.9 * ncols, 2.3 * nrows)

    color_map = {**_GAME_SCATTER_COLORS, **(game_colors or {})}
    rng = np.random.RandomState(seed)
    for i, e in enumerate(enums):
        ax = axes[i // ncols][i % ncols]
        pred = np.asarray(scatter_data[e][pred_key])
        target = np.asarray(scatter_data[e][target_key])
        game_names = np.asarray(scatter_data[e].get("game_names", np.array([])), dtype=object)

        n_pts = len(pred)
        if n_pts > max_points:
            sel = rng.choice(n_pts, size=max_points, replace=False)
            pred = pred[sel]
            target = target[sel]
            if len(game_names) > 0:
                game_names = game_names[sel]

        if len(game_names) == len(pred) and len(set(game_names)) > 1:
            for gi, gname in enumerate(sorted(set(game_names))):
                gmask = game_names == gname
                if not gmask.any():
                    continue
                ax.scatter(
                    target[gmask],
                    pred[gmask],
                    s=6, alpha=0.45, edgecolors="none",
                    color=_get_game_color(gname, color_map, gi),
                    label=gname, rasterized=True,
                )

            handles, labels = ax.get_legend_handles_labels()
            if labels:
                by_label = dict(zip(labels, handles))
                ax.legend(
                    by_label.values(),
                    by_label.keys(),
                    fontsize=6,
                    loc="upper right",
                    framealpha=0.8,
                )
        else:
            ax.scatter(target, pred, s=6, alpha=0.45, edgecolors="none", color="#2166ac")

        # y=x 기준선
        lo = float(min(target.min(), pred.min())) if len(pred) else 0.0
        hi = float(max(target.max(), pred.max())) if len(pred) else 1.0
        ax.plot([lo, hi], [lo, hi], linestyle="--", color="#888", linewidth=1)
        r, slope, intercept = _compute_scatter_trendline(target, pred)
        if np.isfinite(r):
            ax.plot(
                [lo, hi],
                [slope * lo + intercept, slope * hi + intercept],
                linestyle="-",
                color="#1b9e77",
                linewidth=1,
            )

        name = _REWARD_ENUM_NAMES.get(int(e), f"enum_{e}")
        mae = float(np.mean(np.abs(pred - target))) if len(pred) else float("nan")
        r_txt = f"{r:.4f}" if np.isfinite(r) else "nan"
        ax.set_title(f"{name}\nMAE={mae:.4f} | r={r_txt}", fontsize=8)
        ax.set_xlabel("target", fontsize=7)
        ax.set_ylabel("pred", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(alpha=0.25)

    # 빈 subplot 숨김
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)

    fig.suptitle(f"Train-set Prediction Scatter ({space} space)", fontsize=10)
    fig.tight_layout()
    path = os.path.join(out_dir, f"train_scatter_{space}.png")
    fig.savefig(path, dpi=220)
    plt.close(fig)
    logger.info("Scatter plot saved: %s", path)
    return path


def create_regression_scatter_plots_per_enum(
    scatter_data: Dict[int, Dict[str, np.ndarray]],
    out_dir: str,
    max_points: int = 1000,
    seed: int = 0,
    space: str = "raw",
    game_colors: Optional[Dict[str, str]] = None,
) -> Dict[int, str]:
    """Enum별 regression scatter (pred vs target) 이미지를 개별 저장해 경로를 반환한다.

    Returns
    -------
    {reward_enum: image_path}
    """
    if not scatter_data:
        logger.warning("create_regression_scatter_plots_per_enum: empty scatter_data — skipping")
        return {}

    os.makedirs(out_dir, exist_ok=True)
    pred_key = f"pred_{space}"
    target_key = f"target_{space}"

    color_map = {**_GAME_SCATTER_COLORS, **(game_colors or {})}
    pred_paths: Dict[int, str] = {}
    rng = np.random.RandomState(seed)
    for e in sorted(scatter_data.keys()):
        pred = np.asarray(scatter_data[e].get(pred_key, np.array([])))
        target = np.asarray(scatter_data[e].get(target_key, np.array([])))
        game_names = np.asarray(scatter_data[e].get("game_names", np.array([])), dtype=object)

        n_pts = len(pred)
        if n_pts == 0:
            continue

        if n_pts > max_points:
            sel = rng.choice(n_pts, size=max_points, replace=False)
            pred = pred[sel]
            target = target[sel]
            if len(game_names) > 0:
                game_names = game_names[sel]

        fig, ax = plt.subplots()
        fig.set_size_inches(2.9, 2.5)
        if len(game_names) == len(pred) and len(set(game_names)) > 1:
            for gi, gname in enumerate(sorted(set(game_names))):
                gmask = game_names == gname
                if not gmask.any():
                    continue
                ax.scatter(
                    target[gmask], pred[gmask],
                    s=6, alpha=0.45, edgecolors="none",
                    color=_get_game_color(gname, color_map, gi),
                    label=gname, rasterized=True,
                )
            handles, labels = ax.get_legend_handles_labels()
            if labels:
                by_label = dict(zip(labels, handles))
                ax.legend(
                    by_label.values(),
                    by_label.keys(),
                    fontsize=6,
                    loc="upper right",
                    framealpha=0.8,
                )
        else:
            ax.scatter(target, pred, s=6, alpha=0.45, edgecolors="none", color="#2166ac")

        lo = float(min(target.min(), pred.min()))
        hi = float(max(target.max(), pred.max()))
        ax.plot([lo, hi], [lo, hi], linestyle="--", color="#888", linewidth=1)
        r, slope, intercept = _compute_scatter_trendline(target, pred)
        if np.isfinite(r):
            ax.plot(
                [lo, hi],
                [slope * lo + intercept, slope * hi + intercept],
                linestyle="-",
                color="#1b9e77",
                linewidth=1,
            )

        name = _REWARD_ENUM_NAMES.get(int(e), f"enum_{e}")
        mae = float(np.mean(np.abs(pred - target)))
        r_txt = f"{r:.4f}" if np.isfinite(r) else "nan"
        ax.set_title(f"{name}\\nMAE={mae:.4f} | r={r_txt}", fontsize=8)
        ax.set_xlabel("target", fontsize=7)
        ax.set_ylabel("pred", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(alpha=0.25)

        fig.tight_layout()
        path = os.path.join(out_dir, f"train_regression_scatter_{space}_enum_{int(e)}.png")
        fig.savefig(path, dpi=220)
        plt.close(fig)
        pred_paths[int(e)] = path

    if pred_paths:
        logger.info("Regression scatter plots (per enum, %s space) saved: %d", space, len(pred_paths))
    return pred_paths


@partial(jit, static_argnums=(1,))
def extract_embeddings_batch(
    train_state: TrainState,
    mode: str,
    input_ids: jnp.ndarray,
    attention_mask: jnp.ndarray,
    pixel_values: jnp.ndarray,
    reward_enum: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """JIT-컴파일된 배치 임베딩 추출.
    
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
    """데이터셋에서 무작위로 n_samples 개를 뽑아 텍스트·레벨 임베딩을 수집한다.

    JAX 연산은 메인 스레드에서 실행하고 결과를 numpy 로 반환한다.

    Parameters
    ----------
    game_names : 전체 dataset 인덱스에 대응하는 게임 이름 배열.
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
        bidx   = indices[start:end]          # 실제 dataset 인덱스
        actual = len(bidx)

        # 배치 크기 통일 (패딩)
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
        all_gnames.extend(game_names[bidx].tolist())   # bidx = dataset 인덱스

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
    """t-SNE를 계산하고 wandb에 업로드한다.

    Parameters
    ----------
    has_state : mode 에 'state' 가 포함되는지 여부.
                False 이거나 state_embeds 가 모두 0 이면 텍스트 임베딩만 시각화.
    """
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        logger.warning("scikit-learn 이 설치되지 않아 t-SNE 를 건너뜁니다.")
        return

    try:
        # ── 사용할 임베딩 결정 ─────────────────────────────────────────────
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

        # ── L2 정규화 ───────────────────────────────────────────────────────
        norms    = np.linalg.norm(combined, axis=1, keepdims=True)
        combined = combined / (norms + 1e-8)

        # ── t-SNE ───────────────────────────────────────────────────────────
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

        # ── 플롯 ────────────────────────────────────────────────────────────
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

        # ── 범례: 중복 제거 ─────────────────────────────────────────────────
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

        # ── modality 범례 설명 (마커 모양 안내) ────────────────────────────
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
        logger.warning("t-SNE 계산 실패 (epoch %d, tag=%s): %s", epoch, tag, exc)

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

from encoder.visualization.common import GAME_COLORS, TYPE_MARKERS


def save_embedding_npz(
    path: Path,
    embeddings: np.ndarray,
    games: np.ndarray,
    types: np.ndarray,
    texts: np.ndarray,
    source_indices: np.ndarray,
    reward_enums: np.ndarray,
    condition_values: np.ndarray,
    quantized_bins: np.ndarray,
) -> None:
    np.savez_compressed(
        path,
        embeddings=embeddings,
        games=games,
        types=types,
        texts=texts,
        source_indices=source_indices,
        reward_enums=reward_enums,
        condition_values=condition_values,
        quantized_bins=quantized_bins,
    )


def save_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w") as f:
        json.dump(data, f, indent=2, default=str)


def save_csv(
    path: Path,
    coords: np.ndarray,
    games: np.ndarray,
    types: np.ndarray,
    texts: np.ndarray,
    source_indices: np.ndarray,
    reward_enums: np.ndarray,
    condition_values: np.ndarray,
    quantized_bins: np.ndarray,
) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        header = ["x", "y"]
        if coords.shape[1] == 3:
            header.append("z")
        writer.writerow(
            header
            + ["game", "type", "text", "source_index", "reward_enum", "condition_value", "quantized_bin"]
        )
        for xy, game, typ, text, idx, reward_enum, condition_value, quantized_bin in zip(
            coords,
            games,
            types,
            texts,
            source_indices,
            reward_enums,
            condition_values,
            quantized_bins,
        ):
            row = [float(xy[0]), float(xy[1])]
            if coords.shape[1] == 3:
                row.append(float(xy[2]))
            writer.writerow(
                row
                + [
                    game,
                    typ,
                    "" if text is None else str(text),
                    int(idx),
                    int(reward_enum),
                    "" if np.isnan(condition_value) else float(condition_value),
                    int(quantized_bin),
                ]
            )


def save_plot(path: Path, coords: np.ndarray, games: np.ndarray, types: np.ndarray, title: str) -> None:
    unique_games = sorted(set(games.tolist()))
    is_3d = coords.shape[1] == 3
    fig = plt.figure(figsize=(5.4, 5.4))
    ax = fig.add_subplot(111, projection="3d") if is_3d else fig.add_subplot(111)

    for game in unique_games:
        for typ, marker in TYPE_MARKERS.items():
            mask = (games == game) & (types == typ)
            if not np.any(mask):
                continue
            scatter_kwargs = {
                "c": GAME_COLORS.get(game, "#7f7f7f"),
                "marker": marker,
                "s": 14 if typ == "state" else 20,
                "alpha": 0.78,
                "linewidths": 0.6,
                "edgecolors": "white" if typ == "state" else "none",
            }
            if is_3d:
                ax.scatter(coords[mask, 0], coords[mask, 1], coords[mask, 2], **scatter_kwargs)
            else:
                ax.scatter(coords[mask, 0], coords[mask, 1], **scatter_kwargs)

    game_handles = [
        mlines.Line2D(
            [],
            [],
            color=GAME_COLORS.get(game, "#7f7f7f"),
            marker="o",
            linestyle="None",
            markersize=8,
            markeredgecolor="white",
            markeredgewidth=0.8,
            label=game,
        )
        for game in unique_games
    ]
    type_handles = [
        mlines.Line2D([], [], color="grey", marker="o", linestyle="None", markersize=8, label="state"),
        mlines.Line2D([], [], color="grey", marker="x", linestyle="None", markersize=8, label="text"),
    ]

    game_legend = ax.legend(handles=game_handles, loc="upper right", title="Game", frameon=True)
    ax.add_artist(game_legend)
    if len(set(types.tolist())) > 1:
        ax.legend(handles=type_handles, loc="lower right", title="Type", frameon=True)

    ax.set_title(title)
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    if is_3d:
        ax.set_zlabel("t-SNE dim 3")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)

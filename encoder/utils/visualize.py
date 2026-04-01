import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

import matplotlib.colors as mcolors
import matplotlib.cm as cm
from sklearn.manifold import TSNE
from encoder.utils.game_palette import palette_for_games


def create_scatter_plot(df, epoch, config, min_val=0, max_val=1,
                        xlim=(-10, 10), ylim=(-10, 10), postfix=""):

    """
    drawing Ground Truth vs Prediction
    - hue: trade as epoch
    - colorbar range: min_val ~ max_val (n_epochs)
    """

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(6.4, 4.8))

    # color normalize
    norm = mcolors.Normalize(vmin=min_val, vmax=max_val)
    sm = cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])

    # generate Scatter plot
    _ = sns.scatterplot(
        data=df, x="ground_truth", y="prediction",
        hue="reward_id", palette="bright", alpha=0.5, ax=ax
    )

    # add Colorbar
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Epoch")

    ax.set_ylim(*ylim)
    ax.set_xlim(*xlim)
    ax.set_xlabel("Ground Truth")
    ax.set_ylabel("Prediction")

    ax.grid(True)
    plt.tight_layout()

    # save
    os.makedirs(os.path.join(config.exp_dir, config.figure_dir), exist_ok=True)
    fig_path = os.path.join(config.exp_dir, config.figure_dir, f"scatter_epoch_{epoch}{postfix}.png")
    plt.savefig(fig_path)
    plt.close(fig)

    return fig_path


def create_embedding_figure(embed_queue, reward_df: pd.DataFrame, epoch, config, postfix="") -> str:
    reward_ids = [e.reward_id for e in embed_queue]
    embeds = np.array([e.embedding for e in embed_queue])

    # TSNE (2dim)
    tsne = TSNE(n_components=2, random_state=42)
    tsne_embeds = tsne.fit_transform(embeds)

    inst_cols = reward_df.iloc[reward_ids][['reward_enum']].reset_index()
    tsne_df = pd.DataFrame(tsne_embeds, columns=['tsne_x', 'tsne_y']).reset_index()
    df = pd.concat([inst_cols, tsne_df], axis=1).drop(columns=['index'])

    # draw scatter plot
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    sns.scatterplot(
        data=df, x="tsne_x", y="tsne_y",
        hue="reward_enum", palette="bright", alpha=0.9, ax=ax
    )

    ax.set_xlabel("Projection X")
    ax.set_ylabel("Projection Y")
    ax.grid(True)
    plt.tight_layout()

    # save
    os.makedirs(os.path.join(config.exp_dir, config.figure_dir), exist_ok=True)
    fig_path = os.path.join(config.exp_dir, config.figure_dir, f"embed_epoch_{epoch}{postfix}.png")
    plt.savefig(fig_path)
    plt.close(fig)

    return fig_path


def create_clip_embedding_figures(embed_queue, class_id2reward_cond, epoch, config, postfix=""):
    # data set
    state_embeddings = np.array([e.state_embeddings for e in embed_queue])
    text_embeddings = np.array([e.text_embeddings for e in embed_queue])
    class_ids = np.array([e.class_ids for e in embed_queue])

    all_embeddings = np.concatenate([state_embeddings, text_embeddings], axis=0)
    modalities = np.array(["State"] * len(state_embeddings) + ["text"] * len(text_embeddings))

    class_ids = np.concatenate([class_ids, class_ids], axis=0)

    # Extract reward_cond tuples from class_id2reward_cond mapping
    # class_id2reward_cond should be a dictionary: {class_id_int: (game_idx, reward_enum, condition_value)}
    reward_cond_tuples = []
    for cid in class_ids:
        # Ensure cid is a Python int, not numpy int
        cid_int = int(cid)

        # Handle both dict and other possible types
        if isinstance(class_id2reward_cond, dict):
            reward_cond_tuple = class_id2reward_cond.get(cid_int, None)
        else:
            print(f"Warning: class_id2reward_cond is not a dict, it's {type(class_id2reward_cond)}")
            reward_cond_tuple = None

        if reward_cond_tuple is not None:
            reward_cond_tuples.append(reward_cond_tuple)
        else:
            reward_cond_tuples.append(None)

    # Extract game indices from reward_cond tuples
    # reward_cond tuple structure: (game_idx, reward_enum, condition_value)
    game_indices = []
    for rc_tuple in reward_cond_tuples:
        if rc_tuple is not None and isinstance(rc_tuple, tuple) and len(rc_tuple) > 0:
            game_idx = rc_tuple[0]  # First element is game_idx
            game_indices.append(game_idx)
        else:
            game_indices.append(None)

    # Mapping from game_idx to game_name
    game_idx_mapping = {
        0: "Dungeon",
        1: "Sokoban",
        2: "Pokemon",
        3: "Zelda",
        4: "Doom",
    }

    # Convert game indices to game names, filter out None values
    valid_indices = [i for i, g in enumerate(game_indices) if g is not None]
    if len(valid_indices) == 0:
        print("Warning: No valid game labels found")
        return None

    game_labels = [game_idx_mapping.get(game_indices[i], f"Game_{game_indices[i]}") for i in valid_indices]
    all_embeddings = all_embeddings[valid_indices]
    modalities_filtered = [modalities[i] for i in valid_indices]
    modalities = np.array(modalities_filtered)

    # Shared palette (same colors as decoder scatter)
    game_color = palette_for_games(game_labels)

    save_dir = os.path.join(config.exp_dir, config.figure_dir)
    os.makedirs(save_dir, exist_ok=True)

    def compute_tsne_and_df(embeddings, game_labels, modalities):
        tsne = TSNE(n_components=2, random_state=42)
        tsne_embeds = tsne.fit_transform(embeddings)
        df = pd.DataFrame(tsne_embeds, columns=["tsne_x", "tsne_y"])
        unique_games = sorted(set(game_labels))
        df["Game"] = pd.Categorical(game_labels, categories=unique_games, ordered=True)
        df["modality"] = modalities
        return df

    
    def plot_seaborn(df, hue, style, markers, filename, style_order=None):
        fig, ax = plt.subplots(figsize=(6.4, 4.6))
        sns.set_theme(style="whitegrid")
        sns.scatterplot(
            data=df,
            x="tsne_x", y="tsne_y",
            hue=hue,
            style=style,
            markers=markers,
            palette=game_color,
            alpha=0.9,
            s=40,
            ax=ax,
            style_order=style_order
        )
        ax.set_xlabel("Projection X")
        ax.set_ylabel("Projection Y")
        ax.legend(
            bbox_to_anchor=(1.05, 1),
            loc='upper left',
            borderaxespad=0.
        ).set_title(None)
        ax.grid(True)
        plt.tight_layout()
        path = os.path.join(save_dir, filename)
        plt.savefig(path)
        plt.close(fig)
        return path
    
    # total tsne+DF
    tsne_df = compute_tsne_and_df(all_embeddings, game_labels, modalities)

    # graph 2: total + State/Text
    path = plot_seaborn(
        tsne_df,
        hue="Game",
        style="modality",
        markers={"State": "o", "text": "X"},
        filename=f"embed_epoch_{epoch}_modality_all{postfix}.png"
    )

    return path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

import matplotlib.colors as mcolors
import matplotlib.cm as cm
from sklearn.manifold import TSNE


def create_scatter_plot(df, epoch, config, min_val=0, max_val=1,
                        xlim=(-10, 10), ylim=(-10, 10), postfix=""):

    """
    drawing Ground Truth vs Prediction
    - hue: trade as epoch
    - colorbar range: min_val ~ max_val (n_epochs)
    """

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 6))

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
    fig, ax = plt.subplots(figsize=(6, 5))
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
    class_ids = np.array([tuple(e.class_ids) if isinstance(e.class_ids, np.ndarray) else e.class_ids for e in embed_queue])

    all_embeddings = np.concatenate([state_embeddings, text_embeddings], axis=0)
    modalities = np.array(["State"] * len(state_embeddings) + ["text"] * len(text_embeddings))
    class_ids = np.concatenate([class_ids, class_ids], axis=0)
    reward_conds = [class_id2reward_cond.get(cid, "unknown")[0] for cid in class_ids]

    # mapping
    task_map = {
        1: "RG", 6: "RG",
        2: "PL", 7: "PL",
        3: "WC", 8: "WC",
        4: "BC", 9: "BC",
        5: "BD", 10: "BD"
    }
    style_map = {
        1: "Human", 2: "Human", 3: "Human", 4: "Human", 5: "Human",
        6: "AI", 7: "AI", 8: "AI", 9: "AI", 10: "AI"
    }

    task_labels = [task_map.get(r, f"unknown-{r}") for r in reward_conds]
    style_labels = [style_map.get(r, "unknown") for r in reward_conds]

    task_color_map = {
        "RG": "#e41a1c",
        "PL": "#ff7f00",
        "WC": "#377eb8",
        "BC": "#4daf4a",
        "BD": "#984ea3"
    }
    ordered_tasks = ["RG", "PL", "WC", "BC", "BD"]

    save_dir = os.path.join(config.exp_dir, config.figure_dir)
    os.makedirs(save_dir, exist_ok=True)
    def compute_tsne_and_df(embeddings, task_labels, modalities, styles):
        tsne = TSNE(n_components=2, random_state=42)
        tsne_embeds = tsne.fit_transform(embeddings)
        df = pd.DataFrame(tsne_embeds, columns=["tsne_x", "tsne_y"])
        df["Task"] = pd.Categorical(task_labels, categories=ordered_tasks, ordered=True)
        df["modality"] = modalities
        df["Style"] = styles
        return df

    
    def plot_seaborn(df, hue, style, markers, filename, style_order=None):
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.set_theme(style="whitegrid")
        sns.scatterplot(
            data=df,
            x="tsne_x", y="tsne_y",
            hue=hue,
            style=style,
            markers=markers,
            palette=task_color_map,
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
    
    # total tsnd+DF
    tsne_df = compute_tsne_and_df(all_embeddings, task_labels, modalities, style_labels)

    # graph 1: Human vs AI
    path1 = plot_seaborn(
        tsne_df,
        hue="Task",
        style="Style",
        markers={"Human": "o", "AI": "^"},
        filename=f"embed_epoch_{epoch}_human_vs_ai{postfix}.png",
        style_order=["Human", "AI"]
    )

    # graph 2: total + State/Text
    path2 = plot_seaborn(
        tsne_df,
        hue="Task",
        style="modality",
        markers={"State": "o", "text": "X"},
        filename=f"embed_epoch_{epoch}_modality_all{postfix}.png"
    )

    human_mask = np.array(style_labels) == "Human"
    human_embeddings = all_embeddings[human_mask]
    human_task_labels = [task_labels[i] for i in range(len(task_labels)) if human_mask[i]]
    human_modalities = modalities[human_mask]
    human_styles = [style_labels[i] for i in range(len(style_labels)) if human_mask[i]]

    tsne_df_human = compute_tsne_and_df(human_embeddings, human_task_labels, human_modalities, human_styles)

    path3 = plot_seaborn(
        tsne_df_human,
        hue="Task",
        style="modality",
        markers={"State": "o", "text": "X"},
        filename=f"embed_epoch_{epoch}_modality_human_only{postfix}.png"
    )

    return path1, path2, path3

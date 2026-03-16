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



def create_clip_embedding_figures_sketch(embed_queue, class_id2reward_cond, epoch, config, mode="text_state_sketch", postfix=""):
    use_text = "text" in mode
    use_state = "state" in mode
    use_sketch = "sketch" in mode and hasattr(embed_queue[0], 'sketch_embeddings')

    embeddings_list = []
    modalities_list = []
    class_ids_list = []

    if use_state:
        state_embeddings = np.array([e.state_embeddings for e in embed_queue])
        embeddings_list.append(state_embeddings)
        modalities_list += ["State"] * len(state_embeddings)
        class_ids_list.append(np.array([tuple(e.class_ids) if isinstance(e.class_ids, np.ndarray) else e.class_ids for e in embed_queue]))

    if use_text:
        text_embeddings = np.array([e.text_embeddings for e in embed_queue])
        embeddings_list.append(text_embeddings)
        modalities_list += ["text"] * len(text_embeddings)
        class_ids_list.append(np.array([tuple(e.class_ids) if isinstance(e.class_ids, np.ndarray) else e.class_ids for e in embed_queue]))

    if use_sketch:
        sketch_embeddings = np.array([e.sketch_embeddings for e in embed_queue])
        embeddings_list.append(sketch_embeddings)
        modalities_list += ["sketch"] * len(sketch_embeddings)
        class_ids_list.append(np.array([tuple(e.class_ids) if isinstance(e.class_ids, np.ndarray) else e.class_ids for e in embed_queue]))


    if not embeddings_list:
        raise ValueError(
            f"No valid modalities selected from mode='{mode}'. Must include at least one of: 'text', 'state', 'sketch'.")

    all_embeddings = np.concatenate(embeddings_list, axis=0)
    modalities = np.array(modalities_list)
    class_ids = np.concatenate(class_ids_list, axis=0)

    reward_ids = [class_id2reward_cond.get(cid, "unknown")[0] for cid in class_ids]
    raw_style_labels = [class_id2reward_cond.get(cid, "unknown")[2] for cid in class_ids]
    style_labels = ["AI" if lbl == "AI" else "Human" for lbl in raw_style_labels]
    
    condition_lebels = [class_id2reward_cond.get(cid, "unknown")[1] for cid in class_ids]
    cond_style_labels = [f"{cond}_{style}" for cond, style in zip(condition_lebels, raw_style_labels)]
    

    
    
    # label mapping
    task_map = {
        '1': "RG",
        '2': "PL",
        '3': "WC", 
        '4': "BC", 
        '5': "BD", 
    }
    task_labels = [task_map.get(r, f"unknown-{r}") for r in reward_ids]

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
    def compute_tsne_and_df(embeddings, task_labels, modalities, conditions, styles):
        tsne = TSNE(n_components=2, random_state=42)
        tsne_embeds = tsne.fit_transform(embeddings)
        df = pd.DataFrame(tsne_embeds, columns=["tsne_x", "tsne_y"])
        if len(set(task_labels)) > 1:
            df["Task"] = pd.Categorical(task_labels, categories=ordered_tasks, ordered=True)
        else:
            df["Task"] = task_labels
        # df["Task"] = pd.Categorical(task_labels, categories=ordered_tasks, ordered=True)
        df["modality"] = modalities
        df["Condition"] = conditions
        df["Style"] = styles
        return df

    
    def plot_seaborn(df, hue, style, markers, filename, style_order=None, hue_order=None, palette=None):
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.set_theme(style="whitegrid")
        sns.scatterplot(
            data=df,
            x="tsne_x", y="tsne_y",
            hue=hue,
            style=style,
            markers=markers,
            palette=palette,
            alpha=0.9,
            s=40,
            ax=ax,
            style_order=style_order,
            hue_order=hue_order
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

    tsne_df = compute_tsne_and_df(all_embeddings, task_labels, modalities, condition_lebels, style_labels)

    
    # graph 1: Human vs AI
    path1 = plot_seaborn(
        tsne_df,
        hue="Task",
        style="Style",
        markers={"Human": "o", "AI": "^"},
        filename=f"embed_epoch_{epoch}_human_vs_ai{postfix}.png",
        style_order=["Human", "AI"],
        palette=task_color_map
    )

    # graph 2: total + State/Text
    markers_dict = {}
    if use_state: markers_dict["State"] = "o"
    if use_text: markers_dict["text"] = "X"
    if use_sketch: markers_dict["sketch"] = "D"

    path2 = plot_seaborn(
        tsne_df,
        hue="Task",
        style="modality",
        markers=markers_dict,
        filename=f"embed_epoch_{epoch}_modality_all{postfix}.png",
        palette=task_color_map
    )

    human_mask = np.array(style_labels) == "Human"
    human_embeddings = all_embeddings[human_mask]
    human_task_labels = [task_labels[i] for i in range(len(task_labels)) if human_mask[i]]
    human_modalities = modalities[human_mask]
    human_conditions = np.array(condition_lebels)[human_mask]
    human_styles = [style_labels[i] for i in range(len(style_labels)) if human_mask[i]]

    tsne_df_human = compute_tsne_and_df(human_embeddings, human_task_labels, human_modalities, human_conditions, human_styles)

    path3 = plot_seaborn(
        tsne_df_human,
        hue="Task",
        style="modality",
        markers={"State": "o", "text": "X", "sketch": "D"},
        filename=f"embed_epoch_{epoch}_modality_human_only{postfix}.png",
        palette=task_color_map
    )
    
    unique_tasks = sorted(list(set(task_labels))) 
    task_paths = {}
    for task in unique_tasks:
        if task.startswith("unknown"): continue

        # filter current task
        task_mask = np.array(task_labels) == task
        if not np.any(task_mask): continue

        task_embeddings_subset = all_embeddings[task_mask]
        task_modalities_subset = modalities[task_mask]
        task_styles_subset = np.array(style_labels)[task_mask]
        task_condition_styles_subset = np.array(cond_style_labels)[task_mask]
        task_labels_subset = [task] * len(task_embeddings_subset)
        
        # tsnd/DF
        tsne_df_task = compute_tsne_and_df(
            task_embeddings_subset,
            task_labels_subset,
            task_modalities_subset,
            task_condition_styles_subset,
            task_styles_subset
        )
        unique_conditions = sorted(tsne_df_task["Condition"].unique())
        
        colors = sns.color_palette('Paired', len(unique_conditions))
        palette = {cond: colors[i] for i, cond in enumerate(unique_conditions)}
        path = plot_seaborn(
            df=tsne_df_task,
            hue="Condition",
            style="modality",
            markers={"State": "o", "text": "X", "sketch": "D"},
            filename=f"embed_epoch_{epoch}_{task}{postfix}.png",
            palette=palette,
            hue_order=unique_conditions
        )
        task_paths[task] = path
    
    return path1, path2, path3, task_paths



def create_clip_embedding_figures(embed_queue, class_id2reward_cond, epoch, config, postfix=""):
    # data set
    state_embeddings = np.array([e.state_embeddings for e in embed_queue])
    text_embeddings = np.array([e.text_embeddings for e in embed_queue])
    sketch_embeddings = np.array([e.sketch_embeddings for e in embed_queue])
    class_ids = np.array([tuple(e.class_ids) if isinstance(e.class_ids, np.ndarray) else e.class_ids for e in embed_queue])

    all_embeddings = np.concatenate([state_embeddings, text_embeddings, sketch_embeddings], axis=0)
    modalities = np.array(["State"] * len(state_embeddings) + ["text"] * len(text_embeddings) + ["sketch"] * len(sketch_embeddings))
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
        markers={"State": "o", "text": "X", "sketch": "^"},
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
        markers={"State": "o", "text": "X", "sketch": "^"},
        filename=f"embed_epoch_{epoch}_modality_human_only{postfix}.png"
    )

    return path1, path2, path3

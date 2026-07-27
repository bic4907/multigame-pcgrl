"""
image_utils.py
==============
Utilities for sampling evaluation images and overlaying text.
"""
import textwrap

import numpy as np
import wandb
from PIL import Image, ImageDraw, ImageFont


def annotate_image(img_arr: np.ndarray, game: str, instruction: str, conditions: dict) -> Image.Image:
    """Overlay game, instruction, and condition text at the top of a NumPy image.

    Args:
        img_arr    : (H, W, C) uint8 or float32 image.
        game       : game name string.
        instruction: instruction string.
        conditions : {column_name: float} dictionary; NaN values are omitted.

    Returns:
        PIL.Image (RGB).
    """
    if img_arr.dtype != np.uint8:
        img_arr = (np.clip(img_arr, 0, 1) * 255).astype(np.uint8)

    h, w = img_arr.shape[:2]
    font = ImageFont.load_default()

    instr_str = instruction if instruction else ''
    cond_str = '  '.join(
        f"{c.replace('condition_', 'c')}={v:.2f}"
        for c, v in conditions.items()
        if not np.isnan(float(v))
    )
    lines = [f"[{game}]"] + textwrap.wrap(instr_str, width=40) + ([cond_str] if cond_str else [])

    line_h = 14
    pad = line_h * len(lines) + 6
    canvas = Image.new('RGB', (w, h + pad), color=(30, 30, 30))
    draw = ImageDraw.Draw(canvas)
    for li, line in enumerate(lines):
        draw.text((4, 3 + li * line_h), line, fill=(255, 255, 180), font=font)
    canvas.paste(Image.fromarray(img_arr), (0, pad))
    return canvas


def sample_wandb_images(
    df_ctrl_sim,
    eval_env_maps: list,   # Per-batch env_map list; each item is (n_envs, H, W) uint8
    n_rows: int,
    n_samples: int = 16,
    seed: int = 0,
    tile_size: int = 16,
) -> list:
    """Sample non-overlapping conditions from df_ctrl_sim and return wandb.Images.

    Render env_map (state) on demand with render_unified_rgb.

    Args:
        df_ctrl_sim  : all evaluation results DataFrame.
        eval_env_maps: per-batch env_map list; each item is (n_envs, H, W) uint8.
        n_rows       : number of valid samples, used to remove padding.
        n_samples    : maximum number of images to upload (default: 16).
        seed         : random seed for sampling.
        tile_size    : rendered tile cell size (default: 16).
    """
    from envs.probs.multigame import render_multigame_map

    all_env_maps = np.concatenate(eval_env_maps, axis=0)[:n_rows]  # (n_rows, H, W)
    cond_cols = [c for c in df_ctrl_sim.columns if c.startswith('condition_')]

    # Select one seed==0 row per row_i, yielding one sample per unique instruction
    first_per_row = (
        df_ctrl_sim[df_ctrl_sim['seed'] == 0]
        .drop_duplicates(subset='row_i')
        .reset_index()  # Preserve the original DataFrame index (= all_env_maps index)
    )
    sample_df = first_per_row.sample(
        n=min(n_samples, len(first_per_row)),
        random_state=seed,
    ).reset_index(drop=True)

    wandb_images = []
    for _, srow in sample_df.iterrows():
        orig_idx = int(srow['index'])
        env_map = all_env_maps[orig_idx]                           # (H, W) uint8
        img_arr = np.array(
            render_multigame_map(env_map.astype(np.int32), tile_size=tile_size)
        )  # (H*ts, W*ts, 3) uint8
        conditions = {c: srow[c] for c in cond_cols if c in srow}
        pil_img = annotate_image(
            img_arr,
            game=str(srow.get('game', '')),
            instruction=str(srow.get('instruction', '')),
            conditions=conditions,
        )
        caption = f"[{srow.get('game', '')}] {str(srow.get('instruction', ''))[:60]}"
        wandb_images.append(wandb.Image(pil_img, caption=caption))

    return wandb_images

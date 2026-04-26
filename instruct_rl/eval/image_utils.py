"""
image_utils.py
==============
평가 결과 이미지를 샘플링하고 텍스트를 오버레이하는 유틸리티.
"""
import textwrap

import numpy as np
import wandb
from PIL import Image, ImageDraw, ImageFont


def annotate_image(img_arr: np.ndarray, game: str, instruction: str, conditions: dict) -> Image.Image:
    """numpy 이미지 배열 상단에 game / instruction / condition 텍스트를 오버레이.

    Args:
        img_arr    : (H, W, C) uint8 또는 float32 이미지.
        game       : 게임 이름 문자열.
        instruction: 명령어 문자열.
        conditions : {column_name: float} 딕셔너리 (NaN인 값은 생략).

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
    eval_env_maps: list,   # 배치별 env_map list (각 원소: (n_envs, H, W) uint8)
    n_rows: int,
    n_samples: int = 16,
    seed: int = 0,
    tile_size: int = 16,
) -> list:
    """df_ctrl_sim에서 조건이 겹치지 않도록 샘플을 뽑아 wandb.Image 리스트 반환.

    env_map(state)을 받아 render_unified_rgb 로 on-demand 렌더링한다.

    Args:
        df_ctrl_sim  : 전체 평가 결과 DataFrame.
        eval_env_maps: 배치별 env_map list (각 원소: (n_envs, H, W) uint8).
        n_rows       : 실제 유효 샘플 수 (패딩 제거용).
        n_samples    : 최대 업로드 이미지 수 (기본 16).
        seed         : 샘플링 random seed.
        tile_size    : 렌더링 타일 픽셀 크기 (기본 16).
    """
    from envs.probs.multigame import render_multigame_map

    all_env_maps = np.concatenate(eval_env_maps, axis=0)[:n_rows]  # (n_rows, H, W)
    cond_cols = [c for c in df_ctrl_sim.columns if c.startswith('condition_')]

    # seed==0 행에서 row_i별 1개씩 추출 → unique 명령어당 1샘플
    first_per_row = (
        df_ctrl_sim[df_ctrl_sim['seed'] == 0]
        .drop_duplicates(subset='row_i')
        .reset_index()  # 원래 DataFrame 인덱스(= all_env_maps 인덱스) 보존
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


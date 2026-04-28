"""
instruct_rl/utils/render_clip_224.py
======================================
env_map (H, W) int32 → 224x224 CLIP-normalized image 렌더링 유틸.

두 가지 경로:
  - 오프라인 (데이터셋 전처리):  render_map_numpy_224 — PIL/numpy 기반, 고품질
  - 온라인 (RL 학습 중):         build_jax_render_224_fn — 순수 JAX, jit 가능

CLIP 정규화: mean=[0.48145466, 0.4578275, 0.40821073]
             std =[0.26862954, 0.26130258, 0.27577711]
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

CLIP_IMAGE_SIZE = 224  # pretrained CLIP 입력 크기 (16 × 14 = 224)

CLIP_MEAN_NP = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD_NP  = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

_CLIP_MEAN_JAX = jnp.array([0.48145466, 0.4578275, 0.40821073])
_CLIP_STD_JAX  = jnp.array([0.26862954, 0.26130258, 0.27577711])


# ── 오프라인 (numpy / PIL) ─────────────────────────────────────────────────────

def build_numpy_tile_tensor(map_size: int = 16) -> np.ndarray:
    """multigame 타일 이미지를 target_ts × target_ts 로 리사이즈하고 CLIP 정규화한 numpy 배열을 반환한다.

    온라인 JAX 렌더링과 동일한 파이프라인을 오프라인(numpy)으로 구현.

    Returns
    -------
    (num_tiles, target_ts, target_ts, 3) float32, CLIP-normalized.
    """
    from PIL import Image
    from envs.probs.multigame import _load_tile_image, _load_or_color_tile, _BORDER_IMAGE, _CATEGORIES

    target_ts = CLIP_IMAGE_SIZE // map_size   # 14

    # BORDER(0) + category tiles(1~N)
    num_tiles = 1 + len(_CATEGORIES)
    tile_arr = np.zeros((num_tiles, target_ts, target_ts, 3), dtype=np.float32)

    def _pil_to_norm(pil_img):
        pil_img = pil_img.convert("RGB").resize((target_ts, target_ts), Image.BILINEAR)
        arr = np.array(pil_img, dtype=np.float32) / 255.0
        return (arr - CLIP_MEAN_NP) / CLIP_STD_NP

    tile_arr[0] = _pil_to_norm(_load_tile_image(_BORDER_IMAGE))
    for cat_idx in _CATEGORIES:
        tile_arr[cat_idx + 1] = _pil_to_norm(_load_or_color_tile(cat_idx))

    return tile_arr   # (num_tiles, 14, 14, 3)


def render_maps_numpy_224(env_maps: np.ndarray, map_size: int = 16) -> np.ndarray:
    """(N, H, W) int32 → (N, 224, 224, 3) float32 CLIP-normalized.

    타일을 14x14로 미리 리사이즈/정규화하고 조립만 수행 — JAX 온라인 경로와 동일 파이프라인.
    """
    target_ts = CLIP_IMAGE_SIZE // map_size   # 14
    tile_arr = build_numpy_tile_tensor(map_size)   # (num_tiles, 14, 14, 3)

    N, H, W = env_maps.shape
    result = np.zeros((N, H * target_ts, W * target_ts, 3), dtype=np.float32)

    for i, env_map in enumerate(env_maps):
        tiles = tile_arr[env_map]                                   # (H, W, 14, 14, 3)
        result[i] = tiles.transpose(0, 2, 1, 3, 4).reshape(H * target_ts, W * target_ts, 3)

    return result   # (N, 224, 224, 3)


# ── 온라인 (JAX, jit 가능) ────────────────────────────────────────────────────


def build_preprocessed_tile_tensor(tile_tensor_raw: jnp.ndarray, map_size: int = 16) -> jnp.ndarray:
    """타일 이미지를 (CLIP_IMAGE_SIZE // map_size) 크기로 리사이즈하고 CLIP 정규화까지 적용한다.

    map_size=16 이면 target_tile_size = 224 // 16 = 14.
    16 × 14 = 224 이므로 조립 후 reshape 만으로 224x224 완성.

    Parameters
    ----------
    tile_tensor_raw : (num_tiles, ts, ts, 4) uint8
        env.prob.graphics JAX 배열.
    map_size : int
        맵 한 변의 타일 수 (예: 16).

    Returns
    -------
    (num_tiles, target_ts, target_ts, 3) float32  — CLIP 정규화 완료.
    """
    target_ts = CLIP_IMAGE_SIZE // map_size   # 14

    # alpha 제거 + [0,1] 변환: (num_tiles, ts, ts, 3)
    rgb = tile_tensor_raw[..., :3].astype(jnp.float32) / 255.0

    # 각 타일을 target_ts × target_ts 로 리사이즈 (vmap, 학습 전 1회)
    def _resize_tile(tile):
        return jax.image.resize(tile, (target_ts, target_ts, 3), method='bilinear')

    resized = jax.vmap(_resize_tile)(rgb)   # (num_tiles, 14, 14, 3)

    # CLIP 정규화 (학습 전 1회)
    normalized = (resized - _CLIP_MEAN_JAX) / _CLIP_STD_JAX  # (num_tiles, 14, 14, 3)
    return normalized


def build_jax_render_224_fn(tile_tensor_raw: jnp.ndarray, map_size: int = 16):
    """jit 가능한 배치 렌더링 함수를 반환한다.

    타일 이미지를 학습 시작 전에 14x14로 리사이즈하고 CLIP 정규화까지 완료해두어,
    매 스텝의 렌더링은 JAX 인덱싱 + reshape 만으로 처리한다.

    Parameters
    ----------
    tile_tensor_raw : jnp.ndarray, shape (num_tiles, ts, ts, 4) uint8
        env.prob.graphics JAX 배열.
    map_size : int
        맵 한 변의 타일 수 (예: 16). CLIP_IMAGE_SIZE // map_size 가 target 타일 크기.

    Returns
    -------
    render_batch : (B, H, W) int32 → (B, 224, 224, 3) float32, jit 가능.
    """
    target_ts = CLIP_IMAGE_SIZE // map_size   # 14

    # ── 학습 전 1회: 타일 리사이즈 + CLIP 정규화 ────────────────────────────
    preprocessed_tiles = build_preprocessed_tile_tensor(tile_tensor_raw, map_size)
    # preprocessed_tiles: (num_tiles, 14, 14, 3) float32, CLIP-normalized

    def _render_single(env_map):
        """(H, W) int32 → (224, 224, 3) float32."""
        H, W = env_map.shape
        # 타일 인덱싱: (H, W) → (H, W, 14, 14, 3)
        tiles = preprocessed_tiles[env_map]
        # (H, 14, W, 14, 3) → (H*14, W*14, 3) = (224, 224, 3)
        return tiles.transpose(0, 2, 1, 3, 4).reshape(H * target_ts, W * target_ts, 3)

    def render_batch(env_maps):
        """(B, H, W) → (B, 224, 224, 3)."""
        return jax.vmap(_render_single)(env_maps)

    return render_batch

"""tests/test_multigame_render.py

MultigameProblem 렌더링 검증 테스트.

검증 항목
---------
1. init_graphics() 이후 graphics 배열이 올바른 shape인지
2. 각 카테고리 타일 색상이 tile_mapping._category_colors_rgb 와 일치하는지
3. 실제 reset 후 render_env_map() 이 PNG 파일로 저장되는지 (파일 생성 검증)
4. 렌더링된 이미지 크기가 map_shape * tile_size 와 일치하는지
5. 모든 카테고리 타일이 그래픽에 포함되어 있는지 (누락 없음)

샘플 이미지
-----------
    python tests/test_multigame_render.py
    → tests/render_samples/ 에 PNG 파일 생성
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_SAMPLES_DIR = _ROOT / "tests" / "render_samples"

from envs.probs.multigame import (
    NUM_CATEGORIES,
    MultigameTiles,
    _CATEGORIES,
    _CATEGORY_COLORS,
    _TILE_SIZE,
    make_multigame_env,
    render_multigame_map,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def env_with_graphics():
    env, env_params = make_multigame_env()
    env.init_graphics()
    return env, env_params


@pytest.fixture(scope="module")
def sample_env_map(env_with_graphics):
    """reset 후 env_map (16×16 numpy array) 반환."""
    import jax
    from envs.pcgrl_env import gen_dummy_queued_state

    env, env_params = env_with_graphics
    dummy_qs = gen_dummy_queued_state(env)
    rng = jax.random.PRNGKey(0)
    obs, state = env.reset(rng, env_params, dummy_qs)
    return np.array(state.env_map)


@pytest.fixture(scope="module")
def all_tiles_map():
    """16×(NUM_CATEGORIES+1) 맵: 각 행에 하나씩 카테고리 타일 배치."""
    H = NUM_CATEGORIES + 1   # BORDER 포함
    W = 16
    m = np.zeros((H, W), dtype=np.int32)
    for tile in MultigameTiles:
        m[int(tile), :] = int(tile)
    return m


# ═══════════════════════════════════════════════════════════════════════════════
# 1. graphics 배열 shape 검증
# ═══════════════════════════════════════════════════════════════════════════════

def test_graphics_shape(env_with_graphics):
    """init_graphics() 후 graphics.shape == (n_tiles, H, W, 4)."""
    env, _ = env_with_graphics
    g = env.prob.graphics
    n_tiles = len(MultigameTiles)
    assert g.shape[0] == n_tiles, (
        f"graphics 타일 수({g.shape[0]}) != MultigameTiles 수({n_tiles})"
    )
    assert g.shape[1] == _TILE_SIZE and g.shape[2] == _TILE_SIZE, (
        f"타일 크기 불일치: {g.shape[1:3]} != ({_TILE_SIZE}, {_TILE_SIZE})"
    )
    assert g.shape[3] == 4, "RGBA 채널(4) 이어야 합니다"


def test_graphics_no_all_zero_tile(env_with_graphics):
    """어떤 타일도 완전히 검은색(RGBA 모두 0)이면 안 된다."""
    import jax.numpy as jnp
    env, _ = env_with_graphics
    g = np.array(env.prob.graphics)
    for tile in MultigameTiles:
        tile_img = g[int(tile)]   # (H, W, 4)
        mean_val = tile_img[..., :3].mean()
        assert mean_val > 0, (
            f"타일 {tile.name}(idx={int(tile)}) 이 완전 검은색입니다 "
            f"(RGB 평균={mean_val:.2f})"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 2. 카테고리 색상 vs tile_mapping._category_colors_rgb 검증
# ═══════════════════════════════════════════════════════════════════════════════

def test_category_colors_loaded():
    """_CATEGORY_COLORS 가 모든 카테고리를 커버해야 한다."""
    for cat_idx in _CATEGORIES:
        assert cat_idx in _CATEGORY_COLORS, (
            f"category {cat_idx}({_CATEGORIES[cat_idx]}) 의 색상이 없습니다"
        )


def test_category_colors_rgb_range():
    """각 RGB 값이 [0, 255] 범위 안이어야 한다."""
    for cat_idx, rgb in _CATEGORY_COLORS.items():
        for ch, v in enumerate(rgb):
            assert 0 <= v <= 255, (
                f"category {cat_idx} 색상 ch={ch} 값={v} 가 범위 초과"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# 3. 파일 생성 검증 (render_env_map → PNG 저장)
# ═══════════════════════════════════════════════════════════════════════════════

def test_render_saves_png_file(tmp_path, sample_env_map):
    """render_multigame_map() 결과를 저장하면 실제 PNG 파일이 생성된다."""
    img = render_multigame_map(sample_env_map)
    out = tmp_path / "render_test.png"
    img.save(out)

    assert out.exists(), "PNG 파일이 생성되지 않았습니다"
    assert out.stat().st_size > 0, "PNG 파일이 비어 있습니다"

    # 재로드해서 올바른 이미지인지 확인
    reloaded = Image.open(out)
    assert reloaded.size == img.size, "저장/재로드 후 크기가 다릅니다"


def test_render_saves_all_tiles_map(tmp_path, all_tiles_map):
    """모든 타일이 포함된 맵도 정상적으로 저장된다."""
    img = render_multigame_map(all_tiles_map)
    out = tmp_path / "all_tiles.png"
    img.save(out)
    assert out.exists() and out.stat().st_size > 0


# ═══════════════════════════════════════════════════════════════════════════════
# 4. 렌더링 이미지 크기 검증
# ═══════════════════════════════════════════════════════════════════════════════

def test_render_image_size(sample_env_map):
    """렌더링 이미지 크기가 map_shape × tile_size 와 일치해야 한다."""
    H, W = sample_env_map.shape
    img = render_multigame_map(sample_env_map)
    assert img.size == (W * _TILE_SIZE, H * _TILE_SIZE), (
        f"이미지 크기 {img.size} != 기대값 ({W*_TILE_SIZE}, {H*_TILE_SIZE})"
    )


@pytest.mark.parametrize("tile_size", [8, 16, 32])
def test_render_image_size_custom_tile(sample_env_map, tile_size):
    """tile_size 파라미터가 실제 이미지 크기에 반영되어야 한다."""
    H, W = sample_env_map.shape
    img = render_multigame_map(sample_env_map, tile_size=tile_size)
    assert img.size == (W * tile_size, H * tile_size), (
        f"tile_size={tile_size} 일 때 크기 불일치: {img.size}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 5. 모든 카테고리 타일 그래픽 누락 없음 검증
# ═══════════════════════════════════════════════════════════════════════════════

def test_all_category_tiles_present(env_with_graphics):
    """BORDER 포함 모든 MultigameTiles 인덱스에 그래픽이 있어야 한다."""
    env, _ = env_with_graphics
    g = np.array(env.prob.graphics)
    for tile in MultigameTiles:
        idx = int(tile)
        assert idx < g.shape[0], f"{tile.name}(idx={idx}) 가 graphics 범위 밖"
        tile_img = g[idx]
        assert tile_img.shape == (_TILE_SIZE, _TILE_SIZE, 4), (
            f"{tile.name} 그래픽 shape 불일치: {tile_img.shape}"
        )


def test_different_tiles_look_different(env_with_graphics):
    """서로 다른 카테고리 타일은 완전히 동일한 이미지이면 안 된다 (최소 2개 이상 달라야)."""
    env, _ = env_with_graphics
    g = np.array(env.prob.graphics)

    seen: list[np.ndarray] = []
    duplicate_count = 0
    for tile in MultigameTiles:
        img = g[int(tile)]
        for prev in seen:
            if np.array_equal(img, prev):
                duplicate_count += 1
                break
        seen.append(img)

    total = len(MultigameTiles)
    unique = total - duplicate_count
    assert unique >= total // 2, (
        f"서로 다른 타일이 너무 적습니다: unique={unique}/{total}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 6. JSON _category_tile_images 매핑 검증
# ═══════════════════════════════════════════════════════════════════════════════

from envs.probs.multigame import (
    _BORDER_IMAGE,
    _CATEGORY_IMAGE_FILES,
    _TILE_IMS_DIR,
)


def test_json_tile_images_section_exists():
    """tile_mapping.json 에 _category_tile_images 섹션이 존재해야 한다."""
    import json
    from pathlib import Path
    mapping = json.loads(
        (Path(__file__).parent.parent / "dataset" / "multigame" / "tile_mapping.json")
        .read_text(encoding="utf-8")
    )
    assert "_category_tile_images" in mapping, (
        "tile_mapping.json 에 _category_tile_images 섹션이 없습니다"
    )


def test_json_tile_images_has_border():
    """_category_tile_images 에 'border' 키가 있어야 한다."""
    assert _BORDER_IMAGE, "'border' 키가 비어 있습니다"


def test_json_tile_images_covers_all_categories():
    """_category_tile_images 에 모든 category 인덱스가 정의되어 있어야 한다."""
    for cat_idx in _CATEGORIES:
        assert cat_idx in _CATEGORY_IMAGE_FILES, (
            f"category {cat_idx}({_CATEGORIES[cat_idx]}) 에 대한 이미지 매핑이 없습니다"
        )


def test_json_tile_images_files_exist():
    """_category_tile_images 에 지정된 모든 파일이 실제로 존재해야 한다."""
    missing = []
    # border
    if not (_TILE_IMS_DIR / _BORDER_IMAGE).exists():
        missing.append(f"border → {_BORDER_IMAGE}")
    # categories
    for cat_idx, fname in _CATEGORY_IMAGE_FILES.items():
        if not (_TILE_IMS_DIR / fname).exists():
            missing.append(f"category {cat_idx}({_CATEGORIES.get(cat_idx)}) → {fname}")
    assert not missing, (
        f"tile_ims 에 존재하지 않는 파일:\n" + "\n".join(missing)
    )




# ═══════════════════════════════════════════════════════════════════════════════
# 단독 실행 — tests/render_samples/ 에 샘플 이미지 저장
# ═══════════════════════════════════════════════════════════════════════════════

def _save_samples():
    import jax
    from envs.pcgrl_env import gen_dummy_queued_state

    _SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    env, env_params = make_multigame_env()
    env.init_graphics()
    dummy_qs = gen_dummy_queued_state(env)

    # ── ① 랜덤 맵 5장 ────────────────────────────────────────────────────────
    print("\n[1] 랜덤 맵 렌더링...")
    for i in range(5):
        rng = jax.random.PRNGKey(i)
        _, state = env.reset(rng, env_params, dummy_qs)
        env_map = np.array(state.env_map)
        img = render_multigame_map(env_map, tile_size=32)
        out = _SAMPLES_DIR / f"random_map_{i:02d}.png"
        img.save(out)
        print(f"  → {out}  (map shape={env_map.shape}, "
              f"unique tiles={np.unique(env_map).tolist()})")

    # ── ② 타일 범례 이미지 ────────────────────────────────────────────────────
    print("\n[2] 타일 범례 렌더링...")
    _save_tile_legend()

    # ── ③ 카테고리별 solid 맵 (각 카테고리로 가득 찬 16×16) ─────────────────
    print("\n[3] 카테고리별 solid 맵...")
    for cat_idx, cat_name in _CATEGORIES.items():
        tile_val = cat_idx + 1   # BORDER shift
        solid_map = np.full((16, 16), tile_val, dtype=np.int32)
        img = render_multigame_map(solid_map, tile_size=16)
        out = _SAMPLES_DIR / f"solid_{cat_idx:02d}_{cat_name}.png"
        img.save(out)
        print(f"  → {out}")

    print(f"\n✅ 샘플 이미지 저장 완료: {_SAMPLES_DIR}")

    # ── ④ overview: 범례 + 랜덤 맵 5장 합성 ──────────────────────────────────
    print("\n[4] overview 합성...")
    maps = [Image.open(_SAMPLES_DIR / f"random_map_{i:02d}.png") for i in range(5)]
    legend = Image.open(_SAMPLES_DIR / "tile_legend.png")
    MW, MH = maps[0].size
    LW, LH = legend.size
    P = 10
    sw = 5 * MW + 6 * P
    sh = P + LH + P + MH + 22 + P
    sheet = Image.new("RGB", (sw, sh), (20, 20, 20))
    sheet.paste(legend, ((sw - LW) // 2, P))
    from PIL import ImageDraw
    d = ImageDraw.Draw(sheet)
    for i, m in enumerate(maps):
        x = P + i * (MW + P)
        y = P + LH + P
        sheet.paste(m, (x, y))
        d.text((x + 4, y + MH + 2), f"map {i}", fill=(180, 180, 180))
    overview_path = _SAMPLES_DIR / "overview.png"
    sheet.save(str(overview_path))
    print(f"  → {overview_path}  size={sheet.size}")


def _save_tile_legend():
    """각 타일을 한 줄로 배치한 범례 이미지 생성."""
    TILE_SIZE = 32
    PAD = 4
    LABEL_H = 20
    n_tiles = len(MultigameTiles)

    total_w = n_tiles * (TILE_SIZE + PAD) + PAD
    total_h = TILE_SIZE + LABEL_H + PAD * 2

    legend = Image.new("RGB", (total_w, total_h), (30, 30, 30))

    # 타일 이미지 로드
    tile_imgs: dict[int, Image.Image] = {}
    border_path = Path(__file__).parent.parent / "envs" / "probs" / "tile_ims" / "solid.png"
    tile_imgs[0] = (
        Image.open(border_path).convert("RGBA").resize((TILE_SIZE, TILE_SIZE))
        if border_path.exists()
        else _make_color_tile((40, 40, 40), TILE_SIZE)
    )
    for cat_idx in _CATEGORIES:
        tile_imgs[cat_idx + 1] = _load_or_color_tile(cat_idx, TILE_SIZE)

    try:
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(legend)
        font = ImageFont.load_default()
    except Exception:
        draw = None

    for tile in MultigameTiles:
        idx = int(tile)
        x = PAD + idx * (TILE_SIZE + PAD)
        y = PAD

        img = tile_imgs.get(idx)
        if img:
            legend.paste(img.convert("RGB"), (x, y))

        if draw:
            label = tile.name[:6]
            draw.text((x, y + TILE_SIZE + 2), label, fill=(200, 200, 200), font=font)

    out = _SAMPLES_DIR / "tile_legend.png"
    legend.save(out)
    print(f"  → {out}")


# 단독 실행용 import
from envs.probs.multigame import _load_or_color_tile, _make_color_tile

if __name__ == "__main__":
    _save_samples()


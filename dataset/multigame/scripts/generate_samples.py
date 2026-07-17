"""
dataset/multigame/scripts/generate_samples.py
=============================================
dataset class in  sample  text and  samples/ folder in  rendering image  savetext.

create structure:
  samples/
    {game}/
      {game}_raw_01.png      ← game text tile_ims text text (maptext  before )
      {game}_unified_01.png  ← unified 7-category tile_ims text text (maptext  after )
      ...
    overview_raw.png
    overview_unified.png
    compare_raw_vs_unified.png

Usage:
  cd /path/to/multigame-pcgrl
  python -m dataset.multigame.scripts.generate_samples
"""
from __future__ import annotations

import sys
from pathlib import Path

_SCRIPT_DIR   = Path(__file__).resolve().parent
_MULTIGAME    = _SCRIPT_DIR.parent
_DATASET_ROOT = _MULTIGAME.parent
_REPO_ROOT    = _DATASET_ROOT.parent

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from dataset.multigame.base import GameTag
from dataset.multigame.handlers.vglc_handler import VGLCGameHandler
from dataset.multigame.handlers.dungeon_handler import DungeonHandler
from dataset.multigame.tile_utils import (
    CATEGORY_COLORS,
    UNIFIED_CATEGORIES,
    NUM_CATEGORIES,
    to_unified,
)

# ── path text ─────────────────────────────────────────────────────────────────
TILE_IMS_DIR = _REPO_ROOT / "envs" / "probs" / "tile_ims"
SAMPLES_DIR  = _MULTIGAME / "samples"

# ── config ─────────────────────────────────────────────────────────────────────
TILE_SIZE = 16    # tile image   text 16×16 text to  upscale text
N_SAMPLES = 5
LABEL_H   = 22
GAP       = 6
BG        = (30, 30, 30)

# ── text ─────────────────────────────────────────────────────────────────────
def _font(size: int = 11):
    for path in [
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()

FONT    = _font(11)
FONT_LG = _font(12)

# ──────────────────────────────────────────────────────────────────────────────
# 1.  tile_ims text text  to text
# ──────────────────────────────────────────────────────────────────────────────
_SPRITE_CACHE: dict[str, Image.Image] = {}

def _load_sprite(name: str) -> Image.Image:
    """tile_ims/{name}.png   loadtext RGBA to  return. if missing None."""
    if name in _SPRITE_CACHE:
        return _SPRITE_CACHE[name]
    path = TILE_IMS_DIR / f"{name}.png"
    if not path.exists():
        return None
    img = Image.open(path).convert("RGBA")
    _SPRITE_CACHE[name] = img
    return img


# ──────────────────────────────────────────────────────────────────────────────
# 2.  gametext tile_id → sprite name text
#     (envs/probs/dungeon.py DungeonTiles basis)
# ──────────────────────────────────────────────────────────────────────────────
# dungeon: tile_id(int) → sprite filename (expandtext text)
_DUNGEON_SPRITE: dict[int, str] = {
    0: "solid",    # BORDER
    1: "empty",    # EMPTY
    2: "solid",    # WALL
    3: "player",   # PLAYER
    4: "bat",      # BAT
    5: "scorpion", # SCORPION
    6: "spider",   # SPIDER
    7: "key",      # KEY
    8: "door",     # DOOR
}

# unified category index → texttable sprite name
#   dungeon tile_ims  during   of text  text text text  text for
#   hazard(6)  tile_ims in  text to  color fallback
_UNIFIED_SPRITE: dict[int, str | None] = {
    0: "empty",    # empty
    1: "solid",    # wall
    2: "empty",    # floor  (dungeon of  text text)
    3: "bat",      # enemy  (bat  texttable text as )
    4: "key",      # object (key  texttable text text as )
    5: "door",     # spawn  (door  texttable text as )
    6: "lava",     # hazard
}


# ──────────────────────────────────────────────────────────────────────────────
# 3.  text text based level rendering
# ──────────────────────────────────────────────────────────────────────────────
def _color_tile(color_rgb: tuple[int, int, int], size: int = TILE_SIZE) -> Image.Image:
    """text RGBA tile create (sprite text  text fallback)."""
    tile = Image.new("RGBA", (size, size), (*color_rgb, 255))
    return tile


def _composite_sprite(
    sprite: Image.Image | None,
    fallback_color: tuple[int, int, int],
    size: int = TILE_SIZE,
) -> Image.Image:
    """sprite(RGBA)  fallback_color text above in  text RGB to  return."""
    bg = Image.new("RGB", (size, size), fallback_color)
    if sprite is not None:
        sp = sprite.resize((size, size), Image.NEAREST)
        bg.paste(sp, (0, 0), sp)  # RGBA alpha text for
    return bg


def render_raw_sprites(array: np.ndarray, game: str) -> np.ndarray:
    """
    game text tile_id → tile_ims text text to  rendering (maptext  before ).
    dungeontext tile_ims  text to  remaining game  text palette color text for .
    """
    from dataset.multigame.render import array_to_rgb, get_palette

    h, w = array.shape
    canvas = Image.new("RGB", (w * TILE_SIZE, h * TILE_SIZE), BG)

    if game == GameTag.DUNGEON:
        for r in range(h):
            for c in range(w):
                tid = int(array[r, c])
                sname = _DUNGEON_SPRITE.get(tid, "empty")
                sprite = _load_sprite(sname)
                # fallback color: solid→gray, others→category color
                fallback = (80, 80, 80) if "solid" in sname else (20, 20, 20)
                tile_img = _composite_sprite(sprite, fallback)
                canvas.paste(tile_img, (c * TILE_SIZE, r * TILE_SIZE))
    else:
        # VGLC game: text palette color
        palette = get_palette(game)
        small   = array_to_rgb(array, palette)
        big     = np.repeat(np.repeat(small, TILE_SIZE, axis=0), TILE_SIZE, axis=1)
        canvas  = Image.fromarray(big, "RGB")

    return np.array(canvas)


def render_unified_sprites(array: np.ndarray, game: str) -> np.ndarray:
    """
    tile_mapping.json basis unified category to  maptext  after
    unified categorytext texttable tile_ims text text to  rendering (maptext  after ).
    """
    unified = to_unified(array, game, warn_unmapped=False)
    h, w = unified.shape
    canvas = Image.new("RGB", (w * TILE_SIZE, h * TILE_SIZE), BG)

    for r in range(h):
        for c in range(w):
            cat = int(unified[r, c])
            sname    = _UNIFIED_SPRITE.get(cat)
            fallback = CATEGORY_COLORS.get(cat, (128, 0, 128))
            sprite   = _load_sprite(sname) if sname else None
            tile_img = _composite_sprite(sprite, fallback)
            canvas.paste(tile_img, (c * TILE_SIZE, r * TILE_SIZE))

    return np.array(canvas)


# ──────────────────────────────────────────────────────────────────────────────
# 4.  legend image
# ──────────────────────────────────────────────────────────────────────────────
def make_unified_legend() -> Image.Image:
    """unified 7-category legend: sprite + color + name."""
    cell_h = TILE_SIZE + 6
    padding = 6
    label_w = 90
    w = TILE_SIZE + padding * 2 + label_w
    h = cell_h * NUM_CATEGORIES + padding * 2

    canvas = Image.new("RGB", (w, h), (40, 40, 40))
    draw   = ImageDraw.Draw(canvas)

    for idx in range(NUM_CATEGORIES):
        name     = UNIFIED_CATEGORIES[idx]
        sname    = _UNIFIED_SPRITE.get(idx)
        color    = CATEGORY_COLORS[idx]
        sprite   = _load_sprite(sname) if sname else None
        tile_img = _composite_sprite(sprite, color, size=TILE_SIZE)

        y = padding + idx * cell_h
        canvas.paste(tile_img, (padding, y))
        draw.text(
            (padding + TILE_SIZE + 6, y + 2),
            f"{idx}: {name}",
            fill=(220, 220, 220),
            font=FONT,
        )

    return canvas


# ──────────────────────────────────────────────────────────────────────────────
# 5.  common utility
# ──────────────────────────────────────────────────────────────────────────────
def _add_label(img_arr: np.ndarray, text: str) -> np.ndarray:
    h, w = img_arr.shape[:2]
    banner = np.full((LABEL_H, w, 3), (50, 50, 50), dtype=np.uint8)
    pil  = Image.fromarray(banner)
    draw = ImageDraw.Draw(pil)
    draw.text((4, 4), text, fill=(220, 220, 220), font=FONT)
    return np.concatenate([np.array(pil), img_arr], axis=0)


def _save(arr: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, mode="RGB").save(str(path))
    print(f"  saved → {path.relative_to(_REPO_ROOT)}")


def _pad_width(arr: np.ndarray, target_w: int) -> np.ndarray:
    if arr.shape[1] < target_w:
        pad = np.full((arr.shape[0], target_w - arr.shape[1], 3), BG, dtype=np.uint8)
        return np.concatenate([arr, pad], axis=1)
    return arr


def stack_vertical(imgs: list[np.ndarray]) -> np.ndarray:
    if not imgs:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    max_w = max(im.shape[1] for im in imgs)
    rows  = []
    for im in imgs:
        rows.append(_pad_width(im, max_w))
        rows.append(np.full((GAP, max_w, 3), BG, dtype=np.uint8))
    return np.concatenate(rows[:-1], axis=0)


def stack_horizontal(cols: list[np.ndarray]) -> np.ndarray:
    if not cols:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    max_h = max(c.shape[0] for c in cols)
    parts = []
    for c in cols:
        if c.shape[0] < max_h:
            pad = np.full((max_h - c.shape[0], c.shape[1], 3), BG, dtype=np.uint8)
            c = np.concatenate([c, pad], axis=0)
        parts.append(c)
        parts.append(np.full((max_h, GAP, 3), BG, dtype=np.uint8))
    return np.concatenate(parts[:-1], axis=1)


def add_game_header(col: np.ndarray, name: str) -> np.ndarray:
    banner = np.full((28, col.shape[1], 3), (20, 20, 60), dtype=np.uint8)
    pil  = Image.fromarray(banner)
    draw = ImageDraw.Draw(pil)
    draw.text((6, 6), name.upper(), fill=(255, 220, 80), font=FONT_LG)
    return np.concatenate([np.array(pil), col], axis=0)


# ──────────────────────────────────────────────────────────────────────────────
# 6.  text
# ──────────────────────────────────────────────────────────────────────────────
def process_game(game: str, samples: list):
    raw_imgs, unified_imgs = [], []
    for i, sample in enumerate(samples[:N_SAMPLES], 1):
        raw = render_raw_sprites(sample.array, game)
        uni = render_unified_sprites(sample.array, game)
        raw_imgs.append(_add_label(raw, f"{game} #{i}  [original]"))
        unified_imgs.append(_add_label(uni, f"{game} #{i}  [unified]"))
    return raw_imgs, unified_imgs


def main() -> None:
    all_raw_cols:     list[np.ndarray] = []
    all_unified_cols: list[np.ndarray] = []
    all_game_names:   list[str]        = []

    # VGLC game
    for game in [GameTag.ZELDA, GameTag.MARIO, GameTag.LODE_RUNNER, GameTag.DOOM]:
        print(f"\n[{game}] loading...")
        try:
            handler = VGLCGameHandler(game_tag=game)
            entries = handler.list_entries()[:N_SAMPLES]
            samples = [handler.load_sample(e, order=i) for i, e in enumerate(entries)]
        except Exception as e:
            print(f"  ⚠ skip: {e}")
            continue

        raw_imgs, unified_imgs = process_game(game, samples)
        for i, (r, u) in enumerate(zip(raw_imgs, unified_imgs), 1):
            _save(r, SAMPLES_DIR / game / f"{game}_raw_{i:02d}.png")
            _save(u, SAMPLES_DIR / game / f"{game}_unified_{i:02d}.png")

        all_raw_cols.append(stack_vertical(raw_imgs))
        all_unified_cols.append(stack_vertical(unified_imgs))
        all_game_names.append(game)

    # Dungeon (tile_ims text text text for )
    print(f"\n[dungeon] loading...")
    try:
        handler = DungeonHandler()
        entries = handler.list_entries()[:N_SAMPLES]
        samples = [handler.load_sample(e, order=i) for i, e in enumerate(entries)]

        raw_imgs, unified_imgs = process_game(GameTag.DUNGEON, samples)
        for i, (r, u) in enumerate(zip(raw_imgs, unified_imgs), 1):
            _save(r, SAMPLES_DIR / "dungeon" / f"dungeon_raw_{i:02d}.png")
            _save(u, SAMPLES_DIR / "dungeon" / f"dungeon_unified_{i:02d}.png")

        all_raw_cols.append(stack_vertical(raw_imgs))
        all_unified_cols.append(stack_vertical(unified_imgs))
        all_game_names.append(GameTag.DUNGEON)
    except Exception as e:
        print(f"  ⚠ skip dungeon: {e}")

    if not all_raw_cols:
        print("No samples generated.")
        return

    # overview text
    print("\n[overview] generating grids...")
    raw_cols_h     = [add_game_header(c, g) for c, g in zip(all_raw_cols,     all_game_names)]
    unified_cols_h = [add_game_header(c, g) for c, g in zip(all_unified_cols, all_game_names)]
    _save(stack_horizontal(raw_cols_h),     SAMPLES_DIR / "overview_raw.png")
    _save(stack_horizontal(unified_cols_h), SAMPLES_DIR / "overview_unified.png")

    # compare (raw | unified text)
    print("\n[compare] generating side-by-side...")
    sbs_rows = []
    for raw_col, uni_col in zip(all_raw_cols, all_unified_cols):
        divider = np.full((max(raw_col.shape[0], uni_col.shape[0]), 3, 3),
                          (80, 80, 80), dtype=np.uint8)
        # text  text
        max_h = max(raw_col.shape[0], uni_col.shape[0])
        def _pad_h(a):
            if a.shape[0] < max_h:
                p = np.full((max_h - a.shape[0], a.shape[1], 3), BG, dtype=np.uint8)
                return np.concatenate([a, p], axis=0)
            return a
        row = np.concatenate([_pad_h(raw_col), divider, _pad_h(uni_col)], axis=1)
        sbs_rows.append(row)
        sbs_rows.append(np.full((GAP, row.shape[1], 3), BG, dtype=np.uint8))
    _save(np.concatenate(sbs_rows[:-1], axis=0), SAMPLES_DIR / "compare_raw_vs_unified.png")

    # unified legend save
    legend_img = make_unified_legend()
    legend_path = SAMPLES_DIR / "legend_unified.png"
    legend_path.parent.mkdir(parents=True, exist_ok=True)
    legend_img.save(str(legend_path))
    print(f"  saved → {legend_path.relative_to(_REPO_ROOT)}")

    print(f"\n✓ done. images saved under {SAMPLES_DIR.relative_to(_REPO_ROOT)}/")


if __name__ == "__main__":
    main()


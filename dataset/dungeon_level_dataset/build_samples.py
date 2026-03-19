"""
Build sample previews: 10 random levels saved as
  samples/numpy/<key>_<instruction_slug>.npy
  samples/rendered/<key>_<instruction_slug>.png
"""

import csv
import random
import cv2
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).parent
TILE_IMS_DIR = BASE_DIR / "tile_ims"
RANDOM_SEED = 42
N_SAMPLES = 10

# tile value → image filename mapping
TILE_FILES = {
    1: "empty.png",   # floor
    2: "solid.png",   # wall
    3: "bat.png",     # enemy
}

TILE_SIZE = 16  # each tile image is 16×16


def load_tile_images() -> dict:
    """Load tile images as BGR (alpha composited on white background)."""
    tiles = {}
    for val, fname in TILE_FILES.items():
        img = cv2.imread(str(TILE_IMS_DIR / fname), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Tile image not found: {TILE_IMS_DIR / fname}")
        if img.shape[2] == 4:
            # Alpha composite onto white background
            bgr = img[:, :, :3].astype(np.float32)
            alpha = img[:, :, 3:4].astype(np.float32) / 255.0
            white = np.ones_like(bgr) * 255.0
            bgr = (bgr * alpha + white * (1 - alpha)).astype(np.uint8)
            img = bgr
        tiles[val] = img
    return tiles


def render_level(arr: np.ndarray, out_path: Path,
                 tile_imgs: dict, scale: int = 3) -> None:
    h, w = arr.shape
    ts = TILE_SIZE * scale
    canvas = np.ones((h * ts, w * ts, 3), dtype=np.uint8) * 255

    for r in range(h):
        for c in range(w):
            val = int(arr[r, c])
            tile = tile_imgs.get(val, tile_imgs[1])
            tile_resized = cv2.resize(tile, (ts, ts), interpolation=cv2.INTER_NEAREST)
            canvas[r * ts:(r + 1) * ts, c * ts:(c + 1) * ts] = tile_resized

    # Draw grid lines
    grid_color = (180, 180, 180)
    for i in range(h + 1):
        cv2.line(canvas, (0, i * ts), (w * ts, i * ts), grid_color, 1)
    for j in range(w + 1):
        cv2.line(canvas, (j * ts, 0), (j * ts, h * ts), grid_color, 1)

    cv2.imwrite(str(out_path), canvas)


def main():
    archive = np.load(BASE_DIR / "dungeon_levels.npz")

    # CSV 읽기 (pandas 없이)
    with open(BASE_DIR / "dungeon_levels_metadata.csv", newline="", encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))

    # instruction_slug 별로 그룹화
    slug_to_rows: dict = {}
    for row in all_rows:
        slug_to_rows.setdefault(row["instruction_slug"], []).append(row)

    # 슬러그 중에서 N_SAMPLES개 랜덤 선택 후, 각 슬러그에서 1개씩 뽑기
    all_slugs = list(slug_to_rows.keys())
    random.seed(RANDOM_SEED)
    chosen_slugs = random.sample(all_slugs, min(N_SAMPLES, len(all_slugs)))

    rng = random.Random(RANDOM_SEED)
    samples = [rng.choice(slug_to_rows[slug]) for slug in chosen_slugs]

    tile_imgs = load_tile_images()

    # Create output dirs
    npy_dir = BASE_DIR / "samples" / "numpy"
    png_dir = BASE_DIR / "samples" / "rendered"
    npy_dir.mkdir(parents=True, exist_ok=True)
    png_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving {len(samples)} samples …")
    for row in samples:
        key   = row["key"]
        slug  = row["instruction_slug"]
        instr = row["instruction"]
        arr   = archive[key]
        stem  = f"{key}_{slug}"

        np.save(npy_dir / f"{stem}.npy", arr)
        render_level(arr, png_dir / f"{stem}.png", tile_imgs)
        print(f"  [{key}] {instr}")

    print(f"\nDone!")
    print(f"  samples/numpy/    → {len(list(npy_dir.glob('*.npy')))} files")
    print(f"  samples/rendered/ → {len(list(png_dir.glob('*.png')))} files")


if __name__ == "__main__":
    main()


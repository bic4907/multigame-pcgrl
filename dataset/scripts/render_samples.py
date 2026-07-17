"""
dataset/scripts/render_samples.py
---------------------------------
text script: text to text  inside  `dataset` folder in  with textgame dataset in
sample  renderingtext PNG image to  savetext.

Usage (text to text text in ):
    python dataset/scripts/render_samples.py

createwater:
    dataset/samples/zelda_sample.png
    dataset/samples/dungeon_sample.png
    dataset/samples/grid_samples.png

text:   script   to text `dataset` text in  with `multigame` text  direct text.
"""
from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent.parent

from dataset.multigame.dataset import MultiGameDataset
from dataset.multigame.render import save_rendered, save_grid

OUT_DIR = HERE / "samples"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("Loading datasets (this may take a moment)...")
    ds = MultiGameDataset(vglc_games=["zelda", "mario", "lode_runner"], include_dungeon=True)
    print(f"Loaded MultiGameDataset with total samples: {len(ds)}")

    # Zelda sample (text text)
    zelda_pool = ds.by_game("zelda")
    if zelda_pool:
        z = zelda_pool[0]
        out = OUT_DIR / "zelda_sample.png"
        save_rendered(z, out, tile_size=8)
        print("Saved:", out)
    else:
        print("No Zelda samples found")

    # Dungeon sample (text text)
    dungeon_pool = ds.by_game("dungeon")
    if dungeon_pool:
        d = dungeon_pool[0]
        out = OUT_DIR / "dungeon_sample.png"
        save_rendered(d, out, tile_size=8)
        print("Saved:", out)
    else:
        print("No Dungeon samples found")

    # text: Zelda 3text + Dungeon 3text (availabletext text)
    grid_samples = []
    grid_samples.extend(zelda_pool[:3])
    grid_samples.extend(dungeon_pool[:3])
    if grid_samples:
        out = OUT_DIR / "grid_samples.png"
        save_grid(grid_samples, out, cols=3, tile_size=6)
        print("Saved grid:", out)
    else:
        print("Not enough samples for grid")


if __name__ == "__main__":
    main()


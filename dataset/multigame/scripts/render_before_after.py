"""Render raw vs unified mapped image side-by-side for one sample.

Usage:
    python -m dataset.multigame.scripts.render_before_after --game dungeon --index 0 --out outputs/mapping_compare.png
    python -m dataset.multigame.scripts.render_before_after --game boxoban --index 123 --out outputs/boxoban_compare.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from dataset.multigame.base import GameSample
from dataset.multigame.handlers.boxoban_handler import BoxobanHandler
from dataset.multigame.handlers.dungeon_handler import DungeonHandler
from dataset.multigame.render import render_sample
from dataset.multigame.tile_utils import game_mapping_rows, render_unified_rgb, to_unified


def _load_sample(game: str, index: int) -> GameSample:
    if game == "dungeon":
        handler = DungeonHandler()
    else:
        handler = BoxobanHandler(difficulty="both", split="all", n_sample=None)

    n = len(handler)
    if n <= 0:
        raise ValueError(f"No samples found for game={game!r}")

    idx = max(0, min(index, n - 1))
    sid = handler.list_entries()[idx]
    return handler.load_sample(sid, order=idx)


def _render_before_after(sample: GameSample, tile_size: int) -> Image.Image:
    raw_rgb = render_sample(sample, tile_size=tile_size)
    unified = to_unified(sample.array, sample.game, warn_unmapped=False)
    mapped_rgb = render_unified_rgb(unified, tile_size=tile_size)

    gap = 8
    h = max(raw_rgb.shape[0], mapped_rgb.shape[0])
    w = raw_rgb.shape[1] + gap + mapped_rgb.shape[1]
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[:, :] = (30, 30, 30)
    canvas[:raw_rgb.shape[0], :raw_rgb.shape[1]] = raw_rgb
    x2 = raw_rgb.shape[1] + gap
    canvas[:mapped_rgb.shape[0], x2:x2 + mapped_rgb.shape[1]] = mapped_rgb
    return Image.fromarray(canvas, mode="RGB")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render raw/mapped side-by-side image")
    parser.add_argument("--game", required=True, choices=["dungeon", "boxoban"], help="game tag")
    parser.add_argument("--index", type=int, default=0, help="index inside selected game")
    parser.add_argument("--tile-size", type=int, default=16, help="tile pixel size")
    parser.add_argument("--out", type=str, default="outputs/mapping_compare.png", help="output PNG path")
    args = parser.parse_args()

    sample = _load_sample(args.game, args.index)
    img = _render_before_after(sample, tile_size=args.tile_size)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(out))

    print(f"saved: {out}")
    print(f"game={sample.game} source_id={sample.source_id}")
    print("mapping rows:")
    for row in game_mapping_rows(sample.game):
        print(
            f"  {row['raw_id']:>3} {row['raw_name']:<12} -> "
            f"{row['unified_id']} {row['unified_name']}"
        )


if __name__ == "__main__":
    main()


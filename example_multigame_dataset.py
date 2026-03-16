"""Repository-root example for loading level-text pairs from dungeon dataset only.

Run:
    python example_multigame_dataset.py
    python example_multigame_dataset.py --limit 3
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

from dataset.multigame import MultiGameDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", type=str, default="dungeon", help="game tag filter (default: dungeon)")
    parser.add_argument("--limit", type=int, default=5, help="number of samples to print")
    parser.add_argument("--with-text-only", action="store_true", help="keep only samples with instruction text")
    return parser.parse_args()


def ensure_16x16(level, *, game: str, index: int):
    """Ensure level is 16x16; otherwise slice top-left 16x16 and warn."""
    if level.shape == (16, 16):
        return level
    warnings.warn(
        f"sample[{index}] from game '{game}' has shape {level.shape}; slicing to (16, 16)",
        RuntimeWarning,
        stacklevel=2,
    )
    return level[:16, :16]


def main() -> None:
    args = parse_args()

    # Dungeon-only mode: force-disable VGLC loading.
    ds = MultiGameDataset(
        vglc_games=[],
        vglc_root=Path("__disable_vglc__"),
        include_dungeon=True,
    )
    print("total samples:", len(ds))
    print("available games:", ds.available_games())

    samples = ds.by_game("dungeon")
    print("samples in game 'dungeon':", len(samples))

    if args.game and args.game != "dungeon":
        print(f"warning: only 'dungeon' is loaded, ignoring --game={args.game!r}")

    if args.with_text_only:
        samples = [s for s in samples if s.instruction is not None]
        print("text-only samples:", len(samples))

    print("\nfirst samples as (game, level_shape, text):")
    for i, s in enumerate(samples[: args.limit]):
        level = ensure_16x16(s.array, game=s.game, index=i)
        print((s.game, level.shape, s.instruction))


if __name__ == "__main__":
    main()


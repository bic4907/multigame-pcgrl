#!/usr/bin/env python3
"""
dataset/scripts/generate_placeholder_reward_annotations.py
===========================================================
Generate placeholder reward-annotation CSV files for Sokoban, Zelda, Doom, and Pokemon.

Append _placeholder to filenames to mark dummy data.
  Example: sokoban_reward_annotations_placeholder.csv

reward_enum uses the same 1-5 range as Dungeon:
  1 = region        (connected regions / room count)
  2 = path_length   (longest path length)
  3 = block         (wall / water ratio)
  4 = bat_amount    (enemy / object count)
  5 = bat_direction (enemy direction / positional bias)

Usage:
    python -m dataset.scripts.generate_placeholder_reward_annotations
"""
from __future__ import annotations

import csv
from pathlib import Path

_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "dataset" / "reward_annotations"

# game → list of (reward_enum, feature_name, sub_condition, placeholder_condition_value)
# Use the same reward_enum range, 1-5, as Dungeon
_GAME_DEFS: dict[str, list[tuple]] = {
    "sokoban": [
        (1, "region",        "box",       3.0),   # Box-placement region count
        (2, "path_length",   "",         20.0),   # Minimum solution moves
        (3, "block",         "wall",      0.3),   # wall ratio
        (4, "bat_amount",    "box",       3.0),   # Box count
        (5, "bat_direction", "",          0.5),   # Object positional bias
    ],
    "zelda": [
        (1, "region",        "room",      1.0),   # Connected-room count
        (2, "path_length",   "",         15.0),   # Longest inter-room path
        (3, "block",         "wall",      0.3),   # Wall density
        (4, "bat_amount",    "enemy",     3.0),   # Enemy count
        (5, "bat_direction", "enemy",     0.5),   # Enemy positional bias
    ],
    "doom": [
        (1, "region",        "room",      2.0),   # Room count
        (2, "path_length",   "",         30.0),   # Longest traversal path
        (3, "block",         "wall",      0.5),   # wall ratio
        (4, "bat_amount",    "enemy",     5.0),   # Enemy-placement count
        (5, "bat_direction", "enemy",     0.5),   # Enemy directional bias
    ],
    "pokemon": [
        (1, "region",        "",          2.0),   # Connected-region count
        (2, "path_length",   "",         20.0),   # Longest path
        (3, "block",         "wall",      0.4),   # wall ratio
        (4, "bat_amount",    "object",    4.0),   # Object count
        (5, "bat_direction", "",          0.5),   # Object positional bias
    ],
}


def generate_placeholder_csv(game: str, features: list[tuple]) -> Path:
    """
    Create a placeholder CSV for a game.
    Append _placeholder to the filename.
    """
    output_path = _OUTPUT_DIR / f"{game}_reward_annotations_placeholder.csv"

    fieldnames = [
        "game",
        "is_placeholder",   # Dummy-data marker, always "true"
        "reward_enum",
        "feature_name",
        "sub_condition",
        # Use the same condition_1 through condition_5 columns as Dungeon
        "condition_1",      # region
        "condition_2",      # path_length
        "condition_3",      # block
        "condition_4",      # bat_amount
        "condition_5",      # bat_direction
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for reward_enum, feature_name, sub_condition, cond_value in features:
            # Populate only the matching reward_enum column
            conditions = {f"condition_{i}": "" for i in range(1, 6)}
            conditions[f"condition_{reward_enum}"] = cond_value

            writer.writerow({
                "game": game,
                "is_placeholder": "true",
                "reward_enum": reward_enum,
                "feature_name": feature_name,
                "sub_condition": sub_condition,
                **conditions,
            })

    return output_path


def main():
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for game, features in _GAME_DEFS.items():
        output_path = generate_placeholder_csv(game, features)
        print(f"[{game}] → {output_path.name}  "
              f"({len(features)} features, reward_enum 1~5, is_placeholder=true)")

    print("\nDone! Replace *_placeholder.csv files with real per-sample annotations when ready.")


if __name__ == "__main__":
    main()

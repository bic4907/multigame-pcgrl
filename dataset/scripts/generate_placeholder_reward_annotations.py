#!/usr/bin/env python3
"""
dataset/scripts/generate_placeholder_reward_annotations.py
===========================================================
sokoban / zelda / doom / pokemon game in  text placeholder reward annotation CSV  createtext.

filetext in  _placeholder text  text text datatext  text.
  text) sokoban_reward_annotations_placeholder.csv

reward_enum  dungeon and  sametext 1~5 range  text for text:
  1 = region        (text text / text text)
  2 = path_length   (text path text )
  3 = block         (wall / textwater ratio)
  4 = bat_amount    (text / text text)
  5 = bat_direction (text text / abovetext text)

Usage:
    python -m dataset.scripts.generate_placeholder_reward_annotations
"""
from __future__ import annotations

import csv
from pathlib import Path

_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "dataset" / "reward_annotations"

# game → list of (reward_enum, feature_name, sub_condition, placeholder_condition_value)
# reward_enum  dungeon and  sametext 1~5 text for
_GAME_DEFS: dict[str, list[tuple]] = {
    "sokoban": [
        (1, "region",        "box",       3.0),   # text batch text text
        (2, "path_length",   "",         20.0),   # text  minimum move text
        (3, "block",         "wall",      0.3),   # wall ratio
        (4, "bat_amount",    "box",       3.0),   # text count
        (5, "bat_direction", "",          0.5),   # text abovetext text
    ],
    "zelda": [
        (1, "region",        "room",      1.0),   # text text text
        (2, "path_length",   "",         15.0),   # text text text path
        (3, "block",         "wall",      0.3),   # wall text also
        (4, "bat_amount",    "enemy",     3.0),   # text count
        (5, "bat_direction", "enemy",     0.5),   # text abovetext text
    ],
    "doom": [
        (1, "region",        "room",      2.0),   # text text
        (2, "path_length",   "",         30.0),   # text move path
        (3, "block",         "wall",      0.5),   # wall ratio
        (4, "bat_amount",    "enemy",     5.0),   # text batch text
        (5, "bat_direction", "enemy",     0.5),   # text text text
    ],
    "pokemon": [
        (1, "region",        "",          2.0),   # text text text
        (2, "path_length",   "",         20.0),   # text path
        (3, "block",         "wall",      0.4),   # wall ratio
        (4, "bat_amount",    "object",    4.0),   # text text
        (5, "bat_direction", "",          0.5),   # text abovetext text
    ],
}


def generate_placeholder_csv(game: str, features: list[tuple]) -> Path:
    """
    game textabove placeholder CSV  createtext.
    filetext in  _placeholder text  text.
    """
    output_path = _OUTPUT_DIR / f"{game}_reward_annotations_placeholder.csv"

    fieldnames = [
        "game",
        "is_placeholder",   # text data tabletext text (always "true")
        "reward_enum",
        "feature_name",
        "sub_condition",
        # dungeon and  sametext condition_1~5 text text for
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
            # text reward_enum text in text text, remaining text
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


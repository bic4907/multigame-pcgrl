#!/usr/bin/env python3
"""
dataset/scripts/generate_dungeon_reward_annotations.py
======================================================
dungeon_level_dataset metadata of  instruction  scenario_prompt.json basis as
Compute reward annotations (reward_enum, condition values, sub_condition, and actual measures) and
save them to dataset/reward_annotations/dungeon_reward_annotations.csv.

Usage:
    python -m dataset.scripts.generate_dungeon_reward_annotations
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


# ── path config ────────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_INSTRUCT_DIR = _PROJECT_ROOT / "instruct"
_DUNGEON_ROOT = _PROJECT_ROOT / "dataset" / "dungeon_level_dataset"
_OUTPUT_DIR = _PROJECT_ROOT / "dataset" / "reward_annotations"

# reward_enum mapping based on scenario_prompt.json scenario keys
# Condition column indices are also one-based (condition_1 through condition_5)
_FEATURE_TO_ENUM = {
    "region": 1,
    "path_length": 2,
    "block": 3,
    "bat_amount": 4,
    "bat_direction": 5,
}



def _build_instruction_mapping(scenario_path: Path) -> dict:
    """
    Map instructions (text) from scenario_prompt.json to
    Create (feature_name, reward_enum, cond_value, sub_condition) mappings.
    """
    with open(scenario_path, "r", encoding="utf-8") as f:
        sp = json.load(f)

    mapping: dict[str, tuple] = {}

    for feature_name in ["region", "path_length", "block", "bat_amount", "bat_direction"]:
        if feature_name not in sp:
            continue
        enum_val = _FEATURE_TO_ENUM[feature_name]

        for key, entry in sp[feature_name].items():
            value = entry["value"]
            sub_cond = entry.get("sub_condition", "")
            info = (feature_name, enum_val, value, sub_cond)

            # Measured instruction
            mapping[key.lower()] = info
            # similar instructions
            for sim in entry.get("similar", []):
                mapping[sim.lower()] = info

    return mapping


def main():
    scenario_path = _INSTRUCT_DIR / "scenario_prompt.json"
    meta_path = _DUNGEON_ROOT / "dungeon_levels_metadata.csv"

    if not scenario_path.exists():
        print(f"Error: scenario_prompt.json not found at {scenario_path}")
        sys.exit(1)
    if not meta_path.exists():
        print(f"Error: metadata CSV not found at {meta_path}")
        sys.exit(1)

    # ── Build instruction mappings ──────────────────────────────────────────
    instr_mapping = _build_instruction_mapping(scenario_path)
    print(f"[1/3] Built instruction mapping: {len(instr_mapping)} entries")

    # ── metadata load ──────────────────────────────────────────────────────
    metas = []
    with open(meta_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            metas.append(row)
    print(f"[2/3] Loaded metadata: {len(metas)} rows")

    # ── reward annotation create ───────────────────────────────────────────────
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = _OUTPUT_DIR / "dungeon_reward_annotations.csv"

    fieldnames = [
        "key",
        "instruction",
        "level_id",
        "sample_id",
        "reward_enum",
        "feature_name",
        "sub_condition",
        # Five condition columns mapped one-to-one to reward_enum; only the active feature is populated
        "condition_1",  # region
        "condition_2",  # path_length
        "condition_3",  # block (wall)
        "condition_4",  # bat_amount
        "condition_5",  # bat_direction
    ]

    unmapped_count = 0
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, meta in enumerate(metas):
            key = meta["key"]
            instruction = meta["instruction"]
            instr_lower = instruction.lower()

            # Instruction-to-reward mapping
            if instr_lower not in instr_mapping:
                unmapped_count += 1
                continue

            feature_name, reward_enum, cond_value, sub_cond = instr_mapping[instr_lower]

            # Create the condition array with only the active feature populated
            # condition_1 through condition_5 correspond to reward_enum 1 through 5
            conditions = {f"condition_{i}": "" for i in range(1, 6)}
            conditions[f"condition_{reward_enum}"] = cond_value

            row = {
                "key": key,
                "instruction": instruction,
                "level_id": meta["level_id"],
                "sample_id": meta["sample_id"],
                "reward_enum": reward_enum,
                "feature_name": feature_name,
                "sub_condition": sub_cond,
                **conditions,
            }
            writer.writerow(row)

            if (i + 1) % 500 == 0:
                print(f"  ... processed {i + 1}/{len(metas)}")

    print(f"[3/3] Saved reward annotations to {output_path}")
    if unmapped_count > 0:
        print(f"  WARNING: {unmapped_count} rows had unmapped instructions (skipped)")
    print("Done!")


if __name__ == "__main__":
    main()

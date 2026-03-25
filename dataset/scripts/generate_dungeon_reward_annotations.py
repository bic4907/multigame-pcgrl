#!/usr/bin/env python3
"""
dataset/scripts/generate_dungeon_reward_annotations.py
======================================================
dungeon_level_dataset 메타데이터의 instruction을 scenario_prompt.json 기준으로
reward annotation(reward_enum, condition values, sub_condition, 실제 measure)을 계산하여
dataset/reward_annotations/dungeon_reward_annotations.csv 로 저장한다.

Usage:
    python -m dataset.scripts.generate_dungeon_reward_annotations
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


# ── 경로 설정 ────────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_INSTRUCT_DIR = _PROJECT_ROOT / "instruct"
_DUNGEON_ROOT = _PROJECT_ROOT / "dataset" / "dungeon_level_dataset"
_OUTPUT_DIR = _PROJECT_ROOT / "dataset" / "reward_annotations"

# reward_enum 매핑 (scenario_prompt.json의 scenarios 키 기준)
# condition 컬럼 인덱스도 동일하게 1-based (condition_1 ~ condition_5)
_FEATURE_TO_ENUM = {
    "region": 1,
    "path_length": 2,
    "block": 3,
    "bat_amount": 4,
    "bat_direction": 5,
}



def _build_instruction_mapping(scenario_path: Path) -> dict:
    """
    scenario_prompt.json에서 instruction(소문자) →
    (feature_name, reward_enum, cond_value, sub_condition) 매핑을 생성한다.
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

            # 대표 instruction
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

    # ── instruction 매핑 빌드 ────────────────────────────────────────────────
    instr_mapping = _build_instruction_mapping(scenario_path)
    print(f"[1/3] Built instruction mapping: {len(instr_mapping)} entries")

    # ── 메타데이터 로드 ──────────────────────────────────────────────────────
    metas = []
    with open(meta_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            metas.append(row)
    print(f"[2/3] Loaded metadata: {len(metas)} rows")

    # ── reward annotation 생성 ───────────────────────────────────────────────
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
        # condition 배열 (5개 컬럼, reward_enum과 1:1 매칭, 해당 feature만 값 설정, 나머지 빈값)
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

            # instruction → reward 매핑
            if instr_lower not in instr_mapping:
                unmapped_count += 1
                continue

            feature_name, reward_enum, cond_value, sub_cond = instr_mapping[instr_lower]

            # condition 배열 생성 (해당 feature만 값 설정, 나머지 빈값)
            # condition_1~5는 reward_enum 1~5에 대응
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


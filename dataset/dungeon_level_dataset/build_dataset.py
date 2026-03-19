"""
Dungeon Level Dataset Builder
- Packs all .npy files into a single compressed .npz archive
- Creates a metadata CSV with (index, key, instruction, instruction_slug, level_id, sample_id)
"""

import os
import re
import csv
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).parent
NUMPY_DIR = BASE_DIR / "numpy"
OUTPUT_NPZ = BASE_DIR / "dungeon_levels.npz"
OUTPUT_CSV = BASE_DIR / "dungeon_levels_metadata.csv"


def main():
    instructions = sorted([
        d for d in os.listdir(NUMPY_DIR)
        if (NUMPY_DIR / d).is_dir()
    ])

    print(f"Found {len(instructions)} instruction categories")

    data_dict = {}
    csv_rows = []
    global_idx = 0

    for instruction in instructions:
        instr_dir = NUMPY_DIR / instruction
        npy_files = sorted([f for f in os.listdir(instr_dir) if f.endswith(".npy")])

        for filename in npy_files:
            filepath = instr_dir / filename
            stem = Path(filename).stem
            match = re.match(r"^(.+)_level_(\d+)_s(\d+)$", stem)
            if match:
                level_id  = int(match.group(2))
                sample_id = int(match.group(3))
            else:
                level_id  = -1
                sample_id = -1

            arr = np.load(filepath)
            key = f"{global_idx:06d}"
            data_dict[key] = arr

            csv_rows.append({
                "index":            global_idx,
                "key":              key,
                "instruction":      instruction.replace("_", " "),
                "instruction_slug": instruction,
                "level_id":         level_id,
                "sample_id":        sample_id,
            })
            global_idx += 1

    print(f"Total samples: {global_idx}")

    # ── Save .npz ──────────────────────────────────────────────────────────────
    print(f"Saving compressed archive → {OUTPUT_NPZ} ...")
    np.savez_compressed(OUTPUT_NPZ, **data_dict)
    size_mb = OUTPUT_NPZ.stat().st_size / (1024 ** 2)
    print(f"Saved! File size: {size_mb:.2f} MB")

    # ── Save CSV ───────────────────────────────────────────────────────────────
    print(f"Saving metadata CSV → {OUTPUT_CSV} ...")
    fieldnames = ["index", "key", "instruction", "instruction_slug", "level_id", "sample_id"]
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"Saved! Rows: {len(csv_rows)}")

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n=== Dataset Summary ===")
    print(f"  Instruction categories : {len(instructions)}")
    print(f"  Total samples          : {global_idx}")
    print(f"  Array shape (per item) : {arr.shape}  dtype={arr.dtype}")
    print(f"  NPZ archive            : {OUTPUT_NPZ.name}  ({size_mb:.2f} MB)")
    print(f"  Metadata CSV           : {OUTPUT_CSV.name}  ({len(csv_rows)} rows)")
    print("=======================")


if __name__ == "__main__":
    main()

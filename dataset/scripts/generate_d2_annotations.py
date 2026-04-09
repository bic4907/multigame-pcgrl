"""
d2_reward_annotations.csv 생성 스크립트.

dungeon_reward_annotations.csv에서:
  - key 접두사 dg → d2
  - instruction_uni를 instruction_raw로 대체
  - reward_enum == 5 제외
"""
import csv
from pathlib import Path

_HERE = Path(__file__).parent.parent
ANNOT_DIR = _HERE / "reward_annotations"
SRC = ANNOT_DIR / "dungeon_reward_annotations.csv"
DST = ANNOT_DIR / "d2_reward_annotations.csv"


def main():
    rows_out = []
    with open(SRC, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            # reward_enum 5 제외
            if row["reward_enum"] == "5":
                continue
            # key 접두사 변경: dg → d2
            row["key"] = "d2" + row["key"][2:]
            # instruction_raw를 instruction_uni로 설정
            row["instruction_uni"] = row["instruction_raw"]
            rows_out.append(row)

    with open(DST, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows_out:
            writer.writerow(row)

    print(f"Written {len(rows_out)} rows to {DST}")

    enum_counts = {}
    for row in rows_out:
        e = row["reward_enum"]
        enum_counts[e] = enum_counts.get(e, 0) + 1
    for e in sorted(enum_counts):
        print(f"  reward_enum={e}: {enum_counts[e]} rows")


if __name__ == "__main__":
    main()


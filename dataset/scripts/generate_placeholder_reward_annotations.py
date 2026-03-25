#!/usr/bin/env python3
"""
dataset/scripts/generate_placeholder_reward_annotations.py
===========================================================
sokoban / zelda / doom / pokemon 게임에 대한 placeholder reward annotation CSV를 생성한다.

파일명에 _placeholder 접미사를 붙여 더미 데이터임을 명시한다.
  예) sokoban_reward_annotations_placeholder.csv

reward_enum은 dungeon과 동일한 1~5 범위를 사용한다:
  1 = region        (연결 영역 / 방 수)
  2 = path_length   (최장 경로 길이)
  3 = block         (벽 / 장애물 비율)
  4 = bat_amount    (적 / 오브젝트 수)
  5 = bat_direction (적 방향성 / 위치 편향)

Usage:
    python -m dataset.scripts.generate_placeholder_reward_annotations
"""
from __future__ import annotations

import csv
from pathlib import Path

_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "dataset" / "reward_annotations"

# game → list of (reward_enum, feature_name, sub_condition, placeholder_condition_value)
# reward_enum은 dungeon과 동일한 1~5 사용
_GAME_DEFS: dict[str, list[tuple]] = {
    "sokoban": [
        (1, "region",        "box",       3.0),   # 박스 배치 영역 수
        (2, "path_length",   "",         20.0),   # 풀이 최소 이동 수
        (3, "block",         "wall",      0.3),   # 벽 비율
        (4, "bat_amount",    "box",       3.0),   # 박스 개수
        (5, "bat_direction", "",          0.5),   # 박스 위치 편향
    ],
    "zelda": [
        (1, "region",        "room",      1.0),   # 방 연결 수
        (2, "path_length",   "",         15.0),   # 방 간 최장 경로
        (3, "block",         "wall",      0.3),   # 벽 밀도
        (4, "bat_amount",    "enemy",     3.0),   # 적 개수
        (5, "bat_direction", "enemy",     0.5),   # 적 위치 편향
    ],
    "doom": [
        (1, "region",        "room",      2.0),   # 방 수
        (2, "path_length",   "",         30.0),   # 최장 이동 경로
        (3, "block",         "wall",      0.5),   # 벽 비율
        (4, "bat_amount",    "enemy",     5.0),   # 적 배치 수
        (5, "bat_direction", "enemy",     0.5),   # 적 방향 편향
    ],
    "pokemon": [
        (1, "region",        "",          2.0),   # 연결 영역 수
        (2, "path_length",   "",         20.0),   # 최장 경로
        (3, "block",         "wall",      0.4),   # 벽 비율
        (4, "bat_amount",    "object",    4.0),   # 오브젝트 수
        (5, "bat_direction", "",          0.5),   # 오브젝트 위치 편향
    ],
}


def generate_placeholder_csv(game: str, features: list[tuple]) -> Path:
    """
    게임 단위 placeholder CSV를 생성한다.
    파일명에 _placeholder 접미사를 붙인다.
    """
    output_path = _OUTPUT_DIR / f"{game}_reward_annotations_placeholder.csv"

    fieldnames = [
        "game",
        "is_placeholder",   # 더미 데이터 표시 키워드 (항상 "true")
        "reward_enum",
        "feature_name",
        "sub_condition",
        # dungeon과 동일한 condition_1~5 컬럼 사용
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
            # 해당 reward_enum 컬럼에만 값, 나머지 빈값
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


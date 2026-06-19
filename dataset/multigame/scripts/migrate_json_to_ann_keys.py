#!/usr/bin/env python3
"""

dataset/multigame/scripts/migrate_json_to_ann_keys.py
======================================================
기존 {key}.json 캐시를 신규 포맷으로 변환하는 임시 마이그레이션 스크립트.

구 포맷: game, source_id, instruction, order, meta (instruction/meta 포함)
신규 포맷: game, source_id, order, ann_keys (ann.json 키만)

변환 조건:
  - ann.json이 존재하는 경우에만 ann_keys를 계산하여 추가
  - ann.json이 없으면 instruction/meta를 제거만 하고 ann_keys는 비워둠

Usage:
  python dataset/multigame/scripts/migrate_json_to_ann_keys.py
  python dataset/multigame/scripts/migrate_json_to_ann_keys.py --cache-dir dataset/multigame/cache/artifacts
  python dataset/multigame/scripts/migrate_json_to_ann_keys.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from dataset.multigame.cache_utils import (
    load_game_annotations_from_cache,
    update_json_with_ann_keys,
    _stable_json,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_HERE = Path(__file__).parent
_DEFAULT_CACHE_DIR = _HERE.parent / "cache" / "artifacts"


def _is_old_format(entry: dict) -> bool:
    """구 포맷 여부: instruction 또는 meta 필드가 있으면 구 포맷."""
    return "instruction" in entry or "meta" in entry


def migrate_game(cache_dir: Path, game: str, dry_run: bool = False) -> int:
    """한 게임의 .json을 신규 포맷으로 변환한다. 변환된 파일 수를 반환."""
    game_dir = cache_dir / game
    if not game_dir.exists():
        return 0

    json_files = [
        f for f in game_dir.glob("*.json")
        if not f.name.endswith(".info.json") and not f.name.endswith(".ann.json")
    ]
    if not json_files:
        return 0

    converted = 0
    for meta_path in json_files:
        key = meta_path.stem
        entries = json.loads(meta_path.read_text(encoding="utf-8"))

        if not entries:
            continue

        # 이미 신규 포맷이면 ann_keys만 채워지지 않은 경우 보완
        already_new = not _is_old_format(entries[0])

        # ann.json에서 ann_keys 계산
        ann_data = load_game_annotations_from_cache(cache_dir, game, key)

        if already_new and ann_data is None:
            logger.info(f"  [{game}] {key[:12]}….json: 이미 신규 포맷, ann.json 없음 — 건너뜀")
            continue
        if already_new and "ann_keys" in entries[0]:
            logger.info(f"  [{game}] {key[:12]}….json: 이미 신규 포맷 + ann_keys 있음 — 건너뜀")
            continue

        # 신규 포맷 변환
        new_entries = []
        for e in entries:
            new_entry: dict = {
                "game":      e["game"],
                "source_id": e["source_id"],
                "order":     e.get("order"),
            }
            new_entries.append(new_entry)

        n_samples = len(new_entries)
        logger.info(f"  [{game}] {key[:12]}….json: {n_samples}개 샘플 변환"
                    + (" (dry-run)" if dry_run else ""))

        if not dry_run:
            meta_path.write_text(_stable_json(new_entries), encoding="utf-8")
            converted += 1
            # ann_keys 추가
            if ann_data is not None:
                update_json_with_ann_keys(cache_dir, game, key, ann_data)
                logger.info(f"  [{game}] {key[:12]}….json: ann_keys 추가 완료")
            else:
                logger.warning(f"  [{game}] ann.json 없음 — ann_keys 없이 저장 (annotate.py 실행 후 재로드 시 자동 추가)")
        else:
            converted += 1

    return converted


def main() -> None:
    parser = argparse.ArgumentParser(
        description="{key}.json을 신규 포맷(game/source_id/order/ann_keys)으로 마이그레이션"
    )
    parser.add_argument("--cache-dir", type=Path, default=_DEFAULT_CACHE_DIR)
    parser.add_argument("--games", nargs="+",
                        default=["doom", "dungeon", "zelda", "pokemon", "sokoban"])
    parser.add_argument("--dry-run", action="store_true",
                        help="실제로 쓰지 않고 변환 대상만 출력")
    args = parser.parse_args()

    logger.info(f"캐시 디렉토리: {args.cache_dir}")
    if args.dry_run:
        logger.info("(dry-run 모드: 실제 변경 없음)")

    total = 0
    for game in args.games:
        n = migrate_game(args.cache_dir, game, dry_run=args.dry_run)
        if n:
            logger.info(f"[{game}] {n}개 파일 변환 완료")
        total += n

    logger.info(f"\n완료: 총 {total}개 파일 변환")


if __name__ == "__main__":
    main()

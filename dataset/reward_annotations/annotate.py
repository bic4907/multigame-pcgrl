#!/usr/bin/env python3
"""
dataset/annotation/annotate.py
================================
캐시에서 doom/zelda/sokoban/pokemon/dungeon 맵을 읽어
per-sample reward annotation CSV를 생성한다.

tile_mapping.json 기반 unified 카테고리를 사용한다 (dataloader와 동일한 방식):
  raw map → to_unified (0-4) → +1 shift → MultigameTiles (1-5)

Reward enum 정의:
  0 (RG)  region              - 연결된 통로 영역 수  → condition_0
  1 (PL)  path_length         - 가장 긴 경로 길이    → condition_1
  2 (IC)  interactable_count  - Interactive 타일 수  → condition_2
  3 (HC)  hazard_count        - Hazard 타일 수       → condition_3
  4 (CC)  collectable_count   - Collectable 타일 수  → condition_4

passible (region/path_length 기준):
  unified EMPTY(1) + HAZARD(4) + COLLECTABLE(5)  ← 모든 게임 공통

Usage:
  python dataset/annotation/annotate.py
  python dataset/annotation/annotate.py --games doom zelda dungeon
  python dataset/annotation/annotate.py --cache-dir dataset/multigame/cache/artifacts
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# 프로젝트 루트를 경로에 추가 (dataset/annotation/ 두 단계 위)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import jax.numpy as jnp

from evaluator.measures import (
    get_region as eval_get_region,
    get_path_length as eval_get_path_length,
    get_interactive_count,
    get_hazard_count,
    get_collectable_count,
)
from dataset.multigame.tile_utils import to_unified
from envs.probs.multigame import MultigameTiles

# unified passible tiles (MultigameTiles 공간): EMPTY(1) + HAZARD(4) + COLLECTABLE(5)
_UNIFIED_PASSIBLE = jnp.array(
    [int(MultigameTiles.EMPTY), int(MultigameTiles.HAZARD), int(MultigameTiles.COLLECTABLE)],
    dtype=jnp.int32,
)

# ── 경로 ─────────────────────────────────────────────────────────────────────────
_HERE = Path(__file__).parent                              # dataset/annotation/
_CACHE_DIR = _HERE.parent / "multigame" / "cache" / "artifacts"
_ANNOT_DIR = _HERE                                         # CSV를 같은 폴더에 출력

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── 게임별 sub_condition 레이블 (tile_mapping.json unified 카테고리 기준) ──────────
_GAME_CONFIG = {
    "doom": {
        "sub_cond_interactable": "spawn+door+danger",
        "sub_cond_hazard":       "enemy",
        "sub_cond_collectable":  "item",
    },
    "zelda": {
        "sub_cond_interactable": "door+block+start",
        "sub_cond_hazard":       "mob",
        "sub_cond_collectable":  "object",
    },
    "sokoban": {
        "sub_cond_interactable": "box",
        "sub_cond_hazard":       "",
        "sub_cond_collectable":  "",
    },
    "pokemon": {
        "sub_cond_interactable": "spawn+water",
        "sub_cond_hazard":       "enemy",
        "sub_cond_collectable":  "object",
    },
    "dungeon": {
        "sub_cond_interactable": "",
        "sub_cond_hazard":       "enemy",
        "sub_cond_collectable":  "treasure",
    },
}

CSV_HEADER = [
    "key", "instruction_raw", "instruction_uni", "level_id", "sample_id",
    "reward_enum", "feature_name", "sub_condition",
    "condition_0", "condition_1", "condition_2", "condition_3", "condition_4",
]

# 게임별 key prefix (앞 2글자 약어)
GAME_PREFIX: Dict[str, str] = {
    "doom":    "dm",
    "zelda":   "zl",
    "sokoban": "sk",
    "pokemon": "pk",
    "dungeon": "dg",
}


# ── 캐시 로드 ─────────────────────────────────────────────────────────────────────

def _load_cache(cache_dir: Path) -> Optional[List[dict]]:
    """캐시 디렉토리에서 npz + json 캐시를 로드한다."""
    json_files = sorted(
        f for f in cache_dir.glob("*.json")
        if not f.name.endswith(".info.json")
    )
    if not json_files:
        logger.error(f"캐시 json 파일 없음: {cache_dir}")
        return None

    meta_path = json_files[-1]
    npz_path = meta_path.with_suffix(".npz")
    if not npz_path.exists():
        logger.error(f"캐시 npz 파일 없음: {npz_path}")
        return None

    arrays = np.load(npz_path)["arrays"]
    meta: List[dict] = json.loads(meta_path.read_text(encoding="utf-8"))
    assert len(meta) == len(arrays), "캐시 메타/배열 크기 불일치"

    return [
        {**m, "array": arrays[i].astype(np.int32)}
        for i, m in enumerate(meta)
    ]


# ── sample_id 단축 ───────────────────────────────────────────────────────────────

def _shorten_source_id(source_id: str, game: str) -> str:
    """
    source_id를 게임별로 간결한 식별자로 변환한다.

    doom   : /abs/path/Doom/Processed/e1m1.txt|3  → Doom1_e1m1_003
             /abs/path/Doom2/Processed/e1m1.txt|3 → Doom2_e1m1_003
    zelda  : tloz1_1_r0_c0_rot90                  → (그대로)
    sokoban: medium/train/000.txt#5               → medium_000_005
             hard/001.txt#2                       → hard_001_002
    pokemon: pokemon_0042                          → (그대로)
    """
    if game == "doom":
        # source_id 예: "/path/to/Doom2/Processed/e1m1.txt|3"
        if "|" in source_id:
            path_part, slice_idx = source_id.rsplit("|", 1)
        else:
            path_part, slice_idx = source_id, "0"
        p = Path(path_part)
        stem = p.stem  # e.g. "e1m1"
        parts = p.parts
        version = "Doom2" if any("Doom2" in part for part in parts) else "Doom1"
        return f"{version}_{stem}_{int(slice_idx):03d}"

    if game == "sokoban":
        # source_id 예: "medium/train/000.txt#5" or "hard/001.txt#2"
        if "#" in source_id:
            path_part, lvl_idx = source_id.rsplit("#", 1)
        else:
            path_part, lvl_idx = source_id, "0"
        p = Path(path_part)
        # 최상위 폴더(hard/medium) + stem + 인덱스
        parts = p.parts
        difficulty = "hard" if any("hard" in part for part in parts) else "medium"
        return f"{difficulty}_{p.stem}_{int(lvl_idx):03d}"

    # zelda, pokemon: source_id가 이미 짧음
    return source_id


# ── measure 계산 ─────────────────────────────────────────────────────────────────

def _compute_measures(
    env_map: np.ndarray,
    game: str,
) -> Tuple[float, float, float, float, float, float]:
    """
    tile_mapping.json unified 카테고리 기반으로 6가지 measure를 계산한다.
    dataloader(to_unified)와 동일한 매핑 방식을 사용한다.

      raw map → to_unified (0-4) → +1 shift → MultigameTiles (1-5)

    passible = EMPTY(1) + HAZARD(4) + COLLECTABLE(5)  (모든 게임 공통)

    Returns
    -------
    (rg, pl, wc, ic_inter, ic_hazard, ic_coll)
      rg        : region count
      pl        : path length
      wc        : wall tile count         (unified WALL=2)
      ic_inter  : interactable tile count (unified INTERACTIVE=3)
      ic_hazard : hazard tile count       (unified HAZARD=4)
      ic_coll   : collectable tile count  (unified COLLECTABLE=5)
    """
    unified       = to_unified(env_map, game, warn_unmapped=False)
    multigame_map = jnp.array(unified + 1, dtype=jnp.int32)

    rg       = float(eval_get_region(multigame_map, _UNIFIED_PASSIBLE))
    pl       = float(eval_get_path_length(multigame_map, _UNIFIED_PASSIBLE))
    wc       = float(jnp.sum(multigame_map == int(MultigameTiles.WALL)))
    ic_inter = float(get_interactive_count(multigame_map))
    ic_haz   = float(get_hazard_count(multigame_map))
    ic_coll  = float(get_collectable_count(multigame_map))

    return rg, pl, wc, ic_inter, ic_haz, ic_coll


# ── 행 생성 ───────────────────────────────────────────────────────────────────────

def _make_rows(
    samples: List[dict],
    game: str,
    config: dict,
) -> List[dict]:
    """
    한 게임의 전체 샘플에 대해 5×N 행을 생성한다.
    출력 순서: reward_enum 0 전체 → 1 전체 → … → 4 전체 (enum별 그룹화).
    key / level_id 는 게임별 prefix + 게임 내 연속 번호 (reward_enum 바뀌어도 초기화 없음).
    """
    prefix = GAME_PREFIX.get(game, game[:2])

    # 1단계: 전체 샘플 measure 계산
    computed = []
    for order_idx, sample in enumerate(samples):
        env_map     = sample["array"]
        instruction = sample.get("instruction") or ""
        source_id   = _shorten_source_id(
            sample.get("source_id", str(order_idx)), game
        )

        if (order_idx + 1) % 100 == 0 or order_idx == 0:
            logger.info(f"  [{game}] {order_idx + 1}/{len(samples)} …")

        try:
            rg, pl, wc, oc, mc, ic = _compute_measures(env_map, game)
        except Exception as exc:
            logger.warning(f"  measure 실패 ({source_id}): {exc} → 0으로 대체")
            rg, pl, wc, oc, mc, ic = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        computed.append((instruction, order_idx, source_id, rg, pl, wc, oc, mc, ic))

    # 2단계: reward_enum별 그룹으로 행 생성
    sc_object = config["sub_cond_interactable"]
    sc_mob    = config["sub_cond_hazard"]
    sc_item   = config["sub_cond_collectable"]

    # (reward_enum, feature_name, cond_col, val_idx, sub_condition)
    # val_idx: computed 튜플에서의 인덱스 (3=rg, 4=pl, 6=oc, 7=mc, 8=ic)
    # wc(5) 는 unified wall이므로 annotation에서 제외; oc/mc/ic 가 interactable/hazard/collectable
    enum_specs = [
        (0, "region",             "condition_0", 3, ""),
        (1, "path_length",        "condition_1", 4, ""),
        (2, "interactable_count", "condition_2", 6, sc_object),
        (3, "hazard_count",       "condition_3", 7, sc_mob),
        (4, "collectable_count",  "condition_4", 8, sc_item),
    ]

    rows: List[dict] = []
    row_n = 0  # 게임 내 연속 행 번호 (reward_enum 경계에서 초기화하지 않음)

    for reward_enum, feature_name, cond_col, val_idx, sub_cond in enum_specs:
        for instruction, order_idx, source_id, *vals in computed:
            value = vals[val_idx - 3]  # offset: idx 3 → vals[0]
            row: dict = {
                "key":            f"{prefix}{row_n:06d}",
                "instruction_raw": "",
                "instruction_uni": "",
                "level_id":       f"{row_n:06d}",
                "sample_id":     source_id,
                "reward_enum":   reward_enum,
                "feature_name":  feature_name,
                "sub_condition": sub_cond,
                "condition_0":   "",
                "condition_1":   "",
                "condition_2":   "",
                "condition_3":   "",
                "condition_4":   "",
            }
            row[cond_col] = value
            rows.append(row)
            row_n += 1

    return rows


# ── 게임별 처리 ───────────────────────────────────────────────────────────────────

def annotate_game(
    game: str,
    samples: List[dict],
    out_dir: Path,
) -> int:
    """
    한 게임의 annotation CSV를 생성한다.
    기록한 row 수를 반환한다.
    """
    config = _GAME_CONFIG[game]
    logger.info(f"\n=== {game.upper()} ({len(samples)} samples) ===")

    t0 = time.perf_counter()
    rows = _make_rows(samples, game, config)
    elapsed = time.perf_counter() - t0

    out_path = out_dir / f"{game}_reward_annotations.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"  → {out_path.name}  ({len(rows)} rows, {len(samples)} samples × 5)  [{elapsed:.1f}s]")
    return len(rows)


# ── CLI ───────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reward annotation CSV 생성 (doom / zelda / sokoban / pokemon / dungeon)"
    )
    parser.add_argument(
        "--games", nargs="+",
        default=["doom", "zelda", "sokoban", "pokemon", "dungeon"],
        choices=["doom", "zelda", "sokoban", "pokemon", "dungeon"],
        help="처리할 게임 목록 (기본: 전체)",
    )
    parser.add_argument(
        "--cache-dir", type=Path,
        default=_CACHE_DIR,
        help=f"캐시 디렉토리 (기본: {_CACHE_DIR})",
    )
    parser.add_argument(
        "--out-dir", type=Path,
        default=_ANNOT_DIR,
        help=f"CSV 출력 디렉토리 (기본: {_ANNOT_DIR})",
    )
    args = parser.parse_args()

    logger.info(f"캐시 로드: {args.cache_dir}")

    # 게임별 최대 샘플 수 (viewer / MultiGameDataset 의 max_samples 와 일치)
    _MAX_SAMPLES: Dict[str, int] = {
        "doom":    1000,
        "zelda":   1000,
        "sokoban": 1000,
        "pokemon": 1000,
        "dungeon": 4000,
    }

    # 모든 서브디렉토리를 스캔하여 game 태그 기준으로 합산
    # (doom/ + doom2/ 처럼 동일 game 태그가 여러 폴더에 분산된 경우 자동 합산)
    by_game: Dict[str, List[dict]] = {}
    if args.cache_dir.is_dir():
        for sub in sorted(args.cache_dir.iterdir()):
            if not sub.is_dir():
                continue
            samples = _load_cache(sub)
            if not samples:
                continue
            for s in samples:
                g = s.get("game", "")
                if g:
                    by_game.setdefault(g, []).append(s)

    if not by_game:
        # 서브디렉토리 캐시 없음 → 통합 캐시 fallback
        all_samples = _load_cache(args.cache_dir)
        if all_samples is None:
            logger.error("로드된 캐시 없음")
            return
        for s in all_samples:
            by_game.setdefault(s.get("game", ""), []).append(s)

    # max_samples 적용 (캐시에 초과분이 있을 경우 절단)
    for g in list(by_game.keys()):
        limit = _MAX_SAMPLES.get(g)
        if limit is not None and len(by_game[g]) > limit:
            by_game[g] = by_game[g][:limit]

    logger.info(
        "서브디렉토리 캐시 합산 결과: "
        + ", ".join(f"{g}={len(v)}" for g, v in sorted(by_game.items()))
    )

    logger.info(
        "게임별 샘플 수: "
        + ", ".join(f"{g}={len(v)}" for g, v in sorted(by_game.items()))
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    for game in args.games:
        samples = by_game.get(game, [])
        if not samples:
            logger.warning(f"{game}: 캐시에 샘플 없음, 건너뜀")
            continue
        n = annotate_game(game, samples, args.out_dir)
        total_rows += n

        # placeholder CSV 제거 (실제 annotation 파일이 생성됐으므로)
        placeholder = args.out_dir / f"{game}_reward_annotations_placeholder.csv"
        if placeholder.exists():
            placeholder.unlink()
            logger.info(f"  placeholder 삭제: {placeholder.name}")

    logger.info(f"\n완료: 총 {total_rows} rows 생성됨")


if __name__ == "__main__":
    main()

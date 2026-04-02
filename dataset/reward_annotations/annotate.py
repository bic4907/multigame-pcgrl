#!/usr/bin/env python3
"""
dataset/annotation/annotate.py
================================
캐시에서 doom/zelda/sokoban/pokemon/dungeon 맵을 읽어
per-sample reward annotation CSV를 생성한다.

Reward enum 정의:
  1 (RG)  region        - 연결된 통로 영역 수 (passable=Empty+Item) → condition_1
  2 (PL)  path_length   - 가장 긴 경로 (passable=Empty+Item)        → condition_2
  3 (WC)  wall_count    - Wall 그룹 타일 총 개수                    → condition_3
  4 (OC)  object_count  - Object 그룹 타일 총 개수                  → condition_4
  5 (MC)  mob_count     - Mob 그룹 타일 총 개수                     → condition_5
  6 (IC)  item_count    - Item 그룹 타일 총 개수                    → condition_6

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

from multigame_evaluator.measure import doom as doom_m
from multigame_evaluator.measure import zelda as zelda_m
from multigame_evaluator.measure import sokoban as sokoban_m
from multigame_evaluator.measure import pokemon as pokemon_m
from multigame_evaluator.measure import dungeon as dungeon_m

# ── 경로 ─────────────────────────────────────────────────────────────────────────
_HERE = Path(__file__).parent                              # dataset/annotation/
_CACHE_DIR = _HERE.parent / "multigame" / "cache" / "artifacts"
_ANNOT_DIR = _HERE                                         # CSV를 같은 폴더에 출력

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── 게임별 tile 설정 ─────────────────────────────────────────────────────────────
_GAME_CONFIG = {
    "doom": {
        "module":           doom_m,
        "passible":         doom_m.DoomPassible,
        "wall":             doom_m.DoomWall,
        "object":           doom_m.DoomObject,
        "mob":              doom_m.DoomMob,
        "item":             doom_m.DoomItem,
        "sub_cond_wall":    "wall+empty",
        "sub_cond_object":  "spawn+door+danger",
        "sub_cond_mob":     "enemy",
        "sub_cond_item":    "item",
    },
    "zelda": {
        "module":           zelda_m,
        "passible":         zelda_m.ZeldaPassible,
        "wall":             zelda_m.ZeldaWall,
        "object":           zelda_m.ZeldaObject,
        "mob":              zelda_m.ZeldaMob,
        "item":             zelda_m.ZeldaItem,
        "sub_cond_wall":    "wall+hazard+empty",
        "sub_cond_object":  "block+door+start",
        "sub_cond_mob":     "mob",
        "sub_cond_item":    "object",
    },
    "sokoban": {
        "module":           sokoban_m,
        "passible":         sokoban_m.SokobanPassible,
        "wall":             sokoban_m.SokobanWall,
        "object":           sokoban_m.SokobanObject,
        "mob":              sokoban_m.SokobanMob,
        "item":             sokoban_m.SokobanItem,
        "sub_cond_wall":    "wall",
        "sub_cond_object":  "box",
        "sub_cond_mob":     "",
        "sub_cond_item":    "",
    },
    "pokemon": {
        "module":           pokemon_m,
        "passible":         pokemon_m.PokemonPassible,
        "wall":             pokemon_m.PokemonWall,
        "object":           pokemon_m.PokemonObject,
        "mob":              pokemon_m.PokemonMob,
        "item":             pokemon_m.PokemonItem,
        "sub_cond_wall":    "tree+house+fence",
        "sub_cond_object":  "spawn+water",
        "sub_cond_mob":     "",
        "sub_cond_item":    "object",
    },
    "dungeon": {
        "module":           dungeon_m,
        "passible":         dungeon_m.DungeonPassible,
        "wall":             dungeon_m.DungeonWall,
        "object":           dungeon_m.DungeonObject,
        "mob":              dungeon_m.DungeonMob,
        "item":             dungeon_m.DungeonItem,
        "sub_cond_wall":    "wall+border",
        "sub_cond_object":  "",
        "sub_cond_mob":     "bat",
        "sub_cond_item":    "",
    },
}

CSV_HEADER = [
    "key", "instruction", "level_id", "sample_id",
    "reward_enum", "feature_name", "sub_condition",
    "condition_0", "condition_1", "condition_2", "condition_3", "condition_4",
]


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


# ── measure 헬퍼 ──────────────────────────────────────────────────────────────────

def _tile_count(env_map: np.ndarray, tile_ids: jnp.ndarray) -> float:
    """tile_ids 에 포함된 모든 타일 종류의 합산 개수를 반환한다."""
    count = 0
    for tid in np.array(tile_ids, dtype=np.int32):
        count += int((env_map == int(tid)).sum())
    return float(count)


def _compute_measures(
    env_map: np.ndarray,
    config: dict,
) -> Tuple[float, float, float, float, float, float]:
    """
    6가지 measure를 계산한다.

    Returns
    -------
    (rg, pl, wc, oc, mc, ic)
      rg : region count          (passable = Empty + Item)
      pl : path length           (passable = Empty + Item)
      wc : wall tile count
      oc : object tile count
      mc : mob tile count
      ic : item tile count
    """
    module     = config["module"]
    passible   = config["passible"]
    wall_ids   = config["wall"]
    object_ids = config["object"]
    mob_ids    = config["mob"]
    item_ids   = config["item"]

    jmap = jnp.array(env_map)

    rg = float(module.get_region(jmap, passible))
    pl = float(module.get_path_length(jmap, passible))
    wc = _tile_count(env_map, wall_ids)
    oc = _tile_count(env_map, object_ids)
    mc = _tile_count(env_map, mob_ids)
    ic = _tile_count(env_map, item_ids)

    return rg, pl, wc, oc, mc, ic


# ── 행 생성 ───────────────────────────────────────────────────────────────────────

def _make_rows(
    samples: List[dict],
    game: str,
    config: dict,
    key_start: int,
) -> List[dict]:
    """
    한 게임의 전체 샘플에 대해 5×N 행을 생성한다.
    출력 순서: reward_enum 1 전체 → 2 전체 → … → 5 전체 (enum별 그룹화).
    """
    # 1단계: 전체 샘플 measure 계산
    # 각 원소: (instruction, order_idx, source_id, rg, pl, wc, bc, bd)
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
            rg, pl, wc, oc, mc, ic = _compute_measures(env_map, config)
        except Exception as exc:
            logger.warning(f"  measure 실패 ({source_id}): {exc} → 0으로 대체")
            rg, pl, wc, oc, mc, ic = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        computed.append((instruction, order_idx, source_id, rg, pl, wc, oc, mc, ic))

    # 2단계: reward_enum별 그룹으로 행 생성
    sc_object = config["sub_cond_object"]
    sc_mob    = config["sub_cond_mob"]
    sc_item   = config["sub_cond_item"]

    # (reward_enum, feature_name, cond_col, val_idx, sub_condition)
    # val_idx: computed 튜플에서의 인덱스 (3=rg, 4=pl, 5=wc, 6=oc, 7=mc, 8=ic)
    # enum 순서: 1=region, 2=path_length, 3=interactable, 4=hazard, 5=collectable
    enum_specs = [
        (0, "region",             "condition_0", 3, ""),
        (1, "path_length",        "condition_1", 4, ""),
        (2, "interactable_count", "condition_2", 6, sc_object),
        (3, "hazard_count",       "condition_3", 7, sc_mob),
        (4, "collectable_count",  "condition_4", 8, sc_item),
    ]

    rows: List[dict] = []
    key = key_start

    for reward_enum, feature_name, cond_col, val_idx, sub_cond in enum_specs:
        for instruction, order_idx, source_id, *vals in computed:
            value = vals[val_idx - 3]  # offset: idx 3 → vals[0]
            row: dict = {
                "key":           f"{key:06d}",
                "instruction":   instruction,
                "level_id":      f"{order_idx:06d}",
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
            key += 1

    return rows


# ── 게임별 처리 ───────────────────────────────────────────────────────────────────

def annotate_game(
    game: str,
    samples: List[dict],
    out_dir: Path,
    key_start: int = 0,
) -> int:
    """
    한 게임의 annotation CSV를 생성한다.
    placeholder 파일을 대체하며, 기록한 row 수를 반환한다.
    """
    config = _GAME_CONFIG[game]
    logger.info(f"\n=== {game.upper()} ({len(samples)} samples) ===")

    t0 = time.perf_counter()
    rows = _make_rows(samples, game, config, key_start)
    elapsed = time.perf_counter() - t0

    out_path = out_dir / f"{game}_reward_annotations.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"  → {out_path.name}  ({len(rows)} rows, {len(samples)} samples × 6)  [{elapsed:.1f}s]")
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
        n = annotate_game(game, samples, args.out_dir, key_start=total_rows)
        total_rows += n

        # placeholder CSV 제거 (실제 annotation 파일이 생성됐으므로)
        placeholder = args.out_dir / f"{game}_reward_annotations_placeholder.csv"
        if placeholder.exists():
            placeholder.unlink()
            logger.info(f"  placeholder 삭제: {placeholder.name}")

    logger.info(f"\n완료: 총 {total_rows} rows 생성됨")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
dataset/reward_annotations/annotate.py
=======================================
캐시에서 doom/zelda/sokoban/pokemon/dungeon 맵을 읽어
per-sample reward annotation을 계산하고 {key}.ann.json에 저장한다.

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
  python dataset/reward_annotations/annotate.py
  python dataset/reward_annotations/annotate.py --games doom zelda dungeon
  python dataset/reward_annotations/annotate.py --cache-dir dataset/multigame/cache/artifacts
  python dataset/reward_annotations/annotate.py --force  # 기존 ann.json 덮어쓰기
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# 프로젝트 루트를 경로에 추가 (dataset/reward_annotations/ 두 단계 위)
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
from dataset.multigame.cache_utils import (
    save_game_annotations_to_cache,
    load_game_annotations_from_cache,
)
from envs.probs.multigame import MultigameTiles

# unified passible tiles (MultigameTiles 공간): EMPTY(1) + HAZARD(4) + COLLECTABLE(5)
_UNIFIED_PASSIBLE = jnp.array(
    [int(MultigameTiles.EMPTY), int(MultigameTiles.HAZARD), int(MultigameTiles.COLLECTABLE)],
    dtype=jnp.int32,
)

# ── 경로 ─────────────────────────────────────────────────────────────────────────
_HERE = Path(__file__).parent                              # dataset/reward_annotations/
_CACHE_DIR = _HERE.parent / "multigame" / "cache" / "artifacts"

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

# 게임별 key prefix (앞 2글자 약어)
GAME_PREFIX: Dict[str, str] = {
    "doom":    "dm",
    "zelda":   "zl",
    "sokoban": "sk",
    "pokemon": "pk",
    "dungeon": "dg",
}


# ── 캐시 로드 (standalone 모드용) ────────────────────────────────────────────────

def _load_cache_dir(cache_dir: Path) -> Optional[List[dict]]:
    """캐시 디렉토리에서 npz + json 캐시를 로드한다."""
    json_files = sorted(
        f for f in cache_dir.glob("*.json")
        if not f.name.endswith(".info.json") and not f.name.endswith(".ann.json")
    )
    if not json_files:
        return None

    meta_path = json_files[-1]
    npz_path = meta_path.with_suffix(".npz")
    if not npz_path.exists():
        logger.error(f"캐시 npz 파일 없음: {npz_path}")
        return None

    arrays = np.load(npz_path)["arrays"]
    meta: List[dict] = json.loads(meta_path.read_text(encoding="utf-8"))
    if len(meta) != len(arrays):
        logger.error(f"캐시 메타/배열 크기 불일치: {cache_dir}")
        return None

    return [
        {**m, "array": arrays[i].astype(np.int32)}
        for i, m in enumerate(meta)
    ]


def _get_cache_key(cache_dir: Path) -> Optional[str]:
    """캐시 디렉토리에서 npz 파일 이름으로 캐시 키를 추출한다."""
    npz_files = sorted(f for f in cache_dir.glob("*.npz"))
    if not npz_files:
        return None
    return npz_files[-1].stem


# ── sample_id 단축 ───────────────────────────────────────────────────────────────

def _shorten_source_id(source_id: str, game: str) -> str:
    """source_id를 게임별로 간결한 식별자로 변환한다."""
    if game == "doom":
        if "|" in source_id:
            path_part, slice_idx = source_id.rsplit("|", 1)
        else:
            path_part, slice_idx = source_id, "0"
        p = Path(path_part)
        parts = p.parts
        version = "Doom2" if any("Doom2" in part for part in parts) else "Doom1"
        return f"{version}_{p.stem}_{int(slice_idx):03d}"

    if game == "sokoban":
        if "#" in source_id:
            path_part, lvl_idx = source_id.rsplit("#", 1)
        else:
            path_part, lvl_idx = source_id, "0"
        p = Path(path_part)
        parts = p.parts
        difficulty = "hard" if any("hard" in part for part in parts) else "medium"
        return f"{difficulty}_{p.stem}_{int(lvl_idx):03d}"

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

    samples 형식: [{"array": np.ndarray, "source_id": str, "instruction": str|None}, ...]
    """
    prefix = GAME_PREFIX.get(game, game[:2])

    # 1단계: 전체 샘플 measure 계산
    computed = []
    for order_idx, sample in enumerate(samples):
        env_map   = sample["array"]
        source_id = _shorten_source_id(
            sample.get("source_id", str(order_idx)), game
        )

        if (order_idx + 1) % 100 == 0 or order_idx == 0:
            logger.info(f"  [{game}] {order_idx + 1}/{len(samples)} …")

        try:
            rg, pl, wc, oc, mc, ic = _compute_measures(env_map, game)
        except Exception as exc:
            logger.warning(f"  measure 실패 ({source_id}): {exc} → 0으로 대체")
            rg, pl, wc, oc, mc, ic = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        computed.append((order_idx, source_id, rg, pl, wc, oc, mc, ic))

    # 2단계: reward_enum별 그룹으로 행 생성
    sc_object = config["sub_cond_interactable"]
    sc_mob    = config["sub_cond_hazard"]
    sc_item   = config["sub_cond_collectable"]

    # (reward_enum, feature_name, cond_col, val_idx, sub_condition)
    # val_idx: computed 튜플에서의 인덱스 (2=rg, 3=pl, 5=oc, 6=mc, 7=ic)
    enum_specs = [
        (0, "region",             "condition_0", 2, ""),
        (1, "path_length",        "condition_1", 3, ""),
        (2, "interactable_count", "condition_2", 5, sc_object),
        (3, "hazard_count",       "condition_3", 6, sc_mob),
        (4, "collectable_count",  "condition_4", 7, sc_item),
    ]

    rows: List[dict] = []
    row_n = 0  # 게임 내 연속 행 번호

    for reward_enum, feature_name, cond_col, val_idx, sub_cond in enum_specs:
        for order_idx, source_id, *vals in computed:
            value = vals[val_idx - 2]  # offset: idx 2 → vals[0]
            row: dict = {
                "key":             f"{prefix}{row_n:06d}",
                "source_id":       source_id,
                "reward_enum":     reward_enum,
                "feature_name":    feature_name,
                "sub_condition":   sub_cond,
                "condition_0":     None,
                "condition_1":     None,
                "condition_2":     None,
                "condition_3":     None,
                "condition_4":     None,
                "instruction_raw": None,
                "instruction_uni": None,
            }
            row[cond_col] = value
            rows.append(row)
            row_n += 1

    return rows


# ── 외부 호출용 함수 ──────────────────────────────────────────────────────────────

def compute_game_annotations(
    samples,
    game: str,
) -> List[Dict[str, Any]]:
    """
    GameSample 리스트(또는 array/source_id를 가진 dict 리스트)에서
    5개 reward_enum에 대한 annotation 행을 계산하여 반환한다.

    MultiGameDataset 내부 자동 annotation에 사용된다.

    Parameters
    ----------
    samples : List[GameSample] or List[dict]
        각 원소에 .array / .source_id 속성 또는 키가 있어야 함.
    game : str

    Returns
    -------
    List[dict] — _make_rows() 반환값과 동일한 형식
    """
    config = _GAME_CONFIG.get(game, {
        "sub_cond_interactable": "",
        "sub_cond_hazard":       "",
        "sub_cond_collectable":  "",
    })

    # GameSample / dict 양쪽 지원
    sample_dicts: List[dict] = []
    for s in samples:
        if isinstance(s, dict):
            sample_dicts.append(s)
        else:
            sample_dicts.append({
                "array":      s.array,
                "source_id":  s.source_id,
                "instruction": getattr(s, "instruction", None),
            })

    return _make_rows(sample_dicts, game, config)


# ── 게임별 처리 ───────────────────────────────────────────────────────────────────

def annotate_game(
    game: str,
    samples: List[dict],
    cache_dir: Path,
    key: str,
    force: bool = False,
) -> int:
    """
    한 게임의 annotation을 계산하고 {key}.ann.json에 저장한다.
    저장한 row 수를 반환한다.

    force=False이면 ann.json이 이미 존재하는 경우 건너뜀.
    """
    if not force:
        existing = load_game_annotations_from_cache(cache_dir, game, key)
        if existing is not None:
            logger.info(f"  [{game}] ann.json 이미 존재, 건너뜀 (--force로 재생성 가능)")
            return len(existing.get("annotations", []))

    config = _GAME_CONFIG[game]
    logger.info(f"\n=== {game.upper()} ({len(samples)} samples) ===")

    t0 = time.perf_counter()
    rows = _make_rows(samples, game, config)
    elapsed = time.perf_counter() - t0

    save_game_annotations_to_cache(
        cache_dir, game, key, rows,
        has_instructions=False,
        n_samples=len(samples),
    )

    logger.info(
        f"  → {game}/{key[:12]}….ann.json  "
        f"({len(rows)} rows, {len(samples)} samples × 5)  [{elapsed:.1f}s]"
    )
    return len(rows)


# ── CLI ───────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reward annotation 생성 → {key}.ann.json 저장"
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
        "--force", action="store_true",
        help="기존 ann.json이 있어도 덮어쓰기",
    )
    args = parser.parse_args()

    logger.info(f"캐시 로드: {args.cache_dir}")

    # 게임별 최대 샘플 수
    _MAX_SAMPLES: Dict[str, int] = {
        "doom":    1000,
        "zelda":   1000,
        "sokoban": 1000,
        "pokemon": 1000,
        "dungeon": 4000,
    }

    # 모든 서브디렉토리를 스캔하여 game 태그 기준으로 합산
    # {game: {key: samples}} 형태로 수집
    by_game_key: Dict[str, Dict[str, List[dict]]] = {}

    if args.cache_dir.is_dir():
        for sub in sorted(args.cache_dir.iterdir()):
            if not sub.is_dir():
                continue
            key = _get_cache_key(sub)
            if key is None:
                continue
            samples = _load_cache_dir(sub)
            if not samples:
                continue
            for s in samples:
                g = s.get("game", "")
                if g:
                    by_game_key.setdefault(g, {}).setdefault(key, []).append(s)

    if not by_game_key:
        logger.error("로드된 캐시 없음")
        return

    # max_samples 적용
    for g in list(by_game_key.keys()):
        limit = _MAX_SAMPLES.get(g)
        for k in list(by_game_key[g].keys()):
            if limit is not None and len(by_game_key[g][k]) > limit:
                by_game_key[g][k] = by_game_key[g][k][:limit]

    logger.info(
        "게임별 샘플 수: "
        + ", ".join(
            f"{g}={sum(len(v) for v in ks.values())}"
            for g, ks in sorted(by_game_key.items())
        )
    )

    total_rows = 0
    for game in args.games:
        game_keys = by_game_key.get(game, {})
        if not game_keys:
            logger.warning(f"{game}: 캐시에 샘플 없음, 건너뜀")
            continue
        for key, samples in game_keys.items():
            n = annotate_game(game, samples, args.cache_dir, key, force=args.force)
            total_rows += n

    logger.info(f"\n완료: 총 {total_rows} rows 생성됨")


if __name__ == "__main__":
    main()

"""
notebooks/render_level_thumbnails.py

plot_tsne_by_level.py의 figure2-1(embedding plot)에 붙일 게임별 대표 레벨
썸네일 + 대표 instruction 텍스트를 미리 뽑아 저장해두는 별도 스크립트다.
(매번 그림을 다시 그릴 때마다 무거운 데이터셋을 재구축하지 않기 위해
샘플 선택/렌더링과 플로팅을 분리한다.)

렌더링은 MultiGameDataset.render_sample() (dataset/multigame/render.py의
GameLevelRenderer) — 게임마다 다른 raw 타일 스프라이트를 쓰는 렌더러다.
(mgpcgrl 학습/평가 파이프라인이 쓰는 envs/probs/multigame.py:render_multigame_map()은
unified 5-category를 게임 구분 없이 동일한 스프라이트 세트로 그려서, 게임별로
시각적으로 구분이 안 된다 — 여기서는 그림에서 게임을 한눈에 구분할 수 있는 게
목적이라 게임별 raw 렌더러를 쓴다.) raw tile id 배열이 필요하므로
use_tile_mapping=False로 데이터셋을 구성한다 (CLIP 인코더 학습용 파이프라인은
전혀 필요 없다).

각 (game, quantized_bin) 조합에서 후보를 결정적으로 정렬(meta['key'] 기준)한 뒤
--picks로 지정한 인덱스의 샘플을 대표로 고른다 (지정 안 하면 0번). 렌더링한
PNG와 함께 그 샘플의 instruction 텍스트를 captions.json에 저장해서, 플롯에서
img.png 예시처럼 이미지 옆에 자연어 설명을 붙일 수 있게 한다.

quantized_bin은 encoder/data/clip_batch.py가 쓰는 것과 동일한
CUSTOM_THRESHOLDS 기준 np.digitize 로직을 raw GameSample.meta에 직접
적용해서 재현한다 (조건부 값 → 0~7 bin).

Usage:
    python notebooks/render_level_thumbnails.py \
        --reward-enum 1 \
        --bins pokemon=0 zelda=2 sokoban=4 doom=6 dungeon=7 \
        --picks pokemon=1 sokoban=4 \
        --out-dir notebooks/tsne_by_level/levels
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def quantized_bin_of(sample, reward_enum: int) -> int:
    from dataset.reward_annotations.instruction_config import CUSTOM_THRESHOLDS

    feature_name = sample.meta.get("feature_name", "")
    conditions = sample.meta.get("conditions", {})
    condition_value = conditions.get(reward_enum, next(iter(conditions.values()), None))
    thresholds = CUSTOM_THRESHOLDS.get(f"{sample.game}_{feature_name}")
    if thresholds is not None and condition_value is not None:
        return int(np.digitize(condition_value, thresholds))
    return 0


def condition_value_of(sample, reward_enum: int) -> float | None:
    conditions = sample.meta.get("conditions", {})
    return conditions.get(reward_enum, next(iter(conditions.values()), None))


# dataset/multigame/tile_mapping.json 의 "_tile_images"(raw tile_id -> 스프라이트 파일명)가
# "mapping"(raw_id -> unified category)과 어긋나 있는 게 여러 게임에서 확인됐다
# (예: sokoban/doom/zelda/pokemon 모두 WALL(id 1)과 FLOOR/EMPTY(id 2) 이미지가 서로 뒤바뀜,
# dungeon/doom은 ENEMY/SPAWN/TREASURE류 id들도 밀려있음). tile_mapping.json 자체는
# GameLevelRenderer/뷰어 등 다른 곳에서도 공유해서 쓰므로 건드리지 않고, 이 스크립트가
# 쓰는 렌더러 인스턴스에서만 보정한 이미지 파일명으로 덮어쓴다.
TILE_IMAGE_FIXES: dict[str, dict[str, str]] = {
    "sokoban": {"1": "sokoban_wall.png"},
    "doom": {"1": "doom_wall.png", "2": "doom_empty.png", "3": "doom_hazard.png", "4": "doom_interact.png"},
    "zelda": {"1": "wall.png", "2": "floor.png"},
    "pokemon": {"1": "tree.png", "2": "floor_0.png"},
    "dungeon": {"3": "bat.png", "4": "treasure.png", "5": "dungeon_interactable.png"},
}


def make_fixed_renderer():
    """TILE_IMAGE_FIXES를 적용한 GameLevelRenderer를 만들어 반환한다 (원본 tile_mapping.json은 그대로 둠)."""
    import copy

    from dataset.multigame.render import GameLevelRenderer

    renderer = GameLevelRenderer()
    renderer.tile_mapping = copy.deepcopy(renderer.tile_mapping)
    for game, overrides in TILE_IMAGE_FIXES.items():
        renderer.tile_mapping.setdefault(game, {}).setdefault("_tile_images", {}).update(overrides)
    return renderer


def sorted_candidates(samples, game: str, target_bin: int, reward_enum: int):
    """(game, quantized_bin)에 맞는 후보를 결정적 순서로 정렬해서 반환한다.

    condition_value(=path_length 등 raw 값)가 0인 샘플은 대부분 텅 빈/자명한
    맵이라 대표로 부적절하므로, 0이 아닌 후보가 하나라도 있으면 뒤로 미룬다
    (전부 0이면 그냥 그대로 둔다)."""
    cands = [s for s in samples if s.game == game and quantized_bin_of(s, reward_enum) == target_bin]
    cands = sorted(cands, key=lambda s: (str(s.meta.get("key", "")), str(s.source_id)))
    nonzero = [s for s in cands if condition_value_of(s, reward_enum) not in (0, 0.0, None)]
    zero = [s for s in cands if s not in nonzero]
    return nonzero + zero if nonzero else cands


def main() -> None:
    parser = argparse.ArgumentParser(description="게임별 대표 레벨 썸네일 + instruction 텍스트를 미리 뽑아 저장")
    parser.add_argument("--reward-enum", type=int, default=1)
    parser.add_argument("--bins", nargs="+", default=[
        "pokemon=0", "zelda=2", "sokoban=4", "doom=6", "dungeon=7",
    ], help="game=bin 형식으로 게임별 quantized_bin 지정")
    parser.add_argument("--picks", nargs="+", default=["pokemon=1", "sokoban=2"],
                         help="game=index 형식으로 정렬된 후보 중 몇 번째를 대표로 쓸지 지정 (기본 0)")
    parser.add_argument("--tile-size", type=int, default=16)
    parser.add_argument("--out-dir", default=str(ROOT / "notebooks" / "tsne_by_level" / "levels"))
    args = parser.parse_args()

    bin_map = {}
    for item in args.bins:
        game, bin_str = item.split("=")
        bin_map[game.strip()] = int(bin_str)
    pick_map = {}
    for item in args.picks:
        game, idx_str = item.split("=")
        pick_map[game.strip()] = int(idx_str)

    from dataset.multigame import MultiGameDataset

    print("building raw MultiGameDataset (use_tile_mapping=False) ...")
    ds = MultiGameDataset(use_tile_mapping=False)
    samples = ds.by_reward_enum(args.reward_enum)
    print(f"reward_enum={args.reward_enum} samples: {len(samples)}")

    renderer = make_fixed_renderer()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    captions: dict[str, str] = {}
    for game, qbin in bin_map.items():
        cands = sorted_candidates(samples, game, qbin, args.reward_enum)
        if not cands:
            print(f"[skip] no candidate for game={game} bin={qbin}")
            continue
        pick = pick_map.get(game, 0)
        pick = min(pick, len(cands) - 1)
        sample = cands[pick]

        key = f"{game}_bin{qbin}"
        out_path = out_dir / f"{key}.png"
        renderer.render(game=sample.game, level=sample.array, tile_size=args.tile_size, save_path=out_path)
        captions[key] = sample.instruction or ""
        condition_value = sample.meta.get("conditions", {}).get(args.reward_enum)
        print(f"saved: {out_path}  (pick={pick}/{len(cands)}, cond={condition_value})  inst={sample.instruction!r}")

    captions_path = out_dir / "captions.json"
    captions_path.write_text(json.dumps(captions, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved captions: {captions_path}")


if __name__ == "__main__":
    main()

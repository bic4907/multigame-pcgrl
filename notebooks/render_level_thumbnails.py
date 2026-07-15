"""
notebooks/render_level_thumbnails.py

plot_tsne_by_level.py의 figure2-1(embedding plot)에 붙일 게임별 대표 레벨
썸네일 + 대표 instruction 텍스트를 미리 뽑아 저장해두는 별도 스크립트다.
(매번 그림을 다시 그릴 때마다 무거운 데이터셋을 재구축하지 않기 위해
샘플 선택/렌더링과 플로팅을 분리한다.)

envs/probs/multigame.py:render_multigame_map()(mgpcgrl eval 파이프라인이 실제로
쓰는 렌더러)은 unified 5-category를 게임 구분 없이 동일한 스프라이트 세트로
그려서, 게임마다 그림이 달라 보이지 않는다 — 여기서는 게임을 한눈에 구분할 수
있는 게 목적이라, 게임별 raw 스프라이트를 쓰는 MultiGameDataset.render_sample()
(dataset/multigame/render.py의 GameLevelRenderer, dataset/multigame/tile_ims/)을 쓴다.

렌더러뿐 아니라 데이터 로딩 방식도 results/render/table_export/render_assets.py의
_load_dataset_samples_for_gt()와 완전히 동일하게 맞춘다:
  MultiGameDataset(use_tile_mapping=True)  # raw id가 아니라 unified 5-category
  -> preprocess_samples(longtail_cut=True)
  -> apply_tile_offset(samples, 1)         # unified(0~4) -> tile_mapping.json
                                            #   "_tile_images" 키 스킴(1~5)에 맞춤
tile_mapping.json의 "_tile_images"는 raw per-game tile id가 아니라 "unified
category + 1"로 인덱싱되는 딕셔너리였다 — dungeon 기준 key "3"(=unified
interactive+1)이 dungeon_interactable.png인 것과 key "4"(=unified hazard+1)가
bat.png인 게 정확히 맞아떨어진다. use_tile_mapping=False로 raw id를 그대로
넣었던 이전 버전은 애초에 스킴이 다른 값을 렌더러에 넣은 것이었다.

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
    from instruct_rl.utils.dataset_loader_helpers.preprocessing import apply_tile_offset, preprocess_samples

    print("building MultiGameDataset (use_tile_mapping=True, matching render_assets.py) ...")
    ds = MultiGameDataset(use_tile_mapping=True)
    samples = list(ds)
    samples = preprocess_samples(samples, longtail_cut=True)
    samples = apply_tile_offset(samples, 1)
    samples = [s for s in samples if s.meta.get("reward_enum") == args.reward_enum]
    print(f"reward_enum={args.reward_enum} samples: {len(samples)}")

    from dataset.multigame.render import GameLevelRenderer

    renderer = GameLevelRenderer()

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
        renderer.render(
            game=sample.game, level=sample.array, tile_size=args.tile_size, show_tile_numbers=False,
        ).save(str(out_path))
        captions[key] = sample.instruction or ""
        condition_value = sample.meta.get("conditions", {}).get(args.reward_enum)
        print(f"saved: {out_path}  (pick={pick}/{len(cands)}, cond={condition_value})  inst={sample.instruction!r}")

    captions_path = out_dir / "captions.json"
    captions_path.write_text(json.dumps(captions, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved captions: {captions_path}")


if __name__ == "__main__":
    main()

"""
debug/game_sample_stats.py
===========================
train 시 게임별 샘플 수 / dataset_seen_ratio 기준 사용량 / 잘린 수를 출력한다.

실행:
    python debug/game_sample_stats.py [--ratio 0.8]
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from collections import defaultdict

from conf.game_utils import ALL_GAMES, GAME_ABBR, parse_game_str
from dataset.multigame import MultiGameDataset
from instruct_rl.utils.dataset_loader_helpers.filters import _parse_dataset_reward_enum_filter
from instruct_rl.utils.dataset_loader_helpers.preprocessing import preprocess_samples, apply_tile_offset


def run(dataset_game="all", dataset_reward_enum=None, ratio=1.0,
        max_samples_per_game=1000, longtail_cut=True, rl_tile_offset=1):

    load_game = dataset_game or "all"

    if load_game == "all":
        game_names = ALL_GAMES
    elif load_game in GAME_ABBR:
        game_names = GAME_ABBR[load_game]
    elif len(load_game) % 2 == 0 and all(load_game[i:i+2] in GAME_ABBR for i in range(0, len(load_game), 2)):
        includes = parse_game_str(load_game)
        game_names = [n for n in ALL_GAMES if includes.get(f"include_{n}", False)]
    else:
        game_names = [load_game]

    ds = MultiGameDataset(
        include_dungeon=("dungeon" in game_names),
        include_pokemon=("pokemon" in game_names),
        include_sokoban=("sokoban" in game_names),
        include_doom=("doom" in game_names),
        include_doom2=("doom2" in game_names),
        include_zelda=("zelda" in game_names),
        use_tile_mapping=True,
        max_samples_per_game=max_samples_per_game,
    )

    samples = list(ds) if load_game == "all" else ds.by_games(game_names)
    samples = preprocess_samples(samples, longtail_cut=longtail_cut)
    samples = apply_tile_offset(samples, rl_tile_offset)

    re_filter = _parse_dataset_reward_enum_filter(dataset_reward_enum, field_name="dataset_reward_enum")
    if re_filter is not None:
        re_set = set(re_filter)
        samples = [s for s in samples if s.meta.get("reward_enum") in re_set]

    samples = [s for s in samples if "reward_enum" in s.meta and "conditions" in s.meta]

    by_game = defaultdict(int)
    for s in samples:
        by_game[s.game] += 1

    print()
    print(f"  설정: dataset_game={load_game}, dataset_reward_enum={dataset_reward_enum}, "
          f"dataset_seen_ratio={ratio}, max_samples_per_game={max_samples_per_game}")
    print()
    print(f"  {'game':<10} {'원본':>7} {'사용':>7} {'잘림':>7}  {'사용%':>7}  {'잘림%':>7}")
    print("  " + "-" * 54)

    total_orig = total_use = total_cut = 0
    for game in sorted(by_game):
        n_orig = by_game[game]
        n_use  = max(1, int(n_orig * ratio)) if ratio < 1.0 else n_orig
        n_cut  = n_orig - n_use
        used_pct = 100.0 * n_use / n_orig if n_orig else 0.0
        cut_pct  = 100.0 - used_pct if n_orig else 0.0
        total_orig += n_orig; total_use += n_use; total_cut += n_cut
        print(f"  {game:<10} {n_orig:>7,} {n_use:>7,} {n_cut:>7,}  {used_pct:>6.1f}%  {cut_pct:>6.1f}%")

    print("  " + "-" * 54)
    t_used_pct = 100.0 * total_use / total_orig if total_orig else 0.0
    t_cut_pct  = 100.0 - t_used_pct if total_orig else 0.0
    print(f"  {'합계':<10} {total_orig:>7,} {total_use:>7,} {total_cut:>7,}  {t_used_pct:>6.1f}%  {t_cut_pct:>6.1f}%")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default="all")
    parser.add_argument("--reward_enum", default=None)
    parser.add_argument("--ratio", type=float, default=1.0)
    parser.add_argument("--max_samples_per_game", type=int, default=1000)
    args = parser.parse_args()

    run(
        dataset_game=args.game,
        dataset_reward_enum=args.reward_enum,
        ratio=args.ratio,
        max_samples_per_game=args.max_samples_per_game,
    )


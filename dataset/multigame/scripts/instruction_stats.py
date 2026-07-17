"""
gametext·reward_enumtext instruction_uni text also  text.

Usage:
    python -m dataset.multigame.scripts.instruction_stats
    python -m dataset.multigame.scripts.instruction_stats --top-n 20
    python -m dataset.multigame.scripts.instruction_stats --top-n -1              # text
    python -m dataset.multigame.scripts.instruction_stats --no-save               # text
    python -m dataset.multigame.scripts.instruction_stats --out-dir /tmp/stats    # save abovetext text
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

REWARD_ENUM_LABELS = {
    0: "region",
    1: "path_length",
    2: "interactable",
    3: "hazard",
    4: "collectable",
}

DEFAULT_CACHE_DIR = Path(__file__).parents[1] / "cache"


def load_ann_files(cache_dir: Path) -> dict[str, list[dict]]:
    """game → annotation text return."""
    result = {}
    for ann_path in sorted(cache_dir.rglob("*.ann.json")):
        game = ann_path.parent.name
        with ann_path.open() as f:
            data = json.load(f)
        result.setdefault(game, []).extend(data["annotations"])
    return result


def compute_stats(
    annotations_by_game: dict[str, list[dict]],
) -> dict[tuple[str, int], Counter]:
    """(game, reward_enum) → Counter[instruction_uni]."""
    stats: dict[tuple[str, int], Counter] = {}
    for game, anns in annotations_by_game.items():
        for ann in anns:
            key = (game, int(ann["reward_enum"]))
            stats.setdefault(key, Counter())
            inst = ann.get("instruction_uni") or ""
            stats[key][inst] += 1
    return stats


def print_table(
    stats: dict[tuple[str, int], Counter],
    top_n: int = 10,
) -> None:
    unlimited = top_n <= 0
    games = sorted({g for g, _ in stats})
    enums = sorted({e for _, e in stats})

    for game in games:
        print(f"\n{'='*70}")
        print(f"  Game: {game}")
        print(f"{'='*70}")
        for enum in enums:
            key = (game, enum)
            if key not in stats:
                continue
            counter = stats[key]
            total = sum(counter.values())
            n_unique = len(counter)
            label = REWARD_ENUM_LABELS.get(enum, str(enum))
            print(f"\n  [enum={enum}  {label}]  total={total:,}  unique={n_unique:,}")

            if n_unique == 0:
                print("    (no instructions)")
                continue

            top = counter.most_common(None if unlimited else top_n)
            others = total - sum(cnt for _, cnt in top)

            # text text compute
            max_inst_len = max(len(inst) for inst, _ in top)
            max_inst_len = max(max_inst_len, 11)  # "Instruction" text minimum
            max_inst_len = min(max_inst_len, 80)

            header = f"  {'Instruction':<{max_inst_len}}  {'Count':>7}  {'%':>6}"
            print(f"  {'-'*len(header)}")
            print(header)
            print(f"  {'-'*len(header)}")

            for rank, (inst, cnt) in enumerate(top, 1):
                display = inst[:max_inst_len] if inst else "(empty)"
                pct = cnt / total * 100
                print(f"  {display:<{max_inst_len}}  {cnt:>7,}  {pct:>5.1f}%")

            if others > 0:
                pct = others / total * 100
                remainder = f"... +{n_unique - top_n} more"
                print(f"  {remainder:<{max_inst_len}}  {others:>7,}  {pct:>5.1f}%")

            print(f"  {'-'*len(header)}")


DEFAULT_OUT_DIR = DEFAULT_CACHE_DIR / "statics"

FIELDS = ["rank", "instruction", "count", "pct", "total", "n_unique"]


def save_csv_split(
    stats: dict[tuple[str, int], Counter],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for (game, enum), counter in sorted(stats.items()):
        label = REWARD_ENUM_LABELS.get(enum, str(enum))
        game_dir = out_dir / game
        game_dir.mkdir(exist_ok=True)
        out_path = game_dir / f"enum_{enum}_{label}.csv"
        total = sum(counter.values())
        rows = [
            {
                "rank": rank,
                "instruction": inst,
                "count": cnt,
                "pct": round(cnt / total * 100, 4) if total else 0,
                "total": total,
                "n_unique": len(counter),
            }
            for rank, (inst, cnt) in enumerate(counter.most_common(), 1)
        ]
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        saved.append((out_path.resolve(), len(rows)))

    print(f"\nSaved {len(saved)} CSV(s) → {out_dir.resolve()}/")
    for path, n in saved:
        print(f"  {path.relative_to(out_dir.resolve().parent)}  ({n:,} rows)")


def print_summary(stats: dict[tuple[str, int], Counter]) -> None:
    games = sorted({g for g, _ in stats})
    enums = sorted({e for _, e in stats})

    col = 12
    header = f"{'game':<12}" + "".join(
        f"{REWARD_ENUM_LABELS.get(e, str(e)):>{col}}" for e in enums
    )
    print(f"\n{'='*70}")
    print("  Summary: unique instruction count per (game, reward_enum)")
    print(f"{'='*70}")
    print(f"  {header}")
    print(f"  {'-'*len(header)}")
    for game in games:
        row = f"{game:<12}"
        for e in enums:
            n = len(stats.get((game, e), {}))
            row += f"{n:>{col},}"
        print(f"  {row}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Instruction_uni text also  text")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="cache directory (default: dataset/multigame/cache)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="game×enum text textabove Ntext text (default: 10, -1=text)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        metavar="DIR",
        help="CSV save text directory (default: dataset/multigame/cache/statics)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="CSV save text",
    )
    parser.add_argument(
        "--games",
        nargs="*",
        default=None,
        help="text gametext text (text: --games dungeon zelda)",
    )
    parser.add_argument(
        "--enums",
        nargs="*",
        type=int,
        default=None,
        help="text reward_enumtext text (text: --enums 0 1)",
    )
    args = parser.parse_args()

    print(f"Loading .ann.json from: {args.cache_dir}")
    annotations_by_game = load_ann_files(args.cache_dir)

    if args.games:
        annotations_by_game = {
            g: v for g, v in annotations_by_game.items() if g in args.games
        }

    stats = compute_stats(annotations_by_game)

    if args.enums:
        stats = {(g, e): v for (g, e), v in stats.items() if e in args.enums}

    print_summary(stats)
    print_table(stats, top_n=args.top_n)

    if not args.no_save:
        save_csv_split(stats, args.out_dir)


if __name__ == "__main__":
    main()

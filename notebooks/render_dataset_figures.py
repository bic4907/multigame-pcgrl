"""
notebooks/render_dataset_figures.py

Renders dataset example thumbnails for the paper figure: for every
(game, reward_enum) pair it saves 3 PNGs, each from a different level and — where
the data allows — with a different condition value.

Data loading follows notebooks/render_level_thumbnails.py (and
results/render/table_export/render_assets.py) exactly, so the sprites match what
the rest of the pipeline produces:

    MultiGameDataset(use_tile_mapping=True)   # unified 5-category, not raw ids
    -> preprocess_samples(longtail_cut=True)
    -> apply_tile_offset(samples, 1)          # unified(0-4) -> "_tile_images" keys (1-5)

Rendering uses the per-game sprite renderer (dataset/multigame/render.py:
GameLevelRenderer) rather than the unified renderer in envs/probs/multigame.py,
because the figure needs the games to be visually distinguishable.

Selection
---------
Every level carries all five reward_enums, so naively taking the first candidate
per cell yields the same map five times over. Two constraints avoid that:

  * condition spread — the 3 picks of a cell take distinct condition values,
    sampled at even quantiles of the values actually present in that cell.
  * level uniqueness — a level (``source_id``) is used at most once per game, so
    no map is repeated across reward_enums either.

Both fall back gracefully: if a cell has fewer than 3 distinct condition values,
or a game runs short of unused levels, the constraint is relaxed and a warning is
printed rather than dropping the cell.

Ordering is deterministic throughout, so reruns reproduce the same figure.

By default one image is rendered per cell, chosen by the CELL_PICKS table near the
top of this file; ``--pick game:enum=idx`` overrides an entry without editing it.
Asking for more than one image per cell switches back to spreading the picks over
distinct condition values.

Usage:
    python notebooks/render_dataset_figures.py
    python notebooks/render_dataset_figures.py --pick pokemon:4=3 zelda:0=1
    python notebooks/render_dataset_figures.py --games dungeon zelda --enums 0 1
    python notebooks/render_dataset_figures.py --n-per-cell 3 --tile-size 24
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ALL_GAMES = ["dungeon", "pokemon", "sokoban", "doom", "zelda"]
ALL_ENUMS = [0, 1, 2, 3, 4]

ENUM_LABELS = {
    0: "region",
    1: "path_length",
    2: "interactable_count",
    3: "hazard_count",
    4: "collectable_count",
}

# Tile value (after the +1 offset) whose count each count-style enum measures.
COUNT_ENUM_TILE = {2: 3, 3: 4, 4: 5}

# Quantized intensity level (bin) to use for each (game, reward_enum) cell.
#
# The bin is the same quantity encoder/data/clip_batch.py uses: the raw condition
# value passed through np.digitize against CUSTOM_THRESHOLDS, giving 0-7 (every
# feature defines 7 thresholds). Bin 0 is the sparsest level, 7 the densest.
#
# --list-bins prints which bins each cell actually contains, and
# --pick game:enum=bin overrides an entry without editing this table.
# (dungeon, 2), (sokoban, 3) and (sokoban, 4) are absent from the dataset.
CELL_BINS: dict[tuple[str, int], int] = {
    ("dungeon", 0): 6,  ("dungeon", 1): 5,  ("dungeon", 3): 1,  ("dungeon", 4): 7,
    ("pokemon", 0): 3,  ("pokemon", 1): 0,  ("pokemon", 2): 7, ("pokemon", 3): 5,  ("pokemon", 4): 1,
    ("doom",    0): 7,  ("doom",    1): 4,  ("doom",    2): 2, ("doom",    3): 4,  ("doom",    4): 6,
    ("zelda",   0): 0,  ("zelda",   1): 7,  ("zelda",   2): 5, ("zelda",   3): 1,  ("zelda",   4): 2,
    ("sokoban", 0): 5,  ("sokoban", 1): 1,  ("sokoban", 2): 3,
}


def condition_value_of(sample, reward_enum: int):
    conditions = sample.meta.get("conditions", {})
    return conditions.get(reward_enum, next(iter(conditions.values()), None))


def quantized_bin_of(sample, reward_enum: int) -> int:
    """Intensity bin (0-7) of a sample, matching encoder/data/clip_batch.py.

    The raw condition value is digitized against CUSTOM_THRESHOLDS, which defines
    7 thresholds per feature.
    """
    from dataset.reward_annotations.instruction_config import CUSTOM_THRESHOLDS

    feature_name = sample.meta.get("feature_name", "")
    condition_value = condition_value_of(sample, reward_enum)
    thresholds = CUSTOM_THRESHOLDS.get(f"{sample.game}_{feature_name}")
    if thresholds is not None and condition_value is not None:
        return int(np.digitize(condition_value, thresholds))
    return 0


def level_key(sample) -> str:
    """Identity of a level for de-duplication.

    Pokemon stores maps that differ in raw tile ids but collapse to the same
    unified array, so source_id alone lets a visually identical map through.
    Hashing the array itself is what actually prevents duplicate renders.
    """
    return hashlib.md5(np.ascontiguousarray(sample.array)).hexdigest()


def dominance(sample) -> float:
    """Fraction of the map covered by its most common tile.

    A high value means a visually flat level — e.g. a Pokemon map that is 91%
    grass — which reads poorly in a figure even though its condition value is
    correct. Used to prefer varied maps during selection.
    """
    _, counts = np.unique(sample.array, return_counts=True)
    return float(counts.max()) / sample.array.size


def spread_indices(n: int, k: int) -> list[int]:
    """k evenly spread indices over range(n), biased away from the extremes.

    The lowest condition bucket is often an empty or degenerate map, so the
    quantiles start slightly inside the range rather than at 0.
    """
    if n <= k:
        return list(range(n))
    qs = np.linspace(0.15, 0.85, k)
    idx = sorted({int(round(q * (n - 1))) for q in qs})
    # Rounding can collapse two quantiles onto the same index; fill from the end.
    i = 0
    while len(idx) < k and i < n:
        if i not in idx:
            idx.append(i)
        i += 1
    return sorted(idx)[:k]


def pick_every_bin(samples, game: str, reward_enum: int, used_sids: set,
                   select: str = "varied",
                   rng: "np.random.Generator | None" = None):
    """One sample per intensity bin present in this (game, reward_enum) cell.

    Returns [(bin, sample), ...] ordered by bin. Bins with no data are simply
    absent — several features leave gaps between thresholds.
    """
    cands = [
        s for s in samples
        if s.game == game and s.meta.get("reward_enum") == reward_enum
    ]
    if not cands:
        return []

    by_bin: dict[int, list] = defaultdict(list)
    for s in cands:
        by_bin[quantized_bin_of(s, reward_enum)].append(s)

    out = []
    for b in sorted(by_bin):
        pool = [s for s in by_bin[b] if level_key(s) not in used_sids] or by_bin[b]
        pool = sorted(pool, key=lambda s: (str(s.meta.get("key", "")), str(s.source_id)))
        chosen = (pool[int(rng.integers(len(pool)))]
                  if select == "random" and rng is not None
                  else min(pool, key=dominance))
        used_sids.add(level_key(chosen))
        out.append((b, chosen))
    return out


def pick_for_cell(samples, game: str, reward_enum: int, n: int, used_sids: set,
                  overrides: dict[str, int] | None = None,
                  max_dominance: float = 0.8,
                  rng: "np.random.Generator | None" = None,
                  select: str = "varied"):
    """Pick n samples for one (game, reward_enum) cell.

    With n == 1 the condition is taken from CELL_CONDITIONS and the level is chosen
    among the maps carrying it: ``select="varied"`` takes the least flat one,
    ``select="random"`` draws with the seeded generator.
    With n > 1 the picks are spread over distinct condition values instead.
    ``overrides`` maps a slot name (e.g. "pokemon_re4_2") to an index into that
    slot's candidate pool, so a visually poor default can be swapped out by hand.
    Returns (picks, notes) where notes lists any relaxed constraint.
    """
    overrides = overrides or {}
    cands = [
        s for s in samples
        if s.game == game and s.meta.get("reward_enum") == reward_enum
    ]
    if not cands:
        return [], ["no candidate"]

    by_cond: dict[float, list] = defaultdict(list)
    for s in cands:
        cond = condition_value_of(s, reward_enum)
        if cond is None:
            continue
        by_cond[float(cond)].append(s)
    # Within a condition bucket, prefer the most varied map; ties broken
    # deterministically so reruns are stable.
    for v in by_cond.values():
        v.sort(key=lambda s: (dominance(s), str(s.meta.get("key", "")), str(s.source_id)))

    conds = sorted(by_cond)
    # Drop the all-zero bucket when richer maps exist; it is usually an empty map.
    nonzero = [c for c in conds if c != 0.0]
    if nonzero:
        conds = nonzero

    notes: list[str] = []

    if n == 1:
        # Single-panel mode: the caller names an intensity bin, and one of the
        # maps falling in that bin is rendered.
        by_bin: dict[int, list] = defaultdict(list)
        for s in cands:
            by_bin[quantized_bin_of(s, reward_enum)].append(s)

        want = overrides.get((game, reward_enum), CELL_BINS.get((game, reward_enum)))
        available = sorted(by_bin)
        if want is None:
            want = available[len(available) // 2]
            notes.append(f"no bin configured; using {want}")
        elif int(want) not in by_bin:
            nearest = min(available, key=lambda b: (abs(b - int(want)), b))
            notes.append(f"bin={int(want)} NOT PRESENT (have {available}) -> using {nearest}")
            want = nearest

        pool = [s for s in by_bin[int(want)] if level_key(s) not in used_sids]
        if not pool:
            pool = by_bin[int(want)]
            notes.append(f"level reused in bin={int(want)}")

        # Sort first so the pool order never depends on dataset iteration order.
        pool = sorted(pool, key=lambda s: (str(s.meta.get("key", "")), str(s.source_id)))
        if select == "random" and rng is not None:
            chosen = pool[int(rng.integers(len(pool)))]
        else:
            chosen = min(pool, key=dominance)
        used_sids.add(level_key(chosen))
        notes.append(f"bin={int(want)} n={len(by_bin[int(want)])}")
        return [chosen], notes

    # Drop condition values whose best map is still visually flat, as long as
    # enough alternatives remain to keep the spread.
    varied = [c for c in conds if dominance(by_cond[c][0]) <= max_dominance]
    if len(varied) >= n:
        dropped = len(conds) - len(varied)
        if dropped:
            notes.append(f"skipped {dropped} flat condition(s)")
        conds = varied

    chosen_conds = [conds[i] for i in spread_indices(len(conds), n)]
    if len(chosen_conds) < n:
        notes.append(f"only {len(chosen_conds)} distinct condition(s)")

    picks: list = []
    for slot, cond in enumerate(chosen_conds):
        pool = [s for s in by_cond[cond] if level_key(s) not in used_sids]
        if not pool:
            pool = by_cond[cond]
            notes.append(f"level reused at cond={cond:g}")

        idx = 0
        if idx >= len(pool):
            notes.append(f"slot {slot}: pick {idx} out of range ({len(pool)}), using 0")
            idx = 0
        elif idx:
            notes.append(f"slot {slot}: pick={idx}")

        chosen = pool[idx]
        used_sids.add(level_key(chosen))
        picks.append(chosen)

    # Still short (fewer distinct conditions than requested): top up with unused
    # levels from any condition bucket.
    if len(picks) < n:
        rest = sorted(cands, key=lambda s: (str(s.meta.get("key", "")), str(s.source_id)))
        for s in rest:
            if len(picks) >= n:
                break
            if level_key(s) in used_sids:
                continue
            used_sids.add(level_key(s))
            picks.append(s)

    return picks, notes


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render per-(game, reward_enum) dataset example thumbnails"
    )
    parser.add_argument("--games", nargs="+", default=ALL_GAMES)
    parser.add_argument("--enums", nargs="+", type=int, default=ALL_ENUMS)
    parser.add_argument("--n-per-cell", type=int, default=1,
                        help="images per (game, reward_enum) cell")
    parser.add_argument("--tile-size", type=int, default=16)
    parser.add_argument("--out-dir", default=str(ROOT / "fig_dataset"))
    parser.add_argument("--flat", action="store_true",
                        help="write every PNG into one directory instead of per-game subfolders")
    parser.add_argument("--pick", nargs="+", default=[], metavar="GAME:ENUM=BIN",
                        help="override CELL_BINS for a cell, e.g. pokemon:4=3")
    parser.add_argument("--select", choices=["varied", "random"], default="varied",
                        help="among maps with the configured condition, take the "
                             "least flat one (varied) or draw one at random")
    parser.add_argument("--seed", type=int, default=0,
                        help="seed for --select random")
    parser.add_argument("--all-bins", action="store_true",
                        help="render one image per available bin instead of the "
                             "single bin configured in CELL_BINS")
    parser.add_argument("--list-bins", action="store_true",
                        help="print the intensity bins available per cell and exit")
    parser.add_argument("--instruction-field", choices=["raw", "uni"], default="raw",
                        help="which instruction text to record in the manifest "
                             "(raw = game-specific tile names, as the training configs use)")
    parser.add_argument("--max-dominance", type=float, default=0.8,
                        help="skip condition values whose best map is this flat "
                             "(fraction covered by one tile); 1.0 disables the filter")
    args = parser.parse_args()

    overrides: dict[tuple[str, int], int] = {}
    for item in args.pick:
        cell, value = item.split("=")
        game_str, enum_str = cell.split(":")
        overrides[(game_str.strip(), int(enum_str))] = int(value)

    from dataset.multigame import MultiGameDataset
    from dataset.multigame.render import GameLevelRenderer
    from instruct_rl.utils.dataset_loader_helpers.preprocessing import (
        apply_tile_offset,
        preprocess_samples,
    )

    # instruction_field="raw" matches the training configs (conf/config.py); the
    # MultiGameDataset default is "uni", which would caption the figure with the
    # unified category wording instead of the game's own tile names.
    print(f"building MultiGameDataset (use_tile_mapping=True, "
          f"instruction_field={args.instruction_field!r}) ...")
    ds = MultiGameDataset(
        use_tile_mapping=True, instruction_field=args.instruction_field
    )
    samples = list(ds)
    samples = preprocess_samples(samples, longtail_cut=True)
    samples = apply_tile_offset(samples, 1)
    print(f"total samples: {len(samples)}")

    if args.list_bins:
        for game in args.games:
            for reward_enum in args.enums:
                cands = [s for s in samples
                         if s.game == game and s.meta.get("reward_enum") == reward_enum]
                if not cands:
                    continue
                counts: dict[int, int] = defaultdict(int)
                ranges: dict[int, list] = defaultdict(list)
                for s in cands:
                    b = quantized_bin_of(s, reward_enum)
                    counts[b] += 1
                    c = condition_value_of(s, reward_enum)
                    if c is not None:
                        ranges[b].append(float(c))
                want = CELL_BINS.get((game, reward_enum))
                mark = "" if want is None or int(want) in counts else "   <-- configured bin MISSING"
                shown = ", ".join(
                    f"bin{b}: n={counts[b]} cond={min(ranges[b]):g}-{max(ranges[b]):g}"
                    for b in sorted(counts)
                )
                print(f"\n{game} enum={reward_enum} ({ENUM_LABELS.get(reward_enum,'')}) "
                      f"configured bin={want}{mark}\n  {shown}")
        return

    rng = np.random.default_rng(args.seed)
    renderer = GameLevelRenderer()
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, dict] = {}
    n_saved, n_missing = 0, 0

    for game in args.games:
        used_sids: set = set()          # array hashes already used by this game
        for reward_enum in args.enums:
            if args.all_bins:
                pairs = pick_every_bin(samples, game, reward_enum, used_sids,
                                       args.select, rng)
                picks = [s for _, s in pairs]
                bins = [b for b, _ in pairs]
                notes = [f"bins={bins}"] if bins else []
            else:
                picks, notes = pick_for_cell(
                    samples, game, reward_enum, args.n_per_cell, used_sids,
                    overrides, args.max_dominance, rng, args.select
                )
                bins = None
            if not picks:
                print(f"[skip] {game} enum={reward_enum}: no sample "
                      f"(feature undefined for this game)")
                n_missing += 1
                continue

            out_dir = out_root if args.flat else out_root / game
            out_dir.mkdir(parents=True, exist_ok=True)

            conds = []
            for i, sample in enumerate(picks):
                # The per-game subfolder already carries the domain, so the file
                # name omits it; --flat needs the prefix to stay unique.
                if bins is not None:
                    suffix = f"_bin{bins[i]}"
                else:
                    suffix = "" if args.n_per_cell == 1 else f"_{i}"
                name = f"{game}_re{reward_enum}{suffix}"
                stem = name if args.flat else f"re{reward_enum}{suffix}"
                out_path = out_dir / f"{stem}.png"
                renderer.render(
                    game=sample.game,
                    level=sample.array,
                    tile_size=args.tile_size,
                    show_tile_numbers=False,
                ).save(str(out_path))

                cond = condition_value_of(sample, reward_enum)
                conds.append(cond)

                entry = {
                    "game": game,
                    "reward_enum": reward_enum,
                    "feature_name": ENUM_LABELS.get(reward_enum, ""),
                    "source_id": str(sample.source_id),
                    "condition": cond,
                    "bin": quantized_bin_of(sample, reward_enum),
                    "instruction": sample.instruction or "",
                    "path": str(out_path.relative_to(out_root)),
                }
                # Self-check: for count-style features the condition must equal
                # the number of matching tiles actually drawn.
                tile = COUNT_ENUM_TILE.get(reward_enum)
                if tile is not None and cond is not None:
                    actual = int((sample.array == tile).sum())
                    entry["tile_count_check"] = actual
                    if actual != int(cond):
                        print(f"  [MISMATCH] {name}: condition={cond} but "
                              f"{actual} tiles of value {tile}")
                manifest[name] = entry
                n_saved += 1

            note = ("  [" + "; ".join(notes) + "]") if notes else ""
            print(f"saved: {game:8s} enum={reward_enum} "
                  f"({ENUM_LABELS.get(reward_enum,'')})  conds={conds}{note}")

    # ── Duplicate audit over the rendered PNGs ───────────────────────────────
    digests: dict[str, list[str]] = defaultdict(list)
    for name, m in manifest.items():
        data = (out_root / m["path"]).read_bytes()
        digests[hashlib.md5(data).hexdigest()].append(name)
    dups = {k: v for k, v in digests.items() if len(v) > 1}

    manifest_path = out_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\n{n_saved} image(s) saved under {out_root}")
    print(f"unique images: {len(digests)} / {n_saved}")
    if dups:
        print("duplicate renders:")
        for names in dups.values():
            print("  " + " = ".join(names))
    if n_missing:
        print(f"{n_missing} (game, enum) cell(s) had no sample")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()

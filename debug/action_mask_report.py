"""
Print which Multigame actions are masked by build_action_allowed_mask(env).

Examples
--------
    python debug/action_mask_report.py
    python debug/action_mask_report.py --representations narrow turtle wide
    python debug/action_mask_report.py --map-shape 4 4 --show allowed
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.probs.multigame import make_multigame_env
from instruct_rl.utils.action_mask import build_action_allowed_mask


MOVE_NAMES = ("UP", "RIGHT", "DOWN", "LEFT")


@dataclass(frozen=True)
class ActionRow:
    index: int
    allowed: bool
    kind: str
    detail: str
    tile_id: int | None
    tile_name: str | None


def _tile_name_by_id(env) -> dict[int, str]:
    return {int(tile): tile.name for tile in env.prob.tile_enum}


def _editable_tiles(env) -> list[tuple[int, str]]:
    name_by_id = _tile_name_by_id(env)
    return [(int(tile), name_by_id[int(tile)]) for tile in env.rep.editable_tile_enum]


def _describe_tile_action(env, action_idx: int, detail: str = "") -> ActionRow:
    editable = _editable_tiles(env)
    tile_id, tile_name = editable[action_idx % len(editable)]
    suffix = f", {detail}" if detail else ""
    return ActionRow(
        index=action_idx,
        allowed=False,
        kind="build",
        detail=f"set tile to {tile_name}{suffix}",
        tile_id=tile_id,
        tile_name=tile_name,
    )


def _describe_narrow(env, action_idx: int) -> ActionRow:
    return _describe_tile_action(env, action_idx)


def _describe_turtle(env, action_idx: int) -> ActionRow:
    n_tiles = len(_editable_tiles(env))
    if action_idx < n_tiles:
        return _describe_tile_action(env, action_idx, detail="at turtle position")
    move_idx = action_idx - n_tiles
    move_name = MOVE_NAMES[move_idx] if move_idx < len(MOVE_NAMES) else f"MOVE_{move_idx}"
    return ActionRow(
        index=action_idx,
        allowed=False,
        kind="move",
        detail=f"move turtle {move_name}",
        tile_id=None,
        tile_name=None,
    )


def _describe_wide(env, action_idx: int) -> ActionRow:
    h, w = env.map_shape
    editable = _editable_tiles(env)
    n_tiles = len(editable)
    step_product = h * w * n_tiles

    cell_idx, tile_idx = divmod(action_idx, n_tiles)
    tile_id, tile_name = editable[tile_idx]
    if action_idx < step_product:
        x, y = divmod(cell_idx, w)
        target = f"(x={x}, y={y})"
    else:
        target = f"out-of-range action for wide.step product={step_product}"
    return ActionRow(
        index=action_idx,
        allowed=False,
        kind="build",
        detail=f"set {target} to {tile_name}",
        tile_id=tile_id,
        tile_name=tile_name,
    )


def _describe_nca(env, action_idx: int) -> ActionRow:
    return _describe_tile_action(env, action_idx, detail="per-cell NCA channel")


DESCRIBERS = {
    "narrow": _describe_narrow,
    "turtle": _describe_turtle,
    "wide": _describe_wide,
    "nca": _describe_nca,
}


def _iter_rows(env, representation: str) -> list[ActionRow]:
    allowed_mask = np.asarray(build_action_allowed_mask(env), dtype=bool)
    describe = DESCRIBERS.get(representation, _describe_narrow)

    rows: list[ActionRow] = []
    for idx, is_allowed in enumerate(allowed_mask.tolist()):
        row = describe(env, idx)
        rows.append(
            ActionRow(
                index=row.index,
                allowed=bool(is_allowed),
                kind=row.kind,
                detail=row.detail,
                tile_id=row.tile_id,
                tile_name=row.tile_name,
            )
        )
    return rows


def _format_row(row: ActionRow) -> str:
    status = "ALLOW" if row.allowed else "MASK "
    tile = "-" if row.tile_name is None else f"{row.tile_name}({row.tile_id})"
    return f"{row.index:5d}  {status}  {row.kind:5s}  {tile:10s}  {row.detail}"


def _print_summary(
    representation: str,
    env,
    all_rows: list[ActionRow],
    display_rows: list[ActionRow],
    limit: int | None,
) -> None:
    allowed = [row for row in all_rows if row.allowed]
    masked = [row for row in all_rows if not row.allowed]
    by_tile = Counter(row.tile_name or row.kind for row in all_rows)
    masked_by_tile = Counter(row.tile_name or row.kind for row in masked)

    print(f"\n[{representation}]")
    print(f"action_space.n : {env.rep.action_space.n}")
    print(f"rep.builds     : {getattr(env.rep, 'builds', None)}")
    print(f"allowed        : {len(allowed)}")
    print(f"masked         : {len(masked)}")
    print(f"masked by type : {dict(masked_by_tile)}")
    print(f"all by type    : {dict(by_tile)}")
    if representation == "wide":
        h, w = env.map_shape
        n_tiles = len(_editable_tiles(env))
        step_product = h * w * n_tiles
        if env.rep.action_space.n != step_product:
            print(
                "note           : wide.step unravels actions as "
                f"(H, W, editable_tiles)={step_product}, "
                f"but action_space.n={env.rep.action_space.n}"
            )

    if limit is not None and len(display_rows) > limit:
        print(f"rows           : first {limit} of {len(display_rows)}")
        display_rows = display_rows[:limit]

    print("index  mask   kind   tile        meaning")
    for row in display_rows:
        print(_format_row(row))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--representations",
        nargs="+",
        default=["narrow", "turtle", "wide", "nca"],
        choices=sorted(DESCRIBERS),
    )
    parser.add_argument("--map-shape", nargs=2, type=int, default=(4, 4), metavar=("H", "W"))
    parser.add_argument("--act-shape", nargs=2, type=int, default=(1, 1), metavar=("H", "W"))
    parser.add_argument(
        "--show",
        choices=("all", "allowed", "masked"),
        default="all",
        help="Filter action rows after computing the mask.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=80,
        help="Maximum rows to print per representation. Use -1 for no limit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    limit = None if args.limit < 0 else args.limit
    map_shape = tuple(args.map_shape)
    act_shape = tuple(args.act_shape)

    print("Multigame action mask report")
    print("Mask values are read from instruct_rl.utils.action_mask.build_action_allowed_mask(env).")
    print(f"map_shape={map_shape}, act_shape={act_shape}")

    for representation in args.representations:
        try:
            env, _ = make_multigame_env(
                representation=representation,
                map_shape=map_shape,
                act_shape=act_shape,
            )
            rows = _iter_rows(env, representation)
        except Exception as exc:
            print(f"\n[{representation}] failed to instantiate/check: {type(exc).__name__}: {exc}")
            continue

        display_rows = rows
        if args.show == "allowed":
            display_rows = [row for row in rows if row.allowed]
        elif args.show == "masked":
            display_rows = [row for row in rows if not row.allowed]

        _print_summary(representation, env, rows, display_rows, limit)


if __name__ == "__main__":
    main()

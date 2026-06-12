from __future__ import annotations

from bisect import bisect_right
import csv
import json
import math
from pathlib import Path

from .models import RunResult
from .utils import markdown_escape, reward_enum_value, task_name


CONDITION_BUCKETS = (
    ("low", "적을 때"),
    ("mid", "중간"),
    ("high", "많을 때"),
)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _numeric_text(value: str) -> str:
    if value in ("", None):
        return ""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if numeric.is_integer():
        return str(int(numeric))
    return f"{numeric:g}"


def _percentile_text(value: float) -> str:
    if math.isclose(value, round(value)):
        return f"{int(round(value))}%"
    return f"{value:.1f}%"


def _row_identity(row: dict[str, str]) -> tuple[str, str, str]:
    return (
        row.get("row_i", ""),
        row.get("game", ""),
        row.get("reward_enum", ""),
    )


def condition_text(row: dict[str, str]) -> str:
    cond_items = []
    for key in sorted(k for k in row if k.startswith("condition_")):
        value = _numeric_text(row.get(key, ""))
        if value:
            cond_items.append(f"{key}={value}")
    return ", ".join(cond_items) if cond_items else "-"


def condition_value(row: dict[str, str]) -> float | None:
    reward_enum = reward_enum_value(row)
    value = row.get(f"condition_{reward_enum}")
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _latest_ann_files(cache_dir: Path) -> list[tuple[str, Path]]:
    if not cache_dir.exists():
        return []

    ann_files = []
    for game_dir in sorted(path for path in cache_dir.iterdir() if path.is_dir()):
        candidates = sorted(game_dir.glob("*.ann.json"), key=lambda path: path.stat().st_mtime)
        if candidates:
            ann_files.append((game_dir.name.lower(), candidates[-1]))
    return ann_files


def _annotation_condition_value(row: dict, reward_enum: int) -> float | None:
    value = row.get(f"condition_{reward_enum}")
    if value is None and 1 <= reward_enum <= 5:
        value = row.get(f"condition_{reward_enum - 1}")
    if value in (None, ""):
        return None
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value_f):
        return None
    return value_f


def _load_dataset_condition_distributions(
    cache_dir: Path,
) -> dict[tuple[str, int], list[float]]:
    distributions: dict[tuple[str, int], list[float]] = {}
    for fallback_game, ann_path in _latest_ann_files(cache_dir):
        payload = json.loads(ann_path.read_text(encoding="utf-8"))
        game = str(payload.get("game") or fallback_game).lower()
        for row in payload.get("annotations", []):
            try:
                reward_enum = int(row["reward_enum"])
            except (KeyError, TypeError, ValueError):
                continue
            value = _annotation_condition_value(row, reward_enum)
            if value is None:
                continue
            distributions.setdefault((game, reward_enum), []).append(value)

    for values in distributions.values():
        values.sort()
    return distributions


def _condition_percentile(row: dict, distributions: dict[tuple[str, int], list[float]]) -> float | None:
    value = condition_value(row)
    if value is None:
        return None
    values = distributions.get((str(row.get("game", "")).lower(), reward_enum_value(row)))
    if not values:
        return None
    return 100.0 * bisect_right(values, value) / len(values)


def annotate_condition_percentiles(
    rows: list[dict],
    cache_dir: Path = Path("dataset/multigame/cache/artifacts"),
) -> list[dict]:
    distributions = _load_dataset_condition_distributions(cache_dir)
    if not distributions:
        return rows

    for row in rows:
        members = row.get("_pair_members")
        if members:
            target_rows = members
        else:
            target_rows = [row]

        for target_row in target_rows:
            percentile = _condition_percentile(target_row, distributions)
            if percentile is None:
                continue
            target_row["_condition_percentile"] = f"{percentile:.6f}"
            target_row["_condition_percentile_text"] = _percentile_text(percentile)
    return rows


def condition_bucket_key(row: dict[str, str]) -> str:
    return row.get("_condition_bucket", "unknown")


def condition_bucket_label(row: dict[str, str]) -> str:
    if row.get("_condition_bucket_label"):
        return row["_condition_bucket_label"]
    key = condition_bucket_key(row)
    return dict(CONDITION_BUCKETS).get(key, "Condition unknown")


def _condition_bucket_from_value(value: float | None, sorted_values: list[float]) -> str:
    if value is None or not sorted_values:
        return "unknown"
    if len(sorted_values) == 1:
        return "mid"

    index = sorted_values.index(value)
    ratio = index / (len(sorted_values) - 1)
    if ratio < 1 / 3:
        return "low"
    if ratio < 2 / 3:
        return "mid"
    return "high"


def merge_conditions_from_ctrl_sim(
    rows: list[dict[str, str]],
    ctrl_sim_path: Path,
) -> list[dict[str, str]]:
    if not rows or not ctrl_sim_path.exists():
        return rows

    ctrl_by_identity = {}
    for ctrl_row in read_csv_rows(ctrl_sim_path):
        ctrl_by_identity.setdefault(_row_identity(ctrl_row), ctrl_row)

    merged = []
    for row in rows:
        out = dict(row)
        ctrl_row = ctrl_by_identity.get(_row_identity(row))
        if ctrl_row is not None:
            for key, value in ctrl_row.items():
                if key.startswith("condition_") and key not in out:
                    out[key] = value
        merged.append(out)
    return merged


def rows_from_run_csv(run_result: RunResult, max_rows: int) -> list[dict[str, str]]:
    if run_result.csv_dir is None:
        return []

    results_path = run_result.csv_dir / "results.csv"
    ctrl_sim_path = run_result.csv_dir / "ctrl_sim.csv"

    if results_path.exists():
        rows = read_csv_rows(results_path)[:max_rows]
        return merge_conditions_from_ctrl_sim(rows, ctrl_sim_path)

    if not ctrl_sim_path.exists():
        return []

    rows = []
    seen = set()
    for row in read_csv_rows(ctrl_sim_path):
        key = (row.get("row_i"), row.get("game"), row.get("reward_enum"))
        if key in seen:
            continue
        seen.add(key)
        rows.append(row)
        if len(rows) >= max_rows:
            break
    return rows


def reward_enum_game_rows_from_run_csv(
    run_result: RunResult,
    max_rows_per_reward_enum: int,
) -> list[dict[str, str]]:
    rows = rows_from_run_csv(run_result, max_rows=10_000)
    by_reward_enum_game = {}
    for row in rows:
        game = row.get("game")
        reward_enum = row.get("reward_enum")
        if not game or reward_enum is None:
            continue
        key = (reward_enum, game)
        if key in by_reward_enum_game:
            continue
        reward_enum_count = sum(1 for re, _ in by_reward_enum_game if re == reward_enum)
        if reward_enum_count >= max_rows_per_reward_enum:
            continue
        by_reward_enum_game[key] = row
    return list(by_reward_enum_game.values())


def reward_enum_condition_rows_from_run_csv(
    run_result: RunResult,
    max_rows_per_condition: int,
) -> list[dict[str, str]]:
    rows = rows_from_run_csv(run_result, max_rows=10_000)
    rows_by_reward_enum: dict[int, list[dict[str, str]]] = {}
    for row in rows:
        if not row.get("game") or row.get("reward_enum") in (None, ""):
            continue
        rows_by_reward_enum.setdefault(reward_enum_value(row), []).append(row)

    selected = []
    for reward_enum in sorted(rows_by_reward_enum):
        reward_rows = rows_by_reward_enum[reward_enum]
        sorted_values = sorted({v for row in reward_rows if (v := condition_value(row)) is not None})
        seen: set[tuple[str, str]] = set()
        bucket_counts = {bucket: 0 for bucket, _ in CONDITION_BUCKETS}

        for row in reward_rows:
            bucket = _condition_bucket_from_value(condition_value(row), sorted_values)
            if bucket not in bucket_counts:
                continue
            if bucket_counts[bucket] >= max_rows_per_condition:
                continue

            game = row.get("game", "")
            game_bucket_key = (bucket, game)
            if game_bucket_key in seen:
                continue

            out = dict(row)
            out["_condition_bucket"] = bucket
            selected.append(out)
            seen.add(game_bucket_key)
            bucket_counts[bucket] += 1

        for row in reward_rows:
            bucket = _condition_bucket_from_value(condition_value(row), sorted_values)
            if bucket not in bucket_counts:
                continue
            if bucket_counts[bucket] >= max_rows_per_condition:
                continue

            out = dict(row)
            out["_condition_bucket"] = bucket
            selected.append(out)
            bucket_counts[bucket] += 1

    return selected


def _condition_distance(row: dict[str, str], target_value: float) -> tuple[float, float]:
    value = condition_value(row)
    if value is None:
        return (float("inf"), float("inf"))
    return (abs(value - target_value), value)


def condition_contrast_rows_from_run_csv(
    run_result: RunResult,
    targets: list[dict],
) -> list[dict[str, str]]:
    """Select rows only for config-defined game/condition targets.

    Each config target contributes at most one row per listed game for the
    matching reward_enum. If the exact condition_value is absent in eval CSV,
    the nearest available condition row is used so the render table remains
    populated.
    """
    rows = rows_from_run_csv(run_result, max_rows=10_000)
    selected: list[dict[str, str]] = []
    seen: set[tuple[int, str, str]] = set()

    for fallback_i, target in enumerate(targets, start=1):
        target_i = int(target.get("_target_i", fallback_i))
        reward_enum = int(target["reward_enum"])
        target_value = float(target["condition_value"])
        games = target.get("games") or [target.get("game_a"), target.get("game_b")]
        games = [str(game) for game in games if game]

        for game in games:
            candidates = [
                row for row in rows
                if row.get("game") == game and reward_enum_value(row) == reward_enum
            ]
            if not candidates:
                continue

            best = min(candidates, key=lambda row: _condition_distance(row, target_value))
            row_i = best.get("row_i", "")
            key = (target_i, game, row_i)
            if key in seen:
                continue

            out = dict(best)
            out["_condition_bucket"] = f"target_{target_i}"
            out["_condition_bucket_label"] = f"Config target {target_i}: {target.get('note') or f'condition={_numeric_text(str(target_value))}'}"
            out["_target_condition_value"] = _numeric_text(str(target_value))
            out["_target_condition_distance"] = _numeric_text(str(_condition_distance(best, target_value)[0]))
            selected.append(out)
            seen.add(key)

    return selected


def condition_contrast_pair_rows_from_run_csv(
    run_result: RunResult,
    targets: list[dict],
    num_episodes: int,
) -> list[dict]:
    rows = rows_from_run_csv(run_result, max_rows=10_000)
    paired_rows: list[dict] = []
    for fallback_i, target in enumerate(targets, start=1):
        target_i = int(target.get("_target_i", fallback_i))
        reward_enum = int(target["reward_enum"])
        target_value = float(target["condition_value"])
        games = target.get("games") or [target.get("game_a"), target.get("game_b")]
        games = [str(game) for game in games if game]
        if len(games) != 2:
            continue

        member_lists: list[list[dict[str, str]]] = []
        for game in games:
            candidates = [
                row for row in rows
                if row.get("game") == game and reward_enum_value(row) == reward_enum
            ]
            candidates = sorted(candidates, key=lambda row: _condition_distance(row, target_value))

            selected: list[dict[str, str]] = []
            seen_instructions: set[str] = set()
            for row in candidates:
                instruction = row.get("instruction", "")
                if instruction in seen_instructions:
                    continue
                out = dict(row)
                out["_condition_bucket"] = f"target_{target_i}"
                out["_condition_bucket_label"] = f"Config target {target_i}: {target.get('note') or f'condition={_numeric_text(str(target_value))}'}"
                out["_target_condition_value"] = _numeric_text(str(target_value))
                out["_target_condition_distance"] = _numeric_text(str(_condition_distance(row, target_value)[0]))
                selected.append(out)
                seen_instructions.add(instruction)
                if len(selected) >= num_episodes:
                    break

            if len(selected) < num_episodes:
                seen_rows = {row.get("row_i") for row in selected}
                for row in candidates:
                    if row.get("row_i") in seen_rows:
                        continue
                    out = dict(row)
                    out["_condition_bucket"] = f"target_{target_i}"
                    out["_condition_bucket_label"] = f"Config target {target_i}: {target.get('note') or f'condition={_numeric_text(str(target_value))}'}"
                    out["_target_condition_value"] = _numeric_text(str(target_value))
                    out["_target_condition_distance"] = _numeric_text(str(_condition_distance(row, target_value)[0]))
                    selected.append(out)
                    seen_rows.add(row.get("row_i"))
                    if len(selected) >= num_episodes:
                        break

            if not selected:
                break
            member_lists.append(selected)

        if len(member_lists) != 2:
            continue

        bucket_label = member_lists[0][0].get("_condition_bucket_label", f"Config target {target_i}")
        for episode_i in range(num_episodes):
            members = [
                member_lists[0][episode_i % len(member_lists[0])],
                member_lists[1][episode_i % len(member_lists[1])],
            ]
            paired_rows.append(
                {
                    "row_i": str(episode_i),
                    "game": "pair",
                    "reward_enum": str(reward_enum),
                    "_row_key": f"{target_i:02d}_re{reward_enum}_episode{episode_i:02d}",
                    "_condition_bucket": f"target_{target_i}",
                    "_condition_bucket_label": bucket_label,
                    "_episode_i": str(episode_i),
                    "_pair_members": members,
                }
            )
    return paired_rows


def row_key(row: dict[str, str]) -> str:
    if row.get("_row_key"):
        return row["_row_key"]
    row_i = row.get("row_i", "x")
    game = row.get("game", "game")
    reward_enum = row.get("reward_enum", "re")
    return f"{game}_re{reward_enum}_{int(float(row_i)):04d}"


def row_label(row: dict[str, str]) -> str:
    if row.get("_pair_members"):
        lines = []
        for member in row["_pair_members"]:
            instruction = member.get("instruction") or "-"
            c_value = _numeric_text(str(condition_value(member)))
            percentile = member.get("_condition_percentile_text")
            condition_suffix = f"c={c_value}, p={percentile}" if percentile else f"c={c_value}"
            lines.append(f"{member.get('game', '-')}: {instruction} ({condition_suffix})")
        return "<br>".join(markdown_escape(line) for line in lines)

    condition = condition_text(row)
    if row.get("_condition_percentile_text"):
        condition = f"{condition} (p={row['_condition_percentile_text']})"

    parts = [
        f"Game: {row.get('game', '-')}",
        f"Task: {task_name(row)}",
        f"Condition: {condition}",
    ]
    if row.get("_target_condition_value"):
        parts.append(
            f"Target: condition_{reward_enum_value(row)}={row['_target_condition_value']} "
            f"(Δ={row.get('_target_condition_distance', '-')})"
        )
    instruction = row.get("instruction")
    if instruction:
        parts.append(f"Instruction: {instruction}")
    return "<br>".join(markdown_escape(part) for part in parts)


def h5_folder_name(row: dict[str, str]) -> str:
    game = row.get("game", "")
    reward_enum = reward_enum_value(row)
    row_i = int(float(row.get("row_i", 0)))
    return f"{game}_re{reward_enum}_{row_i:04d}"

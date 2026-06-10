from __future__ import annotations

import csv
from pathlib import Path

from .models import RunResult
from .utils import markdown_escape, reward_enum_value, task_name


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


def row_key(row: dict[str, str]) -> str:
    row_i = row.get("row_i", "x")
    game = row.get("game", "game")
    reward_enum = row.get("reward_enum", "re")
    return f"{game}_re{reward_enum}_{int(float(row_i)):04d}"


def row_label(row: dict[str, str]) -> str:
    parts = [
        f"Game: {row.get('game', '-')}",
        f"Task: {task_name(row)}",
        f"Condition: {condition_text(row)}",
    ]
    instruction = row.get("instruction")
    if instruction:
        parts.append(f"Instruction: {instruction}")
    return "<br>".join(markdown_escape(part) for part in parts)


def h5_folder_name(row: dict[str, str]) -> str:
    game = row.get("game", "")
    reward_enum = reward_enum_value(row)
    row_i = int(float(row.get("row_i", 0)))
    return f"{game}_re{reward_enum}_{row_i:04d}"


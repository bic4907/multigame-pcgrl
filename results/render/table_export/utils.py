from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from .models import RunResult


REWARD_ENUM_NAMES = {
    0: "region",
    1: "path_length",
    2: "interactable",
    3: "hazard",
    4: "collectable",
}


def load_config(config_path: Path | str):
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_wandb_run_url(url: str) -> Optional[dict[str, str]]:
    try:
        parsed = urlparse(url)
        parts = [p for p in parsed.path.split("/") if p]
        if "runs" not in parts:
            return None
        run_idx = parts.index("runs")
        if run_idx + 1 >= len(parts) or run_idx < 2:
            return None
        return {
            "entity": parts[run_idx - 2],
            "project": parts[run_idx - 1],
            "run_id": parts[run_idx + 1],
        }
    except Exception:
        return None


def run_url(entity: str, project: str, run_id: str) -> str:
    return f"https://wandb.ai/{entity}/{project}/runs/{run_id}"


def replace_reward_enum_in_run_name(run_name: str, reward_enum: int) -> str:
    return re.sub(r"(--ev_re-)([^_]+)(_[^/]*)", rf"\g<1>{reward_enum}\g<3>", run_name)


def safe_slug(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return value or "run"


def unique_methods(run_results: list[RunResult]) -> list[str]:
    methods = []
    seen = set()
    for result in run_results:
        if result.method in seen:
            continue
        seen.add(result.method)
        methods.append(result.method)
    return methods


def markdown_escape(value) -> str:
    text = "" if value is None else str(value)
    return text.replace("|", "\\|").replace("\n", "<br>")


def reward_enum_value(row: dict[str, str]) -> int:
    try:
        return int(float(row.get("reward_enum", 0)))
    except (TypeError, ValueError):
        return -1


def reward_enum_section_title(reward_enum: int) -> str:
    return f"{REWARD_ENUM_NAMES.get(reward_enum, 'unknown')} (re={reward_enum})"


def task_name(row: dict[str, str]) -> str:
    reward_enum = reward_enum_value(row)
    if reward_enum < 0:
        return f"unknown ({row.get('reward_enum', '')})"
    return reward_enum_section_title(reward_enum)


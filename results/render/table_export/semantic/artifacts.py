from __future__ import annotations

import csv
import json
import math
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from ..wandb_artifacts import (
    download_eval_csv_from_run,
    download_eval_h5_from_run,
)
from ..models import CandidateRow, MethodRewardRun
from .constants import PROJECT_ROOT, RENDER_DIR, _safe_slug
from .metrics import _compute_metric

SCRIPT_DIR = RENDER_DIR


def _num(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def _h5_group(game: str, reward_enum: int, row_i: str) -> str:
    return f"{game}_re{reward_enum}_{int(float(row_i)):04d}"

def _iter_h5_seeds(h5_path: Path, group_name: str) -> list[int]:
    with h5py.File(str(h5_path), "r") as h5:
        if group_name not in h5:
            return []
        seeds = []
        for key in h5[group_name].keys():
            if key.startswith("seed_") and "state" in h5[group_name][key]:
                try:
                    seeds.append(int(key.split("_", 1)[1]))
                except ValueError:
                    continue
        return sorted(seeds)

def _read_state(h5_path: Path, group_name: str, seed: int) -> np.ndarray | None:
    state_path = f"{group_name}/seed_{seed}/state"
    with h5py.File(str(h5_path), "r") as h5:
        if state_path not in h5:
            return None
        return np.asarray(h5[state_path][()], dtype=np.int32)

def _resolve_wandb_api():
    previous_sys_path = list(sys.path)
    try:
        sys.path = [p for p in sys.path if Path(p or ".").resolve() != PROJECT_ROOT]
        import wandb

        return wandb.Api()
    finally:
        sys.path = previous_sys_path

def _run_created_ts(run: Any) -> float:
    for attr in ("created_at", "createdAt"):
        value = getattr(run, attr, None)
        if value is None:
            continue
        if hasattr(value, "timestamp"):
            return float(value.timestamp())
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
            except ValueError:
                pass
    return 0.0

def _is_reward_run(run: Any, reward_enum: int) -> bool:
    text = f"{getattr(run, 'name', '')} {getattr(run, 'id', '')}"
    return bool(re.search(rf"(^|[-_])ev_re-{reward_enum}([_-]|$)", text))

def _resolve_project_run(api: Any, entity: str, project: str, reward_enum: int) -> Any:
    runs = list(api.runs(f"{entity}/{project}", per_page=300))
    candidates = [run for run in runs if _is_reward_run(run, reward_enum)]
    if not candidates:
        raise RuntimeError(f"{project}: ev_re-{reward_enum} run not found")

    def score(run: Any) -> tuple[int, float]:
        state = str(getattr(run, "state", "")).lower()
        finished = 1 if state in {"finished", "crashed", "failed"} else 0
        return finished, _run_created_ts(run)

    return sorted(candidates, key=score)[-1]

def _download_runs(
    entity: str,
    projects: dict[str, str],
    reward_enums: list[int],
    output_dir: Path,
    use_cache_only: bool,
) -> dict[tuple[str, int], MethodRewardRun]:
    runs: dict[tuple[str, int], MethodRewardRun] = {}
    cache_root = SCRIPT_DIR / ".wandb_download" / _safe_slug(entity)

    if use_cache_only:
        for method, project in projects.items():
            project_root = cache_root / _safe_slug(project)
            for reward_enum in reward_enums:
                found = []
                for h5_path in project_root.glob("*/eval.h5"):
                    try:
                        with h5py.File(str(h5_path), "r") as h5:
                            if not any(f"_re{reward_enum}_" in key for key in h5.keys()):
                                continue
                    except OSError:
                        continue
                    csv_dir = h5_path.parent / "csv"
                    if (csv_dir / "ctrl_sim.csv").exists():
                        found.append((h5_path.stat().st_mtime, h5_path, csv_dir))
                if found:
                    _, h5_path, csv_dir = sorted(found)[-1]
                    runs[(method, reward_enum)] = MethodRewardRun(
                        method=method,
                        project=project,
                        reward_enum=reward_enum,
                        run_id=h5_path.parent.name,
                        h5_path=h5_path,
                        csv_dir=csv_dir,
                    )
        return runs

    api = _resolve_wandb_api()
    for method, project in projects.items():
        for reward_enum in reward_enums:
            result = MethodRewardRun(method=method, project=project, reward_enum=reward_enum)
            try:
                run = _resolve_project_run(api, entity, project, reward_enum)
                run_id = getattr(run, "id", "")
                result.run_id = run_id
                result.run_name = getattr(run, "name", "")
                result.run_url = f"https://wandb.ai/{entity}/{project}/runs/{run_id}"
                target_dir = cache_root / _safe_slug(project) / _safe_slug(run_id)
                h5_path, h5_err, _ = download_eval_h5_from_run(run, target_dir)
                csv_dir, csv_err, _ = download_eval_csv_from_run(run, target_dir)
                result.h5_path = h5_path
                result.csv_dir = csv_dir
                result.error = h5_err or csv_err
            except Exception as exc:
                result.error = str(exc)
            runs[(method, reward_enum)] = result

    (output_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                f"{method}_re{reward_enum}": {
                    "project": run.project,
                    "run_id": run.run_id,
                    "run_name": run.run_name,
                    "run_url": run.run_url,
                    "h5_path": str(run.h5_path) if run.h5_path else None,
                    "csv_dir": str(run.csv_dir) if run.csv_dir else None,
                    "error": run.error,
                }
                for (method, reward_enum), run in sorted(runs.items())
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return runs

def _rows_for_run(run: MethodRewardRun, games: list[str]) -> dict[tuple[str, str], list[dict[str, str]]]:
    if run.csv_dir is None:
        return {}
    rows = _read_csv(run.csv_dir / "ctrl_sim.csv")
    out: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("game") not in games:
            continue
        try:
            if int(float(row.get("reward_enum", -1))) != run.reward_enum:
                continue
        except (TypeError, ValueError):
                continue
        key = (row.get("game", ""), row.get("row_i", ""))
        out[key].append(row)
    return out

def _build_candidates(run: MethodRewardRun, games: list[str]) -> dict[tuple[str, str], CandidateRow]:
    rows_by_key = _rows_for_run(run, games)
    candidates: dict[tuple[str, str], CandidateRow] = {}
    if run.h5_path is None:
        return candidates

    for (game, row_i), rows in rows_by_key.items():
        row = rows[0]
        group = _h5_group(game, run.reward_enum, row_i)
        h5_seeds = set(_iter_h5_seeds(run.h5_path, group))
        if not h5_seeds:
            continue
        candidate = CandidateRow(
            method=run.method,
            game=game,
            reward_enum=run.reward_enum,
            row_i=row_i,
            instruction=row.get("instruction", ""),
            target=_num(row.get(f"condition_{run.reward_enum}")),
            h5_group=group,
        )
        for seed_row in rows:
            try:
                seed = int(float(seed_row.get("seed", "")))
            except (TypeError, ValueError):
                continue
            if seed not in h5_seeds:
                continue
            metric = _num(seed_row.get(f"feat_{run.reward_enum}"))
            if metric is not None:
                candidate.seed_metrics[seed] = metric

        if not candidate.seed_metrics:
            # Fallback for unusual CSVs without feat_k columns.
            for seed in sorted(h5_seeds):
                state = _read_state(run.h5_path, group, seed)
                if state is None:
                    continue
                candidate.seed_metrics[seed] = _compute_metric(state, run.reward_enum)
        if candidate.seed_metrics:
            candidates[(game, row_i)] = candidate
    return candidates

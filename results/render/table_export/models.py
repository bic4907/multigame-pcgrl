from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np


@dataclass
class RunResult:
    method: str
    run_url: str
    reward_enum: Optional[int] = None
    run_name: Optional[str] = None
    artifact_name: Optional[str] = None
    csv_artifact_name: Optional[str] = None
    h5_path: Optional[Path] = None
    csv_dir: Optional[Path] = None
    h5_stats: Optional[dict[str, Any]] = None
    error: Optional[str] = None


@dataclass
class MethodRewardRun:
    method: str
    project: str
    reward_enum: int
    run_id: str = ""
    run_name: str = ""
    h5_path: Path | None = None
    csv_dir: Path | None = None
    run_url: str = ""
    error: str | None = None


@dataclass
class CandidateRow:
    method: str
    game: str
    reward_enum: int
    row_i: str
    instruction: str
    target: float | None
    h5_group: str
    seed_metrics: dict[int, float] = field(default_factory=dict)

    @property
    def mean_std(self) -> tuple[float | None, float | None]:
        values = [v for v in self.seed_metrics.values() if math.isfinite(v)]
        if not values:
            return None, None
        return float(np.mean(values)), float(np.std(values, ddof=0))


@dataclass
class RenderCell:
    method: str
    game: str
    feature: str
    low: CandidateRow
    mid: CandidateRow
    high: CandidateRow
    low_seed: int
    mid_seed: int
    high_seed: int
    low_image: Path
    mid_image: Path
    high_image: Path
    low_overlay: Path
    mid_overlay: Path
    high_overlay: Path
    triplet_overlay: Path


RenderConfigPanels = dict[tuple[str, str, str, str], dict[str, Any]]

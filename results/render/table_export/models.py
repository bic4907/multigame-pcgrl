from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


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


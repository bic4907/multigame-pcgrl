"""
Utility: create a timestamped output directory, record run configuration,
and set up a logger that writes to both console and <run_dir>/run.log.

Usage inside a results/ script:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from utils.run_output import make_run_dir, setup_logger

Pipeline integration:
    run_pipeline.py sets two env vars before calling each step:
        PIPELINE_RUN_DIR  → root dir of the running pipeline
        PIPELINE_EXPERIMENT → experiment name (optional)
    When these are set, make_run_dir() creates:
        <PIPELINE_RUN_DIR>/<experiment>/<script_name>/   (experiment 있을 때)
        <PIPELINE_RUN_DIR>/<script_name>/                (experiment 없을 때)
"""
from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

_UTILS_DIR    = Path(__file__).resolve().parent   # results/utils/
_RESULTS_DIR  = _UTILS_DIR.parent                 # results/
_PROJECT_ROOT = _RESULTS_DIR.parent               # project root
_DEFAULT_CFG  = _RESULTS_DIR / "config.json"

# Add project root to sys.path so instruct_rl is importable.
# Use append (not insert) to avoid shadowing results/utils/ with root/utils.py.
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.append(str(_PROJECT_ROOT))

from instruct_rl.utils.log_utils import get_logger  # noqa: E402


def load_cfg(cfg_path: Path | None = None) -> dict:
    """Load config.json from results/. Returns empty dict if not found."""
    path = cfg_path or _DEFAULT_CFG
    if path.is_file():
        with path.open(encoding="utf-8") as f:
            return json.load(f)
    return {}


def make_run_dir(script_name: str, cfg: dict | None = None) -> Path:
    """
    Create and return an output directory for the script.

    Standalone 실행:
        <outputs_root>/<script_name>_<YYYYMMDD_HHMMSS>/

    Pipeline 내 실행 (PIPELINE_RUN_DIR 환경변수 설정 시):
        <PIPELINE_RUN_DIR>/<experiment>/<script_name>/   # PIPELINE_EXPERIMENT 있을 때
        <PIPELINE_RUN_DIR>/<script_name>/                # PIPELINE_EXPERIMENT 없을 때

    Also saves a snapshot of cfg as used_config.json inside the directory.
    """
    if cfg is None:
        cfg = load_cfg()

    pipeline_run_dir = os.environ.get("PIPELINE_RUN_DIR")
    pipeline_experiment = os.environ.get("PIPELINE_EXPERIMENT", "")

    if pipeline_run_dir:
        base = Path(pipeline_run_dir)
        if pipeline_experiment:
            run_dir = base / pipeline_experiment / script_name
        else:
            run_dir = base / script_name
    else:
        raw_root: str = cfg.get("outputs_root", "results/outputs")
        root = Path(raw_root)
        if not root.is_absolute():
            root = (_PROJECT_ROOT / root).resolve()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = root / f"{script_name}_{timestamp}"

    run_dir.mkdir(parents=True, exist_ok=True)

    # Save config snapshot
    snapshot = run_dir / "used_config.json"
    with snapshot.open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    return run_dir


def setup_logger(
    run_dir: Path,
    name: str = "run",
    log_filename: str = "run.log",
    file_level: int = logging.DEBUG,
) -> logging.Logger:
    """
    Create a logger via instruct_rl's get_logger() and attach a FileHandler
    that writes to <run_dir>/<log_filename>.
    """
    logger = get_logger(name)

    log_file = run_dir / log_filename

    # Prevent duplicate FileHandlers for the same log file
    already = any(
        isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None) == str(log_file)
        for h in logger.handlers
    )
    if not already:
        fmt = logging.Formatter(
            "[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(file_level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger




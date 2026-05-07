"""
eval_downloader.py
==================
Downloads eval artifacts uploaded by runner.py from W&B to local storage.

Artifact types
  - eval_csv  (type=dataset): ctrl_sim.csv / results.csv / diversity.csv / summary.csv
  - eval_h5_* (type=dataset): eval.h5

Examples
--------
    python results/eval_downloader.py
    python results/eval_downloader.py --experiment allseen
    python results/eval_downloader.py --experiment unseen_generalizability
    python results/eval_downloader.py --h5
    python results/eval_downloader.py --output wandb_projects
    python results/eval_downloader.py --finished-only --workers 4
"""

import argparse
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from tqdm import tqdm

# ── Add project root to sys.path ─────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from sweep.wandb_utils.config import DEFAULT_ENTITY, DEFAULT_NUM_WORKERS
from sweep.wandb_utils.downloader import get_api

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _load_cfg() -> dict:
    cfg_path = os.path.join(_HERE, "config.json")
    if os.path.isfile(cfg_path):
        import json
        with open(cfg_path, encoding="utf-8") as f:
            return json.load(f)
    return {}

_CFG = _load_cfg()


@dataclass
class _RunResult:
    run_name: str
    status: str          # "ok" | "skipped" | "error"
    error: str = ""


@dataclass
class _ProjectSummary:
    project: str
    ok: int = 0
    skipped: int = 0
    errors: list[_RunResult] = field(default_factory=list)

# Project list to download
# Priority: --experiment > --projects > all experiments combined (config.json fallback)
_EXPERIMENTS: dict[str, list[str]] = {
    name: exp.get("target_projects", [])
    for name, exp in _CFG.get("experiments", {}).items()
}
_ALL_EXPERIMENT_PROJECTS: list[str] = [
    p for projects in _EXPERIMENTS.values() for p in projects
]
# Unique, preserving order
_seen: set[str] = set()
_ALL_EXPERIMENT_PROJECTS_DEDUP: list[str] = []
for _p in _ALL_EXPERIMENT_PROJECTS:
    if _p not in _seen:
        _seen.add(_p)
        _ALL_EXPERIMENT_PROJECTS_DEDUP.append(_p)

TARGET_PROJECTS: list[str] = _ALL_EXPERIMENT_PROJECTS_DEDUP or [
    "aaai27_eval_cpcgrl",
    "aaai27_eval_cpcgrl_all",
]
_DEFAULT_NUM_WORKERS: int = _CFG.get("wandb", {}).get("num_workers", DEFAULT_NUM_WORKERS)
_DEFAULT_OUTPUT: str = "wandb_projects"

# ---------------------------------------------------------------------------
# Download a single run
# ---------------------------------------------------------------------------


def _download_run(
    run,
    output_dir: str,
    download_csv: bool = True,
    download_h5: bool = True,
    skip_if_exists: bool = True,
) -> _RunResult:
    """Download eval artifacts from a single W&B run.
    Returns
    -------
    _RunResult  with status "ok" | "skipped" | "error"
    """
    train_dir, eval_dir = run.name.split("--")
    run_dir = os.path.join(output_dir, train_dir, eval_dir)

    # ── skip check ────────────────────────────────────────────────────────
    if skip_if_exists:
        ctrl_sim = os.path.join(run_dir, "ctrl_sim.csv")
        results  = os.path.join(run_dir, "results.csv")
        if os.path.isfile(ctrl_sim) and os.path.isfile(results):
            return _RunResult(run_name=run.name, status="skipped")

    os.makedirs(run_dir, exist_ok=True)

    # ── list artifacts ────────────────────────────────────────────────────
    try:
        artifacts = list(run.logged_artifacts())
    except Exception as e:
        return _RunResult(run_name=run.name, status="error", error=f"failed to list artifacts: {e}")

    csv_artifact = None
    h5_artifacts = []

    for art in artifacts:
        if art.name.startswith("eval_csv") and art.type == "dataset":
            csv_artifact = art
        elif art.name.startswith("eval_h5") and art.type == "dataset":
            h5_artifacts.append(art)

    errors = []

    # ── download eval_csv ─────────────────────────────────────────────────
    if download_csv:
        if csv_artifact is None:
            errors.append("eval_csv artifact not found")
        else:
            try:
                for f in csv_artifact.files():
                    local_path = os.path.join(run_dir, f.name)
                    if skip_if_exists and os.path.isfile(local_path):
                        continue
                    f.download(root=run_dir, replace=True)
            except Exception as e:
                errors.append(f"eval_csv error: {e}")

    # ── download eval_h5 ──────────────────────────────────────────────────
    if download_h5 and h5_artifacts:
        latest_h5 = h5_artifacts[-1]
        h5_local = os.path.join(run_dir, "eval.h5")
        if not (skip_if_exists and os.path.isfile(h5_local)):
            try:
                for f in latest_h5.files():
                    f.download(root=run_dir, replace=True)
            except Exception as e:
                errors.append(f"eval_h5 error: {e}")

    if errors:
        return _RunResult(run_name=run.name, status="error", error=" | ".join(errors))
    return _RunResult(run_name=run.name, status="ok")


# ---------------------------------------------------------------------------
# Download all runs in a project
# ---------------------------------------------------------------------------


def download_eval_project(
    project: str,
    entity: str = DEFAULT_ENTITY,
    output_dir: str = "wandb_projects",
    download_csv: bool = True,
    download_h5: bool = True,
    skip_if_exists: bool = True,
    n_workers: int = DEFAULT_NUM_WORKERS,
    filters: dict | None = None,
    per_page: int = 200,
) -> _ProjectSummary:
    """Download eval artifacts from all runs in a project."""
    api = get_api()
    proj_output_dir = os.path.join(output_dir, project)
    os.makedirs(proj_output_dir, exist_ok=True)

    runs = list(api.runs(f"{entity}/{project}", filters=filters or {}, per_page=per_page))
    summary = _ProjectSummary(project=project)

    def _worker(run):
        return _download_run(
            run,
            output_dir=proj_output_dir,
            download_csv=download_csv,
            download_h5=download_h5,
            skip_if_exists=skip_if_exists,
        )

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_worker, r): r.name for r in runs}
        with tqdm(total=len(futures), desc=project, unit="run") as pbar:
            for fut in as_completed(futures):
                try:
                    res: _RunResult = fut.result()
                except Exception as e:
                    res = _RunResult(run_name=futures[fut], status="error", error=str(e))

                if res.status == "ok":
                    summary.ok += 1
                elif res.status == "skipped":
                    summary.skipped += 1
                else:
                    summary.errors.append(res)

                pbar.update(1)

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    _exp_names = list(_EXPERIMENTS.keys())
    _exp_choices_str = ", ".join(_exp_names) if _exp_names else "none defined"
    parser = argparse.ArgumentParser(
        description="W&B eval artifact downloader (eval_csv / eval_h5)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
examples:
  python results/eval_downloader.py
  python results/eval_downloader.py --experiment allseen
  python results/eval_downloader.py --experiment unseen_generalizability
  python results/eval_downloader.py --no-h5
  python results/eval_downloader.py --output wandb_projects --finished-only
  python results/eval_downloader.py --projects aaai27_eval_cpcgrl --workers 4

available experiments: {_exp_choices_str}
        """,
    )
    parser.add_argument(
        "--experiment",
        choices=_exp_names if _exp_names else None,
        default=None,
        metavar="EXPERIMENT",
        help=f"Experiment group to download (choices: {_exp_choices_str}). "
             "Overrides --projects.",
    )
    parser.add_argument(
        "--projects",
        nargs="+",
        default=None,
        metavar="PROJECT",
        help="Explicit W&B project list. Ignored when --experiment is set. "
             "Defaults to all projects across all experiments.",
    )
    parser.add_argument(
        "--entity",
        default=DEFAULT_ENTITY,
        help=f"W&B entity (default: {DEFAULT_ENTITY})",
    )
    parser.add_argument(
        "--output",
        default=_DEFAULT_OUTPUT,
        help="Local root path to save downloads (default: wandb_projects)",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Skip downloading eval_csv artifact (ctrl_sim/results/diversity/summary CSVs)",
    )
    parser.add_argument(
        "--no-h5",
        action="store_true",
        default=True,
        help="Skip downloading eval_h5 artifact / eval.h5 (default)",
    )
    parser.add_argument(
        "--h5",
        action="store_false",
        dest="no_h5",
        help="Include eval_h5 artifact (eval.h5) in download",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing files",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=_DEFAULT_NUM_WORKERS,
        help="Number of parallel download threads (default: config.json wandb.num_workers)",
    )
    parser.add_argument(
        "--finished-only",
        action="store_true",
        help="Only download runs with state=finished",
    )
    return parser.parse_args()


def _print_summary(summaries: list[_ProjectSummary]) -> None:
    """Log per-project download results as a table."""
    COL_W = [40, 8, 10, 8]
    headers = ["Project", "OK", "Skipped", "Error"]
    sep = "+" + "+".join("-" * (w + 2) for w in COL_W) + "+"
    fmt = "| " + " | ".join(f"{{:<{w}}}" for w in COL_W) + " |"

    lines = [sep, fmt.format(*headers), sep]
    for s in summaries:
        lines.append(fmt.format(s.project, s.ok, s.skipped, len(s.errors)))
    lines.append(sep)
    logger.info("\n" + "\n".join(lines))

    # Error details
    any_error = any(s.errors for s in summaries)
    if any_error:
        logger.warning("[!] Error details:")
        for s in summaries:
            for r in s.errors:
                logger.warning("  [%s] %s — %s", s.project, r.run_name, r.error)


def main():
    args = parse_args()

    # Resolve project list: --experiment > --projects > all experiments
    if args.experiment:
        projects = _EXPERIMENTS.get(args.experiment, [])
        if not projects:
            logger.error("Experiment '%s' has no target_projects defined.", args.experiment)
            sys.exit(1)
        logger.info("Experiment: %s  (%d projects)", args.experiment, len(projects))
    elif args.projects:
        projects = args.projects
    else:
        projects = TARGET_PROJECTS

    filters = {"state": "finished"} if args.finished_only else None

    summaries: list[_ProjectSummary] = []
    for project in projects:
        summary = download_eval_project(
            project=project,
            entity=args.entity,
            output_dir=args.output,
            download_csv=not args.no_csv,
            download_h5=not args.no_h5,
            skip_if_exists=not args.force,
            n_workers=args.workers,
            filters=filters,
        )
        summaries.append(summary)

    _print_summary(summaries)


if __name__ == "__main__":
    main()

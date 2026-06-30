"""
Download decoder_prediction_csv artifacts and plot decoder training performance.

Inputs:
  - W&B project runs from results/config.json experiment target_projects
  - logged artifact named decoder_prediction_csv
  - W&B history metric train(direction)/continuous_delta_loss

Outputs:
  - decoder_prediction_rows.csv
  - decoder_epoch_metrics.csv
  - decoder_delta_loss_history.csv
  - delta_loss.png/.pdf
  - regression.png/.pdf
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

_HERE = Path(__file__).resolve().parent
_RESULTS_DIR = _HERE.parent.parent
_ROOT = _RESULTS_DIR.parent
if str(_RESULTS_DIR) not in sys.path:
    sys.path.insert(0, str(_RESULTS_DIR))
if str(_ROOT) not in sys.path:
    sys.path.append(str(_ROOT))

from sweep.wandb_utils.config import DEFAULT_ENTITY, DEFAULT_NUM_WORKERS
from sweep.wandb_utils.downloader import get_api
from utils.core.run_output import load_cfg, setup_logger

_CFG = load_cfg()
_LOSS_KEY = "train(direction)/continuous_delta_loss"


def _experiment_names() -> list[str]:
    return list(_CFG.get("experiments", {}).keys())


def _safe_name(value: str) -> str:
    return "".join(c if c.isalnum() or c in "._=-" else "_" for c in value)


def _as_float(value: object) -> float | None:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _resolve_projects(experiment: str | None, projects: list[str] | None) -> list[str]:
    if projects:
        return projects
    if experiment:
        exp_cfg = _CFG.get("experiments", {}).get(experiment, {})
        configured = list(exp_cfg.get("target_projects", []))
        if not configured:
            raise SystemExit(f"Experiment '{experiment}' has no target_projects in results/config.json.")
        return configured
    return ["aaai27_encoder_mgpcgrl__decoderperformance"]


def _make_decoder_run_dir(cfg: dict) -> Path:
    """Use a single decoder_performance folder inside pipeline outputs."""
    pipeline_run_dir = os.environ.get("PIPELINE_RUN_DIR")
    pipeline_experiment = os.environ.get("PIPELINE_EXPERIMENT", "")
    if pipeline_run_dir and pipeline_experiment == "decoder_performance":
        run_dir = Path(pipeline_run_dir) / pipeline_experiment
        run_dir.mkdir(parents=True, exist_ok=True)
        with (run_dir / "used_config.json").open("w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
        return run_dir

    from utils.core.run_output import make_run_dir

    return make_run_dir("decoder_performance", cfg=cfg)


def _default_cache_dir() -> Path:
    return _RESULTS_DIR / "wandb_projects" / "decoder_performance_cache"


def _run_cache_dir(cache_dir: Path, project: str, run) -> Path:
    return cache_dir / project / _safe_name(run.id)


def _is_finished_run(run) -> bool:
    return str(getattr(run, "state", "")).lower() == "finished"


def _select_decoder_artifact(run) -> object | None:
    artifacts = list(run.logged_artifacts())
    candidates = []
    for artifact in artifacts:
        base_name = artifact.name.split(":", 1)[0]
        if base_name == "decoder_prediction_csv" and artifact.type == "dataset":
            candidates.append(artifact)
    return candidates[-1] if candidates else None


def _download_prediction_artifact(
    run,
    project: str,
    cache_dir: Path,
    force: bool,
) -> list[Path]:
    run_dir = _run_cache_dir(cache_dir, project, run)
    artifact_dir = run_dir / "decoder_prediction_csv"
    if _is_finished_run(run) and artifact_dir.exists() and not force:
        cached_csvs = sorted(artifact_dir.rglob("*.csv"))
        if cached_csvs:
            return cached_csvs

    artifact = _select_decoder_artifact(run)
    if artifact is None:
        return []

    if force and artifact_dir.exists():
        shutil.rmtree(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    if not any(artifact_dir.glob("*.csv")):
        artifact.download(root=str(artifact_dir))
    return sorted(artifact_dir.rglob("*.csv"))


def _read_prediction_csvs(
    csv_paths: list[Path],
    *,
    project: str,
    run,
    delta_weight: float | None,
) -> pd.DataFrame:
    frames = []
    for path in csv_paths:
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if df.empty:
            continue
        df["project"] = project
        df["run_id"] = run.id
        df["run_name"] = run.name
        df["source_csv"] = path.name
        if "delta_weight" not in df.columns or df["delta_weight"].isna().all():
            df["delta_weight"] = delta_weight
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _history_rows(
    project: str,
    run,
    delta_weight: float | None,
    cache_dir: Path,
    force: bool,
) -> pd.DataFrame:
    history_cache = _run_cache_dir(cache_dir, project, run) / "decoder_delta_loss_history.csv"
    if _is_finished_run(run) and history_cache.is_file() and not force:
        try:
            cached = pd.read_csv(history_cache)
        except Exception:
            cached = pd.DataFrame()
        if not cached.empty:
            return cached

    try:
        rows = list(run.scan_history(keys=[_LOSS_KEY, "total/epoch"]))
    except Exception:
        return pd.DataFrame()
    if not rows:
        return pd.DataFrame()
    hist = pd.DataFrame(rows)
    if hist.empty or _LOSS_KEY not in hist.columns:
        return pd.DataFrame()

    keep = [c for c in ["_step", "total/epoch", _LOSS_KEY] if c in hist.columns]
    hist = hist[keep].copy()
    hist = hist.dropna(subset=[_LOSS_KEY])
    if hist.empty:
        return hist

    if "total/epoch" in hist.columns:
        hist["epoch"] = pd.to_numeric(hist["total/epoch"], errors="coerce") + 1
    else:
        hist["epoch"] = pd.to_numeric(hist.get("_step"), errors="coerce")
    hist["project"] = project
    hist["run_id"] = run.id
    hist["run_name"] = run.name
    hist["delta_weight"] = delta_weight
    hist = hist.rename(columns={_LOSS_KEY: "continuous_delta_loss"})
    hist = hist[
        ["project", "run_id", "run_name", "delta_weight", "epoch", "continuous_delta_loss"]
    ].dropna(subset=["epoch"])
    if _is_finished_run(run) and not hist.empty:
        history_cache.parent.mkdir(parents=True, exist_ok=True)
        hist.to_csv(history_cache, index=False)
    return hist


def _prediction_epoch_metrics(pred_rows: pd.DataFrame) -> pd.DataFrame:
    if pred_rows.empty:
        return pd.DataFrame()

    df = pred_rows.copy()
    if "epoch_num" in df.columns:
        df["epoch"] = pd.to_numeric(df["epoch_num"], errors="coerce")
    elif "epoch" in df.columns:
        df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce") + 1
    else:
        return pd.DataFrame()

    required = {
        "project",
        "run_id",
        "run_name",
        "delta_weight",
        "epoch",
        "condition_target_raw",
        "condition_pred_raw",
    }
    missing = required.difference(df.columns)
    if missing:
        return pd.DataFrame()

    for col in [
        "delta_weight",
        "condition_target_raw",
        "condition_pred_raw",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(
        subset=[
            "delta_weight",
            "epoch",
            "condition_target_raw",
            "condition_pred_raw",
        ]
    )
    if df.empty:
        return pd.DataFrame()

    df["regression_abs_error_raw"] = (
        df["condition_target_raw"] - df["condition_pred_raw"]
    ).abs()
    if {"condition_target_norm", "condition_pred_norm"}.issubset(df.columns):
        df["condition_target_norm"] = pd.to_numeric(df["condition_target_norm"], errors="coerce")
        df["condition_pred_norm"] = pd.to_numeric(df["condition_pred_norm"], errors="coerce")
        df["regression_abs_error_norm"] = (
            df["condition_target_norm"] - df["condition_pred_norm"]
        ).abs()
    else:
        df["regression_abs_error_norm"] = float("nan")

    group_cols = ["project", "run_id", "run_name", "delta_weight", "epoch"]
    grouped = df.groupby(group_cols, dropna=False).agg(
        regression_mae_raw=("regression_abs_error_raw", "mean"),
        regression_mae_norm=("regression_abs_error_norm", "mean"),
        n_predictions=("regression_abs_error_raw", "size"),
    )
    return grouped.reset_index()


def _epoch_bin(values: pd.Series, bin_size: int) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if bin_size <= 1:
        return numeric.round().astype("Int64")
    return (((numeric - 1) // bin_size) * bin_size + 1).round().astype("Int64")


def _auto_bin_size(epoch_metrics: pd.DataFrame, loss_history: pd.DataFrame, max_points: int) -> int:
    epochs = []
    for df in (epoch_metrics, loss_history):
        if not df.empty and "epoch" in df.columns:
            epochs.extend(pd.to_numeric(df["epoch"], errors="coerce").dropna().tolist())
    if not epochs or max_points <= 0:
        return 1
    n_unique = len(set(int(round(e)) for e in epochs))
    return max(1, int(math.ceil(n_unique / max_points)))


def _aggregate_for_plot(df: pd.DataFrame, value_col: str, epoch_bin_size: int) -> pd.DataFrame:
    if df.empty or value_col not in df.columns:
        return pd.DataFrame()
    work = df.copy()
    work["delta_weight"] = pd.to_numeric(work["delta_weight"], errors="coerce")
    work["epoch_bin"] = _epoch_bin(work["epoch"], epoch_bin_size)
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    work = work.dropna(subset=["delta_weight", "epoch_bin", value_col])
    if work.empty:
        return pd.DataFrame()
    grouped = (
        work.groupby(["delta_weight", "epoch_bin"], dropna=False)[value_col]
        .agg(["mean", "std", "count"])
        .reset_index()
        .sort_values(["delta_weight", "epoch_bin"])
    )
    grouped["sem"] = grouped["std"].fillna(0.0) / grouped["count"].pow(0.5)
    grouped = grouped.rename(columns={"epoch_bin": "epoch", "mean": value_col})
    return grouped


def _plot_performance(
    epoch_metrics: pd.DataFrame,
    loss_history: pd.DataFrame,
    output_base: Path,
    epoch_bin_size: int,
    max_points_per_line: int,
) -> None:
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from matplotlib.ticker import FuncFormatter, MultipleLocator

    if epoch_bin_size <= 0:
        epoch_bin_size = _auto_bin_size(epoch_metrics, loss_history, max_points_per_line)

    plt.style.use("seaborn-v0_8-whitegrid")
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "mathtext.fontset": "custom",
            "mathtext.rm": "Times New Roman",
            "mathtext.it": "Times New Roman:italic",
            "mathtext.bf": "Times New Roman:bold",
        }
    )

    specs = [
        (
            loss_history,
            "continuous_delta_loss",
            "Embedding Alignment Loss",
            output_base.parent / "delta_loss",
        ),
        (
            epoch_metrics,
            "regression_mae_raw",
            "Reward Prediction Loss",
            output_base.parent / "regression",
        ),
    ]

    weights: set[float] = set()
    for source, value_col, _, _ in specs:
        plot_df = _aggregate_for_plot(source, value_col, epoch_bin_size)
        if not plot_df.empty:
            weights.update(float(w) for w in plot_df["delta_weight"].dropna().unique())
    sorted_weights = sorted(weights)
    if sorted_weights:
        norm = mpl.colors.Normalize(vmin=min(sorted_weights), vmax=max(sorted_weights))
        cmap = mpl.colormaps.get_cmap("Blues")
    else:
        norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)
        cmap = mpl.colormaps.get_cmap("Blues")

    markers = ["o", "s", "^", "D", "v", "P", "X"]

    def _epoch_formatter(value: float, _pos: int) -> str:
        if abs(value) >= 1000:
            scaled = value / 1000.0
            if abs(scaled - round(scaled)) < 1e-6:
                return f"{int(round(scaled))}e3"
            return f"{scaled:g}e3"
        if abs(value - round(value)) < 1e-6:
            return str(int(round(value)))
        return f"{value:g}"

    for source, value_col, ylabel, out_path in specs:
        plot_df = _aggregate_for_plot(source, value_col, epoch_bin_size)
        fig, ax = plt.subplots(figsize=(3.6, 2.7))
        if plot_df.empty:
            ax.text(0.5, 0.5, f"No data: {value_col}", ha="center", va="center")
            ax.set_axis_off()
        else:
            for idx, weight in enumerate(sorted_weights):
                wdf = plot_df[plot_df["delta_weight"] == weight].sort_values("epoch")
                if wdf.empty:
                    continue
                color = cmap(0.28 + 0.72 * norm(weight))
                x = wdf["epoch"].astype(float).to_numpy()
                y = wdf[value_col].astype(float).to_numpy()
                sem = wdf["sem"].fillna(0.0).astype(float).to_numpy()
                ax.plot(
                    x,
                    y,
                    color=color,
                    linewidth=1.9,
                    linestyle="-",
                    marker=markers[idx % len(markers)],
                    markersize=4.4,
                    markevery=max(1, len(x) // 12),
                    label=f"{weight:g}",
                )
                ax.fill_between(x, y - sem, y + sem, color=color, alpha=0.16, linewidth=0)
            ax.set_ylabel(ylabel)
            ax.xaxis.set_major_locator(MultipleLocator(1000))
            ax.xaxis.set_major_formatter(FuncFormatter(_epoch_formatter))
            ax.legend(title=r"$\lambda_{dir}$", loc="best", frameon=True)

        ax.set_xlabel("Epoch")
        fig.tight_layout()
        fig.savefig(out_path.with_suffix(".png"), dpi=240)
        fig.savefig(out_path.with_suffix(".pdf"))
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    exp_names = _experiment_names()
    parser = argparse.ArgumentParser(
        description="Download decoder_prediction_csv artifacts and plot decoder performance.",
    )
    parser.add_argument("--experiment", choices=exp_names if exp_names else None, default="decoder_performance")
    parser.add_argument("--projects", nargs="+", default=None)
    parser.add_argument("--entity", default=DEFAULT_ENTITY)
    parser.add_argument("--finished-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument(
        "--cache-dir",
        default=str(_default_cache_dir()),
        help="Persistent cache directory for finished W&B runs.",
    )
    parser.add_argument(
        "--epoch-bin-size",
        type=int,
        default=0,
        help="Epoch bin size for plotting. 0 chooses an automatic bin size.",
    )
    parser.add_argument(
        "--max-points-per-line",
        type=int,
        default=80,
        help="Used only when --epoch-bin-size=0 to avoid overcrowded x-axis lines.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = _make_decoder_run_dir(_CFG)
    log = setup_logger(run_dir, name="decoder_performance")

    projects = _resolve_projects(args.experiment, args.projects)
    cache_dir = Path(args.cache_dir)
    if not cache_dir.is_absolute():
        cache_dir = (_ROOT / cache_dir).resolve()
    api = get_api()
    filters = {"state": "finished"} if args.finished_only else {}

    log.info("experiment : %s", args.experiment)
    log.info("projects   : %s", projects)
    log.info("entity     : %s", args.entity)
    log.info("run_dir    : %s", run_dir)
    log.info("cache_dir  : %s", cache_dir)

    prediction_frames: list[pd.DataFrame] = []
    history_frames: list[pd.DataFrame] = []
    scanned_runs = 0
    artifact_runs = 0

    for project in projects:
        runs = list(api.runs(f"{args.entity}/{project}", filters=filters, per_page=200))
        for run in tqdm(runs, desc=project, unit="run"):
            scanned_runs += 1
            delta_weight = _as_float(run.config.get("delta_weight"))
            csv_paths = _download_prediction_artifact(run, project, cache_dir, force=args.force)
            if csv_paths:
                artifact_runs += 1
                rows = _read_prediction_csvs(
                    csv_paths,
                    project=project,
                    run=run,
                    delta_weight=delta_weight,
                )
                if not rows.empty:
                    prediction_frames.append(rows)

            hist = _history_rows(project, run, delta_weight, cache_dir=cache_dir, force=args.force)
            if not hist.empty:
                history_frames.append(hist)

    prediction_rows = (
        pd.concat(prediction_frames, ignore_index=True)
        if prediction_frames
        else pd.DataFrame()
    )
    loss_history = (
        pd.concat(history_frames, ignore_index=True)
        if history_frames
        else pd.DataFrame()
    )
    epoch_metrics = _prediction_epoch_metrics(prediction_rows)
    if not prediction_rows.empty and not loss_history.empty and "delta_weight" in prediction_rows.columns:
        run_delta = (
            prediction_rows[["run_id", "delta_weight"]]
            .dropna()
            .drop_duplicates("run_id")
            .set_index("run_id")["delta_weight"]
        )
        missing_delta = loss_history["delta_weight"].isna()
        loss_history.loc[missing_delta, "delta_weight"] = (
            loss_history.loc[missing_delta, "run_id"].map(run_delta)
        )

    prediction_rows.to_csv(run_dir / "decoder_prediction_rows.csv", index=False)
    loss_history.to_csv(run_dir / "decoder_delta_loss_history.csv", index=False)
    epoch_metrics.to_csv(run_dir / "decoder_epoch_metrics.csv", index=False)

    if not args.no_plot:
        _plot_performance(
            epoch_metrics=epoch_metrics,
            loss_history=loss_history,
            output_base=run_dir / "decoder_performance_by_delta_weight",
            epoch_bin_size=args.epoch_bin_size,
            max_points_per_line=args.max_points_per_line,
        )

    log.info("runs scanned       : %d", scanned_runs)
    log.info("runs with artifact : %d", artifact_runs)
    log.info("prediction rows    : %d", len(prediction_rows))
    log.info("epoch metric rows  : %d", len(epoch_metrics))
    log.info("loss history rows  : %d", len(loss_history))
    log.info("outputs            : %s", run_dir)


if __name__ == "__main__":
    main()

"""
Download decoder prediction CSV artifacts and plot regression by delta_type.

Inputs:
  - W&B project runs from results/config.json experiment target_projects
  - logged artifact named decoder_prediction_csv

Outputs:
  - delta_type_prediction_rows.csv
  - delta_type_epoch_metrics.csv
  - delta_type_unseen_set_epoch_metrics.csv
  - delta_type_final_summary.csv
  - delta_type_unseen_set_summary.csv
  - regression_by_delta_type.png/.pdf
  - regression_by_delta_type_by_unseen_set.png/.pdf
  - regression_final_by_delta_type.png/.pdf
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

from sweep.wandb_utils.config import DEFAULT_ENTITY
from sweep.wandb_utils.downloader import get_api
from utils.core.run_output import load_cfg, make_run_dir, setup_logger

_CFG = load_cfg()
_DEFAULT_PROJECT = "aaai27_encoder_mgpcgrl__deltatype"
_DELTA_TYPE_ORDER = ["none", "between", "within", "both"]


def _experiment_names() -> list[str]:
    return list(_CFG.get("experiments", {}).keys())


def _safe_name(value: str) -> str:
    return "".join(c if c.isalnum() or c in "._=-" else "_" for c in value)


def _as_str(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value).strip()
    return text or None


def _canonical_unseen_set(row: pd.Series) -> str:
    for col in ("eval_unseen_games", "train_unseen_games"):
        text = _as_str(row.get(col))
        if text:
            return text
    game = _as_str(row.get("game"))
    return json.dumps([game], sort_keys=True) if game else "unknown"


def _pretty_unseen_set(value: object) -> str:
    text = _as_str(value)
    if not text or text == "unknown":
        return "unknown"
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return text
    if isinstance(parsed, list):
        return "+".join(str(v) for v in parsed)
    return str(parsed)


def _resolve_projects(experiment: str | None, projects: list[str] | None) -> list[str]:
    if projects:
        return projects
    if experiment:
        exp_cfg = _CFG.get("experiments", {}).get(experiment, {})
        configured = list(exp_cfg.get("target_projects", []))
        if configured:
            return configured
    return [_DEFAULT_PROJECT]


def _make_run_dir(cfg: dict, experiment: str | None) -> Path:
    pipeline_run_dir = os.environ.get("PIPELINE_RUN_DIR")
    pipeline_experiment = os.environ.get("PIPELINE_EXPERIMENT", "")
    if pipeline_run_dir and pipeline_experiment == experiment:
        run_dir = Path(pipeline_run_dir) / str(experiment)
        run_dir.mkdir(parents=True, exist_ok=True)
        with (run_dir / "used_config.json").open("w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
        return run_dir
    return make_run_dir("encoder_delta_type_regression", cfg=cfg)


def _default_cache_dir() -> Path:
    return _RESULTS_DIR / "wandb_projects" / "encoder_delta_type_regression_cache"


def _run_cache_dir(cache_dir: Path, project: str, run) -> Path:
    return cache_dir / project / _safe_name(run.id)


def _is_finished_run(run) -> bool:
    return str(getattr(run, "state", "")).lower() == "finished"


def _select_prediction_artifact(run) -> object | None:
    candidates = []
    for artifact in run.logged_artifacts():
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

    artifact = _select_prediction_artifact(run)
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
) -> pd.DataFrame:
    frames = []
    config_delta_type = _as_str(run.config.get("delta_type"))
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
        if "delta_type" not in df.columns:
            df["delta_type"] = config_delta_type
        else:
            df["delta_type"] = df["delta_type"].map(_as_str).fillna(config_delta_type)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _prediction_epoch_metrics(pred_rows: pd.DataFrame, min_epoch: int) -> pd.DataFrame:
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
        "delta_type",
        "epoch",
        "condition_target_raw",
        "condition_pred_raw",
    }
    missing = required.difference(df.columns)
    if missing:
        return pd.DataFrame()

    for col in ["condition_target_raw", "condition_pred_raw"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(
        subset=["delta_type", "epoch", "condition_target_raw", "condition_pred_raw"]
    )
    df = df[df["epoch"] >= min_epoch]
    if df.empty:
        return pd.DataFrame()

    df["unseen_game_set"] = df.apply(_canonical_unseen_set, axis=1)
    df["regression_abs_error_raw"] = (
        df["condition_target_raw"] - df["condition_pred_raw"]
    ).abs()
    set_min = df.groupby("unseen_game_set")["condition_target_raw"].transform("min")
    set_max = df.groupby("unseen_game_set")["condition_target_raw"].transform("max")
    set_range = (set_max - set_min).where(lambda s: s.abs() > 1e-12)
    df["target_min_for_unseen_set"] = set_min
    df["target_max_for_unseen_set"] = set_max
    df["target_range_for_unseen_set"] = set_range
    df["condition_target_minmax_norm"] = (df["condition_target_raw"] - set_min) / set_range
    df["condition_pred_minmax_norm"] = (df["condition_pred_raw"] - set_min) / set_range
    df["regression_abs_error_minmax_norm"] = (
        df["regression_abs_error_raw"] / set_range
    )
    df["regression_sq_error_raw"] = (
        df["condition_target_raw"] - df["condition_pred_raw"]
    ) ** 2
    df["regression_sq_error_minmax_norm"] = (
        df["condition_target_minmax_norm"] - df["condition_pred_minmax_norm"]
    ) ** 2
    df["condition_target_minmax_norm_sq"] = df["condition_target_minmax_norm"] ** 2
    if {"condition_target_norm", "condition_pred_norm"}.issubset(df.columns):
        df["condition_target_norm"] = pd.to_numeric(df["condition_target_norm"], errors="coerce")
        df["condition_pred_norm"] = pd.to_numeric(df["condition_pred_norm"], errors="coerce")
        df["regression_abs_error_norm"] = (
            df["condition_target_norm"] - df["condition_pred_norm"]
        ).abs()
    else:
        df["regression_abs_error_norm"] = float("nan")

    group_cols = ["project", "run_id", "run_name", "delta_type", "epoch"]
    grouped = df.groupby(group_cols, dropna=False).agg(
        regression_mae_raw=("regression_abs_error_raw", "mean"),
        regression_mae_minmax_norm=("regression_abs_error_minmax_norm", "mean"),
        regression_mse_raw=("regression_sq_error_raw", "mean"),
        regression_mse_minmax_norm=("regression_sq_error_minmax_norm", "mean"),
        target_minmax_norm_mean=("condition_target_minmax_norm", "mean"),
        target_minmax_norm_sq_mean=("condition_target_minmax_norm_sq", "mean"),
        regression_mae_norm=("regression_abs_error_norm", "mean"),
        n_predictions=("regression_abs_error_raw", "size"),
    )
    grouped = grouped.reset_index()
    grouped["regression_rmse_raw"] = grouped["regression_mse_raw"].pow(0.5)
    grouped["regression_rmse_minmax_norm"] = grouped["regression_mse_minmax_norm"].pow(0.5)
    grouped["target_minmax_norm_var"] = (
        grouped["target_minmax_norm_sq_mean"] - grouped["target_minmax_norm_mean"].pow(2)
    )
    grouped["regression_r2_minmax_norm"] = 1.0 - (
        grouped["regression_mse_minmax_norm"]
        / grouped["target_minmax_norm_var"].where(lambda s: s.abs() > 1e-12)
    )
    return grouped


def _prediction_epoch_metrics_by_unseen_set(
    pred_rows: pd.DataFrame,
    min_epoch: int,
) -> pd.DataFrame:
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
        "delta_type",
        "epoch",
        "condition_target_raw",
        "condition_pred_raw",
    }
    missing = required.difference(df.columns)
    if missing:
        return pd.DataFrame()

    for col in ["condition_target_raw", "condition_pred_raw"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(
        subset=["delta_type", "epoch", "condition_target_raw", "condition_pred_raw"]
    )
    df = df[df["epoch"] >= min_epoch]
    if df.empty:
        return pd.DataFrame()

    df["unseen_game_set"] = df.apply(_canonical_unseen_set, axis=1)
    df["regression_abs_error_raw"] = (
        df["condition_target_raw"] - df["condition_pred_raw"]
    ).abs()
    set_min = df.groupby("unseen_game_set")["condition_target_raw"].transform("min")
    set_max = df.groupby("unseen_game_set")["condition_target_raw"].transform("max")
    set_range = (set_max - set_min).where(lambda s: s.abs() > 1e-12)
    df["target_min_for_unseen_set"] = set_min
    df["target_max_for_unseen_set"] = set_max
    df["target_range_for_unseen_set"] = set_range
    df["condition_target_minmax_norm"] = (df["condition_target_raw"] - set_min) / set_range
    df["condition_pred_minmax_norm"] = (df["condition_pred_raw"] - set_min) / set_range
    df["regression_abs_error_minmax_norm"] = df["regression_abs_error_raw"] / set_range
    df["regression_sq_error_raw"] = (
        df["condition_target_raw"] - df["condition_pred_raw"]
    ) ** 2
    df["regression_sq_error_minmax_norm"] = (
        df["condition_target_minmax_norm"] - df["condition_pred_minmax_norm"]
    ) ** 2
    df["condition_target_minmax_norm_sq"] = df["condition_target_minmax_norm"] ** 2

    group_cols = [
        "project",
        "run_id",
        "run_name",
        "delta_type",
        "unseen_game_set",
        "epoch",
    ]
    grouped = df.groupby(group_cols, dropna=False).agg(
        regression_mae_raw=("regression_abs_error_raw", "mean"),
        regression_mae_minmax_norm=("regression_abs_error_minmax_norm", "mean"),
        regression_mse_raw=("regression_sq_error_raw", "mean"),
        regression_mse_minmax_norm=("regression_sq_error_minmax_norm", "mean"),
        target_minmax_norm_mean=("condition_target_minmax_norm", "mean"),
        target_minmax_norm_sq_mean=("condition_target_minmax_norm_sq", "mean"),
        target_min=("target_min_for_unseen_set", "first"),
        target_max=("target_max_for_unseen_set", "first"),
        target_range=("target_range_for_unseen_set", "first"),
        n_predictions=("regression_abs_error_raw", "size"),
    )
    grouped = grouped.reset_index()
    grouped["regression_rmse_raw"] = grouped["regression_mse_raw"].pow(0.5)
    grouped["regression_rmse_minmax_norm"] = grouped["regression_mse_minmax_norm"].pow(0.5)
    grouped["target_minmax_norm_var"] = (
        grouped["target_minmax_norm_sq_mean"] - grouped["target_minmax_norm_mean"].pow(2)
    )
    grouped["regression_r2_minmax_norm"] = 1.0 - (
        grouped["regression_mse_minmax_norm"]
        / grouped["target_minmax_norm_var"].where(lambda s: s.abs() > 1e-12)
    )
    return grouped


def _unseen_set_summary(pred_rows: pd.DataFrame, min_epoch: int) -> pd.DataFrame:
    if pred_rows.empty:
        return pd.DataFrame()
    df = pred_rows.copy()
    if "epoch_num" in df.columns:
        df["epoch"] = pd.to_numeric(df["epoch_num"], errors="coerce")
    elif "epoch" in df.columns:
        df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce") + 1
    else:
        return pd.DataFrame()
    for col in ["condition_target_raw", "condition_pred_raw"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(
        subset=["delta_type", "epoch", "condition_target_raw", "condition_pred_raw"]
    )
    df = df[df["epoch"] >= min_epoch]
    if df.empty:
        return pd.DataFrame()

    df["unseen_game_set"] = df.apply(_canonical_unseen_set, axis=1)
    set_min = df.groupby("unseen_game_set")["condition_target_raw"].transform("min")
    set_max = df.groupby("unseen_game_set")["condition_target_raw"].transform("max")
    set_range = (set_max - set_min).where(lambda s: s.abs() > 1e-12)
    df["target_min_for_unseen_set"] = set_min
    df["target_max_for_unseen_set"] = set_max
    df["target_range_for_unseen_set"] = set_range
    df["regression_abs_error_raw"] = (
        df["condition_target_raw"] - df["condition_pred_raw"]
    ).abs()
    df["regression_abs_error_minmax_norm"] = df["regression_abs_error_raw"] / set_range

    max_epoch = df.groupby("run_id")["epoch"].transform("max")
    df = df[df["epoch"] == max_epoch]
    grouped = (
        df.groupby(["delta_type", "unseen_game_set"], dropna=False)
        .agg(
            regression_mae_minmax_norm=("regression_abs_error_minmax_norm", "mean"),
            regression_mae_raw=("regression_abs_error_raw", "mean"),
            n_runs=("run_id", "nunique"),
            n_predictions=("regression_abs_error_raw", "size"),
            target_min=("target_min_for_unseen_set", "first"),
            target_max=("target_max_for_unseen_set", "first"),
            target_range=("target_range_for_unseen_set", "first"),
        )
        .reset_index()
    )
    grouped["sort_key"] = grouped["delta_type"].map(_delta_type_sort_key)
    return (
        grouped.sort_values(["unseen_game_set", "sort_key"])
        .drop(columns=["sort_key"])
        .reset_index(drop=True)
    )


def _epoch_bin(values: pd.Series, bin_size: int) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if bin_size <= 1:
        return numeric.round().astype("Int64")
    return (((numeric - 1) // bin_size) * bin_size + 1).round().astype("Int64")


def _auto_bin_size(epoch_metrics: pd.DataFrame, max_points: int) -> int:
    if epoch_metrics.empty or "epoch" not in epoch_metrics.columns or max_points <= 0:
        return 1
    epochs = pd.to_numeric(epoch_metrics["epoch"], errors="coerce").dropna()
    if epochs.empty:
        return 1
    n_unique = len(set(int(round(e)) for e in epochs.tolist()))
    return max(1, int(math.ceil(n_unique / max_points)))


def _delta_type_sort_key(value: str) -> tuple[int, str]:
    if value in _DELTA_TYPE_ORDER:
        return (_DELTA_TYPE_ORDER.index(value), value)
    return (len(_DELTA_TYPE_ORDER), value)


def _aggregate_for_plot(
    epoch_metrics: pd.DataFrame,
    value_col: str,
    epoch_bin_size: int,
) -> pd.DataFrame:
    if epoch_metrics.empty or value_col not in epoch_metrics.columns:
        return pd.DataFrame()
    work = epoch_metrics.copy()
    work["epoch_bin"] = _epoch_bin(work["epoch"], epoch_bin_size)
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    work = work.dropna(subset=["delta_type", "epoch_bin", value_col])
    if work.empty:
        return pd.DataFrame()
    grouped = (
        work.groupby(["delta_type", "epoch_bin"], dropna=False)[value_col]
        .agg(["mean", "std", "count"])
        .reset_index()
        .sort_values(["delta_type", "epoch_bin"])
    )
    grouped["sem"] = grouped["std"].fillna(0.0) / grouped["count"].pow(0.5)
    return grouped.rename(columns={"epoch_bin": "epoch", "mean": value_col})


def _final_summary(epoch_metrics: pd.DataFrame, final_window_epochs: int) -> pd.DataFrame:
    if epoch_metrics.empty:
        return pd.DataFrame()
    work = epoch_metrics.copy()
    work["epoch"] = pd.to_numeric(work["epoch"], errors="coerce")
    work["regression_mae_raw"] = pd.to_numeric(work["regression_mae_raw"], errors="coerce")
    work["regression_mae_minmax_norm"] = pd.to_numeric(
        work["regression_mae_minmax_norm"], errors="coerce"
    )
    work = work.dropna(subset=["delta_type", "epoch", "regression_mae_raw"])
    if work.empty:
        return pd.DataFrame()

    if final_window_epochs > 0:
        max_epoch = work.groupby("run_id")["epoch"].transform("max")
        work = work[work["epoch"] >= max_epoch - final_window_epochs + 1]
        run_final = (
            work.groupby(["project", "run_id", "run_name", "delta_type"], dropna=False)
            .agg(
                final_epoch=("epoch", "max"),
                final_regression_mae_raw=("regression_mae_raw", "mean"),
                final_regression_mae_minmax_norm=("regression_mae_minmax_norm", "mean"),
                final_regression_rmse_minmax_norm=("regression_rmse_minmax_norm", "mean"),
                final_regression_r2_minmax_norm=("regression_r2_minmax_norm", "mean"),
                final_regression_mae_norm=("regression_mae_norm", "mean"),
                n_epoch_points=("epoch", "size"),
                n_predictions=("n_predictions", "sum"),
            )
            .reset_index()
        )
    else:
        idx = work.groupby("run_id")["epoch"].idxmax()
        run_final = work.loc[idx].rename(
            columns={
                "epoch": "final_epoch",
                "regression_mae_raw": "final_regression_mae_raw",
                "regression_mae_minmax_norm": "final_regression_mae_minmax_norm",
                "regression_rmse_minmax_norm": "final_regression_rmse_minmax_norm",
                "regression_r2_minmax_norm": "final_regression_r2_minmax_norm",
                "regression_mae_norm": "final_regression_mae_norm",
            }
        )
        run_final["n_epoch_points"] = 1

    grouped = (
        run_final.groupby("delta_type", dropna=False)
        .agg(
            final_regression_mae_raw_mean=("final_regression_mae_raw", "mean"),
            final_regression_mae_raw_std=("final_regression_mae_raw", "std"),
            final_regression_mae_minmax_norm_mean=("final_regression_mae_minmax_norm", "mean"),
            final_regression_mae_minmax_norm_std=("final_regression_mae_minmax_norm", "std"),
            final_regression_rmse_minmax_norm_mean=("final_regression_rmse_minmax_norm", "mean"),
            final_regression_rmse_minmax_norm_std=("final_regression_rmse_minmax_norm", "std"),
            final_regression_r2_minmax_norm_mean=("final_regression_r2_minmax_norm", "mean"),
            final_regression_r2_minmax_norm_std=("final_regression_r2_minmax_norm", "std"),
            final_regression_mae_norm_mean=("final_regression_mae_norm", "mean"),
            final_epoch_mean=("final_epoch", "mean"),
            n_runs=("run_id", "nunique"),
            n_predictions=("n_predictions", "sum"),
        )
        .reset_index()
    )
    grouped["final_regression_mae_raw_sem"] = (
        grouped["final_regression_mae_raw_std"].fillna(0.0) / grouped["n_runs"].pow(0.5)
    )
    grouped["final_regression_mae_minmax_norm_sem"] = (
        grouped["final_regression_mae_minmax_norm_std"].fillna(0.0)
        / grouped["n_runs"].pow(0.5)
    )
    grouped["final_regression_rmse_minmax_norm_sem"] = (
        grouped["final_regression_rmse_minmax_norm_std"].fillna(0.0)
        / grouped["n_runs"].pow(0.5)
    )
    grouped["final_regression_r2_minmax_norm_sem"] = (
        grouped["final_regression_r2_minmax_norm_std"].fillna(0.0)
        / grouped["n_runs"].pow(0.5)
    )
    grouped["sort_key"] = grouped["delta_type"].map(_delta_type_sort_key)
    return grouped.sort_values("sort_key").drop(columns=["sort_key"]).reset_index(drop=True)


def _plot_epoch_regression(
    epoch_metrics: pd.DataFrame,
    output_path: Path,
    epoch_bin_size: int,
    max_points_per_line: int,
) -> None:
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from matplotlib.ticker import FuncFormatter

    if epoch_bin_size <= 0:
        epoch_bin_size = _auto_bin_size(epoch_metrics, max_points_per_line)
    metric_col = "regression_mae_minmax_norm"
    plot_df = _aggregate_for_plot(epoch_metrics, metric_col, epoch_bin_size)

    plt.style.use("seaborn-v0_8-whitegrid")
    mpl.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})
    colors = {
        "none": "#777777",
        "between": "#2f6fbb",
        "within": "#2a9d8f",
        "both": "#d95f02",
    }
    markers = {"none": "o", "between": "s", "within": "^", "both": "D"}

    def _epoch_formatter(value: float, _pos: int) -> str:
        if abs(value) >= 1000:
            scaled = value / 1000.0
            return f"{scaled:g}e3"
        if abs(value - round(value)) < 1e-6:
            return str(int(round(value)))
        return f"{value:g}"

    fig, ax = plt.subplots(figsize=(3.8, 2.8))
    if plot_df.empty:
        ax.text(0.5, 0.5, "No regression data", ha="center", va="center")
        ax.set_axis_off()
    else:
        delta_types = sorted(plot_df["delta_type"].dropna().unique(), key=_delta_type_sort_key)
        for delta_type in delta_types:
            ddf = plot_df[plot_df["delta_type"] == delta_type].sort_values("epoch")
            x = ddf["epoch"].astype(float).to_numpy()
            y = ddf[metric_col].astype(float).to_numpy()
            sem = ddf["sem"].fillna(0.0).astype(float).to_numpy()
            color = colors.get(delta_type, "#333333")
            ax.plot(
                x,
                y,
                color=color,
                linewidth=1.9,
                marker=markers.get(delta_type, "o"),
                markersize=4.2,
                markevery=max(1, len(x) // 12),
                label=delta_type,
            )
            ax.fill_between(x, y - sem, y + sem, color=color, alpha=0.16, linewidth=0)
        ax.set_ylabel("Unseen-Set Min-Max Normalized MAE")
        ax.xaxis.set_major_formatter(FuncFormatter(_epoch_formatter))
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            fig.legend(
                handles,
                labels,
                title="delta_type",
                loc="upper center",
                ncol=min(len(handles), 4),
                frameon=True,
                fontsize=8,
                title_fontsize=8,
            )
            fig.subplots_adjust(top=0.78)
    ax.set_xlabel("Epoch")
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=240)
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)


def _plot_epoch_regression_by_unseen_set(
    set_epoch_metrics: pd.DataFrame,
    output_path: Path,
    epoch_bin_size: int,
    max_points_per_line: int,
    metric_col: str = "regression_mae_minmax_norm",
    ylabel: str = "Unseen-Set Min-Max Normalized MAE",
    yscale: str = "linear",
) -> None:
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from matplotlib.ticker import FuncFormatter

    if epoch_bin_size <= 0:
        epoch_bin_size = _auto_bin_size(set_epoch_metrics, max_points_per_line)
    set_epoch_metrics = set_epoch_metrics.copy()
    set_epoch_metrics["epoch"] = pd.to_numeric(set_epoch_metrics["epoch"], errors="coerce")
    set_epoch_metrics = set_epoch_metrics[set_epoch_metrics["epoch"] > 0]
    plt.style.use("seaborn-v0_8-whitegrid")
    mpl.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})
    colors = {
        "none": "#777777",
        "between": "#2f6fbb",
        "within": "#2a9d8f",
        "both": "#d95f02",
    }
    markers = {"none": "o", "between": "s", "within": "^", "both": "D"}

    def _epoch_formatter(value: float, _pos: int) -> str:
        if abs(value) >= 1000:
            scaled = value / 1000.0
            return f"{scaled:g}e3"
        if abs(value - round(value)) < 1e-6:
            return str(int(round(value)))
        return f"{value:g}"

    unseen_sets = (
        sorted(set_epoch_metrics["unseen_game_set"].dropna().unique(), key=_pretty_unseen_set)
        if not set_epoch_metrics.empty and "unseen_game_set" in set_epoch_metrics.columns
        else []
    )
    n_cols = min(3, max(1, len(unseen_sets)))
    n_rows = max(1, int(math.ceil(max(1, len(unseen_sets)) / n_cols)))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.0 * n_cols, 2.45 * n_rows),
        sharex=True,
        sharey=False,
        squeeze=False,
    )

    legend_handles = []
    legend_labels = []
    if not unseen_sets:
        ax = axes[0][0]
        ax.text(0.5, 0.5, "No regression data", ha="center", va="center")
        ax.set_axis_off()
    else:
        for idx, unseen_set in enumerate(unseen_sets):
            ax = axes[idx // n_cols][idx % n_cols]
            subset = set_epoch_metrics[set_epoch_metrics["unseen_game_set"] == unseen_set]
            plot_df = _aggregate_for_plot(subset, metric_col, epoch_bin_size)
            delta_types = sorted(
                plot_df["delta_type"].dropna().unique(),
                key=_delta_type_sort_key,
            )
            for delta_type in delta_types:
                ddf = plot_df[plot_df["delta_type"] == delta_type].sort_values("epoch")
                x = ddf["epoch"].astype(float).to_numpy()
                y = ddf[metric_col].astype(float).to_numpy()
                sem = ddf["sem"].fillna(0.0).astype(float).to_numpy()
                color = colors.get(delta_type, "#333333")
                line = ax.plot(
                    x,
                    y,
                    color=color,
                    linewidth=1.7,
                    marker=markers.get(delta_type, "o"),
                    markersize=3.8,
                    markevery=max(1, len(x) // 10),
                    label=delta_type,
                )[0]
                ax.fill_between(x, y - sem, y + sem, color=color, alpha=0.14, linewidth=0)
                if delta_type not in legend_labels:
                    legend_handles.append(line)
                    legend_labels.append(delta_type)
            ax.set_title(_pretty_unseen_set(unseen_set), fontsize=9)
            if yscale != "linear":
                ax.set_yscale(yscale)
            ax.xaxis.set_major_formatter(FuncFormatter(_epoch_formatter))
            ax.tick_params(axis="both", labelsize=8)

        for idx in range(len(unseen_sets), n_rows * n_cols):
            axes[idx // n_cols][idx % n_cols].set_axis_off()

    for row in axes:
        row[0].set_ylabel(ylabel)
    for ax in axes[-1]:
        ax.set_xlabel("Epoch")
    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            title="delta_type",
            loc="upper center",
            ncol=min(len(legend_handles), 4),
            frameon=True,
            fontsize=8,
            title_fontsize=8,
        )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=240)
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)


def _plot_final_summary(summary: pd.DataFrame, output_path: Path) -> None:
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    plt.style.use("seaborn-v0_8-whitegrid")
    mpl.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})
    colors = {
        "none": "#777777",
        "between": "#2f6fbb",
        "within": "#2a9d8f",
        "both": "#d95f02",
    }
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    if summary.empty:
        ax.text(0.5, 0.5, "No final summary", ha="center", va="center")
        ax.set_axis_off()
    else:
        labels = summary["delta_type"].astype(str).tolist()
        values = summary["final_regression_mae_minmax_norm_mean"].astype(float).to_numpy()
        sem = summary["final_regression_mae_minmax_norm_sem"].fillna(0.0).astype(float).to_numpy()
        xs = list(range(len(labels)))
        bars = ax.bar(
            xs,
            values,
            yerr=sem,
            color=[colors.get(label, "#333333") for label in labels],
            edgecolor="none",
            alpha=0.9,
            capsize=3,
        )
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(value, 1e-8) * 0.02,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        ax.set_xticks(xs, labels)
        ax.set_ylabel("Final Unseen-Set Min-Max Normalized MAE")
        ax.set_xlabel("delta_type")
        ax.grid(axis="x", visible=False)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=240)
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    exp_names = _experiment_names()
    parser = argparse.ArgumentParser(
        description="Plot encoder regression performance by delta_type.",
    )
    parser.add_argument(
        "--experiment",
        choices=exp_names if exp_names else None,
        default="encoder_delta_type_regression",
    )
    parser.add_argument("--projects", nargs="+", default=None)
    parser.add_argument("--entity", default=DEFAULT_ENTITY)
    parser.add_argument("--finished-only", action="store_true")
    parser.add_argument("--force", action="store_true")
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
    parser.add_argument(
        "--final-window-epochs",
        type=int,
        default=0,
        help="Average the last N epochs per run for the final bar plot. 0 uses the last epoch only.",
    )
    parser.add_argument(
        "--min-epoch",
        type=int,
        default=0,
        help="Exclude prediction rows before this epoch from metrics and plots.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = _make_run_dir(_CFG, args.experiment)
    log = setup_logger(run_dir, name="encoder_delta_type_regression")

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
    log.info("min_epoch  : %d", args.min_epoch)

    prediction_frames: list[pd.DataFrame] = []
    scanned_runs = 0
    artifact_runs = 0

    for project in projects:
        runs = list(api.runs(f"{args.entity}/{project}", filters=filters, per_page=200))
        for run in tqdm(runs, desc=project, unit="run"):
            scanned_runs += 1
            csv_paths = _download_prediction_artifact(run, project, cache_dir, force=args.force)
            if not csv_paths:
                continue
            artifact_runs += 1
            rows = _read_prediction_csvs(csv_paths, project=project, run=run)
            if not rows.empty:
                prediction_frames.append(rows)

    prediction_rows = (
        pd.concat(prediction_frames, ignore_index=True)
        if prediction_frames
        else pd.DataFrame()
    )
    epoch_metrics = _prediction_epoch_metrics(prediction_rows, args.min_epoch)
    set_epoch_metrics = _prediction_epoch_metrics_by_unseen_set(
        prediction_rows,
        args.min_epoch,
    )
    final_summary = _final_summary(epoch_metrics, args.final_window_epochs)
    unseen_set_summary = _unseen_set_summary(prediction_rows, args.min_epoch)

    prediction_rows.to_csv(run_dir / "delta_type_prediction_rows.csv", index=False)
    epoch_metrics.to_csv(run_dir / "delta_type_epoch_metrics.csv", index=False)
    set_epoch_metrics.to_csv(run_dir / "delta_type_unseen_set_epoch_metrics.csv", index=False)
    final_summary.to_csv(run_dir / "delta_type_final_summary.csv", index=False)
    unseen_set_summary.to_csv(run_dir / "delta_type_unseen_set_summary.csv", index=False)

    if not args.no_plot:
        _plot_epoch_regression(
            epoch_metrics,
            run_dir / "regression_by_delta_type.png",
            epoch_bin_size=args.epoch_bin_size,
            max_points_per_line=args.max_points_per_line,
        )
        _plot_epoch_regression_by_unseen_set(
            set_epoch_metrics,
            run_dir / "regression_by_delta_type_by_unseen_set.png",
            epoch_bin_size=args.epoch_bin_size,
            max_points_per_line=args.max_points_per_line,
            metric_col="regression_mae_raw",
            ylabel="MAE",
        )
        _plot_epoch_regression_by_unseen_set(
            set_epoch_metrics,
            run_dir / "regression_mae_log_by_delta_type_by_unseen_set.png",
            epoch_bin_size=args.epoch_bin_size,
            max_points_per_line=args.max_points_per_line,
            metric_col="regression_mae_raw",
            ylabel="MAE (log)",
            yscale="log",
        )
        _plot_epoch_regression_by_unseen_set(
            set_epoch_metrics,
            run_dir / "regression_rmse_by_delta_type_by_unseen_set.png",
            epoch_bin_size=args.epoch_bin_size,
            max_points_per_line=args.max_points_per_line,
            metric_col="regression_rmse_raw",
            ylabel="RMSE",
        )
        _plot_epoch_regression_by_unseen_set(
            set_epoch_metrics,
            run_dir / "regression_rmse_log_by_delta_type_by_unseen_set.png",
            epoch_bin_size=args.epoch_bin_size,
            max_points_per_line=args.max_points_per_line,
            metric_col="regression_rmse_raw",
            ylabel="RMSE (log)",
            yscale="log",
        )
        _plot_epoch_regression_by_unseen_set(
            set_epoch_metrics,
            run_dir / "regression_r2_by_delta_type_by_unseen_set.png",
            epoch_bin_size=args.epoch_bin_size,
            max_points_per_line=args.max_points_per_line,
            metric_col="regression_r2_minmax_norm",
            ylabel="R2",
        )
        _plot_final_summary(final_summary, run_dir / "regression_final_by_delta_type.png")

    log.info("runs scanned       : %d", scanned_runs)
    log.info("runs with artifact : %d", artifact_runs)
    log.info("prediction rows    : %d", len(prediction_rows))
    log.info("epoch metric rows  : %d", len(epoch_metrics))
    log.info("set epoch rows     : %d", len(set_epoch_metrics))
    log.info("unseen set rows    : %d", len(unseen_set_summary))
    log.info("delta_types        : %s", sorted(epoch_metrics.get("delta_type", pd.Series(dtype=str)).dropna().unique(), key=_delta_type_sort_key))
    log.info("outputs            : %s", run_dir)


if __name__ == "__main__":
    main()

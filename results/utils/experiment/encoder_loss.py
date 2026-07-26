"""
Encoder training loss analysis across methods — artifact-based.

Downloads `decoder_prediction_csv` artifacts from encoder training runs and
computes per-unseen-domain MSE over training epochs.

Inputs:
  W&B artifact `decoder_prediction_csv` (type=dataset) from each target project.
  The artifact contains unseen_regression_predictions_<uid>.csv with columns:
    epoch_num, eval_unseen_games, reward_enum_target,
    condition_target_raw, condition_pred_raw

Outputs:
  encoder_loss_rows.csv             : long-format per-sample rows (all projects)
  encoder_loss_epoch_metrics.csv    : MSE per (method, unseen_game, epoch)
  encoder_loss_domain_table.csv     : pivot (domain × method) — final epoch val MSE
  encoder_loss_domain_min_table.csv : pivot (domain × method) — best epoch val MSE
  encoder_loss_regression_final_table.csv
                                      : domain × method regression metrics at final epoch
  encoder_loss_regression_best_table.csv
                                      : domain × method regression metrics at best epoch
  encoder_loss_appendix.md          : Markdown tables for appendix
  encoder_loss_curves_<game>.png    : loss curve per unseen game
  encoder_loss_curves_overall.png   : overall unseen MSE curves

Usage:
  python results/utils/experiment/encoder_loss.py --experiment encoder_loss
  python results/run_pipeline.py --experiment encoder_loss --steps 15
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import shutil
import sys
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).resolve().parent
_RESULTS_DIR = _HERE.parent.parent
_ROOT = _RESULTS_DIR.parent

for _p in (str(_RESULTS_DIR), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sweep.wandb_utils.config import DEFAULT_ENTITY
from sweep.wandb_utils.downloader import get_api

_RUN_OUTPUT_PATH = _RESULTS_DIR / "utils" / "core" / "run_output.py"
_RUN_OUTPUT_SPEC = importlib.util.spec_from_file_location("_results_run_output", _RUN_OUTPUT_PATH)
if _RUN_OUTPUT_SPEC is None or _RUN_OUTPUT_SPEC.loader is None:
    raise ImportError(f"Cannot load run_output helpers from {_RUN_OUTPUT_PATH}")
_RUN_OUTPUT = importlib.util.module_from_spec(_RUN_OUTPUT_SPEC)
_RUN_OUTPUT_SPEC.loader.exec_module(_RUN_OUTPUT)
load_cfg = _RUN_OUTPUT.load_cfg
make_run_dir = _RUN_OUTPUT.make_run_dir
setup_logger = _RUN_OUTPUT.setup_logger

_CFG = load_cfg()
_ARTIFACT_NAME = "decoder_prediction_csv"
_ARTIFACT_TYPE = "dataset"

_PRETENDARD_CANDIDATES: tuple[Path, ...] = (
    Path(os.environ.get("PRETENDARD_MEDIUM_PATH", "")),
    Path(os.environ.get("PRETENDARD_REGULAR_PATH", "")),
    Path("/Users/inchang/Desktop/MuCap-fin2/mucap/fonts/Pretendard-Medium.otf"),
    Path.home() / "Library/Fonts/Pretendard-Medium.otf",
    Path.home() / "Library/Fonts/Pretendard-Regular.otf",
    Path("/Library/Fonts/Pretendard-Medium.otf"),
    Path("/Library/Fonts/Pretendard-Regular.otf"),
)


# ── helpers ──────────────────────────────────────────────────────────────────

def _safe_name(value: str) -> str:
    return "".join(c if c.isalnum() or c in "._=-" else "_" for c in value)


def _is_finished(run) -> bool:
    return str(getattr(run, "state", "")).lower() == "finished"


def _experiment_cfg(experiment: str | None) -> dict:
    if not experiment:
        return {}
    return _CFG.get("experiments", {}).get(experiment, {})


def _resolve_projects(experiment: str | None, projects: list[str] | None) -> list[str]:
    if projects:
        return projects
    if experiment:
        exp_cfg = _experiment_cfg(experiment)
        configured = list(exp_cfg.get("target_projects", []))
        if configured:
            return configured
    raise SystemExit(
        "No target projects found. Pass --projects or configure 'encoder_loss' in config.json."
    )


def _display_name(project: str, exp_cfg: dict) -> str:
    return exp_cfg.get("project_display_names", {}).get(project, project)


def _cache_dir() -> Path:
    return _RESULTS_DIR / "wandb_projects" / "encoder_loss_cache"


def _run_cache_dir(cache_dir: Path, project: str, run_id: str) -> Path:
    return cache_dir / project / _safe_name(run_id)


def _apply_plot_style(mpl) -> None:
    font_family = "Pretendard"
    from matplotlib import font_manager
    for font_path in _PRETENDARD_CANDIDATES:
        if not str(font_path) or not font_path.is_file():
            continue
        font_manager.fontManager.addfont(str(font_path))
        font_family = font_manager.FontProperties(fname=str(font_path)).get_name()
        break
    mpl.rcParams.update({
        "font.family": font_family,
        "font.sans-serif": [font_family, "Pretendard", "Arial", "Helvetica", "DejaVu Sans"],
        "font.weight": "regular",
        "axes.labelweight": "regular",
        "axes.titleweight": "regular",
    })


# ── artifact download ─────────────────────────────────────────────────────────

def _select_artifact(run):
    """Return latest decoder_prediction_csv artifact for a run, or None."""
    try:
        artifacts = list(run.logged_artifacts())
    except Exception:
        return None
    candidates = [
        a for a in artifacts
        if a.name.split(":", 1)[0] == _ARTIFACT_NAME and a.type == _ARTIFACT_TYPE
    ]
    return candidates[-1] if candidates else None


def _download_artifact_csvs(
    run,
    project: str,
    cache_dir: Path,
    force: bool,
    log,
) -> list[Path]:
    """Download artifact CSVs for one run and return local paths."""
    run_dir = _run_cache_dir(cache_dir, project, run.id)
    artifact_dir = run_dir / _ARTIFACT_NAME

    if _is_finished(run) and artifact_dir.exists() and not force:
        cached = sorted(artifact_dir.rglob("*.csv"))
        if cached:
            log.debug("  Cache hit: %s (%d csv)", artifact_dir, len(cached))
            return cached

    artifact = _select_artifact(run)
    if artifact is None:
        log.warning("  No '%s' artifact found for run %s", _ARTIFACT_NAME, run.id)
        return []

    if force and artifact_dir.exists():
        shutil.rmtree(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    if not any(artifact_dir.glob("*.csv")):
        log.info("  Downloading artifact for run %s …", run.id)
        artifact.download(root=str(artifact_dir))

    return sorted(artifact_dir.rglob("*.csv"))


def _canonical_game_set(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "unknown"
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return "unknown"
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            vals = sorted(str(v).strip() for v in parsed if str(v).strip())
            return "+".join(vals) if vals else "unknown"
    except Exception:
        pass
    return text


def _read_csv(path: Path, project: str, run_name: str, method: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception:
        return pd.DataFrame()
    if df.empty:
        return df

    # Normalise epoch column
    if "epoch_num" in df.columns:
        df["epoch"] = pd.to_numeric(df["epoch_num"], errors="coerce")
    elif "epoch" in df.columns:
        df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")
    else:
        return pd.DataFrame()

    df = df[df["epoch"] > 0].copy()
    if df.empty:
        return df

    # Normalise unseen_game column
    for col in ("eval_unseen_games", "train_unseen_games"):
        if col in df.columns:
            df["unseen_game"] = df[col].map(_canonical_game_set)
            break
    else:
        df["unseen_game"] = "unknown"

    # reward_enum
    if "reward_enum_target" in df.columns:
        df["reward_enum"] = pd.to_numeric(df["reward_enum_target"], errors="coerce")
    else:
        df["reward_enum"] = -1

    df["project"] = project
    df["run_name"] = run_name
    df["method"] = method
    df["source_csv"] = path.name
    return df


# ── metric computation ────────────────────────────────────────────────────────

def _r2(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Coefficient of determination R²."""
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    if ss_tot < 1e-12:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def _pearson_r(y_true: pd.Series, y_pred: pd.Series) -> float:
    if len(y_true) < 2:
        return float("nan")
    std_t = float(y_true.std(ddof=0))
    std_p = float(y_pred.std(ddof=0))
    if std_t < 1e-12 or std_p < 1e-12:
        return float("nan")
    return float(y_true.corr(y_pred))


def _epoch_metrics(rows: pd.DataFrame) -> pd.DataFrame:
    """Compute regression metrics per (method, run_name, unseen_game, epoch).

    Errors are computed on **per-(unseen_game, reward_enum) min-max normalized**
    condition values so that domains with different raw scales are comparable.
    """
    if rows.empty:
        return pd.DataFrame()

    needed = {"condition_target_raw", "condition_pred_raw", "epoch", "unseen_game"}
    if not needed.issubset(rows.columns):
        return pd.DataFrame()

    df = rows.dropna(subset=list(needed)).copy()
    df["condition_target_raw"] = pd.to_numeric(df["condition_target_raw"], errors="coerce")
    df["condition_pred_raw"]   = pd.to_numeric(df["condition_pred_raw"],   errors="coerce")
    df = df.dropna(subset=["condition_target_raw", "condition_pred_raw"])

    # ── per-(unseen_game, reward_enum) min-max normalization ──────────────────
    norm_group = ["unseen_game"]
    if "reward_enum" in df.columns:
        norm_group = ["unseen_game", "reward_enum"]

    target_min = df.groupby(norm_group)["condition_target_raw"].transform("min")
    target_max = df.groupby(norm_group)["condition_target_raw"].transform("max")
    scale = (target_max - target_min).clip(lower=1e-12)

    df["target_norm"] = (df["condition_target_raw"] - target_min) / scale
    df["pred_norm"]   = (df["condition_pred_raw"]   - target_min) / scale

    group_keys = ["method", "project", "run_name", "unseen_game", "epoch"]
    records = []
    for keys, grp in df.groupby(group_keys, dropna=False):
        y_t = grp["target_norm"]
        y_p = grp["pred_norm"]
        mse  = float(((y_t - y_p) ** 2).mean())
        mae  = float((y_t - y_p).abs().mean())
        r2   = _r2(y_t, y_p)
        r    = _pearson_r(y_t, y_p)
        records.append({
            **dict(zip(group_keys, keys if isinstance(keys, tuple) else (keys,))),
            "mse": mse,
            "rmse": mse ** 0.5,
            "mae": mae,
            "r2": r2,
            "pearson_r": r,
            "n": len(grp),
        })
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records).sort_values(["method", "unseen_game", "epoch"])


# ── pivot tables ──────────────────────────────────────────────────────────────

def _domain_pivot(
    metrics: pd.DataFrame,
    method_order: list[str],
    agg: str = "last",  # "last" → final epoch, "min" → best epoch
    value: str = "mse",
) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()

    if agg == "last":
        idx = metrics.groupby(["method", "run_name", "unseen_game"])["epoch"].idxmax()
    else:
        idx = metrics.groupby(["method", "run_name", "unseen_game"])[value].idxmin()

    subset = metrics.loc[idx]
    summary = (
        subset.groupby(["method", "unseen_game"])[value]
        .mean()
        .reset_index()
    )
    pivot = summary.pivot(index="unseen_game", columns="method", values=value)
    ordered = [m for m in method_order if m in pivot.columns]
    rest = [c for c in pivot.columns if c not in ordered]
    pivot = pivot[ordered + rest].sort_index()
    return pivot.round(4)


def _selected_epoch_records(
    metrics: pd.DataFrame,
    agg: str,
    value: str = "mse",
) -> pd.DataFrame:
    """Select one epoch per (method, project, run_name, domain)."""
    if metrics.empty:
        return pd.DataFrame()

    group_keys = ["method", "project", "run_name", "unseen_game"]
    if agg == "last":
        idx = metrics.groupby(group_keys, dropna=False)["epoch"].idxmax()
    elif agg == "min":
        idx = metrics.groupby(group_keys, dropna=False)[value].idxmin()
    else:
        raise ValueError(f"Unknown epoch selector: {agg}")

    return metrics.loc[idx].copy()


def _regression_summary_table(
    metrics: pd.DataFrame,
    method_order: list[str],
    agg: str,
) -> pd.DataFrame:
    """Long-format table: (domain × method) → MSE, RMSE, MAE, R², Pearson r."""
    selected = _selected_epoch_records(metrics, agg=agg)
    if selected.empty:
        return pd.DataFrame()

    grouped = (
        selected
        .groupby(["unseen_game", "method"], dropna=False)
        .agg(
            mse=("mse", "mean"),
            rmse=("rmse", "mean"),
            mae=("mae", "mean"),
            r2=("r2", "mean"),
            pearson_r=("pearson_r", "mean"),
        )
        .reset_index()
    )

    order = {method: i for i, method in enumerate(method_order)}
    grouped["_method_order"] = grouped["method"].map(order).fillna(len(order))
    grouped = grouped.sort_values(["unseen_game", "_method_order"]).drop(columns="_method_order")
    return grouped.round(4)


def _overall_table(metrics: pd.DataFrame, method_order: list[str]) -> pd.DataFrame:
    """Mean metrics per (method, epoch) averaged over all unseen games."""
    if metrics.empty:
        return pd.DataFrame()
    cols = [c for c in ["mse", "mae", "rmse", "r2", "pearson_r"] if c in metrics.columns]
    grouped = (
        metrics.groupby(["method", "epoch"], dropna=False)[cols]
        .mean()
        .reset_index()
        .sort_values(["method", "epoch"])
    )
    return grouped


# Metrics shown in the detailed table (order matters)
_REPORT_METRICS: list[tuple[str, str, int]] = [
    ("mse",       "MSE",       4),
    ("rmse",      "RMSE",      4),
    ("mae",       "MAE",       4),
    ("r2",        "R²",        4),
    ("pearson_r", "Pearson r", 4),
]


def _fmt_cell(v: object, decimals: int = 4) -> str:
    if v is None:
        return "—"
    try:
        f = float(v)  # type: ignore[arg-type]
        return f"{f:.{decimals}f}" if not math.isnan(f) else "—"
    except (TypeError, ValueError):
        return str(v)


def _regression_table_to_md(df: pd.DataFrame, decimals: int = 4) -> str:
    """Render (domain, method, metrics) long-format DataFrame as Markdown.

    Rows are grouped visually by domain (first row of each group shows domain name).
    """
    if df.empty:
        return "_No data._"

    games = sorted(df["unseen_game"].dropna().unique())
    all_methods = df["method"].dropna().unique().tolist()
    metric_cols = [(col, label, dec) for col, label, dec in _REPORT_METRICS if col in df.columns]
    metric_labels = [label for _, label, _ in metric_cols]

    header = "| Domain | Method | " + " | ".join(metric_labels) + " |"
    sep    = "| :--- | :--- | " + " | ".join([":---:"] * len(metric_cols)) + " |"
    rows = [header, sep]

    for game in games:
        gdf = df[df["unseen_game"] == game]
        first = True
        for method in all_methods:
            mrow = gdf[gdf["method"] == method]
            domain_cell = f"**{game}**" if first else ""
            first = False
            if mrow.empty:
                cells = ["—"] * len(metric_cols)
            else:
                r = mrow.iloc[0]
                cells = [_fmt_cell(r.get(col, float("nan")), dec) for col, _, dec in metric_cols]
            rows.append(f"| {domain_cell} | {method} | " + " | ".join(cells) + " |")

    return "\n".join(rows)


# ── Markdown report ───────────────────────────────────────────────────────────

def _pivot_to_md(pivot: pd.DataFrame, decimals: int = 4) -> str:
    cols = list(pivot.columns)
    header = "| Domain | " + " | ".join(cols) + " |"
    sep = "| :--- | " + " | ".join([":---:"] * len(cols)) + " |"
    rows = [header, sep]
    for game, row in pivot.iterrows():
        cells = [_fmt_cell(row[col], decimals) for col in cols]
        rows.append(f"| {game} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def _save_markdown_report(
    pivot_final: pd.DataFrame,
    pivot_min: pd.DataFrame,
    regression_final: pd.DataFrame,
    regression_best: pd.DataFrame,
    overall: pd.DataFrame,
    method_order: list[str],
    run_dir: Path,
    log,
    decimals: int = 4,
) -> None:
    lines: list[str] = [
        "# Encoder Training Loss — Appendix",
        "",
        "Comparison of encoder validation regression metrics across training methods.",
        "**MSE / RMSE / MAE**: lower is better. **R² / Pearson r**: higher is better.",
        "All error metrics are computed on **per-(domain, reward\\_enum) min-max normalized** condition values",
        "so that different reward scales across domains are comparable (range ∈ [0, 1]).",
        "Evaluated on **unseen** domains only.",
        "",
        "**Methods compared:**",
    ]
    for m in method_order:
        lines.append(f"- {m}")
    lines += ["", "---", ""]

    # Best epoch table
    lines += [
        "## Regression Metrics by Unseen Domain and Method (Best Epoch)",
        "",
        "Metrics recorded at the **epoch with minimum MSE** per run/domain, then averaged within each (domain, method) group.",
        "",
        _regression_table_to_md(regression_best, decimals),
        "",
    ]

    md_path = run_dir / "encoder_loss_appendix.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("Saved Markdown report: %s", md_path)


# ── plots ─────────────────────────────────────────────────────────────────────

_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]


def _plot_per_domain(metrics: pd.DataFrame, method_order: list[str], run_dir: Path, log) -> None:
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        _apply_plot_style(matplotlib)
    except ImportError:
        log.warning("matplotlib not available — skipping plots")
        return

    methods = [m for m in method_order if m in metrics["method"].unique()]
    games = sorted(metrics["unseen_game"].dropna().unique())

    for game in games:
        gdf = metrics[metrics["unseen_game"] == game]
        fig, ax = plt.subplots(figsize=(6, 4))
        for i, method in enumerate(methods):
            mdf = (
                gdf[gdf["method"] == method]
                .groupby("epoch")["mse"].mean()
                .reset_index().sort_values("epoch")
            )
            if mdf.empty:
                continue
            ax.plot(mdf["epoch"], mdf["mse"], label=method,
                    color=_COLORS[i % len(_COLORS)], linewidth=1.5, marker="o", markersize=2)
        ax.set_xlabel("Epoch"); ax.set_ylabel("Val MSE")
        ax.set_title(f"Encoder Val MSE — {game} (unseen)")
        ax.legend(fontsize=8); ax.grid(True, linestyle="--", alpha=0.4)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(run_dir / f"encoder_loss_curves_{_safe_name(game)}.{ext}", dpi=300, bbox_inches="tight")
        plt.close(fig)
        log.info("  Saved: encoder_loss_curves_%s", game)


def _plot_overall(overall: pd.DataFrame, method_order: list[str], run_dir: Path, log) -> None:
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        _apply_plot_style(matplotlib)
    except ImportError:
        return
    if overall.empty or "mse" not in overall.columns:
        return
    methods = [m for m in method_order if m in overall["method"].unique()]
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, method in enumerate(methods):
        mdf = overall[overall["method"] == method].sort_values("epoch")
        ax.plot(mdf["epoch"], mdf["mse"], label=method,
                color=_COLORS[i % len(_COLORS)], linewidth=1.8)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Val MSE (overall)")
    ax.set_title("Encoder Unseen Val MSE — Method Comparison")
    ax.legend(fontsize=8); ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(run_dir / f"encoder_loss_curves_overall.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved: encoder_loss_curves_overall")


def _reward_enum_labels() -> dict[int, str]:
    raw = _CFG.get("reward_enums", {}).get("labels", {})
    return {int(k): str(v) for k, v in raw.items() if str(k).isdigit()}


def _plot_scatter_per_domain(
    combined: pd.DataFrame,
    metrics: pd.DataFrame,
    method_order: list[str],
    run_dir: Path,
    log,
) -> None:
    """One figure per method: cols=reward_enums(5), all games overlaid in different colors.

    Shared legend at the bottom shows game names.
    Uses best-epoch data (min MSE epoch per run×domain).
    Points are plotted on per-(domain, reward_enum) normalized scale [0, 1].
    """
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        _apply_plot_style(matplotlib)
    except ImportError:
        log.warning("matplotlib not available — skipping scatter plots")
        return

    if "reward_enum" not in combined.columns:
        log.warning("  No reward_enum column — skipping scatter plots")
        return

    re_labels = _reward_enum_labels()
    all_re = sorted(combined["reward_enum"].dropna().unique().astype(int))
    n_cols = max(5, len(all_re))
    methods = [m for m in method_order if m in combined["method"].unique()]
    games = sorted(combined["unseen_game"].dropna().unique())

    # Color palette per game
    GAME_PALETTE = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#bcbd22",
    ]
    game_colors = {g: GAME_PALETTE[i % len(GAME_PALETTE)] for i, g in enumerate(games)}

    # ── select best epoch rows ────────────────────────────────────────────────
    if metrics.empty:
        return
    best_idx = metrics.groupby(["method", "run_name", "unseen_game"])["mse"].idxmin()
    best_info = metrics.loc[best_idx, ["method", "run_name", "unseen_game", "epoch"]]

    df = combined.copy()
    df["condition_target_raw"] = pd.to_numeric(df["condition_target_raw"], errors="coerce")
    df["condition_pred_raw"]   = pd.to_numeric(df["condition_pred_raw"],   errors="coerce")
    df["reward_enum"] = pd.to_numeric(df["reward_enum"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["condition_target_raw", "condition_pred_raw", "epoch"])

    # per-(domain, reward_enum) min-max normalize
    t_min = df.groupby(["unseen_game", "reward_enum"])["condition_target_raw"].transform("min")
    t_max = df.groupby(["unseen_game", "reward_enum"])["condition_target_raw"].transform("max")
    df["target_norm"] = (df["condition_target_raw"] - t_min) / (t_max - t_min).clip(lower=1e-12)
    df["pred_norm"]   = (df["condition_pred_raw"]   - t_min) / (t_max - t_min).clip(lower=1e-12)

    merged = df.merge(
        best_info.rename(columns={"epoch": "best_epoch"}),
        on=["method", "run_name", "unseen_game"],
        how="inner",
    )
    merged = merged[merged["epoch"] == merged["best_epoch"]]

    scatter_dir = run_dir / "scatter"
    scatter_dir.mkdir(exist_ok=True)

    col_w, row_h = 1.1, 1.1

    for method in methods:
        mdf = merged[merged["method"] == method]
        if mdf.empty:
            continue

        fig, axes = plt.subplots(
            1, n_cols,
            figsize=(col_w * n_cols, row_h),
            squeeze=False,
        )

        for c_idx, re in enumerate(all_re[:n_cols]):
            ax = axes[0][c_idx]
            sub_re = mdf[mdf["reward_enum"] == re]

            for game in games:
                sub = sub_re[sub_re["unseen_game"] == game]
                if sub.empty:
                    continue
                ax.scatter(
                    sub["target_norm"], sub["pred_norm"],
                    s=2, alpha=0.35, linewidths=0,
                    color=game_colors[game],
                    label=game.capitalize(),
                    zorder=2,
                )

            ax.plot([0, 1], [0, 1], color="black", linewidth=0.7, linestyle="--", alpha=0.5, zorder=3)

            # Overall regression line + stats (all games combined for this RE)
            if not sub_re.empty:
                all_t = pd.to_numeric(sub_re["target_norm"], errors="coerce").dropna()
                all_p = pd.to_numeric(sub_re["pred_norm"],   errors="coerce").dropna()
                common = sub_re[["target_norm", "pred_norm"]].dropna()
                if len(common) >= 2:
                    import numpy as _np
                    xt = common["target_norm"].values
                    xp = common["pred_norm"].values
                    coef = _np.polyfit(xt, xp, 1)
                    x_line = _np.array([0, 1])
                    ax.plot(x_line, _np.polyval(coef, x_line),
                            color="dimgray", linewidth=0.9, linestyle="-", alpha=0.8, zorder=4)
                    # Pearson r and R²
                    r_val = _pearson_r(pd.Series(xt), pd.Series(xp))
                    r2_val = _r2(pd.Series(xt), pd.Series(xp))
                    if not math.isnan(r_val) and not math.isnan(r2_val):
                        ax.text(0.04, 0.94,
                                f"r={r_val:.2f}\n$R^2$={r2_val:.2f}",
                                transform=ax.transAxes,
                                fontsize=4.5, va="top", ha="left",
                                color="dimgray",
                                linespacing=1.3)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
            ax.tick_params(labelsize=5, length=2, pad=1)
            re_label = re_labels.get(int(re), f"RE{re}")
            ax.set_title(re_label, fontsize=6.5, pad=2)
            ax.set_xlabel("Normalized Target", fontsize=5.5, labelpad=1)
            if c_idx == 0:
                ax.set_ylabel("Normalized Pred", fontsize=5.5, labelpad=1)
            else:
                ax.set_ylabel("")

        handles = [
            plt.Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=game_colors[g],
                       markersize=4, label=g.capitalize())
            for g in games
        ]
        fig.legend(
            handles=handles,
            loc="lower center",
            ncol=len(games),
            fontsize=5.5,
            frameon=False,
            bbox_to_anchor=(0.5, -0.24),
            handletextpad=0.2,
            columnspacing=0.6,
        )

        fig.tight_layout(pad=0.15, w_pad=0.15, h_pad=0.15)

        import re as _re
        clean = _re.sub(r"[^a-z0-9]+", "_", method.lower()).strip("_")
        fname = f"scatter_{clean}"
        for ext in ("png", "pdf"):
            fig.savefig(scatter_dir / f"{fname}.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.03)
        plt.close(fig)
        log.info("  Saved scatter: %s", fname)




def _make_run_dir(exp_cfg: dict, experiment: str | None) -> Path:
    pipeline_run_dir = os.environ.get("PIPELINE_RUN_DIR")
    pipeline_experiment = os.environ.get("PIPELINE_EXPERIMENT", "")
    if pipeline_run_dir:
        run_dir = Path(pipeline_run_dir) / (pipeline_experiment or "encoder_loss")
        run_dir.mkdir(parents=True, exist_ok=True)
        with (run_dir / "used_config.json").open("w", encoding="utf-8") as f:
            json.dump(exp_cfg, f, indent=2, ensure_ascii=False)
        return run_dir
    return make_run_dir("encoder_loss", cfg=exp_cfg)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    exp_names = list(_CFG.get("experiments", {}).keys())
    p = argparse.ArgumentParser(description="Encoder loss analysis via artifact CSV download")
    p.add_argument("--experiment", default="encoder_loss", metavar="EXPERIMENT",
                   help=f"Experiment name in config.json. Choices: {exp_names}")
    p.add_argument("--projects", nargs="+", default=None,
                   help="Override target_projects")
    p.add_argument("--entity", default=DEFAULT_ENTITY,
                   help=f"W&B entity (default: {DEFAULT_ENTITY})")
    p.add_argument("--force", action="store_true", help="Re-download even if cache exists")
    p.add_argument("--no-plot", action="store_true", help="Skip plot generation")
    return p.parse_args()


# ── main ──────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace | None = None) -> None:
    if args is None:
        args = parse_args()

    exp_cfg = _experiment_cfg(args.experiment)
    projects = _resolve_projects(args.experiment, args.projects)
    method_order = [_display_name(p, exp_cfg) for p in projects]

    run_dir = _make_run_dir(exp_cfg, args.experiment)
    log = setup_logger(run_dir)
    cache_dir = _cache_dir()

    log.info("encoder_loss: comparing %d projects via artifact download", len(projects))
    for p, m in zip(projects, method_order):
        log.info("  %s  →  %s", p, m)

    api = get_api()
    all_rows: list[pd.DataFrame] = []

    for project, method in zip(projects, method_order):
        log.info("Project: %s  (%s)", project, method)
        log.info("  Fetching finished runs …")
        try:
            runs = list(api.runs(f"{args.entity}/{project}", filters={"state": "finished"}))
        except Exception as exc:
            log.warning("  Cannot list runs for %s: %s", project, exc)
            continue

        if not runs:
            log.warning("  No finished runs in %s", project)
            continue

        log.info("  Found %d run(s)", len(runs))
        for i, run in enumerate(runs, 1):
            log.info("  [%d/%d] run=%s (%s)", i, len(runs), run.id, run.name)
            csv_paths = _download_artifact_csvs(run, project, cache_dir, args.force, log)
            if not csv_paths:
                continue
            for path in csv_paths:
                df = _read_csv(path, project, run.name, method)
                if not df.empty:
                    all_rows.append(df)

    if not all_rows:
        log.error("No data collected — check W&B connectivity and artifact names.")
        return

    # Combine all rows
    combined = pd.concat(all_rows, ignore_index=True)
    combined.to_csv(run_dir / "encoder_loss_rows.csv", index=False)
    log.info("Saved raw rows: encoder_loss_rows.csv  (%d rows)", len(combined))

    # Epoch-level metrics
    metrics = _epoch_metrics(combined)
    if metrics.empty:
        log.error("Could not compute metrics — check CSV columns.")
        return
    metrics.to_csv(run_dir / "encoder_loss_epoch_metrics.csv", index=False)
    log.info("Saved epoch metrics: encoder_loss_epoch_metrics.csv  (%d rows)", len(metrics))

    # Pivot tables
    pivot_final = _domain_pivot(metrics, method_order, agg="last")
    pivot_min = _domain_pivot(metrics, method_order, agg="min")
    regression_final = _regression_summary_table(metrics, method_order, agg="last")
    regression_best = _regression_summary_table(metrics, method_order, agg="min")
    if not pivot_final.empty:
        pivot_final.to_csv(run_dir / "encoder_loss_domain_table.csv")
        log.info("Saved: encoder_loss_domain_table.csv")
        log.info("\n%s", pivot_final.to_string())
    if not pivot_min.empty:
        pivot_min.to_csv(run_dir / "encoder_loss_domain_min_table.csv")
        log.info("Saved: encoder_loss_domain_min_table.csv")
    if not regression_final.empty:
        regression_final.to_csv(run_dir / "encoder_loss_regression_final_table.csv", index=False)
        log.info("Saved: encoder_loss_regression_final_table.csv")
    if not regression_best.empty:
        regression_best.to_csv(run_dir / "encoder_loss_regression_best_table.csv", index=False)
        log.info("Saved: encoder_loss_regression_best_table.csv")

    # Overall
    overall = _overall_table(metrics, method_order)
    if not overall.empty:
        overall.to_csv(run_dir / "encoder_loss_overall.csv", index=False)
        log.info("Saved: encoder_loss_overall.csv  (%d rows)", len(overall))

    # Markdown
    _save_markdown_report(
        pivot_final,
        pivot_min,
        regression_final,
        regression_best,
        overall,
        method_order,
        run_dir,
        log,
    )

    # Plots
    if not getattr(args, "no_plot", False):
        _plot_per_domain(metrics, method_order, run_dir, log)
        _plot_overall(overall, method_order, run_dir, log)
        _plot_scatter_per_domain(combined, metrics, method_order, run_dir, log)

    log.info("encoder_loss analysis complete → %s", run_dir)


if __name__ == "__main__":
    main()

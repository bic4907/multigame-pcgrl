"""
condition_shift_perf_drop.py
=============================
aaai27_eval_mgpcgrl_unseen 전용 분석:

  게임 Y가 unseen일 때, evaluation에서 받는 조건(condition) 분포가
  MultiGameDataset(=train_mgpcgrl 학습 데이터 소스)의 조건 분포와
  얼마나 다른지를 JSD(Jensen-Shannon Divergence)로 측정하고,
  그 분포 차이(distribution shift)와 성능 하락(performance drop)의
  상관관계를 RE별 서브플롯으로 시각화한다.

정의
----
  condition_shift(Y | run C) =
      Jensen-Shannon Divergence between
        condition_{re}  (Y in run C, unseen — from ctrl_sim.csv)
      vs
        condition_{re}  (Y in MultiGameDataset — 학습 데이터 분포)

  perf_drop(Y | run C) =
      mean_progress(Y | all seen runs)  −  mean_progress(Y | run C, unseen)

NOTE: seen 기준 condition 분포는 eval CSV가 아닌 MultiGameDataset에서 직접 로드한다.
      이는 train_mgpcgrl이 실제로 학습에 사용한 조건 분포와 동일하기 때문이다.

출력
----
    condition_shift_scatter.png   — 5 subplots (RE 0-4), scatter + 상관계수
    condition_shift_table.csv     — 수치 데이터
    analysis_data.md              — 에이전트 분석용 구조화 Markdown
    condition_shift_report.md     — 한글 분석 리포트

사용법
------
    python results/utils/experiment/condition_shift_perf_drop.py
    python results/utils/experiment/condition_shift_perf_drop.py --no-plot
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import sys as _sys
_HERE        = Path(__file__).resolve().parent
_RESULTS_DIR = _HERE.parent.parent
_ROOT        = _RESULTS_DIR.parent
if str(_RESULTS_DIR) not in _sys.path:
    _sys.path.insert(0, str(_RESULTS_DIR))
if str(_ROOT) not in _sys.path:
    _sys.path.append(str(_ROOT))

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from scipy.spatial.distance import jensenshannon

from utils.core.run_output import load_cfg, make_run_dir, setup_logger
from utils.experiment.benchmark import (
    _bar_plot_setup,
    resolve_input_root,
)

_CFG = load_cfg()
GAME_COLORS: dict[str, str] = _CFG.get("games", {}).get("colors", {
    "doom":    "#1f77b4",
    "dungeon": "#d62728",
    "pokemon": "#2ca02c",
    "sokoban": "#ff7f0e",
    "zelda":   "#9467bd",
})

RE_LABELS: dict[str, str] = {
    str(k): v
    for k, v in _CFG.get("reward_enums", {}).get("labels", {
        "0": "Region",
        "1": "Path Length",
        "2": "Interactable",
        "3": "Hazard",
        "4": "Collectable",
    }).items()
}

_PROJECT = "aaai27_eval_mgpcgrl_unseen"
_SHIFT_COL   = "condition_shift_jsd"
_SHIFT_LABEL = "Condition Shift (JSD)"
_SHIFT_SHORT = "Jensen-Shannon Divergence"


# ---------------------------------------------------------------------------
# MultiGameDataset 조건 분포 로드 (train_mgpcgrl 학습 데이터 소스)
# ---------------------------------------------------------------------------

def load_dataset_conditions(re_keys: list[int] | None = None) -> dict[tuple[str, int], np.ndarray]:
    """
    MultiGameDataset에서 게임×RE별 condition 분포를 로드한다.

    train_mgpcgrl은 MultiGameDataset에서 샘플링한 condition으로 학습하므로,
    이 함수가 반환하는 분포가 모델이 학습 시 접한 '실제 조건 분포'이다.

    Parameters
    ----------
    re_keys : 포함할 reward_enum 값 목록. None 이면 전체.

    Returns
    -------
    {(game, re): np.ndarray of condition float values}
    """
    try:
        from dataset.multigame import MultiGameDataset
    except ImportError:
        raise ImportError("MultiGameDataset을 import할 수 없습니다. _ROOT가 sys.path에 있는지 확인하세요.")

    ds = MultiGameDataset(use_tile_mapping=True)
    re_set = set(re_keys) if re_keys is not None else None

    result: dict[tuple[str, int], list[float]] = defaultdict(list)
    for s in ds._samples:
        re = s.meta.get("reward_enum")
        if re is None:
            continue
        if re_set is not None and re not in re_set:
            continue
        cond_val = s.meta.get("conditions", {}).get(re)
        if cond_val is None:
            # dict 키가 str인 경우 fallback
            cond_val = s.meta.get("conditions", {}).get(str(re))
        if cond_val is None:
            continue
        result[(s.game, int(re))].append(float(cond_val))

    return {k: np.array(v) for k, v in result.items()}


# ---------------------------------------------------------------------------
# 데이터 수집
# ---------------------------------------------------------------------------

def load_all_data(input_root: Path) -> list[dict]:
    project_dir = input_root / _PROJECT
    if not project_dir.exists():
        raise FileNotFoundError(f"프로젝트 디렉토리를 찾을 수 없습니다: {project_dir}")

    rows: list[dict] = []
    for run_dir in sorted(project_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        run_name = run_dir.name
        re_val: int | None = None
        for part in run_name.split("_"):
            if part.startswith("re-"):
                try:
                    re_val = int(part[3:])
                except ValueError:
                    pass
        if re_val is None:
            continue

        eval_dir: Path | None = None
        for sub in run_dir.iterdir():
            if sub.is_dir() and sub.name.startswith(f"ev_re-{re_val}_"):
                eval_dir = sub
                break
        if eval_dir is None:
            continue

        cfg_path = eval_dir / "run_config.json"
        csv_path = eval_dir / "ctrl_sim.csv"
        if not cfg_path.exists() or not csv_path.exists():
            continue

        try:
            run_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        seen_games   = frozenset(run_cfg.get("seen_games",   []))
        unseen_games = frozenset(run_cfg.get("unseen_games", []))

        cond_col = f"condition_{re_val}"
        try:
            df = pd.read_csv(
                csv_path,
                usecols=["game", "reward_enum", cond_col, "progress"],
                dtype={"game": str},
            )
        except Exception:
            continue

        df = df[pd.to_numeric(df["reward_enum"], errors="coerce") == re_val].copy()
        df[cond_col]   = pd.to_numeric(df[cond_col],   errors="coerce")
        df["progress"] = pd.to_numeric(df["progress"], errors="coerce")
        df = df.dropna(subset=[cond_col, "progress"])
        if df.empty:
            continue

        for _, r in df.iterrows():
            game = str(r["game"]).strip()
            split = "seen" if game in seen_games else "unseen" if game in unseen_games else "unknown"
            if split == "unknown":
                continue
            rows.append({
                "run":          run_name,
                "re":           re_val,
                "game":         game,
                "split":        split,
                "condition":    float(r[cond_col]),
                "progress":     float(r["progress"]),
                "seen_games":   seen_games,
                "unseen_games": unseen_games,
            })

    return rows


# ---------------------------------------------------------------------------
# 분포 거리 (JSD)
# ---------------------------------------------------------------------------

def _js_divergence(a: np.ndarray, b: np.ndarray, n_bins: int = 30) -> float:
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    lo  = min(a.min(), b.min())
    hi  = max(a.max(), b.max())
    if lo >= hi:
        return 0.0
    bins = np.linspace(lo, hi, n_bins + 1)
    ha   = np.histogram(a, bins=bins)[0].astype(float) + 1e-9
    hb   = np.histogram(b, bins=bins)[0].astype(float) + 1e-9
    ha  /= ha.sum()
    hb  /= hb.sum()
    return float(jensenshannon(ha, hb))


# ---------------------------------------------------------------------------
# 데이터 포인트 구성
# ---------------------------------------------------------------------------

def build_scatter_points(
    all_rows: list[dict],
    dataset_conds: dict[tuple[str, int], np.ndarray],
) -> dict[int, list[dict]]:
    """
    JSD scatter point를 구성한다.

    Parameters
    ----------
    all_rows      : load_all_data()가 반환한 episode 행 목록
    dataset_conds : load_dataset_conditions()가 반환한 MultiGameDataset 조건 분포
                    {(game, re): np.ndarray} — "seen" 기준 분포로 사용한다.
    """
    # unseen 게임 성능 데이터 수집 (run별)
    # seen 게임 성능 데이터 수집 (all-seen 기준)
    perf_seen_by: dict[tuple[str, int], list[float]] = defaultdict(list)   # (game, re) → seen progress
    unseen_runs: dict[tuple[str, int, str], dict] = {}  # (re, game, run) → run data

    for r in all_rows:
        re, g, key = r["re"], r["game"], r["run"]
        if r["split"] == "seen":
            perf_seen_by[(g, re)].append(r["progress"])
        else:  # unseen
            rkey = (re, g, key)
            if rkey not in unseen_runs:
                unseen_runs[rkey] = {
                    "cond": [], "prog": [],
                    "seen_games": r["seen_games"],
                    "unseen_games": r["unseen_games"],
                }
            unseen_runs[rkey]["cond"].append(r["condition"])
            unseen_runs[rkey]["prog"].append(r["progress"])

    result: dict[int, list[dict]] = defaultdict(list)

    for (re, game, run_key), run_data in unseen_runs.items():
        unseen_cond = np.array(run_data["cond"])
        unseen_prog = np.array(run_data["prog"])
        if len(unseen_cond) < 2:
            continue

        # ── seen 기준 조건 분포: MultiGameDataset에서 로드 ──
        ds_cond = dataset_conds.get((game, re))
        if ds_cond is None or len(ds_cond) < 2:
            continue  # 데이터셋에 해당 (game, re) 조합이 없으면 skip

        # ── seen 기준 성능: eval CSV에서 seen-game rows 사용 ──
        perf_seen_vals = perf_seen_by.get((game, re), [])
        if not perf_seen_vals:
            continue
        perf_seen_mean = float(np.mean(perf_seen_vals))
        perf_unseen_mean = float(unseen_prog.mean())
        perf_drop = perf_seen_mean - perf_unseen_mean

        jsd = _js_divergence(ds_cond, unseen_cond)

        result[re].append({
            "re":                  re,
            "game":                game,
            "run":                 run_key,
            "seen_label":          "+".join(sorted(run_data["seen_games"])),
            "unseen_label":        "+".join(sorted(run_data["unseen_games"])),
            _SHIFT_COL:            jsd,
            "perf_seen":           perf_seen_mean,
            "perf_unseen":         perf_unseen_mean,
            "perf_drop":           perf_drop,
            "n_dataset":           len(ds_cond),
            "n_unseen":            len(unseen_cond),
        })

    return dict(result)


# ---------------------------------------------------------------------------
# 게임 이름 약어
# ---------------------------------------------------------------------------

_GAME_ABBREV: dict[str, str] = {
    "doom": "dm", "dungeon": "dg", "pokemon": "pk", "sokoban": "sk", "zelda": "zd",
}

def _abbrev(game: str) -> str:
    return _GAME_ABBREV.get(game, game[:2])


# ---------------------------------------------------------------------------
# 플롯
# ---------------------------------------------------------------------------

def write_scatter_plot(output_path: Path, points_by_re: dict[int, list[dict]]) -> None:
    plt = _bar_plot_setup()

    re_keys = sorted(points_by_re.keys())
    n_cols  = len(re_keys)
    if n_cols == 0:
        return

    fig, axes = plt.subplots(1, n_cols, figsize=(3.8 * n_cols + 0.4, 3.8), squeeze=False)

    legend_handles: list = []
    legend_labels:  list = []

    for ci, re in enumerate(re_keys):
        ax  = axes[0][ci]
        pts = points_by_re.get(re, [])
        re_label = RE_LABELS.get(str(re), f"RE={re}")
        ax.set_title(re_label, fontsize=10, fontweight="bold")
        ax.set_xlabel(_SHIFT_LABEL, fontsize=8.5)
        if ci == 0:
            ax.set_ylabel("Performance Drop\n(seen − unseen, %)", fontsize=9)

        if not pts:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center", color="gray")
            continue

        xs = np.array([p[_SHIFT_COL] for p in pts], dtype=float)
        ys = np.array([p["perf_drop"] for p in pts], dtype=float)
        valid = np.isfinite(xs) & np.isfinite(ys)
        xs, ys = xs[valid], ys[valid]
        pts_valid = [p for p, v in zip(pts, valid.tolist()) if v]

        for pt in pts_valid:
            g = pt["game"]
            color = GAME_COLORS.get(g, "#888888")
            sc = ax.scatter(
                pt[_SHIFT_COL], pt["perf_drop"],
                color=color, s=55, alpha=0.82, edgecolors="white", linewidths=0.5, zorder=3,
            )
            if g not in legend_labels:
                legend_handles.append(sc)
                legend_labels.append(g)

        if len(xs) >= 3:
            m, b = np.polyfit(xs, ys, 1)
            x_lin = np.linspace(xs.min(), xs.max(), 100)
            ax.plot(x_lin, m * x_lin + b, color="#333333", linewidth=1.2, linestyle="--", alpha=0.7, zorder=2)

        if len(xs) >= 3:
            r_val, p_val = scipy_stats.pearsonr(xs, ys)
            p_str = f"p={p_val:.3f}" if p_val >= 0.001 else "p<0.001"
            ax.annotate(
                f"r = {r_val:+.3f}\n{p_str}",
                xy=(0.97, 0.04), xycoords="axes fraction",
                ha="right", va="bottom", fontsize=8.5,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#aaaaaa", alpha=0.9),
            )

        ax.axhline(0, color="#aaaaaa", linewidth=0.8, linestyle=":", zorder=1)
        ax.grid(alpha=0.25)
        ax.tick_params(labelsize=8)

        for pt in pts_valid:
            seen_abbr  = "+".join(_abbrev(g) for g in pt["seen_label"].split("+"))
            label_text = f"{seen_abbr}→{_abbrev(pt['game'])}"
            ax.annotate(
                label_text,
                xy=(pt[_SHIFT_COL], pt["perf_drop"]),
                xytext=(3, 3), textcoords="offset points",
                fontsize=6.5, color="#444444",
            )

    seen_set: set[str] = set()
    uniq_h, uniq_l = [], []
    for h, l in zip(legend_handles, legend_labels):
        if l not in seen_set:
            uniq_h.append(h); uniq_l.append(l); seen_set.add(l)
    if uniq_h:
        fig.legend(uniq_h, uniq_l, loc="upper center", ncol=min(len(uniq_h), 6),
                   fontsize=8, bbox_to_anchor=(0.5, 1.03), title="Unseen Game", title_fontsize=8)

    fig.suptitle(
        f"Condition Distribution Shift (JSD) vs Performance Drop\n"
        f"(X: {_SHIFT_LABEL}  |  Y: Perf Drop = Seen − Unseen progress)\n"
        f"[Ref: MultiGameDataset training distribution]",
        fontsize=9, y=1.12,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CSV 출력
# ---------------------------------------------------------------------------

def write_csv(output_path: Path, points_by_re: dict[int, list[dict]]) -> None:
    fields = ["re", "re_label", "game", "run", "seen_label", "unseen_label",
              _SHIFT_COL, "perf_seen", "perf_unseen", "perf_drop", "n_dataset", "n_unseen"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for re in sorted(points_by_re.keys()):
            for pt in points_by_re[re]:
                row = dict(pt)
                row["re_label"] = RE_LABELS.get(str(re), f"RE={re}")
                writer.writerow({k: row.get(k, "") for k in fields})


# ---------------------------------------------------------------------------
# 분석용 데이터 MD 출력
# ---------------------------------------------------------------------------

def write_analysis_data_md(output_path: Path, points_by_re: dict[int, list[dict]]) -> None:
    lines: list[str] = []
    lines += [
        "# Condition Distribution Shift vs Performance Drop — Analysis Data",
        "",
        "**Source project**: `aaai27_eval_mgpcgrl_unseen`  ",
        "**Model**: MGPCGRL  ",
        "**Shift metric**: Jensen-Shannon Divergence (JSD, 0–1)  ",
        "**Performance metric**: Progress (%)  ",
        "",
        "**Column definitions**",
        "",
        "| Column | Description |",
        "| --- | --- |",
        "| `re` | Reward enum index (0–4) |",
        "| `re_label` | Human-readable reward type name |",
        "| `unseen_game` | Target game that was **unseen** during training |",
        "| `seen_games` | Games that were **seen** during training in this run |",
        f"| `{_SHIFT_COL}` | JSD between MultiGameDataset condition dist (train ref) and unseen-run eval dist |",
        "| `perf_seen` | Mean progress (%) for this game across all runs where it was **seen** |",
        "| `perf_unseen` | Mean progress (%) for this game in this specific run where it was **unseen** |",
        "| `perf_drop` | `perf_seen − perf_unseen`; positive = performance degraded when unseen |",
        "| `n_dataset` | Number of condition samples in MultiGameDataset (training reference) |",
        "| `n_unseen` | Number of episodes in unseen run |",
        "",
    ]

    all_pts = [p for pts in points_by_re.values() for p in pts]

    # Table 1 — RE별 상관계수
    lines += [
        "## Table 1 — Per-RE Correlation Summary (Pearson r, JSD vs perf drop)",
        "",
        "| RE | Reward Type | N | r (JSD) | p (JSD) | Significant? | Mean JSD | Mean Perf Drop |",
        "| --- | --- | ---: | ---: | ---: | --- | ---: | ---: |",
    ]
    for re in sorted(points_by_re.keys()):
        pts = points_by_re[re]
        re_label = RE_LABELS.get(str(re), f"RE={re}")
        xs = np.array([p[_SHIFT_COL] for p in pts], dtype=float)
        ys = np.array([p["perf_drop"] for p in pts], dtype=float)
        valid = np.isfinite(xs) & np.isfinite(ys)
        n = int(valid.sum())
        if n >= 3:
            r_val, p_val = scipy_stats.pearsonr(xs[valid], ys[valid])
            sig = "✓ Yes (p<0.05)" if p_val < 0.05 else "✗ No"
        else:
            r_val = p_val = float("nan"); sig = "—"
        mean_jsd  = float(xs[valid].mean()) if valid.any() else float("nan")
        mean_drop = float(ys[np.isfinite(ys)].mean()) if np.isfinite(ys).any() else float("nan")
        lines.append(
            f"| {re} | {re_label} | {n} | {r_val:+.4f} | {p_val:.4f} | {sig} "
            f"| {mean_jsd:.4f} | {mean_drop:+.4f} |"
        )
    lines.append("")

    # Table 2 — 게임별 요약
    lines += [
        "## Table 2 — Per-Game Summary",
        "",
        "| Unseen Game | N | Mean Perf Drop | Std Perf Drop | Mean JSD |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    per_game: dict[str, list[dict]] = defaultdict(list)
    for p in all_pts:
        per_game[p["game"]].append(p)
    for g in sorted(per_game.keys()):
        gpts  = per_game[g]
        drops = np.array([p["perf_drop"]  for p in gpts], dtype=float)
        jsds  = np.array([p[_SHIFT_COL]   for p in gpts], dtype=float)
        lines.append(
            f"| **{g}** | {len(gpts)} | {drops.mean():+.4f} | {drops.std():.4f} "
            f"| {jsds[np.isfinite(jsds)].mean():.4f} |"
        )
    lines.append("")

    # Table 3 — RE × Game (JSD)
    games   = sorted(per_game.keys())
    re_keys = sorted(points_by_re.keys())
    lines += [
        "## Table 3 — Mean JSD Shift (rows=RE, cols=Unseen Game)",
        "",
        "| RE | " + " | ".join(f"**{g}**" for g in games) + " |",
        "| --- | " + " | ".join(["---"] * len(games)) + " |",
    ]
    for re in re_keys:
        re_label = RE_LABELS.get(str(re), f"RE={re}")
        cells = [re_label]
        for g in games:
            vals = [p[_SHIFT_COL] for p in points_by_re[re] if p["game"] == g and math.isfinite(p[_SHIFT_COL])]
            cells.append(f"{sum(vals)/len(vals):.4f}" if vals else "—")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    # Table 4 — RE × Game (perf drop)
    lines += [
        "## Table 4 — Mean Performance Drop (rows=RE, cols=Unseen Game)",
        "",
        "| RE | " + " | ".join(f"**{g}**" for g in games) + " |",
        "| --- | " + " | ".join(["---"] * len(games)) + " |",
    ]
    for re in re_keys:
        re_label = RE_LABELS.get(str(re), f"RE={re}")
        cells = [re_label]
        for g in games:
            vals = [p["perf_drop"] for p in points_by_re[re] if p["game"] == g]
            cells.append(f"{sum(vals)/len(vals):+.4f}" if vals else "—")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    # Table 5 — Full raw data
    lines += [
        "## Table 5 — Full Raw Data",
        "",
        "| RE | Reward Type | Unseen Game | Seen Games | JSD Shift | Perf Seen | Perf Unseen | Perf Drop | N dataset | N unseen |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for re in sorted(points_by_re.keys()):
        re_label = RE_LABELS.get(str(re), f"RE={re}")
        for pt in sorted(points_by_re[re], key=lambda p: (p["game"], p["perf_drop"]), reverse=True):
            jsd_s = f"{pt[_SHIFT_COL]:.4f}" if math.isfinite(pt[_SHIFT_COL]) else "—"
            lines.append(
                f"| {re} | {re_label} | **{pt['game']}** | {pt['seen_label']} "
                f"| {jsd_s} | {pt['perf_seen']:.2f} | {pt['perf_unseen']:.2f} "
                f"| {pt['perf_drop']:+.4f} | {pt.get('n_dataset', '—')} | {pt['n_unseen']} |"
            )
    lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# 리포트
# ---------------------------------------------------------------------------

def write_report(output_path: Path, points_by_re: dict[int, list[dict]]) -> None:
    lines: list[str] = []
    lines += [
        "# Condition Distribution Shift (JSD) vs Performance Drop 분석 리포트",
        "",
        "> **분석 대상**: `aaai27_eval_mgpcgrl_unseen` — MGPCGRL 모델의 unseen 게임 일반화 실험",
        ">",
        "> **핵심 질문**: 게임 Y를 학습하지 않았을 때(unseen), Y에 주어지는 조건(condition) 분포가",
        "> Y를 학습했을 때(seen)와 얼마나 다른가? 그 차이(JSD)가 성능 하락과 얼마나 관련이 있는가?",
        "",
        "## 방법론",
        "",
        "| 항목 | 설명 |",
        "| --- | --- |",
        f"| **Condition Shift** | {_SHIFT_SHORT} (JSD) — Y unseen 시 eval condition 분포 vs MultiGameDataset 기준 분포 (train ref) |",
        "| **Performance Drop** | Progress(Y \\| Y seen 평균) − Progress(Y \\| Y unseen in this run) |",
        "| **데이터 포인트** | (run 설정, unseen 게임 Y) 쌍 별 1개 포인트 |",
        "| **상관분석** | Pearson r (p-value 포함) |",
        "| **조건 기준 분포** | MultiGameDataset (train_mgpcgrl 학습 데이터 소스) |",
        "",
        "## RE별 분석 결과", "",
    ]

    summary_rows: list[dict] = []

    for re in sorted(points_by_re.keys()):
        re_label = RE_LABELS.get(str(re), f"RE={re}")
        pts = points_by_re[re]
        lines += [f"### {re_label} (RE={re})", ""]

        if not pts:
            lines += ["> 데이터 없음", ""]; continue

        xs = np.array([p[_SHIFT_COL] for p in pts], dtype=float)
        ys = np.array([p["perf_drop"] for p in pts], dtype=float)
        valid = np.isfinite(xs) & np.isfinite(ys)
        xs, ys = xs[valid], ys[valid]
        pts_valid = [p for p, v in zip(pts, valid.tolist()) if v]
        n = len(xs)

        if n >= 3:
            r_val, p_val = scipy_stats.pearsonr(xs, ys)
            p_str = f"{p_val:.4f}"
            sig = "✓ 유의미" if p_val < 0.05 else "✗ 유의미하지 않음"
        else:
            r_val = p_val = float("nan"); p_str = sig = "—"

        lines += [
            f"- **데이터 포인트 수**: {n}",
            f"- **Pearson r**: {r_val:+.4f}  (p = {p_str})  → {sig}",
            f"- **JSD 범위**: [{xs.min():.4f}, {xs.max():.4f}]  (평균 {xs.mean():.4f})",
            f"- **Perf Drop 범위**: [{ys.min():.4f}, {ys.max():.4f}]  (평균 {ys.mean():.4f})",
            "",
            "| Game | Seen 조합 | JSD Shift | Perf (seen) | Perf (unseen) | Drop |",
            "| --- | --- | --- | ---: | ---: | ---: |",
        ]
        for pt in sorted(pts_valid, key=lambda p: p["perf_drop"], reverse=True):
            s = pt[_SHIFT_COL]
            lines.append(
                f"| **{pt['game']}** | {pt['seen_label']} | {s:.4f} "
                f"| {pt['perf_seen']:.2f} | {pt['perf_unseen']:.2f} | {pt['perf_drop']:+.2f} |"
            )
        lines.append("")

        per_game: dict[str, list[float]] = defaultdict(list)
        per_game_jsd: dict[str, list[float]] = defaultdict(list)
        for pt in pts_valid:
            per_game[pt["game"]].append(pt["perf_drop"])
            if math.isfinite(pt[_SHIFT_COL]):
                per_game_jsd[pt["game"]].append(pt[_SHIFT_COL])

        lines += [
            "**게임별 평균:**", "",
            "| Game | 평균 Drop (%) | 평균 JSD | N |",
            "| --- | ---: | ---: | ---: |",
        ]
        for g in sorted(per_game.keys()):
            jsds = per_game_jsd.get(g, [])
            lines.append(
                f"| **{g}** | {sum(per_game[g])/len(per_game[g]):+.2f} "
                f"| {sum(jsds)/len(jsds):.4f} | {len(per_game[g])} |"
            )
        lines.append("")

        if math.isfinite(r_val):
            summary_rows.append({"re": re, "re_label": re_label, "n": n,
                                  "r": r_val, "p": p_val, "sig": sig,
                                  "jsd_mean": xs.mean(), "drop_mean": ys.mean()})

    # 요약 테이블
    lines += [
        "## 전체 RE 요약", "",
        "| RE | 이름 | N | Pearson r | p-value | 유의성 | 평균 JSD | 평균 Perf Drop |",
        "| --- | --- | ---: | ---: | ---: | --- | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['re']} | {row['re_label']} | {row['n']} "
            f"| {row['r']:+.4f} | {row['p']:.4f} | {row['sig']} "
            f"| {row['jsd_mean']:.4f} | {row['drop_mean']:+.4f} |"
        )
    lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    _exp_names = list(_CFG.get("experiments", {}).keys())
    parser = argparse.ArgumentParser(description="JSD 조건 분포 shift vs 성능 하락 상관관계 분석")
    parser.add_argument("--input", default="wandb_projects")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--experiment", choices=_exp_names if _exp_names else None,
                        default=None, metavar="EXPERIMENT")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args    = parse_args()
    run_dir = make_run_dir("condition_shift_perf_drop", cfg=_CFG)
    log     = setup_logger(run_dir, name=__file__)
    log.info("run_dir    : %s", run_dir)

    input_root = resolve_input_root(args.input, _RESULTS_DIR)
    log.info("input_root : %s", input_root)

    log.info("데이터 로드 중 …")
    try:
        all_rows = load_all_data(input_root)
    except FileNotFoundError as e:
        log.error("%s", e); raise SystemExit(str(e))

    if not all_rows:
        raise SystemExit("데이터를 찾을 수 없습니다.")
    log.info("총 %d 개 episode 로드 완료", len(all_rows))

    # ── MultiGameDataset에서 학습 조건 분포 로드 ──
    all_re_keys = sorted({r["re"] for r in all_rows})
    log.info("MultiGameDataset 조건 분포 로드 중 (re=%s) …", all_re_keys)
    dataset_conds = load_dataset_conditions(all_re_keys)
    log.info(
        "Dataset 조건 로드 완료: %d (game, re) 조합 / 게임별: %s",
        len(dataset_conds),
        {game: len(v) for (game, _), v in dataset_conds.items()} if dataset_conds else {},
    )

    points_by_re = build_scatter_points(all_rows, dataset_conds)
    total_pts = sum(len(v) for v in points_by_re.values())
    log.info("scatter 포인트 수: %d  (RE 별: %s)", total_pts, {k: len(v) for k, v in points_by_re.items()})
    if total_pts == 0:
        raise SystemExit("scatter 포인트가 없습니다.")

    # CSV
    csv_path = run_dir / "condition_shift_table.csv"
    write_csv(csv_path, points_by_re)
    log.info("table      : %s", csv_path)

    # 분석 MD
    analysis_md_path = run_dir / "analysis_data.md"
    write_analysis_data_md(analysis_md_path, points_by_re)
    log.info("analysis   : %s", analysis_md_path)

    # 리포트
    report_path = run_dir / "condition_shift_report.md"
    write_report(report_path, points_by_re)
    log.info("report     : %s", report_path)

    # 플롯
    if not args.no_plot:
        try:
            plot_path = run_dir / "condition_shift_scatter.png"
            write_scatter_plot(plot_path, points_by_re)
            log.info("plot       : %s", plot_path)
        except Exception as e:
            import traceback
            log.error("플롯 생성 실패: %s\n%s", e, traceback.format_exc())

        # seen 1개인 run만 필터링한 scatter (1개 게임으로 학습 → unseen 게임별 전이)
        try:
            singlegame_by_re: dict[int, list[dict]] = {
                re: [
                    pt for pt in pts
                    if len(pt["seen_label"].split("+")) == 1
                ]
                for re, pts in points_by_re.items()
            }
            sg_total = sum(len(v) for v in singlegame_by_re.values())
            log.info("singlegame scatter 포인트 수: %d", sg_total)
            if sg_total > 0:
                sg_plot_path = run_dir / "condition_shift_scatter-singlegame.png"
                write_scatter_plot(sg_plot_path, singlegame_by_re)
                log.info("singlegame plot: %s", sg_plot_path)
            else:
                log.warning("seen=1 & unseen=1 조건을 만족하는 포인트가 없어 singlegame 플롯을 생략합니다.")
        except Exception as e:
            import traceback
            log.error("singlegame 플롯 생성 실패: %s\n%s", e, traceback.format_exc())

    log.info("완료")


if __name__ == "__main__":
    main()


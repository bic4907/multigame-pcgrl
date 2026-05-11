"""
game_impact_analysis.py
=======================
unseen_generalizability 실험 전용:
  어떤 게임의 학습 데이터 유무(seen / unseen)가 다른 게임의 unseen 성능에
  얼마나 영향을 주는지 게임 쌍(X → Y) 단위로 정량화하고 시각화한다.

정의
----
Impact(X → Y) = mean_perf(Y | X seen)  −  mean_perf(Y | X unseen)

  양수(warm): X의 학습 데이터가 Y의 zero-shot 일반화에 기여
  음수(cool) : X가 없어도 Y 성능이 유지되거나 오히려 향상

출력
----
    game_impact_heatmap.png  — 행=target(Y), 열=influencer(X), 컬러=impact
    game_impact_bar.png      — 게임별 평균 impact 바 차트 (어떤 게임이 가장 영향력 큰가)
    game_impact_table.csv    — 수치 데이터
    game_impact_table.md     — Markdown 테이블

사용법
------
    python results/utils/experiment/game_impact_analysis.py
    python results/utils/experiment/game_impact_analysis.py --reward-enum 0
    python results/utils/experiment/game_impact_analysis.py --no-plot
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path

import sys as _sys
_HERE        = Path(__file__).resolve().parent   # results/utils/experiment/
_RESULTS_DIR = _HERE.parent.parent               # results/
_ROOT        = _RESULTS_DIR.parent               # project root
if str(_RESULTS_DIR) not in _sys.path:
    _sys.path.insert(0, str(_RESULTS_DIR))
if str(_ROOT) not in _sys.path:
    _sys.path.append(str(_ROOT))

from utils.core.run_output import load_cfg, make_run_dir, setup_logger
from utils.core.io import (
    normalize_reward_enum,
    parse_run_tokens,
    iter_results_paths,
    load_run_config,
    get_game_split,
)
from utils.core.stats import to_float
from utils.core.normalization import (
    compute_normalization_scale,
    apply_normalization,
    save_normalization_scale,
    load_normalization_scale,
)
from utils.experiment.benchmark import (
    _bar_plot_setup,
    _seed_agg,
    DEFAULT_METRIC_ORDER,
    METRIC_DISPLAY_NAMES,
    resolve_input_root,
    _get_experiment_folder_order,
)

_CFG = load_cfg()
GAME_COLORS: dict[str, str] = _CFG.get("games", {}).get("colors", {})


# ---------------------------------------------------------------------------
# 데이터 수집 (seen_games / unseen_games set 포함)
# ---------------------------------------------------------------------------

def collect_rows_with_game_sets(
    input_root: Path,
    metric_order: list[str],
    target_projects: list[str] | None = None,
) -> list[dict]:
    """results.csv 를 순회하며 seen_games / unseen_games frozenset 을 포함한 row 목록 반환.

    run_config.json 이 없는 run 은 게임 분류 정보가 없으므로 건너뜁니다.
    """
    rows: list[dict] = []
    _cfg_cache: dict[Path, dict] = {}

    for results_path in iter_results_paths(input_root):
        rel = results_path.relative_to(input_root)
        if len(rel.parts) < 3:
            continue

        project   = rel.parts[0]
        run_name  = rel.parts[1]
        eval_name = rel.parts[2] if len(rel.parts) >= 4 else ""

        if target_projects and project not in target_projects:
            continue

        eval_tokens = parse_run_tokens(eval_name)
        run_tokens  = parse_run_tokens(run_name)
        seed = run_tokens.get("s", run_name)

        run_dir = results_path.parent
        if run_dir not in _cfg_cache:
            _cfg_cache[run_dir] = load_run_config(run_dir)
        run_cfg = _cfg_cache[run_dir]

        seen_games_fs   = frozenset(run_cfg.get("seen_games",   []))
        unseen_games_fs = frozenset(run_cfg.get("unseen_games", []))
        if not seen_games_fs and not unseen_games_fs:
            continue  # run_config 없으면 seen/unseen 구분 불가 → 건너뜀

        with results_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                game = (row.get("game") or "").strip() or "unknown"
                if game == "unknown":
                    continue
                re_raw = (row.get("reward_enum") or "").strip()
                reward_enum = normalize_reward_enum(
                    re_raw or eval_tokens.get("re", "unknown")
                )
                metric_values = {
                    m: v for m in metric_order
                    if (v := to_float(row.get(m))) is not None
                }
                if not metric_values:
                    continue

                game_split = get_game_split(game, run_cfg)
                rows.append({
                    "project":     project,
                    "game":        game,
                    "reward_enum": reward_enum,
                    "game_split":  game_split,
                    "seen_games":  seen_games_fs,
                    "unseen_games":unseen_games_fs,
                    "seed":        seed,
                    "metrics":     metric_values,
                })
    return rows


# ---------------------------------------------------------------------------
# Impact 계산
# ---------------------------------------------------------------------------

def compute_game_impact(
    rows: list[dict],
    metric_order: list[str],
    reward_enum: str | None = None,
) -> tuple[list[str], dict[str, dict[tuple[str, str], dict]]]:
    """각 (influencer X, target Y) 쌍에 대한 impact 계산.

    Impact(X → Y) = mean_perf(Y unseen | X seen) − mean_perf(Y unseen | X unseen)

    Returns
    -------
    games      : 정렬된 게임 이름 목록
    impact_map : {metric: {(influencer, target): stat_dict}}
    """
    # unseen 게임 행만 사용 (+ reward_enum 필터)
    filtered = [
        r for r in rows
        if r["game_split"] == "unseen"
        and (reward_enum is None or r["reward_enum"] == reward_enum)
    ]
    if not filtered:
        return [], {}

    # 전체 등장 게임 목록
    games = sorted({r["game"] for r in filtered})

    impact_map: dict[str, dict[tuple, dict]] = {m: {} for m in metric_order}

    for target in games:
        target_rows = [r for r in filtered if r["game"] == target]

        # 이 target 이 unseen인 run 에서 나타나는 모든 influencer 후보
        # (seen_games ∪ unseen_games에서 target이 아닌 것)
        all_games_in_runs = sorted(
            {g for r in target_rows for g in (r["seen_games"] | r["unseen_games"])}
        )

        for influencer in all_games_in_runs:
            if influencer == target:
                continue

            # X seen : influencer 가 seen_games 에 있는 경우
            rows_x_seen   = [r for r in target_rows if influencer in r["seen_games"]]
            # X unseen: influencer 도 unseen_games 에 있는 경우
            rows_x_unseen = [r for r in target_rows if influencer in r["unseen_games"]]

            if not rows_x_seen or not rows_x_unseen:
                continue

            for metric in metric_order:
                stat_seen   = _seed_agg(rows_x_seen,   metric)
                stat_unseen = _seed_agg(rows_x_unseen, metric)

                if stat_seen is None or stat_unseen is None:
                    continue

                impact     = stat_seen["mean"] - stat_unseen["mean"]
                impact_std = (stat_seen["std"] ** 2 + stat_unseen["std"] ** 2) ** 0.5

                impact_map[metric][(influencer, target)] = {
                    "impact":      impact,
                    "std":         impact_std,
                    "perf_seen":   stat_seen["mean"],
                    "perf_unseen": stat_unseen["mean"],
                    "n_seen":      stat_seen["n"],
                    "n_unseen":    stat_unseen["n"],
                }

    return games, impact_map


# ---------------------------------------------------------------------------
# Plot 1: 게임×게임 히트맵
# ---------------------------------------------------------------------------

def write_impact_heatmap(
    output_path: Path,
    games: list[str],
    per_re_impact: dict[str, dict[str, dict[tuple[str, str], dict]]],
    metric_order: list[str],
    heatmap_metric: str = "progress",
) -> None:
    """1행 × n_re열 레이아웃으로 Progress impact 히트맵을 그린다."""
    import numpy as np
    plt = _bar_plot_setup()

    n_games = len(games)
    if n_games < 2:
        return

    re_labels_cfg = _CFG.get("reward_enums", {}).get("labels", {})
    re_keys = sorted(per_re_impact.keys(), key=lambda x: (int(x) if x.isdigit() else 99))
    n_re    = len(re_keys)
    if n_re == 0:
        return

    metric = heatmap_metric if heatmap_metric in metric_order else metric_order[0]
    game_idx = {g: i for i, g in enumerate(games)}

    fig, axes = plt.subplots(
        1, n_re,
        figsize=(2.2 * n_re + 0.4, 2.8 + 0.4),
        squeeze=False,
    )

    for ci, re in enumerate(re_keys):
        ax    = axes[0][ci]
        mat   = np.full((n_games, n_games), np.nan)
        mdata = per_re_impact[re].get(metric, {})

        for (influencer, target), stat in mdata.items():
            xi = game_idx.get(influencer)
            yi = game_idx.get(target)
            if xi is not None and yi is not None:
                mat[yi, xi] = stat["impact"]

        valid    = mat[~np.isnan(mat)]
        re_label = re_labels_cfg.get(re, f"re={re}")
        ax.set_title(re_label, fontsize=8)

        if len(valid) == 0:
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        vmax = max(abs(valid).max(), 1e-6)
        ax.imshow(mat, cmap="RdBu", aspect="auto", vmin=-vmax, vmax=vmax,
                  interpolation="nearest")

        # 셀 값 텍스트 (대각선 포함 모두 숫자만, × 없음)
        for row_i in range(n_games):
            for col_i in range(n_games):
                v = mat[row_i, col_i]
                if np.isnan(v):
                    ax.text(col_i, row_i, "–", ha="center", va="center",
                            fontsize=7, color="#888888")
                else:
                    text_color = "white" if abs(v) > vmax * 0.55 else "black"
                    ax.text(col_i, row_i, f"{v:+.3f}",
                            ha="center", va="center",
                            fontsize=6, color=text_color, fontweight="bold")

        ax.set_xticks(range(n_games))
        ax.set_yticks(range(n_games))
        ax.set_xticklabels(games, rotation=35, ha="right", fontsize=7)
        ax.set_yticklabels(games if ci == 0 else [], fontsize=7)
        ax.tick_params(which="both", bottom=False, left=False)
        ax.grid(False)   # 격자 완전 비활성화

    metric_label = METRIC_DISPLAY_NAMES.get(metric, metric)
    fig.suptitle(
        f"{metric_label} — Cross-Game Training Impact per Reward Type\n"
        r"Impact(X→Y) = perf(Y | X seen) $-$ perf(Y | X unseen)",
        fontsize=9, y=1.02,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 2: influencer 게임별 총 영향력 바 차트 (re별 subplot)
# ---------------------------------------------------------------------------

def write_impact_bar(
    output_path: Path,
    games: list[str],
    per_re_impact: dict[str, dict[str, dict[tuple[str, str], dict]]],
    metric: str = "progress",
) -> None:
    """reward_enum 별 Progress impact 바 차트.

    per_re_impact : {reward_enum_str: {metric: {(influencer, target): stat}}}
    metric        : 표시할 메트릭 (기본: "progress")
    """
    plt = _bar_plot_setup()
    re_labels_cfg = _CFG.get("reward_enums", {}).get("labels", {})
    re_keys = sorted(per_re_impact.keys(), key=lambda x: (int(x) if x.isdigit() else 99))
    n_cols  = len(re_keys)
    if n_cols == 0:
        return

    fig, axes = plt.subplots(1, n_cols, figsize=(2.8 * n_cols + 0.4, 3.0), squeeze=False)

    for ci, re in enumerate(re_keys):
        ax    = axes[0][ci]
        mdata = per_re_impact[re].get(metric, {})

        totals: dict[str, list[float]] = defaultdict(list)
        for (influencer, _target), stat in mdata.items():
            totals[influencer].append(stat["impact"])

        xs = [g for g in games if g in totals]
        re_label = re_labels_cfg.get(re, f"re={re}")
        if not xs:
            ax.set_title(f"Progress\n({re_label})", fontsize=8)
            continue

        ys     = [sum(totals[g]) / len(totals[g]) for g in xs]
        colors = [GAME_COLORS.get(g, "#1f77b4") for g in xs]
        bars   = ax.bar(range(len(xs)), ys, color=colors, edgecolor="white", linewidth=0.8)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.set_xticks(range(len(xs)))
        ax.set_xticklabels(xs, rotation=30, ha="right", fontsize=8)
        ax.set_title(f"Progress\n({re_label})", fontsize=8)
        if ci == 0:
            ax.set_ylabel("Mean impact (seen − unseen)", fontsize=9)
        ax.grid(axis="y", alpha=0.3)

        for bar, y in zip(bars, ys):
            va     = "bottom" if y >= 0 else "top"
            offset = max(abs(y) * 0.05, 0.003) * (1 if y >= 0 else -1)
            ax.text(bar.get_x() + bar.get_width() / 2,
                    y + offset, f"{y:+.3f}",
                    ha="center", va=va, fontsize=7)

    fig.suptitle("Training Impact on Unseen Progress per Reward Type", fontsize=10)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 테이블 출력
# ---------------------------------------------------------------------------

def write_impact_csv(
    output_path: Path,
    games: list[str],
    impact_map: dict[str, dict[tuple[str, str], dict]],
    metric_order: list[str],
) -> None:
    keys = ["impact", "std", "perf_seen", "perf_unseen", "n_seen", "n_unseen"]
    headers = ["influencer", "target"] + [f"{m}_{k}" for m in metric_order for k in keys]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for influencer in games:
            for target in games:
                if influencer == target:
                    continue
                row: dict = {"influencer": influencer, "target": target}
                for m in metric_order:
                    stat = impact_map.get(m, {}).get((influencer, target))
                    for k in keys:
                        row[f"{m}_{k}"] = stat[k] if stat else ""
                writer.writerow(row)


def write_impact_markdown(
    output_path: Path,
    games: list[str],
    impact_map: dict[str, dict[tuple[str, str], dict]],
    metric_order: list[str],
    decimals: int = 4,
) -> None:
    header_cols = ["Influencer→Target"] + [
        METRIC_DISPLAY_NAMES.get(m, m) for m in metric_order
    ]
    lines = [
        "| " + " | ".join(header_cols) + " |",
        "| " + " | ".join(["---"] * len(header_cols)) + " |",
    ]
    for influencer in games:
        for target in games:
            if influencer == target:
                continue
            cells = [f"**{influencer}** → {target}"]
            for m in metric_order:
                stat = impact_map.get(m, {}).get((influencer, target))
                cells.append(
                    f"`{stat['impact']:+.{decimals}f}` (±{stat['std']:.{decimals}f})"
                    if stat else "–"
                )
            lines.append("| " + " | ".join(cells) + " |")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# 분석 리포트
# ---------------------------------------------------------------------------

def write_impact_report(
    output_path: Path,
    games: list[str],
    impact_map: dict[str, dict[tuple[str, str], dict]],       # 전체 re 평균
    per_re_impact: dict[str, dict[str, dict[tuple[str, str], dict]]],
    metric: str = "progress",
    top_n: int = 3,
) -> None:
    """히트맵 분석 결과를 Markdown 리포트로 저장한다.

    포함 내용:
      1. 전체 요약 — 가장 영향력 큰/작은 influencer
      2. Target별 — 어떤 게임이 가장 도움이 됐는가
      3. Influencer별 — 어떤 게임에 가장 기여했는가
      4. Reward enum별 상위 impact 쌍
      5. 주목할 패턴 (음수 impact = 역효과 쌍)
    """
    re_labels_cfg  = _CFG.get("reward_enums", {}).get("labels", {})
    metric_label   = METRIC_DISPLAY_NAMES.get(metric, metric)
    re_keys = sorted(per_re_impact.keys(), key=lambda x: (int(x) if x.isdigit() else 99))

    def _fmt(v: float) -> str:
        sign = "+" if v >= 0 else ""
        return f"{sign}{v:.4f}"

    # ── 전체(re 평균) impact 집계 ────────────────────────────────────────
    mdata_all = impact_map.get(metric, {})

    # influencer별 평균 impact (다른 모든 target에 미치는 평균)
    inf_scores: dict[str, list[float]] = defaultdict(list)
    for (inf, _tgt), stat in mdata_all.items():
        inf_scores[inf].append(stat["impact"])
    inf_avg = {g: sum(v) / len(v) for g, v in inf_scores.items() if v}

    # target별 자신이 받은 평균 impact (다른 모든 influencer로부터)
    tgt_scores: dict[str, list[float]] = defaultdict(list)
    for (_inf, tgt), stat in mdata_all.items():
        tgt_scores[tgt].append(stat["impact"])
    tgt_avg = {g: sum(v) / len(v) for g, v in tgt_scores.items() if v}

    # 전체 top/bottom pairs
    all_pairs = sorted(mdata_all.items(), key=lambda x: x[1]["impact"], reverse=True)

    lines: list[str] = []
    lines += [
        f"# Game Impact Analysis Report — {metric_label}",
        "",
        f"> **Impact(X → Y)** = perf(Y | X **seen**) − perf(Y | X **unseen**)  ",
        "> 양수: X의 학습 데이터가 Y의 unseen 일반화에 기여  ",
        "> 음수: X의 학습 데이터 유무가 Y에 부정적이거나 무관",
        "",
    ]

    # ── 0. 핵심 요약 (자연어 글줄) ────────────────────────────────────────
    lines += ["## 핵심 요약", ""]

    # 가장 영향력 큰 influencer
    top_inf = max(inf_avg, key=inf_avg.get) if inf_avg else None
    bot_inf = min(inf_avg, key=inf_avg.get) if inf_avg else None

    # 가장 수혜받는 target
    top_tgt = max(tgt_avg, key=tgt_avg.get) if tgt_avg else None
    bot_tgt = min(tgt_avg, key=tgt_avg.get) if tgt_avg else None

    # 최강 positive pair
    best_pair  = all_pairs[0]  if all_pairs else None
    worst_pair = all_pairs[-1] if all_pairs else None

    # 음수 쌍 수
    n_neg   = sum(1 for _, s in all_pairs if s["impact"] < 0)
    n_pos   = sum(1 for _, s in all_pairs if s["impact"] >= 0)
    n_total = len(all_pairs)

    summary_bullets: list[str] = []

    if top_inf:
        summary_bullets.append(
            f"- **{top_inf}** 게임의 학습 데이터가 다른 게임의 unseen 일반화에 가장 큰 기여를 함 "
            f"(평균 impact {_fmt(inf_avg[top_inf])})."
        )
    if bot_inf and bot_inf != top_inf:
        summary_bullets.append(
            f"- **{bot_inf}** 게임은 다른 게임에 대한 평균 영향력이 가장 낮음 "
            f"(평균 impact {_fmt(inf_avg[bot_inf])})."
        )
    if top_tgt:
        summary_bullets.append(
            f"- **{top_tgt}** 게임은 다른 게임의 학습 데이터로부터 가장 큰 수혜를 받음 "
            f"(평균 수혜 {_fmt(tgt_avg[top_tgt])})."
        )
    if bot_tgt and bot_tgt != top_tgt:
        summary_bullets.append(
            f"- **{bot_tgt}** 게임은 타 게임 데이터의 도움을 가장 적게 받음 "
            f"(평균 수혜 {_fmt(tgt_avg[bot_tgt])})."
        )
    if best_pair:
        (bi, bt), bs = best_pair
        summary_bullets.append(
            f"- 가장 강한 전이 쌍: **{bi} → {bt}** "
            f"(impact {_fmt(bs['impact'])}, "
            f"seen {bs['perf_seen']:.3f} vs unseen {bs['perf_unseen']:.3f})."
        )
    if worst_pair and worst_pair[1]["impact"] < 0:
        (wi, wt), ws = worst_pair
        summary_bullets.append(
            f"- 가장 부정적 쌍: **{wi} → {wt}** "
            f"(impact {_fmt(ws['impact'])}) — {wi} 데이터가 없을 때 {wt} 성능이 더 높음."
        )
    _pos_ratio = n_pos / n_total if n_total else 0
    if _pos_ratio >= 0.8:
        _verdict = "🟢 **매우 도움됨** — 대부분의 게임 학습이 unseen 일반화에 기여함."
    elif _pos_ratio >= 0.6:
        _verdict = "🔵 **저럭저럭 도움됨** — 과반 이상의 쌍에서 긍정적 전이가 확인됨."
    elif _pos_ratio >= 0.4:
        _verdict = "⚪ **차이없음** — 양·음수 쌍이 비슷하게 섞여 뚜렷한 경향 없음."
    elif _pos_ratio >= 0.2:
        _verdict = "🟠 **도움안됨** — 음수 impact 쌍이 다수로, 멀티게임 학습 효과가 미미함."
    else:
        _verdict = "🔴 **매우 도움안됨** — 대부분의 쌍에서 음수 impact, 게임 간 전이가 역효과."
    summary_bullets.append(
        f"- 전체 **{n_total}개** 쌍 중 **양수 impact: {n_pos}개** ({n_pos}/{n_total}, "
        f"{100*_pos_ratio:.0f}%) / **음수 impact: {n_neg}개** ({n_neg}/{n_total}, "
        f"{100*(1-_pos_ratio):.0f}%)  \n"
        f"  → 종합 평가: {_verdict}"
    )

    lines += summary_bullets
    lines.append("")

    # ── 1. 전체 요약 ──────────────────────────────────────────────────────
    lines += ["## 1. 전체 영향력 순위 (모든 reward_enum 평균)", ""]
    lines += ["### Influencer 순위 — 다른 게임에 미치는 평균 impact", ""]
    lines += ["| 순위 | Influencer | 평균 Impact | 기여한 게임 수 |"]
    lines += ["| --- | --- | --- | --- |"]
    for rank, (g, score) in enumerate(
        sorted(inf_avg.items(), key=lambda x: x[1], reverse=True), 1
    ):
        n = len(inf_scores[g])
        lines.append(f"| {rank} | **{g}** | {_fmt(score)} | {n} |")
    lines.append("")

    lines += ["### Target 순위 — 다른 게임으로부터 받은 평균 impact", ""]
    lines += ["| 순위 | Target | 평균 수혜 Impact | 기여받은 게임 수 |"]
    lines += ["| --- | --- | --- | --- |"]
    for rank, (g, score) in enumerate(
        sorted(tgt_avg.items(), key=lambda x: x[1], reverse=True), 1
    ):
        n = len(tgt_scores[g])
        lines.append(f"| {rank} | **{g}** | {_fmt(score)} | {n} |")
    lines.append("")

    # ── 2. Target별 최대 influencer ───────────────────────────────────────
    lines += ["## 2. Target별 — 학습에 가장 도움이 된 게임", ""]
    lines += ["| Target (Y) | 최고 Influencer (X) | Impact | 최저 Influencer | Impact |"]
    lines += ["| --- | --- | --- | --- | --- |"]
    for tgt in games:
        pairs = [
            ((inf, t), stat) for (inf, t), stat in mdata_all.items() if t == tgt
        ]
        if not pairs:
            continue
        pairs_sorted = sorted(pairs, key=lambda x: x[1]["impact"], reverse=True)
        best_inf,  best_stat  = pairs_sorted[0][0][0],  pairs_sorted[0][1]
        worst_inf, worst_stat = pairs_sorted[-1][0][0], pairs_sorted[-1][1]
        lines.append(
            f"| **{tgt}** | {best_inf} | {_fmt(best_stat['impact'])} "
            f"| {worst_inf} | {_fmt(worst_stat['impact'])} |"
        )
    lines.append("")

    # ── 3. Influencer별 최대 기여 target ─────────────────────────────────
    lines += ["## 3. Influencer별 — 가장 많이 기여한 게임", ""]
    lines += ["| Influencer (X) | 최고 기여 Target (Y) | Impact | 최저 기여 Target | Impact |"]
    lines += ["| --- | --- | --- | --- | --- |"]
    for inf in games:
        pairs = [
            ((i, tgt), stat) for (i, tgt), stat in mdata_all.items() if i == inf
        ]
        if not pairs:
            continue
        pairs_sorted = sorted(pairs, key=lambda x: x[1]["impact"], reverse=True)
        best_tgt,  best_stat  = pairs_sorted[0][0][1],  pairs_sorted[0][1]
        worst_tgt, worst_stat = pairs_sorted[-1][0][1], pairs_sorted[-1][1]
        lines.append(
            f"| **{inf}** | {best_tgt} | {_fmt(best_stat['impact'])} "
            f"| {worst_tgt} | {_fmt(worst_stat['impact'])} |"
        )
    lines.append("")

    # ── 4. Reward enum별 top-N 쌍 ────────────────────────────────────────
    lines += [f"## 4. Reward Enum별 상위 {top_n} 쌍 (Progress)", ""]
    for re in re_keys:
        re_label = re_labels_cfg.get(re, f"re={re}")
        mdata_re = per_re_impact[re].get(metric, {})
        if not mdata_re:
            continue
        pairs_re = sorted(mdata_re.items(), key=lambda x: x[1]["impact"], reverse=True)
        lines += [f"### {re_label}", ""]
        lines += [f"| 순위 | Influencer→Target | Impact | perf(seen) | perf(unseen) |"]
        lines += ["| --- | --- | --- | --- | --- |"]
        for rank, ((inf, tgt), stat) in enumerate(pairs_re[:top_n], 1):
            lines.append(
                f"| {rank} | {inf} → **{tgt}** | {_fmt(stat['impact'])} "
                f"| {stat['perf_seen']:.4f} | {stat['perf_unseen']:.4f} |"
            )
        lines.append("")

    # ── 5. 음수 impact 쌍 (역효과) ───────────────────────────────────────
    neg_pairs = [(pair, stat) for pair, stat in all_pairs if stat["impact"] < 0]
    if neg_pairs:
        lines += ["## 5. 주목할 패턴 — 음수 Impact 쌍 (전체 re 평균)", ""]
        lines += ["> 해당 게임의 학습 데이터가 오히려 없을 때 성능이 더 높은 쌍.", ""]
        lines += ["| Influencer→Target | Impact | perf(seen) | perf(unseen) |"]
        lines += ["| --- | --- | --- | --- |"]
        for (inf, tgt), stat in sorted(neg_pairs, key=lambda x: x[1]["impact"]):
            lines.append(
                f"| {inf} → **{tgt}** | {_fmt(stat['impact'])} "
                f"| {stat['perf_seen']:.4f} | {stat['perf_unseen']:.4f} |"
            )
        lines.append("")

    # ── 6. 전체 Impact 행렬 요약 (re 평균) ─────────────────────────────
    lines += ["## 6. Impact 행렬 요약 (행=Target, 열=Influencer, 전체 re 평균)", ""]
    header = ["Target \\ Influencer"] + games
    lines += ["| " + " | ".join(header) + " |"]
    lines += ["| " + " | ".join(["---"] * len(header)) + " |"]
    for tgt in games:
        cells = [f"**{tgt}**"]
        for inf in games:
            if inf == tgt:
                cells.append("–")
            else:
                stat = mdata_all.get((inf, tgt))
                cells.append(_fmt(stat["impact"]) if stat else "N/A")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    _exp_names = list(_CFG.get("experiments", {}).keys())
    parser = argparse.ArgumentParser(
        description="게임 간 학습 데이터 영향도 분석 (impact heatmap + bar chart)"
    )
    parser.add_argument("--input", default="wandb_projects")
    parser.add_argument("--metrics", nargs="+", default=None)
    parser.add_argument("--decimals", type=int, default=4)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument(
        "--reward-enum", default=None,
        help="특정 reward_enum 만 분석 (기본: 전체 reward_enum 평균)"
    )
    parser.add_argument(
        "--experiment",
        choices=_exp_names if _exp_names else None,
        default="unseen_generalizability",
        metavar="EXPERIMENT",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args    = parse_args()
    run_dir = make_run_dir("game_impact_analysis", cfg=_CFG)
    log     = setup_logger(run_dir, name=__file__)
    log.debug("run_dir    : %s", run_dir)

    experiment   = args.experiment or "unseen_generalizability"
    folder_order = _get_experiment_folder_order(experiment)
    log.info("experiment : %s  folder_order=%s", experiment, folder_order)

    input_root = resolve_input_root(args.input, _RESULTS_DIR)
    log.info("input_root : %s", input_root)

    metric_order: list[str] = args.metrics or DEFAULT_METRIC_ORDER.copy()

    rows = collect_rows_with_game_sets(
        input_root, metric_order,
        target_projects=folder_order or None,
    )
    if not rows:
        raise SystemExit("No valid rows found.")

    if not any(r["game_split"] == "unseen" for r in rows):
        raise SystemExit("unseen 게임 데이터가 없습니다 — run_config.json 의 unseen_games 를 확인하세요.")

    # Normalization
    _global_scale_path = os.environ.get("PIPELINE_NORM_SCALE")
    if _global_scale_path and Path(_global_scale_path).is_file():
        norm_scale = load_normalization_scale(Path(_global_scale_path))
        log.info("norm_scale (global): %s", _global_scale_path)
    else:
        norm_scale = compute_normalization_scale(rows, metric_order)
        save_normalization_scale(norm_scale, run_dir / "normalization_scale.json")
        log.info("norm_scale (local) computed")
    norm_rows = apply_normalization(rows, norm_scale, metric_order)

    # Impact 계산 — 전체(re 평균) + re별
    games, impact_map = compute_game_impact(norm_rows, metric_order, reward_enum=None)
    log.info("games      : %s", games)

    # re별 impact (바 차트용)
    all_re = sorted(
        {r["reward_enum"] for r in norm_rows if r["game_split"] == "unseen"},
        key=lambda x: (int(x) if x.isdigit() else 99),
    )
    per_re_impact: dict[str, dict] = {}
    for re in all_re:
        _, re_map = compute_game_impact(norm_rows, metric_order, reward_enum=re)
        per_re_impact[re] = re_map
    log.info("reward_enums: %s", all_re)

    total_pairs = sum(len(v) for v in impact_map.values())
    log.info("impact pairs: %d", total_pairs)


    if not games:
        raise SystemExit("Impact 계산 결과가 없습니다 — seen/unseen 게임 쌍 데이터를 확인하세요.")

    # ── 테이블 ──────────────────────────────────────────────────────────────
    write_impact_csv(run_dir / "game_impact_table.csv", games, impact_map, metric_order)
    write_impact_markdown(
        run_dir / "game_impact_table.md", games, impact_map, metric_order, args.decimals
    )
    log.info("table      : %s", run_dir / "game_impact_table.md")

    write_impact_report(
        run_dir / "game_impact_report.md",
        games, impact_map, per_re_impact,
        metric="progress",
    )
    log.info("report     : %s", run_dir / "game_impact_report.md")

    # ── 플롯 ────────────────────────────────────────────────────────────────
    if not args.no_plot:
        try:
            write_impact_heatmap(
                run_dir / "game_impact_heatmap.png", games, per_re_impact, metric_order
            )
            log.info("heatmap    : %s", run_dir / "game_impact_heatmap.png")
        except Exception as e:
            log.error("Heatmap 생성 실패: %s", e)

        try:
            write_impact_bar(
                run_dir / "game_impact_bar.png", games, per_re_impact, metric="progress"
            )
            log.info("bar chart  : %s", run_dir / "game_impact_bar.png")
        except Exception as e:
            log.error("Bar 차트 생성 실패: %s", e)

    log.info(
        "done — rows=%d  unseen=%d",
        len(rows), sum(1 for r in rows if r["game_split"] == "unseen"),
    )


if __name__ == "__main__":
    main()


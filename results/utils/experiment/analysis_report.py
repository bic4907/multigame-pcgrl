"""
analysis_report.py
==================
각 실험(allseen / unseen)에 대해
모델 간 수치 비교를 한글로 작성한 분석 리포트를 생성한다.

- Baseline 대비 비교 (절대값 및 %)
- 이전 방법 대비 순차 비교
- 메트릭 방향 고려 (Progress ↑, ViTScore ↑, TPKL-Div ↓, Diversity ↑)

출력:
    analysis_report.md   — 노션에 붙여넣기 편한 한글 Markdown 분석 리포트

사용법:
    python results/utils/experiment/analysis_report.py
    python results/utils/experiment/analysis_report.py --experiment allseen
    python results/utils/experiment/analysis_report.py --experiment unseen
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import sys as _sys

_HERE        = Path(__file__).resolve().parent   # results/utils/experiment/
_RESULTS_DIR = _HERE.parent.parent               # results/
_ROOT        = _HERE.parent.parent.parent        # project root

if str(_RESULTS_DIR) not in _sys.path:
    _sys.path.insert(0, str(_RESULTS_DIR))
if str(_ROOT) not in _sys.path:
    _sys.path.append(str(_ROOT))

from utils.core.run_output import load_cfg, make_run_dir, setup_logger
from utils.experiment.benchmark import (
    collect_plot_rows_from_results,
    resolve_input_root,
    _get_experiment_folder_order,
    _load_project_display_names,
    _project_display_name,
    _sort_folder_for_plot,
    _seed_agg,
    DEFAULT_METRIC_ORDER,
    METRIC_DISPLAY_NAMES,
)

_CFG = load_cfg()

# ── 메트릭 방향 정의 ──────────────────────────────────────────────────────────
# True = 높을수록 좋음, False = 낮을수록 좋음
METRIC_HIGHER_IS_BETTER: dict[str, bool] = {
    "progress":  True,
    "vit_score": True,
    "tpkldiv":   False,
    "diversity": True,
}

METRIC_DIRECTION_LABEL: dict[str, str] = {
    "progress":  "높을수록 좋음 ↑",
    "vit_score": "높을수록 좋음 ↑",
    "tpkldiv":   "낮을수록 좋음 ↓",
    "diversity": "높을수록 좋음 ↑",
}

# ── 실험별 비교 체인 정의 ─────────────────────────────────────────────────────
# target_projects 순서로 결정; 여기서는 config.json 기반으로 사용
# 아래는 fallback 기본값이며, config에서 항상 override됨


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    _exp_names = list(_CFG.get("experiments", {}).keys())
    parser = argparse.ArgumentParser(
        description="실험별 한글 분석 리포트 생성"
    )
    parser.add_argument("--input", default="wandb_projects")
    parser.add_argument("--metrics", nargs="+", default=None)
    parser.add_argument("--decimals", type=int, default=4)
    parser.add_argument(
        "--experiment",
        choices=_exp_names if _exp_names else None,
        default=None,
        metavar="EXPERIMENT",
        help=f"분석할 실험 (선택: {', '.join(_exp_names)}). 미지정 시 모든 실험 분석.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Aggregation: project별 전체 평균 (game / reward_enum 무시)
# ---------------------------------------------------------------------------

def aggregate_by_project(
    plot_rows: list[dict],
    metric_order: list[str],
    project_filter: list[str] | None = None,
) -> dict[str, dict[str, dict]]:
    """project → metric → {mean, std, n} 딕셔너리 반환.

    같은 시드 먼저 평균 → 시드 간 mean / std.
    """
    by_project: dict[str, list[dict]] = defaultdict(list)
    for row in plot_rows:
        proj = row["project"]
        if project_filter is None or proj in project_filter:
            by_project[proj].append(row)

    result: dict[str, dict[str, dict]] = {}
    for proj, rows in by_project.items():
        stats: dict[str, dict] = {}
        for metric in metric_order:
            stat = _seed_agg(rows, metric)
            if stat:
                stats[metric] = stat
        result[proj] = stats
    return result


def aggregate_by_project_and_split(
    plot_rows: list[dict],
    metric_order: list[str],
    project_filter: list[str] | None = None,
) -> dict[tuple, dict]:
    """(project, game_split) → metric → {mean, std, n} 딕셔너리."""
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in plot_rows:
        proj  = row["project"]
        split = row.get("game_split", "unknown")
        if project_filter is None or proj in project_filter:
            grouped[(proj, split)].append(row)

    result: dict[tuple, dict] = {}
    for key, rows in grouped.items():
        stats: dict[str, dict] = {}
        for metric in metric_order:
            stat = _seed_agg(rows, metric)
            if stat:
                stats[metric] = stat
        result[key] = stats
    return result


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------

def _pct_change(new_val: float, ref_val: float, metric: str) -> float:
    """개선율(%) 계산. 항상 양수 = 향상, 음수 = 하락.

    - higher-is-better: (new - ref) / |ref| * 100
    - lower-is-better : (ref - new) / |ref| * 100  (← TPKL-Div)
    """
    if ref_val == 0:
        return float("nan")
    higher = METRIC_HIGHER_IS_BETTER.get(metric, True)
    if higher:
        return (new_val - ref_val) / abs(ref_val) * 100
    else:
        return (ref_val - new_val) / abs(ref_val) * 100


def _fmt_pct(pct: float) -> str:
    if math.isnan(pct):
        return "N/A"
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.1f}%"


def _judge(pct: float) -> str:
    if math.isnan(pct):
        return "데이터 없음"
    if pct > 5:
        return '<span style="color:#16a34a">**향상** 🟢</span>'
    elif pct > 0:
        return '<span style="color:#84cc16">소폭 향상 🟡</span>'
    elif pct > -5:
        return '<span style="color:#f97316">소폭 하락 🟠</span>'
    else:
        return '<span style="color:#dc2626">**하락** 🔴</span>'


def _fmt_stat(stat: dict | None, decimals: int = 4) -> str:
    if stat is None:
        return "-"
    return f"{stat['mean']:.{decimals}f} ± {stat['std']:.{decimals}f}"


# ---------------------------------------------------------------------------
# Markdown section builders
# ---------------------------------------------------------------------------

def _section_overall_table(
    proj_stats: dict[str, dict],
    folder_order: list[str],
    metric_order: list[str],
    decimals: int,
    baseline_proj: str | None,
    label_suffixes: dict[str, str] | None = None,
) -> list[str]:
    """전체 성능 요약 테이블.

    label_suffixes: {proj_key: " *(표시할 레이블)*"} 형태로 추가 접미사 지정.
    baseline_proj 는 자동으로 " *(Baseline)*" 가 붙음 (label_suffixes 미지정 시).
    """
    lines: list[str] = []
    lines.append("## 📊 전체 성능 요약\n")
    lines.append("> 각 수치는 모든 게임 / reward 조건 평균입니다.\n")

    # 헤더 행
    metric_headers = [METRIC_DISPLAY_NAMES.get(m, m) for m in metric_order]
    header_cols = ["모델"] + metric_headers
    lines.append("| " + " | ".join(header_cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(header_cols)) + " |")

    for proj in folder_order:
        if proj not in proj_stats:
            continue
        display = _project_display_name(proj)
        if label_suffixes and proj in label_suffixes:
            display += f" {label_suffixes[proj]}"
        elif proj == baseline_proj:
            display += " *(Baseline)*"
        stats = proj_stats[proj]
        cells = [display]
        for m in metric_order:
            cells.append(_fmt_stat(stats.get(m), decimals))
        lines.append("| " + " | ".join(cells) + " |")

    lines.append("")
    return lines


def _oracle_level_str(new_val: float, ref_val: float, metric: str, decimals: int) -> str:
    """Oracle 대비 '수준' 표현 반환.

    higher-is-better 메트릭: new/ref * 100 → 'Oracle의 X.X% 수준'
    lower-is-better 메트릭 : ref/new * 100 → 'Oracle 기준 X.X% 수준' (낮을수록 좋음)
    """
    if ref_val == 0 or new_val == 0:
        return "참고값 없음"
    higher = METRIC_HIGHER_IS_BETTER.get(metric, True)
    if higher:
        ratio = new_val / ref_val * 100
    else:
        ratio = ref_val / new_val * 100
    return f"Oracle의 {ratio:.1f}% 수준"


def _section_oracle_compare(
    proj_stats: dict[str, dict],
    folder_order: list[str],
    metric_order: list[str],
    decimals: int,
    oracle_proj: str,
    oracle_label: str,
    target_projs: list[str] | None = None,
) -> list[str]:
    """Oracle 대비 비교 — 판정 없이 '수준'으로만 표현.

    직접적 성능 우열을 따지지 않고, 모델 값이 Oracle의 몇 % 수준인지만 기재.
    """
    lines: list[str] = []
    lines.append(f"## 🏅 Oracle ({oracle_label}) 대비 참고 비교\n")
    lines.append("> Oracle은 직접 비교 대상이 아닌 참고 기준입니다. 절대치와 수준(%)만 표기합니다.\n")

    oracle_stats = proj_stats.get(oracle_proj, {})
    if not oracle_stats:
        lines.append(f"> ⚠️ Oracle 데이터({oracle_proj})를 찾을 수 없습니다.\n")
        return lines

    for proj in folder_order:
        if proj == oracle_proj or proj not in proj_stats:
            continue
        if target_projs is not None and proj not in target_projs:
            continue
        display = _project_display_name(proj)
        lines.append(f"### {display} vs {oracle_label}\n")
        stats = proj_stats[proj]
        for m in metric_order:
            m_display = METRIC_DISPLAY_NAMES.get(m, m)
            direction = METRIC_DIRECTION_LABEL.get(m, "")
            new_stat  = stats.get(m)
            ref_stat  = oracle_stats.get(m)
            if new_stat is None or ref_stat is None:
                lines.append(f"- **{m_display}** ({direction}): 데이터 없음")
                continue
            level = _oracle_level_str(new_stat["mean"], ref_stat["mean"], m, decimals)
            lines.append(
                f"- **{m_display}** ({direction}): "
                f"Oracle `{ref_stat['mean']:.{decimals}f}` | 모델 `{new_stat['mean']:.{decimals}f}` "
                f"→ {level}"
            )
        lines.append("")

    return lines


def _section_baseline_compare(
    proj_stats: dict[str, dict],
    folder_order: list[str],
    metric_order: list[str],
    decimals: int,
    baseline_proj: str,
    baseline_label: str,
    target_projs: list[str] | None = None,
) -> list[str]:
    """Baseline 대비 각 모델 비교.

    target_projs 가 주어지면 해당 모델만 비교한다.
    """
    lines: list[str] = []
    lines.append(f"## 🔍 Baseline ({baseline_label}) 대비 비교\n")

    baseline_stats = proj_stats.get(baseline_proj, {})
    if not baseline_stats:
        lines.append(f"> ⚠️ Baseline 데이터({baseline_proj})를 찾을 수 없습니다.\n")
        return lines

    for proj in folder_order:
        if proj == baseline_proj or proj not in proj_stats:
            continue
        if target_projs is not None and proj not in target_projs:
            continue
        display = _project_display_name(proj)
        lines.append(f"### {display} vs {baseline_label}\n")
        stats = proj_stats[proj]
        for m in metric_order:
            m_display    = METRIC_DISPLAY_NAMES.get(m, m)
            direction    = METRIC_DIRECTION_LABEL.get(m, "")
            new_stat     = stats.get(m)
            ref_stat     = baseline_stats.get(m)
            if new_stat is None or ref_stat is None:
                lines.append(f"- **{m_display}** ({direction}): 데이터 없음")
                continue
            pct   = _pct_change(new_stat["mean"], ref_stat["mean"], m)
            judge = _judge(pct)
            lines.append(
                f"- **{m_display}** ({direction}): "
                f"`{ref_stat['mean']:.{decimals}f}` → `{new_stat['mean']:.{decimals}f}` "
                f"| {_fmt_pct(pct)} {judge}"
            )
        lines.append("")

    return lines


def _section_compare_pairs(
    proj_stats: dict[str, dict],
    pairs: list[tuple[str, str]],
    metric_order: list[str],
    decimals: int,
    section_title: str = "## 🔄 이전 방법 대비 비교\n",
    section_note: str = "> 지정된 모델 쌍에 대해 변화량을 표시합니다.\n",
) -> list[str]:
    """명시적 (ref_proj, target_proj) 쌍에 대한 비교.

    pairs: [(ref, target), ...] — target이 새 모델, ref이 이전 모델.
    """
    lines: list[str] = []
    lines.append(section_title)
    lines.append(section_note)

    for ref_proj, tgt_proj in pairs:
        ref_stats = proj_stats.get(ref_proj, {})
        tgt_stats = proj_stats.get(tgt_proj, {})
        if not ref_stats or not tgt_stats:
            continue
        ref_display = _project_display_name(ref_proj)
        tgt_display = _project_display_name(tgt_proj)
        lines.append(f"### {tgt_display} vs {ref_display}\n")
        for m in metric_order:
            m_display = METRIC_DISPLAY_NAMES.get(m, m)
            direction = METRIC_DIRECTION_LABEL.get(m, "")
            r_stat    = ref_stats.get(m)
            t_stat    = tgt_stats.get(m)
            if r_stat is None or t_stat is None:
                lines.append(f"- **{m_display}** ({direction}): 데이터 없음")
                continue
            pct   = _pct_change(t_stat["mean"], r_stat["mean"], m)
            judge = _judge(pct)
            lines.append(
                f"- **{m_display}** ({direction}): "
                f"`{r_stat['mean']:.{decimals}f}` → `{t_stat['mean']:.{decimals}f}` "
                f"| {_fmt_pct(pct)} {judge}"
            )
        lines.append("")

    return lines


def _section_sequential_compare(
    proj_stats: dict[str, dict],
    folder_order: list[str],
    metric_order: list[str],
    decimals: int,
) -> list[str]:
    """이전 방법 대비 순차 비교 (folder_order 순서 기준)."""
    lines: list[str] = []
    lines.append("## 🔄 이전 방법 대비 순차 비교\n")
    lines.append("> 모델 개발 순서(위 → 아래) 기준으로 직전 모델 대비 변화량을 표시합니다.\n")

    available = [p for p in folder_order if p in proj_stats]
    if len(available) < 2:
        lines.append("> ⚠️ 비교할 모델이 2개 이상 필요합니다.\n")
        return lines

    for i in range(1, len(available)):
        prev_proj    = available[i - 1]
        curr_proj    = available[i]
        prev_display = _project_display_name(prev_proj)
        curr_display = _project_display_name(curr_proj)
        lines.append(f"### {curr_display} vs {prev_display}\n")

        prev_stats = proj_stats[prev_proj]
        curr_stats = proj_stats[curr_proj]

        for m in metric_order:
            m_display = METRIC_DISPLAY_NAMES.get(m, m)
            direction = METRIC_DIRECTION_LABEL.get(m, "")
            curr_stat = curr_stats.get(m)
            prev_stat = prev_stats.get(m)
            if curr_stat is None or prev_stat is None:
                lines.append(f"- **{m_display}** ({direction}): 데이터 없음")
                continue
            pct   = _pct_change(curr_stat["mean"], prev_stat["mean"], m)
            judge = _judge(pct)
            lines.append(
                f"- **{m_display}** ({direction}): "
                f"`{prev_stat['mean']:.{decimals}f}` → `{curr_stat['mean']:.{decimals}f}` "
                f"| {_fmt_pct(pct)} {judge}"
            )
        lines.append("")

    return lines


def _section_seen_unseen_compare(
    split_stats: dict[tuple, dict],
    folder_order: list[str],
    metric_order: list[str],
    decimals: int,
) -> list[str]:
    """seen vs unseen 비교 (unseen 전용)."""
    lines: list[str] = []
    lines.append("## 🌐 Seen vs Unseen 일반화 비교\n")
    lines.append("> 학습에 사용된 게임(seen)과 미사용 게임(unseen)에서의 성능 차이를 나타냅니다.\n")

    for proj in folder_order:
        seen_stats   = split_stats.get((proj, "seen"),   {})
        unseen_stats = split_stats.get((proj, "unseen"), {})
        # seen·unseen 둘 다 있는 모델만 표시 (baseline은 unseen 없으므로 제외)
        if not seen_stats or not unseen_stats:
            continue

        display = _project_display_name(proj)
        lines.append(f"### {display}\n")

        metric_header = ["메트릭 (방향)", "Seen", "Unseen", "변화율", "판정"]
        lines.append("| " + " | ".join(metric_header) + " |")
        lines.append("| " + " | ".join(["---"] * len(metric_header)) + " |")

        for m in metric_order:
            m_display = METRIC_DISPLAY_NAMES.get(m, m)
            direction = METRIC_DIRECTION_LABEL.get(m, "")
            s_stat    = seen_stats.get(m)
            u_stat    = unseen_stats.get(m)
            s_str     = _fmt_stat(s_stat, decimals)
            u_str     = _fmt_stat(u_stat, decimals)
            if s_stat and u_stat:
                pct   = _pct_change(u_stat["mean"], s_stat["mean"], m)
                judge = _judge(pct)
                pct_str = _fmt_pct(pct)
            else:
                pct_str = "-"
                judge   = "-"
            lines.append(
                f"| **{m_display}** ({direction}) | {s_str} | {u_str} | {pct_str} | {judge} |"
            )
        lines.append("")

    # 모델 간 unseen 성능 비교
    unseen_projs = [p for p in folder_order if split_stats.get((p, "unseen"))]
    if len(unseen_projs) >= 2:
        lines.append("### 모델 간 Unseen 성능 비교\n")
        for i in range(1, len(unseen_projs)):
            prev_proj    = unseen_projs[i - 1]
            curr_proj    = unseen_projs[i]
            prev_display = _project_display_name(prev_proj)
            curr_display = _project_display_name(curr_proj)
            lines.append(f"#### {curr_display} vs {prev_display} (Unseen 기준)\n")
            prev_s = split_stats.get((prev_proj, "unseen"), {})
            curr_s = split_stats.get((curr_proj, "unseen"), {})
            for m in metric_order:
                m_display = METRIC_DISPLAY_NAMES.get(m, m)
                direction = METRIC_DIRECTION_LABEL.get(m, "")
                c_stat    = curr_s.get(m)
                p_stat    = prev_s.get(m)
                if c_stat is None or p_stat is None:
                    lines.append(f"- **{m_display}**: 데이터 없음")
                    continue
                pct   = _pct_change(c_stat["mean"], p_stat["mean"], m)
                judge = _judge(pct)
                lines.append(
                    f"- **{m_display}** ({direction}): "
                    f"`{p_stat['mean']:.{decimals}f}` → `{c_stat['mean']:.{decimals}f}` "
                    f"| {_fmt_pct(pct)} {judge}"
                )
            lines.append("")

    return lines


def _section_summary_verdict(
    proj_stats: dict[str, dict],
    folder_order: list[str],
    metric_order: list[str],
    baseline_proj: str | None,
    experiment: str,
) -> list[str]:
    """종합 평가 텍스트."""
    lines: list[str] = []
    lines.append("## 📝 종합 평가\n")

    available = [p for p in folder_order if p in proj_stats]
    best_model_proj = available[-1] if available else None

    if best_model_proj and baseline_proj and best_model_proj != baseline_proj:
        best_display = _project_display_name(best_model_proj)
        base_display = _project_display_name(baseline_proj)
        best_stats   = proj_stats.get(best_model_proj, {})
        base_stats   = proj_stats.get(baseline_proj, {})

        verdicts: list[str] = []
        for m in metric_order:
            b_s  = best_stats.get(m)
            bs_s = base_stats.get(m)
            if b_s is None or bs_s is None:
                continue
            pct = _pct_change(b_s["mean"], bs_s["mean"], m)
            m_display = METRIC_DISPLAY_NAMES.get(m, m)
            if not math.isnan(pct):
                direction = "향상" if pct >= 0 else "하락"
                verdicts.append(f"{m_display} {_fmt_pct(pct)} {direction}")

        lines.append(
            f"**{experiment}** 실험에서 최종 모델 **{best_display}**는 "
            f"Baseline(**{base_display}**)에 비해 "
            + ", ".join(verdicts) + "을 보였습니다.\n"
        )
    elif best_model_proj:
        lines.append(
            f"**{experiment}** 실험 최종 모델: **{_project_display_name(best_model_proj)}**\n"
        )

    # 메트릭 방향 안내
    lines.append("\n> **메트릭 방향 안내**")
    for m in metric_order:
        m_display = METRIC_DISPLAY_NAMES.get(m, m)
        direction = METRIC_DIRECTION_LABEL.get(m, "")
        lines.append(f"> - {m_display}: {direction}")
    lines.append("")

    return lines


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------

def build_report_allseen(
    plot_rows: list[dict],
    folder_order: list[str],
    metric_order: list[str],
    decimals: int,
    experiment: str,
    exp_cfg: dict,
) -> list[str]:
    lines: list[str] = []
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines.append(f"# 📋 실험 분석 리포트: `{experiment}`\n")
    lines.append(f"> 생성일시: {now_str}  \n")
    lines.append(f"> 분석 대상 모델: {', '.join(_project_display_name(p) for p in folder_order)}\n")
    lines.append("---\n")

    proj_stats = aggregate_by_project(plot_rows, metric_order, folder_order)

    oracle_proj    = exp_cfg.get("re_oracle_project")
    oracle_label   = exp_cfg.get("re_oracle_label") or (
        _project_display_name(oracle_proj) if oracle_proj else "Oracle"
    )
    baseline_proj  = exp_cfg.get("re_baseline_project")
    baseline_label = exp_cfg.get("re_baseline_label") or (
        _project_display_name(baseline_proj) if baseline_proj else "Baseline"
    )

    # folder_order 내 실제 데이터가 있는 프로젝트 목록
    available = [p for p in folder_order if p in proj_stats]
    last_proj = available[-1] if available else None

    # 비교에서 제외할 프로젝트 목록 (random, oracle, baseline 등)
    exclude_from_compare: list[str] = exp_cfg.get("re_exclude_from_compare", [])

    # 테이블에 Oracle / Baseline 레이블 표시
    label_suffixes: dict[str, str] = {}
    if oracle_proj:
        label_suffixes[oracle_proj] = "*(Oracle)*"
    if baseline_proj:
        label_suffixes[baseline_proj] = "*(Baseline)*"

    # 1. 전체 성능 요약
    lines += _section_overall_table(
        proj_stats, folder_order, metric_order, decimals,
        baseline_proj=None, label_suffixes=label_suffixes,
    )
    lines += ["---\n"]

    # 2. Oracle(CPCGRL) 대비 참고 비교 — 마지막 모델만 (수준 표현)
    if oracle_proj and last_proj and last_proj != oracle_proj:
        lines += _section_oracle_compare(
            proj_stats, folder_order, metric_order, decimals,
            oracle_proj=oracle_proj,
            oracle_label=oracle_label,
            target_projs=[last_proj],
        )
        lines += ["---\n"]

    # 3. Baseline(VIPCGRL) 대비 비교 — 마지막 모델만 (% 향상/하락)
    if baseline_proj and last_proj and last_proj != baseline_proj:
        lines += _section_baseline_compare(
            proj_stats, folder_order, metric_order, decimals,
            baseline_proj=baseline_proj,
            baseline_label=baseline_label,
            target_projs=[last_proj],
        )
        lines += ["---\n"]

    # 4. 이전 방법 대비 비교 — exclude_from_compare 제외한 이전 모델들 vs last
    if last_proj:
        ref_projs = [
            p for p in available
            if p != last_proj and p not in exclude_from_compare
        ]
        pairs = [(ref, last_proj) for ref in ref_projs]
        if pairs:
            ref_names = "·".join(_project_display_name(r) for r in ref_projs)
            lines += _section_compare_pairs(
                proj_stats, pairs, metric_order, decimals,
                section_title="## 🔄 이전 방법 대비 비교\n",
                section_note=f"> {_project_display_name(last_proj)} 를 {ref_names} 등 이전 방법과 비교합니다.\n",
            )
            lines += ["---\n"]

    # 5. 종합 평가 (baseline 기준)
    lines += _section_summary_verdict(proj_stats, folder_order, metric_order, baseline_proj, experiment)

    return lines


def build_report_unseen(
    plot_rows: list[dict],
    folder_order: list[str],
    metric_order: list[str],
    decimals: int,
    experiment: str,
    exp_cfg: dict,
) -> list[str]:
    lines: list[str] = []
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines.append(f"# 📋 실험 분석 리포트: `{experiment}`\n")
    lines.append(f"> 생성일시: {now_str}  \n")
    lines.append(f"> 분석 대상 모델: {', '.join(_project_display_name(p) for p in folder_order)}\n")
    lines.append("---\n")

    # proj별 전체 집계 (seen + unseen 합산)
    proj_stats  = aggregate_by_project(plot_rows, metric_order, folder_order)
    # split별 집계
    split_stats = aggregate_by_project_and_split(plot_rows, metric_order, folder_order)

    baseline_proj  = exp_cfg.get("seen_baseline_project")   # mgpcgrl
    baseline_label = exp_cfg.get("seen_baseline_label") or (
        _project_display_name(baseline_proj) if baseline_proj else "Baseline"
    )
    oracle_proj    = exp_cfg.get("re_oracle_project")        # cpcgrl
    oracle_label   = exp_cfg.get("re_oracle_label") or (
        _project_display_name(oracle_proj) if oracle_proj else "Oracle"
    )

    # folder_order 내 데이터가 있는 프로젝트 / 마지막(target) 모델
    available = [p for p in folder_order if p in proj_stats]
    last_proj = available[-1] if available else None

    # 테이블에 Oracle / Baseline 레이블 표시
    label_suffixes: dict[str, str] = {}
    if oracle_proj:
        label_suffixes[oracle_proj] = "*(Oracle)*"
    if baseline_proj:
        label_suffixes[baseline_proj] = "*(Baseline)*"

    # 1. 전체 성능 요약
    lines += _section_overall_table(
        proj_stats, folder_order, metric_order, decimals,
        baseline_proj=None, label_suffixes=label_suffixes,
    )
    lines += ["---\n"]

    # 2. Seen vs Unseen 일반화 비교 (target 모델만)
    lines += _section_seen_unseen_compare(split_stats, folder_order, metric_order, decimals)
    lines += ["---\n"]

    # 3. Oracle(CPCGRL) 대비 참고 비교 — target 모델만 (수준 표현)
    if oracle_proj and last_proj and last_proj != oracle_proj:
        lines += _section_oracle_compare(
            proj_stats, folder_order, metric_order, decimals,
            oracle_proj=oracle_proj,
            oracle_label=oracle_label,
            target_projs=[last_proj],
        )
        lines += ["---\n"]

    # 4. Baseline(MGPCGRL) 대비 비교 — target 모델만 (% 향상/하락)
    if baseline_proj and last_proj and last_proj != baseline_proj:
        lines += _section_baseline_compare(
            proj_stats, folder_order, metric_order, decimals,
            baseline_proj=baseline_proj,
            baseline_label=baseline_label,
            target_projs=[last_proj],
        )
        lines += ["---\n"]

    # 5. 종합 평가 (baseline 기준)
    lines += _section_summary_verdict(proj_stats, folder_order, metric_order, baseline_proj, experiment)

    return lines


def build_report(
    plot_rows: list[dict],
    folder_order: list[str],
    metric_order: list[str],
    decimals: int,
    experiment: str,
    exp_cfg: dict,
) -> str:
    if experiment == "unseen":
        lines = build_report_unseen(
            plot_rows, folder_order, metric_order, decimals, experiment, exp_cfg
        )
    else:
        lines = build_report_allseen(
            plot_rows, folder_order, metric_order, decimals, experiment, exp_cfg
        )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args    = parse_args()
    run_dir = make_run_dir("analysis_report", cfg=_CFG)
    log     = setup_logger(run_dir, name=__file__)
    log.debug("run_dir   : %s", run_dir)

    script_dir = _RESULTS_DIR
    input_root = resolve_input_root(args.input, script_dir)
    log.info("input_root: %s", input_root)

    metric_order: list[str] = args.metrics or DEFAULT_METRIC_ORDER.copy()

    all_experiments = list(_CFG.get("experiments", {}).keys())
    target_experiments = [args.experiment] if args.experiment else all_experiments

    all_reports: list[str] = []

    for experiment in target_experiments:
        exp_cfg      = _CFG.get("experiments", {}).get(experiment, {})
        folder_order = _get_experiment_folder_order(experiment)

        # benchmark 모듈의 전역 상태 갱신 (정렬용)
        import utils.experiment.benchmark as _bm
        _bm.PREFERRED_PLOT_FOLDER_ORDER = folder_order
        _bm._PROJECT_DISPLAY_NAMES      = _load_project_display_names(experiment)

        # 데이터 수집
        plot_rows = collect_plot_rows_from_results(input_root, metric_order)
        if folder_order:
            plot_rows = [r for r in plot_rows if r["project"] in folder_order]

        if not plot_rows:
            log.warning("[%s] results.csv 데이터를 찾을 수 없습니다. 건너뜁니다.", experiment)
            all_reports.append(f"# 📋 실험 분석 리포트: `{experiment}`\n\n> ⚠️ 데이터 없음\n\n---\n")
            continue

        log.info("[%s] rows=%d, folder_order=%s", experiment, len(plot_rows), folder_order)

        report_text = build_report(plot_rows, folder_order, metric_order, args.decimals, experiment, exp_cfg)
        all_reports.append(report_text)

        # 실험별 개별 파일도 저장
        exp_path = run_dir / f"analysis_report_{experiment}.md"
        exp_path.write_text(report_text, encoding="utf-8")
        log.info("[%s] report → %s", experiment, exp_path)

    # 전체 실험 통합 파일
    if len(target_experiments) > 1:
        combined = "\n\n---\n\n".join(all_reports)
    else:
        combined = all_reports[0] if all_reports else ""

    out_path = run_dir / "analysis_report.md"
    out_path.write_text(combined, encoding="utf-8")
    log.info("analysis_report → %s", out_path)


if __name__ == "__main__":
    main()


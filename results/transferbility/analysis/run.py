"""Orchestrator for the transferability analysis.

Run from the repo root:

    python -m results.transferbility.analysis.run

Produces (in ``results/transferbility/output/``):
    tables/   CSVs for every intermediate result
    *.png     figures
    report.md a narrative summary of the findings
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import config, correlate as C, deeper as Dp, distances as Dist, plots


def _write(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _fmt(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    return df.to_markdown(index=False, floatfmt=floatfmt)


def main() -> None:
    out = config.OUTPUT_DIR
    tables = out / "tables"
    out.mkdir(parents=True, exist_ok=True)

    # ── 1. Distribution descriptors + pair features ─────────────────────────────
    desc = Dist.descriptor_table()
    pairs = Dist.pair_feature_table()
    merged = C.merged_feature_table()
    _write(desc, tables / "distribution_descriptors.csv")
    _write(pairs, tables / "pair_features.csv")
    _write(merged, tables / "merged_feature_delta.csv")

    # ── 2. Naive pooled correlation (per-feature + overall) ─────────────────────
    naive = C.correlations(merged, C.PREDICTORS)
    overall = C.overall_feature_table()
    overall_corr = C.correlations(overall, [p for p in C.PREDICTORS if p in overall.columns])
    per_enum = C.per_enum_correlations(merged, C.PREDICTORS)
    _write(naive, tables / "corr_naive_pooled.csv")
    _write(overall_corr, tables / "corr_overall.csv")
    _write(per_enum, tables / "corr_per_enum.csv")

    # ── 3. Deeper: scale-controlled, partial corr, OLS, head-room, absence ──────
    controlled = Dp.scale_controlled_correlations()
    partials = pd.DataFrame([
        Dp.partial_correlation("js_distance"),
        Dp.partial_correlation("overlap_coef"),
        Dp.partial_correlation("ks_stat"),
    ])
    ols = pd.DataFrame([
        Dp.ols_two_factor("js_distance"),
        Dp.ols_two_factor("overlap_coef"),
    ])
    headroom = Dp.target_headroom_table()
    absence = C.absence_effect()
    exclusion_cmp = Dp.absent_exclusion_comparison()
    partials_excl = Dp.partial_correlations_excluding_absent()
    _write(controlled, tables / "corr_scale_controlled.csv")
    _write(partials, tables / "partial_correlations.csv")
    _write(ols, tables / "ols_two_factor.csv")
    _write(headroom, tables / "target_headroom.csv")
    _write(absence, tables / "absence_effect.csv")
    _write(exclusion_cmp, tables / "corr_absent_excluded_comparison.csv")
    _write(partials_excl, tables / "partial_correlations_absent_excluded.csv")

    # ── 4. Figures ──────────────────────────────────────────────────────────────
    figs = plots.generate_all(out)

    # ── 5. Report ───────────────────────────────────────────────────────────────
    report = _build_report(desc, naive, overall_corr, controlled, partials,
                           ols, headroom, absence, exclusion_cmp, partials_excl,
                           figs)
    (out / "report.md").write_text(report, encoding="utf-8")

    print(f"Analysis complete. Outputs in: {out}")
    for f in figs:
        print(f"  figure: {f.name}")
    print(f"  report: report.md")


def _build_report(desc, naive, overall_corr, controlled, partials, ols,
                  headroom, absence, exclusion_cmp, partials_excl, figs) -> str:
    ols_js = ols[ols["predictor"] == "js_distance"].iloc[0]
    parts = []
    A = parts.append

    A("# Transferability analysis: what drives source->target gain/loss\n")
    A("MGPCGRL (`train_mgpcgrl`) is trained on a **target** game while a "
      "**source** game's data is mixed in. `diff_vs_baseline` is the target "
      "performance change vs. the no-mixing (`source=none`) baseline. We ask "
      "which properties of each game's reward-condition distribution explain "
      "the observed gains and losses.\n")

    A("## 1. Per-game condition distributions\n")
    A("Distributions differ enormously in scale and shape across games and "
      "enums (see `fig_condition_distributions.png`). Structural absences: "
      "dungeon has no *Interactable*, sokoban has no *Hazard* / *Collectable*.\n")
    A(_fmt(desc[["game", "reward_label", "n", "mean", "std", "cv",
                 "entropy", "frac_zero", "present"]]) + "\n")

    A("## 2. Naive pooled correlation is inconclusive\n")
    A("Correlating raw distribution features against `diff_vs_baseline` pooled "
      "across all enums gives weak, sign-inconsistent results (Pearson vs "
      "Spearman disagree): distribution similarity alone does **not** cleanly "
      "explain transfer.\n")
    A(_fmt(naive) + "\n")

    A("## 3. Why: per-enum scale confound\n")
    A("Each enum lives on a different scale (Region ~0-30, Hazard ~0-250), so "
      "raw-scale features let a few high-magnitude enums dominate the pool. "
      "After z-scoring predictor and response **within each enum**, a "
      "consistent picture emerges (Pearson and Spearman now agree in sign):\n")
    A(_fmt(controlled) + "\n")

    A("## 4. Two robust drivers\n")
    A("### 4a. Head-room / ceiling effect (dominant)\n")
    A("The target's own baseline is the strongest driver: low-baseline targets "
      "reliably gain from mixing, high-baseline targets can only stagnate or "
      "lose (see `fig_headroom.png`).\n")
    A(_fmt(headroom) + "\n")
    A("### 4b. Distribution similarity (secondary, independent)\n")
    A("Sources whose condition distribution overlaps the target's help more; "
      "divergent sources help less or hurt. This survives controlling for "
      "head-room (partial correlation):\n")
    A(_fmt(partials) + "\n")
    A(f"A two-factor OLS (within-enum z-scores) `diff ~ baseline + JS-distance` "
      f"gives baseline_beta={ols_js['control_beta']:.3f}, "
      f"JS_beta={ols_js['predictor_beta']:.3f}, R2={ols_js['r2']:.3f} "
      f"(n={int(ols_js['n'])}). Both factors contribute independently.\n")
    A(_fmt(ols) + "\n")

    A("## 5. Structural feature absence hurts (categorical)\n")
    A("When the source structurally lacks the target's feature it can only "
      "dilute that feature's supervision; such rows go negative far more often "
      "(see `fig_absence_effect.png`).\n")
    A(_fmt(absence) + "\n")

    A("## 5b. Robustness: excluding structural-absence rows\n")
    A("Re-running the scale-controlled correlations after **dropping the 12 rows "
      "where the source game lacks the target's feature** (feature not in the "
      "dataset) shows how much of each effect those cases carried:\n")
    A(_fmt(exclusion_cmp) + "\n")
    A("Key changes: the univariate JS/overlap *similarity* signal weakens toward "
      "non-significance (much of it was carried by absent-source rows, which are "
      "simultaneously maximally divergent and harmful), while `coverage` and "
      "`std_ratio` (breadth mismatch) strengthen. **Head-room (`baseline_mean`) "
      "stays robust.** Crucially, after controlling for head-room the similarity "
      "effect is still significant even among feature-present sources, whereas "
      "`std_ratio` is not — i.e. its univariate strength was a head-room "
      "artifact:\n")
    A(_fmt(partials_excl) + "\n")

    A("## 6. Discussion takeaways\n")
    A("- **Primary factor — target head-room.** Gains are largest where the "
      "target baseline is low (dungeon, pokemon); near a performance ceiling "
      "(doom) mixing only adds noise and slightly hurts.\n")
    A("- **Secondary factor — source/target distribution similarity.** Once "
      "scale is controlled, greater condition-distribution overlap (low JS / "
      "KS) predicts more positive transfer, independent of head-room.\n")
    A("- **Structural feature coverage matters.** A source lacking a feature "
      "(sokoban hazard/collectable, dungeon interactable) tends to hurt that "
      "feature's target performance.\n")
    A("- **Anomaly — zelda.** zelda loses under almost every source despite a "
      "mid baseline; its very high-variance, heavy-tailed distributions "
      "(large `std`/`cv` above) make it a candidate for a case study.\n")
    A("\nFigures: " + ", ".join(f.name for f in figs) + "\n")

    return "\n".join(parts)


if __name__ == "__main__":
    main()

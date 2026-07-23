# Transferability analysis

Analyzes how mixing **source**-game data into MGPCGRL (`train_mgpcgrl`) training
changes **target**-game performance, and relates gain/loss to each game's
reward-condition distribution.

## Run

```bash
# from repo root
python -m results.transferbility.analysis.run
```

Outputs land in `results/transferbility/output/`:

- `report.md` — narrative summary of findings.
- `fig_*.png` — figures (distributions, head-room, similarity boxplot, absence effect, per-experiment bars).
- `tables/*.csv` — every intermediate table.

## Inputs

- `src/source_target_table_5seed.csv` — experiment result table
  (`diff_vs_baseline` = target delta vs. `source=none`).
- `dataset/multigame/cache/artifacts/<game>/*.ann.json` — per-game reward-condition
  distributions (loaded via `analysis/dataset_distribution/run.py::load_annotations`).

## Module layout

| module          | responsibility |
|-----------------|----------------|
| `config.py`     | paths, game list, reward-enum mapping, structural feature-absence table |
| `data.py`       | load result table + per-(game, enum) condition arrays |
| `distances.py`  | per-distribution descriptors + directional source→target similarity features |
| `correlate.py`  | merge features with deltas; naive / per-enum / overall correlations; absence effect |
| `deeper.py`     | scale-controlled (within-enum z-scored) correlation, partial correlation, OLS, head-room |
| `plots.py`      | figures |
| `run.py`        | orchestrator + report writer |

## Headline findings

1. **Naive pooled correlation is inconclusive** — per-enum scale differences
   (Region ~0-30 vs Hazard ~0-250) confound raw-feature pooling.
2. **Head-room / ceiling effect (primary)** — low-baseline targets (dungeon,
   pokemon) reliably gain; near-ceiling targets (doom) only stagnate or lose.
3. **Distribution similarity (secondary, independent)** — after within-enum
   z-scoring, higher source↔target overlap (low JS/KS) predicts better transfer;
   survives controlling for head-room (partial r ≈ −0.31, p ≈ 0.004).
4. **Structural feature absence hurts** — a source lacking the target's feature
   (sokoban hazard/collectable, dungeon interactable) dilutes supervision and
   goes negative more often.
5. **zelda anomaly** — loses under almost every source despite a mid baseline;
   heavy-tailed, high-variance distributions make it a case-study candidate.

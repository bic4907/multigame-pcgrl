# results/ Utility Guide

> Configuration values (project names, paths, etc.) → **[CONFIG.md](CONFIG.md)**

---

## Workflow Overview

```
[1] eval_downloader.py         Download raw eval results from W&B  →  wandb_projects/
        ↓
[2] make_gamewise_summary.py   Rebuild per-game results.csv / summary.csv
        ↓
[3] build_benchmark_table.py   Markdown/CSV tables + comparison plots
        ↓
[4] condition_progress_report.py   condition vs metric plot report
    reward_enum_visualizer.py      reward_enum tile-map visualization
        ↓
[5] utils/embed_markdown_images_base64.py  Embed MD images as base64 (for sharing)
    utils/render_markdown_pdf.py           MD → PDF conversion
```

---

## 1. Data Collection

### eval_downloader.py — Download W&B artifacts

```bash
# Default run (TARGET_PROJECTS → wandb_projects/)
python results/eval_downloader.py

# Specific projects, custom output path
python results/eval_downloader.py \
  --projects aaai27_eval_cpcgrl aaai27_eval_cpcgrl_all \
  --output wandb_projects

# Include eval.h5 (skipped by default)
python results/eval_downloader.py --h5

# Finished runs only, 4 parallel threads
python results/eval_downloader.py --finished-only --workers 4

# Force overwrite existing files
python results/eval_downloader.py --force
```

---

## 2. Data Preprocessing

### make_gamewise_summary.py — Rebuild per-game summary/results

Reads `ctrl_sim.csv`, filters rows matching `game-<code>` in the folder name,
and regenerates `results.csv` / `summary.csv`.

```bash
# Target: wandb_projects/aaai27_eval_cpcgrl  (from config.json)
python results/make_gamewise_summary.py
```

> Game code mapping: `dg=dungeon`, `pk=pokemon`, `sk=sokoban`, `dm=doom`, `zd=zelda`

---

## 3. Table Generation

### build_benchmark_table.py — Markdown/CSV tables + comparison plots

```bash
# Default run (input: wandb_projects/)
python results/build_benchmark_table.py

# Specify input path
python results/build_benchmark_table.py --input wandb_projects

# Detailed breakdown: folder + game + reward_enum
python results/build_benchmark_table.py --group-by folder_game_reward_enum

# Select metrics + adjust decimal places
python results/build_benchmark_table.py --metrics progress vit_score --decimals 3

# Tables only, no plots
python results/build_benchmark_table.py --no-plot

# Specify all output paths
python results/build_benchmark_table.py \
  --output-md       results/outputs/table.md \
  --output-csv      results/outputs/table.csv \
  --output-folder-md  results/outputs/folder_mean.md \
  --output-folder-csv results/outputs/folder_mean.csv \
  --plot-file        results/outputs/plot.png \
  --plot-file-simple results/outputs/plot_simple.png
```

**`--group-by` options**

| Value | Grouping key |
|-------|-------------|
| `folder` *(default)* | Top-level folder name |
| `project_game` | Folder + game |
| `folder_game_reward_enum` | Folder + game + reward_enum |
| `game` | Game name |
| `reward_enum` | reward_enum value |

**Output files** (→ `results/outputs/<run_dir>/`)

| File | Description |
|------|-------------|
| `benchmark_table.md` / `.csv` | Aggregated table by `--group-by` |
| `benchmark_folder_mean.md` / `.csv` | Per-folder overall mean |
| `benchmark_game_reward_enum.png` | Game × reward_enum subplots |
| `benchmark_overall_simple.png` | Overall simple bar chart |

---

## 4. Visualization / Reports

### condition_progress_report.py — condition vs metric plots

```bash
# Default run
python results/condition_progress_report.py

# Specify input/output paths
python results/condition_progress_report.py \
  --input-root wandb_projects \
  --output-dir results/outputs/condition_plots \
  --output-md  results/outputs/condition_report.md

# Also generate PDF
python results/condition_progress_report.py \
  --output-pdf results/outputs/condition_report.pdf

# Adjust scatter sample count / seed
python results/condition_progress_report.py \
  --max-scatter-points 3000 \
  --seed 0
```

### reward_enum_visualizer.py — reward_enum representative tile-map visualization

> Requires `eval.h5` files (download with `eval_downloader.py --h5`).

```bash
# Default run
python results/reward_enum_visualizer.py

# Specify input root / output paths
python results/reward_enum_visualizer.py \
  --root wandb_projects/aaai27_eval_cpcgrl \
  --output-dir results/outputs/reward_viz \
  --output-md  results/outputs/reward_viz_report.md

# Specific reward_enums only
python results/reward_enum_visualizer.py --reward-enums 0 1 2

# Adjust tile render size
python results/reward_enum_visualizer.py --render-tile-size 32
```

---

## 5. Document Conversion (`utils/`)

> These scripts are located in `results/utils/`.

### utils/embed_markdown_images_base64.py — Embed images as base64

Embeds local images into Markdown as base64 data URIs for external sharing.

```bash
python results/utils/embed_markdown_images_base64.py input.md output_embedded.md
```

### utils/render_markdown_pdf.py — Markdown → PDF

```bash
# Basic conversion
python results/utils/render_markdown_pdf.py input.md output.pdf

# Compress image pairs onto a single page
python results/utils/render_markdown_pdf.py input.md output.pdf --single-page

# Adjust DPI (default: 300)
python results/utils/render_markdown_pdf.py input.md output.pdf --dpi 150
```

---

## Input File Formats

**`summary.csv`** — `<input>/<project>/<run>/[<eval>/]summary.csv`
```
metric,mean
progress,0.832
vit_score,0.741
tpkldiv,1.23
diversity,0.95
```

**`results.csv`** — used for plots / per-folder mean
```
game,reward_enum,progress,vit_score,tpkldiv,diversity
maze,0.5,0.84,0.73,1.21,0.96
```

run/eval folder names are auto-parsed as `game-maze_re-0.5_seed-42` tokens.

---

## Dependencies

```bash
pip install matplotlib seaborn pandas h5py tqdm
```

# MGPCGRL: Multi-Game Procedural Content Generation via Representation Learning

[![dataset validation](https://github.com/bic4907/multigame-pcgrl/actions/workflows/multigame-cache-tests.yml/badge.svg)](https://github.com/bic4907/multigame-pcgrl/actions/workflows/multigame-cache-tests.yml)


This repository contains the code for **MGPCGRL (Multi-Game PCGRL)**, a
multi-domain reinforcement learning framework for instruction-conditioned
procedural content generation.

MGPCGRL targets a practical gap in PCGRL: rewards and instruction meanings are
usually hand-defined for one game at a time. The framework instead learns shared
representations between design instructions and game levels, then transfers
reward signals across game domains.

---

## What Is Included

- Multi-game dataset loader: `dataset/multigame`
- External datasets:
  - `dataset/TheVGLC` (VGLC levels)
  - `dataset/dungeon_level_dataset` (instruction-level pairs)

---
[train_cpcgrl.py](train_cpcgrl.py)
## Installation

```bash
conda create -n mgpcgrl python=3.11
conda activate mgpcgrl
pip install -r requirements.txt
```[train_cpcgrl.py](train_cpcgrl.py)

---

## Dataset Setup

### Initialize Git Submodules

**Option 1: Clone with all submodules**

```bash
# Clone each submodule
git clone --recursive https://github.com/TheVGLC/TheVGLC dataset/TheVGLC
git clone --recursive https://github.com/bic4907/dungeon-level-dataset dataset/dungeon_level_dataset
git clone --recursive https://github.com/google-deepmind/boxoban-levels dataset/boxoban_levels
git clone --recursive https://github.com/TimMerino1710/five-dollar-model dataset/five-dollar-model
```

**Option 2: Initialize in existing repository**

```bash
git submodule update --init --recursive
```

**Option 3: Update submodules (if already cloned)**

```bash
git -C dataset/TheVGLC pull --ff-only
git -C dataset/dungeon_level_dataset pull --ff-only
git -C dataset/boxoban_levels pull --ff-only
git -C dataset/five-dollar-model pull --ff-only

git submodule update --init --recursive
```

**Verify:**

```bash
git submodule status
```

**Expected submodules** (from `.gitmodules`):
- `dataset/TheVGLC` - VGLC games (Doom, Zelda etc.)
- `dataset/dungeon_level_dataset` - Dungeon with text
- `dataset/boxoban_levels` - Boxoban/Sokoban levels
- `dataset/five-dollar-model` - Pokemon levels

---

## Multi-Game Dataset Quick Start

### 1) Load all available games

```python
from dataset.multigame import MultiGameDataset

ds = MultiGameDataset(include_dungeon=True)
print(len(ds))
print(ds.available_games())

sample = ds[0]
print(sample.game, sample.array.shape, sample.instruction)
```

### 2) Dungeon-only (level-text pairs)

```python
from pathlib import Path
from dataset.multigame import MultiGameDataset

ds = MultiGameDataset(
    vglc_games=[],
    vglc_root=Path("__disable_vglc__"),
    include_dungeon=True,
)

pairs = [(s.game, s.array, s.instruction) for s in ds.with_instruction()]
print(len(pairs))
```

### 3) Filter by game

```python
from dataset.multigame import MultiGameDataset, GameTag

ds = MultiGameDataset(include_dungeon=True)
zelda_samples = ds.by_game(GameTag.ZELDA)
dungeon_samples = ds.by_game(GameTag.DUNGEON)
print(len(zelda_samples), len(dungeon_samples))
```

For more dataset details:
- `dataset/multigame/README.md`

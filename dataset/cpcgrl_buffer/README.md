# CPCGRL Pair Dataset

## Overview

A preprocessed dataset of **consecutive 2-step env_map pairs** `(before, after)`, extracted
from the trajectories collected while training a CPCGRL (Conditional PCGRL) agent.

For five reward_enum values (1=region, 2=path_length, 3=block, 4=bat_amount, 5=bat_direction),
Pairs are extracted per reward_enum from the recorded rollouts, **deduplicated across the
whole set**, and stored in a single `.npz` file.

## file

```
dataset/cpcgrl_buffer/
├── __init__.py                 # CPCGRLBufferDataset, MapTransitionPair export
├── build_pair_dataset.py       # preprocessing script
├── cpcgrl_pair_dataset.npz     # the dataset itself (single file)
├── dataset.py                  # dataset class
├── metadata.json               # build metadata
└── README.md
```

## data Shape

Contents of `cpcgrl_pair_dataset.npz`:
| key | Shape | dtype | description |
| Field | Shape | dtype | Description |
| `env_map_pairs` | `(12655, 2, 16, 16)` | int32 | (before, after) env_map pairs |
| `reward_enums` | `(12655,)` | int32 | reward_enum label of each pair (1-5) |
| `timesteps` | `(12655,)` | int64 | start timestep of each pair (in total_timesteps) |
| `timesteps` | `(12655,)` | int64 | start timestep of each pair (within total_timesteps) |

- `env_map_pairs[:, 0]` → **before** (map at step t)
- `env_map_pairs[:, 1]` → **after** (map at step t+1)
- map size: 16x16, tile ids: dungeon3 integers (1-7)

## reward_enum distribution

| reward_enum | feature | count |
|:-----------:|---------|------:|
| 1 | region | 736 |
| 3 | block | 5,469 |
| 4 | bat_amount | 3,347 |
| 5 | bat_direction | 3,103 |
| **total** | | **12,655** |

> 39,684 pairs before deduplication → **12,655** after.
> Duplicates arise because the same map state recurs across rollouts of different reward_enums.

## metadata.json

Metadata written automatically at build time:

| key | description | example |
|---|---|---|
| `created_at` | build timestamp | `"2026-03-29 16:45:47"` |
| `platform` | OS / architecture | `"macOS-15.7.4-arm64-arm-64bit"` |
| `total_pairs` | pair count after deduplication | `12655` |
| `total_before_dedup` | pair count before deduplication | `39684` |
| `tile_min` / `tile_max` | env_map tile id range | `1` / `7` |
| `env_map_shape` | data shape | `[12655, 2, 16, 16]` |
| `reward_enum_distribution` | pair count per reward_enum | `{"1": 736, "3": 5469, ...}` |
| `seed` | random seed | `42` |

## How it is built

```bash
# Run against a saves/ directory containing finished training runs
python dataset/cpcgrl_buffer/build_pair_dataset.py \
    --saves_dir saves \
    --pairs_per_re 4000 \
    --seed 42
```

### Preprocessing pipeline

1. Discover reward_enum directories under `saves/` via the `_re-{N}_` pattern
2. Load each run's env_map stream and form **consecutive 2-step pairs** `(env_map[t], env_map[t+1])`
   - pairs spanning a `done=True` boundary are dropped
   - pairs with a non-contiguous timestep are dropped
3. Randomly subsample to at most 4,000 pairs per reward_enum
4. Merge everything, then **deduplicate globally** (identical 2-map pairs are removed)
5. Shuffle and write a single `.npz`

### Source trajectories

Trajectories recorded by `BufferCollector` over the 50%-100% window of training:

```
saves/model-contconv_exp-def_game-dungeon_re-{1..5}_vec_ro_s-0/buffer/
    buffer_000000_ts460800.npz
    buffer_000001_ts537600.npz
    ...
```

## Usage

### Loading with defaults

```python
from dataset.cpcgrl_buffer import CPCGRLBufferDataset

ds = CPCGRLBufferDataset()
print(ds)
# CPCGRLBufferDataset(n=12655, reward_enums=[1, 3, 4, 5])

print(len(ds))       # 12655
print(ds.map_shape)  # (16, 16)
```

### MapTransitionPair — before/after accessors

```python
pair = ds[0]
print(pair)
# MapTransitionPair(re=3, ts=537604, map=16x16, changes=1)

pair.before       # (16, 16) int32 — map at step t
pair.after        # (16, 16) int32 — map at step t+1
pair.pair         # (2, 16, 16)    — both, stacked
pair.reward_enum  # 3
pair.timestep     # 537604

# Difference helpers
pair.diff          # (16, 16) int16 — after - before
pair.changed_mask  # (16, 16) bool  — cells that changed
pair.n_changes     # 1              — number of changed tiles
```

### reward_enum filtering

```python
# Keep only the region (re=1) pairs
region_ds = ds.by_reward_enum(1)
print(region_ds)
# CPCGRLBufferDataset(n=736, reward_enums=[1])
ds.after_maps        # (N, 16, 16)    — all `after` maps
# Filter by reward_enum
sub_ds = ds.by_reward_enum(1, 3)
print(sub_ds)
# CPCGRLBufferDataset(n=6205, reward_enums=[1, 3])
```

### Batch access (NumPy arrays)

```python
ds.pairs             # (N, 2, 16, 16) — all
ds.before_maps       # (N, 16, 16)    — all `before` maps
ds.after_maps        # (N, 16, 16) -- all after maps
ds.reward_enums_array  # (N,) int32
ds.timesteps_array     # (N,) int64
```

### Random sampling

```python
pair = ds.sample(seed=42)          # one pair
pairs = ds.sample(n=100, seed=42)  # 100 pairs
```

### Slicing

```python
first_10 = ds[:10]      # CPCGRLBufferDataset(n=10, ...)
```

### Statistics / metadata

```python
ds.summary()
# {'total_pairs': 12655, 'map_shape': (16, 16),
#  'tile_min': 1, 'tile_max': 7,
#  'reward_enum_distribution': {1: 736, 3: 5469, 4: 3347, 5: 3103}}

ds.metadata
# {'created_at': '2026-03-29 16:40:29', 'platform': '...', ...}
```

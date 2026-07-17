# CPCGRL Pair Dataset

## text

CPCGRL (Conditional PCGRL)  in previoustext of  training  during  text trajectory text to text
**text 2-step env_map text** `(before, after)`  extracttext preprocessingtext datasettext.

5text reward_enum (1=region, 2=path_length, 3=block, 4=bat_amount, 5=bat_direction) by
trainingtext  in previoustext of  text in  text extracttext text, **text textabove duplicate  text before text remove**text
text `.npz` file to  savetext.

## file

```
dataset/cpcgrl_buffer/
├── __init__.py                 # CPCGRLBufferDataset, MapTransitionPair export
├── build_pair_dataset.py       # preprocessing script
├── cpcgrl_pair_dataset.npz     # text dataset (text file)
├── dataset.py                  # dataset class
├── metadata.json               # build metadata
└── README.md
```

## data Shape

`cpcgrl_pair_dataset.npz` internal text:

| text | Shape | dtype | text |
|---|---|---|---|
| `env_map_pairs` | `(12655, 2, 16, 16)` | int32 | (before, after) env_map text |
| `reward_enums` | `(12655,)` | int32 | each text of  reward_enum label (1~5) |
| `timesteps` | `(12655,)` | int64 | each text of  start timestep (total_timesteps basis) |

- `env_map_pairs[:, 0]` → **before** (t text of  map)
- `env_map_pairs[:, 1]` → **after** (t+1 text of  map)
- map size: 16×16, tile text: dungeon3 basis integer (1~7)

## reward_enum text distribution

| reward_enum | feature | text text |
|:-----------:|---------|------:|
| 1 | region | 736 |
| 3 | block | 5,469 |
| 4 | bat_amount | 3,347 |
| 5 | bat_direction | 3,103 |
| **text** | | **12,655** |

> duplicate remove  before  39,684 text → remove  after  **12,655** text.
> reward_enum text duplicatetext  different  text  text  in previoustext  text map  repetition createtext text.

## metadata.json

build text automatic createtext  metadata file:

| text | text | text |
|---|---|---|
| `created_at` | create texteach | `"2026-03-29 16:45:47"` |
| `hostname` | create PC name | `"MacBookPro.local"` |
| `platform` | OS/text | `"macOS-15.7.4-arm64-arm-64bit"` |
| `total_pairs` | duplicate remove  after  text text text | `12655` |
| `total_before_dedup` | duplicate remove  before  text text | `39684` |
| `tile_min` / `tile_max` | env_map tile text range | `1` / `7` |
| `env_map_shape` | data shape | `[12655, 2, 16, 16]` |
| `reward_enum_distribution` | retext text text | `{"1": 736, "3": 5469, ...}` |
| `seed` | random seed | `42` |

## create text

```bash
# saves/  in  training text  with text in  Usage
python dataset/cpcgrl_buffer/build_pair_dataset.py \
    --saves_dir saves \
    --pairs_per_re 4000 \
    --seed 42
```

### preprocessing pipeline

1. `saves/`  in  `_re-{N}_` text as  reward_enum text text directory automatic text
2. each text of  env_map    before text loadtext **text 2-step text** `(env_map[t], env_map[t+1])` create
   - `done=True` text text ( in text text text)
   - timestep text text text
3. reward_enum text 4,000text duplicate text  random sampletext
4. all merge  after  **text textabove duplicate remove** (env_map 2text  text before text sametext text remove)
5. text  after  text `.npz`  to  save

### text text text

training 50%~100% bin in  `BufferCollector`  text trajectory:

```
saves/model-contconv_exp-def_game-dungeon_re-{1..5}_vec_ro_s-0/buffer/
    buffer_000000_ts460800.npz
    buffer_000001_ts537600.npz
    ...
```

## Usage

### default text for

```python
from dataset.cpcgrl_buffer import CPCGRLBufferDataset

ds = CPCGRLBufferDataset()
print(ds)
# CPCGRLBufferDataset(n=12655, reward_enums=[1, 3, 4, 5])

print(len(ds))       # 12655
print(ds.map_shape)  # (16, 16)
```

### MapTransitionPair — before/after text text

```python
pair = ds[0]
print(pair)
# MapTransitionPair(re=3, ts=537604, map=16x16, changes=1)

pair.before       # (16, 16) int32 — t text of  map
pair.after        # (16, 16) int32 — t+1 text of  map
pair.pair         # (2, 16, 16)    — text form
pair.reward_enum  # 3
pair.timestep     # 537604

# text text
pair.diff          # (16, 16) int16 — after - before
pair.changed_mask  # (16, 16) bool  — text abovetext
pair.n_changes     # 1              — text tile text
```

### reward_enum filtering

```python
# region(re-1) text  text
region_ds = ds.by_reward_enum(1)
print(region_ds)
# CPCGRLBufferDataset(n=736, reward_enums=[1])

# text reward_enum text filter
sub_ds = ds.by_reward_enum(1, 3)
print(sub_ds)
# CPCGRLBufferDataset(n=6205, reward_enums=[1, 3])
```

### batch text (numpy array)

```python
ds.pairs             # (N, 2, 16, 16) — all
ds.before_maps       # (N, 16, 16)    — text before
ds.after_maps        # (N, 16, 16)    — text after
ds.reward_enums_array  # (N,) int32
ds.timesteps_array     # (N,) int64
```

### random sampletext

```python
pair = ds.sample(seed=42)          # 1text
pairs = ds.sample(n=100, seed=42)  # 100text text
```

### text text

```python
first_10 = ds[:10]      # CPCGRLBufferDataset(n=10, ...)
```

### text / metadata

```python
ds.summary()
# {'total_pairs': 12655, 'map_shape': (16, 16),
#  'tile_min': 1, 'tile_max': 7,
#  'reward_enum_distribution': {1: 736, 3: 5469, 4: 3347, 5: 3103}}

ds.metadata
# {'created_at': '2026-03-29 16:40:29', 'hostname': 'MacBookPro.local', ...}
```


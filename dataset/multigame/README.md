# MultiGame Dataset (Minimal)

This module helps you extract **(game, level, text)** tuples from `TheVGLC` and `dungeon_level_dataset`.

Most users only need:
- `game`: which game this sample belongs to
- `level`: level array (`numpy.ndarray`)
- `text`: instruction (`None` if unavailable)

## Supported Games

| Game | Tag | Text Annotation | Reward Annotation |
|------|-----|:-:|:-:|
| Dungeon | `dungeon` | ✅ per-sample | ✅ per-sample (CSV) |
| Sokoban | `sokoban` | ❌ | ⚠️ placeholder |
| Zelda | `zelda` | ❌ | ⚠️ placeholder |
| Doom | `doom` | ❌ | ⚠️ placeholder |
| Pokemon | `pokemon` | ✅ per-sample | ⚠️ placeholder |

> ⚠️ placeholder: served from `dataset/reward_annotations/{game}_reward_annotations_placeholder.csv`.
> Once real per-sample annotations exist, replace it with `{game}_reward_annotations.csv`.

## Quick Start

```python
from dataset.multigame import MultiGameDataset

ds = MultiGameDataset(include_dungeon=True)

pairs = [(s.game, s.array, s.instruction) for s in ds]
print(len(pairs))
print(pairs[0][0])        # game
print(pairs[0][1].shape)  # level shape
print(pairs[0][2])        # text or None
```

## Get Available Games

```python
from dataset.multigame import MultiGameDataset

ds = MultiGameDataset(include_dungeon=True)
print(ds.available_games())
# ['dungeon', 'sokoban', 'zelda', 'doom', 'pokemon']
```

## Get Samples By Game

```python
from dataset.multigame import MultiGameDataset, GameTag

ds = MultiGameDataset(include_dungeon=True)

zelda_samples   = ds.by_game(GameTag.ZELDA)
dungeon_samples = ds.by_game(GameTag.DUNGEON)

print(len(zelda_samples), len(dungeon_samples))
```

## Get Text-Only Pairs

```python
from dataset.multigame import MultiGameDataset

ds = MultiGameDataset(include_dungeon=True)

text_pairs = [(s.game, s.array, s.instruction) for s in ds.with_instruction()]
print(len(text_pairs))
```

## Reward Annotations

Reward annotations are attached to each sample's `meta` dictionary.

### reward_enum (1-5, shared across games)

| reward_enum | feature_name | description |
|:-----------:|--------------|------|
| 1 | `region` | number of connected regions |
| 2 | `path_length` | length of the longest path |
| 3 | `block` | wall / water ratio |
| 4 | `bat_amount` | number of enemies |
| 5 | `bat_direction` | directional spread of the enemies |

### dungeon — per-sample annotation

```python
from dataset.multigame import MultiGameDataset

ds = MultiGameDataset(include_dungeon=True, include_sokoban=False,
                      include_zelda=False, include_doom=False, include_pokemon=False)

sample = ds.by_game("dungeon")[0]
print(sample.meta["reward_enum"])    # e.g. 2
print(sample.meta["feature_name"])   # e.g. "path_length"
print(sample.meta["sub_condition"])  # e.g. "narrow"
print(sample.meta["conditions"])     # e.g. {2: 40.0}

# Filter samples with reward annotations
annotated = ds.with_reward_annotation()
print(len(annotated))

# Samples with reward_enum=2 (path_length)
path_samples = ds.by_reward_enum(2)
```

### Other games — placeholder (accessing conditions logs a WARNING)

`sokoban`, `zelda`, `doom` and `pokemon` have no per-sample annotations yet, so reading
`conditions` emits a `logging.WARNING`.

```python
import logging
logging.basicConfig(level=logging.WARNING)

ds = MultiGameDataset(include_dungeon=False, include_sokoban=True,
                      include_zelda=False, include_doom=False, include_pokemon=False)

sample = ds.by_game("sokoban")[0]
print(sample.meta["reward_enum"])   # 1 (placeholder default)
print(sample.meta["feature_name"])  # "region"

# Reading conditions logs a WARNING
val = sample.meta["conditions"][1]  # WARNING: sokoban is a placeholder
```

### reward annotation CSV structure

```
dataset/reward_annotations/
├── dungeon_reward_annotations.csv                  ← per-sample annotations
├── sokoban_reward_annotations_placeholder.csv      ← placeholder annotations (game-level)
├── zelda_reward_annotations_placeholder.csv
├── doom_reward_annotations_placeholder.csv
└── pokemon_reward_annotations_placeholder.csv
```

CSV columns:
```
key, instruction, level_id, sample_id,
reward_enum, feature_name, sub_condition,
condition_1, condition_2, condition_3, condition_4, condition_5
```

Extra columns in the placeholder CSV:
```
game, is_placeholder   ← "true" fixed
```

### Replacing a placeholder with real annotations

1. Create `{game}_reward_annotations.csv` in the same format as dungeon
2. `dataset/reward_annotations/`  in  save
3. Delete or keep the placeholder (`*_placeholder.csv`); the real CSV takes precedence

---

## Local Cache (Same Code + Same Args)

`MultiGameDataset` supports local preprocessing cache.

```python
from dataset.multigame import MultiGameDataset

ds = MultiGameDataset(
    include_dungeon=True,
    use_cache=True,
    cache_dir="dataset/multigame/cache/artifacts",
)
```

- same code + same init args → cache hit
- code or args changed → cache miss and rebuild

## Repository-Root Example Script

A runnable sample is added at:
- `example_multigame_dataset.py`

Run from repository root:

```bash
python example_multigame_dataset.py
python example_multigame_dataset.py --game zelda --limit 3
python example_multigame_dataset.py --with-text-only --limit 5
```

## Browser Viewer

Use the built-in viewer to check per-game counts and inspect levels with keyboard navigation.

```bash
python -m dataset.multigame.viewer.server --host 127.0.0.1 --port 8765
```

Open `http://127.0.0.1:8765` and use `Left` / `Right` arrows to move between samples.

**Three rendering modes:**
- **Raw** – Original game-specific tile colors
- **Unified** – 7-category abstraction (empty/wall/floor/enemy/object/spawn/hazard)
- **Symbol** – Tile name text overlay (e.g., "WAL", "ENE", "OBJ")

Switch modes with tabs in the browser UI. Legend updates automatically to show only tiles present in the current level.

## Before/After Mapping View

You can render raw tile image and 7-category mapped image side-by-side.

`render_before_after.py` currently supports only `dungeon` and `boxoban`.

```bash
python -m dataset.multigame.scripts.render_before_after --game dungeon --index 0 --out outputs/dungeon_before_after.png
python -m dataset.multigame.scripts.render_before_after --game boxoban --index 0 --out outputs/boxoban_before_after.png
```

## Notes

- `dungeon_level_dataset` has per-sample instruction text and reward annotation.
- `TheVGLC` (zelda, doom) has `instruction=None` and placeholder reward annotation.
- `pokemon` has instruction text but only placeholder reward annotation.
- For strict level-text pairs, use `ds.with_instruction()`.
- For reward-annotated samples only, use `ds.with_reward_annotation()`.

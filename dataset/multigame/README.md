# MultiGame Dataset (Minimal)

[![dataset validation](https://github.com/bic4907/multigame-pcgrl/actions/workflows/multigame-cache-tests.yml/badge.svg)](https://github.com/bic4907/multigame-pcgrl/actions/workflows/multigame-cache-tests.yml)

This module helps you extract **(game, level, text)** tuples from `TheVGLC` and `dungeon_level_dataset`.

Most users only need:
- `game`: which game this sample belongs to
- `level`: level array (`numpy.ndarray`)
- `text`: instruction (`None` if unavailable)

## Game Tags

- `zelda`
- `mario`
- `lode_runner`
- `kid_icarus`
- `doom`
- `mega_man`
- `dungeon`

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
# example: ['zelda', 'mario', 'lode_runner', 'dungeon']
```

## Get Samples By Game

```python
from dataset.multigame import MultiGameDataset, GameTag

ds = MultiGameDataset(include_dungeon=True)

zelda_samples = ds.by_game(GameTag.ZELDA)
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

- same code + same init args -> cache hit
- code or args changed -> cache miss and rebuild

## Repository-Root Example Script

A runnable sample is added at:
- `example_multigame_dataset.py`

Run from repository root:

```bash
python example_multigame_dataset.py
python example_multigame_dataset.py --game zelda --limit 3
python example_multigame_dataset.py --with-text-only --limit 5
```

## Notes

- `dungeon_level_dataset` usually has instruction text.
- `TheVGLC` usually has `instruction=None`.
- If you only want strict level-text pairs, use `ds.with_instruction()`.

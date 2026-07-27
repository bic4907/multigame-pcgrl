# Dataset Viewer

Lightweight browser viewer for inspecting the multi-game level dataset locally.

## Features

- Per-game sample counts (`dungeon`, `pokemon`, `sokoban`, `doom`, `zelda`, ...)
- Game selection and index-based browsing in the browser
- Keyboard navigation with the left/right arrow keys (`←` / `→`)
- Three rendering modes:
  - **Raw** — original game-specific tile colors. Available for Dungeon, Sokoban/Boxoban,
    DOOM, POKEMON, and Zelda; games without a palette fall back to magenta.
  - **Unified** — the 5-category unified palette
    (`empty` / `wall` / `interactive` / `hazard` / `collectable`). Recommended for
    cross-game comparison, since every game maps into the same color scheme.
  - **Symbol** — tile-name text overlaid on the unified colors.
- Live legend showing only the tiles present in the current level
- Tile mapping panel showing `raw tile -> unified category`, loaded from
  `dataset/multigame/tile_mapping.json`
- Album view showing several samples at once (6/8/12 per page); clicking a card opens
  the single-sample view at that index

## Run

All commands must be run from the project root, otherwise the `dataset.multigame`
package cannot be imported.

```bash
cd /path/to/multigame-pcgrl
python -m dataset.multigame.viewer.server --host 127.0.0.1 --port 8765
```

`python -m dataset.multigame.viewer` is equivalent and forwards to the same entry point.

Running from inside the viewer directory raises `ModuleNotFoundError`. If you must do so,
set `PYTHONPATH` to the project root:

```bash
PYTHONPATH=/path/to/multigame-pcgrl python -m dataset.multigame.viewer.server
```

### Custom dataset paths

Dataset roots are auto-detected. Override them when the submodules live elsewhere:

```bash
python -m dataset.multigame.viewer.server \
  --host 127.0.0.1 \
  --port 8765 \
  --dungeon-root /path/to/dungeon_level_dataset \
  --pokemon-root /path/to/five-dollar-model \
  --boxoban-root /path/to/boxoban_levels \
  --doom-root /path/to/doom_levels
```

## Usage

1. Select a game from the dropdown.
2. Switch rendering mode with the tabs (`Raw` / `Unified` / `Symbol`).
   `Symbol` shows abbreviated tile names such as `WAL`, `INT`, `HAZ`.
3. Navigate samples with the `Prev` / `Next` buttons, the arrow keys, or the
   `Index` input plus `Go`.
4. Switch the view selector to `Album`, choose a page size (6 / 8 / 12), and click a
   thumbnail to return to the single-sample view at that index.

## Notes

- The legend updates dynamically to show only the tiles used in the current level.
- Symbol mode is most readable at a tile size of 12px or more (scaled automatically).
- All rendering happens client-side after the initial JSON fetch, so switching modes is instant.
- Available datasets are detected automatically; missing ones are skipped without error.

## Troubleshooting

### `RuntimeWarning: [doom] ... has shape (133, 96); normalizing to (16, 16)`

DOOM maps are larger than the 16×16 working size. They are normalized with a top-left
slice and zero padding. The warning is informational — rendering still works.

### `RuntimeWarning: [tile_utils] No mapping found for game 'boxoban'`

`tile_mapping.json` registers the game under `sokoban`, while the handler requests
`boxoban`. This is handled automatically and Unified mode renders correctly.

## Mapping Source

The viewer loads its mapping from `dataset/multigame/tile_mapping.json`, exposed through
the `/api/mapping?game=<game_tag>` endpoint. The browser caches the mapping per game and
reuses it while navigating indices.

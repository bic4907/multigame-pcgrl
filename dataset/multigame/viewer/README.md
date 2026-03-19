# Dataset Viewer

Lightweight browser viewer for local dataset inspection.

## Features

- Shows sample counts per game (including `dungeon`, `pokemon`, `boxoban`, `doom`)
- Select game in browser and browse by index
- Keyboard navigation with left/right arrows (`←` / `→`)
- **Three rendering modes:**
  - **🎨 Raw** – Original game-specific tile colors (per-game palette)
  - **🗂 Unified** – 7-category unified palette (empty/wall/floor/enemy/object/spawn/hazard)
  - **🔤 Symbol** – Tile name text overlay on unified colors
- **Live legend** – Shows only tiles present in the current level
- **Tile mapping panel** - Shows `raw tile -> unified category` loaded from `dataset/multigame/tile_mapping.json`
- **Album view** - Shows multiple samples at once (6/8/12 per page), click a card to open single view

## Run

### Method 1: Direct Module Execution (권장)

```bash
python -m dataset.multigame.viewer.server --host 127.0.0.1 --port 8765
```

### Method 2: Run Server Script

```bash
cd dataset/multigame/viewer
python server.py --host 127.0.0.1 --port 8765
```

### Custom Dataset Paths

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

1. **Select game** from dropdown (e.g., `dungeon`, `pokemon`, `boxoban`)
2. **Switch rendering mode** by clicking tabs:
   - `Raw` – See original palette colors
   - `Unified` – See 7-category abstraction (useful for cross-game comparison)
   - `Symbol` – See tile names overlaid (e.g., "WAL", "FLO", "ENE")
3. **Navigate samples:**
   - `Prev` / `Next` buttons
   - Arrow keys: `←` / `→`
   - Jump to specific index with `Index` input + `Go`
4. **Album mode:**
   - Set `보기 = Album`
   - Choose `앨범크기` (6 / 8 / 12)
   - Click a thumbnail card to return to single detail view at that index

## Notes

- Legend panel updates dynamically to show only tiles used in the current level
- Symbol mode is most readable when tile size ≥ 12px (automatically scaled)
- All rendering happens client-side after initial JSON fetch (fast mode switching)
- Viewer automatically detects available datasets (dungeon, pokemon, boxoban)
- Missing datasets are simply skipped without error

## Mapping Source

Viewer mapping is loaded from:

- `dataset/multigame/tile_mapping.json`

API endpoint:

- `/api/mapping?game=<game_tag>`

The browser caches mapping per game and reuses it while navigating indices.

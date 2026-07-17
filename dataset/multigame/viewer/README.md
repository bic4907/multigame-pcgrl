# Dataset Viewer

Lightweight browser viewer for local dataset inspection.

## Features

- Shows sample counts per game (including `dungeon`, `pokemon`, `boxoban`, `doom`)
- Select game in browser and browse by index
- Keyboard navigation with left/right arrows (`←` / `→`)
- **Three rendering modes:**
  - **🎨 Raw** – Original game-specific tile colors (per-game palette)
    - ✅ Dungeon, Boxoban, DOOM text
    - ⚠️ POKEMON: palette undefined → default value text for  (text)
  - **🗂 Unified** – 7-category unified palette (empty/wall/floor/enemy/object/spawn/hazard)
    - ✅ text game textwall text (recommended ✅)
    - game text text in  text
  - **🔤 Symbol** – Tile name text overlay on unified colors
    - ✅ text game text
    - tile name check in  text for
- **Live legend** – Shows only tiles present in the current level
- **Tile mapping panel** - Shows `raw tile -> unified category` loaded from `dataset/multigame/tile_mapping.json`
- **Album view** - Shows multiple samples at once (6/8/12 per page), click a card to open single view

## Run

### ⚠️  during text: text to text text in  Usagetext!

```bash
# text text to text text to  move
cd /home/cilab/Projects/Py/multigame-pcgrl
```

### Method 1: Direct Module Execution (recommended ✅)

```bash
# text to text text in :
python -m dataset.multigame.viewer.server --host 127.0.0.1 --port 8765
```

**❌ warning:  text  text text!**
```bash
cd dataset/multigame/viewer
python -m dataset.multigame.viewer.server  # ← ModuleNotFoundError!
```

### Method 2: Using __main__.py

```bash
# text to text text in :
python -m dataset.multigame.viewer  # __init__.py  server  starttext
```

### Custom Dataset Paths

```bash
# text to text text in :
python -m dataset.multigame.viewer.server \
  --host 127.0.0.1 \
  --port 8765 \
  --dungeon-root /path/to/dungeon_level_dataset \
  --pokemon-root /path/to/five-dollar-model \
  --boxoban-root /path/to/boxoban_levels \
  --doom-root /path/to/doom_levels
```

### PYTHONPATH config (text)

text current directory in  also  Usage available:

```bash
cd dataset/multigame/viewer
PYTHONPATH=/home/cilab/Projects/Py/multigame-pcgrl python -m dataset.multigame.viewer.server
```

## Usage

1. **Select game** from dropdown (e.g., `dungeon`, `pokemon`, `boxoban`, `doom`)
2. **Switch rendering mode** by clicking tabs:
   - `Raw` – See original palette colors
     - **Note:** POKEMON  palette  undefinedtext text tile  same color as  tabletext
     - **solution:** `Unified` mode text for  recommended
   - `Unified` – See 7-category abstraction (useful for cross-game comparison)
     - ✅ text game textwall text (recommended)
   - `Symbol` – See tile names overlaid (e.g., "WAL", "FLO", "ENE")
3. **Navigate samples:**
   - `Prev` / `Next` buttons
   - Arrow keys: `←` / `→`
   - Jump to specific index with `Index` input + `Go`
4. **Album mode:**
   - Set `text = Album`
   - Choose `textsize` (6 / 8 / 12)
   - Click a thumbnail card to return to single detail view at that index

### rendering mode select   text

| game | Raw | Unified | Symbol |
|------|-----|---------|--------|
| Dungeon | ✅ | ✅ | ✅ |
| Sokoban | ✅ | ✅ | ✅ |
| POKEMON | ⚠️ (recommended text) | ✅ **recommended** | ✅ |
| DOOM | ✅ | ✅ | ✅ |
| DOOM 2 | ✅ | ✅ | ✅ |

## Notes

- Legend panel updates dynamically to show only tiles used in the current level
- Symbol mode is most readable when tile size ≥ 12px (automatically scaled)
- All rendering happens client-side after initial JSON fetch (fast mode switching)
- Viewer automatically detects available datasets (dungeon, pokemon, boxoban, doom)
- Missing datasets are simply skipped without error

## issue text

### ❌ POKEMON  text as  tabletext

**cause**: POKEMON game of  palette  tile_mapping.json in  text of text text

**solution**:
1. **Unified mode text for  (recommended)** ✅
   - `Unified` text  text 7-category color as  tabletext
   - text game in  textwalltext text

2. **tile_mapping.json update** (text solution)
   - `dataset/multigame/tile_mapping.json` of  pokemon text in  `_tile_colors` text

### ⚠️ DOOM rendering warning (RuntimeWarning)

**cause**: DOOM map  16x16  text size   text (text: 133x96)
- DOOM maptext  text size  keeptext text
- text  automatic as  top-left 16x16 as  normalizetext

**text**: warning  tabletext rendering  text text
```
RuntimeWarning: [doom] ... has shape (133, 96); normalizing to (16, 16)
```

**solution**:
1. **warning text** (current recommended) - text issue none
2. **DOOM text text enable** (text text)
   - DoomHandler of  text text text  text for text map  text  text as  split

### ⚠️ Boxoban text warning

**cause**: game text  `sokoban` text `boxoban` text to  requesttext

**text**:
```
RuntimeWarning: [tile_utils] No mapping found for game 'boxoban'.
```

**solution**: automatic as  processtext - Unified mode in  text rendering

## Mapping Source

Viewer mapping is loaded from:

- `dataset/multigame/tile_mapping.json`

API endpoint:

- `/api/mapping?game=<game_tag>`

The browser caches mapping per game and reuses it while navigating indices.

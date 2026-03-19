# Dataset Viewer

Lightweight browser viewer for local dataset inspection.

## Features

- Shows sample counts per game (including `boxoban` if available)
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

```bash
python -m dataset.multigame.viewer.server --host 127.0.0.1 --port 8765
```

Then open in your browser:

- http://127.0.0.1:8765

On startup, the server prints per-game counts to terminal.

## Usage

1. **Select game** from dropdown (e.g., `zelda`, `boxoban`, `dungeon`)
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

## Mapping Source

Viewer mapping is loaded from:

- `dataset/multigame/tile_mapping.json`

API endpoint:

- `/api/mapping?game=<game_tag>`

The browser caches mapping per game and reuses it while navigating indices.

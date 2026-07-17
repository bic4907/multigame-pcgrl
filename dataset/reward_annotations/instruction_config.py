"""
dataset/reward_annotations/instruction_config.py
=================================================
generate_instructions.py  in  text for text  text config text.
  - CUSTOM_THRESHOLDS  : annotation_figure.ipynb  of  CUSTOM_THRESHOLDS basis
  - RAW_TILE_COLORS    : gametext text tile ID → RGB (rendering for )
  - RAW_TILE_NAMES     : tile_mapping.json  of  _tile_names  in  automatic load
  - RAW_TILE_DESCS     : gametext text tile text (tile_mapping.json tiletext basis)
  - FEATURE_TILE_DESCS : feature_name → (raw text, unified text)
  - GAME_DESCRIPTIONS  : game text text text
  - FEATURE_DESCRIPTIONS: feature_name text
  - UNIFIED_COLOR_DESCS: unified text color text string
  - FEATURE_ZONE_LABELS: feature_name → 4text zone text text
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── tile_mapping.json load ────────────────────────────────────────────────────────
_MAPPING_FILE = Path(__file__).parent.parent / "multigame" / "tile_mapping.json"
_TILE_MAPPING: dict = json.loads(_MAPPING_FILE.read_text(encoding="utf-8"))

_SUPPORTED_GAMES = ["doom", "zelda", "sokoban", "pokemon", "dungeon"]

# ── Custom Threshold (annotation_figure.ipynb CUSTOM_THRESHOLDS) ─────────────
# None = text (game, feature) text in  threshold none
# threshold 3text → 4 bin: text text / text text / text text / text text
CUSTOM_THRESHOLDS: Dict[str, Optional[List[float]]] = {
    "dungeon_region":             [0.5,  1.5,  3.0,  4.5,  9.5,  14.5, 19.5],
    "dungeon_path_length":        [17.5, 23.5, 28.0, 32.5, 38.5, 44.5, 50.5],
    "dungeon_interactable_count": None,
    "dungeon_hazard_count":       [3.5,  6.5,  8.5,  10.5, 14.5, 18.5, 22.5],
    "dungeon_collectable_count":  [5.5,  8.5,  10.0, 11.5, 13.5, 15.5, 19.5],

    "doom_region":                [0.5,  1.5,  2.0,  2.5,  3.0,  3.5,  4.5],
    "doom_path_length":           [21.5, 23.5, 25.5, 27.5, 29.0, 30.5, 32.5],
    "doom_interactable_count":    [-0.5, 0.5,  2.0,  3.5,  5.0,  6.5,  8.5],
    "doom_hazard_count":          [0.5,  1.5,  2.5,  3.5,  4.5,  5.5,  7.5],
    "doom_collectable_count":     [0.5,  1.5,  2.0,  2.5,  4.0,  5.5,  7.5],

    "zelda_region":               [0.5,  1.5,  2.0,  2.5,  3.5,  4.5,  6.5],
    "zelda_path_length":          [14.5, 16.5, 19.0, 21.5, 22.0, 22.5, 23.5],
    "zelda_interactable_count":   [2.5,  4.5,  6.5,  8.5,  17.5, 26.5, 35.5],
    "zelda_hazard_count":         [3.5,  5.5,  8.0,  10.5, 14.5, 18.5, 22.5],
    "zelda_collectable_count":    [0.5,  1.5,  2.5,  3.5,  9.0,  14.5, 19.5],

    "pokemon_region":             [0.5,  1.5,  2.0,  2.5,  3.5,  4.5,  6.5],
    "pokemon_path_length":        [16.5, 18.5, 21.0, 23.5, 26.5, 29.5, 32.5],
    "pokemon_interactable_count": [-0.5, 0.5,  10.5, 20.5, 40.5, 60.5, 80.5],
    "pokemon_hazard_count":       [5.5,  11.5, 28.0, 44.5, 60.5, 76.5, 92.5],
    "pokemon_collectable_count":  [-0.5, 0.5,  1.5,  2.5,  4.5,  6.5,  9.5],

    "sokoban_region":             [0.5,  1.5,  2.0,  2.5,  3.0,  3.5,  4.5],
    "sokoban_path_length":        [13.5, 17.5, 19.5, 21.5, 23.5, 25.5, 29.5],
    "sokoban_interactable_count": [1.5,  3.5,  5.0,  6.5,  8.0,  9.5,  12.5],
    "sokoban_hazard_count":       None,
    "sokoban_collectable_count":  None,
}

# ── Raw tile color (gametext text tile ID → RGB) ────────────────────────────────────
RAW_TILE_COLORS: Dict[str, Dict[int, Tuple[int, int, int]]] = {
    "doom": {
        0: (30, 30, 30),      # EMPTY  - text of  text (border)
        1: (110, 110, 110),   # WALL   - text text
        2: (215, 195, 165),   # FLOOR  - text text
        3: (230, 40, 40),     # ENEMY  - text
        4: (0, 210, 100),     # SPAWN  - secondstext
        5: (230, 230, 0),     # ITEM   - text
        6: (255, 120, 0),     # DANGER - text
        7: (50, 120, 230),    # DOOR   - text
    },
    "zelda": {
        0: (30, 30, 30),      # EMPTY  - text of  text
        1: (110, 110, 110),   # WALL   - text text
        2: (215, 195, 165),   # FLOOR  - text text
        3: (140, 90, 40),     # DOOR   - text
        4: (100, 130, 50),    # BLOCK  - text
        5: (0, 210, 100),     # START  - text  secondstext
        6: (230, 40, 40),     # MOB    - text
        7: (230, 230, 0),     # OBJECT - text
        8: (80, 160, 240),    # FLOOD  - text
    },
    "sokoban": {
        0: (220, 200, 170),   # EMPTY  - text
        1: (110, 110, 110),   # WALL   - text text
        4: (160, 80, 30),     # BOX    - text
        5: (0, 210, 100),     # PLAYER - text  secondstext
    },
    "pokemon": {
        0:  (30, 30, 30),     # EMPTY  - text of  text
        1:  (110, 110, 110),  # WALL   - text text
        2:  (175, 225, 145),  # FLOOR  - textsecondstext (path)
        3:  (230, 40, 40),    # ENEMY  - text (wild Pokemon)
        4:  (230, 230, 0),    # OBJECT - text (Pokeball)
        5:  (0, 210, 100),    # SPAWN  - text  secondstext (door)
        6:  (50, 150, 240),   # WATER  - text
        7:  (200, 160, 90),   # FENCE  - text
        8:  (0, 150, 0),      # TREE   - text  secondstext
        9:  (210, 100, 50),   # HOUSE  - text
        10: (120, 205, 90),   # GRASS  - textsecondstext
    },
    "dungeon": {
        0: (30, 30, 30),      # UNKNOWN  - text of  text (border)
        1: (215, 195, 165),   # FLOOR    - text text
        2: (110, 110, 110),   # WALL     - text text
        3: (230, 40, 40),     # ENEMY    - text (bat)
        4: (230, 230, 0),     # TREASURE - text
    },
}

# ── Raw tile name: tile_mapping.json  of  _tile_names  in  automatic load ─────────────────
# text: int(tile_id), text: name string (EMPTY, WALL, FLOOR, ENEMY text)
RAW_TILE_NAMES: Dict[str, Dict[int, str]] = {
    game: {
        int(tid): name
        for tid, name in _TILE_MAPPING[game].get("_tile_names", {}).items()
        if int(tid) != 99  # UNKNOWN(99) text — text map in  text text
    }
    for game in _SUPPORTED_GAMES
    if game in _TILE_MAPPING
}

# ── unified text → gametext raw tile name text ───────────────────────────────────
# tile_mapping.json of  mapping in  automatic text: {game: {unified_cat_id: [tile_names]}}
UNIFIED_TILE_GROUPS: Dict[str, Dict[int, List[str]]] = {}
for _game in _SUPPORTED_GAMES:
    if _game not in _TILE_MAPPING:
        continue
    _mapping = _TILE_MAPPING[_game].get("mapping", {})
    _names   = RAW_TILE_NAMES.get(_game, {})
    _groups: Dict[int, List[str]] = {}
    for _raw_str, _uni_id in _mapping.items():
        _raw_id = int(_raw_str)
        _name   = _names.get(_raw_id)
        if _name:
            _groups.setdefault(int(_uni_id), []).append(_name)
    UNIFIED_TILE_GROUPS[_game] = _groups

# ── Raw tile text (tile_mapping.json tiletext basis) ─────────────────────────────────
# RAW_TILE_NAMES  of  name and  text text text.
RAW_TILE_DESCS: Dict[str, Dict[int, str]] = {
    "doom": {
        0: "void",
        1: "wall",
        2: "floor",
        3: "enemy",
        4: "spawn",
        5: "item",
        6: "trap",
        7: "door",
        8: "stair",
    },
    "zelda": {
        0: "void",
        1: "wall",
        2: "floor",
        3: "door",
        4: "block",
        5: "spawn",
        6: "enemy",
        7: "item",
        8: "water",
    },
    "sokoban": {
        0: "floor",
        1: "wall",
        4: "box",
        5: "spawn",
    },
    "pokemon": {
        0:  "void",
        1:  "wall",
        2:  "floor",
        3:  "enemy",
        4:  "object",
        5:  "door",
        6:  "water",
        7:  "fence",
        8:  "tree",
        9:  "house",
        10: "grass",
    },
    "dungeon": {
        0: "border",
        1: "floor",
        2: "wall",
        3: "enemy",
        4: "chest",
    },
}

# ── Featuretext tile text: feature_name → (raw text, unified text) ──────────────────
# passable basis: Empty + Hazard + Collectable (Interactive text)
FEATURE_TILE_DESCS: Dict[str, Dict[str, Tuple[str, str]]] = {
    "doom": {
        "region":             ("passable tiles: FLOOR, STAIR, ENEMY, ITEM",
                               "passable categories: empty, hazard, collectable"),
        "path_length":        ("passable tiles: FLOOR, STAIR, ENEMY, ITEM",
                               "passable categories: empty, hazard, collectable"),
        "interactable_count": ("tiles counted: SPAWN, DANGER, DOOR",
                               "category counted: interactive"),
        "hazard_count":       ("tiles counted: ENEMY",
                               "category counted: hazard"),
        "collectable_count":  ("tiles counted: ITEM (id=5)",
                               "category counted: collectable"),
    },
    "zelda": {
        "region":             ("passable tiles: FLOOR, MOB, OBJECT",
                               "passable categories: empty, hazard, collectable"),
        "path_length":        ("passable tiles: FLOOR, MOB, OBJECT",
                               "passable categories: empty, hazard, collectable"),
        "interactable_count": ("tiles counted: DOOR, BLOCK, START",
                               "category counted: interactive"),
        "hazard_count":       ("tiles counted: MOB",
                               "category counted: hazard"),
        "collectable_count":  ("tiles counted: OBJECT",
                               "category counted: collectable"),
    },
    "sokoban": {
        "region":             ("passable tiles: EMPTY, PLAYER",
                               "passable categories: empty"),
        "path_length":        ("passable tiles: EMPTY, PLAYER",
                               "passable categories: empty"),
        "interactable_count": ("tiles counted: BOX",
                               "category counted: interactive"),
        "hazard_count":       ("tiles counted: (none — Sokoban has no hazard tiles)",
                               "category counted: hazard (N/A for Sokoban)"),
        "collectable_count":  ("tiles counted: (none — Sokoban has no collectable tiles)",
                               "category counted: collectable (N/A for Sokoban)"),
    },
    "pokemon": {
        "region":             ("passable tiles: FLOOR, GRASS, ENEMY, OBJECT",
                               "passable categories: empty, hazard, collectable"),
        "path_length":        ("passable tiles: FLOOR, GRASS, ENEMY, OBJECT",
                               "passable categories: empty, hazard, collectable"),
        "interactable_count": ("tiles counted: SPAWN, WATER",
                               "category counted: interactive"),
        "hazard_count":       ("tiles counted: ENEMY",
                               "category counted: hazard"),
        "collectable_count":  ("tiles counted: OBJECT",
                               "category counted: collectable"),
    },
    "dungeon": {
        "region":             ("passable tiles: FLOOR, ENEMY, TREASURE",
                               "passable categories: empty, hazard, collectable"),
        "path_length":        ("passable tiles: FLOOR, ENEMY, TREASURE",
                               "passable categories: empty, hazard, collectable"),
        "interactable_count": ("tiles counted: (none — Dungeon has no interactable tiles)",
                               "category counted: interactive (N/A for Dungeon)"),
        "hazard_count":       ("tiles counted: ENEMY",
                               "category counted: hazard"),
        "collectable_count":  ("tiles counted: TREASURE",
                               "category counted: collectable"),
    },
}

# ── Count feature(enum 2,3,4) text raw tile ID list ────────────────────────────────
# instruction_raw create text per-tile count compute in  text for .
# tile ID  tile_mapping.json / measure/*.py basis.
FEATURE_COUNT_TILE_IDS: Dict[str, Dict[str, List[int]]] = {
    "doom": {
        "interactable_count": [4, 6, 7],   # SPAWN, DANGER, DOOR
        "hazard_count":       [3],           # ENEMY
        "collectable_count":  [5],           # ITEM
    },
    "zelda": {
        "interactable_count": [3, 4, 5],    # DOOR, BLOCK, START
        "hazard_count":       [6],           # MOB
        "collectable_count":  [7],           # OBJECT
    },
    "sokoban": {
        "interactable_count": [4],           # BOX
        "hazard_count":       [],            # (none)
        "collectable_count":  [],            # (none)
    },
    "pokemon": {
        "interactable_count": [5, 6],        # SPAWN, WATER
        "hazard_count":       [3],           # ENEMY
        "collectable_count":  [4],           # OBJECT
    },
    "dungeon": {
        "interactable_count": [],            # (none)
        "hazard_count":       [3],           # ENEMY
        "collectable_count":  [4],           # TREASURE
    },
}

# ── game text ─────────────────────────────────────────────────────────────────────
GAME_DESCRIPTIONS: Dict[str, str] = {
    "doom":    "Doom (top-down view of a first-person shooter dungeon map)",
    "zelda":   "The Legend of Zelda (top-down dungeon adventure map)",
    "sokoban": "Sokoban (top-down box-pushing puzzle map)",
    "pokemon": "Pokémon (top-down RPG overworld map)",
    "dungeon": "Dungeon adventure (top-down dungeon crawl map)",
}

# ── Feature text ──────────────────────────────────────────────────────────────────
FEATURE_DESCRIPTIONS: Dict[str, str] = {
    "region":             "number of disconnected passable-area clusters — count of separate walkable zones (not their size or content)",
    "path_length":        "length of the longest traversable path through passable tiles",
    "interactable_count": "total count of interactive tiles (doors, objects, spawn points, etc.)",
    "hazard_count":       "total count of hazard/enemy tiles",
    "collectable_count":  "total count of collectable/item tiles",
}

# ── Unified text color text ────────────────────────────────────────────────────
UNIFIED_COLOR_DESCS: Dict[int, str] = {
    0: "RGB(160,140,120) — grayish-tan",
    1: "RGB(80,80,80) — dark gray",
    2: "RGB(0,200,0) — green",
    3: "RGB(220,50,50) — red",
    4: "RGB(200,200,20) — yellow",
}

# ── Zone text text (featuretext) ───────────────────────────────────────────────────────
FEATURE_ZONE_LABELS: Dict[str, List[str]] = {
    "region": [
        "very few regions",
        "few regions",
        "slightly few regions",
        "moderate regions",
        "slightly many regions",
        "several regions",
        "many regions",
        "very many regions",
    ],
    "path_length": [
        "very short path",
        "short path",
        "slightly short path",
        "moderate path",
        "slightly long path",
        "medium path",
        "long path",
        "very long path",
    ],
    "interactable_count": [
        "very few interactive",
        "few interactive",
        "slightly few interactive",
        "moderate interactive",
        "slightly many interactive",
        "several interactive",
        "many interactive",
        "very many interactive",
    ],
    "hazard_count": [
        "very few hazards",
        "few hazards",
        "slightly few hazards",
        "moderate hazards",
        "slightly many hazards",
        "several hazards",
        "many hazards",
        "very many hazards",
    ],
    "collectable_count": [
        "very few collectables",
        "few collectables",
        "slightly few collectables",
        "moderate collectables",
        "slightly many collectables",
        "several collectables",
        "many collectables",
        "very many collectables",
    ],
}

# ── text text: feature × intensity level(0~7) → text tabletext list ────────────────────
# level 0 =  text text/text, level 7 =  text text/text
# existing 4level×4text → 8level×2text (each level  2text of  sub level to  split)
VOCAB_SETS: Dict[str, List[List[str]]] = {
    "region": [
        # level 0 — very few regions (lower)
        ["few",          "sparse"],
        # level 1 — very few regions (upper)
        ["small",        "marginal"],
        # level 2 — somewhat few regions (lower)
        ["some",         "moderate"],
        # level 3 — somewhat few regions (upper)
        ["slight",       "certain"],
        # level 4 — somewhat many regions (lower)
        ["several",      "balanced"],
        # level 5 — somewhat many regions (upper)
        ["multiple",     "partitioned"],
        # level 6 — very many regions (lower)
        ["fragmented",   "numerous"],
        # level 7 — very many regions (upper)
        ["large",        "many"],
    ],

    "path_length": [
        # level 0 — very short (lower)
        ["tiny",         "nano"],
        # level 1 — very short (upper)
        ["minimal",      "micro"],
        # level 2 — somewhat short (lower)
        ["short",        "limited"],
        # level 3 — somewhat short (upper)
        ["restricted",   "condensed"],
        # level 4 — somewhat long (lower)
        ["moderate",     "reasonable"],
        # level 5 — somewhat long (upper)
        ["medium",       "balanced"],
        # level 6 — very long (lower)
        ["long",         "large"],
        # level 7 — very long (upper)
        ["lengthy",      "extensive"],
    ]
}


_COUNT_VOCAB: List[List[str]] = [
    # level 0 — very few (lower)
    ["rare",       "few"],
    # level 1 — very few (upper)
    ["sparse",     "marginal"],
    # level 2 — somewhat few (lower)
    ["some",       "limited"],
    # level 3 — somewhat few (upper)
    ["slight",     "little"],
    # level 4 — moderate (lower)
    ["moderate",   "reasonable"],
    # level 5 — moderate (upper)
    ["decent",     "suitable"],
    # level 6 — very many (lower)
    ["many",       "numerous"],
    # level 7 — very many (upper)
    ["plentiful",  "abundant"],
]


VOCAB_SETS["interactable_count"] = _COUNT_VOCAB
VOCAB_SETS["hazard_count"]       = _COUNT_VOCAB
VOCAB_SETS["collectable_count"]  = _COUNT_VOCAB
"""
dataset/reward_annotations/instruction_config.py
=================================================
generate_instructions.py 에서 사용하는 모든 설정 상수.
  - CUSTOM_THRESHOLDS  : annotation_figure.ipynb 의 CUSTOM_THRESHOLDS 기준
  - RAW_TILE_COLORS    : 게임별 원본 타일 ID → RGB (렌더링용)
  - RAW_TILE_NAMES     : tile_mapping.json 의 _tile_names 에서 자동 로드
  - RAW_TILE_DESCS     : 게임별 원본 타일 설명 (tile_mapping.json 타일명 기준)
  - FEATURE_TILE_DESCS : feature_name → (raw 설명, unified 설명)
  - GAME_DESCRIPTIONS  : 게임 한 줄 설명
  - FEATURE_DESCRIPTIONS: feature_name 설명
  - UNIFIED_COLOR_DESCS: unified 카테고리 색상 설명 문자열
  - FEATURE_ZONE_LABELS: feature_name → 4개 zone 레이블
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── tile_mapping.json 로드 ────────────────────────────────────────────────────────
_MAPPING_FILE = Path(__file__).parent.parent / "multigame" / "tile_mapping.json"
_TILE_MAPPING: dict = json.loads(_MAPPING_FILE.read_text(encoding="utf-8"))

_SUPPORTED_GAMES = ["doom", "zelda", "sokoban", "pokemon", "dungeon"]

# ── Custom Threshold (annotation_figure.ipynb CUSTOM_THRESHOLDS) ─────────────
# None = 해당 (game, feature) 조합에 threshold 없음
# threshold 3개 → 4 구간: 매우 적음 / 다소 적음 / 다소 많음 / 매우 많음
CUSTOM_THRESHOLDS: Dict[str, Optional[List[float]]] = {
    "dungeon_region":             [1.5, 4.5, 10.5],
    "dungeon_path_length":        [23.5, 32.5, 46.5],
    "dungeon_interactable_count": None,
    "dungeon_hazard_count":       [5.5, 10.5, 18.5],
    "dungeon_collectable_count":  [1.5, 4.5, 7.5],

    "doom_region":                [1.5, 2.5, 3.5],
    "doom_path_length":           [23.5, 27.5, 30.5],
    "doom_interactable_count":    [0.5, 3.5, 6.5],
    "doom_hazard_count":          [1.5, 3.5, 5.5],
    "doom_collectable_count":     [1.5, 2.5, 5.5],

    "zelda_region":               [1.5, 2.5, 4.5],
    "zelda_path_length":          [16.5, 21.5, 22.5],
    "zelda_interactable_count":   [4.5, 8.5, 26.5],
    "zelda_hazard_count":         [0.5, 4.5, 8.5],
    "zelda_collectable_count":    [0.5, 1.5, 2.5],

    "pokemon_region":             [1.5, 2.5, 4.5],
    "pokemon_path_length":        [18.5, 24.5, 30.5],
    "pokemon_interactable_count": [0.5, 20.5, 60.5],
    "pokemon_hazard_count":       [1.5, 4.5, 8.5],
    "pokemon_collectable_count":  [0.5, 2.5, 6.5],

    "sokoban_region":             [1.5, 2.5, 3.5],
    "sokoban_path_length":        [17.5, 21.5, 25.5],
    "sokoban_interactable_count": [3.5, 4.5, 5.5],
    "sokoban_hazard_count":       None,
    "sokoban_collectable_count":  None,
}

# ── Raw 타일 색상 (게임별 원본 타일 ID → RGB) ────────────────────────────────────
RAW_TILE_COLORS: Dict[str, Dict[int, Tuple[int, int, int]]] = {
    "doom": {
        0: (30, 30, 30),      # EMPTY  - 거의 검정 (border)
        1: (110, 110, 110),   # WALL   - 어두운 회색
        2: (215, 195, 165),   # FLOOR  - 베이지
        3: (230, 40, 40),     # ENEMY  - 빨강
        4: (0, 210, 100),     # SPAWN  - 초록
        5: (230, 230, 0),     # ITEM   - 노랑
        6: (255, 120, 0),     # DANGER - 오렌지
        7: (50, 120, 230),    # DOOR   - 파랑
    },
    "zelda": {
        0: (30, 30, 30),      # EMPTY  - 거의 검정
        1: (110, 110, 110),   # WALL   - 어두운 회색
        2: (215, 195, 165),   # FLOOR  - 베이지
        3: (140, 90, 40),     # DOOR   - 갈색
        4: (100, 130, 50),    # BLOCK  - 올리브
        5: (0, 210, 100),     # START  - 밝은 초록
        6: (230, 40, 40),     # MOB    - 빨강
        7: (230, 230, 0),     # OBJECT - 노랑
        8: (80, 160, 240),    # FLOOD  - 하늘색
    },
    "sokoban": {
        0: (220, 200, 170),   # EMPTY  - 크림
        1: (110, 110, 110),   # WALL   - 어두운 회색
        4: (160, 80, 30),     # BOX    - 갈색
        5: (0, 210, 100),     # PLAYER - 밝은 초록
    },
    "pokemon": {
        0:  (30, 30, 30),     # EMPTY  - 거의 검정
        1:  (110, 110, 110),  # WALL   - 어두운 회색
        2:  (175, 225, 145),  # FLOOR  - 연초록 (path)
        3:  (230, 40, 40),    # ENEMY  - 빨강 (wild Pokemon)
        4:  (230, 230, 0),    # OBJECT - 노랑 (Pokeball)
        5:  (0, 210, 100),    # SPAWN  - 밝은 초록 (door)
        6:  (50, 150, 240),   # WATER  - 파랑
        7:  (200, 160, 90),   # FENCE  - 황토색
        8:  (0, 150, 0),      # TREE   - 짙은 초록
        9:  (210, 100, 50),   # HOUSE  - 주황갈색
        10: (120, 205, 90),   # GRASS  - 연초록
    },
    "dungeon": {
        0: (30, 30, 30),      # UNKNOWN  - 거의 검정 (border)
        1: (215, 195, 165),   # FLOOR    - 베이지
        2: (110, 110, 110),   # WALL     - 어두운 회색
        3: (230, 40, 40),     # ENEMY    - 빨강 (bat)
        4: (230, 230, 0),     # TREASURE - 노랑
    },
}

# ── Raw 타일 이름: tile_mapping.json 의 _tile_names 에서 자동 로드 ─────────────────
# 키: int(tile_id), 값: 이름 문자열 (EMPTY, WALL, FLOOR, ENEMY 등)
RAW_TILE_NAMES: Dict[str, Dict[int, str]] = {
    game: {
        int(tid): name
        for tid, name in _TILE_MAPPING[game].get("_tile_names", {}).items()
        if int(tid) != 99  # UNKNOWN(99) 제외 — 실제 맵에 등장하지 않음
    }
    for game in _SUPPORTED_GAMES
    if game in _TILE_MAPPING
}

# ── unified 카테고리 → 게임별 raw 타일 이름 그룹 ───────────────────────────────────
# tile_mapping.json의 mapping에서 자동 파생: {game: {unified_cat_id: [tile_names]}}
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

# ── Raw 타일 설명 (tile_mapping.json 타일명 기준) ─────────────────────────────────
# RAW_TILE_NAMES 의 이름과 반드시 일치해야 한다.
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

# ── Feature별 타일 설명: feature_name → (raw 설명, unified 설명) ──────────────────
# passable 기준: Empty + Hazard + Collectable (Interactive 제외)
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

# ── Count feature(enum 2,3,4) 별 raw tile ID 목록 ────────────────────────────────
# instruction_raw 생성 시 per-tile count 계산에 사용.
# tile ID는 tile_mapping.json / measure/*.py 기준.
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
        "hazard_count":       [],            # (없음)
        "collectable_count":  [],            # (없음)
    },
    "pokemon": {
        "interactable_count": [5, 6],        # SPAWN, WATER
        "hazard_count":       [3],           # ENEMY
        "collectable_count":  [4],           # OBJECT
    },
    "dungeon": {
        "interactable_count": [],            # (없음)
        "hazard_count":       [3],           # ENEMY
        "collectable_count":  [4],           # TREASURE
    },
}

# ── 게임 설명 ─────────────────────────────────────────────────────────────────────
GAME_DESCRIPTIONS: Dict[str, str] = {
    "doom":    "Doom (top-down view of a first-person shooter dungeon map)",
    "zelda":   "The Legend of Zelda (top-down dungeon adventure map)",
    "sokoban": "Sokoban (top-down box-pushing puzzle map)",
    "pokemon": "Pokémon (top-down RPG overworld map)",
    "dungeon": "Dungeon adventure (top-down dungeon crawl map)",
}

# ── Feature 설명 ──────────────────────────────────────────────────────────────────
FEATURE_DESCRIPTIONS: Dict[str, str] = {
    "region":             "number of disconnected passable-area clusters — count of separate walkable zones (not their size or content)",
    "path_length":        "length of the longest traversable path through passable tiles",
    "interactable_count": "total count of interactive tiles (doors, objects, spawn points, etc.)",
    "hazard_count":       "total count of hazard/enemy tiles",
    "collectable_count":  "total count of collectable/item tiles",
}

# ── Unified 카테고리 색상 설명 ────────────────────────────────────────────────────
UNIFIED_COLOR_DESCS: Dict[int, str] = {
    0: "RGB(160,140,120) — grayish-tan",
    1: "RGB(80,80,80) — dark gray",
    2: "RGB(0,200,0) — green",
    3: "RGB(220,50,50) — red",
    4: "RGB(200,200,20) — yellow",
}

# ── Zone 레이블 (feature별) ───────────────────────────────────────────────────────
FEATURE_ZONE_LABELS: Dict[str, List[str]] = {
    "region":             ["very few regions",          "somewhat few regions",
                           "somewhat many regions",     "very many regions"],
    "path_length":        ["very short path",           "somewhat short path",
                           "somewhat long path",        "very long path"],
    "interactable_count": ["very few interactive",      "somewhat few interactive",
                           "somewhat many interactive", "very many interactive"],
    "hazard_count":       ["very few hazards",          "somewhat few hazards",
                           "somewhat many hazards",     "very many hazards"],
    "collectable_count":  ["very few collectables",     "somewhat few collectables",
                           "somewhat many collectables","very many collectables"],
}

# ── 어휘 세트: feature × intensity level(0~3) → 추천 표현 목록 ────────────────────
# level 0 = 가장 적음/짧음, level 3 = 가장 많음/긺
VOCAB_SETS: Dict[str, List[List[str]]] = {
    "region": [
        # level 0 — very few regions (fully connected map)
        [
            "few",
            "sparse",
            "small",
            "marginal",
        ],

        # level 1 — somewhat few regions (lightly divided)
        [
            "some",
            "moderate",
            "slight",
            "certain",
        ],

        # level 2 — somewhat many regions (noticeably split)
        [
            "several",
            "balanced",
            "multiple",
            "partitioned",
        ],

        # level 3 — very many regions (heavily fragmented)
        [
            "fragmented",
            "numerous",
            "large",
            "many",
        ],
    ],

    "path_length": [
        # level 0 — very short
        [
            "tiny",
            "nano",
            "minimal",
            "micro",
        ],

        # level 1 — somewhat short
        [
            "short",
            "limited",
            "restricted",
            "condenced",
        ],

        # level 2 — somewhat long
        [
            "moderate",
            "reasonable",
            "medium",
            "balanced",
        ],

        # level 3 — very long
        [
            "long",
            "large",
            "lengthly",
            "extensive"
        ],
    ]
}


_COUNT_VOCAB: List[List[str]] = [
    # level 0 — very few
    [
        "rare",
        "few"
        "sparse",
        "marginal",
    ],

    # level 1 — somewhat few
    [
        "some",
        "limited",
        "slight",
        "little",
    ],

    # level 2 — moderate / somewhat many
    [
        "moderate",
        "reasonable",
        "decent",
        "suitable",
    ],

    # level 3 — very many
    [
        "many",
        "numerous",
        "plentiful",
        "abundant",
    ],
]


VOCAB_SETS["interactable_count"] = _COUNT_VOCAB
VOCAB_SETS["hazard_count"]       = _COUNT_VOCAB
VOCAB_SETS["collectable_count"]  = _COUNT_VOCAB
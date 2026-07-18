from __future__ import annotations

from pathlib import Path
import re


RENDER_DIR = Path(__file__).resolve().parents[2]
PROJECT_ROOT = RENDER_DIR.parents[1]

ENTITY = "st4889ha-gwangju-institute-of-science-and-technology"
DEFAULT_PROJECTS = {
    "VIPCGRL": "aaai27_eval_vipcgrl_renderre1",
    "MGPCGRL": "aaai27_eval_mgpcgrl_renderre1",
}
DEFAULT_GAMES = ["doom", "dungeon", "pokemon"]
DEFAULT_FEATURES = ["path_length"]
DEFAULT_TILE_SIZE = 32
FEATURES = {
    "object_count": -1,
    "region": 0,
    "path_length": 1,
    "interactable_count": 2,
    "hazard_count": 3,
    "collectable_count": 4,
}
OBJECT_COUNT_FEATURE_BY_GAME = {
    "doom": "collectable_count",
    "pokemon": "collectable_count",
    "zelda": "collectable_count",
    "sokoban": "interactable_count",
    "dungeon": "collectable_count",
}
TRANSITION_LABELS = {
    "object_count": "Few objects $\\rightarrow$ Many objects",
    "region": "Few regions $\\rightarrow$ Many regions",
    "path_length": "Short path $\\rightarrow$ Long path",
    "interactable_count": "Few objects $\\rightarrow$ Many objects",
    "hazard_count": "Few hazards $\\rightarrow$ Many hazards",
    "collectable_count": "Few collectables $\\rightarrow$ Many collectables",
}
SIDE_LABELS = {
    "object_count": ("SMALL", "MID", "MANY"),
    "region": ("FEW", "MID", "MANY"),
    "path_length": ("SHORT", "MID", "LONG"),
    "count": ("FEW", "MID", "MANY"),
}
PASSABLE_TILE_IDS = {1, 3, 4, 5}
COUNT_TILE_ID_BY_REWARD_ENUM = {
    2: 3,
    3: 4,
    4: 5,
}
METHOD_ORDER = ["VIPCGRL", "MGPCGRL"]
GAME_LABEL = {
    "doom": "Doom",
    "pokemon": r"Pok\'emon",
    "zelda": "Zelda",
    "sokoban": "Sokoban",
    "dungeon": "Dungeon",
}
GAME_PREVIEW_LABEL = {
    "doom": "Doom",
    "pokemon": "Pokemon",
    "zelda": "Zelda",
    "sokoban": "Sokoban",
    "dungeon": "Dungeon",
}


def _feature_name_for_game(feature: str, game: str) -> str:
    if feature == "object_count":
        return OBJECT_COUNT_FEATURE_BY_GAME.get(game, "collectable_count")
    return feature


def _reward_enum_for_feature_game(feature: str, game: str) -> int:
    feature_name = _feature_name_for_game(feature, game)
    return FEATURES[feature_name]


def _side_labels_for_feature(feature: str) -> tuple[str, str, str]:
    if feature == "path_length":
        return SIDE_LABELS["path_length"]
    if feature == "region":
        return SIDE_LABELS["region"]
    if feature == "object_count":
        return SIDE_LABELS["object_count"]
    return SIDE_LABELS["count"]


def _fmt_num(value: float | None) -> str:
    import math

    if value is None or not math.isfinite(value):
        return "?"
    if math.isclose(value, round(value), abs_tol=1e-6):
        return str(int(round(value)))
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in str(text))


def _safe_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("_") or "x"

from typing import Dict, Iterable

# Shared game color palette used across scatter / t-SNE plots.
# Keys are normalized lowercase names.
GAME_PALETTE: Dict[str, str] = {
    "dungeon": "#e41a1c",
    "sokoban": "#ff7f00",
    "pokemon": "#377eb8",
    "zelda": "#4daf4a",
    "doom": "#984ea3",
    "doom2": "#984ea3",
}


def normalize_game_name(name: str) -> str:
    return str(name).strip().lower()


def palette_for_games(game_names: Iterable[str]) -> Dict[str, str]:
    """Return a seaborn-compatible palette keyed by display game names."""
    palette: Dict[str, str] = {}
    for g in sorted(set(game_names)):
        key = normalize_game_name(g)
        palette[g] = GAME_PALETTE.get(key, "#7f7f7f")
    return palette


"""
conf/game_utils.py
==================
Utilities for selecting games.

abbreviation rule (2text):
    dg = dungeon
    pk = pokemon
    sk = sokoban
    dm = doom  (enables doom and doom2 together)
    zd = zelda

Examples:
    all = all game enable
"""

from __future__ import annotations

import re

from typing import Dict, List, Optional, Set

# ── Two-character game abbreviation <-> include_* flags ─────────────────────────
# 'dm' covers both doom and doom2.
GAME_ABBR: Dict[str, List[str]] = {
    "dg": ["dungeon"],
    "pk": ["pokemon"],
    "sk": ["sokoban"],
    "dm": ["doom", "doom2"],
    "zd": ["zelda"],
}

# Complete list of game names (based on the include_* fields)
ALL_GAMES: List[str] = ["dungeon", "pokemon", "sokoban", "doom", "doom2", "zelda"]

# doom and doom2 count as one game, so the game list has 5 entries.
CANONICAL_GAMES: List[str] = [g for g in ALL_GAMES if g != "doom2"]
CANONICAL_GAMES_TOTAL = 5
assert len(CANONICAL_GAMES) == CANONICAL_GAMES_TOTAL, (
    f"CANONICAL_GAMES must have {CANONICAL_GAMES_TOTAL} entries (doom/doom2 merged), "
    f"got {len(CANONICAL_GAMES)}: {CANONICAL_GAMES}"
)

# Reverse map (full name -> abbr): doom and doom2 both map to dm.
GAME_ABBR_INV: Dict[str, str] = {}
for _abbr, _names in GAME_ABBR.items():
    for _name in _names:
        GAME_ABBR_INV[_name] = _abbr


def parse_game_str(game_str: str) -> Dict[str, bool]:
    """Convert a two-character abbreviation string into an ``include_*`` dict.

    Parameters
    ----------
    game_str : str
        Concatenated two-character abbreviations. ``"all"`` enables every game.

    Returns
    -------
    Dict[str, bool]
        A dict of ``include_dungeon``, ``include_pokemon``, ... flags.

    Examples
    --------
    >>> parse_game_str("dgdm")
    {'include_dungeon': True, 'include_pokemon': False, 'include_sokoban': False,
     'include_doom': True, 'include_doom2': True, 'include_zelda': False}

    >>> parse_game_str("all")
    {'include_dungeon': True, 'include_pokemon': True, 'include_sokoban': True,
     'include_doom': True, 'include_doom2': True, 'include_zelda': True}
    """
    includes = {f"include_{name}": False for name in ALL_GAMES}

    if not game_str:
        return includes

    # Special case: all
    if game_str.lower() == "all":
        return {k: True for k in includes}

    # 2text parsing
    for i in range(0, len(game_str), 2):
        abbr = game_str[i:i + 2]
        if abbr not in GAME_ABBR:
            raise ValueError(
                f"Unknown game abbreviation: '{abbr}'. "
                f"Available: {list(GAME_ABBR.keys())} or 'all'"
            )
        for full_name in GAME_ABBR[abbr]:
            includes[f"include_{full_name}"] = True

    return includes


def parse_unseen_game_names(unseen_str: str) -> set:
    """2text abbreviation string → full game name set.

    Examples
    --------
    parse_unseen_game_names("zd")   -> {'zelda'}
    parse_unseen_game_names("pkzd") -> {'pokemon', 'zelda'}
    """
    names: set = set()
    for i in range(0, len(unseen_str), 2):
        abbr = unseen_str[i:i + 2]
        names.update(GAME_ABBR.get(abbr, []))
    return names


def parse_game_names(game_str: str, *, canonical: bool = False) -> List[str]:
    """Convert a two-character abbreviation string (or ``all``) into full game names.

    With ``canonical=True``, ``doom2`` is reported as the ``doom`` game.
    """
    if not game_str:
        return []

    if game_str.lower() == "all":
        return list(CANONICAL_GAMES if canonical else ALL_GAMES)

    names: List[str] = []
    for i in range(0, len(game_str), 2):
        abbr = game_str[i:i + 2]
        if abbr not in GAME_ABBR:
            return []
        for full_name in GAME_ABBR[abbr]:
            name = "doom" if canonical and full_name == "doom2" else full_name
            if name not in names:
                names.append(name)
    return names


def infer_seen_games_from_ckpt_name(ckpt_name: str) -> List[str]:
    """Canonical seen-game string used in encoder checkpoint folder names.

    Newer zero/few-shot ckpts may carry ``_unseen-XX`` directly. Older
    full-shot subset ckpts only look like ``clip-game-dgpk_exp-def_0``; for
    those, the ``game-`` token is the seen-game subset.
    """
    if not ckpt_name:
        return []

    unseen_match = re.search(r"(?:^|_)unseen-([^_]+)", ckpt_name)
    if unseen_match:
        unseen = set(parse_game_names(unseen_match.group(1), canonical=True))
        return [g for g in CANONICAL_GAMES if g not in unseen]

    game_match = re.search(r"(?:^|[_-])game-([^_]+)", ckpt_name)
    if not game_match:
        return []
    return parse_game_names(game_match.group(1), canonical=True)


def unseen_abbr_from_seen_games(seen_games) -> Optional[str]:
    """Build canonical unseen-game abbreviation from a seen-game list.

    The order follows ``GAME_ABBR`` insertion order, matching existing run-name
    conventions such as ``dgskzd``.
    """
    if not seen_games:
        return None

    seen_game_set = {("doom" if g == "doom2" else g) for g in seen_games}
    abbr_parts: List[str] = []
    seen_abbrs: Set[str] = set()
    for abbr, names in GAME_ABBR.items():
        canonical_names = {("doom" if g == "doom2" else g) for g in names}
        if canonical_names.isdisjoint(seen_game_set) and abbr not in seen_abbrs:
            abbr_parts.append(abbr)
            seen_abbrs.add(abbr)

    return "".join(abbr_parts) or None


def build_game_str(
    include_dungeon: bool = False,
    include_pokemon: bool = False,
    include_sokoban: bool = False,
    include_doom: bool = False,
    include_doom2: bool = False,
    include_zelda: bool = False,
) -> str:
    """Build the game abbreviation string from the ``include_*`` flags.

    When both doom and doom2 are set, ``dm`` is emitted once (no duplicates).
    """
    parts: List[str] = []
    if include_dungeon:
        parts.append("dg")
    if include_pokemon:
        parts.append("pk")
    if include_sokoban:
        parts.append("sk")
    if include_doom or include_doom2:
        parts.append("dm")
    if include_zelda:
        parts.append("zd")
    return "".join(parts)


def compute_seen_unseen_split(seen_games_raw):
    """Compute the canonical seen / unseen split.

    Absorbs ``doom2`` into ``doom`` (canonical list has 5 games), sorts both
    lists, and asserts that ``len(seen) + len(unseen) == 5`` whenever ``seen``
    is non-empty.

    Parameters
    ----------
    seen_games_raw : iterable of str | None
        Game names considered "seen" by the encoder (may contain ``doom2``
        and/or ``doom``).

    Returns
    -------
    (List[str], List[str])
        Tuple of ``(seen_games, unseen_games)`` — both sorted, both drawn from
        ``CANONICAL_GAMES``.
    """
    raw = list(seen_games_raw or [])
    seen = sorted({("doom" if g == "doom2" else g) for g in raw})
    unseen = sorted(set(CANONICAL_GAMES) - set(seen)) if seen else []
    if seen:
        total = len(seen) + len(unseen)
        assert total == CANONICAL_GAMES_TOTAL, (
            f"seen + unseen total must be {CANONICAL_GAMES_TOTAL}, got {total}.\n"
            f"  seen_raw        = {raw}\n"
            f"  seen_games      = {seen}\n"
            f"  unseen_games    = {unseen}\n"
            f"  CANONICAL_GAMES = {CANONICAL_GAMES}"
        )
    return seen, unseen

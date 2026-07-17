"""
conf/game_utils.py
==================
game select text utility.

abbreviation rule (2text):
    dg = dungeon
    pk = pokemon
    sk = sokoban
    dm = doom  (doom + doom2 text enable)
    zd = zelda

text:
    all = all game enable
"""

from __future__ import annotations

import re

from typing import Dict, List, Optional, Set

# ── game 2text abbreviation ↔ include text text ──────────────────────────────────────
# dm   doom + doom2   text in   text.
GAME_ABBR: Dict[str, List[str]] = {
    "dg": ["dungeon"],
    "pk": ["pokemon"],
    "sk": ["sokoban"],
    "dm": ["doom", "doom2"],
    "zd": ["zelda"],
}

# all game name list (include_* text basis)
ALL_GAMES: List[str] = ["dungeon", "pokemon", "sokoban", "doom", "doom2", "zelda"]

# doom  and  doom2   text gametext as  text → text game list  5text
CANONICAL_GAMES: List[str] = [g for g in ALL_GAMES if g != "doom2"]
CANONICAL_GAMES_TOTAL = 5
assert len(CANONICAL_GAMES) == CANONICAL_GAMES_TOTAL, (
    f"CANONICAL_GAMES must have {CANONICAL_GAMES_TOTAL} entries (doom/doom2 merged), "
    f"got {len(CANONICAL_GAMES)}: {CANONICAL_GAMES}"
)

# text text (full name → abbr)  doom, doom2 → dm
GAME_ABBR_INV: Dict[str, str] = {}
for _abbr, _names in GAME_ABBR.items():
    for _name in _names:
        GAME_ABBR_INV[_name] = _abbr


def parse_game_str(game_str: str) -> Dict[str, bool]:
    """2text abbreviation string  ``include_*`` dict  to  converttext.

    Parameters
    ----------
    game_str : str
        2text abbreviation   text text string. ``"all"``  text all enable.

    Returns
    -------
    Dict[str, bool]
        ``include_dungeon``, ``include_pokemon``, ... text   text dict.

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

    # text: all
    if game_str.lower() == "all":
        return {k: True for k in includes}

    # 2text parsing
    for i in range(0, len(game_str), 2):
        abbr = game_str[i:i + 2]
        if abbr not in GAME_ABBR:
            raise ValueError(
                f"text text without game abbreviation: '{abbr}'. "
                f"text for  available: {list(GAME_ABBR.keys())} text  'all'"
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
    """2text abbreviation string text  ``all``  full game name text to  converttext.

    ``canonical=True`` text ``doom2``  text ``doom`` gametext as text returntext.
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
    """Encoder checkpoint folder name in  canonical seen game text  text.

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
    """``include_*`` text as text game abbreviation string  createtext.

    doom, doom2  during  text also  True text ``dm``   text text (duplicate text).
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

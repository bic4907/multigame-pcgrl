"""
conf/game_utils.py
==================
게임 선택 관련 유틸리티.

약어 규칙 (2글자):
    dg = dungeon
    pk = pokemon
    sk = sokoban
    dm = doom  (doom + doom2 동시 활성화)
    zd = zelda

특수값:
    all = 전체 게임 활성화
"""

from __future__ import annotations

from typing import Dict, List

# ── 게임 2글자 약어 ↔ include 플래그 매핑 ──────────────────────────────────────
# dm 은 doom + doom2 를 동시에 가리킨다.
GAME_ABBR: Dict[str, List[str]] = {
    "dg": ["dungeon"],
    "pk": ["pokemon"],
    "sk": ["sokoban"],
    "dm": ["doom", "doom2"],
    "zd": ["zelda"],
}

# 전체 게임 이름 목록 (include_* 필드 기준)
ALL_GAMES: List[str] = ["dungeon", "pokemon", "sokoban", "doom", "doom2", "zelda"]

# 역방향 매핑 (full name → abbr)  doom, doom2 → dm
GAME_ABBR_INV: Dict[str, str] = {}
for _abbr, _names in GAME_ABBR.items():
    for _name in _names:
        GAME_ABBR_INV[_name] = _abbr


def parse_game_str(game_str: str) -> Dict[str, bool]:
    """2글자 약어 문자열을 ``include_*`` dict 로 변환한다.

    Parameters
    ----------
    game_str : str
        2글자 약어를 이어 붙인 문자열. ``"all"`` 이면 전체 활성화.

    Returns
    -------
    Dict[str, bool]
        ``include_dungeon``, ``include_pokemon``, ... 키를 가진 dict.

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

    # 특수값: all
    if game_str.lower() == "all":
        return {k: True for k in includes}

    # 2글자씩 파싱
    for i in range(0, len(game_str), 2):
        abbr = game_str[i:i + 2]
        if abbr not in GAME_ABBR:
            raise ValueError(
                f"알 수 없는 게임 약어: '{abbr}'. "
                f"사용 가능: {list(GAME_ABBR.keys())} 또는 'all'"
            )
        for full_name in GAME_ABBR[abbr]:
            includes[f"include_{full_name}"] = True

    return includes


def build_game_str(
    include_dungeon: bool = False,
    include_pokemon: bool = False,
    include_sokoban: bool = False,
    include_doom: bool = False,
    include_doom2: bool = False,
    include_zelda: bool = False,
) -> str:
    """``include_*`` 불리언으로부터 game 약어 문자열을 생성한다.

    doom, doom2 중 하나라도 True 면 ``dm`` 을 추가한다 (중복 방지).
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


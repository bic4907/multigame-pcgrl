"""
dataset/multigame/handlers/vglc_games/doom.py
==============================================
Doom (TheVGLC) preprocessing handler.

tile text
---------
0  : empty   (-)
1  : wall    (X)
2  : floor   (., ,, :)
3  : enemy   (E)
4  : spawn   (<, T, t, >)
5  : item    (W, A, H, K)
6  : danger  (B)
7  : door    (L, +)
99 : unknown
"""
from __future__ import annotations

from typing import List, Dict, Any
from pathlib import Path
from ...base import BasePreprocessor, TileLegend, enforce_top_left_16x16, GameSample


class DoomTile:
    EMPTY   = 0
    WALL    = 1
    FLOOR   = 2
    ENEMY   = 3
    SPAWN   = 4
    ITEM    = 5
    DANGER  = 6
    DOOR    = 7
    STAIR   = 8
    UNKNOWN = 99



_CHAR_MAP: dict[str, int] = {
    "-": DoomTile.EMPTY,
    " ": DoomTile.EMPTY,
    "X": DoomTile.WALL,
    ".": DoomTile.FLOOR,
    ",": DoomTile.STAIR,
    "E": DoomTile.ENEMY,
    "W": DoomTile.ITEM,
    "A": DoomTile.ITEM,
    "H": DoomTile.ITEM,
    "B": DoomTile.DANGER,
    "K": DoomTile.ITEM,
    "<": DoomTile.SPAWN,
    "T": DoomTile.SPAWN,
    ":": DoomTile.FLOOR,
    "L": DoomTile.DOOR,
    "t": DoomTile.SPAWN,
    "+": DoomTile.DOOR,
    ">": DoomTile.SPAWN
}

DOOM_PALETTE: dict[int, tuple[int, int, int]] = {
    DoomTile.EMPTY:   (20,  20,  20),
    DoomTile.WALL:    (80,  80,  80),
    DoomTile.FLOOR:   (160, 140, 120),
    DoomTile.ENEMY:   (220, 50,  50),
    DoomTile.SPAWN:   (0,   200, 0),
    DoomTile.ITEM:    (230, 230, 20),
    DoomTile.DANGER:  (80,  80,  220),
    DoomTile.DOOR:    (80,  60,  40),
    DoomTile.UNKNOWN: (255, 255, 255),
    DoomTile.STAIR:   (40,  60,  80),
}


def make_legend() -> TileLegend:
    attrs = {
        "-" : ["empty","out of bounds"],
        "X" : ["solid","wall"],
        "." : ["floor","walkable"],
        "," : ["floor","walkable","stairs"],
        "E" : ["enemy","walkable"],
        "W" : ["weapon","walkable"],
        "A" : ["ammo","walkable"],
        "H" : ["health","armor","walkable"],
        "B" : ["explosive barrel","walkable"],
        "K" : ["key","walkable"],
        "<" : ["start","walkable"],
        "T" : ["teleport","walkable","destination"],
        ":" : ["decorative","walkable"],
        "L" : ["door","locked"],
        "t" : ["teleport","source","activatable"],
        "+" : ["door","walkable","activatable"],
        ">" : ["exit","activatable"]
    }
    return TileLegend(char_to_attrs=attrs)


class DoomPreprocessor(BasePreprocessor):
    def char_to_int(self, char: str) -> int:
        return _CHAR_MAP.get(char, DoomTile.UNKNOWN)

    def discover_and_process(
        self,
        files: List[Path],
        config: Any,
        game_tag: str,
        legend: TileLegend,
        cache: Dict[str, Any]
    ) -> List[str]:
        """
        Doom  before  for  file process text text text  to text.
        VGLCGameHandler._discover() in  calltext.
        """
        # config text
        if not hasattr(config, 'doom') or not config.doom.enabled:
            return [str(p) for p in files]

        empty_max = config.doom.empty_max
        floor_empty_max = config.doom.floor_empty_max
        event_count_min = getattr(config.doom, 'event_count_min', 1)
        max_samples = getattr(config.doom, 'max_samples', 1000)

        entries = []
        for txt_path in files:
            # max_samples reach check
            if len(entries) >= max_samples:
                break

            text = txt_path.read_text(encoding='utf-8', errors='replace')
            char_grid = self.parse_txt(text)

            sliced = self.slice_large_map(
                char_grid,
                empty_max=empty_max,
                floor_empty_max=floor_empty_max,
                event_count_min=event_count_min,
            )

            for idx, sliced_data in enumerate(sliced):
                # max_samples reach check
                if len(entries) >= max_samples:
                    break

                source_id = f"{str(txt_path)}|{idx}"
                entries.append(source_id)

                # cache save
                array = self.transform(sliced_data['map'])
                array = enforce_top_left_16x16(
                    array,
                    game=game_tag,
                    source_id=source_id
                )

                cache[source_id] = GameSample(
                    game=game_tag,
                    source_id=source_id,
                    array=array,
                    char_grid=sliced_data['map'],
                    legend=legend,
                    instruction=None,
                    order=None,
                    meta={
                        'file': txt_path.name,
                        'slice_index': idx,
                        'row_start': sliced_data['row_start'],
                        'col_start': sliced_data['col_start'],
                        'empty_count': sliced_data['empty_count'],
                        'floor_count': sliced_data['floor_count'],
                        'event_count': sliced_data['event_count'],
                    }
                )
        return entries

    def slice_large_map(
        self,
        char_grid: List[List[str]],
        empty_max: int = 128,
        floor_empty_max: int = 239,
        event_count_min: int = 1,
    ) -> List[Dict[str, Any]]:
        """
        text map  16x16 text  maptext to  text text

        rule:
        1. text to : 16text text
        2.   to : 16text movetext validtext text. validtext maptext text
        3. validtext: empty("-") <= empty_max AND floor+empty <= floor_empty_max AND event_count >= event_count_min text text text

        Parameters
        ----------
        char_grid : List[List[str]]
            2D character text
        empty_max : int
            validtext map of  maximum empty tile count
        floor_empty_max : int
            validtext map of  floor+empty text of  maximumtext
        event_count_min : int
            validtext map of  enemy+object text of  minimum (default value: 1)

        Returns
        -------
        List[Dict]
            each dict: {
                'map': 16x16 char_grid,
                'row_start': start row,
                'col_start': start column,
                'empty_count': empty tile count,
                'floor_count': floor tile count,
                'event_count': enemy+object tile count,
            }
        """
        if not char_grid:
            return []

        height = len(char_grid)
        width = max(len(row) for row in char_grid) if char_grid else 0

        sliced_maps = []

        # text to  to  16text text
        row = 0
        while row < height:
            row_end = min(row + 16, height)
            row_slice = char_grid[row:row_end]

            #   to  to  16text movetext text
            col = 0
            while col < width:
                col_end = min(col + 16, width)
                # 16x16 map extract
                map_16x16 = []
                for r in row_slice:
                    if col < len(r):
                        row_data = list(r[col:col_end])
                    else:
                        row_data = []

                    #   to  padding (empty '-' to )
                    while len(row_data) < 16:
                        row_data.append('-')

                    map_16x16.append(row_data)

                # text to  padding (empty '-' to )
                while len(map_16x16) < 16:
                    map_16x16.append(['-'] * 16)

                # validtext text (empty_max, floor+empty, event_count check)
                empty_count = sum(1 for r in map_16x16 for cell in r if cell == '-')
                floor_count = sum(1 for r in map_16x16 for cell in r if cell in '.,:')
                # event_count: enemy(E) + object(W,A,H,B,K) sum
                event_count = sum(1 for r in map_16x16 for cell in r if cell in 'EWAHBK')

                if (empty_count <= empty_max and
                    floor_count + empty_count <= floor_empty_max and
                    event_count >= event_count_min):
                    sliced_maps.append({
                        'map': map_16x16,
                        'row_start': row,
                        'col_start': col,
                        'empty_count': empty_count,
                        'floor_count': floor_count,
                        'event_count': event_count,
                    })

                # next abovetext: 16text text
                col += 16

            # next abovetext: 16text text
            row += 16

        return sliced_maps

"""
dataset/multigame/handlers/vglc_games/doom.py
==============================================
Doom (TheVGLC) 전처리 핸들러.

타일 매핑
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
    UNKNOWN = 99



_CHAR_MAP: dict[str, int] = {
    "-": DoomTile.EMPTY,
    " ": DoomTile.EMPTY,
    "X": DoomTile.WALL,
    ".": DoomTile.FLOOR,
    ",": DoomTile.SPAWN,
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
        Doom 전용 파일 처리 및 슬라이싱 로직.
        VGLCGameHandler._discover()에서 호출됨.
        """
        # 설정 체크
        if not hasattr(config, 'doom') or not config.doom.enabled:
            return [str(p) for p in files]

        empty_max = config.doom.empty_max
        floor_empty_max = config.doom.floor_empty_max
        event_count_min = getattr(config.doom, 'event_count_min', 1)
        max_samples = getattr(config.doom, 'max_samples', 1000)

        entries = []
        for txt_path in files:
            # max_samples 도달 확인
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
                # max_samples 도달 확인
                if len(entries) >= max_samples:
                    break
                
                source_id = f"{str(txt_path)}|{idx}"
                entries.append(source_id)

                # 캐시 저장
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
        큰 맵을 16x16 작은 맵들로 슬라이싱
        
        규칙:
        1. 세로: 16줄씩 탐색
        2. 가로: 16칸씩 이동하며 유효성 체크. 유효한 맵만 추가
        3. 유효성: empty("-") <= empty_max AND floor+empty <= floor_empty_max AND event_count >= event_count_min 인 경우만 추가
        
        Parameters
        ----------
        char_grid : List[List[str]]
            2D 문자 그리드
        empty_max : int
            유효한 맵의 최대 empty 타일 개수
        floor_empty_max : int
            유효한 맵의 floor+empty 합의 최대값
        event_count_min : int
            유효한 맵의 enemy+object 합의 최솟값 (기본값: 1)
        
        Returns
        -------
        List[Dict]
            각 dict: {
                'map': 16x16 char_grid,
                'row_start': 시작 행,
                'col_start': 시작 열,
                'empty_count': empty 타일 개수,
                'floor_count': floor 타일 개수,
                'event_count': enemy+object 타일 개수,
            }
        """
        if not char_grid:
            return []
        
        height = len(char_grid)
        width = max(len(row) for row in char_grid) if char_grid else 0
        
        sliced_maps = []
        
        # 세로로 16줄씩 탐색
        row = 0
        while row < height:
            row_end = min(row + 16, height)
            row_slice = char_grid[row:row_end]
            
            # 가로로 16칸씩 이동하며 탐색
            col = 0
            while col < width:
                col_end = min(col + 16, width)
                # 16x16 맵 추출
                map_16x16 = []
                for r in row_slice:
                    if col < len(r):
                        row_data = list(r[col:col_end])
                    else:
                        row_data = []
                    
                    # 가로 패딩 (empty '-'로)
                    while len(row_data) < 16:
                        row_data.append('-')
                    
                    map_16x16.append(row_data)
                
                # 세로 패딩 (empty '-'로)
                while len(map_16x16) < 16:
                    map_16x16.append(['-'] * 16)
                
                # 유효성 체크 (empty_max, floor+empty, event_count 확인)
                empty_count = sum(1 for r in map_16x16 for cell in r if cell == '-')
                floor_count = sum(1 for r in map_16x16 for cell in r if cell in '.,:')
                # event_count: enemy(E) + object(W,A,H,B,K) 합산
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
                
                # 다음 위치: 16칸 뛰어넘기
                col += 16
            
            # 다음 위치: 16줄 뛰어넘기
            row += 16
        
        return sliced_maps

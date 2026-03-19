"""
dataset/multigame/handlers/fdm_game/augmentation.py
===================================================
FDM 데이터 증강 유틸리티.

- 시계방향 90도 회전
- 방향 어휘 자동 변환
"""
from __future__ import annotations

import re
from typing import Dict, Tuple
import numpy as np
from ...base import GameSample
import dataclasses


# 방향 어휘 변환 규칙 (시계방향 90도 회전)
# right, left, up, down
DIRECTION_MAPPING_1 = {
    "right": "down",
    "left": "up",
    "up": "right",
    "down": "left",
}

# north, south, east, west
DIRECTION_MAPPING_2 = {
    "north": "east",
    "south": "west",
    "east": "south",
    "west": "north",
}

# top, bottom, right, left
DIRECTION_MAPPING_3 = {
    "top": "right",
    "bottom": "left",
    "right": "bottom",
    "left": "top",
}

# 모든 맵 통합
ALL_DIRECTION_MAPPINGS = {
    **DIRECTION_MAPPING_1,
    **DIRECTION_MAPPING_2,
    **DIRECTION_MAPPING_3,
}


def rotate_array_cw_90(array: np.ndarray) -> np.ndarray:
    """
    배열을 시계방향으로 90도 회전.
    
    Parameters
    ----------
    array : (H, W) int32
    
    Returns
    -------
    (W, H) int32 회전된 배열
    
    Examples
    --------
    [[1, 2],     [[3, 1],
     [3, 4]]  →   [4, 2]]
    """
    # np.rot90은 반시계방향이므로, k=-1을 사용하여 시계방향 회전
    return np.rot90(array, k=-1).astype(array.dtype)


def transform_instruction_for_rotation(instruction: str) -> str:
    """
    시계방향 90도 회전에 맞게 instruction의 방향 어휘를 변환.
    
    Parameters
    ----------
    instruction : str
        원본 instruction
    
    Returns
    -------
    str
        방향 어휘가 변환된 instruction
    
    Examples
    --------
    "A path to the right" → "A path to the down"
    "Trees on the left side" → "Trees on the up side"
    """
    if not instruction:
        return instruction
    
    # 가장 긴 단어부터 처리하여 중복 매칭 방지
    sorted_words = sorted(ALL_DIRECTION_MAPPINGS.keys(), key=len, reverse=True)
    
    result = instruction
    for original in sorted_words:
        rotated = ALL_DIRECTION_MAPPINGS[original]
        
        # 단어 경계를 고려한 정규식
        pattern = r'\b' + re.escape(original) + r'\b'
        
        # 대소문자 보존하며 치환
        def replace_preserve_case(match):
            matched_text = match.group()
            if matched_text.isupper():
                return rotated.upper()
            elif matched_text and matched_text[0].isupper():
                return rotated.capitalize()
            else:
                return rotated
        
        result = re.sub(pattern, replace_preserve_case, result, flags=re.IGNORECASE)
    
    return result


def create_rotated_sample(sample: GameSample) -> GameSample:
    """
    샘플을 시계방향 90도 회전시킨 새 샘플을 생성.
    
    Parameters
    ----------
    sample : GameSample
        원본 샘플
    
    Returns
    -------
    GameSample
        회전된 샘플 (source_id에 '_rot90' 추가)
    
    Examples
    --------
    sample = GameSample(source_id="map_001", array=...)
    rotated = create_rotated_sample(sample)
    rotated.source_id == "map_001_rot90"  # True
    """
    # 배열 회전
    rotated_array = rotate_array_cw_90(sample.array)
    
    # char_grid 회전 (있으면, 그리고 직사각형이면)
    rotated_char_grid = None
    if sample.char_grid is not None:
        # char_grid는 List[List[str]] 형태
        # 모든 행의 길이가 같아야 numpy array로 변환 가능
        if len(sample.char_grid) > 0:
            row_lengths = [len(row) for row in sample.char_grid]
            # 모든 행의 길이가 같은 경우에만 회전 시도
            if len(set(row_lengths)) == 1:
                try:
                    char_arr = np.array(sample.char_grid)
                    rotated_char_arr = rotate_array_cw_90(char_arr)
                    rotated_char_grid = rotated_char_arr.tolist()
                except (ValueError, TypeError):
                    # 변환 실패 시 None으로 설정
                    rotated_char_grid = None
    
    # instruction 변환
    rotated_instruction = None
    if sample.instruction:
        rotated_instruction = transform_instruction_for_rotation(sample.instruction)
    
    # 새 샘플 생성
    return dataclasses.replace(
        sample,
        source_id=f"{sample.source_id}_rot90",
        array=rotated_array,
        char_grid=rotated_char_grid,
        instruction=rotated_instruction,
        order=None,  # order는 이후 재지정
        meta={**sample.meta, "augmented": "rot90"},
    )


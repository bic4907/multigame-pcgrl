"""
dataset/multigame/handlers/fdm_game/augmentation.py
===================================================
FDM data augmentation utility.

- text 90 also  rotate
- text text automatic convert
"""
from __future__ import annotations

import re
from typing import Dict, Tuple
import numpy as np
from ...base import GameSample
import dataclasses


# text text convert rule (text 90 also  rotate)
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

# text map text
ALL_DIRECTION_MAPPINGS = {
    **DIRECTION_MAPPING_1,
    **DIRECTION_MAPPING_2,
    **DIRECTION_MAPPING_3,
}


def rotate_array_cw_90(array: np.ndarray) -> np.ndarray:
    """
    array  text as  90 also  rotate.

    Parameters
    ----------
    array : (H, W) int32

    Returns
    -------
    (W, H) int32 rotatetext array

    Examples
    --------
    [[1, 2],     [[3, 1],
     [3, 4]]  →   [4, 2]]
    """
    # np.rot90  text text to , k=-1  text for text text rotate
    return np.rot90(array, k=-1).astype(array.dtype)


def transform_instruction_for_rotation(instruction: str) -> str:
    """
    text 90 also  rotate in  text instruction of  text text  convert.

    Parameters
    ----------
    instruction : str
        text instruction

    Returns
    -------
    str
        text text  converttext instruction

    Examples
    --------
    "A path to the right" → "A path to the down"
    "Trees on the left side" → "Trees on the up side"
    """
    if not instruction:
        return instruction

    #  text text text processtext duplicate text text
    sorted_words = sorted(ALL_DIRECTION_MAPPINGS.keys(), key=len, reverse=True)

    result = instruction
    for original in sorted_words:
        rotated = ALL_DIRECTION_MAPPINGS[original]

        # text text  text text
        pattern = r'\b' + re.escape(original) + r'\b'

        # textcharacter preservetext text
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
    sample  text 90 also  rotatetext text sample  create.

    Parameters
    ----------
    sample : GameSample
        text sample

    Returns
    -------
    GameSample
        rotatetext sample (source_id in  '_rot90' text )

    Examples
    --------
    sample = GameSample(source_id="map_001", array=...)
    rotated = create_rotated_sample(sample)
    rotated.source_id == "map_001_rot90"  # True
    """
    # array rotate
    rotated_array = rotate_array_cw_90(sample.array)

    # char_grid rotate (text, text texteachtext text)
    rotated_char_grid = None
    if sample.char_grid is not None:
        # char_grid  List[List[str]] form
        # text row of  text   text numpy array to  convert available
        if len(sample.char_grid) > 0:
            row_lengths = [len(row) for row in sample.char_grid]
            # text row of  text   same text in text rotate text also
            if len(set(row_lengths)) == 1:
                try:
                    char_arr = np.array(sample.char_grid)
                    rotated_char_arr = rotate_array_cw_90(char_arr)
                    rotated_char_grid = rotated_char_arr.tolist()
                except (ValueError, TypeError):
                    # convert failure text None as  config
                    rotated_char_grid = None

    # instruction convert
    rotated_instruction = None
    if sample.instruction:
        rotated_instruction = transform_instruction_for_rotation(sample.instruction)

    # text sample create
    return dataclasses.replace(
        sample,
        source_id=f"{sample.source_id}_rot90",
        array=rotated_array,
        char_grid=rotated_char_grid,
        instruction=rotated_instruction,
        order=None,  # order    after  text
        meta={**sample.meta, "augmented": "rot90"},
    )


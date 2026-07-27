"""
dataset/multigame/handlers/fdm_game/augmentation.py
===================================================
FDM data augmentation utility.

- 90-degree clockwise rotation
- Automatic conversion of directional terms
"""
from __future__ import annotations

import re
from typing import Dict, Tuple
import numpy as np
from ...base import GameSample
import dataclasses


# Directional-term conversion rules for a 90-degree clockwise rotation
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

# Combined mapping
ALL_DIRECTION_MAPPINGS = {
    **DIRECTION_MAPPING_1,
    **DIRECTION_MAPPING_2,
    **DIRECTION_MAPPING_3,
}


def rotate_array_cw_90(array: np.ndarray) -> np.ndarray:
    """
    Rotate an array 90 degrees clockwise.

    Parameters
    ----------
    array : (H, W) int32

    Returns
    -------
    Rotated (W, H) int32 array.

    Examples
    --------
    [[1, 2],     [[3, 1],
     [3, 4]]  →   [4, 2]]
    """
    # np.rot90 rotates counterclockwise, so use k=-1 for clockwise rotation
    return np.rot90(array, k=-1).astype(array.dtype)


def transform_instruction_for_rotation(instruction: str) -> str:
    """
    Convert directional terms in an instruction to match a 90-degree clockwise rotation.

    Parameters
    ----------
    instruction : str
        Original instruction.

    Returns
    -------
    str
        Instruction with converted directional terms.

    Examples
    --------
    "A path to the right" → "A path to the down"
    "Trees on the left side" → "Trees on the up side"
    """
    if not instruction:
        return instruction

    # Process the longest terms first to prevent overlapping matches
    sorted_words = sorted(ALL_DIRECTION_MAPPINGS.keys(), key=len, reverse=True)

    result = instruction
    for original in sorted_words:
        rotated = ALL_DIRECTION_MAPPINGS[original]

        # Regular expression that respects word boundaries
        pattern = r'\b' + re.escape(original) + r'\b'

        # Replace while preserving letter case
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
    Create a new sample rotated 90 degrees clockwise.

    Parameters
    ----------
    sample : GameSample
        Original sample.

    Returns
    -------
    GameSample
        Rotated sample with "_rot90" appended to source_id.

    Examples
    --------
    sample = GameSample(source_id="map_001", array=...)
    rotated = create_rotated_sample(sample)
    rotated.source_id == "map_001_rot90"  # True
    """
    # array rotate
    rotated_array = rotate_array_cw_90(sample.array)

    # Rotate char_grid when present and rectangular
    rotated_char_grid = None
    if sample.char_grid is not None:
        # char_grid  List[List[str]] form
        # All rows must have equal length to convert to a NumPy array
        if len(sample.char_grid) > 0:
            row_lengths = [len(row) for row in sample.char_grid]
            # Attempt rotation only when all rows have equal length
            if len(set(row_lengths)) == 1:
                try:
                    char_arr = np.array(sample.char_grid)
                    rotated_char_arr = rotate_array_cw_90(char_arr)
                    rotated_char_grid = rotated_char_arr.tolist()
                except (ValueError, TypeError):
                    # Set to None if conversion fails
                    rotated_char_grid = None

    # instruction convert
    rotated_instruction = None
    if sample.instruction:
        rotated_instruction = transform_instruction_for_rotation(sample.instruction)

    # Create the new sample
    return dataclasses.replace(
        sample,
        source_id=f"{sample.source_id}_rot90",
        array=rotated_array,
        char_grid=rotated_char_grid,
        instruction=rotated_instruction,
        order=None,  # Reassigned later
        meta={**sample.meta, "augmented": "rot90"},
    )

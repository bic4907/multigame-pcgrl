from enum import IntEnum


class Dungeon3Tiles(IntEnum):
    BORDER = 0
    EMPTY = 1
    WALL = 2
    BAT = 3


def condition2string(row):
    conditions = []

    # Condition 0: Number of regions
    if row['condition_0'] != -1:
        conditions.append(f"Number of regions: {row['condition_0']}"
                          f"(Definition: the number of rooms, which is empty space (1) surrounded by WALL(2) tiles.)"
                          f"(Tip: Make separted round or rectangular rooms with empty tile and fill the remained spaces with walls.)")

    # Condition 1: Path length, diameter
    if row['condition_1'] != -1:
        conditions.append(f"Path length, diameter: {row['condition_1']}"
                          f"(Definition: the longest path length of connecting two arbitrary point of the level.)"
                          f"(Tip: Generate multiple paths which has length of the condition will empty tiles (1) and fill out the most of remained spaces with walls (2).)")

    # Condition 2: Number of blocks
    if row['condition_2'] != -1:
        conditions.append(f"Number of blocks: {row['condition_2']}"
                          f"(Definition: total count of WALL(2) tiles in the level.)"
                          f"(Tip: Place the block connected by each other, you can make some wall clusters.)")

    # Condition 3: Number of bats
    if row['condition_3'] != -1:
        conditions.append(f"Number of bats: {row['condition_3']}"
                            f"(Definition: total count of BAT(3) tiles in the level.)"
                          f"(Tip: Scatter the bats in a way that they are not too close to each other. Do not place the remain spaces only with empty tiles (1), just fill like a game level.)")

    # Condition 4: Direction of bats
    if row['condition_4'] != -1:
        directions = ['West', 'North', 'East', 'South']
        direction_str = directions[int(row['condition_4'])]
        conditions.append(f"Direction of bats: {direction_str}"
                          f"(Definition: spatial distribution of BAT(3) tiles relative to the center of the map.)"
                          f"(Tip: Scatter the bats to the given direction. Not showing as a cluster. Do not place the remain spaces only with empty tiles (1), just fill like a game level.)")

    return ", ".join(conditions) if conditions else "No specific conditions"

RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "level_dimensions",
        "schema": {
            "type": "object",
            "properties": {
                "matrix": {
                    "type": "array",
                    "description": "A three-dimensional array filled with numbers 1, 2, or 3.",
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "array",
                            "items": {
                                "type": "number",
                                "enum": [
                                    1,
                                    2,
                                    3
                                ]
                            }
                        }
                    }
                }
            },
            "required": [
                "matrix"
            ],
            "additionalProperties": False
        }
    }
}

SYSTEM_MESSAGE = """
    You are a professional game level designer for a 2D dungeon game. 
    You have to generate the levels like a human game designer would, and you have to follow the given instruction and condition.

    Block types:
    - 1: Empty
    - 2: Wall
    - 3: Bat

    Return your output strictly as a JSON object formatted as follows:
    {{
        "level": [[int, int, ...], [...], ...]  # A NX16x16 array of integers representing the tiles. N is the number of the generated levels.
    }}
"""

USER_MESSAGE = """
    You have to generate the level to satisfy the following requirements:
    
    You have to generate (revise) the level from this randomly generated level:
    {rnd_level}
    
    Specific condition: {condition}
    Auxiliary information: {instruction}

    Output levels: ({n}, 16, 16) array of integers representing the tiles, 16x16 is the shape of the level and {n} is the batch.
"""

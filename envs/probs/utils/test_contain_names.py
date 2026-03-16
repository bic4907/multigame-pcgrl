from enum import Enum

from .enum_validator import contains_name


class Color(Enum):
    RED = 1
    GREEN = 2
    BLUE = 3


def test_contains_name():
    assert contains_name(Color, "RED") is True
    assert contains_name(Color, "YELLOW") is False

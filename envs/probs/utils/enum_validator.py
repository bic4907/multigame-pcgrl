from enum import Enum
from typing import Union, Iterable


def validate_enum_class(enum_class: Enum) -> None:
    """
    check each tile is validate

    Args:
        enum_class (Enum): class to check

    Raises:
        TypeError: Raises if enum is not valid
    """
    if not isinstance(enum_class, type) or not issubclass(enum_class, Enum):
        raise TypeError(f"{enum_class}is not a valid enum class.")


def contains_name(
    enum_class: Enum,
    name: Union[str, int],
) -> bool:
    """
    Checks whether a given name exists in the specified Enum class.

    Args:
        enum_class (Enum): The Enum class to check
        name (Union[str, int]): The string or integer to look for

    Returns:
        bool: True if the name exists in the Enum, False otherwise

    Raises:
        TypeError: Raised if enum_class is not a valid Enum class.
        AttributeError: Raised if the given name does not exist in the Enum class.
    """
    validate_enum_class(enum_class)

    if any(name in member.name for member in enum_class):
        return True
    else:
        raise AttributeError(
            f"AttributeError: {name} not exists in {enum_class.__name__}."
        )


def contains_names(
    enum_class: Enum,
    names: Iterable[Union[str, int]],
) -> bool:
    """
    Checks whether all specified names exist in the given Enum class.

    Args:
        enum_class (Enum): The Enum class to check
        names (Iterable[Union[str, int]]): An iterable of strings or integers to check for membership

    Returns:
        bool: True if all names exist in the Enum, False otherwise

    Raises:
        TypeError: Raised if enum_class is not a valid Enum class.
        AttributeError: Raised if any of the given names do not exist in the Enum class.
    """

    validate_enum_class(enum_class)

    for name in names:
        if not any(name in member.name for member in enum_class):
            raise AttributeError(
                f"AttributeError: {name} not exists in {enum_class.__name__}."
            )
    return True


def contains_one_of(
    enum_class: Enum,
    names: Iterable[Union[str, int]],
) -> bool:
    """
    Checks whether at least one of the specified names exists in the given Enum class.

    Args:
        enum_class (Enum): The Enum class to check
        names (Iterable[Union[str, int]]): An iterable of strings or integers to check for membership

    Returns:
        bool: True if at least one name exists in the Enum, False otherwise

    Raises:
        TypeError: Raised if enum_class is not a valid Enum class.
        AttributeError: Raised if any of the given names do not exist in the Enum class.
    """

    validate_enum_class(enum_class)

    for name in names:
        if any(name in member.name for member in enum_class):
            return True

    raise AttributeError(
        f"AttributeError: {name} not exists in {enum_class.__name__}."
    )

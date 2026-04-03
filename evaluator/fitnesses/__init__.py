from .amount import get_amount_fitness
from .multigame_amount import get_multigame_amount_fitness
from .region import get_region_fitness
from .path_length import get_path_length_fitness

try:
    from .direction import get_direction_fitness
except ImportError:
    get_direction_fitness = None

__all__ = [
    "get_amount_fitness",
    "get_multigame_amount_fitness",
    "get_path_length_fitness",
    "get_region_fitness",
]

import os
import chex
from flax import struct


__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))


@struct.dataclass
class Instruct:
    reward_i: int
    condition: chex.Array
    embedding: chex.Array
    condition_id: int = None
    level: chex.Array = None
    reward_model_mask: chex.Array = None


@struct.dataclass
class EmbeddingBufferReward:
    embedding: chex.Array
    buffer: chex.Array
    reward: chex.Array

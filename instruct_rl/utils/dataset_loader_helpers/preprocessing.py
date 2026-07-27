"""
instruct_rl/utils/dataset_loader_helpers/preprocessing.py
==========================================================
Common preprocessing utilities for lists of GameSamples.

Apply identically in every data-loading pipeline (CPCGRL, IPCGRL, VIPCGRL,
MGPCGRL, the CLIP encoder, and the MLP encoder).
"""

from __future__ import annotations

import dataclasses
import logging
import os
from os.path import basename

log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logger = logging.getLogger(basename(__file__))
logger.setLevel(getattr(logging, log_level, logging.INFO))

# (game, reward_enum, cutoff): remove samples whose condition >= cutoff
LONGTAIL_CUTOFF = [
    ("dungeon", 1, 80),   # path_length >= 80
    ("pokemon", 2, 150),  # interactive_count >= 150
    ("pokemon", 4, 29),   # collectable_count >= 29
]


def _invalid_instruction(inst) -> bool:
    if inst is None:
        return True
    s = str(inst).strip()
    return s == "" or s.lower() == "none" or s.lower() == "nan"


def apply_longtail_cut(samples: list) -> list:
    """Remove samples with extreme condition values according to LONGTAIL_CUTOFF."""
    def _is_longtail(s) -> bool:
        reward_enum = s.meta.get("reward_enum")
        condition_value = s.meta.get("conditions", {}).get(reward_enum)
        if condition_value is None:
            return False
        return any(
            s.game == game and reward_enum == enum and condition_value >= cutoff
            for game, enum, cutoff in LONGTAIL_CUTOFF
        )
    return [s for s in samples if not _is_longtail(s)]


def apply_tile_offset(samples: list, offset: int) -> list:
    """Return new samples with offset added to every array tile value."""
    if offset == 0:
        return samples
    return [dataclasses.replace(s, array=s.array + offset) for s in samples]


def preprocess_samples(samples: list, *, longtail_cut: bool = True) -> list:
    """common sample preprocessing: invalid instruction filter + longtail cut.

    Apply identically during encoder and RL training.
    """
    n_before = len(samples)
    dropped_combos = sorted(set(
        (s.game, s.meta.get("reward_enum"))
        for s in samples if _invalid_instruction(s.instruction)
    ))
    samples = [s for s in samples if not _invalid_instruction(s.instruction)]
    n_dropped = n_before - len(samples)
    if n_dropped > 0:
        logger.info(
            "Instruction filter: %d → %d (dropped %d). Dropped (game, re) combos: %s",
            n_before, len(samples), n_dropped, dropped_combos,
        )
    else:
        logger.info("Instruction filter: all %d samples valid.", n_before)

    if longtail_cut:
        n_before_lt = len(samples)
        samples = apply_longtail_cut(samples)
        logger.info(
            "Longtail cut: %d → %d (removed %d)",
            n_before_lt, len(samples), n_before_lt - len(samples),
        )

    return samples


def build_effective_instructions(samples: list, *, instruction_prefix) -> list:
    """Return samples with instruction_prefix (name/desc/none) applied to instructions.

    For instruction_prefix in {"name", "desc"}, use the same seed (42) as CLIP
    embedding calculation (_tokenize_texts), producing strings identical to the
    text actually passed to the embedder.

    Use this function in both training and evaluation for consistency.
    """
    raw = [getattr(s, 'instruction', None) for s in samples]

    import random as _random
    from encoder.data.clip_batch import (
        apply_instruction_prefix,
        _normalize_instruction_prefix_mode,
    )

    mode = _normalize_instruction_prefix_mode(instruction_prefix)
    if mode == "none":
        return raw

    # Same fixed seed (42) as _tokenize_texts, matching the embedded text
    _rng = _random.Random(42)
    result = [
        apply_instruction_prefix(inst, s.game, _rng, mode)
        if inst and getattr(s, 'game', None)
        else inst
        for inst, s in zip(raw, samples)
    ]
    logger.info(
        "build_effective_instructions: instruction_prefix='%s', %d samples "
        "(example: '%s')",
        mode,
        len(result),
        result[0][:120] if result else "",
    )
    return result

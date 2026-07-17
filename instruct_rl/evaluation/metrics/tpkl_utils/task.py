"""
task.py
=======
text text create text instruct_df → pred_groups convert utility.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np


def quantize_condition(game: str, feature_name: str, cond_val) -> str:
    """condition text  integer string to  normalize.
    None → 'none', text(int/float/str) → str(round(float(v)))
    """
    if cond_val is None:
        return "none"
    try:
        return str(round(float(cond_val)))
    except (TypeError, ValueError):
        return str(cond_val)


def build_task_key(game: str, reward_enum: int, cond_val,
                   feature_name: str = "") -> str:
    """'{game}_{reward_enum}_{q_bin}' form of  text text text  return.
    feature_name  text in  text text text.
    """
    q = quantize_condition(game, feature_name, cond_val)
    return f"{game}_{reward_enum}_{q}"


def group_states_by_task(
    instruct_df,
    states: np.ndarray,
    n_eps: int,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    instruct_df based as  states  text textabove to  text.

    Parameters
    ----------
    instruct_df : pd.DataFrame  (game, reward_enum, feature_name, condition_value)
    states      : (n_inst * n_eps, H, W)
    n_eps       :  in text(seed) text

    Returns
    -------
    {task_key: (original_indices, levels)}
        original_indices : states  inside  text abovetext (order text for )
        levels           : (k, H, W) int
    """
    task_key_list = []
    for _, row in instruct_df.iterrows():
        game = str(row.get("game", "unknown"))
        re   = int(row.get("reward_enum", 0))
        feat = str(row.get("feature_name", ""))
        cval = row.get("condition_value", None)
        if cval is not None:
            try:
                cval = float(cval)
            except (TypeError, ValueError):
                cval = None
        task_key_list.extend([build_task_key(game, re, cval, feat)] * n_eps)

    groups: dict = defaultdict(list)
    for i, key in enumerate(task_key_list):
        groups[key].append(i)

    return {
        key: (np.array(idxs), states[idxs])
        for key, idxs in groups.items()
    }


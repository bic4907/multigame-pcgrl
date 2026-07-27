"""
task.py
=======
Utilities for creating task keys and converting instruct_df to pred_groups.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np


def quantize_condition(game: str, feature_name: str, cond_val) -> str:
    """Normalize a condition value to an integer string.
    None becomes 'none'; a number (int/float/str) becomes str(round(float(v))).
    """
    if cond_val is None:
        return "none"
    try:
        return str(round(float(cond_val)))
    except (TypeError, ValueError):
        return str(cond_val)


def build_task_key(game: str, reward_enum: int, cond_val,
                   feature_name: str = "") -> str:
    """Return a task key in the form '{game}_{reward_enum}_{q_bin}'.
    feature_name is not included in the key.
    """
    q = quantize_condition(game, feature_name, cond_val)
    return f"{game}_{reward_enum}_{q}"


def group_states_by_task(
    instruct_df,
    states: np.ndarray,
    n_eps: int,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Group states by task using instruct_df.

    Parameters
    ----------
    instruct_df : pd.DataFrame  (game, reward_enum, feature_name, condition_value)
    states      : (n_inst * n_eps, H, W)
    n_eps       : number of episodes (seeds)

    Returns
    -------
    {task_key: (original_indices, levels)}
        original_indices : original positions in states, used to restore order
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

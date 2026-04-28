"""
task.py
=======
태스크 키 생성 및 instruct_df → pred_groups 변환 유틸.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np


def quantize_condition(game: str, feature_name: str, cond_val) -> str:
    """조건 값을 정수 문자열로 정규화.
    None → 'none', 숫자(int/float/str) → str(round(float(v)))
    """
    if cond_val is None:
        return "none"
    try:
        return str(round(float(cond_val)))
    except (TypeError, ValueError):
        return str(cond_val)


def build_task_key(game: str, reward_enum: int, cond_val,
                   feature_name: str = "") -> str:
    """'{game}_{reward_enum}_{q_bin}' 형태의 태스크 식별 키를 반환.
    feature_name은 키에 포함되지 않는다.
    """
    q = quantize_condition(game, feature_name, cond_val)
    return f"{game}_{reward_enum}_{q}"


def group_states_by_task(
    instruct_df,
    states: np.ndarray,
    n_eps: int,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    instruct_df 기반으로 states를 태스크 단위로 그룹화.

    Parameters
    ----------
    instruct_df : pd.DataFrame  (game, reward_enum, feature_name, condition_value)
    states      : (n_inst * n_eps, H, W)
    n_eps       : 에피소드(시드) 수

    Returns
    -------
    {task_key: (original_indices, levels)}
        original_indices : states 내 원래 위치 (순서 복원용)
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


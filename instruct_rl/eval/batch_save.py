"""
instruct_rl/eval/batch_save.py
================================
평가 루프 내 배치 결과 저장 헬퍼.
"""
from __future__ import annotations

import numpy as np

from instruct_rl.eval.hdf5_store import write_sample


def save_batch_results(
    idxes,
    batch_valid_size: int,
    batch_reward_i,
    batch_repetition,
    result,
    last_states,
    instruct_df=None,
    h5=None,
):

    for idx, (row_i, reward_i, repeat_i, feature, state) in enumerate(zip(
        idxes,
        batch_reward_i[:batch_valid_size],
        batch_repetition[:batch_valid_size],
        result.feature[:batch_valid_size],
        last_states.env_state.env_map[0, :][:batch_valid_size],
    )):
        # 폴더명: {game}_re{re}_{row_i:04d}  (메타 없으면 기존 reward_{row_i} 유지)
        if instruct_df is not None and row_i < len(instruct_df):
            meta = instruct_df.iloc[int(row_i)]
            game   = str(meta.get('game', 'unknown'))
            re_val = int(meta.get('reward_enum', int(reward_i[0]) if hasattr(reward_i, '__len__') else int(reward_i)))
            folder_name = f"{game}_re{re_val}_{int(row_i):04d}"
        else:
            folder_name = f"reward_{row_i}"

        # ── HDF5 저장 — state(env_map)만 저장 ────────────────────────────────
        if h5 is not None:
            write_sample(h5, folder_name, int(repeat_i), state=np.array(state))


def build_task_text(reward_i, feature) -> str:
    labels = {1: f"RG: {int(feature[0])} | ",
              2: f"PL: {int(feature[1])} | ",
              3: f"WC: {int(feature[2])} | ",
              4: f"BC: {int(feature[3])} | ",
              5: "BD | "}
    return "".join(v for k, v in labels.items() if k in reward_i)

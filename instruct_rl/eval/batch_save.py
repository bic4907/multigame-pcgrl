"""
instruct_rl/eval/batch_save.py
================================
evaluation loop  inside  batch result save text.
"""
from __future__ import annotations

import numpy as np


def save_batch_results(
    idxes,
    batch_valid_size: int,
    batch_reward_i,
    batch_repetition,
    result,
    last_states,
    instruct_df=None,
    h5_writer=None,   # AsyncH5Writer text
):

    for idx, (row_i, reward_i, repeat_i, feature, state) in enumerate(zip(
        idxes,
        batch_reward_i[:batch_valid_size],
        batch_repetition[:batch_valid_size],
        result.feature[:batch_valid_size],
        last_states.env_state.env_map[0, :][:batch_valid_size],
    )):
        # foldertext: {game}_re{re}_{row_i:04d}  (meta if missing existing reward_{row_i} keep)
        if instruct_df is not None and row_i < len(instruct_df):
            meta = instruct_df.iloc[int(row_i)]
            game   = str(meta.get('game', 'unknown'))
            re_val = int(meta.get('reward_enum', int(reward_i[0]) if hasattr(reward_i, '__len__') else int(reward_i)))
            folder_name = f"{game}_re{re_val}_{int(row_i):04d}"
        else:
            folder_name = f"reward_{row_i}"

        # ── asynchronous HDF5 save — state(env_map)  writer queue in   before text ──────────────
        if h5_writer is not None:
            h5_writer.write(folder_name, int(repeat_i), state)


def build_task_text(reward_i, feature) -> str:
    labels = {1: f"RG: {int(feature[0])} | ",
              2: f"PL: {int(feature[1])} | ",
              3: f"WC: {int(feature[2])} | ",
              4: f"BC: {int(feature[3])} | ",
              5: "BD | "}
    return "".join(v for k, v in labels.items() if k in reward_i)

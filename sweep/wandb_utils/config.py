"""W&B 다운로더에서 사용하는 설정 상수들."""

import os

# config에서 중첩 딕셔너리를 펼칠 키 목록
FLATTEN_KEYS = ["encoder"]

# config에서 제거할 키 목록
REMOVE_KEYS = [
    "lr", "GAMMA", "is_3d", "n_eps", "agents", "n_bins", "n_gpus",
    "VF_COEF", "encoder", "CLIP_EPS", "ENT_COEF", "_img_dir", "_vid_dir",
    "arf_size", "env_name", "eval_dir", "n_agents", "vrf_size", "ANNEAL_LR",
    "act_shape", "ckpt_freq", "eval_freq", "eval_seed", "evaluator",
    "map_width", "num_steps", "overwrite", "pinpoints", "wandb_key",
    "GAE_LAMBDA", "_numpy_dir", "activation", "change_pct", "initialize",
    "multiagent", "n_freezies", "NUM_UPDATES", "empty_start", "hidden_dims",
    "n_eval_envs", "n_eval_maps", "profile_fps", "render_freq",
    "ctrl_metrics", "n_render_eps", "reward_every", "wandb_entity",
    "wandb_resume", "MAX_GRAD_NORM", "eval_map_path", "update_epochs",
    "wandb_project", "MINIBATCH_SIZE", "eval_map_types", "eval_map_width",
    "NUM_MINIBATCHES", "max_board_scans", "metrics_to_keep",
    "static_tile_prob", "current_iteration", "gif_frame_duration",
    "randomize_map_shape", "timestep_chunk_size", "eval_max_board_scans",
    "DEBUG", "eval_randomize_map_shape",
]

# W&B 엔티티 기본값 (.env 의 WANDB_ENTITY 우선)
DEFAULT_ENTITY = os.getenv("WANDB_ENTITY", "st4889ha-gwangju-institute-of-science-and-technology")

# W&B API 타임아웃 (초)
API_TIMEOUT = 600

# 병렬 처리 기본 워커 수
DEFAULT_NUM_WORKERS = 8


"""Configuration constants used by the W&B downloader."""

import os

# Keys whose nested dictionaries are flattened from config.
FLATTEN_KEYS = ["encoder"]

# Keys removed from config.
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

# Default W&B entity. Prefer WANDB_ENTITY from the environment.
DEFAULT_ENTITY = os.getenv("WANDB_ENTITY", "<wandb-entity>")

# W&B API timeout in seconds.
API_TIMEOUT = 600

# Default number of parallel workers.
DEFAULT_NUM_WORKERS = 8

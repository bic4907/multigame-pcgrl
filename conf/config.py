from typing import Iterable, List, Optional, Tuple, Union
from hydra.core.config_store import ConfigStore
from dataclasses import dataclass, field

from conf.game_utils import (                       # noqa: F401  — re-export
    GAME_ABBR, GAME_ABBR_INV, ALL_GAMES,
    parse_game_str, build_game_str,
)
from dataset.multigame.tile_utils import NUM_CATEGORIES

PREFIX = "aaai27_"

@dataclass
class Config:
    lr: float = 1.0e-4
    n_envs: int = 4
    # How many steps do I take in all of my batched environments before doing a gradient update
    num_steps: int = 128
    total_timesteps: int = int(2e7)
    timestep_chunk_size: int = -1
    update_epochs: int = 10
    NUM_MINIBATCHES: int = 4
    GAMMA: float = 0.99
    GAE_LAMBDA: float = 0.95
    CLIP_EPS: float = 0.2
    ENT_COEF: float = 0.01
    VF_COEF: float = 0.5
    MAX_GRAD_NORM: float = 0.5
    activation: str = "relu"
    env_name: str = "PCGRL"
    ANNEAL_LR: bool = False
    DEBUG: bool = False
    exp_name: str = "def"
    random_exp_name: bool = False
    seed: int = 0
    saves_dir: str = "saves"

    # Game selection — two-character abbreviations (dg=dungeon, pk=pokemon, sk=sokoban, dm=doom(+doom2), zd=zelda)
    # e.g. "dg" (dungeon only), "dgdm" (dungeon + doom + doom2), "all" (every game)
    game: str = "all"

    # Parsed from the game string automatically; kept for backward compatibility.
    include_dungeon: bool = True
    include_pokemon: bool = False
    include_sokoban: bool = False
    include_doom: bool = False
    include_doom2: bool = False
    include_zelda: bool = False

    problem: str = "dungeon3"
    representation: str = "turtle"
    model: str = "conv"

    # NLP params
    use_nlp: bool = False
    nlp_input_dim: int = 768

    # CLIP params
    use_clip: bool = False
    # Tile-only input channels = number of unified categories (NUM_CATEGORIES).
    # init_config() adds 2 coordinate channels, for a total of NUM_CATEGORIES + 2.
    clip_input_channel: int = NUM_CATEGORIES

    vec_cont: bool = False
    vec_input_dim: Optional[int] = None
    raw_obs: bool = False

    map_width: int = 16
    randomize_map_shape: bool = False
    is_3d: bool = False
    # ctrl_metrics: Tuple[str] = ('diameter', 'n_regions')
    ctrl_metrics: Tuple[str, ...] = ()
    # Size of the receptive field to be fed to the action subnetwork.
    vrf_size: Optional[
        int
    ] = -1  # -1 means 2 * map_width - 1, i.e. full observation, 31 if map_width=16
    # Size of the receptive field to be fed to the value subnetwork.
    arf_size: Optional[
        int
    ] = -1  # -1 means 2 * map_width - 1, i.e. full observation, 31 if map_width=16

    change_pct: float = -1.0

    # The shape of the (patch of) edit(s) to be made by the edited by the generator at each step.
    act_shape: Tuple[int, int] = (1, 1)

    static_tile_prob: Optional[float] = 0.0
    n_freezies: int = 0
    n_agents: int = 1  # multi-agent is fake and broken
    multiagent: bool = False
    max_board_scans: float = 3.0

    # How many milliseconds to wait between frames of the rendered gifs
    gif_frame_duration: int = 25

    # mutation rate initial map generation
    map_mutation_rate: float = 0.1

    # Resume training from a checkpoint elsewhere. If None, resolved from exp_dir/ckpts.
    init_ckpt_path: Optional[str] = None

    """ DO NOT USE. WILL BE OVERWRITTEN. """
    exp_dir: Optional[str] = None
    n_gpus: int = 1

    # use prev state
    use_prev: bool = True

    # normalize reward
    normal_weigth: float = 5

    # To make the task simpler, always start with an empty map
    empty_start: bool = False

    # In problems with tile-types with specified valid numbers, fix/freeze their random placement at the beginning of
    # each episode.
    pinpoints: bool = False

    hidden_dims: Tuple[int, ...] = (64, 256)

    reward_every: int = 1

    # A toggle, will add `n_envs` to the experiment name if we are profiling training FPS, so that we can distinguish
    # results.
    profile_fps: bool = False

    # NOTE: DO NOT MODIFY THESE. WILL BE SET AUTOMATICALLY AT RUNTIME. ########
    initialize: Optional[bool] = None

    # Wandb (set WANDB_API_KEY in .env or as an environment variable)
    wandb_key: Optional[str] = None
    wandb_project: Optional[str] = 'instruct_pcgrl'
    wandb_entity: Optional[str] = None
    wandb_resume: str = 'allow'
    evaluator: str = 'hr'  # 'vit', 'hr' (heuristic)

    #Ablation study options. default is 1.0
    text_ratio: float = 1.0
    state_ratio: float = 1.0

    # ── Action masking ────────────────────────────────────────────────────
    action_mask: bool = False
    re01_action_mask: bool = True

    exp_group: Optional[str] = None

    _vid_dir: Optional[str] = None
    _img_dir: Optional[str] = None
    _numpy_dir: Optional[str] = None
    _traj_dir: Optional[str] = None

    aug_type: str = "sub_condition"
    embed_type: str = "bert"

    instruct: Optional[str] = None
    instruct_csv: Optional[str] = None

    # MultiGameDataset-based filtering (for CPCGRL)
    dataset_game: Optional[str] = None          # e.g. "dungeon", "pokemon", "doom"
    dataset_reward_enum: Optional[Union[int, str]] = None   # int/list-string (e.g. 0, "01", "0,1") or "all"
    dataset_train_ratio: float = 0.95

    # Common data preprocessing (applied identically across every pipeline)
    longtail_cut: bool = True          # drop samples whose condition falls in the long tail
    max_samples_per_game: int = 1000   # per-game source_id cap (0 = unlimited)
    max_samples_seed: int = 42         # seed used when sampling down to max_samples_per_game
    rl_tile_offset: int = 1            # tile enum offset applied when converting to RL data

    # Multigame tile placement reward weight (sweep target)
    placement_w_amount: float = 1.0
    placement_w_spread: float = 0.0

    # Placement penalty weight for special tiles (interactive/hazard/collectable)
    special_tile_penalty_weight: float = 0.05

@dataclass
class CLIPConfig:
    freeze_text_enc: bool = True
    freeze_state_enc: bool = False
    use_map_array: bool = True
    token_max_len: int = 77


@dataclass
class EncoderConfig(CLIPConfig):
    model: Optional[str] = None
    state: bool = True
    mode: str = "text_state"

    deterministic: bool = True
    num_layers: int = 2
    hidden_dim: int = 256
    output_dim: int = 64

    dropout_rate: float = 0.3
    num_heads: int = 8
    buffer_ratio: float = 1

    ckpt_dir: str = "./encoder_ckpts"
    ckpt: Optional[str] = None
    ckpt_name: Optional[str] = None
    ckpt_path: Optional[str] = None
    trainable: bool = False
    tile_offset: int = 0               # tile enum offset applied to encoder inputs


@dataclass
class DecoderConfig:
    hidden_dim: int = 128
    num_layers: int = 2
    output_dim: int = 1
    num_reward_classes: int = 5
    # Whether the CNN decoder also receives the reward_enum one-hot.
    # If True, a (B, H, W, num_reward_classes) one-hot is concatenated to pixel_values.
    cnn_reward_enum_onehot: bool = False


@dataclass
class TrainConfig(Config):
    overwrite: bool = False
    ckpt_freq: int = int(3e6)
    render_freq: int = 50
    n_render_eps: int = 3
    eval_freq: int = 5000
    n_eval_maps: int = 6
    eval_map_path: str = "user_defined_freezies/binary_eval_maps.json"

    NUM_UPDATES: Optional[int] = None
    MINIBATCH_SIZE: Optional[int] = None

    agents: int = 1
    current_iteration: int = -1
    instruct_freq: int = 1
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    buffer_ratio: float = 1

    coef_human_sim: float = 0.0

    multimodal_condition: bool = False

    use_embedding_cache: bool = True

    # ── instruction prefix mode (train/eval/encoder common) ─────────────────
    # "name" (default): prefix with the game name, e.g. "In Zelda, ..."
    # "desc"          : prefix with a game description, e.g. "(traverse a room, fight creatures, ...) ..."
    # "none"/None     : no prefix
    # Must match the value used at encoder training time, so that RL training and
    # evaluation both see the same embeddings.
    instruction_prefix: Optional[str] = "name"

    # ── instruction field select (train/eval/encoder common) ──────────────────
    # "uni": use instruction_uni (unified wording)
    # "raw" (default): use instruction_raw (game-specific wording)
    instruction_field: str = "raw"


@dataclass
class CPCGRLConfig(TrainConfig):
    problem: str = "multigame"

    game: str = "all"

    dataset_game: Optional[str] = "all"
    dataset_reward_enum: Optional[Union[int, str]] = 0        # int/list-string (e.g. 0, "01", "0,1") or "all"
    dataset_train_ratio: float = 0.95
    # Condition-value filter: "enum_{i}_min_{v}" / "enum_{i}_max_{v}" / "enum_{i}_min_{lo}_max_{hi}"
    # Multiple filters are comma-separated: "enum_0_min_3_max_10,enum_2_max_50"
    dataset_condition_filter: Optional[str] = None

    vec_cont: bool = True
    raw_obs: bool = True
    model: str = "contconv"
    use_nlp: bool = False
    use_clip: bool = False
    vec_input_dim: Optional[int] = 5
    nlp_input_dim: int = 0

    instruct: Optional[str] = None
    instruct_csv: Optional[str] = None
    aug_type: str = "sub_condition"
    embed_type: str = "bert"

    encoder: EncoderConfig = field(default_factory=EncoderConfig)

    wandb_project: Optional[str] = "cpcgrl"


@dataclass
class IPCGRLConfig(CPCGRLConfig):
    """IPCGRL (Instructed PCGRL) — BERT embedding passed through an MLP encoder."""
    use_nlp: bool = True
    vec_cont: bool = False
    model: str = "nlpconv"
    nlp_input_dim: int = 768

    # ── Task variant marker ──
    # IPCGRL and MIPCGRL both use use_nlp=True and encoder='mlp', so path_utils cannot
    # tell them apart. This flag keeps their exp_dir and wandb names distinct.
    # (MIPCGRLConfig overrides it to True.)
    is_mipcgrl: bool = False

    encoder: EncoderConfig = field(default_factory=lambda: EncoderConfig(model="mlp"))

    wandb_project: Optional[str] = f'{PREFIX}train_ipcgrl'

    # ── Encoder unseen-experiment settings (shared with reward / vipcgrl) ────────
    # seen_ratio used at encoder training time — injected from dataset_setting.json.
    # 1.0 = all seen-game data (default); 0.0-1.0 = that leading fraction of it.
    dataset_seen_ratio: float = 1.0

    # unseen_ratio used at encoder training time — injected from dataset_setting.json.
    # None (default) keeps the existing behaviour (per-game ratio filtering disabled).
    dataset_unseen_ratio: Optional[float] = None

    # Games seen at encoder training time — injected from dataset_setting.json as full
    # names, e.g. ["dungeon", "doom", "zelda"]. Written to train_setting.json and logged
    # to WandB so the seen/unseen split can be recovered afterwards.
    reward_seen_games: List[str] = field(default_factory=list)


@dataclass
class VIPCGRLConfig(CPCGRLConfig):
    use_clip: bool = True
    model: str = "cnnclipconv"
    encoder: EncoderConfig = field(default_factory=lambda: EncoderConfig(model="cnnclip"))

    use_nlp: bool = False
    vec_cont: bool = False
    nlp_input_dim: int = 64  # encoder.output_dim (pretrained CLIP latent space)

    # coef_human_sim > 0 enables the human_demo similarity reward (0 disables it)
    coef_human_sim: float = 30.0

    wandb_project: Optional[str] = f"{PREFIX}train_vipcgrl"

    ignore_checkpoint: bool = False

    # ── Encoder unseen-experiment settings (shared with reward) ─────────────────
    # seen_ratio used at encoder training time — injected from dataset_setting.json.
    # 1.0 = all seen-game data (default); 0.0-1.0 = that leading fraction of it.
    dataset_seen_ratio: float = 1.0

    # unseen_ratio used at encoder training time — injected from dataset_setting.json.
    # None (default) keeps the existing behaviour (per-game ratio filtering disabled).
    # For VIPCGRL: 0.0 loads no unseen-game data; 0.0-1.0 loads that leading fraction.
    # ReWARD always injects 1.0, so all unseen-game data is loaded.
    dataset_unseen_ratio: Optional[float] = None

    # ── game_setting_mode: which games RL training covers ──
    # "all"          : every game
    # "encoder_seen" : only the games seen at encoder training time
    #                  (default; resolved from dataset_setting.json)
    game_setting_mode: str = "encoder_seen"

    # Games seen at encoder training time — injected from dataset_setting.json as full
    # names, e.g. ["dungeon", "doom", "zelda"]. Written to train_setting.json and logged
    # to WandB so the seen/unseen split can be recovered afterwards.
    reward_seen_games: List[str] = field(default_factory=list)



@dataclass
class ReWARDConfig(VIPCGRLConfig):
    wandb_project: Optional[str] = f"{PREFIX}train_reward"

    # ReWARD: clip_decoder-based dynamic reward shaping (reward_i / condition)
    use_decoder_reward_shaping: bool = True

    # The similarity reward is available but off by default; enable it explicitly in a config.
    coef_human_sim: float = 0.0

    decoder: DecoderConfig = field(default_factory=DecoderConfig)

    game_setting_mode: str = "all"

    # ── reward_decoder_mode: where the reward condition comes from (ReWARD only) ──
    # "noop"  : use the dataset metadata as-is for every game (no decoder)
    # "all"   : use the CLIP decoder prediction for every game
    # "unseen": dataset metadata for seen games, decoder prediction for unseen games
    reward_decoder_mode: str = "unseen"

    # ── Parameters used to resolve the encoder checkpoint path ───────────────────
    # Unseen-game abbreviation used at encoder training time (e.g. "zd", "zddm")
    train_unseen_abbr: Optional[str] = None
    # Unseen-game data ratio used at encoder training time (0.0 - 1.0)
    train_unseen_ratio: Optional[float] = None
    # Seen-game data ratio used at encoder training time (0.0 - 1.0)
    train_seen_ratio: Optional[float] = None

    # ── reward_unseen_ratio: metadata/decoder split within the unseen games ──────
    # Injected from the unseen_ratio in dataset_setting.json. Each unseen game's samples
    # are split by order:
    #   leading `reward_unseen_ratio` → metadata (ground-truth condition, encoder training data)
    #   remaining (1 - reward_unseen_ratio) → condition predicted by the reward decoder
    # 0.0 (default) applies the decoder to every unseen sample (zero-shot).
    reward_unseen_ratio: float = 0.0

    # ── delta_weight used at encoder training time (logged to wandb for reference) ──
    # Injected from encoder_config.json. 0.0 = baseline (direction alignment only).
    encoder_delta_weight: float = 0.0

    # ReWARD: fraction of unseen-game data to load (default 1.0 = all).
    # Overridable from the CLI; any value other than 1.0 appends a '_uro-XX' suffix
    # to the exp_dir name.
    dataset_unseen_ratio: float = 1.0


@dataclass
class PretrainedCLIPPCGRLConfig(CPCGRLConfig):
    use_clip: bool = True
    model: str = "pretrained_clip"

    encoder: EncoderConfig = field(default_factory=lambda: EncoderConfig(model="clip"))

    use_nlp: bool = False
    vec_cont: bool = False
    nlp_input_dim: int = 512  # encoder.output_dim (pretrained CLIP latent space, no projection)

    # HuggingFace CLIP expects RGB images (3 channels); render_level_from_arr produces them.
    clip_input_channel: int = 3

    use_pretrained_clip_reward: bool = True
    wandb_project: Optional[str] = f'{PREFIX}train_pretrained_clip_pcgrl'

    # Games used for training — derived from config.game and written to train_setting.json
    # as full names, e.g. ["dungeon", "doom", "zelda"]. Logged to WandB so the seen/unseen
    # split can be recovered afterwards.
    reward_seen_games: List[str] = field(default_factory=list)


@dataclass
class FinetunedCLIPPCGRLConfig(PretrainedCLIPPCGRLConfig):
    """Fine-tuned CLIP reward based PCGRL training Config.

    Reuses the observation and model structure of PretrainedCLIPPCGRLConfig, but injects a
    fine-tuned CLIP checkpoint into the RL subtree via `encoder.ckpt_name` (or `ckpt_path`).
    The existing `apply_encoder_params` path is used unchanged.
    """
    wandb_project: Optional[str] = f"{PREFIX}train_finetuned_clip_pcgrl"
    dir_prefix: str = "finetuned-clip-pcgrl-"

    # ── Model selector for RL on top of a fine-tuned CLIP ────────────────────────
    # The parameter tree matches pretrained_clip, and the checkpoint built by
    # `get_finetuned_clip_encoder` shares its trainable-parameter layout
    # (TrainablePretrained*Encoder). A separate model name keeps exp_dir and the
    # encoder hash distinct.
    model: str = "finetuned_clip"

    # ── Encoder unseen-experiment settings (shared with reward / vipcgrl) ──
    dataset_seen_ratio: float = 1.0

    # Unseen-game data ratio used at encoder training time (injected from dataset_setting.json).
    # None keeps the existing behaviour (dataset_seen_ratio applied to every game).
    dataset_unseen_ratio: Optional[float] = None

    reward_seen_games: List[str] = field(default_factory=list)
    game_setting_mode: str = "all"


@dataclass
class EvalConfig(TrainConfig):
    reevaluate: bool = False

    random_agent: bool = False
    n_bins: int = 10
    n_eval_envs: int = 10
    n_eps: int = 10
    eval_exp_name: Optional[str] = None
    eval_map_width: Optional[int] = None
    eval_max_board_scans: Optional[int] = None
    eval_randomize_map_shape: Optional[bool] = None
    eval_seed: int = 0

    # Upload eval.h5 (per-sample rollouts) as a WandB artifact at the end of
    # eval. When False, the local eval.h5 file is also deleted after eval to
    # save disk space (only aggregate metrics are kept).
    upload_h5: bool = False

    eval_aug_type: str = "sub_condition"
    eval_embed_type: str = "bert"
    eval_instruct: str = "scn-1_se-whole"
    eval_instruct_csv: Optional[str] = None
    eval_dir: Optional[str] = None
    eval_map_types: int = 5
    eval_modality: str = "text"
    eval_human_demo_path: str = './human_dataset'

    diversity: bool = True
    vit_score: bool = True
    vit_normalize: bool = False
    tpkldiv: bool = True

    wandb_project: str = 'eval_pcgrl'

    metrics_to_keep: Tuple[str] = ("mean_ep_reward",)
    flush: bool = True

    problem: str = "multigame"



@dataclass
class RandomEvalConfig(EvalConfig):
    """Config for the random baseline evaluation.

    Ignores the network and samples uniformly random actions. The exp_dir name starts
    with "random_" to keep it distinct from the cpcgrl_ runs.
    """

    random_agent: bool = True
    dir_prefix: str = "random_"
    wandb_project: Optional[str] = f"{PREFIX}eval_random"

    dataset_reward_enum: Optional[Union[int, str]] = 0        # int/list-string (e.g. 0, "01", "0,1") or "all"
    eval_games: str = 'all'

    # Number of evaluation samples per (game, reward_enum) group. None uses all of them.
    eval_samples_per_group: Optional[int] = 200

    # reward_enums to evaluate. None falls back to dataset_reward_enum.
    # May be given as a digit string: "12" → [1, 2], "012" → [0, 1, 2]
    # A list is also accepted: [0, 1, 2]
    eval_dataset_reward_enums: Optional[str] = None



@dataclass
class CPCGRLEvalConfig(EvalConfig):
    """Config for CPCGRL evaluation.

    Mirrors the observation and model settings of CPCGRLConfig on top of EvalConfig.
    """
    problem: str = "multigame"

    # ── Same game / dataset defaults as CPCGRLConfig, so the exp_dir name matches ──
    game: str = "all"
    dataset_game: Optional[str] = "all"
    dataset_reward_enum: Optional[int] = 0        # 0=region
    dataset_train_ratio: float = 0.95

    # Games to evaluate on (None means the same as `game`). The checkpoint is resolved from
    # `game`, while the evaluation data comes from `eval_games`. For example, to evaluate a
    # model trained with game="all" on a single game, set eval_games="dg".
    eval_games: str = 'all'

    vec_cont: bool = True
    raw_obs: bool = True
    model: str = "contconv"
    use_nlp: bool = False
    use_clip: bool = False
    vec_input_dim: Optional[int] = 5
    nlp_input_dim: int = 0

    max_samples: Optional[int] = None  # cap the sample count for dry runs (None = no cap)

    # Number of evaluation samples per (game, reward_enum) group. None uses all of them.
    eval_samples_per_group: Optional[int] = 200

    # reward_enums to evaluate. None falls back to dataset_reward_enum.
    # May be given as a digit string: "12" → [1, 2], "012" → [0, 1, 2]
    # A list is also accepted: [0, 1, 2]
    eval_dataset_reward_enums: Optional[str] = None

    # If True, run even when no checkpoint exists (logs a warning). If False (default),
    # a missing checkpoint raises an error.
    ignore_checkpoint: bool = False


    wandb_project: Optional[str] = f"{PREFIX}eval_cpcgrl"

@dataclass
class VIPCGRLEvalConfig(CPCGRLEvalConfig):
    """Config for VIPCGRL evaluation.

    Injects the pretrained CLIP embedding into nlp_obs. No decoder reward shaping is
    applied — only the CLIP embedding is used.
    """
    wandb_project: Optional[str] = f"{PREFIX}eval_vipcgrl"

    encoder: EncoderConfig = field(default_factory=lambda: EncoderConfig(model="cnnclip"))

    use_clip: bool = True
    vec_cont: bool = False
    model: str = "cnnclipconv"
    use_nlp: bool = False
    nlp_input_dim: int = 64  # encoder.output_dim (pretrained CLIP latent space)

    ignore_checkpoint: bool = False

    # ── Encoder unseen-experiment settings (same as reward eval) ────────────────
    # seen_ratio used at encoder training time — injected from dataset_setting.json.
    # Used only to resolve the checkpoint path; it does not filter the eval dataset.
    train_seen_ratio: float = 1.0

    # Seen/unseen game lists from training — injected from dataset_setting.json.
    seen_games: List[str] = field(default_factory=list)
    unseen_games: List[str] = field(default_factory=list)

    # ── game_setting_mode: which games evaluation covers ──
    # Must match the train_vipcgrl default (encoder_seen) so that exp_dir resolves.
    game_setting_mode: str = "encoder_seen"


@dataclass
class PretrainedCLIPEvalConfig(CPCGRLEvalConfig):
    """Config for PretrainedCLIP PCGRL evaluation.

    Evaluates a checkpoint trained by train_pretrained_clip.py. The precomputed CLIP text
    embedding is injected into nlp_obs; no separate encoder checkpoint is loaded, so the
    stock CLIP vision tower is used.
    """
    wandb_project: Optional[str] = f"{PREFIX}eval_pretrained_clip"

    encoder: EncoderConfig = field(default_factory=lambda: EncoderConfig(model="clip"))

    use_clip: bool = True
    vec_cont: bool = False
    model: str = "pretrained_clip"
    use_nlp: bool = False
    nlp_input_dim: int = 512  # pretrained CLIP text embedding dimension (no projection)

    ignore_checkpoint: bool = False

    # Seen/unseen game lists from training — injected from train_setting.json.
    seen_games: List[str] = field(default_factory=list)
    unseen_games: List[str] = field(default_factory=list)


@dataclass
class FinetunedCLIPEvalConfig(PretrainedCLIPEvalConfig):
    """Config for fine-tuned CLIP PCGRL evaluation."""
    wandb_project: Optional[str] = f"{PREFIX}eval_finetuned_clip"
    dir_prefix: str = "finetuned-clip-pcgrl-"
    model: str = "finetuned_clip"


@dataclass
class ReWARDEvalConfig(CPCGRLEvalConfig):
    """Config for ReWARD evaluation.

    Mirrors the observation and model settings of CPCGRLConfig on top of EvalConfig.
    """
    wandb_project: Optional[str] = f"{PREFIX}eval_reward"

    use_decoder_reward_shaping: bool = True

    encoder: EncoderConfig = field(default_factory=lambda: EncoderConfig(model="cnnclip"))
    decoder: DecoderConfig = field(default_factory=DecoderConfig)

    use_clip: bool = True
    nlp_input_dim: int = 64  # encoder.output_dim (pretrained CLIP latent space)

    ignore_checkpoint: bool = False

    # seen_ratio used at encoder training time — injected from dataset_setting.json.
    # Used only to resolve the checkpoint path; it does not filter the eval dataset.
    train_seen_ratio: float = 1.0

    # Kept at the training default ("unseen") so the exp_dir path resolves. The condition
    # source actually used at evaluation is controlled by eval_reward_decoder_mode.
    reward_decoder_mode: str = "unseen"

    # Condition source used during evaluation.
    # "noop"   → ground-truth condition (default; measures generation quality directly)
    # "unseen" → decoder prediction for unseen games
    eval_reward_decoder_mode: str = "noop"

    # Seen/unseen game lists from training — injected from reward_decoder_config.json.
    seen_games: List[str] = field(default_factory=list)
    unseen_games: List[str] = field(default_factory=list)


    # ── Parameters used to resolve the encoder checkpoint path ───────────────────
    # Must match ReWARDConfig at training time for exp_dir to line up.
    train_unseen_abbr: Optional[str] = None
    train_unseen_ratio: Optional[float] = None
    train_seen_ratio: Optional[float] = None

    # ── delta_weight used at encoder training time (logged to wandb for reference) ──
    encoder_delta_weight: float = 0.0

    # Must match training for exp_dir to line up. Any value other than 1.0 adds a
    # '_uro-XX' suffix.
    dataset_unseen_ratio: float = 1.0


@dataclass
class IPCGRLEvalConfig(CPCGRLEvalConfig):
    """Config for IPCGRL evaluation.

    Extends CPCGRLEvalConfig with the BERT-embedding + MLP encoder settings.
    """
    use_nlp: bool = True
    vec_cont: bool = False
    model: str = "nlpconv"
    nlp_input_dim: int = 768

    encoder: EncoderConfig = field(default_factory=lambda: EncoderConfig(model="mlp"))

    dataset_reward_enum: Optional[int] = None

    wandb_project: Optional[str] = f"{PREFIX}eval_ipcgrl"

    # ── seen_ratio used during encoder training (analysis only) ─────────────────
    train_seen_ratio: float = 1.0

    # ── Seen/unseen game lists from training — injected from dataset_setting.json ──
    seen_games: List[str] = field(default_factory=list)
    unseen_games: List[str] = field(default_factory=list)

    # Train-side name used by path_utils/dataset helpers. Eval fills this with
    # the same encoder seen-game metadata so exp_dir matches the trained run.
    reward_seen_games: List[str] = field(default_factory=list)

    # ── MIPCGRL variant marker (same role as in IPCGRLConfig) ──
    is_mipcgrl: bool = False


@dataclass
class MIPCGRLEvalConfig(IPCGRLEvalConfig):
    """Config for MIPCGRL evaluation — same as IPCGRLEvalConfig with is_mipcgrl=True."""
    wandb_project: Optional[str] = f"{PREFIX}eval_mipcgrl"
    is_mipcgrl: bool = True


@dataclass
class CollectBufferConfig(CPCGRLConfig):
    """Config for collecting trajectories during training.

    Collects data from a single environment (env_idx=0) over the
    collect_start_ratio-collect_end_ratio window of training, and writes it as .npz files
    into the buffer/ directory of the experiment folder.
    """
    wandb_project: str = 'collect_buffer'
    dir_prefix: str = "buffer-"

    # ── Collection parameters ──
    buffer_max_samples: int = 10_000        # maximum number of transitions to keep
    collect_start_ratio: float = 0.5        # start of the window (0.5 = 50% into training)
    collect_end_ratio: float = 1.0          # end of the window (1.0 = end of training)
    buffer_save_dir: Optional[str] = None   # output path (None = exp_dir/buffer)

    # Store env_map in each transition (increases file size)
    collect_env_map: bool = True


@dataclass
class BertConfig(Config):

    overwrite: bool = True

    num_samples: int = 100
    batch_size: int = 32
    offline: bool = True
    pretrained_model: str = 'bert'
    model_size: str = "base"
    buffer_path: str = "/mnt/nas/instructed_rl/pcgrl_buffer"
    dataset_path: str = "/mnt/nas/instructed_rl/pcgrl_normalized_dataset"
    fine_tune: bool = False

    deterministic: bool = True
    hidden_dims: int = 512
    num_layers: int = 1  # 1 ~ 3
    output_dim: int = 512
    num_heads: int = 8  # 2, 4, 8, 16, 32, 64 etc

    # decoder parameters
    decoder_hidden_dims: int = 512
    decoder_num_layers: int = 1
    decoder_output_dim: int = 1

    buffer_ratio: float = 1
    instruct: str = "scn-1_se-whole"


@dataclass
class BertTrainConfig(BertConfig):
    wandb_project: str = 'embedding'

    max_length: int = 128

    batch_size: int = 512
    lr: float = 0.001
    n_epochs: int = 100
    n_buffer: int = -1
    use_prev: bool = False

    encoder: EncoderConfig = field(default_factory=EncoderConfig)

@dataclass
class BertEvalConfig(BertConfig):
    wandb_project: str = 'eval_bert'
    use_prev: bool = False
    buffer_ratio: float = 1

    encoder: EncoderConfig = field(default_factory=EncoderConfig)


@dataclass
class RewardConfig(Config):
    dir_prefix: str = "encoder-"
    overwrite: bool = True
    n_max_points: int = 1000
    embed_visualize_freq: int = 5

    num_samples: int = 100
    batch_size: int = 512

    num_layers: int = 2  # 1 ~ 3
    hidden_dim: int = 512
    output_dim: int = 1

    figure_dir: str = "figures"
    buffer_dir: str = "./dataset"
    buffer_raio: float = 1.0
    train_ratio: float = 0.95
    n_epochs: int = 100

    dropout_rate: float = 0.0
    broadcast_dropout: bool = False # Use a broadcasted dropout along batch dims.
    weight_decay: float = 1e-4
    normal_weigth: float = 5

    augment: bool = True
    zero_reward_ratio: Optional[float] = None
    buffer_ratio: float = 1

    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)

    deterministic: bool = True
    use_prev: bool = False

    steps_per_epoch: Optional[int] = None
    warmup_epochs: int = 10  # set 10% of the total timesteps

    max_samples: Optional[int] = None  # cap the sample count for dry runs (None = no cap)



@dataclass
class RewardTrainConfig(RewardConfig):
    wandb_project: str = f'{PREFIX}train_ipcgrl_encoder'

    pretrained_model: str = "bert"
    model_size: str = "base"

    max_len: int = 77

    n_envs: int = 300
    ckpt_freq: int = 5


@dataclass
class CLIPTrainConfig(Config):
    exp_name: str = "def"

    wandb_project: str = f"{PREFIX}train_vipcgrl_encoder"
    seed: int = 0

    overwrite: bool = False
    ckpt_freq: int = int(50)
    ckpt_keep: int = 2

    # Goal img path
    img_data_path: str = "./human_dataset"
    instruct: str = "scn-1_se-whole"

    n_max_points: int = 1000
    embed_visualize_freq: int = 500

    n_epochs: int = 3000
    lr: float = 1.0e-3
    weight_decay: float = 1e-5
    train_ratio: float = 0.99
    batch_size: int = 2048
    buffer_ratio: float = 1.0 # Not implemented for clip yet.
    train_shuffle: bool = False

    dir_prefix: str = "clip-"
    figure_dir: str = "figures"

    steps_per_epoch: Optional[int] = None
    max_samples: Optional[int] = None  # cap the sample count for dry runs (None = no cap)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)

    # Instruction prefix mode: "name" (e.g. "In Zelda, ...") / "desc" / "mix" / "none" (or None)
    instruction_prefix: Optional[str] = "name"

    # instruction field select: "uni" / "raw" (default)
    instruction_field: str = "raw"

    # overwrite
    embed_type: str = "humanai"

    # ── Seen/unseen game split (same semantics as CLIPDecoderTrainConfig) ──
    # Unseen games as two-character abbreviations, e.g. "zd"=zelda, "pkzd"=pokemon+zelda.
    # None/"" keeps the previous behaviour (every game split by train/test ratio).
    unseen_games: Optional[str] = None
    # Few-shot ratio: fraction of unseen-game data used during training
    # (0.0 = zero-shot, 1.0 = all of it)
    unseen_ratio: float = 0.0
    # Fraction of seen-game data used (1.0 = all of it)
    seen_ratio: float = 1.0
    # Seed for the seen/unseen split
    split_seed: int = 42

@dataclass
class FinetunedCLIPEncoderTrainConfig(CLIPTrainConfig):
    """Config for fine-tuning a pretrained HuggingFace CLIP on (image, text) pairs.

    The parameter tree matches `pretrained_clip_model.ContrastiveModule`, so the saved
    checkpoint can be injected into the RL pipeline via `apply_encoder_params` unchanged.
    """
    wandb_project: str = f"{PREFIX}train_finetuned_clip_encoder"
    dir_prefix: str = "finetuned-clip-"

    # HF CLIP expects 224x224 RGB input, so coordinate channels are disabled.
    clip_input_channel: int = 3

    # Encoder settings used for the path / experiment name. RL loads this as 'clip'.
    encoder: EncoderConfig = field(
        default_factory=lambda: EncoderConfig(model="clip", state=True)
    )

    # Fine-tuning the full HF CLIP is unstable at higher learning rates, so keep the rate
    # low and the epoch count modest to limit catastrophic forgetting and overfitting.
    lr: float = 5.0e-6
    weight_decay: float = 0.1
    n_epochs: int = 100
    batch_size: int = 128
    ckpt_freq: int = 50

    embed_type: str = "finetuned_clip"

    instruction_prefix: Optional[str] = "name"



@dataclass
class CLIPEvalConfig(EvalConfig):
    eval_aug_type: str = "test"
    embed_type: str = 'clip'
    eval_embed_type: str = "clip"
    model: str = "cnnclipconv"
    state: bool = True

    wandb_project: str = 'eval_clip_pcgrl'
    encoder: EncoderConfig = field(default_factory=EncoderConfig)

@dataclass
class CLIPDecoderTrainConfig(CLIPTrainConfig):
    """Config for training the CLIP encoder together with the reward decoder.

    Adds a decoder branch on top of the existing contrastive loss, predicting the
    reward_enum (classification) and the condition value (regression) from the state
    embedding.
    """
    wandb_project: str = f'{PREFIX}train_reward_encoder'
    dir_prefix: str = "clipdec-"

    # ── Decoder config ──
    decoder: DecoderConfig = field(default_factory=DecoderConfig)

    # ── loss weight ──
    contrastive_weight: float = 1.0    # contrastive loss weight
    cls_weight: float = 1.0            # reward_enum classification loss weight
    reg_weight: float = 1.0            # condition regression loss weight

    # ── Continuous Task-wise Cross-game Direction Alignment Loss ──
    # Regularizer aligning, across games, the embedding direction along which the
    # condition value increases within the same task. 0.0 disables it (baseline).
    delta_weight: float = 0.03
    delta_min_group_samples: int = 2   # minimum samples per (game, task) group
    delta_var_eps: float = 1e-4        # condition-variance floor; groups below it are skipped
    compute_delta_when_zero: bool = True  # still compute the alignment metric when delta_weight=0

    # ── Regression loss ──
    # "huber": Huber loss (δ=1.0), "mae": Mean Absolute Error
    regression_loss: str = "mae"

    # ── Seen/Unseen game separate config ──
    # Unseen games as two-character abbreviations, e.g. "zd"=zelda, "pkzd"=pokemon+zelda.
    # None means every game is seen.
    unseen_games: Optional[str] = None
    # Few-shot ratio: fraction of unseen-game data used during training
    # (0.0 = zero-shot, 1.0 = all of it)
    unseen_ratio: float = 0.0
    # Fraction of seen-game data used (1.0 = all of it)
    seen_ratio: float = 1.0
    # Seed for the seen/unseen split
    split_seed: int = 42

    n_epochs: int = 3000

    # ── Step-based checkpoint / evaluation intervals ──
    ckpt_freq: int = 1000    # checkpoint interval in steps (0 disables)
    scatter_freq: int = 500  # scatter-plot upload interval in epochs (0 or None disables)

    # ── Unseen-game evaluation intervals ──
    unseen_eval_freq: int = 100     # unseen regression eval interval in epochs (0 disables)
    unseen_scatter_freq: int = 500  # unseen scatter-plot interval in epochs (0 disables)

    # ── Unseen evaluation data ratio ──
    # unseen_ratio      : fraction of unseen-game data mixed into training (train pool)
    # eval_unseen_ratio : fraction of the unseen test set used by the periodic evaluation
    #                     (0.0-1.0, 1.0 = all)
    eval_unseen_ratio: float = 1.0
    export_unseen_predictions_csv: bool = True

    # ── Gradient flow ──
    # True : stop the decoder loss (cls + reg) gradient from reaching the encoder latent space
    # False (default): the decoder loss also updates the encoder
    decoder_nograd: bool = False


@dataclass
class CLIPDecoderUnseenConfig(CLIPDecoderTrainConfig):
    """Seen/Unseen game separate + Few-shot Ratio Sweep Config.

    Trains the CLIP decoder on all seen-game data plus a small fraction of unseen-game
    data, then measures per-game reward_accuracy on a fixed test split.
    """
    wandb_project: str = 'train_clip_decoder_unseen'
    dir_prefix: str = "clipdec-"

    # ── Unseen games as two-character abbreviations, e.g. "zd"=zelda, "pkzd"=pokemon+zelda ──
    unseen_games: Optional[str] = None

    # ── Few-shot ratio ──
    # 0.0 = zero-shot (no unseen training data), 1.0 = all unseen training data
    unseen_ratio: float = 0.01

    # ── Seen game data ratio ──
    # 1.0 = all seen training data (default), 0.0 = none of it
    seen_ratio: float = 1.0

    # ── Split config ──
    # train_ratio: fraction used for training (overrides the CLIPTrainConfig default of 0.8)
    # test ratio = 1.0 - train_ratio
    train_ratio: float = 0.99
    split_seed: int = 42              # seed for the seen/unseen split


@dataclass
class CLIPDecoderUnseenSweepConfig(CLIPDecoderUnseenConfig):
    """Seen/Unseen game separate + Few-shot Ratio **Sweep** Config.

    Extends CLIPDecoderUnseenConfig by sweeping over several unseen_ratios values.
    Used by sweep/runnable_sweep/unseen_games.py.
    """
    # ── Few-shot ratio sweep values ──
    # 0.0 = zero-shot, 1.0 = all unseen training data
    unseen_ratios: Tuple[float, ...] = (0.0, 0.01, 0.03, 0.05, 0.1)


@dataclass
class IPCGRLEncoderMGConfig(RewardConfig):
    """Config for multi-game pretraining of the IPCGRL MLP encoder.

    Annotations come from MultiGameDataset.
    - input : BERT(instruction) -> 768-dim embedding
    - model : MLP encoder + MLP head
    - target: condition value (log1p + per-enum min-max normalisation)
    - unseen_games: games excluded from training, for zero-shot evaluation

    Usage:
        python train_ipcgrl_encoder_mg.py game=all
        python train_ipcgrl_encoder_mg.py game=all unseen_games=zd
    """
    wandb_project: Optional[str] = f"{PREFIX}train_ipcgrl_encoder"
    dir_prefix: str = "ipcgrl-enc-mg-"
    ckpt_freq: int = 10

    # BERT config
    use_nlp: bool = True
    nlp_input_dim: int = 768

    # Unseen games as two-character abbreviations, e.g. "zd"=zelda, "pkzd"=pokemon+zelda.
    # An empty string means no unseen games (train on all of them).
    unseen_games: str = ""

    # ── Seen/unseen data ratios (same semantics as CLIPTrainConfig) ──
    # unseen_ratio: fraction of unseen-game data used during training
    #               (0.0 = zero-shot, 1.0 = all of it)
    unseen_ratio: float = 0.0
    # seen_ratio: fraction of seen-game data used (1.0 = all of it)
    seen_ratio: float = 1.0

    # Annotation dataset config (same options as CLIPTrainConfig)
    # instruction_prefix mode: "name" (default) / "desc" / "none" (or None)
    instruction_prefix: Optional[str] = "name"

    # instruction field select: "uni" / "raw" (default)
    instruction_field: str = "raw"

    # MLP encoder (apply_encoder_model dispatches on model='mlp')
    encoder: EncoderConfig = field(default_factory=lambda: EncoderConfig(model="mlp"))


@dataclass
class MIPCGRLConfig(IPCGRLConfig):
    wandb_project: Optional[str] = f"{PREFIX}train_mipcgrl"
    is_mipcgrl: bool = True


@dataclass
class MIPCGRLEncoderMGConfig(IPCGRLEncoderMGConfig):
    """Config for multi-game pretraining of the MIPCGRL MLP encoder.

    Extends IPCGRL (which only regresses the condition value) with a second head that
    classifies the task (reward_enum) from the same latent z, trained jointly.
        Loss = MSE(condition) + classifier_weight * CrossEntropy(reward_enum)

    Usage:
        python train_mipcgrl_encoder_mg.py game=all
        python train_mipcgrl_encoder_mg.py game=all unseen_games=zd classifier_weight=0.5
    """
    wandb_project: Optional[str] = f"{PREFIX}train_mipcgrl_encoder"
    dir_prefix: str = "mipcgrl-enc-mg-"

    # ── Classifier config ──
    # Loss weight for the task (reward_enum) classification head. A value of 0 matches IPCGRL.
    classifier_weight: float = 1.0
    # Classifier MLP hidden size / depth (output_size is derived from num_classes)
    classifier_num_layers: int = 2
    classifier_hidden_dim: int = 128
    classifier_dropout_rate: float = 0.0
    # Number of classes. None derives it from the reward_enum count in the dataset.
    num_classes: Optional[int] = None


cs = ConfigStore.instance()
cs.store(name="config", node=Config)
cs.store(name="train_pcgrl", node=TrainConfig)
cs.store(name="cpcgrl", node=CPCGRLConfig)
cs.store(name="ipcgrl", node=IPCGRLConfig)
cs.store(name="mipcgrl", node=MIPCGRLConfig)
cs.store(name="vipcgrl", node=VIPCGRLConfig)
cs.store(name="reward", node=ReWARDConfig)
cs.store(name="pretrained_clip_pcgrl", node=PretrainedCLIPPCGRLConfig)
cs.store(name="finetuned_clip_pcgrl_schema", node=FinetunedCLIPPCGRLConfig)
cs.store(name="eval_pcgrl", node=EvalConfig)
cs.store(name="eval_random_schema", node=RandomEvalConfig)
cs.store(name="eval_cpcgrl_schema", node=CPCGRLEvalConfig)
cs.store(name="eval_ipcgrl_schema", node=IPCGRLEvalConfig)
cs.store(name="eval_mipcgrl_schema", node=MIPCGRLEvalConfig)
cs.store(name="eval_vipcgrl_schema", node=VIPCGRLEvalConfig)
cs.store(name="eval_reward_schema", node=ReWARDEvalConfig)
cs.store(name="eval_pretrained_clip_schema", node=PretrainedCLIPEvalConfig)
cs.store(name="eval_finetuned_clip_schema", node=FinetunedCLIPEvalConfig)
cs.store(name="eval_ipcgrl_schema", node=IPCGRLEvalConfig)
cs.store(name="collect_buffer_schema", node=CollectBufferConfig)

# CLIP PCGRL Configs
cs.store(name="train_clip", node=CLIPTrainConfig)
cs.store(name="train_finetuned_clip_encoder_schema", node=FinetunedCLIPEncoderTrainConfig)
cs.store(name="train_clip_decoder_schema", node=CLIPDecoderTrainConfig)
cs.store(name="train_clip_decoder_unseen_schema", node=CLIPDecoderUnseenConfig)
cs.store(name="train_clip_decoder_unseen_sweep_schema", node=CLIPDecoderUnseenSweepConfig)

cs.store(name="train_bert", node=BertTrainConfig)
cs.store(name="eval_bert", node=BertEvalConfig)

cs.store(name="train_reward_model", node=RewardTrainConfig)
cs.store(name="train_ipcgrl_encoder_mg_schema", node=IPCGRLEncoderMGConfig)
cs.store(name="train_mipcgrl_encoder_mg_schema", node=MIPCGRLEncoderMGConfig)

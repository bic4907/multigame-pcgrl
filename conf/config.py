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

    # Game selection — 2text abbreviation text (dg=dungeon, pk=pokemon, sk=sokoban, dm=doom(+doom2), zd=zelda)
    # text: "dg" (dungeontext), "dgdm" (dungeon+doom+doom2), "all" (all)
    game: str = "all"

    # include_* text  game string in  automatic parsingtext (sub text for  as  keep)
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
    # tile-only text text = unified category text (NUM_CATEGORIES).
    # init_config() in  coord text 2text  text total NUM_CATEGORIES+2   text text text  text.
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

    # different path of  checkpoint in  training  starttext text text. None text exp_dir/ckpts in  automatic text.
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

    # common data preprocessing (text pipeline in  sametext apply)
    longtail_cut: bool = True          # text condition text sample remove
    max_samples_per_game: int = 1000   # gametext source_id text (0=text)
    max_samples_seed: int = 42         # max_samples_per_game textsampletext seed
    rl_tile_offset: int = 1            # tile enum text in  text text (RL data to text for )

    # Multigame tile placement reward weight (sweep target)
    placement_w_amount: float = 1.0
    placement_w_spread: float = 0.0

    # Special tile (interactive/hazard/collectable) text penalty weight
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
    tile_offset: int = 0               # tile enum text in  text text (text)


@dataclass
class DecoderConfig:
    hidden_dim: int = 128
    num_layers: int = 2
    output_dim: int = 1
    num_reward_classes: int = 5
    # CNN text in  reward_enum one-hot text  text text text
    # True text pixel_values in  (B, H, W, num_reward_classes) one-hot  concat
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
    # "name" (default): "In Zelda, ..." text  game name prefix
    # "desc"      : "(traverse a room, fight creatures, ...) ..." text  game text prefix
    # "none"/None : prefix textapply
    # text training text text for text text and  RL training/evaluation in  sametext embedding  text.
    instruction_prefix: Optional[str] = "name"

    # ── instruction field select (train/eval/encoder common) ──────────────────
    # "uni": instruction_uni text for  (text tabletext)
    # "raw" (default): instruction_raw text for  (gametext text tabletext)
    instruction_field: str = "raw"


@dataclass
class CPCGRLConfig(TrainConfig):
    problem: str = "multigame"

    game: str = "all"

    dataset_game: Optional[str] = "all"
    dataset_reward_enum: Optional[Union[int, str]] = 0        # int/list-string (e.g. 0, "01", "0,1") or "all"
    dataset_train_ratio: float = 0.95
    # condition text based filter: "enum_{i}_min_{v}" / "enum_{i}_max_{v}" / "enum_{i}_min_{lo}_max_{hi}"
    # text filter  texttable text: "enum_0_min_3_max_10,enum_2_max_50"
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
    """IPCGRL (Instructed PCGRL) — BERT embedding → MLP text."""
    use_nlp: bool = True
    vec_cont: bool = False
    model: str = "nlpconv"
    nlp_input_dim: int = 768

    # ── Task variant marker ──
    # IPCGRL/MIPCGRL text use_nlp=True, encoder=mlp text RL text path_utils  in
    # same text  text text. text baseline  of  exp_dir / wandb name   separatetext abovetext
    #   text  text for text. (MIPCGRLConfig  in  True  to  override)
    is_mipcgrl: bool = False

    encoder: EncoderConfig = field(default_factory=lambda: EncoderConfig(model="mlp"))

    wandb_project: Optional[str] = f'{PREFIX}train_ipcgrl'

    # ── encoder unseen experiment text (mgpcgrl/vipcgrl  and  same text) ──────────────
    # encoder training text text for text seen_ratio — dataset_setting.json in  automatic injecttext.
    # 1.0 = all seen game data text for  (default value), 0.0~1.0 = seen game data prefix ratio
    dataset_seen_ratio: float = 1.0

    # encoder training text text for text unseen_ratio — dataset_setting.json in  automatic injecttext.
    # None(default value) = existing text keep (per-game ratio filtering disabled).
    dataset_unseen_ratio: Optional[float] = None

    # encoder training text seen game list — dataset_setting.json in  automatic injecttext.
    # (full name text, e.g. ["dungeon", "doom", "zelda"]). text text
    # train_setting.json in  seen/unseen split  writetext WandB  to text in  text for text.
    reward_seen_games: List[str] = field(default_factory=list)


@dataclass
class VIPCGRLConfig(CPCGRLConfig):
    use_clip: bool = True
    model: str = "cnnclipconv"
    encoder: EncoderConfig = field(default_factory=lambda: EncoderConfig(model="cnnclip"))

    use_nlp: bool = False
    vec_cont: bool = False
    nlp_input_dim: int = 64  # encoder.output_dim (pretrained CLIP latent space)

    # coef_human_sim > 0: human_demo sim_reward enable text text to  text for  (0 text disabled)
    coef_human_sim: float = 30.0

    wandb_project: Optional[str] = f"{PREFIX}train_vipcgrl"

    ignore_checkpoint: bool = False

    # ── encoder unseen experiment text (mgpcgrl  and  same text) ──────────────────────
    # encoder training text text for text seen_ratio — dataset_setting.json in  automatic injecttext.
    # 1.0 = all seen game data text for  (default value), 0.0~1.0 = seen game data prefix ratio
    dataset_seen_ratio: float = 1.0

    # encoder training text text for text unseen_ratio — dataset_setting.json in  automatic injecttext.
    # None(default value) = existing text keep (per-game ratio filtering disabled).
    # VIPCGRL in text text for : 0.0 = unseen game textload, 0.0~1.0 = unseen game prefix ratio.
    # MGPCGRL  always 1.0(full) as  injecttext text unseen game data  loadtext.
    dataset_unseen_ratio: Optional[float] = None

    # ── game_setting_mode: training in  text for text game range select ──
    # "all"          : all game text for
    # "encoder_seen" : encoder training text seen gametext text for  (default value, dataset_setting.json in  automatic text)
    game_setting_mode: str = "encoder_seen"

    # encoder training text seen game list — dataset_setting.json in  automatic injecttext.
    # (full name text, e.g. ["dungeon", "doom", "zelda"]). text text
    # train_setting.json in  seen/unseen split  writetext WandB  to text in  text for text.
    reward_seen_games: List[str] = field(default_factory=list)



@dataclass
class MGPCGRLConfig(VIPCGRLConfig):
    wandb_project: Optional[str] = f"{PREFIX}train_mgpcgrl"

    # MGPCGRL: clip_decoder based dynamic reward text (reward_i/condition)
    use_decoder_reward_shaping: bool = True

    # sim reward text for  availabletext default value  0.0 (disabled). text to  config text enable.
    coef_human_sim: float = 0.0

    decoder: DecoderConfig = field(default_factory=DecoderConfig)

    game_setting_mode: str = "all"

    # ── reward_decoder_mode: reward/condition text select (MGPCGRL  before  for ) ──
    # "noop"  : text game in  text dataset metadata  as-is text for  (decoder text for )
    # "all"   : text game in  text CLIP decoder text  text for  (default value)
    # "unseen": seen game  dataset metadata, unseen gametext decoder text text for
    reward_decoder_mode: str = "unseen"

    # ── path text for  parameter (encoder ckpt text text) ─────────────────────────
    # encoder training text text for text unseen games abbreviation (e.g. "zd", "zddm")
    train_unseen_abbr: Optional[str] = None
    # encoder training text unseen game data ratio (0.0 ~ 1.0)
    train_unseen_ratio: Optional[float] = None
    # encoder training text seen game data ratio (0.0 ~ 1.0)
    train_seen_ratio: Optional[float] = None

    # ── reward_unseen_ratio: unseen game  inside  metadata/decoder text ─────────────
    # dataset_setting.json  of  unseen_ratio  in  automatic injecttext.
    # each unseen game of  sample  order basis as  split:
    #   front (reward_unseen_ratio ratio) → metadata (GT condition, encoder training data)
    #   remaining (1 - reward_unseen_ratio) → reward decoder  to  condition text
    # 0.0 (default value) = text unseen sample in  decoder apply (zero-shot text)
    reward_unseen_ratio: float = 0.0

    # ── encoder training text text for text delta_weight (wandb  to text/text for ) ──
    # encoder_config.json in  automatic injecttext. 0.0 = baseline (direction alignment text for ).
    encoder_delta_weight: float = 0.0

    # MGPCGRL: unseen game data load ratio (default value 1.0 = all load).
    # CLI in  text availabletext, 1.0  text text exp_dir name in  '_uro-XX' suffix  text text.
    dataset_unseen_ratio: float = 1.0


@dataclass
class PretrainedCLIPPCGRLConfig(CPCGRLConfig):
    use_clip: bool = True
    model: str = "pretrained_clip"

    encoder: EncoderConfig = field(default_factory=lambda: EncoderConfig(model="clip"))

    use_nlp: bool = False
    vec_cont: bool = False
    nlp_input_dim: int = 512  # encoder.output_dim (pretrained CLIP latent space, no projection)

    # HuggingFace CLIP  RGB image(3text)  text — render_level_from_arr  RGB tile image create
    clip_input_channel: int = 3

    use_pretrained_clip_reward: bool = True
    wandb_project: Optional[str] = f'{PREFIX}train_pretrained_clip_pcgrl'

    # training in  text for text game list — config.game  in  automatic text also text train_setting.json  in  writetext.
    # (full name text, e.g. ["dungeon", "doom", "zelda"]). text text
    # train_setting.json  in  seen/unseen split   writetext WandB  to text in  text for text.
    reward_seen_games: List[str] = field(default_factory=list)


@dataclass
class FinetunedCLIPPCGRLConfig(PretrainedCLIPPCGRLConfig):
    """Fine-tuned CLIP reward based PCGRL training Config.

    PretrainedCLIPPCGRLConfig  and  sametext text/text structure  text for text,
    `encoder.ckpt_name` (text  ckpt_path)  as  text fine-tuned CLIP
    checkpoint  RL text subtree  in  inject text. (existing
    `apply_encoder_params` text as-is text for )
    """
    wandb_project: Optional[str] = f"{PREFIX}train_finetuned_clip_pcgrl"
    dir_prefix: str = "finetuned-clip-pcgrl-"

    # ── Finetuned CLIP  before  for  RL text text ──────────────────────────────────
    # pretrained_clip  and  parameter text structure  sametext, `get_finetuned_clip_encoder`
    #  to  text  createtext ckpt  of  trainable parameter text (TrainablePretrained*Encoder)
    #  and  text text. separate model text to  exp_dir / encoder hash text text.
    model: str = "finetuned_clip"

    # ── encoder unseen experiment text (mgpcgrl/vipcgrl  and  same text) ──
    dataset_seen_ratio: float = 1.0

    # encoder training text unseen game data ratio (dataset_setting.json in  automatic inject).
    # None text existing text( before  game in  dataset_seen_ratio apply) keep.
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
    """text before  random text evaluation for  Config.

    NN text  uniform random action  text for text,
    exp_dir name  "random_"  as  starttext (cpcgrl_ text and  text).
    """

    random_agent: bool = True
    dir_prefix: str = "random_"
    wandb_project: Optional[str] = f"{PREFIX}eval_random"

    dataset_reward_enum: Optional[Union[int, str]] = 0        # int/list-string (e.g. 0, "01", "0,1") or "all"
    eval_games: str = 'all'

    # (game, re) text evaluation sample text. None text all text for .
    eval_samples_per_group: Optional[int] = 200

    # evaluation text text reward_enum text. None text dataset_reward_enum text text for .
    # text text string to  text available: "12" → [1,2],  "012" → [0,1,2]
    # text/text also  text for : [0,1,2]
    eval_dataset_reward_enums: Optional[str] = None



@dataclass
class CPCGRLEvalConfig(EvalConfig):
    """CPCGRL evaluation for  Config.

    CPCGRLConfig  and  sametext text/text config  EvalConfig above in  text.
    """
    problem: str = "multigame"

    # ── CPCGRLConfig  and  sametext game / dataset default value → exp_dir name text ──
    game: str = "all"
    dataset_game: Optional[str] = "all"
    dataset_reward_enum: Optional[int] = 0        # 0=region
    dataset_train_ratio: float = 0.95

    # evaluation target game (None text game and  same). checkpoint  to text  game basis, evaluation data  eval_games basis.
    # text: game="all"  to  trainingtext text  text gametext evaluationtext text eval_games="dg" text text.
    eval_games: str = 'all'

    vec_cont: bool = True
    raw_obs: bool = True
    model: str = "contconv"
    use_nlp: bool = False
    use_clip: bool = False
    vec_input_dim: Optional[int] = 5
    nlp_input_dim: int = 0

    max_samples: Optional[int] = None  # dry-run for : data count text (None text all text for )

    # (game, re) text evaluation sample text. None text all text for .
    eval_samples_per_group: Optional[int] = 200

    # evaluation text text reward_enum text. None text dataset_reward_enum text text for .
    # text text string to  text available: "12" → [1,2],  "012" → [0,1,2]
    # text/text also  text for : [0,1,2]
    eval_dataset_reward_enums: Optional[str] = None

    # True text checkpoint text also  textrow (WARNING text). False(default) text checkpoint text  text  in text.
    ignore_checkpoint: bool = False


    wandb_project: Optional[str] = f"{PREFIX}eval_cpcgrl"

@dataclass
class VIPCGRLEvalConfig(CPCGRLEvalConfig):
    """VIPCGRL evaluation for  Config.

    pretrained CLIP embedding  nlp_obs  in  injecttext  evaluation config.
    Decoder reward shaping text  CLIP embeddingtext text for text.
    """
    wandb_project: Optional[str] = f"{PREFIX}eval_vipcgrl"

    encoder: EncoderConfig = field(default_factory=lambda: EncoderConfig(model="cnnclip"))

    use_clip: bool = True
    vec_cont: bool = False
    model: str = "cnnclipconv"
    use_nlp: bool = False
    nlp_input_dim: int = 64  # encoder.output_dim (pretrained CLIP latent space)

    ignore_checkpoint: bool = False

    # ── encoder unseen experiment text (mgpcgrl eval  and  same text) ───────────────
    # encoder training text text for text seen_ratio — dataset_setting.json in  automatic injecttext.
    # text/ to text for  as text text for text, eval dataset filtering in   applytext text.
    train_seen_ratio: float = 1.0

    # training text seen/unseen game list — dataset_setting.json in  automatic injecttext.
    seen_games: List[str] = field(default_factory=list)
    unseen_games: List[str] = field(default_factory=list)

    # ── game_setting_mode: evaluation text text for text game range select ──
    # train_vipcgrl  of  default value(encoder_seen)  and  text exp_dir text  text also text text.
    game_setting_mode: str = "encoder_seen"


@dataclass
class PretrainedCLIPEvalConfig(CPCGRLEvalConfig):
    """PretrainedCLIP PCGRL evaluation for  Config.

    train_pretrained_clip.py  to  trainingtext checkpoint  evaluationtext.
    precomputed CLIP text embedding  nlp_obs  in  injecttext,
    separate of  encoder checkpoint text  text text in  text CLIP vision text  text for text.
    """
    wandb_project: Optional[str] = f"{PREFIX}eval_pretrained_clip"

    encoder: EncoderConfig = field(default_factory=lambda: EncoderConfig(model="clip"))

    use_clip: bool = True
    vec_cont: bool = False
    model: str = "pretrained_clip"
    use_nlp: bool = False
    nlp_input_dim: int = 512  # pretrained CLIP text embedding dimension (projection none)

    ignore_checkpoint: bool = False

    # training text seen/unseen game list — train_setting.json  in  automatic injecttext.
    seen_games: List[str] = field(default_factory=list)
    unseen_games: List[str] = field(default_factory=list)


@dataclass
class FinetunedCLIPEvalConfig(PretrainedCLIPEvalConfig):
    """Fine-tuned CLIP PCGRL evaluation for  Config."""
    wandb_project: Optional[str] = f"{PREFIX}eval_finetuned_clip"
    dir_prefix: str = "finetuned-clip-pcgrl-"
    model: str = "finetuned_clip"


@dataclass
class MGPCGRLEvalConfig(CPCGRLEvalConfig):
    """MGPCGRL evaluation for  Config.

    CPCGRLConfig  and  sametext text/text config  EvalConfig above in  text.
    """
    wandb_project: Optional[str] = f"{PREFIX}eval_mgpcgrl"

    use_decoder_reward_shaping: bool = True

    encoder: EncoderConfig = field(default_factory=lambda: EncoderConfig(model="cnnclip"))
    decoder: DecoderConfig = field(default_factory=DecoderConfig)

    use_clip: bool = True
    nlp_input_dim: int = 64  # encoder.output_dim (pretrained CLIP latent space)

    ignore_checkpoint: bool = False

    # encoder training text text for text seen_ratio — dataset_setting.json in  automatic injecttext.
    # text/ to text for  as text text for text, eval dataset filtering in   applytext text.
    train_seen_ratio: float = 1.0

    # exp_dir path text for  — train and  sametext default value(unseen) keep.
    # text eval condition text  eval_reward_decoder_mode to  separate text.
    reward_decoder_mode: str = "unseen"

    # eval text text to  text for text condition text.
    # "noop" → GT condition text for  (default value, text text  abovetext).
    # "unseen" → unseen gametext decoder text text for .
    eval_reward_decoder_mode: str = "noop"

    # training text seen/unseen game list — reward_decoder_config.json in  automatic injecttext.
    seen_games: List[str] = field(default_factory=list)
    unseen_games: List[str] = field(default_factory=list)


    # ── path text for  parameter (encoder ckpt text text) ─────────────────────────
    # train text MGPCGRLConfig and  sametext text  text exp_dir  text.
    train_unseen_abbr: Optional[str] = None
    train_unseen_ratio: Optional[float] = None
    train_seen_ratio: Optional[float] = None

    # ── encoder training text text for text delta_weight (wandb  to text/text for ) ──
    encoder_delta_weight: float = 0.0

    # train and  sametext text  text exp_dir  text. 1.0  text text '_uro-XX' suffix.
    dataset_unseen_ratio: float = 1.0


@dataclass
class IPCGRLEvalConfig(CPCGRLEvalConfig):
    """IPCGRL evaluation for  Config.

    CPCGRLEvalConfig   text BERT embedding + MLP text config  text text.
    """
    use_nlp: bool = True
    vec_cont: bool = False
    model: str = "nlpconv"
    nlp_input_dim: int = 768

    encoder: EncoderConfig = field(default_factory=lambda: EncoderConfig(model="mlp"))

    dataset_reward_enum: Optional[int] = None

    wandb_project: Optional[str] = f"{PREFIX}eval_ipcgrl"

    # ── encoder training text seen_ratio (analysis only) ──────────────────────────────
    train_seen_ratio: float = 1.0

    # ── training text seen/unseen game list — dataset_setting.json in  automatic injecttext ──
    seen_games: List[str] = field(default_factory=list)
    unseen_games: List[str] = field(default_factory=list)

    # Train-side name used by path_utils/dataset helpers. Eval fills this with
    # the same encoder seen-game metadata so exp_dir matches the trained run.
    reward_seen_games: List[str] = field(default_factory=list)

    # ── MIPCGRL variant text (IPCGRLConfig  and  same text) ──
    is_mipcgrl: bool = False


@dataclass
class MIPCGRLEvalConfig(IPCGRLEvalConfig):
    """MIPCGRL evaluation for  Config — IPCGRLEvalConfig  and  same structure, is_mipcgrl=True."""
    wandb_project: Optional[str] = f"{PREFIX}eval_mipcgrl"
    is_mipcgrl: bool = True


@dataclass
class CollectBufferConfig(CPCGRLConfig):
    """training  during  trajectory text  text  Config.

    training 50%~100% bin(collect_start_ratio~collect_end_ratio) in
    text text text(env_idx=0) basis as  data  text
    experiment folder of  buffer/ directory in  .npz file to  savetext.
    """
    wandb_project: str = 'collect_buffer'
    dir_prefix: str = "buffer-"

    # ── text text parameter ──
    buffer_max_samples: int = 10_000       # text maximum transition text
    collect_start_ratio: float = 0.5        # text start ratio (0.5 = training 50%)
    collect_end_ratio: float = 1.0          # text text ratio (1.0 = training 100%)
    buffer_save_dir: Optional[str] = None   # save path (None text exp_dir/buffer)

    # training  during  env_map  transition in  save (text in  text)
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

    max_samples: Optional[int] = None  # dry-run for : data count text (None text all text for )



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
    max_samples: Optional[int] = None  # dry-run for : data count text (None text all text for )
    encoder: EncoderConfig = field(default_factory=EncoderConfig)

    # instruction prefix mode: "name" (e.g. "In Zelda, ...") / "desc" / "mix" / "none" (text  None)
    instruction_prefix: Optional[str] = "name"

    # instruction field select: "uni" / "raw" (default)
    instruction_field: str = "raw"

    # overwrite
    embed_type: str = "humanai"

    # ── Seen/Unseen game separate config (CLIPDecoderTrainConfig  and  same text) ──
    # unseen game text (2text abbreviation, e.g., "zd"=zelda, "pkzd"=pokemon+zelda).
    # None/""  text existing text (all game  train/test ratio to  split).
    unseen_games: Optional[str] = None
    # few-shot ratio: unseen training text  during  text for text ratio (0.0=zero-shot, 1.0= before text)
    unseen_ratio: float = 0.0
    # seen game data ratio (1.0= before text text for )
    seen_ratio: float = 1.0
    # text split seed (text available)
    split_seed: int = 42

@dataclass
class FinetunedCLIPEncoderTrainConfig(CLIPTrainConfig):
    """HuggingFace pretrained CLIP  text for text of  (image, text) data to
    text abovetext Config.

    parameter text structure  `pretrained_clip_model.ContrastiveModule`  and  sametext to
    savetext checkpoint  as-is RL pipeline(`apply_encoder_params`) in  inject text text text.
    """
    wandb_project: str = f"{PREFIX}train_finetuned_clip_encoder"
    dir_prefix: str = "finetuned-clip-"

    # HF CLIP  224×224 text  text → coordinatetext OFF
    clip_input_channel: int = 3

    # encoder text text (path/exp name text). RL text in   'clip' to   to text.
    encoder: EncoderConfig = field(
        default_factory=lambda: EncoderConfig(model="clip", state=True)
    )

    # HF CLIP  text text trainingtext  text text text, epoch  also  text (5~15) keeptext  text
    # → catastrophic forgetting text + text  also text text
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
    """CLIP Encoder + Reward Decoder training Config.

    existing contrastive loss in  text text text  text text
    state embedding as text reward_enum(text) and  condition(text)  text.
    """
    wandb_project: str = f'{PREFIX}train_mgpcgrl_encoder'
    dir_prefix: str = "clipdec-"

    # ── text config ──
    decoder: DecoderConfig = field(default_factory=DecoderConfig)

    # ── loss weight ──
    contrastive_weight: float = 1.0    # contrastive loss weight
    cls_weight: float = 1.0            # reward_enum text loss weight
    reg_weight: float = 1.0            # condition text loss weight

    # ── Continuous Task-wise Cross-game Direction Alignment Loss ──
    # same task text in  condition  text text text text embedding  text   text
    # game text in  sorttext  regularizer. 0.0 → disabled(baseline text).
    delta_weight: float = 0.03
    delta_min_group_samples: int = 2   # (game, task) text minimum sample text
    delta_var_eps: float = 1e-4        # condition variance text (text text invalid)
    compute_delta_when_zero: bool = True  # delta_weight=0.0 text also  alignment metric compute

    # ── regression loss text ──
    # "huber": Huber loss (δ=1.0), "mae": Mean Absolute Error
    regression_loss: str = "mae"

    # ── Seen/Unseen game separate config ──
    # unseen game text (2text abbreviation, e.g., "zd"=zelda, "pkzd"=pokemon+zelda). None=all seen
    unseen_games: Optional[str] = None
    # few-shot ratio: unseen training text  during  text for text ratio (0.0=zero-shot, 1.0= before text)
    unseen_ratio: float = 0.0
    # seen game data ratio (1.0= before text text for )
    seen_ratio: float = 1.0
    # text split seed (text available)
    split_seed: int = 42

    n_epochs: int = 3000

    # ── Step based checkpoint / evaluation text ──
    ckpt_freq: int = 1000   # checkpoint save text (steps, 0 text disabled)
    scatter_freq: int = 500  # scatter plot upload text (epochs, 0/text disabled)

    # ── Unseen game  before  for   to text text ──
    unseen_eval_freq: int = 100    # unseen regression text  to text text (epochs, 0 text disabled)
    unseen_scatter_freq: int = 500  # unseen scatter plot  to text text (epochs, 0 text disabled)

    # ── Unseen evaluation data ratio ──
    # unseen_ratio  : training data in  text   unseen game data ratio (train pool basis)
    # eval_unseen_ratio : unseen_eval_freq evaluation in  text for text unseen test set ratio (0.0~1.0, 1.0=all)
    eval_unseen_ratio: float = 1.0
    export_unseen_predictions_csv: bool = True

    # ── Gradient text text ──
    # True: decoder loss (cls + reg) of  gradient  encoder(latent space)text  before text text
    # False (default value): decoder loss  encodertext text before text
    decoder_nograd: bool = False


@dataclass
class CLIPDecoderUnseenConfig(CLIPDecoderTrainConfig):
    """Seen/Unseen game separate + Few-shot Ratio Sweep Config.

    Seen game of  all training data and  Unseen game of   text ratio training data to
    CLIP Decoder text  trainingtext, fixedtext text in  gametext reward_accuracy  measuretext.
    """
    wandb_project: str = 'train_clip_decoder_unseen'
    dir_prefix: str = "clipdec-"

    # ── Unseen game text (2text abbreviation, e.g., "zd"=zelda, "pkzd"=pokemon+zelda) ──
    unseen_games: Optional[str] = None

    # ── Few-shot ratio (text Usage for ) ──
    # 0.0 = zero-shot (unseen training data 0%), 1.0 = unseen training text  before text text for
    unseen_ratio: float = 0.01

    # ── Seen game data ratio ──
    # 1.0 = seen training text  before text text for  (default value), 0.0 = seen training data 0%
    seen_ratio: float = 1.0

    # ── text config ──
    # train_ratio: training data ratio (text CLIPTrainConfig text, default 0.99 → text 0.8 to  text of )
    # test ratio = 1.0 - train_ratio
    train_ratio: float = 0.99
    split_seed: int = 42              # text split seed (text available)


@dataclass
class CLIPDecoderUnseenSweepConfig(CLIPDecoderUnseenConfig):
    """Seen/Unseen game separate + Few-shot Ratio **Sweep** Config.

    CLIPDecoderUnseenConfig   text, unseen_ratios text  text  to  text of text.
    sweep/runnable_sweep/unseen_games.py  in  text for text.
    """
    # ── Few-shot ratio sweep config ──
    # 0.0 = zero-shot, 1.0 = unseen training text  before text text for
    unseen_ratios: Tuple[float, ...] = (0.0, 0.01, 0.03, 0.05, 0.1)


@dataclass
class IPCGRLEncoderMGConfig(RewardConfig):
    """IPCGRL MLP text textgame pretraining Config.

    Annotation text MultiGameDataset based.
    - text: BERT(instruction) → 768-dim embedding
    - text: MLP text + MLP text
    - text: condition value text (log1p + per-enum min-max normalize)
    - unseen_games: training in  text game text (zero-shot evaluation for )

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

    # Unseen game config (2text abbreviation, e.g. "zd"=zelda, "pkzd"=pokemon+zelda)
    # text string = text none (all game training)
    unseen_games: str = ""

    # ── Seen/Unseen data ratio (CLIPTrainConfig and  same) ──
    # unseen_ratio: unseen training text  during  text for text ratio (0.0=zero-shot, 1.0= before text)
    unseen_ratio: float = 0.0
    # seen_ratio: seen game data ratio (1.0= before text text for )
    seen_ratio: float = 1.0

    # Annotation dataset config (CLIPTrainConfig  and  sametext text text)
    # instruction_prefix mode: "name" (default) / "desc" / "none" (text  None)
    instruction_prefix: Optional[str] = "name"

    # instruction field select: "uni" / "raw" (default)
    instruction_field: str = "raw"

    # MLP text (apply_encoder_model  in  model='mlp' text text for )
    encoder: EncoderConfig = field(default_factory=lambda: EncoderConfig(model="mlp"))


@dataclass
class MIPCGRLConfig(IPCGRLConfig):
    wandb_project: Optional[str] = f"{PREFIX}train_mipcgrl"
    is_mipcgrl: bool = True


@dataclass
class MIPCGRLEncoderMGConfig(IPCGRLEncoderMGConfig):
    """MIPCGRL MLP text textgame pretraining Config.

    IPCGRL text(condition value text textrow) in  text, same latent z  to text
    task(reward_enum) text head   text  to  trainingtext.
        Loss = MSE(condition) + classifier_weight * CrossEntropy(reward_enum)

    Usage:
        python train_mipcgrl_encoder_mg.py game=all
        python train_mipcgrl_encoder_mg.py game=all unseen_games=zd classifier_weight=0.5
    """
    wandb_project: Optional[str] = f"{PREFIX}train_mipcgrl_encoder"
    dir_prefix: str = "mipcgrl-enc-mg-"

    # ── Classifier config ──
    # task(reward_enum) text head  of  text weight. 0  text IPCGRL  and  same.
    classifier_weight: float = 1.0
    # classifier MLP hidden / layer text (output_size   num_classes  to  automatic text)
    classifier_num_layers: int = 2
    classifier_hidden_dim: int = 128
    classifier_dropout_rate: float = 0.0
    # text class text. None  text dataset of  text reward_enum count to  automatic config.
    num_classes: Optional[int] = None


cs = ConfigStore.instance()
cs.store(name="config", node=Config)
cs.store(name="train_pcgrl", node=TrainConfig)
cs.store(name="cpcgrl", node=CPCGRLConfig)
cs.store(name="ipcgrl", node=IPCGRLConfig)
cs.store(name="mipcgrl", node=MIPCGRLConfig)
cs.store(name="vipcgrl", node=VIPCGRLConfig)
cs.store(name="mgpcgrl", node=MGPCGRLConfig)
cs.store(name="pretrained_clip_pcgrl", node=PretrainedCLIPPCGRLConfig)
cs.store(name="finetuned_clip_pcgrl_schema", node=FinetunedCLIPPCGRLConfig)
cs.store(name="eval_pcgrl", node=EvalConfig)
cs.store(name="eval_random_schema", node=RandomEvalConfig)
cs.store(name="eval_cpcgrl_schema", node=CPCGRLEvalConfig)
cs.store(name="eval_ipcgrl_schema", node=IPCGRLEvalConfig)
cs.store(name="eval_mipcgrl_schema", node=MIPCGRLEvalConfig)
cs.store(name="eval_vipcgrl_schema", node=VIPCGRLEvalConfig)
cs.store(name="eval_mgpcgrl_schema", node=MGPCGRLEvalConfig)
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

cs.store(name="train_reward", node=RewardTrainConfig)
cs.store(name="train_ipcgrl_encoder_mg_schema", node=IPCGRLEncoderMGConfig)
cs.store(name="train_mipcgrl_encoder_mg_schema", node=MIPCGRLEncoderMGConfig)

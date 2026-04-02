from typing import Dict, Iterable, List, Optional, Tuple, Union
from hydra.core.config_store import ConfigStore
from dataclasses import dataclass, field

from conf.game_utils import (                       # noqa: F401  — re-export
    GAME_ABBR, GAME_ABBR_INV, ALL_GAMES,
    parse_game_str, build_game_str,
)

@dataclass
class Config:
    lr: float = 1.0e-4
    n_envs: int = 4
    # How many steps do I take in all of my batched environments before doing a gradient update
    num_steps: int = 128
    total_timesteps: int = int(5e7)
    timestep_chunk_size: int = -1
    update_epochs: int = 10
    NUM_MINIBATCHES: int = 4
    GAMMA: float = 0.99
    GAE_LAMBDA: float = 0.95
    CLIP_EPS: float = 0.2
    ENT_COEF: float = 0.01
    VF_COEF: float = 0.5
    SIM_COEF: float = 1.0
    MAX_GRAD_NORM: float = 0.5
    activation: str = "relu"
    env_name: str = "PCGRL"
    ANNEAL_LR: bool = False
    DEBUG: bool = False
    exp_name: str = "def"
    seed: int = 0

    # Game selection — 2글자 약어 조합 (dg=dungeon, pk=pokemon, sk=sokoban, dm=doom(+doom2), zd=zelda)
    # 예: "dg" (dungeon만), "dgdm" (dungeon+doom+doom2), "all" (전체)
    game: str = "dg"

    # include_* 필드는 game 문자열에서 자동 파싱됨 (하위 호환용으로 유지)
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
    clip_input_channel: int = 3

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

    # Wandb (WANDB_API_KEY 는 .env 파일 또는 환경변수로 설정)
    wandb_key: Optional[str] = None
    wandb_project: Optional[str] = 'instruct_pcgrl'
    wandb_entity: Optional[str] = 'st4889ha-gwangju-institute-of-science-and-technology'
    wandb_resume: str = 'allow'
    evaluator: str = 'hr'  # 'vit', 'hr' (heuristic)
    
    #Ablation study options. default is 1.0
    text_ratio: float = 1.0
    state_ratio: float = 1.0

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
    dataset_reward_enum: Optional[int] = None   # e.g. 0=region, 1=path_length, 2=interactable, 3=hazard, 4=collectable
    dataset_train_ratio: float = 0.95

    # Multigame tile placement reward 가중치 (sweep 대상)
    placement_w_amount: float = 0.4
    placement_w_cluster: float = 0.2
    placement_w_access: float = 0.2
    placement_w_spread: float = 0.2

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


@dataclass
class DecoderConfig:
    hidden_dim: int = 128
    num_layers: int = 2
    output_dim: int = 1
    num_reward_classes: int = 6


@dataclass
class TrainConfig(Config):
    overwrite: bool = False
    ckpt_freq: int = int(5e6)
    render_freq: int = 50
    n_render_eps: int = 3
    eval_freq: int = 5
    n_eval_maps: int = 6
    eval_map_path: str = "user_defined_freezies/binary_eval_maps.json"

    NUM_UPDATES: Optional[int] = None
    MINIBATCH_SIZE: Optional[int] = None

    agents: int = 1
    current_iteration: int = -1
    instruct_freq: int = 1
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    buffer_ratio: float = 1

    use_sim_reward: bool = False
    only_sim_reward: bool = False
    human_demo: bool = True
    human_level: str = "human_20250630_213109"
    human_augment: bool = False

    multimodal_condition: bool = False
    human_demo_path: str = './human_dataset'


@dataclass
class CPCGRLConfig(TrainConfig):
    problem: str = "multigame"
    dataset_game: Optional[str] = "dungeon"
    dataset_reward_enum: Optional[int] = 0        # 0=region
    dataset_train_ratio: float = 0.95

    vec_cont: bool = True
    raw_obs: bool = True
    model: str = "contconv"
    use_nlp: bool = False
    use_clip: bool = False
    vec_input_dim: Optional[int] = 9
    nlp_input_dim: int = 0

    instruct: Optional[str] = None
    instruct_csv: Optional[str] = None
    aug_type: str = "sub_condition"
    embed_type: str = "bert"

    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    use_sim_reward: bool = False
    only_sim_reward: bool = False
    human_demo: bool = False

    wandb_project: Optional[str] = "cpcgrl"


@dataclass
class VIPCGRLConfig(CPCGRLConfig):
    use_clip: bool = True
    model: str = "cnnclipconv"
    encoder: EncoderConfig = field(default_factory=lambda: EncoderConfig(model="cnnclip"))

    use_nlp: bool = False
    vec_cont: bool = False
    nlp_input_dim: int = 64  # encoder.output_dim (pretrained CLIP latent space)

    use_sim_reward: bool = True
    wandb_project: Optional[str] = "vipcgrl"


@dataclass
class MGPCGRLConfig(VIPCGRLConfig):
    wandb_project: Optional[str] = "mgpcgrl"

    # MGPCGRL: clip_decoder 기반 동적 보상 예측 (reward_i/condition)
    use_decoder_reward_shaping: bool = True
    decoder_ckpt_path: Optional[str] = None
    decoder_reward_classes: int = 5
    dummy_decoder: bool = False


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

    eval_aug_type: str = "sub_condition"
    eval_embed_type: str = "bert"
    eval_instruct: str = "scn-1_se-whole"
    eval_instruct_csv: Optional[str] = None
    eval_dir: Optional[str] = None
    eval_map_types: int = 5
    eval_modality: str = "text"
    eval_human_demo_path: str = './human_dataset'

    diversity: bool = True
    human_likeness: bool = True
    vit_normalize: bool = False
    tpkldiv: bool = True

    wandb_project: str = 'eval_pcgrl'

    metrics_to_keep: Tuple[str] = ("mean_ep_reward",)
    flush: bool = True



@dataclass
class CollectBufferConfig(CPCGRLConfig):
    """학습 중 trajectory 버퍼를 수집하는 Config.

    학습 50%~100% 구간(collect_start_ratio~collect_end_ratio)에서
    첫 번째 환경(env_idx=0) 기준으로 데이터를 수집하여
    실험 폴더의 buffer/ 디렉토리에 .npz 파일로 저장한다.
    """
    wandb_project: str = 'collect_buffer'

    # ── 버퍼 수집 파라미터 ──
    buffer_max_samples: int = 10_000       # 수집할 최대 transition 수
    collect_start_ratio: float = 0.5        # 수집 시작 비율 (0.5 = 학습 50%)
    collect_end_ratio: float = 1.0          # 수집 종료 비율 (1.0 = 학습 100%)
    buffer_save_dir: Optional[str] = None   # 저장 경로 (None이면 exp_dir/buffer)

    # 학습 중 env_map을 transition에 저장 (수집에 필요)
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
    batch_size: int = 32

    num_layers: int = 2  # 1 ~ 3
    hidden_dim: int = 512
    output_dim: int = 1

    figure_dir: str = "figures"
    buffer_dir: str = "./pcgrl_buffer"
    n_buffer: int = -1
    train_ratio: float = 0.8
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

@dataclass
class RewardTrainConfig(RewardConfig):
    wandb_project: str = 'train_mlp_encoder'

    n_envs: int = 300
    ckpt_freq: int = 5


@dataclass
class CLIPTrainConfig(Config):
    exp_name: str = "def"
    
    wandb_project: str = 'train_clip'
    seed: int = 0
    
    overwrite: bool = False
    ckpt_freq: int = int(5)

    # Goal img path
    img_data_path: str = "./human_dataset"
    instruct: str = "scn-1_se-whole"
    
    n_max_points: int = 1000
    embed_visualize_freq: int = 5

    n_epochs: int = 100
    lr: float = 1.0e-3
    weight_decay: float = 1e-5
    train_ratio: float = 0.8
    batch_size: int = 128
    buffer_ratio: float = 1.0 # Not implemented for clip yet.
    train_shuffle: bool = False
    
    dir_prefix: str = "clip-"
    figure_dir: str = "figures"
    
    steps_per_epoch: Optional[int] = None
    max_samples: Optional[int] = None  # dry-run용: 데이터 개수 제한 (None이면 전체 사용)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    
    # overwrite
    embed_type: str = "humanai"

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
    """CLIP Encoder + Reward Decoder 학습 Config.

    기존 contrastive loss에 더해 디코더 브랜치를 추가하여
    state embedding으로부터 reward_enum(분류)과 condition(회귀)을 예측한다.
    """
    wandb_project: str = 'train_clip_decoder'
    dir_prefix: str = "clipdec-"

    # ── 디코더 설정 ──
    decoder: DecoderConfig = field(default_factory=DecoderConfig)

    # ── loss 가중치 ──
    contrastive_weight: float = 1.0    # contrastive loss 가중치
    cls_weight: float = 1.0            # reward_enum 분류 loss 가중치
    reg_weight: float = 0.1            # condition 회귀 loss 가중치

cs = ConfigStore.instance()
cs.store(name="config", node=Config)
cs.store(name="train_pcgrl", node=TrainConfig)
cs.store(name="cpcgrl", node=CPCGRLConfig)
cs.store(name="vipcgrl", node=VIPCGRLConfig)
cs.store(name="mgpcgrl", node=MGPCGRLConfig)
cs.store(name="eval_pcgrl", node=EvalConfig)
cs.store(name="collect_buffer_schema", node=CollectBufferConfig)

# CLIP PCGRL Configs
cs.store(name="train_clip", node=CLIPTrainConfig)
cs.store(name="train_clip_decoder_schema", node=CLIPDecoderTrainConfig)

cs.store(name="train_bert", node=BertTrainConfig)
cs.store(name="eval_bert", node=BertEvalConfig)

cs.store(name="train_reward", node=RewardTrainConfig)


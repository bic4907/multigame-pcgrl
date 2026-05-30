from typing import Dict, Iterable, List, Optional, Tuple, Union
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

    # Game selection — 2글자 약어 조합 (dg=dungeon, pk=pokemon, sk=sokoban, dm=doom(+doom2), zd=zelda)
    # 예: "dg" (dungeon만), "dgdm" (dungeon+doom+doom2), "all" (전체)
    game: str = "all"

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
    # tile-only 채널 수 = unified category 수 (NUM_CATEGORIES).
    # init_config()에서 coord 채널 2개를 더해 총 NUM_CATEGORIES+2 가 모델 입력 채널이 된다.
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

    # 다른 경로의 체크포인트에서 학습을 시작할 때 지정. None이면 exp_dir/ckpts에서 자동 복원.
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
    dataset_reward_enum: Optional[Union[int, str]] = None   # int/list-string (e.g. 0, "01", "0,1") or "all"
    dataset_train_ratio: float = 0.95

    # 공통 데이터 전처리 (모든 파이프라인에 동일하게 적용)
    longtail_cut: bool = True          # 극단적 condition 값 샘플 제거
    max_samples_per_game: int = 1000   # 게임별 source_id 상한 (0=무제한)
    max_samples_seed: int = 42         # max_samples_per_game 서브샘플링 시드
    rl_tile_offset: int = 1            # 타일 enum 값에 더할 오프셋 (RL 데이터로더용)

    # Multigame tile placement reward 가중치 (sweep 대상)
    placement_w_amount: float = 1.0
    placement_w_spread: float = 0.0

    # Special tile (interactive/hazard/collectable) 존재 패널티 가중치
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
    tile_offset: int = 0               # 타일 enum 값에 더할 오프셋 (인코더)


@dataclass
class DecoderConfig:
    hidden_dim: int = 128
    num_layers: int = 2
    output_dim: int = 1
    num_reward_classes: int = 5
    # CNN 입력에 reward_enum one-hot 채널을 추가할지 여부
    # True이면 pixel_values에 (B, H, W, num_reward_classes) one-hot을 concat
    cnn_reward_enum_onehot: bool = False


@dataclass
class TrainConfig(Config):
    overwrite: bool = False
    ckpt_freq: int = int(4e6)
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

    # ── instruction prefix mode (train/eval/encoder 공통) ─────────────────
    # "name" (기본): "In Zelda, ..." 같이 게임 이름 prefix
    # "desc"      : "(traverse a room, fight creatures, ...) ..." 같이 게임 설명 prefix
    # "none"/None : prefix 미적용
    # 인코더 학습 시 사용한 값과 RL 학습/평가에서 동일해야 임베딩이 일치한다.
    instruction_prefix: Optional[str] = "name"


@dataclass
class CPCGRLConfig(TrainConfig):
    problem: str = "multigame"

    game: str = "all"

    dataset_game: Optional[str] = "all"
    dataset_reward_enum: Optional[Union[int, str]] = 0        # int/list-string (e.g. 0, "01", "0,1") or "all"
    dataset_train_ratio: float = 0.95
    # condition 값 기반 필터: "enum_{i}_min_{v}" / "enum_{i}_max_{v}" / "enum_{i}_min_{lo}_max_{hi}"
    # 여러 필터는 쉼표 구분: "enum_0_min_3_max_10,enum_2_max_50"
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
    """IPCGRL (Instructed PCGRL) — BERT 임베딩 → MLP 인코더."""
    use_nlp: bool = True
    vec_cont: bool = False
    model: str = "nlpconv"
    nlp_input_dim: int = 768

    encoder: EncoderConfig = field(default_factory=lambda: EncoderConfig(model="mlp"))


    wandb_project: Optional[str] = f'{PREFIX}train_ipcgrl'


@dataclass
class VIPCGRLConfig(CPCGRLConfig):
    use_clip: bool = True
    model: str = "cnnclipconv"
    encoder: EncoderConfig = field(default_factory=lambda: EncoderConfig(model="cnnclip"))

    use_nlp: bool = False
    vec_cont: bool = False
    nlp_input_dim: int = 64  # encoder.output_dim (pretrained CLIP latent space)

    # coef_human_sim > 0: human_demo sim_reward 활성화 및 계수로 사용 (0이면 비활성)
    coef_human_sim: float = 30.0

    wandb_project: Optional[str] = f"{PREFIX}train_vipcgrl"

    ignore_checkpoint: bool = False

    # ── encoder unseen 실험 지원 (mgpcgrl 과 동일 시맨틱) ──────────────────────
    # encoder 학습 시 사용한 seen_ratio — dataset_setting.json에서 자동 주입됨.
    # 1.0 = 전체 seen 게임 데이터 사용 (기본값), 0.0~1.0 = seen 게임 데이터 prefix 비율
    dataset_seen_ratio: float = 1.0

    # ── game_setting_mode: 학습에 사용할 게임 범위 선택 ──
    # "all"          : 전체 게임 사용
    # "encoder_seen" : encoder 학습 시 seen 게임만 사용 (기본값, dataset_setting.json에서 자동 읽음)
    game_setting_mode: str = "encoder_seen"

    # encoder 학습 시 seen 게임 목록 — dataset_setting.json에서 자동 주입됨.
    # (full name 리스트, e.g. ["dungeon", "doom", "zelda"]). 비어있지 않으면
    # train_setting.json에 seen/unseen split이 기록되어 WandB 로깅에 사용된다.
    reward_seen_games: List[str] = field(default_factory=list)



@dataclass
class MGPCGRLConfig(VIPCGRLConfig):
    wandb_project: Optional[str] = f"{PREFIX}train_mgpcgrl"

    # MGPCGRL: clip_decoder 기반 동적 보상 예측 (reward_i/condition)
    use_decoder_reward_shaping: bool = True

    # sim reward 사용 가능하되 기본값은 0.0 (비활성). 양수로 설정 시 활성화.
    coef_human_sim: float = 0.0

    decoder: DecoderConfig = field(default_factory=DecoderConfig)

    game_setting_mode: str = "all"

    # ── reward_decoder_mode: reward/condition 소스 선택 (MGPCGRL 전용) ──
    # "noop"  : 모든 게임에 대해 데이터셋 메타데이터를 그대로 사용 (decoder 미사용)
    # "all"   : 모든 게임에 대해 CLIP decoder 예측값을 사용 (기본값)
    # "unseen": seen 게임은 데이터셋 메타데이터, unseen 게임만 decoder 예측값 사용
    reward_decoder_mode: str = "unseen"

    # ── 경로 식별용 파라미터 (encoder ckpt 해시 대체) ─────────────────────────
    # encoder 학습 시 사용한 unseen games 약어 (e.g. "zd", "zddm")
    train_unseen_abbr: Optional[str] = None
    # encoder 학습 시 unseen game 데이터 비율 (0.0 ~ 1.0)
    train_unseen_ratio: Optional[float] = None
    # encoder 학습 시 seen game 데이터 비율 (0.0 ~ 1.0)
    train_seen_ratio: Optional[float] = None




@dataclass
class PretrainedCLIPPCGRLConfig(CPCGRLConfig):
    use_clip: bool = True
    model: str = "pretrained_clip"

    encoder: EncoderConfig = field(default_factory=lambda: EncoderConfig(model="clip"))

    use_nlp: bool = False
    vec_cont: bool = False
    nlp_input_dim: int = 512  # encoder.output_dim (pretrained CLIP latent space, no projection)

    use_pretrained_clip_reward: bool = True
    wandb_project: Optional[str] = f'{PREFIX}train_pretrained_clip_pcgrl'

    # 학습에 사용된 게임 목록 — config.game 에서 자동 유도되어 train_setting.json 에 기록됨.
    # (full name 리스트, e.g. ["dungeon", "doom", "zelda"]). 비어있지 않으면
    # train_setting.json 에 seen/unseen split 이 기록되어 WandB 로깅에 사용된다.
    reward_seen_games: List[str] = field(default_factory=list)


@dataclass
class FinetunedCLIPPCGRLConfig(PretrainedCLIPPCGRLConfig):
    """Fine-tuned CLIP 보상 기반 PCGRL 학습 Config.

    PretrainedCLIPPCGRLConfig 와 동일한 모델/환경 구조를 사용하되,
    `encoder.ckpt_name` (또는 ckpt_path) 으로 지정된 fine-tuned CLIP
    체크포인트를 RL 인코더 subtree 에 inject 한다. (기존
    `apply_encoder_params` 메커니즘 그대로 활용)
    """
    wandb_project: Optional[str] = f"{PREFIX}train_finetuned_clip_pcgrl"
    dir_prefix: str = "finetuned-clip-pcgrl-"

    # ── Finetuned CLIP 전용 RL 모델 분기 ──────────────────────────────────
    # pretrained_clip 와 파라미터 트리 구조는 동일하지만, `get_finetuned_clip_encoder`
    # 로 모듈을 생성해 ckpt 의 trainable 파라미터 트리 (TrainablePretrained*Encoder)
    # 와 정확히 일치시킨다. 별도 model 식별자로 exp_dir / encoder hash 충돌 회피.
    model: str = "finetuned_clip"

    # ── encoder unseen 실험 지원 (mgpcgrl/vipcgrl 와 동일 시맨틱) ──
    dataset_seen_ratio: float = 1.0
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
    """완전 랜덤 정책 평가용 Config.

    NN 없이 uniform random action을 사용하며,
    exp_dir 이름이 "random_" 으로 시작한다 (cpcgrl_ 접두사와 대응).
    """

    random_agent: bool = True
    dir_prefix: str = "random_"
    wandb_project: Optional[str] = f"{PREFIX}eval_random"

    dataset_reward_enum: Optional[Union[int, str]] = 0        # int/list-string (e.g. 0, "01", "0,1") or "all"
    eval_games: str = 'all'

    # (game, re) 그룹당 평가 샘플 수. None이면 전체 사용.
    eval_samples_per_group: Optional[int] = 200

    # 평가 시 복수 reward_enum 지정. None이면 dataset_reward_enum 단일값 사용.
    # 숫자 연결 문자열로 지정 가능: "12" → [1,2],  "012" → [0,1,2]
    # 리스트/튜플도 허용: [0,1,2]
    eval_dataset_reward_enums: Optional[str] = None



@dataclass
class CPCGRLEvalConfig(EvalConfig):
    """CPCGRL 평가용 Config.

    CPCGRLConfig 와 동일한 모델/환경 설정을 EvalConfig 위에 덮어쓴다.
    """
    problem: str = "multigame"

    # ── CPCGRLConfig 와 동일한 game / dataset 기본값 → exp_dir 이름 일치 ──
    game: str = "all"
    dataset_game: Optional[str] = "all"
    dataset_reward_enum: Optional[int] = 0        # 0=region
    dataset_train_ratio: float = 0.95

    # 평가 대상 게임 (None이면 game과 동일). 체크포인트 로딩은 game 기준, 평가 데이터는 eval_games 기준.
    # 예: game="all" 로 학습된 모델을 특정 게임만 평가할 때 eval_games="dg" 처럼 지정.
    eval_games: str = 'all'

    vec_cont: bool = True
    raw_obs: bool = True
    model: str = "contconv"
    use_nlp: bool = False
    use_clip: bool = False
    vec_input_dim: Optional[int] = 5
    nlp_input_dim: int = 0

    max_samples: Optional[int] = None  # dry-run용: 데이터 개수 제한 (None이면 전체 사용)

    # (game, re) 그룹당 평가 샘플 수. None이면 전체 사용.
    eval_samples_per_group: Optional[int] = 200

    # 평가 시 복수 reward_enum 지정. None이면 dataset_reward_enum 단일값 사용.
    # 숫자 연결 문자열로 지정 가능: "12" → [1,2],  "012" → [0,1,2]
    # 리스트/튜플도 허용: [0,1,2]
    eval_dataset_reward_enums: Optional[str] = None

    # True이면 체크포인트 없어도 진행 (WARNING 출력). False(기본)이면 체크포인트 없을 시 에러.
    ignore_checkpoint: bool = False


    wandb_project: Optional[str] = f"{PREFIX}eval_cpcgrl"

@dataclass
class VIPCGRLEvalConfig(CPCGRLEvalConfig):
    """VIPCGRL 평가용 Config.

    pretrained CLIP 임베딩을 nlp_obs 에 주입하는 평가 설정.
    Decoder reward shaping 없이 CLIP embedding만 사용한다.
    """
    wandb_project: Optional[str] = f"{PREFIX}eval_vipcgrl"

    encoder: EncoderConfig = field(default_factory=lambda: EncoderConfig(model="cnnclip"))

    use_clip: bool = True
    vec_cont: bool = False
    model: str = "cnnclipconv"
    use_nlp: bool = False
    nlp_input_dim: int = 64  # encoder.output_dim (pretrained CLIP latent space)

    ignore_checkpoint: bool = False

    # ── encoder unseen 실험 지원 (mgpcgrl eval 과 동일 시맨틱) ───────────────
    # encoder 학습 시 사용한 seen_ratio — dataset_setting.json에서 자동 주입됨.
    # 분석/로깅용으로만 사용하며, eval 데이터셋 필터링에는 적용되지 않음.
    train_seen_ratio: float = 1.0

    # 학습 시 seen/unseen 게임 목록 — dataset_setting.json에서 자동 주입됨.
    seen_games: List[str] = field(default_factory=list)
    unseen_games: List[str] = field(default_factory=list)

    # ── game_setting_mode: 평가 시 사용할 게임 범위 선택 ──
    # train_vipcgrl 의 기본값(encoder_seen) 과 맞춰서 exp_dir 매칭이 일관되도록 한다.
    game_setting_mode: str = "encoder_seen"


@dataclass
class PretrainedCLIPEvalConfig(CPCGRLEvalConfig):
    """PretrainedCLIP PCGRL 평가용 Config.

    train_pretrained_clip.py 로 학습한 체크포인트를 평가한다.
    사전 계산된 CLIP 텍스트 임베딩을 nlp_obs 에 주입하며,
    별도의 encoder 체크포인트 없이 모델 자체에 포함된 CLIP 비전 인코더를 사용한다.
    """
    wandb_project: Optional[str] = f"{PREFIX}eval_pretrained_clip"

    encoder: EncoderConfig = field(default_factory=lambda: EncoderConfig(model="clip"))

    use_clip: bool = True
    vec_cont: bool = False
    model: str = "pretrained_clip"
    use_nlp: bool = False
    nlp_input_dim: int = 512  # pretrained CLIP 텍스트 임베딩 차원 (projection 없음)

    ignore_checkpoint: bool = False

    # 학습 시 seen/unseen 게임 목록 — train_setting.json 에서 자동 주입됨.
    seen_games: List[str] = field(default_factory=list)
    unseen_games: List[str] = field(default_factory=list)


@dataclass
class FinetunedCLIPEvalConfig(PretrainedCLIPEvalConfig):
    """Fine-tuned CLIP PCGRL 평가용 Config."""
    wandb_project: Optional[str] = f"{PREFIX}eval_finetuned_clip"
    dir_prefix: str = "finetuned-clip-pcgrl-"
    model: str = "finetuned_clip"


@dataclass
class MGPCGRLEvalConfig(CPCGRLEvalConfig):
    """MGPCGRL 평가용 Config.

    CPCGRLConfig 와 동일한 모델/환경 설정을 EvalConfig 위에 덮어쓴다.
    """
    wandb_project: Optional[str] = f"{PREFIX}eval_mgpcgrl"

    use_decoder_reward_shaping: bool = True

    encoder: EncoderConfig = field(default_factory=lambda: EncoderConfig(model="cnnclip"))
    decoder: DecoderConfig = field(default_factory=DecoderConfig)

    use_clip: bool = True
    nlp_input_dim: int = 64  # encoder.output_dim (pretrained CLIP latent space)

    ignore_checkpoint: bool = False

    # encoder 학습 시 사용한 seen_ratio — dataset_setting.json에서 자동 주입됨.
    # 분석/로깅용으로만 사용하며, eval 데이터셋 필터링에는 적용되지 않음.
    train_seen_ratio: float = 1.0

    # exp_dir 경로 매칭용 — train과 동일한 기본값(unseen) 유지.
    # 실제 eval condition 소스는 eval_reward_decoder_mode로 별도 제어한다.
    reward_decoder_mode: str = "unseen"

    # eval 시 실제로 사용할 condition 소스.
    # "noop" → GT condition 사용 (기본값, 공정한 비교를 위해).
    # "unseen" → unseen 게임만 decoder 예측 사용.
    eval_reward_decoder_mode: str = "noop"

    # 학습 시 seen/unseen 게임 목록 — reward_decoder_config.json에서 자동 주입됨.
    seen_games: List[str] = field(default_factory=list)
    unseen_games: List[str] = field(default_factory=list)


    # ── 경로 식별용 파라미터 (encoder ckpt 해시 대체) ─────────────────────────
    # train 시 MGPCGRLConfig와 동일한 값을 지정해야 exp_dir가 일치함.
    train_unseen_abbr: Optional[str] = None
    train_unseen_ratio: Optional[float] = None
    train_seen_ratio: Optional[float] = None


@dataclass
class IPCGRLEvalConfig(CPCGRLEvalConfig):
    """IPCGRL 평가용 Config.

    CPCGRLEvalConfig 를 상속하고 BERT 임베딩 + MLP 인코더 설정을 추가한다.
    """
    use_nlp: bool = True
    vec_cont: bool = False
    model: str = "nlpconv"
    nlp_input_dim: int = 768

    encoder: EncoderConfig = field(default_factory=lambda: EncoderConfig(model="mlp"))

    dataset_reward_enum: Optional[int] = None

    wandb_project: Optional[str] = f"{PREFIX}eval_ipcgrl"


@dataclass
class CollectBufferConfig(CPCGRLConfig):
    """학습 중 trajectory 버퍼를 수집하는 Config.

    학습 50%~100% 구간(collect_start_ratio~collect_end_ratio)에서
    첫 번째 환경(env_idx=0) 기준으로 데이터를 수집하여
    실험 폴더의 buffer/ 디렉토리에 .npz 파일로 저장한다.
    """
    wandb_project: str = 'collect_buffer'
    dir_prefix: str = "buffer-"

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

    max_samples: Optional[int] = None  # dry-run용: 데이터 개수 제한 (None이면 전체 사용)



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
    max_samples: Optional[int] = None  # dry-run용: 데이터 개수 제한 (None이면 전체 사용)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)

    # instruction prefix mode: "name" (e.g. "In Zelda, ...") / "desc" / "none" (또는 None)
    instruction_prefix: Optional[str] = "name"

    # overwrite
    embed_type: str = "humanai"

    # ── Seen/Unseen 게임 분리 설정 (CLIPDecoderTrainConfig 와 동일 시맨틱) ──
    # unseen 게임 지정 (2글자 약어, e.g., "zd"=zelda, "pkzd"=pokemon+zelda).
    # None/"" 이면 기존 동작 (전체 게임을 train/test 비율로 split).
    unseen_games: Optional[str] = None
    # few-shot ratio: unseen 학습 풀 중 사용할 비율 (0.0=zero-shot, 1.0=전부)
    unseen_ratio: float = 0.0
    # seen 게임 데이터 비율 (1.0=전부 사용)
    seen_ratio: float = 1.0
    # 테스트셋 분할 시드 (재현 가능)
    split_seed: int = 42

@dataclass
class FinetunedCLIPEncoderTrainConfig(CLIPTrainConfig):
    """HuggingFace pretrained CLIP을 사용자의 (image, text) 데이터로
    파인튜닝하기 위한 Config.

    파라미터 트리 구조가 `pretrained_clip_model.ContrastiveModule` 과 동일하므로
    저장된 체크포인트를 그대로 RL 파이프라인(`apply_encoder_params`)에서 inject 할 수 있다.
    """
    wandb_project: str = f"{PREFIX}train_finetuned_clip_encoder"
    dir_prefix: str = "finetuned-clip-"

    # HF CLIP은 224×224 입력을 기대 → 좌표채널 OFF
    clip_input_channel: int = 3

    # encoder 모델 식별 (path/exp 이름 일관성). RL 단계에서는 'clip'로 로딩됨.
    encoder: EncoderConfig = field(
        default_factory=lambda: EncoderConfig(model="clip", state=True)
    )

    # HF CLIP은 미세조정 시 학습률을 매우 작게 잡고, epoch 도 적게 (5~15) 유지하는 편
    # → catastrophic forgetting 방지 + 빠른 도메인 적응
    lr: float = 5.0e-6
    weight_decay: float = 0.1
    n_epochs: int = 100
    batch_size: int = 256
    ckpt_freq: int = 50

    embed_type: str = "finetuned_clip"


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
    wandb_project: str = f'{PREFIX}train_mgpcgrl_encoder'
    dir_prefix: str = "clipdec-"

    # ── 디코더 설정 ──
    decoder: DecoderConfig = field(default_factory=DecoderConfig)

    # ── loss 가중치 ──
    contrastive_weight: float = 1.0    # contrastive loss 가중치
    cls_weight: float = 1.0            # reward_enum 분류 loss 가중치
    reg_weight: float = 1.0            # condition 회귀 loss 가중치

    # ── regression loss 종류 ──
    # "huber": Huber loss (δ=1.0), "mae": Mean Absolute Error
    regression_loss: str = "mae"

    # ── Seen/Unseen 게임 분리 설정 ──
    # unseen 게임 지정 (2글자 약어, e.g., "zd"=zelda, "pkzd"=pokemon+zelda). None=전체 seen
    unseen_games: Optional[str] = None
    # few-shot ratio: unseen 학습 풀 중 사용할 비율 (0.0=zero-shot, 1.0=전부)
    unseen_ratio: float = 0.0
    # seen 게임 데이터 비율 (1.0=전부 사용)
    seen_ratio: float = 1.0
    # 테스트셋 분할 시드 (재현 가능)
    split_seed: int = 42

    n_epochs: int = 3000

    # ── Step 기반 체크포인트 / 평가 주기 ──
    ckpt_freq: int = 1000   # 체크포인트 저장 주기 (steps, 0이면 비활성)
    scatter_freq: int = 500  # scatter plot 업로드 주기 (epochs, 0/음수면 비활성)

    # ── Unseen 게임 전용 로깅 주기 ──
    unseen_eval_freq: int = 100    # unseen regression 메트릭 로깅 주기 (epochs, 0이면 비활성)
    unseen_scatter_freq: int = 500  # unseen scatter plot 로깅 주기 (epochs, 0이면 비활성)

    # ── Unseen 평가 데이터 비율 ──
    # unseen_ratio  : 학습 데이터에 흘러들어가는 unseen 게임 데이터 비율 (train pool 기준)
    # eval_unseen_ratio : unseen_eval_freq 평가에 사용할 unseen test set 비율 (0.0~1.0, 1.0=전체)
    eval_unseen_ratio: float = 1.0


@dataclass
class CLIPDecoderUnseenConfig(CLIPDecoderTrainConfig):
    """Seen/Unseen 게임 분리 + Few-shot Ratio Sweep Config.

    Seen 게임의 전체 학습 데이터와 Unseen 게임의 가변 비율 학습 데이터로
    CLIP Decoder 모델을 학습하고, 고정된 테스트셋에서 게임별 reward_accuracy를 측정한다.
    """
    wandb_project: str = 'train_clip_decoder_unseen'
    dir_prefix: str = "clipdec-"

    # ── Unseen 게임 지정 (2글자 약어, e.g., "zd"=zelda, "pkzd"=pokemon+zelda) ──
    unseen_games: Optional[str] = None

    # ── Few-shot ratio (단일 실행용) ──
    # 0.0 = zero-shot (unseen 학습 데이터 0%), 1.0 = unseen 학습 풀 전부 사용
    unseen_ratio: float = 0.01

    # ── Seen 게임 데이터 비율 ──
    # 1.0 = seen 학습 풀 전부 사용 (기본값), 0.0 = seen 학습 데이터 0%
    seen_ratio: float = 1.0

    # ── 테스트셋 설정 ──
    # train_ratio: 학습 데이터 비율 (부모 CLIPTrainConfig 상속, 기본 0.99 → 여기서 0.8로 재정의)
    # test 비율 = 1.0 - train_ratio
    train_ratio: float = 0.99
    split_seed: int = 42              # 테스트셋 분할 시드 (재현 가능)


@dataclass
class CLIPDecoderUnseenSweepConfig(CLIPDecoderUnseenConfig):
    """Seen/Unseen 게임 분리 + Few-shot Ratio **Sweep** Config.

    CLIPDecoderUnseenConfig 를 상속하며, unseen_ratios 리스트를 추가로 정의한다.
    sweep/runnable_sweep/unseen_games.py 에서 사용한다.
    """
    # ── Few-shot ratio sweep 설정 ──
    # 0.0 = zero-shot, 1.0 = unseen 학습 풀 전부 사용
    unseen_ratios: Tuple[float, ...] = (0.0, 0.01, 0.03, 0.05, 0.1)


@dataclass
class IPCGRLEncoderMGConfig(RewardConfig):
    """IPCGRL MLP 인코더 멀티게임 사전학습 Config.

    Annotation 형식 MultiGameDataset 기반.
    - 입력: BERT(instruction) → 768-dim embedding
    - 모델: MLP 인코더 + MLP 디코더
    - 출력: condition value 회귀 (log1p + per-enum min-max 정규화)
    - unseen_games: 학습에서 제외할 게임 지정 (zero-shot 평가용)

    Usage:
        python train_ipcgrl_encoder_mg.py game=all
        python train_ipcgrl_encoder_mg.py game=all unseen_games=zd
    """
    wandb_project: Optional[str] = f"{PREFIX}train_ipcgrl_encoder"
    dir_prefix: str = "ipcgrl-enc-mg-"
    ckpt_freq: int = 10

    # BERT 설정
    use_nlp: bool = True
    nlp_input_dim: int = 768

    # Unseen 게임 설정 (2글자 약어, e.g. "zd"=zelda, "pkzd"=pokemon+zelda)
    # 빈 문자열 = 제외 없음 (전체 게임 학습)
    unseen_games: str = ""

    # Annotation 데이터셋 설정 (CLIPTrainConfig 와 동일한 변인 통제)
    # instruction_prefix mode: "name" (기본) / "desc" / "none" (또는 None)
    instruction_prefix: Optional[str] = "name"

    # MLP 인코더 (apply_encoder_model 에서 model='mlp' 분기 사용)
    encoder: EncoderConfig = field(default_factory=lambda: EncoderConfig(model="mlp"))


cs = ConfigStore.instance()
cs.store(name="config", node=Config)
cs.store(name="train_pcgrl", node=TrainConfig)
cs.store(name="cpcgrl", node=CPCGRLConfig)
cs.store(name="ipcgrl", node=IPCGRLConfig)
cs.store(name="vipcgrl", node=VIPCGRLConfig)
cs.store(name="mgpcgrl", node=MGPCGRLConfig)
cs.store(name="pretrained_clip_pcgrl", node=PretrainedCLIPPCGRLConfig)
cs.store(name="finetuned_clip_pcgrl_schema", node=FinetunedCLIPPCGRLConfig)
cs.store(name="eval_pcgrl", node=EvalConfig)
cs.store(name="eval_random_schema", node=RandomEvalConfig)
cs.store(name="eval_cpcgrl_schema", node=CPCGRLEvalConfig)
cs.store(name="eval_ipcgrl_schema", node=IPCGRLEvalConfig)
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


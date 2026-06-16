import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
import jax
import jax.numpy as jnp
from flax.training import checkpoints
from flax.training.train_state import TrainState
import optax
from tqdm import tqdm
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import umap

_t0 = time.time()

def log(msg: str):
    elapsed = time.time() - _t0
    print(f'[{elapsed:6.1f}s] {msg}', flush=True)

# ── 1. 체크포인트 경로 및 설정 ──────────────────────────────────────────────

BASE_DIR    = '/mnt/nas/mgpcgrl/mgpcgrl_unseen_ratios_0602'
UNSEEN_ABBR = 'zd'  # zelda unseen / 나머지 4개 seen

CKPTS = [
    ('ur=0.1', f'{BASE_DIR}/clipdec-game-all_unseen-{UNSEEN_ABBR}_ur-0.1_exp-def_0/ckpts'),
    ('ur=0.5', f'{BASE_DIR}/clipdec-game-all_unseen-{UNSEEN_ABBR}_ur-0.5_exp-def_0/ckpts'),
    ('ur=1.0', f'{BASE_DIR}/clipdec-game-all_unseen-{UNSEEN_ABBR}_ur-1.0_exp-def_0/ckpts'),
]

ENCODER_OUTPUT_DIM = 64
NUM_REWARD_CLASSES = 5
TOKEN_MAX_LEN      = 77
BATCH_SIZE         = 256
N_SAMPLES_PER_GAME = 50

SAVE_DIR = os.path.dirname(os.path.abspath(__file__))  # notebooks/

# ── 2. 인코더 모듈 생성 ──────────────────────────────────────────────────────

log('importing encoder modules...')
from conf.config import EncoderConfig, DecoderConfig
from encoder.clip_model import get_cnnclip_decoder_encoder

encoder_config = EncoderConfig(
    model='cnnclip',
    state=True,
    output_dim=ENCODER_OUTPUT_DIM,
    token_max_len=TOKEN_MAX_LEN,
    freeze_text_enc=True,
    freeze_state_enc=False,
    dropout_rate=0.3,
)
decoder_config = DecoderConfig(
    num_reward_classes=NUM_REWARD_CLASSES,
    hidden_dim=128,
    num_layers=2,
    cnn_reward_enum_onehot=True,
)

log('building encoder module...')
module, _ = get_cnnclip_decoder_encoder(
    encoder_config,
    decoder_config=decoder_config,
    RL_training=False,
)

log('initializing dummy variables...')
rng = jax.random.PRNGKey(0)
dummy_ids  = jnp.ones((1, TOKEN_MAX_LEN), dtype=jnp.int32)
dummy_mask = jnp.ones((1, TOKEN_MAX_LEN), dtype=jnp.int32)
dummy_pix  = jnp.ones((1, 16, 16, 7),    dtype=jnp.float32)
dummy_enum = jnp.zeros((1,),              dtype=jnp.int32)

variables_template = module.init(
    rng, dummy_ids, dummy_mask, dummy_pix,
    reward_enum=dummy_enum, mode='text_state', training=False,
)
log(f'params keys: {list(variables_template["params"].keys())}')


def load_variables(ckpt_dir: str) -> dict:
    step_dirs = [d for d in os.listdir(ckpt_dir) if d.isdigit()]
    latest    = max(step_dirs, key=int)
    restore_dir = os.path.join(ckpt_dir, latest)
    log(f'  step {latest} from {restore_dir}')
    dummy_state = TrainState.create(
        apply_fn=module.apply,
        params=variables_template,
        tx=optax.identity(),
    )
    restored = checkpoints.restore_checkpoint(restore_dir, target=dummy_state, prefix='')
    return restored.params


log('loading all checkpoints...')
all_variables = {}
for label, ckpt_dir in CKPTS:
    log(f'  [{label}]')
    all_variables[label] = load_variables(ckpt_dir)
log('all checkpoints loaded.')

log('loading CLIP tokenizer...')
from transformers import CLIPTokenizer
tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32')
log('tokenizer loaded.')

# ── 3. 데이터셋 로드 및 전처리 ───────────────────────────────────────────────

log('importing dataset modules...')
from dataset.multigame import MultiGameDataset
from instruct_rl.utils.level_processing_utils import map2onehot_batch, add_coord_channel_batch
from dataset.multigame.tile_utils import NUM_CATEGORIES

log('loading MultiGameDataset...')
ds = MultiGameDataset(
    include_dungeon=True,
    include_pokemon=True,
    include_doom=True,
    include_sokoban=True,
    include_zelda=True,
    use_tile_mapping=True,
)
log(f'dataset loaded: {len(ds)} total samples')

GAMES        = ['dungeon', 'pokemon', 'sokoban', 'doom', 'zelda']  # zelda만 unseen
UNSEEN_GAMES = {'zelda'}

all_arrays, all_texts, all_games = [], [], []

log('sampling per-game levels with instructions...')
for game in GAMES:
    samples = [s for s in ds.by_game(game) if s.instruction is not None]
    rng_np  = np.random.default_rng(42)
    idx     = rng_np.choice(len(samples), size=min(N_SAMPLES_PER_GAME, len(samples)), replace=False)
    samples = [samples[i] for i in idx]
    for s in samples:
        all_arrays.append(s.array)
        all_texts.append(s.instruction)
        all_games.append(game)
    log(f'  {game}: {len(samples)} samples')

log(f'total: {len(all_arrays)} — converting to onehot...')
level_arrays = np.stack(all_arrays, axis=0)
pixel_values = map2onehot_batch(level_arrays, num_classes=NUM_CATEGORIES)
pixel_values = add_coord_channel_batch(pixel_values)
pixel_values = np.array(pixel_values, dtype=np.float32)
game_arr     = np.array(all_games)
log(f'pixel_values: {pixel_values.shape}')

# ── 4. 인코더 임베딩 추출 함수 ───────────────────────────────────────────────

@jax.jit
def _encode_level(pix, vars_):
    return module.apply(
        vars_, pixel_values=pix, reward_enum=None,
        mode='state', training=False,
    )['state_embed']

@jax.jit
def _encode_text(ids, mask, vars_):
    return module.apply(
        vars_, ids, mask, dummy_pix,
        reward_enum=None, mode='text', training=False,
    )['text_embed']


def get_level_embeddings(vars_):
    N, n_batch = len(pixel_values), (len(pixel_values) + BATCH_SIZE - 1) // BATCH_SIZE
    parts = []
    for i, start in enumerate(range(0, N, BATCH_SIZE)):
        end = min(start + BATCH_SIZE, N)
        parts.append(np.array(_encode_level(jnp.array(pixel_values[start:end]), vars_)))
        log(f'    level {i+1}/{n_batch}')
    return np.concatenate(parts, axis=0)


def get_text_embeddings(vars_):
    N, n_batch = len(all_texts), (len(all_texts) + BATCH_SIZE - 1) // BATCH_SIZE
    parts = []
    for i, start in enumerate(range(0, N, BATCH_SIZE)):
        end   = min(start + BATCH_SIZE, N)
        enc   = tokenizer(all_texts[start:end], padding='max_length',
                          max_length=TOKEN_MAX_LEN, truncation=True, return_tensors='np')
        ids   = jnp.array(enc['input_ids'],      dtype=jnp.int32)
        mask  = jnp.array(enc['attention_mask'], dtype=jnp.int32)
        parts.append(np.array(_encode_text(ids, mask, vars_)))
        log(f'    text  {i+1}/{n_batch}')
    return np.concatenate(parts, axis=0)


# ── 5. 각 체크포인트별 임베딩 + UMAP ─────────────────────────────────────────

results = {}   # label → {'emb2d', 'level_embeds', 'text_embeds'}

for label, _ in CKPTS:
    vars_ = all_variables[label]

    log(f'[{label}] encoding level...')
    level_embeds = get_level_embeddings(vars_)
    log(f'[{label}] encoding text...')
    text_embeds  = get_text_embeddings(vars_)

    all_embeds = np.concatenate([level_embeds, text_embeds], axis=0)
    all_types  = np.array(['level'] * len(level_embeds) + ['text'] * len(text_embeds))

    log(f'[{label}] fitting UMAP on {all_embeds.shape}...')
    reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1, random_state=42)
    emb2d   = reducer.fit_transform(all_embeds)
    log(f'[{label}] UMAP done → {emb2d.shape}')

    # cosine similarity 진단
    def normalize(x):
        return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
    lnorm = normalize(level_embeds)
    tnorm = normalize(text_embeds)
    matched_sim = np.sum(lnorm * tnorm, axis=1).mean()
    rand_idx    = np.random.default_rng(0).permutation(len(tnorm))
    random_sim  = np.sum(lnorm * tnorm[rand_idx], axis=1).mean()
    log(f'[{label}] cosine sim — matched: {matched_sim:.4f}  random: {random_sim:.4f}  gap: {matched_sim-random_sim:.4f}')

    results[label] = {
        'emb2d':        emb2d,
        'all_types':    all_types,
        'level_embeds': level_embeds,
        'text_embeds':  text_embeds,
    }

# ── 6. 시각화 ────────────────────────────────────────────────────────────────

GAME_COLORS = {
    'dungeon': '#1f77b4',   # 파랑  (seen)
    'pokemon': '#2ca02c',   # 초록  (seen)
    'sokoban': '#9467bd',   # 보라  (seen)
    'doom':    '#17becf',   # 하늘  (seen)
    'zelda':   '#d62728',   # 빨강  (unseen)
}
TYPE_MARKERS = {'level': 'o', 'text': 'x'}
SEEN_GAMES   = [g for g in GAMES if g not in UNSEEN_GAMES]
UNSEEN_LIST  = [g for g in GAMES if g in UNSEEN_GAMES]
all_games2   = np.concatenate([game_arr, game_arr], axis=0)

n_plots = len(CKPTS)
fig, axes = plt.subplots(1, n_plots, figsize=(9 * n_plots, 8))

for ax, (label, _) in zip(axes, CKPTS):
    emb2d     = results[label]['emb2d']
    all_types = results[label]['all_types']

    for game in GAMES:
        for typ, marker in TYPE_MARKERS.items():
            mask = (all_games2 == game) & (all_types == typ)
            ax.scatter(
                emb2d[mask, 0], emb2d[mask, 1],
                c=GAME_COLORS[game], marker=marker,
                s=80, alpha=0.85, linewidths=1.2, edgecolors='white',
            )

    ax.set_title(label, fontsize=13, fontweight='bold')
    ax.set_xlabel('UMAP dim 1')
    ax.set_ylabel('UMAP dim 2')
    ax.set_aspect('equal', adjustable='datalim')

# ── Legend (마지막 축에만) ────────────────────────────────────────────────────

def game_handle(game):
    return mlines.Line2D(
        [], [], color=GAME_COLORS[game], marker='o', linestyle='None',
        markersize=8, markeredgecolor='white', markeredgewidth=1.2, label=game,
    )

seen_title   = mpatches.Patch(visible=False, label='Seen')
unseen_title = mpatches.Patch(visible=False, label='Unseen')
divider      = mpatches.Patch(visible=False, label='─' * 15)
type_level   = mlines.Line2D([], [], color='grey', marker='o', linestyle='None',
                              markersize=8, markeredgecolor='white', markeredgewidth=1.2, label='Level')
type_text    = mlines.Line2D([], [], color='grey', marker='x', linestyle='None',
                              markersize=8, markeredgewidth=1.8, label='Text')

ax_last = axes[-1]
game_leg = ax_last.legend(
    handles=[seen_title] + [game_handle(g) for g in SEEN_GAMES]
            + [divider, unseen_title] + [game_handle(g) for g in UNSEEN_LIST],
    fontsize=9, handlelength=1.5, loc='upper right', title='Game',
)
for text, handle in zip(game_leg.get_texts(), game_leg.legend_handles):
    if handle in (seen_title, unseen_title):
        text.set_fontweight('bold')
    elif handle is divider:
        text.set_color('grey')
ax_last.add_artist(game_leg)
ax_last.legend(handles=[type_level, type_text], fontsize=9,
               handlelength=1.5, loc='lower right', title='Type')

fig.suptitle(
    f'UMAP of State Embeddings  |  unseen={UNSEEN_ABBR}  |  mgpcgrl_unseen_ratios_0602',
    fontsize=13, y=1.01,
)
plt.tight_layout()

log('saving figure...')
save_path = os.path.join(SAVE_DIR, f'umap_{UNSEEN_ABBR}_ur_comparison.png')
fig.savefig(save_path, dpi=150, bbox_inches='tight')
plt.close(fig)
log(f'saved → {save_path}')

"""
train_finetuned_clip_encoder.py
================================
HuggingFace pretrained CLIP을 사용자의 (맵 이미지, 텍스트 지시) 페어로
파인튜닝하는 entrypoint.

- `MultiGameDataset` + `CLIPDatasetBuilder` 로 224×224 픽셀 이미지 + 토크나이즈된
  텍스트를 구성 (기존 `train_clip.py` 와 동일한 변인 통제).
- 인코더는 `encoder.finetuned_clip_model.get_finetuned_clip_encoder()` → trainable
  버전의 `ContrastiveModule` (파라미터 트리 구조는 RL 쪽 `ContrastiveModule`
  과 완전히 동일).
- 저장된 체크포인트(`pretrained_encoders/finetuned-clip-...`)는 RL 학습 시
  `encoder.ckpt_name=finetuned-clip-...` 로 그대로 inject 된다 (수정 없음).

실행:
    python -m train_finetuned_clip_encoder
    python -m train_finetuned_clip_encoder game=all unseen_games=zd unseen_ratio=0.0
"""
import datetime
import json
import logging
import math
import os
import shutil
from collections import deque
from copy import deepcopy
from functools import partial
from os.path import basename

import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from flax.training import checkpoints
from flax.training.train_state import TrainState
from tqdm import tqdm
from transformers import CLIPProcessor

from conf.config import FinetunedCLIPEncoderTrainConfig
from dataset.multigame import MultiGameDataset
from encoder.data import (CLIPContrastiveBatch, CLIPDataset, CLIPDatasetBuilder,
                          CLIPEmbedData, create_clip_batch)
from encoder.finetuned_clip_model import get_finetuned_clip_encoder
from encoder.schedular import create_learning_rate_fn
from encoder.utils.path import get_ckpt_dir, init_config
from instruct_rl.utils.img_preprocess import (clip_batch_preprocess,
                                              render_level_from_arr)
from instruct_rl.utils.logger import get_wandb_name
from encoder.data.split import (build_train_indices_for_ratio,
                                parse_unseen_game_names, split_dataset_by_game,
                                subset_clip_dataset)

log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logger = logging.getLogger(basename(__file__))
logger.setLevel(getattr(logging, log_level, logging.INFO))


# ── HF CLIP용 RGB pixel_values 변환 ───────────────────────────────────────────
#
# CLIPDatasetBuilder 가 생성하는 pixel_values 는 cnnclip 용 (B, 16, 16, C_onehot+2)
# 포맷이지만 HuggingFace CLIP 은 (B, 224, 224, 3) RGB 입력을 요구한다.
# → one-hot 채널에서 raw tile enum 을 복원 → 타일 렌더링 → 224×224 정규화.

def _render_rgb_pixel_values(pixel_values_5ch: np.ndarray,
                             num_tile_classes: int = 3) -> np.ndarray:
    """(N, 16, 16, num_classes+2) one-hot+coord → (N, 224, 224, 3) HF CLIP RGB."""
    onehot = jnp.asarray(pixel_values_5ch[..., :num_tile_classes])
    raw_enum = jnp.argmax(onehot, axis=-1) + 1  # (N, 16, 16) - tile enum starts from 1
    rendered = jax.vmap(render_level_from_arr)(raw_enum)  # (N, H_px, W_px, 3) uint8
    preprocessed = clip_batch_preprocess(rendered.astype(jnp.float32))
    return np.asarray(preprocessed)


def _replace_pixel_values_with_rgb(dataset: CLIPDataset,
                                   num_tile_classes: int = 3,
                                   chunk_size: int = 256) -> CLIPDataset:
    """CLIPDataset 의 pixel_values 만 HF CLIP RGB 포맷으로 대체한 새 CLIPDataset 반환."""
    n = len(dataset.class_ids)
    if n == 0:
        # 빈 데이터셋: shape만 맞춰 둠
        rgb = np.zeros((0, 224, 224, 3), dtype=np.float32)
    else:
        chunks = []
        for s in range(0, n, chunk_size):
            chunks.append(_render_rgb_pixel_values(
                dataset.pixel_values[s:s + chunk_size], num_tile_classes
            ))
        rgb = np.concatenate(chunks, axis=0)
    return CLIPDataset(
        class_ids=dataset.class_ids,
        reward_cond=dataset.reward_cond,
        input_ids=dataset.input_ids,
        attention_masks=dataset.attention_masks,
        pixel_values=rgb,
        is_train=dataset.is_train,
        reward_enum_targets=getattr(dataset, "reward_enum_targets", None),
        condition_targets=getattr(dataset, "condition_targets", None),
        quantized_condition_targets=getattr(dataset, "quantized_condition_targets", None),
    )


# ── train_step (train_clip.py 와 동일한 contrastive loss) ────────────────────

@partial(jax.jit, static_argnums=(3, 4))
def train_step(train_state: TrainState, batch: CLIPContrastiveBatch,
               rng_key: jax.random.PRNGKey, is_train: bool = True,
               mode: str = "text_state"):
    rng_key, dropout_rng = jax.random.split(rng_key)

    def pair_loss(a, b, temperature):
        logits = jnp.matmul(a, b.T) / jnp.exp(temperature)
        a2b_logp = jax.nn.log_softmax(logits, axis=1)
        b2a_logp = jax.nn.log_softmax(logits, axis=0)
        a2b_pos = a2b_logp - 1e9 * (1 - batch.duplicate_matrix)
        b2a_pos = b2a_logp - 1e9 * (1 - batch.duplicate_matrix)
        a2b_loss = -jnp.mean(jax.scipy.special.logsumexp(a2b_pos, axis=1))
        b2a_loss = -jnp.mean(jax.scipy.special.logsumexp(b2a_pos, axis=0))
        a2b_top1 = jnp.mean(jnp.argmax(a2b_logp, axis=1) == jnp.argmax(a2b_pos, axis=1))
        b2a_top1 = jnp.mean(jnp.argmax(b2a_logp, axis=0) == jnp.argmax(b2a_pos, axis=0))
        return a2b_loss, b2a_loss, a2b_top1, b2a_top1

    def loss_fn(params):
        outputs = train_state.apply_fn(
            params,
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            pixel_values=batch.pixel_values,
            mode=mode,
        )
        text_embed = outputs["text_embed"]
        state_embed = outputs.get("state_embed", jnp.zeros_like(text_embed))
        temp = jnp.clip(outputs["text_state_temperature"], jnp.log(0.01), jnp.log(100))
        s2t_loss, t2s_loss, s2t_top1, t2s_top1 = pair_loss(state_embed, text_embed, temp)
        loss = 0.5 * (s2t_loss + t2s_loss)
        metrics = {
            "total_loss": loss,
            "state2text_loss": s2t_loss,
            "text2state_loss": t2s_loss,
            "state2text_top1": s2t_top1,
            "text2state_top1": t2s_top1,
            "text_state_temperature": temp,
            "state_embed": state_embed,
            "text_embed": text_embed,
        }
        return loss, metrics

    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(train_state.params)
    train_state = jax.lax.cond(
        is_train,
        lambda _: train_state.apply_gradients(grads=grads),
        lambda _: train_state,
        operand=None,
    )
    return train_state, loss, metrics, rng_key


# ── train state 생성 (train_clip.py 의 구조 차용) ───────────────────────────

def get_train_state(config: FinetunedCLIPEncoderTrainConfig, rng_key):
    lr_sched = create_learning_rate_fn(config, config.lr, config.steps_per_epoch)
    encoder, pretrained_params = get_finetuned_clip_encoder(config.encoder)

    def replace_params(params, key, replacement):
        for k in list(params.keys()):
            if k == key:
                params[k] = replacement
                logger.info(f"replaced `{key}` in params")
                return
            if isinstance(params[k], type(params)):
                replace_params(params[k], key, replacement)

    rng_key, init_rng = jax.random.split(rng_key)
    input_ids = jnp.ones((1, config.encoder.token_max_len), dtype=jnp.int32)
    attention_mask = jnp.ones((1, config.encoder.token_max_len), dtype=jnp.int32)
    pixel_values = jnp.ones((1, 224, 224, 3), dtype=jnp.float32)
    params = encoder.init(init_rng, input_ids=input_ids, attention_mask=attention_mask,
                          pixel_values=pixel_values, mode=config.encoder.mode)
    for key, val in pretrained_params.items():
        replace_params(params, key, val)

    tx = optax.adamw(learning_rate=lr_sched, weight_decay=config.weight_decay)
    state = TrainState.create(apply_fn=encoder.apply, params=params, tx=tx)
    return state, lr_sched


def save_checkpoint(config, state, step):
    ckpt_dir = os.path.abspath(get_ckpt_dir(config))
    checkpoints.save_checkpoint(ckpt_dir, target=state, prefix="",
                                step=step, overwrite=True, keep=1)
    logger.info(f"Checkpoint saved at step {step} → {ckpt_dir}")


# ── 메인 학습 루프 ──────────────────────────────────────────────────────────

def make_train(config: FinetunedCLIPEncoderTrainConfig):
    def train(rng_key):
        rng_key, subkey = jax.random.split(rng_key)
        dataset = MultiGameDataset(
            include_dungeon=config.include_dungeon,
            include_pokemon=config.include_pokemon,
            include_sokoban=config.include_sokoban,
            include_doom=config.include_doom,
            include_doom2=config.include_doom2,
            include_zelda=config.include_zelda,
            max_samples_per_game=config.max_samples_per_game,
            max_samples_seed=config.max_samples_seed,
        )

        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        builder = CLIPDatasetBuilder(
            processor=processor, paired_data=dataset, rng_key=subkey,
            max_len=config.encoder.token_max_len, train_ratio=config.train_ratio,
            instruction_prefix=config.instruction_prefix,
            tile_offset=getattr(config.encoder, "tile_offset", 0),
        )
        train_ds, test_ds = builder.get_split_dataset()

        # ── Seen/Unseen 게임 분리 (train_clip.py 와 동일) ──────────────────
        if getattr(config, "unseen_games", None):
            full_dataset = builder.get_dataset()
            unseen_set = parse_unseen_game_names(config.unseen_games)
            all_game_names = np.array([rc["game_name"] for rc in full_dataset.reward_cond])
            unique_games = sorted(set(all_game_names))
            seen = [g for g in unique_games if g not in unseen_set]
            unseen = [g for g in unique_games if g in unseen_set]

            logger.info("=" * 70)
            logger.info("  [Seen/Unseen Split] seen=%s, unseen=%s, seen_ratio=%.3f, unseen_ratio=%.3f",
                        seen, unseen, config.seen_ratio, config.unseen_ratio)

            os.makedirs(config.exp_dir, exist_ok=True)
            with open(os.path.join(config.exp_dir, "dataset_setting.json"), "w") as f:
                json.dump({
                    "all_games": unique_games, "seen_games": seen, "unseen_games": unseen,
                    "unseen_ratio": config.unseen_ratio, "seen_ratio": config.seen_ratio,
                }, f, indent=2, ensure_ascii=False)

            game_train_pool, game_test, _ = split_dataset_by_game(
                full_dataset, unseen_set,
                test_ratio=1.0 - config.train_ratio, test_seed=config.split_seed,
            )
            test_indices = np.concatenate([game_test[g] for g in sorted(game_test.keys())])
            train_indices = build_train_indices_for_ratio(
                game_train_pool, unseen_set,
                ratio=config.unseen_ratio, seen_ratio=config.seen_ratio,
            )
            if len(train_indices) == 0:
                logger.warning("0 training samples — fallback to a single placeholder sample")
                train_indices = np.array([0])
            train_ds = subset_clip_dataset(full_dataset, train_indices)
            test_ds = subset_clip_dataset(full_dataset, test_indices)
            logger.info("  Train=%d, Test=%d", len(train_indices), len(test_indices))

        # dry-run
        if config.max_samples is not None:
            n = config.max_samples

            def _slice(ds, n):
                n = min(n, len(ds.class_ids))
                return CLIPDataset(
                    class_ids=ds.class_ids[:n],
                    reward_cond=ds.reward_cond[:n],
                    input_ids=ds.input_ids[:n],
                    attention_masks=ds.attention_masks[:n],
                    pixel_values=ds.pixel_values[:n],
                    is_train=ds.is_train[:n],
                    quantized_condition_targets=ds.quantized_condition_targets[:n]
                    if ds.quantized_condition_targets is not None else None,
                )
            train_ds = _slice(train_ds, n)
            test_ds = _slice(test_ds, n)

        n_train, n_test = len(train_ds.class_ids), len(test_ds.class_ids)
        n_tr_b = max(1, math.ceil(n_train / config.batch_size))
        n_te_b = max(1, math.ceil(n_test / config.batch_size))
        config.steps_per_epoch = n_tr_b

        mode = "text_state" if config.encoder.state else "text"
        config.encoder.mode = mode

        # ── HF CLIP 입력 포맷으로 pixel_values 변환 (one-time) ───────────────
        # CLIPDatasetBuilder 는 (B, 16, 16, num_classes+2) one-hot+coord 를 만들지만,
        # HuggingFace pretrained CLIP 은 (B, 224, 224, 3) RGB 를 받는다.
        # → raw tile enum 복원 → 타일 렌더 → 224×224 정규화 후 교체.
        # 좌표 채널 2 개 + clip_input_channel(=raw) 으로부터 one-hot 채널 수 계산.
        _num_tile_classes = max(1, int(getattr(config, "clip_input_channel", 5)) - 2)
        logger.info("Rendering RGB pixel_values for HF CLIP (num_tile_classes=%d, n_train=%d, n_test=%d) ...",
                    _num_tile_classes, n_train, n_test)
        train_ds = _replace_pixel_values_with_rgb(train_ds, _num_tile_classes)
        test_ds = _replace_pixel_values_with_rgb(test_ds, _num_tile_classes)
        logger.info("  train pixel_values shape: %s", train_ds.pixel_values.shape)
        logger.info("  test  pixel_values shape: %s", test_ds.pixel_values.shape)

        train_state, lr_sched = get_train_state(config, subkey)
        logger.info("Start fine-tuning HF CLIP encoder ...")

        for epoch in range(config.n_epochs):
            tr = {"total": 0., "s2t": 0., "t2s": 0., "s2t_top1": 0., "t2s_top1": 0., "temp": 0.}
            va = deepcopy(tr)
            rng_key, subkey = jax.random.split(rng_key)

            with tqdm(total=n_tr_b + n_te_b, desc=f"Epoch {epoch+1}") as pbar:
                # Train
                for clip_batch in create_clip_batch(train_ds, config.batch_size, rng_key=subkey):
                    clip_batch = jax.device_put(clip_batch)
                    train_state, loss, m, rng_key = train_step(
                        train_state, clip_batch, rng_key=subkey,
                        is_train=True, mode=mode,
                    )
                    tr["total"] += float(loss)
                    tr["s2t"] += float(m["state2text_loss"])
                    tr["t2s"] += float(m["text2state_loss"])
                    tr["s2t_top1"] += float(m["state2text_top1"])
                    tr["t2s_top1"] += float(m["text2state_top1"])
                    tr["temp"] += float(m["text_state_temperature"])
                    pbar.update(1)
                # Val
                for clip_batch in create_clip_batch(test_ds, config.batch_size, rng_key=subkey):
                    clip_batch = jax.device_put(clip_batch)
                    _, loss, m, rng_key = train_step(
                        train_state, clip_batch, rng_key=subkey,
                        is_train=False, mode=mode,
                    )
                    va["total"] += float(loss)
                    va["s2t"] += float(m["state2text_loss"])
                    va["t2s"] += float(m["text2state_loss"])
                    va["s2t_top1"] += float(m["state2text_top1"])
                    va["t2s_top1"] += float(m["text2state_top1"])
                    pbar.update(1)

            tr = {k: v / n_tr_b for k, v in tr.items()}
            va = {k: v / n_te_b for k, v in va.items()}

            if (epoch + 1) % config.ckpt_freq == 0:
                save_checkpoint(config, train_state, step=epoch + 1)

            if wandb.run is not None:
                wandb.log({
                    "total/train_loss": tr["total"], "total/val_loss": va["total"],
                    "train/state2text_loss": tr["s2t"], "train/text2state_loss": tr["t2s"],
                    "train/state2text_top1": tr["s2t_top1"], "train/text2state_top1": tr["t2s_top1"],
                    "val/state2text_loss": va["s2t"], "val/text2state_loss": va["t2s"],
                    "val/state2text_top1": va["s2t_top1"], "val/text2state_top1": va["t2s_top1"],
                    "train/temperature": tr["temp"],
                    "total/epoch": epoch, "total/lr": float(lr_sched(train_state.step)),
                })

    return train


@hydra.main(version_base=None, config_path="./conf",
            config_name="train_finetuned_clip_encoder")
def main(config: FinetunedCLIPEncoderTrainConfig):
    if config.encoder.model is None:
        config.encoder.model = "clip"
    config = init_config(config)

    rng_key = jax.random.PRNGKey(config.seed)
    np.random.seed(config.seed)

    from instruct_rl.utils.env_loader import get_wandb_key
    wandb_key = get_wandb_key()
    if wandb_key:
        dt = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        wandb_id = f"{get_wandb_name(config)}-{dt}"
        wandb.login(key=wandb_key)
        wandb.init(
            project=config.wandb_project, group=config.instruct,
            entity=config.wandb_entity, name=get_wandb_name(config),
            id=wandb_id, save_code=True,
        )
        wandb.config.update(dict(config), allow_val_change=True)

    exp_dir = config.exp_dir
    logger.info(f"jax devices: {jax.devices()}")
    logger.info(f"running experiment at {exp_dir}")

    if config.overwrite and os.path.exists(exp_dir):
        shutil.rmtree(exp_dir)
    os.makedirs(exp_dir, exist_ok=True)

    make_train(config)(rng_key)

    if wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()

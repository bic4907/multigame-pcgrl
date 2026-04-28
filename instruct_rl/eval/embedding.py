"""
embedding.py
============
eval CSV → Instruct 변환.
CLIP / cnnclip / 기본 BERT 임베딩 세 가지 케이스를 처리한다.
"""
import logging

import jax.numpy as jnp
import wandb
from transformers import CLIPProcessor

from envs.pcgrl_env import PCGRLObs
from instruct_rl.dataclass import Instruct
from instruct_rl.human_data.dataset import DatasetManager
from instruct_rl.utils.level_processing_utils import add_coord_channel_batch, map2onehot_batch
from instruct_rl.vision.data.render import render_array_batch

logger = logging.getLogger(__name__)


def prepare_instruct(config, network, runner_state, instruct_df, init_x) -> Instruct:
    """instruct_df (CSV 로드 결과) → Instruct 반환.

    config.encoder.model 에 따라:
      - 'clip'    : 텍스트 토크나이즈 → 네트워크 forward → text embedding
      - 'cnnclip' : eval_modality ('text' or 'state') → 네트워크 forward → embedding
      - 기타(bert 등): CSV의 embed_* 컬럼을 그대로 사용
    """
    # ── 기본 embedding (CSV embed_* 컬럼) ─────────────────────────────────────
    embedding_df = instruct_df.filter(regex="embed_*")
    embedding_df = embedding_df.reindex(
        sorted(embedding_df.columns, key=lambda x: int(x.split("_")[-1])),
        axis=1,
    )
    embedding = jnp.array(embedding_df.to_numpy())

    if config.nlp_input_dim > embedding.shape[1]:
        embedding = jnp.pad(
            embedding,
            ((0, 0), (0, config.nlp_input_dim - embedding.shape[1])),
            mode="constant",
        )

    # ── condition ─────────────────────────────────────────────────────────────
    condition_df = instruct_df.filter(regex=r'(?<!sub_)condition_')
    condition_df = condition_df.reindex(
        sorted(condition_df.columns, key=lambda x: int(x.split("_")[-1])),
        axis=1,
    )
    condition = jnp.array(condition_df.to_numpy())

    # ── reward_enum ───────────────────────────────────────────────────────────
    reward_enum_list = [
        [int(d) for d in str(num)]
        for num in instruct_df["reward_enum"].to_list()
    ]
    max_len = max(len(x) for x in reward_enum_list)
    reward_enum = jnp.array([x + [0] * (max_len - len(x)) for x in reward_enum_list])

    # ── CLIP / cnnclip embedding override ────────────────────────────────────
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    if config.encoder.model == 'clip':
        input_ids, attention_mask = _tokenize(processor, instruct_df)
        instr_x = _make_instr_obs(init_x, input_ids, attention_mask,
                                   pixel_values=jnp.zeros(
                                       (input_ids.shape[0], 224, 224, config.clip_input_channel),
                                       dtype=jnp.float32,
                                   ))
        _, _, _, embedding, _, _ = network.apply(
            runner_state.train_state.params, x=instr_x,
            return_text_embed=True, return_state_embed=False,
        )
        logger.info(f"Generated clip text embeddings for {input_ids.shape[0]} instructions.")

    elif config.encoder.model == 'cnnclip':
        logger.info(f"Generating cnnclip embeddings with `{config.eval_modality}` modality.")

        if config.eval_modality == 'text':
            input_ids, attention_mask = _tokenize(processor, instruct_df)
            pixel_values = jnp.zeros(
                (input_ids.shape[0], 16, 16, config.clip_input_channel),
                dtype=jnp.float32,
            )
            logger.info(f"Generated cnnclip text embeddings for {input_ids.shape[0]} instructions.")

        elif config.eval_modality == 'state':
            input_ids = jnp.zeros((1, 77), dtype=jnp.int32)
            attention_mask = jnp.ones((1, 77), dtype=jnp.int32)

            language_instr_list = instruct_df["instruction"].to_list()
            dataset_mgr = DatasetManager(config.eval_human_demo_path)
            input_level_raw = dataset_mgr.get_levels(
                instructions=language_instr_list, n=1, squeeze_n=True
            )
            input_level = add_coord_channel_batch(map2onehot_batch(input_level_raw))
            pixel_values = input_level

            assert input_level.shape[-1] == config.clip_input_channel, (
                f"Expected {config.clip_input_channel} channels, got {input_level.shape[-1]}"
            )
            if wandb.run:
                for i, img in enumerate(render_array_batch(input_level_raw)):
                    wandb.log({f"CondState/reward_{i}": wandb.Image(img)})
            logger.info(f"Generated cnnclip state embeddings for {input_level.shape[0]} instructions.")
        else:
            raise ValueError(f"Unknown eval_modality: {config.eval_modality}")

        instr_x = _make_instr_obs(init_x, input_ids, attention_mask, pixel_values=pixel_values)
        _, _, _, text_embed, state_embed = network.apply(
            runner_state.train_state.params, x=instr_x,
            return_text_embed=(config.eval_modality == 'text'),
            return_state_embed=(config.eval_modality == 'state'),
        )
        embedding = text_embed if config.eval_modality == 'text' else state_embed

    return Instruct(
        reward_i=reward_enum,
        condition=condition,
        embedding=embedding,
        condition_id=None,
    )


# ── helpers ───────────────────────────────────────────────────────────────────

def _tokenize(processor, instruct_df):
    tokenized = processor(
        text=instruct_df["instruction"].to_list(),
        return_tensors="jax",
        padding="max_length",
        truncation=True,
        max_length=77,
    )
    return tokenized['input_ids'], tokenized['attention_mask']


def _make_instr_obs(init_x, input_ids, attention_mask, pixel_values):
    n = input_ids.shape[0]
    return PCGRLObs(
        map_obs=jnp.repeat(init_x.map_obs, n, axis=0),
        past_map_obs=None,
        flat_obs=jnp.repeat(init_x.flat_obs, n, axis=0),
        nlp_obs=jnp.repeat(init_x.nlp_obs, n, axis=0),
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
    )


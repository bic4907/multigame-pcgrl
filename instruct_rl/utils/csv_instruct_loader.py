"""
instruct_rl/utils/csv_instruct_loader.py
=========================================
CSV 파일 기반 Instruct 빌더.
train_cpcgrl.py 의 make_train/train 내부에 있던 CSV → Instruct 변환 로직을 분리.

encoder model (clip / cnnclip) 사용 시 네트워크 forward 가 필요하므로,
network 관련 인자를 외부에서 주입받는다.
"""
from __future__ import annotations

from os.path import abspath, dirname, join

import jax.numpy as jnp
import pandas as pd
from transformers import CLIPProcessor

from envs.pcgrl_env import PCGRLObs
from instruct_rl.dataclass import Instruct
from instruct_rl.human_data.dataset import DatasetManager
from instruct_rl.utils.log_utils import get_logger

logger = get_logger(__file__)

# CSV 파일 기준 경로 (train_cpcgrl.py 와 동일한 위치를 가정)
_PROJECT_ROOT = dirname(dirname(dirname(abspath(__file__))))


def load_csv_instruct(
    config,
    *,
    network=None,
    network_params=None,
    init_x=None,
):
    """CSV 파일에서 train/test Instruct 를 빌드한다.

    Parameters
    ----------
    config : CPCGRLConfig / TrainConfig
        instruct_csv, nlp_input_dim, encoder.model, multimodal_condition 등 필요.
    network : flax Module, optional
        encoder model 이 clip / cnnclip 일 때 임베딩 생성에 필요.
    network_params : dict, optional
        network.apply 에 넘길 파라미터.
    init_x : PCGRLObs, optional
        dummy observation (repeat 용).

    Returns
    -------
    (train_inst, test_inst) : tuple[Instruct, Instruct]
    """
    csv_path = abspath(
        join(_PROJECT_ROOT, "instruct", f"{config.instruct_csv}.csv")
    )
    logger.info(f"Loading instruct CSV: {csv_path}")

    instruct_df = pd.read_csv(csv_path)
    instruct_df["cond_id"] = (
        (instruct_df.index // 4) + (instruct_df["reward_enum"] - 1) * 8
    )

    processor = None
    if config.encoder.model in ("clip", "cnnclip"):
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    train_inst = _build_instruct_from_df(
        instruct_df, is_train=True, config=config,
        processor=processor, network=network,
        network_params=network_params, init_x=init_x,
    )
    test_inst = _build_instruct_from_df(
        instruct_df, is_train=False, config=config,
        processor=processor, network=network,
        network_params=network_params, init_x=init_x,
    )

    logger.info(
        f"CSV instruct loaded — train: {train_inst.reward_i.shape[0]}, "
        f"test: {test_inst.reward_i.shape[0]}"
    )
    return train_inst, test_inst


# ── 내부 헬퍼 ──────────────────────────────────────────────────────────────


def _build_instruct_from_df(
    df,
    is_train,
    config,
    processor,
    network,
    network_params,
    init_x,
):
    """DataFrame 의 train/test 파티션에서 Instruct 를 만든다."""
    split_df = df[df["train"] == is_train].copy()
    split_name = "train" if is_train else "test"

    cond_id = jnp.array(split_df["cond_id"].to_list()).reshape(-1, 1)

    # ── embedding (CSV 에 embed_* 컬럼이 있으면 사용) ─────────────────────
    embedding_df = split_df.filter(regex="embed_*")
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

    # ── condition ─────────────────────────────────────────────────────────
    condition_df = split_df.filter(regex=r"(?<!sub_)condition_")
    condition_df = condition_df.reindex(
        sorted(condition_df.columns, key=lambda x: int(x.split("_")[-1])),
        axis=1,
    )
    condition = jnp.array(condition_df.to_numpy())

    # ── reward_enum ───────────────────────────────────────────────────────
    reward_enum_list = [
        [int(digit) for digit in str(num)]
        for num in split_df["reward_enum"].to_list()
    ]
    max_len = max(len(x) for x in reward_enum_list)
    reward_enum = jnp.array(
        [x + [0] * (max_len - len(x)) for x in reward_enum_list]
    )

    # ── encoder-based embedding 재생성 (clip / cnnclip) ───────────────────
    if config.encoder.model == "clip" and processor is not None:
        embedding = _generate_clip_embedding(
            split_df, config, processor, network, network_params, init_x, split_name,
        )
    elif config.encoder.model == "cnnclip" and processor is not None:
        embedding = _generate_cnnclip_embedding(
            split_df, config, processor, network, network_params, init_x, split_name,
        )

    return Instruct(
        reward_i=reward_enum,
        condition=condition,
        embedding=embedding,
        condition_id=cond_id,
    )


def _generate_clip_embedding(df, config, processor, network, network_params, init_x, split_name):
    """CLIP text encoder 로 임베딩을 생성한다."""
    assert network is not None, "network is required for clip embedding generation"

    language_instr_list = df["instruction"].to_list()
    tokenized = processor(
        text=language_instr_list,
        return_tensors="jax",
        padding="max_length",
        truncation=True,
        max_length=77,
    )
    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]

    instr_x = PCGRLObs(
        map_obs=jnp.repeat(init_x.map_obs, input_ids.shape[0], axis=0),
        past_map_obs=None,
        flat_obs=jnp.repeat(init_x.flat_obs, input_ids.shape[0], axis=0),
        nlp_obs=jnp.repeat(init_x.nlp_obs, input_ids.shape[0], axis=0),
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=jnp.zeros(
            (input_ids.shape[0], 224, 224, config.clip_input_channel),
            dtype=jnp.float32,
        ),
    )
    _, _, _, embedding, _, _ = network.apply(
        network_params, x=instr_x,
        return_text_embed=True,
        return_state_embed=False,
        return_sketch_embed=False,
    )
    logger.info(
        f"Generated clip text embeddings {embedding.shape} "
        f"for {input_ids.shape[0]} instructions (split={split_name})"
    )
    return embedding


def _generate_cnnclip_embedding(df, config, processor, network, network_params, init_x, split_name):
    """CNN-CLIP encoder 로 임베딩을 생성한다."""
    assert network is not None, "network is required for cnnclip embedding generation"

    language_instr_list = df["instruction"].to_list()
    tokenized = processor(
        text=language_instr_list,
        return_tensors="jax",
        padding="max_length",
        truncation=True,
        max_length=77,
    )
    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]

    if config.multimodal_condition:
        dataset_mgr = DatasetManager(config.human_demo_path)
        levels = dataset_mgr.get_levels(
            language_instr_list, n=10, to_jax=True, squeeze_n=False, coord_channel=True,
        )
        sketches = dataset_mgr.get_sketches(
            language_instr_list, n=10, to_jax=True, squeeze_n=False, coord_channel=True,
        )

        n_inst, n_samples, H, W, C = levels.shape
        n_inst, n_samples, H_, W_, C_ = sketches.shape

        levels = jnp.reshape(levels, (n_inst * n_samples, H, W, C))
        sketches = jnp.reshape(sketches, (n_inst * n_samples, H_, W_, C_))

        input_ids = jnp.repeat(input_ids, n_samples, axis=0)
        attention_mask = jnp.repeat(attention_mask, n_samples, axis=0)
    else:
        levels = jnp.zeros((input_ids.shape[0], 16, 16, 5), dtype=jnp.float32)
        sketches = jnp.zeros((input_ids.shape[0], 224, 224, 3), dtype=jnp.float32)

    instr_x = PCGRLObs(
        map_obs=jnp.repeat(init_x.map_obs, input_ids.shape[0], axis=0),
        past_map_obs=None,
        flat_obs=jnp.repeat(init_x.flat_obs, input_ids.shape[0], axis=0),
        nlp_obs=jnp.repeat(init_x.nlp_obs, input_ids.shape[0], axis=0),
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=levels,
        sketch_values=sketches,
    )

    _, _, _, embedding_t, embedding_s, embedding_k = network.apply(
        network_params, x=instr_x,
        return_text_embed=True,
        return_state_embed=bool(config.multimodal_condition),
        return_sketch_embed=bool(config.multimodal_condition),
    )

    if config.multimodal_condition:
        embedding = jnp.concatenate([embedding_t, embedding_s, embedding_k], axis=0)
    else:
        embedding = embedding_t

    logger.info(
        f"Generated cnnclip embeddings {embedding.shape} "
        f"for {input_ids.shape[0]} instructions "
        f"(split={split_name}, multimodal={config.multimodal_condition})"
    )
    return embedding


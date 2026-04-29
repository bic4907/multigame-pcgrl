from __future__ import annotations

import hashlib
from glob import glob
from os.path import basename, join
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax.training import checkpoints as flax_ckpts

from instruct_rl.dataclass import Instruct
from instruct_rl.utils.log_utils import get_logger

from .constants import CLIP_EMBED_CACHE_DIR, DECODER_REWARD_CACHE_DIR

logger = get_logger(__file__)


def _build_instruct(sample_list, config):
    """샘플 리스트에서 Instruct 객체를 빌드한다."""
    logger.info(
        "Building Instruct: samples=%d, use_clip=%s, use_nlp=%s, use_decoder=%s",
        len(sample_list),
        getattr(config, "use_clip", False),
        getattr(config, "use_nlp", False),
        hasattr(config, "decoder"),
    )

    use_clip = getattr(config, "use_clip", False)
    needs_clip_embed = use_clip and config.nlp_input_dim > 0
    needs_decoder_pred = use_clip and hasattr(config, "decoder")

    shared_module, shared_variables = None, None
    if needs_clip_embed or needs_decoder_pred:
        logger.info(
            "Loading shared CLIP module + checkpoint once "
            "(needs_embed=%s, needs_decoder=%s)",
            needs_clip_embed,
            needs_decoder_pred,
        )
        shared_module, shared_variables = _load_shared_clip_module_and_ckpt(config)

    embedding = _build_instruct_embedding(
        sample_list,
        config,
        shared_module=shared_module,
        shared_variables=shared_variables,
    )
    reward_i, condition = _build_reward_and_condition(
        sample_list,
        config,
        text_embeddings=embedding,
        shared_module=shared_module,
        shared_variables=shared_variables,
    )
    logger.info(
        "Built Instruct tensors: embedding=%s, reward_i=%s, condition=%s",
        embedding.shape,
        reward_i.shape,
        condition.shape,
    )
    condition_id = jnp.arange(len(sample_list), dtype=jnp.int32).reshape(-1, 1)

    if shared_module is not None or shared_variables is not None:
        del shared_module
        del shared_variables

    return Instruct(
        reward_i=reward_i,
        condition=condition,
        embedding=embedding,
        condition_id=condition_id,
    )


def _load_shared_clip_module_and_ckpt(config):
    """CLIP 인코더 모듈을 init 하고 encoder.ckpt_path 의 체크포인트를 1회 복원한다."""
    encoder_config = config.encoder
    module, variables = _load_clip_encoder_module(config, encoder_config)
    variables = _restore_encoder_checkpoint(encoder_config, variables)
    return module, variables


def _build_instruct_embedding(sample_list, config, *, shared_module=None, shared_variables=None):
    """설정값에 따라 텍스트 임베딩을 계산한다."""
    n = len(sample_list)
    if getattr(config, "use_clip", False) and config.nlp_input_dim > 0:
        embedding = _compute_clip_embeddings(
            sample_list,
            config,
            module=shared_module,
            variables=shared_variables,
        )
        logger.info("Embedding done: shape=%s", embedding.shape)
        return embedding
    if getattr(config, "use_nlp", False) and config.nlp_input_dim > 0:
        embedding = _compute_bert_embeddings(sample_list, config.nlp_input_dim)
        logger.info("Embedding done: shape=%s", embedding.shape)
        return embedding

    fallback_shape = (n, max(1, config.nlp_input_dim))
    logger.info("Embedding disabled: using zeros %s", fallback_shape)
    return jnp.zeros(fallback_shape, dtype=jnp.float32)


def _build_reward_and_condition(
    sample_list,
    config,
    *,
    text_embeddings=None,
    shared_module=None,
    shared_variables=None,
):
    """reward_i 및 condition 벡터를 생성한다."""
    if getattr(config, "use_clip", False) and hasattr(config, "decoder"):
        logger.info("Reward/Condition mode: CLIP decoder prediction path")
        return _build_reward_and_condition_with_decoder(
            sample_list,
            config,
            text_embeddings=text_embeddings,
            module=shared_module,
            variables=shared_variables,
        )

    logger.info("Reward/Condition mode: metadata fallback path")
    reward_i_list = []
    for sample in sample_list:
        reward_i_list.append([sample.meta["reward_enum"]])
    reward_i = jnp.array(reward_i_list, dtype=jnp.int32)

    condition_list = []
    for sample in sample_list:
        conds = sample.meta.get("conditions", {})
        row = []
        for i in range(0, 5):
            val = conds.get(i, conds.get(str(i), -1))
            row.append(float(val))
        condition_list.append(row)
    condition = jnp.array(condition_list, dtype=jnp.float32)
    return reward_i, condition


def _build_reward_and_condition_with_decoder(
    sample_list,
    config,
    *,
    text_embeddings=None,
    module=None,
    variables=None,
):
    """CLIP decoder 추론으로 reward_i/condition을 생성한다."""
    from conf.config import DecoderConfig
    from tqdm import tqdm

    n = len(sample_list)
    num_classes = config.decoder.num_reward_classes

    logger.info(
        "Decoder reward prediction: n=%d, num_classes=%d, ckpt=%s, reuse_ckpt=%s",
        n,
        num_classes,
        config.encoder.ckpt_path,
        module is not None and variables is not None,
    )

    if module is not None and variables is not None:
        decoder_apply_fn = module.apply
        decoder_vars = variables
    else:
        from encoder.utils.decoder_reward import load_decoder

        decoder_cfg = DecoderConfig(
            num_reward_classes=num_classes,
            hidden_dim=getattr(config.decoder, "hidden_dim", DecoderConfig.hidden_dim),
            num_layers=getattr(config.decoder, "num_layers", DecoderConfig.num_layers),
            output_dim=getattr(config.decoder, "output_dim", DecoderConfig.output_dim),
            cnn_reward_enum_onehot=getattr(
                config.decoder,
                "cnn_reward_enum_onehot",
                DecoderConfig.cnn_reward_enum_onehot,
            ),
        )
        decoder_apply_fn, decoder_vars = load_decoder(
            ckpt_dir=config.encoder.ckpt_path,
            encoder_config=config.encoder,
            decoder_config=decoder_cfg,
        )

    ckpt_signature = _checkpoint_signature_for_cache(decoder_vars)
    cache_key, cache_path = _build_decoder_reward_cache_path(
        sample_list,
        ckpt_signature=ckpt_signature,
    )
    if cache_path.exists():
        try:
            cached = np.load(cache_path)
            reward_i_cached = cached["reward_i"]
            condition_cached = cached["condition"]
            expected_reward_shape = (n, 1)
            expected_condition_shape = (n, num_classes)
            if reward_i_cached.shape == expected_reward_shape and condition_cached.shape == expected_condition_shape:
                logger.info(
                    "Decoder reward cache HIT (hash=%s) — loaded from %s",
                    cache_key[:12],
                    cache_path,
                )
                logger.info(
                    "Decoder reward prediction done: reward_i=%s, condition=%s",
                    reward_i_cached.shape,
                    condition_cached.shape,
                )
                return (
                    jnp.array(reward_i_cached, dtype=jnp.int32),
                    jnp.array(condition_cached, dtype=jnp.float32),
                )
            logger.warning(
                "Decoder reward cache invalid shape at %s: reward_i=%s (expected=%s), condition=%s (expected=%s). Recomputing.",
                cache_path,
                reward_i_cached.shape,
                expected_reward_shape,
                condition_cached.shape,
                expected_condition_shape,
            )
        except Exception as e:
            logger.warning(
                "Failed to load decoder reward cache from %s: %s. Recomputing.",
                cache_path,
                e,
            )

    if text_embeddings is None:
        raise ValueError(
            "Decoder reward prediction requires precomputed text embeddings "
            "from Instruct. Got text_embeddings=None."
        )
    text_embeddings = jnp.array(text_embeddings, dtype=jnp.float32)
    if text_embeddings.shape[0] != n:
        raise ValueError(
            f"text_embeddings batch mismatch: got={text_embeddings.shape[0]}, expected={n}"
        )

    @jax.jit
    def _decode_batch(text_embed_batch):
        reward_logits, _, condition_pred_raw = decoder_apply_fn(
            decoder_vars,
            text_embed_batch,
            training=False,
            method=lambda m, embed, training=False: m.decoder(embed, training=training),
        )

        reward_enum_i = jnp.argmax(reward_logits, axis=-1).astype(jnp.int32)
        pred_cond = condition_pred_raw[jnp.arange(condition_pred_raw.shape[0]), reward_enum_i]
        return reward_enum_i, pred_cond

    batch_size = 256
    total_batches = (n + batch_size - 1) // batch_size if n > 0 else 0
    logger.info("Decoder batching: batch_size=%d, total_batches=%d", batch_size, total_batches)
    reward_i_list = []
    cond_list = []
    pbar = tqdm(
        range(0, n, batch_size),
        total=total_batches,
        desc="Decoder reward",
        unit="batch",
    )
    for start_idx in pbar:
        end_idx = min(start_idx + batch_size, n)
        embed_batch = text_embeddings[start_idx:end_idx]
        reward_enum_i, pred_cond = _decode_batch(embed_batch)
        reward_i_list.append(np.array(reward_enum_i))
        cond_list.append(np.array(pred_cond))

    reward_i_flat = np.concatenate(reward_i_list, axis=0)
    pred_cond_raw = np.concatenate(cond_list, axis=0)

    reward_i = jnp.array(reward_i_flat, dtype=jnp.int32).reshape(-1, 1)
    condition = jnp.full((n, num_classes), -1.0, dtype=jnp.float32)
    row_idx = jnp.arange(n)
    condition = condition.at[row_idx, reward_i_flat].set(jnp.array(pred_cond_raw))
    logger.info(
        "Decoder reward prediction done: reward_i=%s, condition=%s",
        reward_i.shape,
        condition.shape,
    )
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            reward_i=np.asarray(reward_i, dtype=np.int32),
            condition=np.asarray(condition, dtype=np.float32),
        )
        logger.info(
            "Decoder reward cache saved (hash=%s): %s",
            cache_key[:12],
            cache_path,
        )
    except Exception as e:
        logger.warning("Failed to save decoder reward cache to %s: %s", cache_path, e)

    return reward_i, condition


def _tokenize_texts(sample_list, encoder_config):
    """샘플 리스트에서 CLIP 토크나이저로 input_ids / attention_mask 를 반환한다."""
    from transformers import CLIPProcessor

    model_name = "openai/clip-vit-base-patch32"
    processor = CLIPProcessor.from_pretrained(model_name)

    texts, has_text = [], []
    for sample in sample_list:
        if sample.instruction is not None and len(sample.instruction.strip()) > 0:
            texts.append(sample.instruction)
            has_text.append(True)
        else:
            texts.append("")
            has_text.append(False)

    num_valid = sum(1 for flag in has_text if flag)
    logger.debug(
        "Tokenize CLIP texts: total=%d, non_empty=%d, max_len=%d",
        len(sample_list),
        num_valid,
        encoder_config.token_max_len,
    )

    inputs = processor(
        text=texts,
        return_tensors="jax",
        padding="max_length",
        truncation=True,
        max_length=encoder_config.token_max_len,
    )
    return (
        jnp.array(inputs["input_ids"]),
        jnp.array(inputs["attention_mask"]),
        has_text,
    )


def _load_clip_encoder_module(config, encoder_config):
    """config에 따라 ContrastiveModule 또는 ContrastiveDecoderModule을 초기화한다."""
    from encoder.clip_model import get_cnnclip_decoder_encoder, get_cnnclip_encoder

    use_decoder = hasattr(config, "decoder")
    if use_decoder:
        from conf.config import DecoderConfig

        decoder_cfg = DecoderConfig(
            num_reward_classes=config.decoder.num_reward_classes,
            cnn_reward_enum_onehot=config.decoder.cnn_reward_enum_onehot,
        )
        module, _ = get_cnnclip_decoder_encoder(
            encoder_config,
            decoder_config=decoder_cfg,
            RL_training=True,
        )
    else:
        module, _ = get_cnnclip_encoder(encoder_config, RL_training=True)

    rng = jax.random.PRNGKey(0)
    dummy_ids = jnp.ones((1, encoder_config.token_max_len), dtype=jnp.int32)
    dummy_mask = jnp.ones((1, encoder_config.token_max_len), dtype=jnp.int32)
    dummy_pix = jnp.ones((1, 16, 16, 6), dtype=jnp.float32)
    mode = "text_state" if encoder_config.state else "text"
    init_kwargs = dict(mode=mode, training=False)
    if use_decoder:
        init_kwargs["reward_enum"] = jnp.zeros((1,), dtype=jnp.int32)

    variables = module.init(rng, dummy_ids, dummy_mask, dummy_pix, **init_kwargs)
    return module, variables


def _format_num_bytes(num_bytes: int) -> str:
    """바이트 수를 사람이 읽기 쉬운 문자열로 변환한다."""
    if num_bytes < 1024:
        return f"{num_bytes} B"
    if num_bytes < 1024**2:
        return f"{num_bytes / 1024:.1f} KB"
    return f"{num_bytes / 1024**2:.2f} MB"


def _compute_tree_signature_hash(tree, *, algo: str = "sha256"):
    """중첩 dict/PyTree의 결정론적 signature hash를 계산한다."""
    from flax.core import FrozenDict
    from flax.traverse_util import flatten_dict

    if isinstance(tree, FrozenDict):
        tree = tree.unfreeze()

    if isinstance(tree, dict):
        flat = flatten_dict(tree)
    else:
        flat = {("root",): tree}

    hasher = hashlib.new(algo)
    leaf_count = 0
    total_bytes = 0

    for path in sorted(flat.keys(), key=lambda p: tuple(str(k) for k in p)):
        leaf = flat[path]
        path_str = "/".join(str(k) for k in path)
        hasher.update(path_str.encode("utf-8"))
        hasher.update(b"\0")

        if leaf is None:
            hasher.update(b"NONE")
            leaf_count += 1
            continue

        try:
            arr = np.asarray(leaf)
            if arr.dtype == np.dtype("O"):
                raise TypeError("object dtype")
            arr = np.ascontiguousarray(arr)
            hasher.update(str(arr.shape).encode("utf-8"))
            hasher.update(b"|")
            hasher.update(str(arr.dtype).encode("utf-8"))
            hasher.update(b"|")
            hasher.update(memoryview(arr.reshape(-1).view(np.uint8)))
            leaf_count += 1
            total_bytes += int(arr.nbytes)
            continue
        except Exception:
            pass

        hasher.update(type(leaf).__name__.encode("utf-8"))
        hasher.update(b"|")
        hasher.update(repr(leaf).encode("utf-8"))
        leaf_count += 1

    return hasher.hexdigest(), leaf_count, total_bytes


def _checkpoint_signature_for_cache(variables) -> str:
    """캐시 키 생성을 위한 체크포인트 시그니처(hex)를 계산한다."""
    try:
        signature_hex, _, _ = _compute_tree_signature_hash(variables)
        return signature_hex
    except Exception:
        return "unknown"


def _build_clip_embedding_cache_path(sample_list, *, ckpt_signature: str) -> tuple[str, Path]:
    """CLIP 임베딩 캐시 키와 저장 경로를 생성한다."""
    hasher = hashlib.sha256()
    hasher.update(b"clip-latent-embedding-cache-v2")
    hasher.update(f"|ckpt_signature={ckpt_signature}".encode("utf-8"))

    for sample in sample_list:
        hasher.update(f"|game={getattr(sample, 'game', '')}".encode("utf-8"))
        hasher.update(f"|source_id={getattr(sample, 'source_id', '')}".encode("utf-8"))
        instr = sample.instruction.strip() if (sample.instruction and sample.instruction.strip()) else ""
        hasher.update(f"|instruction={instr}".encode("utf-8"))

    cache_key = hasher.hexdigest()
    return cache_key, CLIP_EMBED_CACHE_DIR / f"{cache_key}.npy"


def _build_decoder_reward_cache_path(sample_list, *, ckpt_signature: str) -> tuple[str, Path]:
    """디코더 reward/condition 예측 캐시 키와 저장 경로를 생성한다."""
    hasher = hashlib.sha256()
    hasher.update(b"decoder-reward-cache-v4-text-embed")
    hasher.update(f"|ckpt_signature={ckpt_signature}".encode("utf-8"))

    for sample in sample_list:
        hasher.update(f"|game={getattr(sample, 'game', '')}".encode("utf-8"))
        hasher.update(f"|source_id={getattr(sample, 'source_id', '')}".encode("utf-8"))
        instr = sample.instruction.strip() if (sample.instruction and sample.instruction.strip()) else ""
        hasher.update(f"|instruction={instr}".encode("utf-8"))
        arr = np.asarray(sample.array, dtype=np.int32)
        hasher.update(f"|array_shape={arr.shape}".encode("utf-8"))
        hasher.update(memoryview(np.ascontiguousarray(arr).reshape(-1).view(np.uint8)))

    cache_key = hasher.hexdigest()
    return cache_key, DECODER_REWARD_CACHE_DIR / f"{cache_key}.npz"


def _log_checkpoint_signature_hash(state, *, ckpt_dir: str, step: int, fmt: str):
    """복원된 체크포인트 state의 signature hash를 로그로 출력한다."""
    try:
        signature_hex, leaf_count, total_bytes = _compute_tree_signature_hash(state)
        logger.info(
            "Encoder checkpoint signature hash: %s (algo=sha256, step=%d, format=%s, leaves=%d, bytes=%s)",
            signature_hex,
            step,
            fmt,
            leaf_count,
            _format_num_bytes(total_bytes),
        )
    except Exception as e:
        logger.warning(
            "Failed to compute checkpoint signature hash (step=%s, dir=%s): %s",
            step,
            ckpt_dir,
            e,
        )


def _restore_encoder_checkpoint(encoder_config, variables):
    """encoder_config.ckpt_path 에서 가장 최신 체크포인트를 복원한다."""
    ckpt_path = encoder_config.ckpt_path
    if ckpt_path is None:
        logger.warning(
            "encoder.ckpt_path is None — using randomly initialized encoder for text projection"
        )
        return variables

    ckpt_subdirs = glob(join(ckpt_path, "*"))
    ckpt_steps = sorted(
        [int(basename(d)) for d in ckpt_subdirs if basename(d).isdigit()],
        reverse=True,
    )
    if not ckpt_steps:
        logger.warning("No checkpoint steps found in %s", ckpt_path)
        return variables

    ckpt_dir = join(ckpt_path, str(ckpt_steps[0]))
    enc_state = flax_ckpts.restore_checkpoint(ckpt_dir, target=None, prefix="")
    if enc_state is None:
        logger.warning("Checkpoint restore returned None from %s", ckpt_dir)
        return variables

    if "params" in enc_state and "params" in enc_state["params"]:
        logger.info(
            "Encoder checkpoint loaded (TrainState format) from %s (step %s)",
            ckpt_dir,
            ckpt_steps[0],
        )
        restored_state = enc_state["params"]
        _log_checkpoint_signature_hash(
            restored_state,
            ckpt_dir=ckpt_dir,
            step=ckpt_steps[0],
            fmt="TrainState.params",
        )
        return restored_state

    logger.info("Encoder checkpoint loaded from %s (step %s)", ckpt_dir, ckpt_steps[0])
    _log_checkpoint_signature_hash(
        enc_state,
        ckpt_dir=ckpt_dir,
        step=ckpt_steps[0],
        fmt="raw",
    )
    return enc_state


def _encode_texts_batched(module, variables, input_ids, attention_mask, batch_size=256):
    """module.apply 를 배치 단위로 호출해 text_embed 를 수집한다."""
    from tqdm import tqdm

    @jax.jit
    def _encode_batch(variables, ids, mask):
        return module.apply(variables, ids, mask, None, mode="text", training=False)

    n = input_ids.shape[0]
    total_batches = (n + batch_size - 1) // batch_size if n > 0 else 0
    all_embeddings = []
    pbar = tqdm(
        range(0, n, batch_size),
        total=total_batches,
        desc="CLIP text embed",
        unit="batch",
    )
    for start in pbar:
        end = min(start + batch_size, n)
        out = _encode_batch(variables, input_ids[start:end], attention_mask[start:end])
        all_embeddings.append(np.array(out["text_embed"]))

    return np.concatenate(all_embeddings, axis=0)


def _postprocess_embeddings(embeddings, has_text, nlp_input_dim):
    """instruction 없는 행을 zeros 로 채우고, nlp_input_dim 에 맞게 패딩/절삭한다."""
    for i, has_text_flag in enumerate(has_text):
        if not has_text_flag:
            embeddings[i] = 0.0

    embed_dim = embeddings.shape[1]
    if nlp_input_dim > embed_dim:
        pad_width = ((0, 0), (0, nlp_input_dim - embed_dim))
        embeddings = np.pad(embeddings, pad_width, mode="constant")
    elif nlp_input_dim < embed_dim:
        embeddings = embeddings[:, :nlp_input_dim]

    return jnp.array(embeddings, dtype=jnp.float32)


def _compute_clip_embeddings(sample_list, config, *, module=None, variables=None):
    """사전학습된 CLIP encoder를 통해 instruction 텍스트 -> latent embedding을 계산한다."""
    nlp_input_dim = config.nlp_input_dim
    encoder_config = config.encoder

    if module is None or variables is None:
        module, variables = _load_clip_encoder_module(config, encoder_config)
        variables = _restore_encoder_checkpoint(encoder_config, variables)

    ckpt_signature = _checkpoint_signature_for_cache(variables)
    cache_key, cache_path = _build_clip_embedding_cache_path(
        sample_list,
        ckpt_signature=ckpt_signature,
    )
    if cache_path.exists():
        try:
            cached = np.load(cache_path)
            expected_shape = (len(sample_list), nlp_input_dim)
            if cached.shape == expected_shape:
                logger.info(
                    "CLIP latent embedding cache HIT (hash=%s) — loaded from %s",
                    cache_key[:12],
                    cache_path,
                )
                logger.info("CLIP latent embeddings shape: %s", cached.shape)
                return jnp.array(cached, dtype=jnp.float32)
            logger.warning(
                "CLIP latent embedding cache invalid shape at %s: got=%s, expected=%s. Recomputing.",
                cache_path,
                cached.shape,
                expected_shape,
            )
        except Exception as e:
            logger.warning(
                "Failed to load CLIP latent embedding cache from %s: %s. Recomputing.",
                cache_path,
                e,
            )

    try:
        from transformers import CLIPProcessor  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "use_clip=True requires the `transformers` package. "
            "Install it with: pip install transformers"
        ) from e

    logger.info(
        "Computing CLIP latent embeddings (output_dim=%d) for %d samples (reuse_ckpt=%s)",
        encoder_config.output_dim,
        len(sample_list),
        module is not None and variables is not None,
    )

    input_ids, attention_mask, has_text = _tokenize_texts(sample_list, encoder_config)
    clip_embeddings = _encode_texts_batched(module, variables, input_ids, attention_mask)
    result = _postprocess_embeddings(clip_embeddings, has_text, nlp_input_dim)

    logger.info("CLIP latent embeddings shape: %s", result.shape)
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, np.asarray(result, dtype=np.float32))
        logger.info(
            "CLIP latent embedding cache saved (hash=%s): %s",
            cache_key[:12],
            cache_path,
        )
    except Exception as e:
        logger.warning("Failed to save CLIP latent embedding cache to %s: %s", cache_path, e)

    return result


def _compute_bert_embeddings(sample_list, nlp_input_dim):
    """BERT를 사용하여 instruction 텍스트에서 임베딩을 계산한다."""
    try:
        from transformers import AutoTokenizer, FlaxAutoModel
    except ImportError as e:
        raise ImportError(
            "use_nlp=True requires the `transformers` package. "
            "Install it with: pip install transformers"
        ) from e

    model_name = "bert-base-uncased"
    logger.info("Computing BERT embeddings with %s for %d samples", model_name, len(sample_list))

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = FlaxAutoModel.from_pretrained(model_name)

    texts = []
    has_text = []
    for sample in sample_list:
        if sample.instruction is not None and len(sample.instruction.strip()) > 0:
            texts.append(sample.instruction)
            has_text.append(True)
        else:
            texts.append("")
            has_text.append(False)

    from tqdm import tqdm

    bert_batch_size = 64
    all_cls = []
    batches = range(0, len(texts), bert_batch_size)
    for i in tqdm(batches, desc="BERT embeddings", unit="batch"):
        batch_texts = texts[i : i + bert_batch_size]
        tokens = tokenizer(
            batch_texts,
            return_tensors="jax",
            padding=True,
            truncation=True,
            max_length=128,
        )
        outputs = model(**tokens)
        all_cls.append(np.array(outputs.last_hidden_state[:, 0, :]))

    cls_embeddings = np.concatenate(all_cls, axis=0)

    for i, has_text_flag in enumerate(has_text):
        if not has_text_flag:
            cls_embeddings[i] = 0.0

    bert_dim = cls_embeddings.shape[1]
    if nlp_input_dim > bert_dim:
        pad_width = ((0, 0), (0, nlp_input_dim - bert_dim))
        cls_embeddings = np.pad(cls_embeddings, pad_width, mode="constant")
    elif nlp_input_dim < bert_dim:
        cls_embeddings = cls_embeddings[:, :nlp_input_dim]

    logger.info("BERT embeddings shape: %s", cls_embeddings.shape)
    return jnp.array(cls_embeddings, dtype=jnp.float32)


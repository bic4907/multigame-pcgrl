"""
instruct_rl/utils/dataset_loader.py
====================================
MultiGameDataset 기반 Instruct 빌더.
jax.jit 바깥에서 호출하여 데이터셋을 로드하고 Instruct 객체를 빌드한다.
"""
from __future__ import annotations

from collections import Counter

import jax
import jax.numpy as jnp

from instruct_rl.dataclass import Instruct
from instruct_rl.utils.log_utils import get_logger

logger = get_logger(__file__)

# reward_enum → 사람이 읽을 수 있는 이름
REWARD_ENUM_NAMES = {
    1: "region",
    2: "path_length",
    3: "block",
    4: "bat_amount",
    5: "bat_direction",
}


def load_dataset_instruct(config):
    """MultiGameDataset에서 Instruct 객체를 빌드한다.

    Parameters
    ----------
    config : TrainConfig
        dataset_game, dataset_reward_enum, dataset_train_ratio, seed, nlp_input_dim 필요.

    Returns
    -------
    (train_inst, test_inst) : tuple[Instruct, Instruct]
    """
    from dataset.multigame import MultiGameDataset

    logger.info(f"[CPCGRL] Loading MultiGameDataset (game={config.dataset_game}, "
                f"reward_enum={config.dataset_reward_enum})")

    ds = MultiGameDataset(
        include_dungeon=(config.dataset_game == 'dungeon'),
        include_pokemon=(config.dataset_game == 'pokemon'),
        include_sokoban=(config.dataset_game == 'sokoban'),
        include_doom=(config.dataset_game == 'doom'),
        include_doom2=(config.dataset_game == 'doom'),
        include_zelda=(config.dataset_game == 'zelda'),
        use_tile_mapping=False,
    )

    # 게임별 필터링
    samples = ds.by_game(config.dataset_game)

    # reward_enum 필터링
    if config.dataset_reward_enum is not None:
        samples = [s for s in samples if s.meta.get("reward_enum") == config.dataset_reward_enum]

    # reward annotation이 있는 샘플만
    samples = [s for s in samples if "reward_enum" in s.meta and "conditions" in s.meta]

    assert len(samples) > 0, (
        f"No samples found for game={config.dataset_game}, "
        f"reward_enum={config.dataset_reward_enum}. "
        f"Check that reward annotations exist."
    )

    # ── 데이터셋 상세 로그 ──────────────────────────────────────────────
    _log_dataset_summary(config, samples)

    # Train/Test 분할
    n_total = len(samples)
    rng_split = jax.random.PRNGKey(config.seed)
    perm = jax.random.permutation(rng_split, n_total)
    n_train = int(n_total * config.dataset_train_ratio)

    train_indices = perm[:n_train].tolist()
    test_indices = perm[n_train:].tolist()

    if len(train_indices) == 0:
        train_indices = list(range(n_total))
    if len(test_indices) == 0:
        test_indices = list(range(n_total))

    train_samples = [samples[i] for i in train_indices]
    test_samples = [samples[i] for i in test_indices]

    train_inst = _build_instruct(train_samples, config)
    test_inst = _build_instruct(test_samples, config)

    # ── Train/Test 분할 로그 ─────────────────────────────────────────────
    _log_split_summary(train_samples, test_samples, train_inst)

    return train_inst, test_inst


# ── 내부 헬퍼 ──────────────────────────────────────────────────────────────────


def _build_instruct(sample_list, config):
    """샘플 리스트에서 Instruct 객체를 빌드한다."""
    n = len(sample_list)

    reward_i_list = []
    for s in sample_list:
        reward_i_list.append([s.meta["reward_enum"]])
    reward_i = jnp.array(reward_i_list, dtype=jnp.int32)

    condition_list = []
    for s in sample_list:
        conds = s.meta.get("conditions", {})
        row = []
        for i in range(1, 6):
            val = conds.get(i, conds.get(str(i), -1))
            row.append(float(val))
        row.extend([-1.0] * 4)
        condition_list.append(row)
    condition = jnp.array(condition_list, dtype=jnp.float32)

    # ── 임베딩 계산 ──────────────────────────────────────────────────────
    if getattr(config, "use_nlp", False) and config.nlp_input_dim > 0:
        embedding = _compute_bert_embeddings(sample_list, config.nlp_input_dim)
    else:
        embedding = jnp.zeros((n, max(1, config.nlp_input_dim)), dtype=jnp.float32)

    condition_id = jnp.arange(n, dtype=jnp.int32).reshape(-1, 1)

    return Instruct(
        reward_i=reward_i,
        condition=condition,
        embedding=embedding,
        condition_id=condition_id,
    )


def _compute_bert_embeddings(sample_list, nlp_input_dim):
    """BERT를 사용하여 instruction 텍스트에서 임베딩을 계산한다.

    instruction이 None인 샘플은 zeros 임베딩으로 대체한다.
    """
    import numpy as np

    try:
        from transformers import AutoTokenizer, FlaxAutoModel
    except ImportError:
        logger.warning("transformers not installed — falling back to zero embeddings")
        return jnp.zeros((len(sample_list), nlp_input_dim), dtype=jnp.float32)

    model_name = "bert-base-uncased"  # 768-dim
    logger.info(f"Computing BERT embeddings with {model_name} for {len(sample_list)} samples")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = FlaxAutoModel.from_pretrained(model_name)

    texts = []
    has_text = []
    for s in sample_list:
        if s.instruction is not None and len(s.instruction.strip()) > 0:
            texts.append(s.instruction)
            has_text.append(True)
        else:
            texts.append("")  # placeholder
            has_text.append(False)

    # 배치 토크나이즈
    tokens = tokenizer(
        texts, return_tensors="jax",
        padding=True, truncation=True, max_length=128,
    )
    # BERT forward → [CLS] 토큰 임베딩
    outputs = model(**tokens)
    cls_embeddings = np.array(outputs.last_hidden_state[:, 0, :])  # (N, 768)

    # instruction 없는 샘플은 zeros
    for i, ht in enumerate(has_text):
        if not ht:
            cls_embeddings[i] = 0.0

    # nlp_input_dim에 맞게 패딩 또는 절삭
    bert_dim = cls_embeddings.shape[1]
    if nlp_input_dim > bert_dim:
        pad_width = ((0, 0), (0, nlp_input_dim - bert_dim))
        cls_embeddings = np.pad(cls_embeddings, pad_width, mode="constant")
    elif nlp_input_dim < bert_dim:
        cls_embeddings = cls_embeddings[:, :nlp_input_dim]

    logger.info(f"BERT embeddings shape: {cls_embeddings.shape}")
    return jnp.array(cls_embeddings, dtype=jnp.float32)


def _log_dataset_summary(config, samples):
    """데이터셋 요약을 로거로 출력한다."""
    logger.info("=" * 70)
    logger.info("[CPCGRL] Dataset Summary")
    logger.info(f"  Game           : {config.dataset_game}")
    if config.dataset_reward_enum is not None:
        logger.info(f"  Reward Enum    : {config.dataset_reward_enum}"
                    f" ({REWARD_ENUM_NAMES.get(config.dataset_reward_enum, '?')})")
    else:
        logger.info(f"  Reward Enum    : None (all)")
    logger.info(f"  Total Samples  : {len(samples)}")

    # reward_enum별 분포
    re_counter = Counter(s.meta["reward_enum"] for s in samples)
    for re_val in sorted(re_counter.keys()):
        feat_names = set(
            s.meta.get("feature_name", "?")
            for s in samples if s.meta["reward_enum"] == re_val
        )
        logger.info(f"    reward_enum={re_val} ({REWARD_ENUM_NAMES.get(re_val, '?')}): "
                     f"{re_counter[re_val]} samples, features={feat_names}")

    # condition 값 통계
    for re_val in sorted(re_counter.keys()):
        re_samples = [s for s in samples if s.meta["reward_enum"] == re_val]
        cond_vals = set()
        for s in re_samples:
            conds = s.meta.get("conditions", {})
            val = conds.get(re_val, conds.get(str(re_val), None))
            if val is not None:
                cond_vals.add(float(val))
        if cond_vals:
            logger.info(f"    → condition values: {sorted(cond_vals)}")

    logger.info(f"  Train Ratio    : {config.dataset_train_ratio}")
    logger.info("=" * 70)


def _log_split_summary(train_samples, test_samples, train_inst):
    """Train/Test 분할 결과를 로거로 출력한다."""
    train_re = Counter(s.meta["reward_enum"] for s in train_samples)
    test_re = Counter(s.meta["reward_enum"] for s in test_samples)
    logger.info("[CPCGRL] Train/Test Split")
    logger.info(f"  Train : {len(train_samples)} samples  {dict(sorted(train_re.items()))}")
    logger.info(f"  Test  : {len(test_samples)} samples  {dict(sorted(test_re.items()))}")
    logger.info(f"  Instruct reward_i shape : {train_inst.reward_i.shape}")
    logger.info(f"  Instruct condition shape: {train_inst.condition.shape}")


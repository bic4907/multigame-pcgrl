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

    print(f"[CPCGRL] Loading MultiGameDataset (game={config.dataset_game}, "
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
    _print_dataset_summary(config, samples)

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
    _print_split_summary(train_samples, test_samples, train_inst)

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

    embedding = jnp.zeros((n, max(1, config.nlp_input_dim)), dtype=jnp.float32)
    condition_id = jnp.arange(n, dtype=jnp.int32).reshape(-1, 1)

    return Instruct(
        reward_i=reward_i,
        condition=condition,
        embedding=embedding,
        condition_id=condition_id,
    )


def _print_dataset_summary(config, samples):
    """데이터셋 요약을 콘솔에 출력한다."""
    print("=" * 70)
    print("[CPCGRL] Dataset Summary")
    print(f"  Game           : {config.dataset_game}")
    if config.dataset_reward_enum is not None:
        print(f"  Reward Enum    : {config.dataset_reward_enum}"
              f" ({REWARD_ENUM_NAMES.get(config.dataset_reward_enum, '?')})")
    else:
        print(f"  Reward Enum    : None (all)")
    print(f"  Total Samples  : {len(samples)}")

    # reward_enum별 분포
    re_counter = Counter(s.meta["reward_enum"] for s in samples)
    for re_val in sorted(re_counter.keys()):
        feat_names = set(
            s.meta.get("feature_name", "?")
            for s in samples if s.meta["reward_enum"] == re_val
        )
        print(f"    reward_enum={re_val} ({REWARD_ENUM_NAMES.get(re_val, '?')}): "
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
            print(f"    → condition values: {sorted(cond_vals)}")

    print(f"  Train Ratio    : {config.dataset_train_ratio}")
    print("=" * 70)


def _print_split_summary(train_samples, test_samples, train_inst):
    """Train/Test 분할 결과를 콘솔에 출력한다."""
    train_re = Counter(s.meta["reward_enum"] for s in train_samples)
    test_re = Counter(s.meta["reward_enum"] for s in test_samples)
    print("[CPCGRL] Train/Test Split")
    print(f"  Train : {len(train_samples)} samples  {dict(sorted(train_re.items()))}")
    print(f"  Test  : {len(test_samples)} samples  {dict(sorted(test_re.items()))}")
    print(f"  Instruct reward_i shape : {train_inst.reward_i.shape}")
    print(f"  Instruct condition shape: {train_inst.condition.shape}")


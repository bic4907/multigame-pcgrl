# MGPCGRL: Cross-Domain Reward Transfer for Multi-Game Procedural Content Generation Reinforcement Learning

This repository contains the code for **MGPCGRL (Multi-Game PCGRL)**, a
multi-domain reinforcement learning framework for instruction-conditioned
procedural content generation.

MGPCGRL targets a practical gap in PCGRL: rewards and instruction meanings are
usually hand-defined for one game at a time. The framework instead learns shared
representations between design instructions and game levels, then transfers
reward signals across game domains.


## Setup

```bash
conda create -n mgpcgrl python=3.11
conda activate mgpcgrl
pip install -r requirements.txt
```

Initialize external level datasets:

```bash
git submodule update --init --recursive
```

## MGPCGRL Reproduction

The MGPCGRL reproduction has three stages:

1. Train the multi-game CLIP encoder and reward decoder.
2. Train PCGRL policies with the trained encoder checkpoint.
3. Evaluate the trained PCGRL policies.

The sweep files use `/mnt/nas/mgpcgrl/...` as the default checkpoint and result
root. Change `saves_dir` and `encoder.ckpt_dir` in the YAML files if your
machine uses a different path.

Each `wandb sweep ...` command prints a sweep id. Start workers with the printed
`wandb agent <entity>/<project>/<sweep-id>` command.
Replace `<wandb-entity>` with your own W&B entity before running the commands.

### Few-Shot Commands

Train MGPCGRL encoder:

```bash
wandb sweep --project encoder_mgpcgrl_fewshot --entity <wandb-entity> sweep/wandb_sweep/mgpcgrl/fewshot/train_encoder.yaml
```

Train MGPCGRL PCGRL policies:

```bash
wandb sweep --project train_mgpcgrl_fewshot --entity <wandb-entity> sweep/wandb_sweep/mgpcgrl/fewshot/train_pcgrl.yaml
```

Evaluate MGPCGRL:

```bash
wandb sweep --project eval_mgpcgrl_fewshot --entity <wandb-entity> sweep/wandb_sweep/mgpcgrl/fewshot/eval_pcgrl.yaml
```

### Experiment Table

The linked files list the exact WandB sweep commands for encoder training,
PCGRL training, and evaluation. They also include the baseline methods used for
the same setting.

| Setting | Description | Commands |
| --- | --- | --- |
| Zero-shot | Hold out one game domain from encoder training, then evaluate transfer to the unseen game. | [Experiment](experiment/zeroshot.md) |
| Few-shot | Train with a small fraction of the held-out game and compare against zero-shot behavior. Includes the `delta_weight=0.0` ablation. | [Experiment](experiment/fewshot.md) |
| Full-shot | Train with all available data for the target setting. Includes the `delta_weight=0.0` ablation. | [Experiment](experiment/fullshot.md) |
| Full-shot 2 | Full-shot experiments over two-game combinations. | [Experiment](experiment/fullshot_2.md) |
| Full-shot 3 | Full-shot experiments over three-game combinations. | [Experiment](experiment/fullshot_3.md) |

## Reward Prediction

The reward decoder predicts the reward type and target condition from an
instruction embedding. See
[`reward_decode`](encoder/utils/decoder_reward.py#L118) and
[`predict_reward_condition`](encoder/utils/decoder_reward.py#L212).

```python
def predict_reward(text_embedding, decoder_apply_fn, decoder_variables):
    reward_logits, condition_pred, condition_pred_raw = decoder_apply_fn(
        decoder_variables,
        text_embedding,
        training=False,
        method=lambda m, embed, training=False: m.decoder(embed, training=training),
    )

    reward_enum = jnp.argmax(reward_logits, axis=-1)
    condition = condition_pred_raw[jnp.arange(condition_pred_raw.shape[0]), reward_enum]
    return reward_enum, condition
```

## Domain-Cross Loss

MGPCGRL uses a continuous task-wise cross-game direction alignment loss during
encoder training. For each `(game, reward_enum)` group, it estimates the
direction in text-embedding space induced by increasing the normalized condition
value. For the same `reward_enum`, directions from different games are aligned
with cosine distance.

See
[`continuous_direction_alignment`](train_clip_decoder.py#L292) in
`train_clip_decoder.py`.

```python
import jax
import jax.numpy as jnp


def safe_l2_normalize(x, axis=-1, eps=1e-6):
    sq_norm = jnp.sum(x * x, axis=axis, keepdims=True)
    return x * jax.lax.rsqrt(jnp.maximum(sq_norm, eps * eps))


def domain_cross_loss(z_text, game_id, reward_enum, condition, *,
                      num_games, num_tasks, min_count=2, var_eps=1e-4):
    """Continuous task-wise cross-game direction alignment loss."""
    z = safe_l2_normalize(z_text)
    c = condition.astype(jnp.float32)

    game_mask = game_id[None, :] == jnp.arange(num_games)[:, None]
    task_mask = reward_enum[None, :] == jnp.arange(num_tasks)[:, None]
    mask = (game_mask[:, None, :] & task_mask[None, :, :]).astype(jnp.float32)

    n = mask.sum(axis=-1)
    safe_n = jnp.maximum(n, 1.0)

    c_mean = (mask * c[None, None, :]).sum(-1) / safe_n
    z_mean = (mask[..., None] * z[None, None, :, :]).sum(-2) / safe_n[..., None]

    dc = c[None, None, :] - c_mean[..., None]
    dz = z[None, None, :, :] - z_mean[:, :, None, :]

    slope_num = ((mask * dc)[..., None] * dz).sum(-2)
    slope_den = (mask * dc * dc).sum(-1)

    valid = (n >= min_count) & ((slope_den / safe_n) > var_eps)
    slope = slope_num / jnp.where(valid, slope_den, 1.0)[..., None]
    slope = jnp.where(valid[..., None], slope, 0.0)

    direction = safe_l2_normalize(slope)
    direction = jnp.where(valid[..., None], direction, 0.0)

    cosine = jnp.einsum("gtd,htd->ght", direction, direction)
    pair_valid = valid[:, None, :] & valid[None, :, :]
    upper_tri = jnp.triu(jnp.ones((num_games, num_games), dtype=bool), k=1)
    pair_valid = pair_valid & upper_tri[:, :, None]

    pair_loss = jnp.where(pair_valid, 1.0 - cosine, 0.0)
    return pair_loss.sum() / jnp.maximum(pair_valid.astype(jnp.float32).sum(), 1.0)
```

The encoder objective combines this term with contrastive, reward
classification, and condition regression losses:

```python
loss = (
    contrastive_weight * contrastive_loss
    + cls_weight * reward_enum_ce_loss
    + reg_weight * condition_regression_loss
    + delta_weight * domain_cross_loss
)
```

In the provided MGPCGRL sweeps, `delta_weight=0.03` is the default direction
alignment setting. Use the `mgpcgrl_dw0` sweep files to reproduce the ablation
with `delta_weight=0.0`.

## Key Entry Points

- `train_clip_decoder.py`: train the CLIP-style encoder and reward decoder.
- `encoder/utils/decoder_reward.py`: predict `reward_enum` and `condition` from instruction embeddings.
- `train_mgpcgrl.py`: train MGPCGRL PCGRL policies from an encoder checkpoint.
- `eval_mgpcgrl.py`: evaluate trained MGPCGRL policies.
- `conf/train_mgpcgrl.yaml`: MGPCGRL training defaults.
- `conf/eval_mgpcgrl.yaml`: MGPCGRL evaluation defaults.
- `sweep/wandb_sweep/mgpcgrl`: MGPCGRL reproduction sweeps.
- `sweep/wandb_sweep/mgpcgrl_dw0`: domain-cross loss ablation sweeps.

## Dataset Roots

MGPCGRL uses five canonical game domains: Dungeon, Pokemon, Sokoban, Doom, and
Zelda. Doom and Doom2 are loaded separately in code, but reported as one Doom
domain in the seen/unseen split.

| Dataset root | Games used | Source |
| --- | --- | --- |
| `dataset/TheVGLC` | Zelda, Doom, Doom2 | [Repo](https://github.com/TheVGLC/TheVGLC) |
| `dataset/dungeon_level_dataset` | Dungeon | [Repo](https://github.com/bic4907/dungeon-level-dataset) |
| `dataset/boxoban_levels` | Sokoban | [Repo](https://github.com/google-deepmind/boxoban-levels) |
| `dataset/five-dollar-model` | Pokemon | [Repo](https://github.com/TimMerino1710/five-dollar-model) |

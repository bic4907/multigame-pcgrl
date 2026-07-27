### CPCGRL

---
**Train PCGRL**
```bash
wandb sweep --project train_cpcgrl --entity <wandb-entity> sweep/wandb_sweep/cpcgrl/train_pcgrl.yaml
```
**Eval PCGRL**
```bash
wandb sweep --project eval_cpcgrl --entity <wandb-entity> sweep/wandb_sweep/cpcgrl/eval_pcgrl.yaml
```

### ReWARD (Seen Ratios)

---

**Train Encoder**
```bash
wandb sweep --project encoder_reward_unseen_ratios --entity <wandb-entity> sweep/wandb_sweep/reward_unseen_ratios/train_encoder.yaml
```

**Train PCGRL**
```bash
wandb sweep --project train_reward_unseen_ratios --entity <wandb-entity> sweep/wandb_sweep/reward_unseen_ratios/train_pcgrl.yaml
```

**Eval PCGRL**
```bash
wandb sweep --project eval_reward_unseen_ratios --entity <wandb-entity> sweep/wandb_sweep/reward_unseen_ratios/eval_pcgrl.yaml
```


### Random

---

```bash
wandb sweep --project eval_random --entity <wandb-entity> sweep/wandb_sweep/random/eval.yaml
```


### Pretrained CLIP

---

```bash
wandb sweep --project train_pretrained_clip --entity <wandb-entity> sweep/wandb_sweep/pretrained_clip/train_pretrained_clip.yaml
```

```bash
wandb sweep --project eval_pretrained_clip --entity <wandb-entity> sweep/wandb_sweep/pretrained_clip/eval_pretrained_clip.yaml
```

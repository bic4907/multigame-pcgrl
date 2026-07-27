**Train Encoder**
```bash
wandb sweep --project encoder_reward_zeroshot_dw0 --entity <wandb-entity> train_encoder.yaml
```

**Train PCGRL**
```bash
wandb sweep --project train_reward_zeroshot_dw0 --entity <wandb-entity> train_pcgrl.yaml
```

**Eval PCGRL**
```bash
wandb sweep --project eval_reward_zeroshot_dw0 --entity <wandb-entity> eval_pcgrl.yaml
```

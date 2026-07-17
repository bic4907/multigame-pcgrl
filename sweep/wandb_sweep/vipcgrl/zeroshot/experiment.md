**Train Encoder**
```bash
wandb sweep --project encoder_vipcgrl_zeroshot --entity <wandb-entity> train_encoder.yaml
```

**Train PCGRL**
```bash
wandb sweep --project train_vipcgrl_zeroshot --entity <wandb-entity> train_pcgrl.yaml
```

**Eval PCGRL**
```bash
wandb sweep --project eval_vipcgrl_zeroshot --entity <wandb-entity> eval_pcgrl.yaml
```

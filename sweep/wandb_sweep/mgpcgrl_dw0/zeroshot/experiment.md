**Train Encoder**
```bash
wandb sweep --project encoder_mgpcgrl_zeroshot_dw0 --entity <wandb-entity> train_encoder.yaml
```

**Train PCGRL**
```bash
wandb sweep --project train_mgpcgrl_zeroshot_dw0 --entity <wandb-entity> train_pcgrl.yaml
```

**Eval PCGRL**
```bash
wandb sweep --project eval_mgpcgrl_zeroshot_dw0 --entity <wandb-entity> eval_pcgrl.yaml
```

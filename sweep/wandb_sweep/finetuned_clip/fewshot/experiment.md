**Train Encoder**
```bash
wandb sweep --project encoder_finetuned_clip_fewshot --entity <wandb-entity> train_encoder.yaml
```

**Train PCGRL**
```bash
wandb sweep --project train_finetuned_clip_fewshot --entity <wandb-entity> train_pcgrl.yaml
```

**Eval PCGRL**
```bash
wandb sweep --project eval_finetuned_clip_fewshot --entity <wandb-entity> eval_pcgrl.yaml
```

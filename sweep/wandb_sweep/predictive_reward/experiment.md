**Train PCGRL**
```bash
wandb sweep --project train_predictive_reward --entity <wandb-entity> train_pcgrl.yaml
```

**Eval PCGRL**
```bash
wandb sweep --project eval_predictive_reward --entity <wandb-entity> eval_pcgrl.yaml
```

**Train Encoder**
```bash
wandb sweep --project encoder_mgpcgrl_direction0626 --entity st4889ha-gwangju-institute-of-science-and-technology train_encoder.yaml
```

**Train PCGRL**
```bash
wandb sweep --project train_mgpcgrl_direction0626 --entity st4889ha-gwangju-institute-of-science-and-technology train_pcgrl.yaml
```

**Eval PCGRL**
```bash
wandb sweep --project eval_mgpcgrl_direction0626 --entity st4889ha-gwangju-institute-of-science-and-technology eval_pcgrl.yaml
```

**Train Encoder**
```bash
wandb sweep --project aaai27_encoder_ipcgrl_fewshot --entity st4889ha-gwangju-institute-of-science-and-technology train_encoder.yaml
```

**Train PCGRL**
```bash
wandb sweep --project aaai27_train_ipcgrl_fewshot --entity st4889ha-gwangju-institute-of-science-and-technology train_pcgrl.yaml
```

**Eval PCGRL**
```bash
wandb sweep --project aaai27_eval_ipcgrl_fewshot --entity st4889ha-gwangju-institute-of-science-and-technology eval_pcgrl.yaml
```
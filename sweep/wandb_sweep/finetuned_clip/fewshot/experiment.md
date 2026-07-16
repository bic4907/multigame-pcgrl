**Train Encoder**
```bash
wandb sweep --project aaai27_encoder_finetuned_clip_fewshot --entity st4889ha-gwangju-institute-of-science-and-technology train_encoder.yaml
```

**Train PCGRL**
```bash
wandb sweep --project aaai27_train_finetuned_clip_fewshot --entity st4889ha-gwangju-institute-of-science-and-technology train_pcgrl.yaml
```

**Eval PCGRL**
```bash
wandb sweep --project aaai27_eval_finetuned_clip_fewshot --entity st4889ha-gwangju-institute-of-science-and-technology eval_pcgrl.yaml
```

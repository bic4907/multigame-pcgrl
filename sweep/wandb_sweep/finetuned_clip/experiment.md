**Train Encoder**
```bash
wandb sweep --project aaai27_encoder_finetuned_clip_unseen --entity st4889ha-gwangju-institute-of-science-and-technology train_finetuned_clip_encoder_unseen.yaml
```

**Train PCGRL**
```bash
wandb sweep --project aaai27_train_finetuned_clip_unseen --entity st4889ha-gwangju-institute-of-science-and-technology train_finetuned_clip_unseen.yaml
```

**Eval PCGRL**
```bash
wandb sweep --project aaai27_eval_finetuned_clip_unseen --entity st4889ha-gwangju-institute-of-science-and-technology eval_finetuned_clip_unseen.yaml
```

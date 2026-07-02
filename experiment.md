### CPCGRL

---
**Train PCGRL**
```bash
wandb sweep --project aaai27_train_cpcgrl --entity st4889ha-gwangju-institute-of-science-and-technology sweep/wandb_sweep/cpcgrl/train_pcgrl.yaml
```
**Eval PCGRL**
```bash
wandb sweep --project aaai27_eval_cpcgrl --entity st4889ha-gwangju-institute-of-science-and-technology sweep/wandb_sweep/cpcgrl/eval_pcgrl.yaml
```

### MGPCGRL (Seen Ratios)

---

**Train Encoder**
```bash
wandb sweep --project aaai27_encoder_mgpcgrl_unseen_ratios --entity st4889ha-gwangju-institute-of-science-and-technology sweep/wandb_sweep/mgpcgrl_unseen_ratios/train_encoder.yaml
```

**Train PCGRL**
```bash
wandb sweep --project aaai27_train_mgpcgrl_unseen_ratios --entity st4889ha-gwangju-institute-of-science-and-technology sweep/wandb_sweep/mgpcgrl_unseen_ratios/train_pcgrl.yaml
```

**Eval PCGRL**
```bash
wandb sweep --project aaai27_eval_mgpcgrl_unseen_ratios --entity st4889ha-gwangju-institute-of-science-and-technology sweep/wandb_sweep/mgpcgrl_unseen_ratios/eval_pcgrl.yaml
```


### Random

---

```bash
wandb sweep --project aaai27_eval_random --entity st4889ha-gwangju-institute-of-science-and-technology sweep/wandb_sweep/random/eval.yaml
```


### Pretrained CLIP

---

```bash
wandb sweep --project aaai27_train_pretrained_clip --entity st4889ha-gwangju-institute-of-science-and-technology sweep/wandb_sweep/pretrained_clip/train_pretrained_clip.yaml
```

```bash
wandb sweep --project aaai27_eval_pretrained_clip --entity st4889ha-gwangju-institute-of-science-and-technology sweep/wandb_sweep/pretrained_clip/eval_pretrained_clip.yaml
```

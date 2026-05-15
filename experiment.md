### CPCGRL

---
**Train PCGRL**
```bash
wandb sweep --project aaai27_train_cpcgrl --entity st4889ha-gwangju-institute-of-science-and-technology sweep/wandb_sweep/train_cpcgrl.yaml
```
**Eval PCGRL**
```bash
wandb sweep --project aaai27_eval_cpcgrl --entity st4889ha-gwangju-institute-of-science-and-technology sweep/wandb_sweep/eval_cpcgrl.yaml
```



### IPCGRL

---

**Train Encoder**
```bash
bash run_docker.sh python train_ipcgrl_encoder_mg.py saves_dir=/mnt/nas/mgpcgrl/ipcgrl_encoder
```


**Train PCGRL**
```bash
wandb sweep --project aaai27_train_ipcgrl --entity st4889ha-gwangju-institute-of-science-and-technology sweep/wandb_sweep/train_ipcgrl.yaml
```

**Eval PCGRL**
```bash
wandb sweep --project aaai27_eval_ipcgrl --entity st4889ha-gwangju-institute-of-science-and-technology sweep/wandb_sweep/eval_ipcgrl.yaml
```


### VIPCGRL

---

**Train Encoder**
```bash
wandb sweep --project aaai27_train_vipcgrl_encoder --entity st4889ha-gwangju-institute-of-science-and-technology sweep/wandb_sweep/train_vipcgrl_encoder.yaml
```
```bash
bash run_docker.sh python train_clip.py saves_dir=/mnt/nas/mgpcgrl/vipcgrl_encoder
```

**Train PCGRL**
```bash
wandb sweep --project aaai27_train_vipcgrl --entity st4889ha-gwangju-institute-of-science-and-technology sweep/wandb_sweep/train_vipcgrl.yaml
```

**Eval PCGRL**
```bash
wandb sweep --project aaai27_eval_vipcgrl --entity st4889ha-gwangju-institute-of-science-and-technology sweep/wandb_sweep/eval_vipcgrl.yaml
```

### VIPCGRL (`coef_human_sim=0`)

---

**Train PCGRL**
```bash
wandb sweep --project aaai27_train_vipcgrl_nosim --entity st4889ha-gwangju-institute-of-science-and-technology sweep/wandb_sweep/train_vipcgrl_nosim.yaml
```

**Eval PCGRL**
```bash
wandb sweep --project aaai27_eval_vipcgrl_nosim --entity st4889ha-gwangju-institute-of-science-and-technology sweep/wandb_sweep/eval_vipcgrl_nosim.yaml
```


### MGPCGRL

---

**Train Encoder**
```bash
wandb sweep --project aaai27_train_mgpcgrl_encoder --entity st4889ha-gwangju-institute-of-science-and-technology sweep/wandb_sweep/train_mgpcgrl_encoder.yaml
```
```bash
bash run_docker.sh python train_clip_decoder.py saves_dir=/mnt/nas/mgpcgrl/mgpcgrl_encoder
```

**Train PCGRL**
```bash
wandb sweep --project aaai27_train_mgpcgrl --entity st4889ha-gwangju-institute-of-science-and-technology sweep/wandb_sweep/train_mgpcgrl.yaml
```

**Eval PCGRL**
```bash
wandb sweep --project aaai27_eval_mgpcgrl --entity st4889ha-gwangju-institute-of-science-and-technology sweep/wandb_sweep/eval_mgpcgrl.yaml
```

### MGPCGRL (Unseen)

---

**Train Encoder**
```bash
wandb sweep --project aaai27_train_mgpcgrl_encoder_unseen --entity st4889ha-gwangju-institute-of-science-and-technology sweep/wandb_sweep/train_mgpcgrl_encoder_unseen.yaml
```

**Train PCGRL**
```bash
wandb sweep --project aaai27_train_mgpcgrl_unseen --entity st4889ha-gwangju-institute-of-science-and-technology sweep/wandb_sweep/train_mgpcgrl_unseen.yaml
```

**Eval PCGRL**
```bash
wandb sweep --project aaai27_eval_mgpcgrl_unseen --entity st4889ha-gwangju-institute-of-science-and-technology sweep/wandb_sweep/eval_mgpcgrl_unseen.yaml
```

### MGPCGRL (All Decoder Rewards)

---

**Train Encoder (Same to `Unseen`**)
```bash
wandb sweep --project aaai27_train_mgpcgrl_encoder_unseen --entity st4889ha-gwangju-institute-of-science-and-technology sweep/wandb_sweep/train_mgpcgrl_encoder_unseen.yaml
```

**Train PCGRL**
```bash
wandb sweep --project aaai27_train_mgpcgrl_all --entity st4889ha-gwangju-institute-of-science-and-technology sweep/wandb_sweep/train_mgpcgrl_all.yaml
```

**Eval PCGRL**
```bash
wandb sweep --project aaai27_eval_mgpcgrl_all --entity st4889ha-gwangju-institute-of-science-and-technology sweep/wandb_sweep/eval_mgpcgrl_all.yaml
```




### MGPCGRL (Seen Ratios)

---

**Train Encoder**
```bash
wandb sweep --project aaai27_train_mgpcgrl_encoder_unseen_ratios --entity st4889ha-gwangju-institute-of-science-and-technology sweep/wandb_sweep/train_mgpcgrl_encoder_unseen_ratios.yaml
```

**Train PCGRL**
```bash
wandb sweep --project aaai27_train_mgpcgrl_unseen_ratios --entity st4889ha-gwangju-institute-of-science-and-technology sweep/wandb_sweep/train_mgpcgrl_unseen_ratios.yaml
```

**Eval PCGRL**
```bash
wandb sweep --project aaai27_eval_mgpcgrl_unseen_ratios --entity st4889ha-gwangju-institute-of-science-and-technology sweep/wandb_sweep/eval_mgpcgrl_unseen_ratios.yaml
```


### Random

---

```bash
wandb sweep --project aaai27_eval_random --entity st4889ha-gwangju-institute-of-science-and-technology sweep/wandb_sweep/eval_random.yaml
```


### Pretrained CLIP

---

```bash
wandb sweep --project aaai27_train_pretrained_clip --entity st4889ha-gwangju-institute-of-science-and-technology sweep/wandb_sweep/train_pretrained_clip.yaml
```

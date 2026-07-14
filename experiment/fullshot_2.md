### IPCGRL (Fullshot 2)

게임 2개 조합 3가지: dgpk (dungeon+pokemon), skzd (sokoban+zelda), pkdm (pokemon+doom)

---

**Train Encoder**
```bash
wandb sweep --project aaai27_encoder_ipcgrl_fullshot_2 --entity st4889ha-gwangju-institute-of-science-and-technology ../sweep/wandb_sweep/ipcgrl/fullshot_2/train_encoder.yaml
wandb agent st4889ha-gwangju-institute-of-science-and-technology/aaai27_encoder_ipcgrl_fullshot_2/p6ayghbh
```

**Train PCGRL**
```bash
wandb sweep --project aaai27_train_ipcgrl_fullshot_2 --entity st4889ha-gwangju-institute-of-science-and-technology ../sweep/wandb_sweep/ipcgrl/fullshot_2/train_pcgrl.yaml
wandb agent st4889ha-gwangju-institute-of-science-and-technology/aaai27_train_ipcgrl_fullshot_2/ixjgt8sj
```

**Eval PCGRL**
```bash
wandb sweep --project aaai27_eval_ipcgrl_fullshot_2 --entity st4889ha-gwangju-institute-of-science-and-technology ../sweep/wandb_sweep/ipcgrl/fullshot_2/eval_pcgrl.yaml
wandb agent st4889ha-gwangju-institute-of-science-and-technology/aaai27_eval_ipcgrl_fullshot_2/qhnzfty1
```


### MIPCGRL (Fullshot 2)

게임 2개 조합 3가지: dgpk (dungeon+pokemon), skzd (sokoban+zelda), pkdm (pokemon+doom)

---

**Train Encoder**
```bash
wandb sweep --project aaai27_encoder_mipcgrl_fullshot_2 --entity st4889ha-gwangju-institute-of-science-and-technology ../sweep/wandb_sweep/mipcgrl/fullshot_2/train_encoder.yaml
wandb agent st4889ha-gwangju-institute-of-science-and-technology/aaai27_encoder_mipcgrl_fullshot_2/mf09oi05
```

**Train PCGRL**
```bash
wandb sweep --project aaai27_train_mipcgrl_fullshot_2 --entity st4889ha-gwangju-institute-of-science-and-technology ../sweep/wandb_sweep/mipcgrl/fullshot_2/train_pcgrl.yaml
wandb agent st4889ha-gwangju-institute-of-science-and-technology/aaai27_train_mipcgrl_fullshot_2/9e9vnt5j
```

**Eval PCGRL**
```bash
wandb sweep --project aaai27_eval_mipcgrl_fullshot_2 --entity st4889ha-gwangju-institute-of-science-and-technology ../sweep/wandb_sweep/mipcgrl/fullshot_2/eval_pcgrl.yaml
wandb agent st4889ha-gwangju-institute-of-science-and-technology/aaai27_eval_mipcgrl_fullshot_2/qdfgqfgg
```


### VIPCGRL (Fullshot 2)

게임 2개 조합 3가지: dgpk (dungeon+pokemon), skzd (sokoban+zelda), pkdm (pokemon+doom)

---

**Train Encoder**
```bash
wandb sweep --project aaai27_encoder_vipcgrl_fullshot_2 --entity st4889ha-gwangju-institute-of-science-and-technology ../sweep/wandb_sweep/vipcgrl/fullshot_2/train_encoder.yaml
```

**Train PCGRL**
```bash
wandb sweep --project aaai27_train_vipcgrl_fullshot_2 --entity st4889ha-gwangju-institute-of-science-and-technology ../sweep/wandb_sweep/vipcgrl/fullshot_2/train_pcgrl.yaml
```

**Eval PCGRL**
```bash
wandb sweep --project aaai27_eval_vipcgrl_fullshot_2 --entity st4889ha-gwangju-institute-of-science-and-technology ../sweep/wandb_sweep/vipcgrl/fullshot_2/eval_pcgrl.yaml
```


### MGPCGRL (Fullshot 2)

---

**Train Encoder**
```bash
wandb sweep --project aaai27_encoder_mgpcgrl_fullshot_2 --entity st4889ha-gwangju-institute-of-science-and-technology ../sweep/wandb_sweep/mgpcgrl/fullshot_2/train_encoder.yaml
```

**Train PCGRL**
```bash
wandb sweep --project aaai27_train_mgpcgrl_fullshot_2 --entity st4889ha-gwangju-institute-of-science-and-technology ../sweep/wandb_sweep/mgpcgrl/fullshot_2/train_pcgrl.yaml
```

**Eval PCGRL**
```bash
wandb sweep --project aaai27_eval_mgpcgrl_fullshot_2 --entity st4889ha-gwangju-institute-of-science-and-technology ../sweep/wandb_sweep/mgpcgrl/fullshot_2/eval_pcgrl.yaml
```

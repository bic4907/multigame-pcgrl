### IPCGRL (Fullshot 3)

게임 3개 조합 3가지: dgpksk (dungeon+pokemon+sokoban), pkdmzd (pokemon+doom+zelda), skdmzd (sokoban+doom+zelda)

---

**Train Encoder**
```bash
wandb sweep --project aaai27_encoder_ipcgrl_fullshot_3 --entity st4889ha-gwangju-institute-of-science-and-technology ../sweep/wandb_sweep/ipcgrl/fullshot_3/train_encoder.yaml
wandb agent st4889ha-gwangju-institute-of-science-and-technology/aaai27_encoder_ipcgrl_fullshot_3/7gbmubaz
```

**Train PCGRL**
```bash
wandb sweep --project aaai27_train_ipcgrl_fullshot_3 --entity st4889ha-gwangju-institute-of-science-and-technology ../sweep/wandb_sweep/ipcgrl/fullshot_3/train_pcgrl.yaml
wandb agent st4889ha-gwangju-institute-of-science-and-technology/aaai27_train_ipcgrl_fullshot_3/zq9cn98n
```

**Eval PCGRL**
```bash
wandb sweep --project aaai27_eval_ipcgrl_fullshot_3 --entity st4889ha-gwangju-institute-of-science-and-technology ../sweep/wandb_sweep/ipcgrl/fullshot_3/eval_pcgrl.yaml
wandb agent st4889ha-gwangju-institute-of-science-and-technology/aaai27_eval_ipcgrl_fullshot_3/yods92gx
```


### MIPCGRL (Fullshot 3)

게임 3개 조합 3가지: dgpksk (dungeon+pokemon+sokoban), pkdmzd (pokemon+doom+zelda), skdmzd (sokoban+doom+zelda)

---

**Train Encoder**
```bash
wandb sweep --project aaai27_encoder_mipcgrl_fullshot_3 --entity st4889ha-gwangju-institute-of-science-and-technology ../sweep/wandb_sweep/mipcgrl/fullshot_3/train_encoder.yaml
wandb agent st4889ha-gwangju-institute-of-science-and-technology/aaai27_encoder_mipcgrl_fullshot_3/dbdbeq79
```

**Train PCGRL**
```bash
wandb sweep --project aaai27_train_mipcgrl_fullshot_3 --entity st4889ha-gwangju-institute-of-science-and-technology ../sweep/wandb_sweep/mipcgrl/fullshot_3/train_pcgrl.yaml
wandb agent st4889ha-gwangju-institute-of-science-and-technology/aaai27_train_mipcgrl_fullshot_3/raeknn8g
```

**Eval PCGRL**
```bash
wandb sweep --project aaai27_eval_mipcgrl_fullshot_3 --entity st4889ha-gwangju-institute-of-science-and-technology ../sweep/wandb_sweep/mipcgrl/fullshot_3/eval_pcgrl.yaml
wandb agent st4889ha-gwangju-institute-of-science-and-technology/aaai27_eval_mipcgrl_fullshot_3/tuerelb0
```


### VIPCGRL (Fullshot 3)

게임 3개 조합 3가지: dgpksk (dungeon+pokemon+sokoban), pkdmzd (pokemon+doom+zelda), skdmzd (sokoban+doom+zelda)

---

**Train Encoder**
```bash
wandb sweep --project aaai27_encoder_vipcgrl_fullshot_3 --entity st4889ha-gwangju-institute-of-science-and-technology ../sweep/wandb_sweep/vipcgrl/fullshot_3/train_encoder.yaml
```

**Train PCGRL**
```bash
wandb sweep --project aaai27_train_vipcgrl_fullshot_3 --entity st4889ha-gwangju-institute-of-science-and-technology ../sweep/wandb_sweep/vipcgrl/fullshot_3/train_pcgrl.yaml
```

**Eval PCGRL**
```bash
wandb sweep --project aaai27_eval_vipcgrl_fullshot_3 --entity st4889ha-gwangju-institute-of-science-and-technology ../sweep/wandb_sweep/vipcgrl/fullshot_3/eval_pcgrl.yaml
```


### MGPCGRL (Fullshot 3)

---

**Train Encoder**
```bash
wandb sweep --project aaai27_encoder_mgpcgrl_fullshot_3 --entity st4889ha-gwangju-institute-of-science-and-technology ../sweep/wandb_sweep/mgpcgrl/fullshot_3/train_encoder.yaml
```

**Train PCGRL**
```bash
wandb sweep --project aaai27_train_mgpcgrl_fullshot_3 --entity st4889ha-gwangju-institute-of-science-and-technology ../sweep/wandb_sweep/mgpcgrl/fullshot_3/train_pcgrl.yaml
```

**Eval PCGRL**
```bash
wandb sweep --project aaai27_eval_mgpcgrl_fullshot_3 --entity st4889ha-gwangju-institute-of-science-and-technology ../sweep/wandb_sweep/mgpcgrl/fullshot_3/eval_pcgrl.yaml
```

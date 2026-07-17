### IPCGRL (Fullshot 3)

게임 3개 조합 3가지: dgpksk (dungeon+pokemon+sokoban), pkdmzd (pokemon+doom+zelda), skdmzd (sokoban+doom+zelda)

---

**Train Encoder**
```bash
wandb sweep --project encoder_ipcgrl_fullshot_3 --entity <wandb-entity> ../sweep/wandb_sweep/ipcgrl/fullshot_3/train_encoder.yaml
```

**Train PCGRL**
```bash
wandb sweep --project train_ipcgrl_fullshot_3 --entity <wandb-entity> ../sweep/wandb_sweep/ipcgrl/fullshot_3/train_pcgrl.yaml
```

**Eval PCGRL**
```bash
wandb sweep --project eval_ipcgrl_fullshot_3 --entity <wandb-entity> ../sweep/wandb_sweep/ipcgrl/fullshot_3/eval_pcgrl.yaml
```


### MIPCGRL (Fullshot 3)

게임 3개 조합 3가지: dgpksk (dungeon+pokemon+sokoban), pkdmzd (pokemon+doom+zelda), skdmzd (sokoban+doom+zelda)

---

**Train Encoder**
```bash
wandb sweep --project encoder_mipcgrl_fullshot_3 --entity <wandb-entity> ../sweep/wandb_sweep/mipcgrl/fullshot_3/train_encoder.yaml
```

**Train PCGRL**
```bash
wandb sweep --project train_mipcgrl_fullshot_3 --entity <wandb-entity> ../sweep/wandb_sweep/mipcgrl/fullshot_3/train_pcgrl.yaml
```

**Eval PCGRL**
```bash
wandb sweep --project eval_mipcgrl_fullshot_3 --entity <wandb-entity> ../sweep/wandb_sweep/mipcgrl/fullshot_3/eval_pcgrl.yaml
```


### VIPCGRL (Fullshot 3)

게임 3개 조합 3가지: dgpksk (dungeon+pokemon+sokoban), pkdmzd (pokemon+doom+zelda), skdmzd (sokoban+doom+zelda)

---

**Train Encoder**
```bash
wandb sweep --project encoder_vipcgrl_fullshot_3 --entity <wandb-entity> ../sweep/wandb_sweep/vipcgrl/fullshot_3/train_encoder.yaml
```

**Train PCGRL**
```bash
wandb sweep --project train_vipcgrl_fullshot_3 --entity <wandb-entity> ../sweep/wandb_sweep/vipcgrl/fullshot_3/train_pcgrl.yaml
```

**Eval PCGRL**
```bash
wandb sweep --project eval_vipcgrl_fullshot_3 --entity <wandb-entity> ../sweep/wandb_sweep/vipcgrl/fullshot_3/eval_pcgrl.yaml
```


### MGPCGRL (Fullshot 3)

---

**Train Encoder**
```bash
wandb sweep --project encoder_mgpcgrl_fullshot_3 --entity <wandb-entity> ../sweep/wandb_sweep/mgpcgrl/fullshot_3/train_encoder.yaml
```

**Train PCGRL**
```bash
wandb sweep --project train_mgpcgrl_fullshot_3 --entity <wandb-entity> ../sweep/wandb_sweep/mgpcgrl/fullshot_3/train_pcgrl.yaml
```

**Eval PCGRL**
```bash
wandb sweep --project eval_mgpcgrl_fullshot_3 --entity <wandb-entity> ../sweep/wandb_sweep/mgpcgrl/fullshot_3/eval_pcgrl.yaml
```


### MGPCGRL-DW0 (Fullshot 3)

---

**Train Encoder**
```bash
wandb sweep --project encoder_mgpcgrl_fullshot_3_dw0 --entity <wandb-entity> ../sweep/wandb_sweep/mgpcgrl_dw0/fullshot_3/train_encoder.yaml
```

**Train PCGRL**
```bash
wandb sweep --project train_mgpcgrl_fullshot_3_dw0 --entity <wandb-entity> ../sweep/wandb_sweep/mgpcgrl_dw0/fullshot_3/train_pcgrl.yaml
```

**Eval PCGRL**
```bash
wandb sweep --project eval_mgpcgrl_fullshot_3_dw0 --entity <wandb-entity> ../sweep/wandb_sweep/mgpcgrl_dw0/fullshot_3/eval_pcgrl.yaml
```

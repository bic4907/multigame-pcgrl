### IPCGRL (Fullshot 2)

게임 2개 조합 3가지: dgpk (dungeon+pokemon), skzd (sokoban+zelda), pkdm (pokemon+doom)

---

**Train Encoder**
```bash
wandb sweep --project encoder_ipcgrl_fullshot_2 --entity <wandb-entity> ../sweep/wandb_sweep/ipcgrl/fullshot_2/train_encoder.yaml
```

**Train PCGRL**
```bash
wandb sweep --project train_ipcgrl_fullshot_2 --entity <wandb-entity> ../sweep/wandb_sweep/ipcgrl/fullshot_2/train_pcgrl.yaml
```

**Eval PCGRL**
```bash
wandb sweep --project eval_ipcgrl_fullshot_2 --entity <wandb-entity> ../sweep/wandb_sweep/ipcgrl/fullshot_2/eval_pcgrl.yaml
```


### MIPCGRL (Fullshot 2)

게임 2개 조합 3가지: dgpk (dungeon+pokemon), skzd (sokoban+zelda), pkdm (pokemon+doom)

---

**Train Encoder**
```bash
wandb sweep --project encoder_mipcgrl_fullshot_2 --entity <wandb-entity> ../sweep/wandb_sweep/mipcgrl/fullshot_2/train_encoder.yaml
```

**Train PCGRL**
```bash
wandb sweep --project train_mipcgrl_fullshot_2 --entity <wandb-entity> ../sweep/wandb_sweep/mipcgrl/fullshot_2/train_pcgrl.yaml
```

**Eval PCGRL**
```bash
wandb sweep --project eval_mipcgrl_fullshot_2 --entity <wandb-entity> ../sweep/wandb_sweep/mipcgrl/fullshot_2/eval_pcgrl.yaml
```


### VIPCGRL (Fullshot 2)

게임 2개 조합 3가지: dgpk (dungeon+pokemon), skzd (sokoban+zelda), pkdm (pokemon+doom)

---

**Train Encoder**
```bash
wandb sweep --project encoder_vipcgrl_fullshot_2 --entity <wandb-entity> ../sweep/wandb_sweep/vipcgrl/fullshot_2/train_encoder.yaml
```

**Train PCGRL**
```bash
wandb sweep --project train_vipcgrl_fullshot_2 --entity <wandb-entity> ../sweep/wandb_sweep/vipcgrl/fullshot_2/train_pcgrl.yaml
```

**Eval PCGRL**
```bash
wandb sweep --project eval_vipcgrl_fullshot_2 --entity <wandb-entity> ../sweep/wandb_sweep/vipcgrl/fullshot_2/eval_pcgrl.yaml
```


### MGPCGRL (Fullshot 2)

---

**Train Encoder**
```bash
wandb sweep --project encoder_mgpcgrl_fullshot_2 --entity <wandb-entity> ../sweep/wandb_sweep/mgpcgrl/fullshot_2/train_encoder.yaml
```

**Train PCGRL**
```bash
wandb sweep --project train_mgpcgrl_fullshot_2 --entity <wandb-entity> ../sweep/wandb_sweep/mgpcgrl/fullshot_2/train_pcgrl.yaml
```

**Eval PCGRL**
```bash
wandb sweep --project eval_mgpcgrl_fullshot_2 --entity <wandb-entity> ../sweep/wandb_sweep/mgpcgrl/fullshot_2/eval_pcgrl.yaml
```


### MGPCGRL-DW0 (Fullshot 2)

---

**Train Encoder**
```bash
wandb sweep --project encoder_mgpcgrl_fullshot_2_dw0 --entity <wandb-entity> ../sweep/wandb_sweep/mgpcgrl_dw0/fullshot_2/train_encoder.yaml
```

**Train PCGRL**
```bash
wandb sweep --project train_mgpcgrl_fullshot_2_dw0 --entity <wandb-entity> ../sweep/wandb_sweep/mgpcgrl_dw0/fullshot_2/train_pcgrl.yaml
```

**Eval PCGRL**
```bash
wandb sweep --project eval_mgpcgrl_fullshot_2_dw0 --entity <wandb-entity> ../sweep/wandb_sweep/mgpcgrl_dw0/fullshot_2/eval_pcgrl.yaml
```

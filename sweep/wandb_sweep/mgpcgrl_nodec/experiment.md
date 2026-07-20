# MGPCGRL Nodec Sweeps

Encoder training uses `decoder_nograd=true`, so decoder losses do not backpropagate into the encoder embedding.

## Fullshot

**Train Encoder**
```bash
wandb sweep --project aaai27_encoder_mgpcgrl_fullshot_nodec --entity st4889ha-gwangju-institute-of-science-and-technology fullshot/train_encoder.yaml
```

**Train PCGRL**
```bash
wandb sweep --project aaai27_train_mgpcgrl_fullshot_nodec --entity st4889ha-gwangju-institute-of-science-and-technology fullshot/train_pcgrl.yaml
```

**Eval PCGRL**
```bash
wandb sweep --project aaai27_eval_mgpcgrl_fullshot_nodec --entity st4889ha-gwangju-institute-of-science-and-technology fullshot/eval_pcgrl.yaml
```

## Fullshot 2

**Train Encoder**
```bash
wandb sweep --project aaai27_encoder_mgpcgrl_fullshot_2_nodec --entity st4889ha-gwangju-institute-of-science-and-technology fullshot_2/train_encoder.yaml
```

**Train PCGRL**
```bash
wandb sweep --project aaai27_train_mgpcgrl_fullshot_2_nodec --entity st4889ha-gwangju-institute-of-science-and-technology fullshot_2/train_pcgrl.yaml
```

**Eval PCGRL**
```bash
wandb sweep --project aaai27_eval_mgpcgrl_fullshot_2_nodec --entity st4889ha-gwangju-institute-of-science-and-technology fullshot_2/eval_pcgrl.yaml
```

## Fullshot 3

**Train Encoder**
```bash
wandb sweep --project aaai27_encoder_mgpcgrl_fullshot_3_nodec --entity st4889ha-gwangju-institute-of-science-and-technology fullshot_3/train_encoder.yaml
```

**Train PCGRL**
```bash
wandb sweep --project aaai27_train_mgpcgrl_fullshot_3_nodec --entity st4889ha-gwangju-institute-of-science-and-technology fullshot_3/train_pcgrl.yaml
```

**Eval PCGRL**
```bash
wandb sweep --project aaai27_eval_mgpcgrl_fullshot_3_nodec --entity st4889ha-gwangju-institute-of-science-and-technology fullshot_3/eval_pcgrl.yaml
```

# vipcgrl_sample — Pretrained CLIP Encoder Checkpoint

## Overview

| Key | Value |
|-----|-------|
| **Model** | `cnnclip` (CNN-CLIP contrastive encoder) |
| **Config** | `inst-scn-1_se-whole_exp-def_es-64_md-ts_br-1.0_batch-128_lr-0.001` |
| **Epochs** | 30 (checkpoint at epoch 30) |
| **Part size** | ≤ 48 MB each (Git-friendly) |
| **Total (compressed)** | ~189 MB → ~187 MB gzip (4 parts) |

## Restore (Decompress)

```bash
# 1. Move to the target directory
cd pretrained_encoders/vipcgrl_sample

# 2. Concatenate split parts and extract
cat clip-enc-cnnclip_inst-scn-1_se-whole_exp-def_es-64_md-ts_br-1.0_batch-128_lr-0.001_0.tar.gz.part_* | tar xzf -

# The extracted directory structure:
# clip-enc-cnnclip_inst-scn-1_se-whole_exp-def_es-64_md-ts_br-1.0_batch-128_lr-0.001_0/
# ├── ckpts/
# │   └── 30/
# └── figures/
#     └── embed_epoch_*.png
```

### Extract to a custom location

```bash
cat pretrained_encoders/vipcgrl_sample/clip-enc-cnnclip_inst-scn-1_se-whole_exp-def_es-64_md-ts_br-1.0_batch-128_lr-0.001_0.tar.gz.part_* | tar xzf - -C /path/to/destination/
```

## Re-compress (for reference)

```bash
cd saves  # or wherever the original directory lives
tar czf - clip-enc-cnnclip_inst-scn-1_se-whole_exp-def_es-64_md-ts_br-1.0_batch-128_lr-0.001_0 | \
  split -b 48m - ../pretrained_encoders/vipcgrl_sample/clip-enc-cnnclip_inst-scn-1_se-whole_exp-def_es-64_md-ts_br-1.0_batch-128_lr-0.001_0.tar.gz.part_
```

## Verify integrity

```bash
cat clip-enc-cnnclip_inst-scn-1_se-whole_exp-def_es-64_md-ts_br-1.0_batch-128_lr-0.001_0.tar.gz.part_* | tar tzf - | head -20
```


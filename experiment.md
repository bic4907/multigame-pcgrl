

# Main Result

## Encoder Training

**MLP ✅**

```bash
python train_encoder.py batch_size=128

# Lab Server
bash run_docker.sh train_encoder.py batch_size=128
```

**CLIP** 👨‍💻

```bash
python train_clip.py batch_size=128

# Lab Server
bash run_docker.sh train_clip.py batch_size=128 buffer_dir=/raid/inchang/instructed_rl/pcgrl_buffer
```

## RL Training (175)

**CPCGRL (25) ✅**

```bash
bash sbatch_gpu.sh train.py overwrite=True instruct=scn-1_se-1,scn-1_se-2,scn-1_se-3,scn-1_se-4,scn-1_se-5 n_envs=600 seed=0,1,2,3,4 vec_cont=True raw_obs=True wandb_project=cpcgrl
```

**IPCGRL (25) ✅**

```bash
bash sbatch_gpu.sh train.py overwrite=True instruct=scn-1_se-1,scn-1_se-2,scn-1_se-3,scn-1_se-4,scn-1_se-5 n_envs=550 seed=0,1,2,3,4 encoder.model='mlp' wandb_project=ipcgrl
```

**VI-PCGRL (125) ✅**

```bash
bash sbatch_gpu.sh train.py encoder.model=cnnclip overwrite=True n_envs=500 instruct=scn-1_se-1,scn-1_se-2,scn-1_se-3,scn-1_se-4,scn-1_se-5 seed=0,1,2,3,4 SIM_COEF=1.75,3.5,7,15,30 wandb_project=vipcgrl

# Lab Server
bash run_docker.sh train.py encoder.model=cnnclip overwrite=True n_envs=500 instruct=scn-1_se-1 seed=0 SIM_COEF=1.75 wandb_project=vipcgrl
```

## RL Evalution

### Text Modal

**CPCGRL (25) ✅**

```bash
bash sbatch_gpu.sh eval.py instruct=scn-1_se-1,scn-1_se-2,scn-1_se-3,scn-1_se-4,scn-1_se-5 n_envs=300 seed=0,1,2,3,4 vec_cont=True raw_obs=True wandb_project=eval_cpcgrl reevaluate=True

# Lab Server
bash run_docker.sh eval.py overwrite=True instruct=scn-1_se-1 n_envs=100 seed=0 vec_cont=True raw_obs=True wandb_project=eval_cpcgrl total_timesteps=2 n_eps=2
```

**IPCGRL (25) ✅**

```bash
bash sbatch_gpu.sh eval.py instruct=scn-1_se-1,scn-1_se-2,scn-1_se-3,scn-1_se-4,scn-1_se-5 n_envs=200 seed=0,1,2,3,4 encoder.model='mlp' wandb_project=eval_ipcgrl reevaluate=True

# Lab Server
bash run_docker.sh eval.py overwrite=True instruct=scn-1_se-1 n_envs=4 seed=0 encoder.model='mlp' wandb_project=eval_ipcgrl
```

**VI-PCGRL (375)**  👨‍💻 — 3 (1+2)개의 modality에 대해 전부 평가

```bash
bash sbatch_gpu.sh eval.py encoder.model=cnnclip n_envs=100 instruct=scn-1_se-1,scn-1_se-2,scn-1_se-3,scn-1_se-4,scn-1_se-5 seed=0,1,2,3,4 SIM_COEF=1.75,3.5,7,15,30 wandb_project=eval_vipcgrl reevaluate=True eval_instruct=scn-1_se-whole eval_modality=text,state,sketch

bash sbatch_gpu.sh eval_sweep.py encoder.model=cnnclip n_envs=100 instruct=scn-1_se-1,scn-1_se-2,scn-1_se-3,scn-1_se-4,scn-1_se-5 seed=0,1,2,3,4 SIM_COEF=30 wandb_project=vipcgrl reevaluate=True eval_instruct=scn-1_se-whole

# Lab Server
bash run_docker.sh train.py encoder.model=cnnclip overwrite=True n_envs=100 instruct=scn-1_se-1 seed=0 wandb_project=eval_vipcgrl eval_modality=text,image,sketch

```

# Encoder Data ratio Ablation Study

## Encoder Training

- State(0.25, 0.1) / Sketch(0.25,0.1) / Text( 0.25)- 4개의 세팅

```bash
# Sketch 0.25
bash run_docker.sh python train_clip.py train_shuffle=True sketch_ratio=0.25

# Sketch 0.1
bash run_docker.sh python train_clip.py train_shuffle=True sketch_ratio=0.1

# State 0.25
bash run_docker.sh python train_clip.py train_shuffle=True state_ratio=0.25

# State 0.1
bash run_docker.sh python train_clip.py train_shuffle=True state_ratio=0.1

# State 0.25
bash run_docker.sh python train_clip.py train_shuffle=True text_ratio=0.25
```

## RL Training (100)

**VI-PCGRL (sketch ablation)  (50)**

```bash
bash sbatch_gpu.sh train.py encoder.model=cnnclip overwrite=True n_envs=500 instruct=scn-1_se-1,scn-1_se-2,scn-1_se-3,scn-1_se-4,scn-1_se-5 seed=0,1,2,3,4 SIM_COEF=30 sketch_ratio=0.25,0.1 encoder.ckpt_dir=encoder_ckpts_ablation wandb_project=ablation_vipcgrl 
```

**VI-PCGRL (state ablation)  (50)**

```bash
bash sbatch_gpu.sh train.py encoder.model=cnnclip overwrite=True n_envs=500 instruct=scn-1_se-1,scn-1_se-2,scn-1_se-3,scn-1_se-4,scn-1_se-5 seed=0,1,2,3,4 SIM_COEF=30 state_ratio=0.25,0.1 encoder.ckpt_dir=encoder_ckpts_ablation wandb_project=ablation_vipcgrl
```

**VI-PCGRL (text ablation)  (25)**

```bash
bash sbatch_gpu.sh train.py encoder.model=cnnclip overwrite=True n_envs=500 instruct=scn-1_se-1,scn-1_se-2,scn-1_se-3,scn-1_se-4,scn-1_se-5 seed=0,1,2,3,4 SIM_COEF=30 text_ratio=0.25 encoder.ckpt_dir=encoder_ckpts_ablation wandb_project=ablation_vipcgrl
```

## RL Evalution

**VI-PCGRL (sketch ablation)   (50)**  👨‍💻 — sketch만 평가

```bash
bash sbatch_gpu.sh eval.py encoder.model=cnnclip n_envs=100 instruct=scn-1_se-1,scn-1_se-2,scn-1_se-3,scn-1_se-4,scn-1_se-5 seed=0,1,2,3,4 SIM_COEF=30 sketch_ratio=0.1 wandb_project=0722_eval_ablation_vipcgrl reevaluate=True eval_instruct=scn-1_se-whole encoder.ckpt_dir=encoder_ckpts_ablation eval_modality=sketch,text,state
```

**VI-PCGRL (state ablation)   (50)**  👨‍💻 — state만 평가

```bash
bash sbatch_gpu.sh eval.py encoder.model=cnnclip n_envs=100 instruct=scn-1_se-1,scn-1_se-2,scn-1_se-3,scn-1_se-4,scn-1_se-5 seed=0,1,2,3,4 SIM_COEF=30 state_ratio=0.1 wandb_project=0722_eval_ablation_vipcgrl reevaluate=True eval_instruct=scn-1_se-whole encoder.ckpt_dir=encoder_ckpts_ablation eval_modality=state,text,sketch
```

**VI-PCGRL (text ablation)   (25)**  👨‍💻 — text만 평가

```bash
bash sbatch_gpu.sh eval.py encoder.model=cnnclip n_envs=100 instruct=scn-1_se-1,scn-1_se-2,scn-1_se-3,scn-1_se-4,scn-1_se-5 seed=0,1,2,3,4 SIM_COEF=30 text_ratio=0.25 wandb_project=0722_eval_ablation_vipcgrl reevaluate=True eval_instruct=scn-1_se-whole encoder.ckpt_dir=encoder_ckpts_ablation eval_modality=text,state,sketch
```

# Auxiliary Task

**Multimodal Condition-based RL (25) ✅**

```bash
bash sbatch_gpu.sh sweep.py encoder.model=cnnclip overwrite=True n_envs=500 instruct=scn-1_se-1,scn-1_se-2,scn-1_se-3,scn-1_se-4,scn-1_se-5 seed=0,1,2,3,4 SIM_COEF=30 wandb_project=multimodal_vipcgrl multimodal_condition=true exp_name=mc
```

# Encoder Human data Ablation Study

## Encoder Training

```bash
# Sketch 0.25
bash run_docker.sh python train_clip.py embed_type=bert
```

## RL Training (25)

```bash
bash sbatch_gpu.sh train.py encoder.model=cnnclip overwrite=True n_envs=500 instruct=scn-1_se-1,scn-1_se-2,scn-1_se-3,scn-1_se-4,scn-1_se-5 seed=0,1,2,3,4 SIM_COEF=30 encoder.ckpt_dir=encoder_ckpts_human_ablation wandb_project=human_data_ablation_vipcgrl exp_name=hu
```

## RL Evalution

**VI-PCGRL   (25)**  👨‍💻 

```bash
bash sbatch_gpu.sh eval.py encoder.model=cnnclip n_envs=100 instruct=scn-1_se-1,scn-1_se-2,scn-1_se-3,scn-1_se-4,scn-1_se-5 seed=0,1,2,3,4 SIM_COEF=30  wandb_project=eval_human_data_ablation_vipcgrl reevaluate=True eval_instruct=scn-1_se-whole encoder.ckpt_dir=encoder_ckpts_human_ablation  eval_modality=text exp_name=hu
```

# Random

---

```jsx
bash sbatch_gpu.sh eval.py overwrite=True instruct=scn-1_se-whole random_agent=True exp_name=rd wandb_projecet=eval_random seed=0,1,2,3,4
```
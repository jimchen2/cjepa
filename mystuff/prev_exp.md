## Training

### Extracting

```
export PYTHONPATH=$(pwd)
export LD_LIBRARY_PATH=$HOME/miniconda3/envs/cjepa/lib/python3.10/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}

python src/third_party/slotformer/base_slots/extract_videosaur.py \
    --weight "../pusht_videosaur_model.ckpt" \
    --data_root="/home/jichen/.stable_worldmodel" \
    --save_path="./pusht_videosaur_slots.pkl" \
    --dataset="pusht_expert" \
    --videosaur_config="src/third_party/videosaur/configs/videosaur/pusht_dinov2_hf.yml" \
    --params="src/third_party/slotformer/aloe_pusht_params.py"
```

### Download Databases

```
conda activate cjepa
cd /home/jichen/Downloads/cjepa

wget -nc https://huggingface.co/HazelNam/CJEPA/resolve/main/pusht_expert_action_meta.pkl
wget -nc https://huggingface.co/HazelNam/CJEPA/resolve/main/pusht_expert_proprio_meta.pkl
wget -nc https://huggingface.co/HazelNam/CJEPA/resolve/main/pusht_expert_state_meta.pkl
```

### Running the Script

```
#!/usr/bin/env bash
set -e

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cjepa

cd /home/jichen/Downloads/cjepa
export PYTHONPATH=$(pwd)
export LD_LIBRARY_PATH=$HOME/miniconda3/envs/cjepa/lib/python3.10/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}
export NCCL_NVLS_ENABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=1
export WANDB_MODE=disabled


python src/train/train_causalwm_AP_node_pusht_slot.py \
    cache_dir="/home/jichen/.stable_worldmodel" \
    output_model_name="pusht_cjepa" \
    dataset_name="pusht_expert" \
    num_workers=8 \
    batch_size=256 \
    trainer.max_epochs=30 \
    num_masked_slots=1 \
    predictor_lr=5e-4 \
    proprio_encoder_lr=1e-4 \
    action_encoder_lr=5e-4 \
    dinowm.history_size=3 \
    dinowm.num_preds=1 \
    dinowm.proprio_embed_dim=128 \
    dinowm.action_embed_dim=128 \
    frameskip=5 \
    videosaur.NUM_SLOTS=4 \
    videosaur.SLOT_DIM=128 \
    predictor.heads=16 \
    embedding_dir="/home/jichen/Downloads/cjepa/pusht_videosaur_slots.pkl" \
    model.load_weights="/home/jichen/Downloads/pusht_videosaur_model.ckpt" \
    action_dir="/home/jichen/Downloads/cjepa/pusht_expert_action_meta.pkl" \
    proprio_dir="/home/jichen/Downloads/cjepa/pusht_expert_proprio_meta.pkl" \
    state_dir="/home/jichen/Downloads/cjepa/pusht_expert_state_meta.pkl" \
    use_hungarian_matching=false
```


```
nohup /tmp/jepa_run_train_pusht.sh > /tmp/jepa_train_pusht.log 2>&1 &
```



| step | train/loss | val/loss | train/loss_future | val/loss_future | train/proprio_loss | val/proprio_loss |
|------|------------|----------|-------------------|-----------------|--------------------|------------------|
| 0 | 0.004951 | 0.005252 | 0.003052 | 0.003017 | 0.000297 | 0.000719 |
| 1 | 0.004684 | 0.005149 | 0.002961 | 0.002913 | 0.000254 | 0.000811 |
| 2 | 0.004362 | 0.004531 | 0.002821 | 0.002722 | 0.000200 | 0.000567 |
| 3 | 0.004128 | 0.004707 | 0.002669 | 0.002712 | 0.000175 | 0.000772 |
| 4 | 0.004025 | 0.004418 | 0.002618 | 0.002590 | 0.000156 | 0.000713 |
| 5 | 0.003736 | 0.004178 | 0.002454 | 0.002510 | 0.000135 | 0.000620 |
| 6 | 0.003654 | 0.004049 | 0.002448 | 0.002531 | 0.000111 | 0.000468 |
| 7 | 0.003533 | 0.004187 | 0.002389 | 0.002515 | 0.000101 | 0.000643 |
| 8 | 0.003484 | 0.004005 | 0.002335 | 0.002490 | 0.000107 | 0.000522 |
| 9 | 0.003351 | 0.003872 | 0.002254 | 0.002441 | 0.000079 | 0.000485 |
| 10 | 0.003380 | 0.003862 | 0.002299 | 0.002419 | 0.000093 | 0.000516 |
| 11 | 0.003201 | 0.003848 | 0.002188 | 0.002404 | 0.000094 | 0.000533 |
| 12 | 0.003244 | 0.003833 | 0.002211 | 0.002396 | 0.000076 | 0.000549 |
| 13 | 0.003182 | 0.003825 | 0.002200 | 0.002406 | 0.000068 | 0.000516 |
| 14 | 0.003101 | 0.003774 | 0.002104 | 0.002381 | 0.000064 | 0.000513 |
| 15 | 0.003123 | 0.003721 | 0.002138 | 0.002367 | 0.000063 | 0.000476 |
| 16 | 0.003001 | 0.003739 | 0.002051 | 0.002388 | 0.000057 | 0.000470 |
| 17 | 0.003050 | 0.003636 | 0.002136 | 0.002331 | 0.000052 | 0.000456 |
| 18 | 0.003033 | 0.003620 | 0.002117 | 0.002337 | 0.000054 | 0.000426 |
| 19 | 0.002956 | 0.003567 | 0.002046 | 0.002338 | 0.000051 | 0.000397 |
| 20 | 0.002868 | 0.003634 | 0.001988 | 0.002345 | 0.000053 | 0.000440 |
| 21 | 0.002993 | 0.003603 | 0.002077 | 0.002329 | 0.000049 | 0.000438 |
| 22 | 0.002984 | 0.003593 | 0.002054 | 0.002324 | 0.000049 | 0.000437 |
| 23 | 0.002956 | 0.003596 | 0.002058 | 0.002326 | 0.000047 | 0.000434 |
| 24 | - | 0.003598 | - | 0.002330 | - | 0.000433 |



## Testing


## `pusht_videosaur`

```
cat > /tmp/cjepa_eval.sh <<'SH'
set -u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate cjepa
cd /home/jichen/Downloads/cjepa
export PYTHONPATH=$(pwd)
export HYDRA_FULL_ERROR=1
export LD_LIBRARY_PATH=$HOME/miniconda3/envs/cjepa/lib/python3.10/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}
POLICY="pusht_videosaur_1"
for SEED in 0 1 2; do
  echo "=== seed=${SEED} start $(date)"
  python src/plan/run.py \
    seed=${SEED} \
    policy=${POLICY} \
    cache_dir=/home/jichen/.stable_worldmodel \
    world.history_size=1 \
    world.frame_skip=1 \
    plan_config.horizon=5 \
    plan_config.receding_horizon=5 \
    plan_config.action_block=5 \
    eval.eval_budget=50 \
    output.filename=planning_${POLICY}_seed_${SEED}.txt \
    eval.dataset_name=pusht_expert_train \
    eval.goal_offset_steps=25 \
    wandb.use_wandb=false
  echo "=== seed=${SEED} done $(date)"
done
echo "ALL DONE $(date)"
SH

nohup bash /tmp/cjepa_eval.sh > /tmp/cjepa_eval.log 2>&1 &
```



| Seed | Episodes ($N$) | Successes | Success Rate (%) | Total CEM Time (s) | Wall Time (s) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | 50 | 47 | 94.0 | 834.0 | 1253 |
| 1 | 50 | 44 | 88.0 | 824.2 | 1240 |
| 2 | 50 | 44 | 88.0 | 825.5 | 1239 |
| **Total / Mean** | **150** | **135** | **90.0 $\pm$ 3.46\*** | **827.9 (mean)** | **3732 (62.2 min)** |


### `pusht_videosaur_0`

| Seed | Episodes ($N$) | Successes | Success Rate (%) | Total CEM Time (s) | Wall Time (s) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | 50 | 42 | 84.0 | 840.8 | 1258 |
| 1 | 50 | 32 | 64.0 | 835.5 | 1253 |
| 2 | 50 | 38 | 76.0 | 835.3 | 1244 |
| **Total / Mean** | **150** | **112** | **74.7 $\pm$ 10.1** | **837.2 (mean)** | **3755 (62.6 min)** |

---

### `pusht_videosaur_2`

| Seed | Episodes ($N$) | Successes | Success Rate (%) | Total CEM Time (s) | Wall Time (s) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | 50 | 44 | 88.0 | 834.4 | 1255 |
| 1 | 50 | 39 | 78.0 | 845.9 | 1262 |
| 2 | 50 | 41 | 82.0 | 812.6 | 1234 |
| **Total / Mean** | **150** | **124** | **82.7 $\pm$ 5.0** | **831.0 (mean)** | **3751 (62.5 min)** |

```

export PYTHONPATH=$(pwd)
export HYDRA_FULL_ERROR=1
export LD_LIBRARY_PATH=$HOME/miniconda3/envs/cjepa/lib/python3.10/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}

for POLICY in pusht_videosaur_0 pusht_videosaur_2; do
  for SEED in 0 1 2; do
    echo "=== [${POLICY}] seed=${SEED} start $(date)"
    python src/plan/run.py \
      seed=${SEED} \
      policy=${POLICY} \
      cache_dir=/home/jichen/.stable_worldmodel \
      world.history_size=1 \
      world.frame_skip=1 \
      plan_config.horizon=5 \
      plan_config.receding_horizon=5 \
      plan_config.action_block=5 \
      eval.eval_budget=50 \
      output.filename=planning_${POLICY}_seed_${SEED}.txt \
      eval.dataset_name=pusht_expert_train \
      eval.goal_offset_steps=25 \
      wandb.use_wandb=false
    echo "=== [${POLICY}] seed=${SEED} done $(date)"
  done
done

echo "ALL DONE $(date)"
SH

nohup /tmp/run_pusht_eval_0_2.sh > /tmp/pusht_eval_0_2.log 2>&1 &
```



## Environment

Here is the list of environment-tuning commands executed, grouped by action:

### 1. Fix Missing Intel MKL Symlinks
```bash
E=~/miniconda3/envs/cjepa/lib
cd $E
for f in libmkl_*.so.2; do
  base=${f%.2}
  [ -e "$base" ] || ln -s "$f" "$base"
done
```

### 2. Clean Conflicting PyTorch Installations & Reinstall
```bash
SP=~/miniconda3/envs/cjepa/lib/python3.10/site-packages

# Remove corrupt and conflicting PyTorch/Torchvision directories
rm -rf $SP/torch $SP/torchvision $SP/torch-2.4.0-py3.10.egg-info \
       $SP/torchvision-0.19.0-py3.10.egg-info $SP/torch-2.14.0.dist-info \
       $SP/torchvision-0.29.0.dist-info $SP/torchgen $SP/functorch

# Move leftover metadata out of the environment
mkdir -p ~/cjepa_env_backup
mv $SP/torch-2.4.0-py3.10.egg-info ~/cjepa_env_backup/

# Clean reinstall via pip
source ~/miniconda3/etc/profile.d/conda.sh
conda activate cjepa
pip install --force-reinstall --no-deps torch==2.14.0 torchvision==0.29.0
```

### 3. Upgrade Hugging Face `datasets`
```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate cjepa
pip install -q "datasets==4.8.5"
```

### 4. Align Submodule Dependencies
```bash
# Checkout pinned version of stable-worldmodel containing AutoCostModel
cd /home/jichen/Downloads/cjepa/src/third_party/stable-worldmodel
git stash push -m "local pusht env kwargs tweak (pre-pin)"
git checkout 221ac820a1adea75bed99df45ab592bb5f42306c

# Reinstall the package in editable mode
source ~/miniconda3/etc/profile.d/conda.sh
conda activate cjepa
pip install -e . --no-deps -q

# Revert outdated transform workaround in stable-pretraining
cd /home/jichen/Downloads/cjepa/src/third_party/stable-pretraining
git stash push -m "RGB transform workaround for old torchvision" stable_pretraining/data/transforms.py
```

### 5. Export Runtime Dynamic Library Paths
```bash
export LD_LIBRARY_PATH=$HOME/miniconda3/envs/cjepa/lib/python3.10/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$(pwd)
export HYDRA_FULL_ERROR=1
```


## File Changes

### 1. Created File: `custom_models/__init__.py`
Created a new directory and Python module inside the repo root (`/home/jichen/Downloads/cjepa/custom_models/__init__.py`) containing:
* **Checkpoint unpickling aliases:** Added `sys.modules` mappings to redirect references from the legacy `custom_models` package to `src/` modules (`cjepa_predictor`, `dinowm_causal`, `dinowm_causal_AP_node`, `dinowm_causal_savi`).
* **Path injection:** Appended `src/third_party` to `sys.path`.
* **DINOv2 hidden states fix:** Added `_register_dinov2_output_capturing()` to register `Dinov2Model` and `Dinov2Encoder` into `transformers.utils.output_capturing._CAN_RECORD_REGISTRY`.

---

### 2. Modified Repository Submodules via Git
* **`src/third_party/stable-worldmodel/`**:
  * Stashed local uncommitted modifications using `git stash push -m "local pusht env kwargs tweak (pre-pin)"`.
  * Checked out commit `221ac820a1adea75bed99df45ab592bb5f42306c` (altering the working tree of this submodule).
* **`src/third_party/stable-pretraining/`**:
  * Stashed local modifications in `stable_pretraining/data/transforms.py` using `git stash push -m "RGB transform workaround for old torchvision"`.






Setup

```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda3
rm miniconda.sh

source $HOME/miniconda3/bin/activate
conda init bash
conda create -n cjepa python=3.10 -y
conda activate cjepa
conda install -y ffmpeg
pip install --upgrade pip uv
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
uv pip install seaborn webdataset swig einops torchcodec av accelerate tensorboard tensorboardX hickle pycocotools wget hydra-core omegaconf "pytorch-lightning>=2.0,<2.7"
```


```
# --- Download & Configure Push-T Checkpoint ---
cd ..
wget https://huggingface.co/HazelNam/CJEPA/resolve/main/cjepa-ckpts/pusht_videosaur_1_epoch_30_object.ckpt

mkdir -p ~/.stable_worldmodel/checkpoints/
mv pusht_videosaur_1_epoch_30_object.ckpt ~/.stable_worldmodel/checkpoints/pusht_videosaur_1_object.ckpt

pip install 'stable-worldmodel[env]'


# 1. Install stable-pretraining
git clone https://github.com/galilai-group/stable-pretraining.git
cd stable-pretraining
git checkout 92b5841
pip install -e .

# 2. Install stable-worldmodel
cd ..
git clone https://github.com/galilai-group/stable-worldmodel.git
cd stable-worldmodel
git checkout 221ac82
pip install -e .

# 3. Install nerv
cd ..
git clone https://github.com/Wuziyi616/nerv.git
cd nerv
git checkout v0.1.0
pip install -e .
```

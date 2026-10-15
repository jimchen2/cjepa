## Step 1: Generate the (Pendulum) Dataset

https://huggingface.co/datasets/jimchen2/pendulum-cjepa-3k-new/tree/main

```
~/Downloads/pendulum-cjepa-3k-new/videos/train
~/Downloads/pendulum-cjepa-3k-new/videos/val
~/Downloads/pendulum-cjepa-3k-new/wds
```

## Step 2: Run the Training and Extraction through VideoSaur

```
conda activate cjepa
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cudnn/lib:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

cd "$HOME/Downloads/cjepa" || exit 1

DATA="$HOME/Downloads/pendulum-cjepa-3k-new/wds"
RUN="$(date +%Y%m%d-%H%M%S)"
LOGS="$HOME/Downloads/pendulum-videosaur-logs-$RUN"
LOG="$HOME/Downloads/videosaur-training-$RUN.log"
PIDFILE="$HOME/Downloads/videosaur-training-$RUN.pid"

mkdir -p "$LOGS"

train_count=$(find "$DATA/train" -maxdepth 1 -name 'pendulum-train-*.tar' | wc -l)
val_count=$(find "$DATA/val" -maxdepth 1 -name 'pendulum-val-*.tar' | wc -l)

nohup env PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 NCCL_NVLS_ENABLE=0 python src/third_party/videosaur/videosaur/train.py \
  --log-dir "$LOGS" \
  src/third_party/videosaur/configs/videosaur/pendulum_dinov2_hf.yml \
  "dataset.train_shards=$DATA/train/pendulum-train-{000000..000018}.tar" \
  "dataset.val_shards=$DATA/val/pendulum-val-{000000..000004}.tar" "dataset.num_val_workers=5"  \
  > "$LOG" 2>&1 &
echo $! > "$PIDFILE"

tail -f "$LOG"
```

Training Checkpoints: https://huggingface.co/datasets/jimchen2/pendulum-cjepa-3k-clevrer/tree/main/videosaur-checkpoints

Generate an example video

```
cd ~/Downloads/cjepa

PYTHONPATH=. python \
  ~/Downloads/plot.py \
  --video ~/Downloads/pendulum-cjepa-3k-new/videos/train/video_00475.mp4 \
  --checkpoint ~/Downloads/pendulum-cjepa-3k-new/videosaur-checkpoints/videosaur_pendulum_dinov2/checkpoints/step=5000.ckpt \
  --config ~/Downloads/cjepa/src/third_party/videosaur/configs/videosaur/pendulum_dinov2_hf.yml \
  --output ~/Downloads/video_00475_4slot_overlay.mp4 \
  --device cuda
```

Extraction

```
conda activate cjepa
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cudnn/lib:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"


RAW="$HOME/Downloads/pendulum-cjepa-3k-new"
ROOT="$HOME/.stable_worldmodel"

export WEIGHT="$RAW/videosaur-checkpoints/videosaur_pendulum_dinov2/checkpoints/step=5000.ckpt"
export SAVE_DIR="$HOME/Downloads/pendulum_videosaur_4slots.pkl"

# Make the directory layout expected by extract_videosaur.py.
mkdir -p \
  "$ROOT/pendulum_train" \
  "$ROOT/pendulum_val"

ln -sfn "$RAW/videos/train" "$ROOT/pendulum_train/videos"
ln -sfn "$RAW/videos/val" "$ROOT/pendulum_val/videos"

mkdir -p "$(dirname "$SAVE_DIR")"

PYTHONPATH=. python src/third_party/slotformer/base_slots/extract_videosaur.py \
  --weight "$WEIGHT" \
  --data_root "$ROOT" \
  --save_path "$SAVE_DIR" \
  --dataset pendulum \
  --videosaur_config src/third_party/videosaur/configs/videosaur/pendulum_dinov2_hf.yml
```

<!-- 
## Step 3: Run the Training

```
export WANDB_MODE=offline
export PYTHONPATH=$(pwd)
export SLOTPATH="/home/jichen/Downloads/pendulum-cjepa-dataset/videosaur/pendulum_videosaur_4slots.pkl"

python src/train/train_causalwm_from_clevrer_slot.py \
    cache_dir="${HOME}/.stable_worldmodel" \
    output_model_name="pendulum_cjepa" \
    dataset_name="pendulum" \
    num_workers=4 \
    batch_size=32 \
    trainer.max_epochs=30 \
    num_masked_slots=1 \
    predictor_lr=5e-4 \
    dinowm.history_size=6 \
    dinowm.num_preds=10 \
    frameskip=1 \
    videosaur.NUM_SLOTS=4 \
    videosaur.SLOT_DIM=128 \
    predictor.heads=8 \
    embedding_dir="${SLOTPATH}"
```

## Step 4: Run the Testing

Run through the physical simulation. Train on ALOE to answer it. -->

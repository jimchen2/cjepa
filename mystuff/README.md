## Step 1: Generate the (Pendulum) Dataset

https://huggingface.co/datasets/jimchen2/pendulum-cjepa-160f/tree/main

Extract the shards

## Step 2: Run the Extraction through VideoSaur


```
cd "$HOME/Downloads/cjepa" || exit 1

DATA="$HOME/Downloads/pendulum-cjepa-3k-clevrer/wds"
RUN="$(date +%Y%m%d-%H%M%S)"
LOGS="$HOME/Downloads/pendulum-videosaur-logs-$RUN"
LOG="$HOME/Downloads/videosaur-training-$RUN.log"
PIDFILE="$HOME/Downloads/videosaur-training-$RUN.pid"

mkdir -p "$LOGS"

train_count=$(find "$DATA/train" -maxdepth 1 -name 'pendulum-train-*.tar' | wc -l)
val_count=$(find "$DATA/val" -maxdepth 1 -name 'pendulum-val-*.tar' | wc -l)

env PYTHONPATH=. python src/third_party/videosaur/videosaur/train.py \
  --log-dir "$LOGS" \
  src/third_party/videosaur/configs/videosaur/clevrer_dinov2_hf.yml \
  "dataset.train_shards=$DATA/train/pendulum-train-{000000..000018}.tar" \
  "dataset.val_shards=$DATA/val/pendulum-val-{000000..000004}.tar" \
  globals.NUM_GPUS=1 \
  globals.NUM_SLOTS=4 \
  globals.BATCH_SIZE_PER_GPU=32 \
  trainer.devices=1 \
  trainer.max_steps=5000 \
  trainer.val_check_interval=1000 \
  dataset.val_size=600 \
  dataset.num_val_workers=4 \
  dataset.train_pipeline.num_frameskip=1 \
  dataset.val_pipeline.num_frameskip=1 \
  checkpoint_every_n_steps=1000 \
  wandb.enable=false 
```

Checkpoints: https://huggingface.co/datasets/jimchen2/pendulum-cjepa-3k-clevrer/tree/main/videosaur-checkpoints

## Step 3: Run the Training

Pull the thing at https://huggingface.co/datasets/jimchen2/pendulum-cjepa-dataset

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

Run through the physical simulation. Train on ALOE to answer it.

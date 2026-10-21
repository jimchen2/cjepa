## Output layout

Nothing gets dumped loose in `~/Downloads` anymore. Every artifact produced by the steps
below lives under one folder, `~/Downloads/pendulum_output/`:

```
~/Downloads/pendulum_output/
├── dataset/                         # Step 1 (inputs / derived shards)
│   ├── videos/
│   │   ├── train/video_00000.mp4 ... #   pendulum-cjepa-3k-new/videos/train
│   │   └── val/video_*.mp4           #   pendulum-cjepa-3k-new/videos/val
│   └── wds/
│       ├── train/pendulum-train-000000.tar ...
│       └── val/pendulum-val-000000.tar ...
├── model/                           # Step 2 (weights + metrics), one folder per run
│   ├── <RUN>/                       #   e.g. 20250101-120000/
│   │   ├── checkpoints/             #     step=5000.ckpt (last.ckpt, best.ckpt ...)
│   │   ├── metrics/version_0/metrics.csv
│   │   └── tb/                      #     tensorboard event files
│   ├── latest -> <RUN>              #   symlink to the most recent run
│   └── videosaur_pendulum_dinov2/checkpoints/step=5000.ckpt   # downloaded weights
├── logs/                            # Step 2 (process logs), one folder per run
│   └── <RUN>/
│       ├── train.log                #   stdout/stderr of the training process
│       └── train.pid
└── extraction/                      # Step 3 (slot extractions + example renders)
    ├── pendulum_videosaur_4slots_*.pkl
    └── examples/video_00475_4slot_overlay.mp4
```

Set these once per shell (and re-run the `mkdir` after opening a fresh terminal):

```
export PENDULUM_OUT="$HOME/Downloads/pendulum_output"
export REPO="$HOME/Downloads/cjepa"

mkdir -p "$PENDULUM_OUT"/{dataset,model,logs,extraction/examples}
```

## Step 1: Generate the (Pendulum) Dataset

https://huggingface.co/datasets/jimchen2/pendulum-cjepa-3k-new/tree/main

Download the dataset **into the output folder** so it lands at:

```
$PENDULUM_OUT/dataset/videos/train
$PENDULUM_OUT/dataset/videos/val
$PENDULUM_OUT/dataset/wds
```

```
# e.g. with the HF CLI (or use `git clone` / the browser download button)
pip install -q "huggingface_hub[cli]"

mkdir -p "$PENDULUM_OUT/dataset"
huggingface-cli download jimchen2/pendulum-cjepa-3k-new \
  --repo-type dataset \
  --local-dir "$PENDULUM_OUT/dataset"

# make sure the expected layout exists
ls "$PENDULUM_OUT/dataset/videos/train" | head
ls "$PENDULUM_OUT/dataset/wds/train" | head
```

## Step 2: Run the Training through VideoSaur

Checkpoints and metrics land in `$PENDULUM_OUT/model/<RUN>/`; the process log lands in
`$PENDULUM_OUT/logs/<RUN>/`.

```
conda activate cjepa
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cudnn/lib:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

cd "$REPO" || exit 1

DATA="$PENDULUM_OUT/dataset/wds"
RUN="$(date +%Y%m%d-%H%M%S)"

LOGS="$PENDULUM_OUT/logs/$RUN"          # process log + pid
MODEL="$PENDULUM_OUT/model/$RUN"        # checkpoints + metrics.csv + tb
LOG="$LOGS/train.log"
PIDFILE="$LOGS/train.pid"

mkdir -p "$LOGS" "$MODEL"
ln -sfn "$RUN" "$PENDULUM_OUT/model/latest"
ln -sfn "$RUN" "$PENDULUM_OUT/logs/latest"

train_count=$(find "$DATA/train" -maxdepth 1 -name 'pendulum-train-*.tar' | wc -l)
val_count=$(find "$DATA/val" -maxdepth 1 -name 'pendulum-val-*.tar' | wc -l)

nohup env PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 NCCL_NVLS_ENABLE=0 python src/third_party/videosaur/videosaur/train.py \
  --log-dir "$MODEL" --no-sub-logdirs \
  src/third_party/videosaur/configs/videosaur/pendulum_dinov2_hf.yml \
  "dataset.train_shards=$DATA/train/pendulum-train-{000000..000018}.tar" \
  "dataset.val_shards=$DATA/val/pendulum-val-{000000..000004}.tar" "dataset.num_val_workers=5"  \
  > "$LOG" 2>&1 &
echo $! > "$PIDFILE"

tail -f "$LOG"
```

`--no-sub-logdirs` keeps the run flat, so you get exactly:

```
$PENDULUM_OUT/model/<RUN>/checkpoints/step=5000.ckpt
$PENDULUM_OUT/model/<RUN>/metrics/version_0/metrics.csv
$PENDULUM_OUT/model/<RUN>/tb/
```

To resume a run, point `--continue` at the run folder instead of starting a new one:

```
# ... --continue "$PENDULUM_OUT/model/latest" --log-dir "$PENDULUM_OUT/model/latest" ...
```

Pre-trained checkpoints (keep them under `model/`):

```
mkdir -p "$PENDULUM_OUT/model/videosaur_pendulum_dinov2/checkpoints"
wget -c -O "$PENDULUM_OUT/model/videosaur_pendulum_dinov2/checkpoints/step=5000.ckpt" \
  "https://huggingface.co/datasets/jimchen2/pendulum-cjepa-3k-clevrer/resolve/main/videosaur-checkpoints/videosaur_pendulum_dinov2/checkpoints/step%3D5000.ckpt"
```

Training Checkpoints: https://huggingface.co/datasets/jimchen2/pendulum-cjepa-3k-clevrer/tree/main/videosaur-checkpoints

## Step 3: Run the Extraction

Generate an example video (written to `extraction/examples/`):

```
cd "$REPO"

mkdir -p "$PENDULUM_OUT/extraction/examples"

PYTHONPATH=. python \
  mystuff/extraction_reports/pendulum_plot.py \
  --video "$PENDULUM_OUT/dataset/videos/train/video_00475.mp4" \
  --checkpoint "$PENDULUM_OUT/model/videosaur_pendulum_dinov2/checkpoints/step=5000.ckpt" \
  --config src/third_party/videosaur/configs/videosaur/pendulum_dinov2_hf.yml \
  --output "$PENDULUM_OUT/extraction/examples/video_00475_4slot_overlay.mp4" \
  --device cuda
```

Extraction (`.pkl` written to `extraction/`; the script appends the checkpoint name to the
filename):

```
conda activate cjepa
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cudnn/lib:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

RAW="$PENDULUM_OUT/dataset"
ROOT="$HOME/.stable_worldmodel"          # scratch root only, the outputs go to $PENDULUM_OUT

export WEIGHT="$PENDULUM_OUT/model/videosaur_pendulum_dinov2/checkpoints/step=5000.ckpt"
export SAVE_DIR="$PENDULUM_OUT/extraction/pendulum_videosaur_4slots.pkl"

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

Final tree:

```
$PENDULUM_OUT/
├── dataset/{videos/{train,val},wds/{train,val}}
├── model/{latest -> <RUN>, <RUN>/checkpoints, <RUN>/metrics, <RUN>/tb, videosaur_pendulum_dinov2/checkpoints}
├── logs/{latest -> <RUN>, <RUN>/train.log, <RUN>/train.pid}
└── extraction/{pendulum_videosaur_4slots_*.pkl, examples/video_00475_4slot_overlay.mp4}
```

<!--
## Step 3: Run the Training

```
export WANDB_MODE=offline
export PYTHONPATH=$(pwd)
export SLOTPATH="$PENDULUM_OUT/extraction/pendulum_videosaur_4slots.pkl"

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

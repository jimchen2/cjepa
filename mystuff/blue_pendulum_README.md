## Output layout

Everything generated below goes under `~/Downloads/pendulum_output/` — nothing is dumped directly in `~/Downloads`:

```
~/Downloads/pendulum_output/
├── model/
│   └── pendulum-videosaur-<RUN>/               # Step 2 VideoSaur --log-dir
│       └── videosaur_pendulum/<timestamp>_pendulum_dinov2/
│           └── checkpoints/, metrics/, tb/, settings.yaml
├── logs/
│   └── videosaur-training-<RUN>.log, videosaur-training-<RUN>.pid
├── extraction/
│   └── pendulum_videosaur_4slots*.pkl          # Step 3 extracted slots
└── examples/
    └── video_00475_4slot_overlay.mp4           # Step 3 example video
```

Inputs that stay where they are: the repo (`~/Downloads/cjepa`), the blue Hugging Face dataset (`~/Downloads/jimchen2-blue-pendulum-output/pendulum-cjepa-3k-new`), the existing published checkpoints (if present, `~/Downloads/pendulum_output/pendulum-cjepa-3k-new/videosaur-checkpoints`) and the extraction `--data_root` (`~/.stable_worldmodel`). Training outputs still go under `~/Downloads/pendulum_output/`. The blue dataset folder contains the 256×256 videos and WDS shards; it does not include model checkpoints.
Each step sets `OUT` again so the blocks can be run independently.

## Step 1: Generate the (Pendulum) Dataset

https://huggingface.co/jimchen2/jimchen2-blue-pendulum-output/tree/main/pendulum-cjepa-3k-new

Input dataset, expected at:

```
~/Downloads/jimchen2-blue-pendulum-output/pendulum-cjepa-3k-new/videos/train
~/Downloads/jimchen2-blue-pendulum-output/pendulum-cjepa-3k-new/videos/val
~/Downloads/jimchen2-blue-pendulum-output/pendulum-cjepa-3k-new/wds
```

## Step 2: Run the Training through VideoSaur

```
conda activate cjepa
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cudnn/lib:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

cd "$HOME/Downloads/cjepa" || exit 1

OUT="$HOME/Downloads/pendulum_output"
DATASET="$HOME/Downloads/jimchen2-blue-pendulum-output/pendulum-cjepa-3k-new"
DATA="$DATASET/wds"
RUN="$(date +%Y%m%d-%H%M%S)"
MODEL="$OUT/model/pendulum-videosaur-$RUN"
LOG="$OUT/logs/videosaur-training-$RUN.log"
PIDFILE="$OUT/logs/videosaur-training-$RUN.pid"

mkdir -p "$MODEL" "$OUT/logs"

tmux new -d -s "blue-pendulum-train" \
"env PYTHONPATH=. CUDA_VISIBLE_DEVICES=4 NCCL_NVLS_ENABLE=0 python src/third_party/videosaur/videosaur/train.py \
  --log-dir '$MODEL' \
  src/third_party/videosaur/configs/videosaur/pendulum_dinov2_hf.yml \
  'dataset.train_shards=$DATA/train/pendulum-train-{000000..000018}.tar' \
  'dataset.val_shards=$DATA/val/pendulum-val-{000000..000004}.tar' \
  'dataset.num_val_workers=5'; exec bash" \; \
  pipe-pane -o "cat >> $LOG" \; \
  attach

tmux a -t blue-pendulum-train
```

Training Checkpoints: https://huggingface.co/datasets/jimchen2/pendulum-cjepa-3k-clevrer/tree/main/videosaur-checkpoints

## Step 3: Run the Extraction

Generate an example video

```
cd ~/Downloads/cjepa

OUT="$HOME/Downloads/pendulum_output"
mkdir -p "$OUT/examples"

PYTHONPATH=. python \
  mystuff/extraction_reports/pendulum_plot.py \
  --video "$HOME/Downloads/jimchen2-blue-pendulum-output/pendulum-cjepa-3k-new/videos/train/video_00475.mp4" \
  --checkpoint "$HOME/Downloads/pendulum_output/pendulum-cjepa-3k-new/videosaur-checkpoints/videosaur_pendulum_dinov2/checkpoints/step=5000.ckpt" \
  --config ~/Downloads/cjepa/src/third_party/videosaur/configs/videosaur/pendulum_dinov2_hf.yml \
  --output "$OUT/examples/video_00475_4slot_overlay.mp4" \
  --device cuda
```

Extraction

```
conda activate cjepa
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cudnn/lib:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

OUT="$HOME/Downloads/pendulum_output"
RAW="$HOME/Downloads/jimchen2-blue-pendulum-output/pendulum-cjepa-3k-new"
ROOT="$HOME/.stable_worldmodel"

CHECKPOINTS="$HOME/Downloads/pendulum_output/pendulum-cjepa-3k-new/videosaur-checkpoints"
export WEIGHT="$CHECKPOINTS/videosaur_pendulum_dinov2/checkpoints/step=5000.ckpt"
export SAVE_DIR="$OUT/extraction/pendulum_videosaur_4slots.pkl"

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

Note: `extract_videosaur.py` appends the checkpoint name to `--save_path` (and writes any `partial_*.pkl` next to it), so the extracted slots end up in `$OUT/extraction/`.

<!-- 
## Step 3: Run the Training

```
export WANDB_MODE=offline
export PYTHONPATH=$(pwd)
export SLOTPATH="${HOME}/Downloads/pendulum_output/extraction/pendulum_videosaur_4slots.pkl"

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

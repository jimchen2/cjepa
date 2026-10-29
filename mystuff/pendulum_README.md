```
hf upload jimchen2/jimchen2-pendulum-output-1 "/home/jichen/Downloads/pendulum_output" .
```

```
cp ~/Downloads/cjepa/mystuff/config/pendulum_dinov2_hf.yml ~/Downloads/cjepa/src/third_party/videosaur/configs/videosaur/
```

## Output layout

Everything generated below goes under `~/Downloads/pendulum_output/` — nothing is dumped directly in `~/Downloads`:

```
~/Downloads/pendulum_output/
├── model/
│   └── pendulum-videosaur-<RUN>/               # Step 2 VideoSaur --log-dir
│       └── videosaur_pendulum/<timestamp>_pendulum_dinov2/
│           └── checkpoints/, metrics/, tb/, settings.yaml
├── checkpoint-archive/
│   └── pendulum-videosaur-<RUN>/
│       ├── step=<STEP>.ckpt                    # Step 2 archived checkpoints
│       └── step=<STEP>_video_*_4slot_overlay.mp4 # Step 3 example videos
├── logs/
│   └── videosaur-training-<RUN>.log, videosaur-training-<RUN>.pid
└── extraction/
    └── pendulum_videosaur_4slots*.pkl          # Step 3 extracted slots
```

Inputs that stay where they are: the repo (`~/Downloads/cjepa`), the Hugging Face dataset (`~/Downloads/pendulum_output/pendulum-cjepa-3k-new`, incl. its published checkpoints) and the extraction `--data_root` (`~/.stable_worldmodel`).
Each step sets `OUT` again so the blocks can be run independently.

## Step 1: Generate the (Pendulum) Dataset

https://huggingface.co/datasets/jimchen2/pendulum_output/pendulum-cjepa-3k-new/tree/main

Input dataset, expected at:

```
~/Downloads/pendulum_output/pendulum-cjepa-3k-new/videos/train
~/Downloads/pendulum_output/pendulum-cjepa-3k-new/videos/val
~/Downloads/pendulum_output/pendulum-cjepa-3k-new/wds
```

## Step 2: Run the Training through VideoSaur

```
conda activate cjepa
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cudnn/lib:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

cd "$HOME/Downloads/cjepa" || exit 1

OUT="$HOME/Downloads/pendulum_output"
DATA="$HOME/Downloads/pendulum_output/pendulum-cjepa-3k-new/wds"
RUN="$(date +%Y%m%d-%H%M%S)"
MODEL="$OUT/model/pendulum-videosaur-$RUN"
LOG="$OUT/logs/videosaur-training-$RUN.log"
PIDFILE="$OUT/logs/videosaur-training-$RUN.pid"

mkdir -p "$MODEL" "$OUT/logs"

tmux new -d -s "pendulum-train" \
"env PYTHONPATH=. CUDA_VISIBLE_DEVICES=4 NCCL_NVLS_ENABLE=0 python src/third_party/videosaur/videosaur/train.py \
  --log-dir '$MODEL' \
  src/third_party/videosaur/configs/videosaur/pendulum_dinov2_hf.yml \
  'dataset.train_shards=$DATA/train/pendulum-train-{000000..000018}.tar' \
  'dataset.val_shards=$DATA/val/pendulum-val-{000000..000004}.tar' \
  'dataset.num_val_workers=5'; exec bash" \; \
  pipe-pane -o "cat >> $LOG" \; \
  attach

tmux a -t pendulum-train
```

Training Checkpoints: https://huggingface.co/datasets/jimchen2/pendulum-cjepa-3k-clevrer/tree/main/videosaur-checkpoints

Try to archive the checkpoints

```
OUT="$HOME/Downloads/pendulum_output"
MODEL="$(find "$OUT/model" -mindepth 1 -maxdepth 1 -type d \
  -name 'pendulum-videosaur-*' -printf '%T@ %p\n' |
  sort -nr | head -1 | cut -d' ' -f2-)"

if [ -z "$MODEL" ]; then
  echo "Error: No model directory found in $OUT/model. Start Step 2 first." >&2
  exit 1
fi

ARCHIVE="$OUT/checkpoint-archive/$(basename "$MODEL")"
SESSION="pendulum-train"
WATCH_SESSION="checkpoint-copy-$(date +%H%M%S)"
WATCH_SCRIPT="$OUT/logs/$WATCH_SESSION.sh"

mkdir -p "$ARCHIVE" "$OUT/logs"

cat > "$WATCH_SCRIPT" <<'BASH'
#!/usr/bin/env bash
model="$1"
archive="$2"
train_session="$3"
deadline=$((SECONDS + 7 * 24 * 60 * 60))

echo "Watching model dir: $model"
echo "Archiving to:       $archive (every 10,000 steps)"

scan_checkpoints() {
  while IFS= read -r -d '' f; do
    dest="$archive/$(basename "$f")"
    [ -f "$dest" ] && continue

    size1=$(stat -c%s "$f" 2>/dev/null) || continue
    sleep 5
    [ -f "$f" ] || continue
    size2=$(stat -c%s "$f" 2>/dev/null) || continue

    if [ "$size1" = "$size2" ] && [ "$size2" -gt 0 ]; then
      cp -- "$f" "$dest.tmp" && mv -- "$dest.tmp" "$dest"
      echo "[$(date +%H:%M:%S)] Archived $(basename "$f")"
    fi
  done < <(find "$model" -type f -regex '.*/step=[1-9][0-9]*0000\.ckpt' -print0)
}

while (( SECONDS < deadline )); do
  scan_checkpoints

  if [ -f "$archive/step=100000.ckpt" ]; then
    echo "100000-step checkpoint archived; watcher finished."
    exit 0
  fi

  if ! tmux has-session -t "$train_session" 2>/dev/null; then
    echo "Training session ended; doing a final check."
    sleep 6
    scan_checkpoints
    exit 0
  fi

  sleep 3
done

echo "Seven-day safety limit reached; watcher finished."
BASH

chmod +x "$WATCH_SCRIPT"

tmux new-session -d -s "$WATCH_SESSION" \
  "bash '$WATCH_SCRIPT' '$MODEL' '$ARCHIVE' '$SESSION'; exec bash"

echo "Watcher tmux session: $WATCH_SESSION"
echo "Archive directory:    $ARCHIVE"
tmux attach -t "$WATCH_SESSION"
```

## Step 3: Run the Extraction

Generate an example video

```
conda activate cjepa
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cudnn/lib:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
cd ~/Downloads/cjepa

OUT="$HOME/Downloads/pendulum_output"
VID_DIR="$OUT/pendulum-cjepa-3k-new/videos/train"
CONFIG="$HOME/Downloads/cjepa/src/third_party/videosaur/configs/videosaur/pendulum_dinov2_hf.yml"

for item in \
  "pendulum-ft-videosaur-20261022-201603:100000" \
  "pendulum-ft-videosaur-20261022-201603:34000" \
  "pendulum-videosaur-20261022-194905:100000" \
  "pendulum-videosaur-20261028-090204:10000" \
  "pendulum-videosaur-20261028-090204:20000" \
  "pendulum-videosaur-20261028-090204:30000" \
  "pendulum-videosaur-20261028-090204:40000" \
  "pendulum-videosaur-20261028-090204:50000" \
  "pendulum-videosaur-20261028-090204:60000"
do
  RUN="${item%:*}"
  STEP="${item#*:}"
  
  CKPT_DIR="$OUT/checkpoint-archive/$RUN"
  
  for VID in "video_00046" "video_00112" "video_00137"; do
    OUT_FILE="$CKPT_DIR/step=${STEP}_${VID}_4slot_overlay.mp4"
    
    PYTHONPATH=. python mystuff/extraction_reports/pendulum_plot.py \
      --video "$VID_DIR/${VID}.mp4" \
      --checkpoint "$CKPT_DIR/step=${STEP}.ckpt" \
      --config "$CONFIG" \
      --output "$OUT_FILE" \
      --device cuda
  done
done
```

Extraction

```
conda activate cjepa
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cudnn/lib:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

OUT="$HOME/Downloads/pendulum_output"
RAW="$HOME/Downloads/pendulum_output/pendulum-cjepa-3k-new"
ROOT="$HOME/.stable_worldmodel"

export WEIGHT="$RAW/videosaur-checkpoints/videosaur_pendulum_dinov2/checkpoints/step=5000.ckpt"
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

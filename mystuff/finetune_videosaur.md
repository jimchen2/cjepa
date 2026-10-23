## Fine-tune VideoSaur on Pendulum, starting from the original C-JEPA CLEVRER checkpoint

Start from the official `clevrer_videosaur_model.ckpt` (from the main cjepa README, HazelNam/CJEPA) instead of from scratch, then train on Pendulum.

## Step 1: Download the original checkpoint

```
OUT="$HOME/Downloads/pendulum_output"
mkdir -p "$OUT/model"
wget -c -O "$OUT/model/clevrer_videosaur_model.ckpt" \
  https://huggingface.co/HazelNam/CJEPA/resolve/main/clevrer_videosaur_model.ckpt
```

## Step 2: Fine-tune

The important line is `model.load_weights=...`.

**Do not use `--continue`** (the line in `pendulum_README.md`). `--continue` resumes the old run: step 100000, the old optimizer and a decayed LR. `model.load_weights` loads only the weights and starts fresh from step 0.

Also change the slots

```
conda activate cjepa
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cudnn/lib:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
cd "$HOME/Downloads/cjepa" || exit 1

OUT="$HOME/Downloads/pendulum_output"
DATA="$OUT/pendulum-cjepa-3k-new/wds"
INIT="$OUT/model/clevrer_videosaur_model.ckpt"
RUN="$(date +%Y%m%d-%H%M%S)"
MODEL="$OUT/model/pendulum-ft-videosaur-$RUN"
LOG="$OUT/logs/videosaur-ft-$RUN.log"
mkdir -p "$MODEL" "$OUT/logs"

train_max=$(printf "%06d" $(($(find "$DATA/train" -maxdepth 1 -name 'pendulum-train-*.tar' | wc -l) - 1)))
val_max=$(printf "%06d" $(($(find "$DATA/val" -maxdepth 1 -name 'pendulum-val-*.tar' | wc -l) - 1)))

tmux new -d -s "pendulum-ft" \
"env PYTHONPATH=. CUDA_VISIBLE_DEVICES=3 NCCL_NVLS_ENABLE=0 python src/third_party/videosaur/videosaur/train.py \
  --log-dir '$MODEL' \
  src/third_party/videosaur/configs/videosaur/pendulum_dinov2_hf.yml \
  'model.load_weights=$INIT' \
  'dataset.train_shards=$DATA/train/pendulum-train-{000000..$train_max}.tar' \
  'dataset.val_shards=$DATA/val/pendulum-val-{000000..$val_max}.tar' \
  'dataset.num_val_workers=5' \
  'trainer.max_steps=100000' \
  'globals.BASE_LR=0.00003'; exec bash" \; \
  pipe-pane -o "cat >> $LOG" \; \
  attach

tmux a -t pendulum-ft
```

## Step 3: Visualize + extract with the fine-tuned checkpoint

```
cd ~/Downloads/cjepa
OUT="$HOME/Downloads/pendulum_output"
RAW="$OUT/pendulum-cjepa-3k-new"
ROOT="$HOME/.stable_worldmodel"

# newest fine-tuned checkpoint (or pick a step=XXXX.ckpt yourself)
export WEIGHT=$(ls -t "$OUT"/model/pendulum-ft-videosaur-*/videosaur_pendulum/*/checkpoints/*.ckpt | head -1)
echo "$WEIGHT"

mkdir -p "$OUT/examples" "$OUT/extraction"
PYTHONPATH=. python mystuff/extraction_reports/pendulum_plot.py \
  --video "$RAW/videos/train/video_00475.mp4" \
  --checkpoint "$WEIGHT" \
  --config src/third_party/videosaur/configs/videosaur/pendulum_dinov2_hf.yml \
  --output "$OUT/examples/video_00475_4slot_overlay_ft.mp4" \
  --device cuda

mkdir -p "$ROOT/pendulum_train" "$ROOT/pendulum_val"
ln -sfn "$RAW/videos/train" "$ROOT/pendulum_train/videos"
ln -sfn "$RAW/videos/val"   "$ROOT/pendulum_val/videos"

export SAVE_DIR="$OUT/extraction/pendulum_videosaur_ft_4slots.pkl"
PYTHONPATH=. python src/third_party/slotformer/base_slots/extract_videosaur.py \
  --weight "$WEIGHT" \
  --data_root "$ROOT" \
  --save_path "$SAVE_DIR" \
  --dataset pendulum \
  --videosaur_config src/third_party/videosaur/configs/videosaur/pendulum_dinov2_hf.yml
```


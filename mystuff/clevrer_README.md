## Output layout

Everything generated below goes under `~/Downloads/clevrer_output/` — nothing is dumped directly in `~/Downloads`:

```
~/Downloads/clevrer_output/
├── data/
│   ├── video_train.zip, video_validation.zip   # Step 1 downloads
│   ├── train/, val/                            # Step 1 unzipped raw .mp4 files
│   └── clevrer_wds_mp4/{train,val}/            # Step 2 WebDataset shards
├── model/
│   └── clevrer-videosaur-<RUN>/                # Step 3 VideoSaur --log-dir
│       └── videosaur_clevrer/<timestamp>_clevrer_dinov2/
│           └── checkpoints/, metrics/, tb/, settings.yaml
├── logs/
│   └── clevrer-training-<RUN>.log, clevrer-training-<RUN>.pid
├── extraction/
│   └── clevrer_videosaur_7slots*.pkl           # Step 4 extracted slots
└── examples/
    └── video_10000_7slot_overlay.mp4           # Step 4 example video
```

Inputs that stay where they are: the repo (`~/Downloads/cjepa`) and the extraction `--data_root` (`~/.stable_worldmodel`).
Each step sets `OUT` again so the blocks can be run independently.

## Step 1: Dataset

https://clevrer.csail.mit.edu/

```
OUT="$HOME/Downloads/clevrer_output"
mkdir -p "$OUT/data/train" "$OUT/data/val"

wget -c -P "$OUT/data" https://data.csail.mit.edu/clevrer/videos/train/video_train.zip
wget -c -P "$OUT/data" https://data.csail.mit.edu/clevrer/videos/validation/video_validation.zip
unzip -q -n "$OUT/data/video_train.zip" -d "$OUT/data/train"
unzip -q -n "$OUT/data/video_validation.zip" -d "$OUT/data/val"
```

## Step 2: Shard it

```
OUT="$HOME/Downloads/clevrer_output"

for split in train val; do
  find "$OUT/data/${split}" -type f -name "*.mp4" -exec mv {} "$OUT/data/${split}/" \;
  find "$OUT/data/${split}" -type d ! -path "$OUT/data/${split}" -exec rm -rf {} +
done

conda activate cjepa
cd "$HOME/Downloads/cjepa" || exit 1

# Check how ROOT_DIR is currently defined in the script
grep "ROOT_DIR" dataset/clevrer/save_clevrer_webdataset_mp4.py

# Point ROOT_DIR at clevrer_output/data: the script reads $ROOT_DIR/{train,val}/*.mp4
# and writes the shards to $ROOT_DIR/clevrer_wds_mp4/{train,val}
sed -i 's|ROOT_DIR="clevrer_video"|ROOT_DIR="'"$OUT/data"'"|g' dataset/clevrer/save_clevrer_webdataset_mp4.py

# Run the shard conversion script
PYTHONPATH=. python dataset/clevrer/save_clevrer_webdataset_mp4.py

```

## Step 3: Train through Videosaur

```
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cudnn/lib:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

OUT="$HOME/Downloads/clevrer_output"
DATA="$OUT/data/clevrer_wds_mp4"
RUN="$(date +%Y%m%d-%H%M%S)"
MODEL="$OUT/model/clevrer-videosaur-$RUN"
LOG="$OUT/logs/clevrer-training-$RUN.log"
PIDFILE="$OUT/logs/clevrer-training-$RUN.pid"

mkdir -p "$MODEL" "$OUT/logs"

train_max=$(printf "%06d" $(($(find "$DATA/train" -maxdepth 1 -name 'clevrer-train-*.tar' | wc -l) - 1)))
val_max=$(printf "%06d" $(($(find "$DATA/val" -maxdepth 1 -name 'clevrer-val-*.tar' | wc -l) - 1)))

cd ~/Downloads/cjepa

tmux new -d -s "train" \
"env PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 NCCL_NVLS_ENABLE=0 python src/third_party/videosaur/videosaur/train.py \
  --log-dir '$MODEL' \
  src/third_party/videosaur/configs/videosaur/clevrer_dinov2_hf.yml \
  'dataset.train_shards=$DATA/train/clevrer-train-{000000..$train_max}.tar' \
  'dataset.val_shards=$DATA/val/clevrer-val-{000000..$val_max}.tar' \
  'globals.BATCH_SIZE_PER_GPU=128' \
  'dataset.num_workers=16'; exec bash" \; \
  pipe-pane -o "cat >> $LOG" \; \
  attach

tmux a -t train
```

Checkpoint: https://huggingface.co/jimchen2/clevrer_dinov2_checkpoint/blob/main/checkpoints/step%3D43000.ckpt

## Step 4: Extract

Generate an example video

```
cd ~/Downloads/cjepa

OUT="$HOME/Downloads/clevrer_output"
mkdir -p "$OUT/examples" "$OUT/model"

# --checkpoint: a trained .ckpt from $OUT/model/clevrer-videosaur-<RUN>/.../checkpoints/,
# or the Hugging Face checkpoint above downloaded to $OUT/model/clevrer_videosaur_model.ckpt
# --video: any raw CLEVRER .mp4, e.g. one from $OUT/data/train/ (or val/)
PYTHONPATH=. python mystuff/extraction_reports/clevrer_plot.py \
  --video "$OUT/data/train/video_10000.mp4" \
  --checkpoint "$OUT/model/clevrer_videosaur_model.ckpt" \
  --config ~/Downloads/cjepa/src/third_party/videosaur/configs/videosaur/clevrer_dinov2_hf.yml \
  --output "$OUT/examples/video_10000_7slot_overlay.mp4" \
  --side_by_side \
  --device cuda
```

Extraction

```
cd ~/Downloads/cjepa

OUT="$HOME/Downloads/clevrer_output"

export WEIGHT="$OUT/model/clevrer_videosaur_model.ckpt"
export SAVE_DIR="$OUT/extraction/clevrer_videosaur_7slots.pkl"

mkdir -p "$OUT/extraction"

PYTHONPATH=. python src/third_party/slotformer/base_slots/extract_videosaur.py \
  --weight "$WEIGHT" \
  --data_root="$HOME/.stable_worldmodel" \
  --save_path="$SAVE_DIR" \
  --dataset="clevrer" \
  --videosaur_config="src/third_party/videosaur/configs/videosaur/clevrer_dinov2_hf.yml"
```

Note: `extract_videosaur.py` appends the checkpoint name to `--save_path` (and writes any `partial_*.pkl` next to it), so the extracted slots end up in `$OUT/extraction/`.

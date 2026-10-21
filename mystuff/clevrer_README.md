## Output layout

Nothing gets dumped loose in `~/Downloads` anymore. Every artifact produced by the steps
below lives under one folder, `~/Downloads/clevrer_output/`:

```
~/Downloads/clevrer_output/
├── dataset/                      # Step 1 + 2 (inputs / derived shards)
│   ├── train/                    #   video_train.zip       -> video_10000.mp4 ...
│   ├── val/                      #   video_validation.zip  -> video_20000.mp4 ...
│   ├── video_train.zip           #   raw downloads (safe to delete after unzip)
│   ├── video_validation.zip
│   └── clevrer_wds_mp4/
│       ├── train/clevrer-train-000000.tar ...
│       └── val/clevrer-val-000000.tar ...
├── model/                        # Step 3 (weights + metrics), one folder per run
│   └── <RUN>/                    #   e.g. 20250101-120000/
│       ├── checkpoints/          #     step=*.ckpt (last.ckpt, best.ckpt ...)
│       ├── metrics/version_0/metrics.csv
│       └── tb/                   #     tensorboard event files
│   └── latest -> <RUN>           #   symlink to the most recent run
├── logs/                         # Step 3 (process logs), one folder per run
│   └── <RUN>/
│       ├── train.log             #   stdout/stderr of the training process
│       └── train.pid
└── extraction/                   # Step 4 (slot extractions + example renders)
    ├── clevrer_slots_*.pkl       #   extracted slots
    └── examples/video_10000_7slot_overlay.mp4
```

Set these once per shell (and re-run the `mkdir` after opening a fresh terminal):

```
export CLEVRER_OUT="$HOME/Downloads/clevrer_output"
export REPO="$HOME/Downloads/cjepa"

mkdir -p "$CLEVRER_OUT"/{dataset,model,logs,extraction/examples}
```

## Step 1: Dataset

https://clevrer.csail.mit.edu/

Download and unzip **into the output folder**, not into `~/Downloads`:

```
cd "$CLEVRER_OUT/dataset" || exit 1

wget -c https://data.csail.mit.edu/clevrer/videos/train/video_train.zip
wget -c https://data.csail.mit.edu/clevrer/videos/validation/video_validation.zip
unzip -q -n video_train.zip -d train
unzip -q -n video_validation.zip -d val
```

## Step 2: Shard it

Flatten the unzipped videos into `dataset/train` and `dataset/val`, then write the
webdataset shards to `dataset/clevrer_wds_mp4/` (the script always appends
`clevrer_wds_mp4/<split>` to the root it is given):

```
for split in train val; do
  find "$CLEVRER_OUT/dataset/${split}" -type f -name "*.mp4" -exec mv {} "$CLEVRER_OUT/dataset/${split}/" \;
  find "$CLEVRER_OUT/dataset/${split}" -type d ! -path "$CLEVRER_OUT/dataset/${split}" -exec rm -rf {} +
done

conda activate cjepa
cd "$REPO" || exit 1

# Point ROOT_DIR at the output folder
grep "ROOT_DIR" dataset/clevrer/save_clevrer_webdataset_mp4.py
sed -i 's|ROOT_DIR="clevrer_video"|ROOT_DIR="'"$CLEVRER_OUT"'/dataset"|g' \
  dataset/clevrer/save_clevrer_webdataset_mp4.py

# Run the shard conversion script
PYTHONPATH=. python dataset/clevrer/save_clevrer_webdataset_mp4.py
```

Result: `$CLEVRER_OUT/dataset/clevrer_wds_mp4/{train,val}/clevrer-*.tar`.

## Step 3: Train through Videosaur

Checkpoints and metrics land in `$CLEVRER_OUT/model/<RUN>/`; the process log lands in
`$CLEVRER_OUT/logs/<RUN>/`.

```
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cudnn/lib:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

DATA="$CLEVRER_OUT/dataset/clevrer_wds_mp4"
RUN="$(date +%Y%m%d-%H%M%S)"

LOGS="$CLEVRER_OUT/logs/$RUN"          # process log + pid
MODEL="$CLEVRER_OUT/model/$RUN"        # checkpoints + metrics.csv + tb
LOG="$LOGS/train.log"
PIDFILE="$LOGS/train.pid"

mkdir -p "$LOGS" "$MODEL"
ln -sfn "$RUN" "$CLEVRER_OUT/model/latest"
ln -sfn "$RUN" "$CLEVRER_OUT/logs/latest"

train_max=$(printf "%06d" $(($(find "$DATA/train" -maxdepth 1 -name 'clevrer-train-*.tar' | wc -l) - 1)))
val_max=$(printf "%06d" $(($(find "$DATA/val" -maxdepth 1 -name 'clevrer-val-*.tar' | wc -l) - 1)))

cd "$REPO"

tmux new -d -s "train" \
"env PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 NCCL_NVLS_ENABLE=0 python src/third_party/videosaur/videosaur/train.py \
  --log-dir '$MODEL' --no-sub-logdirs \
  src/third_party/videosaur/configs/videosaur/clevrer_dinov2_hf.yml \
  'dataset.train_shards=$DATA/train/clevrer-train-{000000..$train_max}.tar' \
  'dataset.val_shards=$DATA/val/clevrer-val-{000000..$val_max}.tar' \
  'globals.BATCH_SIZE_PER_GPU=128' \
  'dataset.num_workers=16'; exec bash" \; \
  pipe-pane -o "cat >> $LOG" \; \
  attach

tmux a -t train
```

`--no-sub-logdirs` keeps the run flat, so you get exactly:

```
$CLEVRER_OUT/model/<RUN>/checkpoints/step=43000.ckpt
$CLEVRER_OUT/model/<RUN>/metrics/version_0/metrics.csv
$CLEVRER_OUT/model/<RUN>/tb/
```

For a non-tmux background run, write the pid into the same folder:

```
nohup env PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 NCCL_NVLS_ENABLE=0 \
  python src/third_party/videosaur/videosaur/train.py \
  --log-dir "$MODEL" --no-sub-logdirs \
  src/third_party/videosaur/configs/videosaur/clevrer_dinov2_hf.yml \
  "dataset.train_shards=$DATA/train/clevrer-train-{000000..$train_max}.tar" \
  "dataset.val_shards=$DATA/val/clevrer-val-{000000..$val_max}.tar" \
  "globals.BATCH_SIZE_PER_GPU=128" "dataset.num_workers=16" \
  > "$LOG" 2>&1 &
echo $! > "$PIDFILE"
tail -f "$LOG"
```

Pre-trained checkpoint (put it in `model/` too):

```
wget -c -O "$CLEVRER_OUT/model/step=43000.ckpt" \
  "https://huggingface.co/jimchen2/clevrer_dinov2_checkpoint/resolve/main/checkpoints/step%3D43000.ckpt"
```

https://huggingface.co/jimchen2/clevrer_dinov2_checkpoint/blob/main/checkpoints/step%3D43000.ckpt

## Step 4: Extract

Generate an example video (written to `extraction/examples/`):

```
cd "$REPO"

mkdir -p "$CLEVRER_OUT/extraction/examples"

PYTHONPATH=. python mystuff/extraction_reports/clevrer_plot.py \
  --video "$CLEVRER_OUT/dataset/train/video_10000.mp4" \
  --checkpoint "$CLEVRER_OUT/model/step=43000.ckpt" \
  --config src/third_party/videosaur/configs/videosaur/clevrer_dinov2_hf.yml \
  --output "$CLEVRER_OUT/extraction/examples/video_10000_7slot_overlay.mp4" \
  --side_by_side \
  --device cuda
```

Slot extraction (`.pkl` written to `extraction/`; the script appends the checkpoint name
to the filename):

```
export WEIGHT="$CLEVRER_OUT/model/step=43000.ckpt"
export SAVE_DIR="$CLEVRER_OUT/extraction/clevrer_slots.pkl"

mkdir -p "$(dirname "$SAVE_DIR")"

PYTHONPATH=. python src/third_party/slotformer/base_slots/extract_videosaur.py \
  --weight "$WEIGHT" \
  --data_root="$HOME/.stable_worldmodel" \
  --save_path="$SAVE_DIR" \
  --dataset="clevrer" \
  --videosaur_config="src/third_party/videosaur/configs/videosaur/clevrer_dinov2_hf.yml"
```

Final tree:

```
$CLEVRER_OUT/
├── dataset/{train,val,clevrer_wds_mp4/{train,val}}
├── model/{latest -> <RUN>, step=43000.ckpt, <RUN>/checkpoints, <RUN>/metrics, <RUN>/tb}
├── logs/{latest -> <RUN>, <RUN>/train.log, <RUN>/train.pid}
└── extraction/{clevrer_slots_*.pkl, examples/video_10000_7slot_overlay.mp4}
```

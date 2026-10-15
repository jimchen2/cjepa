## Step 1: Dataset

https://clevrer.csail.mit.edu/

```
wget -c https://data.csail.mit.edu/clevrer/videos/train/video_train.zip
wget -c https://data.csail.mit.edu/clevrer/videos/validation/video_validation.zip
unzip -q -n video_train.zip -d train
unzip -q -n video_validation.zip -d val
```

## Step 2: Shard it

```
for split in train val; do
  find "$HOME/Downloads/${split}" -type f -name "*.mp4" -exec mv {} "$HOME/Downloads/${split}/" \;
  find "$HOME/Downloads/${split}" -type d ! -path "$HOME/Downloads/${split}" -exec rm -rf {} +
done

conda activate cjepa
cd "$HOME/Downloads/cjepa" || exit 1

# Check how ROOT_DIR is currently defined in the script
grep "ROOT_DIR" dataset/clevrer/save_clevrer_webdataset_mp4.py

# Automatically replace ROOT_DIR with your ~/Downloads path
sed -i 's|ROOT_DIR="clevrer_video"|ROOT_DIR="'"$HOME/Downloads"'"|g' dataset/clevrer/save_clevrer_webdataset_mp4.py

# Run the shard conversion script
PYTHONPATH=. python dataset/clevrer/save_clevrer_webdataset_mp4.py

```

## Step 3: Train through Videosaur

```
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cudnn/lib:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

DATA="$HOME/Downloads/clevrer_wds_mp4"
RUN="$(date +%Y%m%d-%H%M%S)"
LOGS="$HOME/Downloads/clevrer-videosaur-logs-$RUN"
LOG="$HOME/Downloads/clevrer-training-$RUN.log"
PIDFILE="$HOME/Downloads/clevrer-training-$RUN.pid"

mkdir -p "$LOGS"

train_max=$(printf "%06d" $(($(find "$DATA/train" -maxdepth 1 -name 'clevrer-train-*.tar' | wc -l) - 1)))
val_max=$(printf "%06d" $(($(find "$DATA/val" -maxdepth 1 -name 'clevrer-val-*.tar' | wc -l) - 1)))

cd ~/Downloads/cjepa

tmux new -d -s "train" \
"env PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 NCCL_NVLS_ENABLE=0 python src/third_party/videosaur/videosaur/train.py \
  --log-dir '$LOGS' \
  src/third_party/videosaur/configs/videosaur/clevrer_dinov2_hf.yml \
  'dataset.train_shards=$DATA/train/clevrer-train-{000000..$train_max}.tar' \
  'dataset.val_shards=$DATA/val/clevrer-val-{000000..$val_max}.tar' \
  'globals.BATCH_SIZE_PER_GPU=128' \
  'dataset.num_workers=16'; exec bash" \; \
  pipe-pane -o "cat >> $LOG" \; \
  attach

tmux a -t train
```

Checkpoint: https://huggingface.co/HazelNam/CJEPA/blob/main/clevrer_videosaur_model.ckpt

## Step 4: Extract

Generate an example video

```
PYTHONPATH=. python mystuff/clevrer_plot.py \
  --video ~/Downloads/video_10000.mp4 \
  --checkpoint ~/Downloads/clevrer_videosaur_model.ckpt \
  --config ~/Downloads/cjepa/src/third_party/videosaur/configs/videosaur/clevrer_dinov2_hf.yml \
  --output ~/Downloads/video_10000_7slot_overlay.mp4 \
  --side_by_side \
  --device cuda
```


```
PYTHONPATH=. python src/third_party/slotformer/base_slots/extract_videosaur.py --weight $WEIGHT --data_root="~/.stable_worldmodel"   --save_path=$SAVE_DIR --dataset="clevrer"  --videosaur_config="src/third_party/videosaur/configs/videosaur/clevrer_dinov2_hf.yml" 
```
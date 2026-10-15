## Step 1: Dataset

https://clevrer.csail.mit.edu/

## Step 2: Run the Training and Extraction through VideoSaur

```
PYTHONPATH=. python src/third_party/videosaur/videosaur/train.py \
    src/third_party/videosaur/configs/videosaur/clevrer_dinov2_hf.yml \
    dataset.train_shards="/to/data/path/clevrer-train-{000000..0000XX}.tar" \
    dataset.val_shards="/to/data/path/clevrer-val-{000000..0000XX}.tar" \
    globals.BATCH_SIZE_PER_GPU=32 \
```

Checkpoint: https://huggingface.co/HazelNam/CJEPA/blob/main/clevrer_videosaur_model.ckpt

## Step 3: Extract

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
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

## Step 3: Extract

Generate an example video


```
PYTHONPATH=. python src/third_party/slotformer/base_slots/extract_videosaur.py --weight $WEIGHT --data_root="~/.stable_worldmodel"   --save_path=$SAVE_DIR --dataset="clevrer"  --videosaur_config="src/third_party/videosaur/configs/videosaur/clevrer_dinov2_hf.yml" 
```
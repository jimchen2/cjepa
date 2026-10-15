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
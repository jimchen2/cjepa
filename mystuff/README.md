## Step 1: Generate the Pendulum Dataset

(note the frame and video length, maybe generate questions as well)

https://huggingface.co/datasets/jimchen2/pendulum-cjepa-160f/tree/main

## Step 2: Run the Extraction through VideoSaur

(You need to train the VideoSaur first)

Run pendulum and CLEVRER

https://github.com/jimchen2/cjepa/blob/main/mystuff/extract_four.py

```
OPENBLAS_NUM_THREADS=1 python -u pendulum_run/extract_four.py 
```

It uses

Config:
cjepa/src/third_party/videosaur/configs/videosaur/clevrer_dinov2_hf.yml
—with NUM_SLOTS overridden to 4

Weights:
.cache/clevrer_videosaur_model.ckpt

Input videos:
pendulum_run/cjepa_data_root/pendulum_train/videos/*.mp4
pendulum_run/cjepa_data_root/pendulum_val/videos/*.mp4

Output:
pendulum_run/four_slots/pendulum_videosaur_4slots.pkl

## Step 3: Run the Training

Pull the thing at https://huggingface.co/datasets/jimchen2/pendulum-cjepa-dataset

Causal Jepa is at https://github.com/jimchen2/cjepa

```
export WANDB_MODE=offline
export PYTHONPATH=$(pwd)
export SLOTPATH="/home/jichen/Downloads/pendulum-cjepa-dataset/videosaur/pendulum_videosaur_4slots.pkl"

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

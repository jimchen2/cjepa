```
OPENBLAS_NUM_THREADS=1 python -u pendulum_run/extract_four.py \
  > pendulum_run/four_slots/extraction.log 2>&1
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

Then

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

```
OPENBLAS_NUM_THREADS=1 python -u pendulum_run/extract_four.py \
  > pendulum_run/four_slots/extraction.log 2>&1
```

It uses

```
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
```
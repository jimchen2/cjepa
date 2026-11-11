wandb leet /home/jichen/Downloads/cjepa/wandb/offline-run-20261111_113727-w8a3kgko

Step A succeeded—your C-JEPA rollout slots (160 frames for all 2400 train and 600 val videos) are now saved at:
/home/jichen/Downloads/pendulum_output/extraction/rollout_pendulum_videosaur_4slots_100k_obs128_lr0.0005_mask1.pkl

(cjepa) jichen@uark.edu@eecs-s-sail2025:~/Downloads/cjepa$ python - <<'PY'
import json, os, shutil
from pathlib import Path

home = Path.home()
raw_dir = home / "Downloads/pendulum_output/pendulum-cjepa-3k-new"
aloe_data_root = home / "Downloads/pendulum_output/aloe_data"
(aloe_data_root / "questions").mkdir(parents=True, exist_ok=True)

q_words = set()
a_answers = set()
max_q_len = 0
max_c_len = 0

def tok(s):
    return [w for w in s.lower().replace('?', '').split(' ') if w]

converted_splits = {}
video_fn2path = {"train": {}, "val": {}, "test": {}}

for split in ["train", "val"]:
    src_file = raw_dir / "questions" / f"questions_{split}.json"
    with open(src_file) as f:
        raw = json.load(f)
PYint(f"Wrote ALOE config to: {cfg_path}")se)ple_frames + max_question_len + max_choice_len,ots_100k
train: converted 2400 scenes
val: converted 600 scenes
Vocab saved: 402 question tokens, 23 descriptive answer classes
Max question len: 26, Max choice len: 17
Wrote ALOE config to: /home/jichen/Downloads/cjepa/src/third_party/slotformer/clevrer_vqa/configs/aloe_pendulum_params_rollout.py
(cjepa) jichen@uark.edu@eecs-s-sail2025:~/Downloads/cjepa$ 




conda activate cjepa
cd "$HOME/Downloads/cjepa"

cp src/custom_codes/misc.py src/third_party/nerv/nerv/utils/misc.py
mkdir -p "$HOME/Downloads/pendulum_output/aloe_runs" "$HOME/Downloads/pendulum_output/logs"

tmux new -s aloe-pendulum "
export LD_LIBRARY_PATH=\"$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cudnn/lib:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH\"
export WANDB_MODE=offline
export PYTHONPATH=\"$(pwd):$(pwd)/src/third_party/slotformer\"

CUDA_VISIBLE_DEVICES=0 python src/aloe_train.py \
  --task clevrer_vqa \
  --params src/third_party/slotformer/clevrer_vqa/configs/aloe_pendulum_params_rollout.py \
  --exp_name aloe_pendulum_cjepa100k_ep4 \
  --out_dir \"$HOME/Downloads/pendulum_output/aloe_runs\" \
  --fp16 --cudnn 2>&1 | tee \"$HOME/Downloads/pendulum_output/logs/aloe_pendulum.log\"; exec bash"



conda activate cjepa
cd "$HOME/Downloads/cjepa"

# Change max_n_objects = 4 -> max_n_objects = 3 (since 4 total slots = 3 + 1)
sed -i 's/max_n_objects = 4/max_n_objects = 3/' src/third_party/slotformer/clevrer_vqa/configs/aloe_pendulum_params_rollout.py

# Kill old tmux session if still lingering, then launch clean
tmux kill-session -t aloe-pendulum 2>/dev/null || true

tmux new -s aloe-pendulum "
export LD_LIBRARY_PATH=\"$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cudnn/lib:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH\"
export WANDB_MODE=offline
export PYTHONPATH=\"$(pwd):$(pwd)/src/third_party/slotformer\"

CUDA_VISIBLE_DEVICES=0 python src/aloe_train.py \
  --task clevrer_vqa \
  --params src/third_party/slotformer/clevrer_vqa/configs/aloe_pendulum_params_rollout.py \
  --exp_name aloe_pendulum_cjepa100k_ep4 \
  --out_dir \"$HOME/Downloads/pendulum_output/aloe_runs\" \
  --fp16 --cudnn 2>&1 | tee \"$HOME/Downloads/pendulum_output/logs/aloe_pendulum.log\"; exec bash"



Descriptive accuracy: 76.25%
Explanatory accuracy: 98.00%
Predictive accuracy (uses C-JEPA's rolled-out future frames 128–159): 82.50%
Counterfactual accuracy: 97.08%
Overall Multiple-Choice accuracy: 92.53%

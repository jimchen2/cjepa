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

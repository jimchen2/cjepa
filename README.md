# C-JEPA
## Causal-JEPA: Learning World Models through Object-Level Latent Interventions
by [Heejeong Nam](https://hazel-heejeong-nam.github.io/), [Quentin Le Lidec\*](https://quentinll.github.io/),[Lucas Maes\*](https://lucasmaes.bearblog.dev/), [Yann LeCun](http://yann.lecun.com/), [Randall Balestriero](https://randallbalestriero.github.io/).

* Paper: [https://arxiv.org/abs/2602.11389](https://arxiv.org/abs/2602.11389)
* Project Page: [https://hazel-heejeong-nam.github.io/cjepa/](https://hazel-heejeong-nam.github.io/cjepa/)

## Note 🚨🚨 (Jun 30, 2026)
We will update the repo to make it more user-friendly with the latest stable-worldmodel and stable-pretraining shortly! 

![architecture](static/architecture.png)

## Summary
World models require robust relational understanding to support prediction, reasoning, and control. While object-centric representations provide a useful abstraction, they are not sufficient to capture interaction-dependent dynamics. We therefore propose C-JEPA, a simple and flexible object-centric world model that extends masked joint embedding prediction from image patches to object-centric representations. 
By applying object-level masking that requires an object's state to be inferred from other objects, C-JEPA induces latent interventions with counterfactual-like effects and prevents shortcut solutions, making interaction reasoning essential.
Empirically, C-JEPA leads to consistent gains in visual question answering, with **an absolute improvement of about 20\% in counterfactual reasoning** compared to the same architecture without object-level masking. On agent control tasks, C-JEPA enables substantially more efficient planning by **using only 1\% of the total latent input features required by patch-based world models**, while achieving comparable performance. Finally, we provide a formal analysis demonstrating that object-level masking induces a causal inductive bias via latent interventions.


## Setup
* Please refer to [ENV.md](docs/ENV.md) for environment setup.
* C-JEPA is mainly built on top of [Stable-WorldModel](https://galilai-group.github.io/stable-worldmodel/) and [Stable-Pretraining](https://galilai-group.github.io/stable-pretraining/).
* We adopt [original repo here](https://github.com/martius-lab/videosaur) for training VideoSAUR.
* We adopt [original repo here](https://github.com/pairlab/SlotFormer) for SAVi and VQA model for CLEVRER (a.k.a. ALOE).

## Dataset Preparation
Please refer to [DATASET.md](docs/DATASET.md) for dataset preparation.

## Object-Centric Model / Representations for C-JEPA
C-JEPA relies on object-centric encoders to extract object-centric representations. You can train the encoder by yourself, or download the model checkpoints from HuggingFace, or download the pre-extracted slot representations.

![visualization](static/encoder_vis.png)

### 1. Train Object-Centric Encoder
* Train VideoSAUR on CLEVRER
  ```
  PYTHONPATH=. python src/third_party/videosaur/videosaur/train.py \
      src/third_party/videosaur/configs/videosaur/clevrer_dinov2_hf.yml \
      dataset.train_shards="/to/data/path/clevrer-train-{000000..0000XX}.tar" \
      dataset.val_shards="/to/data/path/clevrer-val-{000000..0000XX}.tar" \
      globals.BATCH_SIZE_PER_GPU=32 \
  ```

* Train SAVi on CLEVRER

  * Refer to slotformer repo to setup data for train savi. Then you have to modify `data_root` and `gpus` in `src/third_party/slotformer/base_slots/configs/stosavi_clevrer_params.py` to point to the data directory. After that, you can run the code below to train SAVi on CLEVRER.

  * Before running the code, replace whole `src/third_party/nerv/nerv/utils/misc.py` with `src/custom_codes/misc.py`. This is because the original code is based on `pytorch-lightning==0.8.*` while we are using `pytorch-lightning==2.6.*`.

  ```
  PYTHONPATH=. torchrun --nproc_per_node=3 src/aloe_train.py \
    --task base_slots \
    --params src/third_party/slotformer/base_slots/configs/stosavi_clevrer_params.py \
    --exp_name clevrer_savi_reproduce \
    --out_dir $OUTDIR \
    --fp16 --ddp --cudnn
  ```

* Train VideoSAUR on PushT
  ```
  PYTHONPATH=. python src/third_party/videosaur/videosaur/train.py \
      src/third_party/videosaur/configs/videosaur/pusht_dinov2_hf.yml \
      dataset.train_shards="pusht_mixed/train/pusht-train-{000000..0000xx}.tar" \
      dataset.val_shards="pusht_mixed/validation/pusht-val-{000000..00000x}.tar"
  ```

### 2. Model Checkpoints
| Dataset      | Encoder    |  Hyperparams                      | Checkpoint Link                                                                                             |
|--------------|------------------|---------------------------|------------------------------------------------------------------------------------------------------------|
| CLEVRER       | VideoSAUR   | clevrer_dinov2_hf.yml        | [Checkpoint](https://huggingface.co/HazelNam/CJEPA/blob/main/clevrer_videosaur_model.ckpt) |
| CLEVRER       | SAVi   | stosavi_clevrer_params.py        | [Checkpoint](https://huggingface.co/HazelNam/CJEPA/blob/main/clevrer_savi_model.pth) |
| Push-T       | VideoSAUR   | pusht_dinov2_hf.yml        | [Checkpoint](https://huggingface.co/HazelNam/CJEPA/blob/main/pusht_videosaur_model.ckpt) |

### 3. Pre-extracted Slot Representations
| Dataset      | Encoder    |  Config                      | Checkpoint Link                                                                                             |
|--------------|------------------|---------------------------|------------------------------------------------------------------------------------------------------------|
| CLEVRER       | VideoSAUR   | clevrer_dinov2_hf.yml        | [Checkpoint](https://huggingface.co/HazelNam/CJEPA/blob/main/clevrer_videosaur_slots.pkl) |
| CLEVRER       | SAVi   | stosavi_clevrer_params.py       | [Checkpoint](https://huggingface.co/HazelNam/CJEPA/blob/main/clevrer_savi_slots.pkl) |
| PUSHT       | VideoSAUR   | pusht_dinov2_hf.yml        | [Checkpoint](https://huggingface.co/HazelNam/CJEPA/blob/main/pusht_videosaur_slots.pkl) |


## Train / Download C-JEPA

### Extract slot representations with the checkpoints

Use this only if you want to extract slots by yourself. If you are using pre-extracted slots, you can skip this step and directly download the pre-extracted slots from HuggingFace.

* CLEVRER SAVi slots 
  *  Refer to slotformer repo to setup data for train savi. Then you have to modify `data_root` in `src/third_party/slotformer/base_slots/configs/stosavi_clevrer_params.py` to point to the data directory. After that, you can run the code below to train SAVi on CLEVRER.
  ```
  PYTHONPATH=. python src/third_party/slotformer/base_slots/extract_slots.py --params src/third_party/slotformer/base_slots/configs/stosavi_clevrer_params.py  --weight $WEIGHT  --save_path clevrer_savi_slots.pkl
  ```
* CLEVRER VideoSAUR 
  ```
  PYTHONPATH=. python src/third_party/slotformer/base_slots/extract_videosaur.py --weight $WEIGHT --data_root="~/.stable_worldmodel"   --save_path=$SAVE_DIR --dataset="clevrer"  --videosaur_config="src/third_party/videosaur/configs/videosaur/clevrer_dinov2_hf.yml" 
  ```
* PushT VideoSAUR Slots
  ```
  PYTHONPATH=. python src/third_party/slotformer/base_slots/extract_videosaur.py --weight $WEIGHT --data_root="~/.stable_worldmodel"   --save_path=$SAVE_DIR  --dataset="pusht_expert"  --videosaur_config="src/third_party/videosaur/configs/videosaur/pusht_dinov2_hf.yml"   --params="src/third_party/slotformer/aloe_pusht_params.py"
  ```

* Extracted pkl will look like:

  ```
  {
      'train': {'0_pixels.mp4': slots, '1_pixels.mp4': slots, ...},  # slots: [T, N, 128] each
      'val': {...},
      'test': {...}
  }
  ```

### 2. Run C-JEPA with pre-extracted slot representations

Use scripts below. You can change the config file and the checkpoint path to run with different settings. For example, you can change `num_masked_slots` to control how many slots are masked.

```sh
sh script/clevrer/train_cjepa_from_slot.sh
sh script/pusht/train_cjepa_from_slot.sh 
```

### 3. Download C-JEPA checkpoints
* We release checkpoints for C-JEPA trained with different object-centric backbones and different number of masked slots. 
* (*) means it performs best. (See the paper for details.)

| Dataset      |   OC-Backbone    |    Masked Slots / Total Slots    | Checkpoint Link                                                 |
|--------------|------------------|---------------------------|------------------------------------------------------------------|
| CLEVRER       | VideoSAUR   | 0/7       | [checkpoint](https://huggingface.co/HazelNam/CJEPA/blob/main/cjepa-ckpts/clevrer_videosaur_0_epoch_30_object.ckpt)                             |
| CLEVRER       | VideoSAUR   | 1/7       | [checkpoint](https://huggingface.co/HazelNam/CJEPA/blob/main/cjepa-ckpts/clevrer_videosaur_1_epoch_30_object.ckpt)                             |
| CLEVRER       | VideoSAUR   | 2/7      | [checkpoint](https://huggingface.co/HazelNam/CJEPA/blob/main/cjepa-ckpts/clevrer_videosaur_2_epoch_30_object.ckpt)                             |
| CLEVRER       | VideoSAUR   | 3/7      | [checkpoint](https://huggingface.co/HazelNam/CJEPA/blob/main/cjepa-ckpts/clevrer_videosaur_3_epoch_30_object.ckpt)                             |
| CLEVRER       | VideoSAUR   | 4/7 (*)     | [checkpoint](https://huggingface.co/HazelNam/CJEPA/blob/main/cjepa-ckpts/clevrer_videosaur_4_epoch_30_object.ckpt)                             |
| CLEVRER       | SAVi        | 0/7      | [checkpoint](https://huggingface.co/HazelNam/CJEPA/blob/main/cjepa-ckpts/clevrer_savi_0_epoch_30_object.ckpt)                             |
| CLEVRER       | SAVi        | 1/7      | [checkpoint](https://huggingface.co/HazelNam/CJEPA/blob/main/cjepa-ckpts/clevrer_savi_1_epoch_30_object.ckpt)                             |
| CLEVRER       | SAVi        | 2/7 (*)     | [checkpoint](https://huggingface.co/HazelNam/CJEPA/blob/main/cjepa-ckpts/clevrer_savi_2_epoch_30_object.ckpt)                             |
| CLEVRER       | SAVi        | 3/7      | [checkpoint](https://huggingface.co/HazelNam/CJEPA/blob/main/cjepa-ckpts/clevrer_savi_3_epoch_30_object.ckpt)                             |
| CLEVRER       | SAVi        | 4/7      | [checkpoint](https://huggingface.co/HazelNam/CJEPA/blob/main/cjepa-ckpts/clevrer_savi_4_epoch_30_object.ckpt)                             |
| PUSHT       | VideoSAUR   | 0/4      | [checkpoint](https://huggingface.co/HazelNam/CJEPA/blob/main/cjepa-ckpts/pusht_videosaur_0_epoch_30_object.ckpt)                             |
| PUSHT       | VideoSAUR   | 1/4  (*)    | [checkpoint](https://huggingface.co/HazelNam/CJEPA/blob/main/cjepa-ckpts/pusht_videosaur_1_epoch_30_object.ckpt)                             |
| PUSHT       | VideoSAUR   | 2/4      | [checkpoint](https://huggingface.co/HazelNam/CJEPA/blob/main/cjepa-ckpts/pusht_videosaur_2_epoch_30_object.ckpt)                             |   

## Evaluation
### Evaluate Control on Push-T
  ```
  sh scripts/pusht/test_planning.sh
  ```

### Evaluate Visual Reasoning on CLEVRER

  * We will first rollout slots (from 128 frame to 160 frame) with C-JEPA checkpoint and pre-extracted slots.

  ```
  # change CKPTPATH and SLOTPATH, `predictor_lr`, `num_masked_slots` before. SLOTPATH should be matched with the checkpoint that you are using (dataset, oc-encoder) and it should point to the pre-extracted slot representations. For example, if you are using `clevrer_savi_2.ckpt`, you should use `clevrer_savi_slots.pkl` and set `num_masked_slots` to 2.
  # `predictor_lr` (default=5e-4), `num_masked_slots` should be matched with config is CKPT that you are using (they go to the output name)
  # output name will be like : rollout_{SLOTPATH name}_{CKPT lr, CKPT masked slot}
  sh scripts/clevrer/rollout_from_slot.sh
  ```
  * This will save output file under the same directory as the pre-extracted slot representations.
  ```
  # rollout_clevrer_slots_{configuration}.pkl :
  {
      'train': {'0_pixels.mp4': slots, '1_pixels.mp4': slots, ...},  # slots: [160, N, 128] each
      'val': {...},
      'test': {...}
  }
  ```
  * Then you can run the code below to evaluate C-JEPA on CLEVRER visual question answering.
  * Again, before running the code, replace whole `src/third_party/nerv/nerv/utils/misc.py` with `src/custom_codes/misc.py`. This is because the thrid-party code is based on `pytorch-lightning==0.8.*` while we are using `pytorch-lightning==2.6.*`.
  * You should change params manually in `src/third_party/slotformer/clevrer_vqa/configs/aloe_clevrer_params-rollout.py`. For example, 
    * `gpu` (it should exactly match the number of the visible devices)
    * `lr`
  * Then run script below to train and test aloe.

  ```
  sh scripts/clevrer/train_aloe.sh
  sh scripts/clevrer/test_aloe.sh
  ```

# Dataset
* If you are using pre-extracted slots for training C-JEPA, you can skip everything here.
* If you are using VideoSAUR for object centric encoder (either by training yourself or downloading the checkpoint), you need to follow the instruction here to prepare the dataset.
* If you are using SAVi for object centric encoder (either by training yourself or downloading the checkpoint), please follow the [instruction](https://github.com/pairlab/SlotFormer/blob/master/docs/data.md) in slotformer repo to setup data. Although, we only use SAVi for CLEVRER dataset, you can also use SAVi for PushT by following the similar data preparation instruction.
* If you are testing downstream (VQA or planning), you need to prepare the dataset for evaluation. 


## CLEVRER
### 1. Download original data (~24G total)
```sh
#!/usr/bin/env bash

ROOT_DIR="./clevrer_video"

mkdir -p \
  ${ROOT_DIR}/train \
  ${ROOT_DIR}/val \
  ${ROOT_DIR}/test

echo "Downloading CLEVRER videos..."

wget -nc -P ${ROOT_DIR}/train \
  http://data.csail.mit.edu/clevrer/videos/train/video_train.zip

wget -nc -P ${ROOT_DIR}/val \
  http://data.csail.mit.edu/clevrer/videos/validation/video_validation.zip

wget -nc -P ${ROOT_DIR}/test \
  http://data.csail.mit.edu/clevrer/videos/test/video_test.zip

echo "Unzipping..."
unzip -q ${ROOT_DIR}/train/video_train.zip -d ${ROOT_DIR}/train
unzip -q ${ROOT_DIR}/val/video_validation.zip -d ${ROOT_DIR}/val
unzip -q ${ROOT_DIR}/test/video_test.zip -d ${ROOT_DIR}/test

echo "Flattening mp4 files..."

for split in train val test; do
  find ${ROOT_DIR}/${split} -type f -name "*.mp4" -exec mv {} ${ROOT_DIR}/${split}/ \;
  find ${ROOT_DIR}/${split} -type d ! -path ${ROOT_DIR}/${split} -exec rm -rf {} +
done

echo "Done."
```

This will give you 
```
ROOT_DIR/
├── train/
│   ├── video_00000.mp4
│   ├── video_00001.mp4
│   └── ...
├── val/
│   ├── video_10000.mp4
│   └── ...
└── test/
    ├── video_15000.mp4
    └── ...
```

### 2. Reformat CLEVRER for Stable-WorldModel
If you are using pre-extracted slots, you can skip this step.
This step is required for extracting slots from object-centric encoders.
```
% set ROOT_DIR in the file first
python dataset/clevrer/clevrer.py
```
* This will create clevrer dataset under stable-wm cache directory (by calling `swm.data.utils.get_cache_dir()`) in a desired format.
* We will use deterministic train / val  setup - your cache directory will look like

```
.stable_worldmodel
├── clevrer_train/
|    ├── data-00000-of-000001.arrow
|    ├── dataset_info.json
|    ├── state.json
|    └── videos
|         └──0_pixels.mp4 ...
├── clevrer_val/
|    ├── data-00000-of-000001.arrow
|    ├── dataset_info.json
|    ├── state.json
|    └── videos
|         └──10000_pixels.mp4 ...
└── clevrer_test/
     ├── data-00000-of-000001.arrow
     ├── dataset_info.json
     ├── state.json
     └── videos
          └──15000_pixels.mp4 ...
```

### 3 Prepare CLEVRER Videosaur dataset
```
% You don't need this if you are not running videosaur for CLEVRER.
% set ROOT_DIR in the file first
python dataset/clevrer/save_clevrer_webdataset_mp4.py
```
This will give you 
```
ROOT_DIR/
├── train/
├── val/
├── test/
└── clevrre_wds_mp4
    ├── train
    |   └── clevrer-train-000000.tar ...
    └── val
        └── clevrer-val-000000.tar ...

```

## Push-T


### 1. Download PushT for Stable-WorldModel
* Download `pusht_expert_{train/val}` data from [link](https://drive.google.com/drive/folders/19ST8nfhZ4rMxrTfh2kgBErCcS5Kk1o-D?usp=sharing).
* Unzip and put them under `swm.data.utils.get_cache_dir()`. Default directory is `~/.stable_worldmodel/`. But you can put them anywhere and set the `cache_dir` argument in the file before running.
* Do not change the folder name. This naming is required when you want to work with your own dataset: {dataset_name}_train and {dataset_name}_val. 


This will give you

```
.stable_worldmodel
├── pusht_expert_train/
|    ├── data-00000-of-000001.arrow
|    ├── dataset_info.json
|    ├── state.json
|    └── videos
|         └──0_pixels.mp4 ...
└── pusht_expert_val/
     ├── data-00000-of-000001.arrow
     ├── dataset_info.json
     ├── state.json
     └── videos
          └──0_pixels.mp4 ...
```

### 2. Prepare PushT Videosaur dataset

* Generate randomly-moving PushT data for better object-centric learning. (10000 for train, 1000 for val)
```
PYTHONPATH=. python dataset/pusht/pusht_all_moving_videogen.py \
    --num_videos 11000 \
    --output_dir my_dataset 
```

* Generate webdataset shards for VideoSAUR training
We will mix the original videos (video_10000.mp4 - video_18684.mp4) with the 10000 randomly moving videos.
You can set the directory paths in the file before running.
```
PYTHONPATH=. python dataset/pusht/save_mixed_pusht_webdataset_mp4.py
```
# Install

We recommend using conda to set up the environment.

### 1. Create and activate conda environment
```
conda create -n cjepa python=3.10 -y
conda activate cjepa
git clone https://github.com/galilai-group/cjepa.git
cd cjepa
```


### 2. Install system dependencies

We use ffmpeg for video processing:
```
conda install anaconda::ffmpeg
```


### 3. Install basic Python dependencies
We recommend using `uv` to install dependencies. You can also use `pip` if you prefer.
```
pip install uv
uv pip install seaborn webdataset swig einops torchcodec av accelerate tensorboard tensorboardX hickle pycocotools wget
```

### 4. Install third-party libraries
* Every third party library should be installed under `src/third_party`. Please follow the instruction to install the library.
* Below will install `stable-pretraining`, `stable-worldmodel` and `nerv`.
* All third-party repositories are installed in editable mode to ensure smooth development.
* Specific commits/tags are pinned for reproducibility.
* The environment is tested with Python 3.10.



```
cd src/third_party
git clone https://github.com/galilai-group/stable-pretraining.git
cd stable-pretraining
git checkout 92b5841
uv pip install -e .

cd ..
git clone https://github.com/galilai-group/stable-worldmodel.git
cd stable-worldmodel
git checkout 221ac82  ## If it doesn't work, please see below
uv pip install -e .


cd ../
git clone https://github.com/Wuziyi616/nerv.git
cd nerv
git checkout v0.1.0   # tested with v0.1.0 release
uv pip install -e .
```

### 5. Stable-Worldmodel Manual Download Instruction

- If git checkout doesn't work, please manually download [https://github.com/galilai-group/stable-worldmodel/tree/221ac820a1adea75bed99df45ab592bb5f42306c](https://github.com/galilai-group/stable-worldmodel/tree/221ac820a1adea75bed99df45ab592bb5f42306c) and locate under third_party folder.

- If the link above also doesn't work, please download the zip file from : [https://drive.google.com/drive/folders/1Oll1ghHa8ySsGjPPTJE6o8XSG0Pvyks0?usp=sharing](https://drive.google.com/drive/folders/1Oll1ghHa8ySsGjPPTJE6o8XSG0Pvyks0?usp=sharing)

Sorry for the inconvenience.



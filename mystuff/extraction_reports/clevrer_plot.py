#!/usr/bin/env python3
"""
Render a CLEVRER MP4 with VideoSAUR slot-attention regions overlaid.
Works with the released CLEVRER VideoSAUR checkpoint
(https://huggingface.co/HazelNam/CJEPA/blob/main/clevrer_videosaur_model.ckpt)
and `clevrer_dinov2_hf.yml` (7 slots, DINOv2-S/14, 196x196 input -> 14x14 patches).
Preprocessing matches `extract_videosaur.py`:
  uint8 -> float [0,1] -> Resize((196,196)) -> ImageNet normalize.
Output:
  The original CLEVRER video (480x320) with a colored hard slot assignment at
  every DINO patch location. Colors are slot indices, not object identities.
  With --side_by_side, the original frame is shown on the left.
"""

import argparse
import math
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from src.third_party.videosaur.videosaur import configuration, models


parser = argparse.ArgumentParser()
parser.add_argument("--video", required=True, help="Input CLEVRER .mp4 (e.g. video_10000.mp4)")
parser.add_argument("--checkpoint", required=True, help="clevrer_videosaur_model.ckpt")
parser.add_argument("--config", required=True, help="clevrer_dinov2_hf.yml")
parser.add_argument("--output", required=True, help="Output .mp4")
parser.add_argument("--device", default="cuda", help="cuda or cpu")
parser.add_argument("--batch", type=int, default=16, help="DINO frame microbatch")
parser.add_argument("--opacity", type=float, default=0.55, help="Overlay opacity, 0 to 1")
parser.add_argument("--frameskip", type=int, default=1,
                    help="Use every k-th frame (extraction uses 1, VideoSAUR training used 2)")
parser.add_argument("--max_frames", type=int, default=None, help="Optional cap on #frames (after frameskip)")
parser.add_argument("--side_by_side", action="store_true", help="Show original | overlay")
parser.add_argument("--seed", type=int, default=None,
                    help="Slot initializer seed. Default is 42 + video number.")
args = parser.parse_args()

video_path = Path(args.video).expanduser()
checkpoint_path = Path(args.checkpoint).expanduser()
output_path = Path(args.output).expanduser()
output_path.parent.mkdir(parents=True, exist_ok=True)

if not video_path.exists():
    raise FileNotFoundError(video_path)
if not checkpoint_path.exists():
    raise FileNotFoundError(checkpoint_path)

device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
print(f"Using device: {device}")

if args.seed is None:
    try:
        args.seed = 42 + int(video_path.stem.split("_")[-1])
    except ValueError:
        args.seed = 42
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)
print(f"Using slot-initializer seed: {args.seed}")

# ------------------------------------------------------------
# Load the CLEVRER VideoSAUR model (slot count comes from the config).
# ------------------------------------------------------------
print("Loading VideoSAUR config and checkpoint...")
config = configuration.load_config(args.config)
NUM_SLOTS = int(config.globals.NUM_SLOTS)          # 7 for CLEVRER
NUM_PATCHES = int(config.globals.NUM_PATCHES)      # 196
GRID = int(round(math.sqrt(NUM_PATCHES)))          # 14
INPUT_SIZE = GRID * 14                             # 196 for DINOv2 patch-14
print(f"NUM_SLOTS={NUM_SLOTS}, grid={GRID}x{GRID}, input={INPUT_SIZE}x{INPUT_SIZE}")

model = models.build(config.model, config.optimizer)
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
state_dict = checkpoint.get("state_dict", checkpoint)
print(model.load_state_dict(state_dict, strict=True))
model.eval().to(device)

# ------------------------------------------------------------
# Decode the original video.
# ------------------------------------------------------------
cap = cv2.VideoCapture(str(video_path))
if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {video_path}")
fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

frames_bgr = []
idx = 0
while True:
    ok, frame_bgr = cap.read()
    if not ok:
        break
    if idx % args.frameskip == 0:
        frames_bgr.append(frame_bgr)
    idx += 1
cap.release()
if args.max_frames:
    frames_bgr = frames_bgr[: args.max_frames]
if not frames_bgr:
    raise RuntimeError("No frames were decoded from the input video.")

fps = fps / args.frameskip
frames_rgb = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_bgr])  # [T,H,W,3]
T, H, W, _ = frames_rgb.shape
print(f"Decoded {T} frames (frameskip={args.frameskip}) at {fps:.2f} FPS, size {W}x{H}")

# ------------------------------------------------------------
# DINOv2 encoder + VideoSAUR processor -> [T, NUM_SLOTS, GRID, GRID] masks.
# ------------------------------------------------------------
mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
video_tensor = torch.from_numpy(frames_rgb).permute(0, 3, 1, 2)  # [T,C,H,W]

encoded_chunks = []
print("Extracting DINO features...")
with torch.inference_mode():
    for chunk in video_tensor.split(args.batch):
        chunk = chunk.to(device).float() / 255.0
        chunk = F.interpolate(chunk, size=(INPUT_SIZE, INPUT_SIZE), mode="bilinear",
                              align_corners=False, antialias=True)
        chunk = (chunk - mean) / std
        encoded_chunks.append(model.encoder(chunk.unsqueeze(0))["features"])
    features = torch.cat(encoded_chunks, dim=1)

    print(f"Computing {NUM_SLOTS} slot-attention maps...")
    initial_slots = model.initializer(batch_size=1).to(device)
    result = model.processor(initial_slots, features)
    masks = result["corrector"]["masks"][0].float().cpu().numpy()  # [T, S, P]

if masks.shape[-1] != NUM_PATCHES:
    raise RuntimeError(f"Expected {NUM_PATCHES} patches per frame, got mask shape {masks.shape}")
masks = masks.reshape(T, NUM_SLOTS, GRID, GRID)
print(f"Attention-map shape: {masks.shape}")
assert np.allclose(masks.sum(axis=1), 1.0, atol=1e-3), "Unexpected slot-mask normalization."

# Per-slot share of the image over the whole video (handy to spot the background slot).
coverage = (masks.argmax(axis=1)[..., None] == np.arange(NUM_SLOTS)).mean(axis=(0, 1, 2))
for s, c in enumerate(coverage):
    print(f"  slot {s + 1}: {100 * c:5.1f}% of patches")

# ------------------------------------------------------------
# Render. Palette is BGR (OpenCV).
# ------------------------------------------------------------
PALETTE_BGR = np.array([
    [255, 190, 60],   # sky blue
    [40, 180, 255],   # orange
    [100, 220, 105],  # green
    [180, 70, 255],   # pink / magenta
    [60, 60, 230],    # red
    [230, 230, 80],   # cyan
    [40, 230, 240],   # yellow
    [200, 120, 120],  # slate
    [150, 150, 150],  # gray
    [90, 40, 160],    # maroon
], dtype=np.uint8)
slot_colors = PALETTE_BGR[np.arange(NUM_SLOTS) % len(PALETTE_BGR)]

out_w = W * 2 if args.side_by_side else W
writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, H))
if not writer.isOpened():
    raise RuntimeError("Could not open the MP4 writer. Try: conda install -c conda-forge ffmpeg")

print(f"Rendering overlay video: {output_path}")
legend_h = 12 + 18 * NUM_SLOTS
for t, original_bgr in enumerate(frames_bgr):
    hard = masks[t].argmax(axis=0).astype(np.uint8)
    conf = masks[t].max(axis=0).astype(np.float32)
    hard_full = cv2.resize(hard, (W, H), interpolation=cv2.INTER_NEAREST)
    conf_full = cv2.resize(conf, (W, H), interpolation=cv2.INTER_NEAREST)

    alpha = (args.opacity * (0.35 + 0.65 * conf_full))[..., None]
    frame = (original_bgr.astype(np.float32) * (1 - alpha)
             + slot_colors[hard_full].astype(np.float32) * alpha).astype(np.uint8)

    cv2.rectangle(frame, (5, 5), (110, 5 + legend_h), (12, 20, 32), -1)
    cv2.rectangle(frame, (5, 5), (110, 5 + legend_h), (225, 235, 245), 1)
    for s, color in enumerate(slot_colors):
        y = 22 + 18 * s
        cv2.rectangle(frame, (13, y - 11), (27, y + 3), color.tolist(), -1)
        cv2.putText(frame, f"Slot {s + 1}", (35, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (240, 245, 250), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Frame {t + 1}/{T}", (W - 120, H - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (255, 255, 255), 1, cv2.LINE_AA)

    if args.side_by_side:
        frame = np.concatenate([original_bgr, frame], axis=1)
    writer.write(frame)

writer.release()
print("\nDone.")
print(f"Created: {output_path.resolve()}")
#!/usr/bin/env python3
"""
Render a pendulum MP4 with VideoSAUR's four slot-attention regions overlaid.

Output:
  The original pendulum video with a colored hard slot assignment at every
  DINO patch location, plus a non-overlapping top header bar for the legend.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from src.third_party.videosaur.videosaur import configuration, models


parser = argparse.ArgumentParser()
parser.add_argument("--video", required=True, help="Input pendulum .mp4")
parser.add_argument("--checkpoint", required=True, help="VideoSAUR step=5000.ckpt")
parser.add_argument("--config", required=True, help="pendulum_dinov2_hf.yml")
parser.add_argument("--output", required=True, help="Output .mp4")
parser.add_argument("--device", default="cuda", help="cuda or cpu")
parser.add_argument("--batch", type=int, default=8, help="DINO frame microbatch")
parser.add_argument("--opacity", type=float, default=0.58, help="Overlay opacity, 0 to 1")
parser.add_argument(
    "--seed",
    type=int,
    default=None,
    help="Slot initializer seed. Default is 42 + video number.",
)
args = parser.parse_args()

video_path = Path(args.video)
checkpoint_path = Path(args.checkpoint)
output_path = Path(args.output)
output_path.parent.mkdir(parents=True, exist_ok=True)

if not video_path.exists():
    raise FileNotFoundError(video_path)
if not checkpoint_path.exists():
    raise FileNotFoundError(checkpoint_path)

device = torch.device(args.device if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Match the stable per-video seeding used for the extraction-style script.
if args.seed is None:
    try:
        video_id = int(video_path.stem.split("_")[-1])
        args.seed = 42 + video_id
    except ValueError:
        args.seed = 42

torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

print(f"Using slot-initializer seed: {args.seed}")

# ------------------------------------------------------------
# Load the exact four-slot Pendulum VideoSAUR model.
# ------------------------------------------------------------
print("Loading VideoSAUR config and checkpoint...")
config = configuration.load_config(args.config)

config.globals.NUM_SLOTS = 4
config.globals.SLOT_DIM = 128

model = models.build(config.model, config.optimizer)

checkpoint = torch.load(
    checkpoint_path,
    map_location="cpu",
    weights_only=False,
)
state_dict = checkpoint.get("state_dict", checkpoint)
load_result = model.load_state_dict(state_dict, strict=True)
print(load_result)

model.eval().to(device)

# ------------------------------------------------------------
# Decode the original video.
# ------------------------------------------------------------
cap = cv2.VideoCapture(str(video_path))
if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {video_path}")

fps = cap.get(cv2.CAP_PROP_FPS)
if not fps or fps <= 0:
    fps = 12.0

frames_bgr = []
frames_rgb = []

while True:
    ok, frame_bgr = cap.read()
    if not ok:
        break

    frames_bgr.append(frame_bgr)
    frames_rgb.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

cap.release()

if not frames_bgr:
    raise RuntimeError("No frames were decoded from the input video.")

frames_rgb = np.stack(frames_rgb)  # [T, H, W, 3]
T, original_h, original_w, _ = frames_rgb.shape

print(f"Decoded {T} frames at {fps:.2f} FPS, size {original_w}x{original_h}")

# ------------------------------------------------------------
# Run DINOv2 encoder + VideoSAUR processor.
# ------------------------------------------------------------
mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

video_tensor = torch.from_numpy(frames_rgb).permute(0, 3, 1, 2)  # [T,C,H,W]
encoded_chunks = []

print("Extracting DINO features...")
with torch.inference_mode():
    for chunk in video_tensor.split(args.batch):
        chunk = chunk.to(device).float() / 255.0
        chunk = F.interpolate(
            chunk,
            size=(196, 196),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )

        normalized = (chunk - mean) / std
        encoder_output = model.encoder(normalized.unsqueeze(0))
        encoded_chunks.append(encoder_output["features"])

    features = torch.cat(encoded_chunks, dim=1)

    print("Computing four slot-attention maps...")
    initial_slots = model.initializer(batch_size=1).to(device)
    result = model.processor(initial_slots, features)

    # [1, T, 4 slots, 196 patches] -> [T, 4, 14, 14]
    masks = result["corrector"]["masks"][0].detach().cpu().numpy()

if masks.shape[-1] != 196:
    raise RuntimeError(
        f"Expected 196 DINO patches per frame, got mask shape {masks.shape}"
    )

masks = masks.reshape(T, 4, 14, 14)
print(f"Attention-map shape: {masks.shape}")

assert np.allclose(masks.sum(axis=1), 1.0, atol=1e-4), (
    "Unexpected slot-mask normalization."
)

# ------------------------------------------------------------
# Render top header bar + original RGB + hard slot assignment overlay.
# ------------------------------------------------------------
slot_colors_bgr = np.array(
    [
        [255, 190, 60],   # Slot 1: sky blue
        [40, 180, 255],   # Slot 2: orange
        [100, 220, 105],  # Slot 3: green
        [180, 70, 255],   # Slot 4: pink / magenta
    ],
    dtype=np.uint8,
)

header_h = 46
out_h = original_h + header_h

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(
    str(output_path),
    fourcc,
    fps,
    (original_w, out_h),
)

if not writer.isOpened():
    raise RuntimeError(
        "Could not open the MP4 writer. Try installing ffmpeg: "
        "conda install -c conda-forge ffmpeg"
    )

print(f"Rendering overlay video: {output_path}")

for frame_idx, original_bgr in enumerate(frames_bgr):
    hard_grid = masks[frame_idx].argmax(axis=0).astype(np.uint8)  # [14,14]
    confidence_grid = masks[frame_idx].max(axis=0).astype(np.float32)

    hard_full = cv2.resize(
        hard_grid,
        (original_w, original_h),
        interpolation=cv2.INTER_NEAREST,
    )
    confidence_full = cv2.resize(
        confidence_grid,
        (original_w, original_h),
        interpolation=cv2.INTER_NEAREST,
    )

    color_layer = slot_colors_bgr[hard_full]

    alpha = args.opacity * (0.35 + 0.65 * confidence_full)
    video_frame = (
        original_bgr.astype(np.float32) * (1.0 - alpha[..., None])
        + color_layer.astype(np.float32) * alpha[..., None]
    ).astype(np.uint8)

    # Create non-overlapping top header bar.
    header = np.full((header_h, original_w, 3), (12, 20, 32), dtype=np.uint8)
    cv2.line(header, (0, header_h - 1), (original_w, header_h - 1), (225, 235, 245), 1)

    # Row 1: Horizontal slot legend.
    col_w = max(original_w // 4, 1)
    for slot_idx, color in enumerate(slot_colors_bgr):
        x = 6 + slot_idx * col_w
        cv2.rectangle(header, (x, 6), (x + 12, 18), color.tolist(), -1)
        cv2.putText(
            header,
            f"S{slot_idx + 1}",
            (x + 16, 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (240, 245, 250),
            1,
            cv2.LINE_AA,
        )

    # Row 2: Frame counter.
    cv2.putText(
        header,
        f"Frame {frame_idx + 1}/{T}",
        (6, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    # Stack header on top of the video frame so nothing overlaps the pendulum.
    output_frame = np.vstack([header, video_frame])
    writer.write(output_frame)

writer.release()

print("\nDone.")
print(f"Created: {output_path.resolve()}")

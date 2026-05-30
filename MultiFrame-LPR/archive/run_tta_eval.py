#!/usr/bin/env python3
"""Test-time augmentation evaluation for ResTranOCR.

Runs each test track through the model N times with mild random augmentations,
averages the CTC log-probabilities, and decodes for improved accuracy.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from itertools import groupby
from typing import Dict, List, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.config import Config
from src.models.restran import ResTranOCR
from src.utils.common import seed_everything


# --------------------------------------------------------------------------- #
#  TTA transforms
# --------------------------------------------------------------------------- #
SEQUENCE_TARGETS = {"image1": "image", "image2": "image", "image3": "image", "image4": "image"}

def get_tta_transforms(img_height: int, img_width: int) -> A.Compose:
    """Mild photometric augmentations suitable for TTA on license plates."""
    return A.Compose([
        A.Resize(height=img_height, width=img_width),
        A.RandomBrightnessContrast(brightness_limit=0.12, contrast_limit=0.12, p=0.8),
        A.HueSaturationValue(hue_shift_limit=3, sat_shift_limit=10, val_shift_limit=10, p=0.4),
        A.GaussNoise(std_range=(0.005, 0.015), mean_range=(0.0, 0.0), p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ], additional_targets=SEQUENCE_TARGETS, is_check_shapes=False)

def get_val_transforms(img_height: int, img_width: int) -> A.Compose:
    return A.Compose([
        A.Resize(height=img_height, width=img_width),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ], additional_targets=SEQUENCE_TARGETS, is_check_shapes=False)


# --------------------------------------------------------------------------- #
#  Dataset helpers
# --------------------------------------------------------------------------- #
def find_test_tracks(test_root: str) -> List[str]:
    import glob
    return sorted(p for p in glob.glob(os.path.join(test_root, "**", "track_*"), recursive=True)
                  if os.path.isdir(p))


def read_label(track_path: str) -> str:
    import json
    json_path = os.path.join(track_path, "annotations.json")
    if not os.path.exists(json_path):
        return ""
    try:
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return ""
    if isinstance(data, list):
        data = data[0] if data else {}
    if not isinstance(data, dict):
        return ""
    return str(data.get("plate_text") or data.get("license_plate") or data.get("text") or "").strip().upper()


def load_frames(track_path: str, n_frames: int = 5) -> List[np.ndarray]:
    import glob
    files: List[str] = []
    for ext in ("png", "jpg", "jpeg"):
        files.extend(glob.glob(os.path.join(track_path, f"lr-*.{ext}")))
    files = sorted(files)
    if not files:
        return []
    if len(files) >= n_frames:
        files = files[:n_frames]
    else:
        files = files + [files[-1]] * (n_frames - len(files))

    frames = []
    for p in files:
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            return []
        frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # Pad all frames to max dimensions (albumentations requires uniform sizes)
    max_h = max(f.shape[0] for f in frames)
    max_w = max(f.shape[1] for f in frames)
    padded = []
    for f in frames:
        h, w = f.shape[:2]
        if h != max_h or w != max_w:
            f = cv2.copyMakeBorder(f, 0, max_h - h, 0, max_w - w, cv2.BORDER_REPLICATE)
        padded.append(f)
    return padded


def apply_transform(frames: List[np.ndarray], transform: A.Compose) -> torch.Tensor:
    """Apply a 5-frame transform and return [5,3,H,W] tensor."""
    result = transform(
        image=frames[0],
        image1=frames[1],
        image2=frames[2],
        image3=frames[3],
        image4=frames[4],
    )
    return torch.stack([result["image"], result["image1"], result["image2"],
                        result["image3"], result["image4"]], dim=0)


# --------------------------------------------------------------------------- #
#  CTC decode
# --------------------------------------------------------------------------- #
def decode_greedy(log_probs: torch.Tensor, idx2char: Dict[int, str]) -> str:
    """Greedy CTC decode from [T, C] log-probabilities."""
    probs = log_probs.exp()
    indices = probs.argmax(dim=1).cpu().numpy()
    chars = []
    for char_idx, _ in groupby(indices):
        if char_idx != 0:
            chars.append(idx2char.get(int(char_idx), ""))
    return "".join(chars)


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TTA evaluation for ResTranOCR.")
    p.add_argument("--checkpoint", default="results/restran_best.pth")
    p.add_argument("--test-data-root", default="data/LRLPR-26-5opEvJTW/test")
    p.add_argument("--output-csv", default="results/test_predictions_tta.csv")
    p.add_argument("--n-augments", type=int, default=8,
                   help="Number of augmented views to average per sample")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use-sr", action="store_true", default=False)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    cfg = Config()

    # Build model (no SR for baseline TTA)
    model = ResTranOCR(
        num_classes=cfg.NUM_CLASSES,
        transformer_heads=cfg.TRANSFORMER_HEADS,
        transformer_layers=cfg.TRANSFORMER_LAYERS,
        transformer_ff_dim=cfg.TRANSFORMER_FF_DIM,
        dropout=cfg.TRANSFORMER_DROPOUT,
        use_stn=cfg.USE_STN,
        pretrained=False,
        use_sr=args.use_sr,
        sr_num_blocks=getattr(cfg, "SR_NUM_BLOCKS", 8),
        sr_scale=getattr(cfg, "SR_SCALE", 2),
        sr_nf=getattr(cfg, "SR_NF", 32),
        sr_gc=getattr(cfg, "SR_GC", 16),
    ).to(device)

    state = torch.load(args.checkpoint, map_location=device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"Missing keys: {len(missing)}")
    model.eval()

    val_tf = get_val_transforms(cfg.IMG_HEIGHT, cfg.IMG_WIDTH)
    tta_tf = get_tta_transforms(cfg.IMG_HEIGHT, cfg.IMG_WIDTH)

    tracks = find_test_tracks(args.test_data_root)
    print(f"Found {len(tracks)} test tracks.")

    total_correct = 0
    total_samples = 0
    rows = []

    for track_path in tqdm(tracks, desc="TTA eval"):
        label = read_label(track_path)
        track_id = os.path.relpath(track_path, args.test_data_root).replace(os.sep, "/")

        frames = load_frames(track_path)
        if not frames:
            continue

        with torch.no_grad():
            # Run N augmented passes and accumulate log-probs
            accumulated: torch.Tensor | None = None
            n_valid = 0

            for aug_idx in range(args.n_augments):
                tf = val_tf if aug_idx == 0 else tta_tf
                try:
                    x = apply_transform(frames, tf)  # [5,3,H,W]
                except Exception:
                    continue
                x = x.unsqueeze(0).to(device)  # [1,5,3,H,W]
                log_probs = model(x)  # [1, T, C]
                if accumulated is None:
                    accumulated = log_probs[0]  # [T, C]
                else:
                    # Average in log-probability space
                    accumulated = torch.logaddexp(accumulated, log_probs[0])
                n_valid += 1

            if accumulated is None or n_valid == 0:
                continue

            # Normalize (divide by n_valid in log space = subtract log(n_valid))
            avg_log_probs = accumulated - np.log(n_valid)

        pred = decode_greedy(avg_log_probs, cfg.IDX2CHAR)
        conf = float(avg_log_probs.exp().max(dim=1).values.mean().item())
        correct = int(pred == label) if label else 0

        if label:
            total_correct += correct
            total_samples += 1

        rows.append((track_id, pred, conf, label, correct))

    acc = 100.0 * total_correct / total_samples if total_samples else 0.0
    print(f"\nTTA Results ({args.n_augments} augments): {total_correct}/{total_samples} = {acc:.2f}%")

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["track_id", "prediction", "confidence", "ground_truth", "correct"])
        for row in rows:
            writer.writerow([row[0], row[1], f"{row[2]:.4f}", row[3], row[4]])
    print(f"Saved {len(rows)} predictions to {args.output_csv}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Apply v4 SR as a preprocessor to test images, then evaluate with the baseline OCR.

No retraining needed. Tests whether SR preprocessing (SR→resize to LR) helps the
already-converged baseline OCR model.

Usage:
    python eval_sr_preprocess_baseline.py
"""
import csv
import os
import sys
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.config import Config
from src.data.dataset import MultiFrameDataset
from src.models.restran import ResTranOCR
from src.utils.postprocess import decode_with_confidence

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

V4_CHECKPOINT = "results/restran_sr_v4_best.pth"
BASELINE_CHECKPOINT = "results/restran_best.pth"
OUTPUT_CSV = "results/test_predictions_sr_preprocess_baseline.csv"


def main() -> None:
    config = Config()
    device = torch.device("cpu")

    # --- Load v4 model just to extract the trained SR module ---
    print("Loading v4 SR module...")
    v4_model = ResTranOCR(
        num_classes=config.NUM_CLASSES,
        transformer_heads=config.TRANSFORMER_HEADS,
        transformer_layers=config.TRANSFORMER_LAYERS,
        transformer_ff_dim=config.TRANSFORMER_FF_DIM,
        dropout=config.TRANSFORMER_DROPOUT,
        use_stn=config.USE_STN,
        pretrained=False,
        use_sr=True,
        sr_num_blocks=config.SR_NUM_BLOCKS,
        sr_scale=config.SR_SCALE,
        sr_nf=config.SR_NF,
        sr_gc=config.SR_GC,
        sr_feed_hr=False,
        sr_use_checkpoint=False,
    ).to(device)
    v4_state = torch.load(V4_CHECKPOINT, map_location="cpu")
    v4_model.load_state_dict(v4_state, strict=False)
    v4_model.eval()
    sr_module = v4_model.sr

    # --- Load baseline OCR model (no SR head) ---
    print("Loading baseline OCR model...")
    baseline_model = ResTranOCR(
        num_classes=config.NUM_CLASSES,
        transformer_heads=config.TRANSFORMER_HEADS,
        transformer_layers=config.TRANSFORMER_LAYERS,
        transformer_ff_dim=config.TRANSFORMER_FF_DIM,
        dropout=config.TRANSFORMER_DROPOUT,
        use_stn=config.USE_STN,
        pretrained=False,
        use_sr=False,
        sr_use_checkpoint=False,
    ).to(device)
    baseline_state = torch.load(BASELINE_CHECKPOINT, map_location="cpu")
    missing, unexpected = baseline_model.load_state_dict(baseline_state, strict=False)
    if missing:
        print(f"  Missing keys: {len(missing)}")
    baseline_model.eval()

    # --- Test dataset ---
    if not os.path.exists(config.TEST_DATA_ROOT):
        print(f"Test data not found: {config.TEST_DATA_ROOT}")
        sys.exit(1)

    test_ds = MultiFrameDataset(
        root_dir=config.TEST_DATA_ROOT,
        mode="test",
        is_test=False,
        img_height=config.IMG_HEIGHT,
        img_width=config.IMG_WIDTH,
        char2idx=config.CHAR2IDX,
        seed=config.SEED,
    )
    if len(test_ds) == 0:
        print("No labeled test samples found.")
        sys.exit(1)

    loader = DataLoader(
        test_ds,
        batch_size=16,
        shuffle=False,
        collate_fn=MultiFrameDataset.collate_fn,
        num_workers=0,
        pin_memory=False,
    )

    rows = []
    correct = 0
    total = 0
    t0 = time.time()

    with torch.no_grad():
        for i, batch in enumerate(loader):
            images, _, _, labels_text, track_ids, _, _ = batch
            b, f, c, h, w = images.shape
            images_flat = images.view(b * f, c, h, w)

            # Denormalize to [0, 1] image domain
            images_01 = (images_flat * _STD + _MEAN).clamp(0.0, 1.0)

            # Apply SR: [B*F, 3, H, W] -> [B*F, 3, 2H, 2W]
            hr = sr_module(images_01)

            # Resize back to original LR size [B*F, 3, H, W]
            lr_sr = F.interpolate(hr, size=(h, w), mode="bilinear", align_corners=False)

            # Re-normalize for OCR backbone
            lr_sr_norm = (lr_sr - _MEAN) / _STD
            lr_sr_norm = lr_sr_norm.view(b, f, c, h, w)

            preds = baseline_model(lr_sr_norm)
            decoded = decode_with_confidence(preds, config.IDX2CHAR)

            for j, (pred_text, conf) in enumerate(decoded):
                gt = labels_text[j]
                is_correct = pred_text == gt
                rows.append((track_ids[j], pred_text, conf, gt, is_correct))
                correct += int(is_correct)
                total += 1

            if (i + 1) % 15 == 0:
                elapsed = time.time() - t0
                remaining = elapsed / (i + 1) * (len(loader) - i - 1)
                print(
                    f"  {total}/{len(test_ds)} | acc: {100*correct/total:.1f}% | ETA {remaining:.0f}s",
                    flush=True,
                )

    elapsed = time.time() - t0
    acc = 100.0 * correct / max(1, total)
    print(f"\nSR-preprocess + baseline OCR: {correct}/{total} = {acc:.2f}%  ({elapsed:.0f}s on CPU)")

    os.makedirs(os.path.dirname(OUTPUT_CSV) if os.path.dirname(OUTPUT_CSV) else ".", exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["track_id", "prediction", "confidence", "ground_truth", "correct"])
        for row in rows:
            writer.writerow([row[0], row[1], f"{row[2]:.4f}", row[3], int(row[4])])
    print(f"Saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()

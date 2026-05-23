#!/usr/bin/env python3
"""CPU-based test evaluation for a saved checkpoint (runs concurrently with GPU training).

Usage:
    python eval_test_cpu.py [checkpoint_path] [output_csv]

Defaults:
    checkpoint_path = results/restran_sr_v4_best.pth
    output_csv      = results/test_predictions_restran_sr_v4_cpu.csv
"""
import csv
import os
import sys
import time

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.config import Config
from src.data.dataset import MultiFrameDataset
from src.models.restran import ResTranOCR
from src.utils.postprocess import decode_with_confidence


def main() -> None:
    checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else "results/restran_sr_v4_best.pth"
    output_csv = sys.argv[2] if len(sys.argv) > 2 else "results/test_predictions_restran_sr_v4_cpu.csv"
    # Optional third positional arg: "feed_hr" to force sr_feed_hr=True regardless of config
    force_feed_hr = len(sys.argv) > 3 and sys.argv[3] == "feed_hr"

    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    config = Config()
    device = torch.device("cpu")
    sr_feed_hr = force_feed_hr or config.SR_FEED_HR

    model = ResTranOCR(
        num_classes=config.NUM_CLASSES,
        transformer_heads=config.TRANSFORMER_HEADS,
        transformer_layers=config.TRANSFORMER_LAYERS,
        transformer_ff_dim=config.TRANSFORMER_FF_DIM,
        dropout=config.TRANSFORMER_DROPOUT,
        use_stn=config.USE_STN,
        pretrained=False,
        use_sr=config.USE_SR,
        sr_num_blocks=config.SR_NUM_BLOCKS,
        sr_scale=config.SR_SCALE,
        sr_nf=config.SR_NF,
        sr_gc=config.SR_GC,
        sr_feed_hr=sr_feed_hr,
        sr_use_checkpoint=False,  # no checkpointing on CPU
    ).to(device)

    state_dict = torch.load(checkpoint_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  Missing keys: {len(missing)}")
    model.eval()

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
        batch_size=32,
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
            images = images.to(device)
            preds = model(images)
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
                print(f"  {total}/{len(test_ds)} | acc so far: {100*correct/total:.1f}% | ETA {remaining:.0f}s", flush=True)

    elapsed = time.time() - t0
    acc = 100.0 * correct / max(1, total)
    print(f"\nTest accuracy: {correct}/{total} = {acc:.2f}%  ({elapsed:.0f}s on CPU)")

    os.makedirs(os.path.dirname(output_csv) if os.path.dirname(output_csv) else ".", exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["track_id", "prediction", "confidence", "ground_truth", "correct"])
        for row in rows:
            writer.writerow([row[0], row[1], f"{row[2]:.4f}", row[3], int(row[4])])

    print(f"Saved predictions to {output_csv}")


if __name__ == "__main__":
    main()

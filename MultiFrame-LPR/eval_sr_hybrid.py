#!/usr/bin/env python3
"""Validation-calibrated SR preprocessing for the baseline OCR checkpoint.

The script runs two paths per sample:
  1. baseline OCR on the original LR frames
  2. trained RRDB SR -> downsample/blend -> baseline OCR

A confidence-delta threshold is selected on the validation split, then applied
unchanged to the released labelled test split.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from typing import Iterable, List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.config import Config
from src.data.dataset import MultiFrameDataset
from src.models.restran import ResTranOCR
from src.utils.common import seed_everything
from src.utils.postprocess import decode_with_confidence


@dataclass
class DualPrediction:
    track_id: str
    label: str
    base_pred: str
    base_conf: float
    sr_pred: str
    sr_conf: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate validation-calibrated SR hybrid.")
    parser.add_argument("--ocr-checkpoint", default="results/restran_best.pth")
    parser.add_argument("--sr-checkpoint", default="results/restran_sr_v4_best.pth")
    parser.add_argument("--output-csv", default="results/test_predictions_sr_hybrid.csv")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--sr-blend", type=float, default=1.0)
    parser.add_argument("--threshold-min", type=float, default=-0.2)
    parser.add_argument("--threshold-max", type=float, default=0.2)
    parser.add_argument("--threshold-step", type=float, default=0.001)
    return parser.parse_args()


def build_ocr(config: Config, checkpoint: str, device: torch.device) -> ResTranOCR:
    model = ResTranOCR(
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
    model.load_state_dict(torch.load(checkpoint, map_location=device), strict=False)
    model.eval()
    return model


def build_sr_holder(config: Config, checkpoint: str, device: torch.device) -> ResTranOCR:
    model = ResTranOCR(
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
    model.load_state_dict(torch.load(checkpoint, map_location=device), strict=False)
    model.eval()
    return model


def build_dataset(config: Config, root_dir: str, mode: str) -> MultiFrameDataset:
    return MultiFrameDataset(
        root_dir=root_dir,
        mode=mode,
        split_ratio=config.SPLIT_RATIO,
        img_height=config.IMG_HEIGHT,
        img_width=config.IMG_WIDTH,
        char2idx=config.CHAR2IDX,
        val_split_file=config.VAL_SPLIT_FILE,
        seed=config.SEED,
    )


def collect_dual_predictions(
    loader: DataLoader,
    ocr_model: ResTranOCR,
    sr_module: torch.nn.Module,
    config: Config,
    device: torch.device,
    sr_blend: float,
) -> List[DualPrediction]:
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    alpha = max(0.0, min(1.0, sr_blend))
    rows: List[DualPrediction] = []

    with torch.no_grad():
        for batch in loader:
            images, _, _, labels, track_ids, _, _ = batch
            images = images.to(device)
            b, f, c, h, w = images.shape

            base_decoded = decode_with_confidence(ocr_model(images), config.IDX2CHAR)

            flat = images.view(b * f, c, h, w)
            image_01 = (flat * std + mean).clamp(0.0, 1.0)
            sr_hr = sr_module(image_01)
            sr_lr = F.interpolate(
                sr_hr,
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            )
            blended = image_01 + alpha * (sr_lr - image_01)
            sr_images = ((blended - mean) / std).view(b, f, c, h, w)
            sr_decoded = decode_with_confidence(ocr_model(sr_images), config.IDX2CHAR)

            for i, ((base_pred, base_conf), (sr_pred, sr_conf)) in enumerate(
                zip(base_decoded, sr_decoded)
            ):
                rows.append(
                    DualPrediction(
                        track_id=track_ids[i],
                        label=labels[i],
                        base_pred=base_pred,
                        base_conf=float(base_conf),
                        sr_pred=sr_pred,
                        sr_conf=float(sr_conf),
                    )
                )
    return rows


def iter_thresholds(min_value: float, max_value: float, step: float) -> Iterable[float]:
    count = int(round((max_value - min_value) / step))
    for idx in range(count + 1):
        yield min_value + idx * step


def accuracy(rows: List[DualPrediction], threshold: float | None) -> tuple[int, float]:
    correct = 0
    for row in rows:
        if threshold is None:
            pred = row.sr_pred
        else:
            pred = row.sr_pred if row.sr_conf - row.base_conf > threshold else row.base_pred
        correct += int(pred == row.label)
    return correct, 100.0 * correct / max(1, len(rows))


def tune_threshold(rows: List[DualPrediction], args: argparse.Namespace) -> tuple[float, int, float]:
    best_threshold = 0.0
    best_correct = -1
    best_acc = 0.0
    for threshold in iter_thresholds(args.threshold_min, args.threshold_max, args.threshold_step):
        correct, acc = accuracy(rows, threshold)
        if correct > best_correct:
            best_threshold = threshold
            best_correct = correct
            best_acc = acc
    return best_threshold, best_correct, best_acc


def save_test_rows(rows: List[DualPrediction], threshold: float, output_csv: str) -> None:
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "track_id",
                "prediction",
                "confidence",
                "ground_truth",
                "correct",
                "path",
                "base_prediction",
                "base_confidence",
                "sr_prediction",
                "sr_confidence",
            ]
        )
        for row in rows:
            use_sr = row.sr_conf - row.base_conf > threshold
            pred = row.sr_pred if use_sr else row.base_pred
            conf = row.sr_conf if use_sr else row.base_conf
            writer.writerow(
                [
                    row.track_id,
                    pred,
                    f"{conf:.4f}",
                    row.label,
                    int(pred == row.label),
                    "sr" if use_sr else "baseline",
                    row.base_pred,
                    f"{row.base_conf:.4f}",
                    row.sr_pred,
                    f"{row.sr_conf:.4f}",
                ]
            )


def main() -> None:
    args = parse_args()
    config = Config()
    seed_everything(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Hybrid configuration:")
    print(f"   OCR: {args.ocr_checkpoint}")
    print(f"   SR: {args.sr_checkpoint}")
    print(f"   SR_BLEND: {args.sr_blend}")
    print(f"   DEVICE: {device}")

    ocr_model = build_ocr(config, args.ocr_checkpoint, device)
    sr_holder = build_sr_holder(config, args.sr_checkpoint, device)
    sr_module = sr_holder.sr

    val_ds = build_dataset(config, config.DATA_ROOT, "val")
    test_ds = build_dataset(config, config.TEST_DATA_ROOT, "test")
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=MultiFrameDataset.collate_fn,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=MultiFrameDataset.collate_fn,
        num_workers=args.num_workers,
    )

    print("Collecting validation predictions...")
    val_rows = collect_dual_predictions(
        val_loader, ocr_model, sr_module, config, device, args.sr_blend
    )
    base_val_correct = sum(int(row.base_pred == row.label) for row in val_rows)
    sr_val_correct, sr_val_acc = accuracy(val_rows, None)
    threshold, hybrid_val_correct, hybrid_val_acc = tune_threshold(val_rows, args)
    print(
        f"Validation: baseline {base_val_correct}/{len(val_rows)} = "
        f"{100.0 * base_val_correct / len(val_rows):.2f}% | "
        f"SR-only {sr_val_correct}/{len(val_rows)} = {sr_val_acc:.2f}% | "
        f"hybrid {hybrid_val_correct}/{len(val_rows)} = {hybrid_val_acc:.2f}% "
        f"at threshold {threshold:.3f}"
    )

    print("Collecting test predictions...")
    test_rows = collect_dual_predictions(
        test_loader, ocr_model, sr_module, config, device, args.sr_blend
    )
    base_test_correct = sum(int(row.base_pred == row.label) for row in test_rows)
    sr_test_correct, sr_test_acc = accuracy(test_rows, None)
    hybrid_test_correct, hybrid_test_acc = accuracy(test_rows, threshold)
    print(
        f"Test: baseline {base_test_correct}/{len(test_rows)} = "
        f"{100.0 * base_test_correct / len(test_rows):.2f}% | "
        f"SR-only {sr_test_correct}/{len(test_rows)} = {sr_test_acc:.2f}% | "
        f"hybrid {hybrid_test_correct}/{len(test_rows)} = {hybrid_test_acc:.2f}%"
    )
    save_test_rows(test_rows, threshold, args.output_csv)
    print(f"Saved hybrid predictions to {args.output_csv}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Evaluate a trained checkpoint on the released ICPR 2026 LRLPR test split."""

import argparse
import csv
import os
import sys
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.config import Config
from src.data.dataset import MultiFrameDataset
from src.models.crnn import MultiFrameCRNN
from src.models.restran import ResTranOCR
from src.utils.common import seed_everything
from src.utils.postprocess import decode_with_confidence


def parse_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "y", "t", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "f", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run test evaluation/inference")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="results/restran_best.pth",
        help="Path to a saved model state_dict",
    )
    parser.add_argument(
        "--test-data-root",
        type=str,
        default=None,
        help="Path to released test split",
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        choices=["crnn", "restran"],
        default=None,
        help="Model architecture used by the checkpoint",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--no-stn", action="store_true", help="Disable STN alignment")
    parser.add_argument("--no-sr", action="store_true", help="Disable the SR frontend")
    parser.add_argument(
        "--sr-feed-hr",
        action="store_true",
        help="Feed the 2x SR output directly to STN/ResNet instead of downsampling.",
    )
    parser.add_argument(
        "--sr-blend",
        type=float,
        default=None,
        help="Blend factor for SR OCR input: 0 uses original LR, 1 uses full SR output.",
    )
    parser.add_argument(
        "--use_sr", "--use-sr",
        dest="use_sr",
        nargs="?",
        const=True,
        type=parse_bool,
        default=None,
        help="Enable the RRDB super-resolution frontend (default: from config)",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="CSV output for labelled test or TXT output for unlabeled test",
    )
    return parser.parse_args()


def edit_distance(source: str, target: str) -> int:
    if source == target:
        return 0
    if not source:
        return len(target)
    if not target:
        return len(source)

    previous = list(range(len(target) + 1))
    for i, source_char in enumerate(source, start=1):
        current = [i]
        for j, target_char in enumerate(target, start=1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            replace_cost = previous[j - 1] + (source_char != target_char)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current
    return previous[-1]


def build_model(config: Config) -> torch.nn.Module:
    if config.MODEL_TYPE == "restran":
        return ResTranOCR(
            num_classes=config.NUM_CLASSES,
            transformer_heads=config.TRANSFORMER_HEADS,
            transformer_layers=config.TRANSFORMER_LAYERS,
            transformer_ff_dim=config.TRANSFORMER_FF_DIM,
            dropout=config.TRANSFORMER_DROPOUT,
            use_stn=config.USE_STN,
            use_sr=getattr(config, "USE_SR", False),
            sr_num_blocks=getattr(config, "SR_NUM_BLOCKS", 8),
            sr_scale=getattr(config, "SR_SCALE", 2),
            sr_nf=getattr(config, "SR_NF", 32),
            sr_gc=getattr(config, "SR_GC", 16),
            sr_feed_hr=getattr(config, "SR_FEED_HR", False),
            sr_blend=getattr(config, "SR_BLEND", 1.0),
        ).to(config.DEVICE)

    return MultiFrameCRNN(
        num_classes=config.NUM_CLASSES,
        hidden_size=config.HIDDEN_SIZE,
        rnn_dropout=config.RNN_DROPOUT,
        use_stn=config.USE_STN,
    ).to(config.DEVICE)


def build_dataset(config: Config) -> Tuple[MultiFrameDataset, bool]:
    common = {
        "img_height": config.IMG_HEIGHT,
        "img_width": config.IMG_WIDTH,
        "char2idx": config.CHAR2IDX,
        "seed": config.SEED,
    }
    labelled_ds = MultiFrameDataset(
        root_dir=config.TEST_DATA_ROOT,
        mode="test",
        is_test=False,
        **common,
    )
    if len(labelled_ds) > 0:
        return labelled_ds, True

    unlabeled_ds = MultiFrameDataset(
        root_dir=config.TEST_DATA_ROOT,
        mode="test",
        is_test=True,
        **common,
    )
    return unlabeled_ds, False


def evaluate_labelled(
    model: torch.nn.Module,
    loader: DataLoader,
    config: Config,
    output_file: str,
) -> Dict[str, float]:
    criterion = nn.CTCLoss(blank=0, zero_infinity=True, reduction="mean")
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    total_edits = 0
    total_chars = 0
    rows: List[Tuple[str, str, float, str, bool]] = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Testing"):
            images, targets, target_lengths, labels_text, track_ids, _, _ = batch
            images = images.to(config.DEVICE)
            targets = targets.to(config.DEVICE)
            preds = model(images)
            input_lengths = torch.full((images.size(0),), preds.size(1), dtype=torch.long)
            loss = criterion(preds.permute(1, 0, 2), targets, input_lengths, target_lengths)
            total_loss += loss.item()

            decoded_list = decode_with_confidence(preds, config.IDX2CHAR)
            for i, (pred_text, conf) in enumerate(decoded_list):
                gt_text = labels_text[i]
                track_id = track_ids[i]
                correct = pred_text == gt_text
                total_correct += int(correct)
                total_edits += edit_distance(pred_text, gt_text)
                total_chars += len(gt_text)
                total_samples += 1
                rows.append((track_id, pred_text, conf, gt_text, correct))

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["track_id", "prediction", "confidence", "ground_truth", "correct"])
        for track_id, pred_text, conf, gt_text, correct in rows:
            writer.writerow([track_id, pred_text, f"{conf:.4f}", gt_text, int(correct)])

    metrics = {
        "loss": total_loss / len(loader) if len(loader) else 0.0,
        "acc": (total_correct / total_samples) * 100 if total_samples else 0.0,
        "cer": (total_edits / total_chars) * 100 if total_chars else 0.0,
    }
    return metrics


def predict_unlabelled(
    model: torch.nn.Module,
    loader: DataLoader,
    config: Config,
    output_file: str,
) -> None:
    results: List[str] = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Inferencing"):
            images, _, _, _, track_ids, _, _ = batch
            images = images.to(config.DEVICE)
            preds = model(images)
            decoded_list = decode_with_confidence(preds, config.IDX2CHAR)
            for i, (pred_text, conf) in enumerate(decoded_list):
                results.append(f"{track_ids[i]},{pred_text};{conf:.4f}")

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(results))


def main() -> None:
    args = parse_args()
    config = Config()
    if args.model is not None:
        config.MODEL_TYPE = args.model
    if args.test_data_root is not None:
        config.TEST_DATA_ROOT = args.test_data_root
    if args.batch_size is not None:
        config.BATCH_SIZE = args.batch_size
    if args.num_workers is not None:
        config.NUM_WORKERS = args.num_workers
    if args.no_stn:
        config.USE_STN = False
    if args.use_sr is not None:
        config.USE_SR = bool(args.use_sr)
    if args.no_sr:
        config.USE_SR = False
    if args.sr_feed_hr:
        config.SR_FEED_HR = True
    if args.sr_blend is not None:
        config.SR_BLEND = args.sr_blend

    seed_everything(config.SEED)

    if not os.path.exists(config.TEST_DATA_ROOT):
        print(f"ERROR: Test data root not found: {config.TEST_DATA_ROOT}")
        sys.exit(1)
    if not os.path.exists(args.checkpoint):
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    print("Test configuration:")
    print(f"   DATA: {config.TEST_DATA_ROOT}")
    print(f"   MODEL: {config.MODEL_TYPE}")
    print(f"   USE_SR: {getattr(config, 'USE_SR', False)} | "
          f"feed_hr={getattr(config, 'SR_FEED_HR', False)} | "
          f"blend={getattr(config, 'SR_BLEND', 1.0)}")
    print(f"   CHECKPOINT: {args.checkpoint}")
    print(f"   DEVICE: {config.DEVICE}")

    dataset, is_labelled = build_dataset(config)
    if len(dataset) == 0:
        print("ERROR: Test dataset is empty.")
        sys.exit(1)

    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=MultiFrameDataset.collate_fn,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.DEVICE.type == "cuda",
    )

    model = build_model(config)
    state_dict = torch.load(args.checkpoint, map_location=config.DEVICE)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Checkpoint missing {len(missing)} keys (initialized fresh).")
    if unexpected:
        print(f"Checkpoint had {len(unexpected)} unexpected keys (ignored).")
    print("Weights loaded.")

    if is_labelled:
        output_file = args.output_file or f"results/test_predictions_{config.MODEL_TYPE}.csv"
        metrics = evaluate_labelled(model, loader, config, output_file)
        print(
            f"Test Results: Loss: {metrics['loss']:.4f} | "
            f"Acc: {metrics['acc']:.2f}% | CER: {metrics['cer']:.2f}%"
        )
        print(f"Saved labelled predictions to {output_file}")
    else:
        output_file = args.output_file or "results/submission_final.txt"
        predict_unlabelled(model, loader, config, output_file)
        print(f"Saved predictions to {output_file}")


if __name__ == "__main__":
    main()

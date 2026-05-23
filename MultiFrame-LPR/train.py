#!/usr/bin/env python3
"""Main entry point for OCR training and released-dataset evaluation."""

import argparse
import os
import sys
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.config import Config
from src.data.dataset import MultiFrameDataset
from src.models.crnn import MultiFrameCRNN
from src.models.restran import ResTranOCR
from src.training.trainer import Trainer
from src.utils.common import seed_everything


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
    parser = argparse.ArgumentParser(
        description="Train Multi-Frame OCR for the released ICPR 2026 LRLPR dataset"
    )
    parser.add_argument(
        "-n", "--experiment-name", type=str, default=None,
        help="Experiment name for checkpoint/result files (default: from config)",
    )
    parser.add_argument(
        "-m", "--model", type=str, choices=["crnn", "restran"], default=None,
        help="Model architecture: crnn or restran (default: from config)",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Training batch size")
    parser.add_argument(
        "--lr", "--learning-rate", type=float, default=None, dest="learning_rate",
        help="Learning rate",
    )
    parser.add_argument(
        "--data-root", type=str, default=None,
        help="Root directory for released train split",
    )
    parser.add_argument(
        "--test-data-root", type=str, default=None,
        help="Root directory for released test split",
    )
    parser.add_argument(
        "--val-split-file", type=str, default=None,
        help="Path to save/load train-to-val split JSON",
    )
    parser.add_argument(
        "--split-ratio", type=float, default=None,
        help="Fraction of train tracks used for training (default: from config)",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--num-workers", type=int, default=None, help="DataLoader workers")
    parser.add_argument("--hidden-size", type=int, default=None, help="LSTM hidden size for CRNN")
    parser.add_argument("--transformer-heads", type=int, default=None)
    parser.add_argument("--transformer-layers", type=int, default=None)
    parser.add_argument(
        "--aug-level",
        type=str,
        choices=["full", "light"],
        default=None,
        help="Augmentation level for training data",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save checkpoints and result files",
    )
    parser.add_argument("--no-stn", action="store_true", help="Disable STN alignment")
    parser.add_argument(
        "--use_sr", "--use-sr",
        dest="use_sr",
        nargs="?",
        const=True,
        type=parse_bool,
        default=None,
        help="Enable the RRDB super-resolution frontend (default: from config; true)",
    )
    parser.add_argument("--no-sr", action="store_true", help="Disable the SR frontend")
    parser.add_argument(
        "--lambda_sr", "--lambda-sr",
        dest="lambda_sr",
        type=float, default=None,
        help="Weight for the SR L1 reconstruction loss (default: 0.1)",
    )
    parser.add_argument(
        "--sr-lr",
        dest="sr_lr",
        type=float, default=None,
        help="Separate learning rate for the SR module. When set, --lr applies only to OCR "
             "and the SR module trains at this (typically higher) rate. Example: --lr 1e-5 --sr-lr 2e-4",
    )
    parser.add_argument(
        "--sr-freeze-epochs", type=int, default=None,
        help="Number of initial epochs to keep the SR module frozen",
    )
    parser.add_argument(
        "--ocr-freeze-epochs", type=int, default=None,
        help="Number of initial epochs to freeze all OCR params (SR-only training phase)",
    )
    parser.add_argument(
        "--sr-feed-hr",
        action="store_true",
        help="Feed the 2x SR output directly to STN/ResNet instead of downsampling.",
    )
    parser.add_argument(
        "--sr-blend",
        dest="sr_blend",
        type=float,
        default=None,
        help="Blend factor for SR OCR input: 0 uses original LR, 1 uses full SR output.",
    )
    parser.add_argument(
        "--init-checkpoint",
        type=str,
        default=None,
        help="Checkpoint to initialize from when the experiment checkpoint does not exist.",
    )
    parser.add_argument(
        "--no-test-eval",
        action="store_true",
        help="Skip labelled test evaluation after training",
    )
    parser.add_argument(
        "--submission-mode",
        action="store_true",
        help="Train on all labelled train data and run test inference/evaluation",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build datasets/model and run one forward pass, then exit",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        choices=["onecycle", "cosine"],
        default=None,
        help="LR scheduler: 'onecycle' (default, with warmup) or 'cosine' (no warmup, starts at --lr)",
    )
    return parser.parse_args()


def apply_cli_overrides(config: Config, args: argparse.Namespace) -> None:
    arg_to_config = {
        "experiment_name": "EXPERIMENT_NAME",
        "model": "MODEL_TYPE",
        "epochs": "EPOCHS",
        "batch_size": "BATCH_SIZE",
        "learning_rate": "LEARNING_RATE",
        "data_root": "DATA_ROOT",
        "test_data_root": "TEST_DATA_ROOT",
        "val_split_file": "VAL_SPLIT_FILE",
        "split_ratio": "SPLIT_RATIO",
        "seed": "SEED",
        "num_workers": "NUM_WORKERS",
        "hidden_size": "HIDDEN_SIZE",
        "transformer_heads": "TRANSFORMER_HEADS",
        "transformer_layers": "TRANSFORMER_LAYERS",
    }

    for arg_name, config_name in arg_to_config.items():
        value = getattr(args, arg_name, None)
        if value is not None:
            setattr(config, config_name, value)

    if args.aug_level is not None:
        config.AUGMENTATION_LEVEL = args.aug_level
    if args.experiment_name is None and args.model is not None:
        config.EXPERIMENT_NAME = config.MODEL_TYPE
    if args.no_stn:
        config.USE_STN = False
    if args.use_sr is not None:
        config.USE_SR = bool(args.use_sr)
    if args.no_sr:
        config.USE_SR = False
    if args.lambda_sr is not None:
        config.LAMBDA_SR = args.lambda_sr
    if args.sr_freeze_epochs is not None:
        config.SR_FREEZE_EPOCHS = args.sr_freeze_epochs
    if getattr(args, "ocr_freeze_epochs", None) is not None:
        config.OCR_FREEZE_EPOCHS = args.ocr_freeze_epochs
    if getattr(args, "sr_lr", None) is not None:
        config.SR_LR = args.sr_lr
    if args.sr_feed_hr:
        config.SR_FEED_HR = True
    if getattr(args, "sr_blend", None) is not None:
        config.SR_BLEND = args.sr_blend
    if getattr(args, "scheduler", None) is not None:
        config.SCHEDULER = args.scheduler
    config.OUTPUT_DIR = args.output_dir


def make_loader(
    dataset: MultiFrameDataset,
    config: Config,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=shuffle,
        collate_fn=MultiFrameDataset.collate_fn,
        num_workers=config.NUM_WORKERS,
        pin_memory=False,
        persistent_workers=config.NUM_WORKERS > 0,
    )


def build_model(config: Config) -> torch.nn.Module:
    if config.MODEL_TYPE == "restran":
        return ResTranOCR(
            num_classes=config.NUM_CLASSES,
            transformer_heads=config.TRANSFORMER_HEADS,
            transformer_layers=config.TRANSFORMER_LAYERS,
            transformer_ff_dim=config.TRANSFORMER_FF_DIM,
            dropout=config.TRANSFORMER_DROPOUT,
            use_stn=config.USE_STN,
            pretrained=getattr(config, "USE_PRETRAINED", True),
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


def build_test_dataset(config: Config) -> Tuple[Optional[MultiFrameDataset], bool]:
    """Build released test dataset, preferring labelled evaluation if annotations exist."""
    if not os.path.exists(config.TEST_DATA_ROOT):
        print(f"WARNING: Test data root not found: {config.TEST_DATA_ROOT}")
        return None, False

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
    if len(unlabeled_ds) > 0:
        return unlabeled_ds, False

    return None, False


def print_config(config: Config, submission_mode: bool, dry_run: bool) -> None:
    print("Configuration:")
    print(f"   EXPERIMENT: {config.EXPERIMENT_NAME}")
    print(f"   MODEL: {config.MODEL_TYPE}")
    print(f"   USE_STN: {config.USE_STN}")
    sr_lr_str = f" | sr_lr={getattr(config, 'SR_LR', None)}" if getattr(config, 'SR_LR', None) else ""
    print(f"   USE_SR: {getattr(config, 'USE_SR', False)} | "
          f"lambda_sr={getattr(config, 'LAMBDA_SR', 0.0)} | "
          f"freeze_epochs={getattr(config, 'SR_FREEZE_EPOCHS', 0)} | "
          f"feed_hr={getattr(config, 'SR_FEED_HR', False)} | "
          f"blend={getattr(config, 'SR_BLEND', 1.0)}{sr_lr_str}")
    print(f"   DATA_ROOT: {config.DATA_ROOT}")
    print(f"   TEST_DATA_ROOT: {config.TEST_DATA_ROOT}")
    print(f"   VAL_SPLIT_FILE: {config.VAL_SPLIT_FILE}")
    print(f"   SPLIT_RATIO: {config.SPLIT_RATIO}")
    print(f"   EPOCHS: {config.EPOCHS}")
    print(f"   BATCH_SIZE: {config.BATCH_SIZE}")
    print(f"   LEARNING_RATE: {config.LEARNING_RATE}")
    print(f"   DEVICE: {config.DEVICE}")
    print(f"   SUBMISSION_MODE: {submission_mode}")
    print(f"   DRY_RUN: {dry_run}")


def run_dry_check(model: torch.nn.Module, train_loader: DataLoader, config: Config) -> None:
    batch = next(iter(train_loader))
    images, targets, target_lengths, labels_text, track_ids, hr_frames, has_hr = batch
    print("Dry run batch:")
    print(f"   images: {tuple(images.shape)}")
    print(f"   targets: {tuple(targets.shape)}")
    print(f"   target_lengths: {target_lengths.tolist()[:8]}")
    print(f"   first_label: {labels_text[0]}")
    print(f"   first_track: {track_ids[0]}")
    print(f"   hr_frames: {tuple(hr_frames.shape)} | has_hr: {has_hr.tolist()[:8]}")

    model.eval()
    with torch.no_grad():
        if getattr(model, "use_sr", False):
            preds, sr_out = model(images.to(config.DEVICE), return_sr=True)
            print(f"   sr_output: {tuple(sr_out.shape)}")
        else:
            preds = model(images.to(config.DEVICE))
    print(f"   model_output: {tuple(preds.shape)}")
    print("Dry run OK.")


def main() -> None:
    args = parse_args()
    config = Config()
    apply_cli_overrides(config, args)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    seed_everything(config.SEED)

    print_config(config, args.submission_mode, args.dry_run)

    if not os.path.exists(config.DATA_ROOT):
        print(f"ERROR: Data root not found: {config.DATA_ROOT}")
        sys.exit(1)
    if config.EPOCHS < 1 and not args.dry_run:
        print("ERROR: --epochs must be >= 1 unless --dry-run is used.")
        sys.exit(1)

    load_hr = config.MODEL_TYPE == "restran" and bool(getattr(config, "USE_SR", False))
    common_ds_params = {
        "split_ratio": config.SPLIT_RATIO,
        "img_height": config.IMG_HEIGHT,
        "img_width": config.IMG_WIDTH,
        "char2idx": config.CHAR2IDX,
        "val_split_file": config.VAL_SPLIT_FILE,
        "seed": config.SEED,
        "augmentation_level": config.AUGMENTATION_LEVEL,
        "load_hr": load_hr,
        "hr_height": getattr(config, "HR_HEIGHT", 64),
        "hr_width": getattr(config, "HR_WIDTH", 256),
    }

    train_ds = MultiFrameDataset(
        root_dir=config.DATA_ROOT,
        mode="train",
        full_train=args.submission_mode,
        **common_ds_params,
    )
    if len(train_ds) == 0:
        print("ERROR: Training dataset is empty.")
        sys.exit(1)

    val_loader = None
    if not args.submission_mode:
        val_ds = MultiFrameDataset(
            root_dir=config.DATA_ROOT,
            mode="val",
            **common_ds_params,
        )
        if len(val_ds) > 0:
            val_loader = make_loader(val_ds, config, shuffle=False)
        else:
            print("WARNING: Validation dataset is empty.")

    train_loader = make_loader(train_ds, config, shuffle=True)
    test_ds, test_is_labelled = build_test_dataset(config)
    test_loader = make_loader(test_ds, config, shuffle=False) if test_ds is not None else None

    model = build_model(config)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model ({config.MODEL_TYPE}): {total_params:,} total params, {trainable_params:,} trainable")

    checkpoint_path = os.path.join(config.OUTPUT_DIR, f"{config.EXPERIMENT_NAME}_best.pth")
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=config.DEVICE)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"  Missing keys (will be initialized fresh): {len(missing)}")
        if unexpected:
            print(f"  Unexpected keys (ignored): {len(unexpected)}")
    elif args.init_checkpoint:
        if not os.path.exists(args.init_checkpoint):
            print(f"ERROR: init checkpoint not found: {args.init_checkpoint}")
            sys.exit(1)
        print(f"Initializing from checkpoint: {args.init_checkpoint}")
        state_dict = torch.load(args.init_checkpoint, map_location=config.DEVICE)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"  Missing keys (will be initialized fresh): {len(missing)}")
        if unexpected:
            print(f"  Unexpected keys (ignored): {len(unexpected)}")
    else:
        print("Starting training from scratch.")

    if args.dry_run:
        run_dry_check(model, train_loader, config)
        return

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        idx2char=config.IDX2CHAR,
    )
    trainer.fit()

    if test_loader is None or args.no_test_eval:
        return

    exp_name = config.EXPERIMENT_NAME
    best_model_path = os.path.join(config.OUTPUT_DIR, f"{exp_name}_best.pth")
    if os.path.exists(best_model_path):
        print(f"Loading best checkpoint for test: {best_model_path}")
        model.load_state_dict(
            torch.load(best_model_path, map_location=config.DEVICE), strict=False,
        )

    if test_is_labelled:
        trainer.evaluate_labeled(
            test_loader,
            split_name="Test",
            output_filename=f"test_predictions_{exp_name}.csv",
        )
    else:
        trainer.predict_test(test_loader, output_filename=f"submission_{exp_name}_test.txt")


if __name__ == "__main__":
    main()

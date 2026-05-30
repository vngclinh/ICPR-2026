"""Train one of the multi-architecture models (svtr / new_svtr / restran / crnn / mamba).

Usage:
    python train_multi_arch.py -n multi_svtr -m svtr --epochs 25 --batch-size 48
    python train_multi_arch.py -n multi_new_svtr -m new_svtr --epochs 25 --batch-size 32
    python train_multi_arch.py -n multi_restran -m restran --epochs 25 --batch-size 16 --no-sr
    python train_multi_arch.py -n multi_crnn -m crnn --epochs 25 --batch-size 48
"""
import argparse
import os
import sys
from dataclasses import replace

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.icpr2026_base import ICPR2026Config
from src.data.dataset import MultiFrameDataset
from src.utils.common import seed_everything


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("-n", "--experiment-name", type=str, required=True)
    p.add_argument("-m", "--model", choices=["svtr", "new_svtr", "restran", "mamba", "crnn"], required=True)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--no-stn", action="store_true")
    p.add_argument("--no-sr", action="store_true")
    p.add_argument("--data-root", type=str, default=None)
    return p.parse_args()


def main():
    args = _parse_args()

    cfg = ICPR2026Config()
    overrides = {
        "EXPERIMENT_NAME": args.experiment_name,
        "EPOCHS": args.epochs,
        "BATCH_SIZE": args.batch_size,
        "LEARNING_RATE": args.lr,
        "NUM_WORKERS": args.num_workers,
        "USE_STN": not args.no_stn,
        "GRAD_CLIP": 2.0,
    }
    if args.data_root:
        overrides["DATA_ROOT"] = args.data_root
    cfg = replace(cfg, **overrides)
    cfg.__post_init__()
    # MODEL_TYPE is read by the trainer for the SVTR-vs-others branch.
    cfg.MODEL_TYPE = args.model

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    seed_everything(cfg.SEED)

    print(f"🚀 Config: {cfg.EXPERIMENT_NAME} | Model: {cfg.MODEL_TYPE} | Epochs: {cfg.EPOCHS}")

    # --- Dataset / loaders ---
    train_ds = MultiFrameDataset(
        root_dir=cfg.DATA_ROOT, mode='train', split_ratio=cfg.SPLIT_RATIO,
        img_height=cfg.IMG_HEIGHT, img_width=cfg.IMG_WIDTH,
        char2idx=cfg.CHAR2IDX, val_split_file=cfg.VAL_SPLIT_FILE,
        seed=cfg.SEED, augmentation_level='full',
        load_hr=True, hr_height=cfg.IMG_HEIGHT, hr_width=cfg.IMG_WIDTH,
    )
    val_ds = MultiFrameDataset(
        root_dir=cfg.DATA_ROOT, mode='val', split_ratio=cfg.SPLIT_RATIO,
        img_height=cfg.IMG_HEIGHT, img_width=cfg.IMG_WIDTH,
        char2idx=cfg.CHAR2IDX, val_split_file=cfg.VAL_SPLIT_FILE, seed=cfg.SEED,
    )
    train_loader = DataLoader(
        train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
        collate_fn=MultiFrameDataset.collate_fn, num_workers=cfg.NUM_WORKERS, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
        collate_fn=MultiFrameDataset.collate_fn, num_workers=cfg.NUM_WORKERS, pin_memory=True,
    ) if len(val_ds) > 0 else None

    # --- Model ---
    if args.model == "svtr":
        from src.models.multi_arch.svtr import SVTROCR
        model = SVTROCR(
            num_classes=cfg.NUM_CLASSES, img_size=(cfg.IMG_HEIGHT, cfg.IMG_WIDTH),
            transformer_heads=cfg.NHEAD, transformer_layers=3, transformer_ff_dim=2048,
            dropout=0.1, use_stn=cfg.USE_STN, max_len=25, attn_weight=0.5,
        )
    elif args.model == "new_svtr":
        from src.models.multi_arch.new_svtr import svtrNew
        model = svtrNew(
            num_classes=cfg.NUM_CLASSES, img_size=(cfg.IMG_HEIGHT, cfg.IMG_WIDTH),
            transformer_heads=cfg.NHEAD, transformer_layers=4, transformer_ff_dim=2048,
            dropout=0.1, use_stn=cfg.USE_STN, use_sr=not args.no_sr,
        )
    elif args.model == "restran":
        from src.models.multi_arch.restran import ResTranOCR
        model = ResTranOCR(
            num_classes=cfg.NUM_CLASSES,
            transformer_heads=cfg.NHEAD, transformer_layers=3, transformer_ff_dim=2048,
            dropout=0.1, use_stn=cfg.USE_STN, use_sr=not args.no_sr,
        )
    elif args.model == "mamba":
        from src.models.multi_arch.mamba import NeuroMambaOCR
        model = NeuroMambaOCR(
            num_classes=cfg.NUM_CLASSES, mamba_layers=3,
            use_stn=cfg.USE_STN, use_sr=not args.no_sr,
        )
    elif args.model == "crnn":
        from src.models.multi_arch.crnn import MultiFrameCRNN
        model = MultiFrameCRNN(num_classes=cfg.NUM_CLASSES, use_stn=cfg.USE_STN)
    else:
        raise ValueError(f"Unknown model {args.model}")

    model = model.to(cfg.DEVICE)
    print(f"   Params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    from src.models.multi_arch.trainer import UniversalTrainer
    trainer = UniversalTrainer(
        model=model, train_loader=train_loader, val_loader=val_loader,
        config=cfg, idx2char=cfg.IDX2CHAR,
    )
    trainer.fit()


if __name__ == "__main__":
    main()

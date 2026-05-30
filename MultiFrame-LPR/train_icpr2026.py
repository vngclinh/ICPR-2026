"""Train one V1-V4 variant of the ICPR 2026 LPR pipeline.

Usage:
    python train_icpr2026.py --variant v1
    python train_icpr2026.py --variant v3 --epochs 30 --batch-size 32
    python train_icpr2026.py --variant v4 --lr 3e-4

Notes:
* The validation split lives in ``configs.icpr2026_base.ICPR2026Config.VAL_SPLIT_FILE``.
  If it does not exist, we generate it by sampling 10% of Scenario-B tracks
  (Section 7 of the design doc).
* All four variants share the same data loaders; only the model architecture
  and loss schedule differ.
* Synthetic LR degradation reuses ``get_degradation_transforms`` and is
  triggered for HR-only synthetic samples inside ``MultiFrameDataset`` when
  ``include_synthetic=True``.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
from dataclasses import replace

from torch.utils.data import DataLoader

from configs.icpr2026_base import ICPR2026Config
from configs.icpr2026_variants import build_config
from src.data.dataset import MultiFrameDataset
from src.models.lpr_variants import VariantConfig, build_variant
from src.training.icpr2026_trainer import ICPR2026Trainer


def _generate_scenario_b_split(data_root: str, val_split_file: str, seed: int, ratio: float = 0.9) -> None:
    """Sample 10% of Scenario-B tracks for validation, save as JSON.

    The 90% Scenario-B + entire Scenario-A go to training (Section 7).
    """
    if os.path.exists(val_split_file):
        return
    scen_b_root = os.path.join(data_root, "Scenario-B")
    if not os.path.isdir(scen_b_root):
        raise FileNotFoundError(f"Scenario-B directory not found: {scen_b_root}")

    tracks = sorted(
        p for p in glob.glob(os.path.join(scen_b_root, "**", "track_*"), recursive=True)
        if os.path.isdir(p)
    )
    if not tracks:
        raise RuntimeError(f"No tracks found under {scen_b_root}")

    rng = random.Random(seed)
    shuffled = list(tracks)
    rng.shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) * (1.0 - ratio))))
    val_tracks = sorted(shuffled[:n_val])

    rel = [os.path.relpath(p, data_root).replace(os.sep, "/") for p in val_tracks]
    os.makedirs(os.path.dirname(val_split_file) or ".", exist_ok=True)
    with open(val_split_file, "w", encoding="utf-8") as f:
        json.dump(rel, f, indent=2)
    print(f"[split] Wrote {len(rel)} Scenario-B val tracks to {val_split_file}")


def _build_dataloaders(config: ICPR2026Config) -> tuple[DataLoader, DataLoader]:
    _generate_scenario_b_split(
        config.DATA_ROOT,
        config.VAL_SPLIT_FILE,
        config.SEED,
        ratio=config.SPLIT_RATIO,
    )
    common = dict(
        root_dir=config.DATA_ROOT,
        split_ratio=config.SPLIT_RATIO,
        img_height=config.IMG_HEIGHT,
        img_width=config.IMG_WIDTH,
        char2idx=config.CHAR2IDX,
        val_split_file=config.VAL_SPLIT_FILE,
        seed=config.SEED,
        augmentation_level="full",
        load_hr=False,
        hr_height=config.HR_HEIGHT,
        hr_width=config.HR_WIDTH,
    )
    train_set = MultiFrameDataset(mode="train", **common)
    val_set = MultiFrameDataset(mode="val", **common)

    train_loader = DataLoader(
        train_set,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        collate_fn=MultiFrameDataset.collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        collate_fn=MultiFrameDataset.collate_fn,
    )
    return train_loader, val_loader


def _build_model(config: ICPR2026Config):
    vc = VariantConfig(
        num_classes=config.NUM_CLASSES,
        num_frames=config.NUM_FRAMES,
        use_stn=config.USE_STN,
        stn_mode=config.STN_MODE,
        stn_tps_x=config.STN_TPS_X,
        stn_tps_y=config.STN_TPS_Y,
        d_model=config.D_MODEL,
        nhead=config.NHEAD,
        encoder_layers=max(1, config.ENCODER_LAYERS),
        encoder_ff=config.ENCODER_FF,
        encoder_dropout=config.ENCODER_DROPOUT,
        aux_tap_layer=config.AUX_TAP_LAYER if config.AUX_TAP_LAYER else None,
        decoder_layers=config.DECODER_LAYERS,
        decoder_ff=config.DECODER_FF,
        decoder_dropout=config.DECODER_DROPOUT,
        decoder_num_queries=config.DECODER_NUM_QUERIES,
        fusion_per_position=config.FUSION_PER_POSITION,
    )
    model = build_variant(config.VARIANT, vc).to(config.DEVICE)
    return model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train an ICPR 2026 LPR variant")
    p.add_argument("--variant", choices=["v1", "v2", "v3", "v4", "v5"], default="v1")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--data-root", type=str, default=None)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--no-stn", action="store_true")
    p.add_argument("--stn-mode", choices=["affine", "tps"], default=None)
    p.add_argument("--grad-clip", type=float, default=None)
    p.add_argument("--lambda-center", type=float, default=None,
                   help="Weight for center loss; lower to avoid late-training instability")
    p.add_argument("--lambda-stn", type=float, default=None)
    p.add_argument("--lambda-aux", type=float, default=None)
    p.add_argument("--pct-start", type=float, default=None,
                   help="OneCycle warmup fraction; default 0.1, raise to slow LR ramp-up")
    p.add_argument("--no-ohem", action="store_true",
                   help="Force-disable OHEM CTC even if the variant default has it on")
    p.add_argument("--no-length-penalty", action="store_true",
                   help="Force-disable length penalty even if the variant default has it on")
    p.add_argument("--resume", type=str, default=None,
                   help="Path to checkpoint to load model weights from before training")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = build_config(args.variant)
    overrides: dict = {}
    if args.epochs is not None:
        overrides["EPOCHS"] = args.epochs
    if args.batch_size is not None:
        overrides["BATCH_SIZE"] = args.batch_size
    if args.lr is not None:
        overrides["LEARNING_RATE"] = args.lr
    if args.seed is not None:
        overrides["SEED"] = args.seed
    if args.data_root:
        overrides["DATA_ROOT"] = args.data_root
    if args.output_dir:
        overrides["OUTPUT_DIR"] = args.output_dir
    if args.no_stn:
        overrides["USE_STN"] = False
    if args.stn_mode:
        overrides["STN_MODE"] = args.stn_mode
    if args.grad_clip is not None:
        overrides["GRAD_CLIP"] = args.grad_clip
    if args.lambda_center is not None:
        overrides["LAMBDA_CENTER"] = args.lambda_center
    if args.lambda_stn is not None:
        overrides["LAMBDA_STN"] = args.lambda_stn
    if args.lambda_aux is not None:
        overrides["LAMBDA_AUX_CTC"] = args.lambda_aux
    if args.pct_start is not None:
        overrides["PCT_START"] = args.pct_start
    if args.no_ohem:
        overrides["USE_OHEM"] = False
    if args.no_length_penalty:
        overrides["USE_LENGTH_PENALTY"] = False
    if overrides:
        cfg = replace(cfg, **overrides)
        cfg.__post_init__()

    train_loader, val_loader = _build_dataloaders(cfg)
    model = _build_model(cfg)

    if args.resume:
        import torch as _torch
        print(f"Resuming weights from {args.resume}")
        state = _torch.load(args.resume, map_location=cfg.DEVICE)
        sd = state.get("model", state) if isinstance(state, dict) else state
        model.load_state_dict(sd, strict=False)

    trainer = ICPR2026Trainer(model, train_loader, val_loader, cfg, cfg.IDX2CHAR)
    trainer.fit()


if __name__ == "__main__":
    main()

"""Run the ICPR 2026 ensemble (V1-V4) on the test set and write a submission.

Usage:
    python run_icpr2026_ensemble.py \\
        --ckpt v1=results/icpr2026_v1_best.pth \\
        --ckpt v2=results/icpr2026_v2_best.pth \\
        --ckpt v3=results/icpr2026_v3_best.pth \\
        --ckpt v4=results/icpr2026_v4_best.pth \\
        --weights 1.0,1.0,1.2,1.2 \\
        --fallback v3
"""
from __future__ import annotations

import argparse
import os
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs.icpr2026_variants import build_config
from src.data.dataset import MultiFrameDataset
from src.inference import ensemble_predictions
from src.models.lpr_variants import VariantConfig, build_variant


def _parse_ckpts(raw: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for item in raw:
        if "=" not in item:
            raise ValueError(f"--ckpt expects 'vN=path', got {item!r}")
        name, path = item.split("=", 1)
        out[name.lower()] = path
    return out


def _build_one(name: str, ckpt_path: str, device: torch.device):
    cfg = build_config(name)
    vc = VariantConfig(
        num_classes=cfg.NUM_CLASSES,
        num_frames=cfg.NUM_FRAMES,
        use_stn=cfg.USE_STN,
        stn_mode=cfg.STN_MODE,
        stn_tps_x=cfg.STN_TPS_X,
        stn_tps_y=cfg.STN_TPS_Y,
        d_model=cfg.D_MODEL,
        nhead=cfg.NHEAD,
        encoder_layers=max(1, cfg.ENCODER_LAYERS),
        encoder_ff=cfg.ENCODER_FF,
        encoder_dropout=cfg.ENCODER_DROPOUT,
        aux_tap_layer=cfg.AUX_TAP_LAYER if cfg.AUX_TAP_LAYER else None,
        decoder_layers=cfg.DECODER_LAYERS,
        decoder_ff=cfg.DECODER_FF,
        decoder_dropout=cfg.DECODER_DROPOUT,
        decoder_num_queries=cfg.DECODER_NUM_QUERIES,
        fusion_per_position=cfg.FUSION_PER_POSITION,
    )
    model = build_variant(name, vc).to(device).eval()
    state = torch.load(ckpt_path, map_location=device)
    sd = state.get("model", state) if isinstance(state, dict) else state
    model.load_state_dict(sd, strict=False)
    return cfg, model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", action="append", required=True,
                   help="vN=path/to/checkpoint.pth (use multiple --ckpt flags)")
    p.add_argument("--weights", type=str, default=None,
                   help="Comma-separated weights, same order as --ckpt arguments")
    p.add_argument("--fallback", type=str, default=None,
                   help="Variant name to use as single-model fallback")
    p.add_argument("--fallback-threshold", type=float, default=-1.5)
    p.add_argument("--data-root", type=str, default="data/LRLPR-26-5opEvJTW/test")
    p.add_argument("--output", type=str, default="results/submission_icpr2026_ensemble.txt")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    return p.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpts = _parse_ckpts(args.ckpt)
    if not ckpts:
        raise SystemExit("No checkpoints provided")

    weights = None
    if args.weights:
        weights = [float(x) for x in args.weights.split(",")]
        if len(weights) != len(ckpts):
            raise SystemExit(f"--weights length {len(weights)} != #ckpts {len(ckpts)}")

    fallback_index = 0
    if args.fallback:
        names = list(ckpts.keys())
        if args.fallback.lower() not in names:
            raise SystemExit(f"Fallback {args.fallback!r} not in {names}")
        fallback_index = names.index(args.fallback.lower())

    # Build all models.
    print("Loading models...")
    models: Dict[str, torch.nn.Module] = {}
    char_cfg = None
    for name, path in ckpts.items():
        cfg, model = _build_one(name, path, device)
        models[name] = model
        char_cfg = cfg
        print(f"  {name} <- {path}")
    assert char_cfg is not None

    # Dataset / loader (test, unlabeled).
    test_set = MultiFrameDataset(
        root_dir=args.data_root,
        mode="test",
        img_height=char_cfg.IMG_HEIGHT,
        img_width=char_cfg.IMG_WIDTH,
        char2idx=char_cfg.CHAR2IDX,
        val_split_file=char_cfg.VAL_SPLIT_FILE,
        seed=char_cfg.SEED,
        is_test=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=MultiFrameDataset.collate_fn,
    )

    submission: List[str] = []
    path_counts = {"average": 0, "vote": 0, "fallback": 0}

    for batch in tqdm(test_loader, desc="Ensembling"):
        images, _, _, _, track_ids, _, _ = batch
        images = images.to(device)

        # Collect log-probs from every variant: list of [B, T_i, C].
        per_variant: List[torch.Tensor] = []
        for name in ckpts.keys():
            out = models[name](images)
            per_variant.append(out["log_probs"])

        # Per-sample ensemble.
        b = images.size(0)
        for i in range(b):
            per_sample = [v[i].detach().cpu() for v in per_variant]
            text, score, debug = ensemble_predictions(
                per_sample,
                char2idx=char_cfg.CHAR2IDX,
                idx2char=char_cfg.IDX2CHAR,
                weights=weights,
                fallback_index=fallback_index,
                fallback_threshold=args.fallback_threshold,
                target_length=char_cfg.TARGET_PLATE_LENGTH,
            )
            path_counts[debug["path"]] = path_counts.get(debug["path"], 0) + 1
            conf = float(min(1.0, max(0.0, (score / char_cfg.TARGET_PLATE_LENGTH))))
            submission.append(f"{track_ids[i]},{text};{conf:.4f}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write("\n".join(submission))
    print(f"Wrote {len(submission)} predictions to {args.output}")
    print(f"Ensemble paths: {path_counts}")


if __name__ == "__main__":
    main()

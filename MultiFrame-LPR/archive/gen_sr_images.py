#!/usr/bin/env python3
"""Run LP-Diff inference and save SR images to disk.

For every track folder found under --data-root, generates one ``sr-{idx:03d}.png``
file per LR frame (idx matching the original frame ordering). The OCR pipeline
can then load these in place of ``lr-*.png`` to train/evaluate on the
super-resolved frames (see ``--use-sr-cache`` on ``train.py``).

DDIM sampling defaults to 50 steps and eta=0 (deterministic).

Usage:
    python gen_sr_images.py --ckpt results/lpdiff/lpdiff_v1_best.pth \\
        --data-root data/LRLPR-26-5opEvJTW/train
    python gen_sr_images.py --ckpt results/lpdiff/lpdiff_v1_best.pth \\
        --data-root data/LRLPR-26-5opEvJTW/test --ddim-steps 30
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.lpdiff_config import LPDiffConfig
from src.data.lpdiff_dataset import (
    LPDiffDataset, _image_files, _read_image, _resize_to, _to_minus1_1,
    lpdiff_collate,
)
from src.models.lpdiff import LPDiffNet


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LP-Diff inference: cache SR PNGs to disk.")
    p.add_argument("--ckpt", type=str, required=True, help="LP-Diff checkpoint (.pth).")
    p.add_argument("--data-root", type=str, required=True,
                   help="Directory containing track_* folders to process.")
    p.add_argument("--val-split-file", type=str,
                   default="data/LRLPR-26-5opEvJTW/val_tracks.json")
    p.add_argument("--ddim-steps", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--use-ema", action="store_true",
                   help="Load the EMA weights if present in the checkpoint.")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-generate even if sr-*.png already exists.")
    p.add_argument("--inner-channel", type=int, default=None,
                   help="Override INNER_CHANNEL (must match training).")
    p.add_argument("--out-prefix", type=str, default="sr",
                   help="Filename prefix for cached SR images. Default 'sr'.")
    return p.parse_args()


def build_model(cfg: LPDiffConfig, args: argparse.Namespace) -> Tuple[LPDiffNet, dict]:
    inner_channel = args.inner_channel or cfg.INNER_CHANNEL
    model = LPDiffNet(
        image_size=cfg.HR_HEIGHT,
        in_channel=6, out_channel=3,
        inner_channel=inner_channel,
        norm_groups=cfg.NORM_GROUPS,
        channel_mults=cfg.CHANNEL_MULTS,
        attn_res=cfg.ATTN_RES,
        res_blocks=cfg.RES_BLOCKS,
        dropout=cfg.DROPOUT,
        beta_schedule=cfg.beta_schedule_dict,
    )
    model.configure_for_device(cfg.DEVICE)

    state = torch.load(args.ckpt, map_location=cfg.DEVICE)
    if isinstance(state, dict) and "model" in state:
        weights_key = "ema" if (args.use_ema and state.get("ema") is not None) else "model"
        print(f"Loading {weights_key} weights from {args.ckpt} (iter={state.get('iteration', '?')})")
        model.load_state_dict(state[weights_key], strict=False)
    else:
        print(f"Loading raw state_dict from {args.ckpt}")
        model.load_state_dict(state, strict=False)
    model.eval()
    return model, state if isinstance(state, dict) else {}


def collect_all_tracks(root_dir: str) -> List[dict]:
    """Walk the tree once; each track contributes one 'job' with all LR frames."""
    import glob
    search_path = os.path.join(os.path.abspath(root_dir), "**", "track_*")
    tracks = sorted(
        p for p in glob.glob(search_path, recursive=True) if os.path.isdir(p)
    )
    jobs = []
    for track in tracks:
        lr_paths = _image_files(track, "lr")
        if not lr_paths:
            continue
        jobs.append({"track_path": track, "lr_paths": lr_paths})
    return jobs


def process_track(
    model: LPDiffNet,
    job: dict,
    cfg: LPDiffConfig,
    ddim_steps: int,
    out_prefix: str,
    overwrite: bool,
) -> int:
    """Run LP-Diff once per LR frame in the track and save the SR PNGs.

    Returns the number of frames written.
    """
    track = job["track_path"]
    lr_paths = job["lr_paths"]

    # Skip if all sr outputs already exist.
    if not overwrite:
        all_done = all(
            os.path.exists(os.path.join(track, f"{out_prefix}-{i+1:03d}.png"))
            for i in range(len(lr_paths))
        )
        if all_done:
            return 0

    # MTA needs three frames. We follow cfg.FRAME_INDICES to choose the context,
    # and run the diffusion once per *target* frame. The "main" LR frame is the
    # target (placed as LR1, with LR2/LR3 = two neighbouring frames for context).
    n = len(lr_paths)
    written = 0
    device = cfg.DEVICE
    for i, target_path in enumerate(lr_paths):
        out_path = os.path.join(track, f"{out_prefix}-{i+1:03d}.png")
        if (not overwrite) and os.path.exists(out_path):
            continue

        # Context frames: pick the two frames temporally closest to i.
        if n >= 3:
            ctx_indices = sorted({max(0, i - 1), min(n - 1, i + 1)})
            while len(ctx_indices) < 2:
                ctx_indices.append(min(n - 1, ctx_indices[-1] + 1))
        else:
            ctx_indices = [0, min(1, n - 1)]
        ctx1_path = lr_paths[ctx_indices[0]]
        ctx2_path = lr_paths[ctx_indices[-1]]

        def _load(path: str) -> torch.Tensor:
            img = _read_image(path)
            if img is None:
                img = np.zeros((cfg.HR_HEIGHT, cfg.HR_WIDTH, 3), dtype=np.uint8)
            img = _resize_to(img, cfg.HR_HEIGHT, cfg.HR_WIDTH)
            return _to_minus1_1(img).unsqueeze(0)

        lr1 = _load(target_path).to(device)
        lr2 = _load(ctx1_path).to(device)
        lr3 = _load(ctx2_path).to(device)

        sr = model.infer(lr1, lr2, lr3, sampler="ddim", num_steps=ddim_steps)
        sr_01 = LPDiffNet.from_diffusion_domain(sr)
        sr_np = (sr_01.squeeze(0).clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
        sr_bgr = cv2.cvtColor(sr_np, cv2.COLOR_RGB2BGR)
        cv2.imwrite(out_path, sr_bgr)
        written += 1
    return written


def main() -> None:
    args = parse_args()
    cfg = LPDiffConfig()

    model, _ = build_model(cfg, args)
    print(f"LP-Diff loaded. DDIM steps = {args.ddim_steps}. Sampling on {cfg.DEVICE}.")

    jobs = collect_all_tracks(args.data_root)
    print(f"Found {len(jobs)} tracks under {args.data_root}.")
    if not jobs:
        print("Nothing to do.")
        return

    total_written = 0
    pbar = tqdm(jobs, desc="LP-Diff inference")
    for job in pbar:
        try:
            written = process_track(
                model, job, cfg, args.ddim_steps, args.out_prefix, args.overwrite,
            )
        except RuntimeError as exc:
            print(f"  WARNING: track {job['track_path']} failed: {exc}")
            continue
        total_written += written
        pbar.set_postfix({"written": total_written})
    print(f"Done. Wrote {total_written} SR images.")


if __name__ == "__main__":
    main()

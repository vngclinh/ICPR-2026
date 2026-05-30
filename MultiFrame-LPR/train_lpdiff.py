#!/usr/bin/env python3
"""Train the LP-Diff super-resolution model standalone.

Phase 1 of the LP-Diff workflow (Phase 2 is OCR training on cached SR images
produced by ``gen_sr_images.py``).

Usage:
    python train_lpdiff.py
    python train_lpdiff.py --iterations 30000 --batch-size 2
    python train_lpdiff.py --resume results/lpdiff/lpdiff_v1_step_50000.pth
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from copy import deepcopy
from typing import Iterator

import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.lpdiff_config import LPDiffConfig
from src.data.lpdiff_dataset import LPDiffDataset, lpdiff_collate
from src.models.lpdiff import LPDiffNet
from src.utils.common import seed_everything


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train LP-Diff (residual diffusion SR).")
    p.add_argument("--data-root", type=str, default=None)
    p.add_argument("--val-split-file", type=str, default=None)
    p.add_argument("--iterations", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--experiment-name", type=str, default=None)
    p.add_argument("--resume", type=str, default=None,
                   help="Path to a *_gen.pth checkpoint to resume from.")
    p.add_argument("--no-amp", action="store_true", help="Disable mixed precision.")
    p.add_argument("--amp-dtype", type=str, default="bf16", choices=["fp16", "bf16"],
                   help="AMP dtype. bf16 is more stable for diffusion (no overflow); "
                        "fp16 is slightly faster but can produce NaN.")
    p.add_argument("--dry-run", action="store_true",
                   help="Build everything and run one forward/backward, then exit.")
    p.add_argument("--print-freq", type=int, default=None)
    p.add_argument("--save-freq", type=int, default=None)
    p.add_argument("--val-freq", type=int, default=None)
    p.add_argument("--inner-channel", type=int, default=None)
    return p.parse_args()


def apply_overrides(cfg: LPDiffConfig, args: argparse.Namespace) -> None:
    mapping = {
        "data_root": "DATA_ROOT",
        "val_split_file": "VAL_SPLIT_FILE",
        "iterations": "N_ITERATIONS",
        "batch_size": "BATCH_SIZE",
        "lr": "LEARNING_RATE",
        "num_workers": "NUM_WORKERS",
        "output_dir": "OUTPUT_DIR",
        "experiment_name": "EXPERIMENT_NAME",
        "print_freq": "PRINT_FREQ",
        "save_freq": "SAVE_FREQ",
        "val_freq": "VAL_FREQ",
        "inner_channel": "INNER_CHANNEL",
    }
    for arg_key, cfg_key in mapping.items():
        v = getattr(args, arg_key, None)
        if v is not None:
            setattr(cfg, cfg_key, v)
    if args.no_amp:
        cfg.USE_AMP = False


def cycle(loader: DataLoader) -> Iterator[dict]:
    while True:
        for batch in loader:
            yield batch


def ema_update(ema_model: LPDiffNet, model: LPDiffNet, decay: float) -> None:
    with torch.no_grad():
        for ema_p, p in zip(ema_model.parameters(), model.parameters()):
            ema_p.mul_(decay).add_(p.detach(), alpha=1.0 - decay)
        # Buffers (BatchNorm running stats, diffusion alphas, etc.) should track
        # the live model exactly so the EMA's noise schedule stays valid.
        for ema_b, b in zip(ema_model.buffers(), model.buffers()):
            ema_b.copy_(b)


@torch.no_grad()
def validate_loss(model: LPDiffNet, val_loader: DataLoader, device: torch.device,
                  max_batches: int = 50) -> float:
    """Approximate validation loss = the same training objective on val data."""
    model.eval()
    total = 0.0
    count = 0
    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break
        lr1 = batch["LR1"].to(device, non_blocking=True)
        lr2 = batch["LR2"].to(device, non_blocking=True)
        lr3 = batch["LR3"].to(device, non_blocking=True)
        hr = batch["HR"].to(device, non_blocking=True)
        loss = model.training_loss(lr1, lr2, lr3, hr)
        b, c, h, w = hr.shape
        total += float(loss.item()) / max(1, b * c * h * w)
        count += 1
    model.train()
    return total / max(1, count)


def main() -> None:
    args = parse_args()
    cfg = LPDiffConfig()
    apply_overrides(cfg, args)
    seed_everything(cfg.SEED)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    device = cfg.DEVICE
    print(f"LP-Diff training: device={device}, batch_size={cfg.BATCH_SIZE}, "
          f"iterations={cfg.N_ITERATIONS}, inner_channel={cfg.INNER_CHANNEL}")

    # ---- Data ----
    train_ds = LPDiffDataset(
        root_dir=cfg.DATA_ROOT, mode="train",
        hr_height=cfg.HR_HEIGHT, hr_width=cfg.HR_WIDTH,
        frame_indices=cfg.FRAME_INDICES, require_hr=True,
        val_split_file=cfg.VAL_SPLIT_FILE, seed=cfg.SEED,
    )
    val_ds = LPDiffDataset(
        root_dir=cfg.DATA_ROOT, mode="val",
        hr_height=cfg.HR_HEIGHT, hr_width=cfg.HR_WIDTH,
        frame_indices=cfg.FRAME_INDICES, require_hr=True,
        val_split_file=cfg.VAL_SPLIT_FILE, seed=cfg.SEED,
    )
    if len(train_ds) == 0:
        print(f"ERROR: No training tracks with HR found under {cfg.DATA_ROOT}.")
        sys.exit(1)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
        num_workers=cfg.NUM_WORKERS, collate_fn=lpdiff_collate,
        pin_memory=(device.type == "cuda"), drop_last=True,
        persistent_workers=cfg.NUM_WORKERS > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
        num_workers=0, collate_fn=lpdiff_collate, pin_memory=False,
    ) if len(val_ds) > 0 else None

    # ---- Model ----
    model = LPDiffNet(
        image_size=cfg.HR_HEIGHT,
        in_channel=6, out_channel=3,
        inner_channel=cfg.INNER_CHANNEL,
        norm_groups=cfg.NORM_GROUPS,
        channel_mults=cfg.CHANNEL_MULTS,
        attn_res=cfg.ATTN_RES,
        res_blocks=cfg.RES_BLOCKS,
        dropout=cfg.DROPOUT,
        beta_schedule=cfg.beta_schedule_dict,
    )
    model.configure_for_device(device)
    model.train()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"LP-Diff parameters: {n_params:,} (~{n_params/1e6:.2f}M)")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
    # bf16 has fp32-equivalent dynamic range -> no GradScaler needed.
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    use_grad_scaler = (cfg.USE_AMP and device.type == "cuda" and amp_dtype == torch.float16)
    scaler = GradScaler("cuda", enabled=use_grad_scaler)
    print(f"AMP: enabled={cfg.USE_AMP}, dtype={args.amp_dtype}, grad_scaler={use_grad_scaler}")

    ema_model = None
    if cfg.USE_EMA:
        ema_model = deepcopy(model)
        for p in ema_model.parameters():
            p.requires_grad = False
        ema_model.eval()

    start_iter = 0
    if args.resume:
        if not os.path.exists(args.resume):
            print(f"ERROR: resume checkpoint not found: {args.resume}")
            sys.exit(1)
        print(f"Resuming from {args.resume}")
        state = torch.load(args.resume, map_location=device)
        if "model" in state:
            model.load_state_dict(state["model"], strict=False)
            if ema_model is not None and "ema" in state:
                ema_model.load_state_dict(state["ema"], strict=False)
            optimizer.load_state_dict(state["optimizer"])
            start_iter = int(state.get("iteration", 0))
        else:
            model.load_state_dict(state, strict=False)
        print(f"Resumed at iteration {start_iter}.")

    # ---- Dry run ----
    if args.dry_run:
        batch = next(iter(train_loader))
        print(f"Batch shapes: LR1={tuple(batch['LR1'].shape)} HR={tuple(batch['HR'].shape)}")
        lr1 = batch["LR1"].to(device); lr2 = batch["LR2"].to(device)
        lr3 = batch["LR3"].to(device); hr = batch["HR"].to(device)
        loss = model.training_loss(lr1, lr2, lr3, hr)
        b, c, h, w = hr.shape
        norm_loss = loss / (b * c * h * w)
        norm_loss.backward()
        print(f"Dry-run OK: loss={float(norm_loss.item()):.4f}")
        return

    # ---- Training loop ----
    data_iter = cycle(train_loader)
    print(f"Starting training from iter {start_iter} -> {cfg.N_ITERATIONS}")
    running_loss = 0.0
    running_count = 0
    log_t0 = time.time()
    best_val = math.inf

    nan_skips = 0
    for it in range(start_iter, cfg.N_ITERATIONS):
        batch = next(data_iter)
        lr1 = batch["LR1"].to(device, non_blocking=True)
        lr2 = batch["LR2"].to(device, non_blocking=True)
        lr3 = batch["LR3"].to(device, non_blocking=True)
        hr = batch["HR"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(
            device_type=device.type,
            enabled=cfg.USE_AMP and device.type == "cuda",
            dtype=amp_dtype,
        ):
            loss = model.training_loss(lr1, lr2, lr3, hr)
            b, c, h, w = hr.shape
            loss = loss / (b * c * h * w)  # match upstream normalization

        # NaN/Inf guard: if the forward produced a non-finite loss, skip the
        # backward+step entirely so we don't pollute model parameters with NaN.
        # GradScaler already skips when *gradients* are non-finite, but not
        # when the loss itself is — that case can leak through to parameters
        # under aggressive AMP scaling.
        if not torch.isfinite(loss):
            nan_skips += 1
            if nan_skips % 10 == 1:
                print(f"  WARNING: non-finite loss at iter {it+1} (skipped {nan_skips} so far).")
            continue

        if use_grad_scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
            optimizer.step()

        # Verify no parameter became NaN (rare, but if it does we want to know
        # immediately rather than 1000 iters later when validation explodes).
        if (it + 1) % cfg.PRINT_FREQ == 0:
            bad = False
            for p in model.parameters():
                if not torch.isfinite(p).all():
                    bad = True
                    break
            if bad:
                print(f"  FATAL: parameters became NaN at iter {it+1}. Stopping.")
                print(f"  Last clean checkpoint should be {cfg.EXPERIMENT_NAME}_step_"
                      f"{((it+1)//cfg.SAVE_FREQ - 1) * cfg.SAVE_FREQ}.pth")
                sys.exit(2)

        if ema_model is not None and it >= 1000:
            ema_update(ema_model, model, cfg.EMA_DECAY)

        running_loss += float(loss.detach().item())
        running_count += 1

        if (it + 1) % cfg.PRINT_FREQ == 0:
            elapsed = time.time() - log_t0
            avg = running_loss / max(1, running_count)
            ips = running_count / max(1e-6, elapsed)
            print(f"[iter {it+1:>7d}/{cfg.N_ITERATIONS}] "
                  f"loss={avg:.4f} | it/s={ips:.2f}")
            running_loss = 0.0
            running_count = 0
            log_t0 = time.time()

        if (it + 1) % cfg.SAVE_FREQ == 0 or (it + 1) == cfg.N_ITERATIONS:
            ckpt = {
                "iteration": it + 1,
                "model": model.state_dict(),
                "ema": ema_model.state_dict() if ema_model is not None else None,
                "optimizer": optimizer.state_dict(),
                "config": cfg.__dict__,
            }
            path = os.path.join(cfg.OUTPUT_DIR, f"{cfg.EXPERIMENT_NAME}_step_{it+1}.pth")
            torch.save(ckpt, path)
            print(f"  Saved checkpoint: {path}")

        if val_loader is not None and ((it + 1) % cfg.VAL_FREQ == 0):
            val_loss = validate_loss(model, val_loader, device, max_batches=50)
            print(f"  [val @ iter {it+1}] loss={val_loss:.4f}")
            if val_loss < best_val:
                best_val = val_loss
                best_ckpt = {
                    "iteration": it + 1,
                    "model": model.state_dict(),
                    "ema": ema_model.state_dict() if ema_model is not None else None,
                    "val_loss": val_loss,
                    "config": cfg.__dict__,
                }
                best_path = os.path.join(cfg.OUTPUT_DIR, f"{cfg.EXPERIMENT_NAME}_best.pth")
                torch.save(best_ckpt, best_path)
                print(f"  New best val loss = {val_loss:.4f}. Saved {best_path}")

    print("Training complete.")


if __name__ == "__main__":
    main()

"""Evaluate a trained ICPR 2026 variant on the val split + write test submission.

Usage:
    python eval_icpr2026.py --variant v1 --ckpt results/icpr2026_v1_best.pth
    python eval_icpr2026.py --variant v1 --ckpt results/icpr2026_v1_best.pth --no-format-decode
"""
from __future__ import annotations

import argparse
import os
from typing import List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs.icpr2026_variants import build_config
from src.data.dataset import MultiFrameDataset
from src.inference import format_constrained_decode
from src.models.lpr_variants import VariantConfig, build_variant
from src.utils.postprocess import decode_with_confidence


def _edit_distance(s: str, t: str) -> int:
    if s == t:
        return 0
    if not s:
        return len(t)
    if not t:
        return len(s)
    prev = list(range(len(t) + 1))
    for i, ca in enumerate(s, start=1):
        cur = [i]
        for j, cb in enumerate(t, start=1):
            cur.append(min(cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _build_model(name: str, ckpt_path: str, device: torch.device):
    cfg = build_config(name)
    state = torch.load(ckpt_path, map_location=device)
    saved_cfg = state.get("config", {}) if isinstance(state, dict) else {}
    stn_mode = saved_cfg.get("STN_MODE", cfg.STN_MODE)
    vc = VariantConfig(
        num_classes=cfg.NUM_CLASSES, num_frames=cfg.NUM_FRAMES, use_stn=cfg.USE_STN,
        stn_mode=stn_mode, stn_tps_x=cfg.STN_TPS_X, stn_tps_y=cfg.STN_TPS_Y,
        d_model=cfg.D_MODEL, nhead=cfg.NHEAD, encoder_layers=cfg.ENCODER_LAYERS,
        encoder_ff=cfg.ENCODER_FF, encoder_dropout=cfg.ENCODER_DROPOUT,
        aux_tap_layer=cfg.AUX_TAP_LAYER, decoder_layers=cfg.DECODER_LAYERS,
        decoder_ff=cfg.DECODER_FF, decoder_dropout=cfg.DECODER_DROPOUT,
        decoder_num_queries=cfg.DECODER_NUM_QUERIES, fusion_per_position=cfg.FUSION_PER_POSITION,
    )
    model = build_variant(name, vc).to(device).eval()
    sd = state.get("model", state) if isinstance(state, dict) else state
    model.load_state_dict(sd, strict=False)
    return cfg, model


@torch.no_grad()
def evaluate_labeled(cfg, model, device, use_format_decode: bool, root: str, mode: str = "val") -> dict:
    val_set = MultiFrameDataset(
        root_dir=root, mode=mode, split_ratio=cfg.SPLIT_RATIO,
        img_height=cfg.IMG_HEIGHT, img_width=cfg.IMG_WIDTH, char2idx=cfg.CHAR2IDX,
        val_split_file=cfg.VAL_SPLIT_FILE, seed=cfg.SEED,
    )
    loader = DataLoader(val_set, batch_size=32, shuffle=False, num_workers=2,
                        collate_fn=MultiFrameDataset.collate_fn)
    correct = 0
    total = 0
    edits = 0
    chars = 0
    for batch in tqdm(loader, desc="Val eval"):
        images, _, _, labels_text, _, _, _ = batch
        images = images.to(device)
        out = model(images)
        log_probs = out["log_probs"]
        if use_format_decode:
            for i in range(images.size(0)):
                text, _ = format_constrained_decode(
                    log_probs[i].detach().cpu(), cfg.CHAR2IDX, cfg.IDX2CHAR,
                    target_length=cfg.TARGET_PLATE_LENGTH,
                )
                gt = labels_text[i]
                correct += int(text == gt)
                edits += _edit_distance(text, gt)
                chars += len(gt)
                total += 1
        else:
            decoded = decode_with_confidence(log_probs, cfg.IDX2CHAR)
            for i, (text, _) in enumerate(decoded):
                gt = labels_text[i]
                correct += int(text == gt)
                edits += _edit_distance(text, gt)
                chars += len(gt)
                total += 1
    return {"acc": correct / total * 100, "cer": edits / max(1, chars) * 100, "n": total}


@torch.no_grad()
def predict_test(cfg, model, device, use_format_decode: bool, output: str) -> int:
    test_set = MultiFrameDataset(
        root_dir=cfg.TEST_DATA_ROOT, mode="test", is_test=True,
        img_height=cfg.IMG_HEIGHT, img_width=cfg.IMG_WIDTH, char2idx=cfg.CHAR2IDX,
        val_split_file=cfg.VAL_SPLIT_FILE, seed=cfg.SEED,
    )
    loader = DataLoader(test_set, batch_size=32, shuffle=False, num_workers=2,
                        collate_fn=MultiFrameDataset.collate_fn)
    lines: List[str] = []
    for batch in tqdm(loader, desc="Test predict"):
        images, _, _, _, track_ids, _, _ = batch
        images = images.to(device)
        out = model(images)
        log_probs = out["log_probs"]
        if use_format_decode:
            for i in range(images.size(0)):
                text, score = format_constrained_decode(
                    log_probs[i].detach().cpu(), cfg.CHAR2IDX, cfg.IDX2CHAR,
                    target_length=cfg.TARGET_PLATE_LENGTH,
                )
                conf = float(min(1.0, max(0.0, score / cfg.TARGET_PLATE_LENGTH + 1.0)))
                lines.append(f"{track_ids[i]},{text};{conf:.4f}")
        else:
            decoded = decode_with_confidence(log_probs, cfg.IDX2CHAR)
            for i, (text, conf) in enumerate(decoded):
                lines.append(f"{track_ids[i]},{text};{conf:.4f}")
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return len(lines)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=["v1", "v2", "v3", "v4", "v5"], required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--no-format-decode", action="store_true",
                   help="Disable format-constrained decoding (use CTC greedy)")
    p.add_argument("--skip-val", action="store_true")
    p.add_argument("--skip-test", action="store_true")
    p.add_argument("--eval-test-labeled", action="store_true",
                   help="Compute accuracy on the labelled test set (3000 tracks).")
    p.add_argument("--output", default=None, help="Test submission path")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg, model = _build_model(args.variant, args.ckpt, device)
    use_format = not args.no_format_decode

    if not args.skip_val:
        metrics = evaluate_labeled(cfg, model, device, use_format, cfg.DATA_ROOT, mode="val")
        tag = "format-decode" if use_format else "greedy"
        print(f"VAL ({tag}): acc={metrics['acc']:.2f}% cer={metrics['cer']:.2f}% n={metrics['n']}")

    if args.eval_test_labeled:
        metrics = evaluate_labeled(cfg, model, device, use_format, cfg.TEST_DATA_ROOT, mode="test")
        tag = "format-decode" if use_format else "greedy"
        print(f"TEST_LABELED ({tag}): acc={metrics['acc']:.2f}% cer={metrics['cer']:.2f}% n={metrics['n']}")

    if not args.skip_test:
        out = args.output or f"results/submission_{args.variant}_{'fmt' if use_format else 'greedy'}.txt"
        n = predict_test(cfg, model, device, use_format, out)
        print(f"TEST: wrote {n} predictions to {out}")


if __name__ == "__main__":
    main()

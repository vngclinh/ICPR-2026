"""Ensemble evaluation on the val split (Scenario-B).

Usage:
    python eval_icpr2026_ensemble.py \\
        --ckpt v1=results/icpr2026_v1_best.pth \\
        --ckpt v2=results/icpr2026_v2_best.pth \\
        --weights 1.0,1.0
"""
from __future__ import annotations

import argparse
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs.icpr2026_variants import build_config
from src.data.dataset import MultiFrameDataset
from src.inference import ensemble_predictions, format_constrained_decode
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


def _parse_ckpts(raw: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for item in raw:
        name, path = item.split("=", 1)
        out[name.lower()] = path
    return out


def _build_one(name: str, ckpt_path: str, device: torch.device):
    cfg = build_config(name)
    state = torch.load(ckpt_path, map_location=device)
    # Override STN_MODE from the checkpoint's saved config so we build the same
    # architecture (V3/V4 were trained with affine STN, default config is TPS).
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
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", action="append", required=True)
    p.add_argument("--weights", type=str, default=None)
    p.add_argument("--fallback", type=str, default=None)
    p.add_argument("--fallback-threshold", type=float, default=-1.5)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpts = _parse_ckpts(args.ckpt)
    weights = [float(x) for x in args.weights.split(",")] if args.weights else None
    fallback_index = list(ckpts).index(args.fallback.lower()) if args.fallback else 0

    print(f"Loading {len(ckpts)} variants: {list(ckpts)}")
    models: Dict[str, torch.nn.Module] = {}
    char_cfg = None
    for name, path in ckpts.items():
        cfg, model = _build_one(name, path, device)
        models[name] = model
        char_cfg = cfg
    assert char_cfg is not None

    val_set = MultiFrameDataset(
        root_dir=char_cfg.DATA_ROOT, mode="val", split_ratio=char_cfg.SPLIT_RATIO,
        img_height=char_cfg.IMG_HEIGHT, img_width=char_cfg.IMG_WIDTH,
        char2idx=char_cfg.CHAR2IDX, val_split_file=char_cfg.VAL_SPLIT_FILE,
        seed=char_cfg.SEED,
    )
    loader = DataLoader(val_set, batch_size=16, shuffle=False, num_workers=2,
                        collate_fn=MultiFrameDataset.collate_fn)

    # Track per-variant accuracy + ensemble accuracy + per-path counts.
    per_variant_correct = {name: 0 for name in ckpts}
    per_variant_fmt_correct = {name: 0 for name in ckpts}
    ens_correct = 0
    total = 0
    ens_edits = 0
    ens_chars = 0
    path_counts = {"average": 0, "vote": 0, "fallback": 0}

    for batch in tqdm(loader, desc="Ensemble eval"):
        images, _, _, labels_text, _, _, _ = batch
        images = images.to(device)
        per_variant_log_probs = []
        per_variant_greedy = []
        for name in ckpts:
            out = models[name](images)
            lp = out["log_probs"]
            per_variant_log_probs.append(lp)
            per_variant_greedy.append(decode_with_confidence(lp, char_cfg.IDX2CHAR))

        b = images.size(0)
        for i in range(b):
            gt = labels_text[i]
            # Per-variant greedy accuracy
            for name, greedy in zip(ckpts, per_variant_greedy):
                if greedy[i][0] == gt:
                    per_variant_correct[name] += 1
            # Per-variant format-decode accuracy
            for name, lp in zip(ckpts, per_variant_log_probs):
                text_fmt, _ = format_constrained_decode(
                    lp[i].detach().cpu(), char_cfg.CHAR2IDX, char_cfg.IDX2CHAR,
                    target_length=char_cfg.TARGET_PLATE_LENGTH,
                )
                if text_fmt == gt:
                    per_variant_fmt_correct[name] += 1

            per_sample = [v[i].detach().cpu() for v in per_variant_log_probs]
            text, score, debug = ensemble_predictions(
                per_sample,
                char2idx=char_cfg.CHAR2IDX, idx2char=char_cfg.IDX2CHAR,
                weights=weights, fallback_index=fallback_index,
                fallback_threshold=args.fallback_threshold,
                target_length=char_cfg.TARGET_PLATE_LENGTH,
            )
            path_counts[debug["path"]] = path_counts.get(debug["path"], 0) + 1
            if text == gt:
                ens_correct += 1
            ens_edits += _edit_distance(text, gt)
            ens_chars += len(gt)
            total += 1

    print("\n=== Val accuracy breakdown ===")
    for name in ckpts:
        ga = per_variant_correct[name] / total * 100
        fa = per_variant_fmt_correct[name] / total * 100
        print(f"  {name}: greedy={ga:.2f}%  format-decode={fa:.2f}%")
    ens_acc = ens_correct / total * 100
    ens_cer = ens_edits / max(1, ens_chars) * 100
    print(f"  ENSEMBLE: acc={ens_acc:.2f}%  cer={ens_cer:.2f}%  paths={path_counts}")


if __name__ == "__main__":
    main()

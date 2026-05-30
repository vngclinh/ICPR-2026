"""Ensemble evaluation with smarter voting + confidence tiebreak.

Strategy per sample:
1. Get each variant's format-constrained decoded string + score.
2. If >=3 of 4 variants agree → use that string.
3. Else: use the variant with the single highest log-score (confidence).
"""
from __future__ import annotations

import argparse
from collections import Counter
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs.icpr2026_variants import build_config
from src.data.dataset import MultiFrameDataset
from src.inference import format_constrained_decode
from src.models.lpr_variants import VariantConfig, build_variant


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


def _build_one(name: str, ckpt_path: str, device: torch.device):
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


def smart_ensemble(
    per_variant: List[tuple], min_agree: int = 3
) -> tuple:
    """Pick best string from list of (string, score). Returns (string, path)."""
    strings = [s for s, _ in per_variant]
    scores = [sc for _, sc in per_variant]

    counts = Counter(strings)
    winner, freq = counts.most_common(1)[0]
    if freq >= min_agree:
        return winner, f"agree_{freq}"

    # No strong majority → pick highest-confidence single variant.
    best_idx = max(range(len(per_variant)), key=lambda i: scores[i])
    return per_variant[best_idx][0], f"conf_{best_idx}"


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", action="append", required=True)
    p.add_argument("--min-agree", type=int, default=3)
    p.add_argument("--target-length", type=int, default=7)
    p.add_argument("--mode", choices=["val", "test", "test_labeled"], default="val")
    p.add_argument("--output", type=str, default=None)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpts: Dict[str, str] = {}
    for item in args.ckpt:
        name, path = item.split("=", 1)
        ckpts[name.lower()] = path

    print(f"Loading {len(ckpts)} variants: {list(ckpts)}")
    models: Dict[str, torch.nn.Module] = {}
    char_cfg = None
    for name, path in ckpts.items():
        cfg, model = _build_one(name, path, device)
        models[name] = model
        char_cfg = cfg

    if args.mode == "val":
        dataset = MultiFrameDataset(
            root_dir=char_cfg.DATA_ROOT, mode="val", split_ratio=char_cfg.SPLIT_RATIO,
            img_height=char_cfg.IMG_HEIGHT, img_width=char_cfg.IMG_WIDTH,
            char2idx=char_cfg.CHAR2IDX, val_split_file=char_cfg.VAL_SPLIT_FILE,
            seed=char_cfg.SEED,
        )
    elif args.mode == "test_labeled":
        dataset = MultiFrameDataset(
            root_dir=char_cfg.TEST_DATA_ROOT, mode="test",
            img_height=char_cfg.IMG_HEIGHT, img_width=char_cfg.IMG_WIDTH,
            char2idx=char_cfg.CHAR2IDX, val_split_file=char_cfg.VAL_SPLIT_FILE,
            seed=char_cfg.SEED,
        )
    else:
        dataset = MultiFrameDataset(
            root_dir=char_cfg.TEST_DATA_ROOT, mode="test", is_test=True,
            img_height=char_cfg.IMG_HEIGHT, img_width=char_cfg.IMG_WIDTH,
            char2idx=char_cfg.CHAR2IDX, val_split_file=char_cfg.VAL_SPLIT_FILE,
            seed=char_cfg.SEED,
        )
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=2,
                        collate_fn=MultiFrameDataset.collate_fn)

    correct = 0
    total = 0
    edits = 0
    chars = 0
    path_counts: Counter = Counter()
    submission_lines: List[str] = []

    for batch in tqdm(loader, desc="Smart ensemble"):
        images, _, _, labels_text, track_ids, _, _ = batch
        images = images.to(device)
        per_variant_log_probs = []
        for name in ckpts:
            out = models[name](images)
            per_variant_log_probs.append(out["log_probs"])

        b = images.size(0)
        for i in range(b):
            decoded = [
                format_constrained_decode(
                    lp[i].detach().cpu(), char_cfg.CHAR2IDX, char_cfg.IDX2CHAR,
                    target_length=args.target_length,
                )
                for lp in per_variant_log_probs
            ]
            text, path = smart_ensemble(decoded, min_agree=args.min_agree)
            path_counts[path] += 1

            # Confidence: use the winning ensemble's score normalised.
            scores = [sc for _, sc in decoded]
            conf = float(min(1.0, max(0.0, max(scores) / args.target_length + 1.0)))
            submission_lines.append(f"{track_ids[i]},{text};{conf:.4f}")

            if args.mode in ("val", "test_labeled"):
                gt = labels_text[i]
                if text == gt:
                    correct += 1
                edits += _edit_distance(text, gt)
                chars += len(gt)
                total += 1

    if args.mode in ("val", "test_labeled"):
        acc = correct / total * 100
        cer = edits / max(1, chars) * 100
        print(f"\nSMART ENSEMBLE [{args.mode}]: acc={acc:.2f}% cer={cer:.2f}% paths={dict(path_counts)}")
    else:
        out_path = args.output or "results/submission_ensemble_smart.txt"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(submission_lines))
        print(f"\nSMART ENSEMBLE test: wrote {len(submission_lines)} predictions to {out_path}")
        print(f"paths={dict(path_counts)}")


if __name__ == "__main__":
    main()

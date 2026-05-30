"""Logit-level ensemble for the multi-architecture pipeline.

Each variant runs format-constrained decoding on its CTC log-probs; the smart
voter returns the majority string when ``>= min_agree`` variants agree, else
the variant with the highest log-score. Supports val, labelled test, and
unlabelled test (for competition submission).

Usage:
    python eval_multi_arch.py \\
        --ckpt svtr=results/multi_svtr_best.pth \\
        --ckpt new_svtr=results/multi_new_svtr_best.pth \\
        --ckpt restran=results/multi_restran_best.pth \\
        --ckpt crnn=results/multi_crnn_best.pth \\
        --mode test_labeled
"""
import argparse
import os
from collections import Counter
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs.icpr2026_base import ICPR2026Config
from src.data.dataset import MultiFrameDataset
from src.inference import format_constrained_decode


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


def _build_model(name: str, ckpt_path: str, device: torch.device, num_classes: int):
    if name == "svtr":
        from src.models.multi_arch.svtr import SVTROCR
        model = SVTROCR(num_classes=num_classes, max_len=25)
    elif name == "new_svtr":
        from src.models.multi_arch.new_svtr import svtrNew
        model = svtrNew(num_classes=num_classes, use_sr=True)
    elif name == "restran":
        from src.models.multi_arch.restran import ResTranOCR
        model = ResTranOCR(num_classes=num_classes, use_sr=False)
    elif name == "mamba":
        from src.models.multi_arch.mamba import NeuroMambaOCR
        model = NeuroMambaOCR(num_classes=num_classes, use_sr=False)
    elif name == "crnn":
        from src.models.multi_arch.crnn import MultiFrameCRNN
        model = MultiFrameCRNN(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model {name}")
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state, strict=False)
    return model.to(device).eval()


def smart_ensemble(per_variant, min_agree: int = 2):
    """List of (string, score) → pick majority if >= min_agree, else highest score."""
    strings = [s for s, _ in per_variant]
    scores = [sc for _, sc in per_variant]
    counts = Counter(strings)
    winner, freq = counts.most_common(1)[0]
    if freq >= min_agree:
        return winner, f"agree_{freq}"
    best_idx = max(range(len(per_variant)), key=lambda i: scores[i])
    return per_variant[best_idx][0], f"conf_{best_idx}"


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", action="append", required=True)
    p.add_argument("--min-agree", type=int, default=2)
    p.add_argument("--mode", choices=["val", "test_labeled", "test"], default="val")
    p.add_argument("--output", type=str, default=None)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = ICPR2026Config()

    ckpts: Dict[str, str] = {}
    for item in args.ckpt:
        name, path = item.split("=", 1)
        ckpts[name.lower()] = path

    print(f"Loading {len(ckpts)} models: {list(ckpts)}")
    models = {name: _build_model(name, path, device, cfg.NUM_CLASSES) for name, path in ckpts.items()}

    if args.mode == "val":
        dataset = MultiFrameDataset(
            root_dir=cfg.DATA_ROOT, mode="val", split_ratio=cfg.SPLIT_RATIO,
            img_height=cfg.IMG_HEIGHT, img_width=cfg.IMG_WIDTH,
            char2idx=cfg.CHAR2IDX, val_split_file=cfg.VAL_SPLIT_FILE, seed=cfg.SEED,
        )
    elif args.mode == "test_labeled":
        dataset = MultiFrameDataset(
            root_dir=cfg.TEST_DATA_ROOT, mode="test",
            img_height=cfg.IMG_HEIGHT, img_width=cfg.IMG_WIDTH,
            char2idx=cfg.CHAR2IDX, val_split_file=cfg.VAL_SPLIT_FILE, seed=cfg.SEED,
        )
    else:
        dataset = MultiFrameDataset(
            root_dir=cfg.TEST_DATA_ROOT, mode="test", is_test=True,
            img_height=cfg.IMG_HEIGHT, img_width=cfg.IMG_WIDTH,
            char2idx=cfg.CHAR2IDX, val_split_file=cfg.VAL_SPLIT_FILE, seed=cfg.SEED,
        )
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=2,
                        collate_fn=MultiFrameDataset.collate_fn)

    correct = 0; total = 0; edits = 0; chars = 0
    path_counts: Counter = Counter()
    per_var_correct = {n: 0 for n in ckpts}
    submission_lines: List[str] = []

    for batch in tqdm(loader, desc=f"Ensemble {args.mode}"):
        images, _, _, labels_text, track_ids, _, _ = batch
        images = images.to(device)
        per_log_probs = []
        for name in ckpts:
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                if name == "svtr":
                    out = models[name](images)
                else:
                    out = models[name](images, return_sr=False)
            per_log_probs.append(out["ocr_logits"].float())

        b = images.size(0)
        for i in range(b):
            decoded = [
                format_constrained_decode(
                    lp[i].detach().cpu(), cfg.CHAR2IDX, cfg.IDX2CHAR,
                    target_length=cfg.TARGET_PLATE_LENGTH,
                )
                for lp in per_log_probs
            ]
            for j, name in enumerate(ckpts):
                if args.mode != "test" and decoded[j][0] == labels_text[i]:
                    per_var_correct[name] += 1
            text, path = smart_ensemble(decoded, min_agree=args.min_agree)
            path_counts[path] += 1
            scores = [sc for _, sc in decoded]
            conf = float(min(1.0, max(0.0, max(scores) / cfg.TARGET_PLATE_LENGTH + 1.0)))
            submission_lines.append(f"{track_ids[i]},{text};{conf:.4f}")

            if args.mode != "test":
                gt = labels_text[i]
                if text == gt:
                    correct += 1
                edits += _edit_distance(text, gt)
                chars += len(gt)
                total += 1

    if args.mode != "test":
        print("\n=== Per-variant accuracy (format-decode) ===")
        for n in ckpts:
            print(f"  {n}: {per_var_correct[n] / max(1, total) * 100:.2f}%")
        acc = correct / max(1, total) * 100
        cer = edits / max(1, chars) * 100
        print(f"ENSEMBLE [{args.mode}]: acc={acc:.2f}% cer={cer:.2f}% paths={dict(path_counts)}")
    else:
        out_path = args.output or "results/submission_multi_ensemble.txt"
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(submission_lines))
        print(f"Wrote {len(submission_lines)} predictions to {out_path}")
        print(f"paths={dict(path_counts)}")


if __name__ == "__main__":
    main()

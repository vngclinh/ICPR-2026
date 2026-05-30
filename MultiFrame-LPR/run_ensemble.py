#!/usr/bin/env python3
"""Ensemble existing checkpoint predictions for the released test set.

Loads multiple ``test_predictions_*.csv`` files and combines them with three
strategies:
  1. Majority vote on the full plate string.
  2. Confidence-weighted top-pick.
  3. Position-level majority (works because plates are fixed length 7).

The position-level voter is the strongest one in practice — single-character
errors in one checkpoint get outvoted by the other 4-6 checkpoints.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import Counter
from typing import Dict, List, Tuple

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ensemble OCR checkpoint predictions.")
    p.add_argument(
        "--predictions", nargs="+",
        default=[
            "results/test_predictions_restran.csv",
            "results/test_predictions_tta_baseline.csv",
            "results/test_predictions_restran_baseline_ocr_v4_sr.csv",
            "results/test_predictions_restran_finetune.csv",
            "results/test_predictions_restran_sr_v7_rerun.csv",
            "results/test_predictions_baseline_tta8.csv",
            "results/test_predictions_sr_preprocess_baseline.csv",
        ],
        help="Prediction CSVs to ensemble (need columns: track_id, prediction, "
             "confidence, ground_truth, correct).",
    )
    p.add_argument(
        "--output-csv", default="results/test_predictions_ensemble.csv",
        help="Where to save the ensemble predictions.",
    )
    p.add_argument(
        "--strategy", choices=["all", "majority", "confidence", "position"],
        default="all",
        help="Which ensemble to compute. 'all' prints stats for each and saves "
             "the position-level one.",
    )
    return p.parse_args()


def load_predictions(paths: List[str]) -> Tuple[Dict[str, List[dict]], Dict[str, str]]:
    """Return per-track prediction list and gt lookup.

    Skips files that don't have the required columns.
    """
    per_track: Dict[str, List[dict]] = {}
    gt: Dict[str, str] = {}
    for path in paths:
        if not os.path.exists(path):
            print(f"  SKIP (missing): {path}")
            continue
        df = pd.read_csv(path)
        required = {"track_id", "prediction", "ground_truth"}
        if not required.issubset(df.columns):
            print(f"  SKIP (missing cols): {path}")
            continue
        for _, row in df.iterrows():
            tid = str(row["track_id"])
            pred = str(row["prediction"])
            conf = float(row["confidence"]) if "confidence" in row and pd.notna(row["confidence"]) else 0.0
            per_track.setdefault(tid, []).append({"pred": pred, "conf": conf, "src": os.path.basename(path)})
            if tid not in gt:
                gt[tid] = str(row["ground_truth"])
    return per_track, gt


def vote_majority(preds: List[dict]) -> Tuple[str, float]:
    """Most common full-string prediction; ties broken by max confidence."""
    counter = Counter()
    by_pred = {}
    for p in preds:
        counter[p["pred"]] += 1
        by_pred.setdefault(p["pred"], []).append(p["conf"])
    best_pred, best_count = max(counter.items(), key=lambda x: (x[1], max(by_pred[x[0]])))
    avg_conf = sum(by_pred[best_pred]) / len(by_pred[best_pred])
    return best_pred, avg_conf


def vote_confidence(preds: List[dict]) -> Tuple[str, float]:
    """Single highest-confidence prediction."""
    best = max(preds, key=lambda p: p["conf"])
    return best["pred"], best["conf"]


def vote_position(preds: List[dict]) -> Tuple[str, float]:
    """Per-position majority over candidates of the same length.

    Filters predictions to the most-common length, then votes per character
    weighted by the source confidence (so a high-confidence checkpoint wins
    its character even if outvoted by two unconfident ones).
    """
    if not preds:
        return "", 0.0
    # Length filter: pick the most-common length so we have aligned positions
    lengths = Counter(len(p["pred"]) for p in preds)
    target_len = lengths.most_common(1)[0][0]
    candidates = [p for p in preds if len(p["pred"]) == target_len] or preds
    out_chars = []
    for i in range(target_len):
        scores: Dict[str, float] = {}
        for c in candidates:
            ch = c["pred"][i] if i < len(c["pred"]) else ""
            scores[ch] = scores.get(ch, 0.0) + max(0.001, c["conf"])
        if scores:
            out_chars.append(max(scores.items(), key=lambda x: x[1])[0])
    avg_conf = sum(c["conf"] for c in candidates) / max(1, len(candidates))
    return "".join(out_chars), avg_conf


def evaluate(per_track: Dict[str, List[dict]], gt: Dict[str, str], voter) -> Tuple[List[dict], float]:
    rows = []
    correct = 0
    for tid in sorted(per_track):
        pred, conf = voter(per_track[tid])
        ok = int(pred == gt[tid])
        correct += ok
        rows.append({
            "track_id": tid,
            "prediction": pred,
            "confidence": f"{conf:.4f}",
            "ground_truth": gt[tid],
            "correct": ok,
        })
    acc = 100.0 * correct / max(1, len(rows))
    return rows, acc


def main() -> None:
    args = parse_args()
    print(f"Loading {len(args.predictions)} prediction files...")
    per_track, gt = load_predictions(args.predictions)
    n_sources = max((len(v) for v in per_track.values()), default=0)
    print(f"Tracks: {len(per_track)}; sources per track (max): {n_sources}")

    strategies = {
        "majority": vote_majority,
        "confidence": vote_confidence,
        "position": vote_position,
    }

    chosen_rows = None
    chosen_label = "position"
    for name, voter in strategies.items():
        if args.strategy != "all" and args.strategy != name:
            continue
        rows, acc = evaluate(per_track, gt, voter)
        print(f"  [{name:>10s}] Accuracy = {acc:.2f}% ({sum(r['correct'] for r in rows)}/{len(rows)})")
        if name == chosen_label or args.strategy == name:
            chosen_rows = rows

    if chosen_rows is None:
        return
    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["track_id", "prediction", "confidence", "ground_truth", "correct"])
        writer.writeheader()
        for row in chosen_rows:
            writer.writerow(row)
    print(f"Saved {len(chosen_rows)} ensemble predictions to {args.output_csv}")


if __name__ == "__main__":
    main()

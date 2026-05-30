#!/usr/bin/env python3
"""Automate SR fine-tuning/evaluation until a target test accuracy is reached."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Optional


TEST_ACC_RE = re.compile(r"Test Results:.*?Acc:\s*([0-9.]+)%")


@dataclass
class Candidate:
    name: str
    epochs: int
    batch_size: int
    learning_rate: float
    lambda_sr: float
    sr_freeze_epochs: int
    aug_level: str
    sr_feed_hr: bool = False


def default_python() -> str:
    venv_python = Path(".venv") / "Scripts" / "python.exe"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable or "python"


def candidates_for_preset(preset: str) -> List[Candidate]:
    if preset == "quick":
        return [
            Candidate("sr_enhance_l001_e4", 4, 64, 1e-4, 0.01, 1, "light"),
            Candidate("sr_enhance_l000_e4", 4, 64, 8e-5, 0.0, 0, "light"),
        ]

    if preset == "hr":
        return [
            Candidate("sr_hr_l001_e8", 8, 32, 8e-5, 0.01, 1, "light", True),
            Candidate("sr_hr_l000_e8", 8, 32, 8e-5, 0.0, 0, "light", True),
            Candidate("sr_hr_l005_e10", 10, 32, 8e-5, 0.05, 1, "light", True),
        ]

    return [
        Candidate("sr_enhance_l001_e10", 10, 64, 1e-4, 0.01, 1, "light"),
        Candidate("sr_enhance_l005_e10", 10, 64, 1e-4, 0.05, 1, "light"),
        Candidate("sr_enhance_l000_e8", 8, 64, 8e-5, 0.0, 0, "light"),
        Candidate("sr_hr_l001_e8", 8, 32, 8e-5, 0.01, 1, "light", True),
        Candidate("sr_hr_l000_e8", 8, 32, 8e-5, 0.0, 0, "light", True),
        Candidate("sr_enhance_full_l001_e10", 10, 64, 1e-4, 0.01, 1, "full"),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train/evaluate SR variants until target test accuracy is reached."
    )
    parser.add_argument("--target-test-acc", type=float, default=75.0)
    parser.add_argument("--baseline-checkpoint", default="results/restran_best.pth")
    parser.add_argument("--output-dir", default="results/sr_auto_search")
    parser.add_argument("--python", default=default_python())
    parser.add_argument("--preset", choices=["quick", "default", "hr"], default="default")
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--test-data-root", default=None)
    parser.add_argument("--val-split-file", default=None)
    parser.add_argument(
        "--skip-existing-train",
        action="store_true",
        help="If a candidate checkpoint already exists, evaluate it without more training.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned commands without running them.",
    )
    return parser.parse_args()


def run_command(cmd: List[str], log_path: Path, dry_run: bool = False) -> int:
    print(" ".join(cmd))
    if dry_run:
        return 0

    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        log.write("$ " + " ".join(cmd) + "\n\n")
        log.flush()
        process = subprocess.run(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=Path(__file__).resolve().parent,
        )
        elapsed = time.time() - start
        log.write(f"\n[exit_code={process.returncode} elapsed_sec={elapsed:.1f}]\n")
        return process.returncode


def csv_accuracy(path: Path) -> Optional[float]:
    if not path.exists():
        return None
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        total = 0
        correct = 0
        for row in reader:
            if "correct" not in row:
                return None
            total += 1
            correct += int(str(row["correct"]).strip() in {"1", "True", "true"})
    if total == 0:
        return None
    return (correct / total) * 100.0


def log_accuracy(path: Path) -> Optional[float]:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8", errors="replace")
    matches = TEST_ACC_RE.findall(text)
    return float(matches[-1]) if matches else None


def append_summary(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def train_command(args: argparse.Namespace, candidate: Candidate, checkpoint: Path) -> List[str]:
    cmd = [
        args.python,
        "train.py",
        "--experiment-name",
        candidate.name,
        "--output-dir",
        args.output_dir,
        "--init-checkpoint",
        args.baseline_checkpoint,
        "--epochs",
        str(candidate.epochs),
        "--batch-size",
        str(candidate.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--lr",
        str(candidate.learning_rate),
        "--aug-level",
        candidate.aug_level,
        "--use-sr",
        "true",
        "--lambda-sr",
        str(candidate.lambda_sr),
        "--sr-freeze-epochs",
        str(candidate.sr_freeze_epochs),
        "--no-test-eval",
    ]
    if args.seed is not None:
        cmd.extend(["--seed", str(args.seed)])
    if args.data_root:
        cmd.extend(["--data-root", args.data_root])
    if args.test_data_root:
        cmd.extend(["--test-data-root", args.test_data_root])
    if args.val_split_file:
        cmd.extend(["--val-split-file", args.val_split_file])
    if candidate.sr_feed_hr:
        cmd.append("--sr-feed-hr")
    return cmd


def test_command(args: argparse.Namespace, candidate: Candidate, checkpoint: Path, output_csv: Path) -> List[str]:
    cmd = [
        args.python,
        "run_test.py",
        "--checkpoint",
        str(checkpoint),
        "--batch-size",
        str(args.eval_batch_size),
        "--num-workers",
        str(args.num_workers),
        "--use-sr",
        "true",
        "--output-file",
        str(output_csv),
    ]
    if args.test_data_root:
        cmd.extend(["--test-data-root", args.test_data_root])
    if candidate.sr_feed_hr:
        cmd.append("--sr-feed-hr")
    return cmd


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    log_dir = output_dir / "logs"
    summary_csv = output_dir / "summary.csv"
    summary_json = output_dir / "summary.json"

    if not Path(args.baseline_checkpoint).exists():
        print(f"ERROR: baseline checkpoint not found: {args.baseline_checkpoint}")
        return 1

    candidates = candidates_for_preset(args.preset)
    if args.max_candidates is not None:
        candidates = candidates[: args.max_candidates]

    print(f"Target test accuracy: {args.target_test_acc:.2f}%")
    print(f"Baseline checkpoint: {args.baseline_checkpoint}")
    print(f"Output dir: {output_dir}")

    if args.dry_run:
        for candidate in candidates:
            checkpoint = output_dir / f"{candidate.name}_best.pth"
            output_csv = output_dir / f"test_predictions_{candidate.name}.csv"
            print(f"\n=== Candidate: {candidate.name} ===")
            print(json.dumps(asdict(candidate), indent=2))
            print(" ".join(train_command(args, candidate, checkpoint)))
            print(" ".join(test_command(args, candidate, checkpoint, output_csv)))
        return 0

    all_rows: List[dict] = []
    for candidate in candidates:
        checkpoint = output_dir / f"{candidate.name}_best.pth"
        output_csv = output_dir / f"test_predictions_{candidate.name}.csv"
        train_log = log_dir / f"{candidate.name}_train.log"
        test_log = log_dir / f"{candidate.name}_test.log"

        print(f"\n=== Candidate: {candidate.name} ===")
        print(json.dumps(asdict(candidate), indent=2))

        train_rc = 0
        if args.skip_existing_train and checkpoint.exists():
            print(f"Checkpoint exists, skipping training: {checkpoint}")
        else:
            train_rc = run_command(train_command(args, candidate, checkpoint), train_log, args.dry_run)

        test_rc = 0
        test_acc: Optional[float] = None
        if train_rc == 0:
            test_rc = run_command(test_command(args, candidate, checkpoint, output_csv), test_log, args.dry_run)
            test_acc = csv_accuracy(output_csv) or log_accuracy(test_log)

        row = {
            **asdict(candidate),
            "checkpoint": str(checkpoint),
            "prediction_csv": str(output_csv),
            "train_log": str(train_log),
            "test_log": str(test_log),
            "train_rc": train_rc,
            "test_rc": test_rc,
            "test_acc": "" if test_acc is None else f"{test_acc:.4f}",
            "target_met": bool(test_acc is not None and test_acc >= args.target_test_acc),
        }
        all_rows.append(row)
        append_summary(summary_csv, [row])
        summary_json.write_text(json.dumps(all_rows, indent=2), encoding="utf-8")

        if test_acc is None:
            print("Candidate finished without a parseable test accuracy.")
        else:
            print(f"Candidate test accuracy: {test_acc:.2f}%")

        if test_acc is not None and test_acc >= args.target_test_acc:
            print(f"TARGET MET: {candidate.name} reached {test_acc:.2f}%")
            return 0

    print("Target was not reached by the configured candidate list.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

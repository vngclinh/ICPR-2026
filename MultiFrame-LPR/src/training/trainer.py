"""Trainer class encapsulating training, validation, and labelled evaluation."""

import csv
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.common import seed_everything
from src.utils.postprocess import decode_with_confidence


PredictionRow = Tuple[str, str, float, str, bool]


class Trainer:
    """Encapsulates training, validation, and inference logic."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        config,
        idx2char: Dict[int, str],
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.idx2char = idx2char
        self.device = config.DEVICE
        self.use_amp = self.device.type == "cuda"
        seed_everything(config.SEED, benchmark=config.USE_CUDNN_BENCHMARK)

        self.criterion = nn.CTCLoss(blank=0, zero_infinity=True, reduction="mean")
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
        )
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config.LEARNING_RATE,
            steps_per_epoch=len(train_loader),
            epochs=config.EPOCHS,
        )
        self.scaler = GradScaler("cuda", enabled=self.use_amp)

        self.best_acc = -1.0
        self.current_epoch = 0

    def _get_output_path(self, filename: str) -> str:
        output_dir = getattr(self.config, "OUTPUT_DIR", "results")
        os.makedirs(output_dir, exist_ok=True)
        return os.path.join(output_dir, filename)

    def _get_exp_name(self) -> str:
        return getattr(self.config, "EXPERIMENT_NAME", "baseline")

    @staticmethod
    def _edit_distance(source: str, target: str) -> int:
        """Compute Levenshtein edit distance."""
        if source == target:
            return 0
        if not source:
            return len(target)
        if not target:
            return len(source)

        previous = list(range(len(target) + 1))
        for i, source_char in enumerate(source, start=1):
            current = [i]
            for j, target_char in enumerate(target, start=1):
                insert_cost = current[j - 1] + 1
                delete_cost = previous[j] + 1
                replace_cost = previous[j - 1] + (source_char != target_char)
                current.append(min(insert_cost, delete_cost, replace_cost))
            previous = current
        return previous[-1]

    def train_one_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        epoch_loss = 0.0
        pbar = tqdm(self.train_loader, desc=f"Ep {self.current_epoch + 1}/{self.config.EPOCHS}")

        for images, targets, target_lengths, _, _ in pbar:
            images = images.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=self.device.type, enabled=self.use_amp):
                preds = self.model(images)
                preds_permuted = preds.permute(1, 0, 2)
                input_lengths = torch.full(
                    size=(images.size(0),),
                    fill_value=preds.size(1),
                    dtype=torch.long,
                )
                loss = self.criterion(preds_permuted, targets, input_lengths, target_lengths)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.GRAD_CLIP)

            scale_before = self.scaler.get_scale()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self.scaler.get_scale() >= scale_before:
                self.scheduler.step()

            epoch_loss += loss.item()
            pbar.set_postfix({"loss": loss.item(), "lr": self.scheduler.get_last_lr()[0]})

        return epoch_loss / len(self.train_loader)

    def _evaluate_loader(
        self,
        loader: DataLoader,
        collect_submission: bool = False,
    ) -> Tuple[Dict[str, float], List[str], List[PredictionRow]]:
        """Evaluate a labelled loader."""
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        total_edits = 0
        total_chars = 0
        submission_data: List[str] = []
        prediction_rows: List[PredictionRow] = []

        with torch.no_grad():
            for images, targets, target_lengths, labels_text, track_ids in loader:
                images = images.to(self.device)
                targets = targets.to(self.device)
                preds = self.model(images)

                input_lengths = torch.full(
                    (images.size(0),),
                    preds.size(1),
                    dtype=torch.long,
                )
                loss = self.criterion(
                    preds.permute(1, 0, 2),
                    targets,
                    input_lengths,
                    target_lengths,
                )
                total_loss += loss.item()

                decoded_list = decode_with_confidence(preds, self.idx2char)
                for i, (pred_text, conf) in enumerate(decoded_list):
                    gt_text = labels_text[i]
                    track_id = track_ids[i]
                    correct = pred_text == gt_text

                    total_correct += int(correct)
                    total_edits += self._edit_distance(pred_text, gt_text)
                    total_chars += len(gt_text)
                    total_samples += 1

                    if collect_submission:
                        submission_data.append(f"{track_id},{pred_text};{conf:.4f}")
                    prediction_rows.append((track_id, pred_text, conf, gt_text, correct))

        avg_loss = total_loss / len(loader) if len(loader) > 0 else 0.0
        acc = (total_correct / total_samples) * 100 if total_samples else 0.0
        cer = (total_edits / total_chars) * 100 if total_chars else 0.0
        return {"loss": avg_loss, "acc": acc, "cer": cer}, submission_data, prediction_rows

    def validate(self) -> Tuple[Dict[str, float], List[str]]:
        """Run validation and return metrics plus submission-style predictions."""
        if self.val_loader is None:
            return {"loss": 0.0, "acc": 0.0, "cer": 0.0}, []

        metrics, submission_data, _ = self._evaluate_loader(
            self.val_loader,
            collect_submission=True,
        )
        return metrics, submission_data

    def save_submission(self, submission_data: List[str]) -> None:
        """Save validation predictions in the original competition text format."""
        exp_name = self._get_exp_name()
        filename = self._get_output_path(f"submission_{exp_name}.txt")
        with open(filename, "w", encoding="utf-8") as f:
            f.write("\n".join(submission_data))
        print(f"Saved {len(submission_data)} lines to {filename}")

    def save_evaluation_csv(self, rows: List[PredictionRow], filename: str) -> None:
        """Save labelled predictions with ground truth for inspection."""
        output_path = self._get_output_path(filename)
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["track_id", "prediction", "confidence", "ground_truth", "correct"])
            for track_id, pred_text, conf, gt_text, correct in rows:
                writer.writerow([track_id, pred_text, f"{conf:.4f}", gt_text, int(correct)])
        print(f"Saved {len(rows)} labelled predictions to {output_path}")

    def save_model(self, path: str | None = None) -> None:
        """Save model checkpoint with experiment name."""
        if path is None:
            exp_name = self._get_exp_name()
            path = self._get_output_path(f"{exp_name}_best.pth")
        torch.save(self.model.state_dict(), path)

    def fit(self) -> None:
        """Run the full training loop."""
        print(f"TRAINING START | Device: {self.device} | Epochs: {self.config.EPOCHS}")

        for epoch in range(self.config.EPOCHS):
            self.current_epoch = epoch
            avg_train_loss = self.train_one_epoch()

            val_metrics, submission_data = self.validate()
            val_loss = val_metrics["loss"]
            val_acc = val_metrics["acc"]
            val_cer = val_metrics.get("cer", 0.0)
            current_lr = self.scheduler.get_last_lr()[0]

            print(
                f"Epoch {epoch + 1}/{self.config.EPOCHS}: "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc:.2f}% | "
                f"Val CER: {val_cer:.2f}% | "
                f"LR: {current_lr:.2e}"
            )

            if val_acc > self.best_acc:
                self.best_acc = val_acc
                self.save_model()
                exp_name = self._get_exp_name()
                model_path = self._get_output_path(f"{exp_name}_best.pth")
                print(f"  Saved best model: {model_path} ({val_acc:.2f}%)")

                if submission_data:
                    self.save_submission(submission_data)

        if self.val_loader is None:
            self.save_model()
            exp_name = self._get_exp_name()
            model_path = self._get_output_path(f"{exp_name}_best.pth")
            print(f"  Saved final model: {model_path}")

        print(f"\nTraining complete. Best Val Acc: {self.best_acc:.2f}%")

    def evaluate_labeled(
        self,
        loader: DataLoader,
        split_name: str = "Test",
        output_filename: Optional[str] = None,
    ) -> Dict[str, float]:
        """Evaluate a labelled validation/test split and optionally save predictions."""
        print(f"Evaluating labelled {split_name.lower()} data...")
        metrics, _, rows = self._evaluate_loader(loader, collect_submission=False)
        print(
            f"{split_name} Results: "
            f"Loss: {metrics['loss']:.4f} | "
            f"Acc: {metrics['acc']:.2f}% | "
            f"CER: {metrics['cer']:.2f}%"
        )
        if output_filename:
            self.save_evaluation_csv(rows, output_filename)
        return metrics

    def predict(self, loader: DataLoader) -> List[Tuple[str, str, float]]:
        """Run inference on an unlabeled data loader."""
        self.model.eval()
        results: List[Tuple[str, str, float]] = []

        with torch.no_grad():
            for images, _, _, _, track_ids in loader:
                images = images.to(self.device)
                preds = self.model(images)

                decoded_list = decode_with_confidence(preds, self.idx2char)
                for i, (pred_text, conf) in enumerate(decoded_list):
                    results.append((track_ids[i], pred_text, conf))

        return results

    def predict_test(self, test_loader: DataLoader, output_filename: str = "submission_final.txt") -> None:
        """Run inference on unlabeled test data and save competition-format output."""
        print("Running inference on test data...")

        results: List[Tuple[str, str, float]] = []
        self.model.eval()
        with torch.no_grad():
            for images, _, _, _, track_ids in tqdm(test_loader, desc="Test Inference"):
                images = images.to(self.device)
                preds = self.model(images)
                decoded_list = decode_with_confidence(preds, self.idx2char)

                for i, (pred_text, conf) in enumerate(decoded_list):
                    results.append((track_ids[i], pred_text, conf))

        submission_data = [
            f"{track_id},{pred_text};{conf:.4f}" for track_id, pred_text, conf in results
        ]
        output_path = self._get_output_path(output_filename)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(submission_data))

        print(f"Saved {len(submission_data)} predictions to {output_path}")

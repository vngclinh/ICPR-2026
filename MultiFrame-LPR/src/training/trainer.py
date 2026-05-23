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

        # SR-specific configuration (safe defaults when attribute is missing).
        self.use_sr = bool(getattr(config, "USE_SR", False)) and getattr(model, "use_sr", False)
        self.lambda_sr = float(getattr(config, "LAMBDA_SR", 0.0))
        self.sr_freeze_epochs = int(getattr(config, "SR_FREEZE_EPOCHS", 0))
        self.ocr_freeze_epochs = int(getattr(config, "OCR_FREEZE_EPOCHS", 0))
        self.sr_baseline_val_acc = float(getattr(config, "SR_BASELINE_VAL_ACC", 77.2))
        self.sr_warning_epoch = int(getattr(config, "SR_WARNING_EPOCH", 5))
        self.sr_baseline_warned = False
        self.tb_writer = self._build_tensorboard_writer()
        self.wandb = self._get_active_wandb()

        self.criterion = nn.CTCLoss(blank=0, zero_infinity=True, reduction="mean")
        self.sr_criterion = nn.L1Loss(reduction="none")

        sr_lr = getattr(config, "SR_LR", None)
        self.dual_lr = self.use_sr and sr_lr is not None
        if self.dual_lr:
            sr_params = [p for n, p in model.named_parameters() if n.startswith("sr.")]
            ocr_params = [p for n, p in model.named_parameters() if not n.startswith("sr.")]
            self.optimizer = optim.AdamW([
                {"params": ocr_params, "lr": config.LEARNING_RATE},
                {"params": sr_params, "lr": sr_lr},
            ], weight_decay=config.WEIGHT_DECAY)
            print(f"Dual-LR mode — OCR: {config.LEARNING_RATE:.1e}, SR: {sr_lr:.1e}")
        else:
            self.optimizer = optim.AdamW(
                model.parameters(),
                lr=config.LEARNING_RATE,
                weight_decay=config.WEIGHT_DECAY,
            )

        scheduler_type = getattr(config, "SCHEDULER", "onecycle")
        total_steps = len(train_loader) * config.EPOCHS
        if scheduler_type == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps,
                eta_min=1e-7,
            )
        else:
            onecycle_max_lr = (
                [config.LEARNING_RATE, getattr(config, "SR_LR", None)]
                if self.dual_lr
                else config.LEARNING_RATE
            )
            self.scheduler = optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=onecycle_max_lr,
                steps_per_epoch=len(train_loader),
                epochs=config.EPOCHS,
            )
        self.scaler = GradScaler("cuda", enabled=self.use_amp)

        self.best_acc = -1.0
        self.epochs_since_improvement = 0
        self.current_epoch = 0

        if self.use_sr and self.sr_freeze_epochs > 0:
            self.model.set_sr_requires_grad(False)
            print(f"SR frontend FROZEN for first {self.sr_freeze_epochs} epoch(s).")
        if self.use_sr and self.ocr_freeze_epochs > 0:
            self.model.set_ocr_requires_grad(False)
            print(f"OCR backbone FROZEN for first {self.ocr_freeze_epochs} epoch(s) — SR-only training phase.")

    def _build_tensorboard_writer(self):
        log_dir = getattr(self.config, "TENSORBOARD_LOG_DIR", None)
        if not log_dir:
            return None
        try:
            from torch.utils.tensorboard import SummaryWriter

            return SummaryWriter(log_dir=log_dir)
        except Exception as exc:
            print(f"WARNING: TensorBoard logging unavailable: {exc}")
            return None

    @staticmethod
    def _get_active_wandb():
        try:
            import wandb

            return wandb if getattr(wandb, "run", None) is not None else None
        except Exception:
            return None

    def _log_epoch_metrics(self, metrics: Dict[str, float], step: int) -> None:
        if self.wandb is not None:
            self.wandb.log(metrics, step=step)
        if self.tb_writer is not None:
            for name, value in metrics.items():
                self.tb_writer.add_scalar(name, value, step)

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

    def _compute_sr_loss(
        self,
        sr_output: torch.Tensor,
        hr_target: torch.Tensor,
        has_hr: torch.Tensor,
    ) -> torch.Tensor:
        """Masked L1 between SR output and HR target.

        Args:
            sr_output: [B*F, 3, H_hr, W_hr] from the SR module.
            hr_target: [B, F, 3, H_hr, W_hr] from the dataloader in [0, 1].
            has_hr:    [B] bool flags indicating which samples have valid HR.
        """
        b, f, c, h_hr, w_hr = hr_target.shape
        hr_flat = hr_target.view(b * f, c, h_hr, w_hr)

        if sr_output.shape[-2:] != hr_flat.shape[-2:]:
            hr_flat = nn.functional.interpolate(
                hr_flat, size=sr_output.shape[-2:], mode="bilinear", align_corners=False,
            )

        per_pixel = self.sr_criterion(sr_output, hr_flat)  # [B*F, 3, H, W]
        per_sample = per_pixel.mean(dim=(1, 2, 3)).view(b, f).mean(dim=1)  # [B]

        mask = has_hr.to(per_sample.dtype)
        denom = mask.sum().clamp_min(1.0)
        return (per_sample * mask).sum() / denom

    def _maybe_unfreeze_sr(self) -> None:
        if not self.use_sr or self.sr_freeze_epochs <= 0:
            return
        if self.current_epoch == self.sr_freeze_epochs:
            self.model.set_sr_requires_grad(True)
            print(f"Epoch {self.current_epoch + 1}: SR frontend UNFROZEN - joint training.")

    def _maybe_unfreeze_ocr(self) -> None:
        if not self.use_sr or self.ocr_freeze_epochs <= 0:
            return
        if self.current_epoch == self.ocr_freeze_epochs:
            self.model.set_ocr_requires_grad(True)
            print(f"Epoch {self.current_epoch + 1}: OCR backbone UNFROZEN - joint fine-tuning begins.")

    def train_one_epoch(self) -> Tuple[float, float]:
        """Train for one epoch. Returns (avg_total_loss, avg_sr_loss)."""
        self._maybe_unfreeze_sr()
        self._maybe_unfreeze_ocr()
        self.model.train()
        epoch_loss = 0.0
        epoch_sr_loss = 0.0
        sr_loss_batches = 0
        pbar = tqdm(self.train_loader, desc=f"Ep {self.current_epoch + 1}/{self.config.EPOCHS}")

        for batch in pbar:
            images, targets, target_lengths, _, _, hr_frames, has_hr = batch
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            hr_frames = hr_frames.to(self.device, non_blocking=True)
            has_hr = has_hr.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=self.device.type, enabled=self.use_amp):
                if self.use_sr:
                    preds, sr_output = self.model(images, return_sr=True)
                else:
                    preds = self.model(images)
                    sr_output = None

                preds_permuted = preds.permute(1, 0, 2)
                input_lengths = torch.full(
                    size=(images.size(0),),
                    fill_value=preds.size(1),
                    dtype=torch.long,
                )
                ctc_loss = self.criterion(preds_permuted, targets, input_lengths, target_lengths)

                sr_loss = torch.zeros((), device=self.device, dtype=ctc_loss.dtype)
                if self.use_sr and sr_output is not None and bool(has_hr.any()):
                    sr_loss = self._compute_sr_loss(sr_output.to(ctc_loss.dtype), hr_frames, has_hr)

                loss = ctc_loss + self.lambda_sr * sr_loss

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.GRAD_CLIP)

            scale_before = self.scaler.get_scale()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self.scaler.get_scale() >= scale_before:
                self.scheduler.step()

            epoch_loss += loss.item()
            sr_loss_val = float(sr_loss.detach().item()) if self.use_sr else 0.0
            if self.use_sr and bool(has_hr.any()):
                epoch_sr_loss += sr_loss_val
                sr_loss_batches += 1

            last_lrs = self.scheduler.get_last_lr()
            postfix = {
                "loss": loss.item(),
                "ctc": ctc_loss.item(),
                "lr": last_lrs[0],
            }
            if self.dual_lr:
                postfix["sr_lr"] = last_lrs[1]
            if self.use_sr:
                postfix["sr"] = sr_loss_val
            pbar.set_postfix(postfix)

        avg_loss = epoch_loss / max(1, len(self.train_loader))
        avg_sr = epoch_sr_loss / sr_loss_batches if sr_loss_batches else 0.0
        return avg_loss, avg_sr

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
            for batch in loader:
                images, targets, target_lengths, labels_text, track_ids, _, _ = batch
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
        if self.use_sr:
            print(f"SR enabled | lambda_sr={self.lambda_sr} | freeze_epochs={self.sr_freeze_epochs}")

        for epoch in range(self.config.EPOCHS):
            self.current_epoch = epoch
            avg_train_loss, avg_sr_loss = self.train_one_epoch()

            val_metrics, submission_data = self.validate()
            val_loss = val_metrics["loss"]
            val_acc = val_metrics["acc"]
            val_cer = val_metrics.get("cer", 0.0)
            current_lr = self.scheduler.get_last_lr()[0]

            sr_log = f" | SR Loss: {avg_sr_loss:.4f}" if self.use_sr else ""
            print(
                f"Epoch {epoch + 1}/{self.config.EPOCHS}: "
                f"Train Loss: {avg_train_loss:.4f}{sr_log} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc:.2f}% | "
                f"Val CER: {val_cer:.2f}% | "
                f"LR: {current_lr:.2e}"
            )
            log_metrics = {
                "train/loss": avg_train_loss,
                "val/loss": val_loss,
                "val/acc": val_acc,
                "val/cer": val_cer,
                "train/lr": current_lr,
            }
            if self.use_sr:
                log_metrics["train/sr_loss"] = avg_sr_loss
            self._log_epoch_metrics(log_metrics, epoch + 1)

            if val_acc > self.best_acc:
                self.best_acc = val_acc
                self.epochs_since_improvement = 0
                self.save_model()
                exp_name = self._get_exp_name()
                model_path = self._get_output_path(f"{exp_name}_best.pth")
                print(f"  Saved best model: {model_path} ({val_acc:.2f}%)")

                if submission_data:
                    self.save_submission(submission_data)
            else:
                self.epochs_since_improvement += 1
                if self.use_sr and self.epochs_since_improvement >= 5:
                    print(
                        "  WARNING: Val accuracy has not improved for "
                        f"{self.epochs_since_improvement} epochs. Last SR L1 ~= {avg_sr_loss:.4f}. "
                        "Check that lambda_sr is not too high (try 0.01-0.05) and that the "
                        "SR module is producing sensible reconstructions."
                    )

            if (
                self.use_sr
                and self.val_loader is not None
                and not self.sr_baseline_warned
                and epoch + 1 >= self.sr_warning_epoch
                and self.best_acc <= self.sr_baseline_val_acc
            ):
                print(
                    "  WARNING: SR run has not exceeded the baseline validation accuracy "
                    f"({self.sr_baseline_val_acc:.2f}%) after {epoch + 1} epochs. "
                    f"Current best: {self.best_acc:.2f}%; last SR L1: {avg_sr_loss:.4f}. "
                    "Check the SR loss magnitude and HR target availability."
                )
                self.sr_baseline_warned = True

        if self.val_loader is None:
            self.save_model()
            exp_name = self._get_exp_name()
            model_path = self._get_output_path(f"{exp_name}_best.pth")
            print(f"  Saved final model: {model_path}")

        print(f"\nTraining complete. Best Val Acc: {self.best_acc:.2f}%")
        if self.tb_writer is not None:
            self.tb_writer.close()

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
            for batch in loader:
                images, _, _, _, track_ids, _, _ = batch
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
            for batch in tqdm(test_loader, desc="Test Inference"):
                images, _, _, _, track_ids, _, _ = batch
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

"""UniversalTrainer for the multi-architecture pipeline.

Single trainer that drives all four models (SVTR / new SVTR / ResTran / CRNN)
with the same recipe: bf16 AMP autocast, AdamW + OneCycleLR, and the composite
loss shown below. Each model returns a dict with at least ``ocr_logits``; the
trainer adds the auxiliary losses only when the relevant keys are present.

Loss composition:
- CTC main loss (weight 1.0) — every model
- Attention decoder cross-entropy (weight 0.5) — emitted by ``ocr_logits`` +
  ``attn_logits`` heads (only the SVTR variant)
- Super-resolution MSE (weight 0.1) — emitted by ``sr_out`` head (new SVTR,
  ResTran when ``use_sr=True``) against the HR target loaded from the dataset
"""
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.common import seed_everything
from src.utils.postprocess import decode_with_confidence


class UniversalTrainer:
    def __init__(self, model, train_loader, val_loader, config, idx2char):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.idx2char = idx2char
        self.device = config.DEVICE
        seed_everything(config.SEED, benchmark=config.USE_CUDNN_BENCHMARK)

        self.criterion = nn.CTCLoss(blank=0, zero_infinity=True, reduction='mean')

        if self.train_loader is not None:
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
            # bf16 doesn't need a real scaler; we still construct one but disabled
            self.scaler = GradScaler("cuda", enabled=False)
        else:
            self.optimizer = None
            self.scheduler = None
            self.scaler = None

        self.best_acc = 0.0
        self.current_epoch = 0

    def _get_output_path(self, filename: str) -> str:
        output_dir = getattr(self.config, 'OUTPUT_DIR', 'results')
        os.makedirs(output_dir, exist_ok=True)
        return os.path.join(output_dir, filename)

    def _get_exp_name(self) -> str:
        return getattr(self.config, 'EXPERIMENT_NAME', 'baseline')

    def train_one_epoch(self) -> float:
        self.model.train()
        epoch_loss = 0.0
        pbar = tqdm(self.train_loader, desc=f"Ep {self.current_epoch + 1}/{self.config.EPOCHS}")
        for batch in pbar:
            # Our dataset returns 7 items; ignore the last (has_hr flags).
            images, targets, target_lengths, labels_text, track_ids, hr_images, _has_hr = batch
            images = images.to(self.device)
            targets = targets.to(self.device)
            target_lengths = target_lengths.to(self.device)

            self.optimizer.zero_grad(set_to_none=True)

            with autocast('cuda', dtype=torch.bfloat16):
                if self.config.MODEL_TYPE == "svtr":
                    outputs = self.model(images, targets=targets, target_lengths=target_lengths)
                else:
                    outputs = self.model(images, return_sr=True)

                ocr_logits = outputs['ocr_logits']
                preds_permuted = ocr_logits.permute(1, 0, 2)
                input_lengths = torch.full(
                    size=(images.size(0),), fill_value=ocr_logits.size(1),
                    dtype=torch.long, device=self.device,
                )
                loss_ctc = self.criterion(preds_permuted.float(), targets, input_lengths, target_lengths)
                loss = loss_ctc
                postfix_dict = {'loss': f"{loss_ctc.item():.3f}"}

                # Aux SR loss
                if 'sr_out' in outputs and hr_images.numel() > 0:
                    hr_images = hr_images.to(self.device)
                    sr_pred = F.interpolate(
                        outputs['sr_out'].float(),
                        size=(self.config.IMG_HEIGHT, self.config.IMG_WIDTH),
                        mode='bilinear', align_corners=False,
                    )
                    loss_sr = F.mse_loss(
                        sr_pred,
                        hr_images.view(-1, 3, self.config.IMG_HEIGHT, self.config.IMG_WIDTH).float(),
                    )
                    loss = loss + 0.1 * loss_sr
                    postfix_dict['sr'] = f"{loss_sr.item():.3f}"

                # Aux attention decoder loss
                if 'attn_logits' in outputs:
                    attn_logits = outputs['attn_logits']
                    B = images.size(0)
                    max_len = attn_logits.size(1)
                    padded_targets = torch.zeros(B, max_len, dtype=torch.long, device=self.device)
                    start_idx = 0
                    for i, length in enumerate(target_lengths):
                        l = length.item()
                        padded_targets[i, :l] = targets[start_idx:start_idx + l]
                        start_idx += l
                    loss_attn = F.cross_entropy(
                        attn_logits.reshape(-1, attn_logits.size(-1)),
                        padded_targets.reshape(-1),
                        ignore_index=0,
                    )
                    loss = loss + 0.5 * loss_attn
                    postfix_dict['attn'] = f"{loss_attn.item():.3f}"

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), getattr(self.config, 'GRAD_CLIP', 2.0))

            scale_before = self.scaler.get_scale()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self.scaler.get_scale() >= scale_before:
                self.scheduler.step()

            epoch_loss += loss.item()
            postfix_dict['lr'] = f"{self.scheduler.get_last_lr()[0]:.2e}"
            pbar.set_postfix(postfix_dict)

        return epoch_loss / max(1, len(self.train_loader))

    def validate(self):
        if self.val_loader is None:
            return {'loss': 0.0, 'acc': 0.0}, []
        self.model.eval()
        val_loss = 0.0
        total_correct = 0
        total_samples = 0
        submission_data: List[str] = []
        with torch.no_grad():
            for batch in self.val_loader:
                images, targets, target_lengths, labels_text, track_ids, _hr, _has_hr = batch
                images = images.to(self.device)
                targets = targets.to(self.device)
                target_lengths = target_lengths.to(self.device)
                with autocast('cuda', dtype=torch.bfloat16):
                    outputs = self.model(images, return_sr=False)
                    ocr_logits = outputs['ocr_logits']
                    input_lengths = torch.full(
                        (images.size(0),), ocr_logits.size(1), dtype=torch.long, device=self.device,
                    )
                    loss = self.criterion(ocr_logits.permute(1, 0, 2).float(), targets, input_lengths, target_lengths)
                val_loss += loss.item()
                decoded_list = decode_with_confidence(ocr_logits, self.idx2char)
                for i, (pred_text, conf) in enumerate(decoded_list):
                    gt_text = labels_text[i]
                    track_id = track_ids[i]
                    if pred_text == gt_text:
                        total_correct += 1
                    submission_data.append(f"{track_id},{pred_text};{conf:.4f}")
                total_samples += len(labels_text)
        avg_val_loss = val_loss / len(self.val_loader) if len(self.val_loader) > 0 else 0.0
        val_acc = (total_correct / total_samples) * 100 if total_samples > 0 else 0.0
        return {'loss': avg_val_loss, 'acc': val_acc}, submission_data

    def save_submission(self, submission_data, filename=None):
        if filename is None:
            filename = f"submission_{self._get_exp_name()}.txt"
        filepath = self._get_output_path(filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("\n".join(submission_data))
        print(f"📝 Saved {len(submission_data)} lines to {filepath}")

    def save_model(self, path=None):
        if path is None:
            path = self._get_output_path(f"{self._get_exp_name()}_best.pth")
        torch.save(self.model.state_dict(), path)

    def fit(self):
        print(f"🚀 TRAINING START | Device: {self.device} | Epochs: {self.config.EPOCHS}")
        for epoch in range(self.config.EPOCHS):
            self.current_epoch = epoch
            avg_train_loss = self.train_one_epoch()
            if self.val_loader is not None:
                val_metrics, submission_data = self.validate()
                val_loss = val_metrics['loss']
                val_acc = val_metrics['acc']
                current_lr = self.scheduler.get_last_lr()[0]
                print(f"Epoch {epoch + 1}/{self.config.EPOCHS}: "
                      f"Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                      f"Val Acc: {val_acc:.2f}% | LR: {current_lr:.2e}")
                if val_acc > self.best_acc:
                    self.best_acc = val_acc
                    self.save_model()
                    print(f"  ⭐ Saved Best Model ({val_acc:.2f}%)")
                    if submission_data:
                        self.save_submission(submission_data)
            else:
                current_lr = self.scheduler.get_last_lr()[0]
                print(f"Epoch {epoch + 1}/{self.config.EPOCHS}: Train Loss: {avg_train_loss:.4f} | LR: {current_lr:.2e}")

        if self.val_loader is None:
            self.save_model(self._get_output_path(f"{self._get_exp_name()}_final.pth"))
        print(f"\n✅ Training complete! Best Val Acc: {self.best_acc:.2f}%")

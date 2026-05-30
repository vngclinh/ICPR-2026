"""Trainer for the ICPR 2026 LPR pipeline (V1-V4).

Owns the full multi-loss schedule (CTC + auxiliary CTC + center + STN
[+ OHEM-CTC + length penalty]) and the AdamW / OneCycle schedule from
Section 7 of the design doc.

Distinct from ``src/training/trainer.py`` so the legacy ResTran + SR pipeline
keeps working unchanged.
"""
from __future__ import annotations

import csv
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.losses import (
    AuxCTC,
    CenterLoss,
    MainCTC,
    length_penalty_loss,
    ohem_ctc_loss,
    stn_regularization_loss,
)
from src.models.lpr_stn import _build_control_points
from src.utils.common import seed_everything
from src.utils.postprocess import decode_with_confidence


PredictionRow = Tuple[str, str, float, str, bool]


class ICPR2026Trainer:
    """Train one of {V1, V2, V3, V4} with the multi-loss schedule."""

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

        # --- Losses ---
        self.main_ctc = MainCTC(blank=0)
        self.aux_ctc = AuxCTC(blank=0)
        self.center_loss = CenterLoss(
            num_classes=config.NUM_CLASSES,
            feat_dim=config.D_MODEL,
            blank_index=0,
        ).to(self.device)

        # STN identity buffer for TPS regularisation.
        if getattr(config, "USE_STN", True) and getattr(config, "STN_MODE", "tps") == "tps":
            ctrl = _build_control_points(config.STN_TPS_X, config.STN_TPS_Y)
            self.stn_identity = ctrl.to(self.device)
        else:
            self.stn_identity = None

        # --- Optimiser ---
        # Center-loss centres are parameters too; they share the same opt.
        params = list(self.model.parameters()) + list(self.center_loss.parameters())
        self.optimizer = optim.AdamW(
            params,
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
        )

        steps_per_epoch = max(1, len(train_loader))
        if getattr(config, "SCHEDULER", "onecycle") == "onecycle":
            self.scheduler = optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=config.LEARNING_RATE,
                steps_per_epoch=steps_per_epoch,
                epochs=config.EPOCHS,
                pct_start=getattr(config, "PCT_START", 0.1),
                div_factor=getattr(config, "DIV_FACTOR", 25.0),
                final_div_factor=getattr(config, "FINAL_DIV_FACTOR", 1e4),
            )
        else:
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=steps_per_epoch * config.EPOCHS,
                eta_min=1e-7,
            )

        # bf16 doesn't underflow → GradScaler not needed; we still use it as a
        # no-op so the scaler.scale/step/update calls below work uniformly.
        self.scaler = GradScaler("cuda", enabled=False)
        self.best_acc = -1.0
        self.current_epoch = 0

    @staticmethod
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

    def _get_output_path(self, fname: str) -> str:
        out = getattr(self.config, "OUTPUT_DIR", "results")
        os.makedirs(out, exist_ok=True)
        return os.path.join(out, fname)

    def _compose_loss(
        self,
        outputs: dict,
        targets: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Build the total loss from the active components."""
        cfg = self.config
        log_probs = outputs["log_probs"]
        components: Dict[str, float] = {}

        loss = cfg.LAMBDA_CTC * self.main_ctc(log_probs, targets, target_lengths)
        components["ctc"] = float(loss.detach().item())

        if outputs.get("aux_features") is not None and cfg.LAMBDA_AUX_CTC > 0:
            aux_loss = self.aux_ctc(
                outputs["aux_features"], self.model.aux_head, targets, target_lengths
            )
            loss = loss + cfg.LAMBDA_AUX_CTC * aux_loss
            components["aux_ctc"] = float(aux_loss.detach().item())

        if cfg.LAMBDA_CENTER > 0 and outputs.get("features") is not None:
            c_loss = self.center_loss(outputs["features"], log_probs)
            loss = loss + cfg.LAMBDA_CENTER * c_loss
            components["center"] = float(c_loss.detach().item())

        if (
            cfg.LAMBDA_STN > 0
            and outputs.get("stn_params") is not None
            and outputs.get("stn_mode") is not None
        ):
            stn_loss = stn_regularization_loss(
                outputs["stn_params"],
                mode=outputs["stn_mode"],
                identity=self.stn_identity if outputs["stn_mode"] == "tps" else None,
            )
            loss = loss + cfg.LAMBDA_STN * stn_loss
            components["stn"] = float(stn_loss.detach().item())

        if getattr(cfg, "USE_OHEM", False) and cfg.LAMBDA_OHEM > 0:
            ohem = ohem_ctc_loss(
                log_probs, targets, target_lengths, top_k=cfg.OHEM_TOP_K, blank=0
            )
            loss = loss + cfg.LAMBDA_OHEM * ohem
            components["ohem"] = float(ohem.detach().item())

        if getattr(cfg, "USE_LENGTH_PENALTY", False) and cfg.LAMBDA_LENGTH > 0:
            lp = length_penalty_loss(log_probs, target_length=cfg.TARGET_PLATE_LENGTH, blank=0)
            loss = loss + cfg.LAMBDA_LENGTH * lp
            components["len"] = float(lp.detach().item())

        components["total"] = float(loss.detach().item())
        return loss, components

    def train_one_epoch(self) -> Dict[str, float]:
        self.model.train()
        self.center_loss.train()
        running: Dict[str, float] = {}
        n_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Ep {self.current_epoch + 1}/{self.config.EPOCHS}")
        for batch in pbar:
            images, targets, target_lengths, _, _, _, _ = batch
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            target_lengths = target_lengths.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.use_amp):
                outputs = self.model(images)
                loss, components = self._compose_loss(outputs, targets, target_lengths)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.GRAD_CLIP)

            scale_before = self.scaler.get_scale()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self.scaler.get_scale() >= scale_before:
                self.scheduler.step()

            n_batches += 1
            for k, v in components.items():
                running[k] = running.get(k, 0.0) + v
            pbar.set_postfix({k: f"{v:.3f}" for k, v in components.items()})

        return {k: v / max(1, n_batches) for k, v in running.items()}

    @torch.no_grad()
    def validate(self) -> Tuple[Dict[str, float], List[str]]:
        if self.val_loader is None:
            return {"loss": 0.0, "acc": 0.0, "cer": 0.0}, []

        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        total_edits = 0
        total_chars = 0
        submission: List[str] = []

        for batch in self.val_loader:
            images, targets, target_lengths, labels_text, track_ids, _, _ = batch
            images = images.to(self.device)
            targets = targets.to(self.device)
            target_lengths = target_lengths.to(self.device)

            outputs = self.model(images)
            log_probs = outputs["log_probs"]
            loss = self.main_ctc(log_probs, targets, target_lengths)
            total_loss += float(loss.item())

            decoded = decode_with_confidence(log_probs, self.idx2char)
            for i, (pred_text, conf) in enumerate(decoded):
                gt = labels_text[i]
                correct = pred_text == gt
                total_correct += int(correct)
                total_edits += self._edit_distance(pred_text, gt)
                total_chars += len(gt)
                total_samples += 1
                submission.append(f"{track_ids[i]},{pred_text};{conf:.4f}")

        avg_loss = total_loss / max(1, len(self.val_loader))
        acc = total_correct / max(1, total_samples) * 100.0
        cer = total_edits / max(1, total_chars) * 100.0
        return {"loss": avg_loss, "acc": acc, "cer": cer}, submission

    def save_model(self, fname: str | None = None) -> None:
        path = self._get_output_path(fname or f"{self.config.EXPERIMENT_NAME}_best.pth")
        torch.save(
            {
                "model": self.model.state_dict(),
                "center_loss": self.center_loss.state_dict(),
                "config": {k: v for k, v in vars(self.config).items() if not k.startswith("_")},
            },
            path,
        )

    def save_submission(self, lines: List[str]) -> None:
        path = self._get_output_path(f"submission_{self.config.EXPERIMENT_NAME}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def fit(self) -> None:
        print(
            f"[ICPR2026] variant={self.config.VARIANT} epochs={self.config.EPOCHS} "
            f"device={self.device} OHEM={self.config.USE_OHEM} "
            f"LengthPen={self.config.USE_LENGTH_PENALTY}"
        )
        for epoch in range(self.config.EPOCHS):
            self.current_epoch = epoch
            train_metrics = self.train_one_epoch()
            val_metrics, submission = self.validate()
            print(
                f"Ep {epoch + 1}/{self.config.EPOCHS} | "
                f"train={train_metrics.get('total', 0):.4f} "
                f"(ctc={train_metrics.get('ctc', 0):.3f}) | "
                f"val_loss={val_metrics['loss']:.4f} "
                f"acc={val_metrics['acc']:.2f}% cer={val_metrics['cer']:.2f}%"
            )
            if val_metrics["acc"] > self.best_acc:
                self.best_acc = val_metrics["acc"]
                self.save_model()
                if submission:
                    self.save_submission(submission)
                print(f"  saved best ({self.best_acc:.2f}%)")

        if self.val_loader is None:
            self.save_model(f"{self.config.EXPERIMENT_NAME}_final.pth")
        print(f"\nDone. Best val acc: {self.best_acc:.2f}%")

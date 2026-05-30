"""Multi-frame CRNN — STN + CNN backbone + BiLSTM + CTC.

Output is wrapped in a ``{'ocr_logits': ...}`` dict so the universal trainer
treats it the same as the SVTR / ResTran outputs.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.multi_arch.components import AttentionFusion, STNBlock


class CNNBackbone(nn.Module):
    """7-layer CNN backbone with two narrow-stride pools to preserve width as sequence length."""

    def __init__(self, out_channels=512):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d((2, 2), (2, 1), (0, 1)),
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d((2, 2), (2, 1), (0, 1)),
            nn.Conv2d(512, out_channels, 2, 1, 0), nn.BatchNorm2d(out_channels), nn.ReLU(True),
        )

    def forward(self, x):
        return self.features(x)


class MultiFrameCRNN(nn.Module):
    def __init__(self, num_classes: int, hidden_size: int = 256,
                 rnn_dropout: float = 0.25, use_stn: bool = True):
        super().__init__()
        self.cnn_channels = 512
        self.use_stn = use_stn
        if self.use_stn:
            self.stn = STNBlock(in_channels=3)
        self.backbone = CNNBackbone(out_channels=self.cnn_channels)
        self.fusion = AttentionFusion(channels=self.cnn_channels)
        self.rnn = nn.LSTM(
            input_size=self.cnn_channels, hidden_size=hidden_size,
            num_layers=2, bidirectional=True, batch_first=True, dropout=rnn_dropout,
        )
        self.head = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x, return_sr: bool = False):
        b, f, c, h, w = x.size()
        x_flat = x.view(b * f, c, h, w)
        if self.use_stn:
            theta = self.stn(x_flat)
            grid = F.affine_grid(theta, x_flat.size(), align_corners=False)
            x_aligned = F.grid_sample(x_flat, grid, align_corners=False)
        else:
            x_aligned = x_flat
        features = self.backbone(x_aligned)
        fused = self.fusion(features)
        seq_input = fused.squeeze(2).permute(0, 2, 1)
        rnn_out, _ = self.rnn(seq_input)
        ocr_logits = self.head(rnn_out).log_softmax(2)
        return {'ocr_logits': ocr_logits}

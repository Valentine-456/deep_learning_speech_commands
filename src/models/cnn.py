import torch
import torch.nn as nn


class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CNNClassifier(nn.Module):
    """CNN on Mel spectrograms. Input: (batch, 1, n_mels, time_frames)."""

    def __init__(self, num_classes: int, dropout: float = 0.25):
        super().__init__()
        self.features = nn.Sequential(
            _ConvBlock(1, 32),
            _ConvBlock(32, 64),
            _ConvBlock(64, 128),
            _ConvBlock(128, 256),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.features(x)))

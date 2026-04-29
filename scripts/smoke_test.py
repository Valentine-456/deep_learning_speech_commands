#!/usr/bin/env python3
"""Quick sanity check — verifies all three models produce correct output shapes."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models import CNNClassifier, RNNClassifier, TransformerClassifier

NUM_CLASSES = 12
BATCH = 4

cnn = CNNClassifier(num_classes=NUM_CLASSES)
rnn = RNNClassifier(num_classes=NUM_CLASSES)
tfm = TransformerClassifier(num_classes=NUM_CLASSES)

mel = torch.randn(BATCH, 1, 128, 101)    # (batch, 1, n_mels, time_frames)
mfcc = torch.randn(BATCH, 101, 40)       # (batch, time_frames, n_mfcc)

print(f"CNN  output: {cnn(mel).shape}   expected: ({BATCH}, {NUM_CLASSES})")
print(f"RNN  output: {rnn(mfcc).shape}  expected: ({BATCH}, {NUM_CLASSES})")
print(f"TFM  output: {tfm(mfcc).shape}  expected: ({BATCH}, {NUM_CLASSES})")
print()
for name, m in [("CNN", cnn), ("RNN", rnn), ("Transformer", tfm)]:
    n = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"{name:<12} {n:>10,} params")

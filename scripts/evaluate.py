#!/usr/bin/env python3
"""Evaluate a checkpoint and save a confusion matrix plot.

Example
-------
python scripts/evaluate.py results/cnn_mel_lr0.001_bs64_best.pt
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.dataset import SpeechCommandsDataset
from src.models import CNNClassifier, RNNClassifier, TransformerClassifier

SILENCE_LABEL = "silence"
UNKNOWN_LABEL = "unknown"


def load_model(ckpt: dict) -> torch.nn.Module:
    a = ckpt["args"]
    n = len(ckpt["classes"])
    if a["model"] == "cnn":
        m = CNNClassifier(num_classes=n, dropout=a.get("dropout", 0.25))
    elif a["model"] == "rnn":
        m = RNNClassifier(
            num_classes=n,
            rnn_type=a.get("rnn_type", "gru"),
            hidden_size=a.get("hidden_size", 128),
            num_layers=a.get("num_layers", 2),
            dropout=a.get("dropout", 0.3),
            bidirectional=not a.get("unidirectional", False),
        )
    else:
        m = TransformerClassifier(
            num_classes=n,
            d_model=a.get("d_model", 128),
            num_heads=a.get("num_heads", 4),
            num_layers=a.get("num_layers", 2),
            dim_feedforward=a.get("dim_feedforward", 256),
            dropout=a.get("dropout", 0.1),
        )
    m.load_state_dict(ckpt["model_state"])
    return m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--data", default="data/train/train")
    parser.add_argument("--split", choices=["test", "val", "train"], default="test")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    model = load_model(ckpt).to(device)
    model.eval()

    saved = ckpt["args"]
    classes: list = ckpt["classes"]
    core = [c for c in classes if c not in (SILENCE_LABEL, UNKNOWN_LABEL)]

    dataset = SpeechCommandsDataset(
        root=args.data,
        split=args.split,
        feature=saved["feature"],
        classes=core,
        include_silence=SILENCE_LABEL in classes,
        include_unknown=UNKNOWN_LABEL in classes,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=True)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for features, labels in tqdm(loader, desc="eval"):
            preds = model(features.to(device)).argmax(1).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    print(f"\nAccuracy ({args.split}): {acc:.4f}")
    print(classification_report(all_labels, all_preds, target_names=classes))

    cm = confusion_matrix(all_labels, all_preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    n = len(classes)
    fig, ax = plt.subplots(figsize=(max(8, n), max(6, n - 1)))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{Path(args.checkpoint).stem}  {args.split} acc={acc:.4f}")
    plt.tight_layout()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_path = out_dir / f"{Path(args.checkpoint).stem}_{args.split}_cm.png"
    fig.savefig(plot_path, dpi=150)
    print(f"Confusion matrix → {plot_path}")


if __name__ == "__main__":
    main()

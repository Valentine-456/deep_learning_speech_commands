"""Evaluate all trained checkpoints in RESULTS_DIR and save confusion matrices.

Usage
-----
    python scripts/evaluate.py
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import PowerNorm
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.dataset import SpeechCommandsDataset
from src.models import CNNClassifier, LSTMClassifier, TransformerClassifier, VisualTransformerClassifier

CACHE_DIR   = "data/cache"
RESULTS_DIR = Path("results")
SPLIT       = "test"
BATCH_SIZE  = 64
WORKERS     = 4

CLASSES = [
    "yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go",
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "silence", "unknown",
]

# Subset for focused confusion matrix — similar-sounding or problematic classes
CONFUSABLE_CLASSES = [
    "no", "go", "two", "zero",
    "on", "one",
    "up", "off",
    "three", "silence", "unknown",
]


def build_model(cfg: dict, num_classes: int) -> torch.nn.Module:
    m = cfg["model"]
    t = m["type"]
    if t == "cnn":
        return CNNClassifier(
            num_classes=num_classes,
            channels=m.get("channels", [32, 64, 128, 256]),
            dropout=m.get("dropout", 0.25),
        )
    if t == "lstm":
        return LSTMClassifier(
            num_classes=num_classes,
            input_size=m.get("input_size", 40),
            hidden_size=m.get("hidden_size", 128),
            num_layers=m.get("num_layers", 2),
            bidirectional=m.get("bidirectional", True),
            dropout=m.get("dropout", 0.3),
        )
    if t == "transformer":
        return TransformerClassifier(
            num_classes=num_classes,
            input_size=m.get("input_size", 40),
            d_model=m.get("d_model", 128),
            num_heads=m.get("num_heads", 4),
            num_layers=m.get("num_layers", 2),
            dim_feedforward=m.get("dim_feedforward", 256),
            dropout=m.get("dropout", 0.1),
        )
    if t == "visual_transformer":
        return VisualTransformerClassifier(
            num_classes=num_classes,
            patch_h=m.get("patch_h", 16),
            patch_w=m.get("patch_w", 16),
            d_model=m.get("d_model", 128),
            num_heads=m.get("num_heads", 4),
            num_layers=m.get("num_layers", 4),
            dim_feedforward=m.get("dim_feedforward", 256),
            dropout=m.get("dropout", 0.1),
        )
    raise ValueError(f"Unknown model type: {t}")


def evaluate_run(run_dir: Path, device: torch.device) -> None:
    results_json = run_dir / "results.json"
    checkpoint   = run_dir / "best_model.pth"

    if not results_json.exists() or not checkpoint.exists():
        print(f"  Skipping {run_dir.name} — missing results.json or best_model.pth")
        return

    cfg = json.loads(results_json.read_text())["args"]
    feature = cfg["data"]["feature"]

    model = build_model(cfg, len(CLASSES)).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()

    dataset = SpeechCommandsDataset(
        cache_dir=CACHE_DIR,
        split=SPLIT,
        feature=feature,
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=WORKERS, pin_memory=True)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for features, labels in tqdm(loader, desc=run_dir.name, leave=False):
            preds = model(features.to(device)).argmax(1).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    print(f"\n{run_dir.name}  {SPLIT} accuracy: {acc:.4f}")
    print(classification_report(all_labels, all_preds, target_names=CLASSES, zero_division=0))

    cm      = confusion_matrix(all_labels, all_preds, labels=list(range(len(CLASSES))))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    annot = np.empty(cm.shape, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm[i, j]}\n{cm_norm[i, j] * 100:.1f}%"

    n = len(CLASSES)
    fig, ax = plt.subplots(figsize=(n + 2, n))
    sns.heatmap(
        cm_norm, annot=annot, fmt="", cmap="Blues",
        xticklabels=CLASSES, yticklabels=CLASSES, ax=ax,
        norm=PowerNorm(gamma=0.4, vmin=0, vmax=1),
        annot_kws={"size": 7},
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{run_dir.name}  {SPLIT} acc={acc:.4f}")
    plt.tight_layout()

    plot_path = run_dir / f"confusion_matrix_{SPLIT}.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"  Confusion matrix       => {plot_path}")

    # Focused matrix — confusable classes only
    conf_idx    = [CLASSES.index(c) for c in CONFUSABLE_CLASSES if c in CLASSES]
    cm_sub      = cm[np.ix_(conf_idx, conf_idx)]
    cm_norm_sub = cm_norm[np.ix_(conf_idx, conf_idx)]

    annot_sub = np.empty(cm_sub.shape, dtype=object)
    for i in range(cm_sub.shape[0]):
        for j in range(cm_sub.shape[1]):
            annot_sub[i, j] = f"{cm_sub[i, j]}\n{cm_norm_sub[i, j] * 100:.1f}%"

    n_sub = len(conf_idx)
    fig2, ax2 = plt.subplots(figsize=(n_sub + 2, n_sub))
    sns.heatmap(
        cm_norm_sub, annot=annot_sub, fmt="", cmap="Blues",
        xticklabels=CONFUSABLE_CLASSES, yticklabels=CONFUSABLE_CLASSES, ax=ax2,
        norm=PowerNorm(gamma=0.4, vmin=0, vmax=1),
        annot_kws={"size": 9},
    )
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("True")
    ax2.set_title(f"{run_dir.name}  {SPLIT} acc={acc:.4f}  [confusable classes]")
    plt.tight_layout()

    plot_path2 = run_dir / f"confusion_matrix_{SPLIT}_confusable.png"
    fig2.savefig(plot_path2, dpi=150)
    plt.close(fig2)
    print(f"  Confusable matrix      => {plot_path2}")


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  split={SPLIT}\n")

    run_dirs = sorted(p for p in RESULTS_DIR.iterdir() if p.is_dir())
    if not run_dirs:
        print(f"No run directories found in {RESULTS_DIR}/")
        return

    for run_dir in run_dirs:
        evaluate_run(run_dir, device)


if __name__ == "__main__":
    main()

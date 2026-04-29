#!/usr/bin/env python3
"""Train a speech command classifier.

Examples
--------
# CNN on 10 core commands + silence + unknown
python scripts/train.py --model cnn

# GRU on only "yes" vs "no" (quick pipeline check)
python scripts/train.py --model rnn --classes yes no --no-silence --no-unknown

# Transformer, more layers
python scripts/train.py --model transformer --num-layers 4 --num-heads 4 --d-model 128
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.dataset import SpeechCommandsDataset
from src.models import CNNClassifier, RNNClassifier, TransformerClassifier


def build_model(args: argparse.Namespace, num_classes: int) -> nn.Module:
    if args.model == "cnn":
        return CNNClassifier(num_classes=num_classes, dropout=args.dropout)
    if args.model == "rnn":
        return RNNClassifier(
            num_classes=num_classes,
            rnn_type=args.rnn_type,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
            bidirectional=not args.unidirectional,
        )
    return TransformerClassifier(
        num_classes=num_classes,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
    )


def run_epoch(model, loader, optimizer, criterion, device, train: bool):
    model.train(train)
    total_loss = correct = total = 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for features, labels in tqdm(loader, leave=False, desc="train" if train else "val"):
            features, labels = features.to(device), labels.to(device)
            if train:
                optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, labels)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/train/train")
    parser.add_argument("--model", choices=["cnn", "rnn", "transformer"], default="cnn")
    parser.add_argument("--feature", choices=["mel", "mfcc"], default=None,
                        help="Defaults to 'mel' for CNN, 'mfcc' for RNN/Transformer")
    parser.add_argument("--classes", nargs="+", default=None,
                        help="Core command classes (default: 10 main commands)")
    parser.add_argument("--no-silence", action="store_true")
    parser.add_argument("--no-unknown", action="store_true")

    # Training
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out-dir", default="results")

    # RNN
    parser.add_argument("--rnn-type", choices=["lstm", "gru"], default="gru")
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--unidirectional", action="store_true")

    # Transformer
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dim-feedforward", type=int, default=256)

    # Shared RNN + Transformer
    parser.add_argument("--num-layers", type=int, default=2)

    args = parser.parse_args()

    if args.feature is None:
        args.feature = "mel" if args.model == "cnn" else "mfcc"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  model={args.model}  feature={args.feature}")

    ds_kwargs = dict(
        root=args.data,
        feature=args.feature,
        classes=args.classes,
        include_silence=not args.no_silence,
        include_unknown=not args.no_unknown,
    )
    train_ds = SpeechCommandsDataset(split="train", **ds_kwargs)
    val_ds = SpeechCommandsDataset(split="val", **ds_kwargs)

    loader_kwargs = dict(batch_size=args.batch_size, num_workers=args.workers, pin_memory=True)
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)

    model = build_model(args, train_ds.num_classes).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"classes={train_ds.classes}")
    print(f"params={n_params:,}  train={len(train_ds)}  val={len(val_ds)}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    criterion = nn.CrossEntropyLoss()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_name = f"{args.model}_{args.feature}_lr{args.lr}_bs{args.batch_size}"

    best_val_acc = 0.0
    log_rows = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, optimizer, criterion, device, train=True)
        va_loss, va_acc = run_epoch(model, val_loader, optimizer, criterion, device, train=False)
        scheduler.step(va_loss)
        print(
            f"epoch {epoch:3d}/{args.epochs}  "
            f"train loss={tr_loss:.4f} acc={tr_acc:.4f}  "
            f"val loss={va_loss:.4f} acc={va_acc:.4f}  "
            f"{time.time()-t0:.1f}s"
        )
        log_rows.append(
            dict(epoch=epoch, train_loss=tr_loss, train_acc=tr_acc, val_loss=va_loss, val_acc=va_acc)
        )
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            ckpt = out_dir / f"{run_name}_best.pt"
            torch.save(
                dict(model_state=model.state_dict(), args=vars(args), classes=train_ds.classes,
                     val_acc=va_acc, epoch=epoch),
                ckpt,
            )

    csv_path = out_dir / f"{run_name}_log.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=log_rows[0].keys())
        w.writeheader()
        w.writerows(log_rows)

    print(f"\nbest val acc={best_val_acc:.4f}  checkpoint={ckpt}  log={csv_path}")


if __name__ == "__main__":
    main()

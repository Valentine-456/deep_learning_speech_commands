"""Train a speech command classifier from a YAML config.

Usage
-----
    python scripts/train.py configs/cnn_baseline.yaml
    python scripts/train.py configs/lstm_baseline.yaml
"""
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.dataset import SpeechCommandsDataset
from src.models import CNNClassifier, LSTMClassifier, TransformerClassifier, VisualTransformerClassifier
from src.utils import save_json, seed_everything


def build_model(cfg: dict, num_classes: int) -> nn.Module:
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
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/cnn_baseline.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    d  = cfg["data"]
    tr = cfg["training"]

    seed_everything(tr.get("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"config={config_path}  device={device}")

    train_ds = SpeechCommandsDataset(cache_dir=d["cache_dir"], split="train",
                                     feature=d["feature"], augment=d.get("augmentation", False))
    val_ds   = SpeechCommandsDataset(cache_dir=d["cache_dir"], split="val",   feature=d["feature"])
    test_ds  = SpeechCommandsDataset(cache_dir=d["cache_dir"], split="test",  feature=d["feature"])

    loader_kwargs = dict(batch_size=tr["batch_size"], num_workers=tr["workers"], pin_memory=True)
    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_ds,  shuffle=False, **loader_kwargs)

    model = build_model(cfg, train_ds.num_classes).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"classes={train_ds.num_classes}  params={n_params:,}  "
          f"train={len(train_ds):,}  val={len(val_ds):,}  test={len(test_ds):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=tr["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    criterion = nn.CrossEntropyLoss()

    timestamp      = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name       = f"{timestamp}_{Path(config_path).stem}"
    out_dir        = Path(tr["out_dir"]) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = out_dir / "best_model.pth"

    best_val_acc  = 0.0
    best_val_loss = float("inf")
    patience      = tr.get("early_stopping_patience", 7)
    no_improve    = 0
    history       = []

    for epoch in range(1, tr["epochs"] + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, optimizer, criterion, device, train=True)
        va_loss, va_acc = run_epoch(model, val_loader,   optimizer, criterion, device, train=False)
        scheduler.step(va_loss)
        print(
            f"epoch {epoch:3d}/{tr['epochs']}  "
            f"train loss={tr_loss:.4f} acc={tr_acc:.4f}  "
            f"val loss={va_loss:.4f} acc={va_acc:.4f}  "
            f"{time.time() - t0:.1f}s"
        )
        history.append(dict(epoch=epoch,
                            train_loss=tr_loss, train_accuracy=tr_acc,
                            valid_loss=va_loss, valid_accuracy=va_acc))
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(model.state_dict(), best_ckpt_path)

        if va_loss < best_val_loss:
            best_val_loss = va_loss
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping: val loss did not improve for {patience} epochs.")
                break

    print("\nEvaluating on test set")
    model.load_state_dict(torch.load(best_ckpt_path, weights_only=True))
    te_loss, te_acc = run_epoch(model, test_loader, None, criterion, device, train=False)
    print(f"test loss={te_loss:.4f}  acc={te_acc:.4f}")

    save_json(out_dir / "results.json", dict(
        args=cfg,
        device=str(device),
        run_name=run_name,
        best_val_accuracy=best_val_acc,
        history=history,
        test_metrics=dict(loss=te_loss, accuracy=te_acc),
    ))
    print(f"\nSaved => {out_dir}/")


if __name__ == "__main__":
    main()

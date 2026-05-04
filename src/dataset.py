import json
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class SpeechCommandsDataset(Dataset):
    """
    Loads pre-computed mel / mfcc .npy features from a cache directory
    produced by scripts/preprocess.py.

    cache_dir layout:
        <cache_dir>/
            metadata.json
            <split>/mel/<label>/<i>.npy
            <split>/mfcc/<label>/<i>.npy

    Args:
        cache_dir:   root of the pre-processed cache (e.g. "data/cache")
        split:       "train" | "val" | "test"
        feature:     "mel" | "mfcc"
        augment:     apply SpecAugment (mel) or time-mask + noise (mfcc); train only
    """

    def __init__(
        self,
        cache_dir: str,
        split: str = "train",
        feature: str = "mel",
        augment: bool = False,
    ):
        self.feature = feature
        self.augment = augment

        root = Path(cache_dir)
        meta = json.loads((root / "metadata.json").read_text())

        self.classes      = meta["classes"]
        self.class_to_idx = meta["class_to_idx"]

        feature_dir = root / split / feature
        if not feature_dir.exists():
            raise FileNotFoundError(
                f"{feature_dir} not found — run scripts/preprocess.py first"
            )

        self.samples: List[Tuple[str, int]] = []
        for label_dir in sorted(feature_dir.iterdir()):
            if not label_dir.is_dir():
                continue
            label = label_dir.name
            idx   = self.class_to_idx[label]
            for npy in sorted(label_dir.glob("*.npy")):
                self.samples.append((str(npy), idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        path, label_idx = self.samples[index]
        x = torch.from_numpy(np.load(path))
        if self.feature == "mel":
            x = x.unsqueeze(0)  # (n_mels, T) → (1, n_mels, T) for Conv2d
        if self.augment:
            x = self._augment(x)
        return x, label_idx

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        if self.feature == "mel":
            # x: (1, n_mels, time_frames) — SpecAugment: frequency + time masking
            x = x.clone()
            n_mels, n_frames = x.shape[1], x.shape[2]
            f = random.randint(0, 15)
            f0 = random.randint(0, max(0, n_mels - f))
            x[:, f0:f0 + f, :] = 0.0
            t = random.randint(0, 20)
            t0 = random.randint(0, max(0, n_frames - t))
            x[:, :, t0:t0 + t] = 0.0
        else:
            # x: (time_frames, n_mfcc) — time masking + Gaussian noise
            x = x.clone()
            t = random.randint(0, 20)
            t0 = random.randint(0, max(0, x.shape[0] - t))
            x[t0:t0 + t, :] = 0.0
            x = x + torch.randn_like(x) * 0.05
        return x

    @property
    def num_classes(self) -> int:
        return len(self.classes)

import random
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torchaudio
from torch.utils.data import Dataset

from .features.extraction import get_mel_spectrogram, get_mfcc, SAMPLE_RATE, N_SAMPLES

CORE_COMMANDS = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
SILENCE_LABEL = "silence"
UNKNOWN_LABEL = "unknown"

# How many 1-second silence clips to generate per noise file
_SILENCE_CLIPS_PER_FILE = 400


class SpeechCommandsDataset(Dataset):
    """
    TF Speech Commands dataset.

    split:   "train" | "val" | "test"
    feature: "mel"   → (1, n_mels, time_frames) for CNN
             "mfcc"  → (time_frames, n_mfcc)    for RNN / Transformer
    classes: core command labels to keep (default: CORE_COMMANDS)

    Silence is generated from _background_noise_ long recordings.
    Unknown covers all labeled commands not in `classes`.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        feature: str = "mel",
        classes: Optional[List[str]] = None,
        include_silence: bool = True,
        include_unknown: bool = True,
        silence_clips_per_file: int = _SILENCE_CLIPS_PER_FILE,
        seed: int = 42,
    ):
        self.root = Path(root)
        self.split = split
        self.feature = feature

        core = list(classes) if classes is not None else list(CORE_COMMANDS)

        all_labels: List[str] = list(core)
        if include_silence:
            all_labels.append(SILENCE_LABEL)
        if include_unknown:
            all_labels.append(UNKNOWN_LABEL)

        self.classes = all_labels
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        val_set = self._load_list("validation_list.txt")
        test_set = self._load_list("testing_list.txt")

        # (path, label_idx, offset_samples_or_None)
        self.samples: List[Tuple[str, int, Optional[int]]] = []

        audio_dir = self.root / "audio"
        for class_dir in sorted(audio_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            name = class_dir.name

            if name == "_background_noise_":
                if include_silence:
                    self._add_silence_samples(
                        class_dir, silence_clips_per_file, seed, val_set, test_set
                    )
                continue

            if name in core:
                label = name
            elif include_unknown:
                label = UNKNOWN_LABEL
            else:
                continue

            label_idx = self.class_to_idx[label]
            for wav in sorted(class_dir.glob("*.wav")):
                rel = f"{name}/{wav.name}"
                if split == "test" and rel not in test_set:
                    continue
                if split == "val" and rel not in val_set:
                    continue
                if split == "train" and (rel in val_set or rel in test_set):
                    continue
                self.samples.append((str(wav), label_idx, None))

    def _load_list(self, filename: str) -> set:
        path = self.root / filename
        if not path.exists():
            return set()
        return {line.strip() for line in path.read_text().splitlines() if line.strip()}

    def _add_silence_samples(
        self,
        noise_dir: Path,
        clips_per_file: int,
        seed: int,
        val_set: set,
        test_set: set,
    ) -> None:
        rng = random.Random(seed)
        label_idx = self.class_to_idx[SILENCE_LABEL]

        for wav in sorted(noise_dir.glob("*.wav")):
            info = torchaudio.info(str(wav))
            total_samples = info.num_frames
            if total_samples <= N_SAMPLES:
                continue

            max_offset = total_samples - N_SAMPLES
            offsets = [rng.randint(0, max_offset) for _ in range(clips_per_file)]

            # Deterministic split for silence: last 10% → test, prev 10% → val
            n = len(offsets)
            val_start = int(n * 0.8)
            test_start = int(n * 0.9)

            for i, offset in enumerate(offsets):
                if self.split == "test" and i >= test_start:
                    self.samples.append((str(wav), label_idx, offset))
                elif self.split == "val" and val_start <= i < test_start:
                    self.samples.append((str(wav), label_idx, offset))
                elif self.split == "train" and i < val_start:
                    self.samples.append((str(wav), label_idx, offset))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label_idx, offset = self.samples[idx]
        if offset is not None:
            waveform, sr = torchaudio.load(path, frame_offset=offset, num_frames=N_SAMPLES)
        else:
            waveform, sr = torchaudio.load(path)

        if sr != SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)

        if self.feature == "mel":
            return get_mel_spectrogram(waveform), label_idx
        return get_mfcc(waveform), label_idx

    @property
    def num_classes(self) -> int:
        return len(self.classes)

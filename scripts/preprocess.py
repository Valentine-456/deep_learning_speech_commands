import json
import random
import sys
import wave
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio.transforms as T
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

CORE_COMMANDS = [
    "yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go",
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
]
SILENCE_LABEL = "silence"
UNKNOWN_LABEL = "unknown"
ALL_CLASSES   = CORE_COMMANDS + [SILENCE_LABEL, UNKNOWN_LABEL]  # 22 total

SAMPLE_RATE = 16_000
N_SAMPLES   = 16_000   # 1 second

N_MFCC     = 40
N_MELS     = 128
N_FFT      = 1024
HOP_LENGTH = 160       # 10 ms @ 16 kHz
WIN_LENGTH = 400       # 25 ms @ 16 kHz

SILENCE_CLIPS_PER_FILE = 400
SILENCE_SEED           = 42


def load_split_sets(data_root: Path):
    def read(name):
        p = data_root / name
        return {l.strip() for l in p.read_text().splitlines() if l.strip()} if p.exists() else set()
    return read("validation_list.txt"), read("testing_list.txt")


def build_file_map(audio_dir: Path, val_set: set, test_set: set):
    """
    Returns {split: {label: [(path, offset)]}}
    offset is None for normal clips, int for silence clips.
    """
    file_map = {s: defaultdict(list) for s in ("train", "val", "test")}

    for class_dir in sorted(audio_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        name = class_dir.name

        if name == "_background_noise_":
            _add_silence(class_dir, file_map, val_set, test_set)
            continue

        if name in CORE_COMMANDS:
            label = name
        else:
            label = UNKNOWN_LABEL

        for wav in sorted(class_dir.glob("*.wav")):
            rel = f"{name}/{wav.name}"
            if rel in test_set:
                split = "test"
            elif rel in val_set:
                split = "val"
            else:
                split = "train"
            file_map[split][label].append((str(wav), None))

    return file_map


def _add_silence(noise_dir: Path, file_map: dict, val_set: set, test_set: set):
    rng = random.Random(SILENCE_SEED)

    for wav in sorted(noise_dir.glob("*.wav")):
        with wave.open(str(wav), "r") as wf:
            total_samples = wf.getnframes()
        if total_samples <= N_SAMPLES:
            continue

        max_offset = total_samples - N_SAMPLES
        offsets = [rng.randint(0, max_offset) for _ in range(SILENCE_CLIPS_PER_FILE)]

        n          = len(offsets)
        val_start  = int(n * 0.8)
        test_start = int(n * 0.9)

        for i, offset in enumerate(offsets):
            if i >= test_start:
                split = "test"
            elif i >= val_start:
                split = "val"
            else:
                split = "train"
            file_map[split][SILENCE_LABEL].append((str(wav), offset))



_mel_transform = T.MelSpectrogram(
    sample_rate=SAMPLE_RATE, n_fft=N_FFT,
    win_length=WIN_LENGTH, hop_length=HOP_LENGTH, n_mels=N_MELS,
)
_db_transform = T.AmplitudeToDB(top_db=80)
_mfcc_transform = T.MFCC(
    sample_rate=SAMPLE_RATE, n_mfcc=N_MFCC,
    melkwargs={"n_fft": N_FFT, "hop_length": HOP_LENGTH,
               "win_length": WIN_LENGTH, "n_mels": 64},
)


def load_waveform(path: str, offset=None) -> torch.Tensor:
    start = offset if offset is not None else 0
    stop  = start + N_SAMPLES if offset is not None else None
    data, sr = sf.read(path, start=start, stop=stop, dtype="float32", always_2d=True)
    # data: (frames, channels) => (channels, frames)
    waveform = torch.from_numpy(data.T)
    if sr != SAMPLE_RATE:
        import torchaudio
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
    if waveform.shape[-1] < N_SAMPLES:
        waveform = F.pad(waveform, (0, N_SAMPLES - waveform.shape[-1]))
    else:
        waveform = waveform[..., :N_SAMPLES]
    return waveform[:1]  # mono


def extract_mel(waveform: torch.Tensor) -> np.ndarray:
    """Returns (n_mels, time_frames) float32 array."""
    return _db_transform(_mel_transform(waveform))[0].numpy()


def extract_mfcc(waveform: torch.Tensor) -> np.ndarray:
    """Returns (time_frames, n_mfcc) float32 array."""
    return _mfcc_transform(waveform)[0].T.numpy()


def save_split(split: str, labels: dict, out_dir: Path):
    total = sum(len(v) for v in labels.values())
    with tqdm(total=total, desc=f"{split:5s}", unit="clip") as pbar:
        for label, samples in sorted(labels.items()):
            mel_dir  = out_dir / split / "mel"  / label
            mfcc_dir = out_dir / split / "mfcc" / label
            mel_dir.mkdir(parents=True, exist_ok=True)
            mfcc_dir.mkdir(parents=True, exist_ok=True)

            for i, (path, offset) in enumerate(samples):
                waveform = load_waveform(path, offset)
                np.save(mel_dir  / f"{i:05d}.npy", extract_mel(waveform))
                np.save(mfcc_dir / f"{i:05d}.npy", extract_mfcc(waveform))
                pbar.update(1)


DATA_ROOT = Path("data/train/train")
OUT_DIR   = Path("data/cache")


def main():
    data_root = DATA_ROOT
    audio_dir = data_root / "audio"
    out_dir   = OUT_DIR

    print("Loading split lists")
    val_set, test_set = load_split_sets(data_root)
    print(f"  val={len(val_set):,}  test={len(test_set):,}")

    print("Scanning audio files")
    file_map = build_file_map(audio_dir, val_set, test_set)

    for split in ("train", "val", "test"):
        counts = {k: len(v) for k, v in file_map[split].items()}
        total  = sum(counts.values())
        print(f"  {split}: {total:,} clips  |  {counts}")

    out_dir.mkdir(parents=True, exist_ok=True)
    class_to_idx = {c: i for i, c in enumerate(ALL_CLASSES)}
    metadata = {
        "classes":      ALL_CLASSES,
        "class_to_idx": class_to_idx,
        "features": {
            "mel":  {"shape": [N_MELS, "time_frames"], "n_mels": N_MELS},
            "mfcc": {"shape": ["time_frames", N_MFCC], "n_mfcc": N_MFCC},
        },
        "splits": {
            split: {label: len(samples) for label, samples in labels.items()}
            for split, labels in file_map.items()
        },
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"\nMetadata saved => {out_dir / 'metadata.json'}")

    print("\nExtracting features")
    for split in ("train", "val", "test"):
        save_split(split, file_map[split], out_dir)

    print(f"\nDone. Cache saved to {out_dir}/")


if __name__ == "__main__":
    main()

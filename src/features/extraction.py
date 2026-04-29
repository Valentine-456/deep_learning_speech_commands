from typing import Optional

import torch
import torchaudio.transforms as T

SAMPLE_RATE = 16000
CLIP_DURATION = 1.0
N_SAMPLES = int(SAMPLE_RATE * CLIP_DURATION)

N_MELS = 128
N_MFCC = 40
N_FFT = 1024
HOP_LENGTH = 160   # 10 ms at 16 kHz
WIN_LENGTH = 400   # 25 ms at 16 kHz


def pad_or_truncate(waveform: torch.Tensor, n_samples: int = N_SAMPLES) -> torch.Tensor:
    length = waveform.shape[-1]
    if length >= n_samples:
        return waveform[..., :n_samples]
    return torch.nn.functional.pad(waveform, (0, n_samples - length))


def get_mel_spectrogram(
    waveform: torch.Tensor,
    sample_rate: int = SAMPLE_RATE,
    n_mels: int = N_MELS,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    win_length: int = WIN_LENGTH,
    top_db: Optional[float] = 80.0,
) -> torch.Tensor:
    """Returns (1, n_mels, time_frames) log-Mel spectrogram."""
    waveform = pad_or_truncate(waveform)
    mel = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        n_mels=n_mels,
    )(waveform)
    if top_db is not None:
        mel = T.AmplitudeToDB(top_db=top_db)(mel)
    return mel[:1]  # (1, n_mels, time_frames)


def get_mfcc(
    waveform: torch.Tensor,
    sample_rate: int = SAMPLE_RATE,
    n_mfcc: int = N_MFCC,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    win_length: int = WIN_LENGTH,
) -> torch.Tensor:
    """Returns (time_frames, n_mfcc) MFCC sequence."""
    waveform = pad_or_truncate(waveform)
    mfcc = T.MFCC(
        sample_rate=sample_rate,
        n_mfcc=n_mfcc,
        melkwargs={
            "n_fft": n_fft,
            "hop_length": hop_length,
            "win_length": win_length,
            "n_mels": 64,
        },
    )(waveform)  # (1, n_mfcc, time_frames)
    return mfcc[0].T  # (time_frames, n_mfcc)

# Speech Commands Classification with Transformers — Project Summary

## Goal

Build and compare speech classification models on the TensorFlow Speech Commands dataset.
Start with a subset of classes, add a Transformer model, and analyze special cases like
"silence" and "unknown".

---

## Team

| Person | Role | Hardware |
|---|---|---|
| Valentyn Bondarenko | Training experiments, data preprocessing, saving results | GPU laptop, RTX 3050 |
| Mohammed Zaid Shaikh | Code structure, configs, confusion matrices, report | CPU |

---

## The Core Pipeline

```
Raw Audio (.wav)
    ↓
Feature Extraction (Mel Spectrogram / MFCC)
    ↓
Model (CNN / RNN / Transformer)
    ↓
Classification (softmax over N classes)
```

Key insight: a Mel spectrogram turns a 1-second clip into a 2D time-frequency
representation (e.g. 128×128) — you can treat it like an image (CNN) or a sequence
(RNN / Transformer).

---

## Two Ways to Think About Audio

### Audio as an Image → CNN / ViT
Convert waveform to spectrogram → 2D grid of time vs frequency. CNNs and Vision
Transformers treat it like an image. Works well because speech has local patterns in
both time and frequency (like textures).

### Audio as a Sequence → RNN / Transformer
MFCCs give a sequence of feature vectors (~one per 10ms frame). LSTMs/GRUs process
step-by-step; Transformers use global attention across all frames at once.

### Other Approaches (not in scope, for reference)
- Conformer — CNN + Transformer hybrid, state-of-the-art for speech
- Wav2Vec 2.0 — works directly on raw waveform, large pretrained model
- 1D CNN on raw waveform — no feature engineering needed

---

## Models to Implement (from scratch)

| # | Architecture | Input | Notes |
|---|---|---|---|
| 1 | CNN baseline | Mel spectrogram (2D) | Simple, strong reference point |
| 2 | LSTM / GRU | MFCC sequence | Sequential, step-by-step |
| 3 | Transformer encoder | MFCC sequence or spectrogram patches | 2-4 layers, multi-head attention, positional encoding |

**Why from scratch:** The project requires parameter investigation — you can only tune
what you control. Pretrained models are black boxes for this purpose.

**Optional 4th comparison:** Fine-tune `MIT/ast-finetuned-audioset-10-10-0.4593`
from HuggingFace (Audio Spectrogram Transformer) as a ceiling reference. Requires
adding `transformers` to requirements.

---

## Steps

### Step 1 — Environment + Data Exploration
1. Download Speech Commands dataset from Kaggle.
2. Explore data structure and audio files.
3. Start with 2 classes only ("yes" vs "no") to verify pipeline works.
4. Convert audio to spectrograms / MFCC (fixed length, pad/truncate to 1 second).
5. Split: train / validation / test.

### Step 2 — Baseline Models
- CNN baseline on Mel spectrograms.
- LSTM/GRU baseline on MFCC sequences.
- Train on 2-class subset first, then expand to 10-12 main classes + silence/unknown.

### Step 3 — Transformer Model
- Implement Transformer encoder (2-4 layers, multi-head attention).
- Input: spectrogram patches or MFCC sequences with positional encoding.
- Compare vs CNN/RNN on validation accuracy.

### Step 4 — Parameter Experiments
Training: learning rate, batch size (tuned for RTX 3050 VRAM).
Model: number of Transformer layers, attention heads, dropout rate.
Output: table of parameter → validation accuracy → test accuracy.

### Step 5 — Silence + Unknown
Test 3 approaches:
1. **Baseline** — single model on all classes including silence/unknown.
2. **Two-stage** — main classifier on 10 commands + separate binary classifier for silence vs unknown.
3. **Oversampling** — oversample silence/unknown during training.

Output: accuracy comparison especially on silence/unknown classes.

### Step 6 — Confusion Matrix Analysis
- Generate confusion matrix for best model on test set.
- Highlight: most common confusions, silence/unknown performance.
- Discussion: why certain classes confuse? (audio length, background noise, phonetic similarity)

### Step 7 — Final Evaluation + Report
- Train best configuration on full dataset.
- Report test accuracy + full confusion matrix.
- Deliverables: code, model comparison table, confusion matrix plot, silence/unknown discussion.

---

## Feature Extraction Options

| Feature | Tool | Notes |
|---|---|---|
| MFCC | `librosa.feature.mfcc` or `torchaudio.transforms.MFCC` | Compact, classic, ~13-40 coefficients/frame |
| Mel Spectrogram | `librosa.feature.melspectrogram` or `torchaudio.transforms.MelSpectrogram` | Richer, better for CNN/ViT |
| Raw waveform | — | Skip for this project |

---

## Dependencies

Install with:
- **CPU:** `pip install -r requirements-cpu.txt`
- **GPU (RTX 3050):** `pip install -r requirements-cuda.txt`

| Package | Purpose |
|---|---|
| `torch`, `torchvision`, `torchaudio` | Models + audio transforms |
| `librosa` | Feature extraction |
| `soundfile` | .wav file backend for librosa |
| `numpy<2` | Array ops (pinned — librosa incompatible with numpy 2.x) |
| `pandas` | CSV result tracking |
| `scikit-learn` | Confusion matrix, train/test split |
| `matplotlib`, `seaborn` | Plots |
| `tqdm` | Training progress bars |

---

## Key Papers

- *Attention Is All You Need* — Vaswani et al., 2017 (original Transformer)
- *AST: Audio Spectrogram Transformer* — Gong et al., 2021 (ViT for audio)
- *SpecAugment* — Park et al., 2019 (data augmentation for audio)

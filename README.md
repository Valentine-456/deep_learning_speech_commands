# Speech Commands Classification with Transformers

This project trains and compares speech classification models (CNN, LSTM/GRU, Transformer) on the TensorFlow Speech Commands dataset using PyTorch.

---

## 1. Clone the Repository

```bash
git clone https://github.com/Valentine-456/deep_learning_speech_commands
cd deep_learning_speech_commands
```

---

## 2. Create Python Virtual Environment

Python 3.10 is recommended.

### Mac / Linux

```bash
python -m venv .venv
source .venv/bin/activate
```

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

---

## 3. Install Dependencies

If your machine has CUDA GPU run this:

```bash
pip install -r requirements-cuda.txt
```

Otherwise run this:

```bash
pip install -r requirements-cpu.txt
```

If you want to automatically lint and format code before commits, run this:

```bash
pre-commit install
```

---

## 4. Download Speech Commands Dataset

Download the dataset from:

[https://www.kaggle.com/datasets/neehakadyan/tensorflow-speech-commands-dataset](https://www.kaggle.com/datasets/neehakadyan/tensorflow-speech-commands-dataset)

Unzip it and place it inside the project directory with the following structure:

```text
  data/
    speech_commands/
      yes/
      no/
      up/
      down/
      left/
      right/
      on/
      off/
      stop/
      go/
      _silence_/
      _unknown_/
      ...
```

Each subfolder contains `.wav` audio files (1 second, 16 kHz).

IMPORTANT:

* Do NOT commit the dataset to GitHub.
* The `data/` folder is ignored by git.

---

## 5. Smoke Test Dataset Loading

After placing the dataset correctly, run:

```bash
python -m src.train --smoke_test
```

Expected output example:

```text
Using device: cuda
Batch audio shape: torch.Size([64, 1, 128, 128])
Batch labels shape: torch.Size([64])
```

If you see similar output, the dataset pipeline is working correctly.

## 6. Train a Baseline Model

To run a real training experiment, use the training entrypoint. The example below trains the CNN baseline for one epoch and stores metrics in `runs/`:

```bash
python -m src.train --epochs 1 --model cnn --feature melspectrogram
```

You can also use the shared YAML config for a reproducible baseline:

```bash
python -m src.train --config configs/baseline_cnn.yaml
```

Useful flags for the report experiments:

* `--model cnn|lstm|transformer`
* `--feature melspectrogram|mfcc`
* `--classes 2` to train on a 2-class subset ("yes"/"no") first
* `--augmentation none|specaugment`
* `--silence_strategy baseline|two_stage|oversample`
* `--seed 42` for reproducibility

---

## Project Structure

```text
src/        - source code
configs/    - experiment configuration files
scripts/    - helper scripts
results/    - experiment summary tables
data/       - dataset (not tracked by git)
runs/       - training outputs (not tracked by git)
```

---

## Next Steps

After confirming dataset loading works:

1. Run the CNN baseline on 2 classes ("yes" vs "no").
2. Expand to 10+ classes including silence and unknown.
3. Implement and compare LSTM/GRU and Transformer encoder.
4. Sweep hyperparameters (learning rate, layers, attention heads, dropout).
5. Evaluate silence/unknown handling strategies.
6. Record validation and test accuracy per model and generate confusion matrices.

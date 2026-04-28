# CINIC-10 Image Classification Project

This project trains and compares image classification models on the CINIC-10 dataset using PyTorch.

---

## 1. Clone the Repository

```bash
git clone <REPO_URL>
cd <REPO_NAME>
````

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

## 4. Download CINIC-10 Dataset

Download the dataset from:

[https://www.kaggle.com/datasets/mengcius/cinic10](https://www.kaggle.com/datasets/mengcius/cinic10)

Unzip it and place it inside the project directory with the following structure:

```text
  cinic10/
    train/
    valid/
    test/
```

Each of the `train`, `valid`, and `test` folders must contain 10 class subfolders.

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
Batch images shape: torch.Size([64, 3, 32, 32])
Batch labels shape: torch.Size([64])
```

If you see similar output, the dataset pipeline is working correctly.

## 6. Train a Baseline Model

To run a real training experiment, use the new training entrypoint. The example below trains the default small CNN for one epoch and stores metrics in `runs/`:

```bash
python -m src.train --epochs 1 --model small_cnn --augmentation standard
```

You can also use the shared YAML config for a reproducible baseline:

```bash
python -m src.train --config configs/baseline_small_cnn.yaml
```

Useful flags for the report experiments:

* `--model small_cnn|resnet18|efficientnet_b0`
* `--pretrained` to enable ImageNet weights for supported backbones
* `--train_fraction 0.25` for reduced-data experiments
* `--few_shot_per_class 5` for few-shot experiments
* `--augmentation none|standard|strong`
* `--advanced_aug none|mixup|cutmix`
* `--seed 42` for reproducibility
* `--transfer_strategy two_phase` to freeze the backbone first and fine-tune later
* `--freeze_epochs 5` and `--finetune_lr 1e-4` for pretrained transfer learning

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

1. Run the baseline CNN experiment.
2. Compare it with at least one pretrained backbone.
3. Sweep training and regularization hyperparameters.
4. Add reduced-data and few-shot runs.
5. Record validation and test accuracy, mean, and standard deviation across repeats.

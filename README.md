# Acacias Sound Monitor

A Sound Event Detection (SED) system targeting urban noises to monitor and improve the acoustic environment in Acacias. This project is a modern, pure PyTorch implementation of the `SB_CNN_SED` model (originally from the [DCASE-models](https://github.com/MTG/DCASE-models) repository), designed for production efficiency, deployability, and maintainability.

## Repository Structure

```text
acacias-sound-monitor/
├── data/
│   ├── raw/             # Raw URBAN-SED audio (.wav) and annotations
│   └── processed/       # Cached PyTorch tensors (.pt) and fitted scaler
├── notebooks/           # Jupyter notebooks for PoC, visualization, and stakeholder demos
├── scripts/             # CLI entry points (prep, train, eval)
│   ├── preprocess_dataset.py
│   └── train.py         # Training loop with wandb
└── src/
    └── sbcnn_sed/       # Core application code
        ├── data/        # Feature extraction, Dataset classes, scaler
        └── model/       # PyTorch Model definitions (SB_CNN_SED)
```

## Quickstart

### 1. Environment Setup

We recommend using a virtual environment. The project uses Python 3.12+.

```bash
# Create and activate environment
uv venv
source .venv/bin/activate

# Install dependencies
uv sync
```

### 2. Prepare the Data

Before training, the raw audio needs to transition through the feature extraction pipeline. This caches exactly matching sequence tensors to disk so the GPU isn't bottlenecked by I/O.

```bash
python scripts/preprocess_dataset.py
```
*Note: Make sure the raw URBAN-SED_v2.0.0 dataset is extracted into `data/raw/` first.*

### 3. Train the Model

Training runs the `SBCNNSed` network over the prepared folds. It requires a free `wandb` account to log the metrics in real time.

```bash
python scripts/train.py
```
Checkpoints will be saved automatically into `model/checkpoints/` whenever validation loss improves.
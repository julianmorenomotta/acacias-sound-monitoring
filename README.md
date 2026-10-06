# Acacias Sound Monitor

A Sound Event Detection (SED) system targeting urban noises to monitor and improve the acoustic environment in Acacias. This project is a modern, pure PyTorch implementation of the `SB_CNN_SED` model (originally from the [DCASE-models](https://github.com/MTG/DCASE-models) repository), designed for production efficiency, deployability, and maintainability.

This README is the single entry point for the project. It documents every component that exists in the repository today: the data pipeline, training, evaluation, the inference pipeline, the backend API, the web UI, the configuration layer, the test suite and the deployment workflows.

## Contents

- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Quickstart](#quickstart)
  - [1. Environment Setup](#1-environment-setup)
  - [2. Prepare the Data](#2-prepare-the-data)
  - [3. Train the Model](#3-train-the-model)
  - [4. Evaluate the Model](#4-evaluate-the-model)
- [Inference Pipeline](#inference-pipeline)
- [Backend API (FastAPI)](#backend-api-fastapi)
- [Web UI (Gradio)](#web-ui-gradio)
- [Configuration (Hydra + OmegaConf)](#configuration-hydra--omegaconf)
- [Testing](#testing)
- [CI/CD & Deployment](#cicd--deployment)
- [Artifacts & Git LFS](#artifacts--git-lfs)
- [Project Documents](#project-documents)
- [Status & Roadmap](#status--roadmap)

## Repository Structure

```text
acacias-sound-monitor/
├── app/                  # FastAPI backend + Gradio web UI (Hugging Face Space)
│   ├── app.py            # Gradio Blocks UI; mounts the FastAPI router at launch
│   ├── api.py            # FastAPI router: GET /api/health, POST /api/predict
│   ├── inference.py      # Loads SoundEventDetector from configs/inference.yaml
│   ├── requirements.txt  # HF Space requirements (compiled from pyproject.toml)
│   └── README.md         # Hugging Face Space card
├── configs/              # Hydra / OmegaConf configuration tree
│   ├── config.yaml       # Primary config — composes the groups via `defaults`
│   ├── inference.yaml    # Runtime inference config used by the app
│   ├── model/            # Architecture-as-data (sb_cnn.yaml)
│   ├── data/             # Dataset paths + class list (urban_sed.yaml)
│   ├── features/         # Mel-spectrogram parameters (default.yaml)
│   ├── train/            # Training settings (default.yaml)
│   └── eval/             # Evaluation protocol (parity.yaml)
├── data/
│   ├── raw/              # Raw URBAN-SED audio (.wav) and annotations (git-ignored)
│   └── processed/        # Cached tensors (.pt) + fitted scaler (Git LFS)
├── docs/
│   ├── roadmap.md        # Development roadmap and current status
│   └── node_device.md    # Low-cost edge-device hardware research
├── models/
│   └── checkpoints/      # Trained checkpoints (best_sed_model.pth, Git LFS)
├── notebooks/            # Jupyter notebooks (poc.ipynb, demo.ipynb)
├── scripts/              # CLI entry points
│   ├── preprocess_dataset.py  # Feature extraction + scaler fitting
│   ├── train.py               # Training loop with W&B + early stopping
│   └── evaluate.py            # Segment-based evaluation → JSON report
├── src/
│   └── sbcnn_sed/        # Core application code (installed as a package)
│       ├── config/       # Typed OmegaConf schemas (schemas.py)
│       ├── data/         # MelSpectrogramExtractor + UrbanSedDataset
│       ├── model/        # SBCNNSed PyTorch model (config-parameterised)
│       ├── pipeline/     # SoundEventDetector inference pipeline
│       └── utils/        # MinMaxScaler + URBAN_SED_CLASSES
├── tests/                # pytest suite
├── pyproject.toml        # Project metadata + dependencies
└── .github/workflows/    # ci.yml (lint + tests) and deploy-hf-space.yml
```

## Requirements

- Python 3.12 (pinned via `requires-python` in `pyproject.toml`)
- [uv](https://docs.astral.sh/uv/) for environment and dependency management
- PyTorch 2.9.1 with CUDA 12.6 wheels (CPU and Apple MPS also supported at runtime)
- A free [Weights & Biases](https://wandb.ai/) account for training logs
- A free [Hugging Face](https://huggingface.co/) account to run/deploy the Space

## Quickstart

### 1. Environment Setup

We recommend using a virtual environment. The project uses Python 3.12.

```bash
# Create and activate environment
uv venv
source .venv/bin/activate

# Install dependencies (also installs the sbcnn_sed package in editable mode)
uv sync
```

### 2. Prepare the Data

Before training, the raw audio needs to transition through the feature extraction pipeline. This caches exactly matching sequence tensors to disk so the GPU isn't bottlenecked by I/O.

Make sure the raw `URBAN-SED_v2.0.0` dataset is extracted into `data/raw/` first, then run the preprocessing **from the `scripts/` directory** (the script resolves its `../data/...` paths relative to its working directory):

```bash
cd scripts
python preprocess_dataset.py
```

This writes mel-spectrogram sequences to `data/processed/URBAN-SED_v2.0.0/features/<fold>/`, event-roll labels to `.../labels/<fold>/`, and fits the `MinMaxScaler`, saving it to `data/processed/URBAN-SED_v2.0.0/scaler.pt`.

### 3. Train the Model

Training runs the `SBCNNSed` network over the prepared folds. It requires a free `wandb` account to log the metrics in real time. Run it **from the repository root** (paths are resolved relative to the working directory):

```bash
python scripts/train.py           # fresh run
python scripts/train.py --resume  # resume from models/checkpoints/best_sed_model.pth
```

Training uses `BCEWithLogitsLoss` + Adam, logs to Weights & Biases and applies early stopping. The best checkpoint is saved to `models/checkpoints/best_sed_model.pth` whenever validation loss improves; a periodic checkpoint is written every 10 epochs.

### 4. Evaluate the Model

Run from the repository root:

```bash
python scripts/evaluate.py
```

This evaluates `models/checkpoints/best_sed_model.pth` on the `test` fold at a decision threshold of `0.3` and computes micro/macro F1, per-class precision/recall/F1 and the Error Rate (ER). A JSON report is written to `logs/reports/<timestamp>_evaluation_results.json`.

## Inference Pipeline

`src/sbcnn_sed/pipeline/inference.py` exposes `SoundEventDetector`, the shared runtime used by both the backend API and the web UI.

```python
from sbcnn_sed.pipeline.inference import SoundEventDetector

detector = SoundEventDetector("configs/inference.yaml")
events = detector.predict("soundscape.wav")
# [{"event": "dog_bark", "start": 1.5, "end": 3.0, "confidence": 0.88}, ...]
```

`predict()` extracts mel-spectrogram sequences, applies the fitted scaler, runs the model and calls `_smooth_events()`, which:

- keeps frames whose probability exceeds `inference.confidence_threshold`,
- merges consecutive detections separated by no more than `inference.merge_gap_seconds`, and
- returns a JSON-serialisable list of `{event, start, end, confidence}` sorted by start time.

Paths and parameters are read from `configs/inference.yaml` (see [Configuration](#configuration-hydra--omegaconf)).

## Backend API (FastAPI)

The REST API is defined in `app/api.py` as a FastAPI `APIRouter` mounted under the `/api` prefix. It is attached to the Gradio app's underlying FastAPI instance when `app/app.py` starts.

| Method | Path | Description | Response |
|--------|------|-------------|----------|
| `GET` | `/api/health` | Liveness/readiness check | `{"status": "ok"}` (`503` if the model failed to load) |
| `POST` | `/api/predict` | Run SED on an uploaded audio file | JSON array of detected events |

**Request** — `POST /api/predict` expects a `multipart/form-data` upload with the file field named `audio`. Allowed extensions: `.wav`, `.mp3`, `.ogg`, `.flac`, `.m4a`, `.aiff`. An unsupported format returns `400`; an inference failure returns `500`.

**Response** — a JSON array of events:

```json
[
  { "event": "dog_bark", "start": 1.5, "end": 3.0, "confidence": 0.88 }
]
```

**Example**

```bash
# after starting the app (see Web UI section below):
curl http://localhost:7860/api/health
curl -X POST http://localhost:7860/api/predict -F "audio=@sample.wav"
```

## Web UI (Gradio)

The Gradio interface is defined in `app/app.py`. It imports its siblings with local imports (`from inference import detector`, `import api`), so launch it **from inside the `app/` directory**:

```bash
cd app
python app.py
```

The app binds to `0.0.0.0` (Gradio default port `7860`) and serves both the web UI and the `/api` endpoints. Users can upload an audio file or record from a microphone; detected events are returned and shown as JSON. `app/README.md` is the Hugging Face Space card (`sdk: gradio`, `app_file: app.py`).

## Configuration (Hydra + OmegaConf)

`configs/` is the configuration layer for training and evaluation. `configs/config.yaml` declares the composition, and each group file maps onto a typed dataclass schema in `src/sbcnn_sed/config/schemas.py`:

```yaml
# configs/config.yaml
defaults:
  - model: sb_cnn
  - data: urban_sed
  - features: default
  - train: default
  - eval: parity
```

| File | Used by | Contents |
|------|---------|----------|
| `configs/model/sb_cnn.yaml` | model instantiation | `_target_`, conv blocks, FC hidden, dropout |
| `configs/data/urban_sed.yaml` | data pipeline | paths, fold, class list |
| `configs/features/default.yaml` | preprocessing/inference | mel + sequence parameters |
| `configs/train/default.yaml` | training | batch size, LR, epochs, patience, W&B |
| `configs/eval/parity.yaml` | evaluation | threshold, merge gap, metric type |
| `configs/inference.yaml` | app runtime | model/scaler paths, audio + inference params |

The schema rejects unknown keys, wrong types and invalid enum values at load time. To inspect the resolved schema (defaults, validation and override behaviour):

```bash
python -m sbcnn_sed.config.schemas
```

> Note: `scripts/train.py` and `scripts/evaluate.py` currently still read hard-coded constants from inside the script; migrating them (and adding a reproducible parity harness) onto this config tree is tracked in `docs/roadmap.md`.

## Testing

The pytest suite lives in `tests/` and covers feature extraction, the scaler, model shape/parameter count, a training smoke test and the typed config layer.

```bash
uv run pytest tests/ -v
```

## CI/CD & Deployment

- **CI** (`.github/workflows/ci.yml`): on every push and pull request to `main`, installs with `uv sync`, lints `src/` and `scripts/` with `ruff`, and runs the pytest suite.
- **Hugging Face deploy** (`.github/workflows/deploy-hf-space.yml`): on push to `main` (or manual `workflow_dispatch`), recompiles `app/requirements.txt` from `pyproject.toml`, strips CUDA-only packages, bundles `app/`, `configs/`, `src/`, the checkpoint and the scaler, and force-pushes the bundle to the configured Hugging Face Space.

## Artifacts & Git LFS

- `models/checkpoints/best_sed_model.pth` and `data/processed/**/*.pt` are tracked with Git LFS (see `.gitattributes`).
- `data/**` is git-ignored except the fitted scaler, and `models/*` is ignored except the best checkpoint (see `.gitignore`).

## Project Documents

- `docs/roadmap.md` — development roadmap and current status.
- `docs/node_device.md` — hardware research for a low-cost edge monitoring device.
- `notebooks/demo.ipynb` — end-to-end visual demo; `notebooks/poc.ipynb` — proof of concept.

## Status & Roadmap

| Area | Status |
|------|--------|
| Data pipeline (mel features, `UrbanSedDataset`, `MinMaxScaler`, preprocessing) | Done |
| Model (`SBCNNSed`, config-parameterised) | Done |
| Training loop (BCE, Adam, early stopping, W&B, checkpointing) | Done |
| Evaluation (`scripts/evaluate.py`, JSON report) | Done |
| Inference pipeline (`SoundEventDetector`) | Done |
| Backend API (`/api/health`, `/api/predict`) | Done |
| Web UI + Hugging Face Space deployment | Done |
| Automated tests + CI | Done |
| Hydra/OmegaConf config layer + typed schemas | In progress (scripts not yet migrated) |
| Reproducible parity report vs. the Keras/DCASE baseline | Pending |

See `docs/roadmap.md` for the full plan.
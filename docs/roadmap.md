# Development Roadmap

The main idea of this document is to present the current status of the project, define the low-hanging fruit and some more long-term goals that could be attainable for this project. Some of the considerations to place in this document is development that takes into account having edge devices run the inference.

---

## Current State

The project is a pure PyTorch reproduction of the `SB_CNN_SED` model (originally from the [DCASE-models](https://github.com/MTG/DCASE-models) repository).

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | PyTorch data pipeline — Mel-spectrogram extraction, `UrbanSedDataset`, `MinMaxScaler`, preprocessing script | Done |
| 2 | PyTorch model — `SBCNNSed` architecture matching the DCASE reference, validated shapes and forward pass | Done |
| 3 | Training loop — `BCEWithLogitsLoss`, Adam optimizer, early stopping, W&B experiment tracking, checkpointing | Done |
| 4 | Formal evaluation — Segment-based metrics (F1, Error Rate) via `evaluate.py` | Done |
| 5 | Stakeholder demo — `demo.ipynb` notebook for interactive visualization | Done |
| 6 | Inference pipeline — `SoundEventDetector` class with event smoothing, configurable threshold and merge gap | Done |
| 7 | UI & deployment — Gradio web app, Hugging Face Spaces deployment with CI/CD workflow | Done |

**What is still missing after the reproduction effort:**
- Automated test suite (unit tests, model shape tests, training smoke test)
- CI pipeline for code quality (linting and tests on every push)
- Formal parity report comparing PyTorch metrics against the original Keras/DCASE baseline
- Backend API for programmatic access beyond the Gradio UI

The hardware research document (`docs/node_device.md`) contains a detailed study of low-cost acoustic monitoring devices and is available for reference.

---

## Immediate Priorities (~4–6 weeks)

These items are ordered by dependency — each phase builds on the previous one.

### Phase A: Code Quality

**A1 — Automated Tests**
Write a test suite covering:
- Feature extraction (shape correctness, numerical stability against reference)
- Scaler (fit, transform, save, load)
- Model architecture (parameter count, intermediate shapes, forward pass, output range)
- Training smoke test (short training run that exercises the full pipeline)

**A2 — CI Pipeline**
Set up a GitHub Actions workflow (`.github/workflows/ci.yml`) that runs:
- `ruff` linting on the `src/` and `scripts/` directories
- `pytest` suite on every push and pull request to `main`

This ensures code quality is maintained as new features are added.

### Phase B: Backend API

Develop a lightweight REST API using FastAPI that:
- Exposes an endpoint (e.g., `POST /predict`) accepting an uploaded audio file
- Runs inference through the existing `SoundEventDetector` pipeline
- Returns structured JSON predictions (events with class, start time, end time, confidence)
- Is integrated with the existing Gradio UI (the Gradio app can serve as the frontend, calling the FastAPI backend internally)
- Is deployed alongside the Gradio app on Hugging Face Spaces

This enables real-time inference from external applications, automated scripts, or future integrations.

### Phase C: Parity Report

Produce a formal parity report that:
- Runs the trained PyTorch model on the URBAN-SED test fold and records metrics (macro/micro F1, Error Rate, per-class breakdown)
- Re-runs the original Keras baseline (from DCASE-models) on the same data, or references the saved evaluation metrics from `trained_models/SB_CNN_SED/URBAN_SED/test/`
- Compares both sets of metrics and documents any discrepancies
- Is saved as a document (e.g., `docs/parity_report.md`) for stakeholder review

---

## Edge Device Planning

The project includes research on low-cost urban acoustic monitoring hardware, documented in `docs/node_device.md`. This covers:
- Original design (Mydlarz et al., 2016) based on Raspberry Pi Model B+
- Modern equivalent components (Raspberry Pi 5 / Zero 2 W, MEMS microphones)
- Software stack considerations (Raspberry Pi OS Lite, Python acquisition, remote administration)

---

## Further Work (No Timeline)

These are exploratory directions with no fixed delivery date.

### Deeper Model Exploration

- Research current state-of-the-art architectures for Sound Event Detection:
  - CRNN-based models (e.g., DCASE challenge baselines)
  - Transformer / attention-based approaches
  - Convolutional architectures with residual connections
- Develop and train a deeper/larger model on URBAN-SED
- Evaluate against the existing SB_CNN_SED baseline to measure improvement

### Edge-Deployable Model

- Based on the SOTA research above, design a lightweight model variant suitable for low-power devices (Raspberry Pi-class hardware)
- Consider model optimization techniques:
  - Quantization (INT8, FP16)
  - Pruning
  - Knowledge distillation from a larger teacher model
- Target: real-time inference within edge device constraints (CPU-only, limited memory, low power budget)
- This would eventually replace the planning-phase research in `docs/node_device.md` with an implementable solution
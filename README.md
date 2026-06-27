# NEUROSEG

<img src="icon.png" width="120" align="right"/>

**Neural segmentation pipeline for calcium imaging data.**  
NEUROSEG uses a JEPA-style self-supervised architecture to segment neuronal somas in TIFF stacks, with a full training suite for evaluating three experimental hypotheses.

---

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
  - [Inference Mode](#inference-mode)
  - [Training Mode](#training-mode)
- [Experiments](#experiments)
  - [H1 — Semi-supervised segmentation](#h1--semi-supervised-segmentation)
  - [H2 — Cross-organism transfer](#h2--cross-organism-transfer)
  - [H3 — Temporal representation stability](#h3--temporal-representation-stability)
- [Outputs](#outputs)
- [Tracking with MLflow](#tracking-with-mlflow)

---

## Overview

NEUROSEG has two top-level modes:

| Mode | Input | Output |
|------|-------|--------|
| **Inference** | Directory of TIFF stacks | Segmentation masks, activity traces, plots |
| **Training** | Directory of TIFF stacks | JEPA model checkpoint, MLflow experiment logs |

The inference pipeline runs: **load → normalise → segment (Cellpose) → extract ΔF/F₀ traces → visualise**.  
The training pipeline runs one of three hypothesis-driven JEPA experiments (H1 / H2 / H3).

---

## Requirements

- Python ≥ 3.10
- [uv](https://docs.astral.sh/uv/) — dependency and environment manager

All Python dependencies (PyTorch, Cellpose, MLflow, LangGraph, etc.) are declared in `pyproject.toml` and installed automatically by `uv`.

---

## Installation

```bash
git clone https://github.com/<your-username>/NEUROSEG.git
cd NEUROSEG
uv sync
```

That's it. No manual pip installs, no conda environments, no editing source files.

---

## Project Structure

```
NEUROSEG/
├── main.py                          # CLI entry point
├── pyproject.toml
├── data/                            # Put your TIFF stacks here
├── src/neuroseg/
│   ├── pipeline.py                  # LangGraph pipeline definition
│   ├── metrics.py                   # dice(), miou()
│   ├── checkpoint.py                # checkpoint save / list
│   ├── cli.py                       # interactive checkpoint picker
│   ├── models/
│   │   ├── state.py                 # LangGraph state schema
│   │   ├── mode.py                  # Mode enum (TRAINING / INFERENCE)
│   │   └── hypothesis.py            # Hypothesis enum (H1 / H2 / H3)
│   ├── nodes/                       # Inference pipeline nodes
│   │   ├── loader.py
│   │   ├── pre_processor.py
│   │   ├── segmenter.py
│   │   ├── activity_trace_calculator.py
│   │   └── visualizer.py
│   └── trainers/                    # Training modules
│       ├── jepa.py                  # JEPA model components
│       ├── dataset.py               # TIFFVideoDataset, LabeledTIFFDataset
│       ├── h1_trainer.py            # H1 — semi-supervised segmentation
│       ├── h2_trainer.py            # H2 — cross-organism transfer (stub)
│       └── h3_trainer.py            # H3 — temporal stability (stub)
└── src/utils/
    └── EB_JEPA_NEUROFINDER.py       # Original research prototype
```

---

## Usage

All commands are run from the repository root via `uv run`.

### Inference Mode

Segment all TIFF stacks in a data directory and write results to an output directory.

```bash
uv run main.py \
  --mode inference \
  --data  /path/to/tiff/stacks \
  --output /path/to/results
```

On first run you will be prompted to pick a model checkpoint (or skip to use the Cellpose baseline):

```
Available checkpoints:
  [1] jepa_pretrained_h1  H1  2024-11-01  Dice=0.8821  mIoU=0.7943
  [2] jepa_h1_finetune_f100  H1  2024-11-02  Dice=0.9104  mIoU=0.8317
  [0] Skip — use Cellpose baseline

Select checkpoint [0-2]:
```

**Quick run on the bundled sample:**

```bash
uv run main.py --mode inference --data data/ --output output/
```

### Training Mode

Training requires exactly one hypothesis flag (`--H1`, `--H2`, or `--H3`).

```bash
uv run main.py \
  --mode  train \
  --data  /path/to/unlabeled/tiff/stacks \
  --output /path/to/checkpoints \
  --H1
```

---

## Experiments

### H1 — Semi-supervised segmentation

**Hypothesis:** Self-supervised JEPA pretraining improves segmentation when labeled data is scarce.

**Protocol:**
1. Pretrain JEPA on unlabeled TIFF stacks.
2. Fine-tune with 1 %, 5 %, 10 %, and 100 % of labeled data.
3. Train a supervised baseline from scratch under identical conditions.
4. Compare Dice / mIoU curves across labeled-data fractions.

**Minimal run (pretraining only, no labeled data):**

```bash
uv run main.py \
  --mode train \
  --data  /path/to/unlabeled/tiffs \
  --output ./checkpoints \
  --H1 \
  --pretrain-epochs 100
```

**Full run (pretraining + fine-tuning at all fractions):**

```bash
uv run main.py \
  --mode train \
  --data  /path/to/unlabeled/tiffs \
  --output ./checkpoints \
  --H1 \
  --pretrain-epochs 100 \
  --labeled-data /path/to/labeled/data \
  --labeled-fractions 0.01 0.05 0.10 1.0 \
  --finetune-epochs 50
```

Labeled data must follow this layout:

```
labeled_data/
  sample_001/
    video.tif   ← multi-frame TIFF stack  (T × H × W)
    mask.tif    ← binary segmentation mask (H × W  or T × H × W)
  sample_002/
    ...
```

**MLflow tags:** `hypothesis=H1`, `labeled_fraction={f}`, `mode={pretrain|finetune|supervised_baseline}`

---

### H2 — Cross-organism transfer

**Hypothesis:** JEPA representations generalise better across organisms than supervised features.

**Protocol:**

1. Pretrain JEPA on the source-organism calcium imaging (unlabeled).
2. Fine-tune on the target organism with a limited epoch budget.
3. Train a supervised baseline from scratch on the target organism.
4. Compare transfer drop (Dice / mIoU) between the two modes.

Any two Neurofinder datasets work as source/target. The organism label is inferred automatically from Neurofinder directory names (e.g., `neurofinder.04.00` → zebrafish, `neurofinder.00.00` → mouse.visual_cortex). A natural split using the bundled Neurofinder data:

- **Source:** `neurofinder.04.xx` (zebrafish)
- **Target:** `neurofinder.00.xx` – `neurofinder.03.xx` (mouse visual cortex)

```bash
uv run main.py \
  --mode train \
  --data  /path/to/source/tiffs \
  --output ./checkpoints \
  --H2 \
  --source-data /path/to/neurofinder.04 \
  --target-data /path/to/neurofinder.00 \
  --pretrain-epochs 100 \
  --finetune-epochs 10
```

**MLflow tags:** `hypothesis=H2`, `source_organism={inferred}`, `target_organism={inferred}`, `mode={pretrain|finetune|supervised_baseline}`

---

### H3 — Temporal representation stability

**Hypothesis:** JEPA embeddings are more stable across time for the same neuron than for different neurons.

**Protocol:**
1. Encode frames with pretrained / baseline / random encoders.
2. Pool per-neuron embeddings using segmentation masks.
3. Measure within-neuron vs. between-neuron cosine similarity gap.

```bash
uv run main.py \
  --mode train \
  --data  /path/to/annotated/tiffs \
  --output ./checkpoints \
  --H3 \
  --pretrained-ckpt ./checkpoints/jepa_pretrained_h1_<run_id>.pt
```

> **Status:** H3 implementation is in progress. The trainer stub documents the full protocol and required config keys.

**MLflow tags:** `hypothesis=H3`, `mode={pretrained|supervised_baseline|no_pretrain}`

---

## Outputs

### Inference

```
output/
  cache/                          ← cached Cellpose masks (skipped on re-run)
    masks+<file>.npy
    flows+<file>.pkl
  segmentation/
    <file>/
      frame_0.png
      frame_1.png
      ...
  traces/
    <file>/
      traces_combined.png
      neuron_1.png
      neuron_2.png
      ...
    traces+<file>.npy
```

### Training

```
checkpoints/
  jepa_pretrained_h1_<run_id>.pt
  jepa_pretrained_h1_<run_id>.json   ← metadata: date, Dice, mIoU, hypothesis
  jepa_h1_finetune_f10_<run_id>.pt
  ...
mlruns/                              ← MLflow experiment store
```

---

## Tracking with MLflow

Every training run is automatically logged. To open the MLflow UI:

```bash
uv run mlflow ui
```

Then visit [http://localhost:5000](http://localhost:5000).

Each run records: hyperparameters, per-epoch loss / Dice / mIoU, and the checkpoint artifact path.

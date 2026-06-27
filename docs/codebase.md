# Codebase Structure

## Directory Layout

```
NEUROSEG/
├── main.py                          ← thin shim; calls neuroseg.__main__:main
├── pyproject.toml                   ← project metadata, dependencies, entry points
├── uv.lock                          ← locked dependency tree for reproducibility
├── config.example.yaml              ← fully annotated config template
├── config.demo.yaml                 ← scaled-down config for the 20-minute demo
├── demo.sh                          ← one-command end-to-end demo
│
├── src/
│   └── neuroseg/                    ← installable Python package
│       ├── __init__.py
│       ├── __main__.py              ← CLI entry point (argparse, config merge)
│       ├── pipeline.py              ← LangGraph StateGraph definition + run()
│       ├── checkpoint.py            ← save/load/list checkpoints
│       ├── cli.py                   ← interactive checkpoint picker for inference
│       ├── metrics.py               ← dice(), miou()
│       │
│       ├── models/                  ← LangGraph state and enum types
│       │   ├── __init__.py
│       │   ├── state.py             ← State TypedDict
│       │   ├── mode.py              ← Mode enum (TRAINING / INFERENCE)
│       │   ├── hypothesis.py        ← Hypothesis enum (H1 / H2 / H3)
│       │   └── node.py              ← Node enum (node name constants)
│       │
│       ├── nodes/                   ← Inference pipeline nodes (one file per node)
│       │   ├── __init__.py
│       │   ├── loader.py            ← LOADER node: read TIFF/AVI from disk
│       │   ├── pre_processor.py     ← PRE_PROCESSOR node: normalise/augment
│       │   ├── segmenter.py         ← SEGMENTER node: JEPA or Cellpose inference
│       │   ├── activity_trace_calculator.py  ← ΔF/F₀ trace extraction
│       │   └── visualizer.py        ← save segmentation overlays + trace plots
│       │
│       └── trainers/                ← Training system
│           ├── __init__.py
│           ├── jepa.py              ← All model classes + factory functions
│           ├── dataset.py           ← TIFFVideoDataset, LabeledTIFFDataset, NeurofinderDataset
│           ├── h1_trainer.py        ← H1Config, pretrain(), finetune(), run_h1()
│           ├── h2_trainer.py        ← run_h2(), _infer_organism()
│           └── h3_trainer.py        ← _compute_similarity_gap(), run_h3()
│
├── tests/
│   └── test_smoke.py                ← 17 smoke tests (model, metrics, checkpoint, datasets)
│
├── scripts/
│   ├── generate_demo_data.py        ← create synthetic Neurofinder dataset
│   └── plot_results.py              ← produce Figure 1 + Figure 2
│
├── docs/
│   ├── overview.md                  ← scientific context, hypotheses, dataset
│   ├── architecture.md              ← JEPA model, loss functions, checkpoints
│   ├── pipeline.md                  ← LangGraph StateGraph, node descriptions
│   ├── experiments.md               ← H1 / H2 / H3 protocols and metrics
│   ├── training.md                  ← training loops, MLflow, reproducibility
│   ├── data.md                      ← data formats, dataset classes
│   ├── metrics.md                   ← Dice, mIoU, cosine similarity
│   ├── configuration.md             ← all CLI flags and YAML keys
│   ├── codebase.md                  ← this file
│   └── pipeline.png                 ← rendered LangGraph graph (optional)
│
└── archive/
    └── EB_JEPA_NEUROFINDER.py       ← original research prototype (reference only)
```

---

## Module Responsibilities

### `__main__.py`

The CLI entry point. Responsibilities:
- Parse all `argparse` arguments.
- Load YAML config and merge with CLI flags.
- Determine `Mode` and `Hypothesis` from flags.
- Handle the inference checkpoint: either interactive picker (`cli.pick_checkpoint`) or direct path (`--checkpoint`).
- Call `pipeline.run()`.

Does **not** contain any model code, training logic, or file I/O beyond config loading.

---

### `pipeline.py`

The LangGraph graph definition. Responsibilities:
- Define all nodes and edges in `build_app()`.
- Implement the two conditional routing functions (`_which_mode`, `_files_remaining`).
- Implement the `_training_node()` dispatch function.
- Expose `run()` as the public API for programmatic use.
- Expose `visualize_pipeline()` for graph rendering.

Does **not** contain training logic or inference logic — those live in `trainers/` and `nodes/` respectively.

---

### `checkpoint.py`

Checkpoint I/O. Responsibilities:
- `save_checkpoint()`: save a plain `state_dict` with a JSON sidecar.
- `save_compound_checkpoint()`: save a structured payload (JEPA + seg head + arch dict) with `"compound": true` in the sidecar.
- `load_compound_checkpoint()`: validate type field and return payload dict.
- `list_checkpoints()`: scan a directory, filter to compound checkpoints, return metadata list.

---

### `cli.py`

Interactive terminal UI for inference checkpoint selection. Presents the list from `list_checkpoints()` with columns for model name, date, Dice, and mIoU. Returns the selected checkpoint path. Bypassed when `--checkpoint` is passed.

---

### `models/`

Typed definitions for the LangGraph state and enumerated values:

- `State` (TypedDict): the full pipeline state schema.
- `Mode` (StrEnum): `TRAINING`, `INFERENCE`.
- `Hypothesis` (StrEnum): `H1`, `H2`, `H3`.
- `Node` (StrEnum): constants for node names (`LOADER`, `PRE_PROCESSOR`, etc.).

These are kept separate from logic to avoid circular imports.

---

### `nodes/`

Each file is one LangGraph node. Every node function signature is:
```python
def <name>_node(state: State) -> dict:
    ...
    return {<keys to update in state>}
```

Nodes only read from `state` and return partial updates. They never modify `state` in-place.

---

### `trainers/jepa.py`

All model classes. No training loops, no file I/O, no MLflow. Pure PyTorch:

| Class | Role |
| ----- | ---- |
| `ResidualBlock` | Standard residual unit |
| `ResNet5` | Context encoder (strides=1, preserves spatial dims) |
| `ResUNet` | U-Net predictor with residual blocks |
| `StateOnlyPredictor` | Wraps ResUNet, implements consecutive-frame prediction |
| `Projector` | MLP for loss computation in projected space |
| `ImageDecoder` | Pixel-space reconstruction probe |
| `JEPA` | Full model: encoder + predictor + regulariser + pred-cost |
| `JEPAProbe` | Wraps JEPA + a probe head (e.g. ImageDecoder) |
| `VCLoss` | VICReg-style regulariser (variance + covariance terms) |
| `HingeStdLoss` | Hinge loss on per-dimension standard deviation |
| `CovarianceLoss` | Off-diagonal covariance penalty |
| `SquareLossSeq` | MSE prediction loss in projected latent space |
| `SomaDetHead` | 3-D soma detection head (prototype, not used in current pipeline) |
| `build_jepa()` | Factory function for the full JEPA model |
| `build_seg_head()` | Factory function for the segmentation head |

---

### `trainers/dataset.py`

Dataset classes and format detection utilities:

| Symbol | Role |
| ------ | ---- |
| `is_neurofinder_dir()` | Returns True if path has an `images/` subdirectory |
| `find_neurofinder_dirs()` | Finds all NF dirs under a root path |
| `TIFFVideoDataset` | Unlabeled clips from flat TIFF stacks |
| `LabeledTIFFDataset` | Labeled clips from `video.tif + mask.tif` layout |
| `NeurofinderDataset` | Unified dataset for Neurofinder format; unlabeled or labeled |

---

### `trainers/h1_trainer.py`

The most complete trainer. Contains:

| Symbol | Role |
| ------ | ---- |
| `H1Config` | Config dataclass with all hyperparameters |
| `build_config()` | Merges pipeline state config into `H1Config` |
| `setup_seed()` | Sets all random seeds for reproducibility |
| `_make_unlabeled_dataset()` | Auto-detects format, returns Dataset |
| `_make_labeled_dataset()` | Auto-detects format + fraction subsampling |
| `pretrain()` | Full pretraining loop, MLflow logging, checkpoint saving |
| `_validate_pretrain()` | Validation step for pretraining |
| `finetune()` | Fine-tuning or supervised-baseline loop, compound checkpoint |
| `_validate_finetune()` | Validation step for fine-tuning |
| `run_h1()` | Orchestrates pretrain + finetune × fractions × modes |

---

### `trainers/h2_trainer.py`

Thin wrapper over H1 functions:
- `_NF_ORGANISM_MAP`: maps Neurofinder dataset IDs to organism names.
- `_infer_organism()`: reads NF directory names to auto-tag MLflow runs.
- `run_h2()`: calls `pretrain()` on source, then `finetune()` on target twice (pretrained and baseline).

---

### `trainers/h3_trainer.py`

Post-hoc representation analysis:
- `_load_encoder()`: loads JEPA from any checkpoint format (plain or compound).
- `_compute_similarity_gap()`: computes within/between neuron cosine similarities using integer masks.
- `_run_mode()`: loads encoder, computes gap, logs to MLflow.
- `run_h3()`: runs all three modes (pretrained, supervised, no_pretrain).

---

## Dependency Graph

```
__main__.py
    └── pipeline.py
            ├── models/* (State, Mode, Hypothesis, Node)
            ├── nodes/*
            │       ├── loader.py
            │       ├── pre_processor.py
            │       ├── segmenter.py
            │       │       ├── checkpoint.py
            │       │       └── trainers/jepa.py
            │       ├── activity_trace_calculator.py
            │       └── visualizer.py
            └── trainers/*
                    ├── h1_trainer.py
                    │       ├── jepa.py
                    │       ├── dataset.py
                    │       ├── checkpoint.py
                    │       └── metrics.py
                    ├── h2_trainer.py
                    │       └── h1_trainer.py (reuses pretrain/finetune)
                    └── h3_trainer.py
                            ├── jepa.py
                            └── dataset.py
```

`metrics.py` and `checkpoint.py` are pure utilities with no upward dependencies. `models/` is also dependency-free to prevent circular imports.

---

## Entry Points

Defined in `pyproject.toml`:
```toml
[project.scripts]
neuroseg = "neuroseg.__main__:main"
```

After `pip install` or `uv sync`, both `neuroseg --help` (CLI command) and `uv run main.py --help` work identically. `main.py` in the project root is a thin shim:

```python
from neuroseg.__main__ import main
if __name__ == "__main__":
    main()
```

---

## Testing

```bash
uv run pytest tests/ -v
```

`tests/test_smoke.py` covers:

| Test | What it checks |
| ---- | -------------- |
| `test_package_imports` | All key modules importable |
| `test_build_jepa` | Factory returns JEPA instance |
| `test_jepa_encoder_shape` | Encoder output shape `(2, 4, 5, 32, 32)` |
| `test_jepa_unroll` | Unroll returns loss > 0 |
| `test_build_seg_head` | Seg head output shape `(2, 1, 32, 32)`, values in [0,1] |
| `test_dice_perfect` | `dice(ones, ones) == 1.0` |
| `test_dice_no_overlap` | `dice(left_half, right_half) == 0.0` |
| `test_dice_all_zeros` | `dice(zeros, zeros) == 1.0` (both-empty edge case) |
| `test_miou_perfect` | `miou(gt, gt) == 1.0` |
| `test_compound_checkpoint_roundtrip` | Save → load → keys match |
| `test_list_checkpoints_filters_pretrain_only` | Non-compound checkpoints hidden |
| `test_is_neurofinder_dir` | Format detection |
| `test_find_neurofinder_dirs` | Nested NF directory discovery |
| `test_h1config_defaults` | Config dataclass defaults correct |
| `test_build_config_overrides` | YAML config override mechanism |

Tests are purely CPU-based and require no data downloads. They run in under 30 seconds on any modern laptop.

# Configuration Reference

## Layers of Configuration

NEUROSEG merges configuration from three sources, in increasing priority:

1. **`H1Config` defaults** — hardcoded dataclass defaults in `trainers/h1_trainer.py`.
2. **YAML config file** (`--config FILE`) — any key matching an `H1Config` field overrides the default.
3. **CLI flags** — specific flags (e.g. `--pretrain-epochs`) override both of the above.

The merge logic in `__main__.py`:

```python
config = _load_yaml(args.config) if args.config else {}

config.setdefault("pretrain_epochs", args.pretrain_epochs)
config.setdefault("finetune_epochs", args.finetune_epochs)
if args.labeled_data is not None:
    config["labeled_data_dir"] = str(args.labeled_data)
...
```

`setdefault` means the YAML value wins if present; the CLI flag wins if the YAML key is absent. Direct assignment (`config["key"] = val`) means the CLI flag always wins.

---

## CLI Flags

### Required flags

| Flag | Values | Description |
| ---- | ------ | ----------- |
| `--mode` | `train`, `inference` | Pipeline operating mode |
| `--data DIR` | any path | Unlabeled data for training; TIFF files for inference |
| `--output DIR` | any path | Checkpoints (train) or results (inference) |

### Hypothesis selection (training only, mutually exclusive)

| Flag | Description |
| ---- | ----------- |
| `--H1` | Semi-supervised segmentation experiment |
| `--H2` | Cross-organism transfer experiment |
| `--H3` | Temporal representation stability analysis |

### H1-specific

| Flag | Default | Description |
| ---- | ------- | ----------- |
| `--labeled-data DIR` | _(none)_ | Labeled data directory for fine-tuning |
| `--labeled-fractions F [F...]` | `0.01 0.05 0.10 1.0` | Labeled fractions to evaluate |
| `--pretrain-epochs N` | `100` | Number of self-supervised pretraining epochs |
| `--finetune-epochs N` | `50` | Number of fine-tuning epochs per fraction |

### H2-specific

| Flag | Default | Description |
| ---- | ------- | ----------- |
| `--source-data DIR` | _(none, required)_ | Source-organism data (unlabeled pretraining) |
| `--target-data DIR` | _(none, required)_ | Target-organism labeled data (fine-tuning) |
| `--pretrain-epochs N` | `100` | Pretraining epochs on source organism |
| `--finetune-epochs N` | `50` | Fine-tuning epochs on target organism |

### H3-specific

| Flag | Default | Description |
| ---- | ------- | ----------- |
| `--pretrained-ckpt FILE` | _(none)_ | Path to pretrained JEPA checkpoint (from H1) |

### General

| Flag | Default | Description |
| ---- | ------- | ----------- |
| `--config FILE` | _(none)_ | YAML config file; CLI flags take precedence |
| `--checkpoint FILE` | _(none)_ | (Inference) Use this checkpoint, skip interactive picker |

---

## YAML Config File

All keys in `H1Config` can be set in YAML:

```yaml
# ── Model architecture ───────────────────────────────────────────────────────
dobs: 1               # input channels (1 = greyscale)
henc: 32              # encoder hidden channels
hpre: 32              # predictor hidden channels
dstc: 8               # latent state (embedding) channels
seg_head_hidden: 16   # segmentation head hidden channels

# ── Data ─────────────────────────────────────────────────────────────────────
seq_len: 10           # frames per clip fed to the model
img_size: 128         # resize all frames to img_size × img_size (square)
batch_size: 8
num_workers: 2        # DataLoader workers (set 0 for notebooks / Windows)
val_split: 0.1        # fraction of data held out for validation

# ── Training schedule ─────────────────────────────────────────────────────────
lr: 0.001
steps: 4              # JEPA unroll steps per batch
seed: 1
pretrain_epochs: 100
finetune_epochs: 50

# ── Loss coefficients ─────────────────────────────────────────────────────────
std_coeff: 10.0       # VCLoss variance penalty weight
cov_coeff: 100.0      # VCLoss covariance penalty weight

# ── H1 specific ───────────────────────────────────────────────────────────────
labeled_data_dir: /path/to/labeled/data
labeled_fractions: [0.01, 0.05, 0.10, 1.0]

# ── H2 specific ───────────────────────────────────────────────────────────────
source_data_dir: /path/to/source/organism
target_data_dir: /path/to/target/organism
finetune_budget: 10   # H2 fine-tuning epoch budget (overrides finetune_epochs for H2)

# ── H3 specific ───────────────────────────────────────────────────────────────
h3_data_dir: /path/to/labeled/data
pretrained_ckpt: ./checkpoints/jepa_pretrained_h1_<run_id>.pt
supervised_ckpt: ./checkpoints/jepa_h1_supervised_f100_<run_id>.pt
```

See `config.example.yaml` in the project root for the full annotated template.

---

## Demo Config (`config.demo.yaml`)

A scaled-down config for the 20-minute reproducibility demo. Reduces model size, epoch count, and image resolution so the full H1 pipeline fits on a CPU in under 20 minutes:

```yaml
dobs: 1
henc: 8          # 4× smaller than default
hpre: 8
dstc: 4          # 2× smaller
seg_head_hidden: 4

seq_len: 5
img_size: 64     # 2× smaller than default
batch_size: 4
num_workers: 0   # no multiprocessing for compatibility

lr: 0.001
steps: 2
seed: 42
val_split: 0.2
pretrain_epochs: 5    # 20× fewer than default
finetune_epochs: 3    # 17× fewer than default

labeled_data_dir: data/demo
labeled_fractions: [0.1, 0.5, 1.0]
```

---

## Environment-Dependent Settings

### GPU vs CPU

Device is selected automatically:
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

No configuration is needed. On GPU, training is significantly faster but otherwise identical.

### HPC / Cluster

When running on a cluster without a display:
- Set `num_workers: 0` in config to avoid multiprocessing issues.
- Matplotlib uses the non-interactive `Agg` backend by default when no display is available; no action needed.
- The interactive checkpoint picker (`cli.py`) is not suitable for batch scripts — use `--checkpoint FILE` to bypass it.

### Notebook Usage

After `pip install git+https://github.com/MohammadKarbalaee/NEUROSEG.git`:
```python
from neuroseg.pipeline import run
from neuroseg.models.mode import Mode
from neuroseg.models.hypothesis import Hypothesis

run(
    data_dir="/path/to/neurofinder",
    output_dir="/path/to/results",
    mode=Mode.TRAINING,
    hypothesis=Hypothesis.H1,
    config={
        "pretrain_epochs": 5,
        "finetune_epochs": 3,
        "labeled_data_dir": "/path/to/neurofinder",
        "img_size": 64,
        "num_workers": 0,
    },
)
```

Set `num_workers: 0` in notebooks to avoid multiprocessing issues with the DataLoader.

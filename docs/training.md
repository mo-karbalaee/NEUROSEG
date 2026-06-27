# Training System

## Overview

Training is routed through the LangGraph `TRAINING` node, which dispatches to one of three trainer modules based on the selected hypothesis. All three trainers share the same:

- Configuration dataclass (`H1Config`)
- Model factory functions (`build_jepa`, `build_seg_head`)
- Dataset auto-detection logic
- MLflow experiment logging

---

## H1Config — The Configuration Dataclass

```python
@dataclass
class H1Config:
    seq_len: int   = 10     # frames per clip
    img_size: int  = 128    # spatial resize target (square)
    batch_size: int = 8
    num_workers: int = 2
    henc: int  = 32         # encoder hidden channels
    hpre: int  = 32         # predictor hidden channels
    dstc: int  = 8          # latent state channels
    dobs: int  = 1          # input channels (1 = greyscale)
    seg_head_hidden: int = 16
    std_coeff: float = 10.0   # VCLoss variance term weight
    cov_coeff: float = 100.0  # VCLoss covariance term weight
    lr: float  = 1e-3
    pretrain_epochs: int = 100
    finetune_epochs: int = 50
    steps: int = 4            # JEPA unroll steps
    seed: int  = 1
    labeled_fractions: list  = [0.01, 0.05, 0.10, 1.0]
    val_split: float = 0.1
```

`H1Config` is used by all three trainers, not just H1. H2 and H3 read the same config object and may override specific fields.

`arch_dict()` extracts the model architecture parameters as a plain dict for embedding in checkpoint files:
```python
{"dobs": 1, "henc": 32, "hpre": 32, "dstc": 8, "seg_head_hidden": 16}
```

---

## Building Config from Pipeline State

```python
def build_config(state: State) -> H1Config:
    cfg = H1Config()
    for k, v in state.get("config", {}).items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg
```

Any key in `state["config"]` that matches a field name in `H1Config` overrides the default. Unknown keys are silently ignored. This is how YAML config files and CLI flags flow through to training: they are merged into `state["config"]` by `__main__.py` and picked up here.

---

## Pretraining (H1 and H2)

### What is trained

Both the JEPA model and a pixel-reconstruction probe are trained jointly:

```python
jepa = build_jepa(cfg.arch_dict(), device)
decoder = ImageDecoder(cfg.dstc, cfg.dobs, hidden_dim=16)
pixel_decoder = JEPAProbe(jepa, decoder, nn.MSELoss())
```

`ImageDecoder` is a small CNN that maps the `dstc`-channel latent feature map back to a 1-channel image. `JEPAProbe` wraps it with the JEPA encoder and MSE loss. The pixel-reconstruction probe is a secondary diagnostic signal — it is not the primary objective.

### Optimizer

Two parameter groups:
```python
Adam([
    {"params": jepa.parameters(), "lr": cfg.lr},
    {"params": pixel_decoder.head.parameters(), "lr": cfg.lr / 10},
])
```

### Training loop

For each epoch and batch:
```python
_, (jepa_loss, *_) = jepa.unroll(x, None, nsteps=cfg.steps, unroll_mode="parallel")
recon_loss = pixel_decoder(x, x)
(jepa_loss + recon_loss).backward()
optimizer.step()
```

The JEPA `unroll()` call returns the total loss `L_VC + L_pred`. The reconstruction loss is additional signal encouraging the latent space to be spatially interpretable.

### Validation

At the end of each epoch, both losses are computed on the held-out validation split with `torch.inference_mode()`. Metrics are logged to MLflow per epoch.

### Output

A **pretrain checkpoint** containing only the encoder `state_dict` (not the seg head) is saved at the end of the run. This checkpoint is consumed by the subsequent fine-tuning phase.

---

## Fine-Tuning (H1 and H2)

### What is trained

The full JEPA encoder is loaded from the pretrain checkpoint and a new seg head is attached:

```python
jepa = build_jepa(cfg.arch_dict(), device)
if mode == "finetune" and pretrained_checkpoint is not None:
    state_dict = torch.load(str(pretrained_checkpoint), map_location=device, weights_only=True)
    jepa.load_state_dict(state_dict)

seg_head = build_seg_head(cfg.dstc, cfg.seg_head_hidden).to(device)
```

For `mode="supervised_baseline"`, `pretrained_checkpoint` is `None` and the JEPA starts from random weights — this is the comparison baseline.

### Optimizer

Two parameter groups with differential learning rates:
```python
Adam([
    {"params": jepa.encoder.parameters(), "lr": cfg.lr / 10},
    {"params": seg_head.parameters(), "lr": cfg.lr},
])
```

The encoder is updated at 10× slower rate than the seg head. This is a common fine-tuning practice: the pretrained features are treated as a warm start, updated slowly to avoid destroying what was learned during pretraining.

### Forward pass

```python
enc_state = jepa.encoder(x)        # (B, dstc, T, H, W)
enc_mean = enc_state.mean(dim=2)   # (B, dstc, H, W) — temporal average
pred = seg_head(enc_mean)          # (B, 1, H, W)
pred = F.interpolate(pred, size=mask_gt.shape[-2:], ...)  # upsample to mask size
loss = BCELoss(pred, mask_gt[:, 0:1].float())
```

The temporal average collapses the `T` frames into a single summary feature map. This is combined with the binary ground-truth mask (first frame, or broadcast if single mask) using binary cross-entropy.

### Validation

At the end of each epoch, Dice score and mIoU are computed on the held-out validation split with a 0.5 threshold on the predicted probability map.

### Output

A **compound checkpoint** containing JEPA + seg head state dicts, the architecture dict, and final Dice/mIoU metrics is saved. These are the inference-ready checkpoints.

---

## H3 — Representation Analysis (No Training)

H3 does not train any model. It:
1. Loads each encoder mode (pretrained, supervised, random) from existing checkpoints.
2. Runs the encoder in `inference_mode` on the labeled dataset.
3. Computes within-neuron and between-neuron cosine similarities.
4. Logs results to MLflow.

See `trainers/h3_trainer.py` and the [experiments doc](experiments.md) for details.

---

## MLflow Logging

Every training run is wrapped in an `mlflow.start_run()` context. The following are logged:

### Pretrain run
- **Params:** `seq_len`, `img_size`, `batch_size`, `henc`, `hpre`, `dstc`, `std_coeff`, `cov_coeff`, `lr`, `pretrain_epochs`, `steps`
- **Metrics (per epoch):** `train/jepa_loss`, `train/recon_loss`, `val/jepa_loss`, `val/recon_loss`
- **Artifacts:** checkpoint `.pt` file path

### Finetune run
- **Tags:** `hypothesis`, `mode`, `labeled_fraction`
- **Params:** `labeled_fraction`, `mode`, `finetune_epochs`
- **Metrics (per epoch):** `train/loss`, `val/dice`, `val/miou`
- **Artifacts:** compound checkpoint `.pt` file path

### H3 run
- **Tags:** `hypothesis`, `mode`
- **Params:** `img_size`, `seq_len`
- **Metrics:** `within_sim`, `between_sim`, `gap`

### Experiment names

| Trainer | Experiment name |
| ------- | --------------- |
| H1 pretrain | `neuroseg-H1-pretrain` |
| H1 finetune | `neuroseg-H1-finetune` |
| H2 pretrain | `neuroseg-H2-pretrain` |
| H2 finetune | `neuroseg-H2-finetune` |
| H3 | `neuroseg-H3` |

Open the MLflow UI:
```bash
uv run mlflow ui   # → http://localhost:5000
```

---

## Reproducibility

All random processes are seeded via:

```python
def setup_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
```

The same seed is used for train/val splits (`torch.Generator().manual_seed(cfg.seed)`) and labeled-fraction subsampling (`np.random.default_rng(seed)`). Given the same seed and data, results are deterministic.

---

## Adding a New Hypothesis

1. Create `src/neuroseg/trainers/h4_trainer.py` with a `run_h4(state: State)` function.
2. Add `H4 = "H4"` to `src/neuroseg/models/hypothesis.py`.
3. Add the routing in `src/neuroseg/pipeline.py`:
   ```python
   elif hypothesis == Hypothesis.H4:
       from neuroseg.trainers.h4_trainer import run_h4
       run_h4(state)
   ```
4. Add `--H4` to the mutually exclusive group in `src/neuroseg/__main__.py`.
5. Write tests in `tests/test_smoke.py`.

No other files need to change.

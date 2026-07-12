# H1 · 5. Metrics, Logging & Outputs

How segmentation quality is measured, where every number is recorded, and what
files a run produces.

---

## 5.1 The metrics

Both are computed on binary masks (prediction thresholded at 0.5 vs binary
ground truth), in `src/neuroseg/metrics.py`.

### Dice coefficient

```
dice = 2 · |P ∩ G| / (|P| + |G|)
```

Overlap between predicted neuron pixels `P` and ground-truth neuron pixels `G`,
ranging 0 (no overlap) to 1 (perfect). **Edge case:** if both `P` and `G` are
empty (a frame with no neurons, correctly predicted empty), `dice = 1.0` — an
empty-vs-empty prediction is treated as perfect, not undefined.

### mIoU (mean Intersection-over-Union)

```
IoU(c) = |P_c ∩ G_c| / |P_c ∪ G_c|,   miou = mean over classes c ∈ {background, neuron}
```

Computed over **2 classes** (background = 0, neuron = 1). A class with empty union
is skipped. Because background usually dominates, mIoU is typically higher and
smoother than Dice.

Both are averaged over all clips in the split. Dice is the primary
segmentation-quality metric; mIoU is the secondary one.

---

## 5.2 The experiment log — `runs.csv`

All logging is a plain CSV written by `RunLogger` (`src/neuroseg/logger.py`) to
`<output>/logs/runs.csv`. There is **no MLflow / no tracking server** — the CSV is
the single source of truth, which keeps runs fully reproducible and portable.

Each training run gets a unique 8-character `run_id`. Two kinds of rows are
written:

- **Epoch rows** — one per epoch; `epoch` is filled, `checkpoint` is empty.
- **Checkpoint (summary) row** — one at the end of a run; `epoch` is empty,
  `test_dice` / `test_miou` hold the final held-out test scores, and `checkpoint`
  holds the saved `.pt` path.

Column schema:

| Column | Meaning |
| ------ | ------- |
| `timestamp` | ISO-8601 time |
| `run_id` | 8-char id shared by all rows of one run |
| `hypothesis` | `H1` |
| `mode` | `pretrain` / `finetune` / `supervised_baseline` |
| `model_name` | checkpoint filename stem |
| `labeled_fraction` | fraction of labels used (fine-tune arms only) |
| `epoch` | epoch index (empty on the summary row) |
| `train_loss` | per-epoch training loss (JEPA loss in pretrain, BCE in fine-tune) |
| `train_recon_loss` | reconstruction loss (pretrain only) |
| `val_jepa_loss`, `val_recon_loss` | validation losses (pretrain only) |
| `val_dice`, `val_miou` | per-epoch validation metrics (fine-tune only) |
| `test_dice`, `test_miou` | **final test-set metrics** (summary row only) |
| `within_sim`, `between_sim`, `gap` | H3-only columns (unused in H1) |
| `checkpoint` | path to the saved checkpoint (summary row only) |

To pull the H1 result out of the CSV:

```python
import pandas as pd
df = pd.read_csv("output/logs/runs.csv")
summary = (
    df[df["checkpoint"].ne("") & df["mode"].isin(["finetune", "supervised_baseline"])]
      .groupby(["mode", "labeled_fraction"])[["test_dice", "test_miou"]].last()
)
print(summary)
```

---

## 5.3 The figures

Generated automatically into `<output>/figures/`.

| File | Content |
| ---- | ------- |
| `pretrain_..._curves.png` | JEPA loss and reconstruction loss vs epoch (one per pretrain run) |
| `finetune_..._curves.png` | 2×2 panel: train loss, val Dice, val mIoU, final test scores (one per fine-tune/baseline run) |
| `h1_dice_comparison.png` | **The headline figure** — grouped bars of test Dice: pretrained vs baseline, at each labeled fraction |
| `h1_miou_comparison.png` | Same, for mIoU |

The comparison figures are produced by `plot_h1_dice()` reading `runs.csv`.

---

## 5.4 The checkpoints

| File pattern | Type | Contents | In inference picker? |
| ------------ | ---- | -------- | -------------------- |
| `jepa_pretrained_h1_<id>.pt` | plain | JEPA `state_dict` only | No (no seg head) |
| `jepa_h1_finetune_f*_<id>.pt` | compound | `{jepa, seg_head, arch}` + metrics | Yes |
| `jepa_h1_supervised_f*_<id>.pt` | compound | `{jepa, seg_head, arch}` + metrics | Yes |

Every checkpoint has a `.json` sidecar with metadata (date, hypothesis, mode,
`arch`, and — for compound checkpoints — final `dice`/`miou`). Because `arch` is
embedded, a checkpoint is **self-contained**: inference rebuilds the exact model
from the file alone. Checkpoints are never overwritten — each run writes a new
file keyed by `run_id`.

---

## 5.5 How to read the headline result

The story H1 wants to tell is a curve:

- **X-axis:** labeled fraction (0.1 → 1.0).
- **Y-axis:** test Dice (or mIoU).
- **Two series:** pretrained vs baseline.

**Hypothesis confirmed** if the pretrained series sits **above** the baseline,
with the **largest gap at the smallest fraction** (that is where self-supervised
features should matter most). If the pretrained series is at or below the
baseline, pretraining is not helping — see
[Results & fixes](06-results-and-fixes.md) for how to diagnose that.

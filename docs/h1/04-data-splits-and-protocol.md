# H1 · 4. Data, Splits & Experiment Protocol

This page covers where the data comes from, how it is turned into training clips,
how the labeled-fraction ablation works, how train/val/test are split, and how
`run_h1()` orchestrates the whole experiment.

---

## 4.1 The data format — Neurofinder

Training uses the **Neurofinder** benchmark layout. Each recording is a directory:

```
neurofinder.00.00/
    images/
        image00000.tiff        one grayscale frame per file (T frames total)
        image00001.tiff
        ...
    regions/
        regions.json           pixel coordinates of each labeled neuron
```

- **`images/`** is the video: one TIFF per time step.
- **`regions.json`** is the annotation: a list of neurons, each a list of
  `(x, y)` pixel coordinates. `_build_nf_mask()` rasterizes these into an integer
  label mask (pixel value = neuron ID), scaled to `img_size`. For H1 fine-tuning
  the mask is then **binarized** (neuron = 1, background = 0).

`find_neurofinder_dirs()` locates every such directory **recursively** under a
root path, so `--data /path/to/whole/benchmark` pools all recordings.

---

## 4.2 Clip extraction

The model consumes fixed-length clips of `seq_len` frames (default 5). A video of
`T` frames is cut into clips at a stride:

```
starts = range(0, T − seq_len + 1, stride)
```

- **Fine-tuning / evaluation**: `stride = seq_len` (non-overlapping clips) — this
  avoids leakage between train/val/test.
- **Pretraining**: `stride = pretrain_clip_stride` (default 2, overlapping) —
  overlapping clips multiply the amount of unlabeled training signal.

Videos shorter than `seq_len` are zero-padded to one clip, so every recording
contributes at least once.

---

## 4.3 Lazy, memory-bounded loading

`NeurofinderDataset` does **not** load pixel data in `__init__`. It builds only a
lightweight index of `(video, start_frame)` pairs plus the small per-video mask.
The actual frames for a clip are read from disk in `__getitem__`:

```
__init__     → list of clip locations (cheap; scales to the whole 20 GB dataset)
__getitem__  → read seq_len TIFFs, normalize, resize, (pad) → one clip tensor
```

This keeps memory flat regardless of dataset size (it replaced an eager version
that loaded everything into RAM and caused out-of-memory failures — see
[Results & fixes](06-results-and-fixes.md)).

**One consequence:** normalization is **per-clip** min–max (over the clip's
`seq_len` frames), not per-whole-video, because the whole video is never held in
memory. Pretraining and fine-tuning both use this dataset, so they remain
consistent with each other.

**Cost:** frames are re-read from disk every epoch, so data loading is I/O-bound.
`num_workers` (default 4) parallelizes it; on HPC, stage data to `$TMPDIR`.

---

## 4.4 The labeled-fraction ablation

The central x-axis of H1. `labeled_fraction` keeps a reproducible random subset of
the labeled **clips**:

```python
n = max(1, int(num_clips · labeled_fraction))
keep = rng.choice(num_clips, size=n, replace=False)   # seeded → reproducible
```

The default sweep is `labeled_fractions: [0.1, 0.5, 0.75, 1.0]` — i.e. train the
segmentation model on 10 %, 50 %, 75 %, and 100 % of the labels. Pretraining
always uses **all** the (unlabeled) data; only fine-tuning is starved of labels.
This is what makes it a *semi-supervised* test: lots of unlabeled data, varying
amounts of labels.

---

## 4.5 Train / validation / test splits

For each fine-tune/baseline run, the labeled clip set is split with a seeded
generator (`test_split = 0.2`, `val_split = 0.1`):

```
test  = 20 % of all clips           → final held-out evaluation
        of the remaining 80 %:
val   = 10 %                         → per-epoch monitoring / early insight
train = 90 %                         → gradient updates
```

The **test split is the number reported** in the comparison figures. The split is
seeded so the same clips land in the same split across runs and fractions.

Pretraining uses a simpler `val_split` (default 0.1) holdout, since it has no test
metric.

---

## 4.6 The full experiment loop — `run_h1()`

```python
def run_h1(state):
    cfg = build_config(state)                 # H1Config overlaid with config.yaml + CLI
    setup_seed(cfg.seed)                       # seed python/numpy/torch
    device = cuda if available else cpu

    # 1. Self-supervised pretraining on ALL unlabeled data (--data)
    pretrained = pretrain(data_dir, cfg, ...)

    # 2. For each labeled fraction, train BOTH arms on the labeled target (--labeled-data)
    for fraction in cfg.labeled_fractions:          # [0.1, 0.5, 0.75, 1.0]
        finetune(pretrained, labeled_dir, fraction, mode="finetune")
        finetune(None,       labeled_dir, fraction, mode="supervised_baseline")

    # 3. Produce the comparison figures
    plot_h1_dice(log_path, figures_dir)
```

So a full H1 run = **1 pretraining** + **2 × 4 = 8 fine-tuning runs** + plotting.

**Data paths (important for a valid test):**

- `--data` → the **large unlabeled** pool for pretraining (point at the whole
  benchmark).
- `--labeled-data` → the **single labeled target** for the fine-tuning sweep
  (e.g. `neurofinder.00.00`).

If both point at the same single recording, the pretrained model has no data
advantage and the hypothesis cannot show a benefit — keep the unlabeled pool
larger than the labeled target.

Next: exactly how quality is measured and logged —
[Metrics, logging & outputs](05-metrics-logging-and-outputs.md).

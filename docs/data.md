# Data Formats and Dataset Classes

## Supported Input Formats

NEUROSEG operates on two data formats:

| Format | Used for | Auto-detected by |
| ------ | -------- | ---------------- |
| **Neurofinder directory** | Training (H1, H2, H3) | presence of `images/` subdirectory |
| **Stacked TIFF** | Inference pipeline | any `.tif` / `.tiff` file |

The format detection runs at dataset construction time — there is no flag to set manually.

---

## Neurofinder Format

The Neurofinder benchmark defines a standard directory layout:

```
neurofinder.04.00/
    images/
        image00000.tiff    ← one file per frame, uint16
        image00001.tiff
        image00002.tiff
        ...
    regions/
        regions.json       ← pixel-coordinate annotations
```

### Frame files

Individual TIFF frames are loaded with `PIL.Image.open()`. Supports any bit-depth (`uint8`, `uint16`). Greyscale or RGB are both accepted; if 3-channel, only the first channel is retained.

### regions.json

A JSON array, one entry per annotated neuron:

```json
[
    { "coordinates": [[x0, y0], [x1, y1], [x2, y2], ...] },
    { "coordinates": [...] },
    ...
]
```

Each entry is a list of pixel coordinates `[x, y]` (column-first, row-second) marking all pixels that belong to that neuron's soma.

### Integer-labelled mask construction

`_build_nf_mask(nf_dir, orig_h, orig_w, img_size)` converts `regions.json` into a pixel-accurate `(img_size, img_size)` integer mask:

- Neuron `n` (1-indexed) gets integer label `n`.
- Background pixels stay at `0`.
- Coordinates are scaled to `img_size` if the video was resized.

This integer-labelled mask is what H3 uses for per-neuron spatial pooling. When binary masks are needed (for BCELoss in H1/H2 fine-tuning), `binarize=True` thresholds the integer mask to `{0, 1}`.

### Format detection

```python
def is_neurofinder_dir(path) -> bool:
    return (Path(path) / "images").is_dir()

def find_neurofinder_dirs(data_dir) -> list[Path]:
    root = Path(data_dir)
    if is_neurofinder_dir(root):
        return [root]
    return sorted(d for d in root.iterdir() if d.is_dir() and is_neurofinder_dir(d))
```

`find_neurofinder_dirs()` accepts either a single Neurofinder directory or a parent directory containing multiple ones. All detected subdirectories are pooled into a single dataset.

---

## Dataset Classes

### TIFFVideoDataset

**Purpose:** Unlabeled pretraining from flat TIFF stacks.

**Input layout:**
```
data_dir/
    stack_001.tif    ← multi-frame TIFF, shape (T, H, W) or (H, W)
    stack_002.tif
    ...
```

**Behaviour:**
- Loads each TIFF, normalises to [0, 1] per file (`(x - min) / (max - min)`).
- Splits each video into non-overlapping clips of length `seq_len`.
- If a video is shorter than `seq_len`, it is zero-padded to `seq_len` and treated as a single clip.
- Resizes spatial dimensions to `img_size × img_size` with bilinear interpolation at access time.

**Item returned:**
```python
{"video": torch.Tensor(1, T, H, W)}   # grayscale, float32, [0,1]
```

---

### LabeledTIFFDataset

**Purpose:** Labeled fine-tuning from a custom sample-directory layout (non-Neurofinder).

**Input layout:**
```
data_dir/
    sample_001/
        video.tif    ← multi-frame TIFF (T × H × W)
        mask.tif     ← integer or binary mask (H × W) or (T × H × W)
    sample_002/
        ...
```

**Labeling fraction:** At construction time `labeled_fraction ∈ (0, 1]` subsamples a random subset of samples reproducibly via `np.random.default_rng(seed)`.

**Behaviour:**
- Loads `video.tif` + `mask.tif` per sample.
- If `mask.tif` is 2-D, it is broadcast across all `T` frames.
- Normalises the video to [0, 1].
- Resizes with bilinear (video) or nearest-neighbour (mask) interpolation.
- If `binarize=True` (default for H1/H2): returns `mask > 0` as float32.
- If `binarize=False` (H3): returns integer neuron IDs as int64.

**Item returned:**
```python
{"video": torch.Tensor(1, T, H, W), "mask": torch.Tensor(T, H, W)}
```

---

### NeurofinderDataset

**Purpose:** Unified dataset for Neurofinder-format directories; supports both pretraining (unlabeled) and fine-tuning / H3 analysis (labeled).

**Mode flags:**

| Flag | Effect |
| ---- | ------ |
| `labeled=False` | Loads only video frames. Returns `{"video": ...}`. Fast, for pretraining. |
| `labeled=True, binarize=True` | Loads video + binary mask. For H1/H2 fine-tuning. |
| `labeled=True, binarize=False` | Loads video + integer-ID mask. For H3 cosine-similarity analysis. |
| `labeled_fraction < 1.0` | Randomly subsamples temporal clips. For H1 labeled-fraction ablation. |

**Behaviour:** Same clip-splitting logic as `TIFFVideoDataset`. When multiple Neurofinder directories are found under `data_dir`, their clips are pooled into a single dataset.

**Mask semantics:**
- For `binarize=True`: the returned `mask` tensor is broadcast across all `T` frames and has shape `(T, H, W)` with values in `{0, 1}`.
- For `binarize=False`: shape `(T, H, W)` with integer neuron IDs (0 = background, ≥1 = neuron).

---

## Clip Segmentation

All three dataset classes slice videos into **fixed-length clips** of `seq_len` frames. Clips are non-overlapping: `start ∈ range(0, T - seq_len + 1, seq_len)`.

If a video has fewer frames than `seq_len`, it is zero-padded:
```python
if T < seq_len:
    padded = np.zeros((seq_len, *video.shape[1:]), dtype=video.dtype)
    padded[:T] = video
```
This ensures every video contributes at least one clip regardless of length.

---

## Preprocessing

### Normalisation

All datasets normalise video frames to [0, 1] per file:
```python
def _normalize(data):
    mn, mx = data.min(), data.max()
    return (data - mn) / (mx - mn + 1e-8)
```

This is global normalisation (over all frames in a file), not per-frame. This preserves relative brightness differences across time — important since calcium transients manifest as changes in brightness over time.

### Spatial resizing

All clips are resized to `img_size × img_size`. Videos use bilinear interpolation; masks use nearest-neighbour to preserve integer labels.

### Data augmentation

No augmentation is applied in the current implementation. This is intentional — calcium imaging data has meaningful spatial structure (neuron locations) and temporal structure (activity timing) that would be distorted by typical vision augmentations (flips, rotations, colour jitter).

---

## Demo Dataset

`scripts/generate_demo_data.py` creates a synthetic Neurofinder-format dataset without any download:

- `n_neurons` circular neuron blobs placed at random locations.
- Fluorescence signal: sinusoidal with random phase per neuron (`0.4 + 0.35 * sin(t + φ)`).
- Background: Gaussian noise (`μ=0.05, σ=0.015`).
- Output: `data/demo/images/image*.tiff` (uint16) + `data/demo/regions/regions.json`.
- Also writes `data/demo_stacks/demo.tif` — the same frames stacked into a single TIFF for use by the inference pipeline.

The synthetic data follows the same statistical structure as real calcium imaging (pulsing neurons over a noisy background) but at low resolution (64×64 by default) and short duration (100 frames) for fast demo runs.

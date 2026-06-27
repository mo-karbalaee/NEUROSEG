# Metrics

## Overview

Two segmentation quality metrics are computed throughout training and reported at the end of inference. Both operate on binary arrays: predicted mask and ground-truth mask, both thresholded to `{True, False}`.

---

## Dice Score

The Dice similarity coefficient (also known as the F1 score for binary segmentation) measures the overlap between two binary regions:

```
Dice = 2 * |P ∩ G| / (|P| + |G|)
```

Where:
- `P` — predicted foreground pixels
- `G` — ground-truth foreground pixels
- `|·|` — count of True pixels

**Range:** [0, 1]. Higher is better.

**Edge case — both empty:** If both prediction and ground truth are entirely background (`|P| + |G| = 0`), the denominator is zero. By convention in medical imaging, this is defined as `1.0` — the model correctly predicted that there are no neurons in this frame.

```python
def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    denom = pred.sum() + gt.sum()
    if denom == 0:
        return 1.0
    return float(2 * (pred & gt).sum() / denom)
```

**Why not accuracy?** In calcium imaging segmentation the foreground (neuron pixels) typically occupies less than 5% of the image. A model that predicts all-background achieves >95% accuracy while being completely useless. Dice penalises this by not counting true negatives.

---

## mIoU — Mean Intersection over Union

Mean IoU (also called the Jaccard index, averaged over classes) measures overlap class by class:

```
IoU(c) = |P_c ∩ G_c| / |P_c ∪ G_c|
mIoU   = mean over all classes c where |P_c ∪ G_c| > 0
```

For binary segmentation (`num_classes=2`):
- Class 0 = background
- Class 1 = foreground (neuron)

```python
def miou(pred: np.ndarray, gt: np.ndarray, num_classes: int = 2) -> float:
    ious = []
    for c in range(num_classes):
        p = pred == c
        g = gt == c
        union = (p | g).sum()
        if union > 0:
            ious.append(float((p & g).sum() / union))
    return float(np.mean(ious)) if ious else 0.0
```

**Range:** [0, 1]. Higher is better.

**Relationship to Dice:** For two classes, `mIoU = (IoU(bg) + IoU(fg)) / 2`. The foreground IoU is related to Dice by:
```
Dice = 2 * IoU(fg) / (1 + IoU(fg))
```
So `mIoU ≤ Dice` for binary problems. Both are reported because they weight errors differently: Dice is more sensitive to foreground errors, while mIoU is more balanced between foreground and background.

---

## H3 Similarity Metrics

H3 uses a different metric family — cosine similarity in embedding space:

```
cosine_sim(a, b) = (a · b) / (||a|| * ||b||)
```

Range: [-1, 1]. A value of 1 means identical direction in embedding space; -1 means opposite; 0 means orthogonal.

Three summary statistics are computed:

| Metric | Formula | Interpretation |
| ------ | ------- | -------------- |
| `within_sim` | mean cosine sim over all (neuron n, frame pairs t1≠t2) | How consistently the same neuron is represented over time |
| `between_sim` | mean cosine sim over all (neuron pairs n1≠n2, frame t) | How similar distinct neurons look at the same time |
| `gap` | `within_sim − between_sim` | Primary H3 metric. Larger = more temporally stable, more discriminative |

A random encoder produces `gap ≈ 0` (within and between are equally random). A good encoder should have `gap > 0`, with JEPA pretraining hopefully producing a larger gap than the supervised baseline.

---

## Where Metrics Are Used

| Context | Metrics | Frequency |
| ------- | ------- | --------- |
| Pretraining validation | JEPA loss, recon loss | Per epoch |
| Fine-tuning validation | `val/dice`, `val/miou` | Per epoch |
| Checkpoint sidecar | `dice`, `miou` | End of run |
| Inference CLI picker | `dice`, `miou` | Displayed for selection |
| H3 analysis | `within_sim`, `between_sim`, `gap` | Per encoder mode |
| `plot_results.py` (Figure 1) | `val/dice` | Queried from MLflow post-hoc |

import numpy as np


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    intersection = (pred & gt).sum()
    return float(2 * intersection / (pred.sum() + gt.sum() + 1e-8))


def miou(pred: np.ndarray, gt: np.ndarray, num_classes: int = 2) -> float:
    ious = []
    for c in range(num_classes):
        p = pred == c
        g = gt == c
        union = (p | g).sum()
        if union > 0:
            ious.append(float((p & g).sum() / union))
    return float(np.mean(ious)) if ious else 0.0

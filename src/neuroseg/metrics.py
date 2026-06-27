import numpy as np


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    denom = pred.sum() + gt.sum()
    if denom == 0:
        return 1.0
    return float(2 * (pred & gt).sum() / denom)


def miou(pred: np.ndarray, gt: np.ndarray, num_classes: int = 2) -> float:
    ious = []
    for c in range(num_classes):
        p = pred == c
        g = gt == c
        union = (p | g).sum()
        if union > 0:
            ious.append(float((p & g).sum() / union))
    return float(np.mean(ious)) if ious else 0.0

import numpy as np


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute the Dice coefficient between a binary prediction and ground-truth mask."""
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    denom = pred.sum() + gt.sum()
    if denom == 0:
        return 1.0
    return float(2 * (pred & gt).sum() / denom)


def miou(pred: np.ndarray, gt: np.ndarray, num_classes: int = 2) -> float:
    """Compute mean IoU over all classes between a predicted and ground-truth label map."""
    ious = []
    for c in range(num_classes):
        p = pred == c
        g = gt == c
        union = (p | g).sum()
        if union > 0:
            ious.append(float((p & g).sum() / union))
    return float(np.mean(ious)) if ious else 0.0


def detection_f1(pred_labeled: np.ndarray, gt_labeled: np.ndarray,
                 iou_threshold: float = 0.5) -> dict:
    """
    Instance-level detection score: match predicted neurons to ground-truth neurons.

    Unlike Dice/mIoU (pixel-overlap, and monotonically related to each other), this
    counts TP/FP/FN over *neurons*: a predicted connected component is a true positive
    if it matches a ground-truth footprint with IoU >= iou_threshold. Matching is greedy
    by descending IoU, each prediction and each ground-truth used at most once.

    Args:
        pred_labeled: integer-labeled predicted instances (0 = background).
        gt_labeled:   integer-labeled ground-truth footprints (0 = background).
    Returns dict with precision, recall, f1, and the raw tp/fp/fn counts.
    """
    pred_ids = [i for i in np.unique(pred_labeled) if i > 0]
    gt_ids = [i for i in np.unique(gt_labeled) if i > 0]

    pairs = []
    for p in pred_ids:
        pmask = pred_labeled == p
        parea = int(pmask.sum())
        for g in np.unique(gt_labeled[pmask]):
            if g == 0:
                continue
            gmask = gt_labeled == g
            inter = int((pmask & gmask).sum())
            union = parea + int(gmask.sum()) - inter
            iou = inter / union if union else 0.0
            if iou >= iou_threshold:
                pairs.append((iou, int(p), int(g)))

    pairs.sort(reverse=True)
    matched_p, matched_g = set(), set()
    for _, p, g in pairs:
        if p in matched_p or g in matched_g:
            continue
        matched_p.add(p)
        matched_g.add(g)

    tp = len(matched_p)
    fp = len(pred_ids) - tp
    fn = len(gt_ids) - len(matched_g)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}

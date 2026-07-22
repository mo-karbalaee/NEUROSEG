import argparse
from pathlib import Path

import numpy as np
import tifffile as tiff

from neuroseg.metrics import detection_f1, dice as dice_fn
from neuroseg.trainers.dataset import _build_nf_mask, find_neurofinder_dirs


def main():
    """
    Cellpose baseline: segment a Neurofinder recording's time-projection and score it
    against the ground-truth footprints with the SAME metrics as our models (Dice + detection F1).

    Cellpose sees only active cells in any single frame, so we run it on a max-projection
    (every neuron that ever fires is visible) — the standard way to segment calcium data.
    """
    ap = argparse.ArgumentParser(description="Cellpose baseline Dice + detection F1 on Neurofinder.")
    ap.add_argument("--data", type=Path, required=True, metavar="DIR",
                    help="Neurofinder recording directory (images/ + regions/).")
    ap.add_argument("--iou-threshold", type=float, default=0.5)
    ap.add_argument("--proj-frames", type=int, default=300, help="Frames sampled for the max-projection.")
    ap.add_argument("--gpu", action="store_true", help="Run Cellpose on GPU.")
    args = ap.parse_args()

    nf = find_neurofinder_dirs(args.data)[0]
    paths = sorted((nf / "images").glob("*.tif*"))
    idx = np.linspace(0, len(paths) - 1, min(args.proj_frames, len(paths))).astype(int)
    proj = np.stack([tiff.imread(str(paths[int(i)])) for i in idx]).astype(np.float32).max(0)
    H, W = proj.shape
    gt = _build_nf_mask(nf, H, W, max(H, W))[:H, :W]

    lo, hi = np.percentile(proj, 1), np.percentile(proj, 99.7)
    img = np.clip((proj - lo) / (hi - lo + 1e-8), 0, 1).astype(np.float32)

    from cellpose import models
    model = models.CellposeModel(gpu=args.gpu)
    masks, _flows, _styles = model.eval([img], diameter=None, channels=[0, 0])
    pred = masks[0]

    det = detection_f1(pred, gt, iou_threshold=args.iou_threshold)
    d = dice_fn((pred > 0).astype(np.uint8), (gt > 0).astype(np.uint8))
    print(f"Cellpose baseline on {nf.name} (max-projection, native {H}x{W}):")
    print(f"  GT neurons = {int(gt.max())}   Cellpose cells = {int(pred.max())}")
    print(f"  detection F1 = {det['f1']:.3f}  (precision {det['precision']:.3f}, recall {det['recall']:.3f})")
    print(f"  Dice         = {d:.3f}")


if __name__ == "__main__":
    main()

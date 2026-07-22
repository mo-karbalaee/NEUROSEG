import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import label as connected_components

from neuroseg.checkpoint import load_compound_checkpoint
from neuroseg.metrics import detection_f1, dice as dice_fn
from neuroseg.nodes.segmenter import _filter_small_components
from neuroseg.trainers.dataset import NeurofinderDataset, _build_nf_mask, find_neurofinder_dirs
from neuroseg.trainers.jepa import build_jepa, build_seg_head


@torch.inference_mode()
def evaluate(checkpoint: str, target_dir: str, seq_len: int, seed: int,
             test_split: float, max_clips: int, iou_threshold: float) -> dict:
    """Aggregate a checkpoint's prediction over the target's test clips and score detection F1 + Dice."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = load_compound_checkpoint(Path(checkpoint))
    arch = payload["arch"]
    img_size = arch.get("img_size", 128)

    jepa = build_jepa(arch, device)
    jepa.load_state_dict(payload["jepa"], strict=False)
    jepa.eval()
    seg_head = build_seg_head(arch["dstc"], arch.get("seg_head_hidden", 16)).to(device)
    seg_head.load_state_dict(payload["seg_head"])
    seg_head.eval()
    thr = arch.get("seg_threshold", 0.5)

    nf_dir = find_neurofinder_dirs(target_dir)[0]
    first = next((nf_dir / "images").glob("*.tif*"))
    import tifffile as tiff
    H, W = tiff.imread(str(first)).shape[:2]
    gt_labeled = _build_nf_mask(nf_dir, H, W, max(H, W))          # native-res, integer per-neuron
    gt_labeled = gt_labeled[:H, :W]

    dataset = NeurofinderDataset(target_dir, seq_len, img_size, labeled=True,
                                 labeled_fraction=1.0, seed=seed, binarize=True)
    n = len(dataset)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist()
    test_idx = perm[:max(1, int(n * test_split))]
    if max_clips and len(test_idx) > max_clips:
        test_idx = list(np.array(test_idx)[np.linspace(0, len(test_idx) - 1, max_clips).astype(int)])

    prob_sum = np.zeros((H, W), dtype=np.float64)
    for i in test_idx:
        x = dataset[int(i)]["video"].unsqueeze(0).to(device)
        enc_mean = jepa.encoder(x).mean(dim=2)
        pred = seg_head(enc_mean)
        pred = F.interpolate(pred, size=(H, W), mode="bilinear", align_corners=False)
        prob_sum += pred.squeeze().cpu().numpy()
    prob = prob_sum / len(test_idx)

    binary = (prob > thr).astype(np.uint8)
    min_size = max(9, (H * W) // 8000)
    pred_labeled = _filter_small_components(connected_components(binary)[0], min_size)

    det = detection_f1(pred_labeled, gt_labeled, iou_threshold=iou_threshold)
    det["dice"] = dice_fn(binary, (gt_labeled > 0).astype(np.uint8))
    det["gt_neurons"] = int(len([i for i in np.unique(gt_labeled) if i > 0]))
    det["pred_neurons"] = int(len([i for i in np.unique(pred_labeled) if i > 0]))
    det["res"] = f"{H}x{W}"
    det["n_test_clips"] = len(test_idx)
    return det


def main():
    """Compute instance-level detection F1 (+ Dice) for one or more checkpoints on a target recording."""
    parser = argparse.ArgumentParser(description="Detection F1 for NEUROSEG checkpoints.")
    parser.add_argument("--checkpoint", type=Path, nargs="+", required=True, metavar="PT",
                        help="One or more compound checkpoints (finetune/supervised).")
    parser.add_argument("--data", type=Path, required=True, metavar="DIR",
                        help="Target Neurofinder recording (contains images/ and regions/).")
    parser.add_argument("--seq-len", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--test-split", type=float, default=0.2)
    parser.add_argument("--max-clips", type=int, default=40)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    args = parser.parse_args()

    print(f"{'checkpoint':44} {'res':>9} {'GT':>4} {'pred':>5} {'prec':>6} {'rec':>6} {'F1':>6} {'Dice':>6}")
    for ckpt in args.checkpoint:
        m = evaluate(str(ckpt), str(args.data), args.seq_len, args.seed,
                     args.test_split, args.max_clips, args.iou_threshold)
        print(f"{ckpt.name:44} {m['res']:>9} {m['gt_neurons']:>4} {m['pred_neurons']:>5} "
              f"{m['precision']:>6.3f} {m['recall']:>6.3f} {m['f1']:>6.3f} {m['dice']:>6.3f}")


if __name__ == "__main__":
    main()

import argparse
from pathlib import Path

import numpy as np
import torch

from neuroseg.checkpoint import find_compound_checkpoint, load_compound_checkpoint
from neuroseg.trainers.dataset import NeurofinderDataset
from neuroseg.trainers.jepa import build_jepa, build_seg_head


def _load_model(checkpoint_path: Path, device: torch.device):
    """Load a JEPA encoder + segmentation head from a compound checkpoint."""
    payload = load_compound_checkpoint(checkpoint_path)
    arch = payload["arch"]
    jepa = build_jepa(arch, device)
    jepa.load_state_dict(payload["jepa"], strict=False)
    jepa.eval()
    seg_head = build_seg_head(arch["dstc"], arch.get("seg_head_hidden", 16)).to(device)
    seg_head.load_state_dict(payload["seg_head"])
    seg_head.eval()
    return jepa, seg_head, arch


@torch.inference_mode()
def _predict(jepa, seg_head, clip: torch.Tensor, threshold: float, device: torch.device) -> np.ndarray:
    """Predict a binary neuron mask from one clip via temporal-mean encoder features."""
    x = clip.unsqueeze(0).to(device)
    enc = jepa.encoder(x).mean(dim=2)
    prob = seg_head(enc).squeeze().cpu().numpy()
    return (prob > threshold).astype(np.uint8)


def _overlay(ax, frame: np.ndarray, mask: np.ndarray, color):
    """Show a grayscale frame with a semi-transparent colored mask overlay."""
    ax.imshow(frame, cmap="gray")
    rgba = np.zeros((*mask.shape, 4))
    rgba[mask > 0] = (*color, 0.55)
    ax.imshow(rgba)
    ax.set_xticks([])
    ax.set_yticks([])


def main():
    """Build a target-organism segmentation gallery: raw / ground truth / JEPA-transfer / baseline."""
    parser = argparse.ArgumentParser(description="H2 qualitative segmentation gallery on the target organism.")
    parser.add_argument("--output", type=Path, required=True, help="Directory with the H2 compound checkpoints.")
    parser.add_argument("--target-data", type=Path, required=True, help="Target-organism Neurofinder directory.")
    parser.add_argument("--frames", type=int, default=4, help="Number of sample clips to show.")
    parser.add_argument("--seq-len", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--out", type=Path, default=None, help="Output figure path.")
    args = parser.parse_args()

    import matplotlib.pyplot as plt

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ft_ckpt = find_compound_checkpoint(args.output, "H2", "finetune")
    bl_ckpt = find_compound_checkpoint(args.output, "H2", "supervised_baseline")
    if ft_ckpt is None or bl_ckpt is None:
        raise SystemExit("Could not find both H2 finetune and supervised_baseline checkpoints in --output.")

    jepa_ft, head_ft, arch = _load_model(ft_ckpt, device)
    jepa_bl, head_bl, _ = _load_model(bl_ckpt, device)

    img_size = arch.get("img_size", 128)
    dataset = NeurofinderDataset(str(args.target_data), args.seq_len, img_size, labeled=True, binarize=True)
    if len(dataset) == 0:
        raise SystemExit("No clips found in --target-data.")

    n = min(args.frames, len(dataset))
    idxs = np.linspace(0, len(dataset) - 1, n).astype(int)

    fig, axes = plt.subplots(n, 4, figsize=(12, 3 * n))
    axes = np.atleast_2d(axes)
    col_titles = ["Raw frame", "Ground truth", "JEPA-transfer", "From-scratch"]

    for row, idx in enumerate(idxs):
        sample = dataset[int(idx)]
        clip = sample["video"]
        gt = sample["mask"][0].numpy().astype(np.uint8)
        frame = clip[0, 0].numpy()

        pred_ft = _predict(jepa_ft, head_ft, clip, args.threshold, device)
        pred_bl = _predict(jepa_bl, head_bl, clip, args.threshold, device)

        axes[row, 0].imshow(frame, cmap="gray"); axes[row, 0].set_xticks([]); axes[row, 0].set_yticks([])
        _overlay(axes[row, 1], frame, gt, (0.30, 0.69, 0.31))
        _overlay(axes[row, 2], frame, pred_ft, (0.13, 0.59, 0.95))
        _overlay(axes[row, 3], frame, pred_bl, (1.00, 0.60, 0.00))

        if row == 0:
            for col in range(4):
                axes[row, col].set_title(col_titles[col], fontsize=11)

    fig.suptitle("H2 — Segmentation on Target Organism", fontsize=13)
    plt.tight_layout()

    out = args.out or (args.output / "figures" / "h2_segmentation_gallery.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()

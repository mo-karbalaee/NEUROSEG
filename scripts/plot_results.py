#!/usr/bin/env python3
"""
Produce result figures from a completed demo or full training run.

Figure 1 — output/figures/h1_dice_comparison.png
    Bar chart: Dice score vs labeled-data fraction.
    Pretrained JEPA vs Supervised Baseline (H1 experiment).

Figure 2 — output/figures/h2_dice_comparison.png
    Bar chart: Dice on target organism.
    Pretrained JEPA vs Supervised Baseline (H2 experiment).

Figure 3 — output/figures/segmentation_preview.png
    Side-by-side comparison: raw frame | segmentation overlay.

Usage
-----
    uv run python scripts/plot_results.py \\
        --output output/figures \\
        --inference-output output/demo_inference \\
        --logs output/demo_checkpoints/logs/runs.csv
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from neuroseg.plots import plot_h1_dice, plot_h2_dice


# ── Figure 3: segmentation preview ───────────────────────────────────────────

def _find_frame_png(inference_dir: Path) -> Path | None:
    seg_dir = inference_dir / "segmentation"
    if not seg_dir.exists():
        return None
    for video_dir in sorted(seg_dir.iterdir()):
        frames = sorted(video_dir.glob("frame_*.png"))
        if frames:
            return frames[0]
    return None


def _find_raw_frame(inference_dir: Path) -> np.ndarray | None:
    traces_npy = sorted((inference_dir / "traces").glob("traces+*.npy"))
    if not traces_npy:
        return None
    stacks_dir = Path("data/demo_stacks")
    tif_files = list(stacks_dir.glob("*.tif")) + list(stacks_dir.glob("*.tiff"))
    if not tif_files:
        return None
    import tifffile
    data = tifffile.imread(str(tif_files[0]))
    frame = data[0].astype(np.float32)
    mn, mx = frame.min(), frame.max()
    return (frame - mn) / (mx - mn + 1e-8)


def plot_segmentation_preview(inference_dir: Path, out_path: Path):
    from PIL import Image

    seg_png = _find_frame_png(inference_dir)
    if seg_png is None:
        print(f"No segmentation PNG found in {inference_dir} — skipping Figure 3.")
        return

    seg_img = np.array(Image.open(str(seg_png)))
    raw_frame = _find_raw_frame(inference_dir)

    if raw_frame is not None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].imshow(raw_frame, cmap="gray")
        axes[0].set_title("Raw Frame")
        axes[0].axis("off")
        axes[1].imshow(seg_img)
        axes[1].set_title("Neuron Segmentation")
        axes[1].axis("off")
        fig.suptitle("Inference — Frame 0", fontsize=13)
    else:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(seg_img)
        ax.set_title("Inference — Neuron Segmentation (frame 0)")
        ax.axis("off")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Figure 3: {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("output/figures"),
                        metavar="DIR", help="Where to save the figures.")
    parser.add_argument("--inference-output", type=Path, default=None,
                        metavar="DIR", help="Directory where inference results were written.")
    parser.add_argument("--logs", type=Path, default=Path("output/demo_checkpoints/logs/runs.csv"),
                        metavar="CSV", help="Path to runs.csv written by the trainer.")
    args = parser.parse_args()

    plot_h1_dice(args.logs, args.output)
    plot_h2_dice(args.logs, args.output)

    if args.inference_output is not None:
        plot_segmentation_preview(args.inference_output, args.output / "segmentation_preview.png")


if __name__ == "__main__":
    main()

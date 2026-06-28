#!/usr/bin/env python3
"""
Produce two result figures from a completed demo run.

Figure 1 — output/figures/h1_dice_comparison.png
    Bar chart: Dice score vs labeled-data fraction.
    Pretrained JEPA vs Supervised Baseline (H1 experiment).
    Reads final val/dice from MLflow neuroseg-H1-finetune experiment.

Figure 2 — output/figures/segmentation_preview.png
    Side-by-side comparison: raw frame | segmentation overlay.
    Taken from the most recent inference output directory.

Usage
-----
    uv run python scripts/plot_results.py \\
        --output output/figures \\
        --inference-output output/demo_inference
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ── Figure 1: H1 Dice vs labeled fraction ─────────────────────────────────────

def _read_csv(log_path: Path) -> dict | None:
    import csv
    if not log_path.exists():
        print(f"No experiment log found at {log_path} — skipping Figure 1.")
        return None

    results: dict[str, dict[float, float]] = {
        "finetune": {},
        "supervised_baseline": {},
    }
    with open(log_path, newline="") as f:
        for row in csv.DictReader(f):
            mode = row.get("mode", "")
            frac_str = row.get("labeled_fraction", "")
            dice_str = row.get("val_dice", "")
            if mode in results and frac_str and dice_str:
                try:
                    results[mode][float(frac_str)] = float(dice_str)
                except ValueError:
                    pass
    return results


def plot_dice_comparison(results: dict, out_path: Path):
    fractions = sorted(
        set(list(results["finetune"]) + list(results["supervised_baseline"]))
    )
    if not fractions:
        print("No dice results to plot — skipping Figure 1.")
        return

    x = np.arange(len(fractions))
    w = 0.35
    jepa_vals = [results["finetune"].get(f, 0.0) for f in fractions]
    base_vals  = [results["supervised_baseline"].get(f, 0.0) for f in fractions]

    fig, ax = plt.subplots(figsize=(8, 5))
    b1 = ax.bar(x - w / 2, jepa_vals, w, label="JEPA Pretrained",      color="#2196F3", alpha=0.9)
    b2 = ax.bar(x + w / 2, base_vals,  w, label="Supervised Baseline",  color="#FF9800", alpha=0.9)

    for bar in (*b1, *b2):
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2, h + 0.015,
            f"{h:.2f}", ha="center", va="bottom", fontsize=8,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(f * 100)}%" for f in fractions])
    ax.set_xlabel("Labeled Data Fraction")
    ax.set_ylabel("Dice Score")
    ax.set_title("H1 — Semi-supervised Segmentation\nDice Score vs Labeled Data Fraction")
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Figure 1: {out_path}")


# ── Figure 2: segmentation preview ───────────────────────────────────────────

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
        print(f"No segmentation PNG found in {inference_dir} — skipping Figure 2.")
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
    print(f"Saved Figure 2: {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("output/figures"),
                        metavar="DIR", help="Where to save the figures.")
    parser.add_argument("--inference-output", type=Path, default=Path("output/demo_inference"),
                        metavar="DIR", help="Directory where inference results were written.")
    parser.add_argument("--logs", type=Path, default=Path("output/demo_checkpoints/logs/runs.csv"),
                        metavar="CSV", help="Path to runs.csv written by the trainer.")
    args = parser.parse_args()

    results = _read_csv(args.logs)
    if results:
        plot_dice_comparison(results, args.output / "h1_dice_comparison.png")

    plot_segmentation_preview(args.inference_output, args.output / "segmentation_preview.png")


if __name__ == "__main__":
    main()

import argparse
from pathlib import Path

import cv2
import numpy as np
import tifffile as tiff

from neuroseg.nodes.visualizer import _overlay_frame
from neuroseg.trainers.dataset import _build_nf_mask, find_neurofinder_dirs


def _resolve_recording(data_dir: Path) -> Path:
    """Return the Neurofinder recording directory (the one holding images/ and regions/)."""
    if (data_dir / "images").is_dir():
        return data_dir
    found = find_neurofinder_dirs(data_dir)
    if not found:
        raise SystemExit(f"No Neurofinder recording (images/ + regions/) found under {data_dir}")
    return found[0]


def main():
    """Render the Neurofinder ground-truth neuron masks over the raw frames as an MP4."""
    parser = argparse.ArgumentParser(
        description="Visualize Neurofinder ground-truth (regions.json) masks as a contour-overlay video."
    )
    parser.add_argument("--data", type=Path, required=True, metavar="DIR",
                        help="Neurofinder recording directory (contains images/ and regions/).")
    parser.add_argument("--output", type=Path, required=True, metavar="DIR",
                        help="Directory to write the ground-truth overlay MP4.")
    parser.add_argument("--num-frames", type=int, default=None, metavar="N",
                        help="Number of frames to render (default: all).")
    parser.add_argument("--fps", type=int, default=10, metavar="F", help="Output video frame rate.")
    args = parser.parse_args()

    nf_dir = _resolve_recording(args.data)
    img_paths = sorted((nf_dir / "images").glob("*.tiff")) or sorted((nf_dir / "images").glob("*.tif"))
    if not img_paths:
        raise SystemExit(f"No frames found under {nf_dir / 'images'}")
    if args.num_frames:
        img_paths = img_paths[: args.num_frames]

    first = tiff.imread(str(img_paths[0]))
    h, w = first.shape[:2]
    if h != w:
        raise SystemExit(f"Expected square frames; got {h}x{w} (Neurofinder mask builder assumes square).")

    mask = _build_nf_mask(nf_dir, h, w, h)
    n_neurons = int(mask.max())
    fg_pct = float((mask > 0).mean() * 100)

    args.output.mkdir(parents=True, exist_ok=True)
    video_path = args.output / f"{nf_dir.name}_groundtruth.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))

    for i, p in enumerate(img_paths):
        rgb = _overlay_frame(tiff.imread(str(p)), mask)
        cv2.putText(rgb, f"GROUND TRUTH  frame {i}  ({n_neurons} neurons, {fg_pct:.1f}% FG)",
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    writer.release()

    print(f"[gt-video] {video_path}")
    print(f"[gt-video] {len(img_paths)} frames | {n_neurons} neurons | {fg_pct:.1f}% foreground @ {args.fps} fps")


if __name__ == "__main__":
    main()

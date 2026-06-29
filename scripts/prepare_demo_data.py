#!/usr/bin/env python3
import argparse
import shutil
from pathlib import Path

import numpy as np
import tifffile


def prepare(
    source_dir: Path,
    out_dir: Path,
    stack_dir: Path,
    n_frames: int,
) -> None:
    src_images = source_dir / "images"
    src_regions = source_dir / "regions" / "regions.json"

    frame_paths = sorted(src_images.glob("*.tiff")) + sorted(src_images.glob("*.tif"))
    frame_paths = frame_paths[:n_frames]
    if not frame_paths:
        raise FileNotFoundError(f"No TIFF frames found in {src_images}")

    out_images = out_dir / "images"
    out_images.mkdir(parents=True, exist_ok=True)
    for i, p in enumerate(frame_paths):
        shutil.copy2(p, out_images / f"image{i:05d}.tiff")

    out_regions = out_dir / "regions"
    out_regions.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_regions, out_regions / "regions.json")

    frames = [tifffile.imread(str(p)) for p in frame_paths]
    stack = np.stack(frames)
    stack_dir.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(stack_dir / "demo.tif"), stack)

    print(f"Prepared {len(frame_paths)} frames from {source_dir.name}.")
    print(f"  Neurofinder dir : {out_dir}")
    print(f"  Stacked TIFF    : {stack_dir / 'demo.tif'}")


def prepare_h2(
    source_dir: Path,
    h2_source_out: Path,
    h2_target_out: Path,
    n_frames: int,
    source_frac: float = 0.7,
) -> None:
    src_images = source_dir / "images"
    src_regions = source_dir / "regions" / "regions.json"

    all_frames = sorted(src_images.glob("*.tiff")) + sorted(src_images.glob("*.tif"))
    all_frames = all_frames[:n_frames]
    if not all_frames:
        raise FileNotFoundError(f"No TIFF frames found in {src_images}")

    n_source = max(1, int(len(all_frames) * source_frac))
    splits = [
        ("source", all_frames[:n_source], h2_source_out),
        ("target", all_frames[n_source:], h2_target_out),
    ]

    for label, frames, out_dir in splits:
        out_images = out_dir / "images"
        out_images.mkdir(parents=True, exist_ok=True)
        for i, p in enumerate(frames):
            shutil.copy2(p, out_images / f"image{i:05d}.tiff")
        out_regions = out_dir / "regions"
        out_regions.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_regions, out_regions / "regions.json")
        print(f"H2 {label}: {len(frames)} frames → {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a real-data demo subset from a Neurofinder dataset."
    )
    parser.add_argument("--source", type=Path, default=Path("data/neurofinder.00.00"),
                        metavar="DIR", help="Source Neurofinder directory.")
    parser.add_argument("--out", type=Path, default=Path("data/demo"),
                        metavar="DIR", help="Output Neurofinder-format directory (H1).")
    parser.add_argument("--stack-out", type=Path, default=Path("data/demo_stacks"),
                        metavar="DIR", help="Output directory for stacked TIFF (inference).")
    parser.add_argument("--frames", type=int, default=100,
                        help="Number of frames to use (default: 100).")
    parser.add_argument("--h2-source-out", type=Path, default=Path("data/demo_h2_source"),
                        metavar="DIR", help="H2 source split output directory.")
    parser.add_argument("--h2-target-out", type=Path, default=Path("data/demo_h2_target"),
                        metavar="DIR", help="H2 target split output directory.")
    args = parser.parse_args()

    prepare(args.source, args.out, args.stack_out, args.frames)
    prepare_h2(args.source, args.h2_source_out, args.h2_target_out, args.frames)


if __name__ == "__main__":
    main()

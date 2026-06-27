#!/usr/bin/env python3
"""
Generate a small synthetic Neurofinder-format dataset for the demo.

Creates fluorescent neuron blobs that pulse sinusoidally over time,
mimicking calcium imaging data — no real dataset download required.

Output
------
<out>/                          Neurofinder-format directory
    images/image00000.tiff ...  individual frames (uint16)
    regions/regions.json        pixel-coordinate annotations

<stack-out>/
    demo.tif                    same frames as a single stacked TIFF
                                (used by the inference pipeline)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import tifffile


def _circle_coords(cx: int, cy: int, radius: int, H: int, W: int) -> list[list[int]]:
    coords = []
    for y in range(max(0, cy - radius), min(H, cy + radius + 1)):
        for x in range(max(0, cx - radius), min(W, cx + radius + 1)):
            if (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2:
                coords.append([x, y])
    return coords


def generate(
    data_dir: Path,
    stack_dir: Path,
    n_frames: int = 100,
    H: int = 64,
    W: int = 64,
    n_neurons: int = 5,
    seed: int = 42,
):
    rng = np.random.default_rng(seed)

    margin = 12
    radius = 4

    centers = [
        (int(rng.integers(margin, W - margin)), int(rng.integers(margin, H - margin)))
        for _ in range(n_neurons)
    ]

    t = np.linspace(0, 4 * np.pi, n_frames)
    signals = [
        0.4 + 0.35 * np.sin(t + rng.uniform(0, 2 * np.pi))
        for _ in range(n_neurons)
    ]

    frames = []
    for i in range(n_frames):
        frame = rng.normal(0.05, 0.015, (H, W)).clip(0, 1)
        for n_idx, (cx, cy) in enumerate(centers):
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    if dx ** 2 + dy ** 2 <= radius ** 2:
                        fy, fx = cy + dy, cx + dx
                        if 0 <= fy < H and 0 <= fx < W:
                            frame[fy, fx] += signals[n_idx][i] * rng.normal(1.0, 0.08)
        frames.append((frame.clip(0, 1) * 65535).astype(np.uint16))

    img_dir = data_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(frames):
        tifffile.imwrite(str(img_dir / f"image{i:05d}.tiff"), frame)

    regions_dir = data_dir / "regions"
    regions_dir.mkdir(parents=True, exist_ok=True)
    regions = [
        {"coordinates": _circle_coords(cx, cy, radius, H, W)}
        for cx, cy in centers
    ]
    with open(regions_dir / "regions.json", "w") as f:
        json.dump(regions, f)

    stack_dir.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(stack_dir / "demo.tif"), np.stack(frames))

    print(f"Generated {n_frames} frames ({H}×{W}), {n_neurons} neurons.")
    print(f"  Neurofinder dir : {data_dir}")
    print(f"  Stacked TIFF    : {stack_dir / 'demo.tif'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/demo"),
                        metavar="DIR", help="Neurofinder output directory.")
    parser.add_argument("--stack-out", type=Path, default=Path("data/demo_stacks"),
                        metavar="DIR", help="Stacked TIFF output directory (for inference).")
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--neurons", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate(args.out, args.stack_out, args.frames, args.size, args.size, args.neurons, args.seed)


if __name__ == "__main__":
    main()

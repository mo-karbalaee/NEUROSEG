#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

echo "[1/2] Building 10-frame demo stack from neurofinder.00.00 ..."
./.venv/bin/python - <<'PY'
import numpy as np, tifffile as tiff
from pathlib import Path
paths = sorted(Path("data/neurofinder.00.00/images").glob("*.tif*"))
frames = np.stack([tiff.imread(str(p)) for p in paths[2400:2410]])  # 10 frames, active region
out = Path("output/cellpose_demo_input"); out.mkdir(parents=True, exist_ok=True)
tiff.imwrite(str(out / "nf0000_10frames.tif"), frames.astype(np.uint16))
print("  wrote", frames.shape, "->", out / "nf0000_10frames.tif")
PY

echo "[2/2] Running Cellpose inference (empty output dir -> Cellpose auto-selected) ..."
rm -rf output/cellpose_infer
PYTORCH_ENABLE_MPS_FALLBACK=1 ./.venv/bin/python main.py --mode inference \
    --data output/cellpose_demo_input --output output/cellpose_infer

echo
echo "Done. Video:  output/cellpose_infer/segmentation/nf0000_10frames.tif.mp4"
echo "Traces:       output/cellpose_infer/traces/"

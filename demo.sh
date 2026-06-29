#!/usr/bin/env bash
# "Reproduce in 20 Minutes" demo.
# Run this from the NEUROSEG repository root.
#
# Usage:
#   bash demo.sh
#
# Requires: data/neurofinder.00.00/ (download from the Neurofinder benchmark).
#
# What it does (6 steps):
#   1. Prepare a 100-frame subset of neurofinder.00.00 as the demo dataset
#      (also creates a subset of neurofinder.04.00 for the H2 cross-domain experiment).
#   2. Train JEPA on that data (H1 — pretrain + finetune, scaled to ~10 min on CPU).
#   3. Train JEPA for cross-domain transfer (H2 — source pretrain + target finetune).
#   4. Measure temporal representation stability (H3 — cosine similarity gap).
#   5. Run inference on a stacked TIFF and write segmentation output.
#   6. Produce result figures from the training and inference outputs.
#
# Output:
#   output/figures/h1_dice_comparison.png  — Dice vs labeled-data fraction (H1)
#   output/figures/h1_miou_comparison.png  — mIoU vs labeled-data fraction (H1)
#   output/figures/h2_dice_comparison.png  — Dice: pretrained vs baseline (H2)
#   output/figures/h2_miou_comparison.png  — mIoU: pretrained vs baseline (H2)
#   output/figures/segmentation_preview.png — raw frame + segmentation overlay
#   output/demo_checkpoints/figures/h3_similarity.png — temporal stability (H3)

set -euo pipefail

CHECKPOINTS_DIR="output/demo_checkpoints"
INFERENCE_DIR="output/demo_inference"
FIGURES_DIR="output/figures"

# ── Step 1: Prepare real data subset ──────────────────────────────────────────
echo ""
echo "── Step 1/6  Preparing demo dataset (neurofinder.00.00, 100 frames) ─────"
uv run python scripts/prepare_demo_data.py \
    --source data/neurofinder.00.00 \
    --out data/demo \
    --stack-out data/demo_stacks \
    --frames 100 \
    --h2-source data/neurofinder.04.00 \
    --h2-source-out data/demo_h2_source

# ── Step 2: Train H1 (pretrain + finetune) ────────────────────────────────────
echo ""
echo "── Step 2/6  Training JEPA — H1 (pretrain + finetune) ───────────────────"
uv run main.py \
    --mode train \
    --H1 \
    --data data/demo \
    --output "$CHECKPOINTS_DIR" \
    --config config.demo.yaml

# ── Step 3: Train H2 (cross-domain transfer) ──────────────────────────────────
echo ""
echo "── Step 3/6  Training JEPA — H2 (cross-domain transfer) ─────────────────"
uv run main.py \
    --mode train \
    --H2 \
    --data data/demo \
    --output "$CHECKPOINTS_DIR" \
    --config config.demo.yaml

# ── Step 4: Run H3 (temporal representation stability) ────────────────────────
echo ""
echo "── Step 4/6  Measuring temporal stability — H3 ───────────────────────────"

# Find the H1 pretrained JEPA checkpoint (non-compound)
PRETRAINED_CKPT=$(python - <<'EOF'
from pathlib import Path
import json
for jf in sorted(Path("output/demo_checkpoints").glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
    try:
        meta = json.loads(jf.read_text())
    except Exception:
        continue
    if meta.get("hypothesis") == "H1" and meta.get("mode") == "pretrain":
        pt = jf.with_suffix(".pt")
        if pt.exists():
            print(pt)
            break
EOF
)

# Find the H1 supervised-baseline f=100% checkpoint (compound)
SUPERVISED_CKPT=$(python - <<'EOF'
from pathlib import Path
import json
for jf in sorted(Path("output/demo_checkpoints").glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
    try:
        meta = json.loads(jf.read_text())
    except Exception:
        continue
    if (meta.get("hypothesis") == "H1"
            and meta.get("mode") == "supervised_baseline"
            and abs(float(meta.get("labeled_fraction", 0)) - 1.0) < 0.01
            and meta.get("compound")):
        pt = jf.with_suffix(".pt")
        if pt.exists():
            print(pt)
            break
EOF
)

H3_CONFIG=$(mktemp /tmp/neuroseg_h3_XXXXXX)
cat > "$H3_CONFIG" <<YAML
h3_data_dir: data/demo
pretrained_ckpt: ${PRETRAINED_CKPT}
supervised_ckpt: ${SUPERVISED_CKPT}
seq_len: 5
img_size: 128
seed: 42
YAML

uv run main.py \
    --mode train \
    --H3 \
    --data data/demo \
    --output "$CHECKPOINTS_DIR" \
    --config "$H3_CONFIG"

rm -f "$H3_CONFIG"

# ── Step 5: Find best compound checkpoint and run inference ───────────────────
echo ""
echo "── Step 5/6  Running inference ───────────────────────────────────────────"

CHECKPOINT_PATH=$(python - <<'EOF'
import json, sys
from pathlib import Path

ckpt_dir = Path("output/demo_checkpoints")
candidates = []
for jf in sorted(ckpt_dir.glob("*.json"), key=lambda p: p.stat().st_mtime):
    try:
        meta = json.loads(jf.read_text())
    except Exception:
        continue
    if not meta.get("compound"):
        continue
    pt = jf.with_suffix(".pt")
    if pt.exists():
        candidates.append((meta.get("dice", 0.0), str(pt)))

if not candidates:
    print("", file=sys.stderr)
    sys.exit(1)

best = max(candidates, key=lambda t: t[0])
print(best[1])
EOF
)

if [ -z "$CHECKPOINT_PATH" ]; then
    echo "ERROR: no compound checkpoint found in $CHECKPOINTS_DIR" >&2
    exit 1
fi

echo "Using checkpoint: $CHECKPOINT_PATH"

uv run main.py \
    --mode inference \
    --data data/demo_stacks \
    --output "$INFERENCE_DIR" \
    --checkpoint "$CHECKPOINT_PATH"

# ── Step 6: Plot results ───────────────────────────────────────────────────────
echo ""
echo "── Step 6/6  Generating result figures ───────────────────────────────────"
uv run python scripts/plot_results.py \
    --output "$FIGURES_DIR" \
    --inference-output "$INFERENCE_DIR" \
    --logs "$CHECKPOINTS_DIR/logs/runs.csv"

echo ""
echo "════════════════════════════════════════════════════════"
echo "  Demo complete."
echo ""
echo "  Figure 1 (H1 Dice comparison):"
echo "    $FIGURES_DIR/h1_dice_comparison.png"
echo ""
echo "  Figure 2 (H1 mIoU comparison):"
echo "    $FIGURES_DIR/h1_miou_comparison.png"
echo ""
echo "  Figure 3 (H2 Dice comparison):"
echo "    $FIGURES_DIR/h2_dice_comparison.png"
echo ""
echo "  Figure 4 (H2 mIoU comparison):"
echo "    $FIGURES_DIR/h2_miou_comparison.png"
echo ""
echo "  Figure 5 (H3 temporal stability):"
echo "    $CHECKPOINTS_DIR/figures/h3_similarity.png"
echo ""
echo "  Figure 6 (Segmentation preview):"
echo "    $FIGURES_DIR/segmentation_preview.png"
echo ""
echo "  Per-run training curves (loss, Dice, mIoU, test scores):"
echo "    $CHECKPOINTS_DIR/figures/"
echo ""
echo "  Training logs:"
echo "    $CHECKPOINTS_DIR/logs/runs.csv"
echo "════════════════════════════════════════════════════════"

#!/usr/bin/env bash
# "Reproduce in 20 Minutes" demo.
# Run this from the NEUROSEG repository root.
#
# Usage:
#   bash demo.sh
#
# Requires: data/neurofinder.00.00/ (download from the Neurofinder benchmark).
#
# What it does (5 steps):
#   1. Prepare a 100-frame subset of neurofinder.00.00 as the demo dataset
#      (also creates 70/30 temporal splits for the H2 cross-domain experiment).
#   2. Train JEPA on that data (H1 — pretrain + finetune, scaled to ~10 min on CPU).
#   3. Train JEPA for cross-domain transfer (H2 — source pretrain + target finetune).
#   4. Run inference on a stacked TIFF and write segmentation output.
#   5. Produce result figures from the training and inference outputs.
#
# Output:
#   output/figures/h1_dice_comparison.png  — Dice vs labeled-data fraction (H1)
#   output/figures/h2_dice_comparison.png  — Dice: pretrained vs baseline (H2)
#   output/figures/segmentation_preview.png — raw frame + segmentation overlay

set -euo pipefail

CHECKPOINTS_DIR="output/demo_checkpoints"
INFERENCE_DIR="output/demo_inference"
FIGURES_DIR="output/figures"

# ── Step 1: Prepare real data subset ──────────────────────────────────────────
echo ""
echo "── Step 1/5  Preparing demo dataset (neurofinder.00.00, 100 frames) ─────"
uv run python scripts/prepare_demo_data.py \
    --source data/neurofinder.00.00 \
    --out data/demo \
    --stack-out data/demo_stacks \
    --frames 100 \
    --h2-source data/neurofinder.04.00 \
    --h2-source-out data/demo_h2_source

# ── Step 2: Train H1 (pretrain + finetune) ────────────────────────────────────
echo ""
echo "── Step 2/5  Training JEPA — H1 (pretrain + finetune) ───────────────────"
uv run main.py \
    --mode train \
    --H1 \
    --data data/demo \
    --output "$CHECKPOINTS_DIR" \
    --config config.demo.yaml

# ── Step 3: Train H2 (cross-domain transfer) ──────────────────────────────────
echo ""
echo "── Step 3/5  Training JEPA — H2 (cross-domain transfer) ─────────────────"
uv run main.py \
    --mode train \
    --H2 \
    --data data/demo \
    --output "$CHECKPOINTS_DIR" \
    --config config.demo.yaml

# ── Step 4: Find best compound checkpoint ─────────────────────────────────────
echo ""
echo "── Step 4/5  Running inference ───────────────────────────────────────────"

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

# ── Step 5: Plot results ───────────────────────────────────────────────────────
echo ""
echo "── Step 5/5  Generating result figures ───────────────────────────────────"
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
echo "  Figure 3 (Segmentation preview):"
echo "    $FIGURES_DIR/segmentation_preview.png"
echo ""
echo "  Per-run training curves (loss, Dice, mIoU, test scores):"
echo "    $CHECKPOINTS_DIR/figures/"
echo ""
echo "  Training logs:"
echo "    $CHECKPOINTS_DIR/logs/runs.csv"
echo "════════════════════════════════════════════════════════"

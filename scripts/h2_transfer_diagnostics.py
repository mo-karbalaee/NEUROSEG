import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from neuroseg.checkpoint import find_compound_checkpoint, load_compound_checkpoint
from neuroseg.metrics import dice
from neuroseg.trainers.dataset import NeurofinderDataset
from neuroseg.trainers.jepa import build_jepa, build_seg_head

THRESHOLDS = np.round(np.linspace(0.05, 0.95, 19), 3)


def _load_model(checkpoint_path: Path, device: torch.device):
    """Load a JEPA encoder + segmentation head from a compound checkpoint."""
    payload = load_compound_checkpoint(Path(checkpoint_path))
    arch = payload["arch"]
    jepa = build_jepa(arch, device)
    jepa.load_state_dict(payload["jepa"], strict=False)
    jepa.eval()
    seg_head = build_seg_head(arch["dstc"], arch.get("seg_head_hidden", 16)).to(device)
    seg_head.load_state_dict(payload["seg_head"])
    seg_head.eval()
    return jepa, seg_head, arch


@torch.inference_mode()
def _collect_probs(jepa, seg_head, dataset, device):
    """Return per-clip predicted probability maps and binary ground-truth masks on a dataset."""
    probs, gts = [], []
    for i in range(len(dataset)):
        sample = dataset[i]
        x = sample["video"].unsqueeze(0).to(device)
        enc = jepa.encoder(x).mean(dim=2)
        p = seg_head(enc).squeeze().cpu().numpy()
        g = (sample["mask"][0].numpy() > 0).astype(np.uint8)
        probs.append(p)
        gts.append(g)
    return probs, gts


def _adabn(jepa, dataset, device, batch_size=8):
    """Recompute the encoder's BatchNorm statistics on unlabeled target data (no labels, no gradient)."""
    for m in jepa.encoder.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.reset_running_stats()
            m.momentum = None
    jepa.encoder.train()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for batch in loader:
            jepa.encoder(batch["video"].to(device))
    jepa.encoder.eval()


def _dice_at(probs, gts, t):
    """Mean per-clip Dice at threshold t."""
    return float(np.mean([dice((p > t).astype(np.uint8), g) for p, g in zip(probs, gts)]))


def _best_threshold(probs, gts):
    """Return (best_threshold, best_mean_Dice, dice_curve) over the threshold sweep."""
    curve = [_dice_at(probs, gts, t) for t in THRESHOLDS]
    j = int(np.argmax(curve))
    return float(THRESHOLDS[j]), float(curve[j]), curve


def _auprc(probs, gts):
    """Threshold-free average precision over all pixels (NaN if no positive pixels)."""
    from sklearn.metrics import average_precision_score
    y = np.concatenate([g.ravel() for g in gts])
    s = np.concatenate([p.ravel() for p in probs])
    if y.sum() == 0:
        return float("nan"), 0.0
    return float(average_precision_score(y, s)), float(y.mean())


def _report(name, probs, gts):
    """Compute and return the diagnostic metrics for one model on the target."""
    d05 = _dice_at(probs, gts, 0.5)
    t_best, d_best, curve = _best_threshold(probs, gts)
    ap, prevalence = _auprc(probs, gts)
    print(f"  {name:24s} Dice@0.5={d05:.3f}  best-Dice={d_best:.3f} (t={t_best:.2f})  "
          f"AUPRC={ap:.3f} (chance={prevalence:.3f})")
    return {"dice05": d05, "t_best": t_best, "dice_best": d_best, "auprc": ap, "curve": curve}


def main():
    """Diagnose H2 zero-shot transfer: thresholding, AUPRC, and AdaBN on the target organism."""
    parser = argparse.ArgumentParser(description="H2 transfer diagnostics (threshold sweep, AUPRC, AdaBN).")
    parser.add_argument("--output", type=Path, required=True, help="Directory with the H2 compound checkpoints.")
    parser.add_argument("--target-data", type=Path, required=True, help="Target-organism Neurofinder directory.")
    parser.add_argument("--jepa-ckpt", type=Path, default=None, help="Explicit JEPA checkpoint (skips discovery).")
    parser.add_argument("--supervised-ckpt", type=Path, default=None, help="Explicit supervised checkpoint.")
    parser.add_argument("--seq-len", type=int, default=5)
    parser.add_argument("--out", type=Path, default=None, help="Output figure path.")
    args = parser.parse_args()

    import matplotlib.pyplot as plt

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    jepa_ckpt = args.jepa_ckpt or find_compound_checkpoint(args.output, "H2", "finetune")
    sup_ckpt = args.supervised_ckpt or find_compound_checkpoint(args.output, "H2", "supervised_baseline")
    if jepa_ckpt is None or sup_ckpt is None:
        raise SystemExit("Could not find both H2 checkpoints. Pass --jepa-ckpt / --supervised-ckpt explicitly.")

    models = [("JEPA-pretrained", jepa_ckpt), ("Supervised", sup_ckpt)]
    results = {}

    for name, ckpt in models:
        jepa, seg_head, arch = _load_model(ckpt, device)
        dataset = NeurofinderDataset(str(args.target_data), args.seq_len, arch.get("img_size", 128),
                                     labeled=True, binarize=True)
        if len(dataset) == 0:
            raise SystemExit("No clips found in --target-data.")

        print(f"\n[{name}]  ({Path(ckpt).name})")
        probs, gts = _collect_probs(jepa, seg_head, dataset, device)
        print("  --- as trained (source BatchNorm stats) ---")
        base = _report("zero-shot", probs, gts)

        _adabn(jepa, dataset, device)
        probs_a, gts_a = _collect_probs(jepa, seg_head, dataset, device)
        print("  --- after AdaBN (target BatchNorm stats) ---")
        ada = _report("zero-shot + AdaBN", probs_a, gts_a)

        results[name] = {"base": base, "ada": ada}

    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = {"JEPA-pretrained": "#2196F3", "Supervised": "#FF9800"}
    for name in results:
        c = colors[name]
        ax.plot(THRESHOLDS, results[name]["base"]["curve"], color=c, linewidth=2,
                label=f"{name} (AUPRC {results[name]['base']['auprc']:.2f})")
        ax.plot(THRESHOLDS, results[name]["ada"]["curve"], color=c, linewidth=2, linestyle="--",
                label=f"{name} + AdaBN (AUPRC {results[name]['ada']['auprc']:.2f})")
    ax.axvline(0.5, color="gray", linewidth=1, linestyle=":")
    ax.set_title("H2 — Zero-shot Transfer Diagnostics on Target Organism\nDice vs threshold (solid = as-trained, dashed = after AdaBN)", fontsize=12)
    ax.set_xlabel("Segmentation threshold")
    ax.set_ylabel("Mean Dice (target)")
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    out = args.out or (args.output / "figures" / "h2_transfer_diagnostics.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()

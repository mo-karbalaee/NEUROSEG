import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from neuroseg.checkpoint import find_compound_checkpoint, load_compound_checkpoint
from neuroseg.metrics import dice, miou
from neuroseg.trainers.dataset import NeurofinderDataset
from neuroseg.trainers.jepa import build_jepa, build_seg_head


def _load_encoder(checkpoint_path, device):
    """Load and freeze a JEPA encoder from a compound checkpoint; return (encoder, arch)."""
    payload = load_compound_checkpoint(Path(checkpoint_path))
    arch = payload["arch"]
    jepa = build_jepa(arch, device)
    jepa.load_state_dict(payload["jepa"], strict=False)
    encoder = jepa.encoder
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    return encoder, arch


@torch.inference_mode()
def _extract(encoder, dataset, device):
    """Precompute frozen encoder features (temporal mean) and binary masks for every clip."""
    feats, masks = [], []
    for i in range(len(dataset)):
        s = dataset[i]
        x = s["video"].unsqueeze(0).to(device)
        feats.append(encoder(x).mean(dim=2).squeeze(0).cpu())
        masks.append((s["mask"][0] > 0).float().unsqueeze(0))
    return feats, masks


def _train_probe(feats, masks, idx, dstc, hidden, epochs, lr, device):
    """Train a fresh segmentation head on frozen features (linear probe)."""
    X = torch.stack([feats[i] for i in idx]).to(device)
    Y = torch.stack([masks[i] for i in idx]).to(device)
    head = build_seg_head(dstc, hidden).to(device)
    opt = Adam(head.parameters(), lr=lr)
    crit = nn.BCELoss()
    bs = 8
    for _ in range(epochs):
        perm = torch.randperm(len(X))
        for j in range(0, len(X), bs):
            b = perm[j:j + bs]
            loss = crit(head(X[b]), Y[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
    head.eval()
    return head


@torch.inference_mode()
def _eval_probe(head, feats, masks, idx, device):
    """Mean test Dice / mIoU of a trained probe head over the held-out clips."""
    ds, ms = [], []
    for i in idx:
        pred = head(feats[i].unsqueeze(0).to(device)).squeeze().cpu().numpy()
        pb = (pred > 0.5).astype(np.uint8)
        gt = masks[i].squeeze().numpy().astype(np.uint8)
        ds.append(dice(pb, gt))
        ms.append(miou(pb, gt, num_classes=2))
    return float(np.mean(ds)), float(np.mean(ms))


def main():
    """Linear-probe transfer: freeze each source encoder, train a head on a mouse fraction, eval on held-out mouse."""
    parser = argparse.ArgumentParser(description="H2 target linear probe (few-shot transfer of frozen encoder features).")
    parser.add_argument("--output", type=Path, required=True, help="Directory with the H2 compound checkpoints.")
    parser.add_argument("--target-data", type=Path, required=True, help="Target-organism Neurofinder directory.")
    parser.add_argument("--fractions", nargs="+", type=float, default=[0.01, 0.05, 0.1])
    parser.add_argument("--probe-epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seq-len", type=int, default=5)
    parser.add_argument("--test-split", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    import matplotlib.pyplot as plt

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    checkpoints = {
        "JEPA-pretrained": find_compound_checkpoint(args.output, "H2", "finetune"),
        "Supervised": find_compound_checkpoint(args.output, "H2", "supervised_baseline"),
    }
    if any(v is None for v in checkpoints.values()):
        raise SystemExit("Could not find both H2 checkpoints in --output.")

    results = {}
    split_idx = None
    for name, ckpt in checkpoints.items():
        encoder, arch = _load_encoder(ckpt, device)
        dataset = NeurofinderDataset(str(args.target_data), args.seq_len, arch.get("img_size", 128),
                                     labeled=True, binarize=True)
        if len(dataset) < 4:
            raise SystemExit("Too few target clips for a probe.")

        if split_idx is None:
            rng = np.random.default_rng(args.seed)
            order = rng.permutation(len(dataset))
            n_test = max(1, int(args.test_split * len(dataset)))
            split_idx = {"test": order[:n_test], "pool": order[n_test:]}
            print(f"Target clips: {len(dataset)} | probe-train pool: {len(split_idx['pool'])} | held-out test: {len(split_idx['test'])}")

        print(f"\n[{name}]  ({Path(ckpt).name})  — extracting frozen features...")
        feats, masks = _extract(encoder, dataset, device)

        results[name] = {"dice": [], "miou": []}
        for frac in args.fractions:
            k = max(1, int(frac * len(split_idx["pool"])))
            train_idx = split_idx["pool"][:k]
            head = _train_probe(feats, masks, train_idx, arch["dstc"],
                                arch.get("seg_head_hidden", 16), args.probe_epochs, args.lr, device)
            d, m = _eval_probe(head, feats, masks, split_idx["test"], device)
            results[name]["dice"].append(d)
            results[name]["miou"].append(m)
            print(f"  frac={frac:>5.0%}  (n_train={k:3d})  test Dice={d:.3f}  mIoU={m:.3f}")

    fracs_pct = [f * 100 for f in args.fractions]
    colors = {"JEPA-pretrained": "#2196F3", "Supervised": "#FF9800"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("H2 — Target Linear-Probe (few-shot transfer of frozen encoder features)", fontsize=12)
    for ax, metric, title in [(axes[0], "dice", "Dice"), (axes[1], "miou", "mIoU")]:
        for name in results:
            ax.plot(fracs_pct, results[name][metric], marker="o", color=colors[name], linewidth=2, label=name)
            for xf, yv in zip(fracs_pct, results[name][metric]):
                ax.text(xf, yv + 0.02, f"{yv:.2f}", color=colors[name], ha="center", va="bottom", fontsize=8)
        ax.set_title(f"Target {title} vs labeled fraction")
        ax.set_xlabel("Mouse labels used for the probe (%)")
        ax.set_ylabel(f"{title} (held-out target)")
        ax.set_ylim(0, 1.0)
        ax.legend(fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    plt.tight_layout()

    out = args.out or (args.output / "figures" / "h2_linear_probe.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()

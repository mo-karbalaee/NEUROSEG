import argparse
from pathlib import Path

import numpy as np
import torch

from neuroseg.checkpoint import find_compound_checkpoint, load_compound_checkpoint
from neuroseg.trainers.dataset import NeurofinderDataset
from neuroseg.trainers.jepa import build_jepa


def _load_encoder(checkpoint_path: Path, device: torch.device):
    """Load only the JEPA encoder (and arch) from a compound checkpoint."""
    payload = load_compound_checkpoint(checkpoint_path)
    arch = payload["arch"]
    jepa = build_jepa(arch, device)
    jepa.load_state_dict(payload["jepa"], strict=False)
    jepa.eval()
    return jepa, arch


@torch.inference_mode()
def _neuron_embeddings(jepa, dataset, device: torch.device):
    """Pool per-neuron encoder embeddings across frames; return (embeddings, neuron_ids)."""
    embs, ids = [], []
    for i in range(len(dataset)):
        sample = dataset[i]
        mask = sample["mask"].numpy().astype(np.int64)
        x = sample["video"].unsqueeze(0).to(device)
        enc = jepa.encoder(x).squeeze(0).cpu().numpy()
        T = enc.shape[1]
        for n in np.unique(mask):
            if n <= 0:
                continue
            feats = []
            for t in range(T):
                region = mask[t] == n
                if region.any():
                    feats.append(enc[:, t, region].mean(axis=-1))
            if feats:
                embs.append(np.mean(feats, axis=0))
                ids.append(int(n))
    return np.array(embs), np.array(ids)


def main():
    """Project per-neuron encoder features to 2-D (PCA) for JEPA-transfer vs from-scratch encoders."""
    parser = argparse.ArgumentParser(description="H2 target-organism feature-space map (PCA of per-neuron embeddings).")
    parser.add_argument("--output", type=Path, required=True, help="Directory with the H2 compound checkpoints.")
    parser.add_argument("--target-data", type=Path, required=True, help="Target-organism Neurofinder directory.")
    parser.add_argument("--seq-len", type=int, default=5)
    parser.add_argument("--out", type=Path, default=None, help="Output figure path.")
    args = parser.parse_args()

    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ft_ckpt = find_compound_checkpoint(args.output, "H2", "finetune")
    bl_ckpt = find_compound_checkpoint(args.output, "H2", "supervised_baseline")
    if ft_ckpt is None or bl_ckpt is None:
        raise SystemExit("Could not find both H2 finetune and supervised_baseline checkpoints in --output.")

    jepa_ft, arch = _load_encoder(ft_ckpt, device)
    jepa_bl, _ = _load_encoder(bl_ckpt, device)

    img_size = arch.get("img_size", 128)
    dataset = NeurofinderDataset(str(args.target_data), args.seq_len, img_size,
                                 labeled=True, binarize=False)
    if len(dataset) == 0:
        raise SystemExit("No clips found in --target-data.")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    fig.suptitle(
        "H2 — Target-neuron Feature Space (PCA)\nsame color = same neuron; tighter clusters = more discriminative features",
        fontsize=12,
    )

    for ax, (jepa, title) in zip(axes, [(jepa_ft, "JEPA-transfer encoder"), (jepa_bl, "From-scratch encoder")]):
        embs, ids = _neuron_embeddings(jepa, dataset, device)
        if len(embs) < 2:
            ax.text(0.5, 0.5, "Not enough neurons", ha="center", va="center", transform=ax.transAxes, color="gray")
            ax.set_title(title)
            continue
        coords = PCA(n_components=2).fit_transform(embs)
        ax.scatter(coords[:, 0], coords[:, 1], c=ids, cmap="tab20", s=28, alpha=0.85, edgecolors="none")
        ax.set_title(title)
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out = args.out or (args.output / "figures" / "h2_feature_map.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()

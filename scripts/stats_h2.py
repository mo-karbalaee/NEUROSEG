import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats

from neuroseg.checkpoint import load_compound_checkpoint
from neuroseg.metrics import dice as dice_fn
from neuroseg.trainers.dataset import NeurofinderDataset
from neuroseg.trainers.h1_trainer import build_config
from neuroseg.trainers.jepa import build_jepa, build_seg_head


def mean_ci(a, z=1.96):
    a = np.asarray(a, dtype=np.float64)
    return a.mean(), z * a.std(ddof=1) / np.sqrt(len(a))


def test_split_indices(dataset, seed, test_split):
    n = len(dataset)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist()
    return perm[:max(1, int(n * test_split))]


@torch.inference_mode()
def per_clip_dice(ckpt, dataset, idx, device):
    p = load_compound_checkpoint(Path(ckpt))
    arch = p["arch"]; thr = arch.get("seg_threshold", 0.5)
    jepa = build_jepa(arch, device); jepa.load_state_dict(p["jepa"], strict=False); jepa.eval()
    head = build_seg_head(arch["dstc"], arch.get("seg_head_hidden", 16)).to(device)
    head.load_state_dict(p["seg_head"]); head.eval()
    out = []
    for i in idx:
        s = dataset[int(i)]
        pred = head(jepa.encoder(s["video"].unsqueeze(0).to(device)).mean(dim=2))
        pred = F.interpolate(pred, size=s["mask"].shape[-2:], mode="bilinear", align_corners=False)
        pb = (pred.squeeze().cpu().numpy() > thr).astype(np.uint8)
        gt = (s["mask"][0].numpy() > 0).astype(np.uint8)
        out.append(dice_fn(pb, gt))
    return np.array(out)


def main():
    """H2 significance: paired per-clip Dice on the mouse target for cross-species SSL,
    same-species SSL, and from-scratch. Tests (a) cross vs same = the cross-species penalty,
    (b) cross vs from-scratch = does cross-species SSL help at all."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/neurofinder.00.00"))
    ap.add_argument("--cross", type=Path, required=True, help="H2 cross-species finetune ckpt")
    ap.add_argument("--same", type=Path, required=True, help="H1 same-species finetune ckpt")
    ap.add_argument("--scratch", type=Path, required=True, help="from-scratch supervised ckpt")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = build_config({"config": {"img_size": 128, "seq_len": 5, "dstc": 32, "seed": 1, "test_split": 0.2}})
    ds = NeurofinderDataset(str(args.data), cfg.seq_len, cfg.img_size, labeled=True,
                            labeled_fraction=1.0, seed=cfg.seed, binarize=True)
    idx = test_split_indices(ds, cfg.seed, cfg.test_split)

    cross = per_clip_dice(str(args.cross), ds, idx, device)
    same = per_clip_dice(str(args.same), ds, idx, device)
    scratch = per_clip_dice(str(args.scratch), ds, idx, device)
    mc, cc = mean_ci(cross); ms, cs = mean_ci(same); msc, csc = mean_ci(scratch)

    print(f"[H2] per-clip Dice on mouse, n={len(idx)} paired clips")
    print(f"    cross-species SSL : {mc:.3f} ± {cc:.3f}")
    print(f"    same-species SSL  : {ms:.3f} ± {cs:.3f}")
    print(f"    from-scratch      : {msc:.3f} ± {csc:.3f}")
    _, p_pen = stats.wilcoxon(cross, same)
    _, p_help = stats.wilcoxon(cross, scratch)
    print(f"    cross vs same  (penalty?): p={p_pen:.4g}  diff {mc-ms:+.3f}"
          f"  -> {'penalty' if p_pen < 0.05 else 'NO significant penalty'}")
    print(f"    cross vs scratch (helps?): p={p_help:.4g}  diff {mc-msc:+.3f}"
          f"  -> {'differs' if p_help < 0.05 else 'no significant difference'}")


if __name__ == "__main__":
    main()

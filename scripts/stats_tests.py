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
from neuroseg.trainers.h3_trainer import _load_encoder
from neuroseg.trainers.jepa import build_jepa, build_seg_head


def _mean_ci(a, z=1.96):
    a = np.asarray(a, dtype=np.float64)
    m = a.mean()
    se = a.std(ddof=1) / np.sqrt(len(a))
    return m, z * se


def _test_split_indices(dataset, seed, test_split):
    n = len(dataset)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist()
    return perm[:max(1, int(n * test_split))]


@torch.inference_mode()
def _per_clip_dice(ckpt, dataset, idx, device):
    p = load_compound_checkpoint(Path(ckpt))
    arch = p["arch"]; thr = arch.get("seg_threshold", 0.5)
    jepa = build_jepa(arch, device); jepa.load_state_dict(p["jepa"], strict=False); jepa.eval()
    head = build_seg_head(arch["dstc"], arch.get("seg_head_hidden", 16)).to(device)
    head.load_state_dict(p["seg_head"]); head.eval()
    out = []
    for i in idx:
        s = dataset[int(i)]
        x = s["video"].unsqueeze(0).to(device)
        pred = head(jepa.encoder(x).mean(dim=2))
        pred = F.interpolate(pred, size=s["mask"].shape[-2:], mode="bilinear", align_corners=False)
        pb = (pred.squeeze().cpu().numpy() > thr).astype(np.uint8)
        gt = (s["mask"][0].numpy() > 0).astype(np.uint8)
        out.append(dice_fn(pb, gt))
    return np.array(out)


def h1_paired(target, jepa_ckpt, sup_ckpt, cfg, device, label):
    ds = NeurofinderDataset(target, cfg.seq_len, cfg.img_size, labeled=True,
                            labeled_fraction=1.0, seed=cfg.seed, binarize=True)
    idx = _test_split_indices(ds, cfg.seed, cfg.test_split)
    dj = _per_clip_dice(jepa_ckpt, ds, idx, device)
    dsup = _per_clip_dice(sup_ckpt, ds, idx, device)
    w, p = stats.wilcoxon(dj, dsup)
    mj, cj = _mean_ci(dj); ms, cs = _mean_ci(dsup)
    print(f"[H1 {label}] per-clip Dice, n={len(idx)} paired clips")
    print(f"    JEPA-pretrained : {mj:.3f} ± {cj:.3f}")
    print(f"    from-scratch    : {ms:.3f} ± {cs:.3f}")
    print(f"    paired Wilcoxon : p={p:.4g}   mean diff {mj-ms:+.3f}"
          f"   -> {'significant' if p < 0.05 else 'NOT significant'} at 0.05")


@torch.inference_mode()
def _within_between(ckpt, dataset, cfg, device, max_clips):
    jepa = _load_encoder(ckpt, cfg, device); jepa.eval()
    n = len(dataset)
    indices = np.linspace(0, n - 1, min(max_clips, n)).astype(int) if max_clips else np.arange(n)
    per_neuron, per_frame = {}, []
    for ci in indices:
        s = dataset[int(ci)]
        enc = jepa.encoder(s["video"].unsqueeze(0).to(device)).squeeze(0).cpu().numpy()
        mask = s["mask"].numpy().astype(np.int64)
        for t in range(enc.shape[1]):
            fe = []
            for nid in np.unique(mask[t]):
                if nid <= 0:
                    continue
                region = mask[t] == nid
                if region.sum() == 0:
                    continue
                v = enc[:, t, region].mean(axis=-1)
                per_neuron.setdefault(int(nid), []).append(v)
                fe.append(v)
            if len(fe) >= 2:
                per_frame.append(np.stack(fe))

    def sims(vecs):
        v = np.asarray(vecs, np.float32)
        v = v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-8)
        s = v @ v.T
        iu = np.triu_indices(len(v), k=1)
        return s[iu]

    within = np.concatenate([sims(v) for v in per_neuron.values() if len(v) >= 2])
    between = np.concatenate([sims(f) for f in per_frame])
    return within, between


def _cohens_d(a, b):
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return (a.mean() - b.mean()) / (sp + 1e-12)


def h3_test(mode, ckpt, dataset, cfg, device, max_clips, rng):
    within, between = _within_between(ckpt, dataset, cfg, device, max_clips)
    u, p = stats.mannwhitneyu(within, between, alternative="greater")
    d = _cohens_d(within, between)
    gaps = np.array([within[rng.integers(0, len(within), len(within))].mean()
                     - between[rng.integers(0, len(between), len(between))].mean() for _ in range(1000)])
    lo, hi = np.percentile(gaps, [2.5, 97.5])
    print(f"[H3 {mode}] within={within.mean():.3f} between={between.mean():.3f} "
          f"gap={within.mean()-between.mean():.3f} (95% CI [{lo:.3f}, {hi:.3f}])")
    print(f"    Mann-Whitney within>between: p={p:.3g} | Cohen's d={d:.2f} "
          f"| {'significant' if p < 0.05 else 'NOT significant'}")


def main():
    ap = argparse.ArgumentParser(description="Statistical tests for H1 (per-clip Dice) and H3 (within/between).")
    ap.add_argument("--data", type=Path, default=Path("data/neurofinder.00.00"))
    ap.add_argument("--v9", type=Path, default=Path("output/H1.v9"))
    ap.add_argument("--max-clips", type=int, default=40)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = build_config({"config": {"img_size": 128, "seq_len": 5, "dstc": 32, "seed": 1, "test_split": 0.2}})
    rng = np.random.default_rng(0)

    def g(pat):
        return str(next(args.v9.glob(pat)))

    def g_pretrain():
        cands = [p for p in args.v9.glob("jepa_pretrained_h1_*.pt") if "_latest" not in p.name]
        return str(cands[0])

    print("=== H1: JEPA-pretrained vs from-scratch (paired per-clip Dice) ===")
    h1_paired(str(args.data), g("jepa_h1_finetune_f100_*.pt"), g("jepa_h1_supervised_f100_*.pt"), cfg, device, "100%")
    h1_paired(str(args.data), g("jepa_h1_finetune_f10_*.pt"), g("jepa_h1_supervised_f10_*.pt"), cfg, device, "10%")

    print("\n=== H3: within- vs between-neuron similarity (per encoder) ===")
    ds = NeurofinderDataset(str(args.data), cfg.seq_len, cfg.img_size, labeled=True,
                            labeled_fraction=1.0, seed=cfg.seed, binarize=False)
    h3_test("pretrained (SSL)", g_pretrain(), ds, cfg, device, args.max_clips, rng)
    h3_test("supervised", g("jepa_h1_supervised_f100_*.pt"), ds, cfg, device, args.max_clips, rng)
    h3_test("random (no_pretrain)", None, ds, cfg, device, args.max_clips, rng)


if __name__ == "__main__":
    main()

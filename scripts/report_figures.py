import argparse
import csv
import glob
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, Ellipse, FancyBboxPatch, Circle
import matplotlib.patheffects as pe

BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
VERM = "#D55E00"
GREY = "#7a7a7a"
INK = "#222222"

_H3_LABELS = {
    "pretrained_crossspecies": "Cross-species\nSSL",
    "pretrained": "Same-species\nSSL",
    "supervised_baseline": "Supervised",
}
_H3_ORDER = ["pretrained_crossspecies", "pretrained", "supervised_baseline"]


def _style() -> None:
    """Apply the shared serif plotting style used across the report figures."""
    plt.rcParams.update({
        "font.size": 11, "font.family": "serif", "figure.dpi": 150,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6, "axes.axisbelow": True,
    })


def _epoch_series(rows: list, mode: str, col: str, fraction: str | None = None) -> tuple:
    """Return sorted (epochs, values) for a per-epoch column of one training mode."""
    out = []
    for r in rows:
        if r.get("mode") != mode:
            continue
        if fraction is not None and r.get("labeled_fraction") != fraction:
            continue
        if r.get("epoch", "") == "" or r.get(col, "") == "":
            continue
        out.append((int(r["epoch"]), float(r[col])))
    out.sort()
    return [e for e, _ in out], [v for _, v in out]


def training_curves(h1_log: Path, out_dir: Path) -> None:
    """Plot pretraining train-vs-held-out loss and fine-tuning validation Dice from the H1 log."""
    _style()
    with open(h1_log, newline="") as f:
        rows = list(csv.DictReader(f))
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(10.6, 3.9))

    e, tr = _epoch_series(rows, "pretrain", "train_loss")
    _, va = _epoch_series(rows, "pretrain", "val_jepa_loss")
    axL.plot(e, tr, "-o", color=BLUE, lw=2, ms=5, label="training loss")
    axL.plot(e, va, "-s", color=VERM, lw=2, ms=5, label="held-out loss")
    axL.set_xlabel("Pretraining epoch")
    axL.set_ylabel("JEPA prediction loss")
    axL.set_title("Self-supervised pretraining", fontsize=10.5)
    axL.legend(frameon=False, fontsize=9.5, loc="center right")

    e1, d1 = _epoch_series(rows, "finetune", "val_dice", "1.0")
    e2, d2 = _epoch_series(rows, "supervised_baseline", "val_dice", "1.0")
    axR.plot(e1, d1, "-", color=BLUE, lw=2, label="SSL-pretrained")
    axR.plot(e2, d2, "-", color=ORANGE, lw=2, label="from scratch")
    axR.set_xlabel("Fine-tuning epoch")
    axR.set_ylabel("Validation Dice")
    axR.set_ylim(0.2, 0.95)
    axR.set_title("Fine-tuning (100\\% labels)", fontsize=10.5)
    axR.legend(frameon=False, fontsize=9.5, loc="lower right")

    fig.tight_layout()
    _save(fig, out_dir, "training-curves")


def h3_stability(h3_log: Path, out_dir: Path) -> None:
    """Plot within/between similarity and the separation gap per encoder from the H3 log."""
    _style()
    data = {}
    with open(h3_log, newline="") as f:
        for r in csv.DictReader(f):
            if r.get("hypothesis") != "H3":
                continue
            m = r.get("mode", "")
            if m in _H3_ORDER and r.get("within_sim", "") != "":
                data[m] = (float(r["within_sim"]), float(r["between_sim"]), float(r["gap"]))
    modes = [m for m in _H3_ORDER if m in data]
    labels = [_H3_LABELS[m] for m in modes]
    within = [data[m][0] for m in modes]
    between = [data[m][1] for m in modes]
    gap = [data[m][2] for m in modes]
    gapcolors = [BLUE, GREEN, VERM][:len(modes)]
    x = np.arange(len(modes))

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(10.6, 3.8))
    w = 0.38
    axL.bar(x - w / 2, within, w, color=BLUE, label="Within-neuron")
    axL.bar(x + w / 2, between, w, color=ORANGE, label="Between-neuron")
    axL.set_xticks(x)
    axL.set_xticklabels(labels, fontsize=8.8)
    axL.set_ylabel("Mean cosine similarity")
    axL.set_ylim(0.5, 1.0)
    axL.legend(frameon=False, fontsize=9.2, loc="lower left")
    axL.set_title("Within- vs between-neuron similarity", fontsize=10.5)

    bars = axR.bar(x, gap, width=0.62, color=gapcolors)
    for b, g in zip(bars, gap):
        axR.text(b.get_x() + b.get_width() / 2, g + 0.004, f"{g:.3f}", ha="center", va="bottom", fontsize=9)
    axR.set_xticks(x)
    axR.set_xticklabels(labels, fontsize=8.8)
    axR.set_ylabel("Gap (within $-$ between)")
    axR.set_ylim(0, max(gap) * 1.15)
    axR.set_title("Separation gap", fontsize=10.5)

    fig.tight_layout()
    _save(fig, out_dir, "h3-stability")


def h3_within_between(data_dir: Path, out_dir: Path) -> None:
    """Illustrate within- vs between-neuron comparisons on two real calcium frames."""
    from neuroseg.trainers.dataset import _build_nf_mask
    import tifffile as tiff
    _style()

    nf = Path(data_dir)
    imgs = sorted(glob.glob(str(nf / "images" / "*.tif*")))
    H, W = tiff.imread(imgs[0]).shape[:2]
    mask = _build_nf_mask(nf, H, W, max(H, W))[:H, :W]
    ids = [i for i in np.unique(mask) if i > 0]
    cent = {i: (np.where(mask == i)[1].mean(), np.where(mask == i)[0].mean(), (mask == i).sum()) for i in ids}
    cand = [i for i in ids if 90 < cent[i][0] < W - 90 and 90 < cent[i][1] < H - 90 and cent[i][2] > 50]
    A = max(cand, key=lambda i: cent[i][2])
    ax_, ay_, _ = cent[A]
    B = max([i for i in cand if i != A and np.hypot(cent[i][0] - ax_, cent[i][1] - ay_) > 150],
            key=lambda i: cent[i][2])

    sample = np.linspace(0, len(imgs) - 1, 300).astype(int)
    mA, mB = (mask == A), (mask == B)
    tA = np.array([tiff.imread(imgs[s]).astype(np.float32)[mA].mean() for s in sample])
    tB = np.array([tiff.imread(imgs[s]).astype(np.float32)[mB].mean() for s in sample])
    t1 = int(sample[np.argmax(tA - tB)])
    t2 = int(sample[np.argmax(tB - tA)])

    def norm(fr):
        lo, hi = np.percentile(fr, 1), np.percentile(fr, 99.5)
        return np.clip((fr - lo) / (hi - lo + 1e-8), 0, 1)

    F1 = norm(tiff.imread(imgs[t1]).astype(np.float32))
    F2 = norm(tiff.imread(imgs[t2]).astype(np.float32))

    fig, ax = plt.subplots(figsize=(9.6, 4.7))
    S, GAP = 4.0, 0.7
    ext1, ext2 = (0, S, 0, S), (S + GAP, 2 * S + GAP, 0, S)
    ax.imshow(F1, cmap="gray", extent=ext1, origin="upper", vmin=0, vmax=1)
    ax.imshow(F2, cmap="gray", extent=ext2, origin="upper", vmin=0, vmax=1)
    ax.set_xlim(-0.2, 2 * S + GAP + 0.2)
    ax.set_ylim(-1.5, S + 1.8)
    ax.axis("off")
    ax.set_aspect("equal")

    def px2d(cx, cy, x0):
        return x0 + cx / W * S, S - cy / H * S

    for ext, x0 in [(ext1, 0.0), (ext2, S + GAP)]:
        for nid, col, lab, side in [(A, BLUE, "A", "left"), (B, ORANGE, "B", "right")]:
            ax.contour((mask == nid).astype(float), levels=[0.5], colors=[col], linewidths=1.6,
                       extent=ext, origin="upper")
            cx, cy, _ = cent[nid]
            dx, dy = px2d(cx, cy, x0)
            ax.add_patch(Circle((dx, dy), 0.32, fill=False, ec=col, lw=2.2))
            ox, ha = (-0.46, "right") if side == "left" else (0.46, "left")
            t = ax.text(dx + ox, dy + 0.30, lab, color=col, fontsize=12, weight="bold", ha=ha)
            t.set_path_effects([pe.withStroke(linewidth=2.4, foreground="black")])

    ax.text(S / 2, -0.35, "frame $t_1$", ha="center", va="top", fontsize=11, color=INK)
    ax.text(S + GAP + S / 2, -0.35, "frame $t_2$", ha="center", va="top", fontsize=11, color=INK)

    cxc = (2 * S + GAP) / 2
    a1 = px2d(*cent[A][:2], 0.0)
    a2 = px2d(*cent[A][:2], S + GAP)
    b2 = px2d(*cent[B][:2], S + GAP)
    ax.add_patch(FancyArrowPatch((a1[0] + 0.28, a1[1]), (a2[0] - 0.28, a2[1]),
                 connectionstyle="arc3,rad=-0.12", arrowstyle="<->", mutation_scale=16, lw=2.4, color=GREEN))
    ax.add_patch(FancyArrowPatch(a2, b2, connectionstyle="arc3,rad=0.22",
                 arrowstyle="<->", mutation_scale=16, lw=2.4, color=VERM))
    ax.text(cxc, S + 1.28, "within-neuron: same neuron, different frames",
            ha="center", fontsize=10.5, color=GREEN, weight="bold")
    ax.text(cxc, S + 0.72, "between-neuron: different neurons, same frame",
            ha="center", fontsize=10.5, color=VERM, weight="bold")
    ax.text(cxc, -1.2,
            r"a good encoder: within-neuron similarity high   $>$   between-neuron similarity",
            ha="center", fontsize=10, color=INK)

    fig.tight_layout()
    _save(fig, out_dir, "h3-within-between")


def h3_embedding_pool(out_dir: Path) -> None:
    """Draw the masked-average-pooling schematic that forms a per-neuron embedding."""
    _style()
    rng = np.random.default_rng(3)
    fig, ax = plt.subplots(figsize=(9.6, 3.4))
    ax.set_xlim(0, 15.2)
    ax.set_ylim(0.55, 6.05)
    ax.axis("off")
    ax.set_aspect("equal")

    ax.add_patch(Rectangle((0.1, 2.6), 1.7, 1.7, facecolor="#101820", edgecolor=INK, lw=1.0))
    ax.add_patch(Ellipse((0.7, 3.6), 0.55, 0.42, facecolor="#8fd3ff", edgecolor="none"))
    ax.add_patch(Ellipse((1.25, 3.1), 0.5, 0.4, facecolor="#c7e9ff", edgecolor="none"))
    ax.text(0.95, 2.35, "frame", ha="center", va="top", fontsize=9, color=INK)
    ax.add_patch(FancyArrowPatch((1.9, 3.45), (2.5, 3.45), arrowstyle="-|>", mutation_scale=13, lw=1.6, color=INK))

    for dx, col in [(0.2, VERM), (0.0, BLUE)]:
        ax.add_patch(FancyBboxPatch((2.7 + dx, 2.6 + (0.2 - dx)), 1.7, 1.9, boxstyle="round,pad=0.04",
                                    facecolor="white", edgecolor=col, lw=1.8))
    ax.text(3.55, 3.5, "encoder", ha="center", va="center", fontsize=9.5, color=INK)
    ax.add_patch(Rectangle((2.75, 5.15), 0.22, 0.22, facecolor=BLUE))
    ax.text(3.05, 5.26, "SSL", va="center", fontsize=7.6, color=BLUE)
    ax.add_patch(Rectangle((4.05, 5.15), 0.22, 0.22, facecolor=VERM))
    ax.text(4.35, 5.26, "sup.", va="center", fontsize=7.6, color=VERM)
    ax.text(3.7, 2.15, "same architecture, 2 weight-sets", ha="center", va="top", fontsize=8.2, color=GREY, style="italic")
    ax.add_patch(FancyArrowPatch((5.05, 3.45), (5.7, 3.45), arrowstyle="-|>", mutation_scale=13, lw=1.6, color=INK))

    cw, N, x0g, y0g = 0.62, 6, 5.9, 1.6
    for i in range(N):
        for j in range(N):
            ax.add_patch(Rectangle((x0g + i * cw, y0g + j * cw), cw, cw, facecolor="#eef1f4", edgecolor="#c9d1d9", lw=0.6))
    ax.text(x0g + N * cw / 2, y0g + N * cw + 0.18, "feature map (32 / location)", ha="center", fontsize=8.6, color=GREY, style="italic")
    for (i, j) in {(1, 2), (2, 2), (3, 2), (1, 3), (2, 3), (3, 3), (2, 4), (2, 1)}:
        ax.add_patch(Rectangle((x0g + i * cw, y0g + j * cw), cw, cw, facecolor=BLUE, alpha=0.34, edgecolor=BLUE, lw=1.0))
    ax.text(x0g + 2 * cw, y0g - 0.15, "footprint", ha="center", va="top", fontsize=8.2, color=BLUE)

    xend = x0g + N * cw
    ax.add_patch(FancyArrowPatch((xend + 0.05, 3.45), (xend + 0.85, 3.45), arrowstyle="-|>", mutation_scale=13, lw=1.6, color=INK))
    ax.text(xend + 0.45, 3.95, "average", ha="center", fontsize=8.6, color=INK)
    ax.text(xend + 0.45, 2.95, r"$\mathbf{e}=\frac{1}{|F|}\sum_{p\in F}\mathbf{f}_p$", ha="center", fontsize=9.5, color=INK)
    vx = xend + 0.95
    cols = plt.get_cmap("viridis")(np.linspace(0, 1, 6))
    for k, c in enumerate(cols):
        ax.add_patch(Rectangle((vx + k * 0.5, 3.2), 0.5, 0.5, facecolor=c, edgecolor="white", lw=0.5))
    ax.add_patch(Rectangle((vx, 3.2), 6 * 0.5, 0.5, fill=False, edgecolor=INK, lw=1.2))
    ax.text(vx + 6 * 0.5 / 2, 3.1, r"$\mathbf{e}\in\mathbb{R}^{32}$", ha="center", va="top", fontsize=9)
    ax.text(vx + 6 * 0.5 / 2, 3.75, "embedding", ha="center", va="bottom", fontsize=8.4, color=GREY, style="italic")

    fig.tight_layout()
    _save(fig, out_dir, "h3-embedding-pool")


def _save(fig, out_dir: Path, name: str) -> None:
    """Write a figure as both PNG and PDF into out_dir and close it."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_dir / f"{name}.png"), bbox_inches="tight")
    fig.savefig(str(out_dir / f"{name}.pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_dir / name}.png")


def main() -> None:
    """Regenerate the report figures from logged training results and Neurofinder data."""
    parser = argparse.ArgumentParser(description="Regenerate NEUROSEG report figures.")
    parser.add_argument("--h1-log", type=Path, default=Path("output/H1.v9/logs/runs.csv"))
    parser.add_argument("--h3-log", type=Path, default=Path("output/H3_v9/logs/runs.csv"))
    parser.add_argument("--data", type=Path, default=Path("data/neurofinder.00.00"))
    parser.add_argument("--output", type=Path, default=Path("output/report_figures"))
    parser.add_argument("--which", nargs="+",
                        default=["training-curves", "h3-stability", "h3-within-between", "h3-embedding"],
                        choices=["training-curves", "h3-stability", "h3-within-between", "h3-embedding"])
    args = parser.parse_args()

    if "training-curves" in args.which and args.h1_log.exists():
        training_curves(args.h1_log, args.output)
    if "h3-stability" in args.which and args.h3_log.exists():
        h3_stability(args.h3_log, args.output)
    if "h3-within-between" in args.which and args.data.exists():
        h3_within_between(args.data, args.output)
    if "h3-embedding" in args.which:
        h3_embedding_pool(args.output)


if __name__ == "__main__":
    main()

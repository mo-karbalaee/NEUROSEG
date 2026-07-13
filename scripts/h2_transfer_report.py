import argparse
import csv
from pathlib import Path


def _read_finetune_dice(log_path: Path, hypothesis: str, mode: str) -> dict:
    """Return {fraction: test_dice} for the given hypothesis+mode summary rows in a runs.csv."""
    out: dict[float, float] = {}
    if not log_path or not Path(log_path).exists():
        return out
    with open(log_path, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("hypothesis", "") != hypothesis or row.get("mode", "") != mode:
                continue
            frac, dice = row.get("labeled_fraction", ""), row.get("test_dice", "")
            if frac and dice:
                try:
                    out[float(frac)] = float(dice)
                except ValueError:
                    pass
    return out


def main():
    """
    Answer H2: does self-supervised pretraining transfer across species?

    Overlays three arms (all fine-tuned on the SAME mouse target/split) — from-scratch,
    cross-species SSL (Drosophila+zebrafish), and same-species SSL (mouse) — and reports
    the per-fraction transfer ratio (cross_gain / same_gain), where gain = arm − from-scratch.
    Ratio ~1 = SSL fully transfers across species; ~0 = it does not.
    """
    parser = argparse.ArgumentParser(description="H2 cross-species SSL transfer report.")
    parser.add_argument("--h1-log", type=Path, required=True, metavar="CSV",
                        help="H1 runs.csv: provides same-species SSL (finetune) + from-scratch (supervised_baseline).")
    parser.add_argument("--h2-log", type=Path, required=True, metavar="CSV",
                        help="H2 runs.csv: provides cross-species SSL (finetune).")
    parser.add_argument("--output", type=Path, required=True, metavar="DIR",
                        help="Directory to write the transfer figure.")
    args = parser.parse_args()

    same_ssl = _read_finetune_dice(args.h1_log, "H1", "finetune")
    scratch = _read_finetune_dice(args.h1_log, "H1", "supervised_baseline")
    cross_ssl = _read_finetune_dice(args.h2_log, "H2", "finetune")
    scratch_h2 = _read_finetune_dice(args.h2_log, "H2", "supervised_baseline")
    if not scratch:
        scratch = scratch_h2

    fractions = sorted(set(same_ssl) | set(cross_ssl) | set(scratch))
    if not fractions:
        raise SystemExit("No H1/H2 finetune rows found in the provided logs.")

    print(f"{'frac':>6} | {'scratch':>8} {'cross-SSL':>9} {'same-SSL':>9} | {'transfer ratio':>14}")
    ratios = {}
    for fr in fractions:
        s = scratch.get(fr)
        c = cross_ssl.get(fr)
        m = same_ssl.get(fr)
        ratio = None
        if s is not None and c is not None and m is not None and (m - s) > 1e-3:
            ratio = (c - s) / (m - s)
            ratios[fr] = ratio
        print(f"{fr:>6.2f} | {('%.3f'%s) if s is not None else '   -':>8} "
              f"{('%.3f'%c) if c is not None else '   -':>9} "
              f"{('%.3f'%m) if m is not None else '   -':>9} | "
              f"{('%.2f'%ratio) if ratio is not None else '   -':>14}")

    import matplotlib.pyplot as plt
    import numpy as np

    args.output.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(fractions))
    xticks = [f"{int(f * 100)}%" for f in fractions]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("H2 — Does self-supervised pretraining transfer across species?", fontsize=12)

    ax = axes[0]
    for label, data, color, marker in [
        ("from-scratch (no SSL)", scratch, "#9E9E9E", "o"),
        ("cross-species SSL (Droso+zebrafish)", cross_ssl, "#2196F3", "s"),
        ("same-species SSL (mouse)", same_ssl, "#4CAF50", "^"),
    ]:
        ys = [data.get(f) for f in fractions]
        xs = [i for i, y in enumerate(ys) if y is not None]
        yv = [y for y in ys if y is not None]
        if yv:
            ax.plot(xs, yv, marker=marker, label=label, color=color, linewidth=2, markersize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(xticks)
    ax.set_xlabel("Mouse labeled fraction (fine-tuning)")
    ax.set_ylabel("Test Dice (held-out mouse)")
    ax.set_title("Fine-tuned mouse segmentation by pretraining source")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[1]
    rf = [f for f in fractions if f in ratios]
    rv = [ratios[f] for f in rf]
    if rv:
        bars = ax.bar([f"{int(f*100)}%" for f in rf], rv, color="#673AB7", alpha=0.9, width=0.5)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.02, f"{h:.2f}", ha="center", va="bottom", fontsize=9)
    ax.axhline(1.0, color="#4CAF50", linewidth=1, linestyle="--", label="fully transfers (=1)")
    ax.axhline(0.0, color="#9E9E9E", linewidth=1, linestyle="--", label="no transfer (=0)")
    ax.set_xlabel("Mouse labeled fraction")
    ax.set_ylabel("Transfer ratio  (cross_gain / same_gain)")
    ax.set_title("Fraction of in-species SSL benefit retained cross-species")
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out = args.output / "h2_transfer.png"
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()

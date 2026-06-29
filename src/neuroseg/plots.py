import csv
from pathlib import Path


def _read_h1_results(log_path: Path) -> dict | None:
    if not log_path.exists():
        return None
    results: dict[str, dict[float, float]] = {"finetune": {}, "supervised_baseline": {}}
    with open(log_path, newline="") as f:
        for row in csv.DictReader(f):
            mode = row.get("mode", "")
            frac_str = row.get("labeled_fraction", "")
            dice_str = row.get("val_dice", "")
            if mode in results and frac_str and dice_str:
                try:
                    results[mode][float(frac_str)] = float(dice_str)
                except ValueError:
                    pass
    has_data = any(results[m] for m in results)
    return results if has_data else None


def _read_h2_results(log_path: Path) -> dict | None:
    if not log_path.exists():
        return None
    results: dict[str, float] = {}
    with open(log_path, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("hypothesis", "") != "H2":
                continue
            mode = row.get("mode", "")
            dice_str = row.get("val_dice", "")
            if mode in ("finetune", "supervised_baseline") and dice_str:
                try:
                    results[mode] = float(dice_str)
                except ValueError:
                    pass
    return results if results else None


def plot_h1_dice(log_path: Path, figures_dir: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    results = _read_h1_results(log_path)
    if not results:
        print(f"No H1 results in {log_path} — skipping H1 plot.")
        return

    fractions = sorted(set(list(results["finetune"]) + list(results["supervised_baseline"])))
    if not fractions:
        print("No H1 Dice results to plot.")
        return

    x = np.arange(len(fractions))
    w = 0.35
    jepa_vals = [results["finetune"].get(f, 0.0) for f in fractions]
    base_vals  = [results["supervised_baseline"].get(f, 0.0) for f in fractions]

    fig, ax = plt.subplots(figsize=(8, 5))
    b1 = ax.bar(x - w / 2, jepa_vals, w, label="JEPA Pretrained",     color="#2196F3", alpha=0.9)
    b2 = ax.bar(x + w / 2, base_vals,  w, label="Supervised Baseline", color="#FF9800", alpha=0.9)

    for bar in (*b1, *b2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.015,
                f"{h:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(f * 100)}%" for f in fractions])
    ax.set_xlabel("Labeled Data Fraction")
    ax.set_ylabel("Dice Score")
    ax.set_title("H1 — Semi-supervised Segmentation\nDice Score vs Labeled Data Fraction")
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    out = figures_dir / "h1_dice_comparison.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_h2_dice(log_path: Path, figures_dir: Path) -> None:
    import matplotlib.pyplot as plt

    results = _read_h2_results(log_path)
    if not results:
        print(f"No H2 results in {log_path} — skipping H2 plot.")
        return

    modes  = ["finetune", "supervised_baseline"]
    labels = ["JEPA Pretrained", "Supervised Baseline"]
    colors = ["#2196F3", "#FF9800"]
    vals   = [results.get(m, 0.0) for m in modes]

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(labels, vals, color=colors, alpha=0.9, width=0.5)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.015,
                f"{h:.2f}", ha="center", va="bottom", fontsize=10)

    ax.set_ylabel("Dice Score")
    ax.set_title("H2 — Cross-organism Transfer\nDice Score on Target Organism")
    ax.set_ylim(0, 1.1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    out = figures_dir / "h2_dice_comparison.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")

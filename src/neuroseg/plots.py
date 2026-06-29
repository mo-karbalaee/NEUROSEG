import csv
from pathlib import Path


def _safe_float(s: str) -> float | None:
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def _read_h1_results(log_path: Path) -> dict | None:
    if not log_path.exists():
        return None
    results: dict[str, dict[float, dict[str, float]]] = {"finetune": {}, "supervised_baseline": {}}
    with open(log_path, newline="") as f:
        for row in csv.DictReader(f):
            mode = row.get("mode", "")
            frac_str = row.get("labeled_fraction", "")
            dice_str = row.get("test_dice", "") or row.get("val_dice", "")
            miou_str = row.get("test_miou", "") or row.get("val_miou", "")
            if mode in results and frac_str and dice_str:
                try:
                    frac = float(frac_str)
                    results[mode][frac] = {
                        "dice": float(dice_str),
                        "miou": float(miou_str) if miou_str else 0.0,
                    }
                except ValueError:
                    pass
    has_data = any(results[m] for m in results)
    return results if has_data else None


def _read_h2_results(log_path: Path) -> dict | None:
    if not log_path.exists():
        return None
    results: dict[str, dict[str, float]] = {}
    with open(log_path, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("hypothesis", "") != "H2":
                continue
            mode = row.get("mode", "")
            dice_str = row.get("test_dice", "") or row.get("val_dice", "")
            miou_str = row.get("test_miou", "") or row.get("val_miou", "")
            if mode in ("finetune", "supervised_baseline") and dice_str:
                try:
                    results[mode] = {
                        "dice": float(dice_str),
                        "miou": float(miou_str) if miou_str else 0.0,
                    }
                except ValueError:
                    pass
    return results if results else None


def _read_run_epochs(log_path: Path, run_id: str) -> list[dict]:
    if not log_path.exists():
        return []
    rows = []
    with open(log_path, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("run_id", "") == run_id and row.get("epoch", "") != "":
                rows.append(row)
    rows.sort(key=lambda r: int(r["epoch"]))
    return rows


def plot_pretrain_curves(
    log_path: Path, run_id: str, model_name: str, figures_dir: Path
) -> None:
    import matplotlib.pyplot as plt

    rows = _read_run_epochs(log_path, run_id)
    if not rows:
        return

    epochs = [int(r["epoch"]) for r in rows]
    train_jepa = [_safe_float(r.get("train_loss", "")) for r in rows]
    val_jepa = [_safe_float(r.get("val_jepa_loss", "")) for r in rows]
    train_recon = [_safe_float(r.get("train_recon_loss", "")) for r in rows]
    val_recon = [_safe_float(r.get("val_recon_loss", "")) for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Pretrain — {model_name}", fontsize=12)

    ax = axes[0]
    if any(v is not None for v in train_jepa):
        ax.plot(epochs, [v or 0 for v in train_jepa], label="train JEPA loss", color="#2196F3")
    if any(v is not None for v in val_jepa):
        ax.plot(epochs, [v or 0 for v in val_jepa], label="val JEPA loss",
                color="#2196F3", linestyle="--", alpha=0.7)
    ax.set_title("JEPA Loss")
    ax.set_xlabel("Epoch")
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[1]
    if any(v is not None for v in train_recon):
        ax.plot(epochs, [v or 0 for v in train_recon], label="train recon loss", color="#FF9800")
    if any(v is not None for v in val_recon):
        ax.plot(epochs, [v or 0 for v in val_recon], label="val recon loss",
                color="#FF9800", linestyle="--", alpha=0.7)
    ax.set_title("Reconstruction Loss")
    ax.set_xlabel("Epoch")
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out = figures_dir / f"pretrain_{model_name}_curves.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_finetune_curves(
    log_path: Path, run_id: str, model_name: str, figures_dir: Path
) -> None:
    import matplotlib.pyplot as plt

    rows = _read_run_epochs(log_path, run_id)
    if not rows:
        return

    epochs = [int(r["epoch"]) for r in rows]
    train_loss = [_safe_float(r.get("train_loss", "")) for r in rows]
    val_dice = [_safe_float(r.get("val_dice", "")) for r in rows]
    val_miou = [_safe_float(r.get("val_miou", "")) for r in rows]

    test_dice, test_miou = None, None
    with open(log_path, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("run_id", "") == run_id and row.get("checkpoint", "") != "":
                test_dice = _safe_float(row.get("test_dice", ""))
                test_miou = _safe_float(row.get("test_miou", ""))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Training — {model_name}", fontsize=12)

    ax = axes[0, 0]
    if any(v is not None for v in train_loss):
        ax.plot(epochs, [v or 0 for v in train_loss], color="#E91E63", label="train loss (BCE)")
    ax.set_title("Train Loss")
    ax.set_xlabel("Epoch")
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[0, 1]
    if any(v is not None for v in val_dice):
        ax.plot(epochs, [v or 0 for v in val_dice], color="#4CAF50", label="val Dice")
    ax.set_title("Dice Score")
    ax.set_xlabel("Epoch")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[1, 0]
    if any(v is not None for v in val_miou):
        ax.plot(epochs, [v or 0 for v in val_miou], color="#9C27B0", label="val mIoU")
    ax.set_title("mIoU")
    ax.set_xlabel("Epoch")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[1, 1]
    if test_dice is not None and test_miou is not None:
        bars = ax.bar(
            ["Test Dice", "Test mIoU"], [test_dice, test_miou],
            color=["#4CAF50", "#9C27B0"], alpha=0.85, width=0.5,
        )
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.015,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=10)
        ax.set_ylim(0, 1.1)
    else:
        ax.text(0.5, 0.5, "No test scores", ha="center", va="center",
                transform=ax.transAxes, color="gray")
    ax.set_title("Final Test Scores")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out = figures_dir / f"finetune_{model_name}_curves.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_h3_similarity(log_path: Path, figures_dir: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    if not log_path.exists():
        print(f"No H3 results in {log_path} — skipping H3 plot.")
        return

    modes_data: dict[str, dict] = {}
    with open(log_path, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("hypothesis", "") != "H3":
                continue
            mode = row.get("mode", "")
            within = _safe_float(row.get("within_sim", ""))
            between = _safe_float(row.get("between_sim", ""))
            gap = _safe_float(row.get("gap", ""))
            if mode and within is not None:
                modes_data[mode] = {"within_sim": within, "between_sim": between, "gap": gap}

    if not modes_data:
        print(f"No H3 results in {log_path} — skipping H3 plot.")
        return

    mode_labels = list(modes_data.keys())
    within_vals = [modes_data[m]["within_sim"] for m in mode_labels]
    between_vals = [modes_data[m]["between_sim"] for m in mode_labels]
    gap_vals = [modes_data[m]["gap"] for m in mode_labels]

    x = np.arange(len(mode_labels))
    w = 0.25
    palette = ["#2196F3", "#FF9800", "#9C27B0"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("H3 — Temporal Representation Stability", fontsize=12)

    ax = axes[0]
    b1 = ax.bar(x - w / 2, within_vals, w, label="Within-neuron", color="#4CAF50", alpha=0.9)
    b2 = ax.bar(x + w / 2, between_vals, w, label="Between-neuron", color="#F44336", alpha=0.9)
    for bar in (*b1, *b2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                f"{h:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(mode_labels, rotation=15, ha="right")
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("Within vs Between Neuron Similarity")
    ax.set_ylim(-1, 1.25)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[1]
    bars = ax.bar(
        mode_labels, gap_vals,
        color=palette[:len(mode_labels)], alpha=0.9, width=0.5,
    )
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                f"{h:.3f}", ha="center", va="bottom", fontsize=10)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_ylabel("Similarity Gap (within − between)")
    ax.set_title("Representation Discriminability Gap")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out = figures_dir / "h3_similarity.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_h1_dice(log_path: Path, figures_dir: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    results = _read_h1_results(log_path)
    if not results:
        print(f"No H1 results in {log_path} — skipping H1 plot.")
        return

    fractions = sorted(set(list(results["finetune"]) + list(results["supervised_baseline"])))
    if not fractions:
        print("No H1 results to plot.")
        return

    x = np.arange(len(fractions))
    w = 0.35
    xtick_labels = [f"{int(f * 100)}%" for f in fractions]

    def _vals(metric: str, mode: str) -> list[float]:
        return [results[mode].get(f, {}).get(metric, 0.0) for f in fractions]

    figures_dir.mkdir(parents=True, exist_ok=True)
    for metric, ylabel, filename in [
        ("dice", "Dice Score", "h1_dice_comparison.png"),
        ("miou", "mIoU",       "h1_miou_comparison.png"),
    ]:
        jepa_vals = _vals(metric, "finetune")
        base_vals = _vals(metric, "supervised_baseline")

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_title(
            f"H1 — Semi-supervised Segmentation\n{ylabel} vs Labeled Data Fraction",
            fontsize=12,
        )
        b1 = ax.bar(x - w / 2, jepa_vals, w, label="JEPA Pretrained",     color="#2196F3", alpha=0.9)
        b2 = ax.bar(x + w / 2, base_vals,  w, label="Supervised Baseline", color="#FF9800", alpha=0.9)
        for bar in (*b1, *b2):
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.015,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(xtick_labels)
        ax.set_xlabel("Labeled Data Fraction")
        ax.set_ylabel(f"{ylabel} (test set)")
        ax.set_ylim(0, 1.1)
        ax.legend()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()

        out = figures_dir / filename
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

    figures_dir.mkdir(parents=True, exist_ok=True)
    for metric, ylabel, filename in [
        ("dice", "Dice Score", "h2_dice_comparison.png"),
        ("miou", "mIoU",       "h2_miou_comparison.png"),
    ]:
        vals = [results.get(m, {}).get(metric, 0.0) for m in modes]

        fig, ax = plt.subplots(figsize=(6, 5))
        bars = ax.bar(labels, vals, color=colors, alpha=0.9, width=0.5)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.015,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=10)
        ax.set_ylabel(f"{ylabel} (test set)")
        ax.set_title(f"H2 — Cross-organism Transfer\n{ylabel} on Target Organism", fontsize=12)
        ax.set_ylim(0, 1.1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()

        out = figures_dir / filename
        plt.savefig(str(out), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out}")

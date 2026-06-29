from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from neuroseg.models.state import State


def _show_mask_overlay(ax: plt.Axes, frame: np.ndarray, mask: np.ndarray):
    """Render a grayscale frame with a semi-transparent colored mask overlay."""
    ax.imshow(frame, cmap="gray")
    if mask.max() > 0:
        colored = np.zeros((*frame.shape[:2], 4), dtype=np.float32)
        for label_id in range(1, int(mask.max()) + 1):
            region = mask == label_id
            color = plt.cm.tab20(label_id % 20)
            colored[region] = [*color[:3], 0.5]
        ax.imshow(colored)
    ax.axis("off")


def _visualize_segmentation(
    data: np.ndarray,
    masks,
    flows: Optional[list],
    output_dir: str,
    file_name: str,
):
    """Save a per-frame segmentation overlay PNG for every frame in the stack."""
    out = Path(output_dir) / "segmentation" / file_name
    out.mkdir(parents=True, exist_ok=True)

    for t in range(data.shape[0]):
        frame = data[t] if data[t].ndim == 2 else data[t][:, :, 0]

        if flows is not None:
            from cellpose import plot
            fig = plt.figure(figsize=(8, 8))
            plot.show_segmentation(fig, frame, masks[t], flows[t][0])
            plt.title(f"Frame {t}")
        else:
            fig, ax = plt.subplots(figsize=(8, 8))
            _show_mask_overlay(ax, frame, masks[t])
            ax.set_title(f"Frame {t}")

        plt.savefig(str(out / f"frame_{t}.png"), dpi=100, bbox_inches="tight")
        plt.close(fig)


def _visualize_traces(traces: np.ndarray, output_dir: str, file_name: str):
    """Save a single combined plot of all neuron activity traces."""
    N, T = traces.shape
    colors = plt.cm.tab20(np.linspace(0, 1, N))
    out = Path(output_dir) / "traces" / file_name
    out.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(15, 6))
    for n in range(N):
        ax.plot(traces[n], linewidth=0.8, color=colors[n], label=f"Neuron {n + 1}")
    ax.set_xlabel("Frame")
    ax.set_ylabel("ΔF/F₀")
    ax.set_xlim(0, T)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), fontsize=8, ncol=5)
    plt.title("Neural Activity Traces")
    plt.tight_layout()
    plt.savefig(str(out / "traces_combined.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_individual_traces(traces: np.ndarray, output_dir: str, file_name: str):
    """Save one PNG per neuron showing its individual ΔF/F₀ trace."""
    N, T = traces.shape
    colors = plt.cm.tab20(np.linspace(0, 1, N))
    out = Path(output_dir) / "traces" / file_name
    out.mkdir(parents=True, exist_ok=True)

    for n in range(N):
        fig, ax = plt.subplots(figsize=(15, 4))
        ax.plot(traces[n], linewidth=0.8, color=colors[n])
        ax.set_xlabel("Frame")
        ax.set_ylabel("ΔF/F₀")
        ax.set_xlim(0, T)
        ax.set_title(f"Neuron {n + 1}")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        plt.savefig(str(out / f"neuron_{n + 1}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)


def visualizer_node(state: State) -> dict:
    """Produce all segmentation and trace visualizations for the current file."""
    _visualize_segmentation(
        state["data"], state["masks"], state["flows"], state["output_dir"], state["file_name"]
    )
    _visualize_traces(state["traces"], state["output_dir"], state["file_name"])
    _save_individual_traces(state["traces"], state["output_dir"], state["file_name"])
    print(f"Finished processing {state['file_name']}")
    return {}

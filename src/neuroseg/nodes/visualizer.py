import os
import numpy as np
import matplotlib.pyplot as plt
from cellpose import plot
from neuroseg.models.state import State


def _visualize_segmentation(data: np.ndarray, masks, flows, file_name: str):
    output_dir = f"output/{file_name}"
    os.makedirs(output_dir, exist_ok=True)
    fig = plt.figure(figsize=(8, 8))
    for t in range(data.shape[0]):
        plt.clf()
        plot.show_segmentation(fig, data[t], masks[t], flows[t][0])
        plt.title(f"Frame {t}")
        plt.savefig(f"{output_dir}/segmentation_frame_{t}.png", dpi=100, bbox_inches="tight")
    plt.close(fig)


def _visualize_traces(traces: np.ndarray, file_name: str):
    N, T = traces.shape
    colors = plt.cm.tab20(np.linspace(0, 1, N))
    output_dir = f"output/{file_name}"
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(15, 6))
    for n in range(N):
        ax.plot(traces[n], linewidth=0.8, color=colors[n], label=f"Neuron {n + 1}")
    ax.set_xlabel("Frame")
    ax.set_ylabel("ΔF/F₀")
    ax.set_xlim(0, T)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fontsize=8, ncol=5)
    plt.title("Neural Activity Traces")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/traces_combined.png", dpi=150, bbox_inches='tight')
    plt.close(fig)


def _save_individual_traces(traces: np.ndarray, file_name: str):
    N, T = traces.shape
    colors = plt.cm.tab20(np.linspace(0, 1, N))
    output_dir = f"output/{file_name}"
    os.makedirs(output_dir, exist_ok=True)

    for n in range(N):
        fig, ax = plt.subplots(figsize=(15, 4))
        ax.plot(traces[n], linewidth=0.8, color=colors[n])
        ax.set_xlabel("Frame")
        ax.set_ylabel("ΔF/F₀")
        ax.set_xlim(0, T)
        ax.set_title(f"Neuron {n + 1}")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/trace_neuron_{n + 1}.png", dpi=150, bbox_inches='tight')
        plt.close(fig)


def visualizer_node(state: State) -> dict:
    data = state["data"]
    file_name = state["file_name"]
    masks = state["masks"]
    flows = state["flows"]
    traces = state["traces"]

    _visualize_segmentation(data, masks, flows, file_name)
    _visualize_traces(traces, file_name)
    _save_individual_traces(traces, file_name)

    print(f"Finished processing {file_name}")
    return {}

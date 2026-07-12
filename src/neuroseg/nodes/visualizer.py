from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from neuroseg.models.state import State


def _to_gray_uint8(frame: np.ndarray) -> np.ndarray:
    """Return a single-channel 0–255 uint8 view of a frame, contrast-stretched."""
    f = frame.astype(np.float32)
    if f.ndim == 3:
        f = f[:, :, 0]
    lo, hi = float(f.min()), float(f.max())
    if hi > lo:
        f = (f - lo) / (hi - lo) * 255.0
    return f.astype(np.uint8)


def _overlay_frame(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Draw per-neuron contour outlines over a grayscale frame; return an RGB uint8 image."""
    gray = _to_gray_uint8(frame)
    rgb = np.stack([gray] * 3, axis=-1).astype(np.uint8)
    if mask is not None and mask.max() > 0:
        for label_id in range(1, int(mask.max()) + 1):
            region = (mask == label_id).astype(np.uint8)
            if not region.any():
                continue
            color = tuple(int(c * 255) for c in plt.cm.tab20(label_id % 20)[:3])
            contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(rgb, contours, -1, color, 1)
    return rgb


def _visualize_segmentation(
    data: np.ndarray,
    masks,
    output_dir: str,
    file_name: str,
    fps: int = 10,
):
    """Write one MP4 of the segmentation overlay for the whole stack (one frame per time step)."""
    out = Path(output_dir) / "segmentation"
    out.mkdir(parents=True, exist_ok=True)
    video_path = out / f"{file_name}.mp4"

    T = data.shape[0]
    first = _overlay_frame(data[0], masks[0])
    h, w = first.shape[:2]
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    for t in range(T):
        rgb = _overlay_frame(data[t], masks[t])
        cv2.putText(rgb, f"frame {t}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 1, cv2.LINE_AA)
        writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    writer.release()
    print(f"[video] {video_path} ({T} frames @ {fps} fps)")


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
        state["data"], state["masks"], state["output_dir"], state["file_name"]
    )
    _visualize_traces(state["traces"], state["output_dir"], state["file_name"])
    _save_individual_traces(state["traces"], state["output_dir"], state["file_name"])
    print(f"Finished processing {state['file_name']}")
    return {}

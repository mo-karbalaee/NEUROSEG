from pathlib import Path

import numpy as np

from neuroseg.models.state import State


def _extract_traces(
    masks: list,
    data: np.ndarray,
    f0_percentile: int = 10,
    dff_epsilon: float = 1e-6,
) -> np.ndarray:
    T = data.shape[0]
    if masks is None or len(masks) == 0:
        return np.zeros((0, T))
    reference_idx = int(np.argmax([np.max(m) for m in masks]))
    reference_mask = masks[reference_idx]
    N = int(np.max(reference_mask))

    traces = np.zeros((N, T))
    for t in range(T):
        for n in range(1, N + 1):
            pixel_values = data[t][reference_mask == n]
            if len(pixel_values) > 0:
                traces[n - 1, t] = pixel_values.mean()

    F0 = np.percentile(traces, f0_percentile, axis=1, keepdims=True)
    return (traces - F0) / (F0 + dff_epsilon)


def _save_traces(traces: np.ndarray, output_dir: str, file_name: str):
    out = Path(output_dir) / "traces"
    out.mkdir(parents=True, exist_ok=True)
    np.save(str(out / f"traces+{file_name}.npy"), traces)


def activity_trace_calculator_node(state: State) -> dict:
    cfg = state.get("config", {})
    traces = _extract_traces(
        state["masks"],
        state["data"],
        f0_percentile=cfg.get("f0_percentile", 10),
        dff_epsilon=cfg.get("dff_epsilon", 1e-6),
    )
    _save_traces(traces, state["output_dir"], state["file_name"])
    return {"traces": traces}

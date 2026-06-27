import pickle
from pathlib import Path

import numpy as np
from cellpose import models

from neuroseg.models.state import State


def _generate_mask(data: np.ndarray):
    model = models.CellposeModel(gpu=True)
    frames = [data[i] for i in range(data.shape[0])]
    masks, flows, _ = model.eval(frames, diameter=None, channels=[0, 0])
    return masks, flows


def _cache_path(output_dir: str, file_name: str, ext: str) -> Path:
    return Path(output_dir) / "cache" / f"{ext}+{file_name}"


def _save_results(masks, flows, output_dir: str, file_name: str):
    cache_dir = Path(output_dir) / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(str(_cache_path(output_dir, file_name, "masks")) + ".npy", masks)
    with open(str(_cache_path(output_dir, file_name, "flows")) + ".pkl", "wb") as f:
        pickle.dump(flows, f)


def _load_results(output_dir: str, file_name: str):
    masks = np.load(
        str(_cache_path(output_dir, file_name, "masks")) + ".npy", allow_pickle=True
    )
    with open(str(_cache_path(output_dir, file_name, "flows")) + ".pkl", "rb") as f:
        flows = pickle.load(f)
    return masks, flows


def segmenter_node(state: State) -> dict:
    file_name = state["file_name"]
    output_dir = state["output_dir"]
    masks_path = _cache_path(output_dir, file_name, "masks").with_suffix(".npy")

    if masks_path.exists():
        print(f"Masks already cached for {file_name}")
        masks, flows = _load_results(output_dir, file_name)
    else:
        print(f"Generating masks for {file_name}")
        masks, flows = _generate_mask(state["data"])
        _save_results(masks, flows, output_dir, file_name)

    return {"masks": masks, "flows": flows}

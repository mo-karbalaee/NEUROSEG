import os
import pickle
import numpy as np
from cellpose import models
from neuroseg.models.state import State


def _generate_mask(data: np.ndarray):
    model = models.CellposeModel(gpu=True)
    frames = [data[i] for i in range(data.shape[0])]
    masks, flows, _ = model.eval(frames, diameter=None, channels=[0, 0])
    return masks, flows


def _save_results(masks, flows, file_name: str):
    os.makedirs("output", exist_ok=True)
    np.save(os.path.join("output", f"masks+{file_name}.npy"), masks)
    with open(os.path.join("output", f"flows+{file_name}.pkl"), "wb") as f:
        pickle.dump(flows, f)


def _load_results(file_name: str):
    masks = np.load(os.path.join("output", f"masks+{file_name}.npy"), allow_pickle=True)
    with open(os.path.join("output", f"flows+{file_name}.pkl"), "rb") as f:
        flows = pickle.load(f)
    return masks, flows


def segmenter_node(state: State) -> dict:
    file_name = state["file_name"]
    masks_path = os.path.join("output", f"masks+{file_name}.npy")

    if os.path.exists(masks_path):
        print(f"Masks already processed for {file_name}")
        masks, flows = _load_results(file_name)
    else:
        print(f"Generating masks for {file_name}")
        masks, flows = _generate_mask(state["data"])
        _save_results(masks, flows, file_name)

    return {"masks": masks, "flows": flows}

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import label as connected_components

from neuroseg.checkpoint import load_compound_checkpoint
from neuroseg.models.state import State
from neuroseg.trainers.jepa import build_jepa, build_seg_head


def _jepa_segment(data: np.ndarray, checkpoint_path: str) -> tuple[list, None]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    payload = load_compound_checkpoint(Path(checkpoint_path))
    arch = payload["arch"]

    jepa = build_jepa(arch, device)
    jepa.load_state_dict(payload["jepa"])
    jepa.eval()

    seg_head = build_seg_head(arch["dstc"], arch.get("seg_head_hidden", 16)).to(device)
    seg_head.load_state_dict(payload["seg_head"])
    seg_head.eval()

    T = data.shape[0]
    masks = []

    with torch.no_grad():
        for t in range(T):
            frame = data[t]
            if frame.ndim == 3:
                frame = frame[:, :, 0]
            H, W = frame.shape

            x = torch.from_numpy(frame).float()
            x = x.unsqueeze(0).unsqueeze(0).unsqueeze(0).to(device)

            enc_state = jepa.encoder(x)
            enc_mean = enc_state.squeeze(2)
            pred = seg_head(enc_mean)
            pred = F.interpolate(pred, size=(H, W), mode="bilinear", align_corners=False)
            binary = (pred.squeeze().cpu().numpy() > 0.5).astype(np.uint8)

            labeled, _ = connected_components(binary)
            masks.append(labeled)

    return masks, None


def _cellpose_segment(data: np.ndarray) -> tuple[list, list]:
    from cellpose import models
    model = models.CellposeModel(gpu=True)
    frames = [data[i] for i in range(data.shape[0])]
    masks, flows, _ = model.eval(frames, diameter=None, channels=[0, 0])
    return masks, flows


def _cache_root(output_dir: str, model_key: str) -> Path:
    return Path(output_dir) / "cache" / model_key


def _save_results(masks, flows: Optional[list], output_dir: str, file_name: str, model_key: str):
    cache_dir = _cache_root(output_dir, model_key)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(str(cache_dir / f"masks+{file_name}.npy"), masks)
    with open(cache_dir / f"flows+{file_name}.pkl", "wb") as f:
        pickle.dump(flows, f)


def _load_results(output_dir: str, file_name: str, model_key: str) -> tuple:
    cache_dir = _cache_root(output_dir, model_key)
    masks = np.load(str(cache_dir / f"masks+{file_name}.npy"), allow_pickle=True)
    with open(cache_dir / f"flows+{file_name}.pkl", "rb") as f:
        flows = pickle.load(f)
    return masks, flows


def _model_key(checkpoint_path: Optional[str]) -> str:
    if checkpoint_path is None:
        return "cellpose"
    return "jepa_" + Path(checkpoint_path).stem


def segmenter_node(state: State) -> dict:
    file_name = state["file_name"]
    output_dir = state["output_dir"]
    checkpoint_path = state.get("checkpoint_path")
    key = _model_key(checkpoint_path)
    cache_file = _cache_root(output_dir, key) / f"masks+{file_name}.npy"

    if cache_file.exists():
        print(f"[{key}] Masks cached for {file_name}")
        masks, flows = _load_results(output_dir, file_name, key)
    elif checkpoint_path:
        print(f"[JEPA] Segmenting {file_name}")
        masks, flows = _jepa_segment(state["data"], checkpoint_path)
        _save_results(masks, flows, output_dir, file_name, key)
    else:
        print(f"[Cellpose] Segmenting {file_name}")
        masks, flows = _cellpose_segment(state["data"])
        _save_results(masks, flows, output_dir, file_name, key)

    return {"masks": masks, "flows": flows}

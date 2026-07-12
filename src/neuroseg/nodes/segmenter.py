import pickle
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import label as connected_components

from neuroseg.checkpoint import load_compound_checkpoint
from neuroseg.models.state import State
from neuroseg.trainers.jepa import build_jepa, build_seg_head


def _filter_small_components(labeled: np.ndarray, min_size: int) -> np.ndarray:
    """Remove connected components smaller than min_size and relabel the remaining ones."""
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    keep = sizes >= min_size
    filtered = np.where(keep[labeled], labeled, 0)
    relabeled, _ = connected_components(filtered > 0)
    return relabeled


def _normalize_frame(frame: np.ndarray) -> np.ndarray:
    """Min-max normalize a single frame to [0, 1], matching the training data pipeline."""
    f = frame.astype(np.float32)
    if f.ndim == 3:
        f = f[:, :, 0]
    mn, mx = float(f.min()), float(f.max())
    return (f - mn) / (mx - mn + 1e-8)


def _jepa_segment(data: np.ndarray, checkpoint_path: str, window: int = 5) -> tuple[list, None]:
    """
    Segment a stack with a loaded JEPA encoder + seg head.

    Matches the training-time input pipeline: each frame is min-max normalized to
    [0, 1], and encoder features are averaged over a temporal window (default 5)
    before the head — the same temporal aggregation used during fine-tuning. Feeding
    raw, unnormalized single frames (the previous behavior) put the encoder far out of
    its trained input range and made it flood ~45% of the frame with mask.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    payload = load_compound_checkpoint(Path(checkpoint_path))
    arch = payload["arch"]
    img_size = arch.get("img_size", 128)

    jepa = build_jepa(arch, device)
    jepa.load_state_dict(payload["jepa"], strict=False)
    jepa.eval()

    seg_head = build_seg_head(arch["dstc"], arch.get("seg_head_hidden", 16)).to(device)
    seg_head.load_state_dict(payload["seg_head"])
    seg_head.eval()

    seg_threshold = arch.get("seg_threshold", 0.5)

    T = data.shape[0]
    frame0 = data[0] if data[0].ndim == 2 else data[0][:, :, 0]
    H, W = frame0.shape[:2]
    min_size = max(9, (H * W) // 8000)

    features = []
    with torch.no_grad():
        for t in range(T):
            frame = _normalize_frame(data[t])
            frame = cv2.resize(frame, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
            x = torch.from_numpy(frame).float().view(1, 1, 1, img_size, img_size).to(device)
            features.append(jepa.encoder(x)[:, :, 0])

    half = window // 2
    masks = []
    with torch.no_grad():
        for t in range(T):
            a, b = max(0, t - half), min(T, t + half + 1)
            feat = torch.cat(features[a:b], dim=0).mean(dim=0, keepdim=True)
            pred = seg_head(feat)
            pred = F.interpolate(pred, size=(H, W), mode="bilinear", align_corners=False)
            binary = (pred.squeeze().cpu().numpy() > seg_threshold).astype(np.uint8)
            labeled = _filter_small_components(connected_components(binary)[0], min_size)
            masks.append(labeled)

    return masks, None


def _cellpose_segment(data: np.ndarray) -> tuple[list, list]:
    """Segment all frames using the Cellpose baseline model."""
    from cellpose import models
    model = models.CellposeModel(gpu=True)
    frames = [data[i] for i in range(data.shape[0])]
    masks, flows, _ = model.eval(frames, diameter=None, channels=[0, 0])
    return masks, flows


def _cache_root(output_dir: str, model_key: str) -> Path:
    """Return the cache directory path for a given model key."""
    return Path(output_dir) / "cache" / model_key


def _save_results(masks, flows: Optional[list], output_dir: str, file_name: str, model_key: str):
    """Persist segmentation masks and flows to the on-disk cache."""
    cache_dir = _cache_root(output_dir, model_key)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(str(cache_dir / f"masks+{file_name}.npy"), masks)
    with open(cache_dir / f"flows+{file_name}.pkl", "wb") as f:
        pickle.dump(flows, f)


def _load_results(output_dir: str, file_name: str, model_key: str) -> tuple:
    """Load previously cached masks and flows from disk."""
    cache_dir = _cache_root(output_dir, model_key)
    masks = np.load(str(cache_dir / f"masks+{file_name}.npy"), allow_pickle=True)
    with open(cache_dir / f"flows+{file_name}.pkl", "rb") as f:
        flows = pickle.load(f)
    return masks, flows


def _model_key(checkpoint_path: Optional[str], h: int, w: int) -> str:
    """Build a cache key that encodes the model and frame resolution."""
    base = "cellpose" if checkpoint_path is None else "jepa_" + Path(checkpoint_path).stem
    return f"{base}_{h}x{w}"


def segmenter_node(state: State) -> dict:
    """Segment the current frame stack, using the cache when available."""
    file_name = state["file_name"]
    output_dir = state["output_dir"]
    checkpoint_path = state.get("checkpoint_path")
    _, h, w = state["data"].shape[0], state["data"].shape[1], state["data"].shape[2]
    key = _model_key(checkpoint_path, h, w)
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

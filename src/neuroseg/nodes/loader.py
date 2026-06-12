from pathlib import Path
import cv2
import numpy as np
import tifffile as tiff
from neuroseg.models.state import State


def _load_avi(path: str) -> np.ndarray:
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return np.array(frames)


def _load(path: str) -> np.ndarray:
    if "avi" in path:
        data = _load_avi(path)
    else:
        data = tiff.imread(path)
    if data.ndim == 2:
        data = data[np.newaxis, ...]
    return data


def loader_node(state: State) -> dict:
    idx = state["current_file_index"]
    file_path = state["file_paths"][idx]
    data = _load(str(file_path))
    file_name = Path(file_path).name
    print(f"Processing {file_name}, dimensions: {data.shape}")
    return {
        "data": data,
        "file_name": file_name,
        "current_file_index": idx + 1,
        "masks": None,
        "flows": None,
        "traces": None,
    }

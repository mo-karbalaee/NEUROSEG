import os
import pickle
import numpy as np
from cellpose import models
from neuroseg.calcium_recording import CalciumRecording
from scipy import stats

class Segmenter:
    def __init__(self):
        pass

    @staticmethod
    def generate_mask(recording: CalciumRecording):
        model = models.CellposeModel(gpu=True)
        data = [recording.data[i] for i in range(recording.data.shape[0])]
        masks, flows, styles = model.eval(data, diameter=None, channels=[0, 0])
        return masks, flows

    @staticmethod
    def extract_traces(masks: list, recording: CalciumRecording) -> np.ndarray:
        T = recording.data.shape[0]
        all_counts = [np.max(m) for m in masks]

        reference_idx = int(np.argmax(all_counts))
        reference_mask = masks[reference_idx]
        N = int(np.max(reference_mask))

        traces = np.zeros((N, T))

        for t in range(T):
            for n in range(1, N + 1):
                pixel_values = recording.data[t][reference_mask == n]
                if len(pixel_values) > 0:
                    F = pixel_values.mean()
                    traces[n - 1, t] = F

        F0 = np.percentile(traces, 10, axis=1, keepdims=True)
        dff = (traces - F0) / (F0 + 1e-6)

        return dff

    @staticmethod
    def save_results(masks, flows, file_name):
        save_dir = "../../output"
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, f"masks+{file_name}.npy"), masks)
        with open(os.path.join(save_dir, f"flows+{file_name}.pkl"), "wb") as f:
            pickle.dump(flows, f)

    @staticmethod
    def load_results(file_name):
        save_dir = "../../output"
        masks = np.load(os.path.join(save_dir, f"masks+{file_name}.npy"), allow_pickle=True)
        with open(os.path.join(save_dir, f"flows+{file_name}.pkl"), "rb") as f:
            flows = pickle.load(f)

        return masks, flows

    @staticmethod
    def save_traces(traces: np.ndarray, file_name: str):
        save_dir = "../../output"
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, f"traces+{file_name}.npy"), traces)

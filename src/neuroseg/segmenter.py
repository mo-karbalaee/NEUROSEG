import pickle

import numpy as np
from cellpose import models

from neuroseg.calcium_recording import CalciumRecording


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
        reference_mask = masks[0]
        N = int(np.max(reference_mask))
        T = recording.data.shape[0]
        traces = np.zeros((N, T))

        for t in range(T):
            for n in range(1, N + 1):
                pixel_values = recording.data[t][reference_mask == n]
                if len(pixel_values) > 0:
                    traces[n - 1, t] = pixel_values.mean()

        return traces

    @staticmethod
    def save_traces(traces: np.ndarray, path: str):
        np.save(path, traces)

    @staticmethod
    def save_results(masks, flows, file_name):
        np.save(f"masks+{file_name}.npy", masks)
        with open(f"flows+{file_name}.pkl", "wb") as f:
            pickle.dump(flows, f)

    @staticmethod
    def load_results(file_name):
        masks = np.load(f"masks+{file_name}.npy", allow_pickle=True)
        with open(f"flows+{file_name}.pkl", "rb") as f:
            flows = pickle.load(f)
        return masks, flows

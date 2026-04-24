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
        stacked = np.array(masks)
        consensus_mask = stats.mode(stacked, axis=0).mode
        N = int(np.max(consensus_mask))
        T = recording.data.shape[0]
        traces = np.zeros((N, T))

        for t in range(T):
            for n in range(1, N + 1):
                pixel_values = recording.data[t][consensus_mask == n]
                if len(pixel_values) > 0:
                    traces[n - 1, t] = pixel_values.mean()

        return traces

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

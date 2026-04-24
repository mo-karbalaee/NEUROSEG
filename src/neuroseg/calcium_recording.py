from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tifffile as tiff
from cellpose import plot


class CalciumRecording:
    def __init__(self, path: str):
        self.data: np.ndarray = self._load(path)
        print("Data dimensions: ", self.data.shape)
        self.path = path
        self.file_name = Path(path).name

    @staticmethod
    def _load_avi(path: str):
        cap = cv2.VideoCapture(path)

        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)

        cap.release()
        return np.array(frames)

    @staticmethod
    def _load(path: str):
        if "avi" in path:
            data = CalciumRecording._load_avi(path)
        else:
            data = tiff.imread(path)

        if data.ndim == 2:
            data = data[np.newaxis, ...]

        return data

    def visualize(self, masks: np.ndarray, flows: np.ndarray):
        fig = plt.figure(figsize=(8, 8))
        for t in range(self.data.shape[0]):
            plt.clf()
            plot.show_segmentation(fig, self.data[t], masks[t], flows[t][0])
            plt.title(f"Frame {t}")
            plt.pause(0.05)

        plt.show()

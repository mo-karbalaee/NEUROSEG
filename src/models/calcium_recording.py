import cv2
import matplotlib.pyplot as plt
import numpy as np
import tifffile as tiff
from cellpose import plot


class CalciumRecording:
    def __init__(self, path: str):
        self.data: np.ndarray = self._load(path)
        self.path = path

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

    def _load(self, path: str):
        if "avi" in path:
            data = CalciumRecording._load_avi(path)
        else:
            data = tiff.imread(path)

        return data

    def visualize(self, masks: np.ndarray, flows: np.ndarray):
        fig = plt.figure(figsize=(10, 10))

        T = self.data.shape[0]

        for t in range(T):
            plot.show_segmentation(
                fig,
                self.data[t],
                masks,
                flows[t][0]
            )


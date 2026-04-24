import cv2
import matplotlib.pyplot as plt
import tifffile as tiff
from cellpose import plot
from pathlib import Path
import numpy as np

class CalciumRecording:
    def __init__(self, path: str):
        self.data: np.ndarray = self._load(path)
        print(self.data.shape)
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

    def _load(self, path: str):
        if "avi" in path:
            data = CalciumRecording._load_avi(path)
        else:
            data = tiff.imread(path)

        return data

    def visualize(self, masks: np.ndarray, flows: np.ndarray):
        from matplotlib.animation import FuncAnimation, FFMpegWriter

        fig = plt.figure(figsize=(20, 20))

        def update(t):
            fig.clf()
            plot.show_segmentation(fig, self.data[t], masks[t], flows[t][0])

        ani = FuncAnimation(fig, update, frames=self.data.shape[0], interval=100)
        writer = FFMpegWriter(fps=10)
        ani.save(f"segmentation+{self.file_name}.mp4", writer=writer)
        plt.close()
        print(f"Saved to segmentation+{self.file_name}.mp4")


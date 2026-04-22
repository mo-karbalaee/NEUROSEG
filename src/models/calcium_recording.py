import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
import cv2

class CalciumRecording:
    def __init__(self,path:str):
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

    def _load(self,path: str):
        if "avi" in path:
            data = CalciumRecording._load_avi(path)
        else:
            data = tiff.imread(path)

        return data

    def visualize_raw(self):
        plt.figure()

        for t in range(0, self.data.shape[0]):
            plt.imshow(self.data[t], cmap="gray")
            plt.title(f"t = {t}")
            plt.axis("off")

            plt.pause(0.05)
            plt.clf()

        plt.close()
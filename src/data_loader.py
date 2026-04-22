import matplotlib.pyplot as plt
import numpy as np
import tifffile as tiff
import cv2
from models.calcium_recording import CalciumRecording

class DataLoader:
    def __init__(self):
        pass

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
    def load(path: str):
        if "avi" in path:
            data = DataLoader._load_avi(path)
        else:
            data = tiff.imread(path)

        return CalciumRecording(data, path)

    @staticmethod
    def visualize(recording: CalciumRecording):
        plt.figure()

        for t in range(0, recording.data.shape[0]):
            plt.imshow(recording.data[t], cmap="gray")
            plt.title(f"t = {t}")
            plt.axis("off")

            plt.pause(0.05)
            plt.clf()

        plt.close()

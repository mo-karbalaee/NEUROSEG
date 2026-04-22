import tifffile as tiff
from models.calcium_recording import CalciumRecording
import matplotlib.pyplot as plt

class DataLoader:
    def __init__(self):
        pass

    @staticmethod
    def load(path: str):
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

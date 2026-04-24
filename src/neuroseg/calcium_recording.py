from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tifffile as tiff
from cellpose import plot
import os

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

    @staticmethod
    def save_individual_traces(traces: np.ndarray, file_name: str):
        N, T = traces.shape
        colors = plt.cm.tab20(np.linspace(0, 1, N))
        output_dir = f"output/{file_name}"
        os.makedirs(output_dir, exist_ok=True)

        for n in range(N):
            fig, ax = plt.subplots(figsize=(15, 4))
            ax.plot(traces[n], linewidth=0.8, color=colors[n])
            ax.set_xlabel("Frame")
            ax.set_ylabel("ΔF/F₀")
            ax.set_xlim(0, T)
            ax.set_title(f"Neuron {n + 1}")
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/trace_neuron_{n + 1}.png", dpi=150, bbox_inches='tight')
            plt.close()

    @staticmethod
    def visualize_traces(traces: np.ndarray, file_name: str):
        N, T = traces.shape
        colors = plt.cm.tab20(np.linspace(0, 1, N))
        output_dir = f"output/{file_name}"
        os.makedirs(output_dir, exist_ok=True)

        fig, ax = plt.subplots(figsize=(15, 6))
        for n in range(N):
            ax.plot(traces[n], linewidth=0.8, color=colors[n], label=f"Neuron {n + 1}")
        ax.set_xlabel("Frame")
        ax.set_ylabel("ΔF/F₀")
        ax.set_xlim(0, T)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fontsize=8, ncol=5)
        plt.title("Neural Activity Traces")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/traces_combined.png", dpi=150, bbox_inches='tight')
        plt.show()
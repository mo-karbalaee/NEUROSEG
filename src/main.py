from src.models.calcium_recording import CalciumRecording
from src.segmenter import Segmenter

recording: CalciumRecording = CalciumRecording(path="../data/6s.tif")

# DataLoader.visualize(recording)

Segmenter.generate_mask(recording)
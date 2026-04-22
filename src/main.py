from src.data_loader import DataLoader
from src.models.calcium_recording import CalciumRecording

recording: CalciumRecording = DataLoader.load("../data/6s.tif")

DataLoader.visualize(recording)
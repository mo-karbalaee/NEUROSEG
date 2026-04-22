from src.data_loader import DataLoader
from src.models.calcium_recording import CalciumRecording

recording: CalciumRecording = DataLoader.load("../data/Medien1.avi")

DataLoader.visualize(recording)
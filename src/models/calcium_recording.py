import numpy as np

class CalciumRecording:
    def __init__(self, data: np.ndarray, path:str):
        self.data = data
        self.path = path

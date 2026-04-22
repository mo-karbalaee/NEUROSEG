from cellpose import models

from models.calcium_recording import CalciumRecording


class Segmenter:
    def __init__(self):
        pass

    @staticmethod
    def generate_mask(recording: CalciumRecording):
        model = models.CellposeModel()
        data = [recording.data[i] for i in range(recording.data.shape[0])]
        masks, flows, styles = model.eval(data, diameter=None, channels=[0, 0])
        return masks, flows
